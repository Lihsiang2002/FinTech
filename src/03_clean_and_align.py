"""Stage 2：法說會雙區塊切分、增強去識別化與嚴格 Point-in-Time 對齊（01_raw -> 02_clean）。

功能：
1. 唯讀讀取 data/01_raw/transcripts_raw/year=*/part.parquet 與 nyse_calendar.parquet。
2. 剝離早期逐字稿之 Executives / Analysts 目錄頁首，採兩層規則將正文切分為
   prepared_remarks_raw 與 qa_session_raw。
3. 並存原文版（*_raw）與三層增強去識別化版（*_anon）：
   (a) 動態公司名 + config/redact_dict.json 品牌／產品字典 -> [COMPANY] / [PRODUCT]
   (b) 結構化 speaker 動態識別高管與分析師姓名 -> [EXECUTIVE] / [ANALYST]
   (c) 金融年份與季度正則（排除 $ 與千分位金額誤傷） -> [YEAR] / [CURRENT_QUARTER]
4. 嚴格以美東時間（America/New_York）與 NYSE 官方交易日曆對齊事件交易日 t=0：
   - 06:00–09:29 且當日開盤（BMO） -> 當日 t=0
   - >= 09:30（AMC/盤中）、休市日、或 00:00–05:59 異常時段 -> 一律保守映射至下一個 NYSE 交易日
   - 同步寫出 pre_end_date (t-1)、post_10d_date (t+10)、post_20d_date (t+20)
5. 異常樣本採軟刪除（is_valid = False + exclude_reason），保留 100% 稽核軌跡，
   並產出含 10 筆分層人工抽檢明細之 outputs/stage2_clean_audit.md。

執行範例：
    python -m src.03_clean_and_align
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.config import (
    CLEAN_CFG,
    CLEAN_TRANSCRIPTS_DIR,
    DATA_CFG,
    QUANT_CFG,
    RAW_NYSE_CALENDAR_PATH,
    RAW_TRANSCRIPTS_DIR,
    REDACT_DICT_PATH,
    STAGE2_AUDIT_MD,
    ensure_project_dirs,
    load_universe,
)
from src.io_utils import atomic_write_parquet


# 嚴格統一所有年份分區之 02_clean Parquet Schema
TRANSCRIPT_CLEAN_SCHEMA = pa.schema(
    [
        pa.field("event_id", pa.string(), nullable=False),
        pa.field("symbol", pa.string(), nullable=False),
        pa.field("raw_symbol", pa.string(), nullable=False),
        pa.field("gics_sector", pa.string(), nullable=False),
        pa.field("company_name", pa.string(), nullable=True),
        pa.field("year", pa.int32(), nullable=False),
        pa.field("quarter", pa.int8(), nullable=False),
        pa.field("raw_date", pa.string(), nullable=False),
        pa.field("event_datetime_et", pa.string(), nullable=True),
        pa.field("session_type", pa.string(), nullable=False),
        pa.field("alignment_flag", pa.string(), nullable=False),
        pa.field("event_trading_date", pa.string(), nullable=True),
        pa.field("pre_end_date", pa.string(), nullable=True),
        pa.field("post_10d_date", pa.string(), nullable=True),
        pa.field("post_20d_date", pa.string(), nullable=True),
        pa.field("qa_split_method", pa.string(), nullable=False),
        pa.field("prepared_remarks_raw", pa.string(), nullable=False),
        pa.field("qa_session_raw", pa.string(), nullable=False),
        pa.field("prepared_remarks_anon", pa.string(), nullable=False),
        pa.field("qa_session_anon", pa.string(), nullable=False),
        pa.field("prepared_word_count", pa.int32(), nullable=False),
        pa.field("qa_word_count", pa.int32(), nullable=False),
        pa.field("total_word_count", pa.int32(), nullable=False),
        pa.field("n_executives_masked", pa.int16(), nullable=False),
        pa.field("n_analysts_masked", pa.int16(), nullable=False),
        pa.field("is_valid", pa.bool_(), nullable=False),
        pa.field("exclude_reason", pa.string(), nullable=False),
    ]
)


# 預編譯年份與季度正則（保護 $ 金額與千分位數字不被誤遮）
_FISCAL_YEAR_RX = re.compile(
    r"\b(?:fiscal\s+year\s+(?:20\d{2}|19\d{2})|FY\s?['’]?\d{2,4})\b",
    flags=re.IGNORECASE,
)
_YEAR_RX = re.compile(
    r"(?<![\$,\d])(?:20(?:0[5-9]|1\d|2\d|30)|199\d)(?![,\d%])"
)
_QUARTER_RX = re.compile(
    r"\b(?:"
    r"Q[1-4](?:\s?['’]?\d{2,4})?|"
    r"[1-4]Q\s?['’]?\d{2,4}|"
    r"(?:first|second|third|fourth|1st|2nd|3rd|4th)\s+(?:fiscal\s+)?quarter|"
    r"(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+quarter"
    r")\b",
    flags=re.IGNORECASE,
)
_QA_HANDOFF_RXS = [re.compile(p, flags=re.IGNORECASE) for p in CLEAN_CFG.qa_handoff_patterns]
_HEADER_SPEAKER_SET = {s.lower() for s in CLEAN_CFG.header_speaker_labels}

# 避免把常見英文單字當作高管名字誤遮（例如 "Will", "Mark", "Bill", "May", "Grant"）
_AMBIGUOUS_FIRST_NAMES: Set[str] = {
    "will", "mark", "bill", "may", "grant", "rich", "art", "pat", "max",
    "ray", "don", "van", "rob", "bob", "ben", "dan", "jay", "lee", "al",
    "ed", "jo", "bo", "guy", "dean", "wade", "lane", "page", "rose", "june",
    "april", "august", "chip", "cliff", "frank", "jack", "john", "drew",
    "chase", "cash", "price", "west", "north", "south", "east", "long",
}


# ==================== 1. Point-in-Time NYSE 交易日對齊 ====================


@dataclass(frozen=True)
class AlignedEventDates:
    """嚴格 Point-in-Time 對齊後之交易日索引結果。"""

    event_datetime_et: Optional[str]
    session_type: str
    alignment_flag: str
    event_trading_date: Optional[str]
    pre_end_date: Optional[str]
    post_10d_date: Optional[str]
    post_20d_date: Optional[str]


def align_event_to_nyse_calendar(
    raw_date: str,
    trading_dates_np: np.ndarray,
) -> AlignedEventDates:
    """將美東時間法說會時間戳嚴格映射至 NYSE 交易日 t=0 及 t-1 / t+10 / t+20。

    對齊紅線（Zero Future Data Leakage）：
    - 僅美東 06:00–09:29 且當日為 NYSE 開盤日（BMO），t=0 才設為當日。
    - 其餘所有情況（>= 09:30 盤中／盤後、休市日、00:00:00 缺時間、00:01–05:59 異常時段），
      t=0 一律保守映射至嚴格大於當日的下一個 NYSE 交易日。
    """
    raw_str = str(raw_date or "").strip()
    ts = pd.to_datetime(raw_str, errors="coerce")
    if pd.isna(ts):
        return AlignedEventDates(None, "INVALID", "INVALID_DATE", None, None, None, None)

    if ts.tzinfo is None:
        ts_et = ts.tz_localize(DATA_CFG.exchange_tz, nonexistent="shift_forward", ambiguous=True)
    else:
        ts_et = ts.tz_convert(DATA_CFG.exchange_tz)

    event_day = np.datetime64(ts_et.date(), "D")
    n_cal = len(trading_dates_np)
    left_idx = int(np.searchsorted(trading_dates_np, event_day, side="left"))
    is_trading_day = (left_idx < n_cal) and (trading_dates_np[left_idx] == event_day)
    has_time_syntax = ":" in raw_str
    is_midnight_zero = (not has_time_syntax) or (
        ts_et.hour == 0 and ts_et.minute == 0 and ts_et.second == 0
    )
    is_abnormal_predawn = is_midnight_zero or (ts_et.hour < DATA_CFG.bmo_earliest_hour)
    is_bmo_window = (
        not is_abnormal_predawn
        and (
            ts_et.hour < DATA_CFG.market_open_hour
            or (
                ts_et.hour == DATA_CFG.market_open_hour
                and ts_et.minute < DATA_CFG.market_open_minute
            )
        )
    )

    if is_bmo_window and is_trading_day:
        t0_idx = left_idx
        session_type = "BMO"
        alignment_flag = "EXACT_BMO"
    elif not is_trading_day:
        t0_idx = int(np.searchsorted(trading_dates_np, event_day, side="right"))
        session_type = "NON_TRADING_DAY"
        alignment_flag = "NON_TRADING_DAY_NEXT"
    elif is_abnormal_predawn:
        t0_idx = int(np.searchsorted(trading_dates_np, event_day, side="right"))
        session_type = "ABNORMAL_OR_MISSING_TIME"
        alignment_flag = "CONSERVATIVE_ABNORMAL_TIME"
    else:
        t0_idx = int(np.searchsorted(trading_dates_np, event_day, side="right"))
        session_type = "AMC_OR_INTRADAY"
        alignment_flag = "EXACT_AMC_NEXT_DAY"

    def _fmt(idx: int) -> Optional[str]:
        if 0 <= idx < n_cal:
            return str(trading_dates_np[idx])
        return None

    return AlignedEventDates(
        event_datetime_et=ts_et.strftime("%Y-%m-%d %H:%M:%S%z"),
        session_type=session_type,
        alignment_flag=alignment_flag,
        event_trading_date=_fmt(t0_idx),
        pre_end_date=_fmt(t0_idx - 1),
        post_10d_date=_fmt(t0_idx + QUANT_CFG.pead_horizons[0]),
        post_20d_date=_fmt(t0_idx + QUANT_CFG.pead_horizons[1]),
    )


# ==================== 2. 雙區塊切分（Prepared Remarks vs Q&A） ====================


def _is_operator(speaker: str) -> bool:
    return "operator" in speaker.lower()


def _extract_person_name(raw_speaker: str) -> str:
    """從含職稱之 speaker 欄位提取純淨人名（如 'Tim Cook - CEO' -> 'Tim Cook'）。"""
    name = str(raw_speaker or "").strip()
    # 先移除括號職稱或破折號職稱
    name = re.split(r"\s+[-–—]\s+|,\s*", name, maxsplit=1)[0].strip()
    name = re.sub(r"\s*\([^)]*\)", "", name).strip()
    return name


def split_prepared_and_qa(
    structured_content: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], str, List[str], List[str]]:
    """將 structured_content 切分為準備發言與 Q&A，並剝離目錄頁首。

    Returns:
        (prepared_segs, qa_segs, split_method, header_exec_texts, header_analyst_texts)
    """
    body_segs: List[Dict[str, str]] = []
    header_exec_texts: List[str] = []
    header_analyst_texts: List[str] = []

    for seg in structured_content:
        speaker = str(seg.get("speaker", "") or "").strip()
        text = str(seg.get("text", "") or "").strip()
        sp_low = speaker.lower()
        if sp_low in _HEADER_SPEAKER_SET:
            if "analyst" in sp_low:
                header_analyst_texts.append(text)
            elif "exec" in sp_low or "corporate" in sp_low or "company" in sp_low:
                header_exec_texts.append(text)
            continue
        if speaker or text:
            body_segs.append({"speaker": speaker, "text": text})

    if not body_segs:
        return [], [], "empty_body", header_exec_texts, header_analyst_texts

    # Tier 1：在累積管理層發言達門檻後，比對明確 Q&A 進入提示語
    exec_words_so_far = 0
    for idx, seg in enumerate(body_segs):
        sp = seg["speaker"]
        txt = seg["text"]
        wc = len(txt.split())

        if exec_words_so_far >= CLEAN_CFG.min_prepared_words_before_qa:
            if any(rx.search(txt) for rx in _QA_HANDOFF_RXS):
                # 若該段是高管長篇發言並在句尾轉進 Q&A，則保留該段於 Prepared，Q&A 從下一段起算
                if not _is_operator(sp) and wc > 120 and idx + 1 < len(body_segs):
                    return (
                        body_segs[: idx + 1],
                        body_segs[idx + 1 :],
                        "tier1_exec_tail_handoff",
                        header_exec_texts,
                        header_analyst_texts,
                    )
                return (
                    body_segs[:idx],
                    body_segs[idx:],
                    "tier1_handoff_marker",
                    header_exec_texts,
                    header_analyst_texts,
                )

        if not _is_operator(sp):
            exec_words_so_far += wc

    # Tier 2：結構回退 — 管理層已發表至少 350 字後，第一個再次出現的 Operator 段落即為 Q&A 起點
    exec_words_so_far = 0
    for idx, seg in enumerate(body_segs):
        sp = seg["speaker"]
        wc = len(seg["text"].split())
        if _is_operator(sp) and exec_words_so_far >= 350 and idx + 2 < len(body_segs):
            return (
                body_segs[:idx],
                body_segs[idx:],
                "tier2_operator_fallback",
                header_exec_texts,
                header_analyst_texts,
            )
        if not _is_operator(sp):
            exec_words_so_far += wc

    return body_segs, [], "no_qa_detected", header_exec_texts, header_analyst_texts


# ==================== 3. 三層增強去識別化 ====================


def load_redact_dict(path: Path = REDACT_DICT_PATH) -> Dict[str, Any]:
    """載入公司別名與旗艦產品遮蔽字典。"""
    if not path.exists():
        raise FileNotFoundError(f"找不到遮蔽字典：{path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def derive_company_core_terms(
    symbol: str,
    raw_symbol: str,
    company_name: str,
    redact_dict: Dict[str, Any],
) -> Tuple[List[str], List[str]]:
    """彙整目標公司的所有名稱別名（[COMPANY]）與旗艦產品詞（[PRODUCT]）。"""
    entry = redact_dict.get("by_symbol", {}).get(symbol, {})
    aliases: Set[str] = set(entry.get("aliases", []))
    products: Set[str] = set(entry.get("products", []))

    # 從 company_name 自動剝離企業後綴（如 'Apple Inc.' -> 'Apple'）
    clean_co = str(company_name or "").strip()
    if clean_co:
        aliases.add(clean_co)
        stripped = clean_co
        for suffix in redact_dict.get("corporate_suffixes", []):
            stripped = re.sub(rf",?\s*\b{re.escape(suffix)}\b\.?", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"^The\s+", "", stripped.strip(" ,."), flags=re.IGNORECASE).strip(" ,.")
        if len(stripped) >= 3:
            aliases.add(stripped)

    # 僅針對長度 >= 3 之股票代碼直接遮蔽（避免單／雙字母代碼如 C, F, D, O, T, V 誤傷英文單字）
    for sym in (symbol, raw_symbol):
        if sym and len(sym) >= 3:
            aliases.add(sym)

    # 依長度由長至短排序，確保先匹配長詞（例如先替換 'Amazon Web Services' 再替換 'Amazon'）
    sorted_aliases = sorted((a.strip() for a in aliases if len(a.strip()) >= 2), key=len, reverse=True)
    sorted_products = sorted((p.strip() for p in products if len(p.strip()) >= 2), key=len, reverse=True)
    return sorted_aliases, sorted_products


def identify_speaker_roles(
    prepared_segs: Sequence[Dict[str, str]],
    qa_segs: Sequence[Dict[str, str]],
) -> Tuple[Set[str], Set[str]]:
    """根據出現區段與 Q&A 發言次數區分管理層（executive）與分析師（analyst）名單。"""
    exec_names: Set[str] = set()
    for seg in prepared_segs:
        sp = seg["speaker"]
        if sp and not _is_operator(sp):
            name = _extract_person_name(sp)
            if len(name) >= 3:
                exec_names.add(name)

    qa_counts: Counter[str] = Counter()
    for seg in qa_segs:
        sp = seg["speaker"]
        if sp and not _is_operator(sp):
            name = _extract_person_name(sp)
            if len(name) >= 3:
                qa_counts[name] += 1

    # 若有人只在 Q&A 出現卻連續發言 >= 4 次，極高機率是專門答題的高管（如 CFO / COO）
    analyst_names: Set[str] = set()
    for name, cnt in qa_counts.items():
        if name in exec_names:
            continue
        if cnt >= 4:
            exec_names.add(name)
        else:
            analyst_names.add(name)

    return exec_names, analyst_names


def _compile_word_boundary_pattern(terms: Sequence[str]) -> Optional[re.Pattern[str]]:
    if not terms:
        return None
    parts = []
    for t in terms:
        esc = re.escape(t)
        # 首尾為英數字時加上 \b 邊界，含符號（如 S&P, AT&T, Disney+）時用前後非英數邊界
        left = r"\b" if t[0].isalnum() else r"(?<!\w)"
        right = r"\b" if t[-1].isalnum() else r"(?!\w)"
        parts.append(f"{left}{esc}{right}")
    return re.compile("|".join(parts), flags=re.IGNORECASE)


def deidentify_text(
    text: str,
    company_terms: Sequence[str],
    product_terms: Sequence[str],
    exec_names: Iterable[str],
    analyst_names: Iterable[str],
) -> str:
    """對法說會文本執行三層確定性去識別化。"""
    if not text:
        return ""

    out = text

    # 1. 產品詞優先於公司名（避免 'Apple Watch' 先被替換成 '[COMPANY] Watch'）
    prod_rx = _compile_word_boundary_pattern(product_terms)
    if prod_rx is not None:
        out = prod_rx.sub("[PRODUCT]", out)

    comp_rx = _compile_word_boundary_pattern(company_terms)
    if comp_rx is not None:
        out = comp_rx.sub("[COMPANY]", out)

    # 2. 高管與分析師全名 + 高管尊稱姓氏 + 直接稱呼名字（如 'Thank you, Tim' / 'Hi Luca'）
    sorted_execs = sorted((n for n in exec_names if len(n) >= 3), key=len, reverse=True)
    exec_full_rx = _compile_word_boundary_pattern(sorted_execs)
    if exec_full_rx is not None:
        out = exec_full_rx.sub("[EXECUTIVE]", out)

    first_names: Set[str] = set()
    last_names: Set[str] = set()
    for full in sorted_execs:
        tokens = [t.strip(".") for t in full.split() if len(t.strip(".")) >= 3]
        if len(tokens) >= 2:
            fn, ln = tokens[0], tokens[-1]
            if fn.lower() not in _AMBIGUOUS_FIRST_NAMES:
                first_names.add(fn)
            if ln.lower() not in _AMBIGUOUS_FIRST_NAMES:
                last_names.add(ln)

    if last_names:
        ln_alt = "|".join(re.escape(n) for n in sorted(last_names, key=len, reverse=True))
        out = re.sub(rf"\b(?:Mr|Ms|Mrs|Dr)\.?\s+(?:{ln_alt})\b", "[EXECUTIVE]", out, flags=re.IGNORECASE)

    if first_names:
        fn_alt = "|".join(re.escape(n) for n in sorted(first_names, key=len, reverse=True))
        out = re.sub(
            rf"\b(thank you|thanks|hi|hello|hey|good morning|good afternoon|over to you),?\s+(?:{fn_alt})\b",
            r"\1, [EXECUTIVE]",
            out,
            flags=re.IGNORECASE,
        )

    sorted_analysts = sorted((n for n in analyst_names if len(n) >= 3), key=len, reverse=True)
    analyst_rx = _compile_word_boundary_pattern(sorted_analysts)
    if analyst_rx is not None:
        out = analyst_rx.sub("[ANALYST]", out)

    # 3. 財年、年份與季度（保留財務金額 $2024 / 2,024）
    out = _FISCAL_YEAR_RX.sub("[FISCAL_YEAR]", out)
    out = _QUARTER_RX.sub("[CURRENT_QUARTER]", out)
    out = _YEAR_RX.sub("[YEAR]", out)

    return out


def format_section_texts(
    segs: Sequence[Dict[str, str]],
    company_terms: Sequence[str],
    product_terms: Sequence[str],
    exec_names: Set[str],
    analyst_names: Set[str],
) -> Tuple[str, str]:
    """將段落清單格式化為原文版與去識別化版字串。"""
    if not segs:
        return "", ""

    raw_parts: List[str] = []
    anon_parts: List[str] = []
    for seg in segs:
        sp = seg["speaker"]
        txt = seg["text"]
        raw_parts.append(f"{sp}: {txt}" if sp else txt)

        if _is_operator(sp):
            role_tag = "[OPERATOR]"
        else:
            person = _extract_person_name(sp)
            role_tag = "[EXECUTIVE]" if person in exec_names else "[ANALYST]"

        anon_body = deidentify_text(txt, company_terms, product_terms, exec_names, analyst_names)
        anon_parts.append(f"{role_tag}: {anon_body}")

    return "\n\n".join(raw_parts), "\n\n".join(anon_parts)


def evaluate_record_validity(
    aligned: AlignedEventDates,
    prepared_wc: int,
    qa_wc: int,
    total_wc: int,
) -> Tuple[bool, str]:
    """執行軟刪除品質檢核，回傳 (is_valid, exclude_reason)。"""
    reasons: List[str] = []
    if aligned.event_trading_date is None:
        reasons.append("invalid_or_out_of_calendar_date")
    if aligned.post_20d_date is None:
        reasons.append("missing_post_20d_trading_date")
    if total_wc < DATA_CFG.min_word_count:
        reasons.append(f"short_transcript_lt_{DATA_CFG.min_word_count}")
    if prepared_wc < 100:
        reasons.append("missing_prepared_remarks")
    if qa_wc < CLEAN_CFG.min_qa_words:
        reasons.append(f"missing_or_short_qa_lt_{CLEAN_CFG.min_qa_words}")

    return (len(reasons) == 0, ";".join(reasons))


# ==================== 主執行管線與 10 筆人工抽檢報告 ====================


def run_clean_and_align(
    raw_transcripts_dir: Path = RAW_TRANSCRIPTS_DIR,
    calendar_path: Path = RAW_NYSE_CALENDAR_PATH,
    clean_out_dir: Path = CLEAN_TRANSCRIPTS_DIR,
    audit_md_path: Path = STAGE2_AUDIT_MD,
) -> Dict[str, Any]:
    """逐年讀取 01_raw 逐字稿，執行切分、去識別化、Point-in-Time 對齊並寫入 02_clean。"""
    ensure_project_dirs()
    uni = load_universe()
    redact_dict = load_redact_dict()

    # 建立 symbol -> GICS sector 對照表
    symbol_to_sector: Dict[str, str] = {}
    for sector, syms in uni["sectors"].items():
        for s in syms:
            symbol_to_sector[s] = sector

    # 載入 NYSE 交易日曆為排序 numpy datetime64[D]
    cal_df = pd.read_parquet(calendar_path)
    trading_dates_np = cal_df["date"].to_numpy(dtype="datetime64[D]")
    trading_dates_np.sort()

    year_files = sorted(raw_transcripts_dir.glob("year=*/part.parquet"))
    if not year_files:
        raise FileNotFoundError(f"在 {raw_transcripts_dir} 找不到任何年份分區檔，請先執行 Stage 1。")

    print(f"[Stage 2] 開始處理 {len(year_files)} 個年份分區...", flush=True)

    total_rows = 0
    valid_rows = 0
    split_method_counter: Counter[str] = Counter()
    alignment_flag_counter: Counter[str] = Counter()
    exclude_reason_counter: Counter[str] = Counter()
    company_leak_count = 0
    seen_event_ids: Set[str] = set()
    audit_pool: List[Dict[str, Any]] = []

    for yf_path in year_files:
        raw_table = pq.read_table(yf_path)
        raw_records = raw_table.to_pylist()
        yr = int(raw_records[0]["year"])
        clean_rows: List[Dict[str, Any]] = []

        for rec in raw_records:
            eid = str(rec["event_id"])
            assert eid not in seen_event_ids, f"偵測到跨分區重複 event_id：{eid}"
            seen_event_ids.add(eid)

            sym = str(rec["symbol"])
            raw_sym = str(rec["raw_symbol"])
            co_name = str(rec.get("company_name", "") or "")
            raw_date = str(rec.get("raw_date", "") or "")

            aligned = align_event_to_nyse_calendar(raw_date, trading_dates_np)
            prep_segs, qa_segs, split_method, _, _ = split_prepared_and_qa(rec["structured_content"])
            exec_names, analyst_names = identify_speaker_roles(prep_segs, qa_segs)
            company_terms, product_terms = derive_company_core_terms(sym, raw_sym, co_name, redact_dict)

            prep_raw, prep_anon = format_section_texts(
                prep_segs, company_terms, product_terms, exec_names, analyst_names
            )
            qa_raw, qa_anon = format_section_texts(
                qa_segs, company_terms, product_terms, exec_names, analyst_names
            )

            prep_wc = len(prep_raw.split())
            qa_wc = len(qa_raw.split())
            total_wc = prep_wc + qa_wc
            is_valid, exclude_reason = evaluate_record_validity(aligned, prep_wc, qa_wc, total_wc)

            # 檢查核心公司別名是否仍有殘留（去識別化品質稽核）
            primary_alias = company_terms[0] if company_terms else ""
            if is_valid and primary_alias and len(primary_alias) >= 4:
                if re.search(rf"\b{re.escape(primary_alias)}\b", prep_anon + " " + qa_anon, flags=re.IGNORECASE):
                    company_leak_count += 1

            clean_row = {
                "event_id": eid,
                "symbol": sym,
                "raw_symbol": raw_sym,
                "gics_sector": symbol_to_sector.get(sym, "Unknown"),
                "company_name": co_name,
                "year": int(rec["year"]),
                "quarter": int(rec["quarter"]),
                "raw_date": raw_date,
                "event_datetime_et": aligned.event_datetime_et,
                "session_type": aligned.session_type,
                "alignment_flag": aligned.alignment_flag,
                "event_trading_date": aligned.event_trading_date,
                "pre_end_date": aligned.pre_end_date,
                "post_10d_date": aligned.post_10d_date,
                "post_20d_date": aligned.post_20d_date,
                "qa_split_method": split_method,
                "prepared_remarks_raw": prep_raw,
                "qa_session_raw": qa_raw,
                "prepared_remarks_anon": prep_anon,
                "qa_session_anon": qa_anon,
                "prepared_word_count": prep_wc,
                "qa_word_count": qa_wc,
                "total_word_count": total_wc,
                "n_executives_masked": len(exec_names),
                "n_analysts_masked": len(analyst_names),
                "is_valid": is_valid,
                "exclude_reason": exclude_reason,
            }
            clean_rows.append(clean_row)

            total_rows += 1
            if is_valid:
                valid_rows += 1
            else:
                exclude_reason_counter[exclude_reason] += 1
            split_method_counter[split_method] += 1
            alignment_flag_counter[aligned.alignment_flag] += 1

            # 收集輕量抽檢池（只存前 240 字片段以節省記憶體）
            if is_valid:
                audit_pool.append(
                    {
                        "event_id": eid,
                        "symbol": sym,
                        "gics_sector": clean_row["gics_sector"],
                        "year": yr,
                        "raw_date": raw_date,
                        "session_type": aligned.session_type,
                        "alignment_flag": aligned.alignment_flag,
                        "pre_end_date": aligned.pre_end_date,
                        "event_trading_date": aligned.event_trading_date,
                        "post_20d_date": aligned.post_20d_date,
                        "qa_split_method": split_method,
                        "prepared_wc": prep_wc,
                        "qa_wc": qa_wc,
                        "prep_raw_snippet": prep_raw[:220].replace("\n", " "),
                        "prep_anon_snippet": prep_anon[:220].replace("\n", " "),
                        "qa_start_snippet": qa_raw[:180].replace("\n", " "),
                    }
                )

        out_table = pa.Table.from_pylist(clean_rows, schema=TRANSCRIPT_CLEAN_SCHEMA)
        out_path = clean_out_dir / f"year={yr}" / "part.parquet"
        atomic_write_parquet(out_table, out_path)
        print(
            f"  - year={yr}: 寫入 {len(clean_rows):,} 筆（有效 {sum(r['is_valid'] for r in clean_rows):,} 筆）",
            flush=True,
        )

    total_size_mb = sum(
        p.stat().st_size for p in clean_out_dir.rglob("*.parquet") if p.is_file()
    ) / (1024 * 1024)

    summary = {
        "total_rows": total_rows,
        "valid_rows": valid_rows,
        "valid_rate_pct": round(valid_rows / max(total_rows, 1) * 100.0, 2),
        "clean_size_mb": round(total_size_mb, 2),
        "company_alias_residual_count": company_leak_count,
        "split_methods": dict(split_method_counter),
        "alignment_flags": dict(alignment_flag_counter),
        "exclude_reasons": dict(exclude_reason_counter),
    }

    _write_stage2_audit_md(summary, audit_pool, audit_md_path)
    print(
        f"[Stage 2] 完成！共 {total_rows:,} 筆，有效 {valid_rows:,} 筆（{summary['valid_rate_pct']:.2f}%），"
        f"磁碟 {total_size_mb:.2f} MB。稽核報告：{audit_md_path}",
        flush=True,
    )
    return summary


def _write_stage2_audit_md(
    summary: Dict[str, Any],
    audit_pool: Sequence[Dict[str, Any]],
    out_path: Path,
) -> None:
    """輸出 Stage 2 品質統計與 10 筆分層人工抽檢對照明細。"""
    rng = np.random.default_rng(QUANT_CFG.random_seed)
    pool_df = pd.DataFrame(audit_pool)

    # 依 GICS 板塊分層抽樣 10 筆（確保跨板塊與跨年份代表性）
    sampled_rows: List[Dict[str, Any]] = []
    if not pool_df.empty:
        sectors = sorted(pool_df["gics_sector"].unique())
        rng.shuffle(sectors)
        for sec in sectors[: CLEAN_CFG.audit_sample_size]:
            sec_df = pool_df[pool_df["gics_sector"] == sec]
            pick_idx = int(rng.integers(0, len(sec_df)))
            sampled_rows.append(sec_df.iloc[pick_idx].to_dict())

    lines = [
        "# Stage 2 清洗、去識別化與 Point-in-Time 對齊稽核報告（02_clean）",
        "",
        f"- **產出時間**：{datetime.now().isoformat(timespec='seconds')}",
        f"- **總筆數（保留全量軟刪除軌跡）**：**{summary['total_rows']:,}** 筆",
        f"- **有效樣本數（`is_valid = True`）**：**{summary['valid_rows']:,}** 筆（**{summary['valid_rate_pct']:.2f}%**）",
        f"- **02_clean 總磁碟占用**：**{summary['clean_size_mb']:.2f} MB**",
        f"- **核心公司別名殘留筆數（稽核指標）**：**{summary['company_alias_residual_count']}** 筆",
        "",
        "## 1. Q&A 雙區塊切分方式分布",
        "| 切分方式 | 筆數 | 占比 |",
        "|---|---:|---:|",
    ]
    for method, cnt in sorted(summary["split_methods"].items(), key=lambda x: -x[1]):
        lines.append(f"| `{method}` | {cnt:,} | {cnt / max(summary['total_rows'], 1) * 100:.2f}% |")

    lines.extend(
        [
            "",
            "## 2. Point-in-Time 交易日對齊旗標分布",
            "| Alignment Flag | 筆數 | 占比 |",
            "|---|---:|---:|",
        ]
    )
    for flag, cnt in sorted(summary["alignment_flags"].items(), key=lambda x: -x[1]):
        lines.append(f"| `{flag}` | {cnt:,} | {cnt / max(summary['total_rows'], 1) * 100:.2f}% |")

    lines.extend(
        [
            "",
            "## 3. 軟刪除排除原因統計（`is_valid = False`）",
            "| 排除原因 | 筆數 |",
            "|---|---:|",
        ]
    )
    if summary["exclude_reasons"]:
        for reason, cnt in sorted(summary["exclude_reasons"].items(), key=lambda x: -x[1]):
            lines.append(f"| `{reason}` | {cnt:,} |")
    else:
        lines.append("| （無排除樣本） | 0 |")

    lines.extend(
        [
            "",
            "## 4. 10 筆跨板塊人工抽檢明細（Stage 2 Verification Gate）",
            "",
            "| # | Event ID | Sector | Raw Date (ET) | Flag | $t-1$ | $t=0$ | $t+20$ | Prep/Q&A 字數 |",
            "|---:|---|---|---|---|---|---|---|---|",
        ]
    )
    for i, s in enumerate(sampled_rows, start=1):
        lines.append(
            f"| {i} | `{s['event_id']}` | {s['gics_sector']} | `{s['raw_date']}` | "
            f"`{s['alignment_flag']}` | `{s['pre_end_date']}` | **`{s['event_trading_date']}`** | "
            f"`{s['post_20d_date']}` | {s['prepared_wc']:,} / {s['qa_wc']:,} |"
        )

    lines.append("")
    lines.append("### 抽檢樣本去識別化與 Q&A 切分起點對照")
    for i, s in enumerate(sampled_rows, start=1):
        lines.extend(
            [
                f"#### {i}. `{s['event_id']}`（{s['gics_sector']}｜{s['qa_split_method']}）",
                f"- **Prepared 原文片段**：`{s['prep_raw_snippet']}...`",
                f"- **Prepared 去識別化**：`{s['prep_anon_snippet']}...`",
                f"- **Q&A 切分起點**：`{s['qa_start_snippet']}...`",
                "",
            ]
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    run_clean_and_align()
