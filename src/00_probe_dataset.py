"""Stage 0：Hugging Face 法說會資料集品質前置探勘腳本。

功能：
1. 串流抽檢 kurry/sp500_earnings_transcripts 欄位 Schema、structured_content 結構與 Q&A 切分標記。
2. 利用 Parquet 欄式裁剪（Column Pruning）高速掃描全量元資料（symbol, year, quarter, date），
   統計各年度時間戳時分秒完整度、盤前（BMO）／盤後（AMC）錨點股時間分布以判定真實時區。
3. 統計 2016 年起各標的季度覆蓋率，輸出結構化探勘報告與候選股票池統計表，供定案 config/universe.json。

執行範例：
    python src/00_probe_dataset.py
    python src/00_probe_dataset.py --schema-samples 100
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
from dotenv import load_dotenv

from src.config import (
    DATA_CFG,
    DEFAULT_RENAME_MAP,
    PROBE_CFG,
    ensure_project_dirs,
)


# ==================== 純函式核心邏輯（易於單元測試） ====================


@dataclass(frozen=True)
class ParsedTimestamp:
    """原始時間戳解析結果。"""

    raw: str
    valid: bool
    has_time_component: bool
    is_midnight_zero: bool
    hour: Optional[int]
    minute: Optional[int]
    tz_info: Optional[str]


def normalize_symbol(symbol: str, rename_map: Dict[str, str] = DEFAULT_RENAME_MAP) -> str:
    """標準化股票代碼並套用歷史更名映射。"""
    cleaned = str(symbol or "").strip().upper()
    return rename_map.get(cleaned, cleaned)


def parse_raw_timestamp(raw_value: Any) -> ParsedTimestamp:
    """解析逐字稿 date 欄位，判斷是否包含有效時分秒與時區資訊。"""
    raw_str = "" if raw_value is None else str(raw_value).strip()
    if not raw_str or raw_str.lower() in {"none", "nan", "nat"}:
        return ParsedTimestamp(
            raw=raw_str,
            valid=False,
            has_time_component=False,
            is_midnight_zero=False,
            hour=None,
            minute=None,
            tz_info=None,
        )

    has_time_syntax = bool(re.search(r"\d{1,2}:\d{2}", raw_str))
    try:
        ts = pd.to_datetime(raw_str, errors="raise")
    except Exception:
        return ParsedTimestamp(
            raw=raw_str,
            valid=False,
            has_time_component=has_time_syntax,
            is_midnight_zero=False,
            hour=None,
            minute=None,
            tz_info=None,
        )

    if pd.isna(ts):
        return ParsedTimestamp(
            raw=raw_str,
            valid=False,
            has_time_component=False,
            is_midnight_zero=False,
            hour=None,
            minute=None,
            tz_info=None,
        )

    is_midnight = (ts.hour == 0 and ts.minute == 0 and ts.second == 0)
    tz_str = str(ts.tzinfo) if ts.tzinfo is not None else None
    return ParsedTimestamp(
        raw=raw_str,
        valid=True,
        has_time_component=has_time_syntax and not is_midnight,
        is_midnight_zero=has_time_syntax and is_midnight,
        hour=int(ts.hour),
        minute=int(ts.minute),
        tz_info=tz_str,
    )


def detect_qa_split_index(
    structured_content: Any,
    patterns: Sequence[str] = PROBE_CFG.qa_marker_patterns,
    min_segment_index: int = 2,
) -> Optional[int]:
    """在 structured_content 發言陣列中偵測 Q&A 環節起點索引。

    略過開場前兩段（min_segment_index=2），避免誤判 Operator 開場白中
    「After the speakers' remarks, there will be a question-and-answer session」。

    Args:
        structured_content: 發言段落清單（每筆為 dict，含 speaker 與 text/content）。
        patterns: 用於匹配 Q&A 進入提示語的正則清單。
        min_segment_index: 開始偵測 Q&A 的最小段落索引（預設 2）。

    Returns:
        命中之段落索引；若無法偵測則回傳 None。
    """
    if not isinstance(structured_content, (list, tuple)) or len(structured_content) <= min_segment_index:
        return None

    compiled = [re.compile(p, flags=re.IGNORECASE) for p in patterns]
    for idx in range(min_segment_index, len(structured_content)):
        seg = structured_content[idx]
        if not isinstance(seg, dict):
            continue
        text = str(seg.get("text", seg.get("content", "")) or "")
        if any(rx.search(text) for rx in compiled):
            return idx
    return None


def estimate_structured_word_count(structured_content: Any) -> int:
    """估算 structured_content 總英文單字數。"""
    if not isinstance(structured_content, (list, tuple)):
        return 0
    total = 0
    for seg in structured_content:
        if isinstance(seg, dict):
            text = str(seg.get("text", seg.get("content", "")) or "")
            total += len(text.split())
    return total


def infer_timezone_hypothesis(
    amc_hours: Sequence[int],
    bmo_hours: Sequence[int],
) -> str:
    """根據固定盤後（AMC）與盤前（BMO）錨點股之發布小時中位數推斷原始時區。"""
    if not amc_hours and not bmo_hours:
        return "UNKNOWN (缺乏有效時分秒樣本)"

    amc_med = float(pd.Series(amc_hours).median()) if amc_hours else None
    bmo_med = float(pd.Series(bmo_hours).median()) if bmo_hours else None

    # 美東時間：AMC 通常 16:00-18:00，BMO 通常 07:00-09:30
    # UTC 時間：AMC 通常 20:00-23:00，BMO 通常 11:00-14:30
    if amc_med is not None and 15.0 <= amc_med <= 19.0:
        if bmo_med is None or 6.0 <= bmo_med <= 10.5:
            return "America/New_York (美東時間，AMC 中位數約 16-18 時、BMO 約 7-10 時)"
    if amc_med is not None and (20.0 <= amc_med <= 23.5 or amc_med <= 1.0):
        if bmo_med is None or 11.0 <= bmo_med <= 15.0:
            return "UTC (協調世界時，AMC 中位數約 20-23 時、BMO 約 11-14 時)"

    return f"AMBIGUOUS (需人工複核：AMC 小時中位數={amc_med}, BMO 小時中位數={bmo_med})"


def summarize_symbol_coverage(
    records: Iterable[Dict[str, Any]],
    start_year: int = DATA_CFG.start_year,
    rename_map: Dict[str, str] = DEFAULT_RENAME_MAP,
) -> pd.DataFrame:
    """彙整各標的在 start_year 之後的季度覆蓋率與時間戳品質統計。"""
    per_symbol_quarters: Dict[str, set[Tuple[int, int]]] = defaultdict(set)
    per_symbol_years: Dict[str, List[int]] = defaultdict(list)
    per_symbol_names: Dict[str, str] = {}
    per_symbol_time_ok: Counter[str] = Counter()
    per_symbol_total: Counter[str] = Counter()

    all_years: List[int] = []
    for rec in records:
        year = rec.get("year")
        quarter = rec.get("quarter")
        symbol = normalize_symbol(rec.get("symbol", ""), rename_map)
        if not symbol or year is None or quarter is None:
            continue
        year_int = int(year)
        quarter_int = int(quarter)
        if year_int < start_year:
            continue

        all_years.append(year_int)
        per_symbol_quarters[symbol].add((year_int, quarter_int))
        per_symbol_years[symbol].append(year_int)
        per_symbol_total[symbol] += 1
        if rec.get("company_name") and symbol not in per_symbol_names:
            per_symbol_names[symbol] = str(rec["company_name"])

        parsed = parse_raw_timestamp(rec.get("date"))
        if parsed.has_time_component:
            per_symbol_time_ok[symbol] += 1

    if not all_years:
        return pd.DataFrame(
            columns=[
                "symbol",
                "company_name",
                "total_records",
                "unique_quarters",
                "min_year",
                "max_year",
                "completeness_ratio",
                "non_zero_time_ratio",
            ]
        )

    max_year_global = max(all_years)
    # 估算主庫涵蓋總季數（以資料內實際最大年度的總季度數為基準）
    global_quarters = len({q for qs in per_symbol_quarters.values() for q in qs})
    denom_quarters = max(global_quarters, 1)

    rows = []
    for sym, qset in per_symbol_quarters.items():
        total_cnt = per_symbol_total[sym]
        rows.append(
            {
                "symbol": sym,
                "company_name": per_symbol_names.get(sym, ""),
                "total_records": total_cnt,
                "unique_quarters": len(qset),
                "min_year": min(per_symbol_years[sym]),
                "max_year": max(per_symbol_years[sym]),
                "completeness_ratio": round(len(qset) / denom_quarters, 4),
                "non_zero_time_ratio": round(per_symbol_time_ok[sym] / max(total_cnt, 1), 4),
            }
        )

    df = pd.DataFrame(rows).sort_values(
        by=["unique_quarters", "non_zero_time_ratio", "symbol"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    df.attrs["max_year_global"] = max_year_global
    df.attrs["global_unique_quarters"] = denom_quarters
    return df


# ==================== 線上資料集串流探勘流程 ====================


def run_probe(schema_samples: int = PROBE_CFG.schema_sample_limit) -> Dict[str, Any]:
    """執行 Stage 0 完整探勘，並輸出 JSON、Markdown 報告與股票覆蓋率 CSV。"""
    from datasets import load_dataset

    load_dotenv()
    ensure_project_dirs()
    hf_token = os.getenv("HF_TOKEN") or None

    print(f"[Stage 0] 開始串流探勘 Hugging Face 資料集：{DATA_CFG.hf_dataset_id}", flush=True)
    print(f"[Stage 0] Pass 1：抽檢前 {schema_samples} 筆完整紀錄以驗證 Schema 與 Q&A 切分結構...", flush=True)

    ds_sample = load_dataset(
        DATA_CFG.hf_dataset_id,
        split=DATA_CFG.hf_split,
        streaming=True,
        token=hf_token,
    )

    columns_seen: List[str] = []
    structured_keys: Counter[str] = Counter()
    qa_detected_count = 0
    word_counts: List[int] = []
    sample_examples: List[Dict[str, Any]] = []

    for idx, row in enumerate(ds_sample):
        if idx == 0:
            columns_seen = list(row.keys())
        if idx >= schema_samples:
            break

        sc = row.get("structured_content")
        if isinstance(sc, list) and sc:
            for seg in sc[:5]:
                if isinstance(seg, dict):
                    structured_keys.update(seg.keys())

        qa_idx = detect_qa_split_index(sc)
        if qa_idx is not None:
            qa_detected_count += 1
        wc = estimate_structured_word_count(sc)
        word_counts.append(wc)

        if idx < 5:
            sample_examples.append(
                {
                    "symbol": row.get("symbol"),
                    "year": row.get("year"),
                    "quarter": row.get("quarter"),
                    "date": str(row.get("date")),
                    "n_segments": len(sc) if isinstance(sc, list) else 0,
                    "qa_split_index": qa_idx,
                    "est_words": wc,
                }
            )

    assert columns_seen, "無法從 Hugging Face 讀取任何資料列，請檢查網路或資料集名稱。"
    required_cols = {"symbol", "year", "quarter", "date", "structured_content"}
    missing_cols = required_cols - set(columns_seen)
    assert not missing_cols, f"資料集缺少關鍵欄位：{missing_cols}，實際欄位：{columns_seen}"

    # Pass 2：利用 Parquet Column Pruning 僅讀取輕量元資料欄位掃描全量紀錄
    meta_cols = [c for c in ("symbol", "company_name", "year", "quarter", "date") if c in columns_seen]
    print(f"[Stage 0] Pass 1 完成（Q&A 命中 {qa_detected_count}/{schema_samples}）。", flush=True)
    print(f"[Stage 0] Pass 2：啟用欄式裁剪讀取全量元資料欄位 {meta_cols}...", flush=True)

    ds_meta = load_dataset(
        DATA_CFG.hf_dataset_id,
        split=DATA_CFG.hf_split,
        streaming=True,
        columns=meta_cols,
        token=hf_token,
    )

    total_rows_all_years = 0
    rows_2016_plus: List[Dict[str, Any]] = []
    yearly_time_stats: Dict[int, Counter[str]] = defaultdict(Counter)
    amc_hours: List[int] = []
    bmo_hours: List[int] = []
    anchor_samples: Dict[str, List[str]] = defaultdict(list)

    amc_set = set(PROBE_CFG.amc_anchor_tickers)
    bmo_set = set(PROBE_CFG.bmo_anchor_tickers)

    for row in ds_meta:
        total_rows_all_years += 1
        year_raw = row.get("year")
        if year_raw is None:
            continue
        year_int = int(year_raw)
        if year_int < DATA_CFG.start_year:
            continue

        rows_2016_plus.append(row)
        parsed = parse_raw_timestamp(row.get("date"))
        yearly_time_stats[year_int]["total"] += 1
        if parsed.valid:
            yearly_time_stats[year_int]["valid"] += 1
        if parsed.has_time_component:
            yearly_time_stats[year_int]["non_zero_time"] += 1
        elif parsed.is_midnight_zero:
            yearly_time_stats[year_int]["midnight_zero"] += 1
        else:
            yearly_time_stats[year_int]["date_only_or_missing"] += 1
        if parsed.tz_info:
            yearly_time_stats[year_int]["has_tz"] += 1

        sym = normalize_symbol(row.get("symbol", ""))
        if sym in amc_set or sym in bmo_set:
            if len(anchor_samples[sym]) < 5:
                anchor_samples[sym].append(f"{year_int}Q{row.get('quarter')}: {parsed.raw}")
            if parsed.has_time_component and parsed.hour is not None:
                if sym in amc_set:
                    amc_hours.append(parsed.hour)
                else:
                    bmo_hours.append(parsed.hour)

    assert rows_2016_plus, f"在 year >= {DATA_CFG.start_year} 區間內查無任何紀錄。"

    coverage_df = summarize_symbol_coverage(rows_2016_plus, start_year=DATA_CFG.start_year)
    coverage_df.to_csv(PROBE_CFG.symbol_coverage_csv, index=False, encoding="utf-8-sig")

    tz_hypothesis = infer_timezone_hypothesis(amc_hours, bmo_hours)
    eligible_90 = int((coverage_df["completeness_ratio"] >= DATA_CFG.min_quarter_completeness).sum())
    eligible_85 = int((coverage_df["completeness_ratio"] >= 0.85).sum())

    yearly_summary = {
        str(yr): {
            "total": int(cnt["total"]),
            "non_zero_time": int(cnt["non_zero_time"]),
            "non_zero_time_pct": round(cnt["non_zero_time"] / max(cnt["total"], 1) * 100, 2),
            "midnight_zero": int(cnt["midnight_zero"]),
            "date_only_or_missing": int(cnt["date_only_or_missing"]),
            "has_explicit_tz": int(cnt["has_tz"]),
        }
        for yr, cnt in sorted(yearly_time_stats.items())
    }

    report: Dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_id": DATA_CFG.hf_dataset_id,
        "columns": columns_seen,
        "structured_content_keys": dict(structured_keys),
        "schema_sample_count": schema_samples,
        "qa_detection_rate_pct": round(qa_detected_count / max(schema_samples, 1) * 100, 2),
        "word_count_stats": {
            "min": int(min(word_counts)) if word_counts else 0,
            "median": int(pd.Series(word_counts).median()) if word_counts else 0,
            "mean": round(float(pd.Series(word_counts).mean()), 1) if word_counts else 0.0,
            "below_1500_count": int(sum(w < DATA_CFG.min_word_count for w in word_counts)),
        },
        "total_rows_all_years": total_rows_all_years,
        "total_rows_since_2016": len(rows_2016_plus),
        "unique_symbols_since_2016": int(len(coverage_df)),
        "global_unique_quarters_since_2016": int(coverage_df.attrs.get("global_unique_quarters", 0)),
        "symbols_with_completeness_ge_90pct": eligible_90,
        "symbols_with_completeness_ge_85pct": eligible_85,
        "timezone_inference": {
            "hypothesis": tz_hypothesis,
            "amc_anchor_hour_distribution": dict(sorted(Counter(amc_hours).items())),
            "bmo_anchor_hour_distribution": dict(sorted(Counter(bmo_hours).items())),
            "anchor_raw_samples": dict(anchor_samples),
        },
        "yearly_timestamp_stats": yearly_summary,
        "sample_records": sample_examples,
    }

    with PROBE_CFG.probe_report_json.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    _write_markdown_summary(report, coverage_df, PROBE_CFG.probe_report_md)
    print(f"[Stage 0] 探勘完成！報告已輸出至：")
    print(f"  - JSON：{PROBE_CFG.probe_report_json}")
    print(f"  - Markdown：{PROBE_CFG.probe_report_md}")
    print(f"  - 標的覆蓋率 CSV：{PROBE_CFG.symbol_coverage_csv}")
    return report


def _write_markdown_summary(report: Dict[str, Any], coverage_df: pd.DataFrame, out_path: Path) -> None:
    """輸出人類易讀之 Stage 0 探勘摘要報告。"""
    lines = [
        "# Stage 0 資料品質前置探勘報告",
        "",
        f"- **產出時間**：{report['generated_at']}",
        f"- **資料集**：`{report['dataset_id']}`",
        f"- **全量筆數（全部年份）**：{report['total_rows_all_years']:,}",
        f"- **2016 年起總筆數**：{report['total_rows_since_2016']:,}",
        f"- **2016 年起涵蓋標的數（更名合併後）**：{report['unique_symbols_since_2016']:,}",
        f"- **2016 年起總季度數**：{report['global_unique_quarters_since_2016']}",
        f"- **季度完整度 ≥ 90% 標的數**：**{report['symbols_with_completeness_ge_90pct']}** 檔（≥ 85%：{report['symbols_with_completeness_ge_85pct']} 檔）",
        "",
        "## 1. Schema 與 Q&A 切分抽檢（前 "
        + str(report["schema_sample_count"])
        + " 筆）",
        f"- **原始欄位**：`{', '.join(report['columns'])}`",
        f"- **structured_content 元素鍵值**：`{report['structured_content_keys']}`",
        f"- **Q&A 標記規則命中率**：{report['qa_detection_rate_pct']}%",
        f"- **字數統計**：中位數 {report['word_count_stats']['median']:,} 字、"
        f"平均 {report['word_count_stats']['mean']:,.1f} 字、"
        f"低於 {DATA_CFG.min_word_count} 字共 {report['word_count_stats']['below_1500_count']} 筆",
        "",
        "## 2. 時間戳與時區判讀結果",
        f"- **推斷時區**：{report['timezone_inference']['hypothesis']}",
        f"- **AMC 錨點股小時分布**：`{report['timezone_inference']['amc_anchor_hour_distribution']}`",
        f"- **BMO 錨點股小時分布**：`{report['timezone_inference']['bmo_anchor_hour_distribution']}`",
        "",
        "### 各年度時間戳品質統計（2016 年起）",
        "| 年份 | 總筆數 | 含非零時分秒 | 比例 (%) | 00:00:00 | 純日期/缺失 | 含顯式時區 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for yr, st in report["yearly_timestamp_stats"].items():
        lines.append(
            f"| {yr} | {st['total']} | {st['non_zero_time']} | {st['non_zero_time_pct']:.1f}% "
            f"| {st['midnight_zero']} | {st['date_only_or_missing']} | {st['has_explicit_tz']} |"
        )

    lines.extend(
        [
            "",
            "### 錨點股原始時間戳範例",
        ]
    )
    for sym, samples in sorted(report["timezone_inference"]["anchor_raw_samples"].items()):
        lines.append(f"- **{sym}**：`{' | '.join(samples)}`")

    lines.extend(
        [
            "",
            "## 3. 完整度前 20 名標的預覽",
            "| Symbol | Company | 總筆數 | 獨立季數 | 區間 | 完整度 | 非零時間比例 |",
            "|---|---|---:|---:|---|---:|---:|",
        ]
    )
    for _, row in coverage_df.head(20).iterrows():
        lines.append(
            f"| {row['symbol']} | {row['company_name']} | {row['total_records']} | "
            f"{row['unique_quarters']} | {row['min_year']}–{row['max_year']} | "
            f"{row['completeness_ratio']:.1%} | {row['non_zero_time_ratio']:.1%} |"
        )

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 0：Hugging Face 法說會資料集前置探勘")
    parser.add_argument(
        "--schema-samples",
        type=int,
        default=PROBE_CFG.schema_sample_limit,
        help="Pass 1 抽檢 structured_content 之樣本筆數（預設 200）",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_probe(schema_samples=args.schema_samples)
