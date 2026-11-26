"""Stage 3A：OpenAI 結構化語義抽取（Smoke Test 與 Batch API 全流程）。

嚴格遵守 AGENTS.md：
1. 僅讀取 data/02_clean/transcripts_clean 中已去識別化之 *_anon 文本，杜絕未來洩漏與記憶作弊。
2. API Key 僅自環境變數或 config.env / .env 載入，嚴禁硬編碼或於終端／檔案中印出。
3. 使用 OpenAI Structured Outputs (strict json_schema) + temperature=0 + 固定 seed，並強制驗證數值邊界。
4. 支援 smoke（5 家公司即時驗證）、prepare / submit / status / collect（Batch API 斷點續傳）與 build（產出特徵表）。
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import duckdb
import pandas as pd
import yaml

from src.config import (
    API_EXTRACT_CACHE_PATH,
    BATCH_WORKDIR,
    CLEAN_TRANSCRIPTS_DIR,
    LLM_CFG,
    LLM_FEATURES_PATH,
    PIPELINE_STATE_DB_PATH,
    RETEST_CACHE_PATH,
    STAGE3_AUDIT_MD,
    STAGE3_PILOT_AUDIT_MD,
    STAGE3_SMOKE_JSON,
    STAGE3_SMOKE_MD,
    ensure_project_dirs,
    load_project_env,
    load_universe,
)
from src.io_utils import atomic_write_parquet


REQUIRED_PAYLOAD_KEYS: Tuple[str, ...] = (
    "prepared_optimism",
    "qa_optimism",
    "ambiguity_score",
    "qa_evasiveness",
    "guidance_revision",
    "optimistic_quote",
    "evasive_quote",
    "brief_rationale",
)

STRICT_RESPONSE_FORMAT: Dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "earnings_call_sentiment_v2",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "prepared_optimism": {
                    "type": "integer",
                    "description": "Integer 0-100 tone score for [PREPARED REMARKS] only (50=S&P500 baseline).",
                },
                "qa_optimism": {
                    "type": "integer",
                    "description": "Integer 0-100 tone score for [Q&A SESSION] only (50=S&P500 baseline).",
                },
                "ambiguity_score": {
                    "type": "integer",
                    "description": "Integer 0-100 hedging/vagueness score across the whole call.",
                },
                "qa_evasiveness": {
                    "type": "integer",
                    "description": "Integer 0-100 deflection/evasiveness score in [Q&A SESSION] only.",
                },
                "guidance_revision": {"type": "integer", "enum": [-2, -1, 0, 1, 2]},
                "optimistic_quote": {"type": "string"},
                "evasive_quote": {"type": "string"},
                "brief_rationale": {"type": "string"},
            },
            "required": list(REQUIRED_PAYLOAD_KEYS),
            "additionalProperties": False,
        },
    },
}


# ==================== Prompt 與 Schema 驗證工具 ====================
def load_prompt_template(path: Path = LLM_CFG.prompt_path) -> Dict[str, Any]:
    """讀取版本化 Prompt YAML 模板並檢查必要欄位。"""
    if not path.exists():
        raise FileNotFoundError(f"找不到 Prompt 模板檔：{path}")
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Prompt 模板格式錯誤：{path}")
    for key in ("version", "system_prompt", "user_prompt_template"):
        if not cfg.get(key):
            raise ValueError(f"Prompt 模板缺少必要欄位 '{key}'：{path}")
    tpl = str(cfg["user_prompt_template"])
    if "{prepared_remarks_anon}" not in tpl or "{qa_session_anon}" not in tpl:
        raise ValueError("user_prompt_template 必須包含 {prepared_remarks_anon} 與 {qa_session_anon} 佔位符。")
    return cfg


def build_chat_messages(
    prepared_remarks_anon: str,
    qa_session_anon: str,
    prompt_cfg: Dict[str, Any],
) -> List[Dict[str, str]]:
    """以去識別化文本組裝 OpenAI Chat Completions 訊息陣列。"""
    prep = (prepared_remarks_anon or "").strip()
    qa = (qa_session_anon or "").strip()
    if not prep and not qa:
        raise ValueError("去識別化文本 prepared_remarks_anon 與 qa_session_anon 不可同時為空。")
    user_content = str(prompt_cfg["user_prompt_template"]).format(
        prepared_remarks_anon=prep or "[NO PREPARED REMARKS]",
        qa_session_anon=qa or "[NO Q&A SESSION]",
    )
    return [
        {"role": "system", "content": str(prompt_cfg["system_prompt"]).strip()},
        {"role": "user", "content": user_content},
    ]


def validate_llm_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Fail-Fast 驗證 v2 LLM 回傳 JSON 的欄位完整性、型別與分數邊界。"""
    if not isinstance(payload, dict):
        raise ValueError(f"LLM 回傳必須為 dict，實際為：{type(payload).__name__}")
    missing = [k for k in REQUIRED_PAYLOAD_KEYS if k not in payload]
    if missing:
        raise ValueError(f"LLM 回傳缺少必要欄位：{missing}")

    int_bounds = {
        "prepared_optimism": (0, 100),
        "qa_optimism": (0, 100),
        "ambiguity_score": (0, 100),
        "qa_evasiveness": (0, 100),
        "guidance_revision": (-2, 2),
    }
    validated: Dict[str, Any] = {}
    for col, (lo, hi) in int_bounds.items():
        val = payload[col]
        if isinstance(val, bool) or not isinstance(val, int):
            raise ValueError(f"欄位 {col} 必須為整數，實際為：{val!r}")
        if not (lo <= val <= hi):
            raise ValueError(f"欄位 {col} 超出範圍 [{lo}, {hi}]：{val}")
        validated[col] = val

    for col in ("optimistic_quote", "evasive_quote", "brief_rationale"):
        val = payload[col]
        if not isinstance(val, str) or not val.strip():
            raise ValueError(f"欄位 {col} 必須為非空字串。")
        validated[col] = " ".join(val.split())

    return validated


def enrich_derived_scores(validated: Dict[str, Any]) -> Dict[str, Any]:
    """由已驗證之 Prepared / Q&A 分數確定性衍生綜合樂觀度與 Q&A 語氣落差。"""
    prep = int(validated["prepared_optimism"])
    qa = int(validated["qa_optimism"])
    return {
        **validated,
        "optimism_score": round(0.5 * (prep + qa), 2),
        "qa_tone_drop": qa - prep,
    }


def estimate_cost_usd(
    prompt_tokens: int,
    completion_tokens: int,
    pricing_mode: str = "sync",
) -> float:
    """估算 OpenAI gpt-4o-mini 的美元費用（sync 即時價或 batch 五折價）。"""
    if pricing_mode == "batch":
        in_rate = LLM_CFG.batch_input_cost_per_1m
        out_rate = LLM_CFG.batch_output_cost_per_1m
    elif pricing_mode == "sync":
        in_rate = LLM_CFG.sync_input_cost_per_1m
        out_rate = LLM_CFG.sync_output_cost_per_1m
    else:
        raise ValueError(f"不支援的計費模式：{pricing_mode}")
    return (prompt_tokens * in_rate + completion_tokens * out_rate) / 1_000_000.0


# ==================== Clean 資料讀取與樣本篩選 ====================
def load_valid_clean_transcripts(clean_dir: Path = CLEAN_TRANSCRIPTS_DIR) -> pd.DataFrame:
    """透過 DuckDB 讀取所有 is_valid=True 的去識別化法說會紀錄（嚴格排除 *_raw 欄位）。"""
    parquet_glob = (clean_dir / "year=*" / "*.parquet").as_posix()
    con = duckdb.connect()
    try:
        df = con.execute(
            f"""
            SELECT
                event_id,
                symbol,
                gics_sector,
                company_name,
                year,
                quarter,
                event_trading_date,
                session_type,
                prepared_remarks_anon,
                qa_session_anon,
                prepared_word_count,
                qa_word_count,
                total_word_count
            FROM read_parquet('{parquet_glob}', hive_partitioning=true)
            WHERE is_valid = TRUE
            ORDER BY symbol, event_trading_date, event_id
            """
        ).df()
    finally:
        con.close()

    if df.empty:
        raise RuntimeError(f"在 {clean_dir} 中找不到任何 is_valid=True 的法說會紀錄。")
    assert df["event_id"].is_unique, "clean 表存在重複 event_id！"
    return df


def select_smoke_records(
    df: pd.DataFrame,
    n: int = LLM_CFG.smoke_test_count,
    pilot_symbols: Sequence[str] = (),
) -> pd.DataFrame:
    """挑選 n 家不同公司（優先不同 GICS 產業與 2024 年樣本）作為 API 連通與邏輯驗證集。"""
    if n <= 0:
        raise ValueError("n 必須大於 0。")

    pool = df.copy()
    pool["is_pilot"] = pool["symbol"].isin(set(pilot_symbols)) if pilot_symbols else False
    pool["prefer_2024"] = pool["year"] == 2024
    # 排序：優先 pilot 股票、優先 2024 年、最新交易日、固定字母序確保完全可重現
    pool = pool.sort_values(
        by=["is_pilot", "prefer_2024", "event_trading_date", "symbol", "event_id"],
        ascending=[False, False, False, True, True],
    )

    # 每家公司取 1 場代表性法說會，並優先覆蓋不同 GICS 產業
    per_symbol = pool.drop_duplicates(subset=["symbol"], keep="first")
    selected_rows: List[pd.Series] = []
    used_sectors: Set[str] = set()
    used_symbols: Set[str] = set()

    for _, row in per_symbol.iterrows():
        sector = str(row["gics_sector"])
        if sector not in used_sectors:
            selected_rows.append(row)
            used_sectors.add(sector)
            used_symbols.add(str(row["symbol"]))
            if len(selected_rows) == n:
                break

    if len(selected_rows) < n:
        for _, row in per_symbol.iterrows():
            sym = str(row["symbol"])
            if sym not in used_symbols:
                selected_rows.append(row)
                used_symbols.add(sym)
                if len(selected_rows) == n:
                    break

    if len(selected_rows) < n:
        raise ValueError(f"可用不同公司數量不足 {n} 家（僅 {len(selected_rows)} 家）。")

    result = pd.DataFrame(selected_rows).drop(columns=["is_pilot", "prefer_2024"]).reset_index(drop=True)
    return result


def select_pilot_records(
    df: pd.DataFrame,
    pilot_symbols: Sequence[str],
    quarters_per_symbol: int = LLM_CFG.pilot_quarters_per_symbol,
) -> pd.DataFrame:
    """針對 20 檔 Pilot 股票各取最近 N 季有效法說會（約 200 場）。"""
    pilot_set = set(pilot_symbols)
    sub = df[df["symbol"].isin(pilot_set)].copy()
    sub = sub.sort_values(by=["symbol", "event_trading_date"], ascending=[True, False])
    sub = sub.groupby("symbol", group_keys=False).head(quarters_per_symbol)
    return sub.sort_values(by=["symbol", "event_trading_date"]).reset_index(drop=True)


def select_retest_records(
    pilot_df: pd.DataFrame,
    n: int = LLM_CFG.retest_sample_size,
) -> pd.DataFrame:
    """從 Pilot 集合中確定性分層抽取 n 場重測樣本（涵蓋所有公司與多年度）。"""
    if len(pilot_df) < n:
        raise ValueError(f"Pilot 樣本數不足以抽取 {n} 場重測（實際僅 {len(pilot_df)} 場）。")
    ordered = pilot_df.sort_values(by=["symbol", "event_trading_date", "event_id"]).reset_index(drop=True)
    # 每個 symbol 先取最新 1 場（涵蓋全 20 檔代表股），其餘由各公司較早期季度均勻補足
    latest_per_sym = ordered.groupby("symbol", group_keys=False).tail(1)
    picked_ids = set(latest_per_sym["event_id"])
    remaining_need = n - len(latest_per_sym)
    if remaining_need <= 0:
        return latest_per_sym.head(n).sort_values(by=["symbol", "event_trading_date"]).reset_index(drop=True)

    rem = ordered[~ordered["event_id"].isin(picked_ids)].copy()
    # 從剩餘紀錄按 (year, symbol) 確定性抽樣，優先涵蓋 2022-2023 不同市場時期
    rem = rem.sort_values(by=["year", "symbol", "event_trading_date", "event_id"])
    step = max(1, len(rem) // remaining_need)
    extra = rem.iloc[::step].head(remaining_need)
    if len(extra) < remaining_need:
        extra_ids = set(extra["event_id"])
        pad = rem[~rem["event_id"].isin(extra_ids)].head(remaining_need - len(extra))
        extra = pd.concat([extra, pad], ignore_index=True)

    combined = pd.concat([latest_per_sym, extra], ignore_index=True)
    return combined.sort_values(by=["symbol", "event_trading_date", "event_id"]).reset_index(drop=True)


# ==================== Append-Only JSONL 快取與 SQLite 狀態庫 ====================
def load_cache_records(
    cache_path: Path = API_EXTRACT_CACHE_PATH,
    prompt_version: str = LLM_CFG.prompt_version,
    model: str = LLM_CFG.model,
) -> Dict[str, Dict[str, Any]]:
    """讀取 Append-Only JSONL 快取，回傳符合版本與模型的 event_id -> record 映射。"""
    records: Dict[str, Dict[str, Any]] = {}
    if not cache_path.exists():
        return records
    with cache_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if item.get("prompt_version") == prompt_version and item.get("model") == model:
                records[str(item["event_id"])] = item
    return records


def append_cache_record(cache_path: Path, record: Dict[str, Any]) -> None:
    """以 Append-Only 方式寫入單筆 LLM 抽取結果至 JSONL 快取。"""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def init_state_db(db_path: Path = PIPELINE_STATE_DB_PATH) -> sqlite3.Connection:
    """初始化 Batch 任務狀態追蹤 SQLite 資料庫。"""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS openai_batches (
            batch_id TEXT PRIMARY KEY,
            input_file_id TEXT NOT NULL,
            output_file_id TEXT,
            error_file_id TEXT,
            shard_file TEXT NOT NULL,
            scope TEXT NOT NULL,
            prompt_version TEXT NOT NULL,
            model TEXT NOT NULL,
            n_requests INTEGER NOT NULL,
            status TEXT NOT NULL,
            collected INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    conn.commit()
    return conn


# ==================== Batch JSONL 切分器 ====================
def build_batch_request_line(
    event_id: str,
    prepared_remarks_anon: str,
    qa_session_anon: str,
    prompt_cfg: Dict[str, Any],
    model: str = LLM_CFG.model,
) -> Dict[str, Any]:
    """組裝單筆 OpenAI Batch API /v1/chat/completions 請求物件。"""
    messages = build_chat_messages(prepared_remarks_anon, qa_session_anon, prompt_cfg)
    return {
        "custom_id": str(event_id),
        "method": "POST",
        "url": LLM_CFG.batch_endpoint,
        "body": {
            "model": model,
            "temperature": LLM_CFG.temperature,
            "seed": LLM_CFG.seed,
            "max_completion_tokens": LLM_CFG.max_completion_tokens,
            "response_format": STRICT_RESPONSE_FORMAT,
            "messages": messages,
        },
    }


def build_batch_jsonl_shards(
    records: pd.DataFrame,
    prompt_cfg: Dict[str, Any],
    out_dir: Path,
    scope: str,
    shard_size: int = LLM_CFG.batch_shard_size,
    max_mb: float = LLM_CFG.max_batch_file_mb,
    done_ids: Optional[Set[str]] = None,
) -> List[Path]:
    """將待處理法說會切分為多個符合 OpenAI 檔案與 Token 限制之 Batch JSONL 分片。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    skip_ids = done_ids or set()
    max_bytes = int(max_mb * 1024 * 1024)
    prompt_ver = str(prompt_cfg["version"])

    shard_paths: List[Path] = []
    shard_idx = 0
    buf_lines: List[str] = []
    buf_bytes = 0

    def _flush() -> None:
        nonlocal shard_idx, buf_lines, buf_bytes
        if not buf_lines:
            return
        shard_idx += 1
        shard_path = out_dir / f"batch_{scope}_{prompt_ver}_shard{shard_idx:03d}.jsonl"
        with shard_path.open("w", encoding="utf-8") as f:
            f.writelines(buf_lines)
        shard_paths.append(shard_path)
        buf_lines = []
        buf_bytes = 0

    for row in records.itertuples(index=False):
        eid = str(row.event_id)
        if eid in skip_ids:
            continue
        req = build_batch_request_line(
            event_id=eid,
            prepared_remarks_anon=str(row.prepared_remarks_anon),
            qa_session_anon=str(row.qa_session_anon),
            prompt_cfg=prompt_cfg,
        )
        line = json.dumps(req, ensure_ascii=False) + "\n"
        line_bytes = len(line.encode("utf-8"))
        if buf_lines and (len(buf_lines) >= shard_size or buf_bytes + line_bytes > max_bytes):
            _flush()
        buf_lines.append(line)
        buf_bytes += line_bytes

    _flush()
    return shard_paths


# ==================== 跨季差分特徵計算（嚴格 Point-in-Time） ====================
def compute_qoq_features(df: pd.DataFrame) -> pd.DataFrame:
    """依 (symbol, event_trading_date) 嚴格時序計算跨季分數變化，首季保留 NaN。"""
    out = df.sort_values(by=["symbol", "event_trading_date", "event_id"]).reset_index(drop=True).copy()
    if "optimism_score" not in out.columns:
        out["optimism_score"] = (0.5 * (out["prepared_optimism"] + out["qa_optimism"])).round(2)
    if "qa_tone_drop" not in out.columns:
        out["qa_tone_drop"] = out["qa_optimism"] - out["prepared_optimism"]

    grp = out.groupby("symbol", sort=False)

    out["qoq_prep_opt_diff"] = grp["prepared_optimism"].diff()
    out["qoq_qa_opt_diff"] = grp["qa_optimism"].diff()
    out["qoq_opt_diff"] = grp["optimism_score"].diff()
    out["qoq_tone_drop_diff"] = grp["qa_tone_drop"].diff()
    out["qoq_amb_diff"] = grp["ambiguity_score"].diff()
    out["qoq_evas_diff"] = grp["qa_evasiveness"].diff()
    out["qoq_guid_diff"] = grp["guidance_revision"].diff()

    # 僅使用「過去 4 季（shift(1)）」之滾動平均計算偏移，嚴禁納入當季或未來資訊
    trailing_opt_mean = grp["optimism_score"].transform(
        lambda s: s.shift(1).rolling(window=4, min_periods=2).mean()
    )
    out["opt_vs_trailing_4q_mean"] = out["optimism_score"] - trailing_opt_mean
    return out


# ==================== OpenAI Client 與即時 Smoke Test ====================
def create_openai_client() -> Tuple[Any, Optional[Path]]:
    """從 config.env / .env 載入 OPENAI_API_KEY 並建立 OpenAI Client（嚴禁輸出金鑰內容）。"""
    env_path = load_project_env(override=False)
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "找不到 OPENAI_API_KEY。請確認 config.env 或 .env 中已設定 OPENAI_API_KEY。"
        )
    from openai import OpenAI

    client = OpenAI(api_key=api_key, timeout=LLM_CFG.request_timeout_sec)
    return client, env_path


def call_openai_single(
    client: Any,
    prepared_remarks_anon: str,
    qa_session_anon: str,
    prompt_cfg: Dict[str, Any],
    model: str = LLM_CFG.model,
) -> Dict[str, Any]:
    """對單場去識別化法說會呼叫 OpenAI Chat Completions 並以指數退避重試。"""
    messages = build_chat_messages(prepared_remarks_anon, qa_session_anon, prompt_cfg)
    last_err: Optional[Exception] = None

    for attempt in range(1, LLM_CFG.max_retries + 1):
        t0 = time.perf_counter()
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=LLM_CFG.temperature,
                seed=LLM_CFG.seed,
                max_completion_tokens=LLM_CFG.max_completion_tokens,
                response_format=STRICT_RESPONSE_FORMAT,
                messages=messages,
            )
            elapsed_sec = round(time.perf_counter() - t0, 2)
            choice = resp.choices[0]
            if choice.message.refusal:
                raise RuntimeError(f"模型拒答（refusal）：{choice.message.refusal}")
            raw_content = choice.message.content
            if not raw_content:
                raise RuntimeError("模型回傳空內容。")
            parsed = enrich_derived_scores(validate_llm_payload(json.loads(raw_content)))
            prompt_tokens = int(getattr(resp.usage, "prompt_tokens", 0) or 0)
            completion_tokens = int(getattr(resp.usage, "completion_tokens", 0) or 0)
            return {
                "payload": parsed,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "finish_reason": str(choice.finish_reason),
                "system_fingerprint": str(getattr(resp, "system_fingerprint", "") or ""),
                "elapsed_sec": elapsed_sec,
            }
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            if attempt < LLM_CFG.max_retries:
                sleep_s = LLM_CFG.retry_backoff_base_sec * (2 ** (attempt - 1))
                time.sleep(sleep_s)

    raise RuntimeError(f"OpenAI 呼叫重試 {LLM_CFG.max_retries} 次後仍失敗：{last_err}") from last_err


def write_smoke_report(
    results: List[Dict[str, Any]],
    total_valid_calls: int,
    env_file_name: str,
    json_path: Path = STAGE3_SMOKE_JSON,
    md_path: Path = STAGE3_SMOKE_MD,
) -> None:
    """輸出 5 家公司 Smoke Test 的 JSON 原始紀錄與 Markdown 稽核報告（含橫斷面離散度檢驗）。"""
    n_calls = len(results)
    tot_in = sum(r["prompt_tokens"] for r in results)
    tot_out = sum(r["completion_tokens"] for r in results)
    avg_in = tot_in / n_calls
    avg_out = tot_out / n_calls
    sync_cost = estimate_cost_usd(tot_in, tot_out, pricing_mode="sync")
    batch_proj_200 = estimate_cost_usd(int(avg_in * 200), int(avg_out * 200), pricing_mode="batch")
    batch_proj_full = estimate_cost_usd(
        int(avg_in * total_valid_calls),
        int(avg_out * total_valid_calls),
        pricing_mode="batch",
    )

    score_cols = (
        ("prepared_optimism", "Prepared Opt (0~100)"),
        ("qa_optimism", "Q&A Opt (0~100)"),
        ("qa_tone_drop", "Q&A Tone Drop (Q&A - Prep)"),
        ("ambiguity_score", "Ambiguity (0~100)"),
        ("qa_evasiveness", "Q&A Evasiveness (0~100)"),
        ("guidance_revision", "Guidance (-2~+2)"),
    )
    dispersion: Dict[str, Dict[str, Any]] = {}
    for key, _ in score_cols:
        vals = pd.Series([r["payload"][key] for r in results], dtype="float64")
        dispersion[key] = {
            "min": float(vals.min()),
            "max": float(vals.max()),
            "range": float(vals.max() - vals.min()),
            "std": round(float(vals.std(ddof=1)), 2) if len(vals) > 1 else 0.0,
            "n_unique": int(vals.nunique()),
        }

    summary_obj = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": LLM_CFG.model,
        "prompt_version": LLM_CFG.prompt_version,
        "env_source": env_file_name,
        "n_calls": n_calls,
        "total_prompt_tokens": tot_in,
        "total_completion_tokens": tot_out,
        "avg_prompt_tokens": round(avg_in, 1),
        "avg_completion_tokens": round(avg_out, 1),
        "actual_sync_cost_usd": round(sync_cost, 6),
        "projected_pilot_200_batch_cost_usd": round(batch_proj_200, 4),
        "projected_full_batch_cost_usd": round(batch_proj_full, 4),
        "total_valid_calls_in_clean": total_valid_calls,
        "dispersion": dispersion,
        "results": results,
    }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary_obj, f, ensure_ascii=False, indent=2)

    lines: List[str] = [
        f"# Stage 3A OpenAI 5 家公司法說會情緒評分報告（Prompt `{LLM_CFG.prompt_version}`）",
        "",
        f"- **測試時間 (UTC)**：`{summary_obj['generated_at_utc']}`",
        f"- **模型與 Prompt 版本**：`{LLM_CFG.model}` / `{LLM_CFG.prompt_version}`（`temperature={LLM_CFG.temperature}`, `seed={LLM_CFG.seed}`, Strict JSON Schema）",
        f"- **金鑰來源檔**：`{env_file_name}`（未輸出任何金鑰內容）",
        f"- **測試筆數**：`{n_calls}` 場（來自 `{n_calls}` 家不同公司與不同 GICS 產業，全部輸入去識別化 `*_anon` 文本）",
        f"- **Token 統計**：總輸入 `{tot_in:,}` tokens（場均 `{avg_in:,.1f}`）、總輸出 `{tot_out:,}` tokens（場均 `{avg_out:,.1f}`）",
        f"- **費用實測與推估**：",
        f"  - 本次 5 場即時呼叫實測費用：`USD {sync_cost:.5f}`",
        f"  - 推估 200 場 Pilot（Batch API 五折價）：`USD {batch_proj_200:.4f}`",
        f"  - 推估全量 {total_valid_calls:,} 場（Batch API 五折價）：`USD {batch_proj_full:.2f}`",
        "",
        "## 一、5 家公司情緒評分總覽",
        "",
        "| Symbol | GICS 產業 | 財季 | Prep Opt | Q&A Opt | Tone Drop | Ambiguity | Q&A Evasiveness | Guidance | Input / Output Tokens | 耗時(s) |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for r in results:
        p = r["payload"]
        lines.append(
            f"| `{r['symbol']}` | {r['gics_sector']} | {r['year']} Q{r['quarter']} "
            f"| **{p['prepared_optimism']}** | **{p['qa_optimism']}** | **{p['qa_tone_drop']:+d}** "
            f"| **{p['ambiguity_score']}** | **{p['qa_evasiveness']}** | **{p['guidance_revision']:+d}** "
            f"| {r['prompt_tokens']:,} / {r['completion_tokens']:,} | {r['elapsed_sec']:.2f} |"
        )

    lines.extend(
        [
            "",
            "## 二、橫斷面分數區分度統計（反中央集中檢驗）",
            "",
            "| 指標 | Min | Max | Range (Max-Min) | Std | 相異值數 (Unique / 5) |",
            "| :--- | :---: | :---: | :---: | :---: | :---: |",
        ]
    )
    for key, label in score_cols:
        d = dispersion[key]
        lines.append(
            f"| {label} | {d['min']:g} | {d['max']:g} | {d['range']:g} | {d['std']:.2f} | **{d['n_unique']} / {n_calls}** |"
        )

    lines.extend(["", "## 三、逐場證據引文與評分理由（人工判讀檢核）", ""])
    for idx, r in enumerate(results, 1):
        p = r["payload"]
        lines.extend(
            [
                f"### {idx}. `{r['symbol']}`（{r['company_name']}｜{r['gics_sector']}｜{r['year']} Q{r['quarter']}｜`{r['event_id']}`）",
                (
                    f"- **分數**：Prepared=`{p['prepared_optimism']}`, Q&A=`{p['qa_optimism']}` "
                    f"(Drop=`{p['qa_tone_drop']:+d}`), Ambiguity=`{p['ambiguity_score']}`, "
                    f"Evasiveness=`{p['qa_evasiveness']}`, Guidance=`{p['guidance_revision']:+d}`"
                ),
                f"- **簡要判讀理由**：{p['brief_rationale']}",
                f"- **樂觀證據原句**：> {p['optimistic_quote']}",
                f"- **迴避／保守原句**：> {p['evasive_quote']}",
                "",
            ]
        )

    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(lines), encoding="utf-8")


def run_smoke_test(n: int = LLM_CFG.smoke_test_count) -> None:
    """執行 5 家公司法說會即時情緒評分測試。"""
    ensure_project_dirs()
    prompt_cfg = load_prompt_template()
    universe = load_universe()
    pilot_symbols = universe.get("pilot_symbols", [])

    df_valid = load_valid_clean_transcripts()
    df_smoke = select_smoke_records(df_valid, n=n, pilot_symbols=pilot_symbols)

    client, env_path = create_openai_client()
    env_name = env_path.name if env_path else "process_env"
    print(
        f"[Smoke Test] 已自 {env_name} 載入 OPENAI_API_KEY，開始測試 {len(df_smoke)} 家公司"
        f"（模型：{LLM_CFG.model}，Prompt：{prompt_cfg['version']}）..."
    )

    results: List[Dict[str, Any]] = []
    for idx, row in enumerate(df_smoke.itertuples(index=False), 1):
        print(
            f"  ({idx}/{len(df_smoke)}) 呼叫 {row.symbol} ({row.gics_sector}) "
            f"{row.year} Q{row.quarter} [{row.total_word_count:,} words]..."
        )
        res = call_openai_single(
            client=client,
            prepared_remarks_anon=str(row.prepared_remarks_anon),
            qa_session_anon=str(row.qa_session_anon),
            prompt_cfg=prompt_cfg,
        )
        p = res["payload"]
        print(
            f"    -> OK ({res['elapsed_sec']}s, in={res['prompt_tokens']:,}, out={res['completion_tokens']:,}) | "
            f"Prep={p['prepared_optimism']}, QA={p['qa_optimism']} (Drop={p['qa_tone_drop']:+d}), "
            f"Amb={p['ambiguity_score']}, Evas={p['qa_evasiveness']}, Guid={p['guidance_revision']:+d}"
        )
        results.append(
            {
                "event_id": str(row.event_id),
                "symbol": str(row.symbol),
                "company_name": str(row.company_name),
                "gics_sector": str(row.gics_sector),
                "year": int(row.year),
                "quarter": int(row.quarter),
                "event_trading_date": str(row.event_trading_date),
                "total_word_count": int(row.total_word_count),
                **res,
            }
        )

    write_smoke_report(
        results=results,
        total_valid_calls=len(df_valid),
        env_file_name=env_name,
    )
    print(f"[Smoke Test] 測試全部通過！報告已寫入：{STAGE3_SMOKE_MD}")


def compute_retest_metrics(df_run1: pd.DataFrame, df_run2: pd.DataFrame) -> pd.DataFrame:
    """計算兩次獨立評分（Run 1 vs. Run 2）之重測信度指標（Spearman、Pearson、MAE、±5 容差一致率）。"""
    merged = df_run1.merge(df_run2, on="event_id", suffixes=("_r1", "_r2"), how="inner")
    if merged.empty:
        raise ValueError("Run 1 與 Run 2 無重疊 event_id，無法計算重測信度。")

    score_specs = (
        ("prepared_optimism", 5.0),
        ("qa_optimism", 5.0),
        ("optimism_score", 5.0),
        ("qa_tone_drop", 5.0),
        ("ambiguity_score", 5.0),
        ("qa_evasiveness", 5.0),
        ("guidance_revision", 1.0),
    )
    rows: List[Dict[str, Any]] = []
    for col, tol in score_specs:
        s1 = merged[f"{col}_r1"].astype(float)
        s2 = merged[f"{col}_r2"].astype(float)
        abs_diff = (s1 - s2).abs()
        spearman = float(s1.rank().corr(s2.rank())) if s1.nunique() > 1 and s2.nunique() > 1 else float("nan")
        pearson = float(s1.corr(s2, method="pearson")) if s1.nunique() > 1 and s2.nunique() > 1 else float("nan")
        rows.append(
            {
                "metric": col,
                "n_pairs": len(merged),
                "mae": round(float(abs_diff.mean()), 3),
                "max_abs_diff": round(float(abs_diff.max()), 3),
                "exact_match_rate": round(float((abs_diff == 0).mean()), 4),
                "within_tol_rate": round(float((abs_diff <= tol).mean()), 4),
                "tolerance": tol,
                "spearman_rho": round(spearman, 4) if pd.notna(spearman) else None,
                "pearson_r": round(pearson, 4) if pd.notna(pearson) else None,
            }
        )
    return pd.DataFrame(rows)


# ==================== OpenAI Batch API 完整流水線 ====================
def run_prepare_batches(scope: str) -> None:
    """將目標範圍（pilot、retest 或 full）尚未快取之紀錄切分為 Batch JSONL 檔。"""
    ensure_project_dirs()
    prompt_cfg = load_prompt_template()
    df_valid = load_valid_clean_transcripts()
    if scope == "pilot":
        pilot_symbols = load_universe().get("pilot_symbols", [])
        target_df = select_pilot_records(df_valid, pilot_symbols)
        cache_path = API_EXTRACT_CACHE_PATH
    elif scope == "retest":
        pilot_symbols = load_universe().get("pilot_symbols", [])
        pilot_df = select_pilot_records(df_valid, pilot_symbols)
        target_df = select_retest_records(pilot_df, n=LLM_CFG.retest_sample_size)
        cache_path = RETEST_CACHE_PATH
    elif scope == "full":
        target_df = df_valid
        cache_path = API_EXTRACT_CACHE_PATH
    else:
        raise ValueError(f"不支援的 scope：{scope}")

    done_ids = set(load_cache_records(cache_path=cache_path, prompt_version=str(prompt_cfg["version"])).keys())
    shards = build_batch_jsonl_shards(
        records=target_df,
        prompt_cfg=prompt_cfg,
        out_dir=BATCH_WORKDIR,
        scope=scope,
        done_ids=done_ids,
    )
    print(
        f"[Prepare] scope={scope}, 目標總筆數={len(target_df)}, 已快取跳過={len(done_ids & set(target_df['event_id']))}, "
        f"產生分片數={len(shards)}（輸出目錄：{BATCH_WORKDIR}）"
    )


def run_submit_batches(scope: str, max_shards: int = 0) -> None:
    """上傳本地 Batch JSONL 分片並建立 OpenAI Batch 任務，紀錄至 SQLite。"""
    ensure_project_dirs()
    prompt_ver = LLM_CFG.prompt_version
    pattern = f"batch_{scope}_{prompt_ver}_shard*.jsonl"
    shard_files = sorted(BATCH_WORKDIR.glob(pattern))
    if not shard_files:
        raise FileNotFoundError(f"找不到分片檔案（{pattern}），請先執行 --mode prepare --scope {scope}")

    conn = init_state_db()
    submitted_files = {
        row[0]
        for row in conn.execute(
            "SELECT shard_file FROM openai_batches WHERE status NOT IN ('failed', 'expired', 'cancelled')"
        ).fetchall()
    }

    pending_files = [p for p in shard_files if p.name not in submitted_files]
    if max_shards > 0:
        pending_files = pending_files[:max_shards]

    if not pending_files:
        print(f"[Submit] scope={scope} 所有分片皆已送出，無待上傳分片。")
        conn.close()
        return

    client, _ = create_openai_client()
    now_iso = datetime.now(timezone.utc).isoformat()
    for p in pending_files:
        n_lines = sum(1 for _ in p.open("r", encoding="utf-8"))
        try:
            with p.open("rb") as fh:
                uploaded = client.files.create(file=fh, purpose="batch")
            batch = client.batches.create(
                input_file_id=uploaded.id,
                endpoint=LLM_CFG.batch_endpoint,
                completion_window=LLM_CFG.batch_completion_window,
                metadata={"scope": scope, "prompt_version": prompt_ver, "shard_file": p.name},
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[Submit] 上傳 {p.name} 時遇限流或錯誤（已停止後續分片，前序分片安全保留）：{exc}")
            break

        conn.execute(
            """
            INSERT OR REPLACE INTO openai_batches
            (batch_id, input_file_id, output_file_id, error_file_id, shard_file, scope,
             prompt_version, model, n_requests, status, collected, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?)
            """,
            (
                batch.id,
                uploaded.id,
                batch.output_file_id,
                batch.error_file_id,
                p.name,
                scope,
                prompt_ver,
                LLM_CFG.model,
                n_lines,
                batch.status,
                now_iso,
                now_iso,
            ),
        )
        conn.commit()
        print(f"[Submit] {p.name} ({n_lines} calls) -> batch_id={batch.id}, status={batch.status}")
    conn.close()


def run_check_status() -> None:
    """向 OpenAI 查詢所有未回收 Batch 任務狀態並同步更新 SQLite，印出彙總統計。"""
    conn = init_state_db()
    rows = conn.execute(
        "SELECT batch_id, shard_file, scope, n_requests, status, collected FROM openai_batches ORDER BY created_at"
    ).fetchall()
    if not rows:
        print("[Status] 目前沒有任何已送出的 Batch 任務。")
        conn.close()
        return

    client, _ = create_openai_client()
    now_iso = datetime.now(timezone.utc).isoformat()
    summary: Dict[str, int] = {"collected": 0, "completed_pending": 0, "in_progress": 0, "validating": 0, "failed": 0, "other": 0}
    total_done_calls = 0
    total_target_calls = 0

    for batch_id, shard_file, scope, n_req, old_status, collected in rows:
        if scope == "full":
            total_target_calls += int(n_req)
        if collected:
            summary["collected"] += 1
            if scope == "full":
                total_done_calls += int(n_req)
            continue
        if old_status in ("failed", "expired", "cancelled"):
            summary["failed"] += 1
            continue

        b = client.batches.retrieve(batch_id)
        conn.execute(
            """
            UPDATE openai_batches
            SET status = ?, output_file_id = ?, error_file_id = ?, updated_at = ?
            WHERE batch_id = ?
            """,
            (b.status, b.output_file_id, b.error_file_id, now_iso, batch_id),
        )
        counts = getattr(b, "request_counts", None)
        done_cnt = int(getattr(counts, "completed", 0) or 0) if counts else 0
        fail_cnt = int(getattr(counts, "failed", 0) or 0) if counts else 0
        if scope == "full":
            total_done_calls += done_cnt

        if b.status == "completed":
            summary["completed_pending"] += 1
        elif b.status in ("in_progress", "finalizing"):
            summary["in_progress"] += 1
        elif b.status == "validating":
            summary["validating"] += 1
        elif b.status in ("failed", "expired", "cancelled"):
            summary["failed"] += 1
        else:
            summary["other"] += 1

        err_note = ""
        if b.status == "failed" and getattr(b, "errors", None):
            err_note = f" | errors={b.errors}"
        print(
            f"[Status] {batch_id} | {scope} | {shard_file} | status={b.status} | "
            f"completed={done_cnt}/{n_req}, failed={fail_cnt} | collected={bool(collected)}{err_note}"
        )
    conn.commit()
    conn.close()
    print(
        f"[Status Summary] collected={summary['collected']}, completed_pending={summary['completed_pending']}, "
        f"in_progress={summary['in_progress']}, validating={summary['validating']}, failed={summary['failed']} | "
        f"full_progress={total_done_calls}/{total_target_calls}"
    )


def run_collect_batches() -> None:
    """下載已完成之 Batch 結果，通過 Schema 驗證後依 scope 寫入對應 Append-Only 快取。"""
    conn = init_state_db()
    rows = conn.execute(
        """
        SELECT batch_id, output_file_id, error_file_id, shard_file, scope, prompt_version, model
        FROM openai_batches
        WHERE status = 'completed' AND collected = 0 AND output_file_id IS NOT NULL
        """
    ).fetchall()
    if not rows:
        print("[Collect] 沒有待回收的已完成 Batch 任務（請先執行 --mode status 更新狀態）。")
        conn.close()
        return

    client, _ = create_openai_client()
    for batch_id, output_file_id, error_file_id, shard_file, scope, prompt_ver, model in rows:
        cache_path = RETEST_CACHE_PATH if scope == "retest" else API_EXTRACT_CACHE_PATH
        done_ids = set(load_cache_records(cache_path=cache_path, prompt_version=prompt_ver, model=model).keys())
        raw_text = client.files.content(output_file_id).text
        ok_cnt = 0
        err_cnt = 0
        for line in raw_text.splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            eid = str(item["custom_id"])
            if eid in done_ids:
                continue
            resp = item.get("response") or {}
            if resp.get("status_code") != 200:
                err_cnt += 1
                continue
            body = resp["body"]
            content = body["choices"][0]["message"]["content"]
            payload = enrich_derived_scores(validate_llm_payload(json.loads(content)))
            usage = body.get("usage") or {}
            record = {
                "event_id": eid,
                "model": model,
                "prompt_version": prompt_ver,
                "batch_id": batch_id,
                "prompt_tokens": int(usage.get("prompt_tokens", 0)),
                "completion_tokens": int(usage.get("completion_tokens", 0)),
                **payload,
            }
            append_cache_record(cache_path, record)
            done_ids.add(eid)
            ok_cnt += 1

        if error_file_id:
            err_text = client.files.content(error_file_id).text
            err_lines = [ln for ln in err_text.splitlines() if ln.strip()]
            err_cnt += len(err_lines)

        conn.execute(
            "UPDATE openai_batches SET collected = 1, updated_at = ? WHERE batch_id = ?",
            (datetime.now(timezone.utc).isoformat(), batch_id),
        )
        conn.commit()
        print(
            f"[Collect] {batch_id} ({scope} / {shard_file}) 回收完成：寫入 {ok_cnt} 筆至 {cache_path.name}，失敗 {err_cnt} 筆。"
        )
    conn.close()


def run_pilot_audit() -> None:
    """彙整 200 場 Pilot 分佈檢驗與 30 場重測信度檢定，輸出 outputs/stage3_pilot_audit.md。"""
    ensure_project_dirs()
    prompt_ver = LLM_CFG.prompt_version
    pilot_cache = load_cache_records(cache_path=API_EXTRACT_CACHE_PATH, prompt_version=prompt_ver)
    retest_cache = load_cache_records(cache_path=RETEST_CACHE_PATH, prompt_version=prompt_ver)
    if not pilot_cache:
        raise RuntimeError(f"找不到任何 Pilot 快取紀錄：{API_EXTRACT_CACHE_PATH}")

    df_valid = load_valid_clean_transcripts()[
        ["event_id", "symbol", "gics_sector", "year", "quarter", "event_trading_date"]
    ]
    df_pilot = df_valid.merge(pd.DataFrame(list(pilot_cache.values())), on="event_id", how="inner")
    df_pilot = compute_qoq_features(df_pilot)

    tot_in_p = int(df_pilot["prompt_tokens"].sum())
    tot_out_p = int(df_pilot["completion_tokens"].sum())
    cost_pilot = estimate_cost_usd(tot_in_p, tot_out_p, pricing_mode="batch")

    avg_in = tot_in_p / len(df_pilot)
    avg_out = tot_out_p / len(df_pilot)
    proj_full_cost = estimate_cost_usd(
        int(avg_in * len(df_valid)),
        int(avg_out * len(df_valid)),
        pricing_mode="batch",
    )

    retest_cost = 0.0
    retest_df: Optional[pd.DataFrame] = None
    if retest_cache:
        df_r2 = pd.DataFrame(list(retest_cache.values()))
        retest_cost = estimate_cost_usd(
            int(df_r2["prompt_tokens"].sum()),
            int(df_r2["completion_tokens"].sum()),
            pricing_mode="batch",
        )
        retest_df = compute_retest_metrics(df_pilot, df_r2)

    lines: List[str] = [
        f"# Stage 3A Pilot（{len(df_pilot)} 場）與重測信度檢定報告（Prompt `{prompt_ver}`）",
        "",
        f"- **產出時間 (UTC)**：`{datetime.now(timezone.utc).isoformat()}`",
        f"- **模型與 Prompt 版本**：`{LLM_CFG.model}` / `{prompt_ver}`（OpenAI Batch API 五折價）",
        f"- **Pilot 有效筆數**：`{len(df_pilot)}` 場（涵蓋 `{df_pilot['symbol'].nunique()}` 檔股票、`{df_pilot['gics_sector'].nunique()}` 個 GICS 產業、年份 `{int(df_pilot['year'].min())}–{int(df_pilot['year'].max())}`）",
        f"- **實際費用結算**：Pilot Batch `USD {cost_pilot:.4f}` + 重測 Batch `USD {retest_cost:.4f}` = **總計 `USD {cost_pilot + retest_cost:.4f}`**",
        f"- **依 200 場實測 Token 推估全量 {len(df_valid):,} 場 Batch 費用**：**`USD {proj_full_cost:.2f}`**（場均輸入 `{avg_in:,.1f}` / 輸出 `{avg_out:,.1f}` tokens）",
        "",
        "## 一、Pilot 200 場分數分佈檢驗（反分數集中檢查）",
        "",
        "| 指標 | Mean | Std | Min | P10 | Median | P90 | Max | 相異值數 |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for col in (
        "prepared_optimism",
        "qa_optimism",
        "optimism_score",
        "qa_tone_drop",
        "ambiguity_score",
        "qa_evasiveness",
        "guidance_revision",
    ):
        s = df_pilot[col].astype(float)
        lines.append(
            f"| `{col}` | {s.mean():.2f} | {s.std(ddof=1):.2f} | {s.min():g} | "
            f"{s.quantile(0.10):.1f} | {s.median():.1f} | {s.quantile(0.90):.1f} | {s.max():g} | **{int(s.nunique())}** |"
        )

    lines.extend(
        [
            "",
            "## 二、分年度均值變化（跨市場時期經濟直覺檢核）",
            "",
            "| Year | 場次 | Prep Opt | Q&A Opt | Tone Drop | Ambiguity | Evasiveness | Guidance |",
            "| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
    )
    by_yr = df_pilot.groupby("year", as_index=False).agg(
        n=("event_id", "count"),
        prep=("prepared_optimism", "mean"),
        qa=("qa_optimism", "mean"),
        drop=("qa_tone_drop", "mean"),
        amb=("ambiguity_score", "mean"),
        evas=("qa_evasiveness", "mean"),
        guid=("guidance_revision", "mean"),
    )
    for r in by_yr.itertuples(index=False):
        lines.append(
            f"| {int(r.year)} | {int(r.n)} | {r.prep:.2f} | {r.qa:.2f} | {r.drop:+.2f} | "
            f"{r.amb:.2f} | {r.evas:.2f} | {r.guid:+.2f} |"
        )

    lines.extend(["", "## 三、30 場重測信度檢定（Test-Retest Reliability）", ""])
    if retest_df is None:
        lines.append("> 尚未回收 `retest` Batch 結果。")
    else:
        lines.extend(
            [
                "| 指標 | 配對數 | Spearman ρ | Pearson r | MAE | Max |Diff| | 完全一致率 | 容差內一致率（±5 / ±1） |",
                "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
            ]
        )
        for r in retest_df.itertuples(index=False):
            sp_str = f"{r.spearman_rho:.4f}" if pd.notna(r.spearman_rho) else "N/A"
            pr_str = f"{r.pearson_r:.4f}" if pd.notna(r.pearson_r) else "N/A"
            lines.append(
                f"| `{r.metric}` | {r.n_pairs} | **{sp_str}** | {pr_str} | {r.mae:.2f} | "
                f"{r.max_abs_diff:g} | {r.exact_match_rate * 100:.1f}% | **{r.within_tol_rate * 100:.1f}%** |"
            )

    STAGE3_PILOT_AUDIT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"[Pilot Audit] 報告已產出：{STAGE3_PILOT_AUDIT_MD}")


def run_build_features() -> None:
    """整合快取評分與 clean 元資料，計算嚴格 Point-in-Time 跨季差分特徵並輸出 Parquet 與稽核報告。"""
    ensure_project_dirs()
    cached = load_cache_records()
    if not cached:
        raise RuntimeError(f"快取檔中沒有任何符合版本的紀錄：{API_EXTRACT_CACHE_PATH}")

    df_valid = load_valid_clean_transcripts()[
        ["event_id", "symbol", "gics_sector", "year", "quarter", "event_trading_date"]
    ]
    df_scores = pd.DataFrame(list(cached.values()))
    merged = df_valid.merge(df_scores, on="event_id", how="inner")
    if merged.empty:
        raise RuntimeError("快取與 clean 表無法對齊任何 event_id。")

    feat_df = compute_qoq_features(merged)
    assert feat_df["event_id"].is_unique, "llm_features 存在重複 event_id！"
    n_symbols = int(feat_df["symbol"].nunique())
    n_first_q_nans = int(feat_df["qoq_opt_diff"].isna().sum())
    assert n_first_q_nans == n_symbols, (
        f"跨季差分首季 NaN 數 ({n_first_q_nans}) 與公司數 ({n_symbols}) 不一致！"
    )

    atomic_write_parquet(feat_df, LLM_FEATURES_PATH)

    tot_in = int(feat_df["prompt_tokens"].sum())
    tot_out = int(feat_df["completion_tokens"].sum())
    tot_cost = estimate_cost_usd(tot_in, tot_out, pricing_mode="batch")
    coverage_pct = len(feat_df) / len(df_valid) * 100.0

    lines: List[str] = [
        f"# Stage 3A 全量 LLM 結構化語義特徵稽核報告（Prompt `{LLM_CFG.prompt_version}`）",
        "",
        f"- **產出時間 (UTC)**：`{datetime.now(timezone.utc).isoformat()}`",
        f"- **模型與 Prompt 版本**：`{LLM_CFG.model}` / `{LLM_CFG.prompt_version}`（OpenAI Batch API 五折價）",
        f"- **特徵表路徑**：`{LLM_FEATURES_PATH}`",
        f"- **覆蓋率**：`{len(feat_df):,} / {len(df_valid):,}` 場有效法說會（**{coverage_pct:.2f}%**），涵蓋 `{n_symbols}` 檔股票、`{feat_df['gics_sector'].nunique()}` 個 GICS 產業、年份 `{int(feat_df['year'].min())}–{int(feat_df['year'].max())}`",
        f"- **Token 與總費用結算**：總輸入 `{tot_in:,}` / 總輸出 `{tot_out:,}` tokens，全量 Batch 實付費用 **`USD {tot_cost:.2f}`**",
        f"- **Point-in-Time 跨季差分檢查**：首季 `qoq_opt_diff` 缺失數 = `{n_first_q_nans}`（精確等於 `{n_symbols}` 檔股票，零跨公司污染）",
        "",
        "## 一、全量語義特徵分佈總覽",
        "",
        "| 特徵欄位 | Mean | Std | Min | P10 | Median | P90 | Max | 缺失率 |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for col in (
        "prepared_optimism",
        "qa_optimism",
        "optimism_score",
        "qa_tone_drop",
        "ambiguity_score",
        "qa_evasiveness",
        "guidance_revision",
        "qoq_opt_diff",
        "qoq_tone_drop_diff",
        "opt_vs_trailing_4q_mean",
    ):
        s = feat_df[col].astype(float)
        na_pct = s.isna().mean() * 100.0
        lines.append(
            f"| `{col}` | {s.mean():.2f} | {s.std(ddof=1):.2f} | {s.min():g} | "
            f"{s.quantile(0.10):.1f} | {s.median():.1f} | {s.quantile(0.90):.1f} | {s.max():g} | {na_pct:.2f}% |"
        )

    lines.extend(
        [
            "",
            "## 二、歷年語義情緒演變（2016–2025 總經週期檢核）",
            "",
            "| Year | 場次 | Prep Opt | Q&A Opt | Tone Drop | Ambiguity | Evasiveness | Guidance |",
            "| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
    )
    by_yr = feat_df.groupby("year", as_index=False).agg(
        n=("event_id", "count"),
        prep=("prepared_optimism", "mean"),
        qa=("qa_optimism", "mean"),
        drop=("qa_tone_drop", "mean"),
        amb=("ambiguity_score", "mean"),
        evas=("qa_evasiveness", "mean"),
        guid=("guidance_revision", "mean"),
    )
    for r in by_yr.itertuples(index=False):
        lines.append(
            f"| {int(r.year)} | {int(r.n)} | {r.prep:.2f} | {r.qa:.2f} | {r.drop:+.2f} | "
            f"{r.amb:.2f} | {r.evas:.2f} | {r.guid:+.2f} |"
        )

    lines.extend(
        [
            "",
            "## 三、GICS 11 產業橫斷面語義均值",
            "",
            "| GICS Sector | 場次 | Prep Opt | Q&A Opt | Tone Drop | Ambiguity | Evasiveness | Guidance |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
    )
    by_sec = (
        feat_df.groupby("gics_sector", as_index=False)
        .agg(
            n=("event_id", "count"),
            prep=("prepared_optimism", "mean"),
            qa=("qa_optimism", "mean"),
            drop=("qa_tone_drop", "mean"),
            amb=("ambiguity_score", "mean"),
            evas=("qa_evasiveness", "mean"),
            guid=("guidance_revision", "mean"),
        )
        .sort_values(by="prep", ascending=False)
    )
    for r in by_sec.itertuples(index=False):
        lines.append(
            f"| {r.gics_sector} | {int(r.n)} | {r.prep:.2f} | {r.qa:.2f} | {r.drop:+.2f} | "
            f"{r.amb:.2f} | {r.evas:.2f} | {r.guid:+.2f} |"
        )

    STAGE3_AUDIT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"[Build] 已產出 {len(feat_df):,} 筆 LLM 語義特徵 -> {LLM_FEATURES_PATH}")
    print(f"[Build] 全量稽核報告已寫入 -> {STAGE3_AUDIT_MD}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 3A OpenAI 法說會結構化語義抽取與 Batch API 工具")
    parser.add_argument(
        "--mode",
        choices=("smoke", "prepare", "submit", "status", "collect", "pilot_audit", "build"),
        default="smoke",
        help="執行模式（預設 smoke：即時測試 5 家公司）",
    )
    parser.add_argument("--n", type=int, default=LLM_CFG.smoke_test_count, help="smoke 模式測試公司數（預設 5）")
    parser.add_argument(
        "--scope",
        choices=("pilot", "retest", "full"),
        default="pilot",
        help="Batch 模式範圍（pilot / retest / full）",
    )
    parser.add_argument("--max-shards", type=int, default=0, help="submit 模式單次最多上傳幾個分片（0=全部）")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "smoke":
        run_smoke_test(n=args.n)
    elif args.mode == "prepare":
        run_prepare_batches(scope=args.scope)
    elif args.mode == "submit":
        run_submit_batches(scope=args.scope, max_shards=args.max_shards)
    elif args.mode == "status":
        run_check_status()
    elif args.mode == "collect":
        run_collect_batches()
    elif args.mode == "pilot_audit":
        run_pilot_audit()
    elif args.mode == "build":
        run_build_features()


if __name__ == "__main__":
    main()
