"""Stage 1A：歷史法說會逐字稿原始分區倉儲建置（Layer 1 -> data/01_raw/transcripts_raw/）。

功能：
1. 串流載入 Hugging Face kurry/sp500_earnings_transcripts（啟用 Parquet 欄式裁剪，直接略過冗餘 content 欄位）。
2. 依據 config/universe.json 之股票池與更名映射過濾 year >= 2016 紀錄。
3. 執行雙重去重：(symbol, year, quarter) 主鍵去重 + 同標的 15 天內重複場次檢核。
4. 依年份分區以 Zstd 壓縮與原子替換寫入 data/01_raw/transcripts_raw/year=YYYY/part.parquet，
   並執行主鍵唯一性與覆蓋率 Fail-Fast 檢核。

執行範例：
    python -m src.01_etl_transcripts --pilot      # 先以 20 檔 Pilot 試跑至獨立目錄驗證
    python -m src.01_etl_transcripts              # 正式建置 120 檔全量 01_raw 分區庫
"""

from __future__ import annotations

import argparse
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import pyarrow as pa
from dotenv import load_dotenv

from src.config import (
    DATA_CFG,
    DATA_DIR,
    RAW_TRANSCRIPTS_DIR,
    ensure_project_dirs,
    load_universe,
)
from src.io_utils import atomic_write_parquet


# 嚴格統一所有年份分區之 Parquet Schema，確保 DuckDB 跨分區讀取零型別衝突
TRANSCRIPT_RAW_SCHEMA = pa.schema(
    [
        pa.field("event_id", pa.string(), nullable=False),
        pa.field("symbol", pa.string(), nullable=False),
        pa.field("raw_symbol", pa.string(), nullable=False),
        pa.field("company_name", pa.string(), nullable=True),
        pa.field("year", pa.int32(), nullable=False),
        pa.field("quarter", pa.int8(), nullable=False),
        pa.field("raw_date", pa.string(), nullable=False),
        pa.field(
            "structured_content",
            pa.list_(
                pa.struct(
                    [
                        pa.field("speaker", pa.string()),
                        pa.field("text", pa.string()),
                    ]
                )
            ),
            nullable=False,
        ),
        pa.field("source", pa.string(), nullable=False),
        pa.field("ingested_at", pa.string(), nullable=False),
    ]
)


def _normalize_structured_content(raw_segments: Any) -> List[Dict[str, str]]:
    """標準化 structured_content 陣列為固定鍵值 {'speaker', 'text'} 清單。"""
    if not isinstance(raw_segments, (list, tuple)):
        return []
    cleaned: List[Dict[str, str]] = []
    for seg in raw_segments:
        if not isinstance(seg, dict):
            continue
        speaker = str(seg.get("speaker", "") or "").strip()
        text = str(seg.get("text", seg.get("content", "")) or "").strip()
        if speaker or text:
            cleaned.append({"speaker": speaker, "text": text})
    return cleaned


def _record_quality_key(rec: Dict[str, Any]) -> Tuple[int, int, int]:
    """評估單筆紀錄完整度，用於重複場次取捨（優先非零時間 > 段落數 > 總字元數）。"""
    raw_date = str(rec.get("raw_date", "") or "")
    has_non_zero_time = int(
        ":" in raw_date and not raw_date.endswith("00:00:00")
    )
    segs = rec.get("structured_content") or []
    n_segs = len(segs)
    n_chars = sum(len(s.get("text", "")) for s in segs)
    return (has_non_zero_time, n_segs, n_chars)


def _parse_date_safe(raw_date: str) -> Optional[pd.Timestamp]:
    """安全解析日期字串，失敗時回傳 None。"""
    if not raw_date:
        return None
    ts = pd.to_datetime(raw_date, errors="coerce")
    if pd.isna(ts):
        return None
    if ts.tzinfo is not None:
        ts = ts.tz_localize(None)
    return ts


def deduplicate_transcript_records(
    records: Sequence[Dict[str, Any]],
    dedup_window_days: int = DATA_CFG.dedup_date_window_days,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """執行雙重去重：先依 (symbol, year, quarter)，再檢核同標的 15 天內重複場次。

    Returns:
        (保留紀錄清單, 被排除之重複紀錄摘要清單)
    """
    dropped: List[Dict[str, Any]] = []

    # 第一層：依 (symbol, year, quarter) 去重
    by_key: Dict[Tuple[str, int, int], Dict[str, Any]] = {}
    for rec in records:
        key = (str(rec["symbol"]), int(rec["year"]), int(rec["quarter"]))
        if key not in by_key:
            by_key[key] = rec
            continue
        existing = by_key[key]
        if _record_quality_key(rec) > _record_quality_key(existing):
            dropped.append(
                {
                    "reason": "duplicate_symbol_year_quarter",
                    "event_id": existing["event_id"],
                    "raw_date": existing["raw_date"],
                }
            )
            by_key[key] = rec
        else:
            dropped.append(
                {
                    "reason": "duplicate_symbol_year_quarter",
                    "event_id": rec["event_id"],
                    "raw_date": rec["raw_date"],
                }
            )

    # 第二層：同標的兩筆紀錄若日期差距 <= dedup_window_days 視為同一場次（防企業財年定義差異重複）
    by_symbol: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for rec in by_key.values():
        by_symbol[str(rec["symbol"])].append(rec)

    kept_all: List[Dict[str, Any]] = []
    for sym, sym_recs in sorted(by_symbol.items()):
        dated = [(_parse_date_safe(r["raw_date"]), r) for r in sym_recs]
        valid_dated = sorted(
            [(ts, r) for ts, r in dated if ts is not None],
            key=lambda x: x[0],
        )
        undated = [r for ts, r in dated if ts is None]

        kept_sym: List[Tuple[pd.Timestamp, Dict[str, Any]]] = []
        for ts, rec in valid_dated:
            if not kept_sym:
                kept_sym.append((ts, rec))
                continue
            prev_ts, prev_rec = kept_sym[-1]
            gap_days = abs((ts - prev_ts).total_seconds()) / 86400.0
            if gap_days <= dedup_window_days:
                if _record_quality_key(rec) > _record_quality_key(prev_rec):
                    dropped.append(
                        {
                            "reason": f"within_{dedup_window_days}d_duplicate",
                            "event_id": prev_rec["event_id"],
                            "raw_date": prev_rec["raw_date"],
                            "kept_event_id": rec["event_id"],
                        }
                    )
                    kept_sym[-1] = (ts, rec)
                else:
                    dropped.append(
                        {
                            "reason": f"within_{dedup_window_days}d_duplicate",
                            "event_id": rec["event_id"],
                            "raw_date": rec["raw_date"],
                            "kept_event_id": prev_rec["event_id"],
                        }
                    )
            else:
                kept_sym.append((ts, rec))

        kept_all.extend(r for _, r in kept_sym)
        kept_all.extend(undated)

    return kept_all, dropped


def write_partitioned_transcripts(
    records: Sequence[Dict[str, Any]],
    output_dir: Path,
) -> Dict[int, int]:
    """將逐字稿紀錄依年份分區以原子替換寫入 Parquet，回傳各年份寫入筆數。"""
    if not records:
        raise ValueError("無任何逐字稿紀錄可寫入，請檢查過濾條件或資料來源。")

    # Fail-Fast 主鍵唯一性檢查
    event_ids = [r["event_id"] for r in records]
    if len(event_ids) != len(set(event_ids)):
        raise ValueError("偵測到重複之 event_id，違反主鍵唯一性約束！")

    by_year: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for rec in records:
        by_year[int(rec["year"])].append(rec)

    year_counts: Dict[int, int] = {}
    for yr in sorted(by_year.keys()):
        yr_rows = sorted(by_year[yr], key=lambda r: (r["symbol"], r["quarter"]))
        table = pa.Table.from_pylist(yr_rows, schema=TRANSCRIPT_RAW_SCHEMA)
        out_path = output_dir / f"year={yr}" / "part.parquet"
        atomic_write_parquet(table, out_path)
        year_counts[yr] = len(yr_rows)

    return year_counts


def run_transcript_etl(
    pilot: bool = False,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """執行 Layer 1 歷史逐字稿串流擷取、雙重去重與分區寫入。"""
    from datasets import load_dataset

    load_dotenv()
    ensure_project_dirs()

    uni = load_universe()
    symbols_list: List[str] = list(uni["pilot_symbols"] if pilot else uni["symbols"])
    target_symbols = set(symbols_list)
    rename_map: Dict[str, str] = dict(uni["rename_map"])

    dest_dir = output_dir or (
        DATA_DIR / "01_raw_pilot" / "transcripts_raw" if pilot else RAW_TRANSCRIPTS_DIR
    )
    hf_token = os.getenv("HF_TOKEN") or None
    ingested_at = datetime.now().isoformat(timespec="seconds")

    mode_label = f"Pilot ({len(target_symbols)} 檔)" if pilot else f"全量 ({len(target_symbols)} 檔)"
    print(f"[Stage 1A] 啟動法說會逐字稿串流擷取 — 模式：{mode_label}", flush=True)
    print(f"[Stage 1A] 輸出目錄：{dest_dir}", flush=True)

    # 欄式裁剪：直接跳過冗餘 content 全文欄位以節省頻寬與記憶體
    use_cols = ["symbol", "company_name", "year", "quarter", "date", "structured_content"]
    ds = load_dataset(
        DATA_CFG.hf_dataset_id,
        split=DATA_CFG.hf_split,
        streaming=True,
        columns=use_cols,
        token=hf_token,
    )

    matched_records: List[Dict[str, Any]] = []
    scanned = 0
    for row in ds:
        scanned += 1
        year_val = row.get("year")
        if year_val is None or int(year_val) < DATA_CFG.start_year:
            continue

        raw_sym = str(row.get("symbol", "") or "").strip().upper()
        norm_sym = rename_map.get(raw_sym, raw_sym)
        if norm_sym not in target_symbols:
            continue

        yr = int(year_val)
        qtr = int(row["quarter"])
        sc = _normalize_structured_content(row.get("structured_content"))
        matched_records.append(
            {
                "event_id": f"{norm_sym}_{yr}_Q{qtr}",
                "symbol": norm_sym,
                "raw_symbol": raw_sym,
                "company_name": str(row.get("company_name", "") or ""),
                "year": yr,
                "quarter": qtr,
                "raw_date": str(row.get("date", "") or "").strip(),
                "structured_content": sc,
                "source": DATA_CFG.hf_dataset_id,
                "ingested_at": ingested_at,
            }
        )

    print(
        f"[Stage 1A] 串流掃描完成：共掃描 {scanned:,} 筆，命中 {len(matched_records):,} 筆。",
        flush=True,
    )

    kept_records, dropped = deduplicate_transcript_records(matched_records)
    if dropped:
        print(f"[Stage 1A] 雙重去重排除 {len(dropped)} 筆重複場次：{dropped[:5]}", flush=True)

    # Fail-Fast 覆蓋率與品質檢查
    covered_symbols = {r["symbol"] for r in kept_records}
    missing_symbols = sorted(target_symbols - covered_symbols)
    assert not missing_symbols, f"目標股票池存在 0 筆紀錄之標的：{missing_symbols}"

    empty_sc = [r["event_id"] for r in kept_records if not r["structured_content"]]
    assert not empty_sc, f"偵測到空 structured_content 紀錄：{empty_sc[:5]}"

    year_counts = write_partitioned_transcripts(kept_records, dest_dir)
    total_written = sum(year_counts.values())
    print(f"[Stage 1A] 寫入完成！共 {total_written:,} 筆，跨 {len(year_counts)} 個年份分區：", flush=True)
    for yr, cnt in year_counts.items():
        print(f"  - year={yr}: {cnt:,} 筆", flush=True)

    return {
        "mode": "pilot" if pilot else "full",
        "output_dir": str(dest_dir),
        "scanned_rows": scanned,
        "matched_rows": len(matched_records),
        "written_rows": total_written,
        "dropped_duplicates": len(dropped),
        "covered_symbols": len(covered_symbols),
        "year_counts": year_counts,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 1A：歷史法說會逐字稿串流建置")
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="僅擷取 20 檔 Pilot 標的至 data/01_raw_pilot/transcripts_raw 進行試跑驗證",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_transcript_etl(pilot=args.pilot)
