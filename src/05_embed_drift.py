"""Stage 3B：OpenAI Batch 區段級金融語義 Embedding 與跨季語義偏移特徵。

嚴格遵守 AGENTS.md：
1. 僅讀取 data/02_clean/transcripts_clean 中已去識別化之 *_anon 文本，杜絕實體記憶與未來洩漏。
2. 針對超過 text-embedding-3-small 8,191-token 上限的長區段，採用零截斷切塊 + Token 加權平均池化 + L2 正規化。
3. 透過 OpenAI Batch API (/v1/embeddings 五折價) 執行，並以 SQLite + 分片 Parquet 實現斷點續傳。
4. 嚴格按 (symbol, event_trading_date) 計算 Point-in-Time 跨季／跨年／過去四季質心語義偏移。
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.config import (
    CLEAN_TRANSCRIPTS_DIR,
    EMBED_BATCH_WORKDIR,
    EMBED_CACHE_DIR,
    EMBED_CFG,
    EMBED_DRIFT_FEATURES_PATH,
    LLM_FEATURES_PATH,
    PIPELINE_STATE_DB_PATH,
    SECTION_EMBEDDINGS_PATH,
    STAGE3B_AUDIT_MD,
    STAGE3B_SMOKE_MD,
    ensure_project_dirs,
    load_project_env,
)
from src.io_utils import atomic_write_parquet


# ==================== 切塊、池化與向量數學工具 ====================
def chunk_text_by_budget(
    text: str,
    max_words: int = EMBED_CFG.max_chunk_words,
    max_chars: int = EMBED_CFG.max_chunk_chars,
) -> List[str]:
    """將長文本安全切分為不超過字數與字元預算的非重疊區塊（零截斷，確保低於 8,191 token）。"""
    cleaned = (text or "").strip()
    if not cleaned:
        return []
    if len(cleaned) <= max_chars and len(cleaned.split()) <= max_words:
        return [cleaned]

    words = cleaned.split()
    chunks: List[str] = []
    cur_words: List[str] = []
    cur_chars = 0

    for w in words:
        # 若單一 token 極長，先按字元強制截分避免單詞爆預算
        if len(w) > max_chars:
            if cur_words:
                chunks.append(" ".join(cur_words))
                cur_words, cur_chars = [], 0
            for start in range(0, len(w), max_chars):
                chunks.append(w[start : start + max_chars])
            continue

        add_len = len(w) + (1 if cur_words else 0)
        if cur_words and (len(cur_words) >= max_words or cur_chars + add_len > max_chars):
            chunks.append(" ".join(cur_words))
            cur_words = [w]
            cur_chars = len(w)
        else:
            cur_words.append(w)
            cur_chars += add_len

    if cur_words:
        chunks.append(" ".join(cur_words))
    return chunks


def pool_and_normalize_vectors(
    vectors: Sequence[np.ndarray],
    weights: Sequence[float],
    expected_dim: int = EMBED_CFG.dimensions,
) -> np.ndarray:
    """以 Token 數為權重對多段切塊向量進行加權平均池化，並 L2 正規化為 float32 單位向量。"""
    if not vectors or len(vectors) != len(weights):
        raise ValueError("vectors 與 weights 不可為空且長度必須相等。")
    arr = np.asarray(vectors, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != expected_dim:
        raise ValueError(f"向量維度異常：預期 (*, {expected_dim})，實際為 {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("Embedding 向量含 NaN 或 Inf！")

    w = np.asarray(weights, dtype=np.float64)
    if np.any(w <= 0):
        w = np.ones(len(vectors), dtype=np.float64)
    pooled = np.average(arr, axis=0, weights=w)
    norm = float(np.linalg.norm(pooled))
    if norm <= 1e-12:
        raise ValueError("池化後向量範數接近 0，無法正規化。")
    return (pooled / norm).astype(np.float32)


def cosine_distance(v1: Optional[np.ndarray], v2: Optional[np.ndarray]) -> float:
    """計算兩單位向量間的餘弦距離（1 - cos_sim）；任一為 None 則回傳 NaN。"""
    if v1 is None or v2 is None:
        return float("nan")
    sim = float(np.dot(v1.astype(np.float64), v2.astype(np.float64)))
    sim = max(-1.0, min(1.0, sim))
    return round(1.0 - sim, 6)


def combine_section_vectors(
    v_prep: Optional[np.ndarray],
    v_qa: Optional[np.ndarray],
) -> Optional[np.ndarray]:
    """將 Prepared 與 Q&A 單位向量等權合成為整場法說會單位向量。"""
    if v_prep is None and v_qa is None:
        return None
    if v_prep is None:
        return v_qa
    if v_qa is None:
        return v_prep
    return pool_and_normalize_vectors([v_prep, v_qa], [1.0, 1.0], expected_dim=len(v_prep))


def estimate_embed_cost_usd(total_tokens: int, pricing_mode: str = "batch") -> float:
    """估算 text-embedding-3-small 費用（batch 五折價或 sync 原價）。"""
    rate = EMBED_CFG.batch_cost_per_1m if pricing_mode == "batch" else EMBED_CFG.sync_cost_per_1m
    return total_tokens * rate / 1_000_000.0


# ==================== Point-in-Time 語義偏移特徵計算 ====================
def compute_semantic_drift_features(
    meta_df: pd.DataFrame,
    prep_vecs: Dict[str, np.ndarray],
    qa_vecs: Dict[str, np.ndarray],
) -> pd.DataFrame:
    """嚴格按 (symbol, event_trading_date, event_id) 計算 8 項 Point-in-Time 語義偏移特徵。"""
    ordered = meta_df.sort_values(by=["symbol", "event_trading_date", "event_id"]).reset_index(drop=True)
    out_rows: List[Dict[str, Any]] = []

    for sym, grp in ordered.groupby("symbol", sort=False):
        hist_prep: List[Optional[np.ndarray]] = []
        hist_qa: List[Optional[np.ndarray]] = []
        hist_full: List[Optional[np.ndarray]] = []

        for row in grp.itertuples(index=False):
            eid = str(row.event_id)
            vp = prep_vecs.get(eid)
            vq = qa_vecs.get(eid)
            vf = combine_section_vectors(vp, vq)

            # 1. 單場內 Prepared vs Q&A 語義背離
            prep_qa_div = cosine_distance(vp, vq)

            # 2. 跨季（QoQ: t vs t-1）語義偏移（嚴格僅看前 1 季）
            prev_p = hist_prep[-1] if hist_prep else None
            prev_q = hist_qa[-1] if hist_qa else None
            prev_f = hist_full[-1] if hist_full else None
            qoq_prep = cosine_distance(vp, prev_p)
            qoq_qa = cosine_distance(vq, prev_q)
            qoq_full = cosine_distance(vf, prev_f)

            # 3. 跨年同季（YoY: t vs t-4）語義偏移（控制季節性）
            yoy_p = hist_prep[-4] if len(hist_prep) >= 4 else None
            yoy_q = hist_qa[-4] if len(hist_qa) >= 4 else None
            yoy_prep = cosine_distance(vp, yoy_p)
            yoy_qa = cosine_distance(vq, yoy_q)

            # 4. 相對於「過去 2~4 季語義質心」之偏移（嚴格排除當季與未來）
            past_4_p = [v for v in hist_prep[-4:] if v is not None]
            if vp is not None and len(past_4_p) >= 2:
                c_p = pool_and_normalize_vectors(past_4_p, [1.0] * len(past_4_p), expected_dim=len(vp))
                drift_c4_p = cosine_distance(vp, c_p)
            else:
                drift_c4_p = float("nan")

            past_4_q = [v for v in hist_qa[-4:] if v is not None]
            if vq is not None and len(past_4_q) >= 2:
                c_q = pool_and_normalize_vectors(past_4_q, [1.0] * len(past_4_q), expected_dim=len(vq))
                drift_c4_q = cosine_distance(vq, c_q)
            else:
                drift_c4_q = float("nan")

            out_rows.append(
                {
                    "event_id": eid,
                    "symbol": str(sym),
                    "gics_sector": str(row.gics_sector),
                    "year": int(row.year),
                    "quarter": int(row.quarter),
                    "event_trading_date": str(row.event_trading_date),
                    "prep_qa_divergence": prep_qa_div,
                    "qoq_prep_drift": qoq_prep,
                    "qoq_qa_drift": qoq_qa,
                    "qoq_full_drift": qoq_full,
                    "yoy_prep_drift": yoy_prep,
                    "yoy_qa_drift": yoy_qa,
                    "prep_drift_vs_4q_centroid": drift_c4_p,
                    "qa_drift_vs_4q_centroid": drift_c4_q,
                }
            )

            hist_prep.append(vp)
            hist_qa.append(vq)
            hist_full.append(vf)

    result = pd.DataFrame(out_rows)
    assert result["event_id"].is_unique, "embed_drift_features 存在重複 event_id！"
    return result


# ==================== 資料讀取、快取與 Batch 分片 ====================
def load_valid_clean_transcripts(clean_dir: Path = CLEAN_TRANSCRIPTS_DIR) -> pd.DataFrame:
    """透過 DuckDB 僅讀取 is_valid=True 的去識別化法說會欄位（嚴禁 *_raw）。"""
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
    assert df["event_id"].is_unique, "clean 表存在重複 event_id！"
    return df


def load_cached_embed_event_ids(cache_dir: Path = EMBED_CACHE_DIR, model: str = EMBED_CFG.model) -> Set[str]:
    """掃描 embed_sections 分片快取目錄，回傳已完成快取之 event_id 集合。"""
    if not cache_dir.exists():
        return set()
    done: Set[str] = set()
    for pf in sorted(cache_dir.glob("*.parquet")):
        tbl = pq.read_table(pf, columns=["event_id", "model"])
        df = tbl.to_pandas()
        done.update(df.loc[df["model"] == model, "event_id"].astype(str).tolist())
    return done


def load_all_cached_section_embeddings(
    cache_dir: Path = EMBED_CACHE_DIR,
    model: str = EMBED_CFG.model,
) -> pd.DataFrame:
    """讀取所有已快取之區段級 Embedding 紀錄（依 event_id + section 去重保留最新）。"""
    files = sorted(cache_dir.glob("*.parquet"))
    if not files:
        return pd.DataFrame()
    dfs = [pq.read_table(pf).to_pandas() for pf in files]
    df = pd.concat(dfs, ignore_index=True)
    df = df[df["model"] == model].drop_duplicates(subset=["event_id", "section"], keep="last")
    return df.reset_index(drop=True)


def init_embed_state_db(db_path: Path = PIPELINE_STATE_DB_PATH) -> sqlite3.Connection:
    """初始化 Embedding Batch 任務狀態表（共用 pipeline_state.sqlite）。"""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS openai_embed_batches (
            batch_id TEXT PRIMARY KEY,
            input_file_id TEXT NOT NULL,
            output_file_id TEXT,
            error_file_id TEXT,
            shard_file TEXT NOT NULL,
            scope TEXT NOT NULL,
            model TEXT NOT NULL,
            n_calls INTEGER NOT NULL,
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


def build_embed_batch_shards(
    records: pd.DataFrame,
    out_dir: Path,
    scope: str,
    shard_calls: int = EMBED_CFG.batch_shard_calls,
    max_mb: float = EMBED_CFG.max_batch_file_mb,
    done_ids: Optional[Set[str]] = None,
    model: str = EMBED_CFG.model,
) -> Tuple[List[Path], int]:
    """將待處理法說會之 Prepared / Q&A 切塊並輸出為 OpenAI Batch /v1/embeddings JSONL 分片。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    skip_ids = done_ids or set()
    max_bytes = int(max_mb * 1024 * 1024)

    shard_paths: List[Path] = []
    shard_idx = 0
    buf_lines: List[str] = []
    buf_bytes = 0
    buf_calls = 0
    total_chunks = 0

    def _flush() -> None:
        nonlocal shard_idx, buf_lines, buf_bytes, buf_calls
        if not buf_lines:
            return
        shard_idx += 1
        sp = out_dir / f"embed_{scope}_shard{shard_idx:03d}.jsonl"
        with sp.open("w", encoding="utf-8") as f:
            f.writelines(buf_lines)
        shard_paths.append(sp)
        buf_lines = []
        buf_bytes = 0
        buf_calls = 0

    for row in records.itertuples(index=False):
        eid = str(row.event_id)
        if eid in skip_ids:
            continue

        call_lines: List[str] = []
        for sec_tag, raw_text in (
            ("prep", str(row.prepared_remarks_anon or "")),
            ("qa", str(row.qa_session_anon or "")),
        ):
            chunks = chunk_text_by_budget(raw_text)
            for c_idx, chunk_txt in enumerate(chunks):
                req = {
                    "custom_id": f"{eid}::{sec_tag}::c{c_idx:02d}",
                    "method": "POST",
                    "url": EMBED_CFG.batch_endpoint,
                    "body": {
                        "model": model,
                        "input": chunk_txt,
                        "encoding_format": "float",
                    },
                }
                call_lines.append(json.dumps(req, ensure_ascii=False) + "\n")

        if not call_lines:
            continue

        call_bytes = sum(len(ln.encode("utf-8")) for ln in call_lines)
        # 確保同一場法說會的所有切塊必定落在同一個分片內，方便分片回收時即時完整池化
        if buf_lines and (buf_calls >= shard_calls or buf_bytes + call_bytes > max_bytes):
            _flush()
        buf_lines.extend(call_lines)
        buf_bytes += call_bytes
        buf_calls += 1
        total_chunks += len(call_lines)

    _flush()
    return shard_paths, total_chunks


def create_openai_client() -> Any:
    """載入 config.env / .env 並建立 OpenAI Client（嚴禁輸出金鑰）。"""
    load_project_env(override=False)
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("找不到 OPENAI_API_KEY，請確認 config.env 或 .env 設定。")
    from openai import OpenAI

    return OpenAI(api_key=api_key, timeout=90.0)


# ==================== CLI 流程函式 ====================
def _select_scope_records(df_valid: pd.DataFrame, scope: str) -> pd.DataFrame:
    if scope == "smoke5":
        syms = set(EMBED_CFG.smoke_symbols)
        sub = df_valid[df_valid["symbol"].isin(syms)].copy()
        if sub.empty:
            raise RuntimeError(f"找不到 smoke5 目標股票：{EMBED_CFG.smoke_symbols}")
        return sub.reset_index(drop=True)
    if scope == "full":
        return df_valid
    raise ValueError(f"不支援的 scope：{scope}")


def run_prepare(scope: str) -> None:
    """切分目標範圍之 Embedding Batch JSONL 分片（自動跳過已快取 event_id）。"""
    ensure_project_dirs()
    df_valid = load_valid_clean_transcripts()
    target_df = _select_scope_records(df_valid, scope)
    done_ids = load_cached_embed_event_ids()

    shards, n_chunks = build_embed_batch_shards(
        records=target_df,
        out_dir=EMBED_BATCH_WORKDIR,
        scope=scope,
        done_ids=done_ids,
    )
    skipped = len(done_ids & set(target_df["event_id"]))
    print(
        f"[Embed Prepare] scope={scope}, 目標場數={len(target_df)}, 已快取跳過={skipped}, "
        f"總切塊請求={n_chunks:,}, 產生分片數={len(shards)}（目錄：{EMBED_BATCH_WORKDIR}）"
    )


def run_submit(scope: str, max_shards: int = 0) -> None:
    """上傳 Embedding Batch JSONL 分片至 OpenAI 並記錄至 SQLite。"""
    ensure_project_dirs()
    pattern = f"embed_{scope}_shard*.jsonl"
    shard_files = sorted(EMBED_BATCH_WORKDIR.glob(pattern))
    if not shard_files:
        raise FileNotFoundError(f"找不到分片檔（{pattern}），請先執行 --mode prepare --scope {scope}")

    conn = init_embed_state_db()
    submitted = {
        r[0]
        for r in conn.execute(
            "SELECT shard_file FROM openai_embed_batches WHERE status NOT IN ('failed', 'expired', 'cancelled')"
        ).fetchall()
    }
    pending = [p for p in shard_files if p.name not in submitted]
    if max_shards > 0:
        pending = pending[:max_shards]
    if not pending:
        print(f"[Embed Submit] scope={scope} 所有分片皆已送出。")
        conn.close()
        return

    client = create_openai_client()
    now_iso = datetime.now(timezone.utc).isoformat()
    for p in pending:
        eids: Set[str] = set()
        n_req = 0
        with p.open("r", encoding="utf-8") as f:
            for ln in f:
                if ln.strip():
                    n_req += 1
                    cid = json.loads(ln)["custom_id"]
                    eids.add(cid.split("::")[0])
        try:
            with p.open("rb") as fh:
                uploaded = client.files.create(file=fh, purpose="batch")
            batch = client.batches.create(
                input_file_id=uploaded.id,
                endpoint=EMBED_CFG.batch_endpoint,
                completion_window=EMBED_CFG.batch_completion_window,
                metadata={"stage": "3b_embed", "scope": scope, "shard_file": p.name},
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[Embed Submit] 上傳 {p.name} 時發生錯誤（前序分片已保留）：{exc}")
            break

        conn.execute(
            """
            INSERT OR REPLACE INTO openai_embed_batches
            (batch_id, input_file_id, output_file_id, error_file_id, shard_file, scope,
             model, n_calls, n_requests, status, collected, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?)
            """,
            (
                batch.id,
                uploaded.id,
                batch.output_file_id,
                batch.error_file_id,
                p.name,
                scope,
                EMBED_CFG.model,
                len(eids),
                n_req,
                batch.status,
                now_iso,
                now_iso,
            ),
        )
        conn.commit()
        print(f"[Embed Submit] {p.name} ({len(eids)} calls / {n_req} chunks) -> batch_id={batch.id}, status={batch.status}")
    conn.close()


def run_status() -> None:
    """查詢未回收的 Embedding Batch 任務狀態並更新 SQLite。"""
    conn = init_embed_state_db()
    rows = conn.execute(
        "SELECT batch_id, shard_file, scope, n_calls, n_requests, status, collected "
        "FROM openai_embed_batches ORDER BY created_at"
    ).fetchall()
    if not rows:
        print("[Embed Status] 目前沒有任何已送出的 Embedding Batch 任務。")
        conn.close()
        return

    client = create_openai_client()
    now_iso = datetime.now(timezone.utc).isoformat()
    summary: Dict[str, int] = {"collected": 0, "completed_pending": 0, "in_progress": 0, "validating": 0, "failed": 0}
    for batch_id, shard_file, scope, n_calls, n_req, old_status, collected in rows:
        if collected:
            summary["collected"] += 1
            continue
        if old_status in ("failed", "expired", "cancelled"):
            summary["failed"] += 1
            continue

        b = client.batches.retrieve(batch_id)
        conn.execute(
            "UPDATE openai_embed_batches SET status=?, output_file_id=?, error_file_id=?, updated_at=? WHERE batch_id=?",
            (b.status, b.output_file_id, b.error_file_id, now_iso, batch_id),
        )
        counts = getattr(b, "request_counts", None)
        done_cnt = int(getattr(counts, "completed", 0) or 0) if counts else 0
        fail_cnt = int(getattr(counts, "failed", 0) or 0) if counts else 0

        if b.status == "completed":
            summary["completed_pending"] += 1
        elif b.status in ("in_progress", "finalizing"):
            summary["in_progress"] += 1
        elif b.status == "validating":
            summary["validating"] += 1
        elif b.status in ("failed", "expired", "cancelled"):
            summary["failed"] += 1

        err_note = f" | errors={b.errors}" if (b.status == "failed" and getattr(b, "errors", None)) else ""
        print(
            f"[Embed Status] {batch_id} | {scope} | {shard_file} ({n_calls} calls) | status={b.status} | "
            f"chunks={done_cnt}/{n_req}, failed={fail_cnt}{err_note}"
        )
    conn.commit()
    conn.close()
    print(
        f"[Embed Status Summary] collected={summary['collected']}, completed_pending={summary['completed_pending']}, "
        f"in_progress={summary['in_progress']}, validating={summary['validating']}, failed={summary['failed']}"
    )


def run_collect() -> None:
    """回收已完成的 Embedding Batch，執行 Token 加權平均池化與 L2 正規化後原子寫入分片 Parquet。"""
    ensure_project_dirs()
    conn = init_embed_state_db()
    rows = conn.execute(
        """
        SELECT batch_id, output_file_id, error_file_id, shard_file, scope, model
        FROM openai_embed_batches
        WHERE status = 'completed' AND collected = 0 AND output_file_id IS NOT NULL
        """
    ).fetchall()
    if not rows:
        print("[Embed Collect] 沒有待回收的已完成任務（請先執行 --mode status）。")
        conn.close()
        return

    client = create_openai_client()
    for batch_id, output_file_id, error_file_id, shard_file, scope, model in rows:
        raw_text = client.files.content(output_file_id).text
        grouped: Dict[Tuple[str, str], List[Tuple[int, np.ndarray, int]]] = defaultdict(list)
        err_cnt = 0

        for line in raw_text.splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            resp = item.get("response") or {}
            if resp.get("status_code") != 200:
                err_cnt += 1
                continue
            cid = str(item["custom_id"])
            eid, sec_tag, chunk_str = cid.split("::")
            c_idx = int(chunk_str.lstrip("c"))
            body = resp["body"]
            vec = np.asarray(body["data"][0]["embedding"], dtype=np.float32)
            toks = int((body.get("usage") or {}).get("prompt_tokens", 1) or 1)
            grouped[(eid, sec_tag)].append((c_idx, vec, toks))

        if error_file_id:
            err_text = client.files.content(error_file_id).text
            err_cnt += sum(1 for ln in err_text.splitlines() if ln.strip())

        section_rows: List[Dict[str, Any]] = []
        for (eid, sec_tag), items in sorted(grouped.items()):
            items.sort(key=lambda x: x[0])
            vecs = [x[1] for x in items]
            toks = [x[2] for x in items]
            pooled = pool_and_normalize_vectors(vecs, toks)
            section_rows.append(
                {
                    "event_id": eid,
                    "section": sec_tag,
                    "model": model,
                    "batch_id": batch_id,
                    "n_chunks": len(items),
                    "total_tokens": int(sum(toks)),
                    "embedding": pooled.tolist(),
                }
            )

        out_path = EMBED_CACHE_DIR / f"{batch_id}.parquet"
        atomic_write_parquet(pd.DataFrame(section_rows), out_path)

        conn.execute(
            "UPDATE openai_embed_batches SET collected = 1, updated_at = ? WHERE batch_id = ?",
            (datetime.now(timezone.utc).isoformat(), batch_id),
        )
        conn.commit()
        n_eids = len({r["event_id"] for r in section_rows})
        tot_tok = sum(r["total_tokens"] for r in section_rows)
        print(
            f"[Embed Collect] {batch_id} ({scope} / {shard_file}) 回收完成："
            f"{n_eids} 場（{len(section_rows)} 區段向量，{tot_tok:,} tokens，失敗 {err_cnt} 筆）-> {out_path.name}"
        )
    conn.close()


def _spearman(s1: pd.Series, s2: pd.Series) -> float:
    both = pd.concat([s1, s2], axis=1).dropna()
    if len(both) < 5 or both.iloc[:, 0].nunique() < 2 or both.iloc[:, 1].nunique() < 2:
        return float("nan")
    return float(both.iloc[:, 0].rank().corr(both.iloc[:, 1].rank()))


def run_report(scope: str) -> None:
    """彙整區段 Embedding、計算跨季偏移特徵並輸出評估報告（full 模式同步寫出 Parquet 特徵表）。"""
    ensure_project_dirs()
    df_valid = load_valid_clean_transcripts()[
        ["event_id", "symbol", "gics_sector", "company_name", "year", "quarter", "event_trading_date"]
    ]
    target_meta = _select_scope_records(df_valid, scope)
    sec_df = load_all_cached_section_embeddings()
    if sec_df.empty:
        raise RuntimeError(f"找不到任何已快取的 Embedding 分片：{EMBED_CACHE_DIR}")

    sec_df = sec_df[sec_df["event_id"].isin(set(target_meta["event_id"]))].copy()
    prep_vecs: Dict[str, np.ndarray] = {}
    qa_vecs: Dict[str, np.ndarray] = {}
    for row in sec_df.itertuples(index=False):
        vec = np.asarray(row.embedding, dtype=np.float32)
        if row.section == "prep":
            prep_vecs[str(row.event_id)] = vec
        elif row.section == "qa":
            qa_vecs[str(row.event_id)] = vec

    covered_ids = set(prep_vecs.keys()) | set(qa_vecs.keys())
    covered_meta = target_meta[target_meta["event_id"].isin(covered_ids)].copy()
    drift_df = compute_semantic_drift_features(covered_meta, prep_vecs, qa_vecs)

    # 費用與 Token 統計
    tot_tokens = int(sec_df["total_tokens"].sum())
    tot_chunks = int(sec_df["n_chunks"].sum())
    n_calls = len(drift_df)
    avg_tokens = tot_tokens / n_calls
    actual_cost = estimate_embed_cost_usd(tot_tokens, pricing_mode="batch")
    proj_full_cost = estimate_embed_cost_usd(int(avg_tokens * len(df_valid)), pricing_mode="batch")

    # 關聯 Stage 3A LLM 特徵（檢驗正交性與極端事件解釋力）
    llm_df: Optional[pd.DataFrame] = None
    if LLM_FEATURES_PATH.exists():
        llm_df = pq.read_table(LLM_FEATURES_PATH).to_pandas()
        merged_3a = drift_df.merge(
            llm_df[
                [
                    "event_id",
                    "optimism_score",
                    "qa_tone_drop",
                    "ambiguity_score",
                    "qa_evasiveness",
                    "guidance_revision",
                    "brief_rationale",
                ]
            ],
            on="event_id",
            how="left",
        )
    else:
        merged_3a = drift_df.copy()

    # 若為 full 模式，寫出正式特徵表與精簡區段向量表
    if scope == "full":
        atomic_write_parquet(drift_df, EMBED_DRIFT_FEATURES_PATH)
        atomic_write_parquet(
            sec_df[["event_id", "section", "model", "n_chunks", "total_tokens", "embedding"]],
            SECTION_EMBEDDINGS_PATH,
        )
        print(f"[Embed Build] 已輸出偏移特徵表 -> {EMBED_DRIFT_FEATURES_PATH} ({len(drift_df):,} 筆)")
        print(f"[Embed Build] 已輸出區段向量表 -> {SECTION_EMBEDDINGS_PATH} ({len(sec_df):,} 筆)")

    # 撰寫 Markdown 評估報告
    title = (
        f"# Stage 3B 5 家公司 OpenAI Batch Embedding 效果評估報告（`{EMBED_CFG.model}`）"
        if scope == "smoke5"
        else f"# Stage 3B 全量區段級 Embedding 與跨季語義偏移稽核報告（`{EMBED_CFG.model}`）"
    )
    lines: List[str] = [
        title,
        "",
        f"- **產出時間 (UTC)**：`{datetime.now(timezone.utc).isoformat()}`",
        f"- **模型與維度**：`{EMBED_CFG.model}`（`{EMBED_CFG.dimensions}` 維，長文本零截斷切塊 + Token 加權池化 + L2 正規化）",
        f"- **覆蓋範圍**：`{n_calls:,} / {len(target_meta):,}` 場有效法說會（`{drift_df['symbol'].nunique()}` 家公司、`{int(drift_df['year'].min())}–{int(drift_df['year'].max())}` 年）",
        f"- **切塊與 Token 統計**：共 `{len(sec_df):,}` 個區段、`{tot_chunks:,}` 個切塊（平均每場 `{tot_chunks / n_calls:.2f}` 塊）、總計 `{tot_tokens:,}` tokens（場均 `{avg_tokens:,.1f}` tokens）",
        f"- **費用實測與全量推估（Batch 五折價 `USD 0.01 / 1M tokens`）**：",
        f"  - 本次實付費用：**`USD {actual_cost:.4f}`**",
        f"  - 推估全量 `{len(df_valid):,}` 場 Batch 總費用：**`USD {proj_full_cost:.3f}`**",
        "",
        "## 一、語義偏移特徵分佈總覽（餘弦距離：數值越大代表語義跳變越劇烈）",
        "",
        "| 特徵欄位 | Mean | Std | Min | P10 | Median | P90 | Max | 缺失率 |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    drift_cols = (
        "prep_qa_divergence",
        "qoq_prep_drift",
        "qoq_qa_drift",
        "qoq_full_drift",
        "yoy_prep_drift",
        "yoy_qa_drift",
        "prep_drift_vs_4q_centroid",
        "qa_drift_vs_4q_centroid",
    )
    for col in drift_cols:
        s = drift_df[col].astype(float)
        lines.append(
            f"| `{col}` | {s.mean():.4f} | {s.std(ddof=1):.4f} | {s.min():.4f} | "
            f"{s.quantile(0.10):.4f} | {s.median():.4f} | {s.quantile(0.90):.4f} | {s.max():.4f} | {s.isna().mean() * 100:.2f}% |"
        )

    # 年度偏移演變
    lines.extend(
        [
            "",
            "## 二、歷年語義偏移演變（檢驗危機與總經轉折偵測力）",
            "",
            "| Year | 場次 | Prep-Q&A 背離 | QoQ Prep Drift | QoQ Q&A Drift | YoY Prep Drift | vs 4Q 質心偏移 (Prep) |",
            "| :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
    )
    by_yr = drift_df.groupby("year", as_index=False).agg(
        n=("event_id", "count"),
        div=("prep_qa_divergence", "mean"),
        qp=("qoq_prep_drift", "mean"),
        qq=("qoq_qa_drift", "mean"),
        yp=("yoy_prep_drift", "mean"),
        c4=("prep_drift_vs_4q_centroid", "mean"),
    )
    for r in by_yr.itertuples(index=False):
        fmt = lambda v: f"{v:.4f}" if pd.notna(v) else "N/A"
        lines.append(
            f"| {int(r.year)} | {int(r.n)} | {fmt(r.div)} | {fmt(r.qp)} | {fmt(r.qq)} | {fmt(r.yp)} | {fmt(r.c4)} |"
        )

    # Top-10 最大跨季敘事劇變場次
    top10 = merged_3a.dropna(subset=["qoq_prep_drift"]).sort_values(by="qoq_prep_drift", ascending=False).head(10)
    lines.extend(
        [
            "",
            "## 三、Top 10 最大跨季語義跳變場次（`qoq_prep_drift` 異常事件偵測檢驗）",
            "",
            "| 排名 | Symbol | 財季 | 交易日 | QoQ Prep Drift | QoQ Q&A Drift | Prep-Q&A 背離 | Stage 3A Opt / Amb | Stage 3A 判讀摘要 |",
            "| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |",
        ]
    )
    for rank, r in enumerate(top10.itertuples(index=False), 1):
        opt_str = f"{r.optimism_score:.1f} / {int(r.ambiguity_score)}" if hasattr(r, "optimism_score") and pd.notna(r.optimism_score) else "N/A"
        rat_str = str(getattr(r, "brief_rationale", "") or "")[:90]
        lines.append(
            f"| {rank} | `{r.symbol}` | {int(r.year)} Q{int(r.quarter)} | {r.event_trading_date} "
            f"| **{r.qoq_prep_drift:.4f}** | {r.qoq_qa_drift:.4f} | {r.prep_qa_divergence:.4f} | {opt_str} | {rat_str} |"
        )

    # 5 家公司同公司 vs 跨公司餘弦相似度矩陣（驗證去識別化後仍保留產業與商業模式語義結構）
    if scope == "smoke5":
        syms_sorted = sorted(drift_df["symbol"].unique().tolist())
        sym_centroids: Dict[str, np.ndarray] = {}
        within_sims: Dict[str, float] = {}
        for sym in syms_sorted:
            eids = drift_df.loc[drift_df["symbol"] == sym, "event_id"].tolist()
            vecs = [combine_section_vectors(prep_vecs.get(e), qa_vecs.get(e)) for e in eids]
            vecs = [v for v in vecs if v is not None]
            mat = np.stack(vecs, axis=0).astype(np.float64)
            sim_mat = mat @ mat.T
            triu = sim_mat[np.triu_indices(len(vecs), k=1)]
            within_sims[sym] = float(triu.mean())
            sym_centroids[sym] = pool_and_normalize_vectors(vecs, [1.0] * len(vecs))

        lines.extend(
            [
                "",
                "## 四、去識別化後之公司間語義結構檢驗（對角線＝同公司跨季平均相似度，非對角＝跨公司質心相似度）",
                "",
                "| Symbol | " + " | ".join(f"`{s}`" for s in syms_sorted) + " |",
                "| :--- | " + " | ".join(":---:" for _ in syms_sorted) + " |",
            ]
        )
        for s1 in syms_sorted:
            cells = []
            for s2 in syms_sorted:
                if s1 == s2:
                    cells.append(f"**{within_sims[s1]:.3f}**")
                else:
                    cs = float(np.dot(sym_centroids[s1].astype(np.float64), sym_centroids[s2].astype(np.float64)))
                    cells.append(f"{cs:.3f}")
            lines.append(f"| `{s1}` | " + " | ".join(cells) + " |")

    # 與 Stage 3A LLM 情緒分數的 Spearman 相關係數（檢驗訊號正交性／增量價值）
    if llm_df is not None:
        lines.extend(
            [
                "",
                "## 五、Stage 3B 語義偏移 vs. Stage 3A LLM 情緒分數之正交性檢驗（Spearman ρ）",
                "",
                "| Stage 3B 偏移特徵 | vs `optimism_score` | vs `qa_tone_drop` | vs `ambiguity_score` | vs `qa_evasiveness` | vs `guidance_revision` |",
                "| :--- | :---: | :---: | :---: | :---: | :---: |",
            ]
        )
        for dcol in ("prep_qa_divergence", "qoq_prep_drift", "qoq_qa_drift", "yoy_prep_drift", "prep_drift_vs_4q_centroid"):
            vals = [
                f"{_spearman(merged_3a[dcol], merged_3a[lcol]):+.3f}"
                for lcol in ("optimism_score", "qa_tone_drop", "ambiguity_score", "qa_evasiveness", "guidance_revision")
            ]
            lines.append(f"| `{dcol}` | " + " | ".join(vals) + " |")

    out_md = STAGE3B_SMOKE_MD if scope == "smoke5" else STAGE3B_AUDIT_MD
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"[Embed Report] 報告已寫入 -> {out_md}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 3B OpenAI Batch Embedding 與跨季語義偏移工具")
    parser.add_argument(
        "--mode",
        choices=("prepare", "submit", "status", "collect", "report"),
        required=True,
        help="執行模式",
    )
    parser.add_argument(
        "--scope",
        choices=("smoke5", "full"),
        default="smoke5",
        help="目標範圍（smoke5=5家公司十年歷史；full=全量120家公司）",
    )
    parser.add_argument("--max-shards", type=int, default=0, help="submit 模式單次最多上傳幾個分片（0=全部）")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "prepare":
        run_prepare(scope=args.scope)
    elif args.mode == "submit":
        run_submit(scope=args.scope, max_shards=args.max_shards)
    elif args.mode == "status":
        run_status()
    elif args.mode == "collect":
        run_collect()
    elif args.mode == "report":
        run_report(scope=args.scope)


if __name__ == "__main__":
    main()
