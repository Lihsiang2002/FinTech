"""Stage 3B 區段級 Embedding 切塊、池化與 Point-in-Time 語義偏移離線單元測試。"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

emb_mod = importlib.import_module("src.05_embed_drift")
chunk_text_by_budget = emb_mod.chunk_text_by_budget
pool_and_normalize_vectors = emb_mod.pool_and_normalize_vectors
cosine_distance = emb_mod.cosine_distance
build_embed_batch_shards = emb_mod.build_embed_batch_shards
compute_semantic_drift_features = emb_mod.compute_semantic_drift_features
estimate_embed_cost_usd = emb_mod.estimate_embed_cost_usd


def test_chunk_text_by_budget_zero_loss_and_limits() -> None:
    """驗證長文本切塊嚴格遵守字數與字元上限且零詞彙流失。"""
    assert chunk_text_by_budget("   ") == []
    words = [f"tok{i}" for i in range(120)]
    text = " ".join(words)
    chunks = chunk_text_by_budget(text, max_words=35, max_chars=250)
    assert len(chunks) >= 4
    for c in chunks:
        assert len(c.split()) <= 35
        assert len(c) <= 250
    reconstructed = " ".join(chunks).split()
    assert reconstructed == words


def test_pool_and_normalize_vectors_weighted_and_guards() -> None:
    """驗證 Token 加權池化、單位向量正規化與異常值防禦。"""
    v1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    v2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    # 權重 3:1 -> 方向 [3, 1, 0] / sqrt(10)
    pooled = pool_and_normalize_vectors([v1, v2], [300.0, 100.0], expected_dim=3)
    assert pooled.dtype == np.float32
    assert pytest.approx(float(np.linalg.norm(pooled)), rel=1e-6) == 1.0
    assert pytest.approx(float(pooled[0] / pooled[1]), rel=1e-6) == 3.0

    with pytest.raises(ValueError, match="維度異常"):
        pool_and_normalize_vectors([v1], [1.0], expected_dim=4)

    bad = np.array([1.0, np.nan, 0.0], dtype=np.float32)
    with pytest.raises(ValueError, match="NaN"):
        pool_and_normalize_vectors([bad], [1.0], expected_dim=3)


def test_build_embed_batch_shards_format_and_skip_done(tmp_path: Path) -> None:
    """驗證 /v1/embeddings Batch JSONL 格式、同場切塊同分片保證與斷點跳過。"""
    df = pd.DataFrame(
        [
            {"event_id": "E1", "prepared_remarks_anon": "alpha beta", "qa_session_anon": "gamma delta"},
            {"event_id": "E2", "prepared_remarks_anon": " ".join(["w"] * 50), "qa_session_anon": "q1 q2"},
            {"event_id": "E3", "prepared_remarks_anon": "p3", "qa_session_anon": "q3"},
        ]
    )
    shards, n_chunks = build_embed_batch_shards(
        records=df,
        out_dir=tmp_path / "emb",
        scope="smoke5",
        shard_calls=1,
        max_mb=10.0,
        done_ids={"E1"},
    )
    # E1 跳過，剩 E2、E3，每片 1 場 -> 2 片，每場 prep+qa 各 1 塊 -> 4 筆請求
    assert len(shards) == 2
    assert n_chunks == 4
    req_0 = [json.loads(ln) for ln in shards[0].read_text(encoding="utf-8").splitlines()]
    assert [r["custom_id"] for r in req_0] == ["E2::prep::c00", "E2::qa::c00"]
    assert req_0[0]["url"] == "/v1/embeddings"
    assert req_0[0]["body"]["encoding_format"] == "float"


def test_compute_semantic_drift_features_point_in_time_exact() -> None:
    """驗證跨季／跨年／質心偏移之餘弦距離精確度與零未來／跨公司洩漏。"""
    # 構造 3 維正交單位向量，方便手算精確檢核
    e0 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    e1 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    e2 = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    # 故意打亂輸入順序，確認函數內部強制按 (symbol, event_trading_date) 排序
    meta = pd.DataFrame(
        [
            {"event_id": "A2", "symbol": "AAA", "gics_sector": "IT", "year": 2024, "quarter": 2, "event_trading_date": "2024-05-01"},
            {"event_id": "B1", "symbol": "BBB", "gics_sector": "Energy", "year": 2024, "quarter": 1, "event_trading_date": "2024-02-01"},
            {"event_id": "A1", "symbol": "AAA", "gics_sector": "IT", "year": 2024, "quarter": 1, "event_trading_date": "2024-02-01"},
            {"event_id": "A3", "symbol": "AAA", "gics_sector": "IT", "year": 2024, "quarter": 3, "event_trading_date": "2024-08-01"},
            {"event_id": "A5", "symbol": "AAA", "gics_sector": "IT", "year": 2025, "quarter": 1, "event_trading_date": "2025-02-01"},
            {"event_id": "A4", "symbol": "AAA", "gics_sector": "IT", "year": 2024, "quarter": 4, "event_trading_date": "2024-11-01"},
        ]
    )
    prep_vecs = {"A1": e0, "A2": e1, "A3": e0, "A4": e0, "A5": e0, "B1": e2}
    qa_vecs = {"A1": e0, "A2": e0, "A3": e2, "A4": e2, "A5": e2, "B1": e2}

    out = compute_semantic_drift_features(meta, prep_vecs, qa_vecs)
    aaa = out[out["symbol"] == "AAA"].reset_index(drop=True)
    bbb = out[out["symbol"] == "BBB"].reset_index(drop=True)

    # 首季跨季必為 NaN，且 BBB 不受 AAA 影響
    assert pd.isna(aaa.loc[0, "qoq_prep_drift"])
    assert pd.isna(bbb.loc[0, "qoq_prep_drift"])
    assert bbb.loc[0, "prep_qa_divergence"] == 0.0

    # A1 prep=e0, qa=e0 -> div=0; A2 prep=e1, qa=e0 -> div=1.0, qoq_prep=1.0 (e1 vs e0)
    assert aaa.loc[0, "prep_qa_divergence"] == 0.0
    assert aaa.loc[1, "prep_qa_divergence"] == 1.0
    assert aaa.loc[1, "qoq_prep_drift"] == 1.0

    # A3 (idx 2): 過去 2 季質心 = norm(e0 + e1) = [1/sqrt2, 1/sqrt2, 0]，與 A3=e0 的 cos = 1/sqrt2
    assert pd.isna(aaa.loc[0, "prep_drift_vs_4q_centroid"])
    assert pd.isna(aaa.loc[1, "prep_drift_vs_4q_centroid"])
    assert pytest.approx(aaa.loc[2, "prep_drift_vs_4q_centroid"], abs=1e-5) == 1.0 - (1.0 / np.sqrt(2.0))

    # YoY: 前 4 季為 NaN，A5 (idx 4) vs A1 (idx 0) 皆為 e0 -> yoy_prep_drift = 0.0
    assert aaa.loc[:3, "yoy_prep_drift"].isna().all()
    assert aaa.loc[4, "yoy_prep_drift"] == 0.0


def test_estimate_embed_cost_batch_half_sync() -> None:
    assert pytest.approx(estimate_embed_cost_usd(1_000_000, "batch"), rel=1e-9) == 0.01
    assert pytest.approx(estimate_embed_cost_usd(1_000_000, "sync"), rel=1e-9) == 0.02
