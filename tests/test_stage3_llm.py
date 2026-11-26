"""Stage 3A OpenAI 結構化語義抽取模組（v2 0~100 尺度與 Prepared/Q&A 雙區段）離線單元測試。"""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path

import pandas as pd
import pytest

from src.config import LLM_CFG, load_project_env

llm_mod = importlib.import_module("src.04_llm_extract")
STRICT_RESPONSE_FORMAT = llm_mod.STRICT_RESPONSE_FORMAT
append_cache_record = llm_mod.append_cache_record
build_batch_jsonl_shards = llm_mod.build_batch_jsonl_shards
build_chat_messages = llm_mod.build_chat_messages
compute_qoq_features = llm_mod.compute_qoq_features
compute_retest_metrics = llm_mod.compute_retest_metrics
enrich_derived_scores = llm_mod.enrich_derived_scores
estimate_cost_usd = llm_mod.estimate_cost_usd
load_cache_records = llm_mod.load_cache_records
load_prompt_template = llm_mod.load_prompt_template
select_retest_records = llm_mod.select_retest_records
select_smoke_records = llm_mod.select_smoke_records
validate_llm_payload = llm_mod.validate_llm_payload


def test_load_project_env_parses_spaced_quoted_format(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """驗證 config.env 使用 `OPENAI_API_KEY = "my api key"`（含空格與雙引號）格式可正確解析。"""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    env_file = tmp_path / "config.env"
    env_file.write_text('OPENAI_API_KEY = "sk-unit-test-placeholder"\n', encoding="utf-8")

    loaded = load_project_env(candidates=(env_file,), override=True)
    assert loaded == env_file
    assert os.environ.get("OPENAI_API_KEY") == "sk-unit-test-placeholder"


def test_load_prompt_template_v2_valid() -> None:
    """驗證 config/prompts/extract_v2.yaml 結構完整且具備必要去識別化佔位符。"""
    cfg = load_prompt_template(LLM_CFG.prompt_path)
    assert cfg["version"] == "v2"
    assert "{prepared_remarks_anon}" in cfg["user_prompt_template"]
    assert "{qa_session_anon}" in cfg["user_prompt_template"]


def test_build_chat_messages_and_empty_guard() -> None:
    """驗證僅接收去識別化文本組裝訊息，且雙空文本立即 Fail-Fast。"""
    cfg = load_prompt_template(LLM_CFG.prompt_path)
    msgs = build_chat_messages(
        prepared_remarks_anon="[COMPANY] delivered solid revenue growth in [CURRENT_QUARTER].",
        qa_session_anon="[ANALYST]: What about margins? [EXECUTIVE]: Margins expanded.",
        prompt_cfg=cfg,
    )
    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
    assert "[COMPANY]" in msgs[1]["content"]

    with pytest.raises(ValueError, match="不可同時為空"):
        build_chat_messages("   ", "", cfg)


def test_validate_llm_payload_v2_bounds_and_derived_scores() -> None:
    """驗證 v2 JSON 0~100 邊界、bool 防護與衍生 qa_tone_drop / optimism_score 計算。"""
    valid = {
        "prepared_optimism": 72,
        "qa_optimism": 58,
        "ambiguity_score": 44,
        "qa_evasiveness": 63,
        "guidance_revision": 1,
        "optimistic_quote": "Demand remained strong across all regions.",
        "evasive_quote": "We will not break out specific comp numbers.",
        "brief_rationale": "Strong scripted confidence softened under analyst margin probes.",
    }
    cleaned = validate_llm_payload(valid)
    enriched = enrich_derived_scores(cleaned)
    assert enriched["optimism_score"] == 65.0
    assert enriched["qa_tone_drop"] == -14

    # 超界應立即報錯（0~100 與 -2~+2）
    with pytest.raises(ValueError, match="超出範圍"):
        validate_llm_payload({**valid, "prepared_optimism": 101})
    with pytest.raises(ValueError, match="超出範圍"):
        validate_llm_payload({**valid, "ambiguity_score": -1})
    with pytest.raises(ValueError, match="超出範圍"):
        validate_llm_payload({**valid, "guidance_revision": -3})

    # bool 不可混充 int
    with pytest.raises(ValueError, match="必須為整數"):
        validate_llm_payload({**valid, "qa_evasiveness": True})

    # 缺欄位與空引文應報錯
    missing = {k: v for k, v in valid.items() if k != "qa_optimism"}
    with pytest.raises(ValueError, match="缺少必要欄位"):
        validate_llm_payload(missing)

    with pytest.raises(ValueError, match="非空字串"):
        validate_llm_payload({**valid, "optimistic_quote": "   "})


def test_estimate_cost_batch_is_half_of_sync() -> None:
    """驗證 Batch API 計費為即時模式之精確五折。"""
    sync_c = estimate_cost_usd(1_000_000, 100_000, pricing_mode="sync")
    batch_c = estimate_cost_usd(1_000_000, 100_000, pricing_mode="batch")
    assert pytest.approx(batch_c, rel=1e-9) == sync_c * 0.5


def test_select_smoke_records_distinct_companies_and_sectors() -> None:
    """驗證 5 家公司測試挑選可覆蓋 5 家不同公司且優先覆蓋不同 GICS 產業。"""
    rows = [
        {"event_id": f"E{i}", "symbol": sym, "gics_sector": sec, "year": 2024, "event_trading_date": f"2024-08-0{i}"}
        for i, (sym, sec) in enumerate(
            [
                ("AAPL", "Information Technology"),
                ("AAPL", "Information Technology"),
                ("JPM", "Financials"),
                ("UNH", "Health Care"),
                ("XOM", "Energy"),
                ("PG", "Consumer Staples"),
                ("MSFT", "Information Technology"),
            ],
            start=1,
        )
    ]
    df = pd.DataFrame(rows)
    picked = select_smoke_records(df, n=5, pilot_symbols=["AAPL", "JPM", "UNH", "XOM", "PG"])
    assert len(picked) == 5
    assert picked["symbol"].nunique() == 5
    assert picked["gics_sector"].nunique() == 5


def test_batch_jsonl_sharding_and_cache_resume(tmp_path: Path) -> None:
    """驗證 Batch JSONL 格式正確性、分片切割與已快取斷點跳過機制。"""
    cfg = load_prompt_template(LLM_CFG.prompt_path)
    df = pd.DataFrame(
        [
            {
                "event_id": f"SYM_2024_Q{q}",
                "prepared_remarks_anon": f"[COMPANY] prepared remarks {q}.",
                "qa_session_anon": f"[ANALYST] question {q}.",
            }
            for q in (1, 2, 3, 4)
        ]
    )

    cache_path = tmp_path / "cache.jsonl"
    append_cache_record(
        cache_path,
        {
            "event_id": "SYM_2024_Q1",
            "model": LLM_CFG.model,
            "prompt_version": cfg["version"],
            "prepared_optimism": 68,
            "qa_optimism": 62,
        },
    )
    done_ids = set(load_cache_records(cache_path, prompt_version=cfg["version"]).keys())
    assert done_ids == {"SYM_2024_Q1"}

    shards = build_batch_jsonl_shards(
        records=df,
        prompt_cfg=cfg,
        out_dir=tmp_path / "batches",
        scope="pilot",
        shard_size=2,
        max_mb=10.0,
        done_ids=done_ids,
    )
    assert len(shards) == 2
    lines_0 = [json.loads(x) for x in shards[0].read_text(encoding="utf-8").splitlines()]
    lines_1 = [json.loads(x) for x in shards[1].read_text(encoding="utf-8").splitlines()]
    assert [x["custom_id"] for x in lines_0] == ["SYM_2024_Q2", "SYM_2024_Q3"]
    assert [x["custom_id"] for x in lines_1] == ["SYM_2024_Q4"]
    assert lines_0[0]["url"] == "/v1/chat/completions"
    assert lines_0[0]["body"]["response_format"] == STRICT_RESPONSE_FORMAT
    assert lines_0[0]["body"]["temperature"] == 0.0


def test_compute_qoq_features_point_in_time_no_leakage() -> None:
    """驗證 v2 跨季差分、語氣落差與滾動偏移嚴格按公司與時間隔離，無未來資訊洩漏。"""
    df = pd.DataFrame(
        [
            {"event_id": "A2", "symbol": "AAA", "event_trading_date": "2024-05-01", "prepared_optimism": 70, "qa_optimism": 66, "ambiguity_score": 40, "qa_evasiveness": 35, "guidance_revision": 1},
            {"event_id": "A1", "symbol": "AAA", "event_trading_date": "2024-02-01", "prepared_optimism": 50, "qa_optimism": 46, "ambiguity_score": 55, "qa_evasiveness": 50, "guidance_revision": 0},
            {"event_id": "A3", "symbol": "AAA", "event_trading_date": "2024-08-01", "prepared_optimism": 42, "qa_optimism": 30, "ambiguity_score": 68, "qa_evasiveness": 72, "guidance_revision": -1},
            {"event_id": "B1", "symbol": "BBB", "event_trading_date": "2024-03-01", "prepared_optimism": 60, "qa_optimism": 60, "ambiguity_score": 45, "qa_evasiveness": 40, "guidance_revision": 0},
        ]
    )
    out = compute_qoq_features(df)
    aaa = out[out["symbol"] == "AAA"].reset_index(drop=True)
    bbb = out[out["symbol"] == "BBB"].reset_index(drop=True)

    # 衍生欄位驗證：A1 opt=48 (drop=-4), A2 opt=68 (drop=-4), A3 opt=36 (drop=-12)
    assert list(aaa["optimism_score"]) == [48.0, 68.0, 36.0]
    assert list(aaa["qa_tone_drop"]) == [-4, -4, -12]

    # 首季差分必為 NaN，且不可跨公司污染
    assert pd.isna(aaa.loc[0, "qoq_opt_diff"])
    assert aaa.loc[1, "qoq_opt_diff"] == 20.0
    assert aaa.loc[2, "qoq_opt_diff"] == -32.0
    assert aaa.loc[2, "qoq_tone_drop_diff"] == -8.0
    assert pd.isna(bbb.loc[0, "qoq_opt_diff"])

    # A3 的 trailing 4q mean 僅看 (A1=48, A2=68) -> mean=58.0，故 36 - 58 = -22.0
    assert pd.isna(aaa.loc[0, "opt_vs_trailing_4q_mean"])
    assert pd.isna(aaa.loc[1, "opt_vs_trailing_4q_mean"])
    assert pytest.approx(aaa.loc[2, "opt_vs_trailing_4q_mean"], rel=1e-9) == -22.0


def test_select_retest_records_covers_all_symbols_deterministically() -> None:
    """驗證重測樣本分層抽樣覆蓋全部公司且完全可重現。"""
    rows = []
    for s_idx in range(20):
        sym = f"SYM{s_idx:02d}"
        for q_idx in range(10):
            yr = 2022 + q_idx // 4
            rows.append(
                {
                    "event_id": f"{sym}_{yr}_Q{q_idx % 4 + 1}_{q_idx}",
                    "symbol": sym,
                    "year": yr,
                    "event_trading_date": f"{yr}-{(q_idx % 4) * 3 + 1:02d}-15",
                }
            )
    pilot_df = pd.DataFrame(rows)
    r1 = select_retest_records(pilot_df, n=30)
    r2 = select_retest_records(pilot_df, n=30)
    assert len(r1) == 30
    assert r1["event_id"].is_unique
    assert r1["symbol"].nunique() == 20
    assert list(r1["event_id"]) == list(r2["event_id"])


def test_compute_retest_metrics_exact_and_near_match() -> None:
    """驗證重測信度指標計算正確（Spearman、MAE、容差一致率）。"""
    run1 = pd.DataFrame(
        [
            {"event_id": "E1", "prepared_optimism": 50, "qa_optimism": 44, "optimism_score": 47.0, "qa_tone_drop": -6, "ambiguity_score": 55, "qa_evasiveness": 42, "guidance_revision": 0},
            {"event_id": "E2", "prepared_optimism": 70, "qa_optimism": 62, "optimism_score": 66.0, "qa_tone_drop": -8, "ambiguity_score": 45, "qa_evasiveness": 36, "guidance_revision": 1},
            {"event_id": "E3", "prepared_optimism": 80, "qa_optimism": 74, "optimism_score": 77.0, "qa_tone_drop": -6, "ambiguity_score": 38, "qa_evasiveness": 30, "guidance_revision": 2},
        ]
    )
    run2 = pd.DataFrame(
        [
            {"event_id": "E1", "prepared_optimism": 52, "qa_optimism": 44, "optimism_score": 48.0, "qa_tone_drop": -8, "ambiguity_score": 55, "qa_evasiveness": 40, "guidance_revision": 0},
            {"event_id": "E2", "prepared_optimism": 70, "qa_optimism": 60, "optimism_score": 65.0, "qa_tone_drop": -10, "ambiguity_score": 45, "qa_evasiveness": 36, "guidance_revision": 1},
            {"event_id": "E3", "prepared_optimism": 79, "qa_optimism": 74, "optimism_score": 76.5, "qa_tone_drop": -5, "ambiguity_score": 40, "qa_evasiveness": 30, "guidance_revision": 2},
        ]
    )
    m = compute_retest_metrics(run1, run2).set_index("metric")
    assert m.loc["prepared_optimism", "spearman_rho"] == 1.0
    assert m.loc["prepared_optimism", "within_tol_rate"] == 1.0
    assert pytest.approx(m.loc["prepared_optimism", "mae"], rel=1e-9) == 1.0
    assert m.loc["guidance_revision", "exact_match_rate"] == 1.0

