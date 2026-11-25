"""Stage 0 探勘腳本與設定模組之單元測試。"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from src.config import DATA_CFG, DEFAULT_RENAME_MAP, PROBE_CFG, QUANT_CFG, load_universe

probe = importlib.import_module("src.00_probe_dataset")


def test_config_invariants() -> None:
    """驗證核心設定防呆條件：Purge Gap 不小於最長預測期程。"""
    assert QUANT_CFG.purge_gap_trading_days >= max(QUANT_CFG.pead_horizons)
    assert DATA_CFG.start_year == 2016
    assert DATA_CFG.market_open_hour == 9 and DATA_CFG.market_open_minute == 30
    assert PROBE_CFG.schema_sample_limit > 0


def test_load_universe_missing_raises(tmp_path: Path) -> None:
    """缺少 universe.json 時必須 Fail-Fast 拋出 FileNotFoundError。"""
    with pytest.raises(FileNotFoundError):
        load_universe(tmp_path / "nonexistent.json")


def test_normalize_symbol_rename_and_whitespace() -> None:
    assert probe.normalize_symbol(" fb ") == "META"
    assert probe.normalize_symbol("ANTM") == "ELV"
    assert probe.normalize_symbol("aapl") == "AAPL"
    # PXD 為 2024 年被 XOM 併購之獨立公司，嚴禁合併為 XOM
    assert probe.normalize_symbol("PXD") == "PXD"
    assert probe.normalize_symbol("") == ""


def test_parse_raw_timestamp_edge_cases() -> None:
    # 完整時間（非零）
    ts_full = probe.parse_raw_timestamp("2024-05-02 17:00:00")
    assert ts_full.valid is True
    assert ts_full.has_time_component is True
    assert ts_full.is_midnight_zero is False
    assert ts_full.hour == 17 and ts_full.minute == 0

    # 退化為 00:00:00（缺少實際時分秒）
    ts_mid = probe.parse_raw_timestamp("2017-01-31 00:00:00")
    assert ts_mid.valid is True
    assert ts_mid.has_time_component is False
    assert ts_mid.is_midnight_zero is True

    # 純日期
    ts_date = probe.parse_raw_timestamp("2018-07-25")
    assert ts_date.valid is True
    assert ts_date.has_time_component is False
    assert ts_date.is_midnight_zero is False

    # 帶顯式時區
    ts_tz = probe.parse_raw_timestamp("2024-08-01T21:00:00Z")
    assert ts_tz.valid is True
    assert ts_tz.has_time_component is True
    assert ts_tz.tz_info is not None

    # 空值與壞格式
    for bad in (None, "", "NaN", "not-a-date"):
        res = probe.parse_raw_timestamp(bad)
        assert res.valid is False
        assert res.has_time_component is False


def test_detect_qa_split_index_and_word_count() -> None:
    segments = [
        {"speaker": "Operator", "text": "Good afternoon. After remarks there will be a question-and-answer session."},
        {"speaker": "CEO", "text": "Revenue grew strongly this quarter across all segments."},
        {"speaker": "Operator", "text": "We will now begin the Question-and-Answer session."},
        {"speaker": "Analyst", "text": "Can you comment on gross margin trajectory?"},
    ]
    # 必須略過開場白 (idx=0)，精準命中真正的 Q&A 起點 (idx=2)
    assert probe.detect_qa_split_index(segments) == 2
    assert probe.estimate_structured_word_count(segments) == 32

    # 邊界條件：無 Q&A 與空資料
    assert probe.detect_qa_split_index([{"speaker": "CEO", "text": "Only remarks."}]) is None
    assert probe.detect_qa_split_index([]) is None
    assert probe.detect_qa_split_index(None) is None
    assert probe.estimate_structured_word_count(None) == 0


def test_universe_json_integrity_and_stage0_coverage() -> None:
    """驗證 config/universe.json 滿足 11 大 GICS 板塊均衡且全數通過 Stage 0 覆蓋率門檻。"""
    import pandas as pd

    uni = load_universe()
    symbols = uni["symbols"]
    sectors = uni["sectors"]
    pilot = uni["pilot_symbols"]

    assert len(symbols) == 120
    assert len(set(symbols)) == 120  # 零重複
    assert len(sectors) == 11  # 涵蓋全部 GICS 11 大板塊
    sector_union = [s for syms in sectors.values() for s in syms]
    assert sorted(sector_union) == sorted(symbols)
    assert len(pilot) == 20 and set(pilot).issubset(set(symbols))
    assert not ({"GOOG", "GOOGL"}.issubset(set(symbols)))  # 無雙重股權重複
    assert uni["rename_map"] == DEFAULT_RENAME_MAP  # 設定與 Single Source of Truth 一致

    if PROBE_CFG.symbol_coverage_csv.exists():
        cov = pd.read_csv(PROBE_CFG.symbol_coverage_csv).set_index("symbol")
        missing = [s for s in symbols if s not in cov.index]
        assert not missing, f"股票池包含 Stage 0 查無紀錄之標的：{missing}"
        low_cov = cov.loc[symbols][cov.loc[symbols, "completeness_ratio"] < DATA_CFG.min_quarter_completeness]
        assert low_cov.empty, f"股票池包含完整度低於 90% 之標的：{low_cov.index.tolist()}"
        dups = cov.loc[symbols][cov.loc[symbols, "total_records"] > cov.loc[symbols, "unique_quarters"]]
        assert dups.empty, f"股票池包含重複季度異常標的（疑有併購誤合併）：{dups.index.tolist()}"


def test_infer_timezone_hypothesis() -> None:
    assert "America/New_York" in probe.infer_timezone_hypothesis([17, 17, 16], [8, 8, 9])
    assert "UTC" in probe.infer_timezone_hypothesis([21, 22, 21], [12, 13, 12])
    assert "UNKNOWN" in probe.infer_timezone_hypothesis([], [])


def test_summarize_symbol_coverage_merges_renames() -> None:
    records = [
        {"symbol": "FB", "company_name": "Facebook", "year": 2020, "quarter": 1, "date": "2020-04-29 17:00:00"},
        {"symbol": "META", "company_name": "Meta", "year": 2022, "quarter": 1, "date": "2022-04-27 17:00:00"},
        {"symbol": "AAPL", "company_name": "Apple", "year": 2020, "quarter": 1, "date": "2020-04-30 00:00:00"},
        {"symbol": "AAPL", "company_name": "Apple", "year": 2015, "quarter": 4, "date": "2015-10-27 17:00:00"},
    ]
    df = probe.summarize_symbol_coverage(records, start_year=2016)
    meta_row = df[df["symbol"] == "META"].iloc[0]
    aapl_row = df[df["symbol"] == "AAPL"].iloc[0]

    assert meta_row["unique_quarters"] == 2
    assert meta_row["completeness_ratio"] == 1.0
    assert meta_row["non_zero_time_ratio"] == 1.0
    assert aapl_row["unique_quarters"] == 1
    assert aapl_row["non_zero_time_ratio"] == 0.0
