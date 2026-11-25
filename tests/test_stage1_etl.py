"""Stage 1 ETL（逐字稿分區、市場量價、EPS Surprise、NYSE 日曆與 DuckDB 稽核）單元測試。"""

from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from src.io_utils import atomic_write_parquet

etl_t = importlib.import_module("src.01_etl_transcripts")
etl_m = importlib.import_module("src.02_etl_market_sue")


def test_atomic_write_parquet_roundtrip_and_no_tmp(tmp_path: Path) -> None:
    df = pd.DataFrame({"symbol": ["AAPL", "NVDA"], "close": [150.0, 900.5]})
    target = tmp_path / "nested" / "sample.parquet"
    atomic_write_parquet(df, target)

    assert target.exists()
    assert not target.with_name("sample.parquet.tmp").exists()
    loaded = pd.read_parquet(target)
    pd.testing.assert_frame_equal(df, loaded)


def test_deduplicate_transcript_records_both_layers() -> None:
    def _rec(sym: str, yr: int, q: int, dt: str, n_segs: int) -> dict:
        return {
            "event_id": f"{sym}_{yr}_Q{q}",
            "symbol": sym,
            "raw_symbol": sym,
            "company_name": "Test Inc",
            "year": yr,
            "quarter": q,
            "raw_date": dt,
            "structured_content": [{"speaker": "CEO", "text": f"seg {i}"} for i in range(n_segs)],
            "source": "test",
            "ingested_at": "2026-11-25T00:00:00",
        }

    records = [
        # 第一層重複：相同 (AAPL, 2023, 1)，應保留段落較多且時間非零者
        _rec("AAPL", 2023, 1, "2023-02-02 00:00:00", 2),
        _rec("AAPL", 2023, 1, "2023-02-02 17:00:00", 5),
        # 第二層重複：相同標的但跨財年標籤不同（相差僅 3 天 <= 15 天）
        _rec("NVDA", 2023, 4, "2024-02-21 17:00:00", 6),
        _rec("NVDA", 2024, 1, "2024-02-24 17:00:00", 3),
        # 正常不同季（相隔 90 天）
        _rec("NVDA", 2024, 2, "2024-05-22 17:00:00", 6),
    ]

    kept, dropped = etl_t.deduplicate_transcript_records(records, dedup_window_days=15)
    assert len(kept) == 3
    assert len(dropped) == 2

    kept_by_id = {r["event_id"]: r for r in kept}
    assert kept_by_id["AAPL_2023_Q1"]["raw_date"] == "2023-02-02 17:00:00"
    assert "NVDA_2023_Q4" in kept_by_id
    assert "NVDA_2024_Q1" not in kept_by_id
    assert "NVDA_2024_Q2" in kept_by_id


def test_write_partitioned_transcripts_schema_and_duplicate_guard(tmp_path: Path) -> None:
    rec1 = {
        "event_id": "AAPL_2023_Q1",
        "symbol": "AAPL",
        "raw_symbol": "AAPL",
        "company_name": "Apple Inc.",
        "year": 2023,
        "quarter": 1,
        "raw_date": "2023-02-02 17:00:00",
        "structured_content": [{"speaker": "CEO", "text": "Strong quarter."}],
        "source": "test",
        "ingested_at": "2026-11-25T00:00:00",
    }
    rec2 = {**rec1, "event_id": "AAPL_2024_Q1", "year": 2024, "raw_date": "2024-02-01 17:00:00"}

    counts = etl_t.write_partitioned_transcripts([rec1, rec2], tmp_path)
    assert counts == {2023: 1, 2024: 1}
    t2023 = pq.read_table(tmp_path / "year=2023" / "part.parquet")
    assert t2023.schema.equals(etl_t.TRANSCRIPT_RAW_SCHEMA)

    # 重複 event_id 必須 Fail-Fast
    with pytest.raises(ValueError, match="event_id"):
        etl_t.write_partitioned_transcripts([rec1, rec1], tmp_path)


def test_build_nyse_calendar_excludes_holidays_and_flags_half_days() -> None:
    cal = etl_m.build_nyse_calendar("2024-01-01", "2024-12-31")
    dates = set(cal["date"].dt.strftime("%Y-%m-%d"))

    assert "2024-01-01" not in dates  # New Year's Day
    assert "2024-12-25" not in dates  # Christmas
    assert "2024-11-28" not in dates  # Thanksgiving
    assert "2024-11-29" in dates  # Black Friday (half day)
    black_friday = cal[cal["date"] == pd.Timestamp("2024-11-29")].iloc[0]
    assert bool(black_friday["is_half_day"]) is True


def test_normalize_yf_ohlcv_batch_single_and_multiindex() -> None:
    idx = pd.to_datetime(["2024-01-02", "2024-01-03"])

    # 1. 單檔平面表（包含異常 NaN 列應被過濾）
    flat = pd.DataFrame(
        {
            "Open": [100.0, np.nan],
            "High": [102.0, np.nan],
            "Low": [99.0, np.nan],
            "Close": [101.0, np.nan],
            "Volume": [1000, np.nan],
        },
        index=idx,
    )
    out_single = etl_m.normalize_yf_ohlcv_batch(flat, ["AAPL"])
    assert len(out_single) == 1
    assert out_single.iloc[0]["symbol"] == "AAPL"
    assert out_single.iloc[0]["close"] == 101.0

    # 2. 多檔 MultiIndex（group_by='ticker' 格式）
    cols = pd.MultiIndex.from_tuples(
        [
            ("AAPL", "Open"), ("AAPL", "High"), ("AAPL", "Low"), ("AAPL", "Close"), ("AAPL", "Volume"),
            ("MSFT", "Open"), ("MSFT", "High"), ("MSFT", "Low"), ("MSFT", "Close"), ("MSFT", "Volume"),
        ]
    )
    multi = pd.DataFrame(
        [
            [100.0, 102.0, 99.0, 101.0, 1000, 300.0, 305.0, 298.0, 304.0, 2000],
            [101.0, 103.0, 100.0, 102.0, 1100, 304.0, 306.0, 301.0, 302.0, 1800],
        ],
        index=idx,
        columns=cols,
    )
    out_multi = etl_m.normalize_yf_ohlcv_batch(multi, ["AAPL", "MSFT"])
    assert len(out_multi) == 4
    assert set(out_multi["symbol"]) == {"AAPL", "MSFT"}


def test_normalize_eps_dates_df_handles_zero_and_future_rows() -> None:
    idx = pd.DatetimeIndex(
        [
            pd.Timestamp("2024-05-02 16:30:00", tz="America/New_York"),
            pd.Timestamp("2024-08-01 16:30:00", tz="America/New_York"),
            pd.Timestamp("2026-12-01 16:30:00", tz="America/New_York"),  # 未來未公布
        ]
    )
    raw = pd.DataFrame(
        {
            "EPS Estimate": [1.50, 0.0, 2.00],
            "Reported EPS": [1.65, 0.10, np.nan],
            "Surprise(%)": [10.0, np.nan, np.nan],
        },
        index=idx,
    )
    out = etl_m.normalize_eps_dates_df("AAPL", raw)
    assert len(out) == 2  # 未來未公布列應被排除
    assert pytest.approx(out.iloc[0]["computed_surprise_pct"], rel=1e-4) == 10.0
    assert pd.isna(out.iloc[1]["computed_surprise_pct"])  # 除零保護，不得為 inf


def test_validate_ohlcv_completeness_catches_delisted_ticker() -> None:
    """驗證單日殘缺股（如私有化下市之 EA）會觸發 Fail-Fast 錯誤。"""
    good_dates = pd.date_range("2024-01-01", periods=100, freq="B")
    df = pd.DataFrame(
        {
            "symbol": ["AAPL"] * 100 + ["EA"],
            "date": list(good_dates) + [pd.Timestamp("2026-08-04")],
            "close": [180.0] * 100 + [140.0],
        }
    )
    # 僅檢核 AAPL 應通過
    etl_m.validate_ohlcv_completeness(df, ["AAPL"], benchmark_days=100, min_coverage_ratio=0.95)

    # 納入只有 1 天的 EA 必須拋出 ValueError
    with pytest.raises(ValueError, match="EA"):
        etl_m.validate_ohlcv_completeness(df, ["AAPL", "EA"], benchmark_days=100, min_coverage_ratio=0.95)


def test_run_stage1_audit_end_to_end_on_synthetic_lake(tmp_path: Path) -> None:
    """端到端驗證 DuckDB 跨表稽核邏輯。"""
    rec = {
        "event_id": "AAPL_2024_Q1",
        "symbol": "AAPL",
        "raw_symbol": "AAPL",
        "company_name": "Apple Inc.",
        "year": 2024,
        "quarter": 1,
        "raw_date": "2024-02-01 17:00:00",
        "structured_content": [{"speaker": "CEO", "text": "Good results."}],
        "source": "test",
        "ingested_at": "2026-11-25T00:00:00",
    }
    etl_t.write_partitioned_transcripts([rec], tmp_path / "transcripts_raw")

    atomic_write_parquet(
        pd.DataFrame(
            {
                "symbol": ["AAPL"],
                "date": [pd.Timestamp("2024-02-01")],
                "open": [185.0],
                "high": [187.0],
                "low": [184.0],
                "close": [186.5],
                "volume": [50_000_000.0],
            }
        ),
        tmp_path / "market_ohlcv_raw.parquet",
    )
    atomic_write_parquet(
        pd.DataFrame(
            {
                "symbol": ["^GSPC", "^VIX"],
                "date": [pd.Timestamp("2024-02-01"), pd.Timestamp("2024-02-01")],
                "open": [4900.0, 14.0],
                "high": [4920.0, 15.0],
                "low": [4890.0, 13.5],
                "close": [4910.0, 14.2],
                "volume": [1.0, 0.0],
            }
        ),
        tmp_path / "macro_index_raw.parquet",
    )
    atomic_write_parquet(
        pd.DataFrame(
            {
                "symbol": ["AAPL"],
                "earnings_datetime_et": ["2024-02-01 16:30:00-0500"],
                "earnings_date": [pd.Timestamp("2024-02-01")],
                "eps_estimate": [2.10],
                "eps_actual": [2.18],
                "reported_surprise_pct": [3.81],
                "computed_surprise_pct": [3.8095],
            }
        ),
        tmp_path / "eps_sue_raw.parquet",
    )
    atomic_write_parquet(
        etl_m.build_nyse_calendar("2024-01-01", "2024-03-01"),
        tmp_path / "nyse_calendar.parquet",
    )

    audit_md = tmp_path / "audit.md"
    res = etl_m.run_stage1_audit(raw_dir=tmp_path, audit_md_path=audit_md)
    assert res["transcript_rows"] == 1
    assert res["ohlcv_rows"] == 1
    assert res["eps_match_pct"] == 100.0
    assert audit_md.exists()
