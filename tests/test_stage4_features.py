"""Stage 4 多模態特徵工程、EPS 配對與零未來洩漏標籤之單元測試。"""

from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest

from src.config import FEATURE_GROUPS, KEY_META_COLUMNS, LABEL_COLUMNS, QUANT_CFG

f_mod = importlib.import_module("src.06_build_features")
prepare_macro_panel = f_mod.prepare_macro_panel
compute_stock_panels = f_mod.compute_stock_panels
match_eps_sue_to_events = f_mod.match_eps_sue_to_events


def test_feature_groups_keys_and_labels_mutually_disjoint() -> None:
    """驗證設定檔中之鍵值、特徵群組與目標標籤完全互斥，杜絕鍵值或標籤混入特徵矩陣。"""
    all_feats = [c for cols in FEATURE_GROUPS.values() for c in cols]
    assert len(all_feats) == len(set(all_feats))
    assert set(KEY_META_COLUMNS).isdisjoint(all_feats)
    assert set(LABEL_COLUMNS).isdisjoint(all_feats)
    assert set(KEY_META_COLUMNS).isdisjoint(LABEL_COLUMNS)
    assert all(c.startswith("label_") for c in LABEL_COLUMNS)


def test_prepare_macro_panel_aligns_vix_and_ffills_backward_only() -> None:
    """驗證 ^VIX 嚴格對齊 ^GSPC 交易日、排除非交易日並僅向後（過去）前填。"""
    dates = pd.bdate_range("2020-01-02", periods=90)
    spx = pd.DataFrame(
        {
            "symbol": "^GSPC",
            "date": dates,
            "open": np.linspace(3000.0, 3100.0, len(dates)),
            "high": np.linspace(3010.0, 3110.0, len(dates)),
            "low": np.linspace(2990.0, 3090.0, len(dates)),
            "close": np.linspace(3000.0, 3100.0, len(dates)),
            "volume": 1e9,
        }
    )
    # 故意缺第 10 天 VIX 並額外插入 1 筆週六非交易日 VIX
    vix_dates = list(dates[:10]) + list(dates[11:]) + [pd.Timestamp("2020-01-04")]
    vix_vals = [20.0 + i * 0.1 for i in range(len(vix_dates) - 1)] + [99.0]
    vix = pd.DataFrame(
        {
            "symbol": "^VIX",
            "date": vix_dates,
            "open": vix_vals,
            "high": vix_vals,
            "low": vix_vals,
            "close": vix_vals,
            "volume": 0.0,
        }
    )
    panel = prepare_macro_panel(pd.concat([spx, vix], ignore_index=True))
    assert len(panel) == len(dates)
    # 週六 99.0 必須被排除，且第 10 天缺值必須前填第 9 天數值（絕非第 11 天未來值！）
    assert (panel["pre_vix_close"] < 90.0).all()
    assert panel.loc[10, "pre_vix_close"] == panel.loc[9, "pre_vix_close"]


def test_compute_stock_panels_strict_point_in_time_and_exact_rv() -> None:
    """驗證事前特徵完全不受 t=0 巨幅跳空污染，且事後 10d/20d 報酬與波動率精確排除隔夜跳空。"""
    n_days = 160
    dates = pd.bdate_range("2022-01-03", periods=n_days)
    rng = np.random.default_rng(42)
    spx_ret = rng.normal(0.0004, 0.01, size=n_days)
    spx_close = 4000.0 * np.cumprod(1.0 + spx_ret)

    macro = pd.concat(
        [
            pd.DataFrame({"symbol": "^GSPC", "date": dates, "open": spx_close, "high": spx_close, "low": spx_close, "close": spx_close, "volume": 1e9}),
            pd.DataFrame({"symbol": "^VIX", "date": dates, "open": 20.0, "high": 20.0, "low": 20.0, "close": 20.0, "volume": 0.0}),
        ],
        ignore_index=True,
    )
    macro_panel = prepare_macro_panel(macro)

    # 構造個股價格序列：在事件日 t0=125 製造 +50% 巨幅隔夜跳空
    t0 = 125
    stock_ret = rng.normal(0.0005, 0.015, size=n_days)
    stock_ret[t0] = 0.50  # 巨幅事件日跳空！
    stock_close = 100.0 * np.cumprod(1.0 + stock_ret)
    stock_open = stock_close.copy()
    stock_open[t0] = stock_close[t0 - 1] * 1.45  # 跳空開高 45%，盤中再漲至 +50%

    ohlcv = pd.DataFrame(
        {
            "symbol": "TEST",
            "date": dates,
            "open": stock_open,
            "high": np.maximum(stock_open, stock_close),
            "low": np.minimum(stock_open, stock_close),
            "close": stock_close,
            "volume": 1_000_000,
        }
    )

    pre_panel, event_panel = compute_stock_panels(ohlcv, macro_panel)
    pre_row = pre_panel.loc[pre_panel["pre_end_date"] == dates[t0 - 1].strftime("%Y-%m-%d")].iloc[0]
    ev_row = event_panel.loc[event_panel["event_trading_date"] == dates[t0].strftime("%Y-%m-%d")].iloc[0]

    # 1. 事前特徵（t-1）絕不可受到 t0 的 +50% 跳空污染
    expected_pre_ret_5d = stock_close[t0 - 1] / stock_close[t0 - 6] - 1.0
    expected_pre_rv_20d = float(np.std(stock_ret[t0 - 20 : t0], ddof=1) * np.sqrt(QUANT_CFG.trading_days_per_year))
    assert pytest.approx(pre_row["pre_ret_5d"], rel=1e-10) == expected_pre_ret_5d
    assert pytest.approx(pre_row["pre_rv_20d"], rel=1e-10) == expected_pre_rv_20d
    assert pre_row["pre_rv_20d"] < 0.60  # 若誤納 +50% 跳空，年化波動會衝破 1.5

    # 2. 事件日特徵精確捕捉跳空與盤中反轉
    assert pytest.approx(ev_row["event_day_ret"], rel=1e-10) == 0.50
    assert pytest.approx(ev_row["overnight_gap"], rel=1e-10) == 0.45

    # 3. 事後 10d/20d 報酬嚴格自 Close(t0) 起算（完全排除 t0 跳空）
    expected_ret_10d = stock_close[t0 + 10] / stock_close[t0] - 1.0
    expected_ret_20d = stock_close[t0 + 20] / stock_close[t0] - 1.0
    assert pytest.approx(ev_row["label_ret_10d"], rel=1e-10) == expected_ret_10d
    assert pytest.approx(ev_row["label_ret_20d"], rel=1e-10) == expected_ret_20d

    # 4. 事後 10d/20d 已實現波動率嚴格僅含 r[t0+1 .. t0+H]，不含 r[t0]=+50%
    expected_rv_10d = float(np.std(stock_ret[t0 + 1 : t0 + 11], ddof=1) * np.sqrt(QUANT_CFG.trading_days_per_year))
    expected_rv_20d = float(np.std(stock_ret[t0 + 1 : t0 + 21], ddof=1) * np.sqrt(QUANT_CFG.trading_days_per_year))
    assert pytest.approx(ev_row["label_rv_10d"], rel=1e-10) == expected_rv_10d
    assert pytest.approx(ev_row["label_rv_20d"], rel=1e-10) == expected_rv_20d
    assert ev_row["label_rv_10d"] < 0.60


def test_match_eps_sue_to_events_window_and_tiebreak() -> None:
    """驗證 EPS Surprise ±5 天窗口配對、同距離優先 <= 事件日與缺失備援。"""
    events = pd.DataFrame(
        [
            {"event_id": "E1", "symbol": "AAPL", "event_trading_date": "2024-05-03"},
            {"event_id": "E2", "symbol": "AAPL", "event_trading_date": "2024-08-15"},
            {"event_id": "E3", "symbol": "MSFT", "event_trading_date": "2024-05-03"},
        ]
    )
    eps = pd.DataFrame(
        [
            # 與 E1 同距離 1 天：5/02 (<= 5/03) 應優先於 5/04 (> 5/03)
            {"symbol": "AAPL", "earnings_date": pd.Timestamp("2024-05-02"), "eps_estimate": 1.50, "eps_actual": 1.65, "reported_surprise_pct": 10.0, "computed_surprise_pct": 10.0},
            {"symbol": "AAPL", "earnings_date": pd.Timestamp("2024-05-04"), "eps_estimate": 1.50, "eps_actual": 1.20, "reported_surprise_pct": -20.0, "computed_surprise_pct": -20.0},
            # 距 E2 超過 5 天（10 天），不可誤配
            {"symbol": "AAPL", "earnings_date": pd.Timestamp("2024-08-01"), "eps_estimate": 1.50, "eps_actual": 1.70, "reported_surprise_pct": 13.3, "computed_surprise_pct": 13.3},
            # reported_surprise_pct 為 NaN，應備援使用 computed_surprise_pct
            {"symbol": "MSFT", "earnings_date": pd.Timestamp("2024-05-03"), "eps_estimate": 2.00, "eps_actual": 1.90, "reported_surprise_pct": np.nan, "computed_surprise_pct": -5.0},
        ]
    )
    matched = match_eps_sue_to_events(events, eps, max_days=5).set_index("event_id")

    assert matched.loc["E1", "sue_available"] == 1
    assert matched.loc["E1", "sue_pct"] == 10.0
    assert matched.loc["E1", "sue_beat_flag"] == 1.0

    assert matched.loc["E2", "sue_available"] == 0
    assert pd.isna(matched.loc["E2", "sue_pct"])

    assert matched.loc["E3", "sue_available"] == 1
    assert matched.loc["E3", "sue_pct"] == -5.0
    assert matched.loc["E3", "sue_beat_flag"] == -1.0
