"""Stage 4：多模態量化特徵工程與嚴格可交易標籤建置。

嚴格遵守 AGENTS.md：
1. 事前量價與市場狀態特徵嚴格截斷至事件前收盤價 Close(t-1)（pre_end_date），零未來資料洩漏。
2. 事後策略標籤嚴格自真實可成交價 Close(t=0)（及 Open(t=0) 敏感度對照）起算，排除無法參與之隔夜跳空。
3. 嚴禁於全資料集上進行全局 Z-Score 標準化、缺失填補或全局 Winsorization（留待 Stage 5 訓練視窗內 fit）。
4. 嚴格執行鍵值、特徵與標籤隔離檢查，產出 data/04_features/model_dataset.parquet 與 outputs/stage4_feature_audit.md。
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.config import (
    CLEAN_TRANSCRIPTS_DIR,
    EMBED_DRIFT_FEATURES_PATH,
    FEATURE_GROUPS,
    KEY_META_COLUMNS,
    LABEL_COLUMNS,
    LLM_FEATURES_PATH,
    MODEL_DATASET_PATH,
    QUANT_CFG,
    RAW_EPS_SUE_PATH,
    RAW_MACRO_INDEX_PATH,
    RAW_MARKET_OHLCV_PATH,
    STAGE4_AUDIT_MD,
    ensure_project_dirs,
)
from src.io_utils import atomic_write_parquet


# ==================== 大盤與 VIX 市場狀態面板（嚴格 NYSE 交易日對齊） ====================
def prepare_macro_panel(macro_df: pd.DataFrame) -> pd.DataFrame:
    """建立嚴格對齊 ^GSPC (NYSE) 交易日之大盤與 VIX 事前狀態與事後基準報酬表。"""
    df = macro_df.copy()
    df["date_str"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

    spx = (
        df[df["symbol"] == "^GSPC"][["date_str", "open", "close"]]
        .drop_duplicates(subset=["date_str"])
        .sort_values("date_str")
        .rename(columns={"open": "spx_open", "close": "spx_close"})
        .reset_index(drop=True)
    )
    vix = (
        df[df["symbol"] == "^VIX"][["date_str", "close"]]
        .drop_duplicates(subset=["date_str"])
        .sort_values("date_str")
        .rename(columns={"close": "vix_close"})
        .reset_index(drop=True)
    )
    if spx.empty or vix.empty:
        raise ValueError("macro_index_raw 缺少 ^GSPC 或 ^VIX 紀錄。")

    # 以 ^GSPC 官方交易日為基準 LEFT JOIN ^VIX，並僅允許嚴格向後（過去）前填
    panel = spx.merge(vix, on="date_str", how="left")
    panel["vix_close"] = panel["vix_close"].ffill()

    ann = float(np.sqrt(QUANT_CFG.trading_days_per_year))
    panel["spx_ret_1d"] = panel["spx_close"].pct_change()
    panel["pre_spx_ret_20d"] = panel["spx_close"].pct_change(20)
    panel["pre_spx_ret_60d"] = panel["spx_close"].pct_change(60)
    panel["pre_spx_ret_120d"] = panel["spx_close"].pct_change(120)
    panel["pre_spx_rv_20d"] = panel["spx_ret_1d"].rolling(20, min_periods=15).std(ddof=1) * ann

    panel["pre_vix_close"] = panel["vix_close"]
    panel["pre_vix_chg_5d"] = panel["vix_close"] - panel["vix_close"].shift(5)
    panel["pre_vix_chg_20d"] = panel["vix_close"] - panel["vix_close"].shift(20)
    # 嚴格過去 252 交易日內之 VIX 分位數（含當日 t-1，零未來資訊）
    panel["pre_vix_pct_rank_252d"] = (
        panel["vix_close"]
        .rolling(252, min_periods=60)
        .apply(lambda w: float(np.mean(w <= w[-1])), raw=True)
    )

    # 事後基準報酬（供事件日 t=0 關聯計算超額報酬，嚴禁進入事前特徵）
    panel["spx_event_day_ret"] = panel["spx_ret_1d"]
    for h in QUANT_CFG.pead_horizons:
        panel[f"label_spx_ret_{h}d"] = panel["spx_close"].shift(-h) / panel["spx_close"] - 1.0
        panel[f"label_spx_open_ret_{h}d"] = panel["spx_close"].shift(-h) / panel["spx_open"] - 1.0

    return panel


# ==================== 個股量價事前特徵與事後可交易標籤 ====================
def compute_stock_panels(
    ohlcv_df: pd.DataFrame,
    macro_panel: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """為每檔股票分別產出（1）截斷至日期 d 收盤之事前特徵表與（2）事件日 t=0 反應及事後標籤表。"""
    df = ohlcv_df.copy()
    df["date_str"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df = df.sort_values(["symbol", "date_str"]).reset_index(drop=True)

    spx_daily = macro_panel[
        [
            "date_str",
            "spx_ret_1d",
            "pre_spx_ret_20d",
            "pre_spx_ret_60d",
            "pre_spx_ret_120d",
            "pre_spx_rv_20d",
            "pre_vix_close",
            "pre_vix_chg_5d",
            "pre_vix_chg_20d",
            "pre_vix_pct_rank_252d",
            "spx_event_day_ret",
            "label_spx_ret_10d",
            "label_spx_ret_20d",
            "label_spx_open_ret_10d",
            "label_spx_open_ret_20d",
        ]
    ]
    df = df.merge(spx_daily, on="date_str", how="inner")

    ann = float(np.sqrt(QUANT_CFG.trading_days_per_year))
    pre_frames: List[pd.DataFrame] = []
    event_frames: List[pd.DataFrame] = []

    for sym, grp in df.groupby("symbol", sort=False):
        g = grp.sort_values("date_str").reset_index(drop=True)
        close = g["close"].astype(float)
        open_p = g["open"].astype(float)
        vol = g["volume"].astype(float)
        ret_1d = close.pct_change()
        spx_r = g["spx_ret_1d"].astype(float)

        # ---- 1. 事前特徵（嚴格 <= date_str，後續僅與 pre_end_date = t-1 關聯） ----
        pre_ret_5d = close.pct_change(5)
        pre_ret_20d = close.pct_change(20)
        pre_ret_60d = close.pct_change(60)
        pre_ret_120d = close.pct_change(120)

        pre_rv_20d = ret_1d.rolling(20, min_periods=15).std(ddof=1) * ann
        pre_rv_60d = ret_1d.rolling(60, min_periods=40).std(ddof=1) * ann
        pre_rv_ratio = pre_rv_20d / pre_rv_60d.replace(0.0, np.nan)

        dollar_vol = (close * vol).replace(0.0, np.nan)
        amihud_daily = ret_1d.abs() / dollar_vol
        pre_amihud_20d = np.log1p(1e9 * amihud_daily.rolling(20, min_periods=15).mean())

        ma5_vol = vol.rolling(5, min_periods=3).mean()
        ma20_vol = vol.rolling(20, min_periods=15).mean()
        pre_vol_ratio_5_20 = ma5_vol / ma20_vol.replace(0.0, np.nan)

        roll_max_60 = close.rolling(60, min_periods=20).max()
        pre_drawdown_60d = close / roll_max_60 - 1.0

        cov_60 = ret_1d.rolling(60, min_periods=40).cov(spx_r)
        var_spx_60 = spx_r.rolling(60, min_periods=40).var().replace(0.0, np.nan)
        pre_beta_60d = cov_60 / var_spx_60

        pre_frames.append(
            pd.DataFrame(
                {
                    "symbol": sym,
                    "pre_end_date": g["date_str"],
                    "pre_ret_5d": pre_ret_5d,
                    "pre_ret_20d": pre_ret_20d,
                    "pre_ret_60d": pre_ret_60d,
                    "pre_ret_120d": pre_ret_120d,
                    "pre_car_20d": pre_ret_20d - g["pre_spx_ret_20d"],
                    "pre_car_60d": pre_ret_60d - g["pre_spx_ret_60d"],
                    "pre_car_120d": pre_ret_120d - g["pre_spx_ret_120d"],
                    "pre_rv_20d": pre_rv_20d,
                    "pre_rv_60d": pre_rv_60d,
                    "pre_rv_ratio_20_60": pre_rv_ratio,
                    "pre_amihud_20d": pre_amihud_20d,
                    "pre_vol_ratio_5_20": pre_vol_ratio_5_20,
                    "pre_drawdown_60d": pre_drawdown_60d,
                    "pre_beta_60d": pre_beta_60d,
                    "pre_spx_ret_20d": g["pre_spx_ret_20d"],
                    "pre_spx_ret_60d": g["pre_spx_ret_60d"],
                    "pre_spx_rv_20d": g["pre_spx_rv_20d"],
                    "pre_vix_close": g["pre_vix_close"],
                    "pre_vix_chg_5d": g["pre_vix_chg_5d"],
                    "pre_vix_chg_20d": g["pre_vix_chg_20d"],
                    "pre_vix_pct_rank_252d": g["pre_vix_pct_rank_252d"],
                }
            )
        )

        # ---- 2. 事件日 t=0 反應特徵與嚴格事後可交易標籤 ----
        prev_close = close.shift(1)
        prev_ma20_vol = ma20_vol.shift(1).replace(0.0, np.nan)
        event_day_ret = close / prev_close - 1.0
        event_day_excess_ret = event_day_ret - g["spx_event_day_ret"]
        overnight_gap = open_p / prev_close - 1.0
        intraday_reversal = close / open_p - 1.0
        event_day_vol_ratio = vol / prev_ma20_vol

        # 事後 H 日報酬：Close(t=0) -> Close(t+H)（嚴格排除 t-1->t=0 隔夜跳空）
        label_ret_10d = close.shift(-10) / close - 1.0
        label_ret_20d = close.shift(-20) / close - 1.0
        label_open_ret_10d = close.shift(-10) / open_p - 1.0
        label_open_ret_20d = close.shift(-20) / open_p - 1.0

        # 事後 H 日年化已實現波動率：嚴格僅含 r(t+1) ... r(t+H)
        # 驗證：ret_1d.rolling(H).std(ddof=1) 在索引 i+H 計算 [i+1..i+H]，shift(-H) 對齊回索引 i
        label_rv_10d = ret_1d.rolling(10, min_periods=10).std(ddof=1).shift(-10) * ann
        label_rv_20d = ret_1d.rolling(20, min_periods=20).std(ddof=1).shift(-20) * ann
        label_log_rv_10d = np.log(np.clip(label_rv_10d, 1e-4, None))
        label_log_rv_20d = np.log(np.clip(label_rv_20d, 1e-4, None))

        event_frames.append(
            pd.DataFrame(
                {
                    "symbol": sym,
                    "event_trading_date": g["date_str"],
                    "event_day_ret": event_day_ret,
                    "event_day_excess_ret": event_day_excess_ret,
                    "overnight_gap": overnight_gap,
                    "intraday_reversal": intraday_reversal,
                    "event_day_vol_ratio": event_day_vol_ratio,
                    "label_ret_10d": label_ret_10d,
                    "label_spx_ret_10d": g["label_spx_ret_10d"],
                    "label_excess_ret_10d": label_ret_10d - g["label_spx_ret_10d"],
                    "label_ret_20d": label_ret_20d,
                    "label_spx_ret_20d": g["label_spx_ret_20d"],
                    "label_excess_ret_20d": label_ret_20d - g["label_spx_ret_20d"],
                    "label_open_excess_ret_10d": label_open_ret_10d - g["label_spx_open_ret_10d"],
                    "label_open_excess_ret_20d": label_open_ret_20d - g["label_spx_open_ret_20d"],
                    "label_rv_10d": label_rv_10d,
                    "label_rv_20d": label_rv_20d,
                    "label_log_rv_10d": label_log_rv_10d,
                    "label_log_rv_20d": label_log_rv_20d,
                }
            )
        )

    pre_panel = pd.concat(pre_frames, ignore_index=True)
    event_panel = pd.concat(event_frames, ignore_index=True)
    return pre_panel, event_panel


# ==================== EPS Surprise 精準時間窗口配對 ====================
def match_eps_sue_to_events(
    events_df: pd.DataFrame,
    eps_df: pd.DataFrame,
    max_days: int = QUANT_CFG.eps_match_max_days,
) -> pd.DataFrame:
    """於 ±max_days 日曆日內將每場法說會配對最接近之當季 EPS Surprise（優先 <= event_trading_date）。"""
    eps = eps_df.copy()
    eps["earn_dt"] = pd.to_datetime(eps["earnings_date"]).dt.normalize()
    eps["sue_pct_val"] = eps["reported_surprise_pct"].combine_first(eps["computed_surprise_pct"])
    eps_by_sym: Dict[str, pd.DataFrame] = {
        sym: g.sort_values("earn_dt").reset_index(drop=True)
        for sym, g in eps.groupby("symbol", sort=False)
    }

    rows: List[Dict[str, Any]] = []
    for row in events_df[["event_id", "symbol", "event_trading_date"]].itertuples(index=False):
        eid = str(row.event_id)
        sym = str(row.symbol)
        ev_dt = pd.Timestamp(row.event_trading_date)
        cand = eps_by_sym.get(sym)

        if cand is None or cand.empty:
            rows.append(
                {
                    "event_id": eid,
                    "sue_pct": np.nan,
                    "sue_diff": np.nan,
                    "sue_beat_flag": np.nan,
                    "sue_available": 0,
                }
            )
            continue

        delta_days = (cand["earn_dt"] - ev_dt).dt.days
        mask = delta_days.abs() <= max_days
        if not mask.any():
            rows.append(
                {
                    "event_id": eid,
                    "sue_pct": np.nan,
                    "sue_diff": np.nan,
                    "sue_beat_flag": np.nan,
                    "sue_available": 0,
                }
            )
            continue

        sub = cand.loc[mask].copy()
        sub["abs_d"] = delta_days.loc[mask].abs()
        # 優先選擇公告日 <= event_trading_date（is_after=0 優於 1）
        sub["is_after"] = (delta_days.loc[mask] > 0).astype(int)
        best = sub.sort_values(["abs_d", "is_after", "earn_dt"]).iloc[0]

        est = float(best["eps_estimate"]) if pd.notna(best["eps_estimate"]) else np.nan
        act = float(best["eps_actual"]) if pd.notna(best["eps_actual"]) else np.nan
        sue_pct = float(best["sue_pct_val"]) if pd.notna(best["sue_pct_val"]) else np.nan
        sue_diff = (act - est) if (pd.notna(act) and pd.notna(est)) else np.nan

        if pd.notna(sue_diff):
            beat_flag = 1.0 if sue_diff > 1e-9 else (-1.0 if sue_diff < -1e-9 else 0.0)
        elif pd.notna(sue_pct):
            beat_flag = 1.0 if sue_pct > 1e-9 else (-1.0 if sue_pct < -1e-9 else 0.0)
        else:
            beat_flag = np.nan

        avail = 1 if (pd.notna(sue_pct) or pd.notna(sue_diff)) else 0
        rows.append(
            {
                "event_id": eid,
                "sue_pct": sue_pct,
                "sue_diff": sue_diff,
                "sue_beat_flag": beat_flag,
                "sue_available": avail,
            }
        )

    return pd.DataFrame(rows)


# ==================== 主特徵表組裝與嚴格防禦性檢核 ====================
def assemble_model_dataset(
    clean_meta: pd.DataFrame,
    llm_df: pd.DataFrame,
    embed_df: pd.DataFrame,
    ohlcv_df: pd.DataFrame,
    macro_df: pd.DataFrame,
    eps_df: pd.DataFrame,
) -> pd.DataFrame:
    """整合所有多模態來源並計算交互項，回傳通過嚴格 Point-in-Time 驗證之主特徵表。"""
    base = clean_meta.sort_values(["symbol", "event_trading_date", "event_id"]).reset_index(drop=True).copy()
    assert base["event_id"].is_unique, "clean_meta 存在重複 event_id！"

    # 文本結構與跨季字數變化（嚴格 Point-in-Time）
    base["is_bmo"] = (base["session_type"] == "BMO").astype(int)
    base["qa_word_ratio"] = base["qa_word_count"] / base["total_word_count"].replace(0, np.nan)
    base["log_total_words"] = np.log1p(base["total_word_count"].astype(float))
    base["qoq_word_count_pct_chg"] = base.groupby("symbol", sort=False)["total_word_count"].pct_change()

    # 市場面板與個股量價面板
    macro_panel = prepare_macro_panel(macro_df)
    pre_panel, event_panel = compute_stock_panels(ohlcv_df, macro_panel)
    eps_matched = match_eps_sue_to_events(base, eps_df)

    # 嚴格對齊合併（左連接皆驗證不增減筆數）
    n0 = len(base)
    df = base.merge(pre_panel, on=["symbol", "pre_end_date"], how="left")
    df = df.merge(event_panel, on=["symbol", "event_trading_date"], how="left")
    df = df.merge(eps_matched, on="event_id", how="left")

    llm_cols = ["event_id", *FEATURE_GROUPS["llm_sentiment"]]
    emb_cols = ["event_id", *FEATURE_GROUPS["embed_drift"]]
    df = df.merge(llm_df[llm_cols], on="event_id", how="inner")
    df = df.merge(embed_df[emb_cols], on="event_id", how="inner")
    assert len(df) == n0, f"多模態合併後筆數異常：預期 {n0}，實際 {len(df)}！"
    assert df["event_id"].is_unique, "合併後存在重複 event_id！"

    # 波動率擴張比標籤（事後 20d RV / 事前 20d RV）
    df["label_rv_ratio_20d"] = df["label_rv_20d"] / df["pre_rv_20d"].replace(0.0, np.nan)

    # 領域多模態交互項（嚴格 Point-in-Time，不涉及任何未來欄位）
    df["opt_x_vix"] = df["optimism_score"] * df["pre_vix_close"]
    df["amb_x_rv20"] = df["ambiguity_score"] * df["pre_rv_20d"]
    df["drift_x_rv20"] = df["qoq_prep_drift"] * df["pre_rv_20d"]
    df["opt_x_sue_sign"] = df["optimism_score"] * df["sue_beat_flag"].fillna(0.0)
    df["tone_drop_x_evas"] = (-df["qa_tone_drop"]) * df["qa_evasiveness"]

    # 嚴格欄位篩選與順序（確保無多餘未受控欄位洩漏）
    all_feat_cols: List[str] = [c for cols in FEATURE_GROUPS.values() for c in cols]
    assert len(all_feat_cols) == len(set(all_feat_cols)), "FEATURE_GROUPS 存在重複特徵名！"
    assert set(KEY_META_COLUMNS).isdisjoint(all_feat_cols), "鍵值欄位不可進入特徵矩陣！"
    assert set(LABEL_COLUMNS).isdisjoint(all_feat_cols), "標籤欄位不可進入特徵矩陣！"

    ordered_cols = [*KEY_META_COLUMNS, *all_feat_cols, *LABEL_COLUMNS]
    missing_cols = [c for c in ordered_cols if c not in df.columns]
    assert not missing_cols, f"主特徵表缺少預期欄位：{missing_cols}"
    df = df[ordered_cols].copy()

    # Fail-Fast 防禦性驗證：時間單調性、零 Inf、核心標籤完整
    assert (df["pre_end_date"] < df["event_trading_date"]).all(), "發現 pre_end_date >= event_trading_date！"
    assert (df["event_trading_date"] < df["post_10d_date"]).all(), "發現 event_trading_date >= post_10d_date！"
    assert (df["post_10d_date"] < df["post_20d_date"]).all(), "發現 post_10d_date >= post_20d_date！"

    num_cols = df.select_dtypes(include=[np.number]).columns
    inf_mask = np.isinf(df[num_cols].to_numpy(dtype=float))
    assert not inf_mask.any(), "主特徵表含 +/-Inf 異常數值！"

    return df


def _spearman(s1: pd.Series, s2: pd.Series) -> float:
    both = pd.concat([s1, s2], axis=1).dropna()
    if len(both) < 20 or both.iloc[:, 0].nunique() < 2 or both.iloc[:, 1].nunique() < 2:
        return float("nan")
    return float(both.iloc[:, 0].rank().corr(both.iloc[:, 1].rank()))


def write_stage4_audit_report(df: pd.DataFrame, md_path: Path = STAGE4_AUDIT_MD) -> None:
    """輸出 Stage 4 多模態特徵表品質稽核報告（僅於 2016–2023 訓練期檢視單因子 RankIC，嚴禁偷看 OOS）。"""
    all_feat_cols = [c for cols in FEATURE_GROUPS.values() for c in cols]
    eps_match_rate = df["sue_available"].mean() * 100.0
    label_na_counts = {c: int(df[c].isna().sum()) for c in LABEL_COLUMNS}
    n_post_cutoff_2025 = int((df["year"] >= QUANT_CFG.post_cutoff_start_year).sum())
    n_post_cutoff_2024_2025 = int((df["year"] >= 2024).sum())

    lines: List[str] = [
        "# Stage 4 多模態特徵表與可交易標籤稽核報告",
        "",
        f"- **產出時間 (UTC)**：`{datetime.now(timezone.utc).isoformat()}`",
        f"- **主特徵表路徑**：`{MODEL_DATASET_PATH}`",
        f"- **總樣本數**：`{len(df):,}` 筆（`{df['symbol'].nunique()}` 檔股票、`{df['gics_sector'].nunique()}` 個 GICS 產業、年份 `{int(df['year'].min())}–{int(df['year'].max())}`）",
        f"- **欄位結構**：鍵值與元資料 `{len(KEY_META_COLUMNS)}` 欄 + 特徵 `{len(all_feat_cols)}` 欄（6 大特徵群組）+ 目標標籤 `{len(LABEL_COLUMNS)}` 欄 = 合計 `{df.shape[1]}` 欄",
        f"- **EPS Surprise 配對率（±{QUANT_CFG.eps_match_max_days} 日曆日）**：**{eps_match_rate:.2f}%**（`{int(df['sue_available'].sum()):,} / {len(df):,}`）",
        f"- **嚴格樣本外保留樣本數**：2024–2025 共 `{n_post_cutoff_2024_2025}` 場（其中 2025 年 Post-Cutoff 獨立驗收集共 `{n_post_cutoff_2025}` 場）",
        f"- **防禦性驗證**：時間順序 `pre_end < event_date < post_10d < post_20d` 100% 通過；全表零 `+/-Inf`；核心目標標籤缺失數 = `{label_na_counts['label_excess_ret_20d']}`（20d 報酬） / `{label_na_counts['label_rv_20d']}`（20d 波動率）",
        "",
        "## 一、六大特徵群組概覽",
        "",
        "| 特徵群組 | 欄位數 | 資訊時間邊界 | 平均缺失率 | 最高缺失欄位 |",
        "| :--- | :---: | :--- | :---: | :--- |",
    ]
    group_notes = {
        "market_pre": "嚴格 <= Close(t-1)",
        "hard_event": "財報發布與文本結構（Open/Close 皆可用）",
        "event_day_close_only": "僅 Close(t=0) 進場可用（Open 敏感度測試自動排除）",
        "llm_sentiment": "Stage 3A 去識別化文本 LLM 評分與跨季差分",
        "embed_drift": "Stage 3B 去識別化區段 Embedding 語義偏移",
        "interactions": "多模態領域交互項（情緒 × 波動／硬財報）",
    }
    for gname, gcols in FEATURE_GROUPS.items():
        na_rates = df[list(gcols)].isna().mean() * 100.0
        worst_col = str(na_rates.idxmax())
        lines.append(
            f"| `{gname}` | {len(gcols)} | {group_notes[gname]} | {na_rates.mean():.2f}% | `{worst_col}` ({na_rates.max():.2f}%) |"
        )

    lines.extend(
        [
            "",
            "## 二、事後可交易目標標籤分佈（嚴格排除隔夜跳空，自 Close(t=0) 起算）",
            "",
            "| 目標標籤 | 有效筆數 | Mean | Std | P10 | Median | P90 | 缺失數 |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
    )
    for lcol in (
        "label_excess_ret_10d",
        "label_excess_ret_20d",
        "label_open_excess_ret_10d",
        "label_open_excess_ret_20d",
        "label_rv_10d",
        "label_rv_20d",
        "label_log_rv_20d",
        "label_rv_ratio_20d",
    ):
        s = df[lcol].astype(float)
        lines.append(
            f"| `{lcol}` | {int(s.notna().sum()):,} | {s.mean():+.4f} | {s.std(ddof=1):.4f} | "
            f"{s.quantile(0.10):+.4f} | {s.median():+.4f} | {s.quantile(0.90):+.4f} | {int(s.isna().sum())} |"
        )

    # 嚴格僅在 2016-2023 訓練觀察期檢視單因子 RankIC，完全隔離 2024-2025（杜絕 Test Set Snooping！）
    train_preview = df[df["year"] <= 2023]
    ic_rows = []
    for col in all_feat_cols:
        ic_ret = _spearman(train_preview[col], train_preview["label_excess_ret_20d"])
        ic_rv = _spearman(train_preview[col], train_preview["label_rv_20d"])
        ic_rows.append((col, ic_ret, ic_rv))

    ic_df = pd.DataFrame(ic_rows, columns=["feature", "ic_ret_20d", "ic_rv_20d"])
    top_ret = ic_df.reindex(ic_df["ic_ret_20d"].abs().sort_values(ascending=False).index).head(12)
    top_rv = ic_df.reindex(ic_df["ic_rv_20d"].abs().sort_values(ascending=False).index).head(12)

    lines.extend(
        [
            "",
            f"## 三、防洩漏與單因子訊號健檢（嚴格限縮於 `2016–2023` 歷史樣本 `n={len(train_preview):,}`，完全封鎖 2024–2025 OOS）",
            "",
            "### 3.1 與 20 日 PEAD 超額報酬 (`label_excess_ret_20d`) 之 Top 12 單因子 Spearman RankIC",
            "",
            "| 排名 | 特徵 | Spearman RankIC (2016–2023) |",
            "| :---: | :--- | :---: |",
        ]
    )
    for idx, r in enumerate(top_ret.itertuples(index=False), 1):
        lines.append(f"| {idx} | `{r.feature}` | **{r.ic_ret_20d:+.4f}** |")

    lines.extend(
        [
            "",
            "### 3.2 與 20 日事後已實現波動率 (`label_rv_20d`) 之 Top 12 單因子 Spearman RankIC",
            "",
            "| 排名 | 特徵 | Spearman RankIC (2016–2023) |",
            "| :---: | :--- | :---: |",
        ]
    )
    for idx, r in enumerate(top_rv.itertuples(index=False), 1):
        lines.append(f"| {idx} | `{r.feature}` | **{r.ic_rv_20d:+.4f}** |")

    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(lines), encoding="utf-8")


def run_stage4() -> None:
    """執行 Stage 4 完整多模態特徵建置與稽核流程。"""
    ensure_project_dirs()
    parquet_glob = (CLEAN_TRANSCRIPTS_DIR / "year=*" / "*.parquet").as_posix()
    con = duckdb.connect()
    try:
        clean_meta = con.execute(
            f"""
            SELECT
                event_id, symbol, gics_sector, year, quarter,
                session_type, alignment_flag,
                event_trading_date, pre_end_date, post_10d_date, post_20d_date,
                prepared_word_count, qa_word_count, total_word_count
            FROM read_parquet('{parquet_glob}', hive_partitioning=true)
            WHERE is_valid = TRUE
            ORDER BY symbol, event_trading_date, event_id
            """
        ).df()
    finally:
        con.close()

    llm_df = pq.read_table(LLM_FEATURES_PATH).to_pandas()
    embed_df = pq.read_table(EMBED_DRIFT_FEATURES_PATH).to_pandas()
    ohlcv_df = pq.read_table(RAW_MARKET_OHLCV_PATH).to_pandas()
    macro_df = pq.read_table(RAW_MACRO_INDEX_PATH).to_pandas()
    eps_df = pq.read_table(RAW_EPS_SUE_PATH).to_pandas()

    model_df = assemble_model_dataset(
        clean_meta=clean_meta,
        llm_df=llm_df,
        embed_df=embed_df,
        ohlcv_df=ohlcv_df,
        macro_df=macro_df,
        eps_df=eps_df,
    )
    atomic_write_parquet(model_df, MODEL_DATASET_PATH)
    write_stage4_audit_report(model_df, STAGE4_AUDIT_MD)
    print(f"[Stage 4] 主特徵表已產出：{MODEL_DATASET_PATH} ({model_df.shape[0]:,} rows x {model_df.shape[1]} cols)")
    print(f"[Stage 4] 品質稽核報告已寫入：{STAGE4_AUDIT_MD}")


if __name__ == "__main__":
    run_stage4()
