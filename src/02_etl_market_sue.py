"""Stage 1B：個股還原日 K、大盤指標、EPS Surprise（SUE）與 NYSE 交易日曆擷取。

功能：
1. 透過 pandas_market_calendars 建立 2015–2027 年 NYSE 官方交易日曆（含半日市標記）。
2. 透過 yfinance（顯式 auto_adjust=True）分批抓取目標股票還原日 K（OHLCV）與大盤指數（^GSPC、^VIX）。
3. 透過 yfinance get_earnings_dates 抓取歷史 EPS 預期與實際公布值，供消融實驗控制硬財報驚喜（SUE）。
4. 以 DuckDB 對全部 01_raw Parquet 執行跨表稽核，產出 outputs/stage1_raw_audit.md。

執行範例：
    python -m src.02_etl_market_sue --pilot      # 先以 20 檔 Pilot 試跑至 data/01_raw_pilot/
    python -m src.02_etl_market_sue              # 正式建置 120 檔全量 data/01_raw/
    python -m src.02_etl_market_sue --audit-only # 僅針對既有 data/01_raw/ 重跑 DuckDB 稽核
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, TypeVar

import duckdb
import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
import yfinance as yf

from src.config import (
    DATA_CFG,
    DATA_DIR,
    MARKET_ETL_CFG,
    OUTPUTS_DIR,
    RAW_DIR,
    RAW_EPS_SUE_PATH,
    RAW_MACRO_INDEX_PATH,
    RAW_MARKET_OHLCV_PATH,
    RAW_NYSE_CALENDAR_PATH,
    RAW_TRANSCRIPTS_DIR,
    STAGE1_AUDIT_MD,
    ensure_project_dirs,
    load_universe,
)
from src.io_utils import atomic_write_parquet

T = TypeVar("T")

OHLCV_COLUMNS = ["symbol", "date", "open", "high", "low", "close", "volume"]
EPS_COLUMNS = [
    "symbol",
    "earnings_datetime_et",
    "earnings_date",
    "eps_estimate",
    "eps_actual",
    "reported_surprise_pct",
    "computed_surprise_pct",
]


def _retry_call(
    fn: Callable[[], T],
    max_retries: int = MARKET_ETL_CFG.max_retries,
    backoff_base: float = MARKET_ETL_CFG.retry_backoff_base_sec,
    label: str = "",
) -> T:
    """指數退避重試包裝器，應對外部行情 API 暫時性限流或逾時。"""
    last_err: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as exc:
            last_err = exc
            if attempt < max_retries - 1:
                wait_sec = backoff_base * (2**attempt)
                print(
                    f"[Retry] {label} 第 {attempt + 1} 次失敗（{exc}），{wait_sec:.1f}s 後重試...",
                    flush=True,
                )
                time.sleep(wait_sec)
    assert last_err is not None
    raise last_err


# ==================== 純函式資料轉換與驗證（易於單元測試） ====================


def build_nyse_calendar(
    start_date: str = MARKET_ETL_CFG.calendar_start_date,
    end_date: str = MARKET_ETL_CFG.calendar_end_date,
    exchange_tz: str = DATA_CFG.exchange_tz,
) -> pd.DataFrame:
    """建立 NYSE 官方開盤日曆表（含美東開收盤時間與半日市標記）。"""
    nyse = mcal.get_calendar(DATA_CFG.nyse_calendar_name)
    sched = nyse.schedule(start_date=start_date, end_date=end_date)
    if sched.empty:
        raise ValueError(f"無法取得 {start_date} 至 {end_date} 之 NYSE 交易日曆。")

    open_et = sched["market_open"].dt.tz_convert(exchange_tz)
    close_et = sched["market_close"].dt.tz_convert(exchange_tz)
    duration_hours = (close_et - open_et).dt.total_seconds() / 3600.0

    cal_df = pd.DataFrame(
        {
            "date": pd.to_datetime(sched.index).normalize(),
            "market_open_et": open_et.dt.strftime("%Y-%m-%d %H:%M:%S%z"),
            "market_close_et": close_et.dt.strftime("%Y-%m-%d %H:%M:%S%z"),
            "is_half_day": (duration_hours < 6.0).astype(bool),
        }
    ).reset_index(drop=True)

    assert cal_df["date"].is_monotonic_increasing, "NYSE 交易日曆日期非嚴格遞增！"
    assert cal_df["date"].is_unique, "NYSE 交易日曆存在重複日期！"
    return cal_df


def normalize_yf_ohlcv_batch(raw_df: pd.DataFrame, tickers: Sequence[str]) -> pd.DataFrame:
    """將 yfinance.download 回傳（單檔平面表或多檔 MultiIndex）標準化為整齊長表。"""
    if raw_df is None or raw_df.empty:
        return pd.DataFrame(columns=OHLCV_COLUMNS)

    frames: List[pd.DataFrame] = []
    if isinstance(raw_df.columns, pd.MultiIndex):
        level0 = set(raw_df.columns.get_level_values(0))
        level1 = set(raw_df.columns.get_level_values(1))
        for sym in tickers:
            if sym in level0:
                sub = raw_df[sym].copy()
            elif sym in level1:
                sub = raw_df.xs(sym, axis=1, level=1).copy()
            else:
                continue
            sub = _standardize_single_ohlcv(sub, sym)
            if not sub.empty:
                frames.append(sub)
    else:
        if len(tickers) != 1:
            raise ValueError(
                f"多檔下載（{len(tickers)} 檔）卻收到非 MultiIndex 欄位：{list(raw_df.columns)}"
            )
        sub = _standardize_single_ohlcv(raw_df.copy(), tickers[0])
        if not sub.empty:
            frames.append(sub)

    if not frames:
        return pd.DataFrame(columns=OHLCV_COLUMNS)

    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(subset=["symbol", "date"], keep="last")
    out = out.sort_values(["symbol", "date"]).reset_index(drop=True)
    return out


def _standardize_single_ohlcv(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """標準化單一標的日 K 欄位型別並過濾無效列。"""
    col_map = {str(c).strip().lower(): c for c in df.columns}
    if "close" not in col_map:
        return pd.DataFrame(columns=OHLCV_COLUMNS)

    dt_idx = pd.to_datetime(df.index, errors="coerce")
    if getattr(dt_idx, "tz", None) is not None:
        dt_idx = dt_idx.tz_localize(None)

    out = pd.DataFrame(
        {
            "symbol": str(symbol),
            "date": dt_idx.normalize(),
            "open": pd.to_numeric(df[col_map["open"]], errors="coerce")
            if "open" in col_map
            else np.nan,
            "high": pd.to_numeric(df[col_map["high"]], errors="coerce")
            if "high" in col_map
            else np.nan,
            "low": pd.to_numeric(df[col_map["low"]], errors="coerce")
            if "low" in col_map
            else np.nan,
            "close": pd.to_numeric(df[col_map["close"]], errors="coerce"),
            "volume": pd.to_numeric(df[col_map["volume"]], errors="coerce")
            if "volume" in col_map
            else np.nan,
        }
    )
    out = out.dropna(subset=["date", "close"])
    out = out[out["close"] > 0].copy()
    return out[OHLCV_COLUMNS]


def normalize_eps_dates_df(
    symbol: str,
    raw_df: Optional[pd.DataFrame],
    exchange_tz: str = DATA_CFG.exchange_tz,
) -> pd.DataFrame:
    """將 yfinance get_earnings_dates 結果標準化並安全計算 EPS Surprise (%)。"""
    if raw_df is None or raw_df.empty:
        return pd.DataFrame(columns=EPS_COLUMNS)

    idx = pd.to_datetime(raw_df.index, errors="coerce")
    if getattr(idx, "tz", None) is None:
        idx_et = idx.tz_localize(exchange_tz, nonexistent="shift_forward", ambiguous="NaT")
    else:
        idx_et = idx.tz_convert(exchange_tz)

    col_lookup = {str(c).strip().lower(): c for c in raw_df.columns}
    est_col = col_lookup.get("eps estimate")
    act_col = col_lookup.get("reported eps")
    surp_col = col_lookup.get("surprise(%)")

    eps_est = (
        pd.to_numeric(raw_df[est_col], errors="coerce").to_numpy(dtype=float)
        if est_col
        else np.full(len(raw_df), np.nan)
    )
    eps_act = (
        pd.to_numeric(raw_df[act_col], errors="coerce").to_numpy(dtype=float)
        if act_col
        else np.full(len(raw_df), np.nan)
    )
    rep_surp = (
        pd.to_numeric(raw_df[surp_col], errors="coerce").to_numpy(dtype=float)
        if surp_col
        else np.full(len(raw_df), np.nan)
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        denom = np.abs(eps_est)
        comp_surp = np.where(
            np.isfinite(eps_est) & np.isfinite(eps_act) & (denom > 1e-9),
            (eps_act - eps_est) / denom * 100.0,
            np.nan,
        )

    out = pd.DataFrame(
        {
            "symbol": str(symbol),
            "earnings_datetime_et": idx_et.strftime("%Y-%m-%d %H:%M:%S%z"),
            "earnings_date": idx_et.tz_localize(None).normalize(),
            "eps_estimate": eps_est,
            "eps_actual": eps_act,
            "reported_surprise_pct": rep_surp,
            "computed_surprise_pct": comp_surp,
        }
    )

    # 排除尚未公布的未來財報預告列（實際 EPS 與驚喜皆空值）
    out = out.dropna(subset=["earnings_date"])
    out = out[~(out["eps_actual"].isna() & out["reported_surprise_pct"].isna())].copy()
    out = out.drop_duplicates(subset=["symbol", "earnings_date"], keep="first")
    out = out.sort_values(["symbol", "earnings_date"]).reset_index(drop=True)
    return out[EPS_COLUMNS]


def validate_ohlcv_completeness(
    ohlcv_df: pd.DataFrame,
    expected_symbols: Sequence[str],
    benchmark_days: int,
    min_coverage_ratio: float = MARKET_ETL_CFG.min_ohlcv_coverage_ratio,
) -> None:
    """Fail-Fast 檢核每檔個股皆達最低交易日覆蓋比例，防止下市／私有化殘缺股漏網。"""
    if benchmark_days <= 0:
        raise ValueError(f"基準交易日數必須大於 0，收到：{benchmark_days}")

    min_required = int(benchmark_days * min_coverage_ratio)
    counts = ohlcv_df.groupby("symbol")["date"].nunique().to_dict() if not ohlcv_df.empty else {}

    missing = sorted(s for s in expected_symbols if s not in counts)
    if missing:
        raise ValueError(f"以下標的無法從 yfinance 取得任何股價：{missing}")

    incomplete = {
        s: int(counts[s])
        for s in expected_symbols
        if int(counts[s]) < min_required
    }
    if incomplete:
        raise ValueError(
            f"以下標的交易日數低於門檻（須 >= {min_required} 天，即大盤 {benchmark_days} 天之 "
            f"{min_coverage_ratio:.0%}），疑為下市／私有化或晚掛牌標的：{incomplete}"
        )


# ==================== 外部下載管線 ====================


def fetch_ohlcv_for_tickers(
    tickers: Sequence[str],
    start_date: str = DATA_CFG.market_data_start_date,
    batch_size: int = MARKET_ETL_CFG.yf_batch_size,
    sleep_sec: float = MARKET_ETL_CFG.yf_batch_sleep_sec,
) -> pd.DataFrame:
    """分批透過 yfinance 下載還原日 K（強制 auto_adjust=True）。"""
    all_batches: List[pd.DataFrame] = []
    ticker_list = list(tickers)

    for start_idx in range(0, len(ticker_list), batch_size):
        batch = ticker_list[start_idx : start_idx + batch_size]
        label = f"OHLCV batch {start_idx + 1}-{start_idx + len(batch)}/{len(ticker_list)}"
        print(f"[Stage 1B] 下載 {label}...", flush=True)

        def _download(b: List[str] = batch) -> pd.DataFrame:
            return yf.download(
                tickers=b,
                start=start_date,
                auto_adjust=True,
                progress=False,
                group_by="ticker",
                threads=True,
            )

        raw = _retry_call(_download, label=label)
        norm = normalize_yf_ohlcv_batch(raw, batch)
        all_batches.append(norm)
        if start_idx + batch_size < len(ticker_list):
            time.sleep(sleep_sec)

    combined = pd.concat(all_batches, ignore_index=True) if all_batches else pd.DataFrame(columns=OHLCV_COLUMNS)
    combined = combined.drop_duplicates(subset=["symbol", "date"], keep="last")
    combined = combined.sort_values(["symbol", "date"]).reset_index(drop=True)

    # 檢查是否有漏抓標的並單檔補抓一次
    got_symbols = set(combined["symbol"].unique())
    missing = [s for s in ticker_list if s not in got_symbols]
    for sym in missing:
        print(f"[Stage 1B] 單檔補抓漏抓標的：{sym}", flush=True)
        raw_one = _retry_call(
            lambda s=sym: yf.download(
                tickers=s, start=start_date, auto_adjust=True, progress=False
            ),
            label=f"OHLCV single {sym}",
        )
        one_norm = normalize_yf_ohlcv_batch(raw_one, [sym])
        if not one_norm.empty:
            combined = pd.concat([combined, one_norm], ignore_index=True)

    combined = combined.drop_duplicates(subset=["symbol", "date"], keep="last")
    combined = combined.sort_values(["symbol", "date"]).reset_index(drop=True)
    return combined


def fetch_eps_sue_for_tickers(
    tickers: Sequence[str],
    limit: int = MARKET_ETL_CFG.yf_eps_limit,
    sleep_sec: float = MARKET_ETL_CFG.yf_eps_sleep_sec,
) -> Tuple[pd.DataFrame, List[str]]:
    """逐檔透過 yfinance 抓取歷史 EPS 預期與實際值。"""
    frames: List[pd.DataFrame] = []
    failed_tickers: List[str] = []
    total = len(tickers)

    for idx, sym in enumerate(tickers, start=1):
        if idx == 1 or idx % 20 == 0 or idx == total:
            print(f"[Stage 1B] 下載 EPS Surprise 進度：{idx}/{total} ({sym})", flush=True)
        try:
            raw_df = _retry_call(
                lambda s=sym: yf.Ticker(s).get_earnings_dates(limit=limit),
                label=f"EPS {sym}",
            )
            norm = normalize_eps_dates_df(sym, raw_df)
            if not norm.empty:
                frames.append(norm)
            else:
                failed_tickers.append(sym)
        except Exception as exc:
            print(f"[Stage 1B] 警告：{sym} EPS 抓取失敗（{exc}）", flush=True)
            failed_tickers.append(sym)
        time.sleep(sleep_sec)

    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=EPS_COLUMNS)
    if not combined.empty:
        combined = combined.drop_duplicates(subset=["symbol", "earnings_date"], keep="first")
        combined = combined.sort_values(["symbol", "earnings_date"]).reset_index(drop=True)
    return combined, failed_tickers


# ==================== DuckDB 跨表稽核報告 ====================


def run_stage1_audit(
    raw_dir: Path = RAW_DIR,
    audit_md_path: Path = STAGE1_AUDIT_MD,
) -> Dict[str, Any]:
    """以 DuckDB 直接檢核 01_raw 全部 Parquet 檔案之筆數、覆蓋率、型別與磁碟體積。"""
    transcripts_glob = str(raw_dir / "transcripts_raw" / "year=*" / "part.parquet").replace("\\", "/")
    ohlcv_path = str(raw_dir / "market_ohlcv_raw.parquet").replace("\\", "/")
    macro_path = str(raw_dir / "macro_index_raw.parquet").replace("\\", "/")
    eps_path = str(raw_dir / "eps_sue_raw.parquet").replace("\\", "/")
    cal_path = str(raw_dir / "nyse_calendar.parquet").replace("\\", "/")

    con = duckdb.connect()
    try:
        t_stats = con.execute(
            f"""
            SELECT
                COUNT(*) AS total_rows,
                COUNT(DISTINCT event_id) AS unique_events,
                COUNT(DISTINCT symbol) AS unique_symbols,
                MIN(year) AS min_year,
                MAX(year) AS max_year
            FROM read_parquet('{transcripts_glob}', hive_partitioning=true)
            """
        ).fetchone()

        m_stats = con.execute(
            f"""
            SELECT
                COUNT(*) AS total_rows,
                COUNT(DISTINCT symbol) AS unique_symbols,
                MIN(date)::VARCHAR AS min_date,
                MAX(date)::VARCHAR AS max_date,
                MIN(close) AS min_close,
                SUM(CASE WHEN close IS NULL OR close <= 0 THEN 1 ELSE 0 END) AS bad_close,
                (SELECT MIN(cnt) FROM (SELECT COUNT(*) AS cnt FROM read_parquet('{ohlcv_path}') GROUP BY symbol)) AS min_days_per_symbol
            FROM read_parquet('{ohlcv_path}')
            """
        ).fetchone()

        macro_rows = con.execute(
            f"""
            SELECT symbol, COUNT(*) AS n_rows, MIN(date)::VARCHAR AS min_d, MAX(date)::VARCHAR AS max_d
            FROM read_parquet('{macro_path}')
            GROUP BY symbol
            ORDER BY symbol
            """
        ).fetchall()

        eps_stats = con.execute(
            f"""
            SELECT
                COUNT(*) AS total_rows,
                COUNT(DISTINCT symbol) AS unique_symbols,
                MIN(earnings_date)::VARCHAR AS min_date,
                MAX(earnings_date)::VARCHAR AS max_date,
                SUM(CASE WHEN computed_surprise_pct IS NOT NULL THEN 1 ELSE 0 END) AS valid_sue_rows
            FROM read_parquet('{eps_path}')
            """
        ).fetchone()

        # 檢核法說會事件在 ±5 天內能匹配到 EPS Surprise 的比例
        eps_match = con.execute(
            f"""
            WITH t AS (
                SELECT event_id, symbol, TRY_CAST(SUBSTR(raw_date, 1, 10) AS DATE) AS t_date
                FROM read_parquet('{transcripts_glob}', hive_partitioning=true)
            ),
            e AS (
                SELECT symbol, CAST(earnings_date AS DATE) AS e_date, computed_surprise_pct
                FROM read_parquet('{eps_path}')
                WHERE computed_surprise_pct IS NOT NULL
            )
            SELECT
                COUNT(DISTINCT t.event_id) AS total_events,
                COUNT(DISTINCT CASE WHEN e.symbol IS NOT NULL THEN t.event_id END) AS matched_events
            FROM t
            LEFT JOIN e
              ON t.symbol = e.symbol
             AND ABS(DATEDIFF('day', t.t_date, e.e_date)) <= 5
            """
        ).fetchone()

        cal_stats = con.execute(
            f"""
            SELECT COUNT(*) AS n_days, MIN(date)::VARCHAR AS min_d, MAX(date)::VARCHAR AS max_d,
                   SUM(CASE WHEN is_half_day THEN 1 ELSE 0 END) AS half_days
            FROM read_parquet('{cal_path}')
            """
        ).fetchone()
    finally:
        con.close()

    # Fail-Fast 完整性斷言
    assert t_stats[0] == t_stats[1], f"法說會主鍵 event_id 存在重複：{t_stats[0]} vs {t_stats[1]}"
    assert m_stats[5] == 0, f"個股行情存在 close <= 0 或空值筆數：{m_stats[5]}"

    total_size_mb = sum(
        p.stat().st_size for p in raw_dir.rglob("*.parquet") if p.is_file()
    ) / (1024 * 1024)
    eps_match_pct = (eps_match[1] / max(eps_match[0], 1)) * 100.0

    lines = [
        "# Stage 1 原始資料層（01_raw）DuckDB 稽核報告",
        "",
        f"- **產出時間**：{datetime.now().isoformat(timespec='seconds')}",
        f"- **稽核目錄**：`{raw_dir}`",
        f"- **01_raw 總磁碟占用**：**{total_size_mb:.2f} MB**",
        "",
        "## 1. 法說會逐字稿（`transcripts_raw/year=*/part.parquet`）",
        f"- 總筆數：**{t_stats[0]:,}**（獨立 `event_id`：{t_stats[1]:,}，主鍵零重複）",
        f"- 涵蓋標的數：**{t_stats[2]}** 檔",
        f"- 年份區間：**{t_stats[3]} – {t_stats[4]}**",
        "",
        "## 2. 個股還原日 K（`market_ohlcv_raw.parquet`）",
        f"- 總筆數：**{m_stats[0]:,}** 列（涵蓋 **{m_stats[1]}** 檔標的，單檔最少交易日：**{m_stats[6]:,}** 天）",
        f"- 日期區間：**{m_stats[2]} – {m_stats[3]}**",
        f"- 最低收盤價：`{m_stats[4]:.4f}`（異常 `close <= 0` 筆數：**{m_stats[5]}**）",
        "",
        "## 3. 大盤與恐慌指標（`macro_index_raw.parquet`）",
        "| Symbol | 筆數 | 起始日 | 結束日 |",
        "|---|---:|---|---|",
    ]
    for sym, n_r, d0, d1 in macro_rows:
        lines.append(f"| `{sym}` | {n_r:,} | {d0} | {d1} |")

    lines.extend(
        [
            "",
            "## 4. 硬財報驚喜 EPS Surprise（`eps_sue_raw.parquet`）",
            f"- 總筆數：**{eps_stats[0]:,}** 筆（有效 SUE：{eps_stats[4]:,} 筆，涵蓋 **{eps_stats[1]}** 檔）",
            f"- 日期區間：**{eps_stats[2]} – {eps_stats[3]}**",
            f"- 與法說會事件（±5 天視窗）匹配率：**{eps_match[1]:,} / {eps_match[0]:,}（{eps_match_pct:.1f}%）**",
            "",
            "## 5. NYSE 官方交易日曆（`nyse_calendar.parquet`）",
            f"- 交易日總數：**{cal_stats[0]:,}** 天（{cal_stats[1]} – {cal_stats[2]}，含 {cal_stats[3]} 天半日市）",
        ]
    )

    if eps_match_pct < 80.0:
        lines.extend(
            [
                "",
                "> **風險提醒**：EPS Surprise 匹配率低於 80%，後續 Stage 4 建議同時建置「隔夜跳空報酬（EAR）」作為市場即時硬驚喜代理變數（依計畫書風險矩陣第 2 項）。",
            ]
        )

    audit_md_path.parent.mkdir(parents=True, exist_ok=True)
    audit_md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[Stage 1 Audit] 稽核報告已產出：{audit_md_path}", flush=True)
    return {
        "total_size_mb": round(total_size_mb, 2),
        "transcript_rows": int(t_stats[0]),
        "transcript_symbols": int(t_stats[2]),
        "ohlcv_rows": int(m_stats[0]),
        "ohlcv_symbols": int(m_stats[1]),
        "eps_rows": int(eps_stats[0]),
        "eps_match_pct": round(eps_match_pct, 2),
        "calendar_days": int(cal_stats[0]),
    }


def run_market_sue_etl(pilot: bool = False) -> Dict[str, Any]:
    """執行 Stage 1B 市場量價、大盤、EPS Surprise 與 NYSE 日曆建置。"""
    ensure_project_dirs()
    uni = load_universe()
    symbols: List[str] = list(uni["pilot_symbols"] if pilot else uni["symbols"])

    raw_dir = DATA_DIR / "01_raw_pilot" if pilot else RAW_DIR
    ohlcv_out = raw_dir / "market_ohlcv_raw.parquet"
    macro_out = raw_dir / "macro_index_raw.parquet"
    eps_out = raw_dir / "eps_sue_raw.parquet"
    cal_out = raw_dir / "nyse_calendar.parquet"
    audit_out = OUTPUTS_DIR / ("stage1_pilot_audit.md" if pilot else STAGE1_AUDIT_MD.name)

    mode_label = f"Pilot ({len(symbols)} 檔)" if pilot else f"全量 ({len(symbols)} 檔)"
    print(f"[Stage 1B] 啟動市場數據、SUE 與日曆擷取 — 模式：{mode_label}", flush=True)

    # 1. NYSE 日曆
    cal_df = build_nyse_calendar()
    atomic_write_parquet(cal_df, cal_out)
    print(f"[Stage 1B] NYSE 日曆已寫入：{len(cal_df):,} 個交易日 -> {cal_out}", flush=True)

    # 2. 大盤與 VIX（先抓大盤以取得基準交易日數）
    macro_df = fetch_ohlcv_for_tickers(list(MARKET_ETL_CFG.macro_tickers), batch_size=2, sleep_sec=0.0)
    missing_macro = sorted(set(MARKET_ETL_CFG.macro_tickers) - set(macro_df["symbol"].unique()))
    assert not missing_macro, f"缺少大盤指標：{missing_macro}"
    gspc_days = int(macro_df.loc[macro_df["symbol"] == "^GSPC", "date"].nunique())
    atomic_write_parquet(macro_df, macro_out)
    print(f"[Stage 1B] 大盤與 VIX 已寫入：{len(macro_df):,} 列（^GSPC {gspc_days:,} 天） -> {macro_out}", flush=True)

    # 3. 個股還原日 K + Fail-Fast 最低交易日覆蓋檢核
    ohlcv_df = fetch_ohlcv_for_tickers(symbols)
    validate_ohlcv_completeness(ohlcv_df, symbols, benchmark_days=gspc_days)
    atomic_write_parquet(ohlcv_df, ohlcv_out)
    print(f"[Stage 1B] 個股還原日 K 已寫入：{len(ohlcv_df):,} 列（{len(symbols)}/{len(symbols)} 達標） -> {ohlcv_out}", flush=True)

    # 4. EPS Surprise
    eps_df, failed_eps = fetch_eps_sue_for_tickers(symbols)
    if failed_eps:
        print(f"[Stage 1B] 注意：有 {len(failed_eps)} 檔標的無 EPS 紀錄：{failed_eps[:10]}", flush=True)
    atomic_write_parquet(eps_df, eps_out)
    print(f"[Stage 1B] EPS Surprise 已寫入：{len(eps_df):,} 筆 -> {eps_out}", flush=True)

    # 5. 若逐字稿也已就緒，自動執行 DuckDB 跨表稽核
    audit_res: Optional[Dict[str, Any]] = None
    if list((raw_dir / "transcripts_raw").glob("year=*/part.parquet")):
        audit_res = run_stage1_audit(raw_dir=raw_dir, audit_md_path=audit_out)

    return {
        "mode": "pilot" if pilot else "full",
        "calendar_days": len(cal_df),
        "ohlcv_rows": len(ohlcv_df),
        "macro_rows": len(macro_df),
        "eps_rows": len(eps_df),
        "failed_eps_tickers": failed_eps,
        "audit": audit_res,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 1B：市場量價、大盤、EPS Surprise 與 NYSE 日曆擷取")
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="僅擷取 20 檔 Pilot 標的至 data/01_raw_pilot/ 進行試跑驗證",
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="僅針對既有 01_raw 執行 DuckDB 跨表稽核",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.audit_only:
        run_stage1_audit(
            raw_dir=DATA_DIR / "01_raw_pilot" if args.pilot else RAW_DIR,
            audit_md_path=OUTPUTS_DIR / ("stage1_pilot_audit.md" if args.pilot else STAGE1_AUDIT_MD.name),
        )
    else:
        run_market_sue_etl(pilot=args.pilot)
