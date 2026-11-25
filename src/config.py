"""專案單一事實來源設定模組（Single Source of Truth）。

集中管理所有目錄路徑、資料源識別碼、時間對齊參數、量化回測超參數與 Stage 0 探勘參數，
確保全專案修改任一設定時僅需改動此處。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple


# ==================== 專案根目錄與分層路徑 ====================
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

CONFIG_DIR: Path = PROJECT_ROOT / "config"
PROMPTS_DIR: Path = CONFIG_DIR / "prompts"
UNIVERSE_PATH: Path = CONFIG_DIR / "universe.json"
REDACT_DICT_PATH: Path = CONFIG_DIR / "redact_dict.json"

DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "01_raw"
CLEAN_DIR: Path = DATA_DIR / "02_clean"
CACHE_DIR: Path = DATA_DIR / "03_cache"
FEATURES_DIR: Path = DATA_DIR / "04_features"

OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"


# ==================== 資料源與樣本範圍設定 ====================
@dataclass(frozen=True)
class DataSettings:
    """資料來源、年份區間與品質門檻設定。"""

    hf_dataset_id: str = "kurry/sp500_earnings_transcripts"
    hf_split: str = "train"
    start_year: int = 2016
    end_year: int = 2026
    market_data_start_date: str = "2015-01-01"
    nyse_calendar_name: str = "NYSE"
    exchange_tz: str = "America/New_York"
    bmo_earliest_hour: int = 6
    market_open_hour: int = 9
    market_open_minute: int = 30
    min_word_count: int = 1500
    dedup_date_window_days: int = 15
    target_universe_min: int = 100
    target_universe_max: int = 150
    min_quarter_completeness: float = 0.90


# ==================== 歷史代碼更名映射（僅限同公司純更名，嚴禁併購標的） ====================
DEFAULT_RENAME_MAP: Dict[str, str] = {
    "FB": "META",
    "ANTM": "ELV",
    "FISV": "FI",
    "COG": "CTRA",
    "WLTW": "WTW",
    "PKI": "RVTY",
    "ABC": "COR",
    "PEAK": "DOC",
    "FLT": "CPAY",
    "RE": "EG",
}


# ==================== Stage 0 探勘參數 ====================
@dataclass(frozen=True)
class ProbeSettings:
    """Stage 0 資料品質前置探勘專用參數。"""

    amc_anchor_tickers: Tuple[str, ...] = ("AAPL", "NVDA", "MSFT", "META", "AMZN")
    bmo_anchor_tickers: Tuple[str, ...] = ("JPM", "BAC", "UNH", "JNJ", "PG")
    qa_marker_patterns: Tuple[str, ...] = (
        r"question[\s\-]+and[\s\-]+answer",
        r"q\s*&\s*a",
        r"first question comes from",
        r"open the (?:floor|line|call) for questions",
        r"we will now begin the question",
        r"our first question",
    )
    sample_years_for_inspection: Tuple[int, ...] = (2016, 2020, 2024)
    schema_sample_limit: int = 200
    probe_report_json: Path = OUTPUTS_DIR / "stage0_probe_report.json"
    probe_report_md: Path = OUTPUTS_DIR / "stage0_probe_summary.md"
    symbol_coverage_csv: Path = OUTPUTS_DIR / "stage0_symbol_coverage.csv"


# ==================== 量化建模與回測參數 ====================
@dataclass(frozen=True)
class QuantSettings:
    """量化特徵、Walk-Forward 切分與交易成本設定。"""

    random_seed: int = 42
    pre_car_windows: Tuple[int, ...] = (20, 60, 120)
    pead_horizons: Tuple[int, ...] = (10, 20)
    purge_gap_trading_days: int = 20
    one_way_cost_bps: float = 10.0
    train_start_year: int = 2016
    initial_train_end_year: int = 2018
    walk_forward_end_year: int = 2024
    post_cutoff_start_year: int = 2025


DATA_CFG = DataSettings()
PROBE_CFG = ProbeSettings()
QUANT_CFG = QuantSettings()


def ensure_project_dirs() -> None:
    """建立專案標準資料夾結構（冪等操作，不覆寫既有資料）。"""
    for directory in (
        CONFIG_DIR,
        PROMPTS_DIR,
        RAW_DIR,
        CLEAN_DIR,
        CACHE_DIR,
        FEATURES_DIR,
        OUTPUTS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)


def load_universe(path: Path = UNIVERSE_PATH) -> Dict[str, object]:
    """讀取股票池設定檔，若檔案不存在則拋出明確錯誤。"""
    if not path.exists():
        raise FileNotFoundError(
            f"找不到股票池設定檔：{path}。請先完成 Stage 0 探勘並產生 config/universe.json。"
        )
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if "symbols" not in data or not isinstance(data["symbols"], list):
        raise ValueError(f"股票池設定檔格式錯誤，缺少 symbols 清單：{path}")
    return data
