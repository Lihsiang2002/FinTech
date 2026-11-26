"""專案單一事實來源設定模組（Single Source of Truth）。

集中管理所有目錄路徑、資料源識別碼、時間對齊參數、量化回測超參數與 Stage 0 探勘參數，
確保全專案修改任一設定時僅需改動此處。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ==================== 專案根目錄與分層路徑 ====================
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

CONFIG_DIR: Path = PROJECT_ROOT / "config"
PROMPTS_DIR: Path = CONFIG_DIR / "prompts"
UNIVERSE_PATH: Path = CONFIG_DIR / "universe.json"
REDACT_DICT_PATH: Path = CONFIG_DIR / "redact_dict.json"

DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "01_raw"
RAW_TRANSCRIPTS_DIR: Path = RAW_DIR / "transcripts_raw"
RAW_MARKET_OHLCV_PATH: Path = RAW_DIR / "market_ohlcv_raw.parquet"
RAW_MACRO_INDEX_PATH: Path = RAW_DIR / "macro_index_raw.parquet"
RAW_EPS_SUE_PATH: Path = RAW_DIR / "eps_sue_raw.parquet"
RAW_NYSE_CALENDAR_PATH: Path = RAW_DIR / "nyse_calendar.parquet"

CLEAN_DIR: Path = DATA_DIR / "02_clean"
CLEAN_TRANSCRIPTS_DIR: Path = CLEAN_DIR / "transcripts_clean"
CACHE_DIR: Path = DATA_DIR / "03_cache"
API_EXTRACT_CACHE_PATH: Path = CACHE_DIR / "api_extract_cache.jsonl"
RETEST_CACHE_PATH: Path = CACHE_DIR / "api_retest_cache.jsonl"
PIPELINE_STATE_DB_PATH: Path = CACHE_DIR / "pipeline_state.sqlite"
BATCH_WORKDIR: Path = CACHE_DIR / "openai_batches"

FEATURES_DIR: Path = DATA_DIR / "04_features"
LLM_FEATURES_PATH: Path = FEATURES_DIR / "llm_features.parquet"

OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"
STAGE1_AUDIT_MD: Path = OUTPUTS_DIR / "stage1_raw_audit.md"
STAGE2_AUDIT_MD: Path = OUTPUTS_DIR / "stage2_clean_audit.md"
STAGE3_SMOKE_JSON: Path = OUTPUTS_DIR / "stage3_smoke_5calls_v2.json"
STAGE3_SMOKE_MD: Path = OUTPUTS_DIR / "stage3_smoke_5calls_v2.md"
STAGE3_PILOT_AUDIT_MD: Path = OUTPUTS_DIR / "stage3_pilot_audit.md"
STAGE3_AUDIT_MD: Path = OUTPUTS_DIR / "stage3_llm_audit.md"

ENV_FILE_CANDIDATES: Tuple[Path, ...] = (
    PROJECT_ROOT / "config.env",
    PROJECT_ROOT / ".env",
)


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


@dataclass(frozen=True)
class MarketETLSettings:
    """Stage 1 市場量價、大盤、EPS Surprise 與交易日曆擷取設定。"""

    macro_tickers: Tuple[str, ...] = ("^GSPC", "^VIX")
    calendar_start_date: str = "2015-01-01"
    calendar_end_date: str = "2027-12-31"
    yf_batch_size: int = 20
    yf_batch_sleep_sec: float = 0.5
    yf_eps_limit: int = 48
    yf_eps_sleep_sec: float = 0.2
    max_retries: int = 3
    retry_backoff_base_sec: float = 1.5
    min_ohlcv_coverage_ratio: float = 0.95
    parquet_compression: str = "zstd"


@dataclass(frozen=True)
class CleanSettings:
    """Stage 2 文本切分、去識別化與軟刪除品質門檻設定。"""

    header_speaker_labels: Tuple[str, ...] = (
        "executives",
        "analysts",
        "corporate participants",
        "conference call participants",
        "company participants",
        "presentation",
    )
    qa_handoff_patterns: Tuple[str, ...] = (
        r"question[\s\-]+and[\s\-]+answer",
        r"\bq\s*&\s*a\b",
        r"first question comes from",
        r"our first question",
        r"open the (?:floor|line|lines|call) for questions",
        r"open it up (?:for|to) questions",
        r"we will now begin the question",
        r"(?:happy|pleased|ready) to take (?:your )?questions",
        r"ready for (?:your )?questions",
        r"concludes our (?:formal|prepared) (?:remarks|comments)",
        r"\[operator instructions?\]",
        r"we(?:'ll| will) (?:go|move) first to",
        r"limit yourself to one",
        r"turn (?:the call )?(?:back )?over to (?:the )?operator",
    )
    min_prepared_words_before_qa: int = 200
    min_qa_words: int = 150
    audit_sample_size: int = 10


@dataclass(frozen=True)
class LLMSettings:
    """Stage 3A OpenAI 結構化語義抽取與 Batch API 設定。"""

    provider: str = "openai"
    model: str = "gpt-4o-mini"
    prompt_version: str = "v2"
    prompt_path: Path = PROMPTS_DIR / "extract_v2.yaml"
    temperature: float = 0.0
    seed: int = 42
    max_completion_tokens: int = 500
    request_timeout_sec: float = 90.0
    max_retries: int = 3
    retry_backoff_base_sec: float = 2.0
    batch_endpoint: str = "/v1/chat/completions"
    batch_completion_window: str = "24h"
    batch_shard_size: int = 100
    max_batch_file_mb: float = 150.0
    smoke_test_count: int = 5
    pilot_quarters_per_symbol: int = 10
    retest_sample_size: int = 30
    # OpenAI gpt-4o-mini 官方參考定價（USD / 1M tokens）
    sync_input_cost_per_1m: float = 0.15
    sync_output_cost_per_1m: float = 0.60
    batch_input_cost_per_1m: float = 0.075
    batch_output_cost_per_1m: float = 0.30


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
MARKET_ETL_CFG = MarketETLSettings()
CLEAN_CFG = CleanSettings()
LLM_CFG = LLMSettings()
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
        BATCH_WORKDIR,
        FEATURES_DIR,
        OUTPUTS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)


def load_project_env(
    candidates: Tuple[Path, ...] = ENV_FILE_CANDIDATES,
    override: bool = False,
) -> Optional[Path]:
    """依序載入專案環境變數檔（預設優先 config.env，其次 .env），回傳實際載入之路徑。"""
    from dotenv import load_dotenv

    for env_path in candidates:
        if env_path.exists():
            load_dotenv(dotenv_path=env_path, override=override)
            return env_path
    return None


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
