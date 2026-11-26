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
EMBED_CACHE_DIR: Path = CACHE_DIR / "embed_sections"
EMBED_BATCH_WORKDIR: Path = CACHE_DIR / "openai_embed_batches"

FEATURES_DIR: Path = DATA_DIR / "04_features"
LLM_FEATURES_PATH: Path = FEATURES_DIR / "llm_features.parquet"
EMBED_DRIFT_FEATURES_PATH: Path = FEATURES_DIR / "embed_drift_features.parquet"
SECTION_EMBEDDINGS_PATH: Path = FEATURES_DIR / "section_embeddings.parquet"
MODEL_DATASET_PATH: Path = FEATURES_DIR / "model_dataset.parquet"

OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"
STAGE1_AUDIT_MD: Path = OUTPUTS_DIR / "stage1_raw_audit.md"
STAGE2_AUDIT_MD: Path = OUTPUTS_DIR / "stage2_clean_audit.md"
STAGE3_SMOKE_JSON: Path = OUTPUTS_DIR / "stage3_smoke_5calls_v2.json"
STAGE3_SMOKE_MD: Path = OUTPUTS_DIR / "stage3_smoke_5calls_v2.md"
STAGE3_PILOT_AUDIT_MD: Path = OUTPUTS_DIR / "stage3_pilot_audit.md"
STAGE3_AUDIT_MD: Path = OUTPUTS_DIR / "stage3_llm_audit.md"
STAGE3B_SMOKE_MD: Path = OUTPUTS_DIR / "stage3b_embed_5co_report.md"
STAGE3B_AUDIT_MD: Path = OUTPUTS_DIR / "stage3b_embed_audit.md"
STAGE4_AUDIT_MD: Path = OUTPUTS_DIR / "stage4_feature_audit.md"

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


@dataclass(frozen=True)
class EmbedSettings:
    """Stage 3B 區段級金融語義 Embedding 與跨季偏移設定。"""

    provider: str = "openai"
    model: str = "text-embedding-3-small"
    dimensions: int = 1536
    batch_endpoint: str = "/v1/embeddings"
    batch_completion_window: str = "24h"
    max_chunk_words: int = 3500
    max_chunk_chars: int = 20000
    batch_shard_calls: int = 200
    max_batch_file_mb: float = 150.0
    smoke_symbols: Tuple[str, ...] = ("AAPL", "JPM", "LLY", "XOM", "MCD")
    # OpenAI text-embedding-3-small 官方參考定價（USD / 1M tokens）
    sync_cost_per_1m: float = 0.02
    batch_cost_per_1m: float = 0.01


# ==================== 量化建模與回測參數 ====================
@dataclass(frozen=True)
class QuantSettings:
    """量化特徵、Walk-Forward 切分與交易成本設定。"""

    random_seed: int = 42
    pre_car_windows: Tuple[int, ...] = (20, 60, 120)
    pead_horizons: Tuple[int, ...] = (10, 20)
    purge_gap_trading_days: int = 20
    eps_match_max_days: int = 5
    trading_days_per_year: int = 252
    one_way_cost_bps: float = 10.0
    train_start_year: int = 2016
    initial_train_end_year: int = 2018
    walk_forward_end_year: int = 2024
    post_cutoff_start_year: int = 2025


# ==================== 特徵分組單一事實來源（供 Stage 4 與 Stage 5 消融實驗共用） ====================
KEY_META_COLUMNS: Tuple[str, ...] = (
    "event_id",
    "symbol",
    "gics_sector",
    "year",
    "quarter",
    "session_type",
    "alignment_flag",
    "event_trading_date",
    "pre_end_date",
    "post_10d_date",
    "post_20d_date",
)

FEATURE_GROUPS: Dict[str, Tuple[str, ...]] = {
    # 嚴格截斷至 Close(t-1) 之事前量價與市場狀態特徵（適用於 Open 與 Close 兩種進場點）
    "market_pre": (
        "pre_ret_5d",
        "pre_ret_20d",
        "pre_ret_60d",
        "pre_ret_120d",
        "pre_car_20d",
        "pre_car_60d",
        "pre_car_120d",
        "pre_rv_20d",
        "pre_rv_60d",
        "pre_rv_ratio_20_60",
        "pre_amihud_20d",
        "pre_vol_ratio_5_20",
        "pre_drawdown_60d",
        "pre_beta_60d",
        "pre_spx_ret_20d",
        "pre_spx_ret_60d",
        "pre_spx_rv_20d",
        "pre_vix_close",
        "pre_vix_chg_5d",
        "pre_vix_chg_20d",
        "pre_vix_pct_rank_252d",
    ),
    # 事件日硬財報、文本結構與事件日可觀測價量反應（Close(t=0) 進場基準）
    "hard_event": (
        "sue_pct",
        "sue_diff",
        "sue_beat_flag",
        "sue_available",
        "is_bmo",
        "prepared_word_count",
        "qa_word_count",
        "total_word_count",
        "qa_word_ratio",
        "log_total_words",
        "qoq_word_count_pct_chg",
    ),
    # 僅在 Close(t=0) 進場點可觀測之事件日反應特徵（Open(t=0) 敏感度測試時自動排除）
    "event_day_close_only": (
        "event_day_ret",
        "event_day_excess_ret",
        "overnight_gap",
        "intraday_reversal",
        "event_day_vol_ratio",
    ),
    # Stage 3A LLM 結構化情緒與跨季差分特徵
    "llm_sentiment": (
        "prepared_optimism",
        "qa_optimism",
        "optimism_score",
        "qa_tone_drop",
        "ambiguity_score",
        "qa_evasiveness",
        "guidance_revision",
        "qoq_prep_opt_diff",
        "qoq_qa_opt_diff",
        "qoq_opt_diff",
        "qoq_tone_drop_diff",
        "qoq_amb_diff",
        "qoq_evas_diff",
        "qoq_guid_diff",
        "opt_vs_trailing_4q_mean",
    ),
    # Stage 3B 區段級 Embedding 跨季語義偏移與背離特徵
    "embed_drift": (
        "prep_qa_divergence",
        "qoq_prep_drift",
        "qoq_qa_drift",
        "qoq_full_drift",
        "yoy_prep_drift",
        "yoy_qa_drift",
        "prep_drift_vs_4q_centroid",
        "qa_drift_vs_4q_centroid",
    ),
    # 多模態領域交互項（情緒／語義 × 市場波動／硬財報）
    "interactions": (
        "opt_x_vix",
        "amb_x_rv20",
        "drift_x_rv20",
        "opt_x_sue_sign",
        "tone_drop_x_evas",
    ),
}

LABEL_COLUMNS: Tuple[str, ...] = (
    "label_ret_10d",
    "label_spx_ret_10d",
    "label_excess_ret_10d",
    "label_ret_20d",
    "label_spx_ret_20d",
    "label_excess_ret_20d",
    "label_open_excess_ret_10d",
    "label_open_excess_ret_20d",
    "label_rv_10d",
    "label_rv_20d",
    "label_log_rv_10d",
    "label_log_rv_20d",
    "label_rv_ratio_20d",
)


DATA_CFG = DataSettings()
MARKET_ETL_CFG = MarketETLSettings()
CLEAN_CFG = CleanSettings()
LLM_CFG = LLMSettings()
EMBED_CFG = EmbedSettings()
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
        EMBED_CACHE_DIR,
        EMBED_BATCH_WORKDIR,
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
