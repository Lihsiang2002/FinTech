"""Stage 2（雙區塊切分、三層去識別化、Point-in-Time 對齊與軟刪除）單元測試。"""

from __future__ import annotations

import importlib

import numpy as np
import pandas as pd

from src.config import load_universe

etl_m = importlib.import_module("src.02_etl_market_sue")
clean = importlib.import_module("src.03_clean_and_align")


def _sample_calendar() -> np.ndarray:
    cal_df = etl_m.build_nyse_calendar("2024-01-01", "2025-03-01")
    arr = cal_df["date"].to_numpy(dtype="datetime64[D]")
    arr.sort()
    return arr


def test_align_bmo_vs_amc_vs_holidays_and_predawn() -> None:
    arr = _sample_calendar()

    # 1. BMO 09:29（開盤前 1 分鐘）-> 當日 t=0
    bmo = clean.align_event_to_nyse_calendar("2024-02-01 09:29:00", arr)
    assert bmo.alignment_flag == "EXACT_BMO"
    assert bmo.event_trading_date == "2024-02-01"
    assert bmo.pre_end_date == "2024-01-31"

    # 2. 09:30 整（已開盤）-> 嚴格順延至隔日 t=0
    at_open = clean.align_event_to_nyse_calendar("2024-02-01 09:30:00", arr)
    assert at_open.alignment_flag == "EXACT_AMC_NEXT_DAY"
    assert at_open.event_trading_date == "2024-02-02"
    assert at_open.pre_end_date == "2024-02-01"

    # 3. 週五盤後 17:00 -> 順延至下週一
    fri_amc = clean.align_event_to_nyse_calendar("2024-02-02 17:00:00", arr)
    assert fri_amc.alignment_flag == "EXACT_AMC_NEXT_DAY"
    assert fri_amc.event_trading_date == "2024-02-05"
    assert fri_amc.pre_end_date == "2024-02-02"

    # 4. 感恩節休市日（2024-11-28）即使早上 08:30 -> 順延至隔日開盤日 2024-11-29
    hol = clean.align_event_to_nyse_calendar("2024-11-28 08:30:00", arr)
    assert hol.alignment_flag == "NON_TRADING_DAY_NEXT"
    assert hol.event_trading_date == "2024-11-29"

    # 5. 缺時分秒 00:00:00 與凌晨異常 03:15:00 -> 保守順延至隔日
    mid = clean.align_event_to_nyse_calendar("2024-02-01 00:00:00", arr)
    assert mid.alignment_flag == "CONSERVATIVE_ABNORMAL_TIME"
    assert mid.event_trading_date == "2024-02-02"

    predawn = clean.align_event_to_nyse_calendar("2024-02-01 03:15:00", arr)
    assert predawn.alignment_flag == "CONSERVATIVE_ABNORMAL_TIME"
    assert predawn.event_trading_date == "2024-02-02"

    # 6. 無效日期
    bad = clean.align_event_to_nyse_calendar("not-a-date", arr)
    assert bad.alignment_flag == "INVALID_DATE"
    assert bad.event_trading_date is None


def test_split_prepared_and_qa_strips_headers_and_ignores_opening_greeting() -> None:
    long_remarks = " ".join(["revenue growth margin expansion strong demand"] * 45)  # 270 words
    segs = [
        {"speaker": "Executives", "text": "Tim Cook - CEO, Luca Maestri - CFO"},
        {"speaker": "Analysts", "text": "Erik Woodring - Morgan Stanley"},
        {
            "speaker": "Operator",
            "text": "Good afternoon. After the speakers' remarks there will be a question-and-answer session.",
        },
        {"speaker": "Tim Cook - CEO", "text": long_remarks},
        {"speaker": "Operator", "text": "We will now begin the Question-and-Answer session."},
        {"speaker": "Erik Woodring", "text": "Thanks Tim, can you discuss services margin?"},
        {"speaker": "Tim Cook - CEO", "text": "Sure, services had a great quarter."},
    ]

    prep, qa, method, hdr_e, hdr_a = clean.split_prepared_and_qa(segs)
    assert method == "tier1_handoff_marker"
    assert len(hdr_e) == 1 and len(hdr_a) == 1
    # 目錄頁首已被剝離，Prepared 僅剩開場 Operator + Tim Cook
    assert len(prep) == 2
    assert len(qa) == 3
    assert qa[0]["speaker"] == "Operator"


def test_split_prepared_and_qa_tier2_operator_fallback() -> None:
    long_remarks = " ".join(["operating leverage free cash flow guidance"] * 70)  # 420 words
    segs = [
        {"speaker": "Operator", "text": "Welcome everyone to the call."},
        {"speaker": "CEO", "text": long_remarks},
        {"speaker": "Operator", "text": "Next we have Jane Smith on the line."},
        {"speaker": "Jane Smith", "text": "Hi, quick question on capex."},
        {"speaker": "CEO", "text": "Capex remains disciplined."},
    ]
    prep, qa, method, _, _ = clean.split_prepared_and_qa(segs)
    assert method == "tier2_operator_fallback"
    assert len(prep) == 2
    assert len(qa) == 3


def test_deidentify_text_masks_entities_and_preserves_financial_numbers() -> None:
    rdict = clean.load_redact_dict()
    comp_terms, prod_terms = clean.derive_company_core_terms("AAPL", "AAPL", "Apple Inc.", rdict)

    sample = (
        "Thank you, Suhasini. Good afternoon. In Q1 Fiscal Year 2024, Apple and AAPL "
        "saw record iPhone and Apple Watch revenue of $2024 million across 2,024 stores "
        "in the March quarter. Tim Cook and Mr. Maestri thanked Erik Woodring."
    )
    anon = clean.deidentify_text(
        sample,
        company_terms=comp_terms,
        product_terms=prod_terms,
        exec_names={"Suhasini Chandramouli", "Tim Cook", "Luca Maestri"},
        analyst_names={"Erik Woodring"},
    )

    assert "[EXECUTIVE]" in anon
    assert "[ANALYST]" in anon
    assert "[COMPANY]" in anon
    assert "[PRODUCT]" in anon
    assert "[CURRENT_QUARTER]" in anon
    assert "Apple" not in anon
    assert "iPhone" not in anon
    assert "Tim Cook" not in anon
    assert "Erik Woodring" not in anon
    assert "Suhasini" not in anon
    # 嚴格保護財務金額與千分位數字不被誤遮！
    assert "$2024 million" in anon
    assert "2,024 stores" in anon


def test_single_letter_ticker_does_not_corrupt_english_words() -> None:
    rdict = clean.load_redact_dict()
    comp_terms, prod_terms = clean.derive_company_core_terms("F", "F", "Ford Motor Company", rdict)
    sample = "Ford delivered strong F-150 demand for our fleet customers."
    anon = clean.deidentify_text(sample, comp_terms, prod_terms, set(), set())
    assert "[COMPANY]" in anon
    assert "[PRODUCT]" in anon
    assert "for our fleet" in anon  # 單字母 'F' 不可破壞英文單字


def test_redact_dict_covers_all_120_universe_symbols() -> None:
    uni = load_universe()
    rdict = clean.load_redact_dict()
    missing = [s for s in uni["symbols"] if s not in rdict["by_symbol"]]
    assert not missing, f"redact_dict.json 缺少股票池標的：{missing}"


def test_evaluate_record_validity_soft_delete_reasons() -> None:
    valid_aligned = clean.AlignedEventDates(
        "2024-02-01 17:00:00-0500", "AMC_OR_INTRADAY", "EXACT_AMC_NEXT_DAY",
        "2024-02-02", "2024-02-01", "2024-02-16", "2024-03-04"
    )
    ok, reason = clean.evaluate_record_validity(valid_aligned, prepared_wc=3000, qa_wc=5000, total_wc=8000)
    assert ok is True and reason == ""

    bad_ok, bad_reason = clean.evaluate_record_validity(valid_aligned, prepared_wc=800, qa_wc=50, total_wc=850)
    assert bad_ok is False
    assert "short_transcript" in bad_reason
    assert "missing_or_short_qa" in bad_reason
