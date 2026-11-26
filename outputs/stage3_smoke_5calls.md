# Stage 3A OpenAI API 連通性與 5 家公司法說會情緒評分測試報告

- **測試時間 (UTC)**：`2026-11-26T08:05:54.020960+00:00`
- **模型與 Prompt 版本**：`gpt-4o-mini` / `v1`（`temperature=0.0`, `seed=42`, Strict JSON Schema）
- **金鑰來源檔**：`config.env`（未輸出任何金鑰內容）
- **測試筆數**：`5` 場（來自 `5` 家不同公司與不同 GICS 產業，全部輸入去識別化 `*_anon` 文本）
- **Token 統計**：總輸入 `65,945` tokens（場均 `13,189.0`）、總輸出 `536` tokens（場均 `107.2`）
- **費用實測與推估**：
  - 本次 5 場即時呼叫實測費用：`USD 0.01021`
  - 推估 200 場 Pilot（Batch API 五折價）：`USD 0.2043`
  - 推估全量 4,412 場（Batch API 五折價）：`USD 4.51`

## 一、5 家公司情緒評分總覽

| Symbol | GICS 產業 | 財季 | 交易日 | 總字數 | Optimism (1~5) | Ambiguity (1~5) | Guidance (-2~+2) | Q&A Evasiveness (1~5) | Input Tokens | Output Tokens | 耗時(s) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `MCD` | Consumer Discretionary | 2024 Q4 | 2025-02-10 | 9,985 | **3** | **3** | **+0** | **2** | 13,084 | 108 | 5.09 |
| `LLY` | Health Care | 2024 Q4 | 2025-02-07 | 8,966 | **4** | **3** | **+1** | **2** | 12,657 | 79 | 5.87 |
| `HON` | Industrials | 2024 Q4 | 2025-02-06 | 10,505 | **3** | **3** | **+0** | **2** | 14,017 | 124 | 13.40 |
| `XOM` | Energy | 2024 Q4 | 2025-02-03 | 9,255 | **4** | **3** | **+1** | **2** | 12,074 | 126 | 2.73 |
| `SHW` | Materials | 2024 Q4 | 2025-01-31 | 10,867 | **3** | **3** | **+0** | **2** | 14,113 | 99 | 11.29 |

## 二、逐場證據引文與評分理由（人工判讀檢核）

### 1. `MCD`（McDonald's Corporation｜Consumer Discretionary｜2024 Q4｜`MCD_2024_Q4`）
- **四維分數**：Optimism=`3`, Ambiguity=`3`, Guidance=`+0`, Q&A Evasiveness=`2`
- **簡要判讀理由**：Management expressed cautious optimism about recovery and growth, but acknowledged ongoing challenges and uncertainties.
- **樂觀證據原句**：> I'm pleased with how it's getting out of the gate.
- **迴避／保守原句**：> I won't get into kind of specific comp numbers, but I think, you know, I think it's certainly reasonable to think of [CURRENT_QUARTER] as kind of a low point quarter for the year.

### 2. `LLY`（Eli Lilly and Company｜Health Care｜2024 Q4｜`LLY_2024_Q4`）
- **四維分數**：Optimism=`4`, Ambiguity=`3`, Guidance=`+1`, Q&A Evasiveness=`2`
- **簡要判讀理由**：Management displayed confidence in growth and execution, but acknowledged some uncertainties and challenges in the market.
- **樂觀證據原句**：> We believe both products are still very early in their launch cycle.
- **迴避／保守原句**：> We don't know the outcome of all that.

### 3. `HON`（Honeywell International Inc.｜Industrials｜2024 Q4｜`HON_2024_Q4`）
- **四維分數**：Optimism=`3`, Ambiguity=`3`, Guidance=`+0`, Q&A Evasiveness=`2`
- **簡要判讀理由**：Management displayed a balanced tone, acknowledging challenges while highlighting positive achievements and future growth potential.
- **樂觀證據原句**：> We finished [YEAR] on a positive note, exceeding or meeting the high end of our guidance for organic sales growth and adjusted earnings growth in the [CURRENT_QUARTER].
- **迴避／保守原句**：> We have to do more work. And that's why we have offered a second half [YEAR] because we won't have ample time to understand it and execute it.

### 4. `XOM`（Exxon Mobil Corporation｜Energy｜2024 Q4｜`XOM_2024_Q4`）
- **四維分數**：Optimism=`4`, Ambiguity=`3`, Guidance=`+1`, Q&A Evasiveness=`2`
- **簡要判讀理由**：Management displayed confidence in growth and operational performance while acknowledging uncertainties in future projections.
- **樂觀證據原句**：> We're confident we'll deliver $20 billion more in earnings and $30 billion more in cash flow, all of which enables us to keep our commitment to sustainable, competitive, and growing shareholder returns.
- **迴避／保守原句**：> I would tell you that that's a challenging, to project that far in the future. There's a lot of variables that go into the developments across all of our upstream portfolios.

### 5. `SHW`（The Sherwin-Williams Company｜Materials｜2024 Q4｜`SHW_2024_Q4`）
- **四維分數**：Optimism=`3`, Ambiguity=`3`, Guidance=`+0`, Q&A Evasiveness=`2`
- **簡要判讀理由**：Management expressed a balanced view with cautious optimism, acknowledging challenges while highlighting strategic strengths and opportunities.
- **樂觀證據原句**：> We are extremely well positioned to continue delivering shareholder value.
- **迴避／保守原句**：> If those were to occur and they're significant, we would need to go and we're prepared to go out with additional price in specific markets and segments as required.
