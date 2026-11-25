# Stage 2 清洗、去識別化與 Point-in-Time 對齊稽核報告（02_clean）

- **產出時間**：2026-11-25T22:30:44
- **總筆數（保留全量軟刪除軌跡）**：**4,478** 筆
- **有效樣本數（`is_valid = True`）**：**4,412** 筆（**98.53%**）
- **02_clean 總磁碟占用**：**148.77 MB**
- **核心公司別名殘留筆數（稽核指標）**：**0** 筆

## 1. Q&A 雙區塊切分方式分布
| 切分方式 | 筆數 | 占比 |
|---|---:|---:|
| `tier1_exec_tail_handoff` | 2,254 | 50.33% |
| `tier1_handoff_marker` | 2,099 | 46.87% |
| `tier2_operator_fallback` | 81 | 1.81% |
| `no_qa_detected` | 44 | 0.98% |

## 2. Point-in-Time 交易日對齊旗標分布
| Alignment Flag | 筆數 | 占比 |
|---|---:|---:|
| `EXACT_AMC_NEXT_DAY` | 2,871 | 64.11% |
| `EXACT_BMO` | 1,543 | 34.46% |
| `CONSERVATIVE_ABNORMAL_TIME` | 44 | 0.98% |
| `NON_TRADING_DAY_NEXT` | 20 | 0.45% |

## 3. 軟刪除排除原因統計（`is_valid = False`）
| 排除原因 | 筆數 |
|---|---:|
| `missing_or_short_qa_lt_150` | 64 |
| `short_transcript_lt_1500;missing_or_short_qa_lt_150` | 2 |

## 4. 10 筆跨板塊人工抽檢明細（Stage 2 Verification Gate）

| # | Event ID | Sector | Raw Date (ET) | Flag | $t-1$ | $t=0$ | $t+20$ | Prep/Q&A 字數 |
|---:|---|---|---|---|---|---|---|---|
| 1 | `UPS_2022_Q2` | Industrials | `2022-07-26 08:30:00` | `EXACT_BMO` | `2022-07-25` | **`2022-07-26`** | `2022-08-23` | 3,703 / 5,561 |
| 2 | `AMD_2023_Q2` | Information Technology | `2023-08-01 17:00:00` | `EXACT_AMC_NEXT_DAY` | `2023-08-01` | **`2023-08-02`** | `2023-08-30` | 2,840 / 6,312 |
| 3 | `OMC_2022_Q2` | Communication Services | `2022-07-19 16:30:00` | `EXACT_AMC_NEXT_DAY` | `2022-07-19` | **`2022-07-20`** | `2022-08-17` | 2,832 / 3,930 |
| 4 | `DUK_2023_Q2` | Utilities | `2023-08-08 10:00:00` | `EXACT_AMC_NEXT_DAY` | `2023-08-08` | **`2023-08-09`** | `2023-09-07` | 2,469 / 2,525 |
| 5 | `SLB_2020_Q4` | Energy | `2021-01-22 08:30:00` | `EXACT_BMO` | `2021-01-21` | **`2021-01-22`** | `2021-02-22` | 2,392 / 6,488 |
| 6 | `AMGN_2017_Q3` | Health Care | `2017-10-25 17:00:00` | `EXACT_AMC_NEXT_DAY` | `2017-10-25` | **`2017-10-26`** | `2017-11-24` | 5,138 / 6,090 |
| 7 | `PM_2023_Q3` | Consumer Staples | `2023-10-19 09:00:00` | `EXACT_BMO` | `2023-10-18` | **`2023-10-19`** | `2023-11-16` | 3,989 / 3,758 |
| 8 | `BLK_2020_Q2` | Financials | `2020-07-17 08:30:00` | `EXACT_BMO` | `2020-07-16` | **`2020-07-17`** | `2020-08-14` | 4,585 / 3,698 |
| 9 | `PLD_2020_Q4` | Real Estate | `2021-01-26 12:00:00` | `EXACT_AMC_NEXT_DAY` | `2021-01-26` | **`2021-01-27`** | `2021-02-25` | 1,864 / 6,198 |
| 10 | `HD_2019_Q4` | Consumer Discretionary | `2020-02-25 09:00:00` | `EXACT_BMO` | `2020-02-24` | **`2020-02-25`** | `2020-03-24` | 3,433 / 5,365 |

### 抽檢樣本去識別化與 Q&A 切分起點對照
#### 1. `UPS_2022_Q2`（Industrials｜tier1_handoff_marker）
- **Prepared 原文片段**：`Operator: Good morning. My name is Steven, and I will be your facilitator today. I would like to welcome everyone to the UPS Investor Relations Second Quarter 2022 Earnings Conference Call. [Operator Instructions]. It is...`
- **Prepared 去識別化**：`[OPERATOR]: Good morning. My name is Steven, and I will be your facilitator today. I would like to welcome everyone to the [COMPANY] Investor Relations [CURRENT_QUARTER] [YEAR] Earnings Conference Call. [Operator Instruc...`
- **Q&A 切分起點**：`Operator: [Operator Instructions]. Our first question will come from the line of Chris Wetherbee of Citi.  Christian Wetherbee: Maybe starting, Brian, where you left off on sort of...`

#### 2. `AMD_2023_Q2`（Information Technology｜tier1_exec_tail_handoff）
- **Prepared 原文片段**：`Operator: Greetings, and welcome to the AMD Second Quarter 2023 Earnings Conference Call. At this time, all participants are in a listen-only mode. A brief question-and-answer session will follow the formal presentation....`
- **Prepared 去識別化**：`[OPERATOR]: Greetings, and welcome to the [COMPANY] [CURRENT_QUARTER] [YEAR] Earnings Conference Call. At this time, all participants are in a listen-only mode. A brief question-and-answer session will follow the formal ...`
- **Q&A 切分起點**：`Mitch Haws: Thank you, Jean. John, we're happy to poll the audience for questions.  Operator: Thank you. We will now be conducting a question-and-answer session. [Operator Instruct...`

#### 3. `OMC_2022_Q2`（Communication Services｜tier1_handoff_marker）
- **Prepared 原文片段**：`Operator: Welcome to the Omnicom Second Quarter 2022 Earnings Release Conference Call. At this time, all participants are in a listen-only mode. Later, we will conduct a question-and-answer session. [Operator Instruction...`
- **Prepared 去識別化**：`[OPERATOR]: Welcome to the [COMPANY] [CURRENT_QUARTER] [YEAR] Earnings Release Conference Call. At this time, all participants are in a listen-only mode. Later, we will conduct a question-and-answer session. [Operator In...`
- **Q&A 切分起點**：`Operator: [Operator Instructions] We'll go to the line of Steven Cahall with Wells Fargo. Please go ahead.  Steven Cahall: So maybe first question, you could just talk a little bit...`

#### 4. `DUK_2023_Q2`（Utilities｜tier1_handoff_marker）
- **Prepared 原文片段**：`Operator: Ladies and gentlemen, welcome to the Duke Energy Second Quarter 2023 Earnings Call. My name is Brian, I will be the operator for today’s call. [Operator Instructions] I will now hand over to your host, Abby Mot...`
- **Prepared 去識別化**：`[OPERATOR]: Ladies and gentlemen, welcome to the [COMPANY] [CURRENT_QUARTER] [YEAR] Earnings Call. My name is Brian, I will be the operator for today’s call. [Operator Instructions] I will now hand over to your host, [EX...`
- **Q&A 切分起點**：`Operator: [Operator Instructions] Our first question comes from Mrs. Shar Pourreza with Guggenheim Partners. The line is now open.  Shar Pourreza: Hey, guys, good morning. Obviousl...`

#### 5. `SLB_2020_Q4`（Energy｜tier1_handoff_marker）
- **Prepared 原文片段**：`Operator: Ladies and gentlemen, thank you for standing by. Welcome to the Schlumberger Earnings Conference Call. At this time, all participants are in a listen-only mode. Later, there will be an opportunity for your ques...`
- **Prepared 去識別化**：`[OPERATOR]: Ladies and gentlemen, thank you for standing by. Welcome to the [COMPANY] Earnings Conference Call. At this time, all participants are in a listen-only mode. Later, there will be an opportunity for your quest...`
- **Q&A 切分起點**：`Operator: Thank you. [Operator Instructions] And our first question is from the line of James West with Evercore ISI. Please go ahead.  James West: Hey. Good morning, guys. So, Oli...`

#### 6. `AMGN_2017_Q3`（Health Care｜tier1_handoff_marker）
- **Prepared 原文片段**：`Operator: My name is Ian, and I will be your conference facilitator today for Amgen's Third Quarter 2017 Financial Results Conference Call. I would now like to introduce Arvind Sood, Vice President of Investor Relations....`
- **Prepared 去識別化**：`[OPERATOR]: My name is Ian, and I will be your conference facilitator today for [COMPANY]'s [CURRENT_QUARTER] [YEAR] Financial Results Conference Call. I would now like to introduce Arvind Sood, Vice President of Investo...`
- **Q&A 切分起點**：`Operator: Our first question is from Chris Raymond from Piper Jaffray.  Christopher Raymond - Piper Jaffray: Hey, thanks. Thanks for taking the question. So just a question on the ...`

#### 7. `PM_2023_Q3`（Consumer Staples｜tier1_handoff_marker）
- **Prepared 原文片段**：`Operator: Good day and welcome to the Philip Morris International Third Quarter 2023 Earnings Conference Call. Today's call is scheduled to last about 1 hour, including remarks by Philip Morris International Management a...`
- **Prepared 去識別化**：`[OPERATOR]: Good day and welcome to the [COMPANY] International [CURRENT_QUARTER] [YEAR] Earnings Conference Call. Today's call is scheduled to last about 1 hour, including remarks by [COMPANY] International Management a...`
- **Q&A 切分起點**：`Operator: Thank you. We will now conduct the question-and-answer portion of the conference. [Operator Instructions] Our first question will come from Vivien Azer with TD Cowen. Ple...`

#### 8. `BLK_2020_Q2`（Financials｜tier1_exec_tail_handoff）
- **Prepared 原文片段**：`Operator: Good morning. My name is Maria, and I will be your conference facilitator today. At this time, I would like to welcome everyone to the BlackRock Incorporated Second Quarter 2020 Earnings Teleconference. Our hos...`
- **Prepared 去識別化**：`[OPERATOR]: Good morning. My name is Maria, and I will be your conference facilitator today. At this time, I would like to welcome everyone to the [COMPANY] Incorporated [CURRENT_QUARTER] [YEAR] Earnings Teleconference. ...`
- **Q&A 切分起點**：`Operator: Thank you. [Operator Instructions] Our first question comes from the line of Dan Fannon of Jefferies.  Dan Fannon: Thanks.  Larry Fink: Good morning Dan.  Dan Fannon: Goo...`

#### 9. `PLD_2020_Q4`（Real Estate｜tier1_handoff_marker）
- **Prepared 原文片段**：`Operator: Welcome to the Prologis Q4 Earnings Conference Call. My name is Amy, and I will be your operator for today's call. [Operator Instructions] Please note that this conference is being recorded. I'd now like to tur...`
- **Prepared 去識別化**：`[OPERATOR]: Welcome to the [COMPANY] [CURRENT_QUARTER] Earnings Conference Call. My name is Amy, and I will be your operator for today's call. [Operator Instructions] Please note that this conference is being recorded. I...`
- **Q&A 切分起點**：`Operator: [Operator Instructions] Your first question today comes from the line of Caitlin Burrows with Goldman Sachs. Please proceed with your question.  Caitlin Burrows: Your gui...`

#### 10. `HD_2019_Q4`（Consumer Discretionary｜tier1_exec_tail_handoff）
- **Prepared 原文片段**：`Operator: Greetings. And welcome to The Home Depot Fourth Quarter 2019 Earnings Call. At this time, all participants are in a listen-only mode. A brief question-and-answer session will follow the formal presentation. [Op...`
- **Prepared 去識別化**：`[OPERATOR]: Greetings. And welcome to [COMPANY] [CURRENT_QUARTER] [YEAR] Earnings Call. At this time, all participants are in a listen-only mode. A brief question-and-answer session will follow the formal presentation. [...`
- **Q&A 切分起點**：`Operator: Thank you. [Operator Instructions] Thank you. Our first question comes from line of Michael Lasser with UBS. Please proceed with your question.  Michael Lasser: Good morn...`

