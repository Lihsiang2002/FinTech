# Stage 3B 5 家公司 OpenAI Batch Embedding 效果評估報告（`text-embedding-3-small`）

- **產出時間 (UTC)**：`2026-11-26T11:16:18.163358+00:00`
- **模型與維度**：`text-embedding-3-small`（`1536` 維，長文本零截斷切塊 + Token 加權池化 + L2 正規化）
- **覆蓋範圍**：`182 / 182` 場有效法說會（`5` 家公司、`2016–2025` 年）
- **切塊與 Token 統計**：共 `364` 個區段、`725` 個切塊（平均每場 `3.98` 塊）、總計 `2,297,272` tokens（場均 `12,622.4` tokens）
- **費用實測與全量推估（Batch 五折價 `USD 0.01 / 1M tokens`）**：
  - 本次實付費用：**`USD 0.0230`**
  - 推估全量 `4,412` 場 Batch 總費用：**`USD 0.557`**

## 一、語義偏移特徵分佈總覽（餘弦距離：數值越大代表語義跳變越劇烈）

| 特徵欄位 | Mean | Std | Min | P10 | Median | P90 | Max | 缺失率 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `prep_qa_divergence` | 0.1950 | 0.0372 | 0.0996 | 0.1509 | 0.1907 | 0.2447 | 0.3125 | 0.00% |
| `qoq_prep_drift` | 0.0887 | 0.0437 | 0.0221 | 0.0384 | 0.0846 | 0.1524 | 0.2477 | 2.75% |
| `qoq_qa_drift` | 0.0899 | 0.0361 | 0.0290 | 0.0559 | 0.0837 | 0.1259 | 0.2778 | 2.75% |
| `qoq_full_drift` | 0.0556 | 0.0215 | 0.0219 | 0.0324 | 0.0505 | 0.0809 | 0.1393 | 2.75% |
| `yoy_prep_drift` | 0.1057 | 0.0446 | 0.0273 | 0.0520 | 0.1033 | 0.1628 | 0.2379 | 10.99% |
| `yoy_qa_drift` | 0.0965 | 0.0354 | 0.0494 | 0.0622 | 0.0893 | 0.1354 | 0.2498 | 10.99% |
| `prep_drift_vs_4q_centroid` | 0.0669 | 0.0326 | 0.0229 | 0.0318 | 0.0599 | 0.1102 | 0.1984 | 5.49% |
| `qa_drift_vs_4q_centroid` | 0.0611 | 0.0243 | 0.0272 | 0.0400 | 0.0562 | 0.0867 | 0.2423 | 5.49% |

## 二、歷年語義偏移演變（檢驗危機與總經轉折偵測力）

| Year | 場次 | Prep-Q&A 背離 | QoQ Prep Drift | QoQ Q&A Drift | YoY Prep Drift | vs 4Q 質心偏移 (Prep) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 2016 | 20 | 0.1880 | 0.0723 | 0.0723 | N/A | 0.0663 |
| 2017 | 19 | 0.1803 | 0.1009 | 0.0750 | 0.0859 | 0.0650 |
| 2018 | 19 | 0.2001 | 0.0882 | 0.1032 | 0.1018 | 0.0616 |
| 2019 | 20 | 0.1932 | 0.0664 | 0.1039 | 0.0900 | 0.0502 |
| 2020 | 19 | 0.1949 | 0.1028 | 0.0927 | 0.1169 | 0.0809 |
| 2021 | 20 | 0.1877 | 0.0729 | 0.0850 | 0.1128 | 0.0619 |
| 2022 | 20 | 0.1968 | 0.1016 | 0.0870 | 0.1080 | 0.0782 |
| 2023 | 20 | 0.1995 | 0.0794 | 0.0802 | 0.1030 | 0.0604 |
| 2024 | 19 | 0.2172 | 0.1128 | 0.1058 | 0.1164 | 0.0829 |
| 2025 | 6 | 0.1879 | 0.0852 | 0.0918 | 0.1423 | 0.0529 |

## 三、Top 10 最大跨季語義跳變場次（`qoq_prep_drift` 異常事件偵測檢驗）

| 排名 | Symbol | 財季 | 交易日 | QoQ Prep Drift | QoQ Q&A Drift | Prep-Q&A 背離 | Stage 3A Opt / Amb | Stage 3A 判讀摘要 |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 1 | `JPM` | 2024 Q2 | 2024-07-12 | **0.2477** | 0.2778 | 0.2366 | 61.0 / 45 | Management displayed confidence in prepared remarks but became more cautious and defensive |
| 2 | `MCD` | 2020 Q1 | 2020-04-30 | **0.2403** | 0.1119 | 0.1873 | 48.0 / 62 | Management's prepared remarks reflect cautious optimism amid significant challenges, while |
| 3 | `JPM` | 2024 Q1 | 2024-04-12 | **0.2195** | 0.2364 | 0.2522 | 59.5 / 42 | Management displayed confidence in prepared remarks but became more cautious and evasive d |
| 4 | `JPM` | 2022 Q1 | 2022-04-14 | **0.2039** | 0.0892 | 0.2151 | 58.0 / 45 | Management displayed a generally positive outlook in prepared remarks but showed some caut |
| 5 | `MCD` | 2017 Q1 | 2017-04-27 | **0.1955** | 0.1335 | 0.0996 | 68.5 / 42 | Management's prepared remarks conveyed strong confidence and positive momentum, while the  |
| 6 | `MCD` | 2018 Q3 | 2018-10-24 | **0.1813** | 0.1119 | 0.1656 | 65.0 / 45 | Management expressed strong confidence in growth strategies during prepared remarks but sh |
| 7 | `XOM` | 2016 Q4 | 2017-02-01 | **0.1812** | 0.0863 | 0.2335 | 58.0 / 45 | Management's prepared remarks conveyed a solid performance and future plans, but the Q&A r |
| 8 | `LLY` | 2017 Q3 | 2017-10-24 | **0.1704** | 0.0377 | 0.1443 | 61.0 / 45 | Management displayed confidence in growth and strategic direction during prepared remarks, |
| 9 | `LLY` | 2020 Q1 | 2020-04-23 | **0.1693** | 0.1011 | 0.1942 | 51.0 / 62 | Management's prepared remarks reflected cautious optimism amid COVID-19 challenges, while  |
| 10 | `JPM` | 2017 Q4 | 2018-01-12 | **0.1676** | 0.1023 | 0.1845 | 65.0 / 45 | Management displayed confidence in prepared remarks, but Q&A revealed some caution and unc |

## 四、去識別化後之公司間語義結構檢驗（對角線＝同公司跨季平均相似度，非對角＝跨公司質心相似度）

| Symbol | `AAPL` | `JPM` | `LLY` | `MCD` | `XOM` |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `AAPL` | **0.929** | 0.853 | 0.824 | 0.867 | 0.842 |
| `JPM` | 0.853 | **0.935** | 0.797 | 0.838 | 0.850 |
| `LLY` | 0.824 | 0.797 | **0.929** | 0.790 | 0.828 |
| `MCD` | 0.867 | 0.838 | 0.790 | **0.916** | 0.839 |
| `XOM` | 0.842 | 0.850 | 0.828 | 0.839 | **0.910** |

## 五、Stage 3B 語義偏移 vs. Stage 3A LLM 情緒分數之正交性檢驗（Spearman ρ）

| Stage 3B 偏移特徵 | vs `optimism_score` | vs `qa_tone_drop` | vs `ambiguity_score` | vs `qa_evasiveness` | vs `guidance_revision` |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `prep_qa_divergence` | +0.191 | +0.046 | -0.146 | -0.127 | -0.055 |
| `qoq_prep_drift` | -0.149 | +0.091 | +0.133 | +0.151 | -0.033 |
| `qoq_qa_drift` | +0.225 | +0.062 | -0.162 | -0.129 | +0.074 |
| `yoy_prep_drift` | -0.202 | +0.038 | +0.130 | +0.104 | +0.026 |
| `prep_drift_vs_4q_centroid` | -0.154 | +0.077 | +0.144 | +0.099 | +0.007 |