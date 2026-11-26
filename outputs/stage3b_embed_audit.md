# Stage 3B 全量區段級 Embedding 與跨季語義偏移稽核報告（`text-embedding-3-small`）

- **產出時間 (UTC)**：`2026-11-26T11:36:23.920853+00:00`
- **模型與維度**：`text-embedding-3-small`（`1536` 維，長文本零截斷切塊 + Token 加權池化 + L2 正規化）
- **覆蓋範圍**：`4,412 / 4,412` 場有效法說會（`120` 家公司、`2016–2025` 年）
- **切塊與 Token 統計**：共 `8,824` 個區段、`16,383` 個切塊（平均每場 `3.71` 塊）、總計 `51,357,252` tokens（場均 `11,640.4` tokens）
- **費用實測與全量推估（Batch 五折價 `USD 0.01 / 1M tokens`）**：
  - 本次實付費用：**`USD 0.5136`**
  - 推估全量 `4,412` 場 Batch 總費用：**`USD 0.514`**

## 一、語義偏移特徵分佈總覽（餘弦距離：數值越大代表語義跳變越劇烈）

| 特徵欄位 | Mean | Std | Min | P10 | Median | P90 | Max | 缺失率 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `prep_qa_divergence` | 0.1887 | 0.0420 | 0.0736 | 0.1427 | 0.1829 | 0.2401 | 0.4500 | 0.00% |
| `qoq_prep_drift` | 0.0829 | 0.0538 | 0.0048 | 0.0340 | 0.0671 | 0.1579 | 0.4166 | 2.72% |
| `qoq_qa_drift` | 0.0924 | 0.0375 | 0.0290 | 0.0567 | 0.0838 | 0.1384 | 0.4418 | 2.72% |
| `qoq_full_drift` | 0.0552 | 0.0258 | 0.0163 | 0.0309 | 0.0486 | 0.0861 | 0.2393 | 2.72% |
| `yoy_prep_drift` | 0.0981 | 0.0562 | 0.0040 | 0.0437 | 0.0826 | 0.1789 | 0.3846 | 10.86% |
| `yoy_qa_drift` | 0.1005 | 0.0382 | 0.0371 | 0.0626 | 0.0912 | 0.1490 | 0.4433 | 10.86% |
| `prep_drift_vs_4q_centroid` | 0.0616 | 0.0394 | 0.0047 | 0.0281 | 0.0507 | 0.1090 | 0.3268 | 5.44% |
| `qa_drift_vs_4q_centroid` | 0.0635 | 0.0260 | 0.0214 | 0.0397 | 0.0576 | 0.0941 | 0.3503 | 5.44% |

## 二、歷年語義偏移演變（檢驗危機與總經轉折偵測力）

| Year | 場次 | Prep-Q&A 背離 | QoQ Prep Drift | QoQ Q&A Drift | YoY Prep Drift | vs 4Q 質心偏移 (Prep) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 2016 | 473 | 0.1886 | 0.0745 | 0.0874 | N/A | 0.0648 |
| 2017 | 476 | 0.1903 | 0.0786 | 0.0881 | 0.0891 | 0.0566 |
| 2018 | 470 | 0.1889 | 0.0837 | 0.0906 | 0.0897 | 0.0592 |
| 2019 | 472 | 0.1890 | 0.0789 | 0.0920 | 0.0949 | 0.0582 |
| 2020 | 472 | 0.1878 | 0.0959 | 0.1003 | 0.1161 | 0.0757 |
| 2021 | 473 | 0.1858 | 0.0813 | 0.0897 | 0.1079 | 0.0632 |
| 2022 | 473 | 0.1870 | 0.0807 | 0.0927 | 0.0936 | 0.0583 |
| 2023 | 476 | 0.1873 | 0.0772 | 0.0910 | 0.0879 | 0.0551 |
| 2024 | 469 | 0.1941 | 0.0954 | 0.0985 | 0.0972 | 0.0674 |
| 2025 | 158 | 0.1862 | 0.0760 | 0.0918 | 0.1229 | 0.0554 |

## 三、Top 10 最大跨季語義跳變場次（`qoq_prep_drift` 異常事件偵測檢驗）

| 排名 | Symbol | 財季 | 交易日 | QoQ Prep Drift | QoQ Q&A Drift | Prep-Q&A 背離 | Stage 3A Opt / Amb | Stage 3A 判讀摘要 |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 1 | `ADI` | 2018 Q1 | 2018-02-28 | **0.4166** | 0.2283 | 0.3747 | 68.5 / 38 | Management displayed strong confidence in prepared remarks but was slightly more cautious  |
| 2 | `GIS` | 2021 Q4 | 2021-06-30 | **0.3547** | 0.1316 | 0.2302 | 51.0 / 62 | Management's prepared remarks conveyed a balanced outlook with some optimism, but the Q&A  |
| 3 | `NEE` | 2017 Q1 | 2017-04-21 | **0.3399** | 0.1540 | 0.2497 | 61.0 / 42 | Management expressed solid results and growth prospects in prepared remarks but showed mor |
| 4 | `PFE` | 2021 Q2 | 2021-08-06 | **0.3394** | 0.2290 | 0.1891 | 65.0 / 45 | Management displayed confidence in their pipeline and execution during prepared remarks, b |
| 5 | `CAT` | 2016 Q4 | 2017-01-27 | **0.3318** | 0.0826 | 0.2686 | 48.5 / 58 | Management's prepared remarks reflected cautious optimism amid challenges, while Q&A showe |
| 6 | `AMZN` | 2020 Q1 | 2020-05-01 | **0.3306** | 0.0996 | 0.1846 | 58.0 / 65 | Management expressed a mix of confidence in demand and operational challenges, with a nota |
| 7 | `ABT` | 2018 Q4 | 2019-01-23 | **0.3303** | 0.1138 | 0.3119 | 68.5 / 38 | Management displayed strong confidence in growth and execution during prepared remarks, bu |
| 8 | `DIS` | 2024 Q3 | 2024-08-07 | **0.3296** | 0.1379 | 0.1768 | 58.0 / 45 | Management displayed confidence in prepared remarks but became more cautious and less spec |
| 9 | `AEP` | 2024 Q1 | 2024-04-30 | **0.3248** | 0.1384 | 0.2171 | 58.0 / 45 | Management displayed confidence in growth initiatives during prepared remarks but showed s |
| 10 | `GIS` | 2023 Q2 | 2022-12-20 | **0.3244** | 0.0926 | 0.2095 | 65.0 / 45 | Management displayed confidence in growth and guidance during prepared remarks but showed  |

## 五、Stage 3B 語義偏移 vs. Stage 3A LLM 情緒分數之正交性檢驗（Spearman ρ）

| Stage 3B 偏移特徵 | vs `optimism_score` | vs `qa_tone_drop` | vs `ambiguity_score` | vs `qa_evasiveness` | vs `guidance_revision` |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `prep_qa_divergence` | -0.038 | -0.014 | +0.011 | +0.061 | -0.022 |
| `qoq_prep_drift` | -0.110 | +0.027 | +0.103 | +0.086 | -0.074 |
| `qoq_qa_drift` | -0.022 | -0.011 | +0.008 | +0.030 | -0.040 |
| `yoy_prep_drift` | -0.129 | +0.034 | +0.121 | +0.111 | -0.078 |
| `prep_drift_vs_4q_centroid` | -0.141 | +0.038 | +0.131 | +0.113 | -0.081 |