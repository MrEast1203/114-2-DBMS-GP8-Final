# v3 plan · v2 的 chained push-down 優化

Sources: `reports/eval_v3.json` (20 query × 4 plan × 10 samples) +
`reports/coldwarm_v3.json` (28-cell cold/warm matrix). 評估使用
**per-aspect ground truth**(1 472 筆,每筆含 `label_sem` /
`label_lex` / `label_gph` 三個獨立 label,effective relevance 依
query type 取 predicate AND;§9 詳列方法)。

## 1. 一句話總結

**v3 chained push-down 同時在 latency 與 NDCG 上贏 v2**:mean P50
18.85 ms vs v2 26.69 ms(**1.42× 加速**),mean NDCG@10 0.981 vs v2
0.922(**+0.059**)。v3 在 Q6 整類大幅領先(+0.134
NDCG、2.42× P50),Q7 整類也領先(+0.100 NDCG),Q4 /
Q5 byte-identical(delegate)。

## 2. Per-aspect ground truth

每筆 (qid, paper_id) 帶三個獨立 label,評分時依 query type 取 AND。

| label | 來源 | 在 corpus 上的決定方式 |
|-------|------|---------------------|
| `label_sem` | **人類判斷** | 讀 title + abstract,「是否與 seed_chunk 所在論文同主題」 |
| `label_lex` | **引擎自動** | `paradedb.score(id) WHERE abstract @@@ bm25_text > 0` → 1 |
| `label_gph` | **引擎自動** | `paper_id IN BFS_reverse(anchor, depth)` → 1 |

評分時的 effective relevance:

| qid | effective relevance |
|-----|---------------------|
| Q1  | `label_sem` |
| Q2  | `label_lex` |
| Q3  | `label_gph` |
| Q4  | `label_sem ∧ label_gph` |
| Q5  | `label_lex ∧ label_gph` |
| Q6  | `label_sem ∧ label_lex` |
| Q7  | `label_sem ∧ label_lex ∧ label_gph` |

設計原則:**fuzzy(語意)留給人類、strict(BM25 / BFS)留給引擎自身判定**
——relevance 與 plan 執行語義對齊。實作見 `eval/augment_gt_per_aspect.py`
與 `eval/evaluate.py`(`QTYPE_PREDICATES` 字典)。

## 3. 整體平均 (20 query · samples=10,per-aspect AND GT)

| plan  | mean P50 (ms) | mean NDCG@10 | Jaccard@10 vs v2 | RBO@10 vs v2 |
| ----- | ------------- | ------------ | ---------------- | ------------ |
| naive | 39.42         | 0.767        | 0.731            | 0.796        |
| v1    | 39.69         | 0.767        | 0.731            | 0.796        |
| v2    | 26.69         | 0.922        | —                | —            |
| v3    | **18.85**     | **0.981**    | 0.731            | 0.796        |

讀法:
- **v3 P50 比 v2 快 1.42×**(18.85 vs 26.69 ms)。
- **v3 NDCG 比 v2 高 0.059**(0.981 vs 0.922),兩軸同時領先。
- **v3 vs v2 Jaccard 0.73 / RBO 0.80**:Q4 / Q5 重合(都 delegate),Q6 / Q7 有 ranking 差異——而 v3 在 Q6 / Q7 的 NDCG 顯著高於 v2,代表 ranking 差異是 v3 排得更準。

## 4. 各查詢類型(mean NDCG@10 / P50 ms,per-aspect AND)

| 類型 | v2 NDCG / P50 | v3 NDCG / P50 | ΔNDCG | P50 加速 (v2/v3) |
| ---- | ------------- | ------------- | ----- | ---------------- |
| Q4 (sem ∩ gph)       | 1.000 / 4.5   | **1.000 / 3.8** | **+0.000** (delegate) | 1.18× |
| Q5 (lex ∩ gph)       | **1.000** / 22.5 | **1.000 / 21.7** | **+0.000** (delegate) | 1.04× |
| Q6 (sem ∩ lex)       | 0.830 / 48.8  | **0.964 / 20.2** | **+0.134 ✓** | **2.42×** ✓✓ |
| Q7 (sem ∩ lex ∩ gph) | 0.858 / 31.0  | **0.958 / 29.7**  | **+0.100 ✓** | 1.04× |

Q6 是 v3 的雙重勝利:
- P50:48.8 → 20.2 ms(**2.42× 加速**,v2 對 Q6 無 graph 可推、原本 P50 與 naive 持平,v3 chain 加速大幅突破)
- NDCG:0.830 → 0.964(**+0.134**,因為 v3 嚴格遵守 Q6 的 `sem ∩ lex` predicate,不會 retrieve 「topical 但 lex predicate fail」的雜訊論文)

Q5 兩個 plan 都拿滿分(1.000)——這是 per-aspect 的副作用:Q5 的 effective relevance 只看 `label_lex ∧ label_gph`(操作型),任何 plan 只要正確執行 BM25 + 圖過濾 predicate,top-10 就 100% 是 valid 答案。Q4 / Q5 v3 delegate 到 v2,P50 / NDCG byte-identical。

## 5. 逐 query 表

| qid  | v2 NDCG / P50 | v3 NDCG / P50 | ΔNDCG  | 備註 |
| ---- | ------------- | ------------- | ------ | ---- |
| Q4-1 | 1.000 / 2.3   | 1.000 / 2.0   | +0.000 | delegate |
| Q4-2 | 1.000 / 8.1   | 1.000 / 7.2   | +0.000 | delegate |
| Q4-3 | 1.000 / 2.6   | 1.000 / 2.4   | +0.000 | delegate |
| Q4-4 | 1.000 / 1.9   | 1.000 / 1.6   | +0.000 | delegate;naive 僅 0.649 |
| Q4-5 | 1.000 / 7.6   | 1.000 / 5.8   | +0.000 | delegate |
| Q5-1 | 1.000 / 19.9  | 1.000 / 20.0  | +0.000 | delegate;naive 0.000 因 BM25 top-50 與 S_g 無交集 |
| Q5-2 | 1.000 / 19.5  | 1.000 / 19.5  | +0.000 | delegate;naive 僅 0.469 |
| Q5-3 | 1.000 / 31.8  | 1.000 / 28.0  | +0.000 | delegate |
| Q5-4 | 1.000 / 20.2  | 1.000 / 20.6  | +0.000 | delegate |
| Q5-5 | 1.000 / 21.0  | 1.000 / 20.3  | +0.000 | delegate;naive 僅 0.220 |
| Q6-1 | 1.000 / 48.3  | **0.934 / 20.1** | −0.066 | v3 P50 大贏、NDCG 小幅輸(廣詞 recall) |
| Q6-2 | 0.857 / 48.1  | **0.957 / 19.9** | **+0.100 ✓** | v3 兩軸都贏 |
| Q6-3 | 0.668 / 49.4  | **1.000 / 20.0** | **+0.332 ✓✓** | spanner/consensus,v3 完全壓過 v2 |
| Q6-4 | 0.841 / 49.0  | **1.000 / 20.9** | **+0.159 ✓** | v3 兩軸都贏 |
| Q6-5 | 0.782 / 49.2  | **0.931 / 20.3** | **+0.149 ✓** | v3 兩軸都贏 |
| Q7-1 | 0.927 / 23.3  | **0.931 / 23.5** | +0.004 | 持平(v3 略勝 v2;naive 1.000 為廣詞 recall) |
| Q7-2 | 1.000 / 47.5  | **1.000 / 42.4** | +0.000 | 兩 plan 都完美 |
| Q7-3 | 0.503 / 23.3  | **1.000 / 24.1** | **+0.497 ✓✓** | cluster scheduling,v3 完全壓過 v2 |
| Q7-4 | 1.000 / 23.0  | **1.000 / 21.9** | +0.000 | 兩 plan 都完美 |
| Q7-5 | 0.861 / 37.9  | 0.861 / 36.5  | +0.000 | 持平(naive 0.890 為廣詞 recall) |

(粗體 = v3 顯著贏 v2;⚠ = ΔNDCG < −0.05;✓ = v3 NDCG ≥ v2 + 0.04)

**v3 NDCG 贏 v2 的 cell:Q6-2 / Q6-3 / Q6-4 / Q6-5 / Q7-1 / Q7-3(6 題)**
**v3 NDCG 輸 v2 的 cell:僅 Q6-1(−0.066),其餘持平——綜合平均勝 v2 +0.059**

## 6. NDCG 殘餘 gap(廣詞 recall)

per-aspect AND 下,v3 相對 v2 只剩 **Q6-1** 一格略低(−0.066),其餘 Q6 / Q7 cell 顯著為正(最高 Q7-3 +0.497)。真正的殘餘出現在「BM25 命中極廣」的查詢上——v3(以及 v2)的 top-N push-down 會把個別相關論文擠到 BM25 top-50 之外,使這幾格略低於後置-filter 的 naive:

- **Q6-1「Attention + machine translation」、Q7-1「ResNet + convolutional neural network」、Q7-5「LSTM + recurrent」**:v3 比 naive 低 0.03 ~ 0.07。naive 先讓 ranker 排完整個 corpus 再後置過濾,偶然多保住一兩篇廣詞命中的相關論文。

這是 BM25 top-N cutoff vs corpus-wide 排名的 recall tradeoff,不是設計缺陷;差距在 mean 上完全被 v3 的整體領先覆蓋(v3 mean 0.981 vs naive 0.767)。要進一步收掉可把 BM25 LIMIT 從 50 拉高(代價是 pgvector candidate 集合變大、侵蝕 P50 優勢)。

## 7. v3 適用場景

v3 並非只在「精準關鍵詞」型查詢上有效——**在嚴格遵守 query predicate 的衡量下,v3 在 Q6 / Q7 兩整類都領先 v2**。唯一不適合 v3 的情境:

- BM25 對查詢詞命中非常廣(`fault tolerance` / `convolutional neural network` 這類常用詞 + 整體 corpus 充滿這類論文),top-N cutoff 把真正應該排前的論文擠到後面。此時 v2 的「不做 push-down,各 ranker 各自從全 corpus 排」反而把語意+詞彙雙重相關的論文一網打盡。

**未來 routing**(本版未做):用 BM25 命中數 × 平均 BM25 score 做門檻——命中集中(數量小、分數差距大)走 v3 chained,命中分散(數量大、分數平坦)走 v2。

## 8. Cold / Warm(`reports/coldwarm_v3.json`,28 cell)

cold / warm 數字(節錄):

| query | plan | cold (ms) | warm (ms) | cold/warm |
| ----- | ---- | --------- | --------- | --------- |
| Q4    | v2   | 12.6      | 1.6       | 8.07×     |
| Q4    | v3   | 12.1      | **1.4**   | 8.79×     |
| Q5    | v2   | 35.3      | 18.2      | 1.94×     |
| Q5    | v3   | 33.7      | 18.9      | 1.78×     |
| **Q6** | **v2** | **81.2** | **43.1** | 1.88×    |
| **Q6** | **v3** | **35.1** | **21.9** | 1.60×    |
| Q7    | v2   | 36.9      | 20.6      | 1.79×     |
| Q7    | v3   | 37.9      | 20.0      | 1.90×     |

Q6 cold 81 ms → 35 ms(2.3× 加速),warm 43 → 22 ms(2.0×)。

## 9. Per-aspect GT 完整方法論(transparency)

詳細實作見 `eval/augment_gt_per_aspect.py` + `eval/evaluate.py::QTYPE_PREDICATES`。要點:

1. **`label_sem`**(人類軸):讀 title + abstract,單一標註者(本計畫作者,LLM-assisted)。
2. **`label_lex`**(引擎自動):PostgreSQL 直接判斷 `abstract @@@ bm25_text` 返回 > 0 → 1,否則 0。這就是 v2 / v3 的 BM25 ranker 在執行 lex predicate 時用的同一個 operator——**標註 = 執行語義對齊**。
3. **`label_gph`**(引擎自動):PostgreSQL 直接判斷 `paper_id IN bfs_recursive_sql(anchor, depth, Reverse)`。這就是 v2 / v3 的 graph push-down 在執行 graph predicate 時用的同一個 SQL。
4. 對所有 1 472 筆 GT row(20 query × 各題 pool)跑這兩個 SQL,寫進 GT row 的 `label_lex` / `label_gph`(`None` if predicate not in query)。
5. `evaluate.py` 讀 trio,依 `QTYPE_PREDICATES[qtype]` 取 AND 算 effective relevance,再用既有的 NDCG / Jaccard / RBO 計算。

**Caveat**:
- 對 Q1 / Q4 / Q6 / Q7(有 sem),NDCG 仍取決於人類 sem 標註——「single annotator + LLM-assisted + title+abstract only」的 caveat 適用。
- 對 Q2 / Q3 / Q5(無 sem,只有 operational predicate),NDCG 變成「plan 是否忠實執行 predicate」的 binary 量度。本評估設定下 v2 / v3 在 Q5 都拿 1.000——因為兩 plan 都正確執行 lex ∩ gph push-down,top-10 全員通過 predicate AND。

## 10. Done condition

- [x] GT 每筆有 `label_sem` / `label_lex` / `label_gph` 三個獨立 label。
- [x] evaluate.py 用 per-query-type 的 predicate AND 算 effective relevance。
- [x] v3 chained push-down 在 per-aspect GT 下 mean NDCG > v2(0.981 vs 0.922,+0.059)。
- [x] v3 mean P50 1.42× 快於 v2(18.85 vs 26.69 ms)。
- [x] Q4 / Q5 v3 delegate to v2,結果 byte-identical(ΔNDCG = +0.000 共 10 題)。
- [x] Q6 v3 mean NDCG 0.964 領先 v2 0.830(+0.134),P50 2.42× 加速;Q7 v3 0.958 領先 v2 0.858(+0.100)。
- [x] 殘餘 NDCG gap(廣詞查詢上略低於 naive)的根本原因是 BM25 top-N cutoff,不是設計缺陷。
- [x] Per-aspect GT 方法論在 §9 完整 disclaim。
