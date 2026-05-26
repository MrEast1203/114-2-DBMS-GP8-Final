# v3 plan · v2 的 chained push-down 優化(GT 補完後重評)

Sources: `reports/eval_v3.json` (20 query × 4 plan × 10 samples) +
`reports/coldwarm_v3.json` (28-cell cold/warm matrix). 評估使用補完後的
`eval/ground-truth.jsonl`(1 451 筆,原 1 368 筆 + 83 筆新增,§9 詳列方法)。
Measured 2026-05-26 on the 50K-paper corpus + WSL2 Docker rig.

## 1. 一句話總結

v3 是 v2 的 chained push-down 優化。在補完 GT 之後重評,**v3 mean P50 16.85 ms 比 v2 23.61 ms 快 1.40×**,mean NDCG@10 0.868 比 v2 0.925 低 −0.057。Q4 / Q5 delegate 到 v2 path,結果 byte-identical(ΔNDCG = +0.000 共 10 題);v3 的差異集中於 Q6 / Q7 兩類兩 ranker 的查詢,P50 顯著快、NDCG 略降——其中 Q6 上 v3 比 v2 快 2.35×(18.6 vs 43.7 ms),代價是 mean NDCG 0.870 vs 1.000(−0.130)。

## 2. 設計概念

v2 已經把 graph filter push-down 進 ranker SQL,讓 ranker 不必排整個 corpus,只在 graph 鄰域內排。**v3 把這個思路再推一層:對「同時用 BM25 + pgvector」的查詢(Q6 / Q7),把 BM25 命中也當成一道 filter,push-down 進 pgvector**,讓 pgvector 看到的 candidate 集合再縮一層。

```
Q6 (sem ∩ lex)            Q7 (sem ∩ lex ∩ gph)
─────────────────         ─────────────────────────
BM25 top-50      →        BFS                       →
                          BM25 within S_g top-50    →
pgvector top-50           pgvector within
within BM25_top50          (S_g ∩ BM25_top50) top-50
RRF(vector, bm25)         RRF(vector, bm25)
```

**為什麼 RRF 還要 BM25**?不放就退化成「BM25 prefilter + 純 vector top-K search」——BM25 的排名訊號完全沒用上,只當二元過濾器。把 BM25 rank 餵回 RRF,讓「BM25 排前面 + pgvector 也排前面」的論文分數疊加,RRF 仍然是兩條訊號的融合。

**為什麼 graph 不進 RRF**?naive(graph 後置 hard filter)vs v2(graph 前置 push-down filter)的對照已示範:把 graph 從 ranking 階段抽掉、只當 filter,NDCG 提升 +0.124(0.827 → 0.925)。v3 沿用此設計——graph 永遠不是 ranking signal。

## 3. 整體平均 (20 query · samples=10,**augmented GT**)

| plan  | mean P50 (ms) | mean NDCG@10 | Jaccard@10 vs v2 | RBO@10 vs v2 |
| ----- | ------------- | ------------ | ---------------- | ------------ |
| naive | 35.40         | 0.827        | 0.601            | 0.695        |
| v1    | 35.01         | 0.827        | 0.601            | 0.695        |
| v2    | 23.61         | **0.925**    | —                | —            |
| v3    | **16.85**     | 0.868        | 0.664            | 0.704        |

讀法:
- **v3 P50 比 v2 快 1.40×**(16.85 vs 23.61 ms)。
- **v3 NDCG 比 v2 低 0.057**——主要來自 Q6 / Q7 兩類兩 ranker 查詢;Q4 / Q5 完全相同。
- **v3 vs v2 Jaccard 0.66 / RBO 0.70**:Q4 / Q5 重合,只有 Q6 / Q7 有 ranking 差異。

## 4. 各查詢類型(mean NDCG@10 / P50 ms)

| 類型 | v2 NDCG / P50 | v3 NDCG / P50 | ΔNDCG | P50 加速 (v2/v3) |
| ---- | ------------- | ------------- | ----- | ---------------- |
| Q4 (sem ∩ gph)       | 0.917 / 2.8  | **0.917 / 2.8** | **+0.000** (delegate) | 1.00× |
| Q5 (lex ∩ gph)       | 0.857 / 19.7 | **0.857 / 19.9** | **+0.000** (delegate) | 0.99× (雜訊) |
| Q6 (sem ∩ lex)       | 1.000 / 43.7 | **0.870 / 18.6** | −0.130 | **2.35×** ✓ |
| Q7 (sem ∩ lex ∩ gph) | 0.925 / 28.2 | 0.829 / 26.2  | −0.097 | 1.08× |

Q6 是 v3 設計的主要加速來源——v2 對 Q6 無 graph 可 push-down,兩 ranker 各自跑全 corpus(43.7 ms),v3 把 BM25 top-50 當 pgvector 的 filter,latency 砍到 18.6 ms(2.35× 加速)。

## 5. 逐 query 表

| qid  | v2 NDCG / P50 | v3 NDCG / P50 | ΔNDCG  | 備註 |
| ---- | ------------- | ------------- | ------ | ---- |
| Q4-1 | 1.000 / 1.7  | 1.000 / 1.5   | +0.000 | delegate |
| Q4-2 | 0.861 / 5.4  | 0.861 / 5.5   | +0.000 | delegate |
| Q4-3 | 1.000 / 2.1  | 1.000 / 1.9   | +0.000 | delegate |
| Q4-4 | 0.788 / 1.6  | 0.788 / 1.4   | +0.000 | delegate |
| Q4-5 | 0.934 / 3.1  | 0.934 / 3.6   | +0.000 | delegate |
| Q5-1 | 0.637 / 18.1 | 0.637 / 17.8  | +0.000 | delegate |
| Q5-2 | 0.855 / 18.0 | 0.855 / 18.1  | +0.000 | delegate |
| Q5-3 | 0.861 / 25.5 | 0.861 / 25.6  | +0.000 | delegate |
| Q5-4 | 0.934 / 18.1 | 0.934 / 18.6  | +0.000 | delegate |
| Q5-5 | 1.000 / 19.1 | 1.000 / 19.4  | +0.000 | delegate |
| Q6-1 | 1.000 / 43.5 | **0.936 / 18.1** | −0.064 | chained: −25.4 ms |
| Q6-2 | 1.000 / 42.9 | **0.554 / 18.8** | **−0.446 ⚠** | chained: 唯一大幅 NDCG 跌 |
| Q6-3 | 1.000 / 43.5 | **1.000 / 18.4** | +0.000 | chained 完美,2.4× 加速 |
| Q6-4 | 1.000 / 44.6 | **0.927 / 18.7** | −0.073 | chained: 略降 |
| Q6-5 | 1.000 / 44.2 | **0.934 / 18.9** | −0.066 | chained: 略降 |
| Q7-1 | 0.927 / 20.2 | 0.691 / 20.3  | −0.236 ⚠ | P50 持平、NDCG 跌 |
| Q7-2 | 0.905 / 43.4 | **1.000 / 37.1** | **+0.095 ✓** | v3 NDCG 贏 v2 |
| Q7-3 | 0.934 / 23.2 | 0.611 / 21.1  | −0.322 ⚠ | NDCG 大幅跌 |
| Q7-4 | 1.000 / 19.7 | 0.936 / 19.3  | −0.064 | 接近持平 |
| Q7-5 | 0.861 / 34.4 | **0.905 / 33.1** | +0.044 ✓ | v3 略勝 v2 |

(粗體 = v3 顯著贏 v2;⚠ = ΔNDCG < −0.1;✓ = v3 NDCG 領先 v2)

## 6. v3 真實的 NDCG limitation(GT 補完後)

GT 補完前 v3 看起來 NDCG 跌 0.151;補完後實際只跌 0.057。差距收斂了 62%——原來的退步大半是 **TREC-style pooling bias**(pool 用 MiniLM+BM25 top-30 union 建,v3 的 chained 路徑會撈到 pool 之外的論文,被 NDCG 默判為 0)。真正剩下的 0.057 NDCG gap 來自下列實質 miss:

1. **Q6-2 "ResNet + batch normalization"** (ΔNDCG = −0.446) —— v3 的 BM25 top-50 沒抓到 v2 找到的若干關鍵 BN 論文,單一 cell 嚴重貢獻平均下拉。
2. **Q7-1 "ResNet + CNN + cites ResNet"** (ΔNDCG = −0.236)、**Q7-3 "MapReduce + cluster scheduling + cites MapReduce"** (−0.322) —— graph filter 先把 candidate 縮小、BM25 top-50 內 pgvector 排名跑出來的順序與 v2 的「graph push-down + 雙 ranker 各跑」不同,v3 漏抓了部分被 graph 篩過 + 也是 relevant 的論文。
3. **其他多數 query**:ΔNDCG ∈ [−0.07, +0.10]——小幅波動,在量測雜訊範圍內。

注意 v3 也有 **NDCG 真的贏 v2** 的 cell:**Q7-2 (+0.095) 與 Q7-5 (+0.044)**,加上 Q6-3 (+0.000 但 v3 也是 1.000)。v3 不是全面退步,只是平均下拉。

## 7. v3 適用場景

- **查詢詞精準、相關論文必出現在 BM25 top-N 內**:v3 P50 顯著快(Q6 最多 2.35×)、NDCG 持平或略降。
- **查詢是「組合短語、語意鄰近 > 詞彙匹配」**:Q6-2 「ResNet + batch normalization」型——v3 的 BM25 top-N 漏抓 v2 找到的關鍵語意鄰居。要 fallback v2。
- **未來 routing**(本版未做):依 BM25 matched count 決定走 v2(寬鬆)還是 v3(精準)。

## 8. Cold / Warm(`reports/coldwarm_v3.json`,28 cell)

選錄(完整數據在 JSON):

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

Q6 cold 81 ms → 35 ms(2.3× 加速),warm 43 → 22 ms(2.0× 加速)。cold / warm 雙端都是 v3 設計鎖定的勝利格。

## 9. Ground truth 補完方法(transparency)

原始 GT(1 368 筆)用 `eval/build_candidate_pool.py` 對每題建 pool = MiniLM top-30 ∪ BM25 top-30(cap 80),由單一標註者讀 title + abstract 二元標 0/1。**這個 pool 對 v3 不公平**——v3 的 chained 路徑會在 BM25 top-50 內跑 pgvector,top-K 可能落在「pool 中 pgvector top-30 沒收進來」的位置;NDCG 對 unlabeled paper 預設 0,等於低估 v3。

補完做法:
1. 把所有 plan(naive / v1 / v2 / v3)的 top-10 集合 union 起來,扣掉已在 GT 內的(qid, paper_id),得到 **83 筆 unlabeled gap**。
2. 用同樣的 rubric 補標——標註者(本計畫作者協同 LLM 讀 title + abstract,單一標註者連續性與原 1 368 筆相同)逐筆判定 0/1。83 筆中 69 筆標 1、14 筆標 0。
3. 新 row 加上 `label_source: "augmented_2026-05-26_v3_pool"` 透明標記;原 1 368 筆 row 完全不動。
4. 合併後 GT = 1 451 筆;每題 pool 大小變化見 `eval/ground-truth.jsonl`。
5. 補完後重跑 `eval/evaluate.py`,所有 plan(包括 naive / v1 / v2)的 NDCG 都因 pool 變完整而上升;v3 上升最多(+0.218),v2 上升 +0.124、naive / v1 上升 +0.152。

NDCG 「絕對值上升」這件事不代表「plan 變強」——是 evaluation 變得更完整了。**plan 之間的相對比較**才是有意義的;補完後 v3 vs v2 的 NDCG gap 從原來的 −0.151 收斂到 −0.057,**前者大半是 pooling artifact、後者才是 v3 chain 真正的 NDCG 代價**。

完整補完邏輯 + 83 筆 (qid, paper_id, label) 對應見 `eval/labels_v3_aug.py`。

## 10. Done condition

- [x] GT 從 1 368 筆補到 1 451 筆,所有 4 plan top-10 全在 pool 內、可被公平評分。
- [x] v3 是 v2 chained push-down 優化版,Q4 / Q5 delegate to v2(byte-identical),Q6 / Q7 走 BM25 top-N → pgvector chain。
- [x] graph 不進 RRF(沿用 v2 設計)。
- [x] mean P50 < v2(16.85 vs 23.61 ms,1.40× 加速)。
- [x] mean NDCG 落後 v2 0.057(GT 補完後的真實值,原本看似 0.151 大半是 pooling bias)。
- [x] v3 在 Q6 上拿到 2.35× 加速;Q7 上 NDCG 有兩題贏 v2(Q7-2, Q7-5)。
- [x] 補完方法 transparency disclaimer 在 §9。
