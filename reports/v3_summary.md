# v3 plan · 結果摘要(誠實版)

Source artifacts: `reports/eval_v3.json` (20 query × 4 plan × 10 samples)
+ `reports/coldwarm_v3.json` (7 query × 4 plan, container-restart cold/warm).
Measured 2026-05-26 on the same 50K-paper corpus + WSL2 Docker rig used by
naive / v1 / v2.

## 1. 一句話總結

v3 **成功示範了 cost-based ordering 可以從 annotation 變成 actionable**(Q5/Q7 的 push-down 先後順序由 cost 公式決定,並寫進 `PlanResult.first_predicate` / `actual_order`),也**重新引入了 graph_distance 作為 RRF 的第三條 ranking 訊號**——但 *作為一個排名品質指標*,在這個 corpus 上,**v3 的 NDCG@10 顯著低於 v2**(0.562 vs 0.801);P50 比 naive / v1 快、但慢於 v2。這是一個有教育意義的負面結果,§5 詳列原因。

## 2. 整體平均 (20 query · samples=10)

| plan  | mean P50 (ms) | mean NDCG@10 | Jaccard@10 vs naive | RBO@10 vs naive | Jaccard@10 vs v2 | RBO@10 vs v2 |
| ----- | ------------- | ------------ | ------------------- | --------------- | ---------------- | ------------ |
| naive | 34.04         | 0.675        | —                   | —               | 0.601            | 0.695        |
| v1    | 34.37         | 0.675        | 1.000               | 1.000           | 0.601            | 0.695        |
| v2    | **23.38**     | **0.801**    | 0.601               | 0.695           | —                | —            |
| v3    | 27.62         | 0.562        | 0.265               | 0.392           | 0.328            | 0.416        |

讀法
- **v2 仍是綜合最佳**:P50 23.4 ms、NDCG@10 0.801,兩個軸都贏。
- **v3 比 naive/v1 快**(27.6 ms vs 34 ms),但慢於 v2;v3 的 NDCG@10 下降到 0.562——比 naive 還低。
- **v3 vs v2 Jaccard 0.33 / RBO 0.42**:多了 graph_distance 與 BM25-as-filter 兩條結構性變更,結果集本來就會不同(這是設計目標,不是錯誤);但 NDCG 把這變化判定為「不貼題」。
- **v3 vs naive Jaccard 0.27 / RBO 0.39**:這個數字只記錄不優化(naive 自己 NDCG 才 0.675,不該成為目標)。

## 3. 各查詢類型平均 (mean P50, ms · NDCG@10)

| 類型 | naive P50 / NDCG | v1 P50 / NDCG | v2 P50 / NDCG | v3 P50 / NDCG | v3 vs v2 (NDCG 差) |
| ---- | ---------------- | ------------- | ------------- | ------------- | ----------------- |
| Q4   | 11.4 / 0.766     | 10.7 / 0.766  | **2.5 / 0.917** | 2.7 / 0.559  | **−0.358 ⚠** |
| Q5   | 32.4 / 0.497     | 31.7 / 0.497  | **19.5 / 0.742** | 36.1 / 0.584 | −0.158 ⚠ |
| Q6   | 42.6 / 0.618     | 42.7 / 0.618  | 44.6 / 0.618  | **36.8 / 0.446** | −0.172 ⚠ |
| Q7   | 47.7 / 0.819     | 48.0 / 0.819  | **27.0 / 0.925** | 34.9 / 0.658 | **−0.267 ⚠** |

註:Q6 是唯一 v3 比 v2 快的類型(36.8 vs 44.6 ms)——因為 v2 對 Q6 沒有 push-down 對象,而 v3 把 BM25 命中集合做 push-down 進 pgvector;代價是 NDCG 同時掉了 0.17。

## 4. 逐 query 表

| qid  | naive | v1 | v2 | v3 | ΔNDCG(v3−v2) | P50: v3 vs v2 |
| ---- | ----- | -- | -- | -- | ------------- | -------------- |
| Q4-1 | 1.000 | 1.000 | 1.000 | 0.521 | **−0.479 ⚠** | 1.5 / 1.5 |
| Q4-2 | 0.726 | 0.726 | 0.861 | 0.647 | −0.214 ⚠ | 5.6 / 4.8 |
| Q4-3 | 0.780 | 0.780 | 1.000 | 0.234 | **−0.766 ⚠** | 1.9 / 1.9 |
| Q4-4 | 0.649 | 0.649 | 0.788 | 0.927 | **+0.139** ✓ | 1.2 / 1.3 |
| Q4-5 | 0.675 | 0.675 | 0.934 | 0.467 | −0.467 ⚠ | 3.2 / 2.9 |
| Q5-1 | 0.000 | 0.000 | 0.637 | 0.637 | +0.000 | 30.5 / 17.4 |
| Q5-2 | 0.469 | 0.469 | 0.855 | 0.797 | −0.059 | 31.0 / 17.4 |
| Q5-3 | 0.861 | 0.861 | 0.861 | 0.498 | −0.363 ⚠ | 52.9 / 24.7 |
| Q5-4 | 0.934 | 0.934 | 0.934 | 0.687 | −0.247 ⚠ | 32.3 / 19.4 |
| Q5-5 | 0.220 | 0.220 | 0.425 | 0.300 | −0.125 ⚠ | 33.8 / 18.7 |
| Q6-1 | 0.779 | 0.779 | 0.779 | 0.782 | +0.003 | 41.6 / 43.8 |
| Q6-2 | 0.849 | 0.849 | 0.849 | 0.359 | **−0.490 ⚠** | 33.2 / 43.0 |
| Q6-3 | 0.396 | 0.396 | 0.396 | 0.149 | −0.248 ⚠ | 33.2 / 44.0 |
| Q6-4 | 0.526 | 0.526 | 0.526 | 0.469 | −0.057 | 39.8 / 45.8 |
| Q6-5 | 0.542 | 0.542 | 0.542 | 0.471 | −0.072 | 36.2 / 46.2 |
| Q7-1 | 0.927 | 0.927 | 0.927 | 0.735 | −0.191 ⚠ | 32.8 / 19.9 |
| Q7-2 | 0.745 | 0.745 | 0.905 | 0.462 | −0.444 ⚠ | 40.6 / 41.6 |
| Q7-3 | 0.861 | 0.861 | 0.934 | 0.357 | **−0.577 ⚠** | 33.3 / 21.2 |
| Q7-4 | 0.649 | 0.649 | 1.000 | 0.936 | −0.064 | 32.9 / 19.1 |
| Q7-5 | 0.915 | 0.915 | 0.861 | 0.801 | −0.060 | 34.9 / 33.0 |

⚠ = 單題 ΔNDCG < −0.1,§5 disclaim 詳列原因。

## 5. v3 在 NDCG 上輸給 v2 的根本原因 — 三條 disclaim

v3 的賣點是「**同時拿到三件事**」:多階段 push-down(graph + lexical 兩個 hard predicate)、cost 決定 push-down 先後、找回 fusion 訊號(graph_distance 第三條 ranking)。其中 cost-actionable 部分目標達成(`first_predicate` 在 Q5/Q7 上由 cost 決定,可從 `last_plan.first_predicate` 後驗);但 NDCG 顯著退步,需要誠實列出原因——這也是 §11 disclaim 的核心:

### 5.1 graph_distance_rank 不是好的 ranking signal(主因)

對 RRF 來說,vector_rank / bm25_rank / graph_distance_rank 三條都被視為 *等權* 排名訊號。但 graph_distance_rank 只是「離 anchor 幾跳」,**對相關性的訊號量遠小於語義或詞彙相似度**。把它跟 vector/bm25 等權混進 RRF,反而把 vector/bm25 對「真的貼題」的判斷稀釋掉了。

最極端的例子是 Q4-3「與 MapReduce 相關 且 cites MapReduce」:v2 把 vector 排出 push-down 後的前 10 個全部標為相關(NDCG = 1.000);v3 把 graph_distance(離 MapReduce 1 hop 的論文,paper_id 升冪)混進 RRF 後,排前面的變成「離 MapReduce 近但語義不接近 MapReduce 主題」的論文,NDCG 跌到 0.234。

**這是 v3 設計裡最重要的負面發現:在這個 corpus 上,把 graph filter 同時當 ranking signal 用,實證上反而傷害排名品質**。要修需要動 RRF 的權重(brief 規定 k=60 不動),或改 fusion 為 score-weighted——不在 v3 範圍內。

### 5.2 BM25-as-filter 對 Q6 是過度過濾

Q6(語義 ∩ 精準)沒有圖過濾,v3 的設計把 BM25 命中集合 (`S_l`) 當作 pgvector 的 push-down filter。S_l 的選擇性取決於查詢詞——「machine translation」、「batch normalization」這類詞 BM25 命中數百~數千篇,v3 把 pgvector 限制在這個池子內找 top-K。

但 v2 在 Q6 對 pgvector 是 **全 corpus** 排 top-N,RRF 跟 BM25 top-N 融合。某些 *語義上跟種子很近、但 abstract 沒出現該詞* 的論文 v2 找得到、v3 找不到(被 BM25 filter 砍掉)。Q6-2「ResNet 相關 + batch normalization」就是典型——若一篇論文寫 "BN" 而沒寫完整 "batch normalization",v3 直接漏抓,NDCG 從 0.849 跌到 0.359。

### 5.3 BFS 在 50K 上多半深度 1–2 集中,graph_distance 訊號離散度太低

實測 reverse BFS depth=2 的結果集裡,**大多數論文 depth=1 或 2**,depth 3 很少。`graph_distance_rank` 的 tie-break 落到 paper_id 升冪——這把 graph_distance 變成「在同 depth 內看哪個 paper_id 小」的 *純結構性* 排名,完全不反映相關性。RRF 把它跟 vector/bm25 等權合,反而把純隨機(paper_id 升冪)的訊號帶進 top-10。

> 後續 work:graph_distance 應該用 *連續分數*(如 `1 / (1 + depth)` × log-degree 加權)而不是離散 rank,讓同 depth 的論文不至於被 paper_id 順序強制排名。本 v3 暫不修。

## 6. P50 latency 解析

雖然 NDCG 不理想,P50 的故事仍有正向訊號:

- Q4:v3 vs v2 幾乎打平(2.7 vs 2.5 ms 平均),Q4-4 還領先(1.2 vs 1.3)。push-down 機制本身對單一 ranker 的 Q4 不會比 v2 慢。
- Q6:v3 36.8 ms 比 v2 44.6 ms 快——v3 的 BM25→pgvector push-down 在 v2 完全不做 push-down 的 Q6 上,latency 確實下降(代價是 NDCG 掉,§5.2)。
- Q5:v3 36.1 ms 比 v2 19.5 ms 慢,**因為 v3 多了一次 BM25 push-down(WHERE @@@ AND id = ANY($S_g))查詢以拿到完整 S_l ∩ S_g 的 score map**,而 v2 只取 top-N 不必算完整命中集合。
- Q7:v3 34.9 ms 比 v2 27.0 ms 慢,同樣是 BM25 push-down 拿完整命中集合 + pgvector 第二次 push-down 的成本。

## 7. 成本決策實際發生了嗎

是的——`reports/eval_v3.json` 每個 v3 row 的 `last_plan.first_predicate` 與 `actual_order` 都記錄了 cost 比較後的選擇。對 20 道題的 Q5/Q7(10 道)逐筆檢查:**全部選了 Engine::Age 先做**(BFS depth=2 的估算 ms_estimate ≈ 0.115,壓倒性低於 BM25 的數十 ms 估算)。

> 這意味著本 corpus + 本參數設定下,v1 的 cost 公式雖然「actionable」了,但實際決定永遠是「BFS 先」——和 v2 / v1 / naive 的執行順序在 graph predicate 存在時其實一致。要真正看到 cost 改變 push-down 方向,需要更高 depth(branching^depth 把 BFS 抬到 BM25 之上)或更選擇性的 BM25 查詢——這是未來 work。

## 8. Cold / Warm (28-cell, `reports/coldwarm_v3.json`)

選錄幾個關鍵 cell(完整數據見 JSON):

| query | plan | cold ms | warm ms | ratio |
| ----- | ---- | ------- | ------- | ----- |
| Q4    | v2   | 10.7    | **1.5** | 7.09× |
| Q4    | v3   | 11.6    | **1.4** | 8.33× |
| Q5    | v2   | 32.3    | **17.4** | 1.85× |
| Q5    | v3   | 46.8    | 30.8    | 1.52× |
| Q6    | v2   | 74.1    | 41.2    | 1.80× |
| Q6    | v3   | 77.3    | **42.8** | 1.80× |
| Q7    | v2   | 38.8    | **19.8** | 1.96× |
| Q7    | v3   | 51.7    | 32.2    | 1.61× |

cold/warm ratio 與 v2 同數量級(都在 1.5–8.3× 範圍),沒有出現 cold cache 災難。

## 9. Done 條件清單

- [x] `reports/eval_v3.json` 寫入 4 plan × 20 query × 10 samples 的完整數據(`results` 為 flat list,80 筆)。
- [x] `reports/coldwarm_v3.json` 寫入 4 plan × 7 query 的 cold/warm 量測(28 筆)。
- [x] `scripts/coldwarm_all_28.py` 對應實現。
- [x] **誠實標出 v3 比 v2 差的 cell**——§4 表的 ⚠ 標記、§5 詳列三條 disclaim。
- [x] **保留** `reports/eval_phase1_e4.json` 不動(naive/v1/v2 baseline 仍可後驗)。
