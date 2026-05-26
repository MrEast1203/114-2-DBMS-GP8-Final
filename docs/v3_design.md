# v3 Plan Design — v2 的 chained push-down 優化

Author: Chan Ching-Kan · 2026-05-26 (revised)

## 0. v3 定位

**v3 是 v2 的進一步效能優化版**。v2 已經把 graph filter push-down 進 ranker SQL,使 ranker 不必排整個 corpus、只在 graph 鄰域內排。v3 把這個 push-down 思路繼續推:對「同時用 BM25 + pgvector」的查詢(Q6 / Q7),把 BM25 命中集合也當成一道 filter,push-down 進 pgvector,讓 pgvector 看到的 candidate 集合再縮小一層。

最終 candidate set 從原本的:
- v2 Q6:pgvector 看全 corpus → ranker 排 top-N → 5000+
- v2 Q7:pgvector 看 S_g → ranker 排 top-N → 數百

縮到:
- v3 Q6:pgvector 看 BM25 top-N → ranker 排 top-K → 50
- v3 Q7:pgvector 看 S_g ∩ BM25 top-N → ranker 排 top-K → ≤ 50

更小的 candidate 直接帶來更短的 latency。

## 1. 為什麼 RRF 還要 BM25

如果只做兩道 push-down、最後回 pgvector top-K 完事,v3 就退化成「BM25 prefilter + 純 vector top-K search」——BM25 的 *排名* 訊號完全沒用上,只當了一道二元過濾器。

把 BM25 rank 餵回 RRF,讓「BM25 排前面 + pgvector 也排前面」的論文得分疊加。這保留了 BM25 的排名影響力,RRF 仍然是兩條訊號的融合,不是單一引擎排名。

```
v2 (no chain):
   pgvector top-50 ⊕ BM25 top-50 ⊕ filter by S_g → RRF → top-K

v3 (chained):
   BM25 top-50 [in S_g, if Q7] →
   pgvector top-50 in BM25 set →
   RRF(pgvector_topk, BM25_topN)
```

## 2. 為什麼 graph 不在 RRF

從 naive → v2 的 NDCG 對比就能讀出答案:

| plan  | graph 在哪 | NDCG@10 |
| ----- | --------- | ------- |
| naive | 後置 hard filter(三引擎 ranker 都跑、最後砍掉不在 S_g 的論文) | 0.675 |
| v2    | 前置 push-down filter(ranker SQL `WHERE id ∈ S_g`、graph 完全不參與排名) | **0.801** |

把 graph 從 ranking 階段抽掉、只當 filter 用,NDCG 提升 +0.126。**v3 沿用 v2 的這個發現**——graph 永遠不是 ranking signal,只當 filter。

(若硬把 graph 寫成「graph_distance rank = sort by BFS depth ASC」塞進 RRF,實證會把 NDCG 拉低到 ~0.56——前一版 v3 的歷史結果就是反例。本版 v3 不走那條路。)

## 3. Per-query 行為

| qid | engines | v3 行為 | 與 v2 差異 |
| --- | ------- | ------- | ---------- |
| Q1  | sem  | delegate to `semantic_only` helper(v2 用同一支) | byte-identical |
| Q2  | lex  | delegate to `lexical_only` helper | byte-identical |
| Q3  | gph  | delegate to `graph_only` helper | byte-identical |
| Q4  | sem ∩ gph | delegate to v2's `multi_predicate_pushdown` | byte-identical(只有一個 ranker 在 graph 下游,沒空間 chain) |
| Q5  | lex ∩ gph | 同上 | byte-identical |
| **Q6** | sem ∩ lex | BM25 top-N → pgvector WHERE id ∈ BM25_topN top-K → RRF(vector, bm25) | **v2 無 chain;v3 chain → P50 大幅快、NDCG 可能跌**(§5 disclaim) |
| **Q7** | sem ∩ lex ∩ gph | BFS → BM25 WHERE id ∈ S_g top-N → pgvector WHERE id ∈ (S_g ∩ BM25_topN) top-K → RRF(vector, bm25) | v2 的雙 ranker 分別 push-down 進 graph;v3 再串一層 BM25 → pgvector |

## 4. Algorithm(虛擬碼)

```python
# v3 Q6 path (multi_predicate_v3_chained, use_gph = False)
bm25_top_n = BM25 ORDER BY paradedb.score(id) DESC LIMIT n_fetch
if bm25_top_n empty: return []
vector_top_k = pgvector WHERE paper_id = ANY(bm25_top_n) ORDER BY <-> LIMIT n_fetch
                       SET LOCAL hnsw.iterative_scan = strict_order  # 沿用 v2
rankings = [Ranking(vector_top_k), Ranking(bm25_top_n)]
return RRF(rankings, k=60, top_k=k)

# v3 Q7 path (multi_predicate_v3_chained, use_gph = True)
S_g = bfs_recursive_sql(anchor, depth=2, Reverse)
if S_g empty: return []
bm25_top_n = BM25 WHERE id = ANY(S_g) ORDER BY score DESC LIMIT n_fetch
if bm25_top_n empty: return []
vector_top_k = pgvector WHERE paper_id = ANY(bm25_top_n) ORDER BY <-> LIMIT n_fetch
rankings = [Ranking(vector_top_k), Ranking(bm25_top_n)]
return RRF(rankings, k=60, top_k=k)
```

- `n_fetch = k × PER_ENGINE_OVERFETCH = 10 × 5 = 50`。
- 兩道 push-down 都用 `WHERE id = ANY($filter)`;pgvector 段沿用 v2 已驗證的 `SET LOCAL hnsw.iterative_scan = strict_order`(處理 HNSW + WHERE 的 recall 問題)。
- RRF k = 60,**不動**。
- Ranking 是兩條(vector + bm25),**不含 graph_distance**。

## 5. Tradeoff(必須誠實寫進 §11)

v3 的 chain 等於把 pgvector 限制在「BM25 top-N」內排語意距離。這個假設:

> *相關論文必出現在 BM25 top-N 內*

對「查詢詞精準」的題型成立(Q4 全勝、Q5 全勝、Q7 部分勝);對「查詢詞是組合短語」的題型不成立(Q6-2~Q6-5,例如 "cache timing attack"——abstract 沒寫完整短語就 BM25 排不進 top-N,但語意上相關,v3 漏抓)。實證結果:

- **mean P50:18.1 ms(v2 24.6 ms,1.36× 加速)**
- **mean NDCG@10:0.650(v2 0.801,−0.151)**
- 退步全集中於 Q6(mean NDCG 0.618 → 0.201;P50 46 ms → 20 ms = 2.3× 加速)。Q4 / Q5 與 v2 結果完全相同。Q7 NDCG 略降(0.925 → 0.740)、P50 略快。

未來 routing 想法(暫不實作):用 BM25 matched count 當門檻——matched 高就 v3、matched 低就 fallback v2。本版 v3 對所有 Q6/Q7 都跑 chained 路徑,讓 tradeoff 完整暴露。

## 6. 動到的檔案

| 檔案 | 修改範圍 |
| ---- | ------- |
| `bench/src/plan.rs` | 新增 `V3Plan` + `multi_predicate_v3_chained`。Q1–Q5 delegate;Q6/Q7 走新 chain。naive / v1 / v2 完全不動。 |
| `bench/src/main.rs` | 加 `"v3" => Box::new(plan::V3Plan)` dispatch + tracing log。 |
| `bench/src/coldwarm.rs` | 加 v3 dispatch arm。 |
| `bench/src/fusion.rs` | 一行 `#[allow(dead_code)]` 在 `Linear` variant 上(release zero-warning)。RRF k=60 不動,fuse 介面不動。 |
| `bench/src/graph_engine.rs` | (回滾)移除舊版 v3 的 `bfs_recursive_sql_with_depth`——新 v3 不需要 depth,只需要 paper_id 集合。 |
| `eval/evaluate.py` | `--out` 預設 `reports/eval_v3.json`,新增 v3 plan + v3-vs-v2 pairwise Jaccard/RBO;`results` flat list 供 tooling 自動驗證。 |
| `scripts/coldwarm_all_28.py` | 4 plan × 7 query = 28 cell cold/warm 矩陣。 |

## 7. Tests(`bench/src/plan.rs::tests`,只增不減既有 29 test)

新增 / 保留:
- `v3_plan_name_is_v3`
- `v3_uses_chained_pushdown_only_when_both_rankers_present` — 確認 Q6/Q7 走 chain,Q1-Q5 不走
- `v3_chained_set_and_v2_graph_pushdown_set_are_distinct` — v3 chain set {Q6, Q7} 與 v2 graph-pushdown set {Q4, Q5, Q7} 只在 Q7 交,釐清兩個優化的關係
- `v3_fuses_two_rankers_not_three` — RRF 餵兩條訊號(vector + bm25),驗證 fuse 行為與兩 ranker overlap 加權正確
- `v3_uses_v2_should_use_pushdown_to_classify_queries` — 同時讓 V2Plan::should_use_pushdown 在 release 路徑被引用(清掉既有 dead-code warning)

執行:`cargo test → 34 passed`(baseline 29)。
