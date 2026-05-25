# v3 Plan Design (一氣呵成版)

Author: Chan Ching-Kan · 2026-05-26

## 0. Why v3

| Plan  | 強項                                | 痛點                                                                 |
| ----- | ---------------------------------- | ------------------------------------------------------------------- |
| naive | 三引擎 ranking 訊號都進 RRF        | 慢;ranker 排整個 corpus 後常被圖過濾砍空。                          |
| v1    | 加 selectivity reorder + BFS cost  | cost 公式只是 annotation,top-K + RRF 拓撲下 reorder 不改變 total work,實測等同 naive。 |
| v2    | Push-down graph filter,P50 1.44×,NDCG +0.126 | graph 退化為硬過濾,RRF 只剩 vector / BM25 兩條訊號;naive 的 三路 fusion 訊號被丟掉。 |

**v3 命題:同時拿到三件事**

1. 把 v2 的 push-down **擴張到多階段**(graph + lexical 兩個 hard predicate)。
2. 把 v1 的 cost 從 *annotation* 升級成 **actionable**:當兩個可 push-down 的 predicate 同時存在(Q5、Q7),用 cost 決定 push-down 先後。
3. 找回 naive 的 **三路 fusion ranking 訊號**——對縮小後的候選 re-score
   `(vector, bm25, graph_distance)`,再餵進 RRF。

核心 insight:v2 把 graph filter 當 *hard predicate* 用(filter context),丟掉了它作為 *ranking signal* 的價值(query context)。**v3 對每個 hard predicate 同時兩用**——filter 階段縮小 candidate,fusion 階段在縮小後的候選上 re-score 餵回 RRF。

---

## 1. Per-query algorithm

符號:`S_g` = graph BFS 集合(以及每個 paper 的最小 depth);`S_l` = BM25 命中集合(以及每個 paper 的 BM25 score);`S_both = S_g ∩ S_l`。

### Q1 / Q2 / Q3 (單引擎)
等同 v2(也等同 v1、naive 的單引擎 path);完全不引入 v3-specific 邏輯。**保證 hard equivalence**——單元測試會檢查 V3Plan 在 Q1/Q2/Q3 上 dispatch 到與 v2 相同的 helper。

### Q4 (semantic ∩ graph)
```
S_g, depth_map = BFS_with_depth(anchor, depth)
if S_g empty → []
vector_rank = pgvector top-(k×5) WHERE paper_id = ANY(S_g)
graph_dist_rank = sort(S_g, key = depth ASC, paper_id ASC)
return RRF([vector_rank, graph_dist_rank], k=60, top_k)
```

### Q5 (lexical ∩ graph) — **cost-driven push-down**
```
cost_bfs  = age_cost_v1(depth, 2.4) * age_ms_unit
cost_bm25 = pg_search_cost(matched, 50.0) * pg_search_ms_unit   # matched 來自 COUNT(*)
first = arg_min(cost_bfs, cost_bm25)       # tie-break by lower selectivity
if first == BFS:
    S_g, depth_map = BFS_with_depth(anchor, depth)
    bm25_hits = BM25 WHERE @@@ AND id = ANY(S_g) ORDER BY score DESC LIMIT k×5
                       # push-down 進 BM25
else:
    bm25_hits = BM25 WHERE @@@ ORDER BY score DESC LIMIT k×5
    S_g, depth_map = BFS_with_depth(anchor, depth)
    bm25_hits = [h for h in bm25_hits if h ∈ S_g]
S_both = ids(bm25_hits) ∩ S_g
bm25_rank        = sort(S_both, key = bm25_score DESC)
graph_dist_rank  = sort(S_both, key = depth ASC, paper_id ASC)
return RRF([bm25_rank, graph_dist_rank], k=60, top_k)
```

### Q6 (semantic ∩ lexical) — no graph, no cost decision
```
S_l, bm25_score_map = BM25 ORDER BY score LIMIT k×5
if S_l empty → []
vector_rank = pgvector top-(k×5) WHERE paper_id = ANY(S_l)
bm25_rank   = S_l (in BM25 order; truncated to top-(k×5))
return RRF([vector_rank, bm25_rank], k=60, top_k)
```
(這比 v2 Q6 多了「先 BM25,push-down 進 pgvector」的捷徑——v2 在 Q6 退化到 v1 path,沒做 push-down。v3 即使沒有圖過濾也仍多吃一層 push-down 效率。)

### Q7 (all three) — **兩階段 push-down + cost-driven ordering**
```
cost_bfs, cost_bm25 = same as Q5
first = arg_min(cost_bfs, cost_bm25)
if first == BFS:
    S_g, depth_map = BFS_with_depth(anchor, depth)
    bm25_hits      = BM25 WHERE @@@ AND id = ANY(S_g) ORDER BY score DESC LIMIT k×5
else:
    bm25_hits      = BM25 WHERE @@@ ORDER BY score DESC LIMIT k×5
    S_g, depth_map = BFS_with_depth(anchor, depth)
    bm25_hits      = [h for h in bm25_hits if h ∈ S_g]
S_both = ids(bm25_hits) ∩ S_g                           # (4)
vector_hits = pgvector top-(k×5) WHERE paper_id = ANY(S_both)   # (3)
vector_rank        = vector_hits                                # by HNSW distance
bm25_rank          = sort(S_both, key = bm25_score DESC)        # (4)
graph_dist_rank    = sort(S_both, key = depth ASC, paper_id)    # (4)
return RRF([vector_rank, bm25_rank, graph_dist_rank], k=60, top_k)   # (5)
```

---

## 2. Cost-based ordering (Q5 / Q7)

公式 reuse,**不重新擬合**:
```
cost_bfs_ms  = age_cost_v1(depth, 2.4) * age_ms_unit           # 2.4 ≈ avg out-degree from microbench
cost_bm25_ms = pg_search_cost(matched, 50.0) * pg_search_ms_unit
```
- `age_cost_v1` 與 `age_ms_unit` 來自 `bench/src/cost.rs`,係數出自 `microbench` 對 5K + 50K corpus BFS P50 的最小平方擬合。
- `matched` 是 `SELECT count(*) FROM papers WHERE abstract @@@ $1`(v1 / v2 已有同樣的 round-trip)。
- Tie-break:cost 平手時,挑 selectivity 更小者(估出的集合更小);兩者皆平則 BFS 優先(graph 對下游 push-down 通常更便宜)。

決策被 **log 出來**(tracing INFO),也寫進 `PlanResult.actual_order` 與 `PlanResult.first_predicate`,讓 evaluation 可以後驗。

> 粗估註明:`avg_branching = 2.4` 是從 microbench 對 50K corpus reverse BFS 的近似平均;`avg_posting = 50` 是 ParadeDB 文件給的粗估。兩者皆已寫死於 v1 / v2 路徑,v3 沿用相同常數以維持「同一 cost model 比較不同 plan」的公平性。

---

## 3. Edge cases & invariants

- **候選不在 BFS 鄰域** → 不出現在 `depth_map`,因此不會進 `graph_dist_rank`。等同 sentinel = "rank 結尾 + 不貢獻 RRF 分數",與 brief 提示的兩種選擇中我選「從 graph_rank list 剔除」這一支(實作較乾淨,RRF 行為定義良好;`plan.rs` `build_graph_distance_rank` 旁有註解說明)。
- **HNSW + WHERE** 仍需 `SET LOCAL hnsw.iterative_scan = strict_order`(沿用 v2 的 `run_semantic_pushdown` helper)。
- **S_g 空** → 提早 return `[]`(同 v2)。
- **S_l 空 (Q5/Q6/Q7)** → 提早 return `[]`。
- **S_both 空 (Q5/Q7)** → 提早 return `[]`。
- **單元測試保證 Q1/Q2/Q3 hard-equivalent to v2**:V3Plan 在這三型直接 delegate 到 v2 採用的 `semantic_only` / `lexical_only` / `graph_only` helper(都是 module 內 free function;v3 不修改它們)。

---

## 4. Touched files

| 檔案 | 修改範圍 |
| ---- | ------- |
| `bench/src/graph_engine.rs` | **加** 新 `bfs_recursive_sql_with_depth(...) -> Vec<(i64, i32)>`(回傳 paper_id + 最小 depth)。既有 `bfs_recursive_sql` 不動。 |
| `bench/src/plan.rs` | **加** `V3Plan` struct + `impl Plan`、`multi_predicate_v3` driver、若干 push-down helper(graph→bm25 with 留 score、bm25 score-map fetcher、graph_distance_rank builder)。NaivePlan / V1Plan / V2Plan 與其輔助 helper(`semantic_only` / `lexical_only` / `graph_only` / `multi_predicate` / `multi_predicate_pushdown` / `run_semantic_topn` / `run_bm25_topn` / `run_semantic_pushdown` / `run_bm25_pushdown`)**完全不動**。 |
| `bench/src/fusion.rs` | 不改 k=60。對 `FusionStrategy::Linear` 補 `#[allow(dead_code)]` 移除既有 release warning(避免拖累 v3 build 條件);測試 + RRF 行為不動。Ranking::engine 在 v3 路徑會被讀來 log(自然解除 dead warning)。 |
| `bench/src/cost.rs` | 不改既有公式。可選地 inline 估算組合於 plan.rs,免改 cost.rs。實作上 plan.rs 直接呼叫既有 `age_cost_v1` / `pg_search_cost` / `normalize`。 |
| `bench/src/main.rs` | dispatch 表新增 `"v3" => Box::new(plan::V3Plan)`。 |
| `eval/evaluate.py` | 把 v3 加進每題 evaluation 列表,輸出新增 `ndcg10_v3` / `jaccard10_v3_*` / `rbo10_v3_*` 欄位。預設 out 改成 `reports/eval_v3.json`(`--out` 仍可覆寫;baseline `eval_phase1_e4.json` 不動)。 |
| `scripts/coldwarm_all_28.py` | 新增,對 4 plan × 7 query 跑 cold/warm。 |
| `docs/report.html` | §4.7 / §8.4 / §9.5 / §11 disclaim、表 7 / 11 / 12 / 13 / 14 增 v3 row 或 column。 |
| `README.md` | TL;DR 表加 v3、"three plans" 改 "four plans"、Reports 區段加三筆。 |

---

## 5. 為什麼 `Plan` trait 不擴充

`Plan::execute` 已經回傳 `PlanResult { predicates, first_predicate, per_engine_rows, round_trips, materializations, actual_order, ... }`,可以**完整描述** v3 的 cost-decision、push-down 順序與每階段集合大小:

- `first_predicate` 紀錄 cost 比較後實際先做的 hard predicate(Q5/Q7 = Engine::Age 或 Engine::PgSearch;Q4 = Age;Q6 = PgSearch;Q1–Q3 = 單引擎本身)。
- `actual_order` 完整紀錄執行順序(Q7 cost 決定後例如 `[Age, PgSearch, Pgvector]` vs `[PgSearch, Age, Pgvector]`)。
- `predicates` 帶完整 cost annotation,evaluation 與 audit 可後驗。
- `per_engine_rows` 紀錄每階段集合大小(S_g / S_l / S_both / vector 候選等),供 §8.4 寫作引用。

不需要新欄位也能滿足 brief 的「**Q5/Q7 的 cost-based ordering 決策有 log 輸出**」要求(透過 tracing + PlanResult 雙管道)。

---

## 6. Fusion 是否要支援第三條 ranking 訊號

`fusion::fuse(&[Ranking], &FusionStrategy::Rrf{k}, intersect_with, top_k)` 已是 **N-way**;feed 三個 `Ranking` 就是三條訊號(Q7)。**不修改 fusion.rs 介面**;只在 `FusionStrategy::Linear` 補一個 `#[allow(dead_code)]` 來清掉既有 release warning(這是 fusion.rs 既存的 dead-code 警告,非由 v3 引入,但 Done condition #1 要求 `cargo build --release` 不能有 warning,所以順便清掉)。Linear 變體仍保留,作為未來 ablation。

`graph_distance_rank` 就是一個普通的 `Ranking { engine: "graph_distance", paper_ids: ... }`——對 RRF 來說它是一條按 depth 升冪排序的 ranking,跟 vector / bm25 的 ranking 完全同型。

---

## 7. Tests

新增於 `bench/src/plan.rs` `tests` mod(只 *增加*,不刪減既有 5 個 plan test):

- `v3_plan_name_is_v3` — `V3Plan.name() == "v3"`。
- `v3_uses_v2_pathway_for_single_engine` — assert 對 Q1/Q2/Q3 而言 v3 dispatch 與 v2 等價(透過 `V2Plan::should_use_pushdown` 回傳 false 的同樣判斷;以此 trigger v2 的 dead-code 解除)。
- `v3_cost_ordering_decision_q5_picks_cheaper` — pure function 測試(把 cost 拆成 helper)選 BFS 還是 BM25。
- `v3_cost_ordering_decision_q7_same` — Q7 cost 決策邏輯。
- `v3_graph_distance_rank_orders_by_depth` — pure helper 測試。
- `v3_graph_distance_rank_drops_unmapped` — sentinel 行為(候選不在 BFS depth_map 內就剔除)。
- `v3_engines_uses_v2_should_use_pushdown_helper` — v3 內部呼叫 `V2Plan::should_use_pushdown` 來判定 "是否有圖 push-down 對象",清除 should_use_pushdown 的 dead-code warning。

(這些測試使 release build 自然引用 `Plan::name` / `Ranking::engine` / `V2Plan::should_use_pushdown`,搭配 `#[allow(dead_code)]` on Linear,讓 `cargo build --release` 達到 zero warning。)

---

## 8. Causal narrative (for §4.7 / §11 / README)

> naive 三路 fusion 訊號齊全但慢 → v1 想用 cost 重排但實證無效(cost 公式只能是 annotation) → v2 用 push-down 同時加速 + 提精度,但 graph 退化成 hard filter、丟掉 ranking 訊號 → **v3 多階段 push-down(graph + lexical 兩個 hard predicate)、cost 決定 push-down 先後順序(v1 的 cost 公式終於從 annotation 升級成 actionable)、在小候選集上 re-score 餵回 RRF(找回 naive 的 fusion 訊號),同時拿到三者**。
