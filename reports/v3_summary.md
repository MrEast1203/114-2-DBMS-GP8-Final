# v3 plan · v2 的 chained push-down 優化(誠實版)

Sources: `reports/eval_v3.json` (20 query × 4 plan × 10 samples) +
`reports/coldwarm_v3.json` (28-cell cold/warm matrix).
Measured 2026-05-26 on the same 50K-paper corpus + WSL2 Docker rig.

## 1. 一句話總結

v3 在 **P50 latency 上明確優於 v2**(mean 18.1 ms vs 24.6 ms,1.36× 加速)。代價是 **mean NDCG@10 降到 0.650**(v2 0.801,−0.151),退步集中於 Q6——v3 把 pgvector 鎖在 BM25 top-N 的命中集合內,落在「語意上接近但 abstract 沒有對應關鍵詞」的相關論文被刷掉。Q4 / Q5 v3 與 v2 結果**完全相同**(delegate to v2 path);Q7 NDCG 略降但 P50 略快。

## 2. v3 是 v2 的什麼優化(設計概念)

v2 已經把 graph filter push-down 進 ranker SQL,使 ranker 不必排整個 corpus,只在 graph 鄰域內排。**v3 把這個思路再推一步:對「同時用 BM25 + pgvector」的查詢(Q6 / Q7),把 BM25 命中也當成一道 filter,push-down 進 pgvector**,讓 pgvector 看到的 candidate 集再縮小一層。

```
Q6 (sem ∩ lex)            Q7 (sem ∩ lex ∩ gph)
─────────────────         ─────────────────────────
BM25 top-N        →       BFS                 →
(用作 pgvector              (用作 BM25 filter)
 filter)                   BM25 within S_g top-N  →
pgvector top-K            pgvector within (S_g ∩ S_l) top-K
within BM25-set
RRF(vector, bm25)         RRF(vector, bm25)
```

**為什麼 RRF 還要 BM25**?單做兩道 push-down,最後只回 pgvector top-K 就退化成「BM25 prefilter + 純 vector search」——BM25 的排名訊號完全沒用上。把 BM25 rank 餵回 RRF,讓「BM25 排前面 + pgvector 也排前面」的論文得分疊加,保留 BM25 對排名的影響力。

**為什麼 graph 不進 RRF**?從 naive → v2 的差異已能看到答案:naive 三引擎都當 ranker、graph 只當後置過濾;v2 graph **只當前置過濾**(push-down filter,不 rank)、結果 NDCG 從 0.675 升到 0.801(+0.126)。把 graph 從 ranking 階段抽掉是 v2 的關鍵勝筆;v3 沿用——graph 永遠不是 ranking signal。

## 3. 整體平均 (20 query · samples=10)

| plan  | mean P50 (ms) | mean NDCG@10 | Jaccard@10 vs v2 | RBO@10 vs v2 |
| ----- | ------------- | ------------ | ---------------- | ------------ |
| v2    | 24.65         | **0.801**    | —                | —            |
| v3    | **18.15**     | 0.650        | 0.664            | 0.704        |

- **v3 P50 比 v2 快 1.36×**(18.1 vs 24.6 ms)。
- **v3 NDCG 比 v2 低 0.151**——以下逐 query 看出規律:Q4/Q5 完全相同,Q6 大幅退步,Q7 小幅退步。
- **v3 vs v2 Jaccard 0.66 / RBO 0.70**:Q4/Q5 完全重合(這兩類 v3 = v2),只有 Q6/Q7 有 ranking 差異。

## 4. 各查詢類型(mean P50 / mean NDCG@10)

| 類型 | v2 P50 / NDCG | v3 P50 / NDCG | P50 加速 (v2/v3) | NDCG 差 (v3−v2) |
| ---- | ------------- | ------------- | ---------------- | ---------------- |
| Q4 (sem ∩ gph)     | 2.8 / 0.917  | **2.7 / 0.917**  | 1.04× | **+0.000** (identical, delegate to v2) |
| Q5 (lex ∩ gph)     | 21.0 / 0.742 | **22.1 / 0.742** | 0.95× (噪音)  | **+0.000** (identical, delegate to v2) |
| Q6 (sem ∩ lex)     | 46.0 / 0.618 | **19.9 / 0.201** | **2.31×** | **−0.417 ⚠** |
| Q7 (sem ∩ lex ∩ gph) | 28.7 / 0.925 | **27.9 / 0.740** | 1.03× | −0.185 ⚠ |

註:Q6 是 v3 唯一能拿到的大幅 P50 加速來源——v2 在 Q6 沒有 graph 可 push-down,所以兩 ranker 各自跑完整 corpus 排 top-N;v3 用「BM25 命中 → pgvector」chain 把 pgvector 的搜尋空間從 5 000+ 縮到 50。

## 5. 逐 query 表

| qid  | v2 P50 / NDCG | v3 P50 / NDCG | v3/v2 P50 | ΔNDCG | 備註 |
| ---- | ------------- | ------------- | --------- | ------ | ---- |
| Q4-1 | 1.7 / 1.000  | 1.5 / 1.000   | 0.88×    | +0.000 | v3=v2 (delegate) |
| Q4-2 | 5.6 / 0.861  | 5.6 / 0.861   | 1.00×    | +0.000 | v3=v2 (delegate) |
| Q4-3 | 2.2 / 1.000  | 1.9 / 1.000   | 0.86×    | +0.000 | v3=v2 (delegate) |
| Q4-4 | 1.3 / 0.788  | 1.3 / 0.788   | 1.00×    | +0.000 | v3=v2 (delegate) |
| Q4-5 | 3.1 / 0.934  | 3.1 / 0.934   | 1.00×    | +0.000 | v3=v2 (delegate) |
| Q5-1 | 18.1 / 0.637 | 18.7 / 0.637  | 1.03×    | +0.000 | v3=v2 (delegate) |
| Q5-2 | 19.4 / 0.855 | 22.0 / 0.855  | 1.13×    | +0.000 | v3=v2 (delegate) |
| Q5-3 | 26.1 / 0.861 | 26.1 / 0.861  | 1.00×    | +0.000 | v3=v2 (delegate) |
| Q5-4 | 19.4 / 0.934 | 21.6 / 0.934  | 1.11×    | +0.000 | v3=v2 (delegate) |
| Q5-5 | 22.2 / 0.425 | 22.1 / 0.425  | 1.00×    | +0.000 | v3=v2 (delegate) |
| Q6-1 | 46.3 / 0.779 | **23.2 / 0.482** | **0.50×** | −0.297 ⚠ | chained: −20.3 ms,NDCG 跌 |
| Q6-2 | 46.1 / 0.849 | **19.0 / 0.359** | **0.41×** | −0.490 ⚠ | chained: −27.1 ms,NDCG 大幅跌 |
| Q6-3 | 46.2 / 0.396 | **19.2 / 0.095** | **0.42×** | −0.302 ⚠ | chained: −27.0 ms |
| Q6-4 | 48.1 / 0.526 | **19.1 / 0.000** | **0.40×** | −0.526 ⚠ | chained: −29.0 ms,NDCG 0 |
| Q6-5 | 43.6 / 0.542 | **18.8 / 0.069** | **0.43×** | −0.473 ⚠ | chained: −24.8 ms |
| Q7-1 | 21.1 / 0.927 | 20.1 / 0.542  | 0.95×    | −0.384 ⚠ | chained: 持平,NDCG 跌 |
| Q7-2 | 45.4 / 0.905 | **41.6 / 0.775** | **0.92×** | −0.130 ⚠ | chained: −3.8 ms |
| Q7-3 | 21.6 / 0.934 | 23.8 / 0.542  | 1.10×    | −0.392 ⚠ | chained: +2.2 ms,NDCG 跌 |
| Q7-4 | 19.6 / 1.000 | 19.5 / 0.936  | 1.00×    | −0.064 | chained: 持平 |
| Q7-5 | 36.0 / 0.861 | **34.7 / 0.905** | **0.96×** | **+0.044** ✓ | 唯一 v3 NDCG 贏 v2 的 cell |

(粗體 = v3 顯著贏 v2;⚠ = ΔNDCG < −0.1;✓ = v3 NDCG 領先 v2)

## 6. v3 為何在 Q6 換到大量 P50 但 NDCG 暴跌(disclaim)

Q6 是 v3 設計的「機制 vs 品質」最尖銳的權衡點:

- **機制(P50 端)**:v2 對 Q6 沒有 push-down 對象,pgvector 跟 BM25 各自在 50K corpus 排 top-50,latency ~46 ms。v3 chain:BM25 top-50 → pgvector WHERE id ∈ 那 50 個,pgvector 只看 50 個候選,latency ~19 ms(2.3× 加速)。**這是 v3 的核心 P50 賣點**。
- **品質(NDCG 端)**:v2 的 RRF 看到 pgvector top-50(全 corpus 最 *語意* 接近)與 BM25 top-50(全 corpus 最 *詞彙* 匹配)的合集,只要任一引擎排前面就有機會進 top-10。v3 的 pgvector 只能在「BM25 命中的 50 篇」內排——**語意上接近但 abstract 沒寫該關鍵詞的論文,從一開始就被 BM25 砍掉**,pgvector 看不到。

最極端的例子是 Q6-4「Meltdown 相關 + cache timing attack」:相關論文很多是 Spectre / 旁通道 / 微架構漏洞主題的論文,**但 abstract 不一定寫完整片語 "cache timing attack"**——v3 直接漏抓,NDCG = 0.000。v2 因為 pgvector 不受 BM25 約束、能憑語意把這些論文拉進 top-50,RRF 後排到前 10,NDCG = 0.526。

**這不是 bug,是設計後果**:v3 用 BM25 top-N 當 pgvector 的硬過濾,等於假設「相關論文必出現在 BM25 top-N 內」。本 corpus 上這個假設對 Q6 的 5 道題都不夠強——只要查詢詞是「組合短語」(machine translation / batch normalization / consensus paxos raft / cache timing attack / fault tolerance replication),v3 都會 miss。

## 7. Q7 為何 NDCG 降但沒像 Q6 那麼慘

Q7 = Q6 + graph filter。graph push-down 已經把 candidate 從 corpus 全域縮到幾百個,所以 BM25 top-N 對 pgvector 的二次縮小空間有限,而 graph 篩出來的論文本來就跟 anchor 主題相近(citation 局部社群),BM25 的詞彙約束沒像 Q6 那樣切掉太多語意候選。

- Q7-4 / Q7-5:NDCG 與 v2 接近(0.936 / 0.905 vs 1.000 / 0.861),P50 略快
- Q7-1 / Q7-3:NDCG 跌 0.4(BM25 top-50 把幾個關鍵候選排到 50 名外),P50 持平或略慢
- Q7-2:NDCG 從 0.905 → 0.775(−0.130),P50 縮 4 ms

## 8. v3 適用場景

- **查詢有確切關鍵詞**(BM25 排名集中、top-N 高覆蓋):Q4-* 全勝、Q7-4 / Q7-5 維持高 NDCG,P50 比 v2 略快。
- **查詢是「同類主題抓取」**(語意鄰近 > 詞彙匹配):v3 不建議用——Q6-2~Q6-5 的 NDCG 都崩。建議在這類查詢上 fallback 回 v2。

未來 work:用 BM25 命中數(matched count)作為 dispatch criterion——matched > 某閾值才走 v3 chain(BM25 很有把握的查詢),否則 fallback 到 v2。本 v3 暫不做這個 routing,**讓 v3 在所有 Q6/Q7 上都跑 chained**,以便完整暴露這個 tradeoff。

## 9. Cold / Warm(`reports/coldwarm_v3.json`,28 cell)

選錄(完整 28 格在 JSON):

| query | plan | cold (ms) | warm (ms) | cold/warm |
| ----- | ---- | --------- | --------- | --------- |
| Q4    | v2   | 12.6      | 1.6       | 8.07×     |
| Q4    | v3   | 12.1      | **1.4**   | 8.79×     |
| Q5    | v2   | 35.3      | 18.2      | 1.94×     |
| Q5    | v3   | 33.7      | 18.9      | 1.78×     |
| **Q6** | **v2** | **81.2** | **43.1**  | 1.88×    |
| **Q6** | **v3** | **35.1** | **21.9**  | 1.60×    |
| Q7    | v2   | 36.9      | 20.6      | 1.79×     |
| Q7    | v3   | 37.9      | 20.0      | 1.90×     |

Q6 cold:v2 81 ms → v3 35 ms(2.3× 加速),warm 43 ms → 22 ms(2.0×)。Q1/Q2/Q3 / Q4 / Q5 / Q7 v3 與 v2 在量測雜訊內。

## 10. Done condition

- [x] v3 是 v2 的 chained push-down 優化版,描述不再提 naive / v1
- [x] Q4 / Q5 delegate 到 v2 path,NDCG bit-identical 已驗證
- [x] Q6 / Q7 走 BM25 → pgvector chain,RRF 只融合 vector + bm25 兩條訊號
- [x] graph 不在 RRF(設計理由見 §2;v2 已示範 graph-only-filter 比 graph-as-ranker 好 +0.126 NDCG)
- [x] mean P50 < v2(18.1 ms < 24.6 ms,1.36× 加速)
- [x] NDCG 退步在 Q6/Q7 上誠實標出,§6 / §7 disclaim 完整
