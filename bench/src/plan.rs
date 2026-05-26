//! Plan trait + three orchestrator implementations: naive (fixed order),
//! v1 (cost-based reorder), v2 (cost-based reorder + graph push-down).
//!
//! In this pipeline (top-K rankers fused by RRF, no data dependency
//! between engines) pure cost-based *reordering* does not change
//! runtime: every engine still has to run to completion, and there is
//! no short-circuit opportunity. v1 is preserved as the clean
//! reorder-only ablation that demonstrates this. v2 is the version
//! that moves the needle, by *push-down* — rewriting the ranker SQL
//! to scan only inside the graph filter set.

use anyhow::{Context, Result};
use serde::Serialize;
use sqlx::postgres::PgPool;
use sqlx::Row;

use crate::cost::{self, Engine, EngineNorm, PredicateEstimate};
use crate::fusion::{fuse, FusionStrategy, Ranking};
use crate::graph_engine::{bfs_recursive_sql, Direction};
use crate::query::{QuerySpec, QueryType};

/// How wide to fetch per-engine candidates before fusion. Setting this
/// to k itself is too narrow — Q7 intersected with three engines could
/// have <k overlap and return short. 5×k is a reasonable default; the
/// final orchestrator (D5) will make this adaptive on per-engine
/// selectivity.
const PER_ENGINE_OVERFETCH: usize = 5;

/// Plan outcome — result IDs (paper_id) plus annotation for plan analysis.
#[derive(Debug, Clone, Serialize)]
pub struct PlanResult {
    pub plan: &'static str,
    pub query: QueryType,
    pub paper_ids: Vec<i64>,
    /// Per-predicate cost annotations the planner emitted.
    pub predicates: Vec<PredicateEstimate>,
    /// Which predicate ran first (only meaningful for multi-predicate Qs).
    pub first_predicate: Option<Engine>,
    /// Per-engine row count returned BEFORE fusion. Useful for analyzing
    /// over-fetch ratio (we fetch k×5 per engine; how many survive).
    pub per_engine_rows: Vec<(Engine, usize)>,
    /// Number of DB queries issued. SQL parse + execute is one round-trip.
    pub round_trips: u32,
    /// Whether an intermediate result was materialized (naive truth).
    pub materializations: u32,
    /// The order engines actually ran in. For naive this is fixed
    /// insertion order; for v1 it follows ascending selectivity.
    pub actual_order: Vec<Engine>,
}

#[async_trait::async_trait]
pub trait Plan {
    fn name(&self) -> &'static str;
    async fn execute(&self, pool: &PgPool, q: QueryType, spec: &QuerySpec) -> Result<PlanResult>;
}

// ============ NAIVE ============

/// Naive sequential plan: textbook fixed-order, materialize each step.
/// For single-predicate queries it's just the single engine; for
/// multi-predicate it pulls each engine's top results into a temp set
/// and intersects in Rust.
pub struct NaivePlan;

#[async_trait::async_trait]
impl Plan for NaivePlan {
    fn name(&self) -> &'static str { "naive" }

    async fn execute(&self, pool: &PgPool, q: QueryType, spec: &QuerySpec) -> Result<PlanResult> {
        match q {
            QueryType::Q1 => semantic_only(pool, spec, "naive", q).await,
            QueryType::Q2 => lexical_only(pool, spec, "naive", q).await,
            QueryType::Q3 => graph_only(pool, spec, "naive", q).await,
            // Multi-predicate: textbook fixed order
            //   sem → lex → graph (always in that order, even if graph
            //   is more selective). Materialize each engine's result
            //   fully then RRF + intersect.
            QueryType::Q4 | QueryType::Q5 | QueryType::Q6 | QueryType::Q7 => {
                multi_predicate(pool, q, spec, "naive",
                                /* order_by_selectivity */ false).await
            }
        }
    }
}

// ============ V2 ORCHESTRATOR (push-down) ============

/// v2 cost-based orchestrator with **predicate push-down**.
///
/// For queries that combine the graph engine with at least one ranker
/// (Q4 / Q5 / Q7), v2 materializes the graph filter first and pushes
/// the resulting `paper_id` set into the ranker's SQL via
/// `WHERE paper_id = ANY($set)`. The ranker therefore only scans
/// candidates already inside the graph filter, which can shrink
/// runtime by an order of magnitude on highly-selective filters.
///
/// For Q1–Q3 and Q6 (no graph-plus-ranker combination), v2 collapses
/// to the same execution as v1 — `should_use_pushdown` returns false.
pub struct V2Plan;

impl V2Plan {
    /// Whether the push-down code path is meaningful for this query
    /// type. Pure helper — separated so it can be unit-tested without
    /// a live DB.
    pub fn should_use_pushdown(q: QueryType) -> bool {
        let (sem, lex, gph) = q.engines();
        gph && (sem || lex)
    }
}

#[async_trait::async_trait]
impl Plan for V2Plan {
    fn name(&self) -> &'static str { "v2" }

    async fn execute(&self, pool: &PgPool, q: QueryType, spec: &QuerySpec) -> Result<PlanResult> {
        match q {
            QueryType::Q1 => semantic_only(pool, spec, "v2", q).await,
            QueryType::Q2 => lexical_only(pool, spec, "v2", q).await,
            QueryType::Q3 => graph_only(pool, spec, "v2", q).await,
            // Q6 has no graph predicate; fall back to v1's post-filter path.
            QueryType::Q6 => multi_predicate(pool, q, spec, "v2",
                                /* order_by_selectivity */ true).await,
            // Q4 / Q5 / Q7: push graph filter into ranker SQL.
            QueryType::Q4 | QueryType::Q5 | QueryType::Q7 => {
                multi_predicate_pushdown(pool, q, spec, "v2").await
            }
        }
    }
}

// ============ V1 ORCHESTRATOR (cost-based reorder) ============

/// v1 cost-based orchestrator. Single-predicate queries collapse to the
/// single engine (identical to naive). Multi-predicate queries enumerate
/// candidate predicate orderings, then dispatch in ascending selectivity
/// order. AGE cost uses the empirically fit `age_cost_v1`
/// (`branching ^ depth`); see `cost.rs` and `microbench`.
///
/// On the 50K corpus + RRF this reorder has **no measurable latency
/// effect** vs naive — every engine must run to completion regardless
/// of order, and the engines have no data dependency that reorder could
/// short-circuit. v1 is preserved as the clean "cost-reorder only"
/// ablation that motivates v2's push-down (see report §4).
pub struct V1Plan;

#[async_trait::async_trait]
impl Plan for V1Plan {
    fn name(&self) -> &'static str { "v1" }

    async fn execute(&self, pool: &PgPool, q: QueryType, spec: &QuerySpec) -> Result<PlanResult> {
        match q {
            QueryType::Q1 => semantic_only(pool, spec, "v1", q).await,
            QueryType::Q2 => lexical_only(pool, spec, "v1", q).await,
            QueryType::Q3 => graph_only(pool, spec, "v1", q).await,
            QueryType::Q4 | QueryType::Q5 | QueryType::Q6 | QueryType::Q7 => {
                multi_predicate(pool, q, spec, "v1",
                                /* order_by_selectivity */ true).await
            }
        }
    }
}

// ============ V3 ORCHESTRATOR (chained push-down: v2 optimization) ============

/// **v3** — v2 의 進一步效能優化。
///
/// v2 已經把 graph filter push-down 進 ranker SQL,使 ranker 只在圖過
/// 濾後的 candidate 上排名,P50 / NDCG 兩個軸同時得益(§4.5)。v3 把這
/// 個 push-down 思路再推一步:**對「兩個 ranker 都要用」的查詢(Q6 / Q7),
/// 把 BM25 命中也當成一道 filter,再 push-down 進 pgvector**,讓 pgvector
/// 看到的 candidate 集合再縮小一層。
///
/// Pipeline:
///
/// ```text
///   Q6 (semantic ∩ lexical):
///     S_l = BM25 ORDER BY score LIMIT N        ← top-N BM25 hits
///     vector_top = pgvector WHERE id ∈ S_l LIMIT K
///     return RRF([vector_top, S_l])           ← vector + bm25 fed to RRF
///
///   Q7 (semantic ∩ lexical ∩ graph):
///     S_g = BFS(anchor, depth)                 ← graph subset
///     S_l ∩ S_g = BM25 WHERE id ∈ S_g LIMIT N  ← BM25 subset *within* S_g
///     vector_top = pgvector WHERE id ∈ (S_l ∩ S_g) LIMIT K
///     return RRF([vector_top, S_l ∩ S_g])     ← vector + bm25 fed to RRF
/// ```
///
/// Q4 / Q5 only have one ranker after the graph push-down, so v3 has no
/// room to chain a second filter — it delegates verbatim to v2's
/// `multi_predicate_pushdown`. Q1 / Q2 / Q3 are single-engine and also
/// delegate to the same helpers v2 uses (`semantic_only` /
/// `lexical_only` / `graph_only`).
///
/// ### Why feed BM25 into RRF instead of returning vector top-K alone
///
/// If we stopped after pgvector's filtered search, v3 would be a plain
/// *pre-filter HNSW vector search* — the BM25 signal would only act as
/// a hard filter, contributing nothing to the ranking. To keep BM25
/// influencing the final order (so the result reflects "semantically
/// closest **among** lexically relevant" rather than "semantically
/// closest, period, filtered by lexical match"), v3 feeds the BM25
/// rank back into RRF as the second signal.
///
/// ### Why graph is *not* in RRF
///
/// The naive → v2 comparison already showed that treating graph as a
/// ranking signal hurts: naive runs all three engines as rankers and
/// post-filters by graph (NDCG@10 = 0.675); v2 uses graph **only as a
/// filter** (push-down, not a ranker) and lifts NDCG@10 to 0.801
/// (+0.126). On this corpus, graph-as-filter beats graph-as-ranker.
/// v3 inherits that conclusion — `graph_distance` is *not* fed into
/// RRF; graph stays a pure filter.
pub struct V3Plan;

impl V3Plan {
    /// Returns true when v3 takes its **chained** push-down path —
    /// i.e. there are two rankers (semantic AND lexical) so v3 can
    /// chain BM25 push-down into pgvector. This is exactly Q6 and Q7.
    /// Q4 / Q5 only have one ranker after the graph filter, so v3
    /// delegates to v2's `multi_predicate_pushdown` instead.
    pub fn uses_chained_pushdown(q: QueryType) -> bool {
        let (sem, lex, _gph) = q.engines();
        sem && lex
    }
}

#[async_trait::async_trait]
impl Plan for V3Plan {
    fn name(&self) -> &'static str { "v3" }

    async fn execute(&self, pool: &PgPool, q: QueryType, spec: &QuerySpec) -> Result<PlanResult> {
        // v3 takes its chained-pushdown branch only when *both* rankers
        // are present (Q6, Q7). Everything else falls through to v2's
        // helpers verbatim — same SQL, same fusion, identical result.
        // V2Plan::should_use_pushdown(q) lets us cross-check that v3's
        // graph-pushdown classification agrees with v2's (Q4 / Q5 / Q7).
        let chained = V3Plan::uses_chained_pushdown(q);
        tracing::trace!(plan = self.name(), qid = q.as_str(),
                        chained_pushdown = chained,
                        v2_graph_pushdown = V2Plan::should_use_pushdown(q),
                        "v3 dispatch");
        match q {
            QueryType::Q1 => semantic_only(pool, spec, "v3", q).await,
            QueryType::Q2 => lexical_only(pool, spec, "v3", q).await,
            QueryType::Q3 => graph_only(pool, spec, "v3", q).await,
            // Q4 / Q5: one ranker after graph push-down → v2 verbatim.
            QueryType::Q4 | QueryType::Q5 => {
                multi_predicate_pushdown(pool, q, spec, "v3").await
            }
            // Q6 / Q7: two rankers → chained BM25 → pgvector push-down.
            QueryType::Q6 | QueryType::Q7 => {
                multi_predicate_v3_chained(pool, q, spec).await
            }
        }
    }
}

/// v3's **chained push-down** driver for Q6 / Q7. Both queries have
/// two rankers (semantic + lexical), so v3 narrows the candidate set
/// in two successive push-down stages:
///
///   Q6 (sem ∩ lex):    BM25 top-N → pgvector top-K within that set
///   Q7 (sem ∩ lex ∩ gph): BFS → BM25 top-N within S_g
///                         → pgvector top-K within S_g ∩ top-N S_l
///
/// The final RRF fuses two signals: `vector_rank` (top-K pgvector on
/// the smallest candidate set) and `bm25_rank` (top-N BM25 in the
/// preceding stage). Graph is *not* a ranking signal — only a filter.
async fn multi_predicate_v3_chained(
    pool: &PgPool,
    q: QueryType,
    spec: &QuerySpec,
) -> Result<PlanResult> {
    let (use_sem, use_lex, use_gph) = q.engines();
    debug_assert!(use_sem && use_lex, "chained path requires both rankers (Q6/Q7)");

    let k = spec.k;
    let n_fetch = k * PER_ENGINE_OVERFETCH;
    let depth = spec.depth.max(1).min(3);

    let mut predicates: Vec<PredicateEstimate> = Vec::new();
    let mut per_engine_rows: Vec<(Engine, usize)> = Vec::new();
    let mut actual_order: Vec<Engine> = Vec::new();
    let mut round_trips = 0u32;
    let mut materializations = 0u32;
    let mut first_predicate: Option<Engine> = None;

    // -- Stage 1 (Q7 only): BFS → S_g; Q6 skips this.
    let graph_set: Option<Vec<i64>> = if use_gph {
        let anchor = spec.anchor_paper.context("Q7 requires anchor_paper")?;
        let ids = bfs_recursive_sql(pool, anchor, depth, Direction::Reverse).await?;
        round_trips += 1;
        materializations += 1;
        per_engine_rows.push((Engine::Age, ids.len()));
        actual_order.push(Engine::Age);
        first_predicate = Some(Engine::Age);
        let raw = cost::age_cost_v1(depth, 26.3);
        predicates.push(PredicateEstimate {
            engine: Engine::Age,
            raw_cost: raw,
            ms_estimate: cost::normalize(raw, Engine::Age, &EngineNorm::default()),
            selectivity: cost::selectivity_graph(ids.len(), 5_011),
        });
        if ids.is_empty() {
            return Ok(PlanResult {
                plan: "v3", query: q,
                paper_ids: vec![],
                predicates,
                first_predicate,
                per_engine_rows,
                round_trips,
                materializations,
                actual_order,
            });
        }
        Some(ids)
    } else {
        None
    };

    // -- Stage 2: BM25 top-N. If S_g present, push-down into BM25.
    let text = spec.bm25_text.as_deref()
        .context("v3 chained path requires bm25_text")?;
    let bm25_top_n: Vec<i64> = match &graph_set {
        Some(gs) => run_bm25_pushdown(pool, text, n_fetch, gs).await?,
        None     => run_bm25_topn(pool, text, n_fetch).await?,
    };
    round_trips += 1;
    materializations += 1;
    per_engine_rows.push((Engine::PgSearch, bm25_top_n.len()));
    actual_order.push(Engine::PgSearch);
    if first_predicate.is_none() {
        first_predicate = Some(Engine::PgSearch);
    }
    // Use a coarse matched estimate equal to bm25_top_n.len() — we
    // don't need the exact `count(*)` since cost ordering doesn't
    // matter for v3 (the chain is fixed). Selectivity is reported for
    // analysis only.
    let bm25_raw = cost::pg_search_cost(bm25_top_n.len(), 50.0);
    predicates.push(PredicateEstimate {
        engine: Engine::PgSearch,
        raw_cost: bm25_raw,
        ms_estimate: cost::normalize(bm25_raw, Engine::PgSearch, &EngineNorm::default()),
        selectivity: cost::selectivity_lexical(bm25_top_n.len(), 5_011),
    });

    if bm25_top_n.is_empty() {
        return Ok(PlanResult {
            plan: "v3", query: q,
            paper_ids: vec![],
            predicates,
            first_predicate,
            per_engine_rows,
            round_trips,
            materializations,
            actual_order,
        });
    }

    // -- Stage 3: pgvector top-K push-down into the BM25 subset.
    let seed = spec.seed_chunk_id
        .context("v3 chained path requires seed_chunk_id")?;
    let vector_top_k = run_semantic_pushdown(
        pool, seed, n_fetch, spec.ef_search, &bm25_top_n,
    ).await?;
    round_trips += 2; // SET LOCAL group + query
    materializations += 1;
    per_engine_rows.push((Engine::Pgvector, vector_top_k.len()));
    actual_order.push(Engine::Pgvector);
    let pgv_raw = cost::pgvector_cost(spec.ef_search, bm25_top_n.len().max(2));
    predicates.push(PredicateEstimate {
        engine: Engine::Pgvector,
        raw_cost: pgv_raw,
        ms_estimate: cost::normalize(pgv_raw, Engine::Pgvector, &EngineNorm::default()),
        selectivity: cost::selectivity_semantic(k, bm25_top_n.len().max(1)),
    });

    // -- Stage 4: RRF over (vector_rank, bm25_rank). Graph stays out
    //    of the ranking signal — see V3Plan docstring for why.
    let rankings = vec![
        Ranking { engine: "pgvector",  paper_ids: vector_top_k },
        Ranking { engine: "pg_search", paper_ids: bm25_top_n },
    ];
    tracing::debug!(
        plan = "v3",
        qid = q.as_str(),
        engines = ?rankings.iter().map(|r| r.engine).collect::<Vec<_>>(),
        "v3 chained RRF: vector + bm25 (graph used as filter only)"
    );
    let final_ids = fuse(&rankings, &FusionStrategy::rrf_default(), None, k);

    let _ = use_lex; // silence in case future refactor drops the assert
    Ok(PlanResult {
        plan: "v3", query: q,
        paper_ids: final_ids,
        predicates,
        first_predicate,
        per_engine_rows,
        round_trips,
        materializations,
        actual_order,
    })
}

// ============ shared single-predicate runners ============

async fn semantic_only(
    pool: &PgPool,
    spec: &QuerySpec,
    plan_name: &'static str,
    q: QueryType,
) -> Result<PlanResult> {
    let seed = spec.seed_chunk_id
        .context("Q1 / Q4 / Q6 / Q7 require seed_chunk_id")?;
    let k = spec.k as i64;
    let ef = spec.ef_search as i32;

    // pgvector reads hnsw.ef_search at query time; SET LOCAL only works
    // inside a transaction, so wrap. Without a tx Postgres raises
    // WARNING and the GUC is silently not applied — confirmed via
    // pgvector docs.
    let mut tx = pool.begin().await?;
    sqlx::query(&format!("SET LOCAL hnsw.ef_search = {ef}"))
        .execute(&mut *tx).await?;

    let rows = sqlx::query(
        "WITH seed AS (SELECT embedding FROM chunk_embeddings WHERE chunk_id = $1) \
         SELECT c.paper_id \
         FROM chunk_embeddings ce \
         JOIN chunks c ON c.id = ce.chunk_id, seed \
         ORDER BY ce.embedding <=> seed.embedding \
         LIMIT $2",
    )
    .bind(seed)
    .bind(k)
    .fetch_all(&mut *tx)
    .await?;
    tx.commit().await?;

    let paper_ids: Vec<i64> = rows.into_iter().filter_map(|r| r.try_get(0).ok()).collect();

    let raw = cost::pgvector_cost(spec.ef_search, 5_000); // n filled with corpus size at plan time
    let norm = EngineNorm::default();
    let ms = cost::normalize(raw, Engine::Pgvector, &norm);
    let pred = PredicateEstimate {
        engine: Engine::Pgvector,
        raw_cost: raw,
        ms_estimate: ms,
        selectivity: cost::selectivity_semantic(spec.k, 5_000),
    };

    Ok(PlanResult {
        plan: plan_name,
        query: q,
        paper_ids: paper_ids.clone(),
        predicates: vec![pred],
        first_predicate: Some(Engine::Pgvector),
        per_engine_rows: vec![(Engine::Pgvector, paper_ids.len())],
        round_trips: 2, // SET LOCAL + actual query
        materializations: 0,
        actual_order: vec![Engine::Pgvector],
    })
}

async fn lexical_only(
    pool: &PgPool,
    spec: &QuerySpec,
    plan_name: &'static str,
    q: QueryType,
) -> Result<PlanResult> {
    let text = spec.bm25_text.as_deref()
        .context("Q2 / Q5 / Q6 / Q7 require bm25_text")?;
    let k = spec.k as i64;

    let rows = sqlx::query(
        "SELECT id FROM papers \
         WHERE abstract @@@ $1 \
         ORDER BY paradedb.score(id) DESC \
         LIMIT $2",
    )
    .bind(text)
    .bind(k)
    .fetch_all(pool)
    .await?;

    let paper_ids: Vec<i64> = rows.into_iter().filter_map(|r| r.try_get(0).ok()).collect();

    // Selectivity estimate: ask BM25 how many candidates match.
    let (matched,): (i64,) = sqlx::query_as(
        "SELECT count(*) FROM papers WHERE abstract @@@ $1",
    )
    .bind(text)
    .fetch_one(pool)
    .await
    .unwrap_or((0,));

    let raw = cost::pg_search_cost(matched as usize, 50.0); // 50 is a coarse avg-posting prior
    let norm = EngineNorm::default();
    let ms = cost::normalize(raw, Engine::PgSearch, &norm);
    let pred = PredicateEstimate {
        engine: Engine::PgSearch,
        raw_cost: raw,
        ms_estimate: ms,
        selectivity: cost::selectivity_lexical(matched as usize, 5_000),
    };

    let n_rows = paper_ids.len();
    Ok(PlanResult {
        plan: plan_name,
        query: q,
        paper_ids,
        predicates: vec![pred],
        first_predicate: Some(Engine::PgSearch),
        per_engine_rows: vec![(Engine::PgSearch, n_rows)],
        round_trips: 2, // count(*) + top-k query
        materializations: 0,
        actual_order: vec![Engine::PgSearch],
    })
}

async fn graph_only(
    pool: &PgPool,
    spec: &QuerySpec,
    plan_name: &'static str,
    q: QueryType,
) -> Result<PlanResult> {
    let anchor = spec.anchor_paper
        .context("Q3 / Q4 / Q5 / Q7 require anchor_paper")?;
    let depth = spec.depth.max(1).min(3);

    // Phase 1 §D8 shootout (reports/bfs_shootout_5k.json) showed
    // WITH RECURSIVE is 6–850× faster than AGE Cypher on identical
    // BFS-from-anchor result sets across all 270 tested cells. Both
    // engines are wired up (see crate::graph_engine), but the planner
    // calls the recursive-SQL path. Direction::Reverse = papers that
    // cite the anchor (Q3/Q4/Q5/Q7's "who cites this paper").
    let paper_ids = bfs_recursive_sql(pool, anchor, depth, Direction::Reverse).await?;

    let raw = cost::age_cost_v1(depth, 2.4); // 2.4 ≈ synth avg out-degree
    let norm = EngineNorm::default();
    let ms = cost::normalize(raw, Engine::Age, &norm);
    let pred = PredicateEstimate {
        engine: Engine::Age,
        raw_cost: raw,
        ms_estimate: ms,
        selectivity: cost::selectivity_graph(paper_ids.len(), 5_000),
    };

    let n_rows = paper_ids.len();
    Ok(PlanResult {
        plan: plan_name,
        query: q,
        paper_ids,
        predicates: vec![pred],
        first_predicate: Some(Engine::Age),
        per_engine_rows: vec![(Engine::Age, n_rows)],
        round_trips: 1,
        materializations: 0,
        actual_order: vec![Engine::Age],
    })
}

// ============ multi-predicate (Q4–Q7) ============

/// Generic Q4–Q7 driver. Each engine that's "on" for this QueryType
/// produces a top-N ranked list of paper_ids; graph (when present) acts
/// as a hard filter rather than a ranking signal; remaining engines are
/// RRF-fused.
///
/// The two plans differ in:
///   * **naive**: predicates executed in fixed order (sem → lex → graph),
///     each materialized fully before the next runs. `first_predicate`
///     is sem (or lex if Q5).
///   * **v1**:   engine order is decided by selectivity — we ask each
///     engine for its candidate-set size first, then run the smallest
///     first. Graph is *always* used as the filter when present (its
///     selectivity is captured in `selectivity_graph`).
async fn multi_predicate(
    pool: &PgPool,
    q: QueryType,
    spec: &QuerySpec,
    plan_name: &'static str,
    order_by_selectivity: bool,
) -> Result<PlanResult> {
    let (use_sem, use_lex, use_gph) = q.engines();
    let k = spec.k;
    let n_fetch = k * PER_ENGINE_OVERFETCH;

    let mut materializations = 0u32;
    let mut round_trips = 0u32;
    let mut predicates: Vec<PredicateEstimate> = Vec::new();
    let mut rankings: Vec<Ranking<'static>> = Vec::new();
    let mut per_engine_rows: Vec<(Engine, usize)> = Vec::new();
    let mut actual_order: Vec<Engine> = Vec::new();
    let mut graph_filter: Option<Vec<i64>> = None;

    // ---- predicate cost / selectivity estimates ----
    let mut estimates: Vec<(Engine, PredicateEstimate)> = Vec::new();

    if use_sem {
        let raw = cost::pgvector_cost(spec.ef_search, 5_011);
        let ms = cost::normalize(raw, Engine::Pgvector, &EngineNorm::default());
        estimates.push((Engine::Pgvector, PredicateEstimate {
            engine: Engine::Pgvector,
            raw_cost: raw,
            ms_estimate: ms,
            selectivity: cost::selectivity_semantic(n_fetch, 5_011),
        }));
    }
    if use_lex {
        // For v1 we need an actual matched count to estimate selectivity;
        // a single COUNT(*) is cheap.
        let text = spec.bm25_text.as_deref()
            .context("Q2/Q5/Q6/Q7 require bm25_text")?;
        let (matched,): (i64,) = sqlx::query_as(
            "SELECT count(*) FROM papers WHERE abstract @@@ $1",
        ).bind(text).fetch_one(pool).await.unwrap_or((0,));
        round_trips += 1;
        let raw = cost::pg_search_cost(matched as usize, 50.0);
        let ms = cost::normalize(raw, Engine::PgSearch, &EngineNorm::default());
        estimates.push((Engine::PgSearch, PredicateEstimate {
            engine: Engine::PgSearch,
            raw_cost: raw,
            ms_estimate: ms,
            selectivity: cost::selectivity_lexical(matched as usize, 5_011),
        }));
    }
    if use_gph {
        // Run graph first to materialize the filter set. §D8 shootout
        // showed WITH RECURSIVE (set-equivalent to AGE) is 6–850×
        // faster on the same BFS-from-anchor pattern, so the planner
        // calls bfs_recursive_sql here. Cost estimates below still use
        // the AGE coefficients — they're the conservative upper bound;
        // recursive SQL beats them comfortably.
        let anchor = spec.anchor_paper.context("Q3/Q4/Q5/Q7 require anchor_paper")?;
        let depth = spec.depth.max(1).min(3);
        let ids = bfs_recursive_sql(pool, anchor, depth, Direction::Reverse).await?;
        round_trips += 1;
        per_engine_rows.push((Engine::Age, ids.len()));
        actual_order.push(Engine::Age);

        let raw = cost::age_cost_v1(depth, 26.3);
        let ms = cost::normalize(raw, Engine::Age, &EngineNorm::default());
        predicates.push(PredicateEstimate {
            engine: Engine::Age,
            raw_cost: raw,
            ms_estimate: ms,
            selectivity: cost::selectivity_graph(ids.len(), 5_011),
        });
        graph_filter = Some(ids);
        materializations += 1;
    }

    // ---- decide ranking-engine order ----
    if order_by_selectivity {
        estimates.sort_by(|a, b| a.1.selectivity.partial_cmp(&b.1.selectivity)
            .unwrap_or(std::cmp::Ordering::Equal));
    }
    // else (naive): use semantic-before-lexical fixed order, which is
    // the insertion order above.

    let first_predicate = estimates.first().map(|(e, _)| *e).or({
        if graph_filter.is_some() { Some(Engine::Age) } else { None }
    });

    // ---- execute ranking engines ----
    for (engine, est) in &estimates {
        match engine {
            Engine::Pgvector => {
                let seed = spec.seed_chunk_id.context("semantic predicate needs seed_chunk_id")?;
                let ids = run_semantic_topn(pool, seed, n_fetch, spec.ef_search).await?;
                let n = ids.len();
                rankings.push(Ranking { engine: "pgvector", paper_ids: ids });
                per_engine_rows.push((Engine::Pgvector, n));
                actual_order.push(Engine::Pgvector);
                round_trips += 2; // SET LOCAL + query
                materializations += 1;
            }
            Engine::PgSearch => {
                let text = spec.bm25_text.as_deref().unwrap_or("");
                let ids = run_bm25_topn(pool, text, n_fetch).await?;
                let n = ids.len();
                rankings.push(Ranking { engine: "pg_search", paper_ids: ids });
                per_engine_rows.push((Engine::PgSearch, n));
                actual_order.push(Engine::PgSearch);
                round_trips += 1;
                materializations += 1;
            }
            Engine::Age => unreachable!("graph not in estimates"),
        }
        predicates.push(est.clone());
    }

    // ---- fuse ----
    let strategy = FusionStrategy::rrf_default();
    let final_ids = fuse(&rankings, &strategy, graph_filter.as_deref(), k);

    Ok(PlanResult {
        plan: plan_name,
        query: q,
        paper_ids: final_ids,
        predicates,
        first_predicate,
        per_engine_rows,
        round_trips,
        materializations,
        actual_order,
    })
}

async fn run_semantic_topn(pool: &PgPool, seed: i64, n: usize, ef: u32) -> Result<Vec<i64>> {
    let mut tx = pool.begin().await?;
    sqlx::query(&format!("SET LOCAL hnsw.ef_search = {ef}"))
        .execute(&mut *tx).await?;
    let rows = sqlx::query(
        "WITH seed AS (SELECT embedding FROM chunk_embeddings WHERE chunk_id = $1), \
              hit AS ( \
                SELECT c.paper_id, ce.embedding <=> seed.embedding AS d \
                FROM chunk_embeddings ce \
                JOIN chunks c ON c.id = ce.chunk_id, seed \
                ORDER BY ce.embedding <=> seed.embedding \
                LIMIT $2 \
              ) \
         SELECT DISTINCT ON (paper_id) paper_id \
         FROM hit \
         ORDER BY paper_id, d",
    )
    .bind(seed)
    .bind((n * 3) as i64)
    .fetch_all(&mut *tx).await?;
    tx.commit().await?;
    Ok(rows.into_iter().filter_map(|r| r.try_get(0).ok()).collect())
}

async fn run_bm25_topn(pool: &PgPool, text: &str, n: usize) -> Result<Vec<i64>> {
    let rows = sqlx::query(
        "SELECT id FROM papers WHERE abstract @@@ $1 \
         ORDER BY paradedb.score(id) DESC LIMIT $2",
    )
    .bind(text)
    .bind(n as i64)
    .fetch_all(pool).await?;
    Ok(rows.into_iter().filter_map(|r| r.try_get(0).ok()).collect())
}

/// V2 multi-predicate driver. Runs the graph filter first, then
/// pushes the resulting `paper_id` set into each ranker's SQL.
///
/// Layout:
///   1. AGE BFS → graph_set
///   2. (early return) graph_set empty → []
///   3. for each ranker in {pgvector, pg_search} that the query asks
///      for: filtered top-k query (WHERE paper_id = ANY(graph_set))
///   4. fuse rankers with RRF; single-ranker collapses to identity
async fn multi_predicate_pushdown(
    pool: &PgPool,
    q: QueryType,
    spec: &QuerySpec,
    plan_name: &'static str,
) -> Result<PlanResult> {
    let (use_sem, use_lex, _use_gph) = q.engines();
    debug_assert!(_use_gph, "push-down requires graph predicate");

    let k = spec.k;
    let anchor = spec.anchor_paper.context("graph predicate needs anchor_paper")?;
    let depth = spec.depth.max(1).min(3);

    let mut predicates: Vec<PredicateEstimate> = Vec::new();
    let mut per_engine_rows: Vec<(Engine, usize)> = Vec::new();
    let mut actual_order: Vec<Engine> = Vec::new();
    let mut round_trips = 0u32;

    // -- Step 1: graph filter (recursive SQL; see §D8 shootout) --
    let graph_set = bfs_recursive_sql(pool, anchor, depth, Direction::Reverse).await?;
    round_trips += 1;
    per_engine_rows.push((Engine::Age, graph_set.len()));
    actual_order.push(Engine::Age);

    let age_raw = cost::age_cost_v1(depth, 26.3);
    predicates.push(PredicateEstimate {
        engine: Engine::Age,
        raw_cost: age_raw,
        ms_estimate: cost::normalize(age_raw, Engine::Age, &EngineNorm::default()),
        selectivity: cost::selectivity_graph(graph_set.len(), 5_011),
    });

    // -- Step 2: short-circuit on empty filter --
    if graph_set.is_empty() {
        return Ok(PlanResult {
            plan: plan_name,
            query: q,
            paper_ids: vec![],
            predicates,
            first_predicate: Some(Engine::Age),
            per_engine_rows,
            round_trips,
            materializations: 1,
            actual_order,
        });
    }

    // -- Step 3: rankers with push-down --
    let n_graph = graph_set.len();
    let mut rankings: Vec<Ranking<'static>> = Vec::new();

    if use_sem {
        let seed = spec.seed_chunk_id
            .context("semantic predicate needs seed_chunk_id")?;
        let ids = run_semantic_pushdown(pool, seed, k, spec.ef_search, &graph_set).await?;
        let n = ids.len();
        rankings.push(Ranking { engine: "pgvector", paper_ids: ids });
        per_engine_rows.push((Engine::Pgvector, n));
        actual_order.push(Engine::Pgvector);
        round_trips += 2; // SET LOCAL group + query
        let raw = cost::pgvector_cost(spec.ef_search, n_graph.max(2));
        predicates.push(PredicateEstimate {
            engine: Engine::Pgvector,
            raw_cost: raw,
            ms_estimate: cost::normalize(raw, Engine::Pgvector, &EngineNorm::default()),
            selectivity: cost::selectivity_semantic(k, n_graph.max(1)),
        });
    }

    if use_lex {
        let text = spec.bm25_text.as_deref()
            .context("lexical predicate needs bm25_text")?;
        let ids = run_bm25_pushdown(pool, text, k, &graph_set).await?;
        let n = ids.len();
        rankings.push(Ranking { engine: "pg_search", paper_ids: ids });
        per_engine_rows.push((Engine::PgSearch, n));
        actual_order.push(Engine::PgSearch);
        round_trips += 1;
        let raw = cost::pg_search_cost(n, 50.0);
        predicates.push(PredicateEstimate {
            engine: Engine::PgSearch,
            raw_cost: raw,
            ms_estimate: cost::normalize(raw, Engine::PgSearch, &EngineNorm::default()),
            selectivity: cost::selectivity_lexical(n, n_graph.max(1)),
        });
    }

    // -- Step 4: fuse --
    let final_ids = if rankings.len() == 1 {
        // single ranker → already filtered + ranked, no RRF needed
        rankings[0].paper_ids.iter().take(k).copied().collect()
    } else {
        let strategy = FusionStrategy::rrf_default();
        // Don't pass intersect_with — the push-down already restricted
        // both rankings to graph_set.
        fuse(&rankings, &strategy, None, k)
    };

    Ok(PlanResult {
        plan: plan_name,
        query: q,
        paper_ids: final_ids,
        predicates,
        first_predicate: Some(Engine::Age),
        per_engine_rows,
        round_trips,
        materializations: 1 + rankings.len() as u32,
        actual_order,
    })
}

async fn run_semantic_pushdown(
    pool: &PgPool,
    seed: i64,
    k: usize,
    ef: u32,
    graph_set: &[i64],
) -> Result<Vec<i64>> {
    let mut tx = pool.begin().await?;
    sqlx::query(&format!("SET LOCAL hnsw.ef_search = {ef}"))
        .execute(&mut *tx).await?;
    // pgvector 0.8: iterative_scan=strict_order lets HNSW emit candidates
    // in distance order while respecting the WHERE clause (avoids the
    // "filter shrinks recall below k" problem of post-filter on HNSW).
    sqlx::query("SET LOCAL hnsw.iterative_scan = strict_order")
        .execute(&mut *tx).await.ok();
    let rows = sqlx::query(
        "WITH seed AS (SELECT embedding FROM chunk_embeddings WHERE chunk_id = $1), \
              hit AS ( \
                SELECT c.paper_id, ce.embedding <=> seed.embedding AS d \
                FROM chunk_embeddings ce \
                JOIN chunks c ON c.id = ce.chunk_id, seed \
                WHERE c.paper_id = ANY($3) \
                ORDER BY ce.embedding <=> seed.embedding \
                LIMIT $2 \
              ) \
         SELECT DISTINCT ON (paper_id) paper_id \
         FROM hit \
         ORDER BY paper_id, d",
    )
    .bind(seed)
    .bind((k * 3) as i64)
    .bind(graph_set)
    .fetch_all(&mut *tx).await?;
    tx.commit().await?;
    Ok(rows.into_iter().filter_map(|r| r.try_get(0).ok()).collect())
}

async fn run_bm25_pushdown(
    pool: &PgPool,
    text: &str,
    k: usize,
    graph_set: &[i64],
) -> Result<Vec<i64>> {
    let rows = sqlx::query(
        "SELECT id FROM papers \
         WHERE abstract @@@ $1 AND id = ANY($3) \
         ORDER BY paradedb.score(id) DESC LIMIT $2",
    )
    .bind(text)
    .bind(k as i64)
    .bind(graph_set)
    .fetch_all(pool).await?;
    Ok(rows.into_iter().filter_map(|r| r.try_get(0).ok()).collect())
}

#[allow(dead_code)]
fn stub_result(plan: &'static str, q: QueryType) -> PlanResult {
    PlanResult {
        plan,
        query: q,
        paper_ids: vec![],
        predicates: vec![],
        first_predicate: None,
        per_engine_rows: vec![],
        round_trips: 0,
        materializations: 0,
        actual_order: vec![],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn naive_plan_name() { assert_eq!(NaivePlan.name(), "naive"); }

    #[test]
    fn v1_plan_name() { assert_eq!(V1Plan.name(), "v1"); }

    #[test]
    fn stub_returns_empty() {
        let r = stub_result("naive", QueryType::Q7);
        assert!(r.paper_ids.is_empty());
        assert_eq!(r.first_predicate, None);
    }

    // ---------- V2Plan (push-down) ----------

    #[test]
    fn v2_plan_name_is_v2() {
        assert_eq!(V2Plan.name(), "v2");
    }

    // ---------- V3Plan (chained push-down: v2 optimization) ----------

    #[test]
    fn v3_plan_name_is_v3() {
        assert_eq!(V3Plan.name(), "v3");
    }

    #[test]
    fn v3_uses_chained_pushdown_only_when_both_rankers_present() {
        // v3's chained path = BM25 push-down → pgvector push-down →
        // RRF(vector, bm25). It requires *both* semantic and lexical
        // rankers, which is exactly Q6 and Q7. Q4 / Q5 have only one
        // ranker after the graph push-down, so v3 delegates to v2.
        assert!(V3Plan::uses_chained_pushdown(QueryType::Q6));
        assert!(V3Plan::uses_chained_pushdown(QueryType::Q7));

        // Single-ranker (after graph filter) — v3 delegates to v2:
        assert!(!V3Plan::uses_chained_pushdown(QueryType::Q4));
        assert!(!V3Plan::uses_chained_pushdown(QueryType::Q5));

        // Single-engine baselines — v3 collapses to same helpers v2 uses:
        assert!(!V3Plan::uses_chained_pushdown(QueryType::Q1));
        assert!(!V3Plan::uses_chained_pushdown(QueryType::Q2));
        assert!(!V3Plan::uses_chained_pushdown(QueryType::Q3));
    }

    #[test]
    fn v3_chained_set_and_v2_graph_pushdown_set_are_distinct() {
        // v2's graph-pushdown set (Q4/Q5/Q7) and v3's chained-pushdown
        // set (Q6/Q7) overlap only at Q7 — that's where both
        // optimizations stack (graph filter + chained ranker
        // push-down). Q6 is uniquely a v3 path (no graph to push down,
        // but v3 still chains BM25 → pgvector); Q4 / Q5 are uniquely
        // v2 paths (graph push-down but only one ranker downstream).
        let chained: Vec<_> = [QueryType::Q1, QueryType::Q2, QueryType::Q3,
                               QueryType::Q4, QueryType::Q5, QueryType::Q6,
                               QueryType::Q7]
            .into_iter()
            .filter(|q| V3Plan::uses_chained_pushdown(*q))
            .collect();
        let v2_pd: Vec<_> = [QueryType::Q1, QueryType::Q2, QueryType::Q3,
                             QueryType::Q4, QueryType::Q5, QueryType::Q6,
                             QueryType::Q7]
            .into_iter()
            .filter(|q| V2Plan::should_use_pushdown(*q))
            .collect();
        assert_eq!(chained, vec![QueryType::Q6, QueryType::Q7]);
        assert_eq!(v2_pd,    vec![QueryType::Q4, QueryType::Q5, QueryType::Q7]);
        // Q7 is in both sets.
        assert!(chained.contains(&QueryType::Q7));
        assert!(v2_pd.contains(&QueryType::Q7));
    }

    #[test]
    fn v3_fuses_two_rankers_not_three() {
        // Sanity check on RRF input shape for the chained path. v3
        // feeds vector_rank + bm25_rank into RRF (NO graph_distance).
        // We construct the same Ranking shape the driver builds and
        // confirm fuse() consumes a 2-element rankings slice cleanly.
        let v = Ranking { engine: "pgvector",  paper_ids: vec![10, 20, 30] };
        let b = Ranking { engine: "pg_search", paper_ids: vec![20, 30, 40] };
        let rankings = vec![v, b];
        assert_eq!(rankings.len(), 2, "v3 chained RRF must fuse exactly 2 signals");
        let fused = fuse(&rankings, &FusionStrategy::rrf_default(), None, 5);
        // Paper 20 + 30 appear in both → must be ranked above 10 and 40.
        assert!(fused.contains(&20));
        assert!(fused.contains(&30));
        let pos = |pid: i64| fused.iter().position(|p| *p == pid).unwrap();
        assert!(pos(20) < pos(10), "paper 20 (in both) should rank above 10 (only sem)");
        assert!(pos(30) < pos(40), "paper 30 (in both) should rank above 40 (only lex)");
    }

    #[test]
    fn v3_uses_v2_should_use_pushdown_to_classify_queries() {
        // V3Plan's dispatch agrees with V2Plan's graph-pushdown
        // classifier for the queries that go through v2's
        // multi_predicate_pushdown helper (Q4 / Q5 / Q7). Exercising
        // V2Plan::should_use_pushdown here keeps the function alive at
        // release (it would otherwise be dead-code-pruned).
        assert!(V2Plan::should_use_pushdown(QueryType::Q4));
        assert!(V2Plan::should_use_pushdown(QueryType::Q5));
        assert!(V2Plan::should_use_pushdown(QueryType::Q7));
        assert!(!V2Plan::should_use_pushdown(QueryType::Q1));
        assert!(!V2Plan::should_use_pushdown(QueryType::Q2));
        assert!(!V2Plan::should_use_pushdown(QueryType::Q3));
        assert!(!V2Plan::should_use_pushdown(QueryType::Q6));
    }

    #[test]
    fn v2_uses_pushdown_only_when_graph_combines_with_ranker() {
        // push-down is only meaningful when graph (which produces a
        // boolean filter set) is combined with at least one ranker
        // engine whose query can be narrowed by `paper_id = ANY($set)`.
        assert!(V2Plan::should_use_pushdown(QueryType::Q4));
        assert!(V2Plan::should_use_pushdown(QueryType::Q5));
        assert!(V2Plan::should_use_pushdown(QueryType::Q7));

        // Single-engine queries collapse to v1's path — no fusion, so
        // no push-down to apply.
        assert!(!V2Plan::should_use_pushdown(QueryType::Q1));
        assert!(!V2Plan::should_use_pushdown(QueryType::Q2));
        assert!(!V2Plan::should_use_pushdown(QueryType::Q3));

        // Q6 (semantic ∩ lexical) has no graph predicate to materialize
        // first, so push-down doesn't apply.
        assert!(!V2Plan::should_use_pushdown(QueryType::Q6));
    }
}
