//! Plan trait + Naive sequential plan + v0 cost-based plan.
//!
//! Phase 1 status:
//!   * Q1 (semantic) — both plans implemented end-to-end. Single-predicate,
//!     so they're functionally identical; benchmarking still distinguishes
//!     orchestrator wrapper overhead.
//!   * Q2 / Q3 — single-predicate, fall back to single engine; same story
//!     as Q1.
//!   * Q4 / Q5 / Q6 / Q7 — TODO. Will land as orchestrator v0 work proceeds.
//!     Currently return PlanResult { rows: vec![], plan: "todo", ... }.

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
    /// insertion order; for v0 it follows ascending selectivity.
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
                                /* order_by_selectivity */ false,
                                AgeModel::V0Linear).await
            }
        }
    }
}

// ============ V0 ORCHESTRATOR ============

/// Which AGE cost model the orchestrator uses to estimate graph cost.
/// V0 uses the linear textbook form; V1 uses the empirical exponential
/// form fit from `bench micro-bench-age` (Phase 1 §D5).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AgeModel {
    V0Linear,
    V1Exponential,
}

/// v0 cost-based orchestrator. Single-predicate queries collapse to the
/// single engine (identical to naive). Multi-predicate queries enumerate
/// candidate predicate orderings, pick by selectivity × cost.
pub struct V0Plan {
    pub norm: EngineNorm,
}

impl V0Plan {
    pub fn new() -> Self {
        Self { norm: EngineNorm::default() }
    }
}

impl Default for V0Plan {
    fn default() -> Self { Self::new() }
}

#[async_trait::async_trait]
impl Plan for V0Plan {
    fn name(&self) -> &'static str { "v0" }

    async fn execute(&self, pool: &PgPool, q: QueryType, spec: &QuerySpec) -> Result<PlanResult> {
        match q {
            QueryType::Q1 => semantic_only(pool, spec, "v0", q).await,
            QueryType::Q2 => lexical_only(pool, spec, "v0", q).await,
            QueryType::Q3 => graph_only(pool, spec, "v0", q).await,
            QueryType::Q4 | QueryType::Q5 | QueryType::Q6 | QueryType::Q7 => {
                multi_predicate(pool, q, spec, "v0",
                                /* order_by_selectivity */ true,
                                AgeModel::V0Linear).await
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
/// to the same execution as v0 — `should_use_pushdown` returns false.
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
            // Q6 has no graph predicate; fall back to v0's post-filter path.
            QueryType::Q6 => multi_predicate(pool, q, spec, "v2",
                                /* order_by_selectivity */ true,
                                AgeModel::V1Exponential).await,
            // Q4 / Q5 / Q7: push graph filter into ranker SQL.
            QueryType::Q4 | QueryType::Q5 | QueryType::Q7 => {
                multi_predicate_pushdown(pool, q, spec, "v2").await
            }
        }
    }
}

// ============ V1 ORCHESTRATOR (D5 empirical AGE cost) ============

/// v1 cost-based orchestrator. Identical to v0 except the AGE cost
/// model uses `age_cost_v1` (branching^depth) from Phase 1 §D5
/// micro-benchmarking. Because the actual SQL it executes is the same,
/// final ranked results are bit-identical to v0; the difference shows
/// up only in the orchestrator's *cost estimates* and (in principle)
/// predicate ordering when graph cost would otherwise be
/// underestimated.
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
                                /* order_by_selectivity */ true,
                                AgeModel::V1Exponential).await
            }
        }
    }
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

    let raw = cost::age_cost(depth, 2.4); // 2.4 ≈ synth avg out-degree
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
///   * **v0**:   engine order is decided by selectivity — we ask each
///     engine for its candidate-set size first, then run the smallest
///     first. Graph is *always* used as the filter when present (its
///     selectivity is captured in `selectivity_graph`).
async fn multi_predicate(
    pool: &PgPool,
    q: QueryType,
    spec: &QuerySpec,
    plan_name: &'static str,
    order_by_selectivity: bool,
    age_model: AgeModel,
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
        // For v0 we need an actual matched count to estimate selectivity;
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

        let raw = match age_model {
            AgeModel::V0Linear      => cost::age_cost(depth, 26.3),
            AgeModel::V1Exponential => cost::age_cost_v1(depth, 26.3),
        };
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
    fn v0_plan_name() { assert_eq!(V0Plan::new().name(), "v0"); }

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

    #[test]
    fn v2_uses_pushdown_only_when_graph_combines_with_ranker() {
        // push-down is only meaningful when graph (which produces a
        // boolean filter set) is combined with at least one ranker
        // engine whose query can be narrowed by `paper_id = ANY($set)`.
        assert!(V2Plan::should_use_pushdown(QueryType::Q4));
        assert!(V2Plan::should_use_pushdown(QueryType::Q5));
        assert!(V2Plan::should_use_pushdown(QueryType::Q7));

        // Single-engine queries collapse to v0's path — no fusion, so
        // no push-down to apply.
        assert!(!V2Plan::should_use_pushdown(QueryType::Q1));
        assert!(!V2Plan::should_use_pushdown(QueryType::Q2));
        assert!(!V2Plan::should_use_pushdown(QueryType::Q3));

        // Q6 (semantic ∩ lexical) has no graph predicate to materialize
        // first, so push-down doesn't apply.
        assert!(!V2Plan::should_use_pushdown(QueryType::Q6));
    }
}
