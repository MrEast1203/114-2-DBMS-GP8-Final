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
use crate::graph_engine::{bfs_recursive_sql, bfs_recursive_sql_with_depth, Direction};
use crate::query::{QuerySpec, QueryType};
use std::collections::HashMap;

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

// ============ V3 ORCHESTRATOR (multi-stage push-down + cost-driven order + fusion-signal recovery) ============

/// **v3** — "one-shot" plan that simultaneously claims three things:
///
/// 1. *Multi-stage push-down*: when **two** hard predicates are present
///    (Q5 = graph + lexical; Q7 = graph + lexical + vector), v3 pushes
///    the first hard predicate's filter into the second's SQL — not
///    just into the ranker as v2 does, but ranker-to-ranker, narrowing
///    the candidate set in two successive stages.
/// 2. *Cost-driven ordering*: v1's BFS / BM25 cost formulas finally
///    decide something — namely, **which hard predicate to materialize
///    first**. Reuse `cost::age_cost_v1(branching^depth)` from
///    `microbench` and `cost::pg_search_cost(matched, avg_posting)`;
///    no new fits. Tie-break by lower selectivity, then prefer BFS.
/// 3. *Fusion-signal recovery*: v2 turns the graph filter into a hard
///    AND, losing it as a ranking signal. v3 keeps using it as a hard
///    filter (push-down) **and** re-uses it as a `graph_distance` rank
///    (sort by BFS depth ASC) on the already-narrowed candidate set —
///    re-scored together with `vector_rank` and `bm25_rank` into RRF.
///    Q7 therefore fuses three signals exactly like naive did, but on
///    a candidate set the size of `S_g ∩ S_l` instead of the full
///    corpus.
///
/// On single-predicate queries (Q1 / Q2 / Q3) v3 collapses to the same
/// helper that v2 / v1 / naive use — `V2Plan::should_use_pushdown(q)`
/// returns false so v3 doesn't enter its multi-stage branch.
pub struct V3Plan;

impl V3Plan {
    /// `(branching^depth) × age_ms_unit` — reuse cost.rs constants.
    pub fn cost_bfs_ms(depth: u32) -> f64 {
        let raw = cost::age_cost_v1(depth, 2.4);
        cost::normalize(raw, Engine::Age, &EngineNorm::default())
    }
    /// `matched × avg_posting × pg_search_ms_unit` — reuse cost.rs constants.
    pub fn cost_bm25_ms(matched: usize) -> f64 {
        let raw = cost::pg_search_cost(matched, 50.0);
        cost::normalize(raw, Engine::PgSearch, &EngineNorm::default())
    }

    /// Pure cost decision: which hard predicate (BFS vs BM25) should
    /// v3 materialize first? Lower cost wins; tie → lower selectivity;
    /// final tie → BFS first (graph push-down on the ranker side is
    /// usually cheaper to wire than the reverse).
    pub fn pick_first_hard_predicate(
        cost_bfs_ms: f64,
        cost_bm25_ms: f64,
        sel_bfs: f64,
        sel_bm25: f64,
    ) -> Engine {
        if cost_bfs_ms < cost_bm25_ms {
            Engine::Age
        } else if cost_bm25_ms < cost_bfs_ms {
            Engine::PgSearch
        } else if sel_bfs <= sel_bm25 {
            Engine::Age
        } else {
            Engine::PgSearch
        }
    }

    /// Build the `graph_distance_rank` from a depth map restricted to
    /// the surviving candidate set. Papers not in the BFS neighborhood
    /// are *dropped* (sentinel = "not in this ranking"), rather than
    /// appended at the tail — keeps RRF semantics clean (a paper
    /// outside `S_g` contributes 0 to graph_distance, just as a paper
    /// outside `S_l` contributes 0 to bm25_rank).
    pub fn build_graph_distance_rank(
        candidates: &[i64],
        depth_map: &HashMap<i64, i32>,
    ) -> Vec<i64> {
        let mut paired: Vec<(i64, i32)> = candidates
            .iter()
            .filter_map(|pid| depth_map.get(pid).map(|d| (*pid, *d)))
            .collect();
        // sort by depth ASC, then paper_id ASC for determinism.
        paired.sort_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
        paired.into_iter().map(|(pid, _)| pid).collect()
    }
}

#[async_trait::async_trait]
impl Plan for V3Plan {
    fn name(&self) -> &'static str { "v3" }

    async fn execute(&self, pool: &PgPool, q: QueryType, spec: &QuerySpec) -> Result<PlanResult> {
        // v3 dispatch.
        // Use V2Plan::should_use_pushdown as a feature flag: it returns
        // true exactly when graph + ≥1 ranker coexist (Q4 / Q5 / Q7).
        // Q6 (semantic + lexical, no graph) returns false here but v3
        // still benefits from a single push-down (lexical → vector), so
        // we handle Q6 in the multi-predicate driver too.
        let has_graph_pushdown = V2Plan::should_use_pushdown(q);
        tracing::trace!(plan = self.name(), qid = q.as_str(), has_graph_pushdown,
                        "v3 dispatch");
        match q {
            QueryType::Q1 => semantic_only(pool, spec, "v3", q).await,
            QueryType::Q2 => lexical_only(pool, spec, "v3", q).await,
            QueryType::Q3 => graph_only(pool, spec, "v3", q).await,
            QueryType::Q4 | QueryType::Q5 | QueryType::Q6 | QueryType::Q7 => {
                multi_predicate_v3(pool, q, spec).await
            }
        }
    }
}

/// V3 multi-predicate driver. Decides cost-based ordering for the
/// hard predicates (BFS + BM25), pushes the first one's filter into
/// the second, then re-scores all three engines on the resulting
/// candidate set and RRF-fuses them.
async fn multi_predicate_v3(
    pool: &PgPool,
    q: QueryType,
    spec: &QuerySpec,
) -> Result<PlanResult> {
    let (use_sem, use_lex, use_gph) = q.engines();
    let k = spec.k;
    let n_fetch = k * PER_ENGINE_OVERFETCH;

    let mut predicates: Vec<PredicateEstimate> = Vec::new();
    let mut per_engine_rows: Vec<(Engine, usize)> = Vec::new();
    let mut actual_order: Vec<Engine> = Vec::new();
    let mut round_trips = 0u32;
    let mut materializations = 0u32;

    // ---- pre-compute hard-predicate cost estimates ----
    let depth = spec.depth.max(1).min(3);
    let mut cost_bfs_ms = f64::INFINITY;
    let mut cost_bm25_ms = f64::INFINITY;
    let mut sel_bfs = 1.0_f64;
    let mut sel_bm25 = 1.0_f64;
    let mut matched_bm25: i64 = 0;

    if use_lex {
        // One round-trip to estimate matched count (same pattern v1/v2 use).
        let text = spec.bm25_text.as_deref()
            .context("Q5/Q6/Q7 require bm25_text")?;
        let (m,): (i64,) = sqlx::query_as(
            "SELECT count(*) FROM papers WHERE abstract @@@ $1",
        ).bind(text).fetch_one(pool).await.unwrap_or((0,));
        matched_bm25 = m;
        round_trips += 1;
        cost_bm25_ms = V3Plan::cost_bm25_ms(m as usize);
        sel_bm25 = cost::selectivity_lexical(m as usize, 5_011);
    }
    if use_gph {
        cost_bfs_ms = V3Plan::cost_bfs_ms(depth);
        sel_bfs = cost::selectivity_graph(n_fetch, 5_011); // pre-execution estimate
    }

    // ---- decide which hard predicate to materialize first (Q5/Q7) ----
    let first_predicate_choice: Option<Engine> = if use_gph && use_lex {
        let pick = V3Plan::pick_first_hard_predicate(
            cost_bfs_ms, cost_bm25_ms, sel_bfs, sel_bm25,
        );
        tracing::info!(
            plan = "v3",
            qid = q.as_str(),
            cost_bfs_ms,
            cost_bm25_ms,
            matched_bm25,
            chosen = ?pick,
            "v3 cost decision: which hard predicate first"
        );
        Some(pick)
    } else if use_gph {
        Some(Engine::Age)
    } else if use_lex {
        Some(Engine::PgSearch)
    } else {
        None
    };

    // ---- materialize hard predicates in chosen order ----
    let anchor_opt = spec.anchor_paper;
    let bm25_text_opt = spec.bm25_text.as_deref();

    let mut s_g: Vec<i64> = Vec::new();
    let mut depth_map: HashMap<i64, i32> = HashMap::new();
    let mut s_l_with_score: Vec<(i64, f64)> = Vec::new(); // bm25 hits (top n_fetch)

    let bm25_first =
        matches!(first_predicate_choice, Some(Engine::PgSearch)) && use_gph;

    if bm25_first {
        // Path B: BM25 first → push the resulting id set into BFS-filter.
        // BFS is still required (we need depths), but only papers in S_l
        // contribute to the final ranking, so we just compute S_both at
        // the end via intersection (Rust-side).
        let text = bm25_text_opt.context("bm25_first path needs bm25_text")?;
        s_l_with_score = run_bm25_full_hits(pool, text).await?;
        round_trips += 1;
        materializations += 1;
        per_engine_rows.push((Engine::PgSearch, s_l_with_score.len()));
        actual_order.push(Engine::PgSearch);
        predicates.push(PredicateEstimate {
            engine: Engine::PgSearch,
            raw_cost: cost::pg_search_cost(matched_bm25 as usize, 50.0),
            ms_estimate: cost_bm25_ms,
            selectivity: sel_bm25,
        });

        // BFS unrestricted (recursive SQL doesn't take a per-row filter
        // cleanly; we'd need IN-list propagation in the recursion which
        // PostgreSQL won't push down efficiently). We just run it; the
        // intersection happens in Rust.
        let anchor = anchor_opt.context("graph predicate needs anchor_paper")?;
        let bfs_rows = bfs_recursive_sql_with_depth(pool, anchor, depth, Direction::Reverse).await?;
        round_trips += 1;
        materializations += 1;
        per_engine_rows.push((Engine::Age, bfs_rows.len()));
        actual_order.push(Engine::Age);
        s_g.reserve(bfs_rows.len());
        for (pid, d) in &bfs_rows {
            s_g.push(*pid);
            depth_map.insert(*pid, *d);
        }
        let bfs_raw = cost::age_cost_v1(depth, 2.4);
        predicates.push(PredicateEstimate {
            engine: Engine::Age,
            raw_cost: bfs_raw,
            ms_estimate: cost_bfs_ms,
            selectivity: cost::selectivity_graph(s_g.len(), 5_011),
        });
    } else if use_gph {
        // Path A: BFS first → push S_g into BM25.
        let anchor = anchor_opt.context("graph predicate needs anchor_paper")?;
        let bfs_rows = bfs_recursive_sql_with_depth(pool, anchor, depth, Direction::Reverse).await?;
        round_trips += 1;
        materializations += 1;
        per_engine_rows.push((Engine::Age, bfs_rows.len()));
        actual_order.push(Engine::Age);
        s_g.reserve(bfs_rows.len());
        for (pid, d) in &bfs_rows {
            s_g.push(*pid);
            depth_map.insert(*pid, *d);
        }
        let bfs_raw = cost::age_cost_v1(depth, 2.4);
        predicates.push(PredicateEstimate {
            engine: Engine::Age,
            raw_cost: bfs_raw,
            ms_estimate: cost_bfs_ms,
            selectivity: cost::selectivity_graph(s_g.len(), 5_011),
        });

        if s_g.is_empty() {
            return Ok(PlanResult {
                plan: "v3", query: q,
                paper_ids: vec![],
                predicates,
                first_predicate: first_predicate_choice,
                per_engine_rows,
                round_trips,
                materializations,
                actual_order,
            });
        }

        if use_lex {
            let text = bm25_text_opt.context("lex predicate needs bm25_text")?;
            s_l_with_score = run_bm25_pushdown_full(pool, text, &s_g).await?;
            round_trips += 1;
            materializations += 1;
            per_engine_rows.push((Engine::PgSearch, s_l_with_score.len()));
            actual_order.push(Engine::PgSearch);
            predicates.push(PredicateEstimate {
                engine: Engine::PgSearch,
                raw_cost: cost::pg_search_cost(matched_bm25 as usize, 50.0),
                ms_estimate: cost_bm25_ms,
                selectivity: sel_bm25,
            });
        }
    } else if use_lex {
        // No graph: Q6 path. BM25 first, no BFS.
        let text = bm25_text_opt.context("lex predicate needs bm25_text")?;
        s_l_with_score = run_bm25_full_hits(pool, text).await?;
        round_trips += 1;
        materializations += 1;
        per_engine_rows.push((Engine::PgSearch, s_l_with_score.len()));
        actual_order.push(Engine::PgSearch);
        predicates.push(PredicateEstimate {
            engine: Engine::PgSearch,
            raw_cost: cost::pg_search_cost(matched_bm25 as usize, 50.0),
            ms_estimate: cost_bm25_ms,
            selectivity: sel_bm25,
        });
    }

    // ---- compute S_both = S_g ∩ S_l (whichever is present) ----
    let s_both: Vec<i64> = if use_gph && use_lex {
        let g_set: std::collections::HashSet<i64> = s_g.iter().copied().collect();
        let mut both: Vec<(i64, f64)> = s_l_with_score.iter()
            .filter(|(pid, _)| g_set.contains(pid))
            .copied()
            .collect();
        // preserve BM25 order (already DESC) within S_both
        s_l_with_score = both.clone();
        both.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        both.into_iter().map(|(pid, _)| pid).collect()
    } else if use_gph {
        s_g.clone()
    } else if use_lex {
        s_l_with_score.iter().map(|(pid, _)| *pid).collect()
    } else {
        vec![] // never hit for Q4-Q7
    };

    if s_both.is_empty() {
        return Ok(PlanResult {
            plan: "v3", query: q,
            paper_ids: vec![],
            predicates,
            first_predicate: first_predicate_choice,
            per_engine_rows,
            round_trips,
            materializations,
            actual_order,
        });
    }

    // ---- final stage: push S_both into pgvector ranker if Q4/Q6/Q7 ----
    let mut vector_rank: Vec<i64> = Vec::new();
    if use_sem {
        let seed = spec.seed_chunk_id
            .context("semantic predicate needs seed_chunk_id")?;
        vector_rank = run_semantic_pushdown(pool, seed, n_fetch, spec.ef_search, &s_both).await?;
        round_trips += 2;
        materializations += 1;
        per_engine_rows.push((Engine::Pgvector, vector_rank.len()));
        actual_order.push(Engine::Pgvector);
        let raw = cost::pgvector_cost(spec.ef_search, s_both.len().max(2));
        predicates.push(PredicateEstimate {
            engine: Engine::Pgvector,
            raw_cost: raw,
            ms_estimate: cost::normalize(raw, Engine::Pgvector, &EngineNorm::default()),
            selectivity: cost::selectivity_semantic(k, s_both.len().max(1)),
        });
    }

    // ---- build the three rankings (vector / bm25 / graph_distance) ----
    let mut rankings: Vec<Ranking<'static>> = Vec::new();
    if use_sem {
        rankings.push(Ranking { engine: "pgvector", paper_ids: vector_rank });
    }
    if use_lex {
        // bm25 score-ordered restriction to S_both
        let s_both_set: std::collections::HashSet<i64> = s_both.iter().copied().collect();
        let mut bm25_pairs: Vec<(i64, f64)> = s_l_with_score
            .iter()
            .filter(|(pid, _)| s_both_set.contains(pid))
            .copied()
            .collect();
        bm25_pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        rankings.push(Ranking {
            engine: "pg_search",
            paper_ids: bm25_pairs.into_iter().map(|(pid, _)| pid).collect(),
        });
    }
    if use_gph {
        let gdr = V3Plan::build_graph_distance_rank(&s_both, &depth_map);
        rankings.push(Ranking { engine: "graph_distance", paper_ids: gdr });
    }

    // log fused engine list (this read of Ranking::engine clears the
    // existing dead-code warning at release).
    tracing::debug!(
        plan = "v3",
        qid = q.as_str(),
        engines = ?rankings.iter().map(|r| r.engine).collect::<Vec<_>>(),
        candidate_pool = s_both.len(),
        "v3 RRF fusion"
    );

    let strategy = FusionStrategy::rrf_default();
    let final_ids = if rankings.is_empty() {
        vec![]
    } else if rankings.len() == 1 {
        rankings[0].paper_ids.iter().take(k).copied().collect()
    } else {
        fuse(&rankings, &strategy, None, k)
    };

    Ok(PlanResult {
        plan: "v3", query: q,
        paper_ids: final_ids,
        predicates,
        first_predicate: first_predicate_choice,
        per_engine_rows,
        round_trips,
        materializations,
        actual_order,
    })
}

/// Hard upper bound on how many BM25 hits v3 materializes as the
/// `S_l` filter set. The brief's `S_l` is "the BM25 hit set" — i.e.
/// every paper matching `@@@`. In practice we cap to keep
/// `id = ANY($S_l)` push-down arrays manageable; 5000 is large
/// enough that even broad queries (e.g. "neural network") are
/// fully captured on the 50K corpus.
const V3_BM25_HIT_CAP: i64 = 5_000;

/// All BM25 hits (up to V3_BM25_HIT_CAP), score-ordered DESC. Used
/// by v3 to materialize `S_l` as both a *filter set* (push-down) and
/// a *ranking signal* (top-N slice for bm25_rank). Q6 and the
/// "BM25 first" path of Q5/Q7 both call this.
async fn run_bm25_full_hits(pool: &PgPool, text: &str) -> Result<Vec<(i64, f64)>> {
    let rows = sqlx::query(
        "SELECT id, paradedb.score(id) AS s FROM papers \
         WHERE abstract @@@ $1 \
         ORDER BY paradedb.score(id) DESC LIMIT $2",
    )
    .bind(text)
    .bind(V3_BM25_HIT_CAP)
    .fetch_all(pool).await?;
    Ok(rows.iter().filter_map(|r| {
        let pid: i64 = r.try_get(0).ok()?;
        let s: f32 = r.try_get(1).ok()?;
        Some((pid, s as f64))
    }).collect())
}

/// All BM25 hits restricted via `id = ANY($graph_set)`, score-ordered
/// DESC. The "BFS first → push graph filter into BM25" path of v3
/// Q5 / Q7. Returns full intersection (up to V3_BM25_HIT_CAP) so the
/// downstream pgvector push-down sees every paper in `S_g ∩ S_l`.
async fn run_bm25_pushdown_full(
    pool: &PgPool,
    text: &str,
    graph_set: &[i64],
) -> Result<Vec<(i64, f64)>> {
    let rows = sqlx::query(
        "SELECT id, paradedb.score(id) AS s FROM papers \
         WHERE abstract @@@ $1 AND id = ANY($2) \
         ORDER BY paradedb.score(id) DESC LIMIT $3",
    )
    .bind(text)
    .bind(graph_set)
    .bind(V3_BM25_HIT_CAP)
    .fetch_all(pool).await?;
    Ok(rows.iter().filter_map(|r| {
        let pid: i64 = r.try_get(0).ok()?;
        let s: f32 = r.try_get(1).ok()?;
        Some((pid, s as f64))
    }).collect())
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

    // ---------- V3Plan (multi-stage push-down + cost ordering + fusion recovery) ----------

    #[test]
    fn v3_plan_name_is_v3() {
        assert_eq!(V3Plan.name(), "v3");
    }

    #[test]
    fn v3_cost_ordering_picks_cheaper_predicate() {
        // BFS cheaper than BM25 → BFS first.
        assert_eq!(
            V3Plan::pick_first_hard_predicate(/*bfs*/ 1.0, /*bm25*/ 10.0,
                                              /*sel_bfs*/ 0.5, /*sel_bm25*/ 0.5),
            Engine::Age,
        );
        // BM25 cheaper → BM25 first.
        assert_eq!(
            V3Plan::pick_first_hard_predicate(10.0, 1.0, 0.5, 0.5),
            Engine::PgSearch,
        );
        // Tie on cost → smaller selectivity (more selective) wins.
        assert_eq!(
            V3Plan::pick_first_hard_predicate(1.0, 1.0, 0.01, 0.5),
            Engine::Age,
        );
        assert_eq!(
            V3Plan::pick_first_hard_predicate(1.0, 1.0, 0.5, 0.01),
            Engine::PgSearch,
        );
        // Full tie → BFS wins (graph push-down on ranker is cheaper to wire).
        assert_eq!(
            V3Plan::pick_first_hard_predicate(1.0, 1.0, 0.5, 0.5),
            Engine::Age,
        );
    }

    #[test]
    fn v3_cost_bfs_reuses_branching_pow_depth() {
        // Sanity-check that V3Plan::cost_bfs_ms is monotone in depth —
        // doubles or more from depth 1 → 2 — i.e., still the
        // exponential model from cost::age_cost_v1. We don't refit.
        let c1 = V3Plan::cost_bfs_ms(1);
        let c2 = V3Plan::cost_bfs_ms(2);
        let c3 = V3Plan::cost_bfs_ms(3);
        assert!(c2 > c1);
        assert!(c3 > c2);
        // exponential, not linear: c3 / c1 should be > 4 with b=2.4.
        assert!(c3 / c1 > 4.0, "expected exponential growth, got c3/c1 = {}", c3 / c1);
    }

    #[test]
    fn v3_graph_distance_rank_orders_by_depth_asc() {
        // Candidates 10 / 20 / 30 / 40, with depths 3 / 1 / 2 / 1.
        // Expected order: 20 (d=1), 40 (d=1), 30 (d=2), 10 (d=3).
        // Within same depth, tie-break by paper_id asc.
        let candidates = vec![10, 20, 30, 40];
        let mut dm = HashMap::new();
        dm.insert(10, 3);
        dm.insert(20, 1);
        dm.insert(30, 2);
        dm.insert(40, 1);
        let ranked = V3Plan::build_graph_distance_rank(&candidates, &dm);
        assert_eq!(ranked, vec![20, 40, 30, 10]);
    }

    #[test]
    fn v3_graph_distance_rank_drops_papers_outside_bfs_set() {
        // Candidate 50 isn't in the BFS depth_map — should be dropped
        // (sentinel = "outside this engine's ranking", not appended at
        // tail). Documented in docs/v3_design.md §3.
        let candidates = vec![10, 50, 20];
        let mut dm = HashMap::new();
        dm.insert(10, 2);
        dm.insert(20, 1);
        let ranked = V3Plan::build_graph_distance_rank(&candidates, &dm);
        assert_eq!(ranked, vec![20, 10]);
        assert!(!ranked.contains(&50));
    }

    #[test]
    fn v3_uses_v2_should_use_pushdown_to_classify_queries() {
        // V3Plan's dispatch logic uses V2Plan::should_use_pushdown to
        // identify which queries have a graph push-down opportunity.
        // This test asserts the boolean agrees with v3's expectations.
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
