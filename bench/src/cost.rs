//! Per-engine cost functions used by the v1 cost-reordering plan.
//!
//! All three engines' costs are dimensionless after normalization. We
//! divide each engine's raw cost by its dry-run median latency (in
//! milliseconds, captured at startup) so the orchestrator picks the
//! predicate whose ACTUAL wall-time we expect to be smallest — not just
//! the one with the smallest theoretical FLOP count.
//!
//! Formulas:
//!   * HNSW       : cost = ef_search × log(n)         (Malkov & Yashunin '20)
//!   * BM25       : cost = candidate_set × avg_posting (ParadeDB docs)
//!   * BFS (AGE)  : cost = branching ^ depth          (geometric — frontier
//!                                                     grows as b^d per
//!                                                     depth level; fit
//!                                                     empirically by
//!                                                     `microbench`)

use serde::Serialize;

/// Per-engine normalization constants. Populated by a dry-run at startup;
/// defaults are reasonable for the 5K synthetic dataset on a 2024-vintage
/// laptop. The orchestrator updates these in-place after warmup.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct EngineNorm {
    pub pgvector_ms_unit: f64,  // ms per `ef_search × log(n)` unit
    pub pg_search_ms_unit: f64, // ms per `candidate × posting` unit
    pub age_ms_unit: f64,       // ms per `branching ^ depth` unit
}

impl Default for EngineNorm {
    fn default() -> Self {
        Self {
            pgvector_ms_unit: 0.005,
            pg_search_ms_unit: 0.0001,
            age_ms_unit: 0.02,
        }
    }
}

/// Engine identifier used in plan output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub enum Engine {
    Pgvector,
    PgSearch,
    Age,
}

/// HNSW theoretical cost. n is the index cardinality.
pub fn pgvector_cost(ef_search: u32, n: usize) -> f64 {
    let n = (n as f64).max(2.0); // log(1) = 0 would zero out the cost
    (ef_search as f64) * n.ln()
}

/// BM25 theoretical cost. candidate_set ≈ the union of posting lists hit
/// by the query terms; avg_posting is the mean posting-list length for
/// terms in this query.
pub fn pg_search_cost(candidate_set: usize, avg_posting: f64) -> f64 {
    (candidate_set as f64) * avg_posting.max(1.0)
}

/// BFS empirical cost: `branching ^ depth`.
///
/// BFS frontier grows geometrically — depth 1 visits b nodes, depth 2
/// visits b², depth 3 visits b³ — so total work is `b^d`. Real BFS
/// latency on the 5K-paper graph fits this model with coefficient
/// ~0.030 ms per unit, captured by `bench micro-bench-age`. We keep
/// cost dimensionless here — normalization happens in
/// `EngineNorm::age_ms_unit`.
pub fn age_cost_v1(depth: u32, avg_branching: f64) -> f64 {
    let depth = depth.max(1) as f64;
    let branching = avg_branching.max(1.0);
    branching.powf(depth)
}

/// Normalize an engine's raw theoretical cost to a per-engine ms estimate.
pub fn normalize(raw: f64, engine: Engine, norm: &EngineNorm) -> f64 {
    let factor = match engine {
        Engine::Pgvector => norm.pgvector_ms_unit,
        Engine::PgSearch => norm.pg_search_ms_unit,
        Engine::Age => norm.age_ms_unit,
    };
    raw * factor
}

/// Selectivity ∈ [0.0, 1.0] — the fraction of rows the predicate is
/// expected to retain. Smaller = more selective = should run earlier.
///
/// Crude per-engine estimators; tighter ones would need full cardinality
/// statistics from each index.
pub fn selectivity_semantic(k_topk: usize, n_corpus: usize) -> f64 {
    if n_corpus == 0 {
        return 1.0;
    }
    (k_topk as f64 / n_corpus as f64).clamp(1e-6, 1.0)
}

pub fn selectivity_lexical(matched: usize, n_corpus: usize) -> f64 {
    if n_corpus == 0 {
        return 1.0;
    }
    (matched as f64 / n_corpus as f64).clamp(1e-6, 1.0)
}

pub fn selectivity_graph(reachable: usize, n_nodes: usize) -> f64 {
    if n_nodes == 0 {
        return 1.0;
    }
    (reachable as f64 / n_nodes as f64).clamp(1e-6, 1.0)
}

/// Annotated per-predicate cost estimate. The plan enumerator builds one
/// of these per (engine, query-fragment) pair and orders them by
/// (selectivity, ms_estimate).
#[derive(Debug, Clone, Serialize)]
pub struct PredicateEstimate {
    pub engine: Engine,
    pub raw_cost: f64,
    pub ms_estimate: f64,
    pub selectivity: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pgvector_cost_grows_with_ef_search() {
        let n = 5_000;
        let low = pgvector_cost(10, n);
        let high = pgvector_cost(320, n);
        assert!(high > low, "ef_search 320 should cost more than 10");
        assert!(low > 0.0);
    }

    #[test]
    fn pgvector_cost_logarithmic_in_n() {
        let ef = 40;
        let small = pgvector_cost(ef, 1_000);
        let big = pgvector_cost(ef, 1_000_000);
        // log(1M) / log(1k) ~ 2, so cost ratio should be ~2 (definitely
        // not 1000×). This catches accidentally linear cost models.
        let ratio = big / small;
        assert!(ratio > 1.5 && ratio < 3.0, "log-scale ratio out of range: {ratio}");
    }

    #[test]
    fn pgvector_cost_guards_log_of_tiny_n() {
        // log(1) == 0 would zero out cost; we floor at log(2).
        assert!(pgvector_cost(40, 1) > 0.0);
        assert!(pgvector_cost(40, 0) > 0.0);
    }

    #[test]
    fn pg_search_cost_zero_posting_floors_to_one() {
        // A query whose terms have empty posting lists still has scan
        // overhead; floor at 1 to keep the estimator monotone.
        let c = pg_search_cost(100, 0.0);
        assert_eq!(c, 100.0);
    }

    #[test]
    fn age_cost_v1_exponential_in_depth() {
        // BFS cost grows multiplicatively per depth increment.
        let b = 10.0_f64;
        let d1 = age_cost_v1(1, b);
        let d2 = age_cost_v1(2, b);
        let d3 = age_cost_v1(3, b);
        // Each step should multiply by ~b.
        assert!((d2 / d1 - b).abs() < 1e-9);
        assert!((d3 / d2 - b).abs() < 1e-9);
    }

    #[test]
    fn age_cost_v1_dominates_at_high_depth() {
        // Geometric growth: at depth=3, b=30 should yield 27_000.
        let b = 30.0_f64;
        assert!(age_cost_v1(3, b) > 20_000.0);
    }

    #[test]
    fn normalize_scales_by_engine() {
        let norm = EngineNorm::default();
        let raw = 1000.0;
        // pg_search_ms_unit is much smaller than pgvector_ms_unit
        let pgv  = normalize(raw, Engine::Pgvector, &norm);
        let pgs  = normalize(raw, Engine::PgSearch, &norm);
        assert!(pgv > pgs, "pgvector unit should normalize to more ms than pg_search at equal raw");
    }

    #[test]
    fn selectivity_in_range() {
        assert_eq!(selectivity_semantic(10, 5_000), 10.0 / 5_000.0);
        assert_eq!(selectivity_lexical(0, 5_000), 1e-6);
        assert_eq!(selectivity_graph(5_000, 5_000), 1.0);
        // Edge: empty corpus → returns 1.0 (don't divide by zero).
        assert_eq!(selectivity_semantic(10, 0), 1.0);
    }

    #[test]
    fn selectivity_clamped() {
        // Even with absurd inputs we never exceed [1e-6, 1.0].
        let s = selectivity_lexical(10_000_000, 100);
        assert!(s <= 1.0);
        assert!(s >= 1e-6);
    }
}
