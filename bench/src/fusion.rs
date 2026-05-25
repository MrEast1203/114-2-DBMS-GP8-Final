//! Multi-engine result fusion for Q4–Q7.
//!
//! Two strategies, chosen at plan build time:
//!   * **RRF** — Reciprocal Rank Fusion (Cormack & Clarke, 2009).
//!     score(d) = Σ 1 / (k + rank_i(d))  for each engine i that retrieved d.
//!     k=60 is the de facto default from the original paper. RRF is
//!     parameter-light and ignores absolute score magnitudes, which is
//!     exactly what we want when fusing across HNSW cosine, BM25, and
//!     AGE-derived boolean filters whose scores live on different scales.
//!   * **Linear** — Σ w_i × normalize(score_i, "minmax").
//!     Used as ablation: gives RRF something to beat. The orchestrator
//!     never picks this (RRF is the production strategy), but the cost
//!     regression suite tracks it so degradation in linear is visible
//!     too.
//!
//! Intersection mode (used by Q4–Q7): when fusion is over a *graph
//! filter* (boolean: in/out), only documents that appear in the filter
//! set survive. RRF still ranks within the survivors.

use std::collections::HashMap;

/// One engine's ranked list, ordered best → worst.
#[derive(Debug, Clone)]
pub struct Ranking<'a> {
    pub engine: &'a str,
    pub paper_ids: Vec<i64>,
}

#[derive(Debug, Clone)]
pub enum FusionStrategy {
    /// Reciprocal Rank Fusion. `k` is the constant in the denominator;
    /// 60 is the original-paper default.
    Rrf { k: f64 },
    /// Linear combination of min-max normalized scores. Engine weights
    /// in the order they appear in the rankings slice.
    Linear { weights: Vec<f64> },
}

impl FusionStrategy {
    pub fn rrf_default() -> Self { Self::Rrf { k: 60.0 } }
}

/// Fuse multiple engine rankings into a single ranking of size `top_k`.
/// `intersect_with` (if Some) restricts the output to papers present in
/// the given set — used to apply a graph predicate as a hard filter.
pub fn fuse(
    rankings: &[Ranking<'_>],
    strategy: &FusionStrategy,
    intersect_with: Option<&[i64]>,
    top_k: usize,
) -> Vec<i64> {
    let allowed: Option<std::collections::HashSet<i64>> =
        intersect_with.map(|v| v.iter().copied().collect());

    let mut scores: HashMap<i64, f64> = HashMap::new();

    match strategy {
        FusionStrategy::Rrf { k } => {
            for ranking in rankings {
                for (rank, pid) in ranking.paper_ids.iter().enumerate() {
                    if let Some(ref a) = allowed {
                        if !a.contains(pid) { continue; }
                    }
                    let contrib = 1.0 / (k + (rank + 1) as f64);
                    *scores.entry(*pid).or_insert(0.0) += contrib;
                }
            }
        }
        FusionStrategy::Linear { weights } => {
            // Map rank → raw score (1.0 at rank 0, 0.0 at last). This is
            // min-max normalization of rank-as-score; tied papers (same
            // rank) get same raw before weighting.
            for (i, ranking) in rankings.iter().enumerate() {
                let w = weights.get(i).copied().unwrap_or(1.0);
                let n = ranking.paper_ids.len().max(1) as f64;
                for (rank, pid) in ranking.paper_ids.iter().enumerate() {
                    if let Some(ref a) = allowed {
                        if !a.contains(pid) { continue; }
                    }
                    let raw = 1.0 - (rank as f64) / n;
                    *scores.entry(*pid).or_insert(0.0) += w * raw;
                }
            }
        }
    }

    let mut entries: Vec<(i64, f64)> = scores.into_iter().collect();
    // Sort by score desc; tie-break by paper id asc for determinism.
    entries.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    entries.into_iter().take(top_k).map(|(pid, _)| pid).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn r<'a>(engine: &'a str, ids: &[i64]) -> Ranking<'a> {
        Ranking { engine, paper_ids: ids.to_vec() }
    }

    #[test]
    fn rrf_single_ranking_preserves_order() {
        let a = r("sem", &[10, 20, 30, 40]);
        let out = fuse(&[a], &FusionStrategy::rrf_default(), None, 4);
        assert_eq!(out, vec![10, 20, 30, 40]);
    }

    #[test]
    fn rrf_two_disjoint_unions_with_first_engine_winning_top() {
        // No overlap: each paper has score 1/(60+1) regardless of engine.
        // Tie-break by paper id asc means lowest id is first.
        let a = r("sem", &[10, 20]);
        let b = r("lex", &[30, 40]);
        let out = fuse(&[a, b], &FusionStrategy::rrf_default(), None, 4);
        assert_eq!(out.len(), 4);
        // 10 and 30 both at rank 0; tie-break → 10 first.
        assert_eq!(out[0], 10);
        assert!(out.contains(&20));
        assert!(out.contains(&30));
        assert!(out.contains(&40));
    }

    #[test]
    fn rrf_overlapping_papers_get_higher_score() {
        // Paper 10 appears in both rankings at rank 0 → score 2/(60+1)
        // Paper 20 only in one ranking → score 1/(60+2)
        // 10 should beat 20.
        let a = r("sem", &[10, 20]);
        let b = r("lex", &[10, 30]);
        let out = fuse(&[a, b], &FusionStrategy::rrf_default(), None, 3);
        assert_eq!(out[0], 10);
        assert_eq!(out.len(), 3);
    }

    #[test]
    fn rrf_higher_rank_dominates() {
        // Paper 10 at rank 0 in both; paper 20 at rank 1 in both.
        // 10's score = 2/61, 20's = 2/62 → 10 first.
        let a = r("sem", &[10, 20]);
        let b = r("lex", &[10, 20]);
        let out = fuse(&[a, b], &FusionStrategy::rrf_default(), None, 2);
        assert_eq!(out, vec![10, 20]);
    }

    #[test]
    fn rrf_intersect_filter_excludes_unallowed() {
        let a = r("sem", &[10, 20, 30]);
        let b = r("lex", &[10, 40, 50]);
        let graph = vec![10, 30]; // graph filter — only these survive
        let out = fuse(&[a, b], &FusionStrategy::rrf_default(), Some(&graph), 5);
        assert!(!out.contains(&20));
        assert!(!out.contains(&40));
        assert!(!out.contains(&50));
        assert!(out.contains(&10));
        assert!(out.contains(&30));
        // 10 in both rankings → higher score → first
        assert_eq!(out[0], 10);
    }

    #[test]
    fn linear_respects_weights() {
        // Lex is weighted 10×; paper present only in lex should beat
        // paper present only in sem despite same rank.
        let a = r("sem", &[10]);
        let b = r("lex", &[20]);
        let s = FusionStrategy::Linear { weights: vec![1.0, 10.0] };
        let out = fuse(&[a, b], &s, None, 2);
        assert_eq!(out[0], 20, "lex-only paper should win when lex weight is higher");
    }

    #[test]
    fn linear_minmax_zero_at_last_rank() {
        // With a single 5-element ranking and linear strategy, the
        // last-ranked paper gets raw score 0 (1 - 4/5 = 0.2) → still in
        // top_k=5 but lowest scored.
        let a = r("sem", &[10, 20, 30, 40, 50]);
        let s = FusionStrategy::Linear { weights: vec![1.0] };
        let out = fuse(&[a], &s, None, 5);
        assert_eq!(out, vec![10, 20, 30, 40, 50]);
    }

    #[test]
    fn fuse_respects_top_k() {
        let a = r("sem", &(0..100).collect::<Vec<_>>());
        let out = fuse(&[a], &FusionStrategy::rrf_default(), None, 7);
        assert_eq!(out.len(), 7);
    }

    #[test]
    fn fuse_empty_with_filter_blocks_all() {
        let a = r("sem", &[10, 20, 30]);
        let nothing: Vec<i64> = vec![];
        let out = fuse(&[a], &FusionStrategy::rrf_default(), Some(&nothing), 10);
        assert!(out.is_empty());
    }
}
