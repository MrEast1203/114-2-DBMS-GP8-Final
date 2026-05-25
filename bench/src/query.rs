//! Seven query types from researchdb-plan.html §Phase 1 "七種查詢組合".
//!
//! Q1 — semantic
//! Q2 — lexical (BM25)
//! Q3 — graph (cited-by / cites BFS)
//! Q4 — semantic ∩ graph
//! Q5 — lexical ∩ graph
//! Q6 — semantic ∩ lexical
//! Q7 — semantic ∩ lexical ∩ graph
//!
//! Phase 1 D1+D2 starts with Q1 end-to-end (template); Q2–Q7 are wired in
//! incrementally as the orchestrator and naive plan implementations land.

use serde::Serialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum QueryType {
    Q1,
    Q2,
    Q3,
    Q4,
    Q5,
    Q6,
    Q7,
}

impl QueryType {
    pub fn from_u8(n: u8) -> Option<Self> {
        match n {
            1 => Some(Self::Q1),
            2 => Some(Self::Q2),
            3 => Some(Self::Q3),
            4 => Some(Self::Q4),
            5 => Some(Self::Q5),
            6 => Some(Self::Q6),
            7 => Some(Self::Q7),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Q1 => "Q1",
            Self::Q2 => "Q2",
            Self::Q3 => "Q3",
            Self::Q4 => "Q4",
            Self::Q5 => "Q5",
            Self::Q6 => "Q6",
            Self::Q7 => "Q7",
        }
    }

    /// Whether this query type touches the semantic / lexical / graph
    /// engine. Used by the orchestrator to decide which predicates to
    /// enumerate.
    pub fn engines(self) -> (bool, bool, bool) {
        // (semantic, lexical, graph)
        match self {
            Self::Q1 => (true, false, false),
            Self::Q2 => (false, true, false),
            Self::Q3 => (false, false, true),
            Self::Q4 => (true, false, true),
            Self::Q5 => (false, true, true),
            Self::Q6 => (true, true, false),
            Self::Q7 => (true, true, true),
        }
    }
}

/// Inputs needed to execute a query. Different query types use different
/// subsets; absent fields are validated by the executor.
#[derive(Debug, Clone)]
pub struct QuerySpec {
    /// Seed chunk id whose embedding drives the semantic predicate.
    pub seed_chunk_id: Option<i64>,
    /// BM25 text expression.
    pub bm25_text: Option<String>,
    /// Anchor paper id for graph predicates.
    pub anchor_paper: Option<i64>,
    /// Top-k results to retain.
    pub k: usize,
    /// AGE BFS depth bound (1..=3 for Phase 1).
    pub depth: u32,
    /// HNSW ef_search; fixed per query type for this benchmark.
    pub ef_search: u32,
}

impl QuerySpec {
    /// Minimal default — caller is expected to set fields relevant to
    /// the chosen QueryType.
    pub fn new() -> Self {
        Self {
            seed_chunk_id: None,
            bm25_text: None,
            anchor_paper: None,
            k: 10,
            depth: 2,
            ef_search: 40,
        }
    }
}

impl Default for QuerySpec {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn engines_match_intersections() {
        assert_eq!(QueryType::Q1.engines(), (true, false, false));
        assert_eq!(QueryType::Q7.engines(), (true, true, true));
        // Q4 = semantic ∩ graph (no lexical)
        assert_eq!(QueryType::Q4.engines(), (true, false, true));
        // Q5 = lexical ∩ graph (no semantic)
        assert_eq!(QueryType::Q5.engines(), (false, true, true));
        // Q6 = semantic ∩ lexical (no graph)
        assert_eq!(QueryType::Q6.engines(), (true, true, false));
    }

    #[test]
    fn from_u8_round_trips() {
        for n in 1u8..=7 {
            let q = QueryType::from_u8(n).unwrap();
            assert_eq!(q.as_str(), format!("Q{n}").as_str());
        }
        assert!(QueryType::from_u8(0).is_none());
        assert!(QueryType::from_u8(8).is_none());
    }
}
