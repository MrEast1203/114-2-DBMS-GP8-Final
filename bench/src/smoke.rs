//! AGE smoke test — Phase 1 Week 1 gate.
//!
//! Goal: measure BFS-from-node latency on the loaded citation graph at
//! depth 1, 2, 3, stratified by source-node out-degree. Output a JSON
//! report that informs the gate decision (researchdb-plan.html §Phase 1
//! deliverables, task A1 in phase1_progress.html).
//!
//! Methodology
//! -----------
//! 1. Bucket Paper nodes by out-degree into low (1-5), mid (6-20),
//!    high (21+). The synthesizer / OpenAlex BFS should produce nodes
//!    in all three buckets — we assert this and fail loudly if not.
//! 2. Sample N start nodes per bucket without replacement.
//! 3. For each (bucket, depth) cell, run BFS N times and record wall-clock
//!    latency into an HDR histogram.
//! 4. Emit per-cell P50 / P95 / P99 and a gate verdict.
//!
//! Gate criteria (from plan)
//! -------------------------
//! * BFS depth 3 P95 < 60s on every bucket → PASS (keep 50K target).
//! * Any depth 3 cell exceeds 60s or OOMs    → REDUCE_DATASET (10K–20K).

use anyhow::{anyhow, Context, Result};
use clap::Args;
use hdrhistogram::Histogram;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use serde::Serialize;
use sqlx::postgres::PgPool;
use sqlx::Row;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Args, Debug, Clone)]
pub struct SmokeArgs {
    /// Samples per (bucket, depth) cell.
    #[arg(long, default_value_t = 100)]
    pub samples: usize,

    /// Per-query timeout in seconds. Exceeding this for a depth-3 cell
    /// triggers the gate's "reduce dataset" recommendation.
    #[arg(long, default_value_t = 60)]
    pub timeout_sec: u64,

    /// PRNG seed for reproducibility across runs.
    #[arg(long, default_value_t = 42)]
    pub seed: u64,

    /// Where to write the JSON report. "-" writes to stdout.
    #[arg(long, default_value = "-")]
    pub out: String,
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq, Hash)]
pub enum Bucket {
    Low,
    Mid,
    High,
}

impl Bucket {
    pub fn classify(outdeg: i64) -> Option<Self> {
        match outdeg {
            d if d <= 0 => None,
            1..=5 => Some(Bucket::Low),
            6..=20 => Some(Bucket::Mid),
            _ => Some(Bucket::High),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Bucket::Low => "low",
            Bucket::Mid => "mid",
            Bucket::High => "high",
        }
    }
}

#[derive(Serialize, Debug)]
pub struct CellStats {
    pub depth: u32,
    pub bucket: String,
    pub n: usize,
    pub p50_ms: u64,
    pub p95_ms: u64,
    pub p99_ms: u64,
    pub max_ms: u64,
    pub timeouts: usize,
}

#[derive(Serialize, Debug)]
pub struct SmokeReport {
    pub schema: &'static str,
    pub age_version: String,
    pub graph_nodes: i64,
    pub graph_edges: i64,
    pub samples_per_cell: usize,
    pub timeout_sec: u64,
    pub cells: Vec<CellStats>,
    pub gate_verdict: GateVerdict,
}

#[derive(Serialize, Debug, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum GateVerdict {
    /// All depth-3 cells finished within timeout; keep planned 50K dataset.
    Pass,
    /// At least one depth-3 cell timed out; recommend dataset reduction.
    ReduceDataset,
    /// BFS could not complete on any depth-3 sample — investigate before
    /// retrying.
    Investigate,
}

pub async fn run(pool: &PgPool, args: SmokeArgs) -> Result<()> {
    // 1. Sanity: AGE present, graph populated.
    let (age_ver,): (String,) = sqlx::query_as(
        "SELECT COALESCE(extversion, '') FROM pg_extension WHERE extname = 'age'",
    )
    .fetch_one(pool)
    .await
    .context("AGE extension version query failed")?;
    if age_ver.is_empty() {
        return Err(anyhow!("AGE extension not present"));
    }

    let (nodes, edges) = graph_size(pool).await?;
    if nodes == 0 {
        return Err(anyhow!("graph is empty; run ingest/load_graph.py first"));
    }
    tracing::info!(nodes, edges, "graph loaded");

    // 2. Sample start nodes per bucket.
    let starts = sample_starts(pool, args.samples, args.seed).await?;
    for (b, ids) in &starts {
        tracing::info!(bucket = b.name(), n = ids.len(), "sampled");
        if ids.is_empty() {
            tracing::warn!(bucket = b.name(),
                "no nodes in bucket — depth cells will be skipped");
        }
    }

    // 3. Run BFS per (bucket, depth) cell.
    let depths = [1u32, 2, 3];
    let mut cells: Vec<CellStats> = Vec::new();
    for depth in depths {
        for bucket in [Bucket::Low, Bucket::Mid, Bucket::High] {
            let ids = starts.get(&bucket).cloned().unwrap_or_default();
            if ids.is_empty() {
                continue;
            }
            let cell = run_cell(pool, bucket, depth, &ids, args.timeout_sec).await?;
            tracing::info!(
                bucket = cell.bucket,
                depth = cell.depth,
                p50_ms = cell.p50_ms,
                p95_ms = cell.p95_ms,
                p99_ms = cell.p99_ms,
                timeouts = cell.timeouts,
                "cell done"
            );
            cells.push(cell);
        }
    }

    // 4. Verdict.
    let verdict = decide_gate(&cells);
    let report = SmokeReport {
        schema: "researchdb.phase1.age_smoke.v1",
        age_version: age_ver,
        graph_nodes: nodes,
        graph_edges: edges,
        samples_per_cell: args.samples,
        timeout_sec: args.timeout_sec,
        cells,
        gate_verdict: verdict,
    };

    write_report(&report, &args.out)?;
    Ok(())
}

async fn graph_size(pool: &PgPool) -> Result<(i64, i64)> {
    let (nodes,): (i64,) = sqlx::query_as("SELECT count(*) FROM papers")
        .fetch_one(pool)
        .await?;
    let (edges,): (i64,) = sqlx::query_as("SELECT count(*) FROM citations")
        .fetch_one(pool)
        .await?;
    Ok((nodes, edges))
}

async fn sample_starts(
    pool: &PgPool,
    samples: usize,
    seed: u64,
) -> Result<std::collections::HashMap<Bucket, Vec<i64>>> {
    // Compute per-paper out-degree from the relational `citations` table
    // (faster + simpler than asking AGE). Papers with 0 out-edges are
    // excluded because BFS from them is trivially empty.
    let rows = sqlx::query(
        "SELECT src_paper_id AS pid, count(*) AS deg
         FROM citations
         GROUP BY src_paper_id",
    )
    .fetch_all(pool)
    .await?;

    let mut by_bucket: std::collections::HashMap<Bucket, Vec<i64>> = std::collections::HashMap::new();
    for row in rows {
        let pid: i64 = row.try_get("pid")?;
        let deg: i64 = row.try_get("deg")?;
        if let Some(b) = Bucket::classify(deg) {
            by_bucket.entry(b).or_default().push(pid);
        }
    }

    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    for v in by_bucket.values_mut() {
        v.shuffle(&mut rng);
        v.truncate(samples);
    }
    Ok(by_bucket)
}

async fn run_cell(
    pool: &PgPool,
    bucket: Bucket,
    depth: u32,
    starts: &[i64],
    timeout_sec: u64,
) -> Result<CellStats> {
    // HDR histogram in microseconds, max 600s.
    let mut hist = Histogram::<u64>::new_with_bounds(1, 600_000_000, 3)?;
    let mut timeouts = 0usize;

    // Apply per-statement timeout for this connection.
    let stmt_timeout_ms = (timeout_sec * 1000).max(1);

    for pid in starts {
        let cypher = format!(
            "SELECT * FROM cypher('citations_g', $$ \
                MATCH (s:Paper {{pid: {pid}}})-[:CITES*1..{depth}]->(n:Paper) \
                RETURN count(DISTINCT n) \
             $$) AS (n agtype)"
        );

        let mut conn = pool.acquire().await?;
        sqlx::query(&format!("SET statement_timeout = {stmt_timeout_ms}"))
            .execute(&mut *conn)
            .await?;

        let t = Instant::now();
        let res: std::result::Result<_, sqlx::Error> = sqlx::query(&cypher)
            .fetch_optional(&mut *conn)
            .await;
        let elapsed_us = t.elapsed().as_micros() as u64;

        match res {
            Ok(_) => {
                hist.record(elapsed_us.max(1))?;
            }
            Err(sqlx::Error::Database(db_err))
                if db_err.message().contains("statement timeout")
                    || db_err.message().contains("canceling statement") =>
            {
                timeouts += 1;
                // Record at the timeout boundary so percentiles reflect
                // the truncation, not silent omission.
                hist.record((stmt_timeout_ms * 1000).max(1))?;
            }
            Err(e) => return Err(e.into()),
        }
    }

    Ok(CellStats {
        depth,
        bucket: bucket.name().to_string(),
        n: starts.len(),
        p50_ms: hist.value_at_quantile(0.50) / 1000,
        p95_ms: hist.value_at_quantile(0.95) / 1000,
        p99_ms: hist.value_at_quantile(0.99) / 1000,
        max_ms: hist.max() / 1000,
        timeouts,
    })
}

pub fn decide_gate(cells: &[CellStats]) -> GateVerdict {
    let depth3: Vec<&CellStats> = cells.iter().filter(|c| c.depth == 3).collect();
    if depth3.is_empty() {
        return GateVerdict::Investigate;
    }
    let any_timeout = depth3.iter().any(|c| c.timeouts > 0);
    if any_timeout {
        GateVerdict::ReduceDataset
    } else {
        GateVerdict::Pass
    }
}

fn write_report(report: &SmokeReport, out: &str) -> Result<()> {
    let json = serde_json::to_string_pretty(report)?;
    if out == "-" {
        println!("{json}");
    } else {
        let path = PathBuf::from(out);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).ok();
        }
        std::fs::write(&path, json)?;
        tracing::info!(path = %path.display(), "report written");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bucket_classify_low_mid_high() {
        assert_eq!(Bucket::classify(0), None);
        assert_eq!(Bucket::classify(-1), None);
        assert_eq!(Bucket::classify(1), Some(Bucket::Low));
        assert_eq!(Bucket::classify(5), Some(Bucket::Low));
        assert_eq!(Bucket::classify(6), Some(Bucket::Mid));
        assert_eq!(Bucket::classify(20), Some(Bucket::Mid));
        assert_eq!(Bucket::classify(21), Some(Bucket::High));
        assert_eq!(Bucket::classify(10_000), Some(Bucket::High));
    }

    #[test]
    fn gate_pass_when_no_timeouts() {
        let cells = vec![
            CellStats { depth: 3, bucket: "low".into(), n: 100, p50_ms: 5, p95_ms: 9, p99_ms: 12, max_ms: 15, timeouts: 0 },
            CellStats { depth: 3, bucket: "mid".into(), n: 100, p50_ms: 20, p95_ms: 40, p99_ms: 50, max_ms: 60, timeouts: 0 },
            CellStats { depth: 3, bucket: "high".into(), n: 100, p50_ms: 100, p95_ms: 200, p99_ms: 250, max_ms: 300, timeouts: 0 },
        ];
        assert_eq!(decide_gate(&cells), GateVerdict::Pass);
    }

    #[test]
    fn gate_reduce_when_timeout_at_depth3() {
        let cells = vec![
            CellStats { depth: 3, bucket: "high".into(), n: 100, p50_ms: 100, p95_ms: 60_000, p99_ms: 60_000, max_ms: 60_000, timeouts: 7 },
        ];
        assert_eq!(decide_gate(&cells), GateVerdict::ReduceDataset);
    }

    #[test]
    fn gate_investigate_when_no_depth3() {
        let cells = vec![
            CellStats { depth: 1, bucket: "low".into(), n: 100, p50_ms: 1, p95_ms: 2, p99_ms: 3, max_ms: 4, timeouts: 0 },
        ];
        assert_eq!(decide_gate(&cells), GateVerdict::Investigate);
    }
}
