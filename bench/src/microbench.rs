//! AGE cost-function micro-benchmark (Phase 1 §D5 — first iteration).
//!
//! v0's cost model is `cost = depth × branching`, a textbook linear-in-
//! both estimate. Real AGE BFS latency on our 5K-paper graph grows much
//! faster than that — visible in the §A1 smoke report (depth-3 high
//! P50 is ~1.6 s, while v0 predicts only 3× over depth-1).
//!
//! This subcommand runs BFS depth ∈ {1, 2, 3} from anchors sampled
//! across three out-degree buckets (low / mid / high), records P50
//! latency, and fits two candidate models against the data:
//!
//!   1. **v0 (linear)**: `cost ∝ depth × branching`
//!   2. **v1 (exponential)**: `cost ∝ branching^depth`
//!
//! Output a JSON report and a log line that suggests the better fit.

use anyhow::Result;
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
pub struct MicroArgs {
    #[arg(long, default_value_t = 30)] pub samples: usize,
    #[arg(long, default_value_t = 42)] pub seed: u64,
    #[arg(long, default_value = "reports/microbench_age.json")]
    pub out: String,
}

#[derive(Serialize, Clone)]
pub struct Cell {
    pub depth: u32,
    pub bucket: &'static str,
    pub branching: f64,    // mean out-degree for sampled anchors
    pub p50_ms: f64,
    pub n_anchors: usize,
}

#[derive(Serialize)]
pub struct Report {
    pub schema: &'static str,
    pub cells:  Vec<Cell>,
    pub v0_residual_sse:  f64,
    pub v1_residual_sse:  f64,
    pub better:           &'static str,
    pub v0_coef:          f64,
    pub v1_coef:          f64,
}

pub async fn run(pool: &PgPool, args: MicroArgs) -> Result<()> {
    let buckets = [
        ("low",  1, 5),
        ("mid",  6, 20),
        ("high", 21, 100_000),
    ];

    let mut rng = rand::rngs::StdRng::seed_from_u64(args.seed);
    let mut cells: Vec<Cell> = Vec::new();

    for (label, lo, hi) in buckets {
        // Sample anchors in this out-degree bucket.
        let rows = sqlx::query(
            "SELECT src_paper_id, count(*) AS deg \
             FROM citations \
             GROUP BY src_paper_id \
             HAVING count(*) BETWEEN $1 AND $2"
        )
        .bind(lo as i64).bind(hi as i64)
        .fetch_all(pool).await?;

        let mut anchors: Vec<(i64, i64)> = rows.into_iter()
            .filter_map(|r| Some((r.try_get::<i64,_>(0).ok()?, r.try_get::<i64,_>(1).ok()?)))
            .collect();
        anchors.shuffle(&mut rng);
        let anchors: Vec<(i64, i64)> = anchors.into_iter().take(args.samples).collect();

        if anchors.is_empty() {
            tracing::warn!(bucket = label, "no anchors in bucket");
            continue;
        }
        let mean_branching: f64 = anchors.iter().map(|(_, d)| *d as f64).sum::<f64>()
            / anchors.len() as f64;

        for depth in [1u32, 2, 3] {
            let mut hist = Histogram::<u64>::new_with_bounds(1, 60_000_000, 3)?;
            for (anchor, _) in &anchors {
                let cypher = format!(
                    "SELECT count(*)::bigint FROM cypher('citations_g', $$ \
                        MATCH (s:Paper {{pid: {anchor}}})-[:CITES*1..{depth}]->(n:Paper) \
                        RETURN count(DISTINCT n) \
                     $$) AS (n agtype)"
                );
                let t = Instant::now();
                let _ = sqlx::query(&cypher).fetch_optional(pool).await?;
                hist.record((t.elapsed().as_micros() as u64).max(1))?;
            }
            let p50_ms = hist.value_at_quantile(0.5) as f64 / 1000.0;
            tracing::info!(bucket = label, depth, branching = mean_branching,
                           p50_ms, "cell");
            cells.push(Cell {
                depth,
                bucket: label,
                branching: mean_branching,
                p50_ms,
                n_anchors: anchors.len(),
            });
        }
    }

    // Fit two models. Both are single-coefficient: latency = k × predictor.
    // Fit k by least squares: k = Σ(y x) / Σ(x²).
    let mut sum_v0 = 0.0; let mut sumsq_v0 = 0.0;
    let mut sum_v1 = 0.0; let mut sumsq_v1 = 0.0;
    for c in &cells {
        let y = c.p50_ms;
        let x0 = c.depth as f64 * c.branching;
        let x1 = c.branching.powf(c.depth as f64);
        sum_v0 += y * x0;  sumsq_v0 += x0 * x0;
        sum_v1 += y * x1;  sumsq_v1 += x1 * x1;
    }
    let v0_coef = if sumsq_v0 > 0.0 { sum_v0 / sumsq_v0 } else { 0.0 };
    let v1_coef = if sumsq_v1 > 0.0 { sum_v1 / sumsq_v1 } else { 0.0 };

    // Residual SSE per model.
    let mut sse0 = 0.0; let mut sse1 = 0.0;
    for c in &cells {
        let y = c.p50_ms;
        let pred0 = v0_coef * (c.depth as f64 * c.branching);
        let pred1 = v1_coef * c.branching.powf(c.depth as f64);
        sse0 += (y - pred0).powi(2);
        sse1 += (y - pred1).powi(2);
    }
    let better = if sse1 < sse0 { "v1 (branching^depth)" } else { "v0 (depth × branching)" };

    let report = Report {
        schema: "researchdb.phase1.microbench_age.v1",
        cells: cells.clone(),
        v0_residual_sse: sse0,
        v1_residual_sse: sse1,
        better,
        v0_coef,
        v1_coef,
    };

    let json = serde_json::to_string_pretty(&report)?;
    let path = PathBuf::from(&args.out);
    if let Some(p) = path.parent() { std::fs::create_dir_all(p).ok(); }
    std::fs::write(&path, &json)?;
    println!("{json}");
    tracing::info!(
        v0_sse = sse0, v1_sse = sse1, better,
        "AGE cost model fit complete"
    );
    Ok(())
}
