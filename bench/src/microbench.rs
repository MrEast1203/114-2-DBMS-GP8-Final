//! BFS cost-function micro-benchmark.
//!
//! BFS frontier grows geometrically — at depth d with average
//! branching b, the cost should scale as `b^d` (depth 1 visits b
//! nodes, depth 2 visits b², depth 3 visits b³). This subcommand
//! confirms that empirically: runs BFS depth ∈ {1, 2, 3} from anchors
//! sampled across three out-degree buckets (low / mid / high), records
//! P50 latency, and fits two candidate single-coefficient models
//! against the data — a linear baseline and the exponential model — so
//! the residual difference can be reported numerically. The
//! exponential form is what `cost::age_cost_v1` uses.

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
    pub linear_residual_sse:      f64,
    pub exponential_residual_sse: f64,
    pub better:                   &'static str,
    pub linear_coef:              f64,
    pub exponential_coef:         f64,
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
    let mut sum_lin = 0.0; let mut sumsq_lin = 0.0;
    let mut sum_exp = 0.0; let mut sumsq_exp = 0.0;
    for c in &cells {
        let y = c.p50_ms;
        let x_lin = c.depth as f64 * c.branching;
        let x_exp = c.branching.powf(c.depth as f64);
        sum_lin += y * x_lin;  sumsq_lin += x_lin * x_lin;
        sum_exp += y * x_exp;  sumsq_exp += x_exp * x_exp;
    }
    let linear_coef      = if sumsq_lin > 0.0 { sum_lin / sumsq_lin } else { 0.0 };
    let exponential_coef = if sumsq_exp > 0.0 { sum_exp / sumsq_exp } else { 0.0 };

    // Residual SSE per model.
    let mut sse_lin = 0.0; let mut sse_exp = 0.0;
    for c in &cells {
        let y = c.p50_ms;
        let pred_lin = linear_coef      * (c.depth as f64 * c.branching);
        let pred_exp = exponential_coef * c.branching.powf(c.depth as f64);
        sse_lin += (y - pred_lin).powi(2);
        sse_exp += (y - pred_exp).powi(2);
    }
    let better = if sse_exp < sse_lin { "exponential" } else { "linear" };

    let report = Report {
        schema: "researchdb.phase1.microbench_age.v2",
        cells: cells.clone(),
        linear_residual_sse:      sse_lin,
        exponential_residual_sse: sse_exp,
        better,
        linear_coef,
        exponential_coef,
    };

    let json = serde_json::to_string_pretty(&report)?;
    let path = PathBuf::from(&args.out);
    if let Some(p) = path.parent() { std::fs::create_dir_all(p).ok(); }
    std::fs::write(&path, &json)?;
    println!("{json}");
    tracing::info!(
        linear_sse = sse_lin, exponential_sse = sse_exp, better,
        "BFS cost model fit complete"
    );
    Ok(())
}
