//! Graph BFS engine shootout — AGE Cypher vs. PostgreSQL WITH RECURSIVE.
//!
//! Phase 1 §D8: validate whether the v2 push-down graph filter can use a
//! plain recursive CTE on `citations` instead of AGE. Both engines must
//! produce the same set of paper_ids for a given (anchor, depth); the
//! decision is made on latency.

use anyhow::{anyhow, Context, Result};
use clap::Args;
use hdrhistogram::Histogram;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use serde::Serialize;
use sqlx::postgres::PgPool;
use sqlx::Row;
use std::collections::HashSet;
use std::path::PathBuf;
use std::time::Instant;

/// Traversal direction for BFS over the directed citations graph.
/// `Forward` follows the edge as stored (src → dst means "src cites dst"),
/// so anchor → cited papers. `Reverse` follows edges backward, so anchor
/// → papers that cite anchor — the direction Q3/Q4/Q5/Q7 want.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    Forward,
    Reverse,
}

/// BFS via Apache AGE Cypher. Returns sorted unique paper_ids reachable
/// from `anchor` within `depth` hops (excluding the anchor itself).
pub async fn bfs_age(
    pool: &PgPool,
    anchor: i64,
    depth: u32,
    dir: Direction,
) -> Result<Vec<i64>> {
    if depth == 0 {
        return Ok(Vec::new());
    }
    // Forward: anchor -[:CITES*]-> p   (papers anchor cites)
    // Reverse: p -[:CITES*]-> anchor   (papers that cite anchor)
    let pattern = match dir {
        Direction::Forward => format!(
            "MATCH (s:Paper {{pid: {anchor}}})-[:CITES*1..{depth}]->(p:Paper)"
        ),
        Direction::Reverse => format!(
            "MATCH (p:Paper)-[:CITES*1..{depth}]->(a:Paper {{pid: {anchor}}})"
        ),
    };
    let cypher = format!(
        "SELECT (n::text)::bigint FROM cypher('citations_g', $$ \
            {pattern} \
            RETURN DISTINCT p.pid AS n \
         $$) AS (n agtype)"
    );
    let rows = sqlx::query(&cypher)
        .fetch_all(pool)
        .await
        .with_context(|| format!("AGE BFS failed (anchor={anchor}, depth={depth}, dir={dir:?})"))?;
    let mut out: Vec<i64> = rows
        .iter()
        .filter_map(|r| r.try_get::<i64, _>(0).ok())
        .collect();
    out.sort_unstable();
    out.dedup();
    Ok(out)
}

/// BFS via PostgreSQL WITH RECURSIVE on the `citations` table.
pub async fn bfs_recursive_sql(
    pool: &PgPool,
    anchor: i64,
    depth: u32,
    dir: Direction,
) -> Result<Vec<i64>> {
    if depth == 0 {
        return Ok(Vec::new());
    }
    // Forward: walk src → dst, starting from anchor as src.
    // Reverse: walk dst ← src, starting from anchor as dst.
    let sql = match dir {
        Direction::Forward => "\
            WITH RECURSIVE bfs(paper_id, d) AS ( \
                SELECT dst_paper_id, 1 FROM citations WHERE src_paper_id = $1 \
                UNION \
                SELECT c.dst_paper_id, b.d + 1 \
                FROM citations c JOIN bfs b ON c.src_paper_id = b.paper_id \
                WHERE b.d < $2 \
            ) \
            SELECT DISTINCT paper_id FROM bfs ORDER BY paper_id",
        Direction::Reverse => "\
            WITH RECURSIVE bfs(paper_id, d) AS ( \
                SELECT src_paper_id, 1 FROM citations WHERE dst_paper_id = $1 \
                UNION \
                SELECT c.src_paper_id, b.d + 1 \
                FROM citations c JOIN bfs b ON c.dst_paper_id = b.paper_id \
                WHERE b.d < $2 \
            ) \
            SELECT DISTINCT paper_id FROM bfs ORDER BY paper_id",
    };
    let rows = sqlx::query(sql)
        .bind(anchor)
        .bind(depth as i32)
        .fetch_all(pool)
        .await
        .with_context(|| format!("recursive SQL BFS failed (anchor={anchor}, depth={depth}, dir={dir:?})"))?;
    Ok(rows
        .iter()
        .filter_map(|r| r.try_get::<i64, _>(0).ok())
        .collect())
}

/// BFS via PostgreSQL WITH RECURSIVE, returning each reached `paper_id`
/// paired with its **minimum depth** from the anchor. Used by V3Plan
/// (Q4 / Q5 / Q7) to derive a `graph_distance_rank` — papers closer to
/// the anchor (smaller depth) rank higher, and the result is fed back
/// into RRF as a third ranking signal alongside vector and BM25.
///
/// The CTE already tracks `d` per row; we just keep the MIN per
/// paper_id (a paper reachable at depth 1 and depth 2 should count as
/// depth 1).
pub async fn bfs_recursive_sql_with_depth(
    pool: &PgPool,
    anchor: i64,
    depth: u32,
    dir: Direction,
) -> Result<Vec<(i64, i32)>> {
    if depth == 0 {
        return Ok(Vec::new());
    }
    let sql = match dir {
        Direction::Forward => "\
            WITH RECURSIVE bfs(paper_id, d) AS ( \
                SELECT dst_paper_id, 1 FROM citations WHERE src_paper_id = $1 \
                UNION \
                SELECT c.dst_paper_id, b.d + 1 \
                FROM citations c JOIN bfs b ON c.src_paper_id = b.paper_id \
                WHERE b.d < $2 \
            ) \
            SELECT paper_id, min(d)::int FROM bfs GROUP BY paper_id ORDER BY paper_id",
        Direction::Reverse => "\
            WITH RECURSIVE bfs(paper_id, d) AS ( \
                SELECT src_paper_id, 1 FROM citations WHERE dst_paper_id = $1 \
                UNION \
                SELECT c.src_paper_id, b.d + 1 \
                FROM citations c JOIN bfs b ON c.dst_paper_id = b.paper_id \
                WHERE b.d < $2 \
            ) \
            SELECT paper_id, min(d)::int FROM bfs GROUP BY paper_id ORDER BY paper_id",
    };
    let rows = sqlx::query(sql)
        .bind(anchor)
        .bind(depth as i32)
        .fetch_all(pool)
        .await
        .with_context(|| format!("recursive SQL BFS (with depth) failed (anchor={anchor}, depth={depth}, dir={dir:?})"))?;
    Ok(rows
        .iter()
        .filter_map(|r| Some((r.try_get::<i64, _>(0).ok()?, r.try_get::<i32, _>(1).ok()?)))
        .collect())
}

// ----- shootout subcommand -------------------------------------------------

#[derive(Args, Debug, Clone)]
pub struct ShootoutArgs {
    /// Anchors per (bucket, depth) cell.
    #[arg(long, default_value_t = 30)]
    pub samples: usize,
    /// Depths to test (comma separated).
    #[arg(long, default_value = "1,2,3")]
    pub depths: String,
    /// Per-query timeout (seconds). Cells exceeding this are recorded as TIMEOUT.
    #[arg(long, default_value_t = 60)]
    pub timeout_sec: u64,
    /// PRNG seed.
    #[arg(long, default_value_t = 42)]
    pub seed: u64,
    /// Output path (JSON). "-" means stdout-only.
    #[arg(long, default_value = "reports/bfs_shootout.json")]
    pub out: String,
    /// If true, also assert set-equivalence per anchor (slower but
    /// catches semantic divergence).
    #[arg(long, default_value_t = true)]
    pub check_equiv: bool,
    /// BFS direction: "forward" (anchor → cited) or "reverse"
    /// (papers citing anchor — what plan.rs uses).
    #[arg(long, default_value = "forward")]
    pub direction: String,
    /// Engines to run: "both", "age", "recursive". Use "recursive" when
    /// AGE projection is unavailable (e.g. 50K corpus where AGE edge
    /// projection times out) to measure recursive-SQL gate latency only.
    #[arg(long, default_value = "both")]
    pub engines: String,
}

#[derive(Serialize, Clone)]
pub struct ShootoutCell {
    pub depth: u32,
    pub bucket: &'static str,
    pub n_anchors: usize,
    pub age_p50_ms: f64,
    pub age_p95_ms: f64,
    pub age_p99_ms: f64,
    pub rec_p50_ms: f64,
    pub rec_p95_ms: f64,
    pub rec_p99_ms: f64,
    pub speedup_p50: f64, // age_p50 / rec_p50  — >1 means recursive faster
    pub equiv_mismatches: usize,
    pub mean_result_size: f64,
}

#[derive(Serialize)]
pub struct ShootoutReport {
    pub schema: &'static str,
    pub samples_per_cell: usize,
    pub cells: Vec<ShootoutCell>,
    pub overall_speedup_p50_geomean: f64,
    pub equiv_total_mismatches: usize,
    pub verdict: String,
}

pub async fn run(pool: &PgPool, args: ShootoutArgs) -> Result<()> {
    let depths: Vec<u32> = args
        .depths
        .split(',')
        .filter_map(|s| s.trim().parse::<u32>().ok())
        .collect();
    if depths.is_empty() {
        return Err(anyhow!("--depths produced no valid integers"));
    }
    let dir = match args.direction.as_str() {
        "forward" => Direction::Forward,
        "reverse" => Direction::Reverse,
        other => return Err(anyhow!("--direction must be 'forward' or 'reverse' (got {other})")),
    };
    let (run_age, run_rec) = match args.engines.as_str() {
        "both" => (true, true),
        "age" => (true, false),
        "recursive" => (false, true),
        other => return Err(anyhow!("--engines must be 'both', 'age', or 'recursive' (got {other})")),
    };

    let buckets: [(&'static str, i64, i64); 3] = [
        ("low", 1, 5),
        ("mid", 6, 20),
        ("high", 21, 100_000),
    ];
    let mut rng = rand::rngs::StdRng::seed_from_u64(args.seed);
    let mut cells: Vec<ShootoutCell> = Vec::new();
    let mut total_mismatches: usize = 0;

    // Anchor degree bucket is computed in the direction we'll traverse:
    // out-degree for forward, in-degree for reverse.
    let anchor_sql = match dir {
        Direction::Forward => "SELECT src_paper_id FROM citations \
                               GROUP BY src_paper_id \
                               HAVING count(*) BETWEEN $1 AND $2",
        Direction::Reverse => "SELECT dst_paper_id FROM citations \
                               GROUP BY dst_paper_id \
                               HAVING count(*) BETWEEN $1 AND $2",
    };
    for (label, lo, hi) in buckets {
        let rows = sqlx::query(anchor_sql)
        .bind(lo)
        .bind(hi)
        .fetch_all(pool)
        .await?;
        let mut anchors: Vec<i64> = rows
            .into_iter()
            .filter_map(|r| r.try_get::<i64, _>(0).ok())
            .collect();
        if anchors.is_empty() {
            tracing::warn!(bucket = label, "no anchors in bucket — skipping");
            continue;
        }
        anchors.shuffle(&mut rng);
        anchors.truncate(args.samples);

        for &depth in &depths {
            let mut h_age = Histogram::<u64>::new_with_bounds(1, 600_000_000, 3)?;
            let mut h_rec = Histogram::<u64>::new_with_bounds(1, 600_000_000, 3)?;
            let mut mismatches = 0usize;
            let mut sum_size = 0.0f64;
            let mut n_size = 0usize;

            for &anchor in &anchors {
                let mut age_set: Vec<i64> = Vec::new();
                if run_age {
                    let t = Instant::now();
                    age_set = bfs_age(pool, anchor, depth, dir).await?;
                    h_age.record((t.elapsed().as_micros() as u64).max(1))?;
                }

                let mut rec_set: Vec<i64> = Vec::new();
                if run_rec {
                    let t = Instant::now();
                    rec_set = bfs_recursive_sql(pool, anchor, depth, dir).await?;
                    h_rec.record((t.elapsed().as_micros() as u64).max(1))?;
                }

                if args.check_equiv && run_age && run_rec {
                    let a: HashSet<i64> = age_set.iter().copied().collect();
                    let r: HashSet<i64> = rec_set.iter().copied().collect();
                    if a != r {
                        mismatches += 1;
                        if mismatches <= 3 {
                            let only_age: Vec<_> = a.difference(&r).take(5).collect();
                            let only_rec: Vec<_> = r.difference(&a).take(5).collect();
                            tracing::warn!(
                                anchor, depth, bucket = label,
                                age_only = ?only_age, rec_only = ?only_rec,
                                "BFS set mismatch"
                            );
                        }
                    }
                }
                sum_size += rec_set.len() as f64;
                n_size += 1;
            }

            let cell = ShootoutCell {
                depth,
                bucket: label,
                n_anchors: anchors.len(),
                age_p50_ms: h_age.value_at_quantile(0.5) as f64 / 1000.0,
                age_p95_ms: h_age.value_at_quantile(0.95) as f64 / 1000.0,
                age_p99_ms: h_age.value_at_quantile(0.99) as f64 / 1000.0,
                rec_p50_ms: h_rec.value_at_quantile(0.5) as f64 / 1000.0,
                rec_p95_ms: h_rec.value_at_quantile(0.95) as f64 / 1000.0,
                rec_p99_ms: h_rec.value_at_quantile(0.99) as f64 / 1000.0,
                speedup_p50: if h_rec.value_at_quantile(0.5) > 0 {
                    h_age.value_at_quantile(0.5) as f64 / h_rec.value_at_quantile(0.5) as f64
                } else {
                    f64::NAN
                },
                equiv_mismatches: mismatches,
                mean_result_size: if n_size > 0 { sum_size / n_size as f64 } else { 0.0 },
            };
            tracing::info!(
                bucket = label, depth, n = anchors.len(),
                age_p50_ms = cell.age_p50_ms,
                rec_p50_ms = cell.rec_p50_ms,
                speedup = cell.speedup_p50,
                mismatches,
                "shootout cell"
            );
            total_mismatches += mismatches;
            cells.push(cell);
        }
    }

    // Geometric mean of speedup_p50 across cells (only where defined and >0).
    let speedups: Vec<f64> = cells
        .iter()
        .map(|c| c.speedup_p50)
        .filter(|s| s.is_finite() && *s > 0.0)
        .collect();
    let geom = if speedups.is_empty() {
        f64::NAN
    } else {
        let log_sum: f64 = speedups.iter().map(|s| s.ln()).sum();
        (log_sum / speedups.len() as f64).exp()
    };

    let verdict = if total_mismatches > 0 {
        format!(
            "MISMATCH — {total_mismatches} anchors disagree; keep AGE for correctness"
        )
    } else if !geom.is_finite() {
        "INSUFFICIENT_DATA".into()
    } else if geom >= 5.0 {
        format!("SWITCH_TO_RECURSIVE — geomean speedup {geom:.2}× ≥ 5×")
    } else if geom < 2.0 {
        format!("KEEP_AGE — geomean speedup {geom:.2}× < 2×")
    } else {
        format!("JUDGMENT_CALL — geomean speedup {geom:.2}× in 2–5× band")
    };

    let report = ShootoutReport {
        schema: "researchdb.phase1.bfs_shootout.v1",
        samples_per_cell: args.samples,
        cells,
        overall_speedup_p50_geomean: geom,
        equiv_total_mismatches: total_mismatches,
        verdict: verdict.clone(),
    };
    let json = serde_json::to_string_pretty(&report)?;
    if args.out != "-" {
        let path = PathBuf::from(&args.out);
        if let Some(p) = path.parent() {
            std::fs::create_dir_all(p).ok();
        }
        std::fs::write(&path, &json)?;
    }
    println!("{json}");
    tracing::info!(verdict, "shootout complete");
    Ok(())
}
