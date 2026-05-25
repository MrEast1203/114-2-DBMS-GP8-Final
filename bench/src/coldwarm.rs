//! Cold vs warm latency report (Phase 1 §E1).
//!
//! Two "cold" strategies:
//!   * **fresh-conn** (default) — between each sample, open a brand-new
//!     PgConnection (not from the pool), run DISCARD ALL to clear
//!     session-side state, then time the query. This invalidates plan
//!     caches and prepared statements but leaves the OS page cache and
//!     PostgreSQL shared_buffers intact. Captures "first-query-of-session"
//!     latency, which is what most production callers actually see.
//!
//!   * **os-drop** — shells out to `docker exec <container> bash -c
//!     "sync && echo 3 > /proc/sys/vm/drop_caches"` before each sample.
//!     Drops the OS page cache; shared_buffers still holds blocks unless
//!     we restart Postgres (we do not). Captures lukewarm filesystem
//!     state. Requires Docker access from the host running this binary.
//!
//! Warm measurements run inside a single pooled connection, after a
//! configurable warmup phase. This is the production-cache-hit case.

use anyhow::{Context, Result};
use clap::Args;
use hdrhistogram::Histogram;
use serde::Serialize;
use sqlx::postgres::{PgConnectOptions, PgConnection, PgPool};
use sqlx::{ConnectOptions, Connection};
use std::path::PathBuf;
use std::process::Command;
use std::str::FromStr;
use std::time::Instant;

use crate::plan::{NaivePlan, Plan, V0Plan, V1Plan, V2Plan};
use crate::query::{QuerySpec, QueryType};

#[derive(Args, Debug, Clone)]
pub struct ColdWarmArgs {
    #[arg(long)] pub query: u8,
    #[arg(long, default_value = "v0")]    pub plan: String,
    #[arg(long, default_value_t = 30)]    pub samples: usize,
    #[arg(long, default_value_t = 5)]     pub warmup: usize,
    #[arg(long, default_value = "fresh-conn")]
    pub cold_method: String,
    #[arg(long, default_value = "researchdb-db")]
    pub container: String,
    // Query-spec args mirror `seven`.
    #[arg(long, default_value_t = 10)]    pub k: usize,
    #[arg(long, default_value_t = 40)]    pub ef_search: u32,
    #[arg(long, default_value_t = 2)]     pub depth: u32,
    #[arg(long, default_value_t = 1)]     pub seed_chunk: i64,
    #[arg(long, default_value_t = 1)]     pub anchor_paper: i64,
    #[arg(long, default_value = "neural network")]
    pub bm25_text: String,
    #[arg(long, default_value = "reports/cold_warm.json")]
    pub out: String,
}

#[derive(Serialize)]
struct Bucket {
    p50_us: u64,
    p95_us: u64,
    p99_us: u64,
    max_us: u64,
    n: usize,
}

#[derive(Serialize)]
struct Report {
    schema: &'static str,
    plan: String,
    query: String,
    cold_method: String,
    warmup: usize,
    cold: Bucket,
    warm: Bucket,
    cold_over_warm_p50: f64,
    cold_over_warm_p95: f64,
}

pub async fn run(dsn: &str, args: ColdWarmArgs) -> Result<()> {
    let q = QueryType::from_u8(args.query)
        .ok_or_else(|| anyhow::anyhow!("query must be 1..=7"))?;
    let spec = QuerySpec {
        seed_chunk_id: Some(args.seed_chunk),
        bm25_text:     Some(args.bm25_text.clone()),
        anchor_paper:  Some(args.anchor_paper),
        k: args.k,
        depth: args.depth,
        ef_search: args.ef_search,
    };
    let plan_kind = args.plan.clone();

    // ---- COLD ----
    let mut cold = Histogram::<u64>::new_with_bounds(1, 600_000_000, 3)?;
    for _ in 0..args.samples {
        if args.cold_method == "os-drop" {
            os_drop_caches(&args.container)?;
        } else if args.cold_method == "container-restart" {
            container_restart(&args.container)?;
        }
        // Open a fresh connection — bypasses pool reuse.
        let mut conn = open_fresh(dsn).await?;
        sqlx::query("DISCARD ALL").execute(&mut conn).await.ok();
        let pool = wrap_in_pool(dsn).await?;
        let plan_impl = make_plan(&plan_kind)?;
        let t = Instant::now();
        let _ = plan_impl.execute(&pool, q, &spec).await?;
        let us = t.elapsed().as_micros() as u64;
        cold.record(us.max(1))?;
        conn.close().await.ok();
    }

    // ---- WARM ----
    // Warmup first.
    let pool_warm = wrap_in_pool(dsn).await?;
    let plan_impl_warm = make_plan(&plan_kind)?;
    for _ in 0..args.warmup {
        let _ = plan_impl_warm.execute(&pool_warm, q, &spec).await?;
    }
    let mut warm = Histogram::<u64>::new_with_bounds(1, 600_000_000, 3)?;
    for _ in 0..args.samples {
        let t = Instant::now();
        let _ = plan_impl_warm.execute(&pool_warm, q, &spec).await?;
        let us = t.elapsed().as_micros() as u64;
        warm.record(us.max(1))?;
    }

    let cold_p50 = cold.value_at_quantile(0.50);
    let warm_p50 = warm.value_at_quantile(0.50);
    let cold_p95 = cold.value_at_quantile(0.95);
    let warm_p95 = warm.value_at_quantile(0.95);

    let report = Report {
        schema: "researchdb.phase1.cold_warm.v1",
        plan: plan_kind,
        query: q.as_str().to_string(),
        cold_method: args.cold_method.clone(),
        warmup: args.warmup,
        cold: Bucket {
            p50_us: cold_p50,
            p95_us: cold_p95,
            p99_us: cold.value_at_quantile(0.99),
            max_us: cold.max(),
            n: args.samples,
        },
        warm: Bucket {
            p50_us: warm_p50,
            p95_us: warm_p95,
            p99_us: warm.value_at_quantile(0.99),
            max_us: warm.max(),
            n: args.samples,
        },
        cold_over_warm_p50: cold_p50 as f64 / warm_p50.max(1) as f64,
        cold_over_warm_p95: cold_p95 as f64 / warm_p95.max(1) as f64,
    };

    let json = serde_json::to_string_pretty(&report)?;
    let path = PathBuf::from(&args.out);
    if let Some(p) = path.parent() { std::fs::create_dir_all(p).ok(); }
    std::fs::write(&path, &json)?;
    println!("{json}");
    Ok(())
}

async fn open_fresh(dsn: &str) -> Result<PgConnection> {
    let mut opts = PgConnectOptions::from_str(dsn)?;
    opts = opts.application_name("researchdb-bench-cold");
    let conn = opts.connect().await.context("fresh PgConnection")?;
    Ok(conn)
}

async fn wrap_in_pool(dsn: &str) -> Result<PgPool> {
    crate::db::connect(dsn).await
}

fn make_plan(name: &str) -> Result<Box<dyn Plan + Send + Sync>> {
    Ok(match name {
        "naive" => Box::new(NaivePlan),
        "v0"    => Box::new(V0Plan::new()),
        "v1"    => Box::new(V1Plan),
        "v2"    => Box::new(V2Plan),
        other   => return Err(anyhow::anyhow!("unknown plan {other}")),
    })
}

fn os_drop_caches(container: &str) -> Result<()> {
    // Requires docker on host PATH + container running. Inside the
    // container we need root, which is the default for postgres image.
    let out = Command::new("docker")
        .args(["exec", container, "bash", "-c", "sync && echo 3 > /proc/sys/vm/drop_caches"])
        .output()
        .context("invoke docker exec; is `docker` on PATH?")?;
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        return Err(anyhow::anyhow!("drop_caches failed: {stderr}"));
    }
    Ok(())
}

/// Portable "fully cold" simulation: restart the container. Postgres
/// shared_buffers is empty after restart; the host kernel page cache
/// usually retains data file pages, so this is "cold Postgres + warm
/// host disk cache" — closer to a true cold-start than fresh-conn but
/// not as severe as actual host page cache drop. Works on macOS / WSL
/// where /proc/sys/vm/drop_caches is read-only.
fn container_restart(container: &str) -> Result<()> {
    let restart = Command::new("docker")
        .args(["restart", container])
        .output()
        .context("invoke docker restart")?;
    if !restart.status.success() {
        let stderr = String::from_utf8_lossy(&restart.stderr);
        return Err(anyhow::anyhow!("docker restart failed: {stderr}"));
    }
    // Wait for healthcheck to pass — container takes a few seconds to
    // accept connections after restart.
    for _ in 0..30 {
        std::thread::sleep(std::time::Duration::from_secs(1));
        let probe = Command::new("docker")
            .args(["exec", container, "pg_isready", "-U", "researchdb"])
            .output();
        if let Ok(o) = probe { if o.status.success() { return Ok(()); } }
    }
    Err(anyhow::anyhow!("container did not become ready after restart"))
}
