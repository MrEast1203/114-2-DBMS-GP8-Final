//! Phase 1 §9 limitation fix: concurrent workload benchmark.
//!
//! Spawns N tokio tasks running the same plan/query in parallel.
//! Measures throughput (QPS) and per-completion latency at the
//! aggregate level, so we can see how P50/P95/P99 degrade under
//! contention.
//!
//! Design notes
//! ------------
//! * Uses the shared sqlx pool (default max 10 connections in
//!   db::connect). Tasks beyond pool capacity will block on acquire,
//!   which is exactly the contention we want to surface.
//! * Each task runs for `--duration-sec` and records every per-query
//!   latency into a shared HdrHistogram protected by Mutex (cheap; the
//!   only contention is the tail record, μs-scale).
//! * Throughput is computed as total completions / wall clock.

use anyhow::Result;
use clap::Args;
use hdrhistogram::Histogram;
use serde::Serialize;
use sqlx::postgres::PgPool;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Mutex;

use crate::plan::{self, NaivePlan, V1Plan, V2Plan};
use crate::query::{QuerySpec, QueryType};

#[derive(Args, Debug, Clone)]
pub struct ConcurrentArgs {
    #[arg(long)] pub query: u8,
    #[arg(long, default_value = "v1")] pub plan: String,
    /// Comma-separated concurrency levels to test, e.g. "1,4,8,16".
    #[arg(long, default_value = "1,4,8,16,32")]
    pub concurrency: String,
    /// Seconds each concurrency level runs.
    #[arg(long, default_value_t = 10)] pub duration_sec: u64,
    /// pgvector / pg_search args (shared).
    #[arg(long, default_value_t = 10)] pub k: usize,
    #[arg(long, default_value_t = 40)] pub ef_search: u32,
    #[arg(long, default_value_t = 2)]  pub depth: u32,
    #[arg(long, default_value_t = 1)]  pub seed_chunk: i64,
    #[arg(long, default_value_t = 1)]  pub anchor_paper: i64,
    #[arg(long, default_value = "neural network")] pub bm25_text: String,
    #[arg(long, default_value = "reports/concurrent.json")] pub out: String,
}

#[derive(Serialize)]
pub struct ConcurrentLevel {
    pub concurrency: usize,
    pub duration_sec: f64,
    pub completed: u64,
    pub qps: f64,
    pub p50_ms: f64,
    pub p95_ms: f64,
    pub p99_ms: f64,
    pub max_ms: f64,
}

#[derive(Serialize)]
pub struct ConcurrentReport {
    pub schema: &'static str,
    pub query: String,
    pub plan: String,
    pub levels: Vec<ConcurrentLevel>,
}

pub async fn run(pool: &PgPool, args: ConcurrentArgs) -> Result<()> {
    let q = QueryType::from_u8(args.query)
        .ok_or_else(|| anyhow::anyhow!("query must be 1..=7"))?;
    let levels: Vec<usize> = args.concurrency.split(',')
        .filter_map(|s| s.trim().parse().ok()).collect();
    if levels.is_empty() {
        return Err(anyhow::anyhow!("--concurrency produced no valid integers"));
    }

    let spec = Arc::new(QuerySpec {
        seed_chunk_id: Some(args.seed_chunk),
        bm25_text:     Some(args.bm25_text.clone()),
        anchor_paper:  Some(args.anchor_paper),
        k:             args.k,
        depth:         args.depth,
        ef_search:     args.ef_search,
    });

    let mut report_levels = Vec::new();
    for &c in &levels {
        tracing::info!(concurrency = c, "starting level");
        let hist = Arc::new(Mutex::new(
            Histogram::<u64>::new_with_bounds(1, 600_000_000, 3)?
        ));
        let stop_at = Instant::now() + Duration::from_secs(args.duration_sec);
        let mut tasks = Vec::with_capacity(c);
        for _ in 0..c {
            let pool = pool.clone();
            let spec = spec.clone();
            let hist = hist.clone();
            let plan_name = args.plan.clone();
            tasks.push(tokio::spawn(async move {
                let plan_impl = make_plan(&plan_name).expect("plan name");
                let mut local_completions: u64 = 0;
                while Instant::now() < stop_at {
                    let t = Instant::now();
                    if plan_impl.execute(&pool, q, &spec).await.is_err() {
                        continue;
                    }
                    let us = t.elapsed().as_micros() as u64;
                    let mut h = hist.lock().await;
                    let _ = h.record(us.max(1));
                    local_completions += 1;
                }
                local_completions
            }));
        }
        let wall_start = Instant::now();
        let mut total: u64 = 0;
        for t in tasks { total += t.await.unwrap_or(0); }
        let elapsed = wall_start.elapsed().as_secs_f64();
        let h = hist.lock().await;
        let lvl = ConcurrentLevel {
            concurrency: c,
            duration_sec: elapsed,
            completed: total,
            qps: total as f64 / elapsed,
            p50_ms: h.value_at_quantile(0.50) as f64 / 1000.0,
            p95_ms: h.value_at_quantile(0.95) as f64 / 1000.0,
            p99_ms: h.value_at_quantile(0.99) as f64 / 1000.0,
            max_ms: h.max() as f64 / 1000.0,
        };
        tracing::info!(
            concurrency = c, qps = lvl.qps,
            p50_ms = lvl.p50_ms, p95_ms = lvl.p95_ms, p99_ms = lvl.p99_ms,
            "level done"
        );
        report_levels.push(lvl);
    }

    let report = ConcurrentReport {
        schema: "researchdb.phase1.concurrent.v1",
        query: q.as_str().to_string(),
        plan: args.plan.clone(),
        levels: report_levels,
    };
    let json = serde_json::to_string_pretty(&report)?;
    let path = PathBuf::from(&args.out);
    if let Some(p) = path.parent() { std::fs::create_dir_all(p).ok(); }
    std::fs::write(&path, &json)?;
    println!("{json}");
    Ok(())
}

fn make_plan(name: &str) -> Result<Box<dyn plan::Plan + Send + Sync>> {
    Ok(match name {
        "naive" => Box::new(NaivePlan),
        "v1"    => Box::new(V1Plan),
        "v2"    => Box::new(V2Plan),
        other   => return Err(anyhow::anyhow!("unknown plan {other}")),
    })
}
