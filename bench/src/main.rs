//! ResearchDB Phase 1 benchmark harness.
//!
//! Subcommands:
//!   * `health`     — connect to DB, verify AGE + schema, print version info.
//!   * `age-smoke`  — Phase 1 Week 1 gate: BFS depth 1/2/3 latency across
//!                    out-degree buckets, JSON report to stdout/--out.
//!   * `seven`      — Run one (plan, query type) cell N times, JSON report.
//!                    Phase 1 D1/D2 work; multi-predicate Q4–Q7 currently
//!                    return stubs while orchestrator lands.

use anyhow::Result;
use clap::{Parser, Subcommand};
use serde::Serialize;
use std::time::Instant;

mod coldwarm;
mod concurrent;
mod cost;
mod db;
mod efsearch;
mod fusion;
mod graph_engine;
mod microbench;
mod plan;
mod query;
mod smoke;
mod storage;

#[derive(Parser, Debug)]
#[command(name = "researchdb-bench", version, about)]
struct Cli {
    /// Postgres DSN. Falls back to $DATABASE_URL.
    #[arg(long, env = "DATABASE_URL", global = true)]
    dsn: Option<String>,

    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// DB connectivity + extension check.
    Health,
    /// Apply embedded SQL migrations in order (idempotent).
    Migrate,
    /// Phase 1 Week 1 gate: AGE BFS latency smoke test.
    AgeSmoke(smoke::SmokeArgs),
    /// Phase 1 D1/D2: run one (plan, query) cell N times.
    Seven(SevenArgs),
    /// Phase 1 E2: ef_search latency–recall sweep.
    EfSweep(efsearch::EfSweepArgs),
    /// Phase 1 E5: storage overhead report.
    Storage(storage::StorageArgs),
    /// Phase 1 E1: cold vs warm latency report for one (plan, query).
    ColdWarm(coldwarm::ColdWarmArgs),
    /// Phase 1 D5: AGE cost-function micro-benchmark + model fit.
    MicroBenchAge(microbench::MicroArgs),
    /// Phase 1 D8: AGE vs WITH RECURSIVE BFS shootout.
    BfsShootout(graph_engine::ShootoutArgs),
    /// Phase 1 §9: concurrent workload — QPS + tail latency curve.
    Concurrent(concurrent::ConcurrentArgs),
}

#[derive(clap::Args, Debug)]
struct SevenArgs {
    /// 1..=7
    #[arg(long)]
    query: u8,
    /// "naive", "v1", or "v2"
    #[arg(long, default_value = "v1")]
    plan: String,
    /// Iterations per cell.
    #[arg(long, default_value_t = 30)]
    samples: usize,
    /// Top-k.
    #[arg(long, default_value_t = 10)]
    k: usize,
    /// HNSW ef_search (Q1/Q4/Q6/Q7 use this).
    #[arg(long, default_value_t = 40)]
    ef_search: u32,
    /// Graph BFS depth (Q3/Q4/Q5/Q7).
    #[arg(long, default_value_t = 2)]
    depth: u32,
    /// Seed chunk id (Q1/Q4/Q6/Q7). Defaults to chunk_id = 1.
    #[arg(long, default_value_t = 1)]
    seed_chunk: i64,
    /// Anchor paper id (Q3/Q4/Q5/Q7). Defaults to paper id 1.
    #[arg(long, default_value_t = 1)]
    anchor_paper: i64,
    /// BM25 text (Q2/Q5/Q6/Q7).
    #[arg(long, default_value = "cluster methodology")]
    bm25_text: String,
}

#[derive(Serialize)]
struct CellReport {
    plan: String,
    query: String,
    samples: usize,
    p50_us: u64,
    p95_us: u64,
    p99_us: u64,
    max_us: u64,
    first_result_count: usize,
    last_plan: plan::PlanResult,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_target(false)
        .compact()
        .init();

    let cli = Cli::parse();
    let dsn = cli
        .dsn
        .clone()
        .unwrap_or_else(|| "postgres://researchdb:researchdb@localhost:5432/researchdb".into());

    match cli.cmd {
        Cmd::Health => {
            let pool = db::connect(&dsn).await?;
            db::health_check(&pool).await?;
            println!("ok");
            Ok(())
        }
        Cmd::Migrate => {
            let pool = db::connect(&dsn).await?;
            db::run_migrations(&pool).await?;
            println!("ok");
            Ok(())
        }
        Cmd::AgeSmoke(args) => {
            let pool = db::connect(&dsn).await?;
            smoke::run(&pool, args).await
        }
        Cmd::Seven(args) => run_seven(&dsn, args).await,
        Cmd::EfSweep(args) => {
            let pool = db::connect(&dsn).await?;
            efsearch::run(&pool, args).await
        }
        Cmd::Storage(args) => {
            let pool = db::connect(&dsn).await?;
            storage::run(&pool, args).await
        }
        Cmd::ColdWarm(args) => coldwarm::run(&dsn, args).await,
        Cmd::MicroBenchAge(args) => {
            let pool = db::connect(&dsn).await?;
            microbench::run(&pool, args).await
        }
        Cmd::BfsShootout(args) => {
            let pool = db::connect(&dsn).await?;
            graph_engine::run(&pool, args).await
        }
        Cmd::Concurrent(args) => {
            let pool = db::connect(&dsn).await?;
            concurrent::run(&pool, args).await
        }
    }
}

async fn run_seven(dsn: &str, args: SevenArgs) -> Result<()> {
    let pool = db::connect(dsn).await?;
    let q = query::QueryType::from_u8(args.query)
        .ok_or_else(|| anyhow::anyhow!("query must be in 1..=7"))?;

    let spec = query::QuerySpec {
        seed_chunk_id: Some(args.seed_chunk),
        bm25_text:     Some(args.bm25_text.clone()),
        anchor_paper:  Some(args.anchor_paper),
        k:             args.k,
        depth:         args.depth,
        ef_search:     args.ef_search,
    };

    let plan_impl: Box<dyn plan::Plan + Send + Sync> = match args.plan.as_str() {
        "naive" => Box::new(plan::NaivePlan),
        "v1"    => Box::new(plan::V1Plan),
        "v2"    => Box::new(plan::V2Plan),
        "v3"    => Box::new(plan::V3Plan),
        other   => return Err(anyhow::anyhow!("unknown plan: {other}")),
    };
    tracing::debug!(plan = plan_impl.name(), query = q.as_str(),
                    samples = args.samples, "seven cell start");

    let mut hist = hdrhistogram::Histogram::<u64>::new_with_bounds(1, 600_000_000, 3)?;
    let mut first_count = 0usize;
    let mut last: Option<plan::PlanResult> = None;

    for i in 0..args.samples {
        let t = Instant::now();
        let res = plan_impl.execute(&pool, q, &spec).await?;
        let us = t.elapsed().as_micros() as u64;
        hist.record(us.max(1))?;
        if i == 0 { first_count = res.paper_ids.len(); }
        last = Some(res);
    }

    let report = CellReport {
        plan:    args.plan.clone(),
        query:   q.as_str().to_string(),
        samples: args.samples,
        p50_us:  hist.value_at_quantile(0.50),
        p95_us:  hist.value_at_quantile(0.95),
        p99_us:  hist.value_at_quantile(0.99),
        max_us:  hist.max(),
        first_result_count: first_count,
        last_plan: last.unwrap(),
    };

    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}
