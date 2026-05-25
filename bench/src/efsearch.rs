//! ef_search latency–recall sweep (Phase 1 §E2).
//!
//! For a set of test queries (Q1 semantic), runs the HNSW retrieval at
//! ef_search ∈ {10, 20, 40, 80, 160, 320, 640} and records both latency
//! and recall@10 against a "gold" run at ef_search = 1000.
//!
//! Result feeds the writeup's adaptive-ef_search claim: per-query-type
//! optima differ, so the orchestrator should pick ef_search at plan
//! time, not fix it globally.

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
pub struct EfSweepArgs {
    /// Seed chunks to use as queries. Sampled from chunks with non-null
    /// embeddings.
    #[arg(long, default_value_t = 30)]
    pub n_queries: usize,

    /// Iterations per (ef_search, query) cell.
    #[arg(long, default_value_t = 20)]
    pub samples: usize,

    /// Top-k.
    #[arg(long, default_value_t = 10)]
    pub k: usize,

    /// PRNG seed for choosing test queries.
    #[arg(long, default_value_t = 42)]
    pub seed: u64,

    /// Where to write the JSON report. "-" writes to stdout.
    #[arg(long, default_value = "reports/ef_sweep.json")]
    pub out: String,
}

#[derive(Serialize)]
pub struct CellSweep {
    pub ef_search: u32,
    pub p50_us: u64,
    pub p95_us: u64,
    pub p99_us: u64,
    pub mean_recall_at_10: f64,
}

#[derive(Serialize)]
pub struct EfSweepReport {
    pub schema:    &'static str,
    pub n_queries: usize,
    pub samples:   usize,
    pub k:         usize,
    pub corpus_size: i64,
    pub gold_ef_search: u32,
    pub cells: Vec<CellSweep>,
}

pub async fn run(pool: &PgPool, args: EfSweepArgs) -> Result<()> {
    let (n_chunks,): (i64,) = sqlx::query_as(
        "SELECT count(*) FROM chunk_embeddings WHERE embedding IS NOT NULL"
    )
    .fetch_one(pool)
    .await?;
    tracing::info!(n_chunks, "corpus");

    // Sample query chunks deterministically.
    let mut rng = rand::rngs::StdRng::seed_from_u64(args.seed);
    let mut all: Vec<i64> = sqlx::query(
        "SELECT chunk_id FROM chunk_embeddings WHERE embedding IS NOT NULL ORDER BY chunk_id"
    )
    .fetch_all(pool).await?
    .into_iter().filter_map(|r| r.try_get::<i64,_>(0).ok()).collect();
    all.shuffle(&mut rng);
    let queries: Vec<i64> = all.into_iter().take(args.n_queries).collect();
    tracing::info!(n = queries.len(), "queries chosen");

    // Gold standard at ef_search = 1000 — within a session, but high
    // enough to approximate exact-NN for k=10 on a 5K-vector corpus.
    let gold_ef: u32 = 1000;
    let mut gold: Vec<Vec<i64>> = Vec::with_capacity(queries.len());
    for &q in &queries {
        gold.push(top_k(pool, q, args.k, gold_ef).await?);
    }
    tracing::info!("gold computed");

    let efs = [10u32, 20, 40, 80, 160, 320, 640];
    let mut cells: Vec<CellSweep> = Vec::with_capacity(efs.len());

    for &ef in &efs {
        let mut hist = Histogram::<u64>::new_with_bounds(1, 60_000_000, 3)?;
        let mut recalls: Vec<f64> = Vec::with_capacity(queries.len());

        for (i, &q) in queries.iter().enumerate() {
            // Run `samples` timings; for recall measure once (deterministic).
            for _ in 0..args.samples {
                let t = Instant::now();
                let _ = top_k(pool, q, args.k, ef).await?;
                let us = t.elapsed().as_micros() as u64;
                hist.record(us.max(1))?;
            }
            // Measure recall once.
            let result = top_k(pool, q, args.k, ef).await?;
            let g: std::collections::HashSet<i64> = gold[i].iter().copied().collect();
            let hits = result.iter().filter(|p| g.contains(p)).count();
            let r = hits as f64 / args.k as f64;
            recalls.push(r);
        }
        let mean_recall = recalls.iter().sum::<f64>() / recalls.len() as f64;
        let cell = CellSweep {
            ef_search: ef,
            p50_us: hist.value_at_quantile(0.50),
            p95_us: hist.value_at_quantile(0.95),
            p99_us: hist.value_at_quantile(0.99),
            mean_recall_at_10: mean_recall,
        };
        tracing::info!(
            ef, p50_ms = cell.p50_us / 1000, p95_ms = cell.p95_us / 1000,
            recall10 = mean_recall, "cell"
        );
        cells.push(cell);
    }

    let report = EfSweepReport {
        schema: "researchdb.phase1.ef_sweep.v1",
        n_queries: queries.len(),
        samples: args.samples,
        k: args.k,
        corpus_size: n_chunks,
        gold_ef_search: gold_ef,
        cells,
    };
    let json = serde_json::to_string_pretty(&report)?;
    if args.out == "-" {
        println!("{json}");
    } else {
        let path = PathBuf::from(&args.out);
        if let Some(p) = path.parent() { std::fs::create_dir_all(p).ok(); }
        std::fs::write(&path, json)?;
        tracing::info!(path = %path.display(), "report written");
    }
    Ok(())
}

async fn top_k(pool: &PgPool, seed_chunk: i64, k: usize, ef: u32) -> Result<Vec<i64>> {
    // SET LOCAL only works inside a transaction; without one Postgres
    // raises WARNING and the GUC is not applied. Open an explicit tx
    // so ef_search actually takes effect for this single query.
    let mut tx = pool.begin().await?;
    sqlx::query(&format!("SET LOCAL hnsw.ef_search = {ef}"))
        .execute(&mut *tx).await?;
    let rows = sqlx::query(
        "WITH seed AS (SELECT embedding FROM chunk_embeddings WHERE chunk_id = $1) \
         SELECT ce.chunk_id FROM chunk_embeddings ce, seed \
         ORDER BY ce.embedding <=> seed.embedding LIMIT $2",
    )
    .bind(seed_chunk)
    .bind(k as i64)
    .fetch_all(&mut *tx).await?;
    tx.commit().await?;
    Ok(rows.into_iter().filter_map(|r| r.try_get::<i64,_>(0).ok()).collect())
}
