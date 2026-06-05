//! `serve` — a tiny HTTP server that puts the four-plan orchestrator
//! behind a browser GUI so the project can be *run*, not just measured.
//!
//! It drives the exact same `plan::{NaivePlan,V1Plan,V2Plan,V3Plan}`
//! code paths as the `seven` benchmark subcommand, so what the GUI shows
//! is what the orchestrator actually does — engine execution order,
//! graph push-down, per-engine row counts, round-trips, and the fused
//! top-K papers with their titles.
//!
//! Deliberately dependency-free: a minimal single-request-per-connection
//! HTTP/1.1 server on `tokio::net` (no axum/hyper), which keeps the
//! research binary's dependency tree unchanged. It is meant for local,
//! single-user use on `localhost`, not as a production server.

use anyhow::{Context, Result};
use serde::Deserialize;
use serde_json::{json, Value};
use sqlx::postgres::PgPool;
use sqlx::Row;
use std::time::Instant;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

use crate::cost::Engine;
use crate::plan::{self, Plan};
use crate::query::{QuerySpec, QueryType};

/// The interactive GUI, baked into the binary so `serve` needs no files
/// on disk at runtime.
const APP_HTML: &str = include_str!("../../web/app.html");
/// The 20 curated eval queries, shipped as one-click presets.
const PRESETS_JSONL: &str = include_str!("../../eval/queries.jsonl");

#[derive(clap::Args, Debug)]
pub struct ServeArgs {
    /// Port to listen on (localhost).
    #[arg(long, default_value_t = 8080)]
    pub port: u16,
}

pub async fn run(pool: PgPool, args: ServeArgs) -> Result<()> {
    let addr = format!("127.0.0.1:{}", args.port);
    let listener = TcpListener::bind(&addr)
        .await
        .with_context(|| format!("binding {addr}"))?;

    println!("researchdb GUI → http://{addr}");
    println!("(Ctrl-C to stop)");

    loop {
        let (stream, _peer) = match listener.accept().await {
            Ok(x) => x,
            Err(e) => {
                tracing::warn!(error = %e, "accept failed");
                continue;
            }
        };
        let pool = pool.clone();
        tokio::spawn(async move {
            if let Err(e) = handle(stream, pool).await {
                tracing::debug!(error = %e, "connection handler error");
            }
        });
    }
}

// ---------------------------------------------------------------------
// Minimal HTTP plumbing
// ---------------------------------------------------------------------

struct Req {
    method: String,
    path: String,
    body: Vec<u8>,
}

async fn read_request(stream: &mut TcpStream) -> Result<Option<Req>> {
    let mut buf: Vec<u8> = Vec::with_capacity(8192);
    let mut tmp = [0u8; 8192];

    loop {
        // Do we already have the full header block?
        if let Some(hdr_end) = find(&buf, b"\r\n\r\n") {
            let head = String::from_utf8_lossy(&buf[..hdr_end]).to_string();
            let content_length = head
                .lines()
                .find_map(|l| {
                    let (k, v) = l.split_once(':')?;
                    if k.trim().eq_ignore_ascii_case("content-length") {
                        v.trim().parse::<usize>().ok()
                    } else {
                        None
                    }
                })
                .unwrap_or(0);
            let body_start = hdr_end + 4;
            if buf.len() - body_start >= content_length {
                let mut lines = head.lines();
                let first = lines.next().unwrap_or("");
                let mut parts = first.split_whitespace();
                let method = parts.next().unwrap_or("").to_string();
                let path = parts.next().unwrap_or("/").to_string();
                let body = buf[body_start..body_start + content_length].to_vec();
                return Ok(Some(Req { method, path, body }));
            }
        }
        let n = stream.read(&mut tmp).await?;
        if n == 0 {
            return Ok(None); // client closed before a full request
        }
        buf.extend_from_slice(&tmp[..n]);
        if buf.len() > 1 << 20 {
            anyhow::bail!("request too large");
        }
    }
}

fn find(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|w| w == needle)
}

async fn write_response(
    stream: &mut TcpStream,
    status: &str,
    content_type: &str,
    body: &[u8],
) -> Result<()> {
    let head = format!(
        "HTTP/1.1 {status}\r\n\
         Content-Type: {content_type}\r\n\
         Content-Length: {}\r\n\
         Cache-Control: no-store\r\n\
         Connection: close\r\n\r\n",
        body.len()
    );
    stream.write_all(head.as_bytes()).await?;
    stream.write_all(body).await?;
    stream.flush().await?;
    Ok(())
}

async fn handle(mut stream: TcpStream, pool: PgPool) -> Result<()> {
    let req = match read_request(&mut stream).await? {
        Some(r) => r,
        None => return Ok(()),
    };

    match (req.method.as_str(), req.path.as_str()) {
        ("GET", "/") | ("GET", "/index.html") => {
            write_response(&mut stream, "200 OK", "text/html; charset=utf-8", APP_HTML.as_bytes())
                .await
        }
        ("GET", "/presets") => {
            let body = presets_json().to_string();
            write_response(&mut stream, "200 OK", "application/json", body.as_bytes()).await
        }
        ("POST", "/search") => {
            let body = match run_search(&pool, &req.body).await {
                Ok(v) => v,
                Err(e) => json!({ "ok": false, "error": e.to_string() }),
            };
            write_response(&mut stream, "200 OK", "application/json", body.to_string().as_bytes())
                .await
        }
        _ => write_response(&mut stream, "404 Not Found", "text/plain", b"not found").await,
    }
}

// ---------------------------------------------------------------------
// Presets — parse the committed eval queries into a friendly list
// ---------------------------------------------------------------------

fn presets_json() -> Value {
    let mut out = Vec::new();
    for line in PRESETS_JSONL.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let Ok(mut v) = serde_json::from_str::<Value>(line) else { continue };
        // type "Q4" → numeric 4 for the form.
        if let Some(t) = v.get("type").and_then(|t| t.as_str()) {
            if let Some(n) = t.strip_prefix('Q').and_then(|s| s.parse::<u8>().ok()) {
                v["qtype"] = json!(n);
            }
        }
        out.push(v);
    }
    json!({ "ok": true, "presets": out })
}

// ---------------------------------------------------------------------
// Search — run the real orchestrator and serialize what it did
// ---------------------------------------------------------------------

#[derive(Deserialize)]
struct SearchReq {
    query: u8,
    plan: String,
    #[serde(default)]
    k: Option<usize>,
    #[serde(default)]
    depth: Option<u32>,
    #[serde(default)]
    ef_search: Option<u32>,
    #[serde(default)]
    seed_chunk: Option<i64>,
    #[serde(default)]
    anchor_paper: Option<i64>,
    #[serde(default)]
    bm25_text: Option<String>,
    #[serde(default)]
    samples: Option<usize>,
}

async fn run_search(pool: &PgPool, body: &[u8]) -> Result<Value> {
    let req: SearchReq = serde_json::from_slice(body).context("invalid request body")?;

    let q = QueryType::from_u8(req.query)
        .ok_or_else(|| anyhow::anyhow!("query must be in 1..=7"))?;
    let k = req.k.unwrap_or(10).clamp(1, 50);
    let samples = req.samples.unwrap_or(5).clamp(1, 25);

    let spec = QuerySpec {
        seed_chunk_id: Some(req.seed_chunk.unwrap_or(1)),
        bm25_text: req.bm25_text.clone().filter(|s| !s.trim().is_empty()),
        anchor_paper: Some(req.anchor_paper.unwrap_or(1)),
        k,
        depth: req.depth.unwrap_or(2).clamp(1, 3),
        ef_search: req.ef_search.unwrap_or(40).clamp(10, 400),
    };

    let plan_impl: Box<dyn Plan + Send + Sync> = match req.plan.as_str() {
        "naive" => Box::new(plan::NaivePlan),
        "v1" => Box::new(plan::V1Plan),
        "v2" => Box::new(plan::V2Plan),
        "v3" => Box::new(plan::V3Plan),
        other => anyhow::bail!("unknown plan: {other}"),
    };

    // One warm-up (HNSW / BM25 caches), then `samples` measured runs.
    let _ = plan_impl.execute(pool, q, &spec).await?;
    let mut durations_us: Vec<u64> = Vec::with_capacity(samples);
    let mut last = None;
    for _ in 0..samples {
        let t = Instant::now();
        let res = plan_impl.execute(pool, q, &spec).await?;
        durations_us.push(t.elapsed().as_micros() as u64);
        last = Some(res);
    }
    let res = last.unwrap();

    // Enrich the top-K paper ids with titles in the orchestrator's order.
    let hits = enrich(pool, &res.paper_ids, k).await?;

    let (sem, lex, gph) = q.engines();
    let p50 = pct(&durations_us, 0.50);
    let p95 = pct(&durations_us, 0.95);
    let pmin = *durations_us.iter().min().unwrap_or(&0);

    Ok(json!({
        "ok": true,
        "plan": res.plan,
        "query": q.as_str(),
        "qtype": req.query,
        "engines": { "semantic": sem, "lexical": lex, "graph": gph },
        "timing_ms": {
            "p50": p50 as f64 / 1000.0,
            "p95": p95 as f64 / 1000.0,
            "min": pmin as f64 / 1000.0,
            "samples": samples,
        },
        "orchestration": {
            "actual_order": res.actual_order.iter().map(|e| engine_label(*e)).collect::<Vec<_>>(),
            "first_predicate": res.first_predicate.map(engine_label),
            "round_trips": res.round_trips,
            "materializations": res.materializations,
            "pushdown": res.materializations > 0 && gph && (sem || lex),
            "per_engine_rows": res.per_engine_rows.iter()
                .map(|(e, n)| json!({ "engine": engine_label(*e), "rows": n }))
                .collect::<Vec<_>>(),
            "predicates": res.predicates.iter()
                .map(|p| json!({
                    "engine": engine_label(p.engine),
                    "selectivity": p.selectivity,
                    "ms_estimate": p.ms_estimate,
                    "raw_cost": p.raw_cost,
                }))
                .collect::<Vec<_>>(),
        },
        "result_count": res.paper_ids.len(),
        "results": hits,
    }))
}

async fn enrich(pool: &PgPool, paper_ids: &[i64], k: usize) -> Result<Value> {
    let top: Vec<i64> = paper_ids.iter().take(k).copied().collect();
    if top.is_empty() {
        return Ok(json!([]));
    }
    let rows = sqlx::query(
        "SELECT id, title, publish_year, venue, cited_count \
         FROM papers WHERE id = ANY($1)",
    )
    .bind(&top)
    .fetch_all(pool)
    .await?;

    // id → row, so we can emit them in the orchestrator's ranking order.
    let mut by_id = std::collections::HashMap::new();
    for r in &rows {
        let id: i64 = r.try_get("id").unwrap_or(0);
        let title: String = r.try_get("title").unwrap_or_default();
        let year: Option<i32> = r.try_get("publish_year").ok().flatten();
        let venue: Option<String> = r.try_get("venue").ok().flatten();
        let cited: i32 = r.try_get("cited_count").unwrap_or(0);
        by_id.insert(id, (title, year, venue, cited));
    }

    let mut out = Vec::with_capacity(top.len());
    for (i, id) in top.iter().enumerate() {
        let (title, year, venue, cited) = by_id
            .get(id)
            .cloned()
            .unwrap_or_else(|| (format!("(paper #{id} — title unavailable)"), None, None, 0));
        out.push(json!({
            "rank": i + 1,
            "paper_id": id,
            "title": title,
            "year": year,
            "venue": venue,
            "cited_count": cited,
        }));
    }
    Ok(Value::Array(out))
}

/// Human-friendly engine name. Note the graph engine is `Engine::Age`
/// historically, but every plan reaches the citation graph via
/// `WITH RECURSIVE`, so we surface it as "graph".
fn engine_label(e: Engine) -> &'static str {
    match e {
        Engine::Pgvector => "pgvector · semantic",
        Engine::PgSearch => "pg_search · lexical",
        Engine::Age => "graph · WITH RECURSIVE",
    }
}

fn pct(sorted_input: &[u64], q: f64) -> u64 {
    if sorted_input.is_empty() {
        return 0;
    }
    let mut v = sorted_input.to_vec();
    v.sort_unstable();
    let idx = ((v.len() as f64 - 1.0) * q).round() as usize;
    v[idx.min(v.len() - 1)]
}
