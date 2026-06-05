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
/// Per-(qid, paper_id) relevance labels for those eval queries, so a preset
/// run can be scored against ground truth right in the browser. Baked in for
/// the same reason as the presets: `serve` needs no files on disk at runtime.
const GT_JSONL: &str = include_str!("../../eval/ground-truth.jsonl");

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

    // Split the request target into path + optional query string so exact
    // routing still works for `/resolve?chunk=1`.
    let (route, query) = req.path.split_once('?').unwrap_or((req.path.as_str(), ""));

    match (req.method.as_str(), route) {
        ("GET", "/") | ("GET", "/index.html") => {
            write_response(&mut stream, "200 OK", "text/html; charset=utf-8", APP_HTML.as_bytes())
                .await
        }
        ("GET", "/presets") => {
            let body = presets_json().to_string();
            write_response(&mut stream, "200 OK", "application/json", body.as_bytes()).await
        }
        ("GET", "/resolve") => {
            let body = match resolve_anchor(&pool, query).await {
                Ok(v) => v,
                Err(e) => json!({ "ok": false, "error": e.to_string() }),
            };
            write_response(&mut stream, "200 OK", "application/json", body.to_string().as_bytes())
                .await
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
    /// Eval-query id (e.g. "Q4-1") when the run came from a preset. Present →
    /// the response is scored against ground truth; absent → custom query, no GT.
    #[serde(default)]
    qid: Option<String>,
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

    let (sem, lex, gph) = q.engines();

    // If this run came from a labelled eval query, score it against ground
    // truth. Effective relevance is the AND of the aspect labels for exactly
    // the engines this query type uses (mirrors eval/evaluate.py). Done after
    // the timing loop, so GT lookup never pollutes the measured latency.
    let gt_rel = req
        .qid
        .as_deref()
        .filter(|s| !s.is_empty())
        .map(|qid| ground_truth_for(qid, sem, lex, gph))
        .filter(|m| !m.is_empty());

    // Enrich the top-K paper ids with titles (+ GT mark) in ranking order.
    let hits = enrich(pool, &res.paper_ids, k, gt_rel.as_ref()).await?;

    let p50 = pct(&durations_us, 0.50);
    let p95 = pct(&durations_us, 0.95);
    let pmin = *durations_us.iter().min().unwrap_or(&0);

    let ground_truth = match &gt_rel {
        Some(rel) => {
            let topk: Vec<i64> = res.paper_ids.iter().take(k).copied().collect();
            let hits_n = topk.iter().filter(|p| rel.get(p).is_some_and(|r| r.relevant)).count();
            let pool_relevant = rel.values().filter(|r| r.relevant).count();
            let ndcg = ndcg_at_k(&res.paper_ids, rel, k);
            json!({
                "available": true,
                "qid": req.qid,
                "ndcg10": if ndcg.is_nan() {
                    Value::Null
                } else {
                    let r = (ndcg * 1000.0).round() / 1000.0;
                    json!(if r == 0.0 { 0.0 } else { r }) // normalize -0.0 → 0.0
                },
                "hits": hits_n,
                "returned": topk.len(),
                "relevant_in_pool": pool_relevant,
            })
        }
        None => json!({ "available": false }),
    };

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
        "ground_truth": ground_truth,
    }))
}

async fn enrich(
    pool: &PgPool,
    paper_ids: &[i64],
    k: usize,
    gt_rel: Option<&std::collections::HashMap<i64, GtRow>>,
) -> Result<Value> {
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
        // true = labelled relevant, false = in pool but not relevant,
        // null = no GT for this query / paper not in the labelled pool.
        let row = gt_rel.and_then(|m| m.get(id));
        let relevant = match row {
            Some(r) => json!(r.relevant),
            None => Value::Null,
        };
        // Per-aspect labels for the engines this query type demands, so the GUI
        // can show which predicate a non-relevant paper failed (§6.3.6).
        let aspects: Vec<Value> = row
            .map(|r| r.aspects.iter().map(|(k, ok)| json!({ "k": k, "ok": ok })).collect())
            .unwrap_or_default();
        out.push(json!({
            "rank": i + 1,
            "paper_id": id,
            "title": title,
            "year": year,
            "venue": venue,
            "cited_count": cited,
            "relevant": relevant,
            "aspects": aspects,
        }));
    }
    Ok(Value::Array(out))
}

// ---------------------------------------------------------------------
// Ground truth — score a preset run against the labelled eval pool, using
// the SAME effective-relevance rule as eval/evaluate.py: a paper counts as
// relevant for a query only if the aspect labels for every engine that query
// uses are all 1. A demanded aspect labelled null means the row is unjudged
// and is dropped from the pool (NDCG then treats it as a 0-gain result).
// ---------------------------------------------------------------------

/// One labelled paper in a query's pool: the effective relevance (AND of the
/// demanded aspects) plus the demanded per-aspect labels themselves, so the GUI
/// can show *why* a paper is (not) relevant — e.g. sem✓ lex✗ → fails Q6.
struct GtRow {
    relevant: bool,
    aspects: Vec<(&'static str, bool)>, // only the aspects this query type demands
}

fn ground_truth_for(
    qid: &str,
    want_sem: bool,
    want_lex: bool,
    want_gph: bool,
) -> std::collections::HashMap<i64, GtRow> {
    // Cheap pre-filter on the quoted qid token (colon spacing varies by
    // writer), then confirm the parsed qid to stay correct regardless.
    let needle = format!("\"{qid}\"");
    let mut rel = std::collections::HashMap::new();
    for line in GT_JSONL.lines() {
        if !line.contains(&needle) {
            continue;
        }
        let Ok(v) = serde_json::from_str::<Value>(line) else { continue };
        if v.get("qid").and_then(Value::as_str) != Some(qid) {
            continue;
        }
        let Some(pid) = v.get("paper_id").and_then(Value::as_i64) else { continue };
        let mut all_one = true;
        let mut unjudged = false;
        let mut aspects: Vec<(&'static str, bool)> = Vec::new();
        for (want, key, name) in [
            (want_sem, "label_sem", "sem"),
            (want_lex, "label_lex", "lex"),
            (want_gph, "label_gph", "gph"),
        ] {
            if !want {
                continue;
            }
            // label_sem predates augmentation as plain `label`; fall back to it.
            let mut val = v.get(key).and_then(Value::as_i64);
            if val.is_none() && key == "label_sem" {
                val = v.get("label").and_then(Value::as_i64);
            }
            match val {
                None => {
                    unjudged = true;
                    break;
                }
                Some(x) => {
                    let ok = x == 1;
                    if !ok {
                        all_one = false;
                    }
                    aspects.push((name, ok));
                }
            }
        }
        if unjudged {
            continue;
        }
        rel.insert(pid, GtRow { relevant: all_one, aspects });
    }
    rel
}

/// Binary-relevance NDCG@k over a ranking, matching eval/evaluate.py: NaN when
/// the pool has no relevant papers (so the caller can render it as "—").
fn ndcg_at_k(ranked: &[i64], rel: &std::collections::HashMap<i64, GtRow>, k: usize) -> f64 {
    let dcg: f64 = ranked
        .iter()
        .take(k)
        .enumerate()
        .map(|(i, pid)| {
            let g = if rel.get(pid).is_some_and(|r| r.relevant) { 1.0 } else { 0.0 };
            g / ((i as f64) + 2.0).log2()
        })
        .sum();
    let n_pos = rel.values().filter(|r| r.relevant).count();
    if n_pos == 0 {
        return f64::NAN;
    }
    let idcg: f64 = (0..k.min(n_pos)).map(|i| 1.0 / ((i as f64) + 2.0).log2()).sum();
    if idcg > 0.0 {
        dcg / idcg
    } else {
        0.0
    }
}

// ---------------------------------------------------------------------
// Resolve — turn a raw seed_chunk / anchor_paper id into something a human
// recognizes, so the GUI can answer "what does seed_chunk = 1 actually mean".
// ---------------------------------------------------------------------

async fn resolve_anchor(pool: &PgPool, query: &str) -> Result<Value> {
    let mut chunk_id: Option<i64> = None;
    let mut paper_id: Option<i64> = None;
    for pair in query.split('&') {
        if let Some(v) = pair.strip_prefix("chunk=") {
            chunk_id = v.parse().ok();
        } else if let Some(v) = pair.strip_prefix("paper=") {
            paper_id = v.parse().ok();
        }
    }

    // seed_chunk → the paper it belongs to + a short snippet of the chunk text.
    if let Some(id) = chunk_id {
        let row = sqlx::query(
            "SELECT c.paper_id, p.title, p.publish_year, c.text \
             FROM chunks c JOIN papers p ON p.id = c.paper_id \
             WHERE c.id = $1",
        )
        .bind(id)
        .fetch_optional(pool)
        .await?;
        return Ok(match row {
            Some(r) => {
                let pid: i64 = r.try_get("paper_id").unwrap_or(0);
                let title: String = r.try_get("title").unwrap_or_default();
                let year: Option<i32> = r.try_get("publish_year").ok().flatten();
                let text: String = r.try_get("text").unwrap_or_default();
                json!({ "ok": true, "kind": "chunk", "id": id, "paper_id": pid,
                        "title": title, "year": year, "snippet": snippet(&text, 140) })
            }
            None => json!({ "ok": false, "error": format!("chunk {id} 不存在") }),
        });
    }

    // anchor_paper → just the paper title/year.
    if let Some(id) = paper_id {
        let row = sqlx::query("SELECT title, publish_year FROM papers WHERE id = $1")
            .bind(id)
            .fetch_optional(pool)
            .await?;
        return Ok(match row {
            Some(r) => {
                let title: String = r.try_get("title").unwrap_or_default();
                let year: Option<i32> = r.try_get("publish_year").ok().flatten();
                json!({ "ok": true, "kind": "paper", "id": id, "title": title, "year": year })
            }
            None => json!({ "ok": false, "error": format!("paper {id} 不存在") }),
        });
    }

    anyhow::bail!("missing chunk= or paper= query param")
}

/// First `max_chars` characters of `text`, trimmed, with an ellipsis if cut.
/// Char-based (not byte) so multi-byte abstracts never split mid-codepoint.
fn snippet(text: &str, max_chars: usize) -> String {
    let t = text.trim();
    let mut s: String = t.chars().take(max_chars).collect();
    if t.chars().count() > max_chars {
        s.push('…');
    }
    s
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
