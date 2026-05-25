//! Storage-overhead report (Phase 1 §E5).
//!
//! Captures per-table heap size + per-index size for the three Phase 1
//! engines (pgvector HNSW, pg_search BM25, AGE graph). Output supports
//! the writeup's storage-overhead table.

use anyhow::Result;
use clap::Args;
use serde::Serialize;
use sqlx::postgres::PgPool;
use sqlx::Row;

#[derive(Args, Debug, Clone)]
pub struct StorageArgs {
    #[arg(long, default_value = "reports/storage_overhead.json")]
    pub out: String,
}

#[derive(Serialize, Default)]
pub struct SizedItem {
    pub name: String,
    pub kind: &'static str, // "table" or "index"
    pub bytes: i64,
}

#[derive(Serialize)]
pub struct StorageReport {
    pub schema: &'static str,
    pub items:  Vec<SizedItem>,
    pub totals: serde_json::Value,
}

pub async fn run(pool: &PgPool, args: StorageArgs) -> Result<()> {
    let table_names = [
        "papers", "authors", "paper_authors", "chunks",
        "chunk_embeddings", "citations",
    ];
    let mut items: Vec<SizedItem> = Vec::new();
    let mut table_total = 0i64;

    // We look up by relname across schemas (tables land in ag_catalog
    // because AGE prepends that to search_path; pg_class oid → size is
    // schema-agnostic).
    for t in table_names {
        let row = sqlx::query(
            "SELECT n.nspname, pg_relation_size(c.oid) AS sz \
             FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace \
             WHERE c.relname = $1 AND c.relkind = 'r' \
             ORDER BY (n.nspname = 'public') DESC LIMIT 1"
        )
        .bind(t)
        .fetch_optional(pool).await?;
        if let Some(r) = row {
            let nsp: String = r.try_get("nspname").unwrap_or_default();
            let sz:  i64 = r.try_get("sz").unwrap_or(0);
            items.push(SizedItem { name: format!("{nsp}.{t}"), kind: "table", bytes: sz });
            table_total += sz;
        }
    }

    // Index sizes across all user schemas (matches HNSW / BM25 / PKs).
    let indexes_q = sqlx::query(
        "SELECT n.nspname, c.relname, pg_relation_size(c.oid) AS sz \
         FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace \
         WHERE c.relkind = 'i' \
           AND n.nspname NOT IN ('pg_catalog','information_schema','citations_g') \
           AND (c.relname LIKE '%hnsw%' \
                OR c.relname LIKE '%bm25%' \
                OR c.relname LIKE '%_pkey')"
    )
    .fetch_all(pool).await?;

    let mut hnsw_total = 0i64;
    let mut bm25_total = 0i64;
    let mut pk_total   = 0i64;

    for row in indexes_q {
        let nsp:  String = row.try_get("nspname")?;
        let name: String = row.try_get("relname")?;
        let sz:   i64 = row.try_get("sz")?;
        let qname = format!("{nsp}.{name}");
        if name.contains("hnsw") { hnsw_total += sz; }
        else if name.contains("bm25") { bm25_total += sz; }
        else { pk_total += sz; }
        items.push(SizedItem { name: qname, kind: "index", bytes: sz });
    }

    // AGE graph is itself a schema with vertex / edge tables. Probe via
    // pg_class oid → pg_relation_size to avoid the regclass casing
    // hazard with mixed-case label names like "Paper" / "CITES".
    let age_q = sqlx::query(
        "SELECT c.relname, c.relkind::text, pg_relation_size(c.oid) AS sz \
         FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace \
         WHERE n.nspname = 'citations_g' AND c.relkind IN ('r','i')"
    )
    .fetch_all(pool).await?;

    let mut age_table_total = 0i64;
    let mut age_index_total = 0i64;
    for row in age_q {
        let name: String = row.try_get("relname")?;
        let kind: String = row.try_get("relkind")?;
        let sz:  i64 = row.try_get("sz")?;
        let qname = format!("citations_g.{name}");
        if kind == "i" {
            items.push(SizedItem { name: qname, kind: "index", bytes: sz });
            age_index_total += sz;
        } else {
            items.push(SizedItem { name: qname, kind: "table", bytes: sz });
            age_table_total += sz;
        }
    }

    let bytes_per_mb = 1024.0 * 1024.0;
    let totals = serde_json::json!({
        "relational_tables_bytes": table_total,
        "relational_tables_mb":    table_total as f64 / bytes_per_mb,
        "hnsw_index_bytes":        hnsw_total,
        "hnsw_index_mb":           hnsw_total as f64 / bytes_per_mb,
        "bm25_index_bytes":        bm25_total,
        "bm25_index_mb":           bm25_total as f64 / bytes_per_mb,
        "age_table_bytes":         age_table_total,
        "age_table_mb":            age_table_total as f64 / bytes_per_mb,
        "age_index_bytes":         age_index_total,
        "age_index_mb":            age_index_total as f64 / bytes_per_mb,
        "primary_keys_bytes":      pk_total,
        "total_index_bytes":       hnsw_total + bm25_total + age_table_total + age_index_total + pk_total,
        "overhead_ratio":          (hnsw_total + bm25_total + age_table_total + age_index_total) as f64
                                       / (table_total.max(1) as f64),
    });

    let report = StorageReport {
        schema: "researchdb.phase1.storage.v1",
        items,
        totals,
    };
    let json = serde_json::to_string_pretty(&report)?;
    let path = std::path::PathBuf::from(&args.out);
    if let Some(p) = path.parent() { std::fs::create_dir_all(p).ok(); }
    std::fs::write(&path, &json)?;
    println!("{}", serde_json::to_string_pretty(&report.totals)?);
    tracing::info!(path = %path.display(), "storage report written");
    Ok(())
}
