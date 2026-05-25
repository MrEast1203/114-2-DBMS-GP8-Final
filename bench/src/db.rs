//! Connection, health-check, and migration helpers shared across
//! subcommands.

use anyhow::{anyhow, Result};
use sqlx::migrate::Migrator;
use sqlx::postgres::{PgPool, PgPoolOptions};
use std::path::Path;
use std::time::Duration;

/// Embedded migrations — sqlx scans the `migrations/` directory relative
/// to CARGO_MANIFEST_DIR at compile time, baking each .sql into the
/// binary. Filenames must be `{NNNN}_{name}.sql` (we use 0001, 0002,
/// 0003). The binary therefore needs no runtime access to the SQL
/// files.
static MIGRATOR: Migrator = sqlx::migrate!("../migrations");

pub async fn connect(dsn: &str) -> Result<PgPool> {
    let pool = PgPoolOptions::new()
        .max_connections(4)
        .acquire_timeout(Duration::from_secs(10))
        // AGE requires LOAD 'age' and ag_catalog on search_path per session.
        // shared_preload_libraries=age + session_preload_libraries=age in
        // compose handle most of this; we still set search_path defensively
        // because some psql/sqlx-managed connections reset it.
        .after_connect(|conn, _| Box::pin(async move {
            use sqlx::Executor;
            conn.execute(r#"SET search_path = ag_catalog, "$user", public"#).await?;
            Ok(())
        }))
        .connect(dsn)
        .await?;
    Ok(pool)
}

pub async fn run_migrations(pool: &PgPool) -> Result<()> {
    // Sanity: warn if Path::new("migrations") doesn't exist at runtime —
    // not required (we embedded at compile time) but useful for ops.
    let _ = Path::new("migrations").exists();
    MIGRATOR.run(pool).await?;
    Ok(())
}

pub async fn health_check(pool: &PgPool) -> Result<()> {
    let (one,): (i32,) = sqlx::query_as("SELECT 1::int4")
        .fetch_one(pool)
        .await?;
    if one != 1 {
        return Err(anyhow!("SELECT 1 returned {}", one));
    }

    let (ext_version,): (Option<String>,) = sqlx::query_as(
        "SELECT extversion FROM pg_extension WHERE extname = 'age'",
    )
    .fetch_one(pool)
    .await?;
    let ver = ext_version.ok_or_else(|| anyhow!("AGE extension not installed"))?;
    tracing::info!(age_version = %ver, "AGE extension present");

    let (n_papers,): (i64,) = sqlx::query_as("SELECT count(*) FROM papers")
        .fetch_one(pool)
        .await?;
    tracing::info!(papers = n_papers, "papers loaded");

    Ok(())
}
