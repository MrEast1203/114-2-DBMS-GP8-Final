-- Runs once on first postgres init (mounted into docker-entrypoint-initdb.d).
-- Creates the three Phase 1 extensions: pgvector, pg_search, AGE.
-- Idempotent — IF NOT EXISTS guards keep this safe to keep around.

CREATE EXTENSION IF NOT EXISTS vector;     -- pgvector (semantic)
CREATE EXTENSION IF NOT EXISTS pg_search;  -- ParadeDB BM25 (lexical)
CREATE EXTENSION IF NOT EXISTS age;        -- Apache AGE (graph)

-- Make AGE's catalog visible without per-query qualifier.
ALTER ROLE researchdb SET search_path TO ag_catalog, "$user", public;
