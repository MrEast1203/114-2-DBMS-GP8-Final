-- ResearchDB Phase 1 · 0003_indexes_vector_bm25
--
-- Promotes chunk_embeddings.embedding from TEXT to vector(384) and lays
-- down the two non-AGE indices used by the orchestrator:
--   * HNSW over chunk_embeddings.embedding (pgvector, cosine ops).
--   * BM25 over papers.title + papers.abstract (pg_search).
--
-- Vector dimension 384 matches sentence-transformers/all-MiniLM-L6-v2 —
-- the Phase 1 default. Cheap to embed locally, no paid API, dim is small
-- enough that 5K abstracts fit comfortably in memory for HNSW build. Phase
-- 2+ can introduce a second column (e.g. embedding_1024) without rewriting
-- this one; the orchestrator picks at query time.
-- Runs inside sqlx::migrate!()'s implicit transaction.

-- pgvector + pg_search are CREATE EXTENSION'd at initdb time
-- (scripts/init-extensions.sql); we re-assert here so a fresh DB that
-- somehow missed initdb still gets them.
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_search;

-- Switch embedding to vector(384). Idempotent: only ALTER if column
-- type is still TEXT (i.e. first run). Re-running this migration on a
-- DB where embedding is already vector(384) would otherwise NULL-out
-- the real embeddings via the USING NULL clause.
DO $maybe_alter$
BEGIN
    IF (
        SELECT pg_catalog.format_type(atttypid, atttypmod)
        FROM pg_attribute
        WHERE attrelid = 'chunk_embeddings'::regclass
          AND attname = 'embedding'
          AND NOT attisdropped
    ) <> 'vector(384)' THEN
        ALTER TABLE chunk_embeddings
            ALTER COLUMN embedding DROP DEFAULT,
            ALTER COLUMN embedding TYPE vector(384) USING NULL,
            ALTER COLUMN model SET DEFAULT 'sentence-transformers/all-MiniLM-L6-v2';
    END IF;
END $maybe_alter$;

-- HNSW with cosine distance. m / ef_construction defaults are pgvector's
-- recommendation for general-purpose retrieval; ef_search is left to the
-- orchestrator to set per-query (researchdb-plan.html §Phase 1 / E2).
CREATE INDEX IF NOT EXISTS chunk_embeddings_hnsw_cos
    ON chunk_embeddings
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 200);

-- pg_search BM25 index over papers.title + papers.abstract. Both fields
-- get a default tokenizer; orchestrator chooses field weighting at query
-- time. key_field must be the table's PK.
CREATE INDEX IF NOT EXISTS papers_bm25_idx
    ON papers
    USING bm25 (id, title, abstract)
    WITH (
        key_field = 'id',
        text_fields = '{
            "title":    {"tokenizer": {"type": "default"}, "fast": true},
            "abstract": {"tokenizer": {"type": "default"}, "fast": true}
        }'
    );

