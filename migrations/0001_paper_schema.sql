-- ResearchDB Phase 1 · 0001_paper_schema
--
-- Pure-paper schema (researchdb-plan.html §Phase 1 / B3). Vector/BM25
-- indices are deferred to a later migration once pgvector + pg_search
-- extensions are available; for the AGE smoke test we only need papers
-- and citations to exist as relational tables that AGE can project from.
--
-- Executed inside sqlx::migrate!()'s implicit transaction — no explicit
-- BEGIN/COMMIT here.

-- Authors. Minimal — OpenAlex gives us an id + display_name; we ignore
-- affiliations at Phase 1 (they belong to the multi-source Phase 2 schema).
CREATE TABLE IF NOT EXISTS authors (
    id           BIGSERIAL PRIMARY KEY,
    openalex_id  TEXT UNIQUE NOT NULL,
    display_name TEXT NOT NULL,
    inserted_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Papers. abstract is nullable because OpenAlex returns abstracts as
-- inverted indices and some records have none; the ingestion script
-- reconstructs them where possible.
CREATE TABLE IF NOT EXISTS papers (
    id           BIGSERIAL PRIMARY KEY,
    openalex_id  TEXT UNIQUE NOT NULL,
    title        TEXT NOT NULL,
    abstract     TEXT,
    publish_year INT,
    venue        TEXT,
    cited_count  INT NOT NULL DEFAULT 0,
    inserted_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS papers_year_idx ON papers (publish_year);
CREATE INDEX IF NOT EXISTS papers_cited_idx ON papers (cited_count DESC);

-- Many-to-many: paper ↔ author, ordered.
CREATE TABLE IF NOT EXISTS paper_authors (
    paper_id   BIGINT NOT NULL REFERENCES papers(id)  ON DELETE CASCADE,
    author_id  BIGINT NOT NULL REFERENCES authors(id) ON DELETE CASCADE,
    position   INT NOT NULL,
    PRIMARY KEY (paper_id, author_id)
);

CREATE INDEX IF NOT EXISTS paper_authors_paper_idx ON paper_authors (paper_id);
CREATE INDEX IF NOT EXISTS paper_authors_author_idx ON paper_authors (author_id);

-- Abstract-level chunks (Phase 1: 1 chunk per paper = the abstract).
-- Schema is chunk-shaped from day one so Phase 2 section-aware chunking
-- is a non-breaking extension rather than a schema migration.
CREATE TABLE IF NOT EXISTS chunks (
    id          BIGSERIAL PRIMARY KEY,
    paper_id    BIGINT NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    ordinal     INT NOT NULL,
    text        TEXT NOT NULL,
    span_start  INT NOT NULL DEFAULT 0,
    span_end    INT NOT NULL DEFAULT 0,
    UNIQUE (paper_id, ordinal)
);

CREATE INDEX IF NOT EXISTS chunks_paper_idx ON chunks (paper_id);

-- Embeddings — column kept as TEXT until pgvector lands. Migration that
-- adds the extension will ALTER this to vector(N). Keeping it nullable
-- ensures Phase 1 smoke test does not need any embeddings at all.
CREATE TABLE IF NOT EXISTS chunk_embeddings (
    chunk_id    BIGINT PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
    embedding   TEXT,
    model       TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Citation edges. Composite PK prevents duplicate edges; src→dst means
-- "src cites dst". Self-loops blocked by CHECK because they break BFS.
CREATE TABLE IF NOT EXISTS citations (
    src_paper_id BIGINT NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    dst_paper_id BIGINT NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    PRIMARY KEY (src_paper_id, dst_paper_id),
    CHECK (src_paper_id <> dst_paper_id)
);

CREATE INDEX IF NOT EXISTS citations_dst_idx ON citations (dst_paper_id);

-- Migration tracking is now handled by sqlx in _sqlx_migrations; the
-- legacy schema_migrations table is no longer required and is omitted.
