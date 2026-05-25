#!/usr/bin/env python3
"""Load papers / citations / authors JSONL into Postgres + project AGE graph.

Idempotent: re-running on the same data is a no-op (UPSERT semantics on
openalex_id). The AGE projection is rebuilt from scratch each run (drop
+ recreate labels) so it always reflects the current relational state.

Batching strategy
-----------------
* SQL inserts use psycopg's `executemany` with a transaction per ~5k rows.
* AGE node CREATE uses a single Cypher per batch via UNWIND of a list
  literal — one parse + plan per batch instead of per row.
* AGE edge CREATE same — UNWIND a list of {src, dst} maps.

Run with `uv run python ingest/load_graph.py --src data/synth`.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import psycopg


# --- DSN helper ------------------------------------------------------------

def get_dsn() -> str:
    return os.environ.get(
        "DATABASE_URL",
        "postgres://researchdb:researchdb@localhost:5432/researchdb",
    )


# --- JSONL streaming -------------------------------------------------------

def stream_jsonl(path: Path):
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


# --- relational load -------------------------------------------------------

def load_authors(conn: psycopg.Connection, authors_path: Path) -> int:
    rows = [(a["openalex_id"], a["display_name"]) for a in stream_jsonl(authors_path)]
    with conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO authors (openalex_id, display_name) VALUES (%s, %s) "
            "ON CONFLICT (openalex_id) DO NOTHING",
            rows,
        )
    return len(rows)


def load_papers(conn: psycopg.Connection, papers_path: Path) -> tuple[int, dict[str, int]]:
    """Insert papers; return (count, openalex_id → numeric id map)."""
    paper_rows = []
    paper_authors_rows: list[tuple[str, str, int]] = []  # (paper_oa, author_oa, position)
    for p in stream_jsonl(papers_path):
        paper_rows.append((
            p["openalex_id"],
            p["title"],
            p.get("abstract"),
            p.get("year"),
            p.get("venue"),
            p.get("cited_count", 0),
        ))
        for pos, aoa in enumerate(p.get("authors", [])):
            paper_authors_rows.append((p["openalex_id"], aoa, pos))

    with conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO papers (openalex_id, title, abstract, publish_year, venue, cited_count) "
            "VALUES (%s, %s, %s, %s, %s, %s) "
            "ON CONFLICT (openalex_id) DO NOTHING",
            paper_rows,
        )

        # Build openalex_id → id map (papers + authors).
        cur.execute("SELECT openalex_id, id FROM papers")
        paper_id_map = dict(cur.fetchall())
        cur.execute("SELECT openalex_id, id FROM authors")
        author_id_map = dict(cur.fetchall())

        # paper_authors junction. Skip rows whose author is missing.
        junction = [
            (paper_id_map[poa], author_id_map[aoa], pos)
            for (poa, aoa, pos) in paper_authors_rows
            if poa in paper_id_map and aoa in author_id_map
        ]
        cur.executemany(
            "INSERT INTO paper_authors (paper_id, author_id, position) VALUES (%s, %s, %s) "
            "ON CONFLICT (paper_id, author_id) DO NOTHING",
            junction,
        )

        # Single-chunk abstract per paper (Phase 1).
        cur.execute(
            "INSERT INTO chunks (paper_id, ordinal, text, span_start, span_end) "
            "SELECT p.id, 0, p.abstract, 0, length(p.abstract) "
            "FROM papers p "
            "WHERE p.abstract IS NOT NULL "
            "  AND NOT EXISTS (SELECT 1 FROM chunks c WHERE c.paper_id = p.id AND c.ordinal = 0)"
        )

    return len(paper_rows), paper_id_map


def load_citations(
    conn: psycopg.Connection,
    citations_path: Path,
    paper_id_map: dict[str, int],
) -> int:
    rows: list[tuple[int, int]] = []
    skipped = 0
    for c in stream_jsonl(citations_path):
        src_oa, dst_oa = c["src"], c["dst"]
        sid = paper_id_map.get(src_oa)
        did = paper_id_map.get(dst_oa)
        if sid is None or did is None or sid == did:
            skipped += 1
            continue
        rows.append((sid, did))

    with conn.cursor() as cur:
        # Chunk inserts to keep prepared-statement memory bounded.
        CHUNK = 5000
        for i in range(0, len(rows), CHUNK):
            cur.executemany(
                "INSERT INTO citations (src_paper_id, dst_paper_id) VALUES (%s, %s) "
                "ON CONFLICT DO NOTHING",
                rows[i:i + CHUNK],
            )
    print(f"  citation rows: {len(rows)} loaded, {skipped} skipped (missing paper)", file=sys.stderr)
    return len(rows)


# --- AGE projection --------------------------------------------------------

def reset_age_labels(conn: psycopg.Connection) -> None:
    """Drop + recreate Paper / CITES labels for a clean projection,
    then rebuild the property / edge indexes from migration 0004.

    AGE's drop_label() drops the underlying label *table*; any index
    on it (including paper_props_gin / cites_start_idx / cites_end_idx
    from migration 0004) is dropped by CASCADE. Without rebuilding,
    project_edges()'s MATCH-by-pid falls back to Seq Scan over all
    36K Paper vertices per edge — empirically >90 min unfinished on
    50K. Mirroring migration 0004 here brings that back to ~84 s.
    """
    with conn.cursor() as cur:
        cur.execute("LOAD 'age'")
        cur.execute("SET search_path = ag_catalog, \"$user\", public")
        for label in ("Paper", "CITES"):
            try:
                cur.execute("SELECT drop_label('citations_g', %s)", (label,))
            except psycopg.Error:
                conn.rollback()
        conn.commit()
        cur.execute("SELECT create_vlabel('citations_g', 'Paper')")
        cur.execute("SELECT create_elabel('citations_g', 'CITES')")
        # Rebuild the indexes drop_label() destroyed. Mirrors
        # migrations/0004_age_property_index.sql so a fresh AGE
        # projection still gets the GIN-backed MATCH-by-pid lookup.
        cur.execute(
            'CREATE INDEX IF NOT EXISTS paper_props_gin '
            'ON citations_g."Paper" USING gin (properties)'
        )
        cur.execute(
            'CREATE INDEX IF NOT EXISTS cites_start_idx '
            'ON citations_g."CITES" (start_id)'
        )
        cur.execute(
            'CREATE INDEX IF NOT EXISTS cites_end_idx '
            'ON citations_g."CITES" (end_id)'
        )
    conn.commit()


def project_nodes(conn: psycopg.Connection, batch: int = 1000) -> int:
    """Project all papers as :Paper nodes via UNWIND batches."""
    with conn.cursor() as cur:
        cur.execute("SELECT id FROM papers ORDER BY id")
        ids = [r[0] for r in cur.fetchall()]

    total = 0
    with conn.cursor() as cur:
        cur.execute("LOAD 'age'")
        cur.execute("SET search_path = ag_catalog, \"$user\", public")
        for i in range(0, len(ids), batch):
            chunk = ids[i:i + batch]
            # Build a Cypher list literal: [1, 2, 3, ...]
            list_literal = "[" + ",".join(str(x) for x in chunk) + "]"
            cypher = (
                f"UNWIND {list_literal} AS pid "
                f"CREATE (:Paper {{pid: pid}})"
            )
            cur.execute(
                "SELECT * FROM cypher('citations_g', $$ " + cypher + " $$) AS (a agtype)"
            )
            total += len(chunk)
    conn.commit()
    return total


def project_edges(conn: psycopg.Connection, batch: int = 1000) -> int:
    """Project all citation rows as :CITES edges via UNWIND batches."""
    with conn.cursor() as cur:
        cur.execute("SELECT src_paper_id, dst_paper_id FROM citations")
        edges = cur.fetchall()

    total = 0
    with conn.cursor() as cur:
        cur.execute("LOAD 'age'")
        cur.execute("SET search_path = ag_catalog, \"$user\", public")
        for i in range(0, len(edges), batch):
            chunk = edges[i:i + batch]
            # Cypher map list: [{s:1,d:2},...]
            pairs = ",".join(f"{{s:{s},d:{d}}}" for s, d in chunk)
            cypher = (
                f"UNWIND [{pairs}] AS p "
                f"MATCH (s:Paper {{pid: p.s}}), (d:Paper {{pid: p.d}}) "
                f"CREATE (s)-[:CITES]->(d)"
            )
            cur.execute(
                "SELECT * FROM cypher('citations_g', $$ " + cypher + " $$) AS (a agtype)"
            )
            total += len(chunk)
    conn.commit()
    return total


# --- verify ----------------------------------------------------------------

def verify(conn: psycopg.Connection) -> dict:
    out: dict = {}
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM papers")
        out["sql_papers"] = cur.fetchone()[0]
        cur.execute("SELECT count(*) FROM citations")
        out["sql_citations"] = cur.fetchone()[0]
        cur.execute("LOAD 'age'")
        cur.execute("SET search_path = ag_catalog, \"$user\", public")
        cur.execute(
            "SELECT * FROM cypher('citations_g', $$ MATCH (p:Paper) RETURN count(p) $$) AS (n agtype)"
        )
        out["age_nodes"] = int(str(cur.fetchone()[0]))
        cur.execute(
            "SELECT * FROM cypher('citations_g', $$ MATCH ()-[r:CITES]->() RETURN count(r) $$) AS (n agtype)"
        )
        out["age_edges"] = int(str(cur.fetchone()[0]))

        # Out-degree distribution buckets.
        cur.execute(
            "SELECT * FROM cypher('citations_g', $$ "
            "MATCH (p:Paper)-[r:CITES]->() "
            "WITH p, count(r) AS d "
            "RETURN d "
            "$$) AS (deg agtype)"
        )
        degs = [int(str(r[0])) for r in cur.fetchall()]
        buckets = {"0": 0, "1-5": 0, "6-20": 0, "21-50": 0, "51-100": 0, "100+": 0}
        # Papers with 0 out-degree don't appear in the MATCH; compute separately.
        zero = out["age_nodes"] - len(degs)
        buckets["0"] = zero
        for d in degs:
            if d <= 5: buckets["1-5"] += 1
            elif d <= 20: buckets["6-20"] += 1
            elif d <= 50: buckets["21-50"] += 1
            elif d <= 100: buckets["51-100"] += 1
            else: buckets["100+"] += 1
        out["outdeg_buckets"] = buckets
    return out


# --- main ------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", type=Path, required=True,
                   help="Directory containing papers.jsonl / citations.jsonl / authors.jsonl")
    p.add_argument("--dsn", default=None, help="Postgres DSN (default: $DATABASE_URL or local).")
    p.add_argument("--skip-graph", action="store_true",
                   help="Load relational tables only; skip AGE projection.")
    p.add_argument("--reset", action="store_true",
                   help="TRUNCATE papers/authors/citations before loading (DESTRUCTIVE).")
    args = p.parse_args()

    src = args.src
    dsn = args.dsn or get_dsn()

    for name in ("papers.jsonl", "citations.jsonl", "authors.jsonl"):
        if not (src / name).exists():
            print(f"missing {src / name}", file=sys.stderr)
            sys.exit(2)

    t0 = time.time()
    with psycopg.connect(dsn, autocommit=False) as conn:
        if args.reset:
            with conn.cursor() as cur:
                cur.execute("TRUNCATE papers, authors, citations, paper_authors, chunks, "
                            "chunk_embeddings RESTART IDENTITY CASCADE")
            conn.commit()
            print("  truncated", file=sys.stderr)

        n_authors = load_authors(conn, src / "authors.jsonl")
        conn.commit()
        print(f"  authors:    {n_authors}", file=sys.stderr)

        n_papers, paper_id_map = load_papers(conn, src / "papers.jsonl")
        conn.commit()
        print(f"  papers:     {n_papers}", file=sys.stderr)

        n_cites = load_citations(conn, src / "citations.jsonl", paper_id_map)
        conn.commit()
        print(f"  citations:  {n_cites}", file=sys.stderr)

        if not args.skip_graph:
            reset_age_labels(conn)
            n_nodes = project_nodes(conn)
            print(f"  AGE nodes:  {n_nodes}", file=sys.stderr)
            n_edges = project_edges(conn)
            print(f"  AGE edges:  {n_edges}", file=sys.stderr)

        stats = verify(conn)
    stats["elapsed_sec"] = round(time.time() - t0, 2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
