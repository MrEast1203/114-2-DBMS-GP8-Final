#!/usr/bin/env python3
"""Build the candidate pool for ground-truth labelling.

For each query in eval/queries.jsonl we collect ≤30 candidate papers by
unioning top-10 from each engine (kitchen sink). Output:

  eval/candidates.jsonl  — one line per (query, candidate) pair with
                           paper metadata; ready for binary labelling.

Phase 1 / C2 deliverable.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import psycopg


def dsn():
    return os.environ.get(
        "DATABASE_URL", "postgres://researchdb:researchdb@localhost:5432/researchdb"
    )


def fetch_meta(conn: psycopg.Connection, pids: list[int]) -> dict[int, dict]:
    if not pids:
        return {}
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, openalex_id, title, publish_year, venue, left(abstract, 600) FROM papers WHERE id = ANY(%s)",
            (pids,),
        )
        out: dict[int, dict] = {}
        for row in cur.fetchall():
            pid, oa, title, year, venue, abstract = row
            out[pid] = {
                "paper_id":      pid,
                "openalex_id":   oa,
                "title":         title,
                "year":          year,
                "venue":         venue,
                "abstract":      abstract,
            }
        return out


def top_semantic(conn, seed_chunk: int, n: int) -> list[int]:
    with conn.cursor() as cur:
        cur.execute("SET LOCAL hnsw.ef_search = 80")
        cur.execute(
            "WITH seed AS (SELECT embedding FROM chunk_embeddings WHERE chunk_id = %s),"
            "  hit AS ("
            "    SELECT c.paper_id, ce.embedding <=> seed.embedding AS d"
            "    FROM chunk_embeddings ce JOIN chunks c ON c.id = ce.chunk_id, seed"
            "    ORDER BY ce.embedding <=> seed.embedding LIMIT %s) "
            "SELECT DISTINCT ON (paper_id) paper_id FROM hit ORDER BY paper_id, d",
            (seed_chunk, n * 3),
        )
        return [r[0] for r in cur.fetchall()]


def top_lexical(conn, text: str, n: int) -> list[int]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id FROM papers WHERE abstract @@@ %s "
            "ORDER BY paradedb.score(id) DESC LIMIT %s",
            (text, n),
        )
        return [r[0] for r in cur.fetchall()]


def graph_filter(conn, anchor: int, depth: int) -> list[int]:
    """Papers that cite anchor (transitively, depth 1..N). Uses
    PostgreSQL WITH RECURSIVE on the citations table (§D8 — set-equal
    with AGE Cypher and 6-850× faster)."""
    with conn.cursor() as cur:
        cur.execute(
            """
            WITH RECURSIVE bfs(paper_id, d) AS (
              SELECT src_paper_id, 1 FROM citations WHERE dst_paper_id = %s
              UNION
              SELECT c.src_paper_id, b.d + 1
              FROM citations c JOIN bfs b ON c.dst_paper_id = b.paper_id
              WHERE b.d < %s
            )
            SELECT DISTINCT paper_id FROM bfs
            """,
            (anchor, depth),
        )
        return [r[0] for r in cur.fetchall()]


def build_pool(conn, q: dict, k: int = 30, max_pool: int = 80) -> list[int]:
    """Build a kitchen-sink candidate pool per query. Wider than the
    initial Phase-1 (k=10) version — at 50K corpus the relevant set is
    bigger, and the labelled pool needs to overlap with whatever the
    plans actually surface or NDCG will be biased low by unlabelled
    top-rank picks.

    Sources unioned (de-duped, capped at max_pool):
      * top-k MiniLM (cosine over chunk embeddings)
      * top-k pg_search BM25 over abstracts
      * full graph filter (anchor's k-hop neighbourhood — usually small)
      * intersection: graph ∩ MiniLM-top-(k×2)  for Q4/Q5/Q7
      * intersection: graph ∩ BM25-top-(k×2)    for Q5/Q7
    """
    pool: list[int] = []
    seen: set[int] = set()

    def add_all(ids):
        for pid in ids:
            if pid not in seen:
                seen.add(pid)
                pool.append(pid)

    qtype = q["type"]
    semantic_ids: list[int] = []
    lexical_ids:  list[int] = []
    graph_ids:    list[int] = []

    if q.get("seed_chunk") is not None:
        semantic_ids = top_semantic(conn, q["seed_chunk"], k)
    if q.get("bm25_text"):
        lexical_ids = top_lexical(conn, q["bm25_text"], k)
    if q.get("anchor_paper") is not None and q.get("depth") is not None:
        graph_ids = graph_filter(conn, q["anchor_paper"], q["depth"])

    needs_graph = qtype in ("Q4", "Q5", "Q7")
    if needs_graph:
        graph_set = set(graph_ids)
        # Intersection of plan candidates with graph filter — these are
        # what v2 push-down actually surfaces in top-k. Highest priority.
        sem_in_graph = [p for p in semantic_ids if p in graph_set]
        lex_in_graph = [p for p in lexical_ids  if p in graph_set]
        add_all(sem_in_graph)
        add_all(lex_in_graph)
        # Then everyone else in the graph filter (capped).
        rest = [p for p in graph_ids if p not in seen]
        add_all(rest[: max_pool - len(pool)])
    else:
        add_all(semantic_ids)
        add_all(lexical_ids)

    return pool[:max_pool]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--queries", type=Path, default=Path("eval/queries.jsonl"))
    p.add_argument("--out", type=Path, default=Path("eval/candidates.jsonl"))
    p.add_argument("--k", type=int, default=30, help="Per-engine top-k.")
    p.add_argument("--max-pool", type=int, default=80,
                   help="Cap on per-query candidate pool size.")
    args = p.parse_args()

    queries = [json.loads(line) for line in args.queries.read_text().splitlines() if line.strip()]

    out_lines: list[str] = []
    with psycopg.connect(dsn(), autocommit=False) as conn:
        for q in queries:
            pool = build_pool(conn, q, args.k, args.max_pool)
            metas = fetch_meta(conn, pool)
            print(f"  {q['qid']:5}  {q['type']}  pool={len(pool)}  metas={len(metas)}", file=sys.stderr)
            for pid in pool:
                m = metas.get(pid)
                if not m:
                    continue
                out_lines.append(json.dumps({
                    "qid":    q["qid"],
                    "qtype":  q["type"],
                    "qdesc":  q["desc"],
                    "paper_id":      m["paper_id"],
                    "openalex_id":   m["openalex_id"],
                    "title":         m["title"],
                    "year":          m["year"],
                    "venue":         m["venue"],
                    "abstract_clip": m["abstract"],
                    "label":         None,  # to be filled by labelling step
                }, ensure_ascii=False))

    args.out.write_text("\n".join(out_lines) + "\n")
    print(f"wrote {len(out_lines)} (query, candidate) pairs → {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
