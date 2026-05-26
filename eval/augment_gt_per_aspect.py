#!/usr/bin/env python3
"""Add per-aspect labels (label_sem / label_lex / label_gph) to ground-truth.jsonl.

Background: the original GT was labelled with a single fuzzy "topical
relevance" rubric. Multi-predicate queries (Q4 = sem ∩ gph, Q5 = lex ∩
gph, Q6 = sem ∩ lex, Q7 = all three) need separate judgements per
predicate so the AND can be computed cleanly at eval time:

  label_sem = HUMAN topical judgment  ← existing `label` column
  label_lex = OPERATIONAL: BM25 @@@ matches abstract        (automated)
  label_gph = OPERATIONAL: paper_id ∈ BFS_reverse(anchor,d) (automated)

Effective relevance per query type at eval time:

  Q1 sem      → label_sem
  Q2 lex      → label_lex
  Q3 gph      → label_gph
  Q4 sem∩gph  → label_sem ∧ label_gph
  Q5 lex∩gph  → label_lex ∧ label_gph
  Q6 sem∩lex  → label_sem ∧ label_lex
  Q7 all      → label_sem ∧ label_lex ∧ label_gph

This decoupling fixes Q6/Q7 specifically: papers semantically about the
seed topic but with abstracts that don't BM25-match the lex predicate
should NOT count as relevant for Q6 (the query explicitly demands lex
match). Under the previous single-label scheme they did — biasing
evaluation against any plan (notably v3 chained push-down) that
correctly enforces the lex predicate at retrieval time.

This script is idempotent: re-running just refreshes label_lex /
label_gph without touching label_sem.

Output: rewrites `eval/ground-truth.jsonl` in place. Each row now has
both the original `label` field (kept for backwards compatibility,
copied from label_sem) and three new fields:
  - label_sem  (int 0/1)
  - label_lex  (int 0/1, or None if query has no lex predicate)
  - label_gph  (int 0/1, or None if query has no graph predicate)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import psycopg


DSN = "postgres://researchdb:researchdb@localhost:5432/researchdb"


def build_lex_set(conn, bm25_text: str) -> set[int]:
    """All paper_ids whose abstract BM25-matches the query text."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id FROM papers WHERE abstract @@@ %s",
            (bm25_text,),
        )
        return {r[0] for r in cur.fetchall()}


def build_gph_set(conn, anchor: int, depth: int) -> set[int]:
    """All paper_ids reachable via reverse BFS from anchor within depth hops."""
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
        return {r[0] for r in cur.fetchall()}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--queries", type=Path, default=Path("eval/queries.jsonl"))
    ap.add_argument("--gt", type=Path, default=Path("eval/ground-truth.jsonl"))
    args = ap.parse_args()

    queries = {q["qid"]: q for q in (json.loads(l) for l in args.queries.read_text().splitlines() if l.strip())}

    # Build per-query lex and gph sets (one SQL each per query, then in-memory).
    print("Building per-query lex / gph sets ...", file=sys.stderr)
    lex_sets: dict[str, set[int]] = {}
    gph_sets: dict[str, set[int]] = {}
    with psycopg.connect(DSN) as conn:
        for qid, q in queries.items():
            if q.get("bm25_text"):
                lex_sets[qid] = build_lex_set(conn, q["bm25_text"])
            if q.get("anchor_paper") is not None and q.get("depth") is not None:
                gph_sets[qid] = build_gph_set(conn, q["anchor_paper"], q["depth"])
            print(f"  {qid:6} lex={len(lex_sets.get(qid, set())):>5}  gph={len(gph_sets.get(qid, set())):>5}",
                  file=sys.stderr)

    # Augment each row.
    print("Augmenting GT rows ...", file=sys.stderr)
    rows_out = []
    n_sem = n_lex = n_gph = 0
    for line in args.gt.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        qid = r["qid"]
        pid = r["paper_id"]

        # Preserve existing label as label_sem (the human topical judgment).
        label_sem = r.get("label")
        r["label_sem"] = label_sem

        # Operational labels.
        if qid in lex_sets:
            r["label_lex"] = 1 if pid in lex_sets[qid] else 0
            n_lex += 1
        else:
            r["label_lex"] = None
        if qid in gph_sets:
            r["label_gph"] = 1 if pid in gph_sets[qid] else 0
            n_gph += 1
        else:
            r["label_gph"] = None
        if label_sem is not None:
            n_sem += 1

        rows_out.append(r)

    # Sanity check: print per-qid counts of relevant under each aspect.
    print()
    print("Per-qid label counts (rows | sem=1 | lex=1 | gph=1):")
    from collections import defaultdict
    cnt = defaultdict(lambda: [0, 0, 0, 0])  # rows, sem, lex, gph
    for r in rows_out:
        c = cnt[r["qid"]]
        c[0] += 1
        if r.get("label_sem") == 1: c[1] += 1
        if r.get("label_lex") == 1: c[2] += 1
        if r.get("label_gph") == 1: c[3] += 1
    for qid in sorted(cnt):
        rows, sem, lex, gph = cnt[qid]
        print(f"  {qid:6}  rows={rows:>3}  sem={sem:>3}  lex={lex:>3}  gph={gph:>3}")

    # Rewrite.
    with args.gt.open("w") as f:
        for r in rows_out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print()
    print(f"wrote {len(rows_out)} rows to {args.gt}")


if __name__ == "__main__":
    main()
