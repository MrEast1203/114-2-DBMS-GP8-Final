#!/usr/bin/env python3
"""Synthesize a paper citation graph for the AGE smoke test.

Why this exists: real OpenAlex fetching is slow + flaky, and the smoke
test (Phase 1 Week 1 gate) needs realistic-shape data to measure BFS
latency at depth 1/2/3 across out-degree buckets. A synthetic graph with
a power-law out-degree distribution gives us deterministic, reproducible
data that exercises the same code paths as real OpenAlex citations.

When real fetching is available, the loader (load_graph.py) takes the
same JSONL schema, so the smoke test does not change.

JSONL schemas
-------------
papers.jsonl    : {openalex_id, title, abstract, year, venue, cited_count}
citations.jsonl : {src, dst}   (both openalex_id strings)
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path


# --- distribution helpers --------------------------------------------------

def power_law_outdegree(rng: random.Random, alpha: float, max_deg: int) -> int:
    """Sample an out-degree from a discrete Pareto / Zipf-like distribution.

    Real citation networks follow heavy-tailed out-degree distributions
    (most papers cite few; a handful cite hundreds). alpha ~ 2.5 gives a
    reasonable shape for CS literature. We cap at max_deg to keep tests
    bounded.
    """
    # Inverse CDF sampling for Pareto-like discrete distribution.
    u = rng.random()
    # Avoid 0; scale to [1, max_deg]
    x = int(math.floor((1 - u) ** (-1 / (alpha - 1))))
    return max(1, min(x, max_deg))


# --- generator -------------------------------------------------------------

def generate(
    n_papers: int,
    seed: int,
    out_dir: Path,
    alpha: float = 2.5,
    max_deg: int = 200,
) -> dict:
    """Generate n_papers papers with a forward-citation graph.

    A paper at index i may cite any paper with j < i (no cycles, no
    self-loops). Out-degree is sampled from a power law; targets are
    biased toward earlier (= more cited, age-weighted) papers via
    inverse-rank sampling — a crude proxy for the rich-get-richer
    dynamic seen in real citation networks.
    """
    rng = random.Random(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    papers_path     = out_dir / "papers.jsonl"
    citations_path  = out_dir / "citations.jsonl"
    authors_path    = out_dir / "authors.jsonl"

    # Make seeds reproducible by writing a manifest.
    manifest = {
        "n_papers":  n_papers,
        "seed":      seed,
        "alpha":     alpha,
        "max_deg":   max_deg,
        "schema":    "researchdb.phase1.synth.v1",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    n_edges = 0
    outdeg_hist: dict[int, int] = {}

    with papers_path.open("w") as pf, \
         citations_path.open("w") as cf, \
         authors_path.open("w") as af:

        # Pre-generate a small author pool so paper_authors is non-empty.
        author_pool = []
        for a_idx in range(max(20, n_papers // 30)):
            a = {
                "openalex_id":  f"A{a_idx:07d}",
                "display_name": f"Author {a_idx:04d}",
            }
            af.write(json.dumps(a) + "\n")
            author_pool.append(a["openalex_id"])

        for i in range(n_papers):
            pid = f"W{i:07d}"
            year = 1990 + (i * 35 // n_papers)  # spread across ~35 years
            venue = f"Venue-{(i % 17):02d}"
            # Out-degree power-law, but i must have enough predecessors;
            # for early papers cap to whatever is available.
            target_deg = power_law_outdegree(rng, alpha, max_deg)
            target_deg = min(target_deg, i)
            outdeg_hist[target_deg] = outdeg_hist.get(target_deg, 0) + 1

            # Cited-count placeholder — actual count is computed after the
            # forward pass by counting in-edges; we leave 0 here and let
            # the loader fix it up (or skip — Phase 1 does not depend on
            # cited_count for the smoke test).
            paper = {
                "openalex_id":  pid,
                "title":        f"Paper {pid}: synthetic title",
                "abstract":     (
                    f"This synthetic abstract describes a fictional study "
                    f"in cluster {i % 7}. Topic seed {(i * 1009) % 9973}. "
                    "Methodology, findings, and conclusion are intentionally "
                    "vague to keep token counts modest while still exercising "
                    "BM25 and embedding code paths."
                ),
                "year":         year,
                "venue":        venue,
                "cited_count":  0,
                "authors":      rng.sample(author_pool, k=min(3, len(author_pool))),
            }
            pf.write(json.dumps(paper) + "\n")

            # Choose target_deg distinct predecessors with bias toward
            # earlier indices (richer = older + more known). We use a
            # power-law weight on j+1 (so j=0 is most likely).
            if target_deg > 0:
                # Pre-build weighted sample using exponentiation; cheap
                # because target_deg is small.
                seen = set()
                tries = 0
                while len(seen) < target_deg and tries < target_deg * 5:
                    tries += 1
                    # Sample j ∈ [0, i) biased toward small values
                    j = int(i * (rng.random() ** 2))
                    if 0 <= j < i and j not in seen:
                        seen.add(j)
                        cf.write(json.dumps({"src": pid, "dst": f"W{j:07d}"}) + "\n")
                        n_edges += 1

    stats = {
        **manifest,
        "edges":             n_edges,
        "outdeg_buckets":    summarize_outdeg(outdeg_hist),
        "papers_path":       str(papers_path),
        "citations_path":    str(citations_path),
        "authors_path":      str(authors_path),
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    return stats


def summarize_outdeg(hist: dict[int, int]) -> dict:
    """Bucket the out-degree histogram for human-readable summary."""
    buckets = {"0": 0, "1-5": 0, "6-20": 0, "21-50": 0, "51-100": 0, "100+": 0}
    for deg, cnt in hist.items():
        if deg == 0:
            buckets["0"] += cnt
        elif deg <= 5:
            buckets["1-5"] += cnt
        elif deg <= 20:
            buckets["6-20"] += cnt
        elif deg <= 50:
            buckets["21-50"] += cnt
        elif deg <= 100:
            buckets["51-100"] += cnt
        else:
            buckets["100+"] += cnt
    return buckets


# --- CLI -------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=5000,
                   help="Number of papers (default 5000 — Phase 1 smoke gate target).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=Path("data/synth"),
                   help="Output directory for JSONL + manifest.")
    p.add_argument("--alpha", type=float, default=2.5,
                   help="Pareto alpha for out-degree (lower = heavier tail).")
    p.add_argument("--max-deg", type=int, default=200,
                   help="Cap out-degree at this many edges per paper.")
    args = p.parse_args()

    stats = generate(args.n, args.seed, args.out, args.alpha, args.max_deg)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
