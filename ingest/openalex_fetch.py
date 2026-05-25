#!/usr/bin/env python3
"""Fetch a paper citation subgraph from OpenAlex via BFS from seeds.

Output schema matches synthesize_graph.py so the loader is agnostic to
source. Polite-pool friendly: include OPENALEX_MAILTO in env or pass
--mailto on CLI.

Phase 1 dataset target: 10K–50K papers from OpenAlex CS subset,
2–3 hop BFS from seeds in systems / security / ML. This script writes
incrementally so partial failures don't lose work; resumes by checking
existing papers.jsonl.

Note: AGE smoke test (Week 1 gate) uses synthesize_graph.py for
reproducibility — this real fetcher is for the actual Phase 1 dataset
once the gate has passed and dataset upper bound is chosen.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import deque
from pathlib import Path
from urllib.parse import urlencode

try:
    import urllib.request as urlreq
    from urllib.error import HTTPError, URLError
except ImportError:  # pragma: no cover
    print("urllib not available?", file=sys.stderr)
    raise


OPENALEX_BASE = "https://api.openalex.org"
BATCH_SIZE    = 50          # OpenAlex filter ids.openalex accepts up to 50
RATE_DELAY    = 0.12        # ~8 req/sec — well within polite pool
MAX_RETRIES   = 4


# --- API client ------------------------------------------------------------

def fetch_works(ids: list[str], mailto: str) -> list[dict]:
    """Fetch a batch of up to 50 works by their OpenAlex IDs."""
    if not ids:
        return []
    filter_str = "ids.openalex:" + "|".join(ids)
    params = {"filter": filter_str, "per-page": str(len(ids)), "mailto": mailto}
    url = f"{OPENALEX_BASE}/works?{urlencode(params)}"

    for attempt in range(MAX_RETRIES):
        try:
            req = urlreq.Request(url, headers={"User-Agent": f"ResearchDB/0.1 (mailto:{mailto})"})
            with urlreq.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
                return data.get("results", [])
        except HTTPError as e:
            if e.code == 429:
                wait = (2 ** attempt) * 1.0
                print(f"  rate-limited; sleeping {wait}s", file=sys.stderr)
                time.sleep(wait)
                continue
            elif e.code >= 500:
                time.sleep(2 ** attempt)
                continue
            else:
                raise
        except URLError as e:
            print(f"  network error: {e}; retry {attempt+1}/{MAX_RETRIES}", file=sys.stderr)
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Exhausted retries for batch: {ids[:3]}...")


def reconstruct_abstract(idx: dict | None) -> str | None:
    """OpenAlex returns abstracts as inverted indices: {word: [positions]}.

    Reconstruct the plain text by placing words at their positions.
    """
    if not idx:
        return None
    # Find max position to size the buffer.
    max_pos = -1
    for positions in idx.values():
        for p in positions:
            if p > max_pos:
                max_pos = p
    if max_pos < 0:
        return None
    tokens: list[str] = [""] * (max_pos + 1)
    for word, positions in idx.items():
        for p in positions:
            if 0 <= p <= max_pos:
                tokens[p] = word
    return " ".join(t for t in tokens if t).strip() or None


def normalize_id(raw: str) -> str:
    """OpenAlex IDs come as full URLs like https://openalex.org/W123. We
    canonicalize to the short form (W123)."""
    if raw.startswith("https://openalex.org/"):
        return raw[len("https://openalex.org/"):]
    return raw


def extract_paper(work: dict) -> dict:
    """Map an OpenAlex work record to our schema."""
    venue = None
    loc = work.get("primary_location") or {}
    src = loc.get("source") if isinstance(loc, dict) else None
    if isinstance(src, dict):
        venue = src.get("display_name")

    authorships = work.get("authorships") or []
    authors = []
    for a in authorships[:20]:  # cap at 20 authors per paper
        au = a.get("author") or {}
        aid = au.get("id")
        name = au.get("display_name")
        if aid and name:
            authors.append({
                "openalex_id":  normalize_id(aid),
                "display_name": name,
            })

    refs = []
    for r in work.get("referenced_works") or []:
        nid = normalize_id(r)
        if nid.startswith("W"):
            refs.append(nid)

    return {
        "paper": {
            "openalex_id":  normalize_id(work["id"]),
            "title":        (work.get("title") or "").strip() or "(untitled)",
            "abstract":     reconstruct_abstract(work.get("abstract_inverted_index")),
            "year":         work.get("publication_year"),
            "venue":        venue,
            "cited_count":  work.get("cited_by_count", 0),
            "authors":      [a["openalex_id"] for a in authors],
        },
        "authors":  authors,
        "refs":     refs,
    }


# --- BFS driver ------------------------------------------------------------

def bfs_fetch(
    seeds: list[str],
    target_n: int,
    max_depth: int,
    out_dir: Path,
    mailto: str,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    papers_path    = out_dir / "papers.jsonl"
    citations_path = out_dir / "citations.jsonl"
    authors_path   = out_dir / "authors.jsonl"
    state_path     = out_dir / "fetch_state.json"

    seen_papers: set[str] = set()
    seen_authors: set[str] = set()

    # Resume from existing state if present.
    if state_path.exists():
        state = json.loads(state_path.read_text())
        seen_papers  = set(state.get("seen_papers", []))
        seen_authors = set(state.get("seen_authors", []))
        print(f"resuming with {len(seen_papers)} papers already fetched", file=sys.stderr)
        mode = "a"
    else:
        mode = "w"

    pf = papers_path.open(mode)
    cf = citations_path.open(mode)
    af = authors_path.open(mode)

    # BFS frontier: list of (paper_id, depth)
    frontier: deque[tuple[str, int]] = deque((s, 0) for s in seeds if s not in seen_papers)

    n_fetched = 0
    try:
        while frontier and len(seen_papers) < target_n:
            # Pull a batch from frontier at the same depth so we BFS evenly.
            batch: list[str] = []
            depth = frontier[0][1]
            while frontier and len(batch) < BATCH_SIZE and frontier[0][1] == depth:
                pid, d = frontier.popleft()
                if pid in seen_papers:
                    continue
                batch.append(pid)

            if not batch:
                continue

            works = fetch_works(batch, mailto)
            for work in works:
                rec = extract_paper(work)
                paper = rec["paper"]
                pid = paper["openalex_id"]
                if pid in seen_papers:
                    continue
                seen_papers.add(pid)
                pf.write(json.dumps(paper) + "\n")
                n_fetched += 1

                for au in rec["authors"]:
                    if au["openalex_id"] not in seen_authors:
                        seen_authors.add(au["openalex_id"])
                        af.write(json.dumps(au) + "\n")

                for ref in rec["refs"]:
                    cf.write(json.dumps({"src": pid, "dst": ref}) + "\n")
                    if depth + 1 <= max_depth and ref not in seen_papers:
                        frontier.append((ref, depth + 1))

            pf.flush(); cf.flush(); af.flush()
            print(f"depth {depth}  fetched {n_fetched}  seen {len(seen_papers)}  frontier {len(frontier)}",
                  file=sys.stderr)
            time.sleep(RATE_DELAY)

    finally:
        pf.close(); cf.close(); af.close()
        state_path.write_text(json.dumps({
            "seen_papers":  sorted(seen_papers),
            "seen_authors": sorted(seen_authors),
        }))

    return {
        "papers_fetched":  n_fetched,
        "total_seen":      len(seen_papers),
        "papers_path":     str(papers_path),
        "citations_path":  str(citations_path),
        "authors_path":    str(authors_path),
    }


# --- seeds -----------------------------------------------------------------

DEFAULT_SEEDS = [
    # Systems
    "W2122465391",  # MapReduce: Simplified Data Processing on Large Clusters
    "W2013409485",  # Spanner: Google's globally-distributed database
    "W2119565742",  # The Google file system
    # Security
    "W2883613460",  # Meltdown: Reading kernel memory from user space
    "W2781723315",  # Spectre Attacks: Exploiting Speculative Execution
    # ML
    "W2626778328",  # Attention Is All You Need
    "W2194775991",  # Deep Residual Learning for Image Recognition (ResNet)
]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", nargs="*", default=DEFAULT_SEEDS,
                   help="OpenAlex IDs to seed BFS (default: hand-picked CS seeds).")
    p.add_argument("--seeds-file", type=Path,
                   help="File with one OpenAlex ID per line; overrides --seeds.")
    p.add_argument("--n", type=int, default=5000,
                   help="Target paper count (Phase 1 smoke = 5K, Phase 1 main = 10K–50K).")
    p.add_argument("--max-depth", type=int, default=3)
    p.add_argument("--out", type=Path, default=Path("data/openalex"))
    p.add_argument("--mailto",
                   default=os.environ.get("OPENALEX_MAILTO"),
                   help="Email for OpenAlex polite pool. Required.")
    args = p.parse_args()

    if not args.mailto:
        print("ERROR: --mailto or OPENALEX_MAILTO required for polite pool", file=sys.stderr)
        sys.exit(2)

    seeds = args.seeds
    if args.seeds_file:
        seeds = [line.strip() for line in args.seeds_file.read_text().splitlines() if line.strip()]

    stats = bfs_fetch(seeds, args.n, args.max_depth, args.out, args.mailto)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
