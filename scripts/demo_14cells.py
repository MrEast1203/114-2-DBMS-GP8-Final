#!/usr/bin/env python3
"""Phase 1 §E6 — 14-cell demo runner.

Runs all 7 query types × 2 plans (naive, v0) on the current DB and
prints a comparison table; writes a JSON summary to reports/.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path


def run_cell(bench: str, plan: str, q: int, args) -> dict:
    cmd = [
        bench, "seven",
        "--plan", plan,
        "--query", str(q),
        "--samples", str(args.samples),
        "--seed-chunk", str(args.seed_chunk),
        "--anchor-paper", str(args.anchor_paper),
        "--depth", str(args.depth),
        "--bm25-text", args.bm25_text,
        "--ef-search", str(args.ef_search),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if res.returncode != 0:
        return {"error": res.stderr.strip() or "non-zero exit"}
    try:
        return json.loads(res.stdout)
    except json.JSONDecodeError as e:
        return {"error": f"json: {e}; stdout: {res.stdout[:200]!r}"}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bench", default=os.environ.get("BENCH", "./target/release/researchdb-bench"))
    p.add_argument("--samples", type=int, default=20)
    p.add_argument("--seed-chunk", type=int, default=1)
    p.add_argument("--anchor-paper", type=int, default=1)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--bm25-text", default="neural network")
    p.add_argument("--ef-search", type=int, default=40)
    p.add_argument("--out", type=Path,
                   default=Path(f"reports/demo_14cells_{dt.datetime.now():%Y%m%d_%H%M%S}.json"))
    args = p.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    print()
    print(f"{'Q':<4} {'plan':<6} {'P50(ms)':>9} {'P95(ms)':>9} {'P99(ms)':>9} "
          f"{'results':>8} {'rtrip':>5} {'first_pred':<12}")
    print("-" * 78)

    cells = []
    for q in range(1, 8):
        for plan in ("naive", "v0", "v1", "v2"):
            r = run_cell(args.bench, plan, q, args)
            if "error" in r:
                print(f"  Q{q}  {plan:<5}  ERROR  {r['error']}")
                continue

            lp = r.get("last_plan", {}) or {}
            fp = lp.get("first_predicate") or "-"
            rt = lp.get("round_trips", 0)

            print(f"  Q{q}  {plan:<5}  "
                  f"{r['p50_us']/1000:>7.2f}   "
                  f"{r['p95_us']/1000:>7.2f}   "
                  f"{r['p99_us']/1000:>7.2f}   "
                  f"{r['first_result_count']:>5}    "
                  f"{rt:>3}    "
                  f"{fp}")

            cells.append({
                "query":   r["query"],
                "plan":    r["plan"],
                "p50_us":  r["p50_us"],
                "p95_us":  r["p95_us"],
                "p99_us":  r["p99_us"],
                "result_count": r["first_result_count"],
                "round_trips":  rt,
                "materializations": lp.get("materializations", 0),
                "first_predicate":  fp,
                "actual_order":     lp.get("actual_order", []),
                "per_engine_rows":  lp.get("per_engine_rows", []),
                "predicates":       lp.get("predicates", []),
            })

    out = {
        "schema": "researchdb.phase1.demo.v1",
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "samples_per_cell": args.samples,
        "params": {
            "seed_chunk":   args.seed_chunk,
            "anchor_paper": args.anchor_paper,
            "depth":        args.depth,
            "ef_search":    args.ef_search,
            "bm25_text":    args.bm25_text,
        },
        "cells": cells,
    }
    args.out.write_text(json.dumps(out, indent=2, ensure_ascii=False))

    print()
    print(f"→ JSON: {args.out}")


if __name__ == "__main__":
    main()
