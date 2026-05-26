#!/usr/bin/env python3
"""Cold OS-drop full sweep — 28-cell cold/warm matrix.

Runs bench cold-warm with --cold-method container-restart on all 28
cells (7 queries × 4 plans: naive, v1, v2, v3). Writes one JSON per
cell into reports/coldwarm_q{1..7}_{plan}.json plus a consolidated
reports/coldwarm_v3.json.

container-restart strategy is portable across macOS / WSL / Linux —
unlike /proc/sys/vm/drop_caches which is read-only on macOS Docker.
It empties Postgres shared_buffers; host page cache is still warm, so
this is "cold Postgres + warm host disk", closer to true cold than
fresh-conn but not as severe as host purge.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

BENCH = "./target/release/researchdb-bench"

PLANS = ("naive", "v1", "v2", "v3")


def run_cell(query: int, plan: str, samples: int = 5) -> dict:
    out_path = f"reports/coldwarm_q{query}_{plan}.json"
    cmd = [
        BENCH, "cold-warm",
        "--query", str(query),
        "--plan", plan,
        "--samples", str(samples),
        "--warmup", "5",
        "--cold-method", "container-restart",
        "--seed-chunk", "1",
        "--anchor-paper", "1",
        "--depth", "2",
        "--bm25-text", "neural network",
        "--ef-search", "40",
        "--out", out_path,
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if res.returncode != 0:
        return {"error": res.stderr.strip()[:200]}
    return json.loads(Path(out_path).read_text())


def main():
    cells = []
    print(f"{'cell':<12} {'cold ms':>9} {'warm ms':>9} {'ratio':>7}", file=sys.stderr)
    print("-" * 47, file=sys.stderr)
    for q in range(1, 8):
        for plan in PLANS:
            r = run_cell(q, plan)
            if "error" in r:
                print(f"Q{q} {plan:<5}  ERROR  {r['error']}", file=sys.stderr)
                continue
            cold_ms = r["cold"]["p50_us"] / 1000
            warm_ms = r["warm"]["p50_us"] / 1000
            ratio = r["cold_over_warm_p50"]
            print(f"Q{q} {plan:<5}  {cold_ms:>7.2f}  {warm_ms:>7.2f}  {ratio:>5.2f}x",
                  file=sys.stderr)
            cells.append({
                "query": q, "plan": plan,
                "cold_p50_ms": cold_ms, "warm_p50_ms": warm_ms,
                "cold_p95_ms": r["cold"]["p95_us"] / 1000,
                "warm_p95_ms": r["warm"]["p95_us"] / 1000,
                "ratio_p50": ratio,
                "ratio_p95": r["cold_over_warm_p95"],
            })

    out = {
        "schema": "researchdb.phase1.coldwarm_full.v3",
        "method": "container-restart",
        "samples_per_cell": 5,
        "warmup_per_cell": 5,
        "cells": cells,
    }
    Path("reports/coldwarm_v3.json").write_text(json.dumps(out, indent=2))
    print("\nwrote reports/coldwarm_v3.json")


if __name__ == "__main__":
    main()
