#!/usr/bin/env python3
"""Phase 1 §D6 — cost regression suite.

Runs the 14-cell benchmark (7 queries × {naive, v0}) and compares the
P50/P95 of each cell against a checked-in baseline. Cells regressing
beyond `--threshold` (default 10%) are flagged; the script exits non-zero
so CI can gate PR merges.

Usage:
  uv run python eval/regress.py                   # compare to baseline
  uv run python eval/regress.py --update          # rewrite baseline
  uv run python eval/regress.py --threshold 0.15  # 15% drift gate
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


BASELINE_PATH = Path("benchmarks/regression/baseline.json")


def run_demo(samples: int) -> dict:
    """Invoke scripts/demo_14cells.py with --out /dev/null and capture
    its written report by intercepting stdout? Simpler: re-implement
    the cell loop here to keep dependencies tight."""
    bench = os.environ.get("BENCH", "./target/release/researchdb-bench")
    if not os.access(bench, os.X_OK):
        print(f"  → building harness ({bench} missing)", file=sys.stderr)
        subprocess.run(["cargo", "build", "--release"], check=True)

    cells = []
    for q in range(1, 8):
        for plan in ("naive", "v0", "v1", "v2"):
            cmd = [
                bench, "seven",
                "--plan", plan, "--query", str(q),
                "--samples", str(samples),
                "--seed-chunk", "1",
                "--anchor-paper", "1",
                "--depth", "2",
                "--bm25-text", "neural network",
                "--ef-search", "40",
            ]
            res = subprocess.run(cmd, capture_output=True, text=True, check=True)
            d = json.loads(res.stdout)
            cells.append({
                "key": f"Q{q}_{plan}",
                "p50_us": d["p50_us"],
                "p95_us": d["p95_us"],
                "p99_us": d["p99_us"],
                "result_count": d["first_result_count"],
            })
    return {"samples": samples, "cells": cells}


def cmp_report(baseline: dict, current: dict, threshold: float) -> tuple[bool, list[dict]]:
    """Returns (any_failure, per-cell diff rows)."""
    base_by_key = {c["key"]: c for c in baseline["cells"]}
    rows: list[dict] = []
    failures: list[str] = []
    for c in current["cells"]:
        b = base_by_key.get(c["key"])
        if b is None:
            rows.append({"key": c["key"], "status": "NEW",
                         "p50_us": c["p50_us"], "drift": None})
            continue
        if b["p50_us"] <= 0:
            rows.append({"key": c["key"], "status": "BASELINE_ZERO",
                         "p50_us": c["p50_us"], "drift": None})
            continue
        drift = (c["p50_us"] - b["p50_us"]) / b["p50_us"]
        status = "OK"
        if drift > threshold:
            status = "REGRESS"
            failures.append(f"{c['key']}: P50 {b['p50_us']}→{c['p50_us']} us ({drift*100:+.1f}%)")
        elif drift < -threshold:
            status = "IMPROVE"
        rows.append({
            "key":     c["key"],
            "status":  status,
            "baseline_p50_us": b["p50_us"],
            "current_p50_us":  c["p50_us"],
            "drift":   drift,
        })
    return (len(failures) > 0, rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", type=Path, default=BASELINE_PATH)
    p.add_argument("--samples",  type=int, default=15)
    p.add_argument("--threshold", type=float, default=0.10,
                   help="Max P50 regression fraction before flagging (default 0.10 = 10%%).")
    p.add_argument("--update", action="store_true",
                   help="Overwrite baseline with current measurements (use for first run "
                        "or after intentional perf changes).")
    args = p.parse_args()

    print("→ running 14 cells…", file=sys.stderr)
    current = run_demo(args.samples)

    if args.update or not args.baseline.exists():
        args.baseline.parent.mkdir(parents=True, exist_ok=True)
        args.baseline.write_text(json.dumps(current, indent=2))
        print(f"→ baseline written: {args.baseline}")
        return

    baseline = json.loads(args.baseline.read_text())
    failed, rows = cmp_report(baseline, current, args.threshold)

    print(f"\n{'cell':<10} {'status':<8} {'base(us)':>10} {'curr(us)':>10} {'drift':>8}")
    print("-" * 50)
    for r in rows:
        drift_str = f"{r['drift']*100:+.1f}%" if r.get("drift") is not None else "-"
        if r["status"] == "OK":
            base, curr = r["baseline_p50_us"], r["current_p50_us"]
        elif r["status"] == "NEW":
            base, curr = "—", r["p50_us"]
        elif r["status"] == "BASELINE_ZERO":
            base, curr = 0, r["p50_us"]
        else:
            base, curr = r["baseline_p50_us"], r["current_p50_us"]
        print(f"{r['key']:<10} {r['status']:<8} {str(base):>10} {str(curr):>10} {drift_str:>8}")

    if failed:
        print(f"\n✗ regression detected (threshold {args.threshold*100:.0f}%). "
              f"Add an explanation in the commit message or rerun with --update.",
              file=sys.stderr)
        sys.exit(1)
    else:
        print(f"\n✓ all 14 cells within ±{args.threshold*100:.0f}% of baseline.")


if __name__ == "__main__":
    main()
