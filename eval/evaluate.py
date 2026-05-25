#!/usr/bin/env python3
"""Run naive vs v0 across all labelled queries and compute Phase 1 §E4
consistency metrics: NDCG@10, Jaccard@10, RBO@10.

For each query in eval/queries.jsonl, we invoke researchdb-bench seven
once per plan and parse the resulting top-k. Latency is collected
inline too (median across the harness's per-call timings).

Output: reports/eval_phase1_e4.json
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

BENCH = "./target/release/researchdb-bench"


# ---------- metrics --------------------------------------------------------

def jaccard(a: list[int], b: list[int], k: int = 10) -> float:
    sa, sb = set(a[:k]), set(b[:k])
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / max(1, len(sa | sb))


def rbo(a: list[int], b: list[int], p: float = 0.9, k: int = 10) -> float:
    """Rank-biased overlap (Webber, Moffat, Zobel 2010), length-normalized
    so identical finite lists of any length return exactly 1.0.

    p closer to 1 weights deeper ranks more; p=0.9 is the typical IR
    default. The plain "RBO_min" variant from the paper gives
    1 - p^n < 1 for identical lists of length n, which we normalize away
    by dividing by that ideal value.
    """
    a, b = a[:k], b[:k]
    n = max(len(a), len(b))
    if n == 0:
        return 1.0 if not a and not b else 0.0
    sa, sb = set(), set()
    summed = 0.0
    for d in range(1, n + 1):
        if d - 1 < len(a):
            sa.add(a[d - 1])
        if d - 1 < len(b):
            sb.add(b[d - 1])
        summed += (p ** (d - 1)) * (len(sa & sb) / d)
    ideal = (1 - p ** n) / (1 - p)  # RBO_min of identical lists
    return summed / ideal if ideal > 0 else 0.0


def ndcg(ranked: list[int], relevance: dict[int, int], k: int = 10) -> float:
    """Binary-relevance NDCG@k."""
    if not ranked:
        return 0.0
    gains = [relevance.get(pid, 0) for pid in ranked[:k]]
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
    n_pos = sum(1 for v in relevance.values() if v == 1)
    if n_pos == 0:
        return float("nan")
    ideal_len = min(k, n_pos)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_len))
    return dcg / idcg if idcg > 0 else 0.0


# ---------- harness driver ------------------------------------------------

def run_bench(plan: str, q: dict, k: int, samples: int) -> tuple[list[int], float, float]:
    args = [
        BENCH, "seven",
        "--plan", plan,
        "--query", str(int(q["type"][1])),
        "--samples", str(samples),
        "--k", str(k),
    ]
    if q.get("seed_chunk") is not None:
        args += ["--seed-chunk", str(q["seed_chunk"])]
    if q.get("anchor_paper") is not None:
        args += ["--anchor-paper", str(q["anchor_paper"])]
    if q.get("depth") is not None:
        args += ["--depth", str(q["depth"])]
    if q.get("bm25_text"):
        args += ["--bm25-text", q["bm25_text"]]

    res = subprocess.run(args, capture_output=True, text=True, check=False)
    if res.returncode != 0:
        return [], float("nan"), float("nan")
    try:
        d = json.loads(res.stdout)
    except json.JSONDecodeError:
        return [], float("nan"), float("nan")
    top = d.get("last_plan", {}).get("paper_ids", []) or []
    return top, d.get("p50_us", 0) / 1000.0, d.get("p95_us", 0) / 1000.0


# ---------- main ----------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--queries", type=Path, default=Path("eval/queries.jsonl"))
    ap.add_argument("--gt", type=Path, default=Path("eval/ground-truth.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("reports/eval_phase1_e4.json"))
    ap.add_argument("--samples", type=int, default=15)
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()

    # Load queries + ground-truth (per-query relevance dict).
    queries = [json.loads(l) for l in args.queries.read_text().splitlines() if l.strip()]
    rel: dict[str, dict[int, int]] = defaultdict(dict)
    for l in args.gt.read_text().splitlines():
        if not l.strip(): continue
        r = json.loads(l)
        if r.get("label") is None: continue
        rel[r["qid"]][r["paper_id"]] = int(r["label"])

    rows = []
    for q in queries:
        qid = q["qid"]
        if qid not in rel or not rel[qid]:
            print(f"  skip {qid}: no ground truth (empty pool)", file=sys.stderr)
            continue

        naive_top, n_p50, n_p95 = run_bench("naive", q, args.k, args.samples)
        v0_top,    v_p50, v_p95 = run_bench("v0",    q, args.k, args.samples)
        v1_top,    v1_p50, v1_p95 = run_bench("v1",  q, args.k, args.samples)
        v2_top,    v2_p50, v2_p95 = run_bench("v2",  q, args.k, args.samples)

        # Hard / soft equivalence depending on QueryType.
        is_single = q["type"] in ("Q1", "Q2", "Q3")
        row = {
            "qid":   qid,
            "type":  q["type"],
            "desc":  q["desc"],
            "naive": {"top": naive_top, "p50_ms": n_p50, "p95_ms": n_p95},
            "v0":    {"top": v0_top,    "p50_ms": v_p50, "p95_ms": v_p95},
            "v1":    {"top": v1_top,    "p50_ms": v1_p50, "p95_ms": v1_p95},
            "v2":    {"top": v2_top,    "p50_ms": v2_p50, "p95_ms": v2_p95},
            "ndcg10_naive": ndcg(naive_top, rel[qid], args.k),
            "ndcg10_v0":    ndcg(v0_top,    rel[qid], args.k),
            "ndcg10_v1":    ndcg(v1_top,    rel[qid], args.k),
            "ndcg10_v2":    ndcg(v2_top,    rel[qid], args.k),
            "jaccard10_v0_naive": jaccard(naive_top, v0_top, args.k),
            "jaccard10_v1_v0":    jaccard(v0_top,    v1_top, args.k),
            "jaccard10_v2_v0":    jaccard(v0_top,    v2_top, args.k),
            "rbo10_v0_naive":     rbo(naive_top, v0_top, p=0.9, k=args.k),
            "rbo10_v1_v0":        rbo(v0_top,    v1_top, p=0.9, k=args.k),
            "rbo10_v2_v0":        rbo(v0_top,    v2_top, p=0.9, k=args.k),
            "equiv":        "hard" if is_single else "soft",
        }
        rows.append(row)
        print(f"  {qid:>6}  type={q['type']}  "
              f"naive ndcg={row['ndcg10_naive']:.2f} p50={n_p50:5.1f} | "
              f"v0 ndcg={row['ndcg10_v0']:.2f} p50={v_p50:5.1f} | "
              f"v1 ndcg={row['ndcg10_v1']:.2f} p50={v1_p50:5.1f} | "
              f"v2 ndcg={row['ndcg10_v2']:.2f} p50={v2_p50:5.1f} | "
              f"J(v2,v0)={row['jaccard10_v2_v0']:.2f}",
              file=sys.stderr)

    # Aggregate
    def mean(xs):
        xs = [x for x in xs if not math.isnan(x)]
        return statistics.mean(xs) if xs else float("nan")

    summary = {
        "n_queries":          len(rows),
        "mean_ndcg10_naive":  mean(r["ndcg10_naive"] for r in rows),
        "mean_ndcg10_v0":     mean(r["ndcg10_v0"]    for r in rows),
        "mean_ndcg10_v1":     mean(r["ndcg10_v1"]    for r in rows),
        "mean_ndcg10_v2":     mean(r["ndcg10_v2"]    for r in rows),
        "mean_jaccard10_v0_naive": mean(r["jaccard10_v0_naive"] for r in rows),
        "mean_jaccard10_v1_v0":    mean(r["jaccard10_v1_v0"]    for r in rows),
        "mean_jaccard10_v2_v0":    mean(r["jaccard10_v2_v0"]    for r in rows),
        "mean_rbo10_v0_naive":     mean(r["rbo10_v0_naive"]     for r in rows),
        "mean_rbo10_v1_v0":        mean(r["rbo10_v1_v0"]        for r in rows),
        "mean_rbo10_v2_v0":        mean(r["rbo10_v2_v0"]        for r in rows),
        "naive_mean_p50_ms":  mean(r["naive"]["p50_ms"] for r in rows),
        "v0_mean_p50_ms":     mean(r["v0"]["p50_ms"]    for r in rows),
        "v1_mean_p50_ms":     mean(r["v1"]["p50_ms"]    for r in rows),
        "v2_mean_p50_ms":     mean(r["v2"]["p50_ms"]    for r in rows),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "schema":  "researchdb.phase1.e4_eval.v1",
        "summary": summary,
        "rows":    rows,
    }, indent=2))
    print("\nsummary:", json.dumps(summary, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
