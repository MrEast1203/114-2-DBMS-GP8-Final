#!/usr/bin/env python3
"""Run naive / v1 / v2 / v3 across all labelled queries and compute
Phase 1 §E4 consistency metrics: NDCG@10, Jaccard@10, RBO@10.

Jaccard / RBO are reported pair-wise:
  * v1 / v2 / v3 vs naive (baseline)
  * v3 vs v2 (diagnostic: how much does the multi-stage push-down +
    fusion-signal recovery move the result set?)

For each query in eval/queries.jsonl, we invoke researchdb-bench seven
once per plan and parse the resulting top-k. Latency is collected
inline too (median across the harness's per-call timings).

Output: reports/eval_v3.json by default (override with --out). The
pre-v3 baseline `reports/eval_phase1_e4.json` is deliberately not
overwritten.
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
    ap.add_argument("--out", type=Path, default=Path("reports/eval_v3.json"))
    ap.add_argument("--samples", type=int, default=15)
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()

    # Load queries + ground-truth.
    #
    # Each GT row now carries three independent labels (see
    # eval/augment_gt_per_aspect.py for how they're populated):
    #   * label_sem (int 0/1)            — human topical judgment vs the
    #                                       seed_chunk
    #   * label_lex (int 0/1 or None)    — operational: does BM25 match
    #                                       the abstract for the query's
    #                                       bm25_text? None when the
    #                                       query has no lex predicate
    #                                       (Q1 / Q3 / Q4).
    #   * label_gph (int 0/1 or None)    — operational: is the paper in
    #                                       reverse-BFS(anchor, depth)?
    #                                       None when the query has no
    #                                       graph predicate (Q1 / Q2 / Q6).
    #
    # For every (qid, paper_id), the *effective relevance* the scoring
    # code uses is the AND of the predicate labels actually demanded by
    # that query type:
    #
    #   Q1 sem      → label_sem
    #   Q2 lex      → label_lex
    #   Q3 gph      → label_gph
    #   Q4 sem ∩ gph→ label_sem ∧ label_gph
    #   Q5 lex ∩ gph→ label_lex ∧ label_gph
    #   Q6 sem ∩ lex→ label_sem ∧ label_lex
    #   Q7 all three→ label_sem ∧ label_lex ∧ label_gph
    #
    # This decoupling exists so that papers can't be credited for a
    # predicate they don't actually satisfy (e.g. a paper semantically
    # about ResNet whose abstract doesn't contain "batch normalization"
    # should NOT count as relevant for Q6, even if the human marked it
    # topically related).
    queries_list = [json.loads(l) for l in args.queries.read_text().splitlines() if l.strip()]
    queries_by_qid = {q["qid"]: q for q in queries_list}

    # Predicates each query type composes.
    QTYPE_PREDICATES = {
        "Q1": ("sem",),
        "Q2": ("lex",),
        "Q3": ("gph",),
        "Q4": ("sem", "gph"),
        "Q5": ("lex", "gph"),
        "Q6": ("sem", "lex"),
        "Q7": ("sem", "lex", "gph"),
    }

    # rel[qid][paper_id] = effective binary label (after AND of predicates).
    rel: dict[str, dict[int, int]] = defaultdict(dict)
    per_aspect: dict[str, dict[int, dict[str, int | None]]] = defaultdict(dict)
    for l in args.gt.read_text().splitlines():
        if not l.strip(): continue
        r = json.loads(l)
        qid = r["qid"]
        pid = r["paper_id"]
        # Backwards compat: if the GT row predates the augmentation,
        # treat `label` as label_sem and the operational labels as None.
        lab_sem = r.get("label_sem", r.get("label"))
        lab_lex = r.get("label_lex")
        lab_gph = r.get("label_gph")
        per_aspect[qid][pid] = {
            "sem": lab_sem, "lex": lab_lex, "gph": lab_gph,
        }
        # Compute effective relevance for this qid.
        qtype = queries_by_qid.get(qid, {}).get("type", r.get("qtype"))
        preds = QTYPE_PREDICATES.get(qtype, ())
        aspect_vals = []
        for p in preds:
            v = {"sem": lab_sem, "lex": lab_lex, "gph": lab_gph}[p]
            if v is None:
                # Predicate label missing — fall through to skip this row.
                aspect_vals = None
                break
            aspect_vals.append(int(v))
        if aspect_vals is None:
            continue
        rel[qid][pid] = 1 if all(a == 1 for a in aspect_vals) else 0
    queries = queries_list

    PLANS = ("naive", "v1", "v2", "v3")

    rows = []      # one per (qid) — bundles all four plans (back-compat shape)
    results = []   # FLAT list of (plan, qid, ...) — used by automated checks
    for q in queries:
        qid = q["qid"]
        if qid not in rel or not rel[qid]:
            print(f"  skip {qid}: no ground truth (empty pool)", file=sys.stderr)
            continue

        per_plan = {}
        for plan in PLANS:
            top, p50, p95 = run_bench(plan, q, args.k, args.samples)
            per_plan[plan] = {"top": top, "p50_ms": p50, "p95_ms": p95}

        # NDCG per plan
        ndcgs = {p: ndcg(per_plan[p]["top"], rel[qid], args.k) for p in PLANS}

        # Pair-wise Jaccard / RBO (all vs naive, plus v3 vs v2 diagnostic)
        def J(a, b): return jaccard(a, b, args.k)
        def R(a, b): return rbo(a, b, p=0.9, k=args.k)

        naive_top = per_plan["naive"]["top"]
        v2_top    = per_plan["v2"]["top"]
        v3_top    = per_plan["v3"]["top"]

        is_single = q["type"] in ("Q1", "Q2", "Q3")

        row = {
            "qid":   qid,
            "type":  q["type"],
            "desc":  q["desc"],
            **{p: per_plan[p] for p in PLANS},
            "ndcg10_naive":         ndcgs["naive"],
            "ndcg10_v1":            ndcgs["v1"],
            "ndcg10_v2":            ndcgs["v2"],
            "ndcg10_v3":            ndcgs["v3"],
            "jaccard10_v1_naive":   J(naive_top, per_plan["v1"]["top"]),
            "jaccard10_v2_naive":   J(naive_top, v2_top),
            "jaccard10_v3_naive":   J(naive_top, v3_top),
            "jaccard10_v3_v2":      J(v2_top,    v3_top),
            "rbo10_v1_naive":       R(naive_top, per_plan["v1"]["top"]),
            "rbo10_v2_naive":       R(naive_top, v2_top),
            "rbo10_v3_naive":       R(naive_top, v3_top),
            "rbo10_v3_v2":          R(v2_top,    v3_top),
            "equiv":                "hard" if is_single else "soft",
        }
        rows.append(row)

        # Flat per-(plan, qid) result rows for automated checks.
        for plan in PLANS:
            results.append({
                "plan":   plan,
                "qid":    qid,
                "type":   q["type"],
                "p50_ms": per_plan[plan]["p50_ms"],
                "p95_ms": per_plan[plan]["p95_ms"],
                "top":    per_plan[plan]["top"],
                "ndcg10": ndcgs[plan],
            })

        print(
            f"  {qid:>6}  type={q['type']}  "
            f"naive ndcg={ndcgs['naive']:.2f} p50={per_plan['naive']['p50_ms']:5.1f} | "
            f"v1 ndcg={ndcgs['v1']:.2f} p50={per_plan['v1']['p50_ms']:5.1f} | "
            f"v2 ndcg={ndcgs['v2']:.2f} p50={per_plan['v2']['p50_ms']:5.1f} | "
            f"v3 ndcg={ndcgs['v3']:.2f} p50={per_plan['v3']['p50_ms']:5.1f} | "
            f"J(v3,v2)={row['jaccard10_v3_v2']:.2f}",
            file=sys.stderr,
        )

    # Aggregate
    def mean(xs):
        xs = [x for x in xs if not math.isnan(x)]
        return statistics.mean(xs) if xs else float("nan")

    summary = {
        "n_queries":               len(rows),
        "mean_ndcg10_naive":       mean(r["ndcg10_naive"] for r in rows),
        "mean_ndcg10_v1":          mean(r["ndcg10_v1"]    for r in rows),
        "mean_ndcg10_v2":          mean(r["ndcg10_v2"]    for r in rows),
        "mean_ndcg10_v3":          mean(r["ndcg10_v3"]    for r in rows),
        "mean_jaccard10_v1_naive": mean(r["jaccard10_v1_naive"] for r in rows),
        "mean_jaccard10_v2_naive": mean(r["jaccard10_v2_naive"] for r in rows),
        "mean_jaccard10_v3_naive": mean(r["jaccard10_v3_naive"] for r in rows),
        "mean_jaccard10_v3_v2":    mean(r["jaccard10_v3_v2"]    for r in rows),
        "mean_rbo10_v1_naive":     mean(r["rbo10_v1_naive"]     for r in rows),
        "mean_rbo10_v2_naive":     mean(r["rbo10_v2_naive"]     for r in rows),
        "mean_rbo10_v3_naive":     mean(r["rbo10_v3_naive"]     for r in rows),
        "mean_rbo10_v3_v2":        mean(r["rbo10_v3_v2"]        for r in rows),
        "naive_mean_p50_ms":       mean(r["naive"]["p50_ms"] for r in rows),
        "v1_mean_p50_ms":          mean(r["v1"]["p50_ms"]    for r in rows),
        "v2_mean_p50_ms":          mean(r["v2"]["p50_ms"]    for r in rows),
        "v3_mean_p50_ms":          mean(r["v3"]["p50_ms"]    for r in rows),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "schema":  "researchdb.phase1.e4_eval.v3",
        "summary": summary,
        "rows":    rows,
        "results": results,
    }, indent=2))
    print("\nsummary:", json.dumps(summary, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
