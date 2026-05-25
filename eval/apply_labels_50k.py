#!/usr/bin/env python3
"""Build eval/ground-truth.jsonl for the 50K corpus.

Strategy:
1. Load old labels (5K ground truth) — index by openalex_id (stable across reloads).
2. Load new LLM labels for 50K candidate pool — from eval/labels_50k_new.py.
3. For each (qid, paper) in eval/candidates_50k.jsonl, look up its label:
   - first in new labels (by qid + openalex_id),
   - else in old labels (by openalex_id alone — any qid that labeled this paper).
4. Emit one JSONL line per labelled (qid, paper) with metadata.

Output: eval/ground-truth.jsonl (overwrites the 5K version).
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import psycopg


def dsn():
    import os
    return os.environ.get("DATABASE_URL", "postgres://researchdb:researchdb@localhost:5432/researchdb")


def load_new_labels() -> dict[tuple[str, str], int]:
    spec = importlib.util.spec_from_file_location("lbl", "eval/labels_50k_new.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)  # type: ignore[union-attr]
    return m.NEW_LABELS


def load_old_labels() -> dict[str, int]:
    """Old labels indexed by openalex_id. Take the strongest label across qids
    (any positive → positive, otherwise majority)."""
    by_oa: dict[str, list[int]] = {}
    for line in open("eval/ground-truth.jsonl"):
        r = json.loads(line)
        if r.get("label") is None:
            continue
        by_oa.setdefault(r["openalex_id"], []).append(int(r["label"]))
    # Heuristic: positive if any annotator marked 1.
    return {oa: max(vs) for oa, vs in by_oa.items()}


def main():
    new_lbls = load_new_labels()
    old_lbls = load_old_labels()

    out_lines: list[str] = []
    unlabelled = 0
    by_qid_stats: dict[str, dict[str, int]] = {}
    with psycopg.connect(dsn(), autocommit=False) as conn, conn.cursor() as cur:
        for line in open("eval/candidates_50k.jsonl"):
            r = json.loads(line)
            qid = r["qid"]
            oa = r["openalex_id"]
            key = (qid, oa)
            if key in new_lbls:
                lbl = int(new_lbls[key])
            elif oa in old_lbls:
                lbl = int(old_lbls[oa])
            else:
                unlabelled += 1
                continue

            cur.execute("SELECT id, title, publish_year, venue, left(abstract, 600) FROM papers WHERE openalex_id = %s", (oa,))
            row = cur.fetchone()
            if row is None:
                unlabelled += 1
                continue
            pid, title, year, venue, abstract = row

            out_lines.append(json.dumps({
                "qid":           qid,
                "qtype":         r["qtype"],
                "qdesc":         r["qdesc"],
                "paper_id":      pid,
                "openalex_id":   oa,
                "title":         title,
                "year":          year,
                "venue":         venue,
                "abstract_clip": abstract,
                "label":         lbl,
            }, ensure_ascii=False))

            s = by_qid_stats.setdefault(qid, {"pos": 0, "neg": 0})
            s["pos" if lbl == 1 else "neg"] += 1

    out_path = Path("eval/ground-truth.jsonl")
    # Back up the 5K version once.
    backup = Path("eval/ground-truth_5k.jsonl")
    if not backup.exists():
        backup.write_text(out_path.read_text())
        print(f"  backed up 5K ground truth → {backup}")
    out_path.write_text("\n".join(out_lines) + "\n")

    print(f"\nwrote {len(out_lines)} labelled (qid, paper) → {out_path}")
    print(f"unlabelled: {unlabelled}\n")
    print(f"{'qid':<6} {'pos':>5} {'neg':>5} {'rate':>6}")
    total_pos = total_neg = 0
    for qid, s in sorted(by_qid_stats.items()):
        rate = s["pos"] / (s["pos"] + s["neg"]) if (s["pos"] + s["neg"]) > 0 else 0
        print(f"{qid:<6} {s['pos']:>5} {s['neg']:>5} {rate:>5.0%}")
        total_pos += s["pos"]; total_neg += s["neg"]
    rate = total_pos / (total_pos + total_neg)
    print(f"{'total':<6} {total_pos:>5} {total_neg:>5} {rate:>5.0%}")


if __name__ == "__main__":
    main()
