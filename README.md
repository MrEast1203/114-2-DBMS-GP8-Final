# Hybrid Search Orchestration on PostgreSQL

Course: **NTU 114-2 DBMS · Final Project, Group 8**
Author: Chan Ching-Kan (chingkan.chan1203@gmail.com)

A research-grade benchmark of **four plans** for running
**semantic + lexical + graph** search inside a single PostgreSQL instance.
The corpus is 36 740 OpenAlex paper abstracts; the three engines are
`pgvector` (HNSW), `pg_search` (BM25, ParadeDB), and PostgreSQL's native
`WITH RECURSIVE` over the citation graph. A Rust orchestrator on top of
the database fuses the three engines with Reciprocal Rank Fusion and
implements increasingly sophisticated plans
(`naive` → `v1` → `v2` → `v3`) that progress from textbook fixed-order,
to cost-aware reorder, to **graph filter push-down into the ranker SQL**,
to **multi-stage push-down + cost-driven ordering + three-way fusion**.

The full write-up — motivation, schema, four plans, ground-truth
construction, results, and reproduction notes — lives in
`docs/report.html`. This README is the public entry point.

---

## TL;DR results (50K papers, WSL2, warm cache)

Two snapshots — the original baseline (samples=15, naive/v1/v2 from
`reports/eval_phase1_e4.json`) and a same-run v3 comparison (samples=10,
all four plans measured back-to-back on the same machine, see
`reports/eval_v3.json`):

| plan  | mean P50 | NDCG@10 | top-10 vs naive       |
| ----- | -------- | ------- | --------------------- |
| naive | 51.36 ms | 0.675   | — (baseline)          |
| v1    | 50.92 ms | 0.675   | Jaccard 1.000 (same)  |
| v2    | **35.75 ms** | **0.801** | Jaccard 0.601 (differs) |
| v3    | 27.62 ms (same-run baseline 23.38) | 0.562 ⚠ | Jaccard 0.265 vs naive · 0.328 vs v2 |

- **v2 push-down** cuts mean latency 1.44× and lifts NDCG@10 by
  +0.126 by restricting the ranker to the graph-filtered candidate set
  rather than ranking the whole corpus and filtering afterward.
- **v3 multi-stage push-down + cost ordering + three-way fusion**:
  P50 beats naive/v1 (1.23× vs naive) and beats v2 on Q6 (1.21×) and
  Q4 (1.35×), but loses on Q5/Q7. ⚠ **NDCG drops to 0.562** — lower
  than v2's 0.801 and even lower than naive's 0.675. Root cause: the
  third ranking signal (`graph_distance` = BFS depth) added to RRF is
  weak relative to vector/BM25 and, at equal RRF weight (k=60, unchanged
  per spec), dilutes the two stronger signals. v3 is preserved as the
  honest "negative result" — the cost-actionable + multi-stage push-down
  *mechanism* works, but on this corpus the fusion-signal recovery
  hurts ranking quality. See `reports/v3_summary.md` and
  `docs/report.html` §4.7 / §8.4 / §11.2 for the full disclaim.
- `naive` and `v1` return **bit-identical top-10s** on every query.
  Cost-based reorder by itself does *not* change latency or quality in
  this pipeline — every engine must run to completion regardless of
  order (RRF needs all three rankings; no engine has data
  dependencies the others can short-circuit). v1 is preserved as the
  clean "reorder only" ablation that motivates v2's push-down.
- BFS shootout: `WITH RECURSIVE` beats Apache AGE Cypher by 71×–7 380×
  across all 9 depth × out-degree cells, with zero result mismatches.

Detailed per-query numbers and methodology are in `reports/*.json` (see
[Reports](#reports) below).

---

## Repository layout

```
.
├── bench/              # Rust orchestrator + benchmark CLI (researchdb-bench)
│   └── src/
│       ├── plan.rs         # naive / v1 / v2 / v3 plan implementations
│       ├── fusion.rs       # RRF (k=60) ranking fusion
│       ├── cost.rs         # cost models per engine
│       ├── coldwarm.rs     # cold-vs-warm cache measurement
│       ├── storage.rs      # disk usage breakdown
│       └── graph_engine.rs # BFS shootout: AGE vs WITH RECURSIVE
├── ingest/             # Python data pipeline
│   ├── openalex_fetch.py   # BFS over OpenAlex from 7 seed papers
│   ├── load_graph.py       # JSONL → PostgreSQL + AGE projection
│   └── embed_chunks.py     # MiniLM (all-MiniLM-L6-v2) embeddings
├── eval/               # Evaluation harness
│   ├── queries.jsonl       # 20 hand-crafted Q4–Q7 queries
│   ├── ground-truth.jsonl  # 1 368 hand-labelled (query, paper) pairs
│   ├── build_candidate_pool.py  # TREC-style pooling
│   └── evaluate.py         # NDCG@10 / Jaccard / RBO computation
├── migrations/         # 4 SQL migrations (paper schema, AGE, indexes)
├── scripts/            # Helper scripts (coldwarm_all_21.py, coldwarm_all_28.py)
├── reports/            # All measurement JSON outputs
├── docker/             # Custom PG 16 + pgvector + pg_search image
├── Cargo.toml, pyproject.toml, Makefile, docker-compose.yml
└── data/openalex_50k/  # Not committed — fetched per §Reproduction
```

---

## Architecture

```
 ┌──────────┐    ┌──────────────────┐    ┌─────────────────────────────┐
 │ user/TA  │ →  │ researchdb-bench │ →  │ PostgreSQL 16               │
 │ (CLI)    │    │  Orchestrator    │    │  · pgvector 0.8 (HNSW)      │
 └──────────┘    │  (Rust + sqlx)   │    │  · pg_search 0.23 (BM25)    │
                 │  · selectivity   │    │  · WITH RECURSIVE (citation)│
                 │  · cost estimate │    │  · Apache AGE (kept, unused)│
                 │  · push-down(v2) │    └─────────────────────────────┘
                 │  · RRF (k=60)    │
                 └──────────────────┘
                          │
                          ▼
              top-k paper_id list → eval/evaluate.py → NDCG / Jaccard / RBO
```

The orchestrator is **not** inside PostgreSQL. It lives in the Rust
application layer and decides per-engine execution order, cost
estimates, and (in `v2`) whether to materialize the graph-filter set
and inject it as `WHERE id = ANY($filter)` into the ranker SQL — the
key trick that makes v2 faster *and* more precise.

### The four plans

| capability                                                                 | naive | v1 | v2 | v3 |
| -------------------------------------------------------------------------- | ----- | -- | -- | -- |
| RRF fusion across engines (k=60)                                           |  ✓   | ✓ | ✓ | ✓ |
| reorder engines by selectivity (cost-aware annotation)                     |       | ✓ | ✓ | ✓ |
| push graph filter down into ranker SQL (single-stage)                      |       |    | ✓ | ✓ |
| **multi-stage push-down (graph + lexical, Q5 / Q7)**                       |       |    |    | ✓ |
| **cost decides push-down ordering (actionable, not just annotation)**      |       |    |    | ✓ |
| **three-way fusion (vector + BM25 + graph_distance) on narrowed candidate**|       |    |    | ✓ |

The cost-based reorder in v1 does *not* change latency or quality in
this pipeline (see TL;DR): every engine has to run to completion for
RRF, and the engines have no data dependency that reorder could
short-circuit. v1 is kept as the clean ablation that shows this; v2's
push-down is what actually moves the needle.

v3 attempts to claim **three things simultaneously**: (a) extend v2's
single-stage push-down to multi-stage (so when *both* graph and lexical
are hard predicates — Q5 + Q7 — the cost formula finally has somewhere
to act, i.e. *which* hard predicate to materialize first); (b) re-score
the graph filter as a `graph_distance` ranking signal and feed it
back into RRF as a third input alongside vector / BM25, recovering the
three-way fusion that naive had; (c) keep RRF's k=60 unchanged. **All
three mechanisms work, but on this corpus the equal-weight RRF
dilutes vector / BM25 with the weaker graph_distance signal — v3's mean
NDCG@10 drops to 0.562**. v3 is preserved as the honest negative
result: push-down is a pure win, but turning a hard predicate into a
ranking signal at equal weight is *not*. See `docs/report.html` §4.7
and `reports/v3_summary.md` for the full causal chain.

The BFS cost formula `branching^depth` is used by v1 only for
annotation — it's the empirical fit from `bench micro-bench-age` (see
`bench/src/microbench.rs`). v2's push-down decision is triggered by
query type (Q4 / Q5 / Q7), not by reading cost at runtime, so the
cost formula is not listed as a v2 capability. **v3 is the first plan
where the BFS cost formula is *actionable*** (it decides the
push-down order between graph and lexical in Q5 / Q7); however, on
this corpus + depth=2 the BFS estimate is always cheaper, so all 10
Q5/Q7 cells end up picking BFS first — the mechanism is in place but
the cost decision happens to be the same as v2's hard-coded
graph-first ordering.

### The seven query types

| #  | type             | engines used                  |
| -- | ---------------- | ----------------------------- |
| Q1 | semantic         | pgvector                      |
| Q2 | lexical          | pg_search (BM25)              |
| Q3 | graph BFS        | WITH RECURSIVE                |
| Q4 | semantic ∩ graph | pgvector + recursion          |
| Q5 | lexical  ∩ graph | pg_search + recursion         |
| Q6 | semantic ∩ lexical | pgvector + pg_search        |
| Q7 | all three        | pgvector + pg_search + recursion |

Evaluation focuses on Q4–Q7 (the truly multi-engine queries); Q1–Q3
exist to confirm baseline parity across plans.

---

## Reports

All numbers cited in the project report are reproducible from the JSON
artifacts committed under `reports/`:

| file                              | purpose                                        |
| --------------------------------- | ---------------------------------------------- |
| `eval_phase1_e4.json`             | 20 queries × 3 plans (naive/v1/v2) · P50 + NDCG + Jaccard + RBO (baseline, samples=15) |
| `eval_v3.json`                    | **20 queries × 4 plans (naive/v1/v2/v3)** · same metrics + v3-vs-v2 pairwise · `results` flat list for tooling |
| `v3_summary.md`                   | **v3 markdown summary** — mean P50 / NDCG / per-query diff vs v2 / honest disclaim of regressions |
| `coldwarm_full_21.json`           | 7 queries × 3 plans · cold-vs-warm P50 matrix (baseline) |
| `coldwarm_v3.json`                | **7 queries × 4 plans · cold-vs-warm P50 matrix** (28 cells, includes v3) |
| `storage.json`                    | per-relation / per-index disk usage breakdown  |
| `bfs_shootout.json`               | AGE Cypher vs `WITH RECURSIVE`, 3 depth × 3 bucket |
| `coldwarm_q{1..7}_{naive,v1,v2,v3}.json` | individual cells of the cold/warm matrix (v3 cells overwrite the prior `naive,v1,v2` per-cell files when re-run via `scripts/coldwarm_all_28.py`) |

If you just want to inspect the data, read those JSONs directly — no
need to set up the full pipeline.

---

## Reproduction

> Want to look at numbers? Read `reports/*.json` and `eval/ground-truth.jsonl` directly.
> Want to **rerun** end-to-end on a fresh machine? Follow this section.
> **Strongly recommend running `make reset` first** to start from a known-clean container/volume/reports state (your OpenAlex raw data under `data/` is preserved — fetching it again costs 30–60 min of polite-pool rate-limited API calls).

### Prerequisites

- **Docker** with Compose support
- **Rust** ≥ 1.81 (for `cargo build --release`)
- **uv** (Python package manager — `curl -LsSf https://astral.sh/uv/install.sh | sh`)
- **OpenAlex contact email** (only if you choose to re-fetch the corpus from OpenAlex — set `export OPENALEX_MAILTO=you@example.com`)

### Step-by-step (50K papers, WSL2 Docker reference timing ≈ 3.5–4 h)

```bash
# 0 · Clone
git clone https://github.com/MrEast1203/114-2-DBMS-GP8-Final.git
cd 114-2-DBMS-GP8-Final

# 1 · Recommended: clean slate (keeps data/ but resets container + volume + reports/)
make reset

# 2 · Python deps (the --extra embed flag is required for sentence-transformers)
uv sync --extra embed

# 3 · Start PostgreSQL + pgvector + pg_search container
make up

# 4 · Apply migrations (creates the schema + empty HNSW/BM25 indexes)
make migrate

# 5 · Get the 50K corpus JSONL
#     Option A (recommended, ~1 min): Dropbox snapshot — preserves the exact
#     row order that ground-truth.jsonl's paper_id values were assigned against.
#     Option B (30–60 min, will drift): re-fetch from OpenAlex. Paper IDs will
#     not align with ground-truth.jsonl, so NDCG drops sharply — that's an
#     ID-mapping artifact, not a plan regression.
mkdir -p data/openalex_50k
curl -L "https://www.dropbox.com/scl/fo/en2hrtwp5o66o5m2930up/AKSJBt5-uWShOC6znXaFCj8?rlkey=yaaloc46din3csufwa8karbzt&dl=1" \
    -o /tmp/openalex_50k.zip
unzip -j /tmp/openalex_50k.zip -d data/openalex_50k/
rm /tmp/openalex_50k.zip
# verify: 36 740 lines, first row should be ResNet (W2194775991)
wc -l data/openalex_50k/papers.jsonl

# 6 · Load relational tables + AGE projection (WSL2 Docker: 22–35 min)
uv run python ingest/load_graph.py --src data/openalex_50k --reset

# 7 · Embed abstracts with MiniLM (CPU: 2–3 h for 36K abstracts;
#     first run downloads ~80 MB model)
uv run python ingest/embed_chunks.py --strategy minilm --force

# 8 · Reproduce the headline tables (now 4 plans, including v3)
uv run python eval/evaluate.py \
    --samples 10 \
    --queries eval/queries.jsonl \
    --gt eval/ground-truth.jsonl \
    --out reports/eval_v3.json
uv run python scripts/coldwarm_all_28.py   # 4 plans × 7 queries = 28 cells
./target/release/researchdb-bench storage      --out reports/storage.json
./target/release/researchdb-bench bfs-shootout --samples 30
# Optional · regenerate the pre-v3 baseline files (do not commit if you want
# to keep eval_phase1_e4.json and coldwarm_full_21.json unchanged):
# uv run python scripts/coldwarm_all_21.py
```

### Cleanup targets

| command       | stop container | drop volume | delete reports/ | delete data/ |
| ------------- | -------------- | ----------- | --------------- | ------------ |
| `make clean`  |       —        |     —       |      ✓          |     —        |
| `make nuke`   |       ✓        |     ✓       |      —          |     —        |
| `make reset`  |       ✓        |     ✓       |      ✓          |     —        |

No target ever deletes `data/openalex_50k` or `data/openalex` — those
take 30–60 min of rate-limited API calls to refetch. If you really need
to re-fetch, `rm -rf data/openalex_50k` manually.

---

## Troubleshooting

- **`uv sync` fails on `sentence-transformers`** — you didn't pass
  `--extra embed`. Re-run `uv sync --extra embed`.
- **NDCG@10 is much lower than reported** — almost always paper_id
  drift from Option B (re-fetching OpenAlex). Use Option A
  (Dropbox snapshot) to guarantee `paper_id` alignment with
  `eval/ground-truth.jsonl`.
- **`load_graph.py` hangs after "AGE nodes: 36740"** — old build
  without GIN-index rebuild after `drop_label()`. Confirm `grep
  paper_props_gin ingest/load_graph.py` matches; if not, `git pull`.
- **`cold-warm` subcommand not found** — the bench binary's subcommand
  is `cold-warm` (hyphenated). The 21-cell matrix is produced by
  `scripts/coldwarm_all_21.py`, not by the binary directly.

---

## License & attribution

OpenAlex data is licensed under CC0. Code in this repository is for
academic coursework; no warranty.
