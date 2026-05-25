# Hybrid Search Orchestration on PostgreSQL

Course: **NTU 114-2 DBMS · Final Project, Group 8**
Author: Chan Ching-Kan (chingkan.chan1203@gmail.com)

A research-grade benchmark of three query-planning strategies for running
**semantic + lexical + graph** search inside a single PostgreSQL instance.
The corpus is 36 740 OpenAlex paper abstracts; the three engines are
`pgvector` (HNSW), `pg_search` (BM25, ParadeDB), and PostgreSQL's native
`WITH RECURSIVE` over the citation graph. A Rust orchestrator on top of
the database fuses the three engines with Reciprocal Rank Fusion and
implements increasingly sophisticated plans (`naive` → `v1` → `v2`) that
culminate in **graph filter push-down into the ranker SQL**.

The full write-up — motivation, schema, four plans, ground-truth
construction, results, and reproduction notes — lives in the local HTML
report (not pushed to this repo; kept by the author for archival).
This README is the public entry point.

---

## TL;DR results (50K papers, WSL2, warm cache)

| plan  | mean P50 | NDCG@10 | top-10 vs naive       |
| ----- | -------- | ------- | --------------------- |
| naive | 51.36 ms | 0.675   | — (baseline)          |
| v1    | 50.92 ms | 0.675   | Jaccard 1.000 (same)  |
| v2    | **35.75 ms** | **0.801** | Jaccard 0.601 (differs) |

- **v2 push-down** cuts mean latency 1.44× and lifts NDCG@10 by
  +0.126 by restricting the ranker to the graph-filtered candidate set
  rather than ranking the whole corpus and filtering afterward.
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
│       ├── plan.rs         # naive / v1 / v2 plan implementations
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
├── scripts/            # Helper scripts (e.g. coldwarm_all_21.py)
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

### The three plans

| capability                                | naive | v1 | v2 |
| ----------------------------------------- | ----- | -- | -- |
| RRF fusion across engines                 |  ✓   | ✓ | ✓ |
| reorder engines by selectivity            |       | ✓ | ✓ |
| push graph filter down into ranker SQL    |       |    | ✓ |

The cost-based reorder in v1 does *not* change latency or quality in
this pipeline (see TL;DR): every engine has to run to completion for
RRF, and the engines have no data dependency that reorder could
short-circuit. v1 is kept as the clean ablation that shows this; v2's
push-down is what actually moves the needle.

The BFS cost formula `branching^depth` is used by v1 only for
annotation — it's the empirical fit from `bench micro-bench-age` (see
`bench/src/microbench.rs`). v2's push-down decision is triggered by
query type (Q4 / Q5 / Q7), not by reading cost at runtime, so the
cost formula is not listed as a v2 capability.

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
| `eval_phase1_e4.json`             | 20 queries × 3 plans · P50 + NDCG + Jaccard + RBO |
| `coldwarm_full_21.json`           | 7 queries × 3 plans · cold-vs-warm P50 matrix  |
| `storage.json`                    | per-relation / per-index disk usage breakdown  |
| `bfs_shootout.json`               | AGE Cypher vs `WITH RECURSIVE`, 3 depth × 3 bucket |
| `coldwarm_q{1..7}_{naive,v1,v2}.json` | individual cells of the cold/warm matrix |

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

# 8 · Reproduce the headline tables
uv run python eval/evaluate.py \
    --samples 10 \
    --queries eval/queries.jsonl \
    --gt eval/ground-truth.jsonl
uv run python scripts/coldwarm_all_21.py
./target/release/researchdb-bench storage      --out reports/storage.json
./target/release/researchdb-bench bfs-shootout --samples 30
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
