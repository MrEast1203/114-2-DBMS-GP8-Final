# ResearchDB · convenience targets
#
# Phase 1 Week 1 path (gate):
#   make up          # start postgres + AGE
#   make migrate     # apply migrations 0001 + 0002
#   make synth       # generate 5K paper synthetic graph
#   make load        # load JSONL into PG + project AGE
#   make smoke       # run AGE smoke test → reports/age_smoke_*.json
#
# Phase 1 main body (later):
#   make ingest      # fetch real OpenAlex subgraph (requires OPENALEX_MAILTO)
#   make bench       # full 7 × 2 benchmark cells (after orchestrator lands)

.PHONY: help up down logs migrate synth load smoke ingest test build clean reset fmt

DB_CMD = docker exec researchdb-db psql -U researchdb -d researchdb -v ON_ERROR_STOP=1

help:
	@awk 'BEGIN{FS=":.*##"} /^[a-zA-Z0-9_-]+:.*##/{printf "  %-12s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

up:                  ## Start postgres + AGE container
	docker compose up -d db
	@echo "waiting for healthy..."
	@until [ "$$(docker inspect -f '{{.State.Health.Status}}' researchdb-db 2>/dev/null)" = "healthy" ]; do sleep 1; done
	@echo "ready"

down:                ## Stop containers + remove orphans (keeps volume)
	docker compose down --remove-orphans

nuke:                ## Stop + delete postgres volume + remove orphans (DESTRUCTIVE)
	docker compose down -v --remove-orphans

logs:                ## Tail db logs
	docker compose logs -f db

migrate: build       ## Apply embedded sqlx migrations (idempotent)
	./target/release/researchdb-bench migrate

synth:               ## Generate 5K synthetic paper graph
	uv run python ingest/synthesize_graph.py --n 5000 --seed 42 --out data/synth

load:                ## Load data/synth into postgres + project AGE
	uv run python ingest/load_graph.py --src data/synth --reset

embed:               ## Populate chunk_embeddings with random unit vectors (placeholder)
	uv run python ingest/embed_chunks.py --strategy random

ingest:              ## Fetch real OpenAlex BFS subgraph (needs OPENALEX_MAILTO)
	uv run python ingest/openalex_fetch.py --n 5000 --out data/openalex

build:               ## Build Rust workspace (release)
	cargo build --release

test:                ## Run Rust unit tests
	cargo test --release

smoke: build         ## Run AGE smoke test → reports/age_smoke_$(shell date +%Y%m%d_%H%M%S).json
	@OUT=reports/age_smoke_$$(date +%Y%m%d_%H%M%S).json; \
	RUST_LOG=info ./target/release/researchdb-bench age-smoke \
	   --samples 100 --timeout-sec 60 --out $$OUT; \
	echo "→ $$OUT"

demo: build          ## 14-cell demo · 7 queries × 2 plans · table + JSON
	uv run python scripts/demo_14cells.py

eval: build          ## §E4 NDCG / Jaccard / RBO evaluation against ground truth
	uv run python eval/evaluate.py --samples 10

ef-sweep: build      ## §E2 ef_search latency-recall sweep
	./target/release/researchdb-bench ef-sweep --n-queries 20 --samples 10

storage: build       ## §E5 storage overhead report
	./target/release/researchdb-bench storage

cold-warm: build     ## §E1 cold/warm latency for one query (default Q1 v0)
	./target/release/researchdb-bench cold-warm --query 1 --plan v0 --samples 30 --warmup 5

health: build        ## DB connectivity + AGE extension check
	./target/release/researchdb-bench health

fmt:                 ## Format Rust + Python
	cargo fmt
	uv run ruff format ingest/

clean:               ## Clear reports/*.json only (keeps fetched data, container, volume)
	rm -f reports/*.json

reset: nuke          ## Drop container + volume + reports/*.json (fetched data preserved)
	rm -f reports/*.json
	@echo "  → reset complete. data/ preserved (refetching OpenAlex is 30-60 min)."
	@echo "  → next: make up && make migrate, then load_graph.py from §10.2 step 5."
