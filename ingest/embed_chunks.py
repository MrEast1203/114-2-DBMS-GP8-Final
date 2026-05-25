#!/usr/bin/env python3
"""Populate chunk_embeddings with 384-dim unit vectors.

Three strategies:
  * `random`     — deterministic random vectors. Sufficient for latency
                   benchmarks (HNSW behavior is structure-driven, not
                   value-driven); insufficient for relevance metrics.
  * `bag-of-words` — hash-the-words mock that gives chunks containing
                   shared words similar vectors. Zero-dep semantic
                   signal, good for sanity checks.
  * `minilm`     — `sentence-transformers/all-MiniLM-L6-v2`. Real 384-dim
                   sentence embedding model. CPU-friendly (~50ms / chunk
                   on Apple Silicon). Required for meaningful NDCG@10
                   numbers in the Phase 1 writeup.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
from typing import Iterable

import psycopg
import random
import math

VECTOR_DIM = 384


# --- vector strategies -----------------------------------------------------

def rand_unit_vector(seed: int) -> list[float]:
    rng = random.Random(seed)
    v = [rng.gauss(0.0, 1.0) for _ in range(VECTOR_DIM)]
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def make_minilm_encoder():
    """Lazy-load all-MiniLM-L6-v2 so the import only happens when needed.
    Returns a callable: (texts: list[str]) -> list[list[float]]."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    def encode(texts: list[str]) -> list[list[float]]:
        embs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return [list(map(float, v)) for v in embs]
    return encode


def bag_of_words_vector(text: str) -> list[float]:
    """Hash each word to one of VECTOR_DIM buckets and accumulate; normalize.

    Two chunks containing the same word will share that bucket, so cosine
    similarity reflects word overlap. Crude but real semantic signal,
    zero dependencies."""
    v = [0.0] * VECTOR_DIM
    for word in text.lower().split():
        h = int.from_bytes(hashlib.md5(word.encode("utf-8")).digest()[:4], "big")
        v[h % VECTOR_DIM] += 1.0
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def vector_literal(v: list[float]) -> str:
    return "[" + ",".join(f"{x:.6f}" for x in v) + "]"


# --- driver ----------------------------------------------------------------

def populate(conn: psycopg.Connection, strategy: str, batch: int = 256,
             force: bool = False) -> int:
    """Embed all chunks whose chunk_embeddings row is missing (or all when
    --force). The minilm strategy batches at 256 to amortize tokenizer
    overhead; random / bag-of-words at 1000 since they have no GPU dance."""
    query = "SELECT c.id, c.text FROM chunks c"
    if not force:
        query += (" LEFT JOIN chunk_embeddings ce ON ce.chunk_id = c.id"
                  " WHERE ce.chunk_id IS NULL")
    with conn.cursor() as cur:
        cur.execute(query)
        rows = cur.fetchall()

    print(f"  to embed: {len(rows)} chunks", file=sys.stderr)
    if not rows:
        return 0

    encoder = None
    if strategy == "minilm":
        encoder = make_minilm_encoder()
    elif strategy == "random":
        batch = 1000
    elif strategy == "bag-of-words":
        batch = 1000
    else:
        raise ValueError(f"unknown strategy: {strategy}")

    inserted = 0
    with conn.cursor() as cur:
        for i in range(0, len(rows), batch):
            chunk = rows[i:i + batch]
            cids  = [c[0] for c in chunk]
            texts = [(c[1] or "") for c in chunk]

            if strategy == "minilm":
                vectors = encoder(texts)
            elif strategy == "random":
                vectors = [rand_unit_vector(seed=cid) for cid in cids]
            else:  # bag-of-words
                vectors = [bag_of_words_vector(t) for t in texts]

            args = [
                (cid, vector_literal(v), f"phase1.{strategy}")
                for cid, v in zip(cids, vectors)
            ]
            cur.executemany(
                "INSERT INTO chunk_embeddings (chunk_id, embedding, model) "
                "VALUES (%s, %s::vector, %s) "
                "ON CONFLICT (chunk_id) DO UPDATE "
                "  SET embedding = EXCLUDED.embedding, model = EXCLUDED.model",
                args,
            )
            inserted += len(chunk)
            print(f"  {inserted}/{len(rows)}", file=sys.stderr)
    conn.commit()
    return inserted


def get_dsn() -> str:
    return os.environ.get(
        "DATABASE_URL",
        "postgres://researchdb:researchdb@localhost:5432/researchdb",
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--strategy", choices=["random", "bag-of-words", "minilm"],
                   default="minilm",
                   help="Vector synthesis strategy. minilm = real sentence-transformers; "
                        "random = deterministic per chunk_id; bag-of-words = word-hash similarity.")
    p.add_argument("--dsn", default=None)
    p.add_argument("--force", action="store_true",
                   help="Re-embed all chunks even if chunk_embeddings already populated.")
    args = p.parse_args()

    dsn = args.dsn or get_dsn()
    with psycopg.connect(dsn, autocommit=False) as conn:
        n = populate(conn, args.strategy, force=args.force)
    print(f"embedded {n} chunks with strategy={args.strategy}")


if __name__ == "__main__":
    main()
