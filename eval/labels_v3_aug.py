#!/usr/bin/env python3
"""Augmented GT labels for the v3 chained-pushdown evaluation.

The original eval/ground-truth.jsonl was built by pooling MiniLM top-30
∪ BM25 top-30 per query and hand-labelling each candidate. v3's chained
push-down (BM25 top-N → pgvector) surfaces papers that fall *outside*
that original pool — pgvector now sees only BM25-matched candidates,
so its top-K can include papers that the pool-build pgvector pass
never retrieved. Those papers are unlabelled and NDCG treats them as 0,
biasing v3's score downward unfairly.

This file holds **additional labels** for every (qid, paper_id) pair
that appears in any plan's top-10 of `reports/eval_v3.json` but is
missing from `eval/ground-truth.jsonl`. 83 pairs total — labelled by
the same single annotator (with LLM-assisted reading of title +
abstract) per the original rubric in `eval/apply_labels.py`:

  1 = relevant   — title/abstract topic clearly overlaps with query
                   intent. For Q4/Q5/Q7 the graph filter is already
                   applied, so we judge only topical relevance.
  0 = not relevant — off-topic w.r.t. the query's stated intent.

Reviewer's note: these 83 additions are LLM-judged in the same way the
original 1 368 labels were (single annotator, no inter-annotator
agreement). They are *not* domain-expert labels — same caveat as the
original pool.
"""

# (qid, paper_id) → label (0 or 1)
EXTRA_LABELS: dict[tuple[str, int], int] = {
    # --- Q4-2  HOG 視覺特徵 + cites HOG (graph filter applied) ---
    # SUN database / object-detection survey — both clearly HOG-era visual feature work.
    ("Q4-2",   278): 1,
    ("Q4-2",   303): 1,

    # --- Q4-3  MapReduce + cites MapReduce ---
    # Percolator — Google's incremental indexing on top of MapReduce-class infra.
    ("Q4-3",    98): 1,

    # --- Q4-5  LSTM 序列建模 + cites LSTM ---
    ("Q4-5",    59): 1,   # Generating Sequences With RNNs (LSTM-based generative model)
    ("Q4-5",   103): 1,   # Deep Recurrent Models w/ Fast-Forward for NMT (LSTM/GRU based)
    ("Q4-5",   124): 1,   # LSTM-Networks for Machine Reading (LSTM extension)
    ("Q4-5",   909): 1,   # On the difficulty of training RNNs (LSTM motivation)

    # --- Q5-5  distributed computing parallel + cites MapReduce ---
    ("Q5-5",  1503): 1,   # Dryad — distributed data-parallel execution engine
    ("Q5-5",  4926): 1,   # Distributed GraphLab — data-parallel framework
    ("Q5-5",  6404): 1,   # MadLINQ — distributed matrix computation
    ("Q5-5", 18683): 1,   # Ad-hoc data processing in the cloud (distributed)
    ("Q5-5", 23574): 1,   # Cloud Computing vs Grid (distributed/parallel)
    ("Q5-5", 27007): 1,   # Landscape of Parallel Computing Research (Berkeley)
    ("Q5-5", 33011): 1,   # SnowFlock — VM fork for distributed deployment

    # --- Q6-1  Attention 相關 + machine translation ---
    # MT papers — most clearly relevant; example-based MT is an off-paradigm outlier.
    ("Q6-1",   986): 1,   # Word Alignment Quality for SMT (alignment = attention precursor)
    ("Q6-1",  1253): 1,   # Neural LM for MT
    ("Q6-1", 13657): 1,   # Improved Alignment Models for SMT
    ("Q6-1", 13694): 1,   # Statistical MT textbook (alignment chapter)
    ("Q6-1", 14515): 1,   # NMT deployment study
    ("Q6-1", 14516): 1,   # Attention-based encoder-decoder NMT
    ("Q6-1", 14517): 1,   # Bidirectional attention-based NMT
    ("Q6-1", 15076): 0,   # Example-based MT — different paradigm, not attention-adjacent

    # --- Q6-2  ResNet 相關 + batch normalization ---
    ("Q6-2",   896): 0,   # Divisive normalization for image rep (image processing, not BN)
    ("Q6-2",   946): 0,   # 1990s batch-learning paper for MLP (not modern BN)
    ("Q6-2",  1194): 1,   # Weight Normalization — explicitly inspired by BN, deep nets
    ("Q6-2",  1204): 0,   # Vocab Manipulation for NMT — about MT, not ResNet/BN
    ("Q6-2",  1305): 0,   # Online vs batch learning (classical, not BN)
    ("Q6-2",  1314): 1,   # ELUs — deep network training, discusses BN
    ("Q6-2",  3734): 0,   # 1990s NN learning rule comparison
    ("Q6-2",  6973): 0,   # Hierarchical Matching Pursuit — image features, not ResNet/BN

    # --- Q6-3  Spanner 相關 + consensus paxos raft ---
    # Paxos / Raft / consensus papers — all clearly on-topic.
    ("Q6-3",    99): 1,   # Walter — geo-replicated transactional storage
    ("Q6-3",   128): 1,   # Part-time Parliament — Paxos original
    ("Q6-3",   133): 1,   # PNUTS — Yahoo distributed DB
    ("Q6-3",   147): 1,   # Consensus on Transaction Commit (Paxos Commit)
    ("Q6-3",  1382): 1,   # Paxos made live
    ("Q6-3",  2972): 1,   # Etna — fault-tolerant DHT w/ replication
    ("Q6-3",  3044): 1,   # Cheap Paxos
    ("Q6-3",  3054): 1,   # Stoppable Paxos
    ("Q6-3",  3060): 1,   # Correctness of Paxos with Replica-Set Views
    ("Q6-3", 20015): 1,   # Generalized Consensus and Paxos
    ("Q6-3", 23789): 1,   # ABCD's of Paxos
    ("Q6-3", 23840): 1,   # Improving Fast Paxos
    ("Q6-3", 32251): 1,   # Raft — In search of understandable consensus

    # --- Q6-4  Meltdown 相關 + cache timing attack ---
    # Cache side-channel / timing attack papers — directly on topic.
    ("Q6-4",   132): 1,   # FLUSH+RELOAD — canonical cache side-channel attack
    ("Q6-4",   165): 1,   # KeyDrown — keystroke timing side-channel
    ("Q6-4",  2262): 1,   # Branch Prediction Analysis (Meltdown/Spectre primitive)
    ("Q6-4",  2278): 0,   # DTLS plaintext recovery — TLS attack, not cache timing
    ("Q6-4",  2836): 1,   # Improved Brumley-Boneh SSL timing attack
    ("Q6-4",  2875): 1,   # Timing-attack-resistant AES-GCM (cache-timing context)
    ("Q6-4",  2973): 1,   # Covert timing channels, caching, cryptography
    ("Q6-4",  3031): 1,   # Cache Attacks on Intel SGX
    ("Q6-4",  3230): 1,   # Partitioned Cache Architecture (defence)
    ("Q6-4", 29492): 1,   # Time-Driven Cache Attacks on Mobile Devices
    ("Q6-4", 29502): 1,   # Cache Timing Analysis of HC-256
    ("Q6-4", 29779): 1,   # Eliminating Cache and Timing Side Channels
    ("Q6-4", 34333): 1,   # Time-Driven Cache Attacks on Mobile (extended)

    # --- Q6-5  MapReduce 相關 + fault tolerance replication ---
    ("Q6-5",    92): 1,   # Google cluster architecture (FT, MapReduce-era)
    ("Q6-5",    98): 1,   # Percolator (MapReduce-class infra)
    ("Q6-5",  1561): 1,   # Eager DB replication protocols
    ("Q6-5",  1562): 1,   # Middleware-based data replication
    ("Q6-5",  1634): 1,   # Distributed file system replication / FT
    ("Q6-5",  1722): 1,   # Transparent FT for parallel apps
    ("Q6-5",  2546): 1,   # Fault-tolerant computing fundamentals
    ("Q6-5",  2849): 1,   # Distributed Systems textbook (FT + replication chapters)
    ("Q6-5",  9870): 0,   # Fault tolerant NEURAL nets — wrong topic
    ("Q6-5", 17387): 1,   # Fault tolerance, principles and practice
    ("Q6-5", 22902): 1,   # DepSpace — tuple space replication FT
    ("Q6-5", 32283): 1,   # Crash-tolerant systems hardening (FT/replication)

    # --- Q7-1  ResNet 相關 + convolutional neural network + cites ResNet ---
    ("Q7-1",     2): 0,   # Attention Is All You Need — proposes NOT using CNN
    ("Q7-1",    75): 1,   # Convolutional Seq2Seq — CNN-based architecture
    ("Q7-1",  1134): 0,   # Speech recognition (cites ResNet but topic is speech, not CNN)
    ("Q7-1",  4472): 1,   # CNN for medical imaging (clearly CNN-based)

    # --- Q7-2  HOG 相關 + object detection feature + cites HOG ---
    ("Q7-2",   264): 1,   # Deep NN for Object Detection — survey
    ("Q7-2",   269): 1,   # DeePM — deep part-based object detection
    ("Q7-2",   393): 1,   # DeepID-Net — deep CNN for object detection
    ("Q7-2",   672): 1,   # BING — gradient-based objectness (HOG-adjacent)
    ("Q7-2",  5419): 1,   # Color attributes for object detection
    ("Q7-2",  6010): 1,   # Efficient large-scale object detection

    # --- Q7-3  MapReduce 相關 + cluster scheduling + cites MapReduce ---
    ("Q7-3",    98): 1,   # Percolator
    ("Q7-3",  1523): 0,   # CloudTPS — transactions, not cluster scheduling
    ("Q7-3",  2282): 0,   # Bulk insertion into distributed table — not cluster scheduling
    ("Q7-3", 14344): 1,   # Borg — cluster scheduling at Google (canonical)
    ("Q7-3", 27485): 0,   # Data center traffic — observational, not scheduling
}


if __name__ == "__main__":
    # Sanity check
    print(f"Total extra labels: {len(EXTRA_LABELS)}")
    print(f"Positive: {sum(1 for v in EXTRA_LABELS.values() if v == 1)}")
    print(f"Negative: {sum(1 for v in EXTRA_LABELS.values() if v == 0)}")
