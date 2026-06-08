# Hybrid Retrieval

!pip -q install rank-bm25 sentence-transformers faiss-cpu

import time
import pickle
import faiss
import torch
import numpy as np
import pandas as pd

from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer

print("="*80)
print("Loading Retrieval Assets")
print("="*80)

chunks_df = pd.read_parquet(
    "retrieval_corpus_50k.parquet"
)

with open(
    "bm25_index.pkl",
    "rb"
) as f:
    bm25 = pickle.load(f)

hnsw_index = faiss.read_index(
    "hnsw_index.faiss"
)

print(
    f"Corpus Size: {len(chunks_df):,}"
)

print(
    f"Indexed Vectors: {hnsw_index.ntotal:,}"
)

print("\n" + "="*80)
print("Loading Dense Retrieval Model")
print("="*80)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

model = SentenceTransformer(
    "intfloat/multilingual-e5-base",
    device=device
)

print(
    f"Device: {device}"
)

print("\n" + "="*80)
print("Hybrid Retrieval Configuration")
print("="*80)

TOP_K = 10
RRF_K = 60

print(
    f"TOP_K: {TOP_K}"
)

print(
    f"RRF Constant: {RRF_K}"
)

def reciprocal_rank_fusion(
    bm25_ids,
    dense_ids,
    rrf_k=60
):

    scores = {}

    for rank, doc_id in enumerate(
        bm25_ids
    ):
        scores[doc_id] = (
            scores.get(doc_id, 0)
            +
            1/(rrf_k + rank + 1)
        )

    for rank, doc_id in enumerate(
        dense_ids
    ):
        scores[doc_id] = (
            scores.get(doc_id, 0)
            +
            1/(rrf_k + rank + 1)
        )

    ranked = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return [
        doc_id
        for doc_id, _
        in ranked[:TOP_K]
    ]

def weighted_fusion(
    bm25_scores,
    dense_scores,
    bm25_weight=0.4,
    dense_weight=0.6
):

    fused = {}

    bm25_scores = (
        bm25_scores
        -
        bm25_scores.min()
    ) / (
        bm25_scores.max()
        -
        bm25_scores.min()
        +
        1e-9
    )

    dense_scores = (
        dense_scores
        -
        dense_scores.min()
    ) / (
        dense_scores.max()
        -
        dense_scores.min()
        +
        1e-9
    )

    for idx in range(
        len(bm25_scores)
    ):

        fused[idx] = (

            bm25_weight *
            bm25_scores[idx]

            +

            dense_weight *
            dense_scores[idx]

        )

    ranked = sorted(
        fused.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return [
        doc_id
        for doc_id, _
        in ranked[:TOP_K]
    ]

print("\n" + "="*80)
print("Creating Evaluation Queries")
print("="*80)

evaluation_df = (
    chunks_df
    .sample(
        1000,
        random_state=42
    )
)

queries = (
    evaluation_df[
        "document_title"
    ]
    .fillna("")
    .astype(str)
    .tolist()
)

print(
    f"Queries: {len(queries):,}"
)

print("\n" + "="*80)
print("Running Hybrid Benchmark")
print("="*80)

latencies = []

for query in tqdm(
    queries
):

    start = time.time()

    bm25_scores = bm25.get_scores(
        query.lower().split()
    )

    bm25_ids = np.argsort(
        bm25_scores
    )[::-1][:100]

    query_emb = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    dense_scores, dense_ids = (
        hnsw_index.search(
            query_emb.astype(
                np.float32
            ),
            100
        )
    )

    hybrid_results = reciprocal_rank_fusion(
        bm25_ids,
        dense_ids[0]
    )

    latency = (
        time.time()
        -
        start
    ) * 1000

    latencies.append(
        latency
    )

avg_latency = np.mean(
    latencies
)

p95_latency = np.percentile(
    latencies,
    95
)

qps = (
    1000
    /
    avg_latency
)

print(
    f"Average Latency: {avg_latency:.3f} ms"
)

print(
    f"P95 Latency: {p95_latency:.3f} ms"
)

print(
    f"QPS: {qps:.2f}"
)

print("\n" + "="*80)
print("Hybrid Retrieval Example")
print("="*80)

sample_query = queries[0]

bm25_scores = bm25.get_scores(
    sample_query.lower().split()
)

bm25_ids = np.argsort(
    bm25_scores
)[::-1][:100]

query_emb = model.encode(
    [sample_query],
    normalize_embeddings=True,
    convert_to_numpy=True
)

dense_scores, dense_ids = (
    hnsw_index.search(
        query_emb.astype(
            np.float32
        ),
        100
    )
)

hybrid_ids = reciprocal_rank_fusion(
    bm25_ids,
    dense_ids[0]
)

print(
    f"Query: {sample_query}"
)

display(

    chunks_df.iloc[
        hybrid_ids[:5]
    ][[
        "document_title",
        "language",
        "chunk_text"
    ]]

)

print("\n" + "="*80)
print("Hybrid Benchmark Summary")
print("="*80)

benchmark_df = pd.DataFrame({

    "Metric":[

        "Corpus Size",
        "Top K",
        "Average Latency (ms)",
        "P95 Latency (ms)",
        "Queries Per Second"

    ],

    "Value":[

        len(chunks_df),
        TOP_K,
        round(avg_latency,3),
        round(p95_latency,3),
        round(qps,2)

    ]
})

print(
    benchmark_df
)

benchmark_df.to_csv(
    "hybrid_retrieval_benchmark.csv",
    index=False
)

print("\n" + "="*80)
print("Saving Artifacts")
print("="*80)

print(
    "hybrid_retrieval_benchmark.csv"
)

print("\n" + "="*80)
print("Step 9 Completed Successfully")
print("="*80)

def hybrid_retrieve(
    query,
    top_k=10
):

    bm25_scores = bm25.get_scores(
        query.lower().split()
    )

    bm25_ids = np.argsort(
        bm25_scores
    )[::-1][:100]

    query_emb = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    dense_scores, dense_ids = (
        hnsw_index.search(
            query_emb.astype(
                np.float32
            ),
            100
        )
    )

    hybrid_ids = reciprocal_rank_fusion(
        bm25_ids,
        dense_ids[0]
    )

    return chunks_df.iloc[
        hybrid_ids[:top_k]
    ][[
        "document_title",
        "language",
        "chunk_text"
    ]]

# Example:
hybrid_retrieve("Who invented the telephone?")
