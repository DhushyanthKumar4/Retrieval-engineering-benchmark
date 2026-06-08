# Cross-Encoder Reranking

!pip -q install sentence-transformers

import time
import pickle
import numpy as np
import pandas as pd
import faiss

from tqdm.auto import tqdm
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder

print("="*80)
print("Loading Retrieval Assets")
print("="*80)

chunks_df = pd.read_parquet(
    "retrieval_corpus_50k.parquet"
)

embeddings = np.load(
    "multilingual_e5_embeddings.npy"
).astype(np.float32)

index = faiss.read_index(
    "hnsw_index.faiss"
)

with open(
    "bm25_index.pkl",
    "rb"
) as f:
    bm25 = pickle.load(f)

print(
    f"Corpus Size: {len(chunks_df):,}"
)

print(
    f"Indexed Vectors: {index.ntotal:,}"
)

print("\n"+"="*80)
print("Loading Dense Retriever")
print("="*80)

dense_model = SentenceTransformer(
    "intfloat/multilingual-e5-base",
    device="cuda"
)

print(
    "Dense Retriever Loaded"
)

print("\n"+"="*80)
print("Loading Cross Encoder")
print("="*80)

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
    device="cuda",
    max_length=512
)

print(
    "Cross Encoder Loaded"
)

TOP_K_RETRIEVAL = 50
TOP_K_FINAL = 10
RRF_K = 60

print("\n"+"="*80)
print("Reranker Configuration")
print("="*80)

print(f"Initial Retrieval: {TOP_K_RETRIEVAL}")
print(f"Final Results: {TOP_K_FINAL}")

def hybrid_retrieve(
    query,
    top_k=TOP_K_RETRIEVAL
):

    bm25_scores = bm25.get_scores(
        query.lower().split()
    )

    bm25_ids = np.argsort(
        bm25_scores
    )[::-1][:top_k]

    q_emb = dense_model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    _, dense_ids = index.search(
        q_emb.astype(np.float32),
        top_k
    )

    dense_ids = dense_ids[0]

    rrf_scores = {}

    for rank, doc_id in enumerate(
        bm25_ids
    ):

        rrf_scores[doc_id] = (
            rrf_scores.get(doc_id, 0)
            + 1/(RRF_K + rank + 1)
        )

    for rank, doc_id in enumerate(
        dense_ids
    ):

        rrf_scores[doc_id] = (
            rrf_scores.get(doc_id, 0)
            + 1/(RRF_K + rank + 1)
        )

    ranked_docs = sorted(
        rrf_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return [
        doc_id
        for doc_id, _
        in ranked_docs[:top_k]
    ]

def rerank(
    query,
    top_k=TOP_K_FINAL
):

    candidate_ids = hybrid_retrieve(
        query
    )

    candidate_texts = (
        chunks_df.iloc[
            candidate_ids
        ]["chunk_text"]
        .tolist()
    )

    pairs = [
        (query, text)
        for text in candidate_texts
    ]

    scores = reranker.predict(
        pairs,
        batch_size=16
    )

    ranking = np.argsort(
        scores
    )[::-1]

    final_ids = [
        candidate_ids[i]
        for i in ranking[:top_k]
    ]

    return (
        chunks_df.iloc[
            final_ids
        ][[
            "document_title",
            "language",
            "chunk_text"
        ]],
        scores
    )

print("\n"+"="*80)
print("Creating Evaluation Queries")
print("="*80)

evaluation_queries = (
    chunks_df["document_title"]
    .dropna()
    .sample(
        500,
        random_state=42
    )
    .tolist()
)

print(
    f"Queries: {len(evaluation_queries):,}"
)

latencies = []

print("\n"+"="*80)
print("Running Reranker Benchmark")
print("="*80)

for query in tqdm(
    evaluation_queries
):

    start = time.time()

    rerank(query)

    latencies.append(
        (time.time()-start)*1000
    )

avg_latency = np.mean(
    latencies
)

p95_latency = np.percentile(
    latencies,
    95
)

qps = (
    1000 / avg_latency
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

print("\n"+"="*80)
print("Reranker Example")
print("="*80)

results, _ = rerank(
    "Who invented the telephone?"
)

display(
    results.head(10)
)

benchmark = pd.DataFrame({

    "Metric":[
        "Corpus Size",
        "Initial Retrieval K",
        "Final Top K",
        "Average Latency (ms)",
        "P95 Latency (ms)",
        "Queries Per Second"
    ],

    "Value":[
        len(chunks_df),
        TOP_K_RETRIEVAL,
        TOP_K_FINAL,
        round(avg_latency,3),
        round(p95_latency,3),
        round(qps,2)
    ]
})

print("\n"+"="*80)
print("Reranker Benchmark Summary")
print("="*80)

display(
    benchmark
)

benchmark.to_csv(
    "reranker_benchmark.csv",
    index=False
)

print("\n"+"="*80)
print("Saving Artifacts")
print("="*80)

print(
    "reranker_benchmark.csv"
)

print("\n"+"="*80)
print("Step 10 Completed Successfully")
print("="*80)
