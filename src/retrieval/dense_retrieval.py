# Dense Retrieval

!pip -q install sentence-transformers faiss-cpu

import time
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

index = faiss.read_index(
    "hnsw_index.faiss"
)

print(
    f"Chunks: {len(chunks_df):,}"
)

print(
    f"Indexed Vectors: {index.ntotal:,}"
)

print("\n" + "="*80)
print("Loading Dense Retrieval Model")
print("="*80)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(
    f"Device: {device}"
)

model = SentenceTransformer(
    "intfloat/multilingual-e5-base",
    device=device
)

print(
    "Model Loaded Successfully"
)

print("\n" + "="*80)
print("Creating Evaluation Queries")
print("="*80)

evaluation_df = (
    chunks_df
    .sample(
        1000,
        random_state=42
    )
    .reset_index(drop=True)
)

queries = (
    evaluation_df["document_title"]
    .fillna("")
    .astype(str)
    .tolist()
)

print(
    f"Queries: {len(queries):,}"
)

print("\n" + "="*80)
print("Dense Retrieval Benchmark")
print("="*80)

latencies = []

for query in tqdm(
    queries
):

    start = time.time()

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    scores, ids = index.search(
        query_embedding.astype(
            np.float32
        ),
        10
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
print("Example Retrieval")
print("="*80)

sample_query = queries[0]

query_embedding = model.encode(
    [sample_query],
    normalize_embeddings=True,
    convert_to_numpy=True
)

scores, ids = index.search(
    query_embedding.astype(
        np.float32
    ),
    5
)

print(
    f"Query: {sample_query}"
)

results = chunks_df.iloc[
    ids[0]
][[
    "document_title",
    "language",
    "chunk_text"
]]

display(
    results
)

print("\n" + "="*80)
print("Dense Retrieval Benchmark Summary")
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
        10,
        round(avg_latency,3),
        round(p95_latency,3),
        round(qps,2)
    ]
})

print(
    benchmark_df
)

benchmark_df.to_csv(
    "dense_retrieval_benchmark.csv",
    index=False
)

print("\n" + "="*80)
print("Saving Artifacts")
print("="*80)

print(
    "dense_retrieval_benchmark.csv"
)

print("\n" + "="*80)
print("Step 8 Completed Successfully")
print("="*80)

def dense_retrieve(
    query,
    top_k=10
):

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    scores, ids = index.search(
        query_embedding.astype(
            np.float32
        ),
        top_k
    )

    return chunks_df.iloc[
        ids[0]
    ][[
        "document_title",
        "language",
        "chunk_text"
    ]]

# Example:
dense_retrieve(
    "Who invented the telephone?"
    )
