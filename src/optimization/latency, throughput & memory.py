# Latency & Systems Optimization

import time
import psutil
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

embeddings = np.load(
    "multilingual_e5_embeddings.npy"
).astype(np.float32)

index = faiss.read_index(
    "hnsw_index.faiss"
)

print(
    f"Corpus Size: {len(chunks_df):,}"
)

print(
    f"Vectors: {index.ntotal:,}"
)

print("\n"+"="*80)
print("Loading Embedding Model")
print("="*80)

model = SentenceTransformer(
    "intfloat/multilingual-e5-base",
    device="cuda"
)

print("Model Loaded")

print("\n"+"="*80)
print("Creating Benchmark Queries")
print("="*80)

queries = (
    chunks_df["document_title"]
    .dropna()
    .sample(
        1000,
        random_state=42
    )
    .tolist()
)

print(
    f"Queries: {len(queries):,}"
)

print("\n"+"="*80)
print("Embedding Benchmark")
print("="*80)

baseline_times = []

for query in tqdm(
    queries,
    leave=False
):

    start = time.time()

    _ = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    baseline_times.append(
        (time.time()-start)*1000
    )

baseline_embedding_latency = np.mean(
    baseline_times
)

print(
    f"Baseline Embedding Latency: "
    f"{baseline_embedding_latency:.3f} ms"
)

print("\n"+"="*80)
print("Embedding Cache Benchmark")
print("="*80)

embedding_cache = {}

cached_times = []

for query in tqdm(
    queries,
    leave=False
):

    start = time.time()

    if query not in embedding_cache:

        embedding_cache[query] = model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )

    _ = embedding_cache[query]

    cached_times.append(
        (time.time()-start)*1000
    )

cached_embedding_latency = np.mean(
    cached_times
)

print(
    f"Cached Embedding Latency: "
    f"{cached_embedding_latency:.3f} ms"
)

print("\n"+"="*80)
print("Mixed Precision Benchmark")
print("="*80)

fp16_times = []

for query in tqdm(
    queries,
    leave=False
):

    start = time.time()

    with torch.autocast(
        device_type="cuda",
        dtype=torch.float16
    ):

        _ = model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )

    fp16_times.append(
        (time.time()-start)*1000
    )

mixed_precision_latency = np.mean(
    fp16_times
)

print(
    f"Mixed Precision Latency: "
    f"{mixed_precision_latency:.3f} ms"
)

print("\n"+"="*80)
print("HNSW Retrieval Benchmark")
print("="*80)

retrieval_times = []

for query in tqdm(
    queries,
    leave=False
):

    q_emb = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    start = time.time()

    index.search(
        q_emb.astype(np.float32),
        10
    )

    retrieval_times.append(
        (time.time()-start)*1000
    )

baseline_retrieval_latency = np.mean(
    retrieval_times
)

print(
    f"Baseline Retrieval Latency: "
    f"{baseline_retrieval_latency:.3f} ms"
)

print("\n"+"="*80)
print("HNSW Tuning Benchmark")
print("="*80)

index.hnsw.efSearch = 128

optimized_times = []

for query in tqdm(
    queries,
    leave=False
):

    q_emb = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    start = time.time()

    index.search(
        q_emb.astype(np.float32),
        10
    )

    optimized_times.append(
        (time.time()-start)*1000
    )

optimized_latency = np.mean(
    optimized_times
)

print(
    f"Optimized Retrieval Latency: "
    f"{optimized_latency:.3f} ms"
)

memory_usage = (
    psutil.Process()
    .memory_info()
    .rss
    /1024**2
)

results = pd.DataFrame({

    "Metric":[

        "Baseline Embedding Latency (ms)",

        "Cached Embedding Latency (ms)",

        "Mixed Precision Latency (ms)",

        "Baseline Retrieval Latency (ms)",

        "Optimized Retrieval Latency (ms)",

        "Embedding Throughput (QPS)",

        "Retrieval Throughput (QPS)",

        "Memory Usage (MB)"
    ],

    "Value":[

        round(
            baseline_embedding_latency,
            3
        ),

        round(
            cached_embedding_latency,
            3
        ),

        round(
            mixed_precision_latency,
            3
        ),

        round(
            baseline_retrieval_latency,
            3
        ),

        round(
            optimized_latency,
            3
        ),

        round(
            1000 /
            baseline_embedding_latency,
            2
        ),

        round(
            1000 /
            optimized_latency,
            2
        ),

        round(
            memory_usage,
            2
        )
    ]
})

print("\n"+"="*80)
print("Optimization Summary")
print("="*80)

display(results)

results.to_csv(
    "systems_optimization_results.csv",
    index=False
)

print("\n"+"="*80)
print("Saved Artifacts")
print("="*80)

print(
    "systems_optimization_results.csv"
)

print("\n"+"="*80)
print("Step 12 Completed Successfully")
print("="*80)
