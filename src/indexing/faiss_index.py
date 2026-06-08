# Vector Indexing System

!pip -q install faiss-gpu-cu12

import os
import time
import faiss
import numpy as np
import pandas as pd

print("="*80)
print("Loading Embeddings")
print("="*80)

embeddings = np.load(
    "multilingual_e5_embeddings.npy"
).astype(np.float32)

print(
    f"Embedding Shape: {embeddings.shape}"
)

n_vectors = embeddings.shape[0]
dimension = embeddings.shape[1]

print(
    f"Vectors: {n_vectors:,}"
)
print(
    f"Dimension: {dimension}"
)

print("\n" + "="*80)
print("Building Flat Index")
print("="*80)

start = time.time()

flat_index = faiss.IndexFlatIP(
    dimension
)

flat_index.add(
    embeddings
)

flat_build_time = (
    time.time() - start
)

faiss.write_index(
    flat_index,
    "flat_index.faiss"
)

flat_size_mb = (
    os.path.getsize(
        "flat_index.faiss"
    ) / (1024**2)
)

print(
    f"Build Time: {flat_build_time:.2f} sec"
)
print(
    f"Index Size: {flat_size_mb:.2f} MB"
)

print("\n" + "="*80)
print("Building IVF Index")
print("="*80)

nlist = 256

quantizer = faiss.IndexFlatIP(
    dimension
)

ivf_index = faiss.IndexIVFFlat(
    quantizer,
    dimension,
    nlist,
    faiss.METRIC_INNER_PRODUCT
)

start = time.time()

ivf_index.train(
    embeddings
)

ivf_index.add(
    embeddings
)

ivf_build_time = (
    time.time() - start
)

ivf_index.nprobe = 16

faiss.write_index(
    ivf_index,
    "ivf_index.faiss"
)

ivf_size_mb = (
    os.path.getsize(
        "ivf_index.faiss"
    ) / (1024**2)
)

print(
    f"Build Time: {ivf_build_time:.2f} sec"
)

print(
    f"Index Size: {ivf_size_mb:.2f} MB"
)

print("\n" + "="*80)
print("Building HNSW Index")
print("="*80)

start = time.time()

hnsw_index = faiss.IndexHNSWFlat(
    dimension,
    32
)

hnsw_index.hnsw.efConstruction = 200

hnsw_index.add(
    embeddings
)

hnsw_build_time = (
    time.time() - start
)

faiss.write_index(
    hnsw_index,
    "hnsw_index.faiss"
)

hnsw_size_mb = (
    os.path.getsize(
        "hnsw_index.faiss"
    ) / (1024**2)
)

print(
    f"Build Time: {hnsw_build_time:.2f} sec"
)

print(
    f"Index Size: {hnsw_size_mb:.2f} MB"
)

print("\n" + "="*80)
print("Latency Benchmark")
print("="*80)

np.random.seed(42)

query_vectors = embeddings[
    np.random.choice(
        n_vectors,
        100,
        replace=False
    )
]

def benchmark_latency(
    index,
    queries,
    top_k=10
):

    start = time.time()

    index.search(
        queries,
        top_k
    )

    total_time = (
        time.time() - start
    )

    avg_ms = (
        total_time
        /
        len(queries)
    ) * 1000

    return avg_ms

flat_latency = benchmark_latency(
    flat_index,
    query_vectors
)

ivf_latency = benchmark_latency(
    ivf_index,
    query_vectors
)

hnsw_latency = benchmark_latency(
    hnsw_index,
    query_vectors
)

print(
    f"Flat Latency : {flat_latency:.3f} ms"
)

print(
    f"IVF Latency  : {ivf_latency:.3f} ms"
)

print(
    f"HNSW Latency : {hnsw_latency:.3f} ms"
)

print("\n" + "="*80)
print("Benchmark Summary")
print("="*80)

benchmark_df = pd.DataFrame({

    "Index":[
        "Flat",
        "IVF",
        "HNSW"
    ],

    "Build_Time_sec":[
        round(flat_build_time,2),
        round(ivf_build_time,2),
        round(hnsw_build_time,2)
    ],

    "Latency_ms":[
        round(flat_latency,3),
        round(ivf_latency,3),
        round(hnsw_latency,3)
    ],

    "Index_Size_MB":[
        round(flat_size_mb,2),
        round(ivf_size_mb,2),
        round(hnsw_size_mb,2)
    ]
})

print(
    benchmark_df
)

benchmark_df.to_csv(
    "faiss_index_benchmark.csv",
    index=False
)

print("\n" + "="*80)
print("Saved Artifacts")
print("="*80)

print("flat_index.faiss")
print("ivf_index.faiss")
print("hnsw_index.faiss")
print("faiss_index_benchmark.csv")

print("\n" + "="*80)
print("Primary Production Index")
print("="*80)

print(
    "HNSW selected as production index."
)

print("\nStep 6 Completed Successfully")
