# BM25 Baseline

!pip -q install rank-bm25

import time
import pickle
import numpy as np
import pandas as pd

from tqdm.auto import tqdm
from rank_bm25 import BM25Okapi

print("="*80)
print("Loading Retrieval Corpus")
print("="*80)

chunks_df = pd.read_parquet(
    "retrieval_corpus_50k.parquet"
)

print(
    f"Chunks Loaded: {len(chunks_df):,}"
)

print("\n" + "="*80)
print("Preparing BM25 Documents")
print("="*80)

documents = (
    chunks_df["chunk_text"]
    .fillna("")
    .astype(str)
    .tolist()
)

tokenized_corpus = [

    doc.lower().split()

    for doc in tqdm(
        documents,
        desc="Tokenizing"
    )

]

print(
    f"Corpus Size: {len(tokenized_corpus):,}"
)

print("\n" + "="*80)
print("Building BM25 Index")
print("="*80)

start_time = time.time()

bm25 = BM25Okapi(
    tokenized_corpus
)

build_time = (
    time.time()
    -
    start_time
)

print(
    f"Build Time: {build_time:.2f} sec"
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
print("Running Retrieval Benchmark")
print("="*80)

latencies = []

for query in tqdm(
    queries
):

    query_tokens = (
        query
        .lower()
        .split()
    )

    start = time.time()

    scores = bm25.get_scores(
        query_tokens
    )

    top_ids = np.argsort(
        scores
    )[::-1][:10]

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
print("BM25 Example Retrieval")
print("="*80)

sample_query = queries[0]

sample_scores = bm25.get_scores(
    sample_query.lower().split()
)

top_results = np.argsort(
    sample_scores
)[::-1][:5]

print(
    f"Query: {sample_query}"
)

display(

    chunks_df.iloc[
        top_results
    ][[
        "document_title",
        "language",
        "chunk_text"
    ]]

)

print("\n" + "="*80)
print("Saving BM25 Artifacts")
print("="*80)

with open(
    "bm25_index.pkl",
    "wb"
) as f:

    pickle.dump(
        bm25,
        f
    )

benchmark_df = pd.DataFrame({

    "Metric":[

        "Corpus Size",
        "Build Time (sec)",
        "Average Latency (ms)",
        "P95 Latency (ms)",
        "Queries Per Second"

    ],

    "Value":[

        len(tokenized_corpus),
        round(build_time,2),
        round(avg_latency,3),
        round(p95_latency,3),
        round(qps,2)

    ]

})

benchmark_df.to_csv(
    "bm25_benchmark.csv",
    index=False
)

print(
    "Saved: bm25_index.pkl"
)

print(
    "Saved: bm25_benchmark.csv"
)

print("\n" + "="*80)
print("BM25 Baseline Summary")
print("="*80)

print(
    benchmark_df
)

print("\n" + "="*80)
print("Step 7 Completed Successfully")
print("="*80)
