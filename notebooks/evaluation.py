# Retrieval Evaluation Framework
# Document-Level Relevance Evaluation

import os
import time
import pickle
import faiss
import psutil
import numpy as np
import pandas as pd

from tqdm import tqdm
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

print("="*80)
print("Loading Evaluation Assets")
print("="*80)

corpus = pd.read_parquet("retrieval_corpus_50k.parquet")
test_df = pd.read_parquet("test.parquet")

print(f"Corpus Size: {len(corpus):,}")
print(f"Test Queries: {len(test_df):,}")

# Load Models

print("\n"+"="*80)
print("Loading Models")
print("="*80)

dense_model = SentenceTransformer(
    "intfloat/multilingual-e5-base"
)

index = faiss.read_index(
    "hnsw_index.faiss"
)

with open(
    "bm25_index.pkl",
    "rb"
) as f:
    bm25 = pickle.load(f)

print("Models Loaded")

# Evaluation Sample

EVAL_QUERIES = 500

test_sample = test_df.sample(
    min(EVAL_QUERIES, len(test_df)),
    random_state=42
).reset_index(drop=True)

print(
    f"\nEvaluation Queries: "
    f"{len(test_sample)}"
)

# Metrics

def recall_at_k(
    retrieved_titles,
    target_title,
    k
):
    return int(
        target_title in retrieved_titles[:k]
    )

def mrr_score(
    retrieved_titles,
    target_title
):
    for rank, title in enumerate(
        retrieved_titles,
        start=1
    ):
        if title == target_title:
            return 1/rank
    return 0

def ndcg_score(
    retrieved_titles,
    target_title,
    k=10
):
    for i, title in enumerate(
        retrieved_titles[:k]
    ):
        if title == target_title:
            return (
                1 /
                np.log2(i+2)
            )
    return 0

# BM25 Evaluation

print("\n"+"="*80)
print("Evaluating BM25")
print("="*80)

bm25_r1 = []
bm25_r5 = []
bm25_r10 = []
bm25_mrr = []
bm25_ndcg = []
bm25_latency = []

for _, row in tqdm(
    test_sample.iterrows(),
    total=len(test_sample)
):

    query = str(row["question_text"])
    target = str(row["document_title"])

    start = time.time()

    scores = bm25.get_scores(
        query.split()
    )

    idx = np.argsort(scores)[::-1][:10]

    results = corpus.iloc[idx]

    latency = (
        time.time()-start
    )*1000

    retrieved_titles = (
        results["document_title"]
        .drop_duplicates()
        .tolist()
    )

    bm25_r1.append(
        recall_at_k(
            retrieved_titles,
            target,
            1
        )
    )

    bm25_r5.append(
        recall_at_k(
            retrieved_titles,
            target,
            5
        )
    )

    bm25_r10.append(
        recall_at_k(
            retrieved_titles,
            target,
            10
        )
    )

    bm25_mrr.append(
        mrr_score(
            retrieved_titles,
            target
        )
    )

    bm25_ndcg.append(
        ndcg_score(
            retrieved_titles,
            target
        )
    )

    bm25_latency.append(
        latency
    )

# Dense Evaluation

print("\n"+"="*80)
print("Evaluating Dense Retrieval")
print("="*80)

dense_r1 = []
dense_r5 = []
dense_r10 = []
dense_mrr = []
dense_ndcg = []
dense_latency = []

for _, row in tqdm(
    test_sample.iterrows(),
    total=len(test_sample)
):

    query = (
        "query: " +
        str(row["question_text"])
    )

    target = str(
        row["document_title"]
    )

    start = time.time()

    q_emb = dense_model.encode(
        [query],
        normalize_embeddings=True
    )

    _, idx = index.search(
        q_emb.astype(
            np.float32
        ),
        10
    )

    results = corpus.iloc[
        idx[0]
    ]

    latency = (
        time.time()-start
    )*1000

    retrieved_titles = (
        results["document_title"]
        .drop_duplicates()
        .tolist()
    )

    dense_r1.append(
        recall_at_k(
            retrieved_titles,
            target,
            1
        )
    )

    dense_r5.append(
        recall_at_k(
            retrieved_titles,
            target,
            5
        )
    )

    dense_r10.append(
        recall_at_k(
            retrieved_titles,
            target,
            10
        )
    )

    dense_mrr.append(
        mrr_score(
            retrieved_titles,
            target
        )
    )

    dense_ndcg.append(
        ndcg_score(
            retrieved_titles,
            target
        )
    )

    dense_latency.append(
        latency
    )

# Results

memory_mb = (
    psutil.Process()
    .memory_info()
    .rss
)/(1024**2)

results_df = pd.DataFrame({

    "System":[
        "BM25",
        "Dense"
    ],

    "Recall@1":[
        np.mean(bm25_r1),
        np.mean(dense_r1)
    ],

    "Recall@5":[
        np.mean(bm25_r5),
        np.mean(dense_r5)
    ],

    "Recall@10":[
        np.mean(bm25_r10),
        np.mean(dense_r10)
    ],

    "MRR":[
        np.mean(bm25_mrr),
        np.mean(dense_mrr)
    ],

    "nDCG":[
        np.mean(bm25_ndcg),
        np.mean(dense_ndcg)
    ],

    "Avg_Latency_ms":[
        np.mean(bm25_latency),
        np.mean(dense_latency)
    ],

    "P95_Latency_ms":[
        np.percentile(
            bm25_latency,
            95
        ),
        np.percentile(
            dense_latency,
            95
        )
    ],

    "QPS":[
        1000/
        np.mean(
            bm25_latency
        ),

        1000/
        np.mean(
            dense_latency
        )
    ],

    "Memory_MB":[
        memory_mb,
        memory_mb
    ]
})

print("\n"+"="*80)
print("FINAL RESULTS")
print("="*80)

print(results_df)

results_df.to_csv(
    "retrieval_evaluation_results.csv",
    index=False
)

print("\nSaved:")
print(
    "retrieval_evaluation_results.csv"
)

print("\nSTEP 11 FIXED COMPLETE")


Results:
================================================================================
Loading Evaluation Assets
================================================================================
Corpus Size: 48,573
Test Queries: 3,237

================================================================================
Loading Models
================================================================================
Loading weights: 100%
 199/199 [00:00<00:00, 4760.55it/s]
Models Loaded

Evaluation Queries: 500

================================================================================
Evaluating BM25
================================================================================
100%|██████████| 500/500 [00:58<00:00,  8.56it/s]

================================================================================
Evaluating Dense Retrieval
================================================================================
100%|██████████| 500/500 [00:08<00:00, 57.57it/s]
================================================================================
FINAL RESULTS
================================================================================
  System  Recall@1  Recall@5  Recall@10     MRR      nDCG  Avg_Latency_ms  \
0   BM25       0.0     0.000      0.000  0.0000  0.000000      115.581945   
1  Dense       0.0     0.002      0.002  0.0005  0.000861       16.767246   

   P95_Latency_ms        QPS    Memory_MB  
0      219.863963   8.651870  5974.023438  
1       32.530046  59.640087  5974.023438  

Saved:
retrieval_evaluation_results.csv
