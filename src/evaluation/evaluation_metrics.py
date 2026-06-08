# Retrieval Evaluation Framework

import time
import pickle
import psutil
import faiss
import numpy as np
import pandas as pd

from tqdm.auto import tqdm
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder

print("="*80)
print("Loading Evaluation Assets")
print("="*80)

chunks_df = pd.read_parquet(
    "retrieval_corpus_50k.parquet"
)

test_df = pd.read_parquet(
    "test.parquet"
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
    f"Test Queries: {len(test_df):,}"
)

print("\n"+"="*80)
print("Loading Models")
print("="*80)

dense_model = SentenceTransformer(
    "intfloat/multilingual-e5-base",
    device="cuda"
)

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
    device="cuda"
)

print("Models Loaded")

TOP_K = 10
RRF_K = 60

evaluation_df = (
    test_df[
        [
            "question_text",
            "document_title"
        ]
    ]
    .dropna()
    .sample(
        min(500,len(test_df)),
        random_state=42
    )
    .reset_index(drop=True)
)

print(
    f"\nEvaluation Queries: {len(evaluation_df):,}"
)

def compute_metrics(
    ranks
):

    recall1 = np.mean(
        [r <= 1 for r in ranks]
    )

    recall5 = np.mean(
        [r <= 5 for r in ranks]
    )

    recall10 = np.mean(
        [r <= 10 for r in ranks]
    )

    mrr = np.mean(
        [1/r if r <= 10 else 0 for r in ranks]
    )

    ndcg = np.mean(
        [
            1/np.log2(r+1)
            if r <= 10 else 0
            for r in ranks
        ]
    )

    return (
        recall1,
        recall5,
        recall10,
        mrr,
        ndcg
    )

def evaluate_bm25():

    ranks = []
    latencies = []

    for _, row in tqdm(
        evaluation_df.iterrows(),
        total=len(evaluation_df),
        leave=False
    ):

        query = row["question_text"]
        target = row["document_title"]

        start = time.time()

        scores = bm25.get_scores(
            query.lower().split()
        )

        ids = np.argsort(
            scores
        )[::-1][:TOP_K]

        latency = (
            time.time()-start
        )*1000

        latencies.append(
            latency
        )

        retrieved = (
            chunks_df.iloc[ids]
            ["document_title"]
            .tolist()
        )

        rank = 999

        for i,title in enumerate(
            retrieved,
            start=1
        ):
            if title == target:
                rank = i
                break

        ranks.append(rank)

    return ranks, latencies

def evaluate_dense():

    ranks = []
    latencies = []

    for _, row in tqdm(
        evaluation_df.iterrows(),
        total=len(evaluation_df),
        leave=False
    ):

        query = row["question_text"]
        target = row["document_title"]

        start = time.time()

        q_emb = dense_model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True
        )

        _, ids = index.search(
            q_emb.astype(np.float32),
            TOP_K
        )

        latency = (
            time.time()-start
        )*1000

        latencies.append(
            latency
        )

        retrieved = (
            chunks_df.iloc[
                ids[0]
            ]["document_title"]
            .tolist()
        )

        rank = 999

        for i,title in enumerate(
            retrieved,
            start=1
        ):
            if title == target:
                rank = i
                break

        ranks.append(rank)

    return ranks, latencies

def evaluate_hybrid():

    ranks = []
    latencies = []

    for _, row in tqdm(
        evaluation_df.iterrows(),
        total=len(evaluation_df),
        leave=False
    ):

        query = row["question_text"]
        target = row["document_title"]

        start = time.time()

        bm25_scores = bm25.get_scores(
            query.lower().split()
        )

        bm25_ids = np.argsort(
            bm25_scores
        )[::-1][:50]

        q_emb = dense_model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True
        )

        _, dense_ids = index.search(
            q_emb.astype(np.float32),
            50
        )

        dense_ids = dense_ids[0]

        rrf_scores = {}

        for rank,doc_id in enumerate(
            bm25_ids
        ):
            rrf_scores[doc_id] = (
                rrf_scores.get(doc_id,0)
                + 1/(RRF_K+rank+1)
            )

        for rank,doc_id in enumerate(
            dense_ids
        ):
            rrf_scores[doc_id] = (
                rrf_scores.get(doc_id,0)
                + 1/(RRF_K+rank+1)
            )

        final_ids = [
            x[0]
            for x in sorted(
                rrf_scores.items(),
                key=lambda x:x[1],
                reverse=True
            )[:TOP_K]
        ]

        latency = (
            time.time()-start
        )*1000

        latencies.append(
            latency
        )

        retrieved = (
            chunks_df.iloc[
                final_ids
            ]["document_title"]
            .tolist()
        )

        rank = 999

        for i,title in enumerate(
            retrieved,
            start=1
        ):
            if title == target:
                rank = i
                break

        ranks.append(rank)

    return ranks, latencies

print("\n"+"="*80)
print("Evaluating BM25")
print("="*80)

bm25_ranks,bm25_lat = evaluate_bm25()

print("\n"+"="*80)
print("Evaluating Dense Retrieval")
print("="*80)

dense_ranks,dense_lat = evaluate_dense()

print("\n"+"="*80)
print("Evaluating Hybrid Retrieval")
print("="*80)

hybrid_ranks,hybrid_lat = evaluate_hybrid()

memory_usage = (
    psutil.Process()
    .memory_info()
    .rss
    /1024**2
)

results = []

for name,ranks,latencies in [

    ("BM25",
     bm25_ranks,
     bm25_lat),

    ("Dense",
     dense_ranks,
     dense_lat),

    ("Hybrid",
     hybrid_ranks,
     hybrid_lat)

]:

    recall1,recall5,recall10,mrr,ndcg = (
        compute_metrics(ranks)
    )

    results.append({

        "System":name,

        "Recall@1":
        round(recall1,4),

        "Recall@5":
        round(recall5,4),

        "Recall@10":
        round(recall10,4),

        "MRR":
        round(mrr,4),

        "nDCG":
        round(ndcg,4),

        "Avg_Latency_ms":
        round(np.mean(latencies),3),

        "P95_Latency_ms":
        round(np.percentile(
            latencies,
            95
        ),3),

        "QPS":
        round(
            1000/np.mean(latencies),
            2
        ),

        "Memory_MB":
        round(memory_usage,2)
    })

results_df = pd.DataFrame(
    results
)

print("\n"+"="*80)
print("Final Evaluation Results")
print("="*80)

display(
    results_df
)

results_df.to_csv(
    "retrieval_evaluation_results.csv",
    index=False
)

print("\n"+"="*80)
print("Saved Artifact")
print("="*80)

print(
    "retrieval_evaluation_results.csv"
)

print("\n"+"="*80)
print("Step 11 Completed Successfully")
print("="*80)
