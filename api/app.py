# FastAPI Deployment

from pathlib import Path

app_code = '''
import time
import faiss
import numpy as np
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

print("Loading Retrieval Assets...")

chunks_df = pd.read_parquet(
    "retrieval_corpus_50k.parquet"
)

index = faiss.read_index(
    "hnsw_index.faiss"
)

model = SentenceTransformer(
    "intfloat/multilingual-e5-base",
    device="cpu"
)

print("System Ready")

app = FastAPI(
    title="Multilingual Retrieval System",
    description="Hybrid Retrieval and Reranking API",
    version="1.0"
)

class QueryRequest(BaseModel):
    query: str
    top_k: int = 10

@app.get("/")
def home():

    return {
        "project":
        "Multilingual Retrieval-Augmented QA Benchmarking System",

        "status":
        "running"
    }

@app.post("/retrieve")
def retrieve(
    request: QueryRequest
):

    start_time = time.time()

    query_embedding = model.encode(
        [request.query],
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    scores, ids = index.search(
        query_embedding.astype(
            np.float32
        ),
        request.top_k
    )

    results = []

    for score, idx in zip(
        scores[0],
        ids[0]
    ):

        row = chunks_df.iloc[idx]

        results.append({

            "document_title":
                str(
                    row["document_title"]
                ),

            "language":
                str(
                    row["language"]
                ),

            "score":
                float(score),

            "chunk_text":
                str(
                    row["chunk_text"]
                )[:500]
        })

    latency_ms = (
        time.time()
        -
        start_time
    ) * 1000

    return {

        "query":
            request.query,

        "top_k":
            request.top_k,

        "latency_ms":
            round(
                latency_ms,
                3
            ),

        "results":
            results
    }

@app.get("/health")
def health():

    return {

        "status":
            "healthy",

        "documents":
            len(
                chunks_df
            ),

        "index_vectors":
            index.ntotal
    }
'''

Path("app.py").write_text(app_code)

print("="*80)
print("FastAPI Application Generated")
print("="*80)

print("Saved: app.py")

print("\nEndpoints")
print("- GET  /")
print("- GET  /health")
print("- POST /retrieve")

print("\nStep 15 Completed Successfully")
print("="*80)
