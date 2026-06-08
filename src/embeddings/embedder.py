# Embedding Pipeline

!pip -q install sentence-transformers

import gc
import os
import time
import torch
import numpy as np
import pandas as pd

from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer

print("="*80)
print("Loading Retrieval Corpus")
print("="*80)

chunks_df = pd.read_parquet(
    "retrieval_corpus.parquet"
)

print(
    f"Original Chunks: {len(chunks_df):,}"
)

print("\n" + "="*80)
print("Creating Retrieval Subset")
print("="*80)

MAX_CHUNKS = 50000

if len(chunks_df) > MAX_CHUNKS:

    chunks_df = (
        chunks_df
        .groupby("language", group_keys=False)
        .apply(
            lambda x: x.sample(
                min(
                    len(x),
                    int(MAX_CHUNKS / chunks_df["language"].nunique())
                ),
                random_state=42
            )
        )
        .reset_index(drop=True)
    )

print(
    f"Chunks Used: {len(chunks_df):,}"
)

chunks_df.to_parquet(
    "retrieval_corpus_50k.parquet",
    index=False
)

texts = chunks_df[
    "chunk_text"
].tolist()

print("\n" + "="*80)
print("Loading Embedding Model")
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
    "\nModel Loaded Successfully"
)

print("\n" + "="*80)
print("Generating Embeddings")
print("="*80)

BATCH_SIZE = 16

all_embeddings = []

start_time = time.time()

for start in tqdm(
    range(
        0,
        len(texts),
        256
    )
):

    batch = texts[
        start:start+256
    ]

    embeddings = model.encode(

        batch,

        batch_size=BATCH_SIZE,

        normalize_embeddings=True,

        convert_to_numpy=True,

        show_progress_bar=False

    )

    all_embeddings.append(
        embeddings
    )

    torch.cuda.empty_cache()

    gc.collect()

embedding_matrix = np.vstack(
    all_embeddings
)

total_time = (
    time.time()
    -
    start_time
)

chunks_per_second = (
    len(texts)
    /
    total_time
)

print("\n" + "="*80)
print("Embedding Statistics")
print("="*80)

print(
    f"Embedding Shape: {embedding_matrix.shape}"
)

print(
    f"Dimension: {embedding_matrix.shape[1]}"
)

print(
    f"Total Time: {total_time:.2f} sec"
)

print(
    f"Chunks / Second: {chunks_per_second:.2f}"
)

print("\n" + "="*80)
print("Saving Artifacts")
print("="*80)

np.save(
    "multilingual_e5_embeddings.npy",
    embedding_matrix
)

benchmark_df = pd.DataFrame({

    "metric":[
        "total_chunks",
        "embedding_dimension",
        "total_time_seconds",
        "chunks_per_second"
    ],

    "value":[
        len(texts),
        embedding_matrix.shape[1],
        round(total_time,2),
        round(chunks_per_second,2)
    ]
})

benchmark_df.to_csv(
    "embedding_benchmark.csv",
    index=False
)

print(
    "Saved: multilingual_e5_embeddings.npy"
)

print(
    "Saved: embedding_benchmark.csv"
)

print(
    "Saved: retrieval_corpus_50k.parquet"
)

del texts
del all_embeddings

gc.collect()

torch.cuda.empty_cache()

print("\n" + "="*80)
print("Step 5 Completed Successfully")
print("="*80)
