# Retrieval Preprocessing Pipeline

!pip -q install pyarrow tqdm

import gc
import math
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from tqdm.auto import tqdm

print("="*80)
print("Loading Training Split")
print("="*80)

train_df = pd.read_parquet(
    "train.parquet"
)

print(
    f"Training Documents: {len(train_df):,}"
)

print("\n" + "="*80)
print("Fixed Chunk Configuration")
print("="*80)

CHUNK_SIZE = 256
OVERLAP = 50

print(
    f"Chunk Size: {CHUNK_SIZE}"
)

print(
    f"Overlap: {OVERLAP}"
)

def fixed_chunk_text(
    text,
    chunk_size=256,
    overlap=50
):

    if not isinstance(text, str):
        return []

    words = text.split()

    if len(words) == 0:
        return []

    chunks = []

    start = 0

    while start < len(words):

        end = start + chunk_size

        chunk = " ".join(
            words[start:end]
        )

        chunks.append(
            chunk
        )

        start += (
            chunk_size - overlap
        )

    return chunks

print("\n" + "="*80)
print("Streaming Chunk Generation")
print("="*80)

OUTPUT_FILE = (
    "retrieval_corpus.parquet"
)

BATCH_SIZE = 5000

writer = None

batch_records = []

total_chunks = 0

for document_id, row in tqdm(
    train_df.iterrows(),
    total=len(train_df)
):

    chunks = fixed_chunk_text(
        row["document_plaintext"],
        CHUNK_SIZE,
        OVERLAP
    )

    for chunk_id, chunk in enumerate(
        chunks
    ):

        batch_records.append({

            "document_id":
                int(document_id),

            "chunk_id":
                int(chunk_id),

            "chunk_position":
                round(
                    chunk_id /
                    max(len(chunks), 1),
                    4
                ),

            "chunk_text":
                chunk,

            "language":
                row["language"],

            "document_title":
                row["document_title"]

        })

        total_chunks += 1

    if len(batch_records) >= BATCH_SIZE:

        batch_df = pd.DataFrame(
            batch_records
        )

        table = pa.Table.from_pandas(
            batch_df,
            preserve_index=False
        )

        if writer is None:

            writer = pq.ParquetWriter(
                OUTPUT_FILE,
                table.schema
            )

        writer.write_table(
            table
        )

        batch_records = []

        gc.collect()

if len(batch_records) > 0:

    batch_df = pd.DataFrame(
        batch_records
    )

    table = pa.Table.from_pandas(
        batch_df,
        preserve_index=False
    )

    if writer is None:

        writer = pq.ParquetWriter(
            OUTPUT_FILE,
            table.schema
        )

    writer.write_table(
        table
    )

if writer:

    writer.close()

print("\n" + "="*80)
print("Chunking Completed")
print("="*80)

print(
    f"Total Chunks Created: {total_chunks:,}"
)

print(
    f"Output File: {OUTPUT_FILE}"
)

print("\n" + "="*80)
print("Quick Validation")
print("="*80)

sample_df = pd.read_parquet(
    OUTPUT_FILE
)

print(
    f"Stored Chunks: {len(sample_df):,}"
)

display(
    sample_df.head()
)

avg_chunks = (
    len(sample_df)
    /
    len(train_df)
)

print(
    f"\nAverage Chunks Per Document: "
    f"{avg_chunks:.2f}"
)

print("\nStep 4 Completed Successfully")
