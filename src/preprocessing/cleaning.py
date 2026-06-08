# Data Cleaning & Document Processing

!pip -q install datasets

import re
import unicodedata
import pandas as pd
from datasets import load_dataset

print("="*80)
print("Loading TyDiQA")
print("="*80)

dataset = load_dataset(
    "google-research-datasets/tydiqa",
    "primary_task",
    split="train[:30000]"
)

df = dataset.to_pandas()

print(f"\nOriginal Rows: {len(df):,}")

print("\n" + "="*80)
print("Unicode Normalization")
print("="*80)

def normalize_text(text):

    if pd.isna(text):
        return ""

    text = str(text)

    text = unicodedata.normalize(
        "NFKC",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text

text_columns = [
    "question_text",
    "document_plaintext",
    "document_title"
]

for col in text_columns:

    df[col] = df[col].apply(
        normalize_text
    )

print("\n" + "="*80)
print("Removing Corrupted Rows")
print("="*80)

before = len(df)

df = df[
    df["question_text"].notna()
]

df = df[
    df["document_plaintext"].notna()
]

df = df[
    df["language"].notna()
]

after = len(df)

print(
    f"Removed: {before-after:,}"
)

print("\n" + "="*80)
print("Removing Empty Documents")
print("="*80)

before = len(df)

df = df[
    df["question_text"].str.len() > 0
]

df = df[
    df["document_plaintext"].str.len() > 0
]

after = len(df)

print(
    f"Removed: {before-after:,}"
)

print("\n" + "="*80)
print("Removing Extremely Small Documents")
print("="*80)

before = len(df)

df["document_word_count"] = (
    df["document_plaintext"]
    .str.split()
    .str.len()
)

df = df[
    df["document_word_count"] >= 20
]

after = len(df)

print(
    f"Removed: {before-after:,}"
)

print("\n" + "="*80)
print("Removing Duplicate Documents")
print("="*80)

before = len(df)

df = df.drop_duplicates(
    subset=[
        "document_plaintext"
    ]
)

after = len(df)

print(
    f"Removed: {before-after:,}"
)

print("\n" + "="*80)
print("Removing Duplicate Question-Document Pairs")
print("="*80)

before = len(df)

df = df.drop_duplicates(
    subset=[
        "question_text",
        "document_plaintext"
    ]
)

after = len(df)

print(
    f"Removed: {before-after:,}"
)

print("\n" + "="*80)
print("Basic Quality Statistics")
print("="*80)

df["question_word_count"] = (
    df["question_text"]
    .str.split()
    .str.len()
)

summary = pd.DataFrame({

    "Metric":[
        "Final Rows",
        "Languages",
        "Avg Question Length",
        "Avg Document Length",
        "Max Document Length"
    ],

    "Value":[
        len(df),
        df["language"].nunique(),
        round(
            df["question_word_count"].mean(),
            2
        ),
        round(
            df["document_word_count"].mean(),
            2
        ),
        int(
            df["document_word_count"].max()
        )
    ]
})

display(summary)

print("\n" + "="*80)
print("Language Distribution")
print("="*80)

display(
    df["language"]
    .value_counts()
)

print("\n" + "="*80)
print("Saving Clean Corpus")
print("="*80)

df = df.reset_index(
    drop=True
)

df.to_parquet(
    "clean_tydiqa.parquet",
    index=False
)

df.to_csv(
    "clean_tydiqa.csv",
    index=False
)

print(
    f"\nClean Dataset Shape: {df.shape}"
)

print(
    "\nSaved Files:"
)

print(
    "clean_tydiqa.parquet"
)

print(
    "clean_tydiqa.csv"
)

print("\nStep 2 Completed Successfully")
