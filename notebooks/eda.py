# Data Understanding & EDA

!pip -q install datasets sentence-transformers seaborn

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

sns.set_style("whitegrid")

print("="*80)
print("Loading TyDiQA")
print("="*80)

dataset = load_dataset(
    "google-research-datasets/tydiqa",
    "primary_task",
    split="train"
)

print(dataset)

print("\nConverting sample for semantic analysis...")

semantic_sample_size = 2000

semantic_df = (
    dataset
    .shuffle(seed=42)
    .select(range(semantic_sample_size))
    .to_pandas()
)

print("\nCreating full metadata dataframe...")

languages = dataset["language"]

question_lengths = [
    len(str(q).split())
    for q in dataset["question_text"]
]

document_lengths = [
    len(str(doc).split())
    for doc in dataset["document_plaintext"]
]

document_char_lengths = [
    len(str(doc))
    for doc in dataset["document_plaintext"]
]

eda_summary = pd.DataFrame({

    "Metric":[
        "Total QA Pairs",
        "Total Languages",
        "Average Question Length",
        "Average Passage Length",
        "Average Document Characters",
        "Maximum Passage Length",
        "Maximum Question Length"
    ],

    "Value":[
        len(dataset),
        len(set(languages)),
        np.mean(question_lengths),
        np.mean(document_lengths),
        np.mean(document_char_lengths),
        np.max(document_lengths),
        np.max(question_lengths)
    ]
})

print("\n")
print("="*80)
print("Dataset Summary")
print("="*80)

display(eda_summary)

print("\n")
print("="*80)
print("Language Distribution")
print("="*80)

language_counts = (
    pd.Series(languages)
    .value_counts()
    .sort_values(ascending=False)
)

display(language_counts)

plt.figure(figsize=(12,6))
language_counts.plot(kind="bar")
plt.title("Language Distribution")
plt.ylabel("Count")
plt.xlabel("Language")
plt.show()

print("\n")
print("="*80)
print("Question Length Distribution")
print("="*80)

plt.figure(figsize=(12,6))
sns.histplot(
    question_lengths,
    bins=50
)
plt.title("Question Length Distribution")
plt.xlabel("Words")
plt.show()

print("\n")
print("="*80)
print("Document Length Distribution")
print("="*80)

plt.figure(figsize=(12,6))
sns.histplot(
    document_lengths,
    bins=50
)
plt.title("Document Length Distribution")
plt.xlabel("Words")
plt.show()

print("\n")
print("="*80)
print("Long Document Analysis")
print("="*80)

long_doc_threshold = 1000

long_doc_pct = (
    np.mean(
        np.array(document_lengths)
        > long_doc_threshold
    )
    * 100
)

print(
    f"Documents > {long_doc_threshold} words: "
    f"{long_doc_pct:.2f}%"
)

print("\n")
print("="*80)
print("Chunking Complexity Analysis")
print("="*80)

chunk_size = 256

chunk_counts = [
    int(np.ceil(x/chunk_size))
    for x in document_lengths
]

print(
    f"Average Chunks Per Document: "
    f"{np.mean(chunk_counts):.2f}"
)

print(
    f"Maximum Chunks Per Document: "
    f"{np.max(chunk_counts)}"
)

plt.figure(figsize=(12,6))
sns.histplot(
    chunk_counts,
    bins=50
)
plt.title(
    "Estimated Chunks Per Document"
)
plt.xlabel("Chunk Count")
plt.show()

print("\n")
print("="*80)
print("Semantic Structure Analysis")
print("="*80)

model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

sample_questions = (
    semantic_df["question_text"]
    .fillna("")
    .astype(str)
    .tolist()
)

sample_docs = (
    semantic_df["document_plaintext"]
    .fillna("")
    .astype(str)
    .str[:1000]
    .tolist()
)

question_embeddings = model.encode(
    sample_questions,
    batch_size=64,
    show_progress_bar=True,
    normalize_embeddings=True
)

document_embeddings = model.encode(
    sample_docs,
    batch_size=64,
    show_progress_bar=True,
    normalize_embeddings=True
)

semantic_similarity = np.sum(
    question_embeddings *
    document_embeddings,
    axis=1
)

print(
    f"\nAverage Question-Document Similarity: "
    f"{semantic_similarity.mean():.4f}"
)

plt.figure(figsize=(12,6))
sns.histplot(
    semantic_similarity,
    bins=40
)
plt.title(
    "Question-Document Semantic Similarity"
)
plt.xlabel("Cosine Similarity")
plt.show()

print("\n")
print("="*80)
print("Embedding Cluster Visualization")
print("="*80)

pca = PCA(n_components=2)

reduced_embeddings = pca.fit_transform(
    question_embeddings[:1000]
)

plot_df = pd.DataFrame({

    "PC1": reduced_embeddings[:,0],
    "PC2": reduced_embeddings[:,1],
    "language": semantic_df[
        "language"
    ].iloc[:1000].values

})

plt.figure(figsize=(12,8))

sns.scatterplot(
    data=plot_df,
    x="PC1",
    y="PC2",
    hue="language",
    s=40
)

plt.title(
    "Multilingual Question Embedding Clusters"
)

plt.legend(
    bbox_to_anchor=(1.05,1),
    loc="upper left"
)

plt.show()

print("\n")
print("="*80)
print("Vocabulary Overlap Analysis")
print("="*80)

language_vocab = {}

sample_vocab_df = (
    semantic_df
    .groupby("language")
    .head(200)
)

for lang in sample_vocab_df[
    "language"
].unique():

    text = " ".join(

        sample_vocab_df[
            sample_vocab_df["language"]
            == lang
        ]["question_text"]
        .astype(str)
        .tolist()

    )

    language_vocab[lang] = set(
        text.lower().split()
    )

languages_list = list(
    language_vocab.keys()
)

overlap_matrix = []

for lang1 in languages_list:

    row = []

    for lang2 in languages_list:

        intersection = len(
            language_vocab[lang1]
            &
            language_vocab[lang2]
        )

        union = len(
            language_vocab[lang1]
            |
            language_vocab[lang2]
        )

        row.append(
            intersection/union
            if union > 0
            else 0
        )

    overlap_matrix.append(row)

overlap_df = pd.DataFrame(
    overlap_matrix,
    index=languages_list,
    columns=languages_list
)

plt.figure(figsize=(10,8))

sns.heatmap(
    overlap_df,
    annot=True,
    fmt=".2f"
)

plt.title(
    "Vocabulary Overlap Across Languages"
)

plt.show()

print("\n")
print("="*80)
print("Hypothesis Validation")
print("="*80)

print("""
1. Dense Retrieval is expected to outperform BM25
   because semantic similarity exists across languages.

2. Hybrid Retrieval should improve recall
   by combining lexical and semantic matching.

3. Long documents generate many chunks,
   increasing retrieval complexity.

4. Multilingual embedding clusters indicate
   strong cross-lingual retrieval potential.

5. Retrieval benchmarking is necessary because
   language distributions are highly imbalanced.
""")

print("\nEDA Completed Successfully")


Results:

Dataset Summary
================================================================================
Metric	Value
0	Total QA Pairs	166916.000000
1	Total Languages	11.000000
2	Average Question Length	5.491301
3	Average Passage Length	2335.611296
4	Average Document Characters	17810.803835
5	Maximum Passage Length	106469.000000
6	Maximum Question Length	1344.000000


================================================================================
Language Distribution
================================================================================
count
telugu	24558
arabic	23092
swahili	17613
japanese	16288
finnish	15285
indonesian	14952
russian	12803
thai	11365
korean	10981
bengali	10768
english	9211

dtype: int64



================================================================================
Question Length Distribution
================================================================================



================================================================================
Document Length Distribution
================================================================================



================================================================================
Long Document Analysis
================================================================================
Documents > 1000 words: 52.82%


================================================================================
Chunking Complexity Analysis
================================================================================
Average Chunks Per Document: 9.63
Maximum Chunks Per Document: 416



================================================================================
Semantic Structure Analysis
================================================================================
modules.json: 100%
 229/229 [00:00<00:00, 13.2kB/s]
config_sentence_transformers.json: 100%
 122/122 [00:00<00:00, 8.16kB/s]
README.md: 100%
 3.89k/3.89k [00:00<00:00, 269kB/s]
sentence_bert_config.json: 100%
 53.0/53.0 [00:00<00:00, 5.55kB/s]
config.json: 100%
 645/645 [00:00<00:00, 44.3kB/s]
model.safetensors: 100%
 471M/471M [00:05<00:00, 89.0MB/s]
Loading weights: 100%
 199/199 [00:00<00:00, 4575.67it/s]
tokenizer_config.json: 100%
 526/526 [00:00<00:00, 53.7kB/s]
tokenizer.json: 100%
 9.08M/9.08M [00:00<00:00, 17.4MB/s]
special_tokens_map.json: 100%
 239/239 [00:00<00:00, 24.2kB/s]
config.json: 100%
 190/190 [00:00<00:00, 16.7kB/s]
Batches: 100%
 32/32 [00:01<00:00, 27.31it/s]
Batches: 100%
 32/32 [00:05<00:00,  5.82it/s]

Average Question-Document Similarity: 0.5337



================================================================================
Embedding Cluster Visualization
================================================================================



================================================================================
Vocabulary Overlap Analysis
================================================================================



================================================================================
Hypothesis Validation
================================================================================

1. Dense Retrieval is expected to outperform BM25
   because semantic similarity exists across languages.

2. Hybrid Retrieval should improve recall
   by combining lexical and semantic matching.

3. Long documents generate many chunks,
   increasing retrieval complexity.

4. Multilingual embedding clusters indicate
   strong cross-lingual retrieval potential.

5. Retrieval benchmarking is necessary because
   language distributions are highly imbalanced.


EDA Completed Successfully
