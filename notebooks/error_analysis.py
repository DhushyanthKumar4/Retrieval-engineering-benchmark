# Error Analysis

import warnings
warnings.filterwarnings("ignore")

import gc
import time
import faiss
import pickle
import numpy as np
import pandas as pd

from tqdm.auto import tqdm
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

print("="*80)
print("Loading Assets")
print("="*80)

chunks_df = pd.read_parquet(
    "retrieval_corpus_50k.parquet"
)

test_df = pd.read_parquet(
    "test.parquet"
)

index = faiss.read_index(
    "hnsw_index.faiss"
)

with open(
    "bm25_index.pkl",
    "rb"
) as f:
    bm25 = pickle.load(f)

print(f"Corpus Size: {len(chunks_df):,}")
print(f"Test Queries: {len(test_df):,}")

print("\n" + "="*80)
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

print("\n" + "="*80)
print("Creating Evaluation Sample")
print("="*80)

N_QUERIES = min(500, len(test_df))

eval_df = (
    test_df
    .sample(
        N_QUERIES,
        random_state=42
    )
    .reset_index(drop=True)
)

print(f"Queries: {len(eval_df):,}")

TOP_K = 10

language_stats = {}

failure_examples = []

bm25_failures = 0
dense_failures = 0
reranker_fixes = 0

for _, row in tqdm(
    eval_df.iterrows(),
    total=len(eval_df)
):

    query = str(
        row["question_text"]
    )

    target_title = str(
        row["document_title"]
    )

    language = str(
        row["language"]
    )

    if language not in language_stats:

        language_stats[language] = {
            "queries": 0,
            "bm25_hits": 0,
            "dense_hits": 0,
            "reranker_hits": 0
        }

    language_stats[language]["queries"] += 1

    # BM25

    bm25_scores = bm25.get_scores(
        query.lower().split()
    )

    bm25_idx = np.argsort(
        bm25_scores
    )[::-1][:TOP_K]

    bm25_titles = set(
        chunks_df.iloc[
            bm25_idx
        ]["document_title"]
    )

    bm25_hit = (
        target_title
        in bm25_titles
    )

    # Dense

    q_emb = dense_model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    _, dense_idx = index.search(
        q_emb.astype(np.float32),
        50
    )

    dense_idx = dense_idx[0]

    dense_titles = set(
        chunks_df.iloc[
            dense_idx[:TOP_K]
        ]["document_title"]
    )

    dense_hit = (
        target_title
        in dense_titles
    )

    # Reranker

    candidate_texts = (
        chunks_df
        .iloc[dense_idx]
        ["chunk_text"]
        .tolist()
    )

    pairs = [
        [query, text]
        for text in candidate_texts
    ]

    rerank_scores = reranker.predict(
        pairs,
        batch_size=32,
        show_progress_bar=False
    )

    rerank_order = np.argsort(
        rerank_scores
    )[::-1]

    rerank_idx = dense_idx[
        rerank_order[:TOP_K]
    ]

    rerank_titles = set(
        chunks_df.iloc[
            rerank_idx
        ]["document_title"]
    )

    rerank_hit = (
        target_title
        in rerank_titles
    )

    # Metrics

    if bm25_hit:
        language_stats[language]["bm25_hits"] += 1
    else:
        bm25_failures += 1

    if dense_hit:
        language_stats[language]["dense_hits"] += 1
    else:
        dense_failures += 1

    if rerank_hit:
        language_stats[language]["reranker_hits"] += 1

    if (
        (not dense_hit)
        and rerank_hit
    ):
        reranker_fixes += 1

    if (
        len(failure_examples) < 50
        and not dense_hit
    ):

        failure_examples.append({

            "language": language,

            "query": query[:300],

            "target_document":
                target_title,

            "bm25_success":
                bm25_hit,

            "dense_success":
                dense_hit,

            "reranker_success":
                rerank_hit

        })

print("\n" + "="*80)
print("Language Failure Analysis")
print("="*80)

language_rows = []

for lang, stats in language_stats.items():

    total = stats["queries"]

    language_rows.append({

        "language": lang,

        "queries": total,

        "bm25_recall":
            stats["bm25_hits"]/total,

        "dense_recall":
            stats["dense_hits"]/total,

        "reranker_recall":
            stats["reranker_hits"]/total,

        "bm25_failure_rate":
            1 - (
                stats["bm25_hits"]/total
            ),

        "dense_failure_rate":
            1 - (
                stats["dense_hits"]/total
            )

    })

language_results = pd.DataFrame(
    language_rows
).sort_values(
    "dense_failure_rate",
    ascending=False
)

print(
    language_results.round(4)
)

print("\n" + "="*80)
print("Failure Summary")
print("="*80)

summary = pd.DataFrame({

    "Metric":[

        "Total Queries",

        "BM25 Failures",

        "Dense Failures",

        "Reranker Fixes",

        "BM25 Failure Rate",

        "Dense Failure Rate",

        "Reranker Recovery Rate"

    ],

    "Value":[

        len(eval_df),

        bm25_failures,

        dense_failures,

        reranker_fixes,

        round(
            bm25_failures/
            len(eval_df),
            4
        ),

        round(
            dense_failures/
            len(eval_df),
            4
        ),

        round(
            reranker_fixes/
            max(
                dense_failures,
                1
            ),
            4
        )
    ]
})

print(summary)

print("\n" + "="*80)
print("Sample Failure Cases")
print("="*80)

failure_df = pd.DataFrame(
    failure_examples
)

display(
    failure_df.head(20)
)

print("\n" + "="*80)
print("Saving Artifacts")
print("="*80)

language_results.to_csv(
    "language_failure_analysis.csv",
    index=False
)

summary.to_csv(
    "failure_summary.csv",
    index=False
)

failure_df.to_csv(
    "failure_examples.csv",
    index=False
)

print("Saved:")
print("language_failure_analysis.csv")
print("failure_summary.csv")
print("failure_examples.csv")

print("\n" + "="*80)
print("Research Insights Produced")
print("="*80)

print("""
1. Language-specific retrieval weaknesses
2. BM25 failure patterns
3. Dense retrieval failure patterns
4. Reranker recovery effectiveness
5. Cross-lingual robustness analysis
6. Real failure case inspection
""")

print("\nStep 13 Completed Successfully")
print("="*80)

gc.collect()



Results:

================================================================================
Loading Assets
================================================================================
Corpus Size: 48,573
Test Queries: 3,237

================================================================================
Loading Models
================================================================================
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
WARNING:huggingface_hub.utils._http:Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Loading weights: 100%
 199/199 [00:00<00:00, 722.76it/s]
Loading weights: 100%
 105/105 [00:00<00:00, 517.74it/s]
Models Loaded

================================================================================
Creating Evaluation Sample
================================================================================
Queries: 500
100%
 500/500 [04:44<00:00,  2.02it/s]

================================================================================
Language Failure Analysis
================================================================================
      language  queries  bm25_recall  dense_recall  reranker_recall  \
0       telugu       65       0.0000        0.0000           0.0000   
1     japanese       62       0.0000        0.0000           0.0000   
2       arabic       86       0.0000        0.0000           0.0000   
3      swahili       33       0.0000        0.0000           0.0000   
4      finnish       54       0.0000        0.0000           0.0000   
5      bengali       27       0.0000        0.0000           0.0000   
6         thai       45       0.0000        0.0000           0.0000   
8      english       33       0.0000        0.0000           0.0000   
9   indonesian       33       0.0000        0.0000           0.0000   
10      korean       19       0.0000        0.0000           0.0000   
7      russian       43       0.0233        0.0233           0.0233   

    bm25_failure_rate  dense_failure_rate  
0              1.0000              1.0000  
1              1.0000              1.0000  
2              1.0000              1.0000  
3              1.0000              1.0000  
4              1.0000              1.0000  
5              1.0000              1.0000  
6              1.0000              1.0000  
8              1.0000              1.0000  
9              1.0000              1.0000  
10             1.0000              1.0000  
7              0.9767              0.9767  

================================================================================
Failure Summary
================================================================================
                   Metric    Value
0           Total Queries  500.000
1           BM25 Failures  499.000
2          Dense Failures  499.000
3          Reranker Fixes    0.000
4       BM25 Failure Rate    0.998
5      Dense Failure Rate    0.998
6  Reranker Recovery Rate    0.000

================================================================================
Sample Failure Cases
================================================================================
language	query	target_document	bm25_success	dense_success	reranker_success
0	telugu	లంక చిత్ర నిడివి ఎంత?	లంక	False	False	False
1	japanese	イタリア王国海軍は第一次世界大戦中に何隻戦艦をつくった?	イタリア海軍	False	False	False
2	arabic	ما هي اهم مؤلفات مصطفى جواد ؟	مصطفى جواد	False	False	False
3	telugu	హుస్సేన్‌ సాగర్‌ ఎక్కడ ఉంది?	హుసేన్ సాగర్	False	False	False
4	japanese	電車が初めて開発されたのはいつ	日本の電車史	False	False	False
5	japanese	アバクロンビー&フィッチの姉妹ブランドはなんといわれるブランドですか?	西武百貨店	False	False	False
6	swahili	Venezuela ina idadi ya watu wangapi?	Venezuela	False	False	False
7	telugu	జగతిపల్లి ఎస్.టి.డి కోడ్ ఎంత?	మర్రిపాడు (సీతంపేట)	False	False	False
8	swahili	Je, mto Niagara una urefu gani?	Mto Niagara	False	False	False
9	swahili	Dikteta wa kwanza duniani alikuwa nani?	Dikteta	False	False	False
10	finnish	Missä AHL-liiga pelaa?	American Hockey League	False	False	False
11	japanese	アポロ1号の乗組員は全員死んだ?	アポロ1号	False	False	False
12	bengali	শাহজালালের মায়ের নাম কি ছিল ?	শাহজালাল আহম্মদ	False	False	False
13	thai	บาหลี มีประชากรกี่คนในปี2018?	ประเทศอินโดนีเซีย	False	False	False
14	arabic	في أي عام أعدم عبد الوهاب الإنكليزي ؟	عبد الوهاب الإنكليزي	False	False	False
15	russian	Кем была жена Виталия Чуркина?	Чуркин, Виталий Иванович	False	False	False
16	japanese	アッパー半島の面積は?	アッパー半島	False	False	False
17	japanese	フェルキッシュという語を生んだのは誰	アドルフ・ヨーゼフ・ランツ	False	False	False
18	thai	อาณาจักรล้านนามีเมืองหลวงที่สําคัญชื่อว่าเมือง...	อาณาจักรล้านนา	False	False	False
19	japanese	日本最大の図書館は何	図書館	False	False	False

================================================================================
Saving Artifacts
================================================================================
Saved:
language_failure_analysis.csv
failure_summary.csv
failure_examples.csv

================================================================================
Research Insights Produced
================================================================================

1. Language-specific retrieval weaknesses
2. BM25 failure patterns
3. Dense retrieval failure patterns
4. Reranker recovery effectiveness
5. Cross-lingual robustness analysis
6. Real failure case inspection
