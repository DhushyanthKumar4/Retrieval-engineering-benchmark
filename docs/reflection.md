#  Reflection: Lessons Learned from Building a Multilingual Retrieval System

## Overview

The original objective of this project was to benchmark multilingual retrieval approaches on the TyDi QA dataset using sparse retrieval, dense retrieval, hybrid ranking, and reranking techniques.

As the project evolved, the primary outcome became less about comparing models and more about understanding retrieval systems as engineering artifacts. The most valuable lessons emerged from system behavior, evaluation failures, indexing tradeoffs, and deployment constraints rather than from any individual model.

This reflection summarizes what worked, what failed, and what was learned throughout the project.

---

# 1. What Worked

## 1.1 Dense Retrieval Scaled Better Than Expected

One of the clearest findings was the effectiveness of dense retrieval when paired with approximate nearest neighbor search.

Using multilingual-e5-base embeddings and FAISS HNSW indexing, the system achieved:

* Average retrieval latency of approximately 16.9 ms
* Throughput of approximately 59 queries per second
* Stable multilingual behavior across 11 languages

Compared with BM25:

* Throughput increased by approximately 3.2×
* Latency decreased by approximately 69%

The experiment reinforced a common observation in modern retrieval systems:

> Once vector representations are available, retrieval performance becomes heavily dependent on index design rather than embedding generation.

The embedding model mattered, but the indexing strategy ultimately determined system responsiveness.

---

## 1.2 HNSW Was the Most Impactful Infrastructure Decision

Among all architectural choices, selecting HNSW had the largest measurable effect on retrieval speed.

Measured latency:

* Flat Index: 11.149 ms
* HNSW: 0.204 ms

This represents approximately:

54.7× lower retrieval latency.

The result highlighted an important systems lesson:

> Retrieval quality improvements often receive more attention than infrastructure improvements, but infrastructure choices can produce significantly larger practical gains.

In production environments, reducing latency by an order of magnitude is often more valuable than marginal improvements in ranking metrics.

---

## 1.3 Hybrid Retrieval Improved Robustness

Sparse and dense retrieval exhibited different failure modes.

BM25 performed well when:

* exact terminology matched
* named entities appeared directly in documents
* lexical overlap was high

Dense retrieval performed well when:

* wording differed
* paraphrasing occurred
* semantic matching was required

Combining both methods through reciprocal rank fusion produced a more robust retrieval pipeline.

The experiment demonstrated that:

> Dense retrieval does not replace sparse retrieval; it complements it.

This finding aligns with modern production search systems, where hybrid retrieval is increasingly standard.

---

## 1.4 Cross Encoder Reranking Improved Ranking Precision

The reranker successfully introduced a second-stage precision layer.

Rather than searching the entire corpus, the reranker operated only on the top candidate documents.

This architecture provided:

* better ranking control
* semantic pairwise scoring
* more interpretable final rankings

The experiment reinforced a key retrieval principle:

> Retrieval and ranking should be treated as separate optimization problems.

Candidate generation and candidate ranking require different architectures and should be optimized independently.

---

# 2. What Failed

## 2.1 The Original Evaluation Was Incorrect

The most important failure occurred during evaluation.

Initial metrics suggested extremely poor retrieval performance:

* Recall@K near zero
* MRR near zero
* nDCG near zero

At first glance, these results appeared to indicate model failure.

Further investigation revealed that the issue was not model quality.

The retrieval corpus had been constructed exclusively from training documents.

When corpus overlap was measured:

* Corpus titles: 12,623
* Test titles: 3,231
* Shared titles: 54

Overlap was only:

1.67%

The majority of test answers were therefore impossible to retrieve.

This was not a retrieval failure.

It was an evaluation design failure.

This became the most important lesson of the project.

---

## 2.2 Data Coverage Dominated Model Performance

The corpus mismatch revealed an important reality:

> Retrieval quality cannot exceed corpus quality.

Regardless of embedding model, indexing strategy, or reranker sophistication, missing information cannot be retrieved.

The experiment demonstrated that:

* Corpus coverage matters.
* Dataset alignment matters.
* Evaluation assumptions matter.

These factors had a larger effect on benchmark results than the retrieval architecture itself.

---

## 2.3 Cross Encoders Introduce Significant Latency Costs

The reranker substantially improved ranking quality but introduced the largest latency penalty in the pipeline.

Measured latency:

* Dense retrieval: 16.9 ms
* Reranking pipeline: 396.9 ms

This represents approximately:

23.5× additional latency.

The experiment highlighted a common production tradeoff:

> Ranking quality and latency are often directly opposed.

Systems requiring low latency may not be able to afford aggressive reranking.

---

## 2.4 Embedding Models Were Not the Primary Bottleneck

Considerable effort was initially spent comparing multilingual embedding models.

However, practical bottlenecks emerged elsewhere:

* GPU memory constraints
* indexing efficiency
* evaluation methodology
* corpus construction

The project demonstrated that:

> Retrieval systems are often constrained more by engineering factors than by model architecture.

This was a valuable shift in perspective.

---

# 3. What I Learned About Retrieval Systems

## 3.1 Retrieval Is Primarily a Systems Problem

Before this project, retrieval appeared to be a modeling problem.

After building the full pipeline, it became clear that retrieval is largely a systems engineering challenge.

Successful retrieval requires:

* corpus construction
* chunking strategy
* embedding generation
* ANN indexing
* ranking architecture
* evaluation methodology
* deployment infrastructure

Each component influences the final outcome.

Model quality alone is insufficient.

---

## 3.2 Evaluation Design Is as Important as Model Design

The corpus mismatch incident fundamentally changed how evaluation is viewed.

A retrieval benchmark is only meaningful when:

* documents exist in the index
* relevance definitions are valid
* metrics measure achievable outcomes

Incorrect evaluation assumptions can invalidate otherwise correct experiments.

Future projects will prioritize evaluation verification earlier in the pipeline.

---

## 3.3 Latency Is a First-Class Metric

Traditional machine learning projects often focus primarily on accuracy.

Retrieval systems introduce additional objectives:

* latency
* throughput
* memory consumption
* scalability

The project demonstrated that:

A model that is slightly less accurate but substantially faster may be the better production choice.

This perspective is central to real-world retrieval engineering.

---

## 3.4 Multi-Stage Retrieval Is the Practical Architecture

The final architecture evolved into:

Query
→ Dense Retrieval
→ HNSW Search
→ Hybrid Fusion
→ Cross Encoder Reranking
→ Final Ranking

This structure mirrors modern retrieval pipelines used in search engines and Retrieval-Augmented Generation systems.

The key lesson was that no single model solves retrieval.

Performance emerges from the interaction of multiple stages.

---

# 4. Future Work

Several directions would improve the system further.

## Retrieval Improvements

* Build retrieval corpus from full dataset
* Recompute evaluation metrics with complete coverage
* Introduce hard-negative mining
* Evaluate multilingual BGE-M3 at larger scale

## Ranking Improvements

* Evaluate larger rerankers
* Compare MonoT5 and ColBERT architectures
* Investigate late-interaction retrieval

## Infrastructure Improvements

* GPU FAISS indexing
* Distributed retrieval architecture
* Online caching systems
* Query analytics and monitoring

## Research Extensions

* Retrieval-Augmented Generation
* Agentic retrieval workflows
* Multi-hop retrieval benchmarks
* Cross-lingual retrieval evaluation

---

# Final Reflection

The most important outcome of this project was not a benchmark score.

It was the realization that retrieval performance emerges from the interaction between data, indexing, ranking, infrastructure, and evaluation.

The strongest lesson learned was:

> Evaluation was dominated by corpus coverage mismatch rather than model quality.

That single discovery explained more of the observed system behavior than any architectural change.

Ultimately, the project evolved from a model comparison exercise into a study of retrieval engineering itself.

Building the full pipeline provided practical experience with the same categories of problems encountered in modern search, recommendation, and Retrieval-Augmented Generation systems: scalability, latency, ranking, evaluation, and system design.

The final result is not simply a retrieval model, but a retrieval system whose behavior, strengths, limitations, and tradeoffs are explicitly understood and documented.
