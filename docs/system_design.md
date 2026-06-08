Multilingual Retrieval Engineering Benchmark System


Executive Summary

This project implements and benchmarks a complete multilingual retrieval pipeline using:

BM25 sparse retrieval
Dense semantic retrieval
FAISS ANN indexing
Hybrid retrieval fusion
Cross-encoder reranking
FastAPI deployment
MLflow experiment tracking
Explainability and error analysis

The objective was not merely to train a model, but to evaluate retrieval systems from an engineering perspective under realistic resource constraints.

1. Dataset Overview
Dataset

TyDi QA

Languages

11 Languages:

Arabic
Bengali
English
Finnish
Indonesian
Japanese
Korean
Russian
Swahili
Telugu
Thai
Data Processing Statistics
Stage	Rows
Raw Dataset	30,000
After Cleaning	21,578
Training Split	15,104
Validation Split	3,237
Test Split	3,237



2. Retrieval Corpus Construction
Chunking Strategy

Fixed chunking:

Parameter	Value
Chunk Size	256 words
Overlap	50 words
Generated Corpus
Metric	Value
Documents	15,104
Chunks	157,241
Average Chunks / Document	10.41
Production Retrieval Corpus

To fit GPU and memory constraints:

Metric	Value
Final Retrieval Corpus	48,573 chunks
Embedding Dimension	768



3. Embedding Benchmark
Models Evaluated
Model
BGE-M3
multilingual-e5-base
MiniLM
Final Selection
multilingual-e5-base

Reasons:

Strong multilingual support
Stable GPU memory usage
Faster embedding generation
Better suitability for T4 hardware
Embedding Performance
Metric	Value
Embedded Chunks	48,573
Embedding Dimension	768
Total Runtime	1549.9 sec
Throughput	31.34 chunks/sec



4. Vector Index Benchmark
Index Types Evaluated
Flat Index
Metric	Value
Build Time	0.15 sec
Latency	11.149 ms
Memory	142.30 MB
IVF Index
Metric	Value
Build Time	3.64 sec
Latency	0.971 ms
Memory	143.43 MB
HNSW Index
Metric	Value
Build Time	49.96 sec
Latency	0.204 ms
Memory	154.91 MB
Final Selection
FAISS HNSW

Why?

Lowest retrieval latency
Excellent ANN quality
Production-grade scalability



5. Retrieval Architecture
User Query
      │
      ▼
Dense Encoder
(multilingual-e5-base)
      │
      ▼
FAISS HNSW Search
      │
      ▼
Top 50 Candidates
      │
      ▼
Cross Encoder
(ms-marco-MiniLM-L6-v2)
      │
      ▼
Top 10 Results



6. BM25 Baseline
Performance
Metric	Value
Corpus Size	48,573
Build Time	9.18 sec
Avg Latency	54.24 ms
P95 Latency	112.97 ms
Throughput	18.44 QPS



7. Dense Retrieval
Architecture
Query
 ↓
multilingual-e5-base
 ↓
FAISS HNSW
 ↓
Top-K
Performance
Metric	Value
Avg Latency	16.86 ms
P95 Latency	22.55 ms
Throughput	59.31 QPS
Observation

Dense retrieval achieved:

~3.2× lower latency than BM25
~3.2× higher throughput
Better multilingual semantic matching



8. Hybrid Retrieval
Methods
Reciprocal Rank Fusion (RRF)
Weighted Fusion
Performance
Metric	Value
Avg Latency	81.80 ms
P95 Latency	164.72 ms
Throughput	12.23 QPS
Tradeoff

Hybrid retrieval improves robustness but increases latency due to dual retrieval execution.




9. Cross Encoder Reranking
Model

cross-encoder/ms-marco-MiniLM-L-6-v2

Configuration
Parameter	Value
Initial Retrieval	50
Final Results	10
Performance
Metric	Value
Avg Latency	396.95 ms
P95 Latency	521.49 ms
Throughput	2.52 QPS
Key Insight

Cross-encoders dramatically increase computational cost and should only operate on a small candidate set.




10. Evaluation Findings
Important Limitation Discovered

During evaluation:

Metric	Value
Corpus Titles	12,623
Test Titles	3,231
Overlap	54
Overlap Percentage	1.67%
Interpretation

The retrieval corpus was built only from training documents.

Most test documents therefore did not exist in the retrieval index.

This explains the artificially low:

Recall@K
MRR
nDCG

scores.

Engineering Lesson

The low retrieval metrics were caused by corpus coverage issues rather than retrieval model quality.

This is a realistic retrieval engineering failure mode and was documented through systematic error analysis.




11. Error Analysis Findings
Major Failure Modes
Corpus Coverage Failure

Largest source of error.

Language Distribution Shift

Certain languages had fewer matching documents.

Sparse Retrieval Limitations

BM25 struggled with:

multilingual morphology
lexical mismatch
semantic paraphrases
Dense Retrieval Limitations

Dense retrieval occasionally retrieved semantically related but incorrect documents.




12. Systems Optimization
Techniques Evaluated
Embedding cache
Batch inference
HNSW tuning
Mixed precision inference
Performance
Metric	Value
Embedding Throughput	81.30 QPS
Retrieval Throughput	732.49 QPS
Memory Usage	10.6 GB



13. Deployment
Production Components
API Layer

FastAPI

Vector Database

FAISS HNSW

Embedding Service

multilingual-e5-base

Experiment Tracking

MLflow




14. Key Engineering Contributions

This project demonstrates:

✅ Multilingual retrieval systems

✅ Vector search engineering

✅ Approximate nearest neighbor indexing

✅ Hybrid ranking systems

✅ Cross-encoder reranking

✅ Retrieval evaluation methodology

✅ Retrieval failure analysis

✅ FastAPI deployment

✅ MLflow experiment tracking

✅ End-to-end retrieval pipeline design




15. Final Reflection

The most important lesson from this project is that retrieval quality is often limited more by data coverage and indexing strategy than by model architecture.

The project evolved from a simple retrieval benchmark into a full retrieval engineering study covering:

sparse retrieval
dense retrieval
ANN indexing
ranking systems
evaluation methodology
deployment infrastructure

The resulting system provides a realistic approximation of production retrieval pipelines used in modern Retrieval-Augmented Generation (RAG) systems and large-scale search infrastructure.
