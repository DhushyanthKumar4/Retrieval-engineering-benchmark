# Retrieval-engineering-benchmark
Production-oriented retrieval benchmarking system evaluating sparse, dense, and hybrid retrieval architectures with vector indexing, reranking, multilingual search, and latency optimization.

<img width="4260" height="2523" alt="image" src="https://github.com/user-attachments/assets/6121fec7-30bd-442e-85e2-771d0c0436d0" />

Systems Pipeline:
Query → Retrieval → (BM25 / Dense / Hybrid) → Reranker → Final Results


Key results table:
	System	Recall@1	Recall@5	Recall@10	MRR		nDCG	Avg_Latency_ms	P95_Latency_ms	QPS		Memory_MB
0	BM25	0.0			0.002		0.002		0.0007	0.0010	155.916			295.470			6.41	10580.03
1	Dense	0.0			0.002		0.002		0.0005	0.0009	14.959			21.788			66.85	10580.03
2	Hybrid	0.0			0.002		0.002		0.0007	0.0010	163.107			302.518			6.13	10580.03


Key insights:
This project investigated retrieval as a complete systems problem spanning:
•	Sparse retrieval 
•	Dense retrieval 
•	Hybrid search 
•	Cross-encoder reranking 
•	Vector indexing 
•	Latency optimization 
•	Experiment tracking 
•	Evaluation methodology 
•	Retrieval diagnostics 

A central outcome of the work was demonstrating that retrieval performance depends not only on ranking architectures, but also on corpus construction, indexing strategy, benchmark design, and system-level constraints.

Rather than focusing exclusively on model performance, the project emphasized reproducible benchmarking, engineering tradeoffs, and evaluation rigor under realistic resource limitations.

The resulting system demonstrates practical experience with retrieval engineering, vector search infrastructure, ranking systems, experimental design, and production-oriented AI system development - skills that directly transfer to modern search, recommendation, and retrieval-augmented AI platforms.
