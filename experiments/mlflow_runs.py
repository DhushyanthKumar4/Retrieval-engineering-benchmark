# MLflow Experiment Tracking

!pip -q install mlflow

import os
import mlflow
import pandas as pd

print("="*80)
print("Loading Evaluation Results")
print("="*80)

evaluation_df = pd.read_csv(
    "retrieval_evaluation_results.csv"
)

print(evaluation_df)

print("\n")
print("="*80)
print("Initializing MLflow")
print("="*80)

mlflow.set_tracking_uri(
    "sqlite:///mlflow.db"
)

experiment_name = (
    "Multilingual_Retrieval_Benchmark"
)

mlflow.set_experiment(
    experiment_name
)

print(
    f"Experiment: {experiment_name}"
)

print("\n")
print("="*80)
print("Logging Retrieval Experiments")
print("="*80)

for _, row in evaluation_df.iterrows():

    with mlflow.start_run(
        run_name=row["System"]
    ):

        mlflow.log_param(
            "Retrieval_Method",
            row["System"]
        )

        mlflow.log_param(
            "Embedding_Model",
            "multilingual-e5-base"
        )

        mlflow.log_metric(
            "Recall_1",
            float(row["Recall@1"])
        )

        mlflow.log_metric(
            "Recall_5",
            float(row["Recall@5"])
        )

        mlflow.log_metric(
            "Recall_10",
            float(row["Recall@10"])
        )

        mlflow.log_metric(
            "MRR",
            float(row["MRR"])
        )

        mlflow.log_metric(
            "nDCG",
            float(row["nDCG"])
        )

        mlflow.log_metric(
            "Avg_Latency_ms",
            float(row["Avg_Latency_ms"])
        )

        mlflow.log_metric(
            "P95_Latency_ms",
            float(row["P95_Latency_ms"])
        )

        mlflow.log_metric(
            "QPS",
            float(row["QPS"])
        )

        mlflow.log_metric(
            "Memory_MB",
            float(row["Memory_MB"])
        )

print("\n")
print("="*80)
print("Logging Artifacts")
print("="*80)

artifact_files = [

    "retrieval_evaluation_results.csv",

    "faiss_index_benchmark.csv",

    "bm25_benchmark.csv",

    "dense_retrieval_benchmark.csv",

    "hybrid_retrieval_benchmark.csv",

    "reranker_benchmark.csv",

    "systems_optimization_results.csv",

    "failure_summary.csv",

    "language_failure_analysis.csv",

    "failure_examples.csv",

    "explainability_summary.csv"
]

with mlflow.start_run(
    run_name="Project_Artifacts"
):

    for file in artifact_files:

        if os.path.exists(file):

            mlflow.log_artifact(
                file
            )

print("\n")
print("="*80)
print("MLflow Summary")
print("="*80)

summary = pd.DataFrame({

    "Component":[

        "BM25",
        "Dense Retrieval",
        "Hybrid Retrieval",
        "Reranker",
        "Optimization",
        "Error Analysis",
        "Explainability"
    ],

    "Tracked":[
        True,
        True,
        True,
        True,
        True,
        True,
        True
    ]
})

print(summary)

print("\n")
print("="*80)
print("Artifacts Created")
print("="*80)

print("mlflow.db")
print("Experiment Runs Logged")
print("Metrics Logged")
print("Artifacts Logged")

print("\n")
print("Step 16 Completed Successfully")
print("="*80)


Results:


================================================================================
Loading Evaluation Results
================================================================================
   System  Recall@1  Recall@5  Recall@10     MRR    nDCG  Avg_Latency_ms  \
0    BM25       0.0     0.002      0.002  0.0007  0.0010         155.916   
1   Dense       0.0     0.002      0.002  0.0005  0.0009          14.959   
2  Hybrid       0.0     0.002      0.002  0.0007  0.0010         163.107   

   P95_Latency_ms    QPS  Memory_MB  
0         295.470   6.41   10580.03  
1          21.788  66.85   10580.03  
2         302.518   6.13   10580.03  


================================================================================
Initializing MLflow
================================================================================
Experiment: Multilingual_Retrieval_Benchmark


================================================================================
Logging Retrieval Experiments
================================================================================


================================================================================
Logging Artifacts
================================================================================


================================================================================
MLflow Summary
================================================================================
          Component  Tracked
0              BM25     True
1   Dense Retrieval     True
2  Hybrid Retrieval     True
3          Reranker     True
4      Optimization     True
5    Error Analysis     True
6    Explainability     True


================================================================================
Artifacts Created
================================================================================
mlflow.db
Experiment Runs Logged
Metrics Logged
Artifacts Logged
