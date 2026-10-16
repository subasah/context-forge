"""
ContextForge Comparative Retrieval Benchmark.
Evaluates Dense, Sparse (BM25), Hybrid RRF, and Reranked pipelines
on Hit Rate @ K, MRR @ K, and Latency.
"""

from typing import Any, Dict, List, Tuple
import time
import pandas as pd
import numpy as np

from context_forge.pipeline import RAGPipeline


class RetrievalBenchmarkHarness:
    """Automated benchmark testbed for comparative retrieval evaluation."""

    def __init__(self, pipeline: RAGPipeline):
        self.pipeline = pipeline

    def run_benchmark(
        self,
        eval_dataset: List[Dict[str, Any]],  # List of {"query": str, "target_doc_id": str}
        top_k: int = 3,
    ) -> pd.DataFrame:
        """
        Runs evaluation queries against:
        1. Dense Retrieval
        2. Sparse (BM25)
        3. Hybrid RRF
        4. Hybrid RRF + Cross-Encoder Rerank
        """
        modes = [
            ("Dense", "dense", False),
            ("Sparse (BM25)", "sparse", False),
            ("Hybrid (RRF)", "hybrid_rrf", False),
            ("Hybrid + Rerank", "hybrid_rrf", True),
        ]

        summary_rows = []

        for label, mode, use_rerank in modes:
            hits = 0
            reciprocal_ranks: List[float] = []
            latencies_ms: List[float] = []

            for item in eval_dataset:
                query = item["query"]
                target_doc = item["target_doc_id"]

                t0 = time.perf_counter()
                results = self.pipeline.retrieve(
                    query=query,
                    top_k=top_k,
                    mode=mode,
                    use_reranker=use_rerank,
                )
                duration_ms = (time.perf_counter() - t0) * 1000.0
                latencies_ms.append(duration_ms)

                # Check if target_doc is in retrieved chunks
                found_rank = 0
                for rank, res in enumerate(results, start=1):
                    if res.chunk.document_id == target_doc or target_doc in res.chunk.metadata.get("title", ""):
                        found_rank = rank
                        break

                if found_rank > 0:
                    hits += 1
                    reciprocal_ranks.append(1.0 / found_rank)
                else:
                    reciprocal_ranks.append(0.0)

            total_q = len(eval_dataset)
            hit_rate = hits / total_q if total_q > 0 else 0.0
            mrr = float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0
            avg_latency = float(np.mean(latencies_ms)) if latencies_ms else 0.0

            summary_rows.append({
                "Strategy": label,
                f"Hit Rate @ {top_k}": f"{hit_rate:.1%}",
                f"MRR @ {top_k}": round(mrr, 3),
                "Avg Latency (ms)": round(avg_latency, 2),
                "Total Queries": total_q,
            })

        return pd.DataFrame(summary_rows)
