#!/usr/bin/env python3
"""
ContextForge Hybrid Retrieval Deep-Dive.
Compares Dense vs Sparse (BM25) vs Hybrid RRF on conceptual vs exact keyword queries.
"""

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from context_forge import RAGPipeline, PipelineConfig

console = Console()

def main():
    console.print(Panel.fit("[bold magenta]ContextForge: Dense vs BM25 vs Hybrid RRF Comparative Study[/bold magenta]"))

    pipeline = RAGPipeline()

    # Ingest multiple sample corpora
    pipeline.ingest_document("data/sample_rag_architecture.md", title="RAG Architecture")
    pipeline.ingest_document("data/enterprise_policies.json", title="Enterprise Compliance Policies")

    queries = [
        ("Conceptual Query", "What architectural techniques reduce memory overhead and context degradation in transformers?"),
        ("Exact Code / Keyword Query", "What is the GDPR compliance timeline specified in SEC-POL-002?"),
    ]

    for q_type, q in queries:
        console.print(f"\n[bold yellow]Scenario: {q_type}[/bold yellow]")
        console.print(f"Query: [cyan]{q}[/cyan]\n")

        dense_res = pipeline.retrieve(query=q, top_k=3, mode="dense", use_reranker=False)
        sparse_res = pipeline.retrieve(query=q, top_k=3, mode="sparse", use_reranker=False)
        hybrid_res = pipeline.retrieve(query=q, top_k=3, mode="hybrid_rrf", use_reranker=False)

        table = Table(title=f"Rank #1 Comparison for '{q_type}'")
        table.add_column("Retrieval Strategy", style="bold")
        table.add_column("Rank 1 Score", justify="right")
        table.add_column("Rank 1 Source", style="cyan")
        table.add_column("Content Snippet", style="white")

        if dense_res:
            table.add_row(
                "Dense Vector",
                f"{dense_res[0].score:.4f}",
                dense_res[0].chunk.metadata.get("title", ""),
                dense_res[0].chunk.text.replace("\n", " ")[:80] + "...",
            )
        if sparse_res:
            table.add_row(
                "Sparse (BM25)",
                f"{sparse_res[0].score:.4f}",
                sparse_res[0].chunk.metadata.get("title", ""),
                sparse_res[0].chunk.text.replace("\n", " ")[:80] + "...",
            )
        if hybrid_res:
            table.add_row(
                "Hybrid RRF",
                f"{hybrid_res[0].score:.4f}",
                hybrid_res[0].chunk.metadata.get("title", ""),
                hybrid_res[0].chunk.text.replace("\n", " ")[:80] + "...",
            )

        console.print(table)

if __name__ == "__main__":
    main()
