#!/usr/bin/env python3
"""
ContextForge 30-Second Quickstart Demo.
Demonstrates end-to-end RAG: Ingestion -> Hybrid RRF -> Reranking -> Grounding Audit.
"""

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from context_forge import RAGPipeline, PipelineConfig

console = Console()

def main():
    console.print(Panel.fit("[bold cyan]ContextForge End-to-End RAG Quickstart[/bold cyan]", border_style="cyan"))

    # 1. Initialize Pipeline
    config = PipelineConfig()
    pipeline = RAGPipeline(config=config)
    console.print("[green]✔[/green] Initialized ContextForge Pipeline with Hybrid RRF & Grounding Guardrails.")

    # 2. Ingest Sample Documents
    corpus_path = "data/sample_rag_architecture.md"
    chunks = pipeline.ingest_document(corpus_path, title="RAG Architecture Spec", chunk_strategy="recursive")
    console.print(f"[green]✔[/green] Ingested '{corpus_path}' -> generated [bold]{len(chunks)}[/bold] chunks.")

    # 3. Query the Knowledge Base
    query = "How does Reciprocal Rank Fusion combine dense and sparse rankings?"
    console.print(f"\n[yellow]Query:[/yellow] [bold]{query}[/bold]\n")

    answer, grounding_report, assembled_ctx, trace = pipeline.query(query=query)

    # 4. Display Retrieved Chunks
    retrieval_table = Table(title="Retrieved & Reranked Context Chunks")
    retrieval_table.add_column("Rank", justify="center", style="cyan")
    retrieval_table.add_column("Score", justify="right", style="magenta")
    retrieval_table.add_column("Mode", justify="center", style="green")
    retrieval_table.add_column("Preview", style="white")

    for res in assembled_ctx.retrieved_chunks:
        preview = res.chunk.text.replace("\n", " ")[:90] + "..."
        retrieval_table.add_row(str(res.rank), f"{res.score:.4f}", res.retrieval_mode, preview)

    console.print(retrieval_table)

    # 5. Display Answer
    console.print(Panel(f"[bold white]{answer}[/bold white]", title="[bold green]Synthesized & Grounded Answer[/bold green]", border_style="green"))

    # 6. Display Grounding & Faithfulness Audit
    console.print("\n[bold]Grounding & Hallucination Audit:[/bold]")
    console.print(f"- Grounding Faithfulness Score: [bold green]{grounding_report.grounding_score:.1%}[/bold green]")
    console.print(f"- Citation Coverage: [bold]{grounding_report.citation_coverage:.1%}[/bold]")
    console.print(f"- Hallucination Detected: [bold]{'YES' if grounding_report.hallucination_detected else 'NO'}[/bold]")
    console.print(f"- Verified Claims Count: [bold]{len(grounding_report.claims)}[/bold]")

    # 7. Execution Trace Summary
    console.print("\n[bold]Microsecond Execution Trace:[/bold]")
    for step in trace.steps:
        console.print(f"  • {step.name:35} [cyan]{step.duration_ms:>6.2f} ms[/cyan] (Inputs: {step.input_count}, Outputs: {step.output_count})")
    console.print(f"  [bold]Total End-to-End Latency:[/bold] [bold yellow]{trace.total_duration_ms:.2f} ms[/bold yellow]\n")

if __name__ == "__main__":
    main()
