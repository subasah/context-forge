#!/usr/bin/env python3
"""
ContextForge Parent-Child Chunking Demo.
Demonstrates decoupling the retrieval representation (small child chunks)
from generation context (larger parent chunks).
"""

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from context_forge import RAGPipeline, PipelineConfig

console = Console()

def main():
    console.print(Panel.fit("[bold blue]ContextForge: Parent-Child Hierarchical Context Demo[/bold blue]"))

    config = PipelineConfig()
    config.chunking.strategy = "parent_child"
    pipeline = RAGPipeline(config=config)

    # Ingest document with parent-child strategy
    chunks = pipeline.ingest_document("data/sample_rag_architecture.md", chunk_strategy="parent_child")
    console.print(f"[green]✔[/green] Generated [bold]{len(chunks)}[/bold] child chunks with linked parent context windows.")

    query = "What is the mathematical formulation of Reciprocal Rank Fusion?"
    console.print(f"\n[yellow]Query:[/yellow] [bold]{query}[/bold]\n")

    results = pipeline.retrieve(query=query, top_k=2)

    for idx, res in enumerate(results, start=1):
        console.print(f"[bold cyan]Result #{idx}:[/bold cyan] Child ID: {res.chunk.id} (Tokens: {res.chunk.token_count})")
        console.print(Panel(res.chunk.text, title="[yellow]Matched Child Chunk (Retrieval Unit)[/yellow]"))

        if res.chunk.parent_text:
            console.print(Panel(res.chunk.parent_text, title=f"[green]Unpacked Parent Chunk Context ({res.chunk.parent_id})[/green]"))
        console.print("-" * 60)

if __name__ == "__main__":
    main()
