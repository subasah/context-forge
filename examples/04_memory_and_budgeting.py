#!/usr/bin/env python3
"""
ContextForge Memory & Token Budgeting Demo.
Demonstrates Conversational State, Persistent Episodic Memory,
Token Allocation, and Lost-in-the-Middle Mitigation.
"""

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from context_forge import RAGPipeline, PipelineConfig

console = Console()

def main():
    console.print(Panel.fit("[bold green]ContextForge: Conversational Memory & Token Budgeting[/bold green]"))

    config = PipelineConfig()
    pipeline = RAGPipeline(config=config)
    pipeline.ingest_document("data/generative_ai_systems.txt", title="AI Systems Guide")

    user_id = "user_engineer_42"

    # 1. Seed Persistent Episodic Memory
    console.print("\n[bold]1. Storing Persistent User Profile & Entities:[/bold]")
    pipeline.persistent_memory.set(user_id, "profile", "role", "Principal AI Architect")
    pipeline.persistent_memory.set(user_id, "preference", "latency_target", "p95 < 200ms")
    pipeline.persistent_memory.set(user_id, "preference", "output_format", "concise markdown with bullet points")
    console.print("[green]✔[/green] Saved 3 persistent facts to SQLite store.")

    # 2. Simulate Dialogue Buffer Turns
    console.print("\n[bold]2. Simulating Multi-Turn Conversational Buffer:[/bold]")
    pipeline.buffer_memory.add_turn("user", "Hello, I am architecting an enterprise RAG service.")
    pipeline.buffer_memory.add_turn("assistant", "Greetings! I am ready to assist with retrieval, chunking, and context engineering.")
    pipeline.buffer_memory.add_turn("user", "What are the recommended token budgeting boundaries?")
    console.print(f"[green]✔[/green] Dialogue turns in buffer: [bold]{len(pipeline.buffer_memory.history)}[/bold]")

    # 3. Retrieve and Assemble Context with Token Budget Allocation
    query = "Explain the token bucket allocation breakdown for production agents."
    results = pipeline.retrieve(query=query, top_k=3)
    assembled = pipeline.assemble_context(query=query, retrieved_chunks=results, user_id=user_id)

    # 4. Display Token Allocation Table
    alloc = assembled.token_allocation
    alloc_table = Table(title="Context Budget Allocation Breakdown (Tokens)")
    alloc_table.add_column("Context Tier", style="cyan")
    alloc_table.add_column("Tokens Used", justify="right", style="green")
    alloc_table.add_column("Percentage", justify="right", style="yellow")

    total = max(alloc.total_tokens, 1)
    alloc_table.add_row("System Prompt", str(alloc.system_prompt_tokens), f"{alloc.system_prompt_tokens/total:.1%}")
    alloc_table.add_row("Instructions & Constraints", str(alloc.instruction_tokens), f"{alloc.instruction_tokens/total:.1%}")
    alloc_table.add_row("User Memory & Dialogue History", str(alloc.memory_history_tokens), f"{alloc.memory_history_tokens/total:.1%}")
    alloc_table.add_row("Retrieved Knowledge Context", str(alloc.retrieved_context_tokens), f"{alloc.retrieved_context_tokens/total:.1%}")
    alloc_table.add_row("Reserved Generation Margin", str(alloc.reserved_generation_tokens), f"{alloc.reserved_generation_tokens/alloc.budget_limit:.1%}")
    alloc_table.add_row("[bold]Total Consumed / Budget[/bold]", f"[bold]{alloc.total_tokens} / {alloc.budget_limit}[/bold]", "[bold]100%[/bold]")

    console.print(alloc_table)
    console.print(f"Lost-in-the-Middle Attention Mitigation Applied: [bold green]{assembled.lost_in_the_middle_applied}[/bold green]\n")

    # 5. Display Assembled Prompt Preview
    console.print(Panel(assembled.assembled_prompt[:600] + "\n\n... [TRUNCATED] ...", title="[bold cyan]Assembled Prompt Payload[/bold cyan]"))

if __name__ == "__main__":
    main()
