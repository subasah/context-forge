#!/usr/bin/env python3
"""
ContextForge Grounding & Hallucination Audit Demo.
Demonstrates citation verification, claim-level entailment checking,
and detection of unsupported hallucinations.
"""

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from context_forge import RAGPipeline

console = Console()

def main():
    console.print(Panel.fit("[bold red]ContextForge: Grounding & Hallucination Guardrail Demo[/bold red]"))

    pipeline = RAGPipeline()
    pipeline.ingest_document("data/enterprise_policies.json", title="Security Policies")

    query = "What are the rules regarding data erasure under GDPR Article 17?"
    results = pipeline.retrieve(query=query, top_k=2)

    console.print(f"[bold yellow]Query:[/bold yellow] {query}\n")

    # Scenario A: Faithful, Grounded Answer
    grounded_answer = (
        "Under GDPR Article 17 erasure requests, the Data Protection Officer must execute "
        "cryptographic deletion across all relational tables and vector indexes within 30 calendar days [Doc 1]."
    )

    console.print("[bold green]Scenario A: Evaluating Grounded Answer[/bold green]")
    report_a = pipeline.grounding_verifier.verify_grounding(grounded_answer, results)
    console.print(f"Faithfulness Score: [bold green]{report_a.grounding_score:.1%}[/bold green]")
    console.print(f"Hallucination Detected: [bold]{report_a.hallucination_detected}[/bold]")
    console.print(f"Verified Citations: [bold]{len(report_a.citations)}[/bold]")

    # Scenario B: Hallucinated Answer (fabricated numbers and non-existent policy)
    hallucinated_answer = (
        "Under GDPR Article 17, customer records are automatically transferred to external backup "
        "drives in Switzerland within 5 minutes, and customers receive a \$500 compensation voucher [Doc 1]."
    )

    console.print("\n[bold red]Scenario B: Evaluating Hallucinated Answer[/bold red]")
    report_b = pipeline.grounding_verifier.verify_grounding(hallucinated_answer, results)
    console.print(f"Faithfulness Score: [bold red]{report_b.grounding_score:.1%}[/bold red]")
    console.print(f"Hallucination Detected: [bold red]{report_b.hallucination_detected}[/bold red]")
    console.print(f"Refusal Recommended: [bold red]{report_b.refusal_recommended}[/bold red]")

    # Table of verified claims
    table = Table(title="Claim-Level Entailment Breakdown (Scenario B)")
    table.add_column("Claim Text", style="white")
    table.add_column("Supported?", justify="center")
    table.add_column("Confidence", justify="right")
    table.add_column("Audit Reasoning", style="cyan")

    for claim in report_b.claims:
        status_style = "[bold green]YES[/bold green]" if claim.is_supported else "[bold red]NO[/bold red]"
        table.add_row(claim.claim_text, status_style, f"{claim.confidence:.2f}", claim.reasoning)

    console.print(table)

if __name__ == "__main__":
    main()
