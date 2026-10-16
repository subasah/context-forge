#!/usr/bin/env python3
"""
ContextForge MCP Agent Client Simulator.
Demonstrates how an autonomous LLM Agent interacts with ContextForge
via standard Model Context Protocol (MCP) tools and resources.
"""

from rich.console import Console
from rich.panel import Panel
import asyncio

from context_forge import RAGPipeline
from context_forge.mcp_server.server import create_mcp_server

console = Console()

async def main():
    console.print(Panel.fit("[bold cyan]ContextForge: MCP (Model Context Protocol) Agent Client[/bold cyan]"))

    # 1. Instantiate Pipeline & Seed Knowledge Base
    pipeline = RAGPipeline()
    pipeline.ingest_document("data/sample_rag_architecture.md", title="RAG Architecture Spec")
    pipeline.ingest_document("data/generative_ai_systems.txt", title="AI Systems Guide")

    # 2. Spin up MCP Server instance
    mcp_server = create_mcp_server(pipeline=pipeline)
    console.print(f"[green]✔[/green] Created MCP Server: [bold]{mcp_server.name}[/bold]")

    # 3. List available MCP Tools
    tools = await mcp_server.list_tools()
    console.print(f"\n[bold]Discovered {len(tools)} MCP Tools for Agents:[/bold]")
    for t in tools:
        console.print(f"  • [cyan]{t.name}[/cyan]: {t.description}")

    # 4. Agent simulates calling `search_knowledge_base`
    console.print("\n[bold yellow]Agent Action 1: Calling 'search_knowledge_base'[/bold yellow]")
    search_query = "What is the role of Cross-Encoder rerankers in two-stage retrieval?"
    search_result = await mcp_server.call_tool(
        "search_knowledge_base",
        arguments={"query": search_query, "top_k": 2, "mode": "hybrid_rrf"},
    )
    console.print(Panel(search_result.content[0].text[:400] + "...", title="Tool Call Output: search_knowledge_base"))

    # 5. Agent simulates recording a user fact via `save_memory_fact`
    console.print("\n[bold yellow]Agent Action 2: Calling 'save_memory_fact'[/bold yellow]")
    save_result = await mcp_server.call_tool(
        "save_memory_fact",
        arguments={
            "user_id": "recruiter_alex",
            "category": "evaluation_goal",
            "key": "interview_topic",
            "value": "Candidate's deep understanding of RAG, RRF, and Token Budgeting",
        },
    )
    console.print(Panel(save_result.content[0].text, title="Tool Call Output: save_memory_fact"))

    # 6. Agent queries memory via `query_memory`
    console.print("\n[bold yellow]Agent Action 3: Calling 'query_memory'[/bold yellow]")
    query_mem_result = await mcp_server.call_tool(
        "query_memory",
        arguments={"user_id": "recruiter_alex"},
    )
    console.print(Panel(query_mem_result.content[0].text, title="Tool Call Output: query_memory"))

    console.print("\n[bold green]✔ MCP Agent Interaction Cycle Completed Successfully![/bold green]\n")

if __name__ == "__main__":
    asyncio.run(main())
