"""
ContextForge Model Context Protocol (MCP) Server.
Standard MCP 2.x server exposing RAG tools and resources
for agentic context-on-demand delivery.
"""

from __future__ import annotations
import json
import logging
from typing import Any, Dict, List, Optional
from mcp.server.mcpserver import MCPServer

from context_forge.core.document import Document, Chunk, RetrievalResult
from context_forge.pipeline import RAGPipeline
from context_forge.memory.persistent_memory import PersistentMemoryStore

logger = logging.getLogger(__name__)


def create_mcp_server(
    pipeline: Optional[RAGPipeline] = None,
    memory_store: Optional[PersistentMemoryStore] = None,
    server_name: str = "context-forge-mcp",
) -> MCPServer:
    """Factory creating a fully configured MCP Server."""
    server = MCPServer(
        name=server_name,
        instructions=(
            "ContextForge Model Context Protocol Server. "
            "Enables intelligent agents to dynamically pull contextual knowledge, "
            "inspect document chunks, audit grounding faithfulness, and query persistent memory."
        ),
    )

    _pipeline = pipeline or RAGPipeline()
    _memory = memory_store or PersistentMemoryStore()

    # --- TOOLS ---

    @server.tool(name="search_knowledge_base", description="Search indexed documents using Hybrid RAG (Dense + BM25 + Rerank).")
    def search_knowledge_base(
        query: str,
        top_k: int = 5,
        mode: str = "hybrid_rrf",
    ) -> str:
        """Search knowledge base and return ranked contextual chunks."""
        results = _pipeline.retrieve(query=query, top_k=top_k, mode=mode)
        if not results:
            return "No matching context found for query."

        output_lines = [f"Found {len(results)} relevant chunks for query: '{query}':\n"]
        for idx, res in enumerate(results, start=1):
            doc_title = res.chunk.metadata.get("title", "Document")
            output_lines.append(
                f"### [Chunk #{idx}] (ID: {res.chunk.id}) | Score: {res.score:.4f} | Mode: {res.retrieval_mode}\n"
                f"**Source**: {doc_title}\n"
                f"**Text**: {res.chunk.text}\n"
            )
        return "\n".join(output_lines)

    @server.tool(name="fetch_chunk_details", description="Retrieve full text, parent chunk, and section metadata for a chunk ID.")
    def fetch_chunk_details(chunk_id: str) -> str:
        """Retrieve full details of a specific chunk."""
        for c in _pipeline.chunks:
            if c.id == chunk_id:
                details = {
                    "id": c.id,
                    "document_id": c.document_id,
                    "title": c.metadata.get("title", "Unknown"),
                    "text": c.text,
                    "enriched_text": c.enriched_text,
                    "contextual_header": c.contextual_header,
                    "parent_id": c.parent_id,
                    "parent_text": c.parent_text,
                    "token_count": c.token_count,
                    "metadata": c.metadata,
                }
                return json.dumps(details, indent=2)
        return f"Chunk ID '{chunk_id}' not found."

    @server.tool(name="query_memory", description="Query persistent user memory for preferences, entities, or rules.")
    def query_memory(user_id: str, keyword: str = "") -> str:
        """Search persistent episodic memory for a specific user."""
        if keyword:
            items = _memory.search(user_id, keyword)
        else:
            items = _memory.get_all_for_user(user_id)

        if not items:
            return f"No memory items found for user '{user_id}' with keyword '{keyword}'."

        lines = [f"Memory items for user '{user_id}':"]
        for it in items:
            lines.append(f"- [{it.category.upper()}] {it.key}: {it.value} (confidence: {it.confidence})")
        return "\n".join(lines)

    @server.tool(name="save_memory_fact", description="Save or update a persistent fact or preference for a user.")
    def save_memory_fact(user_id: str, category: str, key: str, value: str) -> str:
        """Store a fact in persistent memory."""
        _memory.set(user_id=user_id, category=category, key=key, value=value)
        return f"Successfully saved memory fact for user '{user_id}': [{category}] {key} = {value}"

    @server.tool(name="verify_grounding", description="Verify if a candidate response claim is supported by retrieved context.")
    def verify_grounding(claim: str, chunk_ids: List[str]) -> str:
        """Verify claim faithfulness against specified chunk IDs."""
        target_chunks: List[RetrievalResult] = []
        for cid in chunk_ids:
            for c in _pipeline.chunks:
                if c.id == cid:
                    target_chunks.append(RetrievalResult(chunk=c, score=1.0))
                    break

        if not target_chunks:
            return "Error: None of the specified chunk IDs were found in the knowledge base."

        report = _pipeline.grounding_verifier.verify_grounding(
            generated_answer=claim,
            retrieved_results=target_chunks,
        )

        result_summary = {
            "claim": claim,
            "is_fully_grounded": report.is_fully_grounded,
            "grounding_score": report.grounding_score,
            "hallucination_detected": report.hallucination_detected,
            "verified_claims": [
                {
                    "text": c.claim_text,
                    "supported": c.is_supported,
                    "confidence": c.confidence,
                    "reasoning": c.reasoning,
                }
                for c in report.claims
            ],
        }
        return json.dumps(result_summary, indent=2)

    # --- RESOURCES ---

    @server.resource(uri="rag://documents/{doc_id}", name="Document Content")
    def read_document_resource(doc_id: str) -> str:
        """Returns document content by ID."""
        for doc in _pipeline.documents:
            if doc.id == doc_id:
                return f"# {doc.title}\n\n{doc.text}"
        return f"Document '{doc_id}' not found."

    @server.resource(uri="memory://users/{user_id}/profile", name="User Profile")
    def read_user_profile_resource(user_id: str) -> str:
        """Returns formatted user profile memory."""
        profile = _memory.format_for_context(user_id)
        return profile or f"No profile records for user '{user_id}'."

    return server
