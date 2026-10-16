"""Tests for Model Context Protocol (MCP) server endpoints."""

import pytest
import json
from context_forge import RAGPipeline
from context_forge.mcp_server.server import create_mcp_server


@pytest.mark.asyncio
async def test_mcp_tools():
    pipeline = RAGPipeline()
    pipeline.ingest_document(
        "Redis cluster requires at least 3 master nodes and 3 replica nodes for quorum.",
        title="Redis Architecture",
    )

    server = create_mcp_server(pipeline=pipeline)

    # 1. Test search_knowledge_base
    res = await server.call_tool("search_knowledge_base", {"query": "how many nodes for Redis quorum?", "top_k": 1})
    content = res.content[0].text
    assert "Redis" in content
    assert "quorum" in content

    # 2. Test memory tools
    await server.call_tool(
        "save_memory_fact",
        {"user_id": "test_agent", "category": "preference", "key": "tone", "value": "direct"},
    )
    mem_res = await server.call_tool("query_memory", {"user_id": "test_agent"})
    assert "tone" in mem_res.content[0].text
    assert "direct" in mem_res.content[0].text
