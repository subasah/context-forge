"""Tests for Memory Management, Token Budgeting, and Context Assembly."""

import pytest
from context_forge.core.document import Chunk, RetrievalResult
from context_forge.memory.buffer_memory import BufferMemory
from context_forge.memory.summary_memory import SummaryMemory
from context_forge.memory.persistent_memory import PersistentMemoryStore
from context_forge.memory.token_budgeter import ContextAssembler, LostInTheMiddleSorter


def test_buffer_memory():
    mem = BufferMemory(window_size=3)
    mem.add_turn("user", "Turn 1")
    mem.add_turn("assistant", "Response 1")
    mem.add_turn("user", "Turn 2")
    mem.add_turn("assistant", "Response 2")

    turns = mem.get_recent_turns()
    assert len(turns) == 3
    assert turns[-1].content == "Response 2"
    assert mem.total_tokens() > 0


def test_summary_memory():
    summary_mem = SummaryMemory()
    buf = BufferMemory()
    buf.add_turn("user", "I need to configure semantic caching with Redis.")
    buf.add_turn("assistant", "Redis semantic cache should use cosine threshold 0.85.")

    summary = summary_mem.update_summary(buf.history)
    assert len(summary) > 0
    assert "User inquired" in summary


def test_persistent_memory_store():
    store = PersistentMemoryStore(db_path=":memory:")
    user_id = "test_user_01"

    store.set(user_id, "preference", "theme", "dark_mode")
    store.set(user_id, "entity", "project_name", "ContextForge")

    item = store.get(user_id, "preference", "theme")
    assert item is not None
    assert item.value == "dark_mode"

    search_results = store.search(user_id, "ContextForge")
    assert len(search_results) == 1
    assert search_results[0].key == "project_name"

    formatted = store.format_for_context(user_id)
    assert "dark_mode" in formatted
    assert "ContextForge" in formatted


def test_lost_in_the_middle_sorter():
    chunks = [
        RetrievalResult(chunk=Chunk(document_id="d", text=f"Doc {i}", chunk_index=i), score=1.0 - (i * 0.1), rank=i)
        for i in range(1, 6)
    ]
    reordered = LostInTheMiddleSorter.reorder(chunks)
    assert len(reordered) == 5
    # Rank 1 should be first
    assert reordered[0].rank == 1
    # Rank 2 should be last
    assert reordered[-1].rank == 2


def test_context_assembler():
    assembler = ContextAssembler(max_context_budget=1000, reserved_generation_tokens=200)
    chunks = [
        RetrievalResult(chunk=Chunk(document_id="d", text="Chunk A with useful info.", chunk_index=0), score=0.9, rank=1),
        RetrievalResult(chunk=Chunk(document_id="d", text="Chunk B with second info.", chunk_index=1), score=0.8, rank=2),
    ]

    ctx = assembler.assemble(
        query="What is in the chunks?",
        system_prompt="You are a helpful assistant.",
        retrieved_chunks=chunks,
        dialogue_history="USER: Hi\nASSISTANT: Hello",
        persistent_memory_context="- Preference: Concise",
    )

    assert ctx.token_allocation.total_tokens <= 1000
    assert "<source id=\"Doc 1\"" in ctx.assembled_prompt
    assert "USER QUERY" in ctx.assembled_prompt
