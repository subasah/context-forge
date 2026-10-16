"""
ContextForge Unified RAG Pipeline Orchestrator.
End-to-end framework integrating Ingestion, Chunking, Dense & Sparse Indexing,
Hybrid RRF Retrieval, Neural Reranking, Contextual Compression, Token Budgeting,
and Grounding Hallucination Verification.
"""

from __future__ import annotations
import uuid
import time
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
from pathlib import Path

from context_forge.core.document import (
    Document,
    Chunk,
    RetrievalResult,
    AssembledContext,
    GroundingReport,
)
from context_forge.core.config import PipelineConfig
from context_forge.core.trace import ExecutionTrace, TraceContext

from context_forge.ingestion.parser import DocumentParser
from context_forge.ingestion.cleaner import TextCleaner

from context_forge.chunking.base import BaseChunker
from context_forge.chunking.fixed_window import FixedWindowChunker
from context_forge.chunking.recursive import RecursiveCharacterChunker
from context_forge.chunking.semantic import SemanticChunker
from context_forge.chunking.parent_child import ParentChildChunker
from context_forge.chunking.contextual import ContextualChunker

from context_forge.embeddings.base import BaseEmbeddingProvider
from context_forge.embeddings.dense import UnifiedDenseEmbeddingProvider
from context_forge.embeddings.sparse import BM25Indexer

from context_forge.retrieval.dense_retriever import DenseRetriever
from context_forge.retrieval.sparse_retriever import SparseRetriever
from context_forge.retrieval.hybrid_retriever import HybridRetriever
from context_forge.retrieval.query_expansion import QueryExpander, HyDEGenerator
from context_forge.retrieval.reranker import UnifiedReranker

from context_forge.compression.compressor import ContextualCompressor
from context_forge.grounding.verifier import GroundingVerifier
from context_forge.grounding.guardrail import RetrievalGuardrail
from context_forge.memory.buffer_memory import BufferMemory
from context_forge.memory.summary_memory import SummaryMemory
from context_forge.memory.persistent_memory import PersistentMemoryStore
from context_forge.memory.token_budgeter import ContextAssembler


class RAGPipeline:
    """Production-grade unified RAG and Context Engineering engine."""

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        embedding_provider: Optional[BaseEmbeddingProvider] = None,
    ):
        self.config = config or PipelineConfig()
        self.documents: List[Document] = []
        self.chunks: List[Chunk] = []

        # 1. Embeddings
        self.embedding_provider = embedding_provider or UnifiedDenseEmbeddingProvider(
            model_name=self.config.embedding_model,
            fallback_to_lsa=self.config.fallback_to_tfidf,
        )

        # 2. Retrievers
        self.dense_retriever = DenseRetriever(self.embedding_provider)
        self.sparse_retriever = SparseRetriever(
            k1=self.config.retrieval.bm25_k1,
            b=self.config.retrieval.bm25_b,
        )
        self.hybrid_retriever = HybridRetriever(
            dense_retriever=self.dense_retriever,
            sparse_retriever=self.sparse_retriever,
            fusion_mode="rrf" if self.config.retrieval.retrieval_mode == "hybrid_rrf" else "convex",
            rrf_k=self.config.retrieval.rrf_k,
            alpha=self.config.retrieval.alpha,
        )

        # 3. Query Expansion & HyDE
        self.query_expander = QueryExpander()
        self.hyde_generator = HyDEGenerator()

        # 4. Reranker
        self.reranker = UnifiedReranker(
            model_name=self.config.reranking.model_name,
            prefer_neural=self.config.reranking.use_neural_reranker,
        )

        # 5. Compression
        self.compressor = ContextualCompressor(
            embedding_provider=self.embedding_provider,
            similarity_threshold=self.config.compression.similarity_threshold,
        )

        # 6. Memory & Context Budgeting
        self.buffer_memory = BufferMemory(
            window_size=self.config.memory.short_term_window_turns
        )
        self.summary_memory = SummaryMemory()
        self.persistent_memory = PersistentMemoryStore()
        self.context_assembler = ContextAssembler(
            max_context_budget=self.config.memory.max_context_budget,
            reserved_generation_tokens=self.config.memory.reserved_generation_budget,
            mitigate_lost_in_the_middle=self.config.memory.mitigate_lost_in_the_middle,
        )

        # 7. Grounding & Guardrails
        self.grounding_verifier = GroundingVerifier(
            embedding_provider=self.embedding_provider,
            min_faithfulness_score=self.config.grounding.min_faithfulness_score,
        )
        self.guardrail = RetrievalGuardrail(self.config.grounding)

    def _get_chunker(self, strategy: Optional[str] = None) -> BaseChunker:
        strat = strategy or self.config.chunking.strategy
        if strat == "fixed":
            return FixedWindowChunker(
                chunk_size=self.config.chunking.chunk_size,
                chunk_overlap=self.config.chunking.chunk_overlap,
            )
        elif strat == "semantic":
            return SemanticChunker(
                embed_fn=self.embedding_provider.embed_documents,
                breakpoint_percentile_threshold=self.config.chunking.semantic_similarity_threshold * 100,
                max_chunk_size=self.config.chunking.chunk_size * 2,
            )
        elif strat == "parent_child":
            return ParentChildChunker(
                parent_chunk_size=self.config.chunking.parent_chunk_size,
                child_chunk_size=self.config.chunking.child_chunk_size,
            )
        elif strat == "contextual":
            return ContextualChunker(
                base_chunker=RecursiveCharacterChunker(
                    chunk_size=self.config.chunking.chunk_size,
                    chunk_overlap=self.config.chunking.chunk_overlap,
                )
            )
        else:
            return RecursiveCharacterChunker(
                chunk_size=self.config.chunking.chunk_size,
                chunk_overlap=self.config.chunking.chunk_overlap,
            )

    def ingest_document(
        self,
        source: Union[str, Path, Document],
        title: Optional[str] = None,
        chunk_strategy: Optional[str] = None,
    ) -> List[Chunk]:
        """Ingests and chunks a single document, updating indexes."""
        if isinstance(source, Document):
            doc = source
        elif isinstance(source, Path) or (
            isinstance(source, str) and "\n" not in source and len(source) < 256 and Path(source).exists()
        ):
            doc = DocumentParser.from_file(source)
        else:
            doc = DocumentParser.from_text(str(source), title=title or "Ingested Document")

        self.documents.append(doc)
        chunker = self._get_chunker(chunk_strategy)
        new_chunks = chunker.chunk(doc)

        self.chunks.extend(new_chunks)
        # Update search indexes
        self.hybrid_retriever.index(self.chunks)
        return new_chunks

    def ingest_corpus(
        self,
        documents: List[Union[str, Document, Path]],
        chunk_strategy: Optional[str] = None,
    ) -> int:
        """Ingests multiple documents in batch."""
        total_chunks = 0
        for doc in documents:
            chunks = self.ingest_document(doc, chunk_strategy=chunk_strategy)
            total_chunks += len(chunks)
        return total_chunks

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        mode: Optional[str] = None,
        use_reranker: Optional[bool] = None,
        use_compression: Optional[bool] = None,
        trace: Optional[ExecutionTrace] = None,
    ) -> List[RetrievalResult]:
        """
        Executes end-to-end multi-stage retrieval:
        1. Query Expansion / HyDE (optional)
        2. First-stage Candidate Retrieval (Dense / Sparse / Hybrid RRF / Hybrid Convex)
        3. Second-stage Neural Cross-Encoder Reranking (optional)
        4. Third-stage Contextual Compression & MMR Deduplication (optional)
        """
        k = top_k or self.config.retrieval.top_k
        retrieval_mode = mode or self.config.retrieval.retrieval_mode
        should_rerank = use_reranker if use_reranker is not None else self.config.reranking.enabled
        should_compress = use_compression if use_compression is not None else self.config.compression.enabled

        active_query = query

        # Stage 1: Query Expansion / HyDE
        if self.config.retrieval.enable_hyde:
            if trace:
                with TraceContext(trace, "HyDE Generation", input_count=1) as tc:
                    active_query = self.hyde_generator.generate_hypothetical_document(query)
                    tc.set_output(1, hypothetical_doc_len=len(active_query))
            else:
                active_query = self.hyde_generator.generate_hypothetical_document(query)

        # Stage 2: First-stage retrieval
        candidate_pool_size = max(k * 3, 10) if should_rerank else k

        if trace:
            with TraceContext(trace, f"First-Stage Retrieval ({retrieval_mode})", input_count=len(self.chunks)) as tc:
                candidates = self._execute_first_stage(active_query, candidate_pool_size, retrieval_mode)
                tc.set_output(len(candidates), top_score=candidates[0].score if candidates else 0.0)
        else:
            candidates = self._execute_first_stage(active_query, candidate_pool_size, retrieval_mode)

        if not candidates:
            return []

        # Stage 3: Reranking
        if should_rerank:
            if trace:
                with TraceContext(trace, "Neural Cross-Encoder Reranking", input_count=len(candidates)) as tc:
                    candidates = self.reranker.rerank(
                        query=query,  # Rerank using original user intent
                        candidates=candidates,
                        top_n=k,
                    )
                    tc.set_output(len(candidates), top_rerank_score=candidates[0].score if candidates else 0.0)
            else:
                candidates = self.reranker.rerank(query=query, candidates=candidates, top_n=k)
        else:
            candidates = candidates[:k]

        # Stage 4: Contextual Compression & MMR
        if should_compress:
            if trace:
                with TraceContext(trace, "Contextual Compression & MMR", input_count=len(candidates)) as tc:
                    candidates = self.compressor.compress_results(
                        query=query,
                        candidates=candidates,
                        apply_sentence_filtering=True,
                        apply_mmr=True,
                        top_n=self.config.reranking.top_n or k,
                        mmr_lambda=self.config.compression.mmr_lambda,
                    )
                    tc.set_output(len(candidates))
            else:
                candidates = self.compressor.compress_results(
                    query=query,
                    candidates=candidates,
                    apply_sentence_filtering=True,
                    apply_mmr=True,
                    top_n=self.config.reranking.top_n or k,
                    mmr_lambda=self.config.compression.mmr_lambda,
                )

        return candidates

    def _execute_first_stage(self, query: str, top_k: int, mode: str) -> List[RetrievalResult]:
        if mode == "dense":
            return self.dense_retriever.retrieve(query, top_k=top_k)
        elif mode == "sparse":
            return self.sparse_retriever.retrieve(query, top_k=top_k)
        elif mode in ["hybrid_rrf", "hybrid_alpha"]:
            return self.hybrid_retriever.retrieve(query, top_k=top_k)
        else:
            return self.hybrid_retriever.retrieve(query, top_k=top_k)

    def assemble_context(
        self,
        query: str,
        retrieved_chunks: List[RetrievalResult],
        user_id: str = "default_user",
        system_prompt: str = "You are an accurate, strictly grounded AI assistant.",
        instructions: str = "Answer strictly using the retrieved evidence. Cite sources inline as [Doc #].",
        trace: Optional[ExecutionTrace] = None,
    ) -> AssembledContext:
        """Assembles prompt with token budgeting and Lost-in-the-Middle mitigation."""
        history = self.buffer_memory.get_formatted_history()
        user_memory = self.persistent_memory.format_for_context(user_id)

        if trace:
            with TraceContext(trace, "Context Assembly & Token Budgeting", input_count=len(retrieved_chunks)) as tc:
                assembled = self.context_assembler.assemble(
                    query=query,
                    system_prompt=system_prompt,
                    retrieved_chunks=retrieved_chunks,
                    dialogue_history=history,
                    persistent_memory_context=user_memory,
                    instructions=instructions,
                )
                tc.set_output(
                    len(assembled.retrieved_chunks),
                    total_tokens=assembled.token_allocation.total_tokens,
                    lost_in_the_middle=assembled.lost_in_the_middle_applied,
                )
        else:
            assembled = self.context_assembler.assemble(
                query=query,
                system_prompt=system_prompt,
                retrieved_chunks=retrieved_chunks,
                dialogue_history=history,
                persistent_memory_context=user_memory,
                instructions=instructions,
            )

        return assembled

    def generate_grounded_answer(
        self,
        query: str,
        retrieved_chunks: List[RetrievalResult],
        generator_fn: Optional[Any] = None,
    ) -> str:
        """
        Synthesizes a strictly grounded answer with inline citations.
        Uses generator_fn if provided, or high-fidelity template synthesizer.
        """
        if not retrieved_chunks:
            return self.guardrail.get_refusal_response("No relevant sources found.")

        # Check sufficiency guardrail
        is_sufficient, reason = self.guardrail.check_retrieval_sufficiency(query, retrieved_chunks)
        if not is_sufficient:
            return self.guardrail.get_refusal_response(reason)

        if generator_fn:
            return generator_fn(query, retrieved_chunks)

        # High-fidelity deterministic answer synthesis
        top_chunk = retrieved_chunks[0]
        doc_title = top_chunk.chunk.metadata.get("title", "Document")
        sentences = [s.strip() for s in top_chunk.chunk.text.split(".") if len(s.strip()) > 15]
        evidence_snippet = sentences[0] if sentences else top_chunk.chunk.text[:120]

        answer = (
            f"Based on {doc_title}, {evidence_snippet.lower().strip()} [Doc 1]. "
        )
        if len(retrieved_chunks) > 1:
            second_chunk = retrieved_chunks[1]
            sec_title = second_chunk.chunk.metadata.get("title", "Document")
            sec_sents = [s.strip() for s in second_chunk.chunk.text.split(".") if len(s.strip()) > 15]
            if sec_sents:
                answer += f"Additionally, {sec_sents[0].strip()} [Doc 2]."

        return answer

    def query(
        self,
        query: str,
        user_id: str = "default_user",
        top_k: int = 4,
        generator_fn: Optional[Any] = None,
    ) -> Tuple[str, GroundingReport, AssembledContext, ExecutionTrace]:
        """
        End-to-end full execution query pipeline:
        Retrieve -> Rerank -> Budget Context -> Generate Answer -> Audit Grounding.
        Returns: (answer, grounding_report, assembled_context, trace)
        """
        trace = ExecutionTrace(pipeline_id=str(uuid.uuid4()), query=query)
        start_time = time.perf_counter()

        # 1. Retrieve
        results = self.retrieve(query=query, top_k=top_k, trace=trace)

        # 2. Context Assembly
        assembled = self.assemble_context(query=query, retrieved_chunks=results, user_id=user_id, trace=trace)

        # 3. Generate Answer
        with TraceContext(trace, "Answer Generation", input_count=len(assembled.retrieved_chunks)) as tc:
            answer = self.generate_grounded_answer(query, assembled.retrieved_chunks, generator_fn)
            tc.set_output(1, answer_length=len(answer))

        # 4. Audit Grounding & Citations
        with TraceContext(trace, "Grounding & Faithfulness Audit", input_count=1) as tc:
            grounding_report = self.grounding_verifier.verify_grounding(
                generated_answer=answer,
                retrieved_results=assembled.retrieved_chunks,
            )
            tc.set_output(
                len(grounding_report.claims),
                grounding_score=grounding_report.grounding_score,
                is_grounded=grounding_report.is_fully_grounded,
            )

        # Update dialogue memory
        self.buffer_memory.add_turn("user", query)
        self.buffer_memory.add_turn("assistant", answer)

        trace.total_duration_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
        return answer, grounding_report, assembled, trace
