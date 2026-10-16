"""
ContextForge Interactive RAG & Context Engineering Studio.
Visual playground for Ingestion, Chunking, Hybrid RRF Retrieval,
Cross-Encoder Reranking, Token Budgeting, and Grounding Audits.
"""

import time
import json
import streamlit as st
import pandas as pd
import numpy as np

from context_forge import RAGPipeline, PipelineConfig
from context_forge.core.document import Document
from context_forge.chunking.fixed_window import FixedWindowChunker
from context_forge.chunking.recursive import RecursiveCharacterChunker
from context_forge.chunking.semantic import SemanticChunker
from context_forge.chunking.parent_child import ParentChildChunker
from context_forge.chunking.contextual import ContextualChunker
from context_forge.mcp_server.server import create_mcp_server

st.set_page_config(
    page_title="ContextForge: Enterprise RAG & Context Engineering",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #3b82f6, #8b5cf6, #ec4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #64748b;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8fafc;
        border-radius: 8px;
        padding: 16px;
        border-left: 4px solid #3b82f6;
        margin-bottom: 12px;
    }
    .badge {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 5px;
    }
    .badge-blue { background-color: #dbeafe; color: #1e40af; }
    .badge-purple { background-color: #f3e8ff; color: #6b21a8; }
    .badge-green { background-color: #dcfce7; color: #166534; }
    .badge-orange { background-color: #ffedd5; color: #9a3412; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_cached_pipeline():
    config = PipelineConfig()
    pipeline = RAGPipeline(config=config)
    # Seed default corpora
    pipeline.ingest_document("data/sample_rag_architecture.md", title="RAG Architecture Spec")
    pipeline.ingest_document("data/enterprise_policies.json", title="Corporate Security Policies")
    pipeline.ingest_document("data/generative_ai_systems.txt", title="AI Systems Guide")
    return pipeline


pipeline = get_cached_pipeline()

# --- SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.markdown("### ⚙️ Pipeline Controls")
    
    selected_corpus = st.selectbox(
        "Active Knowledge Corpus",
        ["All Ingested Corpora (3 documents)", "RAG Architecture Spec", "Corporate Security Policies", "AI Systems Guide", "Custom Upload/Text"],
    )

    st.markdown("---")
    st.markdown("#### 🔍 Retrieval Engine")
    retrieval_mode = st.selectbox(
        "Retrieval Mode",
        ["hybrid_rrf", "dense", "sparse", "hybrid_alpha"],
        index=0,
        format_func=lambda x: {
            "hybrid_rrf": "⚡ Hybrid RRF (Dense + BM25)",
            "dense": "🧠 Dense Semantic Vector",
            "sparse": "📖 Sparse Lexical (BM25 Okapi)",
            "hybrid_alpha": "⚖️ Hybrid Convex Combination (α)",
        }[x],
    )

    enable_reranker = st.toggle("Cross-Encoder Reranker", value=True)
    enable_compression = st.toggle("Contextual Compression & MMR", value=False)
    enable_hyde = st.toggle("HyDE Query Expansion", value=False)
    mitigate_litm = st.toggle("Lost-in-the-Middle Attention Ordering", value=True)

    st.markdown("---")
    st.markdown("#### 📏 Context Window Budget")
    max_budget = st.slider("Total Context Budget (Tokens)", 1024, 16384, 4096, 512)
    top_k = st.slider("Top K Retrieval Chunks", 1, 10, 4)

    pipeline.config.retrieval.retrieval_mode = retrieval_mode
    pipeline.config.reranking.enabled = enable_reranker
    pipeline.config.compression.enabled = enable_compression
    pipeline.config.retrieval.enable_hyde = enable_hyde
    pipeline.config.memory.mitigate_lost_in_the_middle = mitigate_litm
    pipeline.config.memory.max_context_budget = max_budget


# --- MAIN HEADER ---
st.markdown("<div class='main-title'>ContextForge</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Enterprise-Grade Retrieval, Grounding, Memory & Context Engineering Framework</div>", unsafe_allow_html=True)

st.markdown("""
<span class='badge badge-blue'>Hybrid RRF Retrieval</span>
<span class='badge badge-purple'>Cross-Encoder Joint Attention</span>
<span class='badge badge-green'>Strict Grounding & Entailment</span>
<span class='badge badge-orange'>MCP Context-on-Demand</span>
""", unsafe_allow_html=True)

st.write("")

# --- TABS ---
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🚀 End-to-End RAG Studio",
    "🧩 Chunking Comparator",
    "⚖️ Hybrid RRF Benchmark",
    "🧠 Memory & Token Budgeter",
    "🔌 MCP Agent Server",
    "🎓 Recruiter & Architecture Masterclass",
])


# ==========================================
# TAB 1: RAG STUDIO
# ==========================================
with tab1:
    st.markdown("### Interactive Retrieval & Grounded Synthesis")
    st.write("Test end-to-end query execution with real-time vector scoring, reranking, and citation entailment verification.")

    col_q1, col_q2 = st.columns([3, 1])
    with col_q1:
        query_input = st.text_input(
            "Enter User Query:",
            value="How does Reciprocal Rank Fusion combine dense and sparse rankings?",
        )
    with col_q2:
        st.write("")
        st.write("")
        run_btn = st.button("⚡ Execute Pipeline", use_container_width=True, type="primary")

    col_presets = st.columns(4)
    with col_presets[0]:
        if st.button("📌 Preset: RRF Formula", use_container_width=True):
            query_input = "What is the mathematical formulation of Reciprocal Rank Fusion?"
            run_btn = True
    with col_presets[1]:
        if st.button("📌 Preset: GDPR Article 17", use_container_width=True):
            query_input = "What is the GDPR compliance timeline specified in SEC-POL-002?"
            run_btn = True
    with col_presets[2]:
        if st.button("📌 Preset: Token Budgeting", use_container_width=True):
            query_input = "How does token budgeting handle context limits in AI agents?"
            run_btn = True
    with col_presets[3]:
        if st.button("📌 Preset: Cross-Encoder", use_container_width=True):
            query_input = "Why are cross-encoders more accurate than bi-encoders for reranking?"
            run_btn = True

    if run_btn:
        with st.spinner("Processing multi-stage retrieval, reranking, and grounding audit..."):
            answer, grounding_report, assembled_ctx, trace = pipeline.query(
                query=query_input,
                top_k=top_k,
            )

        col_left, col_right = st.columns([3, 2])

        with col_left:
            st.markdown("#### 📝 Synthesized & Grounded Answer")
            st.info(answer)

            # Grounding Audit Card
            st.markdown("#### 🛡️ Grounding & Faithfulness Audit")
            col_g1, col_g2, col_g3 = st.columns(3)
            with col_g1:
                st.metric("Faithfulness Score", f"{grounding_report.grounding_score:.1%}")
            with col_g2:
                st.metric("Citation Coverage", f"{grounding_report.citation_coverage:.1%}")
            with col_g3:
                risk_status = "🚨 High Risk" if grounding_report.hallucination_detected else "✅ Zero Hallucination"
                st.metric("Hallucination Risk", risk_status)

            if grounding_report.claims:
                with st.expander("🔬 View Claim-Level Entailment Analysis", expanded=False):
                    claims_df = pd.DataFrame([
                        {
                            "Claim Assertion": c.claim_text,
                            "Supported?": "✅ Supported" if c.is_supported else "❌ Ungrounded",
                            "Confidence": f"{c.confidence:.3f}",
                            "Reasoning": c.reasoning,
                        }
                        for c in grounding_report.claims
                    ])
                    st.dataframe(claims_df, use_container_width=True)

            if grounding_report.citations:
                with st.expander("📚 View Extracted Citations", expanded=False):
                    for cit in grounding_report.citations:
                        st.write(f"- **[Doc #{cit.citation_id}]**: *{cit.source_title}* | Quote Match: `{cit.char_match_ratio:.1%}` | Valid: `{'YES' if cit.is_valid else 'NO'}`")

        with col_right:
            st.markdown("#### ⏱️ Microsecond Execution Trace")
            trace_df = pd.DataFrame([
                {
                    "Pipeline Stage": s.name,
                    "Latency (ms)": s.duration_ms,
                    "In": s.input_count,
                    "Out": s.output_count,
                }
                for s in trace.steps
            ])
            st.dataframe(trace_df, use_container_width=True, hide_index=True)
            st.caption(f"**Total End-to-End Latency:** {trace.total_duration_ms:.2f} ms")

            st.markdown("#### 📦 Retrieved & Reranked Chunks")
            for res in assembled_ctx.retrieved_chunks:
                with st.expander(f"Rank #{res.rank} | Score: {res.score:.4f} ({res.retrieval_mode})", expanded=(res.rank == 1)):
                    st.caption(f"**Source**: {res.chunk.metadata.get('title', 'Doc')} | **Tokens**: {res.chunk.token_count}")
                    if res.dense_score is not None:
                        st.caption(f"Dense Score: `{res.dense_score:.4f}` | Sparse BM25: `{res.sparse_score or 0.0:.4f}`")
                    st.write(res.chunk.text)


# ==========================================
# TAB 2: CHUNKING COMPARATOR
# ==========================================
with tab2:
    st.markdown("### Chunking Architecture Visualizer")
    st.write("Compare 5 production chunking strategies on the same source text side-by-side.")

    sample_doc_text = Path("data/sample_rag_architecture.md").read_text()
    chunk_test_doc = Document(text=sample_doc_text[:2000], metadata={"title": "RAG Architecture"})

    col_c1, col_c2 = st.columns([1, 1])
    with col_c1:
        strategy_1 = st.selectbox("Strategy A", ["recursive", "fixed", "semantic", "parent_child", "contextual"], index=0)
    with col_c2:
        strategy_2 = st.selectbox("Strategy B", ["recursive", "fixed", "semantic", "parent_child", "contextual"], index=4)

    def run_chunker(strategy_name, doc):
        if strategy_name == "fixed":
            return FixedWindowChunker(chunk_size=300, chunk_overlap=50).chunk(doc)
        elif strategy_name == "recursive":
            return RecursiveCharacterChunker(chunk_size=300, chunk_overlap=50).chunk(doc)
        elif strategy_name == "semantic":
            return SemanticChunker(max_chunk_size=500, min_chunk_size=100).chunk(doc)
        elif strategy_name == "parent_child":
            return ParentChildChunker(parent_chunk_size=600, child_chunk_size=200).chunk(doc)
        elif strategy_name == "contextual":
            return ContextualChunker(base_chunker=RecursiveCharacterChunker(chunk_size=300, chunk_overlap=50)).chunk(doc)

    chunks_a = run_chunker(strategy_1, chunk_test_doc)
    chunks_b = run_chunker(strategy_2, chunk_test_doc)

    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.markdown(f"#### Strategy A: `{strategy_1}`")
        st.metric("Total Chunks", len(chunks_a))
        st.metric("Avg Tokens / Chunk", f"{np.mean([c.token_count for c in chunks_a]):.1f}" if chunks_a else "0")
        for i, c in enumerate(chunks_a[:3], 1):
            with st.expander(f"Chunk #{i} ({c.token_count} tokens)"):
                st.code(c.enriched_text if hasattr(c, "enriched_text") else c.text, language="markdown")

    with col_m2:
        st.markdown(f"#### Strategy B: `{strategy_2}`")
        st.metric("Total Chunks", len(chunks_b))
        st.metric("Avg Tokens / Chunk", f"{np.mean([c.token_count for c in chunks_b]):.1f}" if chunks_b else "0")
        for i, c in enumerate(chunks_b[:3], 1):
            with st.expander(f"Chunk #{i} ({c.token_count} tokens)"):
                st.code(c.enriched_text if hasattr(c, "enriched_text") else c.text, language="markdown")


# ==========================================
# TAB 3: HYBRID RRF BENCHMARK
# ==========================================
with tab3:
    st.markdown("### Dense vs BM25 vs Hybrid RRF Comparative Lab")
    st.write("Demonstrates why hybrid retrieval outperforms single-mode search by evaluating across lexical and semantic queries.")

    benchmark_queries = [
        {"type": "Exact Alphanumeric Identifier", "query": "What is SEC-POL-001 policy on hardware security keys?", "gold": "SEC-POL-001"},
        {"type": "Conceptual Reasoning", "query": "How to minimize attention decay in long context windows for LLMs?", "gold": "Lost-in-the-Middle"},
        {"type": "Temporal & Quantitative Threshold", "query": "What is the 72 hour breach notification deadline?", "gold": "SEC-POL-003"},
        {"type": "Algorithmic Formulation", "query": "Okapi BM25 formula document length normalization parameter b", "gold": "BM25"},
    ]

    bench_df = pd.DataFrame(benchmark_queries)
    st.dataframe(bench_df, use_container_width=True)

    if st.button("🚀 Run Comparative Retrieval Benchmark"):
        rows = []
        for bq in benchmark_queries:
            q = bq["query"]
            dense_top = pipeline.retrieve(q, top_k=1, mode="dense", use_reranker=False)
            sparse_top = pipeline.retrieve(q, top_k=1, mode="sparse", use_reranker=False)
            hybrid_top = pipeline.retrieve(q, top_k=1, mode="hybrid_rrf", use_reranker=False)
            rerank_top = pipeline.retrieve(q, top_k=1, mode="hybrid_rrf", use_reranker=True)

            def get_title(res_list):
                return res_list[0].chunk.metadata.get("title", "") if res_list else "None"

            rows.append({
                "Query Type": bq["type"],
                "Dense Rank 1": get_title(dense_top),
                "BM25 Rank 1": get_title(sparse_top),
                "Hybrid RRF Rank 1": get_title(hybrid_top),
                "Hybrid + Rerank": get_title(rerank_top),
            })

        st.markdown("#### Benchmark Results:")
        st.dataframe(pd.DataFrame(rows), use_container_width=True)


# ==========================================
# TAB 4: MEMORY & TOKEN BUDGETER
# ==========================================
with tab4:
    st.markdown("### Context Memory & Token Allocation Simulator")
    st.write("Simulates strict token budgeting across System Prompt, Instructions, Memory, and Retrieved Knowledge.")

    c_b1, c_b2 = st.columns([2, 1])
    with c_b1:
        st.markdown("#### Dynamic Budget Stack")
        # Visual breakdown chart
        chart_data = pd.DataFrame({
            "Context Tier": ["System Prompt", "Task Instructions", "Dialogue Memory", "User Profile Facts", "Retrieved Chunks", "Reserved Generation"],
            "Tokens": [300, 200, 450, 250, 1850, 1024],
        })
        st.bar_chart(chart_data.set_index("Context Tier"))

    with c_b2:
        st.markdown("#### Lost-in-the-Middle Mitigation")
        st.info("""
        **U-Shaped Attention Phenomenon:**
        LLMs attend strongly to context edges (primacy & recency) and suffer attention degradation in the middle.
        
        **ContextForge Reordering:**
        - Top-1 chunk -> First position (top)
        - Top-2 chunk -> Last position (bottom)
        - Top-3/4 -> Inner middle
        """)

    st.markdown("#### SQLite Persistent Episodic Memory Store")
    col_u1, col_u2, col_u3, col_u4 = st.columns(4)
    with col_u1:
        u_id = st.text_input("User ID", value="user_principal_dev")
    with col_u2:
        cat = st.selectbox("Category", ["profile", "preference", "entity", "constraint"])
    with col_u3:
        k_name = st.text_input("Key", value="target_latency")
    with col_u4:
        v_name = st.text_input("Value", value="p99 < 150ms")

    if st.button("💾 Save Fact to Persistent Store"):
        pipeline.persistent_memory.set(u_id, cat, k_name, v_name)
        st.success(f"Stored `{k_name}: {v_name}` for `{u_id}` in SQLite!")

    st.markdown("##### Current Records for User:")
    items = pipeline.persistent_memory.get_all_for_user(u_id)
    if items:
        st.dataframe(pd.DataFrame([{"Category": it.category, "Key": it.key, "Value": it.value, "Updated": it.updated_at} for it in items]))
    else:
        st.write("No records found for this user ID.")


# ==========================================
# TAB 5: MCP AGENT SERVER
# ==========================================
with tab5:
    st.markdown("### Model Context Protocol (MCP) Server Inspector")
    st.write("Demonstrates how autonomous agents pull context dynamically via MCP tools instead of static prompt stuffing.")

    mcp_srv = create_mcp_server(pipeline=pipeline)

    col_t1, col_t2 = st.columns([1, 2])
    with col_t1:
        st.markdown("#### Available MCP Tools")
        st.code("""
• search_knowledge_base
  - query: string
  - top_k: int
  - mode: string
  
• fetch_chunk_details
  - chunk_id: string

• query_memory
  - user_id: string
  - keyword: string

• save_memory_fact
  - user_id: string
  - category: string
  - key: string
  - value: string

• verify_grounding
  - claim: string
  - chunk_ids: list[string]
        """, language="yaml")

    with col_t2:
        st.markdown("#### Test MCP Tool Invocation")
        tool_to_test = st.selectbox("Select MCP Tool", ["search_knowledge_base", "query_memory"])
        if tool_to_test == "search_knowledge_base":
            t_query = st.text_input("Tool Argument: query", value="Cross-Encoder reranking")
            if st.button("Invoke MCP Tool"):
                import asyncio
                res = asyncio.run(mcp_srv.call_tool("search_knowledge_base", {"query": t_query, "top_k": 2}))
                st.code(res.content[0].text, language="markdown")
        elif tool_to_test == "query_memory":
            t_user = st.text_input("Tool Argument: user_id", value="user_principal_dev")
            if st.button("Invoke MCP Tool"):
                import asyncio
                res = asyncio.run(mcp_srv.call_tool("query_memory", {"user_id": t_user}))
                st.code(res.content[0].text, language="markdown")


# ==========================================
# TAB 6: RECRUITER MASTERCLASS
# ==========================================
with tab6:
    st.markdown("### Technical Interview & Recruiter Showcase Masterclass")
    st.write("High-impact architecture explanations, trade-off analyses, and design decisions for senior and staff engineering interviews.")

    qa_list = [
        (
            "1. Why do Bi-Encoders fail on exact alphanumeric queries, and why does BM25 fail on semantic queries?",
            """
            - **Bi-Encoders** project passages and queries into a continuous vector space where words are compressed into dense latent coordinates. Rare alphanumeric tokens (e.g. `ERR-9042`, `SEC-POL-002`) suffer from out-of-vocabulary hashing or token fragmentation, yielding weak dot product alignment.
            - **BM25 (Sparse)** counts exact lexical overlap weighted by inverse document frequency (IDF). It has no concept of semantic synonyms—searching for 'automobile repair' will return 0 hits for a document stating 'car maintenance' unless those exact words appear.
            - **Hybrid Solution**: Reciprocal Rank Fusion (RRF) merges dense and sparse rankings, guaranteeing that either a strong semantic match OR an exact keyword hit will surface in the top-k!
            """
        ),
        (
            "2. What is Reciprocal Rank Fusion (RRF) and why is it preferred over Score Averaging?",
            """
            - **Score Averaging Problem**: Cosine similarity is bounded in [-1, 1], while BM25 scores are unbounded $[0, \infty)$ and scale with document length and corpus size. Min-max normalization depends heavily on outliers.
            - **RRF Equation**:
              $$RRF(d) = \\sum_{m \\in \\{dense, sparse\\}} \\frac{1}{k + r_m(d)}$$
            - **Advantage**: RRF is purely rank-based. It is scale-invariant, monotonic, robust to outliers, and requires zero hyperparameter tuning across differing embedding models.
            """
        ),
        (
            "3. What is the difference between Bi-Encoders and Cross-Encoders, and what is the computational complexity trade-off?",
            """
            - **Bi-Encoder**: Encodes query $\mathbf{q} = f(Q)$ and document $\mathbf{d} = g(D)$ separately into independent vectors. Retrieval is simple dot product $\mathbf{q} \cdot \mathbf{d}$, which scales in $O(1)$ to $O(\\log N)$ using vector indexes (HNSW, ScaNN).
            - **Cross-Encoder**: Concatenates $[CLS] \\circ Q \\circ [SEP] \\circ D$ and computes full $O((L_Q + L_D)^2)$ pairwise self-attention across all layers. Every token in the query directly attends to every token in the passage.
            - **Production Strategy (Two-Stage)**: Use Bi-Encoder + BM25 to retrieve top 50 candidates in < 20ms, then run a lightweight Cross-Encoder (e.g. `ms-marco-MiniLM-L-6-v2`) on only the top 50 in < 50ms!
            """
        ),
        (
            "4. How do you mitigate the 'Lost-in-the-Middle' phenomenon in long context windows?",
            """
            - **The Problem (Liu et al., 2023)**: Transformer attention displays a pronounced U-shaped curve: tokens at the start (primacy) and end (recency) receive significantly higher attention weights than tokens in the middle.
            - **The Solution**: Rather than ordering retrieved chunks strictly descending $[1, 2, 3, 4, 5]$, ContextForge employs alternating U-curve sorting:
              - Rank 1 -> Top of context
              - Rank 2 -> Bottom of context
              - Ranks 3/4 -> Inner middle
            - This ensures the highest-confidence evidence resides in the highest-attention zones.
            """
        ),
        (
            "5. What is Anthropic Contextual Retrieval, and what problem does it solve?",
            """
            - **Context Loss Problem**: Traditional chunking cuts documents into 300-token chunks. A chunk stating: 'Revenue grew by 24% year-over-year' lacks the surrounding context of which company, year, or division it refers to.
            - **Contextual Retrieval**: Prepends 50-100 tokens of situational context (e.g., `[Doc: ACME Corp 10-K > Section: Cloud Services]`) to each chunk prior to embedding and BM25 indexing.
            - **Result**: Increases retrieval recall by 35-50% without altering chunk boundaries.
            """
        ),
        (
            "6. How does the Model Context Protocol (MCP) transform context delivery for agents?",
            """
            - **Old Pattern (Push / Stuffing)**: All potential search documents are unconditionally stuffed into the system prompt upfront. This leads to prompt pollution, high token costs, and context distraction.
            - **Modern Pattern (Pull / On-Demand via MCP)**: The agent receives lightweight tool descriptions (`search_knowledge_base`, `fetch_chunk_details`). The agent reasoning loop dynamically evaluates information gaps and pulls chunks only when needed, reducing token overhead by up to 85%.
            """
        ),
    ]

    for question, answer_md in qa_list:
        with st.expander(f"💡 {question}", expanded=False):
            st.markdown(answer_md)

    st.markdown("---")
    st.markdown("#### 💼 Portfolio & Resume Bullet Points")
    st.code("""
• Architected 'ContextForge', an end-to-end production RAG framework featuring Hybrid Reciprocal Rank Fusion (RRF), Cross-Encoder neural reranking, and Anthropic contextual retrieval.
• Engineered conversational state and SQLite-backed persistent episodic memory with dynamic token budgeting and Lost-in-the-Middle attention mitigation.
• Built a claim-level NLI grounding verifier and citation attribution engine, enforcing zero-hallucination defensive refusal guardrails.
• Implemented an Anthropic Model Context Protocol (MCP 2.x) standard server, enabling autonomous agents to pull contextual knowledge on demand.
    """, language="markdown")
