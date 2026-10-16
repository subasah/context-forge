# Enterprise RAG & Context Engineering Architecture Specification

## Section 1: Ingestion & Document Parsing
Enterprise Retrieval-Augmented Generation (RAG) pipelines require resilient multi-modal and multi-format document parsing. Unstructured documents (PDFs, Markdown, Docx, HTML, and Source Code) must be normalized to strip boilerplate artifacts, headers, footers, and encoding anomalies before chunking. Document ingestion tracks metadata including document ID, source URI, creation timestamp, author, and semantic tags.

## Section 2: Chunking Strategies & Trade-offs
Chunking transforms continuous documents into atomic discrete text segments. The optimal chunking strategy balances semantic cohesion against vector similarity granularity:
1. **Fixed-Window Chunking**: Splits text into invariant token or character windows (e.g., 500 characters with 100 character sliding overlap). While computationally trivial, it risks splitting sentences and concepts mid-thought.
2. **Recursive Character Chunking**: Recursively splits along structural boundaries (paragraphs `\n\n`, newlines `\n`, sentences `. `, clauses `, `). It preserves semantic integrity and natural narrative flow.
3. **Semantic Chunking**: Computes embedding distance gradients between adjacent sentences. When cosine distance exceeds a statistical breakpoint percentile threshold (e.g., the 80th percentile), a chunk boundary is instantiated.
4. **Parent-Child Hierarchical Chunking**: Decouples retrieval representation from generation context. Small child chunks (200-300 characters) are embedded for high-precision vector search. Upon retrieval, the system unpacks the parent chunk (1200 characters) to supply the LLM with full context.
5. **Contextual Retrieval Enrichment**: Introduced by Anthropic, each chunk is enriched with situational context headers (document title and section hierarchy) prepended to the raw text, preventing orphan context loss during retrieval.

## Section 3: Embeddings & Vector Space
Dense embeddings map discrete text passages into a continuous $D$-dimensional manifold (e.g., 384 dimensions for MiniLM, 1536 for OpenAI text-embedding-3-small). Vectors are L2-normalized so that the inner dot product $\mathbf{u} \cdot \mathbf{v}$ is identical to cosine similarity $\cos(\theta)$. For sparse keyword matching, Okapi BM25 scores term frequency penalized by document length and scaled by inverse document frequency (IDF).

## Section 4: Hybrid Retrieval & Reciprocal Rank Fusion
Dense vector retrieval excels at conceptual and semantic queries ("how to optimize prompt costs") but struggles with exact alphanumeric identifiers, error codes, and unique SKUs (e.g., "Error Code ERR-9042"). Sparse BM25 retrieval excels at lexical matching but misses semantic synonyms.
**Reciprocal Rank Fusion (RRF)** combines dense and sparse rankings without requiring score normalization:
$$RRF(d) = \sum_{m \in \{dense, sparse\}} \frac{1}{k + r_m(d)}$$
where $k=60$ is the standard smoothing constant and $r_m(d)$ is the 1-based rank of document $d$.

## Section 5: Cross-Encoder Reranking
First-stage retrieval operates with bi-encoders for high-throughput candidate generation ($O(1)$ to $O(\log N)$). Second-stage retrieval uses Cross-Encoders: the query and candidate passage are concatenated and passed jointly through full self-attention layers. This models all pairwise token interactions, yielding superior ranking precision at the cost of higher latency.

## Section 6: Contextual Compression & MMR
Retrieved passages often contain 60-80% extraneous content. Contextual compression filters out sentences with cosine similarity below a threshold $\tau = 0.55$. Maximal Marginal Relevance (MMR) balances query relevance against novelty, suppressing near-duplicate passages:
$$MMR = \arg\max_{d_i \in R \setminus S} \left[ \lambda \cdot \text{Sim}(d_i, Q) - (1 - \lambda) \cdot \max_{d_j \in S} \text{Sim}(d_i, d_j) \right]$$

## Section 7: Grounding, Citations & Hallucination Mitigation
Grounded generation guarantees that every factual claim in an answer is supported by evidence:
- **Inline Attribution**: Answers must include verifiable citations mapping directly to source chunks `[Doc #]`.
- **Claim-Level Entailment**: Natural Language Inference (NLI) breaks answers into atomic statements and validates entailment against the retrieved evidence.
- **Defensive Refusal Guardrail**: If top retrieval confidence falls below threshold $\tau < 0.35$, the pipeline declines to answer rather than hallucinating.

## Section 8: Memory & Lost-in-the-Middle Context Assembly
Attention in Transformer models exhibits a U-shaped distribution: models pay high attention to the beginning (primacy effect) and end (recency effect) of context windows, but degrade in the middle. The **Lost-in-the-Middle** sorter arranges chunks so that the highest-ranked documents are placed at the very top and very bottom of the assembled prompt.
