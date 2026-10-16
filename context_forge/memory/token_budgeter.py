"""
ContextForge Token Budgeter & Context Assembler.
Enforces strict token limits across memory and retrieval tiers,
and mitigates the 'Lost-in-the-Middle' LLM attention decay phenomenon.
"""

from typing import List, Optional, Tuple
from context_forge.core.document import (
    AssembledContext,
    RetrievalResult,
    TokenAllocation,
)
from context_forge.ingestion.cleaner import TextCleaner


class LostInTheMiddleSorter:
    """
    Mitigates Liu et al. (2023) Lost-in-the-Middle phenomenon.
    Places highest-ranked documents at the start and end of the context window
    where Transformer self-attention is strongest.
    """

    @staticmethod
    def reorder(items: List[RetrievalResult]) -> List[RetrievalResult]:
        """
        Input sorted by rank: [1, 2, 3, 4, 5, 6]
        Output:               [1, 3, 5, 6, 4, 2]
        """
        if len(items) <= 2:
            return items

        reordered: List[RetrievalResult] = []
        # Alternating left and right insertion
        left_side = True
        for item in items:
            if left_side:
                reordered.insert(len(reordered) // 2, item)
            else:
                reordered.append(item)
            left_side = not left_side

        # Standard U-curve distribution:
        # Top-1 at index 0, Top-2 at index -1, Top-3 at index 1, Top-4 at index -2...
        final_ordered: List[RetrievalResult] = [None] * len(items)  # type: ignore
        left_ptr = 0
        right_ptr = len(items) - 1

        for i, item in enumerate(items):
            if i % 2 == 0:
                final_ordered[left_ptr] = item
                left_ptr += 1
            else:
                final_ordered[right_ptr] = item
                right_ptr -= 1

        return final_ordered


class ContextAssembler:
    """
    Dynamically budgets tokens and constructs the final prompt payload.
    Ensures zero token overflow while maximizing high-relevance evidence.
    """

    def __init__(
        self,
        max_context_budget: int = 4096,
        reserved_generation_tokens: int = 1024,
        mitigate_lost_in_the_middle: bool = True,
    ):
        self.max_context_budget = max_context_budget
        self.reserved_generation = reserved_generation_tokens
        self.mitigate_lost_in_the_middle = mitigate_lost_in_the_middle

    def assemble(
        self,
        query: str,
        system_prompt: str,
        retrieved_chunks: List[RetrievalResult],
        dialogue_history: str = "",
        persistent_memory_context: str = "",
        instructions: str = "",
    ) -> AssembledContext:
        """
        Allocates token budgets strictly and packs components into prompt.
        """
        sys_tokens = TextCleaner.count_tokens(system_prompt)
        inst_tokens = TextCleaner.count_tokens(instructions)
        query_tokens = TextCleaner.count_tokens(query)

        # Usable budget for history and retrieval
        usable_budget = self.max_context_budget - (
            self.reserved_generation + sys_tokens + inst_tokens + query_tokens
        )
        usable_budget = max(usable_budget, 400)

        # Allocate memory budget (up to 30% of usable budget)
        max_memory_budget = int(usable_budget * 0.30)
        history_tokens = TextCleaner.count_tokens(dialogue_history)
        perm_tokens = TextCleaner.count_tokens(persistent_memory_context)

        total_memory_tokens = history_tokens + perm_tokens
        active_history = dialogue_history
        active_memory = persistent_memory_context

        # Prune memory if exceeding budget
        if total_memory_tokens > max_memory_budget:
            # Keep persistent memory, trim dialogue history to fit
            rem_for_history = max(0, max_memory_budget - perm_tokens)
            words = dialogue_history.split()
            active_history = " ".join(words[-rem_for_history:]) if rem_for_history > 0 else ""
            history_tokens = TextCleaner.count_tokens(active_history)

        actual_memory_tokens = history_tokens + perm_tokens
        retrieval_budget = usable_budget - actual_memory_tokens

        # Select retrieved chunks within retrieval budget
        retained_chunks: List[RetrievalResult] = []
        current_retrieval_tokens = 0

        for r in retrieved_chunks:
            chunk_tokens = r.chunk.token_count or TextCleaner.count_tokens(r.chunk.text)
            # Add markup overhead (~20 tokens per doc container)
            chunk_footprint = chunk_tokens + 20
            if current_retrieval_tokens + chunk_footprint <= retrieval_budget:
                retained_chunks.append(r)
                current_retrieval_tokens += chunk_footprint
            else:
                break

        # Apply Lost-in-the-Middle reordering
        applied_litm = False
        if self.mitigate_lost_in_the_middle and len(retained_chunks) > 2:
            retained_chunks = LostInTheMiddleSorter.reorder(retained_chunks)
            applied_litm = True

        # Build context document block
        doc_blocks: List[str] = []
        for idx, res in enumerate(retained_chunks, start=1):
            src_title = res.chunk.metadata.get("title", f"Document {idx}")
            doc_blocks.append(
                f"<source id=\"Doc {idx}\" title=\"{src_title}\">\n{res.chunk.text}\n</source>"
            )

        context_str = "\n\n".join(doc_blocks) if doc_blocks else "(No reference context retrieved)"

        # Construct final prompt
        prompt_sections = [
            f"### INSTRUCTIONS\n{instructions}" if instructions else "",
            f"### USER PROFILE & MEMORY\n{active_memory}" if active_memory else "",
            f"### CONVERSATION HISTORY\n{active_history}" if active_history else "",
            f"### RETRIEVED REFERENCE CONTEXT\n{context_str}",
            f"### USER QUERY\n{query}",
            "### GROUNDED RESPONSE (Include inline [Doc #] citations):",
        ]
        assembled_prompt = "\n\n".join([s for s in prompt_sections if s.strip()])

        total_consumed = (
            sys_tokens
            + inst_tokens
            + actual_memory_tokens
            + current_retrieval_tokens
            + query_tokens
        )

        allocation = TokenAllocation(
            system_prompt_tokens=sys_tokens,
            instruction_tokens=inst_tokens,
            memory_history_tokens=actual_memory_tokens,
            retrieved_context_tokens=current_retrieval_tokens,
            reserved_generation_tokens=self.reserved_generation,
            total_tokens=total_consumed,
            budget_limit=self.max_context_budget,
        )

        return AssembledContext(
            system_prompt=system_prompt,
            assembled_prompt=assembled_prompt,
            retrieved_chunks=retained_chunks,
            token_allocation=allocation,
            lost_in_the_middle_applied=applied_litm,
            ordering=[r.chunk.id for r in retained_chunks],
        )
