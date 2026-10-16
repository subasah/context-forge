"""
ContextForge Maximal Marginal Relevance (MMR).
Balances query relevance against redundancy to select diverse information nuggets.
"""

from typing import List
import numpy as np

from context_forge.core.document import RetrievalResult


def compute_cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (norm1 * norm2))


def maximal_marginal_relevance(
    query_vector: np.ndarray,
    candidate_vectors: np.ndarray,
    candidates: List[RetrievalResult],
    top_n: int = 3,
    lambda_param: float = 0.7,
) -> List[RetrievalResult]:
    """
    Selects top_n candidates maximizing MMR:
    MMR = argmax_{d in R \\ S} [ lambda * Sim(d, Q) - (1 - lambda) * max_{s in S} Sim(d, s) ]
    """
    if len(candidates) == 0:
        return []

    top_n = min(top_n, len(candidates))
    selected_indices: List[int] = []
    unselected_indices = list(range(len(candidates)))

    # Compute relevance to query for all candidates
    relevance_scores = np.array([
        compute_cosine_similarity(query_vector, candidate_vectors[i])
        for i in unselected_indices
    ])

    # Pick the first candidate with highest relevance to query
    first_choice = int(np.argmax(relevance_scores))
    selected_indices.append(first_choice)
    unselected_indices.remove(first_choice)

    while len(selected_indices) < top_n and unselected_indices:
        best_mmr_score = -float("inf")
        best_candidate_idx = -1

        for idx in unselected_indices:
            cand_vec = candidate_vectors[idx]
            relevance = relevance_scores[idx]

            # Max similarity to already selected candidates
            diversity_penalties = [
                compute_cosine_similarity(cand_vec, candidate_vectors[s_idx])
                for s_idx in selected_indices
            ]
            max_sim_to_selected = max(diversity_penalties) if diversity_penalties else 0.0

            mmr_score = (lambda_param * relevance) - ((1.0 - lambda_param) * max_sim_to_selected)

            if mmr_score > best_mmr_score:
                best_mmr_score = mmr_score
                best_candidate_idx = idx

        if best_candidate_idx != -1:
            selected_indices.append(best_candidate_idx)
            unselected_indices.remove(best_candidate_idx)
        else:
            break

    results: List[RetrievalResult] = []
    for rank, idx in enumerate(selected_indices, start=1):
        res = candidates[idx].model_copy(deep=True)
        res.rank = rank
        res.retrieval_mode = "compressed"
        results.append(res)

    return results
