import numpy as np
from typing import List, Dict, Any


def hit_rate(retrieved: List[str], relevant: List[str], k: int = 5) -> float:
    retrieved_k = retrieved[:k]
    return 1.0 if any(item in retrieved_k for item in relevant) else 0.0


def mrr(retrieved: List[str], relevant: List[str], k: int = 5) -> float:
    for rank, item in enumerate(retrieved[:k], 1):
        if item in relevant:
            return 1.0 / rank
    return 0.0


def precision_at_k(retrieved: List[str], relevant: List[str], k: int = 5) -> float:
    if k == 0:
        return 0.0
    retrieved_k = retrieved[:k]
    relevant_in_k = sum(1 for item in retrieved_k if item in relevant)
    return relevant_in_k / k


def recall_at_k(retrieved: List[str], relevant: List[str], k: int = 5) -> float:
    if len(relevant) == 0:
        return 0.0
    retrieved_k = set(retrieved[:k])
    relevant_set = set(relevant)
    return len(retrieved_k & relevant_set) / len(relevant_set)


def ndcg_at_k(retrieved: List[str], relevant: List[str], k: int = 5) -> float:
    if len(relevant) == 0:
        return 0.0

    dcg = 0.0
    for i, item in enumerate(retrieved[:k]):
        if item in relevant:
            dcg += 1.0 / np.log2(i + 2)

    ideal_dcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def evaluate_retrieval(retrieved: List[str], relevant: List[str], k: int = 5) -> Dict[str, float]:
    """Run all retrieval metrics at once."""
    return {
        "hit_rate": hit_rate(retrieved, relevant, k),
        "mrr": mrr(retrieved, relevant, k),
        "precision@k": precision_at_k(retrieved, relevant, k),
        "recall@k": recall_at_k(retrieved, relevant, k),
        "ndcg@k": ndcg_at_k(retrieved, relevant, k),
    }