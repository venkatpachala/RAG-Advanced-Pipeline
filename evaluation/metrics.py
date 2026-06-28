from typing import List, Set


def _normalize(text: str) -> str:
    """Simple normalization for better matching"""
    return text.lower().strip()


def hit_rate(retrieved: List[str], relevant: Set[str]) -> float:
    """1 if at least one relevant chunk appears in retrieved results"""
    retrieved_norm = [_normalize(r) for r in retrieved]
    relevant_norm = {_normalize(r) for r in relevant}

    for rel in relevant_norm:
        if any(rel in ret for ret in retrieved_norm):
            return 1.0
    return 0.0


def mrr(retrieved: List[str], relevant: Set[str]) -> float:
    """Mean Reciprocal Rank"""
    retrieved_norm = [_normalize(r) for r in retrieved]
    relevant_norm = {_normalize(r) for r in relevant}

    for rank, ret in enumerate(retrieved_norm, 1):
        if any(rel in ret for rel in relevant_norm):
            return 1.0 / rank
    return 0.0


def precision_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    if not retrieved:
        return 0.0

    retrieved_k = [_normalize(r) for r in retrieved[:k]]
    relevant_norm = {_normalize(r) for r in relevant}

    relevant_count = sum(
        1 for ret in retrieved_k 
        if any(rel in ret for rel in relevant_norm)
    )
    return relevant_count / len(retrieved_k)


def recall_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    if not relevant:
        return 0.0

    retrieved_k = {_normalize(r) for r in retrieved[:k]}
    relevant_norm = {_normalize(r) for r in relevant}

    matched = sum(
        1 for rel in relevant_norm 
        if any(rel in ret for ret in retrieved_k)
    )
    return matched / len(relevant_norm)


def ndcg_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    if not relevant:
        return 0.0

    retrieved_norm = [_normalize(r) for r in retrieved[:k]]
    relevant_norm = {_normalize(r) for r in relevant}

    dcg = 0.0
    for i, ret in enumerate(retrieved_norm):
        if any(rel in ret for rel in relevant_norm):
            dcg += 1.0 / (i + 2) ** 0.5   # Simplified DCG

    ideal_dcg = sum(1.0 / (i + 2) ** 0.5 for i in range(min(len(relevant), k)))
    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0