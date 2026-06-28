import logging
from typing import List, Dict, Any, Set
from evaluation.metrics import hit_rate, mrr, precision_at_k, recall_at_k, ndcg_at_k

logger = logging.getLogger(__name__)


class RAGEvaluator:
    """
    Evaluation Framework for RAG Retrieval Quality
    """

    def __init__(self, query_engine):
        self.query_engine = query_engine

    def evaluate(
        self,
        test_cases: List[Dict[str, Any]],
        k: int = 5
    ) -> Dict[str, float]:
        """
        Run evaluation on a set of test cases.

        test_cases format:
        [
            {
                "query": "How does cold start latency work?",
                "relevant_chunks": ["text snippet 1", "text snippet 2", ...]
            },
            ...
        ]
        """
        if not test_cases:
            return {}

        results = {
            "hit_rate": 0.0,
            "mrr": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "ndcg": 0.0
        }

        for case in test_cases:
            query = case["query"]
            relevant = set(case.get("relevant_chunks", []))

            # Get retrieved results from your QueryEngine
            retrieved_results = self.query_engine.query(query, limit=k)
            retrieved_texts = [r.get("text", "") for r in retrieved_results]

            # Calculate all metrics
            results["hit_rate"] += hit_rate(retrieved_texts, relevant)
            results["mrr"] += mrr(retrieved_texts, relevant)
            results["precision"] += precision_at_k(retrieved_texts, relevant, k)
            results["recall"] += recall_at_k(retrieved_texts, relevant, k)
            results["ndcg"] += ndcg_at_k(retrieved_texts, relevant, k)

        # Average the scores
        total = len(test_cases)
        for metric in results:
            results[metric] /= total

        logger.info(f"Evaluation Results @ K={k}: {results}")
        return results