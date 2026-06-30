from typing import List, Dict, Any
from evaluation.metrics import evaluate_retrieval
from pipeline.rag_pipeline import RAGPipeline


class RetrievalEvaluator:
    """
    Evaluates retrieval quality of the RAG system.
    """

    def __init__(self, pipeline: RAGPipeline, k: int = 5):
        self.pipeline = pipeline
        self.k = k

    def evaluate(self, test_cases: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Run retrieval evaluation on a list of test cases.
        
        Each test case should have:
        - "query": str
        - "relevant_texts": List[str]
        """
        if not test_cases:
            return {}

        all_scores = {
            "hit_rate": [],
            "mrr": [],
            "precision@k": [],
            "recall@k": [],
            "ndcg@k": []
        }

        for case in test_cases:
            query = case["query"]
            relevant_texts = case["relevant_texts"]

            # Get retrieved chunks using the pipeline
            retrieved_chunks = self.pipeline.retrieve_only(query, top_k=self.k)
            retrieved_texts = [chunk.get("text", "") for chunk in retrieved_chunks]

            # Calculate metrics
            scores = evaluate_retrieval(retrieved_texts, relevant_texts, k=self.k)

            for metric, value in scores.items():
                all_scores[metric].append(value)

        # Calculate average scores
        avg_scores = {metric: round(sum(values) / len(values), 4) 
                      for metric, values in all_scores.items()}

        return avg_scores

    def evaluate_single(self, query: str, relevant_texts: List[str]) -> Dict[str, float]:
        """Evaluate a single query."""
        retrieved_chunks = self.pipeline.retrieve_only(query, top_k=self.k)
        retrieved_texts = [chunk.get("text", "") for chunk in retrieved_chunks]
        return evaluate_retrieval(retrieved_texts, relevant_texts, k=self.k)