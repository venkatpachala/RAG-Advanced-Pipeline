from pipeline.rag_pipeline import RAGPipeline
from typing import Dict, Any, List


class RAGApp:
    """
    Clean wrapper around RAGPipeline for DeepEval evaluation.
    """

    def __init__(self, collection_name: str = "rag_chunks_v1"):
        self.pipeline = RAGPipeline(collection_name=collection_name)
        print("RAGApp initialized successfully.")

    def generate_answer(self, query: str, top_k: int = 6) -> Dict[str, Any]:
        """
        Runs the query through the RAG pipeline and returns structured output
        required by DeepEval.
        """
        # Get full result from pipeline
        result = self.pipeline.ask(query, top_k=top_k)

        # Get retrieved context separately (can be optimized later)
        retrieved_chunks = self.pipeline.retrieve_only(query, top_k=top_k)
        retrieved_context = [chunk.get("text", "") for chunk in retrieved_chunks]

        return {
            "answer": result.get("answer", ""),
            "retrieved_context": retrieved_context,
            "citations": result.get("citations", []),
            "grounded": result.get("grounded", False),
            "rewritten_query": result.get("rewritten_query", query)
        }