import logging
from typing import Dict, Any, List
from ingestion.retrieval.query_engine import QueryEngine
from generation.grounded_generator import GroundedGenerator

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    High-level RAG Pipeline that combines:
    - Grounded Query Rewriting
    - Hybrid Retrieval + Reranking
    - Grounded Generation + Citations
    """

    def __init__(
        self,
        collection_name: str = "rag_chunks_v1",
        llm_model: str = "qwen2.5:7b"
    ):
        self.query_engine = QueryEngine(
            collection_name=collection_name,
            enable_reranking=True,
            enable_query_rewriting=True
        )
        self.generator = GroundedGenerator(model=llm_model)
        logger.info("RAGPipeline initialized successfully")

    def ask(self, user_query: str, top_k: int = 8) -> Dict[str, Any]:
        """
        Main method to ask a question to the RAG system.
        """
        logger.info(f"Processing query: {user_query}")

        # Step 1 & 2: Retrieval (includes rewriting + reranking)
        retrieved_chunks = self.query_engine.query(user_query, limit=top_k)

        if not retrieved_chunks:
            return {
                "query": user_query,
                "answer": "I don't have enough relevant information in the documents to answer this question.",
                "citations": [],
                "grounded": False,
                "chunks_used": 0
            }

        # Step 3: Grounded Generation with Citations
        result = self.generator.generate(user_query, retrieved_chunks)

        return {
            "query": user_query,
            "answer": result["answer"],
            "citations": result.get("citations", []),
            "grounded": result.get("grounded", False),
            "chunks_used": len(retrieved_chunks)
        }

    def retrieve_only(self, user_query: str, top_k: int = 8) -> List[Dict[str, Any]]:
        """Only retrieve chunks without generation (useful for debugging)"""
        return self.query_engine.query(user_query, limit=top_k)