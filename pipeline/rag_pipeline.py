import logging
from typing import Dict, Any, List
from observability import log_stage, generate_request_id
from retrieval.query_engine import QueryEngine
from generation.grounded_generator import GroundedGenerator

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Production-oriented RAG Pipeline with structured logging and observability.
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
        logger.info("RAGPipeline initialized")

    def ask(self, user_query: str, top_k: int = 8) -> Dict[str, Any]:
        """
        Main method that handles the full RAG flow with logging.
        """
        request_id = generate_request_id()

        # Log the incoming request
        logger.info({
            "event": "request_received",
            "request_id": request_id,
            "query": user_query
        })

        # === Stage 1: Query Rewriting ===
        with log_stage("query_rewriting", {
            "request_id": request_id,
            "original_query": user_query
        }):
            rewritten_query = self.query_engine.rewriter.rewrite_query(user_query) \
                if self.query_engine.rewriter else user_query

        # === Stage 2: Retrieval ===
        with log_stage("retrieval", {
            "request_id": request_id,
            "query": rewritten_query
        }):
            retrieved_chunks = self.query_engine.query(rewritten_query, limit=top_k)

        # === Stage 3: Generation ===
        with log_stage("generation", {
            "request_id": request_id,
            "num_chunks": len(retrieved_chunks)
        }):
            result = self.generator.generate(rewritten_query, retrieved_chunks)

        # Final response logging
        logger.info({
            "event": "request_completed",
            "request_id": request_id,
            "grounded": result.get("grounded", False),
            "chunks_used": result.get("chunks_used", 0),
            "answer_length": len(result.get("answer", ""))
        })

        return {
            "query": user_query,
            "rewritten_query": rewritten_query,
            "answer": result["answer"],
            "citations": result.get("citations", []),
            "grounded": result.get("grounded", False),
            "chunks_used": result.get("chunks_used", 0),
            "request_id": request_id
        }

    def retrieve_only(self, user_query: str, top_k: int = 8) -> List[Dict[str, Any]]:
        """Only retrieve chunks (useful for debugging)"""
        return self.query_engine.query(user_query, limit=top_k)