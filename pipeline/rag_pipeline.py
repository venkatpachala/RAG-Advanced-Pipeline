import logging
from typing import Dict, Any, List, Optional

from observability import log_stage, generate_request_id
from retrieval.query_engine import QueryEngine
from generation.grounded_generator import GroundedGenerator

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Production-oriented RAG Pipeline with structured observability,
    request tracing, and stage-level latency tracking.
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

    def ask(
        self, 
        user_query: str, 
        top_k: int = 8
    ) -> Dict[str, Any]:
        """
        Main RAG pipeline method with full observability.
        Tracks query rewriting, retrieval, and generation stages with latency.
        """
        request_id: str = generate_request_id()

        # Log incoming request
        logger.info(
            "Request received",
            extra={
                "request_id": request_id,
                "event": "request_received",
                "query": user_query
            }
        )

        # === Stage 1: Query Rewriting ===
        with log_stage(
            "query_rewriting",
            request_id=request_id,
            original_query=user_query
        ):
            if self.query_engine.rewriter:
                rewritten_query = self.query_engine.rewriter.rewrite_query(user_query)
            else:
                rewritten_query = user_query

        # === Stage 2: Retrieval (Hybrid + Reranking) ===
        with log_stage(
            "retrieval",
            request_id=request_id,
            rewritten_query=rewritten_query,
            top_k=top_k
        ):
            retrieved_chunks: List[Dict[str, Any]] = self.query_engine.query(
                rewritten_query, 
                limit=top_k
            )

        # === Stage 3: Grounded Generation ===
        with log_stage(
            "generation",
            request_id=request_id,
            num_chunks=len(retrieved_chunks)
        ):
            result: Dict[str, Any] = self.generator.generate(
                user_query=rewritten_query,
                retrieved_chunks=retrieved_chunks,
                request_id=request_id
            )

        # Log completion
        logger.info(
            "Request completed",
            extra={
                "request_id": request_id,
                "event": "request_completed",
                "grounded": result.get("grounded", False),
                "chunks_used": result.get("chunks_used", 0),
                "answer_length": len(result.get("answer", "")),
                "citations_count": len(result.get("citations", []))
            }
        )

        return {
            "query": user_query,
            "rewritten_query": rewritten_query,
            "answer": result.get("answer", ""),
            "citations": result.get("citations", []),
            "grounded": result.get("grounded", False),
            "chunks_used": result.get("chunks_used", 0),
            "request_id": request_id
        }

    def retrieve_only(
        self, 
        user_query: str, 
        top_k: int = 8
    ) -> List[Dict[str, Any]]:
        """
        Retrieve chunks only (useful for debugging and evaluation).
        Does NOT go through generation.
        """
        return self.query_engine.query(user_query, limit=top_k)

    def health_check(self) -> Dict[str, Any]:
        """Simple health check for the pipeline."""
        return {
            "status": "healthy",
            "collection": self.query_engine.collection_name if hasattr(self.query_engine, "collection_name") else "unknown",
            "llm_model": self.generator.model if hasattr(self.generator, "model") else "unknown"
        }