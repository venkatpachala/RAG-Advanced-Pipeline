# api/dependencies.py
import logging
from pipeline.rag_pipeline import RAGPipeline

logger = logging.getLogger(__name__)

_rag_pipeline = None

def get_rag_pipeline() -> RAGPipeline:
    """
    Returns a singleton RAGPipeline (lazy loading).
    The heavy models will only load on the first request.
    """
    global _rag_pipeline
    if _rag_pipeline is None:
        logger.info("Initializing RAG Pipeline for the first time...")
        _rag_pipeline = RAGPipeline(
            collection_name="rag_chunks_v1",
            llm_model="qwen2.5:7b"
        )
        logger.info("RAG Pipeline initialized successfully.")
    return _rag_pipeline