import logging
from typing import List, Dict, Any, Optional
from ingestion.retrieval.retriever import Retriever
from ingestion.embedding.embedder import Embedder
from ingestion.retrieval.query_rewriter import QueryRewriter
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)


class QueryEngine:
    """
    Advanced Query Engine with Hybrid Search + Reranking.
    Uses sentence-transformers CrossEncoder for stable reranking.
    """

    def __init__(
        self,
        collection_name: str = "rag_chunks_v1",
        enable_reranking: bool = True,
        enable_query_rewriting: bool = True
    ):
        self.retriever = Retriever(collection_name=collection_name)
        self.embedder = Embedder()
        self.rewriter = QueryRewriter() if enable_query_rewriting else None
        self.enable_reranking = enable_reranking
        self.enable_query_rewriting = enable_query_rewriting

        if enable_reranking:
            try:
                self.reranker = CrossEncoder('BAAI/bge-reranker-base', max_length=512)
                logger.info("Reranker initialized successfully (sentence-transformers)")
            except Exception as e:
                logger.warning(f"Failed to load reranker: {e}")
                self.enable_reranking = False

    def query(
        self,
        user_query: str,
        limit: int = 8,
        use_rewriting: bool = True,
        rerank_top_k: int = 20,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        if not user_query or not user_query.strip():
            logger.warning("Empty query provided.")
            return []

        # Step 1: Query Rewriting (optional)
        final_query = user_query
        if self.enable_query_rewriting and use_rewriting and self.rewriter:
            final_query = self.rewriter.rewrite_query(user_query)

        # Step 2: Generate embeddings
        try:
            query_embedding = self.embedder.embed_texts([final_query])[0]
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            return []

        dense_vector = query_embedding.get("dense")
        sparse_vector = query_embedding.get("sparse")

        if not dense_vector:
            logger.error("Dense vector missing.")
            return []

        # Step 3: Hybrid Search
        candidates = self.retriever.hybrid_search(
            query_dense=dense_vector,
            query_sparse=sparse_vector,
            limit=rerank_top_k,
            filter_dict=filter_dict
        )

        if not candidates:
            return []

        # Step 4: Reranking (using sentence-transformers)
        if self.enable_reranking and len(candidates) > 1:
            try:
                pairs = [[final_query, item.get("text", "")] for item in candidates]
                scores = self.reranker.predict(pairs)

                ranked_results = sorted(
                    zip(candidates, scores), 
                    key=lambda x: x[1], 
                    reverse=True
                )
                final_results = [item for item, score in ranked_results[:limit]]
            except Exception as e:
                logger.warning(f"Reranking failed: {e}. Using original results.")
                final_results = candidates[:limit]
        else:
            final_results = candidates[:limit]

        logger.info(f"Retrieved {len(final_results)} chunks for query")
        return final_results