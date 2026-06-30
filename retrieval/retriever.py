import logging
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Filter,
    FieldCondition,
    MatchValue,
    Prefetch,
    FusionQuery,
    Fusion
)

logger = logging.getLogger(__name__)


class Retriever:
    """
    Hybrid Search Retriever using Qdrant's modern query API.
    Works with collections that have named vectors: "dense" + "sparse".
    """

    def __init__(self, collection_name: str = "rag_chunks_v1"):
        self.client = QdrantClient(host="localhost", port=6333)
        self.collection_name = collection_name
        logger.info(f"Retriever initialized for collection: {collection_name}")

    def hybrid_search(
        self,
        query_dense: List[float],
        query_sparse: Optional[Dict[str, Any]] = None,
        limit: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search (dense + sparse) using Prefetch + Fusion.
        
        query_sparse should be in format:
        {"indices": [...], "values": [...]}
        """
        query_filter = self._build_filter(filter_dict)

        try:
            prefetch = [
                Prefetch(
                    query=query_dense,
                    using="dense",
                    limit=limit * 2,
                    filter=query_filter
                )
            ]

            # Add sparse prefetch only if sparse vector is provided
            if query_sparse:
                prefetch.append(
                    Prefetch(
                        query=query_sparse,
                        using="sparse",
                        limit=limit * 2,
                        filter=query_filter
                    )
                )

            results = self.client.query_points(
                collection_name=self.collection_name,
                prefetch=prefetch,
                query=FusionQuery(fusion=Fusion.RRF),  # Reciprocal Rank Fusion
                limit=limit,
                with_payload=True,
            )

            return [point.payload for point in results.points]

        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            return []

    def _build_filter(self, filter_dict: Optional[Dict[str, Any]]) -> Optional[Filter]:
        if not filter_dict:
            return None

        conditions = [
            FieldCondition(key=key, match=MatchValue(value=value))
            for key, value in filter_dict.items()
        ]
        return Filter(must=conditions)

    def dense_search(
        self,
        query_vector: List[float],
        limit: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Simple dense-only search"""
        query_filter = self._build_filter(filter_dict)

        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            using="dense",
            limit=limit,
            with_payload=True,
            query_filter=query_filter
        )
        return [hit.payload for hit in results]