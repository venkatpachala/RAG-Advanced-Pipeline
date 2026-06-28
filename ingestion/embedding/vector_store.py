import logging
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    SparseVectorParams,
    SparseIndexParams,
    PointStruct,
    PayloadSchemaType,
    SparseVector,
    Filter,
    FieldCondition,
    MatchValue
)

logger = logging.getLogger(__name__)


class VectorStore:
    def __init__(self, host: str = "localhost", port: int = 6333):
        self.client = QdrantClient(host=host, port=port)
        logger.info("Connected to Qdrant")

    def create_collection(
        self, 
        collection_name: str, 
        dense_dim: int = 1024
    ):
        """
        Create collection with separate dense and sparse vector configs.
        """
        try:
            if self.client.collection_exists(collection_name):
                logger.info(f"Collection '{collection_name}' already exists")
                return

            self.client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    "dense": VectorParams(
                        size=dense_dim,
                        distance=Distance.COSINE,
                        on_disk=True
                    )
                },
                sparse_vectors_config={
                    "sparse": SparseVectorParams(
                        index=SparseIndexParams(on_disk=False)
                    )
                }
            )
            logger.info(f"Collection '{collection_name}' created successfully")

        except Exception as e:
            logger.error(f"Failed to create collection '{collection_name}': {e}")
            raise  # Re-raise so the pipeline can stop

    def create_payload_indexes(self, collection_name: str):
        fields = ["element_type", "section_path", "chunk_level", "source_file"]

        for field in fields:
            try:
                self.client.create_payload_index(
                    collection_name=collection_name,
                    field_name=field,
                    field_schema=PayloadSchemaType.KEYWORD
                )
                logger.info(f"Created index for field: {field}")
            except Exception as e:
                logger.warning(f"Could not create index for {field}: {e}")

    def upsert_points(self, collection_name: str, points: List[Dict[str, Any]]):
        """
        Upsert points with proper sparse vector formatting.
        Expects:
        {
            "id": str/int,
            "vector": {
                "dense": List[float],
                "sparse": {"indices": List[int], "values": List[float]}
            },
            "payload": dict
        }
        """
        qdrant_points = []
        for point in points:
            vector = point["vector"]
            sparse_vector = None

            if vector.get("sparse"):
                sparse_data = vector["sparse"]
                sparse_vector = SparseVector(
                    indices=sparse_data["indices"],
                    values=sparse_data["values"]
                )

            qdrant_points.append(
                PointStruct(
                    id=point["id"],
                    vector={
                        "dense": vector["dense"],
                        "sparse": sparse_vector
                    },
                    payload=point["payload"]
                )
            )

        self.client.upsert(collection_name=collection_name, points=qdrant_points)
        logger.info(f"Upserted {len(points)} points into '{collection_name}'")

    def search(
        self,
        collection_name: str,
        query_vector: List[float],
        limit: int = 10,
        filter_conditions: Optional[Dict] = None
    ):
        """Basic dense search"""
        query_filter = None
        if filter_conditions:
            conditions = [
                FieldCondition(key=key, match=MatchValue(value=value))
                for key, value in filter_conditions.items()
            ]
            query_filter = Filter(must=conditions)

        return self.client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=limit,
            with_payload=True,
            query_filter=query_filter
        )