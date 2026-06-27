import logging
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, SparseVectorParams, SparseIndexParams,
    PointStruct, Filter, FieldCondition, MatchValue, PayloadSchemaType
)

logger = logging.getLogger(__name__)


class VectorStore:
    def __init__(self, host: str = "localhost", port: int = 6333):
        self.client = QdrantClient(host=host, port=port)
        logger.info("Connected to Qdrant")

    def create_collection(self, collection_name: str):
        """Create collection with hybrid vector support"""
        try:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    "dense": VectorParams(
                        size=1024,
                        distance=Distance.COSINE,
                        on_disk=True
                    ),
                    "sparse": SparseVectorParams(
                        index=SparseIndexParams(on_disk=False)
                    )
                }
            )
            logger.info(f"Collection '{collection_name}' created with hybrid config")
        except Exception:
            logger.info(f"Collection '{collection_name}' already exists")

    def create_payload_indexes(self, collection_name: str):
        """Create indexes for fast filtering (Very Important)"""
        fields = ["element_type", "section_path", "chunk_level", "source_file"]
        
        for field in fields:
            try:
                self.client.create_payload_index(
                    collection_name=collection_name,
                    field_name=field,
                    field_schema=PayloadSchemaType.KEYWORD
                )
                logger.info(f"Created payload index for: {field}")
            except Exception as e:
                logger.warning(f"Index already exists or failed for {field}: {e}")

    def upsert_points(self, collection_name: str, points: List[Dict[str, Any]]):
        """Insert or update points"""
        qdrant_points = []
        for point in points:
            qdrant_points.append(
                PointStruct(
                    id=point["id"],
                    vector={
                        "dense": point["vector"]["dense"],
                        "sparse": point["vector"].get("sparse")
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
        """Basic dense vector search"""
        results = self.client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=limit,
            with_payload=True,
            query_filter=filter_conditions
        )
        return results