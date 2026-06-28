import logging
from typing import List
from ingestion.chunking.chunk import Chunk
from ingestion.embedding.embedder import Embedder
from ingestion.embedding.vector_store import VectorStore

logger = logging.getLogger(__name__)


class EmbeddingPipeline:
    def __init__(self, embedding_model: str = "BAAI/bge-m3", collection_name: str = "rag_chunks_v1"):
        self.embedder = Embedder(model_name=embedding_model)
        self.vector_store = VectorStore()
        self.collection_name = collection_name

        self.vector_store.create_collection(self.collection_name)
        self.vector_store.create_payload_indexes(self.collection_name)

    def embed_and_store(self, chunks: List[Chunk]):
        if not chunks:
            return

        texts = [chunk.text for chunk in chunks]
        embeddings = self.embedder.embed_texts(texts)

        points = []
        for i, chunk in enumerate(chunks):
            points.append({
                "id": chunk.chunk_id,
                "vector": embeddings[i],
                "payload": {"text": chunk.text, **chunk.metadata}
            })

        self.vector_store.upsert_points(self.collection_name, points)