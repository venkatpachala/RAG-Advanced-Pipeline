import logging
from typing import List
from ingestion.chunking.chunk import Chunk
from ingestion.embedding.embedder import Embedder
from ingestion.embedding.vector_store import VectorStore

logger = logging.getLogger(__name__)


class EmbeddingPipeline:
    def __init__(
        self,
        embedding_model: str = "BAAI/bge-m3",
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        collection_name: str = "rag_chunks"
    ):
        self.embedder = Embedder(model_name=embedding_model)
        self.vector_store = VectorStore(host=qdrant_host, port=qdrant_port)
        self.collection_name = collection_name

        # Create collection if not exists
        self.vector_store.create_collection(self.collection_name)

    def embed_and_store(self, chunks: List[Chunk]):
        """Embed chunks and store them in Qdrant."""
        if not chunks:
            logger.warning("No chunks to embed.")
            return

        logger.info(f"Embedding {len(chunks)} chunks...")

        texts = [chunk.text for chunk in chunks]
        embeddings = self.embedder.embed_texts(texts)

        points = []
        for i, chunk in enumerate(chunks):
            points.append({
                "id": chunk.chunk_id,
                "vector": embeddings[i],
                "payload": {
                    "text": chunk.text,
                    **chunk.metadata
                }
            })

        self.vector_store.upsert_points(self.collection_name, points)
        logger.info(f"Successfully stored {len(chunks)} chunks in Qdrant.")