import logging
from typing import List, Dict, Any
from FlagEmbedding import BGEM3FlagModel

logger = logging.getLogger(__name__)


class Embedder:
    """
    Embedding class using bge-m3 for dense + sparse embeddings.
    """

    def __init__(self, model_name: str = "BAAI/bge-m3", use_fp16: bool = True):
        logger.info(f"Loading embedding model: {model_name}")
        self.model = BGEM3FlagModel(model_name, use_fp16=use_fp16)
        logger.info("Embedding model loaded successfully")

    def embed_texts(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Generate both dense and sparse embeddings for a list of texts.
        """
        if not texts:
            return []

        logger.info(f"Embedding {len(texts)} texts...")

        output = self.model.encode(
            texts,
            batch_size=12,
            max_length=8192,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False
        )

        embeddings = []
        for i in range(len(texts)):
            embeddings.append({
                "dense": output['dense_vecs'][i].tolist(),
                "sparse": {
                    "indices": list(output['lexical_weights'][i].keys()),
                    "values": list(output['lexical_weights'][i].values())
                }
            })

        return embeddings

    def embed_single_text(self, text: str) -> Dict[str, Any]:
        """Embed a single text."""
        result = self.embed_texts([text])
        return result[0] if result else {}