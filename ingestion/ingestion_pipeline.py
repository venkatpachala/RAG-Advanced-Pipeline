import logging
from typing import List
from pathlib import Path

from ingestion.schema_normalizer import LlamaParseJSONNormalizer
from ingestion.image_visualizer import ImageVisionPipeline          # ← Fixed import
from ingestion.chunking.hierarchical_chunker import HierarchicalChunker
from ingestion.embedding.embedding_pipeline import EmbeddingPipeline
from ingestion.table_semantic_enricher import TableSemanticEnricher
from ingestion.cache_manager import IngestionCache
from models.document_element import DocumentElement

logger = logging.getLogger(__name__)


class IngestionPipeline:
    SUPPORTED_FORMATS = {".pdf", ".docx", ".pptx", ".html", ".htm"}

    def __init__(
        self,
        llama_parse_api_key: str,
        vision_model: str = "llama3.2-vision:latest",
        embedding_model: str = "BAAI/bge-m3",
        collection_name: str = "rag_chunks_v1",
        enable_cache: bool = True,
        cache_dir: str = "cache/ingestion",
        image_dir: str = "extracted_images",
        parser_version: str = "v4-full-pipeline"
    ):
        self.parser_version = parser_version
        self.enable_cache = enable_cache
        self.collection_name = collection_name

        self.normalizer = LlamaParseJSONNormalizer(api_key=llama_parse_api_key, image_dir=image_dir)
        self.image_vision = ImageVisionPipeline(vision_model=vision_model)
        self.chunker = HierarchicalChunker()
        self.embedder = EmbeddingPipeline(embedding_model=embedding_model, collection_name=collection_name)
        self.table_enricher = TableSemanticEnricher()
        self.cache = IngestionCache(cache_dir=cache_dir) if enable_cache else None

        logger.info("IngestionPipeline initialized successfully")

    def process_file(self, file_path: str) -> List[DocumentElement]:
        if not Path(file_path).exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = Path(file_path).suffix.lower()
        if ext not in self.SUPPORTED_FORMATS:
            logger.warning(f"Unsupported format: {ext}")
            return []

        if self.cache:
            cached = self.cache.get(file_path, self.parser_version)
            if cached:
                logger.info(f"Cache hit for {Path(file_path).name}")
                return cached

        logger.info(f"Processing file: {file_path}")

        # Phase 1: Normalizer + Vision
        elements = self.normalizer.parse_and_normalize(file_path)
        elements = self.image_vision.enrich_images_and_vision(elements, file_path)

        # Phase 2: Hierarchical Chunking
        chunks = self.chunker.chunk(elements, file_path)

        # Phase 3: Embedding + Storage
        self.embedder.embed_and_store(chunks)

        # Table enrichment (can stay here or move before chunking)
        elements = self.table_enricher.enrich_tables(elements)

        if self.cache:
            self.cache.set(file_path, self.parser_version, elements)

        logger.info(f"Finished: {Path(file_path).name} → {len(chunks)} chunks stored in Qdrant")
        return elements

    def process_directory(self, directory: str, recursive: bool = True):
        all_elements = []
        dir_path = Path(directory)
        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        pattern = "**/*" if recursive else "*"
        for file_path in dir_path.glob(pattern):
            if file_path.suffix.lower() in self.SUPPORTED_FORMATS:
                try:
                    elements = self.process_file(str(file_path))
                    all_elements.extend(elements)
                except Exception as e:
                    logger.error(f"Failed to process {file_path}: {e}")
        return all_elements

    def clear_cache(self):
        if self.cache:
            self.cache.clear()