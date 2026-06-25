# src/ingestion/ingestion_pipeline.py
import logging
from typing import List, Optional
from pathlib import Path

from ingestion.schema_normalizer import LlamaParseJSONNormalizer
from ingestion.figure_enricher import FigureVisionEnricher
from ingestion.table_semantic_enricher import TableSemanticEnricher
from ingestion.cache_manager import IngestionCache
from models.document_element import DocumentElement
from ingestion.figure_enricher import FigureVisionEnricher
from ingestion.image_visualizer import ImageVisionPipeline


logger = logging.getLogger(__name__)


class IngestionPipeline:
    """
    Clean, production-ready Ingestion Pipeline (Updated)
    """

    SUPPORTED_FORMATS = {".pdf", ".docx", ".pptx", ".html", ".htm"}

    def __init__(
        self,
        llama_parse_api_key: str,
        vision_model: str = "lama3.2-vision:latest",
        enable_cache: bool = True,
        cache_dir: str = "cache/ingestion",
        image_dir: str = "extracted_images",
        parser_version: str = "v2-final"
    ):
        self.parser_version = parser_version
        self.enable_cache = enable_cache

        # Initialize components
        self.normalizer = LlamaParseJSONNormalizer(
            api_key=llama_parse_api_key,
            image_dir=image_dir
        )
        self.figure_enricher = FigureVisionEnricher(vision_model=vision_model)
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

        # Check cache first
        if self.cache:
            cached = self.cache.get(file_path, self.parser_version)
            if cached:
                logger.info(f"Cache hit for {Path(file_path).name}")
                return cached

        logger.info(f"Processing file: {file_path}")

        # 1. Parse + Normalize (with image downloading)
        elements = self.normalizer.parse_and_normalize(file_path)
        image_vision = ImageVisionPipeline(vision_model="llama3.2-vision:latest")
        elements = image_vision.enrich_images_and_vision(elements, file_path)

        # 2. Enrich figures with vision model (only if images exist)
        elements = self.figure_enricher.enrich_figures(elements)

        # 3. Add semantic understanding to tables
        elements = self.table_enricher.enrich_tables(elements)
        enricher = FigureVisionEnricher(vision_model="llama3.2-vision:latest")
        elements = enricher.enrich_figures(elements)

        # 4. Save to cache
        if self.cache:
            self.cache.set(file_path, self.parser_version, elements)

        logger.info(f"Finished: {Path(file_path).name} → {len(elements)} elements")
        return elements
    
    

    def process_directory(
        self,
        directory: str,
        recursive: bool = True,
        file_extensions: Optional[List[str]] = None
    ) -> List[DocumentElement]:
        if file_extensions is None:
            file_extensions = list(self.SUPPORTED_FORMATS)

        all_elements: List[DocumentElement] = []
        dir_path = Path(directory)

        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        pattern = "**/*" if recursive else "*"
        for file_path in dir_path.glob(pattern):
            if file_path.suffix.lower() in file_extensions:
                try:
                    elements = self.process_file(str(file_path))
                    all_elements.extend(elements)
                except Exception as e:
                    logger.error(f"Failed to process {file_path}: {e}")

        return all_elements

    def clear_cache(self):
        if self.cache:
            self.cache.clear()
    
    