# src/ingestion/document_loader_v2.py
import os
import logging
from typing import List
from pathlib import Path

from ingestion.schema_normalizer import LlamaParseJSONNormalizer
from ingestion.figure_enricher import FigureVisionEnricher
from ingestion.cache_manager import IngestionCache
from models.document_element import DocumentElement

logger = logging.getLogger(__name__)


class DocumentLoaderV2:
    """
    Step 8: Full pipeline with Semantic Table Understanding
    """

    SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".pptx", ".html", ".htm"}

    def __init__(
        self,
        llama_parse_api_key: str,
        vision_model: str = "qwen2.5-vl:7b",
        enable_cache: bool = True,
        cache_dir: str = "cache/ingestion"
    ):
        self.normalizer = LlamaParseJSONNormalizer(api_key=llama_parse_api_key)
        self.figure_enricher = FigureVisionEnricher(vision_model=vision_model)
        self.table_enricher = TableSemanticEnricher()
        self.cache = IngestionCache(cache_dir=cache_dir) if enable_cache else None
        self.parser_version = "v2-step8"

    def load_file(self, file_path: str) -> List[DocumentElement]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = Path(file_path).suffix.lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            return []

        # Check cache first
        if self.cache:
            cached = self.cache.get(file_path, self.parser_version)
            if cached:
                return cached

        logger.info(f"Processing {ext.upper()}: {file_path}")

        # 1. Parse & Normalize
        elements = self.normalizer.parse_and_normalize(file_path)

        # 2. Enrich figures with vision
        elements = self.figure_enricher.enrich_figures(elements)

        # 3. Add semantic understanding to tables (NEW in Step 8)
        elements = self.table_enricher.enrich_tables(elements)

        # Save to cache
        if self.cache:
            self.cache.set(file_path, self.parser_version, elements)

        return elements

    def load_directory(self, directory: str, recursive: bool = True) -> List[DocumentElement]:
        all_elements = []
        for root, _, files in os.walk(directory):
            for file in files:
                if Path(file).suffix.lower() in self.SUPPORTED_EXTENSIONS:
                    path = os.path.join(root, file)
                    elements = self.load_file(path)
                    all_elements.extend(elements)
        return all_elements