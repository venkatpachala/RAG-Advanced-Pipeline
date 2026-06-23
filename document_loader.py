# src/ingestion/document_loader_v2.py
import os
import logging
from typing import List
from pathlib import Path

from src.ingestion.schema_normalizer import LlamaParseJSONNormalizer
from src.ingestion.figure_enricher import FigureVisionEnricher
from src.models.document_element import DocumentElement

logger = logging.getLogger(__name__)


class DocumentLoaderV2:
    """
    Step 4: Full ingestion with JSON parsing + Figure Vision Enrichment
    """

    def __init__(self, llama_parse_api_key: str, vision_model: str = "qwen2.5-vl:7b"):
        self.normalizer = LlamaParseJSONNormalizer(api_key=llama_parse_api_key)
        self.figure_enricher = FigureVisionEnricher(vision_model=vision_model)

    def load_file(self, file_path: str) -> List[DocumentElement]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = Path(file_path).suffix.lower()

        if ext != ".pdf":
            logger.warning(f"Only PDF is fully supported in current version. Got: {ext}")
            return []

        # Step 1: Parse and normalize
        elements = self.normalizer.parse_and_normalize(file_path)

        # Step 2: Enrich figures with vision model
        elements = self.figure_enricher.enrich_figures(elements)

        return elements

    def load_directory(self, directory: str) -> List[DocumentElement]:
        all_elements = []
        for root, _, files in os.walk(directory):
            for file in files:
                if file.lower().endswith(".pdf"):
                    path = os.path.join(root, file)
                    elements = self.load_file(path)
                    all_elements.extend(elements)
        return all_elements