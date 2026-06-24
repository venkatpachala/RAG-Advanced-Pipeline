# src/ingestion/table_semantic_enricher.py
import logging
from typing import List
from models.document_element import DocumentElement

logger = logging.getLogger(__name__)


class TableSemanticEnricher:
    """
    Improved semantic table classification.
    Now works better with the enhanced table parsing from LlamaParse.
    """

    def enrich_tables(self, elements: List[DocumentElement]) -> List[DocumentElement]:
        for element in elements:
            if element.content_type == "table":
                self._classify_table(element)
        return elements

    def _classify_table(self, element: DocumentElement):
        structured = element.structured_content or {}
        headers = [str(h).lower() for h in structured.get("headers", [])]
        rows = structured.get("rows", [])
        header_text = " ".join(headers)

        table_type = "general"

        # Time series detection
        time_keywords = ["year", "month", "date", "quarter", "time", "epoch", "step"]
        if any(kw in header_text for kw in time_keywords):
            table_type = "time_series"

        # Financial / Performance tables
        elif any(kw in header_text for kw in ["accuracy", "bram", "lut", "power", "latency", "resource"]):
            table_type = "performance_comparison"

        # Comparison tables
        elif len(headers) >= 3 and len(rows) >= 3:
            table_type = "comparison"

        # Key-value style tables
        elif len(headers) <= 2:
            table_type = "key_value"

        structured["table_type"] = table_type
        element.structured_content = structured
        element.metadata["table_type"] = table_type

        logger.info(f"Table classified as: {table_type} (Page {element.page_number})")