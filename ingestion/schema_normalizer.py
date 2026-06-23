# src/ingestion/schema_normalizer.py
import logging
from typing import List, Dict, Any
from pathlib import Path
from datetime import datetime

from llama_parse import LlamaParse
from src.models.document_element import DocumentElement

logger = logging.getLogger(__name__)


class LlamaParseJSONNormalizer:
    """
    Step 5: Section Relationships + Rich Metadata + Quality Scoring
    """

    def __init__(self, api_key: str):
        self.parser = LlamaParse(
            api_key=api_key,
            result_type="json",
            parsing_instruction=(
                "Extract text, tables, section headers, figures, and document structure. "
                "Identify headings and their levels clearly."
            ),
        )

    def parse_and_normalize(self, file_path: str) -> List[DocumentElement]:
        logger.info(f"[Step 5] Parsing with relationships & metadata: {file_path}")

        try:
            json_result = self.parser.get_json_result(file_path)
            if not json_result:
                return []

            pages = json_result[0].get("pages", [])
            all_elements: List[DocumentElement] = []

            for page in pages:
                page_num = page.get("page", 0)
                all_elements.extend(self._process_page(page, file_path, page_num))

            return all_elements

        except Exception as e:
            logger.error(f"Failed to parse {file_path}: {e}")
            return []

    def _process_page(self, page: Dict[str, Any], source_file: str, page_number: int) -> List[DocumentElement]:
        elements = []
        items = page.get("items", [])
        current_section_path: List[str] = []
        element_index = 0

        for idx, item in enumerate(items):
            item_type = item.get("type", "").lower()
            content = item.get("content") or item.get("text", "")
            metadata = item.get("metadata", {})

            element_index += 1

            # === SECTION HEADERS ===
            if item_type in ["heading", "header", "title"]:
                level = metadata.get("level", 1)
                current_section_path = current_section_path[:level-1] + [content]

                element = DocumentElement(
                    content_type="section_header" if level == 1 else "subsection_header",
                    text=content,
                    source_file=source_file,
                    document_type=Path(source_file).suffix.lower(),
                    page_number=page_number,
                    section_path=current_section_path.copy(),
                    relationships={"level": level, "parent_section": current_section_path[-2] if len(current_section_path) > 1 else None},
                    metadata=self._build_rich_metadata(metadata, element_index, source_file, page_number),
                    quality_score=self._calculate_quality_score(content, item_type)
                )
                elements.append(element)

            # === TABLES ===
            elif item_type == "table":
                table_data = self._parse_table(item)
                element = DocumentElement(
                    content_type="table",
                    text=table_data.get("as_markdown", ""),
                    structured_content=table_data,
                    source_file=source_file,
                    document_type=Path(source_file).suffix.lower(),
                    page_number=page_number,
                    section_path=current_section_path.copy(),
                    relationships={"parent_section": current_section_path[-1] if current_section_path else None},
                    metadata=self._build_rich_metadata(metadata, element_index, source_file, page_number),
                    quality_score=self._calculate_quality_score(str(table_data), item_type)
                )
                elements.append(element)

            # === FIGURES ===
            elif item_type in ["image", "figure", "chart", "diagram"]:
                figure_data = self._parse_figure_with_context(item, items, idx, page_number)
                element = DocumentElement(
                    content_type="figure",
                    text=figure_data.get("caption", ""),
                    structured_content=figure_data,
                    source_file=source_file,
                    document_type=Path(source_file).suffix.lower(),
                    page_number=page_number,
                    section_path=current_section_path.copy(),
                    caption=figure_data.get("caption"),
                    nearby_context=figure_data.get("nearby_context"),
                    relationships={"parent_section": current_section_path[-1] if current_section_path else None},
                    metadata=self._build_rich_metadata(metadata, element_index, source_file, page_number),
                    quality_score=self._calculate_quality_score(figure_data.get("caption", ""), item_type)
                )
                elements.append(element)

            # === TEXT ===
            elif item_type in ["text", "paragraph"]:
                element = DocumentElement(
                    content_type="text",
                    text=content,
                    source_file=source_file,
                    document_type=Path(source_file).suffix.lower(),
                    page_number=page_number,
                    section_path=current_section_path.copy(),
                    relationships={"parent_section": current_section_path[-1] if current_section_path else None},
                    metadata=self._build_rich_metadata(metadata, element_index, source_file, page_number),
                    quality_score=self._calculate_quality_score(content, item_type)
                )
                elements.append(element)

        return elements

    # ==================== HELPERS ====================

    def _build_rich_metadata(self, item_metadata: Dict, element_index: int, source_file: str, page_number: int) -> Dict[str, Any]:
        """Build rich, consistent metadata for every element."""
        return {
            "element_index_on_page": element_index,
            "source_file_name": Path(source_file).name,
            "full_source_path": source_file,
            "page_number": page_number,
            "extraction_timestamp": datetime.utcnow().isoformat(),
            "parser_version": "v2-step5",
            "original_item_metadata": item_metadata,
        }

    def _calculate_quality_score(self, content: str, item_type: str) -> float:
        """Simple quality scoring (can be improved later)."""
        if not content:
            return 0.3

        score = 0.5
        length = len(content)

        if item_type == "table":
            score = 0.85 if length > 50 else 0.6
        elif item_type in ["figure", "image"]:
            score = 0.75
        elif item_type in ["heading", "header"]:
            score = 0.9
        else:
            if length > 300:
                score = 0.85
            elif length > 100:
                score = 0.75
            else:
                score = 0.6

        return round(score, 2)

    def _parse_table(self, table_item: Dict[str, Any]) -> Dict[str, Any]:
        headers = table_item.get("headers", [])
        rows = table_item.get("rows", [])
        return {
            "headers": headers,
            "rows": rows,
            "num_rows": len(rows),
            "num_columns": len(headers),
            "as_markdown": table_item.get("markdown", ""),
        }

    def _parse_figure_with_context(self, figure_item, all_items, current_index, page_number):
        caption = figure_item.get("content") or figure_item.get("text", "")
        image_path = figure_item.get("image_path") or figure_item.get("path")

        nearby_context = ""
        if current_index > 0:
            prev = all_items[current_index - 1]
            if prev.get("type") in ["text", "paragraph"]:
                nearby_context += prev.get("content", "") + " "
        if current_index + 1 < len(all_items):
            next_item = all_items[current_index + 1]
            if next_item.get("type") in ["text", "paragraph"]:
                nearby_context += next_item.get("content", "")

        return {
            "caption": caption,
            "image_path": image_path,
            "page_number": page_number,
            "nearby_context": nearby_context.strip(),
        }