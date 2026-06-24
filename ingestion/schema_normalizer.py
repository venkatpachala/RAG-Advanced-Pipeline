# src/ingestion/schema_normalizer.py
import logging
import re
from typing import List, Dict, Any
from pathlib import Path
from datetime import datetime

from llama_parse import LlamaParse
from models.document_element import DocumentElement

logger = logging.getLogger(__name__)


class LlamaParseJSONNormalizer:
    """
    Step 5: Section Relationships + Rich Metadata + Quality Scoring
    """

    def __init__(self, api_key: str, image_dir: str = "cache/ingestion_images"):
        self.image_dir = Path(image_dir)
        self.parser = LlamaParse(
            api_key=api_key,
            result_type="json",
            aggressive_table_extraction=True,
            extract_charts=True,
            extract_layout=True,
            save_images=True,
            specialized_image_parsing=True,
            system_prompt=(
                "Extract text, tables, section headers, figures, and document structure. "
                "Identify headings and their levels clearly. Preserve tables as table "
                "elements and expose figures, diagrams, charts, and images as image assets."
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
            image_path_map = self._download_image_assets(json_result, file_path)

            for page in pages:
                page_num = page.get("page", 0)
                all_elements.extend(self._process_page(page, file_path, page_num, image_path_map))

            return all_elements

        except Exception as e:
            logger.error(f"Failed to parse {file_path}: {e}")
            return []

    def _process_page(
        self,
        page: Dict[str, Any],
        source_file: str,
        page_number: int,
        image_path_map: Dict[str, str],
    ) -> List[DocumentElement]:
        elements = []
        items = page.get("items", [])
        current_section_path: List[str] = []
        element_index = 0

        for idx, item in enumerate(items):
            item_type = item.get("type", "").lower()
            content = item.get("content") or item.get("text") or item.get("value") or item.get("md") or ""
            metadata = item.get("metadata", {})

            element_index += 1

            # === SECTION HEADERS ===
            if item_type in ["heading", "header", "title"]:
                level = metadata.get("level") or item.get("lvl") or 1
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

        elements.extend(
            self._extract_visual_asset_elements(
                page=page,
                source_file=source_file,
                page_number=page_number,
                section_path=current_section_path.copy(),
                image_path_map=image_path_map,
                start_index=element_index,
            )
        )
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
            "parser_version": "v2-step5.1",
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
        if not headers and rows:
            headers = rows[0]
        return {
            "headers": headers,
            "rows": rows,
            "num_rows": len(rows),
            "num_columns": len(headers),
            "as_markdown": table_item.get("markdown") or table_item.get("md") or table_item.get("value") or "",
        }

    def _parse_figure_with_context(self, figure_item, all_items, current_index, page_number):
        caption = figure_item.get("content") or figure_item.get("text") or figure_item.get("value") or figure_item.get("md") or ""
        image_path = figure_item.get("image_path") or figure_item.get("path")

        nearby_context = ""
        if current_index > 0:
            prev = all_items[current_index - 1]
            if prev.get("type") in ["text", "paragraph"]:
                nearby_context += (prev.get("content") or prev.get("text") or prev.get("value") or prev.get("md") or "") + " "
        if current_index + 1 < len(all_items):
            next_item = all_items[current_index + 1]
            if next_item.get("type") in ["text", "paragraph"]:
                nearby_context += next_item.get("content") or next_item.get("text") or next_item.get("value") or next_item.get("md") or ""

        return {
            "caption": caption,
            "image_path": image_path,
            "page_number": page_number,
            "nearby_context": nearby_context.strip(),
        }

    def _download_image_assets(self, json_result: List[Dict[str, Any]], source_file: str) -> Dict[str, str]:
        download_dir = self.image_dir / Path(source_file).stem
        download_dir.mkdir(parents=True, exist_ok=True)

        try:
            self.parser.get_images(json_result, str(download_dir))
        except Exception as exc:
            logger.warning("Could not download LlamaParse image assets: %s", exc)

        return {path.name: str(path) for path in download_dir.iterdir() if path.is_file()}

    def _extract_visual_asset_elements(
        self,
        page: Dict[str, Any],
        source_file: str,
        page_number: int,
        section_path: List[str],
        image_path_map: Dict[str, str],
        start_index: int,
    ) -> List[DocumentElement]:
        elements: List[DocumentElement] = []
        seen_names = set()

        for image in page.get("images") or []:
            name = image.get("name") or ""
            if not self._is_primary_visual_asset(image) or name in seen_names:
                continue

            seen_names.add(name)
            caption = self._find_nearby_caption(page.get("items", []), image)
            nearby_context = self._find_nearby_text(page.get("items", []), image)
            structured_content = {
                "image_name": name,
                "image_path": image_path_map.get(name),
                "bbox": {
                    "x": image.get("x"),
                    "y": image.get("y"),
                    "width": image.get("width"),
                    "height": image.get("height"),
                },
                "asset_type": image.get("type") or "visual_asset",
            }

            elements.append(
                DocumentElement(
                    content_type="figure",
                    text=caption or name,
                    structured_content=structured_content,
                    source_file=source_file,
                    document_type=Path(source_file).suffix.lower(),
                    page_number=page_number,
                    section_path=section_path.copy(),
                    caption=caption,
                    nearby_context=nearby_context,
                    relationships={"parent_section": section_path[-1] if section_path else None},
                    metadata=self._build_rich_metadata(image, start_index + len(elements) + 1, source_file, page_number),
                    quality_score=self._calculate_quality_score(caption or name, "figure"),
                )
            )

        for chart in page.get("charts") or []:
            name = chart.get("name") or ""
            if not name or name in seen_names:
                continue

            seen_names.add(name)
            caption = self._find_nearby_caption(page.get("items", []), chart)
            structured_content = {
                "image_name": name,
                "image_path": image_path_map.get(name),
                "bbox": {
                    "x": chart.get("x"),
                    "y": chart.get("y"),
                    "width": chart.get("width"),
                    "height": chart.get("height"),
                },
                "asset_type": "chart",
            }
            elements.append(
                DocumentElement(
                    content_type="figure",
                    text=caption or name,
                    structured_content=structured_content,
                    source_file=source_file,
                    document_type=Path(source_file).suffix.lower(),
                    page_number=page_number,
                    section_path=section_path.copy(),
                    caption=caption,
                    nearby_context=self._find_nearby_text(page.get("items", []), chart),
                    relationships={"parent_section": section_path[-1] if section_path else None},
                    metadata=self._build_rich_metadata(chart, start_index + len(elements) + 1, source_file, page_number),
                    quality_score=self._calculate_quality_score(caption or name, "chart"),
                )
            )

        return elements

    def _is_primary_visual_asset(self, image: Dict[str, Any]) -> bool:
        name = image.get("name") or ""
        return name.startswith(("img_", "chart_"))

    def _find_nearby_caption(self, items: List[Dict[str, Any]], visual: Dict[str, Any]) -> str:
        visual_y = visual.get("y")
        caption_candidates = []
        for item in items:
            text = self._item_text(item)
            if not text:
                continue
            item_type = (item.get("type") or "").lower()
            if item_type == "caption" or re.search(r"\b(fig\.?|figure|chart)\s*\d*", text, re.IGNORECASE):
                distance = self._vertical_distance(visual_y, item)
                caption_candidates.append((distance, text))

        if not caption_candidates:
            return ""

        caption_candidates.sort(key=lambda candidate: candidate[0])
        return caption_candidates[0][1]

    def _find_nearby_text(self, items: List[Dict[str, Any]], visual: Dict[str, Any]) -> str:
        visual_y = visual.get("y")
        text_candidates = []
        for item in items:
            if (item.get("type") or "").lower() not in {"text", "paragraph", "heading"}:
                continue
            text = self._item_text(item)
            if text:
                text_candidates.append((self._vertical_distance(visual_y, item), text))

        text_candidates.sort(key=lambda candidate: candidate[0])
        return " ".join(text for _, text in text_candidates[:3])[:1000]

    def _vertical_distance(self, visual_y: Any, item: Dict[str, Any]) -> float:
        if visual_y is None:
            return 999999.0
        bbox = item.get("bBox") or item.get("bbox") or {}
        item_y = bbox.get("y")
        if item_y is None:
            return 999999.0
        return abs(float(item_y) - float(visual_y))

    def _item_text(self, item: Dict[str, Any]) -> str:
        return item.get("content") or item.get("text") or item.get("value") or item.get("md") or ""
