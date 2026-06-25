import logging
import asyncio
from typing import List, Dict, Any
from pathlib import Path
from datetime import datetime

from llama_parse import LlamaParse
from models.document_element import DocumentElement

logger = logging.getLogger(__name__)


class LlamaParseJSONNormalizer:
    """
    Fixed version using asyncio.run() for LlamaParse compatibility on Windows.
    """

    def __init__(self, api_key: str, image_dir: str = "extracted_images"):
        self.image_dir = Path(image_dir)
        self.parser = LlamaParse(
            api_key=api_key,
            result_type="json",
            aggressive_table_extraction=True,
            extract_charts=True,
            extract_layout=True,
            save_images=True,
            specialized_image_parsing=True,
        )

    def parse_and_normalize(self, file_path: str) -> List[DocumentElement]:
        """Synchronous wrapper"""
        return asyncio.run(self._parse_and_normalize_async(file_path))

    async def _parse_and_normalize_async(self, file_path: str) -> List[DocumentElement]:
        try:
            json_result = await self.parser.aget_json_result(file_path)
            if not json_result:
                return []

            pages = json_result[0].get("pages", [])
            all_elements: List[DocumentElement] = []

            # Download images
            image_files = await self._download_image_assets_async(json_result, file_path)
            available_images = list(image_files.values()) if image_files else []

            for page in pages:
                page_num = page.get("page", 0)
                all_elements.extend(
                    self._process_page(page, file_path, page_num, available_images)
                )

            return all_elements

        except Exception as e:
            logger.error(f"Failed to parse {file_path}: {e}")
            return []

    def _process_page(
        self,
        page: Dict[str, Any],
        source_file: str,
        page_number: int,
        available_images: List[str],
    ) -> List[DocumentElement]:
        # (Same as previous version - sequential assignment)
        elements = []
        items = page.get("items", [])
        current_section_path: List[str] = []
        element_index = 0

        for idx, item in enumerate(items):
            item_type = item.get("type", "").lower()
            content = item.get("content") or item.get("text") or item.get("value") or item.get("md") or ""
            metadata = item.get("metadata", {})

            element_index += 1

            if item_type in ["heading", "header", "title"]:
                level = metadata.get("level") or item.get("lvl") or 1
                current_section_path = current_section_path[:level-1] + [content]
                elements.append(DocumentElement(
                    content_type="section_header" if level == 1 else "subsection_header",
                    text=content,
                    source_file=source_file,
                    document_type=Path(source_file).suffix.lower(),
                    page_number=page_number,
                    section_path=current_section_path.copy(),
                    metadata=self._build_rich_metadata(metadata, element_index, source_file, page_number),
                    quality_score=self._calculate_quality_score(content, item_type)
                ))

            elif item_type == "table":
                table_data = self._parse_table(item)
                elements.append(DocumentElement(
                    content_type="table",
                    text=table_data.get("as_markdown", ""),
                    structured_content=table_data,
                    source_file=source_file,
                    document_type=Path(source_file).suffix.lower(),
                    page_number=page_number,
                    metadata=self._build_rich_metadata(metadata, element_index, source_file, page_number),
                    quality_score=self._calculate_quality_score(str(table_data), item_type)
                ))

            elif item_type in ["image", "figure", "chart", "diagram"]:
                figure_data = self._parse_figure_with_context(
                    figure_item=item,
                    all_items=items,
                    current_index=idx,
                    page_number=page_number,
                    available_images=available_images
                )
                elements.append(DocumentElement(
                    content_type="figure",
                    text=figure_data.get("caption", ""),
                    structured_content=figure_data,
                    source_file=source_file,
                    document_type=Path(source_file).suffix.lower(),
                    page_number=page_number,
                    caption=figure_data.get("caption"),
                    nearby_context=figure_data.get("nearby_context"),
                    metadata=self._build_rich_metadata(metadata, element_index, source_file, page_number),
                    quality_score=self._calculate_quality_score(figure_data.get("caption", ""), item_type)
                ))

            elif item_type in ["text", "paragraph"]:
                elements.append(DocumentElement(
                    content_type="text",
                    text=content,
                    source_file=source_file,
                    document_type=Path(source_file).suffix.lower(),
                    page_number=page_number,
                    metadata=self._build_rich_metadata(metadata, element_index, source_file, page_number),
                    quality_score=self._calculate_quality_score(content, item_type)
                ))

        return elements

    def _parse_figure_with_context(self, figure_item, all_items, current_index, page_number, available_images=None):
        caption = figure_item.get("content") or figure_item.get("text") or figure_item.get("value") or figure_item.get("md") or ""

        image_path = None
        if available_images and len(available_images) > 0:
            image_path = available_images.pop(0)

        return {
            "caption": caption,
            "image_path": image_path,
            "page_number": page_number,
            "nearby_context": ""
        }

    async def _download_image_assets_async(self, json_result, source_file):
        download_dir = self.image_dir / Path(source_file).stem
        download_dir.mkdir(parents=True, exist_ok=True)

        try:
            await self.parser.aget_images(json_result, str(download_dir))
        except Exception as exc:
            logger.warning("Could not download images: %s", exc)

        return {path.name: str(path) for path in download_dir.iterdir() if path.is_file()}

    def _build_rich_metadata(self, item_metadata, element_index, source_file, page_number):
        return {
            "element_index_on_page": element_index,
            "source_file_name": Path(source_file).name,
            "full_source_path": source_file,
            "page_number": page_number,
            "extraction_timestamp": datetime.utcnow().isoformat(),
            "parser_version": "v2-final",
        }

    def _calculate_quality_score(self, content, item_type):
        if not content: return 0.3
        length = len(content)
        if item_type == "table": return 0.85 if length > 50 else 0.6
        if item_type in ["figure", "image"]: return 0.75
        if item_type in ["heading", "header"]: return 0.9
        return 0.85 if length > 300 else (0.75 if length > 100 else 0.6)

    def _parse_table(self, table_item):
        headers = table_item.get("headers", [])
        rows = table_item.get("rows", [])
        if not headers and rows: headers = rows[0]
        return {
            "headers": headers, "rows": rows,
            "num_rows": len(rows), "num_columns": len(headers),
            "as_markdown": table_item.get("markdown") or ""
        }

    # Keep other methods if needed (they are not critical for now)
    def _extract_visual_asset_elements(self, *args, **kwargs):
        return []