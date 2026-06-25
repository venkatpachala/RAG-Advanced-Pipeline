# ============================================================
# FILE: multimodal_ingestion.py
# ============================================================
import logging
import ollama
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

from llama_parse import LlamaParse
from models.document_element import DocumentElement

logger = logging.getLogger(__name__)


class MultimodalDocumentParser:
    """
    Modern LlamaParse + Vision Model pipeline.
    - LlamaParse handles text, tables, layout
    - Vision model handles figure/diagram description
    """

    def __init__(
        self,
        llama_cloud_api_key: str,
        vision_model: str = "llama3.2-vision:latest",
        image_output_dir: str = "extracted_images"
    ):
        self.vision_model = vision_model
        self.image_dir = Path(image_output_dir)

        # Recommended LlamaParse configuration (as per current docs)
        self.parser = LlamaParse(
            api_key=llama_cloud_api_key,
            result_type="json",                    # We need JSON for image handling
            aggressive_table_extraction=True,
            extract_charts=True,
            extract_layout=True,
            save_images=True,                      # Important for image extraction
            specialized_image_parsing=True,
            system_prompt="Extract text, tables, headings, figures, charts and diagrams with high accuracy."
        )

    def parse_document(self, file_path: str) -> List[DocumentElement]:
        """Main entry point: Parse document + enrich figures with vision model."""
        logger.info(f"Starting multimodal parsing: {file_path}")

        # Step 1: Parse with LlamaParse
        json_result = self.parser.get_json_result(file_path)
        if not json_result:
            return []

        # Step 2: Download all images from the document
        image_path_map = self._download_images(json_result, file_path)

        # Step 3: Process pages and create DocumentElements
        pages = json_result[0].get("pages", [])
        all_elements: List[DocumentElement] = []

        for page in pages:
            page_elements = self._process_page(page, file_path, image_path_map)
            all_elements.extend(page_elements)

        # Step 4: Enrich figures using vision model
        self._enrich_figures_with_vision(all_elements)

        logger.info(f"Parsing complete. Total elements: {len(all_elements)}")
        return all_elements

    def _download_images(self, json_result: List[Dict], source_file: str) -> Dict[str, str]:
        """Download images using LlamaParse's get_images method."""
        download_dir = self.image_dir / Path(source_file).stem
        download_dir.mkdir(parents=True, exist_ok=True)

        try:
            self.parser.get_images(json_result, str(download_dir))
            logger.info(f"Images downloaded to: {download_dir}")
        except Exception as e:
            logger.warning(f"Could not download images: {e}")

        # Return mapping of image name → local path
        return {
            path.name: str(path)
            for path in download_dir.iterdir()
            if path.is_file()
        }

    def _process_page(self, page: Dict[str, Any], source_file: str, image_path_map: Dict[str, str]) -> List[DocumentElement]:
        elements = []
        items = page.get("items", [])
        current_section_path: List[str] = []

        for idx, item in enumerate(items):
            item_type = item.get("type", "").lower()
            content = item.get("content") or item.get("text") or item.get("value") or ""

            if item_type in ["heading", "header", "title"]:
                level = item.get("metadata", {}).get("level", 1)
                current_section_path = current_section_path[:level-1] + [content]
                elements.append(self._create_element("section_header", content, source_file, page.get("page"), current_section_path))

            elif item_type == "table":
                elements.append(self._create_element("table", content, source_file, page.get("page"), current_section_path,
                                                     structured={"markdown": item.get("markdown", "")}))

            elif item_type in ["image", "figure", "chart", "diagram"]:
                figure_data = self._get_figure_data(item, items, idx, image_path_map)
                elements.append(self._create_element(
                    "figure",
                    figure_data.get("caption", ""),
                    source_file,
                    page.get("page"),
                    current_section_path,
                    structured=figure_data,
                    caption=figure_data.get("caption"),
                    nearby_context=figure_data.get("nearby_context")
                ))

            elif item_type in ["text", "paragraph"]:
                elements.append(self._create_element("text", content, source_file, page.get("page"), current_section_path))

        return elements

    def _get_figure_data(self, item, all_items, current_index, image_path_map):
        caption = item.get("content") or item.get("text") or ""
        image_path = item.get("image_path") or item.get("path")

        # Fallback image matching
        if not image_path and image_path_map:
            for name, path in image_path_map.items():
                if str(item.get("name", "")).lower() in name.lower():
                    image_path = path
                    break
            if not image_path and image_path_map:
                image_path = list(image_path_map.values())[0]

        # Get nearby context
        nearby_context = ""
        if current_index > 0:
            prev = all_items[current_index - 1]
            if prev.get("type") in ["text", "paragraph"]:
                nearby_context = prev.get("content") or prev.get("text") or ""

        return {
            "caption": caption,
            "image_path": image_path,
            "nearby_context": nearby_context.strip()
        }

    def _create_element(self, content_type, text, source_file, page_number, section_path,
                        structured=None, caption=None, nearby_context=None):
        return DocumentElement(
            content_type=content_type,
            text=text,
            source_file=source_file,
            page_number=page_number,
            section_path=section_path.copy() if section_path else [],
            structured_content=structured or {},
            caption=caption,
            nearby_context=nearby_context,
            metadata={"parsed_at": datetime.utcnow().isoformat()}
        )

    def _enrich_figures_with_vision(self, elements: List[DocumentElement]):
        """Send figure images to vision model and mix description back."""
        for element in elements:
            if element.content_type != "figure":
                continue

            image_path = element.structured_content.get("image_path")
            if not image_path or not Path(image_path).exists():
                element.vision_summary = "No image available"
                continue

            try:
                prompt = self._build_vision_prompt(element.caption, element.nearby_context)

                with open(image_path, "rb") as f:
                    image_bytes = f.read()

                response = ollama.chat(
                    model=self.vision_model,
                    messages=[{"role": "user", "content": prompt, "images": [image_bytes]}],
                    options={"temperature": 0.2}
                )

                vision_desc = response["message"]["content"].strip()

                # === Mix vision description into the element ===
                element.vision_summary = vision_desc

                # Add to main text (useful for embedding)
                if element.text:
                    element.text += f"\n\n[Vision Description]: {vision_desc}"
                else:
                    element.text = f"[Vision Description]: {vision_desc}"

                # Store separately in structured_content
                element.structured_content["vision_description"] = vision_desc

                logger.info(f"Vision enriched: Page {element.page_number}")

            except Exception as e:
                logger.error(f"Vision enrichment failed: {e}")
                element.vision_summary = "Vision model failed"

    def _build_vision_prompt(self, caption: str, context: str) -> str:
        prompt = (
            "Describe this figure, diagram or chart in detail. "
            "Explain what it shows, key components, flow, relationships, and any visible text. "
            "Be clear and structured."
        )
        if caption:
            prompt += f"\nCaption: {caption}"
        if context:
            prompt += f"\nContext: {context[:400]}"
        return prompt


# ============================================================
# USAGE EXAMPLE
# ============================================================
if __name__ == "__main__":
    import nest_asyncio
    nest_asyncio.apply()

    logging.basicConfig(level=logging.INFO)

    parser = MultimodalDocumentParser(
        llama_cloud_api_key="llx-P2TtlNbb0aHrbTeKlkDeVh4pZruvMxIjTgrvQxuMfFUzUsIK",   # ← Replace
        vision_model="llama3.2-vision:latest",
        image_output_dir="extracted_images"
    )

    elements = parser.parse_document("sample.pdf")

    # Example: Print enriched figures
    for elem in elements:
        if elem.content_type == "figure" and elem.vision_summary:
            print(f"\nPage {elem.page_number} | {elem.caption}")
            print(f"Vision: {elem.vision_summary[:200]}...")