# src/ingestion/figure_enricher.py
import logging
import ollama
from typing import List
from models.document_element import DocumentElement

logger = logging.getLogger(__name__)


class FigureVisionEnricher:
    def __init__(self, vision_model: str = "qwen2.5-vl:7b"):
        self.vision_model = vision_model

    def enrich_figures(self, elements: List[DocumentElement]) -> List[DocumentElement]:
        for element in elements:
            if element.content_type == "figure":
                self._enrich_single_figure(element)
        return elements

    def _enrich_single_figure(self, element: DocumentElement):
        structured = element.structured_content or {}
        image_path = structured.get("image_path")

        if not image_path or not Path(image_path).exists():
            element.vision_summary = "No valid image file available."
            return

        try:
            prompt = self._build_prompt(element.caption, element.nearby_context)

            with open(image_path, "rb") as f:
                image_data = f.read()

            response = ollama.chat(
                model=self.vision_model,
                messages=[{"role": "user", "content": prompt, "images": [image_data]}],
                options={"temperature": 0.3}
            )
            element.vision_summary = response["message"]["content"].strip()
            logger.info(f"Vision summary generated for page {element.page_number}")

        except Exception as e:
            logger.error(f"Vision failed: {e}")
            element.vision_summary = "Vision model failed to describe this figure."

    def _build_prompt(self, caption: str, context: str) -> str:
        prompt = "Describe this figure or diagram in detail. "
        if caption:
            prompt += f"Caption: {caption}. "
        if context:
            prompt += f"Context: {context[:400]}. "
        prompt += "Explain what it shows clearly."
        return prompt