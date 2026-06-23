# src/ingestion/figure_enricher.py
import logging
import ollama
from typing import List
from models.document_element import DocumentElement

logger = logging.getLogger(__name__)


class FigureVisionEnricher:
    """
    Step 4: Enriches figures using local vision model.
    Passes image + caption + nearby context for better descriptions.
    """

    def __init__(self, vision_model: str = "qwen2.5-vl:7b"):
        self.vision_model = vision_model

    def enrich_figures(self, elements: List[DocumentElement]) -> List[DocumentElement]:
        """Enrich all figure elements with vision-generated summaries."""
        enriched_elements = []

        for element in elements:
            if element.content_type == "figure":
                enriched_element = self._enrich_single_figure(element)
                enriched_elements.append(enriched_element)
            else:
                enriched_elements.append(element)

        return enriched_elements

    def _enrich_single_figure(self, element: DocumentElement) -> DocumentElement:
        """Generate vision summary for one figure."""
        structured = element.structured_content or {}
        image_path = structured.get("image_path")
        caption = element.caption or ""
        nearby_context = element.nearby_context or ""

        # If no image path, we still store what we have
        if not image_path or not isinstance(image_path, str):
            logger.warning(f"No image path found for figure on page {element.page_number}")
            element.vision_summary = "No image available for description."
            return element

        try:
            # Build rich prompt with context
            prompt = self._build_vision_prompt(caption, nearby_context)

            with open(image_path, "rb") as f:
                image_data = f.read()

            response = ollama.chat(
                model=self.vision_model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                        "images": [image_data]
                    }
                ],
                options={"temperature": 0.3}
            )

            vision_summary = response["message"]["content"].strip()
            element.vision_summary = vision_summary

            logger.info(f"Generated vision summary for figure on page {element.page_number}")

        except Exception as e:
            logger.error(f"Vision enrichment failed: {e}")
            element.vision_summary = "Vision model failed to describe this figure."

        return element

    def _build_vision_prompt(self, caption: str, nearby_context: str) -> str:
        """Create a rich prompt using caption and nearby text."""
        prompt = "Describe this figure, chart, or diagram in detail. "

        if caption:
            prompt += f"The caption is: '{caption}'. "

        if nearby_context:
            prompt += f"Relevant surrounding text: '{nearby_context[:500]}'. "

        prompt += (
            "Explain what the figure shows, key components, trends, or relationships. "
            "Be precise and structured."
        )
        return prompt