import logging
from pathlib import Path
from typing import List

import fitz  # PyMuPDF
from models.document_element import DocumentElement
from ingestion.figure_enricher import FigureVisionEnricher

logger = logging.getLogger(__name__)


class ImageVisionPipeline:
    """
    Dedicated pipeline for:
    - Image extraction (PyMuPDF)
    - Figure detection + caption + nearby context matching
    - Vision model enrichment
    """

    def __init__(self, vision_model: str = "llama3.2-vision:latest", image_dir: str = "extracted_images"):
        self.image_dir = Path(image_dir)
        self.vision_enricher = FigureVisionEnricher(vision_model=vision_model)

    def enrich_images_and_vision(
        self, 
        elements: List[DocumentElement], 
        file_path: str
    ) -> List[DocumentElement]:
        """
        Takes elements from main normalizer and enriches them with:
        - image_path (from PyMuPDF)
        - vision_summary
        """
        if not elements:
            return elements

        # Step 1: Extract images using PyMuPDF
        page_images = self._extract_images_with_pymupdf(file_path)

        # Step 2: Link images to figures (match by page)
        enriched_elements = self._link_images_to_figures(elements, page_images)

        # Step 3: Run vision model on figures that have image_path
        enriched_elements = self.vision_enricher.enrich_figures(enriched_elements)

        return enriched_elements

    def _extract_images_with_pymupdf(self, file_path: str) -> dict:
        """Extract all images from PDF using PyMuPDF"""
        doc = fitz.open(file_path)
        page_images = {}

        for page_num in range(len(doc)):
            page = doc[page_num]
            images_on_page = []

            for img_index, img in enumerate(page.get_images(full=True)):
                xref = img[0]
                base_image = doc.extract_image(xref)
                if not base_image:
                    continue

                image_bytes = base_image["image"]
                image_ext = base_image["ext"]

                image_filename = f"page{page_num+1}_img{img_index+1}.{image_ext}"
                image_path = self.image_dir / Path(file_path).stem / image_filename
                image_path.parent.mkdir(parents=True, exist_ok=True)

                with open(image_path, "wb") as f:
                    f.write(image_bytes)

                images_on_page.append({
                    "path": str(image_path),
                    "page": page_num + 1
                })

            if images_on_page:
                page_images[page_num + 1] = images_on_page

        doc.close()
        return page_images

    def _link_images_to_figures(self, elements: List[DocumentElement], page_images: dict) -> List[DocumentElement]:
        """Link extracted images to figure elements by page number"""
        for element in elements:
            if element.content_type != "figure":
                continue

            page_num = element.page_number
            images_on_page = page_images.get(page_num, [])

            if images_on_page and not element.structured_content.get("image_path"):
                # Assign first available image on this page
                element.structured_content = element.structured_content or {}
                element.structured_content["image_path"] = images_on_page[0]["path"]

        return elements