import logging
from typing import List, Dict
from pathlib import Path

from models.document_element import DocumentElement
from ingestion.chunking.chunk import Chunk

logger = logging.getLogger(__name__)


class HierarchicalChunker:
    """
    Hierarchical Chunker that creates parent (section-level) 
    and child (element-level) chunks with relationships.
    """

    def __init__(self):
        pass

    def chunk(
        self, 
        elements: List[DocumentElement], 
        file_path: str
    ) -> List[Chunk]:
        """
        Main method to convert DocumentElements into hierarchical chunks.
        """
        if not elements:
            return []

        # Step 1: Group elements by sections
        sections = self._group_by_sections(elements)

        all_chunks: List[Chunk] = []

        for section_elements in sections:
            if not section_elements:
                continue

            # Step 2: Create Parent Chunk (Section level)
            parent_chunk = self._create_parent_chunk(section_elements, file_path)

            # Step 3: Create Child Chunks (Element level)
            child_chunks: List[Chunk] = []

            for element in section_elements:
                child_chunk = self._create_child_chunk(element, file_path)
                child_chunk.parent_chunk_id = parent_chunk.chunk_id
                parent_chunk.child_chunk_ids.append(child_chunk.chunk_id)
                child_chunks.append(child_chunk)

            all_chunks.append(parent_chunk)
            all_chunks.extend(child_chunks)

        logger.info(f"Created {len(all_chunks)} hierarchical chunks for {Path(file_path).name}")
        return all_chunks

    def _group_by_sections(self, elements: List[DocumentElement]) -> List[List[DocumentElement]]:
        """Group elements under their respective sections."""
        sections = []
        current_section = []

        for element in elements:
            if element.content_type in ["section_header", "subsection_header"]:
                if current_section:
                    sections.append(current_section)
                current_section = [element]
            else:
                current_section.append(element)

        if current_section:
            sections.append(current_section)

        return sections

    def _create_parent_chunk(self, section_elements: List[DocumentElement], file_path: str) -> Chunk:
        """Create a parent chunk at section level."""
        # Get section heading
        heading = ""
        for el in section_elements:
            if el.content_type in ["section_header", "subsection_header"]:
                heading = el.text
                break

        # Combine text from all elements for parent context (light)
        text_parts = []
        for el in section_elements:
            if el.content_type == "text" and el.text:
                text_parts.append(el.text[:300])  # Limit context

        text = f"{heading}\n" + "\n".join(text_parts[:3])  # Keep it light

        metadata = {
            "source_file": Path(file_path).name,
            "page_numbers": list({el.page_number for el in section_elements if el.page_number}),
            "section_path": section_elements[0].section_path if section_elements else [],
            "element_type": "section",
            "chunk_level": "parent",
        }

        return Chunk(text=text, metadata=metadata)

    def _create_child_chunk(self, element: DocumentElement, file_path: str) -> Chunk:
        """Create a child chunk for individual elements."""
        text = element.text or ""

        # For figures and tables, include vision summary or structured content
        if element.content_type == "figure" and element.vision_summary:
            text = f"{element.caption or ''}\n{element.vision_summary}"
        elif element.content_type == "table" and element.structured_content:
            text = element.structured_content.get("as_markdown", element.text or "")

        metadata = {
            "source_file": Path(file_path).name,
            "page_numbers": [element.page_number] if element.page_number else [],
            "section_path": element.section_path,
            "element_type": element.content_type,
            "chunk_level": "child",
        }

        # Add multimodal fields if present
        if element.content_type == "figure":
            if element.structured_content:
                metadata["image_path"] = element.structured_content.get("image_path")
                metadata["bbox"] = element.structured_content.get("bbox")
            metadata["vision_summary"] = element.vision_summary

        return Chunk(text=text, metadata=metadata)