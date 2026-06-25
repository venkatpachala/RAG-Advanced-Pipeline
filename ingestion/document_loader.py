# src/ingestion/document_loader.py

import os
import logging
import base64
from typing import List, Optional
from llama_index.core import SimpleDirectoryReader, Document
from llama_parse import LlamaParse
import ollama

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnterpriseDocumentLoader:
    def __init__(
        self,
        data_dir: str,
        vision_model: str = "llama3.2-vision:latest",
        use_llama_parse: bool = True,
        llama_parse_api_key: Optional[str] = None
    ):
        self.data_dir = data_dir
        self.vision_model = vision_model
        self.use_llama_parse = use_llama_parse
        self.llama_parse_api_key = llama_parse_api_key or os.getenv("LLAMA_CLOUD_API_KEY")

    def load_documents(self) -> List[Document]:
        """Main production-grade document loader with image & table handling."""
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Directory not found: {self.data_dir}")

        documents = []

        # === Step 1: Load with LlamaParse (Best for PDFs with tables/images) ===
        if self.use_llama_parse and self.llama_parse_api_key:
            logger.info("Using LlamaParse for high-quality PDF parsing...")
            parser = LlamaParse(
                api_key=self.llama_parse_api_key,
                result_type="markdown",
                parsing_instruction=(
                    "Extract all text, tables as clean markdown, and describe any images, "
                    "diagrams, or flowcharts in detail. Maintain document structure."
                )
            )
            file_extractor = {".pdf": parser}
        else:
            file_extractor = {}

        reader = SimpleDirectoryReader(
            input_dir=self.data_dir,
            recursive=True,
            filename_as_id=True,
            file_extractor=file_extractor,
        )

        raw_documents = reader.load_data()
        logger.info(f"Loaded {len(raw_documents)} raw documents")

        # === Step 2: Process each document (Images + Tables) ===
        for doc in raw_documents:
            enriched_doc = self._process_document(doc)
            documents.append(enriched_doc)

        logger.info(f"Successfully processed {len(documents)} documents with images/tables handled.")
        return documents

    def _process_document(self, doc: Document) -> Document:
        """Enhance document with image descriptions and table handling."""
        text = doc.text or ""
        metadata = doc.metadata or {}

        # Extract and describe images (if any image references exist)
        image_descriptions = self._extract_and_describe_images(doc)
        
        if image_descriptions:
            text += "\n\n### Visual Elements in Document:\n"
            text += "\n".join(image_descriptions)
            metadata["has_images"] = True
            metadata["image_count"] = len(image_descriptions)
        else:
            metadata["has_images"] = False

        # Tables are already well handled by LlamaParse (as markdown)
        # We just mark them for better retrieval later
        if "table" in text.lower() or "|" in text:
            metadata["contains_tables"] = True

        doc.text = text
        doc.metadata = metadata
        return doc

    def _extract_and_describe_images(self, doc: Document) -> List[str]:
        """Extract images and generate descriptions using local vision model."""
        descriptions = []
        
        # LlamaParse sometimes includes image references in markdown
        # For now, we check metadata or look for image markers
        image_paths = doc.metadata.get("image_paths", [])
        
        # If no explicit paths, we can still describe if images are embedded
        # (LlamaParse markdown often includes ![alt](image) style references)
        
        for img_path in image_paths:
            try:
                if os.path.exists(img_path):
                    description = self._describe_image_with_ollama(img_path)
                    descriptions.append(f"- Image: {description}")
                    logger.info(f"Described image: {img_path}")
            except Exception as e:
                logger.warning(f"Failed to describe image {img_path}: {e}")

        return descriptions

    def _describe_image_with_ollama(self, image_path: str) -> str:
        """Use local vision model to describe image/flowchart."""
        try:
            with open(image_path, "rb") as f:
                image_data = f.read()

            response = ollama.chat(
                model=self.vision_model,
                messages=[
                    {
                        "role": "user",
                        "content": (
                            "Describe this image or flowchart in detail. "
                            "If it's a diagram or flowchart, explain the flow and key components. "
                            "Be precise and structured."
                        ),
                        "images": [image_data]
                    }
                ],
                options={"temperature": 0.3}
            )
            return response['message']['content']
        except Exception as e:
            logger.error(f"Vision model error on {image_path}: {e}")
            return "Image could not be described."


# ==================== Quick Test ====================
if __name__ == "__main__":
    loader = EnterpriseDocumentLoader(
        data_dir="data/research_papers",
        vision_model="llama3.2-vision:latest",
        use_llama_parse=True
    )
    
    docs = loader.load_documents()
    
    for i, doc in enumerate(docs[:2]):  # Preview first 2
        print(f"\n=== Document {i+1} ===")
        print(f"Source: {doc.metadata.get('source')}")
        print(f"Has Images: {doc.metadata.get('has_images')}")
        print(f"Contains Tables: {doc.metadata.get('contains_tables')}")
        print(f"Text Preview:\n{doc.text[:800]}...\n")