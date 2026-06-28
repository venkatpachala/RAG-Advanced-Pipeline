import os
import json
from pathlib import Path
from dotenv import load_dotenv

from ingestion.schema_normalizer import LlamaParseJSONNormalizer
from ingestion.image_visualizer import ImageVisionPipeline
from ingestion.chunking.hierarchical_chunker import HierarchicalChunker

load_dotenv()


def test_full_pipeline(file_path: str = "test_data/research_papers/2508.11287v1.pdf"):
    api_key = os.getenv("LLAMA_CLOUD_API_KEY")
    if not api_key:
        print("ERROR: LLAMA_CLOUD_API_KEY not found in .env")
        return

    print(f"\n{'='*90}")
    print(f"FULL PIPELINE TEST: Normalizer → Vision → Hierarchical Chunker")
    print(f"File: {file_path}")
    print(f"{'='*90}\n")

    # ===================== STAGE 1: NORMALIZER =====================
    print("[Stage 1] Running LlamaParseJSONNormalizer...")
    normalizer = LlamaParseJSONNormalizer(api_key=api_key)
    elements = normalizer.parse_and_normalize(file_path)

    print(f"  → Elements after Normalizer : {len(elements)}")

    # ===================== STAGE 2: IMAGE + VISION =====================
    print("\n[Stage 2] Running ImageVisionPipeline...")
    image_vision = ImageVisionPipeline(vision_model="llama3.2-vision:latest ")
    enriched_elements = image_vision.enrich_images_and_vision(elements, file_path)

    figures = [el for el in enriched_elements if el.content_type == "figure"]
    figures_with_path = [
        el for el in enriched_elements 
        if el.content_type == "figure" 
        and el.structured_content 
        and el.structured_content.get("image_path")
    ]

    print(f"  → Total Elements after Vision : {len(enriched_elements)}")
    print(f"  → Figures detected            : {len(figures)}")
    print(f"  → Figures with image_path     : {len(figures_with_path)}")

    # ===================== STAGE 3: HIERARCHICAL CHUNKING =====================
    print("\n[Stage 3] Running HierarchicalChunker...")
    chunker = HierarchicalChunker()
    chunks = chunker.chunk(enriched_elements, file_path)

    parent_chunks = [c for c in chunks if c.metadata.get("chunk_level") == "parent"]
    child_chunks = [c for c in chunks if c.metadata.get("chunk_level") == "child"]

    print(f"  → Total Chunks created        : {len(chunks)}")
    print(f"  → Parent Chunks (Sections)    : {len(parent_chunks)}")
    print(f"  → Child Chunks (Elements)     : {len(child_chunks)}")

    # ===================== SAMPLE OUTPUT =====================
    print("\n[Sample Output] First 3 Child Chunks:\n")

    for i, chunk in enumerate(child_chunks[:3]):
        print(f"--- Child Chunk {i+1} ---")
        print(f"Chunk ID      : {chunk.chunk_id}")
        print(f"Text (first 150 chars): {chunk.text[:150]}...")
        print(f"Parent ID     : {chunk.parent_chunk_id}")
        print(f"Metadata      : {json.dumps(chunk.metadata, indent=2)[:300]}...")
        print()

    # ===================== SAVE TO JSON =====================
    output_dir = Path("debug_scripts")
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f"full_pipeline_chunks_{Path(file_path).stem}.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump([c.to_dict() for c in chunks], f, indent=2, ensure_ascii=False)

    print(f"{'='*90}")
    print(f"Full pipeline completed successfully!")
    print(f"Chunks saved to: {output_file}")
    print(f"{'='*90}")


if __name__ == "__main__":
    test_full_pipeline()