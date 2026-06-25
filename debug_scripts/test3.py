import os
import json
from pathlib import Path
from dotenv import load_dotenv

from ingestion.image_visualizer import ImageVisionPipeline
from ingestion.schema_normalizer import LlamaParseJSONNormalizer

load_dotenv()


def test_image_vision_pipeline(file_path: str = "test_data/research_papers/2508.11287v1.pdf"):
    api_key = os.getenv("LLAMA_CLOUD_API_KEY")
    if not api_key:
        print("ERROR: LLAMA_CLOUD_API_KEY not found in .env")
        return

    print(f"\n{'='*80}")
    print(f"Testing Image + Vision Pipeline on: {file_path}")
    print(f"{'='*80}\n")

    # Step 1: Get base elements from main normalizer (structure + figure detection)
    normalizer = LlamaParseJSONNormalizer(api_key=api_key)
    base_elements = normalizer.parse_and_normalize(file_path)

    print(f"Base elements from Normalizer: {len(base_elements)}")

    # Step 2: Run dedicated Image + Vision Pipeline
    image_vision = ImageVisionPipeline(vision_model="llama3.2-vision:latest")
    enriched_elements = image_vision.enrich_images_and_vision(base_elements, file_path)

    # Step 3: Analyze results
    figures = [el for el in enriched_elements if el.content_type == "figure"]

    print(f"\nTotal Figures after enrichment: {len(figures)}\n")

    enriched_data = []

    for i, fig in enumerate(figures):
        structured = fig.structured_content or {}
        image_path = structured.get("image_path")

        figure_data = {
            "figure_index": i + 1,
            "page_number": fig.page_number,
            "caption": fig.caption,
            "image_path": image_path,
            "file_exists": Path(image_path).exists() if image_path else False,
            "vision_summary": fig.vision_summary,
            "bbox": structured.get("bbox"),
        }

        enriched_data.append(figure_data)

        print(f"--- Figure {i+1} ---")
        print(f"Page          : {fig.page_number}")
        print(f"Caption       : {fig.caption[:90] if fig.caption else 'N/A'}...")
        print(f"Image Path    : {image_path}")
        print(f"File Exists   : {figure_data['file_exists']}")
        print(f"Vision Summary: {fig.vision_summary[:200] if fig.vision_summary else 'N/A'}...")
        print()

    # Save enriched output to JSON
    output_file = Path("debug_scripts") / f"enriched_figures_{Path(file_path).stem}.json"
    output_file.parent.mkdir(exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(enriched_data, f, indent=2, ensure_ascii=False)

    print(f"{'='*80}")
    print(f"Enriched JSON saved to: {output_file}")
    print(f"Figures with image_path     : {sum(1 for d in enriched_data if d['image_path'])}")
    print(f"Figures with vision_summary : {sum(1 for d in enriched_data if d['vision_summary'] and d['vision_summary'] != 'No valid image file available.')}")
    print(f"{'='*80}")


if __name__ == "__main__":
    test_image_vision_pipeline()