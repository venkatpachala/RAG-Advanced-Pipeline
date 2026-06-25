# # tests/test_vision_and_paths.py

# import os
# import json
# from pathlib import Path
# from dotenv import load_dotenv

# from ingestion.ingestion_pipeline import IngestionPipeline

# load_dotenv()


# def test_vision_and_paths(input_dir: str, output_dir: str = "processed_data"):
#     api_key = os.getenv("LLAMA_CLOUD_API_KEY")
#     if not api_key:
#         print("ERROR: LLAMA_CLOUD_API_KEY not found in .env file")
#         return

#     pipeline = IngestionPipeline(
#         llama_parse_api_key=api_key,
#         vision_model="qwen2.5-vl:7b",
#         enable_cache=True
#     )

#     pdf_files = list(Path(input_dir).glob("*.pdf"))
#     print(f"Testing {len(pdf_files)} files for image paths and vision output...\n")

#     for pdf_file in pdf_files:
#         print(f"{'='*90}")
#         print(f"File: {pdf_file.name}")
#         print(f"{'='*90}")

#         try:
#             elements = pipeline.process_file(str(pdf_file))

#             # Filter figures
#             figures = [el for el in elements if el.content_type == "figure"]

#             print(f"\nTotal figures detected: {len(figures)}")

#             for i, fig in enumerate(figures[:5]):  # Check first 5 figures
#                 print(f"\n--- Figure {i+1} ---")

#                 # 1. Image Path
#                 structured = fig.structured_content or {}
#                 image_path = structured.get("image_path", "NOT FOUND")
#                 print(f"Image Path in structured_content : {image_path}")

#                 # Check if file actually exists
#                 if image_path and Path(image_path).exists():
#                     print(f"File exists on disk             : Yes")
#                 else:
#                     print(f"File exists on disk             : No")

#                 # 2. Vision Model Output
#                 vision_summary = fig.vision_summary or "NO VISION SUMMARY"
#                 print(f"Vision Summary (first 400 chars):")
#                 print(vision_summary[:400] if len(vision_summary) > 400 else vision_summary)

#             # Save full elements.json
#             doc_folder = Path(output_dir) / pdf_file.stem
#             doc_folder.mkdir(parents=True, exist_ok=True)

#             json_path = doc_folder / "elements.json"
#             with open(json_path, "w", encoding="utf-8") as f:
#                 json.dump([e.model_dump() for e in elements], f, indent=2, ensure_ascii=False)

#             print(f"\nFull elements.json saved at: {json_path}")
#             print("You can open this file and search for 'vision_summary' and 'image_path'.")

#         except Exception as e:
#             print(f"Error processing {pdf_file.name}: {e}")

#         print("\n")


# if __name__ == "__main__":
#     test_vision_and_paths("test_data/research_papers")

# tests/test_image_linking.py

import os
from pathlib import Path
from dotenv import load_dotenv

from ingestion.ingestion_pipeline import IngestionPipeline

load_dotenv()


def test_image_linking(input_dir: str = "test_data/research_papers"):
    api_key = os.getenv("LLAMA_CLOUD_API_KEY")
    if not api_key:
        print("ERROR: LLAMA_CLOUD_API_KEY not found in .env")
        return

    pipeline = IngestionPipeline(
        llama_parse_api_key=api_key,
        vision_model="llama3.2-vision:latest",
        enable_cache=True
    )

    pdf_files = list(Path(input_dir).glob("*.pdf"))
    print(f"\nTesting image linking on {len(pdf_files)} files...\n")

    for pdf_file in pdf_files:
        print(f"{'='*80}")
        print(f"File: {pdf_file.name}")
        print(f"{'='*80}")

        try:
            elements = pipeline.process_file(str(pdf_file))

            # Filter figures
            figures = [el for el in elements if el.content_type == "figure"]
            figures_with_path = [
                el for el in figures 
                if el.structured_content and el.structured_content.get("image_path")
            ]
            figures_with_existing_file = [
                el for el in figures_with_path 
                if Path(el.structured_content["image_path"]).exists()
            ]

            print(f"Total figures detected          : {len(figures)}")
            print(f"Figures with image_path         : {len(figures_with_path)}")
            print(f"Figures with existing image file: {len(figures_with_existing_file)}")

            # Show sample of first 3 figures
            print("\nSample Figures:")
            for i, fig in enumerate(figures[:3]):
                structured = fig.structured_content or {}
                img_path = structured.get("image_path")
                exists = Path(img_path).exists() if img_path else False

                print(f"\n  Figure {i+1}:")
                print(f"    Caption     : {fig.caption[:80] if fig.caption else 'N/A'}...")
                print(f"    Image Path  : {img_path}")
                print(f"    File Exists : {exists}")
                print(f"    Vision Summary: {fig.vision_summary[:100] if fig.vision_summary else 'N/A'}...")

            success_rate = (len(figures_with_existing_file) / len(figures) * 100) if figures else 0
            print(f"\nSuccess Rate: {success_rate:.1f}% figures have valid images\n")

        except Exception as e:
            print(f"Error processing {pdf_file.name}: {e}\n")


if __name__ == "__main__":
    test_image_linking()