# tests/test_image_and_table.py

import os
from pathlib import Path
from dotenv import load_dotenv

from ingestion.ingestion_pipeline import IngestionPipeline

load_dotenv()


def test_image_download_and_tables(input_dir: str):
    api_key = os.getenv("LLAMA_CLOUD_API_KEY")
    if not api_key:
        print("ERROR: LLAMA_CLOUD_API_KEY not found in .env file")
        return

    pipeline = IngestionPipeline(
        llama_parse_api_key=api_key,
        vision_model="qwen2.5-vl:7b",
        enable_cache=True
    )

    pdf_files = list(Path(input_dir).glob("*.pdf"))
    print(f"Found {len(pdf_files)} PDF files to test.\n")

    for pdf_file in pdf_files:
        print(f"{'='*80}")
        print(f"Testing File: {pdf_file.name}")
        print(f"{'='*80}")

        try:
            elements = pipeline.process_file(str(pdf_file))

            # === 1. Check Figures / Images ===
            figures = [el for el in elements if el.content_type == "figure"]
            figures_with_image_path = [
                el for el in figures 
                if el.structured_content and el.structured_content.get("image_path")
            ]

            print(f"\n[1] IMAGE DOWNLOAD CHECK")
            print(f"    Total figures detected          : {len(figures)}")
            print(f"    Figures with image_path         : {len(figures_with_image_path)}")

            # Check if image files actually exist on disk
            existing_images = 0
            for el in figures_with_image_path:
                img_path = el.structured_content.get("image_path")
                if img_path and Path(img_path).exists():
                    existing_images += 1

            print(f"    Images actually found on disk   : {existing_images}")

            if figures_with_image_path:
                print(f"    Sample image path               : {figures_with_image_path[0].structured_content.get('image_path')}")

            # === 2. Check Tables ===
            tables = [el for el in elements if el.content_type == "table"]
            tables_with_structure = [
                el for el in tables 
                if el.structured_content 
                and el.structured_content.get("headers") 
                and el.structured_content.get("rows")
            ]

            print(f"\n[2] TABLE STRUCTURING CHECK")
            print(f"    Total tables detected           : {len(tables)}")
            print(f"    Tables with proper structure    : {len(tables_with_structure)}")

            if tables_with_structure:
                sample_table = tables_with_structure[0].structured_content
                print(f"    Sample table headers            : {sample_table.get('headers')}")
                print(f"    Sample table rows count         : {len(sample_table.get('rows', []))}")

            print(f"\nSummary for {pdf_file.name}:")
            print(f"  - Image paths present     : {'Yes' if figures_with_image_path else 'No'}")
            print(f"  - Images exist on disk    : {'Yes' if existing_images > 0 else 'No'}")
            print(f"  - Tables properly structured : {'Yes' if tables_with_structure else 'No'}")

        except Exception as e:
            print(f"ERROR processing {pdf_file.name}: {e}")

        print("\n")


if __name__ == "__main__":
    test_image_download_and_tables("test_data/research_papers")