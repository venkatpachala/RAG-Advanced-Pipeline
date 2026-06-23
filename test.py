# test_ingestion_pipeline.py
import os
from ingestion.ingestion_pipeline import IngestionPipeline

def main():
    print("=" * 70)
    print("TESTING INGESTION PIPELINE (Step 9)")
    print("=" * 70)

    # Initialize the pipeline
    pipeline = IngestionPipeline(
        llama_parse_api_key=os.getenv("LLAMA_CLOUD_API_KEY"),
        vision_model="llama3.2-vision:latest",
        enable_cache=True,
        cache_dir="cache/ingestion"
    )

    # ====================== TEST SINGLE FILE ======================
    test_file = "data/sample.pdf"   # ← Change this to your test PDF

    print(f"\n[1] Processing single file: {test_file}")
    elements = pipeline.process_file(test_file)

    print(f"\nTotal elements extracted: {len(elements)}")

    # Show summary
    print("\n--- Element Type Summary ---")
    type_count = {}
    for elem in elements:
        type_count[elem.content_type] = type_count.get(elem.content_type, 0) + 1

    for content_type, count in type_count.items():
        print(f"  {content_type:20}: {count}")

    # Show first few elements with details
    print("\n--- Sample Elements ---")
    for i, elem in enumerate(elements[:6]):
        print(f"\n[{i+1}] Type: {elem.content_type}")
        print(f"    Page: {elem.page_number}")
        print(f"    Section: {elem.section_path}")
        print(f"    Quality Score: {elem.quality_score}")
        print(f"    Text Preview: {str(elem.text)[:120]}...")

        if elem.content_type == "figure":
            print(f"    Caption: {elem.caption}")
            print(f"    Vision Summary: {elem.vision_summary[:150] if elem.vision_summary else 'N/A'}...")

        if elem.content_type == "table":
            table_type = elem.metadata.get("table_type", "unknown")
            print(f"    Table Type: {table_type}")

    # ====================== TEST CACHING ======================
    print("\n" + "=" * 70)
    print("[2] Testing Caching (Run same file again)")
    print("=" * 70)

    elements2 = pipeline.process_file(test_file)
    print(f"Second run completed. Elements returned: {len(elements2)}")
    print("→ If this was fast, caching is working!")

    print("\n" + "=" * 70)
    print("TEST COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()