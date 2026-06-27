import os
from pathlib import Path
from dotenv import load_dotenv
from qdrant_client import QdrantClient

from ingestion.ingestion_pipeline import IngestionPipeline

load_dotenv()


def test_full_pipeline_until_vectordb(file_path: str = "test_data/research_papers/2508.11287v1.pdf"):
    api_key = os.getenv("LLAMA_CLOUD_API_KEY")
    if not api_key:
        print("ERROR: LLAMA_CLOUD_API_KEY not found")
        return

    collection_name = "rag_chunks_v1"

    print("\n" + "="*100)
    print("FULL PIPELINE TEST (Until Vector Storage in Qdrant)")
    print(f"File: {file_path}")
    print("="*100 + "\n")

    # Run Full Pipeline
    pipeline = IngestionPipeline(
        llama_parse_api_key=api_key,
        vision_model="llama3.2-vision:latest",
        embedding_model="BAAI/bge-m3",
        collection_name=collection_name,
        enable_cache=False
    )

    elements = pipeline.process_file(file_path)

    # Verify in Qdrant
    print("\n[Verification] Checking Qdrant...\n")
    try:
        client = QdrantClient(host="localhost", port=6333)
        info = client.get_collection(collection_name)

        print(f"Collection Name       : {collection_name}")
        print(f"Total Points Stored   : {info.points_count}")
        print(f"Vector Size           : {info.config.params.vectors.size}")

        if info.points_count > 0:
            print("\n✅ Phase 3 SUCCESS: Embeddings generated and stored in Qdrant.")
            print(f"   → {info.points_count} vectors are now searchable.")
        else:
            print("\n⚠️ Phase 3 WARNING: No vectors found in Qdrant.")

    except Exception as e:
        print(f"\n❌ ERROR: Could not connect to Qdrant. Details: {e}")

    print("\n" + "="*100)
    print("TEST COMPLETED")
    print("="*100 + "\n")


if __name__ == "__main__":
    test_full_pipeline_until_vectordb()