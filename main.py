"""
Final Clean Example: Full Advanced RAG Pipeline
Ingestion (Phases 1-3) + Retrieval (Phase 4)
"""

from ingestion.ingestion_pipeline import IngestionPipeline
from ingestion.retrieval.query_engine import QueryEngine
from ingestion.retrieval.filter_builder import FilterBuilder

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    # ====================== CONFIGURATION ======================
    LLAMA_PARSE_API_KEY = "llx-itdagRKTP9JNkKypdda6jaA50vcSS0EQkLiTLO3ITK5B0aCm"   # ← Replace this
    DOCUMENT_PATH = "test_data/research_papers/2508.11287v1.pdf"
    COLLECTION_NAME = "rag_chunks_v1"

    print("\n" + "="*75)
    print("PHASE 1-3: DOCUMENT INGESTION & EMBEDDING")
    print("="*75)

    # Initialize Ingestion Pipeline
    ingestion = IngestionPipeline(
        llama_parse_api_key=LLAMA_PARSE_API_KEY,
        vision_model="llama3.2-vision:latest",
        embedding_model="BAAI/bge-m3",
        collection_name=COLLECTION_NAME,
        enable_cache=False
    )

    # Process the document
    elements = ingestion.process_file(DOCUMENT_PATH)
    print(f"\n✅ Document processed successfully!")
    print(f"   - Total elements extracted : {len(elements)}")

    print("\n" + "="*75)
    print("PHASE 4: RETRIEVAL (Hybrid Search + Reranking + Rewriting)")
    print("="*75)

    # Initialize Query Engine
    query_engine = QueryEngine(
        collection_name=COLLECTION_NAME,
        enable_reranking=True,
        enable_query_rewriting=True
    )

    # ====================== EXAMPLE 1: Simple Query ======================
    query1 = "How does the system optimize cold start latency?"
    print(f"\n🔍 Query: {query1}")

    results = query_engine.query(
        user_query=query1,
        limit=6
    )

    print(f"\nTop {len(results)} results:\n")
    for i, result in enumerate(results, 1):
        text_preview = result.get("text", "")[:220].replace("\n", " ")
        print(f"{i}. {text_preview}...")
        print("-" * 75)

    # ====================== EXAMPLE 2: With Metadata Filtering ======================
    print("\n" + "="*75)
    print("Example with Metadata Filtering (Only Figures)")
    print("="*75)

    filter_builder = FilterBuilder()
    filter_builder.add("element_type", "figure")

    query2 = "Performance comparison of different optimization strategies"
    print(f"\n🔍 Query: {query2} (Filtered by: figures only)")

    results = query_engine.query(
        user_query=query2,
        limit=5,
        filter_dict=filter_builder.build()
    )

    print(f"\nTop {len(results)} figure results:\n")
    for i, result in enumerate(results, 1):
        text_preview = result.get("text", "")[:220].replace("\n", " ")
        print(f"{i}. {text_preview}...")
        print("-" * 75)


if __name__ == "__main__":
    main()