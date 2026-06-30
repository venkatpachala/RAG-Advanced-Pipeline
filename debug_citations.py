from retrieval.query_engine import QueryEngine

def inspect_chunk_metadata():
    print("\n" + "="*80)
    print("INSPECTING CHUNK METADATA (Especially page_number)")
    print("="*80)

    query_engine = QueryEngine(
        collection_name="rag_chunks_v1",
        enable_reranking=True,
        enable_query_rewriting=True
    )

    # Test query
    query = "cold start"
    chunks = query_engine.query(query, limit=5)

    print(f"\nTotal chunks retrieved: {len(chunks)}\n")

    for i, chunk in enumerate(chunks, 1):
        print(f"--- Chunk {i} ---")
        print(f"Source File : {chunk.get('source_file', 'N/A')}")
        print(f"Page Number : {chunk.get('page_number', 'N/A')}")
        print(f"Text (first 150 chars): {chunk.get('text', '')[:150]}...")
        print()

    print("="*80)


if __name__ == "__main__":
    inspect_chunk_metadata()