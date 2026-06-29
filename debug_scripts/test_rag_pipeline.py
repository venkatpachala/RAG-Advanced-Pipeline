from rag_pipeline import RAGPipeline

def test_rag_pipeline():
    print("\n" + "="*90)
    print("TESTING FULL RAG PIPELINE (Retrieval + Grounded Generation + Citations)")
    print("="*90)

    pipeline = RAGPipeline(collection_name="rag_chunks_v1", llm_model="qwen2.5:7b")

    test_queries = [
        "What is the main idea of this paper?",
        "How does the system reduce cold start latency?",
        "What does Figure 3 show?"
    ]

    for query in test_queries:
        print(f"\n{'='*90}")
        print(f"User Query: {query}\n")

        result = pipeline.ask(query, top_k=6)

        print("Answer:\n")
        print(result["answer"])
        print(f"\nChunks Used     : {result['chunks_used']}")
        print(f"Grounded        : {result['grounded']}")
        print(f"Citations       : {result['citations']}")

    print("\n" + "="*90)
    print("TEST COMPLETED")
    print("="*90 + "\n")


if __name__ == "__main__":
    test_rag_pipeline()