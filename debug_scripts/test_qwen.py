from ingestion.retrieval.query_engine import QueryEngine

def test_query_engine():
    print("\n" + "="*85)
    print("TESTING GROUNDED QUERY ENGINE (with Context-Aware Query Rewriting)")
    print("="*85)

    query_engine = QueryEngine(
        collection_name="rag_chunks_v1",
        enable_reranking=True,
        enable_query_rewriting=True
    )

    test_queries = [
        "cold start",
        "how to reduce latency",
        "figure 3 performance"
    ]

    for query in test_queries:
        print(f"\n{'='*85}")
        print(f" Original Query      : {query}")

        # Show rewritten query
        if query_engine.rewriter:
            rewritten = query_engine.rewriter.rewrite_query(query)
            print(f" Rewritten Query     : {rewritten}")
        else:
            print(" Rewritten Query     : (Rewriting disabled)")

        # Get results
        results = query_engine.query(query, limit=5)

        print(f"\n Retrieved {len(results)} chunks:\n")
        for i, result in enumerate(results, 1):
            text = result.get("text", "")[:230].replace("\n", " ")
            print(f"{i}. {text}...")
            print("-" * 85)

    print("\n" + "="*85)
    print("TEST COMPLETED")
    print("="*85 + "\n")


if __name__ == "__main__":
    test_query_engine()