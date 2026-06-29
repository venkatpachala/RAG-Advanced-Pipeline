from ingestion.retrieval.query_engine import QueryEngine
from evaluation.evaluator import RAGEvaluator

def test_evaluation():
    print("\n" + "="*70)
    print("EVALUATION FRAMEWORK TEST (Improved Test Cases)")
    print("="*70)

    # Initialize QueryEngine (same configuration as main.py)
    query_engine = QueryEngine(
        collection_name="rag_chunks_v1",
        enable_reranking=True,
        enable_query_rewriting=True
    )

    evaluator = RAGEvaluator(query_engine)

    # Improved test cases with longer, more unique text from the paper
    test_cases = [
        {
            "query": "How does the system optimize cold start latency?",
            "relevant_chunks": [
                "In this paper, we tackle the challenge of inference cold-start latency in wireless networks composed of heterogeneous devices. We propose a novel dynamic programming–based layer allocation algorithm to minimize latency.",
                "The cold start problem in personal devices, where long idle periods between inference requests are typical, can lead to substantial latency as each task may require loading the model from storage into memory from scratch."
            ]
        },
        {
            "query": "What does Figure 3 show in the paper?",
            "relevant_chunks": [
                "Figure 3: Performance comparison of the proposed DP algorithm against baseline methods. Figure 3(b): Detailed analysis of cold-start latency across different token lengths.",
                "As shown in Figure 3, the proposed DP algorithm consistently achieves the lowest cold-start latency across all evaluated token lengths (256 to 8192)"
            ]
        },
        {
            "query": "What is the main contribution of this paper?",
            "relevant_chunks": [
                "1.2 Contributions To tackle the cold-start challenge in wireless distributed collaborative inference, this paper proposes a latency-aware pipeline scheduling algorithm tailored for edge environments.",
                "We propose a novel dynamic programming–based layer allocation algorithm to minimize latency."
            ]
        }
    ]

    print(f"\nRunning evaluation on {len(test_cases)} test cases...\n")

    # Run evaluation at K=5
    results = evaluator.evaluate(test_cases, k=5)

    print("\n" + "="*70)
    print("EVALUATION RESULTS @ K=5")
    print("="*70)
    for metric, score in results.items():
        print(f"{metric.upper():12} : {score:.4f}")
    print("="*70 + "\n")


if __name__ == "__main__":
    test_evaluation()