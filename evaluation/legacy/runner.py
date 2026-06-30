import json
from pathlib import Path
from datetime import datetime
from pipeline.rag_pipeline import RAGPipeline
from evaluation.test_case_loader import load_test_cases
from evaluation.retrieval_evaluator import RetrievalEvaluator


def run_evaluation(
    test_case_file: str = "evaluation/test_cases/sample_test_cases.json",
    k: int = 5,
    save_report: bool = True
):
    print("=" * 70)
    print("RAG RETRIEVAL EVALUATION")
    print("=" * 70)

    # Load test cases
    print(f"\nLoading test cases from: {test_case_file}")
    test_cases = load_test_cases(test_case_file)
    print(f"Loaded {len(test_cases)} test cases")

    # Initialize pipeline
    print("\nInitializing RAG Pipeline...")
    pipeline = RAGPipeline(collection_name="rag_chunks_v1")
    print("Pipeline ready")

    # Run evaluation
    print(f"\nRunning retrieval evaluation (k={k})...")
    evaluator = RetrievalEvaluator(pipeline=pipeline, k=k)
    scores = evaluator.evaluate(test_cases)

    # Print results
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    for metric, score in scores.items():
        print(f"{metric:15}: {score:.4f}")
    print("=" * 70)

    # Save report
    if save_report:
        report = {
            "timestamp": datetime.now().isoformat(),
            "test_case_file": test_case_file,
            "k": k,
            "num_test_cases": len(test_cases),
            "scores": scores
        }

        report_path = Path("evaluation/reports") / f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)

        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        print(f"\nReport saved to: {report_path}")

    return scores


if __name__ == "__main__":
    run_evaluation()