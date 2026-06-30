import sys
from pathlib import Path

# Fix import path
sys.path.append(str(Path(__file__).parent.parent.parent))

from typing import List
from deepeval.test_case import LLMTestCase
from deepeval import evaluate

from evaluation.deep_eval.rag_app import RAGApp
from evaluation.deep_eval.dataset import load_deep_eval_dataset
from evaluation.deep_eval.metrics import (
    get_faithfulness_metric,
    get_answer_relevancy_metric
)

import json
from datetime import datetime
from pathlib import Path


def run_deep_eval(
    dataset_path: str = "evaluation/deep_eval/test_dataset.json",
    save_report: bool = True
):
    print("=" * 85)
    print("🚀 DEEPEVAL - RAG PIPELINE EVALUATION (Answer Relevancy + Faithfulness)")
    print("=" * 85)

    print("\n[1/3] Initializing RAG Pipeline...")
    rag_app = RAGApp()

    print(f"[2/3] Loading test dataset from: {dataset_path}")
    test_dataset = load_deep_eval_dataset(dataset_path)
    print(f"Loaded {len(test_dataset)} test cases")

    print("[3/3] Preparing metrics...")
    metrics = [
        get_faithfulness_metric(),
        get_answer_relevancy_metric()
    ]

    test_cases: List[LLMTestCase] = []

    for item in test_dataset:
        result = rag_app.generate_answer(item["input"])

        test_case = LLMTestCase(
            input=item["input"],
            actual_output=result["answer"],
            expected_output=item["expected_output"],
            retrieval_context=item["retrieval_context"],
            context=item["retrieval_context"]
        )
        test_cases.append(test_case)

    print(f"\nRunning evaluation on {len(test_cases)} test cases...\n")

    evaluate(
        test_cases=test_cases,
        metrics=metrics
    )

    # Save report
    if save_report:
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_test_cases": len(test_cases),
            "metrics_used": [m.__class__.__name__ for m in metrics],
            "dataset_path": dataset_path
        }

        report_dir = Path("evaluation/reports")
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / f"deepeval_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        print(f"\n✅ Report saved to: {report_path}")

    print("\n" + "=" * 85)
    print("✅ EVALUATION COMPLETED")
    print("=" * 85)


if __name__ == "__main__":
    run_deep_eval()