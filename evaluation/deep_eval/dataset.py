from typing import List, Dict, Any
import json
from pathlib import Path


def load_deep_eval_dataset(file_path: str) -> List[Dict[str, Any]]:
    """
    Load a dataset formatted for DeepEval evaluation.
    
    Expected JSON structure:
    [
        {
            "input": "How does the system reduce cold start latency?",
            "expected_output": "The system reduces cold start latency by using a dynamic programming-based layer allocation algorithm...",
            "retrieval_context": [
                "The algorithm uses dynamic programming to allocate model layers across heterogeneous devices.",
                "It overlaps model loading with ongoing computation and communication to hide cold start latency."
            ]
        },
        ...
    ]
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"DeepEval dataset file not found: {file_path}")

    with open(path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    # Basic validation
    for i, item in enumerate(dataset):
        if "input" not in item:
            raise ValueError(f"Test case at index {i} is missing 'input'")
        if "expected_output" not in item:
            raise ValueError(f"Test case at index {i} is missing 'expected_output'")
        if "retrieval_context" not in item:
            raise ValueError(f"Test case at index {i} is missing 'retrieval_context'")

    return dataset


def get_sample_dataset() -> List[Dict[str, Any]]:
    """Returns a small sample dataset for quick testing."""
    return [
        {
            "input": "How does the system reduce cold start latency?",
            "expected_output": "The system reduces cold start latency by proposing a dynamic programming-based layer allocation algorithm that overlaps model loading with computation and communication phases.",
            "retrieval_context": [
                "The proposed method uses dynamic programming to allocate model layers.",
                "It overlaps model loading with ongoing computation and communication to hide latency."
            ]
        },
        {
            "input": "What is the main contribution of this paper?",
            "expected_output": "The main contribution is a latency-aware pipeline scheduling algorithm using dynamic programming to minimize cold-start latency in wireless collaborative edge LLM systems.",
            "retrieval_context": [
                "proposes a latency-aware pipeline scheduling algorithm tailored for edge environments",
                "dynamic programming–based layer allocation algorithm to minimize latency"
            ]
        }
    ]