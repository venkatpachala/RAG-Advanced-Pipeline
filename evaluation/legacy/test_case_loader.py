import json
from pathlib import Path
from typing import List, Dict, Any


def load_test_cases(file_path: str) -> List[Dict[str, Any]]:
    """
    Load test cases from a JSON file.
    
    Expected JSON structure:
    [
        {
            "query": "How does the system reduce cold start latency?",
            "relevant_texts": [
                "dynamic programming-based layer allocation algorithm",
                "overlaps model loading with ongoing computation"
            ]
        },
        ...
    ]
    """
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Test case file not found: {file_path}")
    
    with open(path, "r", encoding="utf-8") as f:
        test_cases = json.load(f)
    
    # Basic validation
    for i, case in enumerate(test_cases):
        if "query" not in case:
            raise ValueError(f"Test case at index {i} is missing 'query' field")
        if "relevant_texts" not in case:
            raise ValueError(f"Test case at index {i} is missing 'relevant_texts' field")
    
    return test_cases


def get_test_case_files(directory: str = "evaluation/test_cases") -> List[str]:
    """Get all JSON test case files from a directory."""
    path = Path(directory)
    if not path.exists():
        return []
    
    return [str(f) for f in path.glob("*.json")]