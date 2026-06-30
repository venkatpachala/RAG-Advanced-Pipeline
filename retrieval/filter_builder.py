from typing import Dict, Any


class FilterBuilder:
    """
    Helper to build metadata filters easily.
    """

    def __init__(self):
        self.filters: Dict[str, Any] = {}

    def add(self, key: str, value: Any):
        """Add a filter condition."""
        self.filters[key] = value
        return self

    def build(self) -> Dict[str, Any]:
        return self.filters.copy()

    def reset(self):
        self.filters = {}
        return self