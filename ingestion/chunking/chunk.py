from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import uuid


@dataclass
class Chunk:
    """
    Represents a single chunk in the hierarchical structure.
    """
    chunk_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    text: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    parent_chunk_id: Optional[str] = None
    child_chunk_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk to dictionary (useful for JSON export)."""
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "metadata": self.metadata,
            "parent_chunk_id": self.parent_chunk_id,
            "child_chunk_ids": self.child_chunk_ids,
        }