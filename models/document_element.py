from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class DocumentElement(BaseModel):
    content_type: str
    text: str = ""
    source_file: str
    document_type: str
    page_number: int = 0
    section_path: List[str] = Field(default_factory=list)
    relationships: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    structured_content: Optional[Dict[str, Any]] = None
    caption: Optional[str] = None
    nearby_context: Optional[str] = None
    vision_summary: Optional[str] = None
    quality_score: float = 0.0
