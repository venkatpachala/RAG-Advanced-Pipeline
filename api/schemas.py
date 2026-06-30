# api/schemas.py
from pydantic import BaseModel
from typing import List, Optional

class QueryRequest(BaseModel):
    question: str
    top_k: int = 8

class QueryResponse(BaseModel):
    query: str
    answer: str
    citations: List[str]
    grounded: bool
    chunks_used: int

class IngestResponse(BaseModel):
    message: str
    filename: str
    status: str = "success"