# api/routers/query.py
from fastapi import APIRouter, Depends
from api.schemas import QueryRequest, QueryResponse
from api.dependencies import get_rag_pipeline

router = APIRouter(prefix="/query", tags=["Query"])

@router.post("/", response_model=QueryResponse)
async def query_rag(
    request: QueryRequest,
    pipeline=Depends(get_rag_pipeline)
):
    result = pipeline.ask(request.question, top_k=request.top_k)
    return result