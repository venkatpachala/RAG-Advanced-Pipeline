import logging
from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Any

from api.schemas import QueryRequest, QueryResponse
from api.dependencies import get_rag_pipeline
from observability import generate_request_id  # Optional: for router-level tracing

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/query", tags=["Query"])


@router.post("/", response_model=QueryResponse)
async def query_rag(
    request: QueryRequest,
    pipeline=Depends(get_rag_pipeline)
) -> Dict[str, Any]:
    """
    Main RAG query endpoint.
    Accepts a question and returns a grounded answer with citations and request_id.
    """
    try:
        # Call the pipeline
        result = pipeline.ask(
            user_query=request.question,
            top_k=request.top_k or 8
        )

        # Log the API-level request completion
        logger.info(
            "Query request processed",
            extra={
                "request_id": result.get("request_id"),
                "event": "api_query_completed",
                "query": request.question,
                "chunks_used": result.get("chunks_used", 0),
                "grounded": result.get("grounded", False)
            }
        )

        return result

    except Exception as e:
        logger.error(
            "Query request failed",
            extra={
                "event": "api_query_failed",
                "query": request.question,
                "error": str(e)
            },
            exc_info=True
        )
        raise HTTPException(
            status_code=500,
            detail="An error occurred while processing your query. Please try again."
        )