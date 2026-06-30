# api/main.py
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
import time
import logging
import json

from api.routers import ingest, query
from observability import generate_request_id

logger = logging.getLogger("rag_api")

app = FastAPI(
    title="Advanced RAG Pipeline API",
    description="Production-grade RAG with Grounded Generation, Citations & Observability",
    version="1.0.0"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = generate_request_id()
        request.state.request_id = request_id

        start_time = time.time()

        # Log incoming request
        logger.info(json.dumps({
            "event": "http_request_start",
            "request_id": request_id,
            "method": request.method,
            "url": str(request.url),
            "client_ip": request.client.host if request.client else None
        }))

        response = await call_next(request)

        duration = round(time.time() - start_time, 3)

        # Log response
        logger.info(json.dumps({
            "event": "http_request_end",
            "request_id": request_id,
            "status_code": response.status_code,
            "duration_seconds": duration
        }))

        # Add request_id to response headers (useful for clients)
        response.headers["X-Request-ID"] = request_id

        return response


# Add middleware
app.add_middleware(RequestLoggingMiddleware)

# Include routers
app.include_router(ingest.router)
app.include_router(query.router)


@app.get("/")
def root():
    return {"message": "Advanced RAG Pipeline API is running"}