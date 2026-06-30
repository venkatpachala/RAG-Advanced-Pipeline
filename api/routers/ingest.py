# api/routers/ingest.py
import os
import tempfile
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException
from api.schemas import IngestResponse
from ingestion.ingestion_pipeline import IngestionPipeline
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ingest", tags=["Ingestion"])

# Supported file types
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".pptx", ".txt"}

@router.post("/", response_model=IngestResponse)
async def ingest_document(file: UploadFile = File(...)):
    # Validate file type
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file_ext}. Supported: {SUPPORTED_EXTENSIONS}"
        )

    # Save to temporary file
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name

        # Initialize ingestion pipeline
        # TODO: Load API key from environment variable in production
        llama_api_key = os.getenv("LLAMA_PARSE_API_KEY", "llx-itdagRKTP9JNkKypdda6jaA50vcSS0EQkLiTLO3ITK5B0aCm")

        ingestion_pipeline = IngestionPipeline(llama_parse_api_key=llama_api_key)
        ingestion_pipeline.process_file(tmp_path)

        logger.info(f"Successfully ingested file: {file.filename}")

        return IngestResponse(
            message="Document ingested successfully",
            filename=file.filename
        )

    except Exception as e:
        logger.error(f"Ingestion failed for {file.filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to ingest document: {str(e)}")

    finally:
        # Clean up temporary file
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.unlink(tmp_path)