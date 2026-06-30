import logging
import json
import time
import uuid
from contextlib import contextmanager
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, Optional

# Create logs directory if it doesn't exist
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

LOG_FILE = LOG_DIR / "rag_pipeline.jsonl"


class JsonFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging."""
    def format(self, record: logging.LogRecord) -> str:
        log_record = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add extra fields if present
        if hasattr(record, "request_id"):
            log_record["request_id"] = record.request_id
        if hasattr(record, "stage"):
            log_record["stage"] = record.stage
        if hasattr(record, "duration_ms"):
            log_record["duration_ms"] = record.duration_ms
        if hasattr(record, "status"):
            log_record["status"] = record.status
        if hasattr(record, "extra"):
            log_record.update(record.extra)

        return json.dumps(log_record, ensure_ascii=False)


def setup_logging(level: int = logging.INFO) -> None:
    """Initialize structured JSON logging to both console and file."""
    logger = logging.getLogger()
    logger.setLevel(level)

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(JsonFormatter())
    logger.addHandler(console_handler)

    # File handler (rotating)
    file_handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5,
        encoding="utf-8"
    )
    file_handler.setFormatter(JsonFormatter())
    logger.addHandler(file_handler)

    logging.info("Observability logging initialized", extra={"extra": {"log_file": str(LOG_FILE)}})


def generate_request_id() -> str:
    """Generate a unique request ID for tracing."""
    return str(uuid.uuid4())[:12]


@contextmanager
def log_stage(
    stage_name: str,
    request_id: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
    **extra_data: Any
):
    """
    Context manager to log the start, end, and duration of any pipeline stage.
    
    Usage:
        with log_stage("retrieval", request_id=req_id, num_chunks=8):
            # do retrieval work
    """
    if logger is None:
        logger = logging.getLogger("rag.observability")

    start_time = time.perf_counter()
    rid = request_id or generate_request_id()

    # Log stage start
    logger.info(
        f"Stage started: {stage_name}",
        extra={
            "request_id": rid,
            "stage": stage_name,
            "status": "started",
            "extra": extra_data
        }
    )

    try:
        yield rid
        status = "success"
    except Exception as e:
        status = "failed"
        logger.error(
            f"Stage failed: {stage_name} - {str(e)}",
            extra={
                "request_id": rid,
                "stage": stage_name,
                "status": "failed",
                "extra": {"error": str(e), **extra_data}
            },
            exc_info=True
        )
        raise
    finally:
        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)

        # Log stage completion with latency
        logger.info(
            f"Stage completed: {stage_name}",
            extra={
                "request_id": rid,
                "stage": stage_name,
                "duration_ms": duration_ms,
                "status": status,
                "extra": extra_data
            }
        )