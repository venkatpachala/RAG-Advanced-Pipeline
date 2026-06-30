import logging
import json
import time
import uuid
from contextlib import contextmanager
from typing import Any, Dict, Optional

# Configure JSON logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",  # We will output JSON manually
)

logger = logging.getLogger("rag_pipeline")


def get_logger(name: str = "rag_pipeline"):
    return logging.getLogger(name)


def generate_request_id() -> str:
    """Generate a unique request ID for tracing."""
    return str(uuid.uuid4())


@contextmanager
def log_stage(stage_name: str, extra: Optional[Dict[str, Any]] = None):
    """
    Context manager to log the start, duration, and result of a pipeline stage.
    Usage:
        with log_stage("retrieval", {"query": query}):
            # do something
    """
    if extra is None:
        extra = {}

    request_id = extra.get("request_id", generate_request_id())
    start_time = time.time()

    logger.info(json.dumps({
        "event": "stage_start",
        "stage": stage_name,
        "request_id": request_id,
        **extra
    }))

    try:
        yield request_id
        duration = round(time.time() - start_time, 3)

        logger.info(json.dumps({
            "event": "stage_end",
            "stage": stage_name,
            "request_id": request_id,
            "duration_seconds": duration,
            "status": "success",
            **extra
        }))

    except Exception as e:
        duration = round(time.time() - start_time, 3)
        logger.error(json.dumps({
            "event": "stage_end",
            "stage": stage_name,
            "request_id": request_id,
            "duration_seconds": duration,
            "status": "error",
            "error": str(e),
            **extra
        }))
        raise