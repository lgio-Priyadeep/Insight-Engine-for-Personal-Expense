import json
import logging
import contextvars
import uuid
from datetime import datetime, timezone

import config

# Context variable to hold the pipeline_run_id seamlessly across the module
pipeline_run_id_ctx: contextvars.ContextVar[str] = contextvars.ContextVar(
    "pipeline_run_id",
    default="no_pipeline_context"
)

def generate_new_run_id() -> str:
    """Generate a new pipeline_run_id."""
    return str(uuid.uuid4())

class JSONFormatter(logging.Formatter):
    """
    Standard Library JSON Formatter for structured logging.
    Injects contextvars and explicitly provided kwargs nicely.
    """
    def format(self, record):
        exc_type = record.exc_info[0] if record.exc_info else None
        log_record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "event_type": getattr(record, "event_type", "unspecified_event"),
            "pipeline_run_id": pipeline_run_id_ctx.get(),
            "stage": getattr(record, "stage", "unknown_stage"),
            "message": record.getMessage(),
            "error_type": exc_type.__name__ if exc_type else None,
            "traceback": self.formatException(record.exc_info) if record.exc_info else None,
        }

        if hasattr(record, "metrics"):
            log_record["metrics"] = record.metrics

        return json.dumps(log_record)

def get_logger(name: str) -> logging.Logger:
    """
    Returns a logger configured with JSONFormatter.
    """
    logger = logging.getLogger(name)
    
    # Avoid attaching multiple handlers if already configured
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
        logger.setLevel(getattr(logging, config.LOG_LEVEL))
        logger.propagate = False  # Prevent firing to root logger which might be basic text
        
    return logger
