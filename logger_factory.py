import json
import logging
import contextvars
import uuid
from datetime import datetime, timezone

# Context variable to hold the pipeline_run_id seamlessly across the module
pipeline_run_id_ctx = contextvars.ContextVar("pipeline_run_id", default="UNKNOWN_RUN")

def generate_new_run_id() -> str:
    """Generate and set a new pipeline_run_id in the current context."""
    run_id = f"run_{uuid.uuid4().hex[:8]}"
    pipeline_run_id_ctx.set(run_id)
    return run_id

class JSONFormatter(logging.Formatter):
    """
    Standard Library JSON Formatter for structured logging.
    Injects contextvars and explicitly provided kwargs nicely.
    """
    def format(self, record):
        log_record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "pipeline_run_id": pipeline_run_id_ctx.get(),
        }

        if hasattr(record, "event_type"):
            log_record["event_type"] = record.event_type
        
        if hasattr(record, "metrics"):
            log_record["metrics"] = record.metrics
            
        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)

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
        logger.setLevel(logging.INFO)
        logger.propagate = False  # Prevent firing to root logger which might be basic text
        
    return logger
