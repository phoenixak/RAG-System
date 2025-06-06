"""
Structured Logging Configuration
Provides comprehensive logging setup with JSON formatting and structured output.
"""

import logging
import sys
from typing import Any, Dict

import structlog
from structlog.stdlib import LoggerFactory

from src.core.config import get_settings

settings = get_settings()


def setup_logging() -> None:
    """Configure structured logging for the application."""

    # Configure structlog
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer()
            if settings.log_format == "json"
            else structlog.dev.ConsoleRenderer(colors=True),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelName(settings.log_level)
        ),
        logger_factory=LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Configure standard logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, settings.log_level.upper()),
    )

    # Set third-party library log levels
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("uvicorn.access").setLevel(logging.INFO)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)


def get_logger(name: str) -> structlog.BoundLogger:
    """Get a structured logger instance."""
    return structlog.get_logger(name)


class LoggerMixin:
    """Mixin class to add logging capabilities to other classes."""

    @property
    def logger(self) -> structlog.BoundLogger:
        """Get logger instance for this class."""
        return get_logger(self.__class__.__name__)


def log_request_response(
    method: str,
    url: str,
    status_code: int,
    response_time: float,
    user_id: str = None,
    **kwargs,
) -> None:
    """Log HTTP request/response with structured data."""
    logger = get_logger("api")

    log_data = {
        "method": method,
        "url": url,
        "status_code": status_code,
        "response_time_ms": round(response_time * 1000, 2),
        **kwargs,
    }

    if user_id:
        log_data["user_id"] = user_id

    if status_code >= 400:
        logger.error("HTTP request failed", **log_data)
    elif status_code >= 300:
        logger.warning("HTTP request redirected", **log_data)
    else:
        logger.info("HTTP request completed", **log_data)


def log_security_event(
    event_type: str,
    user_id: str = None,
    ip_address: str = None,
    details: Dict[str, Any] = None,
) -> None:
    """Log security-related events."""
    logger = get_logger("security")

    log_data = {
        "event_type": event_type,
    }

    if user_id:
        log_data["user_id"] = user_id
    if ip_address:
        log_data["ip_address"] = ip_address
    if details:
        log_data.update(details)

    logger.warning("Security event occurred", **log_data)


def log_performance_metric(
    operation: str, duration: float, metadata: Dict[str, Any] = None
) -> None:
    """Log performance metrics for monitoring."""
    logger = get_logger("performance")

    log_data = {
        "operation": operation,
        "duration_ms": round(duration * 1000, 2),
    }

    if metadata:
        log_data.update(metadata)

    # Log as warning if operation is slow
    if duration > 5.0:  # 5 seconds threshold
        logger.warning("Slow operation detected", **log_data)
    else:
        logger.info("Operation completed", **log_data)
