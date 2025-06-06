"""
Health Check Endpoints
Provides system health monitoring and service status endpoints.
"""

import time
from typing import Any, Dict

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.core.config import get_settings
from src.core.logging import get_logger

router = APIRouter()
settings = get_settings()
logger = get_logger(__name__)


class HealthResponse(BaseModel):
    """Health check response model."""

    status: str
    timestamp: float
    version: str
    environment: str
    uptime_seconds: float


class DetailedHealthResponse(BaseModel):
    """Detailed health check response model."""

    status: str
    timestamp: float
    version: str
    environment: str
    uptime_seconds: float
    services: Dict[str, str]
    metrics: Dict[str, Any]


# Application start time for uptime calculation
_start_time = time.time()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Basic health check endpoint.
    Returns simple status information.
    """
    current_time = time.time()
    uptime = current_time - _start_time

    health_data = {
        "status": "healthy",
        "timestamp": current_time,
        "version": settings.app_version,
        "environment": settings.environment,
        "uptime_seconds": round(uptime, 2),
    }

    logger.info("Health check requested", **health_data)

    return health_data


@router.get("/health/detailed", response_model=DetailedHealthResponse)
async def detailed_health_check():
    """
    Detailed health check endpoint.
    Returns comprehensive system status including service health.
    """
    current_time = time.time()
    uptime = current_time - _start_time

    # Check service health (placeholder for now)
    services_status = await check_services_health()

    # Collect basic metrics
    metrics = {
        "uptime_seconds": round(uptime, 2),
        "memory_usage_mb": 0,  # TODO: Implement actual memory monitoring
        "active_connections": 0,  # TODO: Implement connection monitoring
        "requests_processed": 0,  # TODO: Implement request counter
    }

    # Determine overall status
    overall_status = "healthy"
    if any(status != "healthy" for status in services_status.values()):
        overall_status = "degraded"

    health_data = {
        "status": overall_status,
        "timestamp": current_time,
        "version": settings.app_version,
        "environment": settings.environment,
        "uptime_seconds": round(uptime, 2),
        "services": services_status,
        "metrics": metrics,
    }

    logger.info("Detailed health check requested", **health_data)

    return health_data


@router.get("/health/ready")
async def readiness_check():
    """
    Kubernetes readiness probe endpoint.
    Returns 200 if the application is ready to serve traffic.
    """
    # Check if critical services are available
    services_status = await check_services_health()

    # Check for critical service failures
    critical_services = ["api_gateway"]  # Add more as services are implemented

    for service in critical_services:
        if services_status.get(service) != "healthy":
            logger.warning(f"Readiness check failed: {service} is not healthy")
            raise HTTPException(status_code=503, detail=f"Service not ready: {service}")

    return {"status": "ready"}


@router.get("/health/live")
async def liveness_check():
    """
    Kubernetes liveness probe endpoint.
    Returns 200 if the application is alive.
    """
    # Simple liveness check - if we can respond, we're alive
    current_time = time.time()
    uptime = current_time - _start_time

    # Consider the application dead if it's been running for too long without restart
    # This is a safety mechanism to force restarts in case of memory leaks
    max_uptime_hours = 24  # 24 hours
    max_uptime_seconds = max_uptime_hours * 3600

    if uptime > max_uptime_seconds and settings.environment == "production":
        logger.warning(
            "Liveness check failed: application uptime exceeded maximum",
            uptime_seconds=uptime,
            max_uptime_seconds=max_uptime_seconds,
        )
        raise HTTPException(
            status_code=503, detail="Application uptime exceeded maximum threshold"
        )

    return {"status": "alive", "uptime_seconds": round(uptime, 2)}


async def check_services_health() -> Dict[str, str]:
    """
    Check the health of all system services.
    Returns a dictionary with service names and their health status.
    """
    services_status = {
        "api_gateway": "healthy",  # Always healthy if we can respond
    }

    # Check ChromaDB connection
    try:
        from src.vector_store.chroma_client import get_chroma_client
        chroma_client = get_chroma_client()
        await chroma_client.health_check()
        services_status["vector_store"] = "healthy"
    except Exception as e:
        logger.error("Vector store health check failed", error=str(e))
        services_status["vector_store"] = "unhealthy"

    # Check embedding service
    try:
        from src.documents.embeddings import get_embedding_generator
        embedding_generator = get_embedding_generator()
        model_info = embedding_generator.get_model_info()
        services_status["embedding_service"] = "healthy"
    except Exception as e:
        logger.error("Embedding service health check failed", error=str(e))
        services_status["embedding_service"] = "unhealthy"

    # Check LLM service
    try:
        from src.llm.service import get_llm_service
        llm_service = get_llm_service()
        connection_test = await llm_service.test_connection()
        
        if connection_test["success"]:
            services_status["llm_service"] = "healthy"
        elif connection_test.get("demo_mode"):
            services_status["llm_service"] = "demo_mode"
        else:
            services_status["llm_service"] = "unhealthy"
    except Exception as e:
        logger.error("LLM service health check failed", error=str(e))
        services_status["llm_service"] = "unhealthy"

    # Check Redis connection (optional)
    try:
        # Redis is not critical for basic functionality in this implementation
        services_status["redis"] = "not_implemented"
    except Exception as e:
        logger.error("Redis health check failed", error=str(e))
        services_status["redis"] = "unhealthy"

    return services_status
