"""
Document Deduplication API Endpoints
Provides endpoints for managing document deduplication and cleanup operations.
"""

from typing import Dict, List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.core.config import get_settings
from src.core.logging import get_logger
from src.documents.deduplication import get_deduplication_service
from src.vector_store.chroma_client import get_chroma_client

settings = get_settings()
logger = get_logger(__name__)

router = APIRouter()


class DuplicateInfo(BaseModel):
    """Duplicate document information."""

    document_id: str = Field(..., description="Document ID")
    filename: str = Field(..., description="Document filename")
    file_hash: str = Field(..., description="File hash")
    metadata: Dict = Field(..., description="Document metadata")


class DuplicateGroup(BaseModel):
    """Group of duplicate documents."""

    file_hash: str = Field(..., description="File hash")
    documents: List[DuplicateInfo] = Field(
        ..., description="List of duplicate documents"
    )
    count: int = Field(..., description="Number of duplicates")


class DuplicateScanResponse(BaseModel):
    """Response for duplicate scan operation."""

    total_documents: int = Field(..., description="Total documents scanned")
    duplicate_groups: List[DuplicateGroup] = Field(
        ..., description="Groups of duplicate documents"
    )
    total_duplicates: int = Field(
        ..., description="Total number of duplicate documents"
    )
    total_groups: int = Field(..., description="Number of duplicate groups")


class CleanupStats(BaseModel):
    """Cleanup operation statistics."""

    groups_processed: int = Field(
        ..., description="Number of duplicate groups processed"
    )
    documents_removed: int = Field(..., description="Number of documents removed")
    errors: int = Field(..., description="Number of errors encountered")


class CleanupRequest(BaseModel):
    """Request for cleanup operation."""

    keep_strategy: str = Field(
        default="oldest",
        description="Strategy for which duplicate to keep ('oldest' or 'newest')",
    )
    dry_run: bool = Field(
        default=True,
        description="If true, only report what would be deleted without actually deleting",
    )


@router.get("/scan", response_model=DuplicateScanResponse)
async def scan_for_duplicates():
    """
    Scan for duplicate documents in the system.

    Returns:
        DuplicateScanResponse: Information about found duplicates
    """
    try:
        chroma_client = get_chroma_client()
        deduplication_service = get_deduplication_service(chroma_client)

        # Find existing duplicates
        duplicates = await deduplication_service.find_existing_duplicates()

        # Convert to response format
        duplicate_groups = []
        total_duplicates = 0

        for file_hash, duplicate_docs in duplicates.items():
            documents = []
            for doc in duplicate_docs:
                documents.append(
                    DuplicateInfo(
                        document_id=doc["id"],
                        filename=doc["metadata"].get("filename", "unknown"),
                        file_hash=file_hash,
                        metadata=doc["metadata"],
                    )
                )

            duplicate_groups.append(
                DuplicateGroup(
                    file_hash=file_hash, documents=documents, count=len(documents)
                )
            )
            total_duplicates += len(documents)

        # Get total document count
        stats = await chroma_client.get_collection_stats()
        total_documents = stats.get("document_count", 0)

        logger.info(
            "Duplicate scan completed",
            total_documents=total_documents,
            duplicate_groups=len(duplicate_groups),
            total_duplicates=total_duplicates,
        )

        return DuplicateScanResponse(
            total_documents=total_documents,
            duplicate_groups=duplicate_groups,
            total_duplicates=total_duplicates,
            total_groups=len(duplicate_groups),
        )

    except Exception as e:
        logger.error("Failed to scan for duplicates", error=str(e))
        raise HTTPException(
            status_code=500, detail=f"Failed to scan for duplicates: {str(e)}"
        )


@router.post("/cleanup", response_model=CleanupStats)
async def cleanup_duplicates(request: CleanupRequest):
    """
    Clean up duplicate documents from the system.

    Args:
        request: Cleanup configuration

    Returns:
        CleanupStats: Statistics about the cleanup operation
    """
    try:
        # Validate keep strategy
        if request.keep_strategy not in ["oldest", "newest"]:
            raise HTTPException(
                status_code=400, detail="keep_strategy must be 'oldest' or 'newest'"
            )

        chroma_client = get_chroma_client()
        deduplication_service = get_deduplication_service(chroma_client)

        # Perform cleanup
        stats = await deduplication_service.cleanup_duplicates(
            keep_strategy=request.keep_strategy, dry_run=request.dry_run
        )

        logger.info(
            "Duplicate cleanup completed",
            keep_strategy=request.keep_strategy,
            dry_run=request.dry_run,
            **stats,
        )

        return CleanupStats(**stats)

    except Exception as e:
        logger.error("Failed to cleanup duplicates", error=str(e))
        raise HTTPException(
            status_code=500, detail=f"Failed to cleanup duplicates: {str(e)}"
        )


@router.post("/cleanup-orphaned", response_model=CleanupStats)
async def cleanup_orphaned_files():
    """
    Clean up orphaned files in the uploads directory.

    Returns:
        CleanupStats: Statistics about the cleanup operation
    """
    try:
        chroma_client = get_chroma_client()
        deduplication_service = get_deduplication_service(chroma_client)

        # Cleanup orphaned files
        stats = await deduplication_service.cleanup_orphaned_files()

        # Convert to expected format
        cleanup_stats = CleanupStats(
            groups_processed=0,  # Not applicable for orphaned files
            documents_removed=stats.get("files_removed", 0),
            errors=stats.get("errors", 0),
        )

        logger.info(
            "Orphaned file cleanup completed",
            files_scanned=stats.get("files_scanned", 0),
            files_removed=stats.get("files_removed", 0),
            errors=stats.get("errors", 0),
        )

        return cleanup_stats

    except Exception as e:
        logger.error("Failed to cleanup orphaned files", error=str(e))
        raise HTTPException(
            status_code=500, detail=f"Failed to cleanup orphaned files: {str(e)}"
        )


@router.get("/check/{file_hash}")
async def check_duplicate_by_hash(file_hash: str):
    """
    Check if a document with the given hash already exists.

    Args:
        file_hash: SHA256 hash of the file

    Returns:
        Information about existing document or null if not found
    """
    try:
        chroma_client = get_chroma_client()
        deduplication_service = get_deduplication_service(chroma_client)

        # Check for duplicate
        duplicate_info = await deduplication_service.check_duplicate_by_hash(file_hash)

        if duplicate_info:
            return {
                "exists": True,
                "document_id": duplicate_info["id"],
                "metadata": duplicate_info["metadata"],
            }
        else:
            return {"exists": False}

    except Exception as e:
        logger.error(
            "Failed to check duplicate by hash", error=str(e), file_hash=file_hash
        )
        raise HTTPException(
            status_code=500, detail=f"Failed to check duplicate: {str(e)}"
        )


@router.get("/statistics")
async def get_deduplication_statistics():
    """
    Get statistics about duplicates in the system.

    Returns:
        Statistics about duplicates and system state
    """
    try:
        chroma_client = get_chroma_client()
        deduplication_service = get_deduplication_service(chroma_client)

        # Get collection stats
        collection_stats = await chroma_client.get_collection_stats()

        # Find duplicates
        duplicates = await deduplication_service.find_existing_duplicates()

        # Calculate statistics
        total_documents = collection_stats.get("document_count", 0)
        duplicate_groups = len(duplicates)
        total_duplicates = sum(len(docs) for docs in duplicates.values())
        unique_documents = (
            total_documents - total_duplicates + duplicate_groups
            if duplicate_groups > 0
            else total_documents
        )

        return {
            "total_documents": total_documents,
            "unique_documents": unique_documents,
            "duplicate_groups": duplicate_groups,
            "total_duplicates": total_duplicates,
            "duplication_rate": (total_duplicates / total_documents * 100)
            if total_documents > 0
            else 0.0,
            "space_saved_potential": total_duplicates - duplicate_groups
            if duplicate_groups > 0
            else 0,
        }

    except Exception as e:
        logger.error("Failed to get deduplication statistics", error=str(e))
        raise HTTPException(
            status_code=500, detail=f"Failed to get statistics: {str(e)}"
        )
