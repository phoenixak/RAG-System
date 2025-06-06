"""
Document Management API Routes
FastAPI routes for document upload, processing, and management.
"""

import mimetypes
from typing import Optional
from uuid import UUID

from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    Query,
    UploadFile,
    status,
)

from src.auth.models import User
from src.auth.security import get_current_user
from src.core.config import get_settings
from src.core.logging import get_logger
from src.documents.models import (
    BulkDeleteRequest,
    BulkDeleteResponse,
    ChunkListResponse,
    DocumentListResponse,
    DocumentProcessingRequest,
    DocumentProcessingResponse,
    DocumentResponse,
    DocumentStatsResponse,
    ProcessingStatus,
)
from src.documents.service import get_document_service

settings = get_settings()
logger = get_logger(__name__)
router = APIRouter()


@router.post(
    "/upload",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload document for processing",
)
async def upload_document(
    file: UploadFile = File(..., description="Document to upload"),
    current_user: User = Depends(get_current_user),
) -> DocumentResponse:
    """
    Upload one or more documents for processing.

    - **files**: Document files (PDF, DOCX, TXT, CSV)
    - Maximum file size: 50MB per file
    - Supported formats: .pdf, .docx, .txt, .csv

    Returns document metadata with processing status.
    """
    if not file or not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="No file provided"
        )

    document_service = await get_document_service()

    try:
        # Validate file
        if not file.filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Filename is required",
            )

        # Read file content
        file_content = await file.read()

        # Validate file size
        if len(file_content) > settings.max_file_size:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File {file.filename} exceeds maximum size of {settings.max_file_size} bytes",
            )

        # Validate file type
        content_type = (
            file.content_type
            or mimetypes.guess_type(file.filename)[0]
            or "application/octet-stream"
        )

        # Upload and queue document
        doc_metadata = await document_service.upload_and_queue_document(
            file_content=file_content,
            filename=file.filename,
            content_type=content_type,
            uploaded_by=current_user.email,
            metadata={
                "upload_timestamp": "utcnow",
                "user_id": str(
                    current_user.user_id
                ),  # Fixed: use user_id instead of id
                "ip_address": "unknown",  # Would be extracted from request in real implementation
            },
        )

        # Convert to response model
        response = DocumentResponse(
            id=doc_metadata.id,
            filename=doc_metadata.filename,
            document_type=doc_metadata.document_type,
            size=doc_metadata.size,
            processing_status=doc_metadata.processing_status,
            created_at=doc_metadata.created_at,
            updated_at=doc_metadata.updated_at,
            uploaded_by=doc_metadata.uploaded_by,
            title=doc_metadata.title,
            author=doc_metadata.author,
            subject=doc_metadata.subject,
        )

        logger.info(
            "Document uploaded successfully",
            document_id=str(doc_metadata.id),
            filename=file.filename,
            size=len(file_content),
            user_email=current_user.email,
        )

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Document upload failed",
            filename=file.filename,
            error=str(e),
            user_email=current_user.email,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to upload {file.filename}: {str(e)}",
        )


@router.get(
    "",
    response_model=DocumentListResponse,
    summary="List documents with pagination and filtering",
)
async def list_documents(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(50, ge=1, le=100, description="Items per page"),
    status_filter: Optional[ProcessingStatus] = Query(
        None, description="Filter by processing status"
    ),
    search: Optional[str] = Query(None, description="Search in filename or title"),
    current_user: User = Depends(get_current_user),
) -> DocumentListResponse:
    """
    List documents with pagination and filtering.

    - **page**: Page number (starts from 1)
    - **page_size**: Number of items per page (1-100)
    - **status_filter**: Filter by processing status
    - **search**: Search in filename or title
    """
    document_service = await get_document_service()

    try:
        documents, total_count = await document_service.list_documents(
            page=page, page_size=page_size, status_filter=status_filter
        )

        # Convert to response models
        document_responses = [
            DocumentResponse(
                id=doc.id,
                filename=doc.filename,
                document_type=doc.document_type,
                size=doc.size,
                processing_status=doc.processing_status,
                text_length=doc.text_length,
                chunk_count=doc.chunk_count,
                page_count=doc.page_count,
                created_at=doc.created_at,
                updated_at=doc.updated_at,
                uploaded_by=doc.uploaded_by,
                title=doc.title,
                author=doc.author,
                subject=doc.subject,
                processing_error=doc.processing_error,
            )
            for doc in documents
        ]

        total_pages = (total_count + page_size - 1) // page_size

        return DocumentListResponse(
            documents=document_responses,
            total=total_count,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
        )

    except Exception as e:
        logger.error(
            "Failed to list documents", error=str(e), user_email=current_user.email
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve documents",
        )


@router.get(
    "/{document_id}", response_model=DocumentResponse, summary="Get document details"
)
async def get_document(
    document_id: UUID, current_user: User = Depends(get_current_user)
) -> DocumentResponse:
    """
    Get detailed information about a specific document.

    - **document_id**: Document UUID
    """
    document_service = await get_document_service()

    try:
        doc_metadata = await document_service.get_document(document_id)

        if not doc_metadata:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document {document_id} not found",
            )

        return DocumentResponse(
            id=doc_metadata.id,
            filename=doc_metadata.filename,
            document_type=doc_metadata.document_type,
            size=doc_metadata.size,
            processing_status=doc_metadata.processing_status,
            text_length=doc_metadata.text_length,
            chunk_count=doc_metadata.chunk_count,
            page_count=doc_metadata.page_count,
            created_at=doc_metadata.created_at,
            updated_at=doc_metadata.updated_at,
            uploaded_by=doc_metadata.uploaded_by,
            title=doc_metadata.title,
            author=doc_metadata.author,
            subject=doc_metadata.subject,
            processing_error=doc_metadata.processing_error,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Failed to get document",
            document_id=str(document_id),
            error=str(e),
            user_email=current_user.email,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve document",
        )


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete document and all associated data",
)
async def delete_document(
    document_id: UUID, current_user: User = Depends(get_current_user)
) -> None:
    """
    Delete a document and all its associated chunks and embeddings.

    - **document_id**: Document UUID

    This operation cannot be undone.
    """
    document_service = await get_document_service()

    try:
        # Check if document exists
        doc_metadata = await document_service.get_document(document_id)
        if not doc_metadata:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document {document_id} not found",
            )

        # Delete document
        success = await document_service.delete_document(document_id)

        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to delete document",
            )

        logger.info(
            "Document deleted",
            document_id=str(document_id),
            filename=doc_metadata.filename,
            user_email=current_user.email,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Failed to delete document",
            document_id=str(document_id),
            error=str(e),
            user_email=current_user.email,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete document",
        )


@router.get(
    "/{document_id}/chunks",
    response_model=ChunkListResponse,
    summary="Get document chunks with pagination",
)
async def get_document_chunks(
    document_id: UUID,
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(50, ge=1, le=100, description="Items per page"),
    current_user: User = Depends(get_current_user),
) -> ChunkListResponse:
    """
    Get chunks for a specific document with pagination.

    - **document_id**: Document UUID
    - **page**: Page number (starts from 1)
    - **page_size**: Number of chunks per page (1-100)
    """
    document_service = await get_document_service()

    try:
        # Check if document exists
        doc_metadata = await document_service.get_document(document_id)
        if not doc_metadata:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document {document_id} not found",
            )

        chunks, total_count = await document_service.get_document_chunks(
            document_id=document_id, page=page, page_size=page_size
        )

        # Convert to response models
        chunk_responses = [
            {
                "id": chunk.id,
                "document_id": chunk.document_id,
                "chunk_index": chunk.chunk_index,
                "text": chunk.text,
                "token_count": chunk.token_count,
                "page_number": chunk.page_number,
                "metadata": chunk.metadata,
                "created_at": chunk.created_at,
            }
            for chunk in chunks
        ]

        total_pages = (total_count + page_size - 1) // page_size

        return ChunkListResponse(
            chunks=chunk_responses,
            document_id=document_id,
            total=total_count,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Failed to get document chunks",
            document_id=str(document_id),
            error=str(e),
            user_email=current_user.email,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve document chunks",
        )


@router.get(
    "/{document_id}/status",
    response_model=DocumentProcessingResponse,
    summary="Get document processing status",
)
async def get_processing_status(
    document_id: UUID, current_user: User = Depends(get_current_user)
) -> DocumentProcessingResponse:
    """
    Get the current processing status of a document.

    - **document_id**: Document UUID
    """
    document_service = await get_document_service()

    try:
        status = await document_service.get_processing_status(document_id)
        error = await document_service.get_processing_error(document_id)

        if status is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document {document_id} not found",
            )

        message = {
            ProcessingStatus.QUEUED: "Document is queued for processing",
            ProcessingStatus.PROCESSING: "Document is currently being processed",
            ProcessingStatus.COMPLETED: "Document processing completed successfully",
            ProcessingStatus.FAILED: f"Document processing failed: {error or 'Unknown error'}",
        }.get(status, "Unknown status")

        return DocumentProcessingResponse(
            document_id=document_id, status=status, message=message
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Failed to get processing status",
            document_id=str(document_id),
            error=str(e),
            user_email=current_user.email,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve processing status",
        )


@router.post(
    "/{document_id}/reprocess",
    response_model=DocumentProcessingResponse,
    summary="Reprocess document",
)
async def reprocess_document(
    document_id: UUID,
    request: DocumentProcessingRequest,
    current_user: User = Depends(get_current_user),
) -> DocumentProcessingResponse:
    """
    Reprocess a document (force re-extraction and re-chunking).

    - **document_id**: Document UUID
    - **force_reprocess**: Force reprocessing even if already completed
    """
    # This would require additional implementation in the service
    # For now, return a placeholder response
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Document reprocessing is not yet implemented",
    )


@router.delete(
    "/bulk", response_model=BulkDeleteResponse, summary="Bulk delete documents"
)
async def bulk_delete_documents(
    request: BulkDeleteRequest, current_user: User = Depends(get_current_user)
) -> BulkDeleteResponse:
    """
    Delete multiple documents at once.

    - **document_ids**: List of document UUIDs to delete
    - **confirm_deletion**: Must be true to confirm bulk deletion

    This operation cannot be undone.
    """
    document_service = await get_document_service()

    deleted_count = 0
    failed_deletions = []

    for document_id in request.document_ids:
        try:
            success = await document_service.delete_document(document_id)
            if success:
                deleted_count += 1
            else:
                failed_deletions.append(
                    {
                        "document_id": str(document_id),
                        "error": "Delete operation returned false",
                    }
                )
        except Exception as e:
            failed_deletions.append({"document_id": str(document_id), "error": str(e)})

    logger.info(
        "Bulk delete completed",
        requested_count=len(request.document_ids),
        deleted_count=deleted_count,
        failed_count=len(failed_deletions),
        user_email=current_user.email,
    )

    return BulkDeleteResponse(
        deleted_count=deleted_count,
        failed_deletions=failed_deletions,
        total_requested=len(request.document_ids),
    )


@router.get(
    "/stats", response_model=DocumentStatsResponse, summary="Get document statistics"
)
async def get_document_stats(
    current_user: User = Depends(get_current_user),
) -> DocumentStatsResponse:
    """
    Get overall document statistics and system status.
    """
    document_service = await get_document_service()

    try:
        queue_size = await document_service.get_queue_size()

        # This would require additional implementation to get real stats
        # For now, return placeholder data
        return DocumentStatsResponse(
            total_documents=0,
            total_chunks=0,
            processing_queue_size=queue_size,
            storage_size_bytes=0,
            status_counts={},
            type_counts={},
        )

    except Exception as e:
        logger.error(
            "Failed to get document stats", error=str(e), user_email=current_user.email
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve document statistics",
        )
