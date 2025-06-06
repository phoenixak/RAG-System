"""
Document Processing Models
Pydantic models for document upload, processing, and management.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator


class DuplicateDocumentError(Exception):
    """Exception raised when a duplicate document is detected."""

    def __init__(
        self,
        filename: str,
        existing_document_id: str,
        existing_filename: str,
        file_hash: str,
    ):
        self.filename = filename
        self.existing_document_id = existing_document_id
        self.existing_filename = existing_filename
        self.file_hash = file_hash
        super().__init__(
            f"Duplicate document detected: '{filename}' already exists as '{existing_filename}' "
            f"(ID: {existing_document_id}, Hash: {file_hash[:8]}...)"
        )


class ProcessingStatus(str, Enum):
    """Document processing status enumeration."""

    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class DocumentType(str, Enum):
    """Supported document types."""

    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"
    CSV = "csv"


class DocumentUpload(BaseModel):
    """Document upload request model."""

    filename: str = Field(..., description="Original filename")
    content_type: str = Field(..., description="MIME content type")
    size: int = Field(..., description="File size in bytes")

    @field_validator("size")
    @classmethod
    def validate_file_size(cls, v):
        max_size = 50 * 1024 * 1024  # 50MB
        if v > max_size:
            raise ValueError(f"File size {v} exceeds maximum allowed size {max_size}")
        return v

    @field_validator("filename")
    @classmethod
    def validate_filename(cls, v):
        if not v or len(v) > 255:
            raise ValueError("Filename must be between 1 and 255 characters")

        allowed_extensions = [".pdf", ".docx", ".txt", ".csv"]
        if not any(v.lower().endswith(ext) for ext in allowed_extensions):
            raise ValueError(f"File type not supported. Allowed: {allowed_extensions}")
        return v


class DocumentMetadata(BaseModel):
    """Document metadata model."""

    id: UUID = Field(default_factory=uuid4, description="Document unique identifier")
    filename: str = Field(..., description="Original filename")
    document_type: DocumentType = Field(..., description="Document type")
    size: int = Field(..., description="File size in bytes")
    content_type: str = Field(..., description="MIME content type")

    # Processing information
    processing_status: ProcessingStatus = Field(default=ProcessingStatus.QUEUED)
    processing_started_at: Optional[datetime] = None
    processing_completed_at: Optional[datetime] = None
    processing_error: Optional[str] = None

    # Content information
    text_length: Optional[int] = None
    chunk_count: Optional[int] = None
    page_count: Optional[int] = None

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    # User information
    uploaded_by: str = Field(..., description="User who uploaded the document")

    # File storage information
    file_path: Optional[str] = Field(None, description="Path to stored file")

    # Additional metadata extracted from document
    title: Optional[str] = None
    author: Optional[str] = None
    subject: Optional[str] = None
    keywords: Optional[List[str]] = None
    language: Optional[str] = None

    class Config:
        use_enum_values = True


class DocumentChunk(BaseModel):
    """Document chunk model."""

    id: UUID = Field(default_factory=uuid4, description="Chunk unique identifier")
    document_id: UUID = Field(..., description="Parent document ID")
    chunk_index: int = Field(..., description="Chunk sequence number")

    # Content
    text: str = Field(..., description="Chunk text content")
    token_count: int = Field(..., description="Number of tokens in chunk")

    # Position information
    start_char: int = Field(..., description="Start character position in document")
    end_char: int = Field(..., description="End character position in document")
    page_number: Optional[int] = None

    # Embedding information
    embedding: Optional[List[float]] = None
    embedding_model: Optional[str] = None

    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        validate_assignment = True


class DocumentResponse(BaseModel):
    """Document response model."""

    id: UUID
    filename: str
    document_type: DocumentType
    size: int
    processing_status: ProcessingStatus
    text_length: Optional[int] = None
    chunk_count: Optional[int] = None
    page_count: Optional[int] = None
    created_at: datetime
    updated_at: datetime
    uploaded_by: str

    # Optional detailed information
    title: Optional[str] = None
    author: Optional[str] = None
    subject: Optional[str] = None
    processing_error: Optional[str] = None

    class Config:
        use_enum_values = True


class ChunkResponse(BaseModel):
    """Document chunk response model."""

    id: UUID
    document_id: UUID
    chunk_index: int
    text: str
    token_count: int
    page_number: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime


class DocumentListResponse(BaseModel):
    """Document list response with pagination."""

    documents: List[DocumentResponse]
    total: int
    page: int
    page_size: int
    total_pages: int


class ChunkListResponse(BaseModel):
    """Chunk list response with pagination."""

    chunks: List[ChunkResponse]
    document_id: UUID
    total: int
    page: int
    page_size: int
    total_pages: int


class DocumentProcessingRequest(BaseModel):
    """Document processing request model."""

    document_id: UUID
    force_reprocess: bool = False


class DocumentProcessingResponse(BaseModel):
    """Document processing response model."""

    document_id: UUID
    status: ProcessingStatus
    message: str
    estimated_completion_time: Optional[datetime] = None


class DocumentStatsResponse(BaseModel):
    """Document statistics response."""

    total_documents: int
    total_chunks: int
    processing_queue_size: int
    avg_processing_time_seconds: Optional[float] = None
    storage_size_bytes: int

    # Status breakdown
    status_counts: Dict[ProcessingStatus, int] = Field(default_factory=dict)

    # Type breakdown
    type_counts: Dict[DocumentType, int] = Field(default_factory=dict)


class BulkDeleteRequest(BaseModel):
    """Bulk document deletion request."""

    document_ids: List[UUID] = Field(..., min_items=1, max_items=100)
    confirm_deletion: bool = Field(..., description="Must be True to confirm deletion")

    @field_validator("confirm_deletion")
    @classmethod
    def validate_confirmation(cls, v):
        if not v:
            raise ValueError("Deletion must be explicitly confirmed")
        return v


class BulkDeleteResponse(BaseModel):
    """Bulk document deletion response."""

    deleted_count: int
    failed_deletions: List[Dict[str, str]] = Field(default_factory=list)
    total_requested: int
