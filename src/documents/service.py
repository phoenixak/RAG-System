"""
Document Processing Service
Complete document processing service with end-to-end pipeline orchestration.
"""

import asyncio
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from uuid import UUID

from src.core.config import get_settings
from src.core.logging import get_logger
from src.documents.chunking import get_text_chunker
from src.documents.deduplication import get_deduplication_service
from src.documents.embeddings import get_embedding_generator
from src.documents.models import (
    DocumentChunk,
    DocumentMetadata,
    DuplicateDocumentError,
    ProcessingStatus,
)
from src.documents.processors import DocumentProcessorFactory
from src.vector_store.chroma_client import get_chroma_client

settings = get_settings()
logger = get_logger(__name__)


class DocumentService:
    """
    Document processing service with async queue management and status tracking.
    Provides end-to-end pipeline orchestration for document processing.
    """

    def __init__(self):
        self.processor_factory = DocumentProcessorFactory()
        self.text_chunker = get_text_chunker()
        self.embedding_service = get_embedding_generator()
        self.chroma_client = None
        self.deduplication_service = None
        self.processing_queue = asyncio.Queue()
        self.processing_task = None
        self._documents: Dict[UUID, DocumentMetadata] = {}
        self._chunks: Dict[UUID, List[DocumentChunk]] = {}

    async def initialize(self):
        """Initialize the document service."""
        try:
            self.chroma_client = get_chroma_client()
            # Initialize deduplication service with ChromaDB client
            self.deduplication_service = get_deduplication_service(self.chroma_client)
            # No initialization needed for embedding generator
            pass

            # Start background processing task
            self.processing_task = asyncio.create_task(self._process_queue())
            logger.info("Document service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize document service: {e}")
            raise

    async def shutdown(self):
        """Shutdown the document service."""
        try:
            if self.processing_task:
                self.processing_task.cancel()
                try:
                    await self.processing_task
                except asyncio.CancelledError:
                    pass

            if hasattr(self.embedding_service, "close"):
                self.embedding_service.close()
            logger.info("Document service shutdown completed")
        except Exception as e:
            logger.error(f"Error during document service shutdown: {e}")

    async def upload_and_queue_document(
        self,
        file_content: bytes,
        filename: str,
        content_type: str,
        uploaded_by: str,
        metadata: Optional[Dict] = None,
    ) -> DocumentMetadata:
        """
        Upload document and queue for processing.

        Args:
            file_content: Raw file bytes
            filename: Original filename
            content_type: MIME type
            uploaded_by: User email
            metadata: Additional metadata

        Returns:
            DocumentMetadata: Document metadata with processing status
        """
        try:
            # Check for duplicates first if deduplication service is available
            if self.deduplication_service:
                duplicate_info = (
                    await self.deduplication_service.check_duplicate_by_content(
                        file_content, filename
                    )
                )
                if duplicate_info:
                    existing_metadata = duplicate_info["metadata"]
                    raise DuplicateDocumentError(
                        filename=filename,
                        existing_document_id=duplicate_info["id"],
                        existing_filename=existing_metadata.get("filename", "unknown"),
                        file_hash=self.deduplication_service.calculate_file_hash(
                            file_content
                        ),
                    )

            doc_id = uuid.uuid4()

            # Save file temporarily
            temp_dir = Path(
                getattr(settings, "upload_dir", None)
                or settings.upload_path
                or "uploads"
            )
            temp_dir.mkdir(exist_ok=True)

            temp_file = temp_dir / f"{doc_id}_{filename}"
            with open(temp_file, "wb") as f:
                f.write(file_content)

            # Create document metadata
            doc_metadata = DocumentMetadata(
                id=doc_id,
                filename=filename,
                document_type=self._get_document_type(filename),
                size=len(file_content),
                processing_status=ProcessingStatus.QUEUED,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                uploaded_by=uploaded_by,
                file_path=str(temp_file),
                content_type=content_type,
            )

            # Store metadata
            self._documents[doc_id] = doc_metadata

            # Queue for processing
            await self.processing_queue.put(doc_id)

            logger.info(
                "Document queued for processing",
                document_id=str(doc_id),
                filename=filename,
                size=len(file_content),
            )

            return doc_metadata

        except Exception as e:
            logger.error(f"Failed to upload document {filename}: {e}")
            raise

    async def get_document(self, document_id: UUID) -> Optional[DocumentMetadata]:
        """Get document metadata by ID."""
        return self._documents.get(document_id)

    async def list_documents(
        self,
        page: int = 1,
        page_size: int = 50,
        status_filter: Optional[ProcessingStatus] = None,
    ) -> Tuple[List[DocumentMetadata], int]:
        """
        List documents with pagination and filtering.

        Returns:
            Tuple of (documents, total_count)
        """
        documents = list(self._documents.values())

        # Apply status filter
        if status_filter:
            documents = [d for d in documents if d.processing_status == status_filter]

        total_count = len(documents)

        # Apply pagination
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        paginated_docs = documents[start_idx:end_idx]

        return paginated_docs, total_count

    async def delete_document(self, document_id: UUID) -> bool:
        """Delete document and all associated data."""
        try:
            doc_metadata = self._documents.get(document_id)
            if not doc_metadata:
                return False

            # Delete from vector store
            if self.chroma_client:
                try:
                    # Delete chunks associated with this document
                    await self.chroma_client.delete_documents_by_filter(
                        {"document_id": str(document_id)}
                    )
                except Exception as e:
                    logger.warning(f"Failed to delete from vector store: {e}")

            # Delete temporary file
            if doc_metadata.file_path and Path(doc_metadata.file_path).exists():
                Path(doc_metadata.file_path).unlink()

            # Remove from memory
            self._documents.pop(document_id, None)
            self._chunks.pop(document_id, None)

            logger.info("Document deleted", document_id=str(document_id))
            return True

        except Exception as e:
            logger.error(f"Failed to delete document {document_id}: {e}")
            return False

    async def get_document_chunks(
        self, document_id: UUID, page: int = 1, page_size: int = 50
    ) -> Tuple[List[DocumentChunk], int]:
        """Get chunks for a document with pagination."""
        chunks = self._chunks.get(document_id, [])
        total_count = len(chunks)

        # Apply pagination
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        paginated_chunks = chunks[start_idx:end_idx]

        return paginated_chunks, total_count

    async def get_processing_status(
        self, document_id: UUID
    ) -> Optional[ProcessingStatus]:
        """Get processing status for a document."""
        doc_metadata = self._documents.get(document_id)
        return doc_metadata.processing_status if doc_metadata else None

    async def get_processing_error(self, document_id: UUID) -> Optional[str]:
        """Get processing error for a document."""
        doc_metadata = self._documents.get(document_id)
        return doc_metadata.processing_error if doc_metadata else None

    async def get_queue_size(self) -> int:
        """Get current processing queue size."""
        return self.processing_queue.qsize()

    async def _process_queue(self):
        """Background task to process documents from queue."""
        logger.info("Document processing queue started")

        while True:
            try:
                # Wait for document to process
                document_id = await self.processing_queue.get()
                await self._process_document_internal(document_id)
                self.processing_queue.task_done()

            except asyncio.CancelledError:
                logger.info("Document processing queue cancelled")
                break
            except Exception as e:
                logger.error(f"Error in document processing queue: {e}")
                # Continue processing other documents
                continue

    async def _process_document_internal(self, document_id: UUID):
        """Internal document processing method."""
        try:
            doc_metadata = self._documents.get(document_id)
            if not doc_metadata:
                logger.error(f"Document {document_id} not found for processing")
                return

            # Update status to processing
            doc_metadata.processing_status = ProcessingStatus.PROCESSING
            doc_metadata.updated_at = datetime.utcnow()

            logger.info(
                "Starting document processing",
                document_id=str(document_id),
                filename=doc_metadata.filename,
            )

            # Get processor for file type
            processor = self.processor_factory.get_processor(
                doc_metadata.filename, doc_metadata.content_type
            )

            # Extract text using the saved file path
            extracted_text, metadata = await processor.extract_text(
                doc_metadata.file_path
            )

            # Update document metadata with extracted info
            doc_metadata.text_length = len(extracted_text)
            doc_metadata.title = metadata.get("title")
            doc_metadata.author = metadata.get("author")
            doc_metadata.subject = metadata.get("subject")
            doc_metadata.page_count = metadata.get("page_count")

            # Chunk text
            chunks = self.text_chunker.chunk_text(
                text=extracted_text,
                document_id=document_id,
                metadata={
                    "filename": doc_metadata.filename,
                    **metadata,
                },
            )

            # Update chunk count
            doc_metadata.chunk_count = len(chunks)

            # Generate embeddings and store in vector database
            if self.chroma_client and chunks:
                try:
                    # Prepare data for ChromaDB
                    texts = [chunk.text for chunk in chunks]
                    embeddings = await self.embedding_service.generate_embeddings_batch(
                        texts
                    )

                    # Calculate file hash for deduplication
                    file_hash = None
                    if self.deduplication_service and doc_metadata.file_path:
                        try:
                            file_hash = self.deduplication_service.calculate_file_hash_from_path(
                                doc_metadata.file_path
                            )
                        except Exception as e:
                            logger.warning(f"Failed to calculate file hash: {e}")

                    ids = [str(chunk.id) for chunk in chunks]
                    metadatas = []
                    for chunk in chunks:
                        chunk_metadata = {
                            "chunk_id": str(chunk.id),
                            "document_id": str(document_id),
                            "chunk_index": chunk.chunk_index,
                            "filename": doc_metadata.filename,
                            "token_count": chunk.token_count,
                            "page_number": chunk.page_number or 0,
                        }
                        # Add file hash for deduplication
                        if file_hash:
                            chunk_metadata["file_hash"] = file_hash
                        metadatas.append(chunk_metadata)

                    # Add to vector store using the client's add_documents method
                    await self.chroma_client.add_documents(
                        documents=texts,
                        embeddings=embeddings,
                        metadata=metadatas,
                        document_ids=ids,
                    )

                    logger.info(
                        "Added chunks to vector store",
                        document_id=str(document_id),
                        chunk_count=len(chunks),
                    )

                except Exception as e:
                    logger.error(f"Failed to add chunks to vector store: {e}")
                    doc_metadata.processing_error = f"Vector store error: {str(e)}"
                    doc_metadata.processing_status = ProcessingStatus.FAILED
                    return

            # Store chunks
            self._chunks[document_id] = chunks

            # Update status to completed
            doc_metadata.processing_status = ProcessingStatus.COMPLETED
            doc_metadata.updated_at = datetime.utcnow()

            logger.info(
                "Document processing completed",
                document_id=str(document_id),
                filename=doc_metadata.filename,
                chunk_count=len(chunks),
                text_length=len(extracted_text),
            )

        except Exception as e:
            logger.error(
                "Document processing failed",
                document_id=str(document_id),
                error=str(e),
            )

            # Update status to failed
            if doc_metadata:
                doc_metadata.processing_status = ProcessingStatus.FAILED
                doc_metadata.processing_error = str(e)
                doc_metadata.updated_at = datetime.utcnow()

    def _get_document_type(self, filename: str) -> str:
        """Get document type from filename."""
        suffix = Path(filename).suffix.lower()
        type_mapping = {
            ".pdf": "pdf",
            ".docx": "docx",
            ".txt": "txt",
            ".csv": "csv",
        }
        return type_mapping.get(suffix, "unknown")


# Global service instance
_document_service: Optional[DocumentService] = None


async def get_document_service() -> DocumentService:
    """Get the global document service instance."""
    global _document_service

    if _document_service is None:
        _document_service = DocumentService()
        await _document_service.initialize()

    return _document_service


async def shutdown_document_service():
    """Shutdown the global document service."""
    global _document_service

    if _document_service:
        await _document_service.shutdown()
        _document_service = None
