"""
Tests for document processing components.
Tests DocumentProcessorFactory, TextChunker, DocumentService, and models.
"""

import tempfile
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from src.documents.models import (
    DocumentChunk,
    DocumentMetadata,
    DocumentType,
    DocumentUpload,
    ProcessingStatus,
)


class TestDocumentModels:
    """Tests for document Pydantic models."""

    def test_document_upload_valid(self):
        """Test valid document upload model."""
        upload = DocumentUpload(
            filename="test.pdf",
            content_type="application/pdf",
            size=1024,
        )
        assert upload.filename == "test.pdf"
        assert upload.size == 1024

    def test_document_upload_file_too_large(self):
        """Test document upload rejects oversized files."""
        with pytest.raises(ValueError):
            DocumentUpload(
                filename="large.pdf",
                content_type="application/pdf",
                size=100 * 1024 * 1024,  # 100MB exceeds 50MB limit
            )

    def test_document_upload_invalid_extension(self):
        """Test document upload rejects unsupported file types."""
        with pytest.raises(ValueError):
            DocumentUpload(
                filename="test.exe",
                content_type="application/octet-stream",
                size=1024,
            )

    def test_document_metadata_defaults(self):
        """Test DocumentMetadata default values."""
        meta = DocumentMetadata(
            filename="test.txt",
            document_type=DocumentType.TXT,
            size=100,
            content_type="text/plain",
            uploaded_by="test_user",
        )
        assert meta.processing_status == ProcessingStatus.QUEUED
        assert meta.id is not None
        assert meta.chunk_count is None

    def test_document_chunk_creation(self):
        """Test DocumentChunk creation with required fields."""
        chunk = DocumentChunk(
            document_id=uuid.uuid4(),
            chunk_index=0,
            text="Sample chunk text",
            token_count=3,
            start_char=0,
            end_char=17,
        )
        assert chunk.chunk_index == 0
        assert chunk.embedding is None
        assert chunk.id is not None

    def test_processing_status_values(self):
        """Test ProcessingStatus enum values."""
        assert ProcessingStatus.QUEUED == "queued"
        assert ProcessingStatus.PROCESSING == "processing"
        assert ProcessingStatus.COMPLETED == "completed"
        assert ProcessingStatus.FAILED == "failed"

    def test_document_type_values(self):
        """Test DocumentType enum values."""
        assert DocumentType.PDF == "pdf"
        assert DocumentType.DOCX == "docx"
        assert DocumentType.TXT == "txt"
        assert DocumentType.CSV == "csv"


class TestDocumentProcessorFactory:
    """Tests for DocumentProcessorFactory."""

    def test_get_processor_pdf(self):
        """Test getting PDF processor."""
        from src.documents.processors import document_processor_factory

        processor = document_processor_factory.get_processor(
            "test.pdf", "application/pdf"
        )
        assert processor is not None

    def test_get_processor_txt(self):
        """Test getting TXT processor."""
        from src.documents.processors import document_processor_factory

        processor = document_processor_factory.get_processor("test.txt", "text/plain")
        assert processor is not None

    def test_get_processor_docx(self):
        """Test getting DOCX processor."""
        from src.documents.processors import document_processor_factory

        processor = document_processor_factory.get_processor(
            "test.docx",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )
        assert processor is not None

    def test_get_processor_csv(self):
        """Test getting CSV processor."""
        from src.documents.processors import document_processor_factory

        processor = document_processor_factory.get_processor("test.csv", "text/csv")
        assert processor is not None

    def test_get_processor_unsupported(self):
        """Test getting processor for unsupported type returns None."""
        from src.documents.processors import document_processor_factory

        processor = document_processor_factory.get_processor(
            "test.exe", "application/octet-stream"
        )
        assert processor is None

    def test_get_supported_types(self):
        """Test listing supported file types."""
        from src.documents.processors import document_processor_factory

        supported = document_processor_factory.get_supported_types()
        assert isinstance(supported, list)
        assert len(supported) > 0

    def test_validate_file_type_valid(self):
        """Test file type validation for supported type."""
        from src.documents.processors import document_processor_factory

        assert (
            document_processor_factory.validate_file_type("test.pdf", "application/pdf")
            is True
        )

    def test_validate_file_type_invalid(self):
        """Test file type validation for unsupported type."""
        from src.documents.processors import document_processor_factory

        assert (
            document_processor_factory.validate_file_type(
                "test.exe", "application/octet-stream"
            )
            is False
        )


class TestTXTProcessor:
    """Tests for TXT document processor."""

    @pytest.fixture
    def txt_file(self, tmp_path):
        """Create a temporary text file."""
        file_path = tmp_path / "test.txt"
        file_path.write_text(
            "This is a test document.\nWith multiple lines.\nFor testing purposes."
        )
        return str(file_path)

    def test_can_process_txt(self):
        """Test TXTProcessor recognizes .txt files."""
        from src.documents.processors import TXTProcessor

        processor = TXTProcessor()
        assert processor.can_process("document.txt", "text/plain") is True

    def test_can_process_non_txt(self):
        """Test TXTProcessor rejects non-.txt files."""
        from src.documents.processors import TXTProcessor

        processor = TXTProcessor()
        assert processor.can_process("document.pdf", "application/pdf") is False

    def test_get_document_type(self):
        """Test TXTProcessor returns TXT document type."""
        from src.documents.processors import TXTProcessor

        processor = TXTProcessor()
        assert processor.get_document_type() == DocumentType.TXT

    @pytest.mark.asyncio
    async def test_extract_text(self, txt_file):
        """Test text extraction from .txt file."""
        from src.documents.processors import TXTProcessor

        processor = TXTProcessor()
        text, metadata = await processor.extract_text(txt_file)
        assert "This is a test document" in text
        assert isinstance(metadata, dict)


class TestTextChunker:
    """Tests for TextChunker."""

    def test_count_tokens(self):
        """Test token counting."""
        from src.documents.chunking import TextChunker

        chunker = TextChunker(chunk_size=100, chunk_overlap=20)
        count = chunker.count_tokens("Hello world, this is a test.")
        assert isinstance(count, int)
        assert count > 0

    def test_chunk_text_basic(self, sample_text_content):
        """Test basic text chunking."""
        from src.documents.chunking import TextChunker

        chunker = TextChunker(chunk_size=50, chunk_overlap=10)
        doc_id = uuid.uuid4()
        chunks = chunker.chunk_text(
            sample_text_content,
            document_id=doc_id,
            metadata={"source_file": "test.txt", "document_type": "txt"},
        )
        assert len(chunks) > 0
        for chunk in chunks:
            assert isinstance(chunk, DocumentChunk)
            assert chunk.document_id == doc_id
            assert chunk.text
            assert chunk.token_count > 0

    def test_chunk_text_empty(self):
        """Test chunking empty text returns empty list."""
        from src.documents.chunking import TextChunker

        chunker = TextChunker(chunk_size=100, chunk_overlap=20)
        chunks = chunker.chunk_text(
            "",
            document_id=uuid.uuid4(),
            metadata={"source_file": "empty.txt", "document_type": "txt"},
        )
        assert chunks == []

    def test_chunk_indices_sequential(self, sample_text_content):
        """Test that chunk indices are sequential starting from 0."""
        from src.documents.chunking import TextChunker

        chunker = TextChunker(chunk_size=50, chunk_overlap=10)
        chunks = chunker.chunk_text(
            sample_text_content,
            document_id=uuid.uuid4(),
            metadata={"source_file": "test.txt", "document_type": "txt"},
        )
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i


class TestRecursiveCharacterTextSplitter:
    """Tests for RecursiveCharacterTextSplitter."""

    def test_chunk_with_separators(self, sample_text_content):
        """Test chunking with recursive separators."""
        from src.documents.chunking import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=10)
        chunks = splitter.chunk_text(
            sample_text_content,
            document_id=uuid.uuid4(),
            metadata={"source_file": "test.txt", "document_type": "txt"},
        )
        assert len(chunks) > 0
        for chunk in chunks:
            assert isinstance(chunk, DocumentChunk)

    def test_get_text_chunker_factory(self):
        """Test the get_text_chunker factory function."""
        from src.documents.chunking import get_text_chunker, TextChunker

        chunker = get_text_chunker(strategy="recursive")
        assert isinstance(chunker, TextChunker)


class TestDocumentDeduplication:
    """Tests for document deduplication service."""

    def test_calculate_file_hash(self):
        """Test file hash calculation."""
        from src.documents.deduplication import DocumentDeduplicationService

        service = DocumentDeduplicationService()
        hash1 = service.calculate_file_hash(b"test content")
        hash2 = service.calculate_file_hash(b"test content")
        hash3 = service.calculate_file_hash(b"different content")
        assert hash1 == hash2
        assert hash1 != hash3
        assert isinstance(hash1, str)

    def test_calculate_file_hash_deterministic(self):
        """Test hash is deterministic."""
        from src.documents.deduplication import DocumentDeduplicationService

        service = DocumentDeduplicationService()
        content = b"reproducible content"
        assert service.calculate_file_hash(content) == service.calculate_file_hash(
            content
        )
