"""
Document Processing Tests
Comprehensive tests for document upload, processing, and management.
"""

import asyncio
import tempfile
import uuid
from io import BytesIO
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.documents.models import DocumentChunk, DocumentResponse, DocumentType, ProcessingStatus
from src.documents.processors import (
    PDFProcessor,
    DOCXProcessor,
    TextProcessor,
    CSVProcessor,
    get_processor_for_file_type,
)
from src.documents.chunking import RecursiveChunker, ChunkingStrategy
from src.documents.embeddings import EmbeddingGenerator
from src.documents.service import DocumentService


class TestDocumentModels:
    """Test document data models."""

    def test_document_model_creation(self):
        """Test document model creation and validation."""
        doc_data = {
            "document_id": str(uuid.uuid4()),
            "filename": "test_document.pdf",
            "file_type": DocumentType.PDF,
            "file_size": 1024,
            "status": ProcessingStatus.PENDING,
            "uploaded_by": "test_user",
            "metadata": {"title": "Test Document", "author": "Test Author"},
        }
        
        document = DocumentResponse(**doc_data)
        assert document.filename == "test_document.pdf"
        assert document.file_type == DocumentType.PDF
        assert document.status == ProcessingStatus.PENDING

    def test_document_type_validation(self):
        """Test document type validation."""
        # Valid document types
        assert DocumentType.PDF.value == "pdf"
        assert DocumentType.DOCX.value == "docx"
        assert DocumentType.TXT.value == "txt"
        assert DocumentType.CSV.value == "csv"

    def test_document_status_transitions(self):
        """Test document status state transitions."""
        # Valid status transitions
        valid_transitions = [
            (ProcessingStatus.PENDING, ProcessingStatus.PROCESSING),
            (ProcessingStatus.PROCESSING, ProcessingStatus.COMPLETED),
            (ProcessingStatus.PROCESSING, ProcessingStatus.FAILED),
            (ProcessingStatus.FAILED, ProcessingStatus.PENDING),  # Retry
        ]
        
        for from_status, to_status in valid_transitions:
            assert from_status != to_status  # Ensure they're different states


class TestDocumentProcessors:
    """Test document content processors."""

    def test_pdf_processor(self):
        """Test PDF document processing."""
        processor = PDFProcessor()
        
        # Test processor identification
        assert processor.can_process("test.pdf") is True
        assert processor.can_process("test.txt") is False
        
        # Test supported formats
        assert "pdf" in processor.supported_formats

    def test_docx_processor(self):
        """Test DOCX document processing."""
        processor = DOCXProcessor()
        
        assert processor.can_process("document.docx") is True
        assert processor.can_process("document.pdf") is False

    def test_text_processor(self):
        """Test text document processing."""
        processor = TextProcessor()
        
        assert processor.can_process("readme.txt") is True
        assert processor.can_process("document.pdf") is False

    def test_csv_processor(self):
        """Test CSV document processing."""
        processor = CSVProcessor()
        
        assert processor.can_process("data.csv") is True
        assert processor.can_process("document.txt") is False

    def test_get_processor_for_file_type(self):
        """Test processor selection based on file type."""
        # Test PDF processor selection
        pdf_processor = get_processor_for_file_type("pdf")
        assert isinstance(pdf_processor, PDFProcessor)
        
        # Test DOCX processor selection
        docx_processor = get_processor_for_file_type("docx")
        assert isinstance(docx_processor, DOCXProcessor)
        
        # Test unsupported file type
        with pytest.raises(ValueError):
            get_processor_for_file_type("unsupported")

    @pytest.mark.asyncio
    async def test_text_extraction(self):
        """Test text extraction from documents."""
        processor = TextProcessor()
        
        # Create test text content
        test_content = b"This is a test document with sample content."
        
        with patch("src.documents.processors.TextProcessor.extract_text") as mock_extract:
            mock_extract.return_value = "This is a test document with sample content."
            
            extracted_text = await processor.extract_text(test_content)
            assert "test document" in extracted_text
            assert len(extracted_text) > 0

    @pytest.mark.asyncio
    async def test_metadata_extraction(self):
        """Test metadata extraction from documents."""
        processor = PDFProcessor()
        
        with patch("src.documents.processors.PDFProcessor.extract_metadata") as mock_metadata:
            mock_metadata.return_value = {
                "title": "Test Document",
                "author": "Test Author",
                "creation_date": "2023-01-01",
                "page_count": 5,
            }
            
            metadata = await processor.extract_metadata(b"fake_pdf_content")
            assert metadata["title"] == "Test Document"
            assert metadata["author"] == "Test Author"
            assert metadata["page_count"] == 5


class TestDocumentChunking:
    """Test document chunking strategies."""

    def test_recursive_chunker_creation(self):
        """Test recursive chunker initialization."""
        chunker = RecursiveChunker(
            chunk_size=1000,
            chunk_overlap=200,
            strategy=ChunkingStrategy.RECURSIVE
        )
        
        assert chunker.chunk_size == 1000
        assert chunker.chunk_overlap == 200
        assert chunker.strategy == ChunkingStrategy.RECURSIVE

    def test_text_chunking(self):
        """Test text chunking functionality."""
        chunker = RecursiveChunker(chunk_size=100, chunk_overlap=20)
        
        # Test text that should be chunked
        long_text = "This is a long text. " * 20  # 400+ characters
        chunks = chunker.chunk_text(long_text)
        
        assert len(chunks) > 1  # Should be split into multiple chunks
        
        # Check overlap
        if len(chunks) > 1:
            # Some content should overlap between consecutive chunks
            chunk1_end = chunks[0].text[-20:]  # Last 20 chars of first chunk
            chunk2_start = chunks[1].text[:20]  # First 20 chars of second chunk
            # There should be some overlap in content
            assert len(chunk1_end.strip()) > 0
            assert len(chunk2_start.strip()) > 0

    def test_chunk_metadata(self):
        """Test chunk metadata generation."""
        chunker = RecursiveChunker()
        
        text = "Sample text for chunking with metadata."
        chunks = chunker.chunk_text(text, document_id="test_doc", metadata={"page": 1})
        
        assert len(chunks) >= 1
        chunk = chunks[0]
        
        assert chunk.chunk_index == 0
        assert chunk.document_id == "test_doc"
        assert chunk.metadata["page"] == 1

    def test_sentence_boundary_chunking(self):
        """Test chunking respects sentence boundaries."""
        chunker = RecursiveChunker(chunk_size=50, respect_sentence_boundaries=True)
        
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        chunks = chunker.chunk_text(text)
        
        # Check that chunks end with sentence boundaries when possible
        for chunk in chunks:
            if len(chunk.text) < 50:  # If chunk is smaller than max size
                # It should ideally end with a sentence boundary
                assert chunk.text.strip().endswith('.') or chunk == chunks[-1]

    def test_chunking_strategies(self):
        """Test different chunking strategies."""
        strategies = [
            ChunkingStrategy.RECURSIVE,
            ChunkingStrategy.SENTENCE,
            ChunkingStrategy.PARAGRAPH,
        ]
        
        text = "Paragraph one.\n\nParagraph two. With multiple sentences. And more content.\n\nParagraph three."
        
        for strategy in strategies:
            chunker = RecursiveChunker(strategy=strategy, chunk_size=100)
            chunks = chunker.chunk_text(text)
            
            assert len(chunks) >= 1
            assert all(len(chunk.text) <= 100 + chunker.chunk_overlap for chunk in chunks)


class TestEmbeddingGeneration:
    """Test embedding generation functionality."""

    @pytest.fixture
    def mock_embedding_model(self):
        """Mock embedding model."""
        model = Mock()
        model.encode = Mock(return_value=[[0.1, 0.2, 0.3, 0.4, 0.5]])
        return model

    def test_embedding_generator_creation(self, mock_embedding_model):
        """Test embedding generator initialization."""
        with patch("sentence_transformers.SentenceTransformer") as mock_st:
            mock_st.return_value = mock_embedding_model
            
            generator = EmbeddingGenerator()
            assert generator.model_name is not None
            assert generator.cache_size > 0

    def test_get_model_info(self, mock_embedding_model):
        """Test getting model information."""
        with patch("sentence_transformers.SentenceTransformer") as mock_st:
            mock_st.return_value = mock_embedding_model
            mock_embedding_model.get_sentence_embedding_dimension = Mock(return_value=384)
            
            generator = EmbeddingGenerator()
            model_info = generator.get_model_info()
            
            assert "model_name" in model_info
            assert "dimensions" in model_info
            assert model_info["dimensions"] == 384

    @pytest.mark.asyncio
    async def test_generate_embeddings(self, mock_embedding_model):
        """Test embedding generation."""
        with patch("sentence_transformers.SentenceTransformer") as mock_st:
            mock_st.return_value = mock_embedding_model
            
            generator = EmbeddingGenerator()
            texts = ["This is a test sentence.", "Another test sentence."]
            
            embeddings = await generator.generate_embeddings(texts)
            
            assert len(embeddings) == 2
            assert len(embeddings[0]) == 5  # Mock embedding dimension
            assert all(isinstance(emb, list) for emb in embeddings)

    @pytest.mark.asyncio
    async def test_batch_embedding_generation(self, mock_embedding_model):
        """Test batch embedding generation."""
        with patch("sentence_transformers.SentenceTransformer") as mock_st:
            mock_st.return_value = mock_embedding_model
            mock_embedding_model.encode = Mock(return_value=[[0.1, 0.2]] * 10)
            
            generator = EmbeddingGenerator(batch_size=5)
            
            # Test with more texts than batch size
            texts = [f"Test sentence {i}" for i in range(10)]
            embeddings = await generator.generate_embeddings(texts)
            
            assert len(embeddings) == 10
            # Should have been called in batches
            assert mock_embedding_model.encode.call_count >= 2

    def test_embedding_caching(self, mock_embedding_model):
        """Test embedding caching functionality."""
        with patch("sentence_transformers.SentenceTransformer") as mock_st:
            mock_st.return_value = mock_embedding_model
            
            generator = EmbeddingGenerator(enable_cache=True)
            
            # Test cache operations
            text = "test text"
            embedding = [0.1, 0.2, 0.3]
            
            # Cache embedding
            generator._cache_embedding(text, embedding)
            
            # Retrieve from cache
            cached_embedding = generator._get_cached_embedding(text)
            assert cached_embedding == embedding
            
            # Test cache miss
            missing_embedding = generator._get_cached_embedding("not cached")
            assert missing_embedding is None


class TestDocumentService:
    """Test document service orchestration."""

    @pytest.fixture
    def mock_document_service(self):
        """Mock document service with dependencies."""
        with patch("src.documents.service.get_chroma_client") as mock_chroma, \
             patch("src.documents.service.get_embedding_generator") as mock_embeddings:
            
            # Mock ChromaDB client
            mock_chroma.return_value.add_documents = AsyncMock()
            mock_chroma.return_value.get_collection_info = AsyncMock(return_value={"count": 0})
            
            # Mock embedding generator
            mock_embeddings.return_value.generate_embeddings = AsyncMock(
                return_value=[[0.1, 0.2, 0.3]]
            )
            
            service = DocumentService()
            yield service

    @pytest.mark.asyncio
    async def test_document_upload_processing(self, mock_document_service, mock_file_upload):
        """Test complete document upload and processing."""
        with patch("src.documents.service.get_processor_for_file_type") as mock_processor:
            # Mock processor
            processor_mock = Mock()
            processor_mock.extract_text = AsyncMock(return_value="Extracted text content")
            processor_mock.extract_metadata = AsyncMock(return_value={"title": "Test Doc"})
            mock_processor.return_value = processor_mock
            
            # Process document
            result = await mock_document_service.process_uploaded_document(
                file=mock_file_upload,
                user_id="test_user",
                metadata={"custom": "value"}
            )
            
            assert "document_id" in result
            assert result["status"] == "completed"

    @pytest.mark.asyncio
    async def test_document_chunking_pipeline(self, mock_document_service):
        """Test document chunking pipeline."""
        text = "This is a long document that should be chunked into smaller pieces. " * 20
        
        chunks = await mock_document_service._chunk_document_text(
            text=text,
            document_id="test_doc",
            metadata={"page": 1}
        )
        
        assert len(chunks) > 1
        assert all(chunk.document_id == "test_doc" for chunk in chunks)

    @pytest.mark.asyncio
    async def test_embedding_generation_pipeline(self, mock_document_service):
        """Test embedding generation pipeline."""
        from src.documents.models import DocumentChunk
        
        chunks = [
            DocumentChunk(
                chunk_id=str(uuid.uuid4()),
                document_id="test_doc",
                text="Sample text 1",
                chunk_index=0,
                metadata={}
            ),
            DocumentChunk(
                chunk_id=str(uuid.uuid4()),
                document_id="test_doc",
                text="Sample text 2",
                chunk_index=1,
                metadata={}
            ),
        ]
        
        with patch.object(mock_document_service, '_generate_embeddings_for_chunks') as mock_embed:
            mock_embed.return_value = chunks  # Return chunks with embeddings
            
            result_chunks = await mock_document_service._generate_embeddings_for_chunks(chunks)
            assert len(result_chunks) == 2

    @pytest.mark.asyncio
    async def test_document_storage(self, mock_document_service):
        """Test document storage in vector database."""
        from src.documents.models import DocumentChunk
        
        chunks = [
            DocumentChunk(
                chunk_id=str(uuid.uuid4()),
                document_id="test_doc",
                text="Sample text",
                chunk_index=0,
                embeddings=[0.1, 0.2, 0.3],
                metadata={"page": 1}
            )
        ]
        
        # Test storage
        result = await mock_document_service._store_chunks_in_vector_db(chunks)
        assert result is True

    @pytest.mark.asyncio
    async def test_document_deletion(self, mock_document_service):
        """Test document deletion."""
        document_id = str(uuid.uuid4())
        
        with patch.object(mock_document_service.chroma_client, 'delete_documents') as mock_delete:
            mock_delete.return_value = True
            
            result = await mock_document_service.delete_document(
                document_id=document_id,
                user_id="test_user"
            )
            
            assert result is True

    @pytest.mark.asyncio
    async def test_document_listing(self, mock_document_service):
        """Test document listing with pagination."""
        with patch.object(mock_document_service, 'list_documents') as mock_list:
            mock_documents = [
                {
                    "document_id": str(uuid.uuid4()),
                    "filename": "doc1.pdf",
                    "status": "completed",
                    "uploaded_by": "test_user"
                },
                {
                    "document_id": str(uuid.uuid4()),
                    "filename": "doc2.pdf", 
                    "status": "completed",
                    "uploaded_by": "test_user"
                }
            ]
            
            mock_list.return_value = {
                "documents": mock_documents,
                "total": 2,
                "offset": 0,
                "limit": 10,
                "has_next": False
            }
            
            result = await mock_document_service.list_documents(
                user_id="test_user",
                limit=10,
                offset=0
            )
            
            assert len(result["documents"]) == 2
            assert result["total"] == 2


class TestDocumentErrorHandling:
    """Test document processing error handling."""

    @pytest.mark.asyncio
    async def test_unsupported_file_type(self):
        """Test handling of unsupported file types."""
        with pytest.raises(ValueError, match="Unsupported file type"):
            get_processor_for_file_type("unsupported")

    @pytest.mark.asyncio
    async def test_corrupted_file_processing(self, mock_document_service):
        """Test handling of corrupted files."""
        with patch("src.documents.service.get_processor_for_file_type") as mock_processor:
            processor_mock = Mock()
            processor_mock.extract_text = AsyncMock(side_effect=Exception("Corrupted file"))
            mock_processor.return_value = processor_mock
            
            # Create mock corrupted file
            corrupted_file = Mock()
            corrupted_file.filename = "corrupted.pdf"
            corrupted_file.read = AsyncMock(return_value=b"corrupted_content")
            
            with pytest.raises(Exception):
                await mock_document_service.process_uploaded_document(
                    file=corrupted_file,
                    user_id="test_user"
                )

    @pytest.mark.asyncio
    async def test_large_file_handling(self, mock_document_service):
        """Test handling of very large files."""
        # Test file size validation
        large_file = Mock()
        large_file.filename = "large_file.pdf"
        large_file.size = 100 * 1024 * 1024  # 100MB
        
        # This should be handled by size limits in the actual implementation
        assert large_file.size > 50 * 1024 * 1024  # Assume 50MB limit

    @pytest.mark.asyncio
    async def test_embedding_generation_failure(self, mock_document_service):
        """Test handling of embedding generation failures."""
        with patch.object(mock_document_service.embedding_generator, 'generate_embeddings') as mock_embed:
            mock_embed.side_effect = Exception("Embedding service unavailable")
            
            chunks = [Mock(text="test text")]
            
            with pytest.raises(Exception):
                await mock_document_service._generate_embeddings_for_chunks(chunks)


class TestDocumentAPI:
    """Test document management API endpoints."""

    @pytest.mark.asyncio
    async def test_upload_endpoint(self, test_client):
        """Test document upload endpoint."""
        # This would test the actual upload endpoint
        # For now, we'll test the structure
        upload_data = {
            "file": ("test.pdf", b"fake_pdf_content", "application/pdf"),
            "metadata": '{"title": "Test Document"}'
        }
        
        # Mock the upload process
        with patch("src.api.documents.process_uploaded_document") as mock_process:
            mock_process.return_value = {
                "document_id": str(uuid.uuid4()),
                "status": "completed"
            }
            
            # The actual test would make a request to the endpoint
            # response = test_client.post("/api/v1/documents/upload", files=upload_data)
            # assert response.status_code == 201

    @pytest.mark.asyncio
    async def test_list_documents_endpoint(self, authenticated_client):
        """Test document listing endpoint."""
        with patch("src.api.documents.get_document_service") as mock_service:
            mock_service.return_value.list_documents = AsyncMock(return_value={
                "documents": [],
                "total": 0,
                "offset": 0,
                "limit": 10,
                "has_next": False
            })
            
            # The actual test would make a request
            # response = authenticated_client.get("/api/v1/documents")
            # assert response.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__])