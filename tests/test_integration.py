"""
Integration tests for the RAG system.
Tests cross-module interactions and end-to-end flows.
"""

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.documents.models import (
    DocumentChunk,
    DocumentMetadata,
    DocumentType,
    ProcessingStatus,
)
from src.search.models import SearchResult, SearchResponse, SearchType


class TestDocumentToSearchFlow:
    """Tests for document processing to search pipeline."""

    def test_document_metadata_to_search_result_mapping(self):
        """Test that document metadata fields map correctly to search results."""
        doc_id = uuid.uuid4()
        meta = DocumentMetadata(
            id=doc_id,
            filename="test_report.pdf",
            document_type=DocumentType.PDF,
            size=2048,
            content_type="application/pdf",
            uploaded_by="integration_test_user",
            processing_status=ProcessingStatus.COMPLETED,
        )

        # Create a search result referencing this document
        result = SearchResult(
            chunk_id=uuid.uuid4(),
            document_id=doc_id,
            text="Content from the test report.",
            chunk_index=0,
            score=0.92,
            document_filename=meta.filename,
            document_type=meta.document_type,
        )

        assert result.document_id == meta.id
        assert result.document_filename == meta.filename

    def test_chunk_creation_maintains_document_reference(self):
        """Test that chunks maintain proper document ID references."""
        doc_id = uuid.uuid4()

        chunks = [
            DocumentChunk(
                document_id=doc_id,
                chunk_index=i,
                text=f"Chunk {i} content",
                token_count=5,
                start_char=i * 100,
                end_char=(i + 1) * 100,
            )
            for i in range(3)
        ]

        for chunk in chunks:
            assert chunk.document_id == doc_id
        assert [c.chunk_index for c in chunks] == [0, 1, 2]


class TestSearchResponseConstruction:
    """Tests for search response construction."""

    def test_search_response_with_results(self):
        """Test building a complete SearchResponse."""
        results = [
            SearchResult(
                chunk_id=uuid.uuid4(),
                document_id=uuid.uuid4(),
                text=f"Result {i}",
                chunk_index=i,
                score=0.9 - i * 0.1,
                document_filename=f"doc{i}.pdf",
                document_type="pdf",
            )
            for i in range(3)
        ]

        response = SearchResponse(
            query="test query",
            search_type=SearchType.HYBRID,
            results=results,
            total_results=3,
            limit=10,
            offset=0,
            has_next=False,
            search_time_ms=45.2,
        )

        assert response.total_results == 3
        assert response.search_type == SearchType.HYBRID
        assert response.results[0].score > response.results[-1].score

    def test_search_response_empty_results(self):
        """Test SearchResponse with no results."""
        response = SearchResponse(
            query="obscure query with no matches",
            search_type=SearchType.SEMANTIC,
            results=[],
            total_results=0,
            limit=10,
            offset=0,
            has_next=False,
            search_time_ms=12.5,
        )

        assert response.total_results == 0
        assert len(response.results) == 0


class TestModelInteroperability:
    """Tests for model interoperability between modules."""

    def test_processing_status_transitions(self):
        """Test valid processing status transitions."""
        meta = DocumentMetadata(
            filename="test.pdf",
            document_type=DocumentType.PDF,
            size=1024,
            content_type="application/pdf",
            uploaded_by="test",
        )
        assert meta.processing_status == ProcessingStatus.QUEUED

        # Simulate status transitions
        meta.processing_status = ProcessingStatus.PROCESSING
        assert meta.processing_status == ProcessingStatus.PROCESSING

        meta.processing_status = ProcessingStatus.COMPLETED
        assert meta.processing_status == ProcessingStatus.COMPLETED

    def test_search_result_score_bounds(self):
        """Test search result scores are within valid bounds."""
        result = SearchResult(
            chunk_id=uuid.uuid4(),
            document_id=uuid.uuid4(),
            text="test",
            chunk_index=0,
            score=0.5,
            document_filename="test.pdf",
            document_type="pdf",
        )
        assert 0.0 <= result.score <= 1.0

    def test_search_result_score_out_of_bounds_rejected(self):
        """Test that invalid scores are rejected."""
        with pytest.raises(ValueError):
            SearchResult(
                chunk_id=uuid.uuid4(),
                document_id=uuid.uuid4(),
                text="test",
                chunk_index=0,
                score=1.5,  # Out of bounds
                document_filename="test.pdf",
                document_type="pdf",
            )
