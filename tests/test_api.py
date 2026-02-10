"""
Tests for API endpoint definitions and routing.
Tests router configuration, request/response models, and endpoint behaviors.
"""

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.documents.models import (
    BulkDeleteRequest,
    DocumentListResponse,
    DocumentResponse,
    DocumentStatsResponse,
    DocumentType,
    ProcessingStatus,
)
from src.search.models import (
    HybridSearchRequest,
    MetadataFilter,
    SearchRequest,
    SearchType,
    SemanticSearchRequest,
    SortOrder,
)


class TestSearchRequestModels:
    """Tests for search request model validation."""

    def test_search_request_valid(self):
        """Test valid search request."""
        req = SearchRequest(query="machine learning")
        assert req.query == "machine learning"
        assert req.limit == 10
        assert req.offset == 0

    def test_search_request_empty_query_rejected(self):
        """Test empty query is rejected."""
        with pytest.raises(ValueError):
            SearchRequest(query="")

    def test_search_request_query_too_long(self):
        """Test overly long query is rejected."""
        with pytest.raises(ValueError):
            SearchRequest(query="x" * 1001)

    def test_search_request_limit_bounds(self):
        """Test limit must be within bounds."""
        with pytest.raises(ValueError):
            SearchRequest(query="test", limit=0)
        with pytest.raises(ValueError):
            SearchRequest(query="test", limit=101)

    def test_semantic_search_request(self):
        """Test SemanticSearchRequest with custom threshold."""
        req = SemanticSearchRequest(
            query="deep learning",
            similarity_threshold=0.5,
            include_embeddings=True,
        )
        assert req.similarity_threshold == 0.5
        assert req.include_embeddings is True

    def test_hybrid_search_request_weights(self):
        """Test HybridSearchRequest weight validation."""
        req = HybridSearchRequest(
            query="neural networks",
            semantic_weight=0.7,
            keyword_weight=0.3,
        )
        assert req.semantic_weight == 0.7
        assert req.keyword_weight == 0.3

    def test_metadata_filter(self):
        """Test MetadataFilter model."""
        filt = MetadataFilter(
            document_type="pdf",
            uploaded_by="user123",
        )
        assert filt.document_type == "pdf"

    def test_sort_order_values(self):
        """Test SortOrder enum."""
        assert SortOrder.RELEVANCE == "relevance"
        assert SortOrder.DATE_ASC == "date_asc"
        assert SortOrder.DATE_DESC == "date_desc"

    def test_search_type_values(self):
        """Test SearchType enum."""
        assert SearchType.SEMANTIC == "semantic"
        assert SearchType.KEYWORD == "keyword"
        assert SearchType.HYBRID == "hybrid"


class TestDocumentResponseModels:
    """Tests for document response models."""

    def test_document_response(self):
        """Test DocumentResponse model."""
        resp = DocumentResponse(
            id=uuid.uuid4(),
            filename="test.pdf",
            document_type=DocumentType.PDF,
            size=1024,
            processing_status=ProcessingStatus.COMPLETED,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            uploaded_by="test_user",
        )
        assert resp.filename == "test.pdf"
        assert resp.processing_status == ProcessingStatus.COMPLETED

    def test_document_list_response(self):
        """Test DocumentListResponse model."""
        doc = DocumentResponse(
            id=uuid.uuid4(),
            filename="test.pdf",
            document_type=DocumentType.PDF,
            size=1024,
            processing_status=ProcessingStatus.QUEUED,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            uploaded_by="user1",
        )
        resp = DocumentListResponse(
            documents=[doc],
            total=1,
            page=1,
            page_size=50,
            total_pages=1,
        )
        assert resp.total == 1
        assert len(resp.documents) == 1

    def test_bulk_delete_request_validation(self):
        """Test BulkDeleteRequest requires confirmation."""
        with pytest.raises(ValueError):
            BulkDeleteRequest(
                document_ids=[uuid.uuid4()],
                confirm_deletion=False,
            )

    def test_bulk_delete_request_valid(self):
        """Test valid BulkDeleteRequest."""
        req = BulkDeleteRequest(
            document_ids=[uuid.uuid4()],
            confirm_deletion=True,
        )
        assert len(req.document_ids) == 1

    def test_document_stats_response(self):
        """Test DocumentStatsResponse model."""
        stats = DocumentStatsResponse(
            total_documents=100,
            total_chunks=500,
            processing_queue_size=3,
            storage_size_bytes=1024 * 1024,
            status_counts={ProcessingStatus.COMPLETED: 90, ProcessingStatus.QUEUED: 10},
            type_counts={DocumentType.PDF: 60, DocumentType.TXT: 40},
        )
        assert stats.total_documents == 100
        assert stats.total_chunks == 500


class TestHealthEndpoint:
    """Tests for health check endpoint logic."""

    def test_health_router_exists(self):
        """Test health router is properly defined."""
        from src.api.health import router

        assert router is not None
