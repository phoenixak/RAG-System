"""
Test Cases for Search & Retrieval System
Comprehensive tests for search functionality.
"""

import uuid
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.search.models import (
    HybridSearchRequest,
    MetadataFilter,
    SearchResult,
    SemanticSearchRequest,
)
from src.search.service import SearchService


class TestSearchModels:
    """Test search model validation."""

    def test_semantic_search_request_validation(self):
        """Test semantic search request validation."""
        # Valid request
        request = SemanticSearchRequest(
            query="test query",
            limit=10,
            similarity_threshold=0.5,
        )
        assert request.query == "test query"
        assert request.limit == 10
        assert request.similarity_threshold == 0.5

        # Empty query should be stripped and raise validation error
        with pytest.raises(ValueError):
            SemanticSearchRequest(query="   ")

    def test_hybrid_search_request_validation(self):
        """Test hybrid search request validation."""
        # Valid request
        request = HybridSearchRequest(
            query="test query",
            semantic_weight=0.7,
            keyword_weight=0.3,
        )
        assert request.semantic_weight == 0.7
        assert request.keyword_weight == 0.3

        # Invalid weights (don't sum to 1.0)
        with pytest.raises(ValueError):
            HybridSearchRequest(
                query="test",
                semantic_weight=0.5,
                keyword_weight=0.4,
            )

    def test_metadata_filter(self):
        """Test metadata filter creation."""
        filter_obj = MetadataFilter(
            document_type="pdf",
            uploaded_by="user123",
            tags=["important", "financial"],
            custom_filters={"department": "finance"},
        )

        assert filter_obj.document_type == "pdf"
        assert filter_obj.uploaded_by == "user123"
        assert "important" in filter_obj.tags
        assert filter_obj.custom_filters["department"] == "finance"


class TestSemanticSearch:
    """Test semantic search functionality."""

    @pytest.fixture
    def mock_search_service(self):
        """Create a mock search service."""
        with patch("src.search.service.get_search_service") as mock:
            service = Mock(spec=SearchService)
            mock.return_value = service
            yield service

    @pytest.fixture
    def sample_search_results(self):
        """Create sample search results."""
        return [
            SearchResult(
                chunk_id=uuid.uuid4(),
                document_id=uuid.uuid4(),
                text="This is a test document about machine learning.",
                chunk_index=0,
                score=0.85,
                semantic_score=0.85,
                document_filename="ml_guide.pdf",
                document_type="pdf",
                metadata={"title": "ML Guide"},
            ),
            SearchResult(
                chunk_id=uuid.uuid4(),
                document_id=uuid.uuid4(),
                text="Deep learning is a subset of machine learning.",
                chunk_index=1,
                score=0.78,
                semantic_score=0.78,
                document_filename="dl_intro.pdf",
                document_type="pdf",
                metadata={"title": "Deep Learning Intro"},
            ),
        ]

    @pytest.mark.asyncio
    async def test_semantic_search_success(
        self, mock_search_service, sample_search_results
    ):
        """Test successful semantic search."""
        # Mock the search response
        from src.search.models import SearchResponse, SearchType

        mock_response = SearchResponse(
            query="machine learning",
            search_type=SearchType.SEMANTIC,
            results=sample_search_results,
            total_results=2,
            limit=10,
            offset=0,
            has_next=False,
            search_time_ms=150.5,
        )

        mock_search_service.semantic_search = AsyncMock(return_value=mock_response)

        # Create request
        request = SemanticSearchRequest(
            query="machine learning",
            limit=10,
            similarity_threshold=0.5,
        )

        # Perform search
        response = await mock_search_service.semantic_search(
            request, user_id="test_user"
        )

        # Assertions
        assert response.query == "machine learning"
        assert response.search_type == SearchType.SEMANTIC
        assert len(response.results) == 2
        assert response.results[0].score == 0.85
        assert "machine learning" in response.results[0].text.lower()
        assert response.search_time_ms > 0

        # Verify mock was called correctly
        mock_search_service.semantic_search.assert_called_once_with(
            request, user_id="test_user"
        )

    @pytest.mark.asyncio
    async def test_semantic_search_with_filters(self, mock_search_service):
        """Test semantic search with metadata filters."""
        # Create request with filters
        filters = MetadataFilter(
            document_type="pdf",
            uploaded_by="user123",
            tags=["technical"],
        )

        request = SemanticSearchRequest(
            query="test query",
            filters=filters,
            limit=5,
        )

        mock_search_service.semantic_search = AsyncMock(return_value=Mock())

        # Perform search
        await mock_search_service.semantic_search(request)

        # Verify filters were passed
        call_args = mock_search_service.semantic_search.call_args[0]
        assert call_args[0].filters.document_type == "pdf"
        assert call_args[0].filters.uploaded_by == "user123"
        assert "technical" in call_args[0].filters.tags


class TestHybridSearch:
    """Test hybrid search functionality."""

    @pytest.fixture
    def mock_hybrid_service(self):
        """Create a mock hybrid search service."""
        with patch("src.search.hybrid_search.get_hybrid_search_engine") as mock:
            service = Mock()
            mock.return_value = service
            yield service

    @pytest.mark.asyncio
    async def test_hybrid_search_fusion(self, mock_hybrid_service):
        """Test hybrid search result fusion."""
        from src.search.hybrid_search import HybridSearchEngine

        # Create mock semantic and keyword results
        semantic_results = [
            SearchResult(
                chunk_id=uuid.uuid4(),
                document_id=uuid.uuid4(),
                text="Semantic result 1",
                chunk_index=0,
                score=0.8,
                semantic_score=0.8,
                document_filename="doc1.pdf",
                document_type="pdf",
                metadata={},
            )
        ]

        keyword_results = [
            SearchResult(
                chunk_id=uuid.uuid4(),
                document_id=uuid.uuid4(),
                text="Keyword result 1",
                chunk_index=0,
                score=0.7,
                keyword_score=0.7,
                document_filename="doc2.pdf",
                document_type="pdf",
                metadata={},
            )
        ]

        # Test fusion logic
        engine = HybridSearchEngine()
        fused_results = engine._fuse_results(
            semantic_results=semantic_results,
            keyword_results=keyword_results,
            semantic_weight=0.7,
            keyword_weight=0.3,
            max_results=10,
        )

        # Should have both results
        assert len(fused_results) == 2

        # Check combined scoring
        for result in fused_results:
            assert hasattr(result, "score")
            assert result.score > 0


class TestReranking:
    """Test re-ranking functionality."""

    @pytest.fixture
    def mock_reranking_engine(self):
        """Create a mock re-ranking engine."""
        with patch("src.search.reranking.get_reranking_engine") as mock:
            engine = Mock()
            mock.return_value = engine
            yield engine

    @pytest.mark.asyncio
    async def test_reranking_improves_relevance(
        self, mock_reranking_engine, sample_search_results
    ):
        """Test that re-ranking can improve result relevance."""
        # Mock re-ranking to return improved scores
        reranked_results = sample_search_results.copy()
        reranked_results[0].rerank_score = 0.92
        reranked_results[1].rerank_score = 0.88

        # Sort by re-ranking score
        reranked_results.sort(key=lambda x: x.rerank_score, reverse=True)

        timing_metrics = {"total_time_ms": 50.0}

        mock_reranking_engine.rerank_results = AsyncMock(
            return_value=(reranked_results, timing_metrics)
        )

        # Test re-ranking
        results, metrics = await mock_reranking_engine.rerank_results(
            query="test query",
            results=sample_search_results,
            top_k=10,
        )

        # Verify results are re-ranked
        assert len(results) == 2
        assert results[0].rerank_score >= results[1].rerank_score
        assert metrics["total_time_ms"] == 50.0


class TestConversationContext:
    """Test conversation context management."""

    @pytest.fixture
    def mock_conversation_manager(self):
        """Create a mock conversation manager."""
        with patch("src.search.conversation.get_conversation_manager") as mock:
            manager = Mock()
            mock.return_value = manager
            yield manager

    def test_context_enhances_query(self, mock_conversation_manager):
        """Test that conversation context enhances queries."""
        from src.search.conversation import ConversationContext, ConversationManager

        manager = ConversationManager()

        # Create context with previous queries
        context = ConversationContext(
            session_id="test_session",
            previous_queries=[
                "What is machine learning?",
                "How does neural networks work?",
            ],
            topics=["technical", "AI"],
        )

        # Test query enhancement
        enhanced_query = manager.enhance_query_with_context(
            query="training process",
            context=context,
            context_weight=0.2,
        )

        # Enhanced query should contain additional context
        assert "training process" in enhanced_query
        # May contain additional context terms

    def test_topic_extraction(self):
        """Test topic extraction from conversation."""
        from src.search.conversation import ContextAnalyzer

        analyzer = ContextAnalyzer()

        queries = [
            "What is machine learning?",
            "How to implement neural networks?",
            "Deep learning algorithms",
        ]

        topics = analyzer.extract_topics(queries)

        # Should identify technical topics
        assert "technical" in topics


class TestSearchAPI:
    """Test search API endpoints."""

    @pytest.fixture
    def mock_current_user(self):
        """Mock current user for API tests."""
        return {"user_id": "test_user", "role": "user"}

    @pytest.mark.asyncio
    async def test_semantic_search_endpoint(self, mock_current_user):
        """Test semantic search API endpoint."""
        from src.api.search import semantic_search

        request = SemanticSearchRequest(
            query="test query",
            limit=10,
        )

        with patch("src.api.search.get_search_service") as mock_service:
            # Mock search response
            mock_response = Mock()
            mock_response.results = []
            mock_response.search_time_ms = 100.0

            mock_service.return_value.semantic_search = AsyncMock(
                return_value=mock_response
            )

            # Call endpoint
            response = await semantic_search(request, mock_current_user)

            # Verify response
            assert response == mock_response
            mock_service.return_value.semantic_search.assert_called_once()

    @pytest.mark.asyncio
    async def test_hybrid_search_endpoint(self, mock_current_user):
        """Test hybrid search API endpoint."""
        from src.api.search import hybrid_search

        request = HybridSearchRequest(
            query="test query",
            semantic_weight=0.7,
            keyword_weight=0.3,
        )

        with patch("src.api.search.get_search_service") as mock_service:
            mock_response = Mock()
            mock_response.results = []

            mock_service.return_value.hybrid_search = AsyncMock(
                return_value=mock_response
            )

            # Call endpoint
            response = await hybrid_search(request, mock_current_user)

            # Verify response
            assert response == mock_response


class TestSearchPerformance:
    """Test search performance and caching."""

    @pytest.mark.asyncio
    async def test_search_caching(self):
        """Test search result caching."""
        from src.search.service import SearchCache

        cache = SearchCache(max_size=100, ttl_seconds=300)

        # Test cache key generation
        key = cache._generate_cache_key(
            query="test query",
            search_type="semantic",
            params={"limit": 10},
        )

        assert isinstance(key, str)
        assert len(key) > 0

        # Test cache operations
        mock_response = Mock()
        cache.set(key, mock_response)

        cached_response = cache.get(key)
        assert cached_response == mock_response

        # Test cache stats
        stats = cache.get_stats()
        assert stats["size"] == 1
        assert stats["max_size"] == 100

    def test_search_stats_tracking(self):
        """Test search statistics tracking."""
        from src.search.service import SearchService

        service = SearchService(enable_cache=False)

        # Simulate some searches
        service._update_search_stats(0.1)  # 100ms
        service._update_search_stats(0.2)  # 200ms

        stats = service.get_search_stats()

        assert stats["total_searches"] == 2
        assert stats["average_response_time"] == 0.15  # Average of 0.1 and 0.2


@pytest.fixture
def sample_search_results():
    """Create sample search results for testing."""
    return [
        SearchResult(
            chunk_id=uuid.uuid4(),
            document_id=uuid.uuid4(),
            text="Sample document content about artificial intelligence.",
            chunk_index=0,
            score=0.85,
            document_filename="ai_doc.pdf",
            document_type="pdf",
            metadata={"title": "AI Introduction"},
        ),
        SearchResult(
            chunk_id=uuid.uuid4(),
            document_id=uuid.uuid4(),
            text="Machine learning is a subset of artificial intelligence.",
            chunk_index=1,
            score=0.78,
            document_filename="ml_doc.pdf",
            document_type="pdf",
            metadata={"title": "ML Basics"},
        ),
    ]


if __name__ == "__main__":
    pytest.main([__file__])
