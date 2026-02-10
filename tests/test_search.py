"""
Tests for search service and search cache.
Tests SearchCache, SearchService orchestration, and search models.
"""

import hashlib
import time
import uuid
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from src.search.models import (
    ContextualSearchRequest,
    HybridSearchRequest,
    SearchResponse,
    SearchResult,
    SearchType,
    SemanticSearchRequest,
)


class TestSearchCache:
    """Tests for SearchCache."""

    def test_cache_init(self):
        """Test SearchCache initialization."""
        from src.search.service import SearchCache

        cache = SearchCache(max_size=100, ttl_seconds=60)
        assert cache.max_size == 100
        assert cache.ttl_seconds == 60

    def test_cache_set_and_get(self):
        """Test setting and getting cache entries."""
        from src.search.service import SearchCache

        cache = SearchCache()
        mock_response = MagicMock(spec=SearchResponse)
        cache.set("test-key", mock_response)
        result = cache.get("test-key")
        assert result is mock_response

    def test_cache_miss(self):
        """Test cache miss returns None."""
        from src.search.service import SearchCache

        cache = SearchCache()
        result = cache.get("nonexistent-key")
        assert result is None

    def test_cache_expiration(self):
        """Test cache entries expire after TTL."""
        from src.search.service import SearchCache

        cache = SearchCache(ttl_seconds=0)  # Expire immediately
        mock_response = MagicMock(spec=SearchResponse)
        cache.set("expire-key", mock_response)
        time.sleep(0.01)
        result = cache.get("expire-key")
        assert result is None

    def test_cache_clear(self):
        """Test clearing the cache."""
        from src.search.service import SearchCache

        cache = SearchCache()
        cache.set("key1", MagicMock())
        cache.set("key2", MagicMock())
        cache.clear()
        assert cache.get("key1") is None
        assert cache.get("key2") is None

    def test_cache_stats(self):
        """Test cache statistics."""
        from src.search.service import SearchCache

        cache = SearchCache(max_size=500, ttl_seconds=900)
        stats = cache.get_stats()
        assert stats["size"] == 0
        assert stats["max_size"] == 500
        assert stats["ttl_seconds"] == 900

    def test_cache_eviction(self):
        """Test cache evicts old entries when full."""
        from src.search.service import SearchCache

        cache = SearchCache(max_size=3)
        for i in range(5):
            cache.set(f"key-{i}", MagicMock())
        stats = cache.get_stats()
        assert stats["size"] <= 3

    def test_generate_cache_key_deterministic(self):
        """Test cache key generation is deterministic."""
        from src.search.service import SearchCache

        cache = SearchCache()
        key1 = cache._generate_cache_key(
            "test query", SearchType.SEMANTIC, {"limit": 10}
        )
        key2 = cache._generate_cache_key(
            "test query", SearchType.SEMANTIC, {"limit": 10}
        )
        assert key1 == key2

    def test_generate_cache_key_different_queries(self):
        """Test different queries produce different cache keys."""
        from src.search.service import SearchCache

        cache = SearchCache()
        key1 = cache._generate_cache_key("query one", SearchType.SEMANTIC, {})
        key2 = cache._generate_cache_key("query two", SearchType.SEMANTIC, {})
        assert key1 != key2


class TestSearchService:
    """Tests for SearchService."""

    @pytest.fixture
    def mock_search_service(self):
        """Create SearchService with mocked dependencies."""
        with (
            patch("src.search.service.get_semantic_search_engine") as mock_sem,
            patch("src.search.service.get_hybrid_search_engine") as mock_hyb,
            patch("src.search.service.get_reranking_engine") as mock_rerank,
            patch("src.search.service.get_fallback_reranking_engine") as mock_fallback,
            patch("src.search.service.get_conversation_manager") as mock_conv,
        ):
            mock_sem.return_value = MagicMock()
            mock_hyb.return_value = MagicMock()
            mock_rerank.return_value = MagicMock()
            mock_fallback.return_value = MagicMock()
            mock_conv.return_value = MagicMock()

            from src.search.service import SearchService

            service = SearchService(enable_cache=True)

            # Setup mock returns
            mock_results = [
                SearchResult(
                    chunk_id=uuid.uuid4(),
                    document_id=uuid.uuid4(),
                    text="test result",
                    chunk_index=0,
                    score=0.9,
                    document_filename="test.pdf",
                    document_type="pdf",
                ),
            ]
            service.semantic_engine.search = AsyncMock(
                return_value=(
                    mock_results,
                    {"total_time_ms": 50.0, "embedding_time_ms": 10.0},
                )
            )
            service.hybrid_engine.search = AsyncMock(
                return_value=(mock_results, {"total_time_ms": 80.0})
            )
            service.reranking_engine.rerank_results = AsyncMock(
                return_value=(mock_results, {"total_time_ms": 20.0})
            )

            return service

    @pytest.mark.asyncio
    async def test_semantic_search(self, mock_search_service):
        """Test semantic search execution."""
        request = SemanticSearchRequest(query="machine learning")
        response = await mock_search_service.semantic_search(
            request, user_id="test-user"
        )
        assert isinstance(response, SearchResponse)
        assert response.search_type == SearchType.SEMANTIC
        assert response.total_results > 0

    @pytest.mark.asyncio
    async def test_hybrid_search(self, mock_search_service):
        """Test hybrid search execution."""
        request = HybridSearchRequest(
            query="neural networks",
            semantic_weight=0.7,
            keyword_weight=0.3,
            enable_rerank=False,
        )
        response = await mock_search_service.hybrid_search(request, user_id="test-user")
        assert isinstance(response, SearchResponse)
        assert response.search_type == SearchType.HYBRID

    def test_get_search_stats(self, mock_search_service):
        """Test getting search statistics."""
        stats = mock_search_service.get_search_stats()
        assert isinstance(stats, dict)
        assert "total_searches" in stats
        assert "cache_hits" in stats
        assert "error_count" in stats

    def test_clear_cache(self, mock_search_service):
        """Test clearing the search cache."""
        mock_search_service.clear_cache()
        # Should not raise

    @pytest.mark.asyncio
    async def test_semantic_search_caching(self, mock_search_service):
        """Test that repeated queries hit the cache."""
        request = SemanticSearchRequest(query="cached query test")

        # First call - cache miss
        await mock_search_service.semantic_search(request)
        # Second call - should hit cache
        await mock_search_service.semantic_search(request)

        assert mock_search_service.search_stats["cache_hits"] >= 1
