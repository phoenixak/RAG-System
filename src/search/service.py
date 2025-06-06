"""
Search Service
Main search orchestration service that coordinates all search engines.
"""

import hashlib
import time
from typing import Dict, List, Optional, Tuple

from src.core.config import get_settings
from src.core.logging import LoggerMixin
from src.search.conversation import get_conversation_manager
from src.search.hybrid_search import get_hybrid_search_engine
from src.search.models import (
    ContextualSearchRequest,
    HybridSearchRequest,
    QuerySuggestionsRequest,
    QuerySuggestionsResponse,
    SearchResponse,
    SearchResult,
    SearchType,
    SemanticSearchRequest,
    SimilarDocumentsRequest,
)
from src.search.reranking import get_fallback_reranking_engine, get_reranking_engine
from src.search.semantic_search import get_semantic_search_engine
from src.vector_store.chroma_client import get_chroma_client

settings = get_settings()


class SearchCache:
    """Simple in-memory search result cache."""

    def __init__(self, max_size: int = 1000, ttl_seconds: int = 1800):
        self.cache: Dict[str, Tuple[SearchResponse, float]] = {}
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds

    def _generate_cache_key(
        self,
        query: str,
        search_type: SearchType,
        params: Dict,
    ) -> str:
        """Generate cache key for search request."""
        key_data = {
            "query": query.lower().strip(),
            "search_type": search_type,
            "params": sorted(params.items()),
        }
        key_string = str(key_data)
        return hashlib.md5(key_string.encode()).hexdigest()

    def get(self, key: str) -> Optional[SearchResponse]:
        """Get cached search response."""
        if key in self.cache:
            response, timestamp = self.cache[key]

            # Check if cache entry is still valid
            if time.time() - timestamp < self.ttl_seconds:
                return response
            else:
                # Remove expired entry
                del self.cache[key]

        return None

    def set(self, key: str, response: SearchResponse) -> None:
        """Cache search response."""
        # Clear old entries if cache is full
        if len(self.cache) >= self.max_size:
            self._evict_old_entries()

        self.cache[key] = (response, time.time())

    def _evict_old_entries(self) -> None:
        """Remove oldest 20% of cache entries."""
        if not self.cache:
            return

        # Sort by timestamp and remove oldest entries
        sorted_items = sorted(
            self.cache.items(),
            key=lambda x: x[1][1],  # Sort by timestamp
        )

        evict_count = len(self.cache) // 5  # Remove 20%
        for key, _ in sorted_items[:evict_count]:
            del self.cache[key]

    def clear(self) -> None:
        """Clear all cache entries."""
        self.cache.clear()

    def get_stats(self) -> Dict[str, any]:
        """Get cache statistics."""
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "ttl_seconds": self.ttl_seconds,
        }


class SearchService(LoggerMixin):
    """Main search service orchestrating all search engines."""

    def __init__(self, enable_cache: bool = True):
        self.semantic_engine = get_semantic_search_engine()
        self.hybrid_engine = get_hybrid_search_engine()
        self.reranking_engine = get_reranking_engine()
        self.fallback_reranking = get_fallback_reranking_engine()
        self.conversation_manager = get_conversation_manager()

        # Initialize cache
        self.cache = SearchCache() if enable_cache else None

        # Performance tracking
        self.search_stats = {
            "total_searches": 0,
            "cache_hits": 0,
            "average_response_time": 0.0,
            "error_count": 0,
        }

    async def semantic_search(
        self,
        request: SemanticSearchRequest,
        user_id: Optional[str] = None,
    ) -> SearchResponse:
        """
        Perform semantic search.

        Args:
            request: Semantic search request
            user_id: Optional user ID for tracking

        Returns:
            Search response
        """
        start_time = time.time()

        try:
            self.logger.info(
                "Starting semantic search",
                query=request.query,
                user_id=user_id,
            )

            # Check cache
            cache_key = None
            if self.cache:
                cache_key = self.cache._generate_cache_key(
                    request.query,
                    SearchType.SEMANTIC,
                    request.dict(exclude={"query"}),
                )
                cached_response = self.cache.get(cache_key)
                if cached_response:
                    self.search_stats["cache_hits"] += 1
                    self.logger.info(
                        "Returning cached semantic search result",
                        query=request.query,
                        user_id=user_id,
                    )
                    return cached_response

            # Perform search
            results, timing_metrics = await self.semantic_engine.search(
                request, user_id
            )

            # Build response
            response = SearchResponse(
                query=request.query,
                search_type=SearchType.SEMANTIC,
                results=results,
                total_results=len(results),
                limit=request.limit,
                offset=request.offset,
                has_next=len(results) == request.limit,
                search_time_ms=timing_metrics.get("total_time_ms", 0.0),
                embedding_time_ms=timing_metrics.get("embedding_time_ms"),
                filters_applied=request.filters,
                parameters=request.dict(exclude={"query", "filters"}),
            )

            # Cache response
            if self.cache and cache_key:
                self.cache.set(cache_key, response)

            # Update stats
            self._update_search_stats(time.time() - start_time)

            return response

        except Exception as e:
            self.search_stats["error_count"] += 1
            self.logger.error(
                "Semantic search failed",
                query=request.query,
                error=str(e),
                user_id=user_id,
                exc_info=True,
            )
            raise

    async def hybrid_search(
        self,
        request: HybridSearchRequest,
        user_id: Optional[str] = None,
    ) -> SearchResponse:
        """
        Perform hybrid search with optional re-ranking.

        Args:
            request: Hybrid search request
            user_id: Optional user ID for tracking

        Returns:
            Search response
        """
        start_time = time.time()

        try:
            self.logger.info(
                "Starting hybrid search",
                query=request.query,
                enable_rerank=request.enable_rerank,
                user_id=user_id,
            )

            # Check cache
            cache_key = None
            if self.cache:
                cache_key = self.cache._generate_cache_key(
                    request.query,
                    SearchType.HYBRID,
                    request.dict(exclude={"query"}),
                )
                cached_response = self.cache.get(cache_key)
                if cached_response:
                    self.search_stats["cache_hits"] += 1
                    self.logger.info(
                        "Returning cached hybrid search result",
                        query=request.query,
                        user_id=user_id,
                    )
                    return cached_response

            # Perform hybrid search
            results, timing_metrics = await self.hybrid_engine.search(request, user_id)

            # Apply re-ranking if enabled
            rerank_time_ms = None
            if request.enable_rerank and results:
                try:
                    rerank_start = time.time()
                    (
                        reranked_results,
                        rerank_timing,
                    ) = await self.reranking_engine.rerank_results(
                        query=request.query,
                        results=results,
                        top_k=request.limit,
                    )
                    results = reranked_results
                    rerank_time_ms = rerank_timing.get("total_time_ms", 0.0)

                    self.logger.info(
                        "Re-ranking completed",
                        query=request.query,
                        rerank_time_ms=rerank_time_ms,
                    )

                except Exception as e:
                    self.logger.warning(
                        "Re-ranking failed, using fallback",
                        query=request.query,
                        error=str(e),
                    )

                    # Try fallback re-ranking
                    try:
                        (
                            reranked_results,
                            rerank_timing,
                        ) = await self.fallback_reranking.rerank_results(
                            query=request.query,
                            results=results,
                            top_k=request.limit,
                        )
                        results = reranked_results
                        rerank_time_ms = rerank_timing.get("total_time_ms", 0.0)
                    except Exception as fallback_error:
                        self.logger.error(
                            "Fallback re-ranking also failed",
                            query=request.query,
                            error=str(fallback_error),
                        )
                        # Continue with original results

            # Build response
            response = SearchResponse(
                query=request.query,
                search_type=SearchType.HYBRID,
                results=results,
                total_results=len(results),
                limit=request.limit,
                offset=request.offset,
                has_next=len(results) == request.limit,
                search_time_ms=timing_metrics.get("total_time_ms", 0.0),
                embedding_time_ms=timing_metrics.get("semantic_embedding_time_ms"),
                rerank_time_ms=rerank_time_ms,
                filters_applied=request.filters,
                parameters=request.dict(exclude={"query", "filters"}),
            )

            # Cache response
            if self.cache and cache_key:
                self.cache.set(cache_key, response)

            # Update stats
            self._update_search_stats(time.time() - start_time)

            return response

        except Exception as e:
            self.search_stats["error_count"] += 1
            self.logger.error(
                "Hybrid search failed",
                query=request.query,
                error=str(e),
                user_id=user_id,
                exc_info=True,
            )
            raise

    async def contextual_search(
        self,
        request: ContextualSearchRequest,
        user_id: Optional[str] = None,
    ) -> SearchResponse:
        """
        Perform context-aware search using conversation history.

        Args:
            request: Contextual search request
            user_id: Optional user ID for tracking

        Returns:
            Search response
        """
        start_time = time.time()

        try:
            self.logger.info(
                "Starting contextual search",
                query=request.query,
                session_id=request.session_id,
                use_context=request.use_context,
                user_id=user_id,
            )

            # Get or create conversation context
            if request.session_id and request.use_context:
                context = self.conversation_manager.get_or_create_session(
                    request.session_id
                )

                # Enhance query with context
                enhanced_query = self.conversation_manager.enhance_query_with_context(
                    request.query,
                    context,
                    request.context_weight,
                )

                # Add query to session history
                context = self.conversation_manager.add_query_to_session(
                    request.session_id,
                    request.query,
                )

                self.logger.debug(
                    "Enhanced query with context",
                    original_query=request.query,
                    enhanced_query=enhanced_query,
                    session_id=request.session_id,
                )
            else:
                enhanced_query = request.query
                context = None

            # Convert to hybrid search request
            hybrid_request = HybridSearchRequest(
                query=enhanced_query,
                limit=request.limit,
                offset=request.offset,
                filters=request.filters,
                min_score=request.min_score,
                sort_by=request.sort_by,
                semantic_weight=0.7,  # Default weights for contextual search
                keyword_weight=0.3,
                enable_rerank=True,
            )

            # Perform hybrid search
            response = await self.hybrid_search(hybrid_request, user_id)

            # Update response metadata
            response.query = request.query  # Keep original query
            response.parameters["enhanced_query"] = enhanced_query
            response.parameters["context_used"] = request.use_context
            response.parameters["session_id"] = request.session_id

            # Update conversation with results
            if request.session_id and context:
                self.conversation_manager.add_query_to_session(
                    request.session_id,
                    request.query,
                    len(response.results),
                )

            return response

        except Exception as e:
            self.search_stats["error_count"] += 1
            self.logger.error(
                "Contextual search failed",
                query=request.query,
                session_id=request.session_id,
                error=str(e),
                user_id=user_id,
                exc_info=True,
            )
            raise

    async def find_similar_documents(
        self,
        request: SimilarDocumentsRequest,
        user_id: Optional[str] = None,
    ) -> List[SearchResult]:
        """
        Find documents similar to a reference document.

        Args:
            request: Similar documents request
            user_id: Optional user ID for tracking

        Returns:
            List of similar search results
        """
        try:
            self.logger.info(
                "Finding similar documents",
                reference_document_id=request.document_id,
                limit=request.limit,
                user_id=user_id,
            )

            results = await self.semantic_engine.find_similar_documents(
                document_id=str(request.document_id),
                limit=request.limit,
                similarity_threshold=request.similarity_threshold,
                filters=request.filters,
                exclude_same_document=request.exclude_same_document,
            )

            return results

        except Exception as e:
            self.logger.error(
                "Similar documents search failed",
                reference_document_id=request.document_id,
                error=str(e),
                user_id=user_id,
                exc_info=True,
            )
            raise

    async def get_query_suggestions(
        self,
        request: QuerySuggestionsRequest,
        user_id: Optional[str] = None,
    ) -> QuerySuggestionsResponse:
        """
        Get query suggestions and autocomplete.

        Args:
            request: Query suggestions request
            user_id: Optional user ID for tracking

        Returns:
            Query suggestions response
        """
        start_time = time.time()

        try:
            self.logger.info(
                "Generating query suggestions",
                partial_query=request.partial_query,
                user_id=user_id,
            )

            # For now, return a simple implementation
            # In production, this would use a dedicated suggestion engine
            suggestions = []

            # Simple completion suggestions
            if "completion" in request.suggestion_types:
                completion_suggestions = [
                    f"{request.partial_query} documents",
                    f"{request.partial_query} analysis",
                    f"{request.partial_query} report",
                    f"{request.partial_query} summary",
                    f"{request.partial_query} overview",
                ]

                for i, suggestion in enumerate(completion_suggestions[: request.limit]):
                    suggestions.append(
                        {
                            "suggestion": suggestion,
                            "score": 1.0 - (i * 0.1),
                            "type": "completion",
                            "metadata": {},
                        }
                    )

            response = QuerySuggestionsResponse(
                query=request.partial_query,
                suggestions=suggestions,
                generation_time_ms=(time.time() - start_time) * 1000,
            )

            return response

        except Exception as e:
            self.logger.error(
                "Query suggestions failed",
                partial_query=request.partial_query,
                error=str(e),
                user_id=user_id,
                exc_info=True,
            )
            raise

    def _update_search_stats(self, response_time: float) -> None:
        """Update search performance statistics."""
        self.search_stats["total_searches"] += 1

        # Update running average
        current_avg = self.search_stats["average_response_time"]
        total_searches = self.search_stats["total_searches"]

        new_avg = (
            (current_avg * (total_searches - 1)) + response_time
        ) / total_searches
        self.search_stats["average_response_time"] = new_avg

    def get_search_stats(self) -> Dict[str, any]:
        """
        Get search service statistics.

        Returns:
            Dictionary of search statistics
        """
        stats = self.search_stats.copy()

        # Add cache stats if enabled
        if self.cache:
            stats["cache"] = self.cache.get_stats()
            stats["cache_hit_rate"] = self.search_stats["cache_hits"] / max(
                self.search_stats["total_searches"], 1
            )

        # Add conversation stats
        try:
            stats["conversation"] = self.conversation_manager.get_conversation_stats()
        except Exception:
            pass

        return stats

    def clear_cache(self) -> None:
        """Clear search cache."""
        if self.cache:
            self.cache.clear()
            self.logger.info("Search cache cleared")

    async def get_document_for_api(self, document_id: str) -> Optional[Dict[str, any]]:
        """
        Get document metadata for API endpoints.

        This method uses get_document_metadata() which filters out internal
        processing fields like file_path that shouldn't be exposed to API consumers.

        Args:
            document_id: Document ID

        Returns:
            Clean document metadata for API response
        """
        try:
            chroma_client = get_chroma_client()
            return await chroma_client.get_document_metadata(document_id)
        except Exception as e:
            self.logger.error(
                "Failed to get document for API",
                error=str(e),
                document_id=document_id,
                exc_info=True,
            )
            raise

    async def _process_document_internal(
        self, document_id: str
    ) -> Optional[Dict[str, any]]:
        """
        Internal method for document processing that needs access to file_path.

        This method uses _get_document_metadata_internal() which includes all
        metadata fields including internal processing information like file_path.

        Args:
            document_id: Document ID

        Returns:
            Complete document metadata including internal fields
        """
        try:
            chroma_client = get_chroma_client()
            metadata = await chroma_client._get_document_metadata_internal(document_id)

            if metadata and "file_path" in metadata:
                # Example: Use file_path for internal processing
                file_path = metadata["file_path"]
                self.logger.debug(
                    "Processing document with file path",
                    document_id=document_id,
                    file_path=file_path,
                )
                # Perform internal processing that requires file_path...

            return metadata
        except Exception as e:
            self.logger.error(
                "Failed to process document internally",
                error=str(e),
                document_id=document_id,
                exc_info=True,
            )
            raise


# Global search service instance
_search_service: Optional[SearchService] = None


def get_search_service() -> SearchService:
    """Get the global search service instance."""
    global _search_service
    if _search_service is None:
        _search_service = SearchService()
    return _search_service


async def close_search_service():
    """Close the global search service."""
    global _search_service
    if _search_service:
        if _search_service.cache:
            _search_service.cache.clear()
        _search_service = None
