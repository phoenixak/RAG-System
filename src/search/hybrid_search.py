"""
Hybrid Search Engine
Combines semantic search with keyword search using BM25 and implements result fusion.
"""

import asyncio
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple

from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from src.core.config import get_settings
from src.core.logging import LoggerMixin
from src.search.models import HybridSearchRequest, SearchResult
from src.search.semantic_search import get_semantic_search_engine
from src.vector_store.chroma_client import get_chroma_client

settings = get_settings()


class QueryPreprocessor:
    """Query preprocessing for keyword search."""

    def __init__(self):
        self.stop_words = set(ENGLISH_STOP_WORDS)
        # Add more stop words specific to document search
        self.stop_words.update(
            {
                "document",
                "file",
                "text",
                "page",
                "section",
                "chapter",
                "paragraph",
                "line",
                "content",
                "information",
                "data",
            }
        )

        # Common synonyms for query expansion
        self.synonyms = {
            "find": ["search", "locate", "discover"],
            "show": ["display", "present", "demonstrate"],
            "how": ["method", "way", "approach"],
            "what": ["which", "definition"],
            "why": ["reason", "cause", "explanation"],
        }

    def preprocess_query(self, query: str) -> List[str]:
        """
        Preprocess query for keyword search.

        Args:
            query: Raw search query

        Returns:
            List of processed tokens
        """
        # Convert to lowercase
        query = query.lower()

        # Remove punctuation except apostrophes
        query = re.sub(r"[^\w\s']", " ", query)

        # Split into tokens
        tokens = query.split()

        # Remove stop words and short tokens
        tokens = [
            token for token in tokens if len(token) > 2 and token not in self.stop_words
        ]

        # Apply basic stemming (simple suffix removal)
        tokens = [self._simple_stem(token) for token in tokens]

        return tokens

    def expand_query(self, tokens: List[str]) -> List[str]:
        """
        Expand query with synonyms.

        Args:
            tokens: Preprocessed query tokens

        Returns:
            Expanded token list
        """
        expanded = tokens.copy()

        for token in tokens:
            if token in self.synonyms:
                expanded.extend(self.synonyms[token])

        return expanded

    def _simple_stem(self, word: str) -> str:
        """Simple stemming by removing common suffixes."""
        suffixes = ["ing", "ed", "er", "est", "ly", "tion", "ness", "ment"]

        for suffix in suffixes:
            if word.endswith(suffix) and len(word) > len(suffix) + 2:
                return word[: -len(suffix)]

        return word


class KeywordSearchEngine(LoggerMixin):
    """BM25-based keyword search engine."""

    def __init__(self):
        self.preprocessor = QueryPreprocessor()
        self.vector_store = get_chroma_client()
        self.executor = ThreadPoolExecutor(max_workers=2)

        # Cache for BM25 indices
        self._bm25_cache: Dict[str, BM25Okapi] = {}
        self._document_cache: Dict[str, List[Dict]] = {}

    async def search(
        self,
        query: str,
        limit: int = 10,
        offset: int = 0,
        metadata_filter: Optional[Dict] = None,
    ) -> Tuple[List[SearchResult], Dict[str, float]]:
        """
        Perform keyword search using BM25.

        Args:
            query: Search query
            limit: Maximum number of results
            offset: Number of results to skip
            metadata_filter: Optional metadata filter

        Returns:
            Tuple of (search_results, timing_metrics)
        """
        start_time = time.time()
        timing_metrics = {}

        try:
            self.logger.info(
                "Starting keyword search",
                query=query,
                limit=limit,
                offset=offset,
            )

            # Get document corpus
            corpus_start = time.time()
            documents, doc_metadata = await self._get_document_corpus(metadata_filter)
            timing_metrics["corpus_load_time_ms"] = (time.time() - corpus_start) * 1000

            if not documents:
                return [], timing_metrics

            # Preprocess query
            query_tokens = self.preprocessor.preprocess_query(query)
            expanded_tokens = self.preprocessor.expand_query(query_tokens)

            # Build BM25 index if not cached
            index_start = time.time()
            bm25 = await self._get_or_build_bm25_index(documents, metadata_filter)
            timing_metrics["index_build_time_ms"] = (time.time() - index_start) * 1000

            # Perform BM25 search
            search_start = time.time()
            scores = await asyncio.get_event_loop().run_in_executor(
                self.executor, bm25.get_scores, expanded_tokens
            )
            timing_metrics["bm25_search_time_ms"] = (time.time() - search_start) * 1000

            # Rank and filter results
            ranking_start = time.time()
            results = self._rank_and_filter_results(
                documents, doc_metadata, scores, query, limit, offset
            )
            timing_metrics["ranking_time_ms"] = (time.time() - ranking_start) * 1000

            timing_metrics["total_time_ms"] = (time.time() - start_time) * 1000

            self.logger.info(
                "Keyword search completed",
                query=query,
                results_count=len(results),
                total_time_ms=timing_metrics["total_time_ms"],
            )

            return results, timing_metrics

        except Exception as e:
            self.logger.error(
                "Keyword search failed",
                query=query,
                error=str(e),
                exc_info=True,
            )
            raise

    async def _get_document_corpus(
        self, metadata_filter: Optional[Dict] = None
    ) -> Tuple[List[List[str]], List[Dict]]:
        """
        Get document corpus for BM25 indexing.

        Args:
            metadata_filter: Optional metadata filter

        Returns:
            Tuple of (tokenized_documents, metadata_list)
        """
        # For now, we'll get all documents from ChromaDB
        # In production, this should be optimized with proper corpus management

        # Get documents from vector store
        # This is a simplified approach - in production you'd want a dedicated corpus store
        try:
            # Use a large n_results to get all documents (limited approach)
            dummy_embedding = [0.0] * settings.embedding_dimensions

            raw_docs, metadata_list, _ = await self.vector_store.search_similar(
                query_embedding=dummy_embedding,
                n_results=10000,  # Large number to get most documents
                metadata_filter=metadata_filter,
            )

            # Tokenize documents
            tokenized_docs = []
            for doc in raw_docs:
                tokens = self.preprocessor.preprocess_query(doc)
                tokenized_docs.append(tokens)

            return tokenized_docs, metadata_list

        except Exception as e:
            self.logger.error("Error getting document corpus", error=str(e))
            return [], []

    async def _get_or_build_bm25_index(
        self,
        documents: List[List[str]],
        metadata_filter: Optional[Dict] = None,
    ) -> BM25Okapi:
        """
        Get or build BM25 index for the document corpus.

        Args:
            documents: Tokenized documents
            metadata_filter: Metadata filter (for cache key)

        Returns:
            BM25Okapi index
        """
        # Create cache key based on filter
        cache_key = str(hash(str(metadata_filter))) if metadata_filter else "default"

        if cache_key in self._bm25_cache:
            return self._bm25_cache[cache_key]

        # Build new BM25 index
        self.logger.info(
            "Building BM25 index",
            document_count=len(documents),
            cache_key=cache_key,
        )

        bm25 = await asyncio.get_event_loop().run_in_executor(
            self.executor, BM25Okapi, documents
        )

        # Cache the index (with size limit)
        if len(self._bm25_cache) >= 10:  # Limit cache size
            # Remove oldest entry
            oldest_key = next(iter(self._bm25_cache))
            del self._bm25_cache[oldest_key]

        self._bm25_cache[cache_key] = bm25
        return bm25

    def _rank_and_filter_results(
        self,
        documents: List[List[str]],
        metadata_list: List[Dict],
        scores: List[float],
        original_query: str,
        limit: int,
        offset: int,
    ) -> List[SearchResult]:
        """
        Rank and filter BM25 results.

        Args:
            documents: Tokenized documents
            metadata_list: Document metadata
            scores: BM25 scores
            original_query: Original search query
            limit: Maximum results
            offset: Results offset

        Returns:
            List of SearchResult objects
        """
        # Combine documents with scores and metadata
        combined_results = list(zip(documents, metadata_list, scores))

        # Sort by score (descending)
        combined_results.sort(key=lambda x: x[2], reverse=True)

        # Filter out zero scores and apply offset/limit
        filtered_results = [
            (doc, metadata, score)
            for doc, metadata, score in combined_results
            if score > 0.0
        ][offset : offset + limit]

        # Convert to SearchResult objects
        results = []
        for doc_tokens, metadata, score in filtered_results:
            try:
                # Reconstruct text from tokens (approximate)
                text = " ".join(doc_tokens)

                # Normalize BM25 score to 0-1 range (approximate)
                normalized_score = min(1.0, max(0.0, score / 10.0))

                result = SearchResult(
                    chunk_id=metadata.get("chunk_id"),
                    document_id=metadata.get("document_id"),
                    text=text,
                    chunk_index=metadata.get("chunk_index", 0),
                    score=normalized_score,
                    keyword_score=normalized_score,
                    document_title=metadata.get("title"),
                    document_filename=metadata.get("filename", "Unknown"),
                    document_type=metadata.get("document_type", "unknown"),
                    page_number=metadata.get("page_number"),
                    metadata=metadata,
                )

                results.append(result)

            except Exception as e:
                self.logger.error(
                    "Error creating keyword search result",
                    error=str(e),
                    metadata=metadata,
                )
                continue

        return results


class HybridSearchEngine(LoggerMixin):
    """Hybrid search engine combining semantic and keyword search."""

    def __init__(self):
        self.semantic_engine = get_semantic_search_engine()
        self.keyword_engine = KeywordSearchEngine()

    async def search(
        self,
        request: HybridSearchRequest,
        user_id: Optional[str] = None,
    ) -> Tuple[List[SearchResult], Dict[str, float]]:
        """
        Perform hybrid search combining semantic and keyword search.

        Args:
            request: Hybrid search request
            user_id: Optional user ID for logging

        Returns:
            Tuple of (search_results, timing_metrics)
        """
        start_time = time.time()
        timing_metrics = {}

        try:
            self.logger.info(
                "Starting hybrid search",
                query=request.query,
                semantic_weight=request.semantic_weight,
                keyword_weight=request.keyword_weight,
                user_id=user_id,
            )

            # Build metadata filter
            metadata_filter = self.semantic_engine._build_metadata_filter(
                request.filters
            )

            # Perform both searches concurrently
            semantic_task = asyncio.create_task(
                self._perform_semantic_search(request, metadata_filter)
            )
            keyword_task = asyncio.create_task(
                self._perform_keyword_search(request, metadata_filter)
            )

            # Wait for both searches to complete
            semantic_results, semantic_timing = await semantic_task
            keyword_results, keyword_timing = await keyword_task

            # Merge timing metrics
            timing_metrics.update(
                {f"semantic_{k}": v for k, v in semantic_timing.items()}
            )
            timing_metrics.update(
                {f"keyword_{k}": v for k, v in keyword_timing.items()}
            )

            # Fuse results
            fusion_start = time.time()
            fused_results = self._fuse_results(
                semantic_results,
                keyword_results,
                request.semantic_weight,
                request.keyword_weight,
                request.limit + request.offset,  # Get extra for offset
            )
            timing_metrics["fusion_time_ms"] = (time.time() - fusion_start) * 1000

            # Apply offset and limit
            final_results = fused_results[
                request.offset : request.offset + request.limit
            ]

            timing_metrics["total_time_ms"] = (time.time() - start_time) * 1000

            self.logger.info(
                "Hybrid search completed",
                query=request.query,
                semantic_results=len(semantic_results),
                keyword_results=len(keyword_results),
                fused_results=len(final_results),
                total_time_ms=timing_metrics["total_time_ms"],
                user_id=user_id,
            )

            return final_results, timing_metrics

        except Exception as e:
            self.logger.error(
                "Hybrid search failed",
                query=request.query,
                error=str(e),
                user_id=user_id,
                exc_info=True,
            )
            raise

    async def _perform_semantic_search(
        self,
        request: HybridSearchRequest,
        metadata_filter: Optional[Dict],
    ) -> Tuple[List[SearchResult], Dict[str, float]]:
        """Perform semantic search component."""
        # Convert hybrid request to semantic request
        from src.search.models import SemanticSearchRequest

        semantic_request = SemanticSearchRequest(
            query=request.query,
            limit=min(request.rerank_top_k, 100),  # Get more results for fusion
            offset=0,  # Don't apply offset at component level
            filters=request.filters,
            similarity_threshold=request.similarity_threshold,
            min_score=request.min_score,
            sort_by=request.sort_by,
        )

        return await self.semantic_engine.search(semantic_request)

    async def _perform_keyword_search(
        self,
        request: HybridSearchRequest,
        metadata_filter: Optional[Dict],
    ) -> Tuple[List[SearchResult], Dict[str, float]]:
        """Perform keyword search component."""
        return await self.keyword_engine.search(
            query=request.query,
            limit=min(request.rerank_top_k, 100),
            offset=0,
            metadata_filter=metadata_filter,
        )

    def _fuse_results(
        self,
        semantic_results: List[SearchResult],
        keyword_results: List[SearchResult],
        semantic_weight: float,
        keyword_weight: float,
        max_results: int = 50,
    ) -> List[SearchResult]:
        """
        Fuse semantic and keyword search results using weighted combination.

        Args:
            semantic_results: Results from semantic search
            keyword_results: Results from keyword search
            semantic_weight: Weight for semantic scores
            keyword_weight: Weight for keyword scores
            max_results: Maximum number of results to return

        Returns:
            Fused and ranked search results
        """
        # Create combined results dictionary by chunk_id
        combined_results = {}

        # Add semantic results
        for result in semantic_results:
            chunk_id = str(result.chunk_id)
            combined_results[chunk_id] = result
            # Initialize keyword score to 0
            result.keyword_score = 0.0

        # Add or update with keyword results
        for result in keyword_results:
            chunk_id = str(result.chunk_id)

            if chunk_id in combined_results:
                # Update existing result with keyword score
                existing = combined_results[chunk_id]
                existing.keyword_score = result.keyword_score
            else:
                # New result from keyword search
                result.semantic_score = 0.0
                combined_results[chunk_id] = result

        # Calculate combined scores
        for result in combined_results.values():
            semantic_score = result.semantic_score or 0.0
            keyword_score = result.keyword_score or 0.0

            # Weighted combination
            combined_score = (
                semantic_weight * semantic_score + keyword_weight * keyword_score
            )

            result.score = combined_score

        # Sort by combined score and limit results
        fused_results = sorted(
            combined_results.values(), key=lambda x: x.score, reverse=True
        )[:max_results]

        return fused_results


# Global hybrid search engine instance
_hybrid_search_engine: Optional[HybridSearchEngine] = None


def get_hybrid_search_engine() -> HybridSearchEngine:
    """Get the global hybrid search engine instance."""
    global _hybrid_search_engine
    if _hybrid_search_engine is None:
        _hybrid_search_engine = HybridSearchEngine()
    return _hybrid_search_engine


async def close_hybrid_search_engine():
    """Close the global hybrid search engine."""
    global _hybrid_search_engine
    if _hybrid_search_engine:
        if hasattr(_hybrid_search_engine.keyword_engine, "executor"):
            _hybrid_search_engine.keyword_engine.executor.shutdown(wait=True)
        _hybrid_search_engine = None
