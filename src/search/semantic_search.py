"""
Semantic Search Engine
Implements vector-based similarity search using ChromaDB and sentence transformers.
"""

import time
from typing import Dict, List, Optional, Tuple

from src.core.config import get_settings
from src.core.logging import LoggerMixin
from src.documents.embeddings import get_embedding_generator
from src.search.models import (
    MetadataFilter,
    SearchResult,
    SemanticSearchRequest,
)
from src.vector_store.chroma_client import get_chroma_client

settings = get_settings()


class SemanticSearchEngine(LoggerMixin):
    """Semantic search engine using vector similarity."""

    def __init__(self):
        self.embedding_generator = get_embedding_generator()
        self.vector_store = get_chroma_client()

    async def search(
        self, request: SemanticSearchRequest, user_id: Optional[str] = None
    ) -> Tuple[List[SearchResult], Dict[str, float]]:
        """
        Perform semantic search using vector similarity.

        Args:
            request: Semantic search request
            user_id: Optional user ID for logging

        Returns:
            Tuple of (search_results, timing_metrics)
        """
        start_time = time.time()
        timing_metrics = {}

        try:
            self.logger.info(
                "Starting semantic search",
                query=request.query,
                limit=request.limit,
                user_id=user_id,
                similarity_threshold=request.similarity_threshold,
            )

            # Generate query embedding
            embedding_start = time.time()
            query_embedding = await self.embedding_generator.generate_embedding(
                request.query
            )
            timing_metrics["embedding_time_ms"] = (time.time() - embedding_start) * 1000

            # Build metadata filter
            metadata_filter = self._build_metadata_filter(request.filters)

            # Perform similarity search
            search_start = time.time()
            (
                documents,
                metadata_list,
                distances,
            ) = await self.vector_store.search_similar(
                query_embedding=query_embedding,
                n_results=min(
                    request.limit + request.offset, 1000
                ),  # Get extra for offset
                metadata_filter=metadata_filter,
            )
            timing_metrics["vector_search_time_ms"] = (
                time.time() - search_start
            ) * 1000

            # Convert distances to similarity scores (ChromaDB returns cosine distances)
            # ChromaDB cosine distance = 2 * (1 - cosine_similarity)
            # Therefore: cosine_similarity = 1 - (distance / 2)
            similarity_scores = [
                max(0.0, 1.0 - (distance / 2.0)) for distance in distances
            ]

            # Filter by similarity threshold and apply offset/limit
            filtered_results = []
            for i, (doc, metadata, score) in enumerate(
                zip(documents, metadata_list, similarity_scores)
            ):
                if score >= request.similarity_threshold:
                    if i >= request.offset:
                        filtered_results.append((doc, metadata, score))
                        if len(filtered_results) >= request.limit:
                            break

            # Convert to SearchResult objects
            results = self._convert_to_search_results(
                filtered_results, request.include_embeddings
            )

            timing_metrics["total_time_ms"] = (time.time() - start_time) * 1000

            self.logger.info(
                "Semantic search completed",
                query=request.query,
                results_count=len(results),
                total_time_ms=timing_metrics["total_time_ms"],
                user_id=user_id,
            )

            return results, timing_metrics

        except Exception as e:
            self.logger.error(
                "Semantic search failed",
                query=request.query,
                error=str(e),
                user_id=user_id,
                exc_info=True,
            )
            raise

    async def find_similar_documents(
        self,
        document_id: str,
        limit: int = 10,
        similarity_threshold: float = 0.5,
        filters: Optional[MetadataFilter] = None,
        exclude_same_document: bool = True,
    ) -> List[SearchResult]:
        """
        Find documents similar to a reference document.

        Args:
            document_id: Reference document ID
            limit: Maximum number of results
            similarity_threshold: Minimum similarity score
            filters: Optional metadata filters
            exclude_same_document: Whether to exclude chunks from the same document

        Returns:
            List of similar search results
        """
        try:
            self.logger.info(
                "Finding similar documents",
                reference_document_id=document_id,
                limit=limit,
                similarity_threshold=similarity_threshold,
            )

            # Get reference document embedding
            reference_doc = await self.vector_store.get_document(document_id)
            if not reference_doc or not reference_doc.get("embedding"):
                raise ValueError(
                    f"Document {document_id} not found or has no embedding"
                )

            reference_embedding = reference_doc["embedding"]

            # Build metadata filter
            metadata_filter = self._build_metadata_filter(filters)

            # Add exclusion filter if needed
            if exclude_same_document and reference_doc.get("metadata"):
                ref_doc_id = reference_doc["metadata"].get("document_id")
                if ref_doc_id:
                    if metadata_filter is None:
                        metadata_filter = {}
                    # ChromaDB filter to exclude same document
                    metadata_filter["document_id"] = {"$ne": ref_doc_id}

            # Perform similarity search
            (
                documents,
                metadata_list,
                distances,
            ) = await self.vector_store.search_similar(
                query_embedding=reference_embedding,
                n_results=limit + 1,  # +1 in case reference doc is included
                metadata_filter=metadata_filter,
            )

            # Convert and filter results
            similarity_scores = [
                max(0.0, 1.0 - (distance / 2.0)) for distance in distances
            ]
            filtered_results = [
                (doc, metadata, score)
                for doc, metadata, score in zip(
                    documents, metadata_list, similarity_scores
                )
                if score >= similarity_threshold
            ]

            # Remove reference document if it somehow got included
            if not exclude_same_document:
                filtered_results = [
                    (doc, metadata, score)
                    for doc, metadata, score in filtered_results
                    if metadata.get("chunk_id") != document_id
                ][:limit]

            results = self._convert_to_search_results(
                filtered_results, include_embeddings=False
            )

            self.logger.info(
                "Similar documents search completed",
                reference_document_id=document_id,
                results_count=len(results),
            )

            return results

        except Exception as e:
            self.logger.error(
                "Similar documents search failed",
                reference_document_id=document_id,
                error=str(e),
                exc_info=True,
            )
            raise

    def _build_metadata_filter(
        self, filters: Optional[MetadataFilter]
    ) -> Optional[Dict]:
        """
        Build ChromaDB metadata filter from request filters.

        Args:
            filters: Optional metadata filters

        Returns:
            ChromaDB-compatible metadata filter
        """
        if not filters:
            return None

        metadata_filter = {}

        # Simple equality filters
        if filters.document_type:
            metadata_filter["document_type"] = filters.document_type

        if filters.uploaded_by:
            metadata_filter["uploaded_by"] = filters.uploaded_by

        if filters.title:
            metadata_filter["title"] = {"$contains": filters.title}

        if filters.author:
            metadata_filter["author"] = {"$contains": filters.author}

        # Date range filters
        if filters.date_from or filters.date_to:
            date_filter = {}
            if filters.date_from:
                date_filter["$gte"] = filters.date_from.isoformat()
            if filters.date_to:
                date_filter["$lte"] = filters.date_to.isoformat()
            metadata_filter["created_at"] = date_filter

        # Tags filter (if document has tags)
        if filters.tags:
            metadata_filter["tags"] = {"$in": filters.tags}

        # Custom filters
        if filters.custom_filters:
            metadata_filter.update(filters.custom_filters)

        return metadata_filter if metadata_filter else None

    def _convert_to_search_results(
        self,
        raw_results: List[Tuple[str, Dict, float]],
        include_embeddings: bool = False,
    ) -> List[SearchResult]:
        """
        Convert raw search results to SearchResult objects.

        Args:
            raw_results: List of (document_text, metadata, score) tuples
            include_embeddings: Whether to include embeddings in results

        Returns:
            List of SearchResult objects
        """
        results = []

        for text, metadata, score in raw_results:
            try:
                # Extract required fields with defaults
                # ChromaDB might store the chunk ID as the document ID or in metadata
                chunk_id = metadata.get("chunk_id") or metadata.get("id")
                document_id = metadata.get("document_id")

                if not document_id:
                    self.logger.warning(
                        "Skipping result with missing document_id",
                        chunk_id=chunk_id,
                        document_id=document_id,
                        metadata_keys=list(metadata.keys()),
                        metadata=metadata,
                    )
                    continue

                # Use a fallback chunk_id if not present
                if not chunk_id:
                    chunk_id = f"{document_id}_chunk_{metadata.get('chunk_index', 0)}"
                    self.logger.debug(
                        "Generated fallback chunk_id",
                        chunk_id=chunk_id,
                        document_id=document_id,
                    )

                result = SearchResult(
                    chunk_id=chunk_id,
                    document_id=document_id,
                    text=text,
                    chunk_index=metadata.get("chunk_index", 0),
                    score=score,
                    semantic_score=score,
                    document_title=metadata.get("title"),
                    document_filename=metadata.get("filename", "Unknown"),
                    document_type=metadata.get("document_type", "unknown"),
                    page_number=metadata.get("page_number"),
                    metadata=metadata,
                )

                results.append(result)

            except Exception as e:
                self.logger.error(
                    "Error converting search result",
                    error=str(e),
                    metadata=metadata,
                    text_length=len(text) if text else 0,
                )
                continue

        return results

    async def get_search_stats(self) -> Dict[str, any]:
        """
        Get semantic search engine statistics.

        Returns:
            Dictionary of search statistics
        """
        try:
            # Get vector store stats
            chroma_stats = await self.vector_store.get_collection_stats()

            # Get embedding model info
            model_info = self.embedding_generator.get_model_info()

            return {
                "engine_type": "semantic",
                "vector_store": chroma_stats,
                "embedding_model": model_info,
                "supported_features": [
                    "similarity_search",
                    "metadata_filtering",
                    "batch_processing",
                    "caching",
                ],
            }

        except Exception as e:
            self.logger.error("Error getting search stats", error=str(e))
            return {"error": str(e)}


# Global semantic search engine instance
_semantic_search_engine: Optional[SemanticSearchEngine] = None


def get_semantic_search_engine() -> SemanticSearchEngine:
    """Get the global semantic search engine instance."""
    global _semantic_search_engine
    if _semantic_search_engine is None:
        _semantic_search_engine = SemanticSearchEngine()
    return _semantic_search_engine


async def close_semantic_search_engine():
    """Close the global semantic search engine."""
    global _semantic_search_engine
    if _semantic_search_engine:
        # No explicit cleanup needed for now
        _semantic_search_engine = None
