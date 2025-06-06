"""
Re-ranking System
Cross-encoder based re-ranking to improve search result relevance.
"""

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple

import torch
from sentence_transformers import CrossEncoder

from src.core.config import get_settings
from src.core.logging import LoggerMixin
from src.search.models import SearchResult

settings = get_settings()


class RerankingEngine(LoggerMixin):
    """Cross-encoder based re-ranking engine."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str = None,
        batch_size: int = 16,
    ):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.model: Optional[CrossEncoder] = None
        self.executor = ThreadPoolExecutor(max_workers=2)

        # Model loading is deferred to first use
        self._model_loaded = False

    async def _load_model(self) -> None:
        """Load the cross-encoder model."""
        if self._model_loaded:
            return

        try:
            self.logger.info(
                "Loading re-ranking model",
                model_name=self.model_name,
                device=self.device,
            )

            # Load model in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            self.model = await loop.run_in_executor(
                self.executor, lambda: CrossEncoder(self.model_name, device=self.device)
            )

            self._model_loaded = True

            self.logger.info(
                "Re-ranking model loaded successfully",
                model_name=self.model_name,
                device=self.device,
            )

        except Exception as e:
            self.logger.error(
                "Failed to load re-ranking model",
                model_name=self.model_name,
                error=str(e),
                exc_info=True,
            )
            raise

    async def rerank_results(
        self,
        query: str,
        results: List[SearchResult],
        top_k: Optional[int] = None,
    ) -> Tuple[List[SearchResult], Dict[str, float]]:
        """
        Re-rank search results using cross-encoder.

        Args:
            query: Original search query
            results: Search results to re-rank
            top_k: Number of top results to return (None for all)

        Returns:
            Tuple of (reranked_results, timing_metrics)
        """
        start_time = time.time()
        timing_metrics = {}

        if not results:
            return results, {"total_time_ms": 0.0}

        try:
            self.logger.info(
                "Starting result re-ranking",
                query=query,
                results_count=len(results),
                top_k=top_k,
            )

            # Load model if not already loaded
            model_load_start = time.time()
            await self._load_model()
            timing_metrics["model_load_time_ms"] = (
                time.time() - model_load_start
            ) * 1000

            # Prepare query-document pairs
            pairs_start = time.time()
            query_doc_pairs = [(query, result.text) for result in results]
            timing_metrics["pairs_prep_time_ms"] = (time.time() - pairs_start) * 1000

            # Perform re-ranking in batches
            ranking_start = time.time()
            rerank_scores = await self._predict_relevance_scores(query_doc_pairs)
            timing_metrics["ranking_time_ms"] = (time.time() - ranking_start) * 1000

            # Update results with re-ranking scores
            update_start = time.time()
            updated_results = self._update_results_with_scores(
                results, rerank_scores, top_k
            )
            timing_metrics["update_time_ms"] = (time.time() - update_start) * 1000

            timing_metrics["total_time_ms"] = (time.time() - start_time) * 1000

            self.logger.info(
                "Re-ranking completed",
                query=query,
                original_count=len(results),
                reranked_count=len(updated_results),
                total_time_ms=timing_metrics["total_time_ms"],
            )

            return updated_results, timing_metrics

        except Exception as e:
            self.logger.error(
                "Re-ranking failed",
                query=query,
                results_count=len(results),
                error=str(e),
                exc_info=True,
            )

            # Return original results as fallback
            return results, {"total_time_ms": (time.time() - start_time) * 1000}

    async def _predict_relevance_scores(
        self, query_doc_pairs: List[Tuple[str, str]]
    ) -> List[float]:
        """
        Predict relevance scores for query-document pairs.

        Args:
            query_doc_pairs: List of (query, document) pairs

        Returns:
            List of relevance scores
        """
        if not self.model:
            raise RuntimeError("Model not loaded")

        all_scores = []

        # Process in batches
        for i in range(0, len(query_doc_pairs), self.batch_size):
            batch = query_doc_pairs[i : i + self.batch_size]

            self.logger.debug(
                "Processing re-ranking batch",
                batch_num=i // self.batch_size + 1,
                batch_size=len(batch),
                total_batches=(len(query_doc_pairs) + self.batch_size - 1)
                // self.batch_size,
            )

            # Predict scores in thread pool
            loop = asyncio.get_event_loop()
            batch_scores = await loop.run_in_executor(
                self.executor, self._predict_batch_sync, batch
            )

            all_scores.extend(batch_scores)

        return all_scores

    def _predict_batch_sync(self, batch: List[Tuple[str, str]]) -> List[float]:
        """Synchronously predict scores for a batch."""
        try:
            with torch.no_grad():
                scores = self.model.predict(batch)

            # Convert scores to Python floats and normalize to 0-1 range
            # Cross-encoder scores are typically in a wider range
            normalized_scores = []
            for score in scores:
                # Apply sigmoid to normalize to 0-1 range
                normalized = float(torch.sigmoid(torch.tensor(score)))
                normalized_scores.append(normalized)

            return normalized_scores

        except Exception as e:
            self.logger.error(
                "Batch prediction failed",
                batch_size=len(batch),
                error=str(e),
            )
            # Return neutral scores as fallback
            return [0.5] * len(batch)

    def _update_results_with_scores(
        self,
        results: List[SearchResult],
        rerank_scores: List[float],
        top_k: Optional[int] = None,
    ) -> List[SearchResult]:
        """
        Update search results with re-ranking scores and sort.

        Args:
            results: Original search results
            rerank_scores: Re-ranking scores
            top_k: Number of top results to return

        Returns:
            Updated and sorted search results
        """
        if len(results) != len(rerank_scores):
            self.logger.warning(
                "Mismatch between results and scores",
                results_count=len(results),
                scores_count=len(rerank_scores),
            )
            # Take minimum length to avoid index errors
            min_length = min(len(results), len(rerank_scores))
            results = results[:min_length]
            rerank_scores = rerank_scores[:min_length]

        # Update results with re-ranking scores
        updated_results = []
        for result, rerank_score in zip(results, rerank_scores):
            # Create a copy of the result to avoid modifying the original
            updated_result = result.copy(deep=True)
            updated_result.rerank_score = rerank_score

            # Update the main score to be the re-ranking score
            # Keep original scores for reference
            updated_result.score = rerank_score

            updated_results.append(updated_result)

        # Sort by re-ranking score (descending)
        updated_results.sort(key=lambda x: x.rerank_score or 0.0, reverse=True)

        # Apply top_k limit if specified
        if top_k is not None:
            updated_results = updated_results[:top_k]

        return updated_results

    def get_model_info(self) -> Dict[str, any]:
        """
        Get information about the re-ranking model.

        Returns:
            Dictionary of model information
        """
        return {
            "model_name": self.model_name,
            "device": self.device,
            "batch_size": self.batch_size,
            "model_loaded": self._model_loaded,
            "supported_features": [
                "cross_encoder_reranking",
                "batch_processing",
                "score_normalization",
                "fallback_handling",
            ],
        }

    def close(self) -> None:
        """Clean up resources."""
        try:
            if self.executor:
                self.executor.shutdown(wait=True)

            # Clear model from memory
            if self.model:
                del self.model
                self.model = None

            # Clear CUDA cache if using GPU
            if self.device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()

            self._model_loaded = False

            self.logger.info("Re-ranking engine closed")

        except Exception as e:
            self.logger.error("Error closing re-ranking engine", error=str(e))


class FallbackRerankingEngine(LoggerMixin):
    """Fallback re-ranking using simple text similarity metrics."""

    def __init__(self):
        pass

    async def rerank_results(
        self,
        query: str,
        results: List[SearchResult],
        top_k: Optional[int] = None,
    ) -> Tuple[List[SearchResult], Dict[str, float]]:
        """
        Simple fallback re-ranking using text overlap metrics.

        Args:
            query: Original search query
            results: Search results to re-rank
            top_k: Number of top results to return

        Returns:
            Tuple of (reranked_results, timing_metrics)
        """
        start_time = time.time()

        try:
            self.logger.info(
                "Using fallback re-ranking",
                query=query,
                results_count=len(results),
            )

            # Simple token overlap scoring
            query_tokens = set(query.lower().split())

            for result in results:
                doc_tokens = set(result.text.lower().split())

                # Calculate Jaccard similarity
                intersection = len(query_tokens & doc_tokens)
                union = len(query_tokens | doc_tokens)

                jaccard_score = intersection / union if union > 0 else 0.0

                # Update result
                result.rerank_score = jaccard_score
                result.score = jaccard_score

            # Sort by fallback score
            results.sort(key=lambda x: x.rerank_score or 0.0, reverse=True)

            # Apply top_k limit
            if top_k is not None:
                results = results[:top_k]

            timing_metrics = {
                "total_time_ms": (time.time() - start_time) * 1000,
                "fallback_used": True,
            }

            return results, timing_metrics

        except Exception as e:
            self.logger.error(
                "Fallback re-ranking failed",
                query=query,
                error=str(e),
                exc_info=True,
            )

            # Return original results without modification
            timing_metrics = {
                "total_time_ms": (time.time() - start_time) * 1000,
                "fallback_failed": True,
            }

            return results, timing_metrics


# Global re-ranking engine instance
_reranking_engine: Optional[RerankingEngine] = None
_fallback_engine: Optional[FallbackRerankingEngine] = None


def get_reranking_engine() -> RerankingEngine:
    """Get the global re-ranking engine instance."""
    global _reranking_engine
    if _reranking_engine is None:
        _reranking_engine = RerankingEngine()
    return _reranking_engine


def get_fallback_reranking_engine() -> FallbackRerankingEngine:
    """Get the fallback re-ranking engine instance."""
    global _fallback_engine
    if _fallback_engine is None:
        _fallback_engine = FallbackRerankingEngine()
    return _fallback_engine


async def close_reranking_engines():
    """Close all re-ranking engines."""
    global _reranking_engine, _fallback_engine

    if _reranking_engine:
        _reranking_engine.close()
        _reranking_engine = None

    if _fallback_engine:
        _fallback_engine = None
