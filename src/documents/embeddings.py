"""
Embedding Generation System
Handles text embedding generation with caching and batch processing.
"""

import asyncio
import hashlib
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

import redis.asyncio as redis_async
import torch
from sentence_transformers import SentenceTransformer

from src.core.config import get_settings
from src.core.logging import LoggerMixin
from src.documents.models import DocumentChunk

settings = get_settings()


class EmbeddingCache(LoggerMixin):
    """Redis-backed embedding cache with in-memory fallback."""

    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self._redis_client = None
        self._fallback_cache: Dict[str, Tuple[List[float], float]] = {}
        self._use_fallback = False
        self._logger = logging.getLogger(__name__)
        self._key_prefix = "rag:embedding_cache:"
        self._ttl = settings.cache_ttl_embeddings
        self._init_redis()

    def _init_redis(self) -> None:
        """Initialize async Redis connection for embedding cache."""
        try:
            self._redis_client = redis_async.from_url(
                settings.redis_url,
                decode_responses=True,
            )
        except Exception as e:
            self._logger.warning(
                "Redis unavailable for embedding cache, falling back to in-memory: %s",
                e,
            )
            self._use_fallback = True

    def _get_cache_key(self, text: str, model_name: str) -> str:
        """Generate cache key for text and model."""
        text_hash = hashlib.sha256(text.encode()).hexdigest()
        return f"{model_name}:{text_hash}"

    async def get(self, text: str, model_name: str) -> Optional[List[float]]:
        """Get embedding from cache."""
        key = self._get_cache_key(text, model_name)
        if self._use_fallback:
            return self._get_fallback(key)
        try:
            data = await self._redis_client.get(f"{self._key_prefix}{key}")
            if data is not None:
                return json.loads(data)
            return None
        except Exception as e:
            self._logger.warning(
                "Redis GET failed for embedding cache, using fallback: %s", e
            )
            self._use_fallback = True
            return self._get_fallback(key)

    async def set(self, text: str, model_name: str, embedding: List[float]) -> None:
        """Store embedding in cache."""
        key = self._get_cache_key(text, model_name)
        if self._use_fallback:
            self._set_fallback(key, embedding)
            return
        try:
            data = json.dumps(embedding)
            await self._redis_client.setex(
                f"{self._key_prefix}{key}",
                self._ttl,
                data,
            )
        except Exception as e:
            self._logger.warning(
                "Redis SET failed for embedding cache, using fallback: %s", e
            )
            self._use_fallback = True
            self._set_fallback(key, embedding)

    def _get_fallback(self, key: str) -> Optional[List[float]]:
        """Get from in-memory fallback cache."""
        if key in self._fallback_cache:
            embedding, timestamp = self._fallback_cache[key]
            if time.time() - timestamp < self._ttl:
                return embedding
            else:
                del self._fallback_cache[key]
        return None

    def _set_fallback(self, key: str, embedding: List[float]) -> None:
        """Set in in-memory fallback cache."""
        if len(self._fallback_cache) >= self.max_size:
            sorted_items = sorted(self._fallback_cache.items(), key=lambda x: x[1][1])
            evict_count = max(1, len(self._fallback_cache) // 5)
            for k, _ in sorted_items[:evict_count]:
                del self._fallback_cache[k]
        self._fallback_cache[key] = (embedding, time.time())

    def clear(self) -> None:
        """Clear all cache entries (sync-safe for use in close/cleanup)."""
        self._fallback_cache.clear()
        # Redis keys will expire via TTL; for immediate clear use async clear_async
        if self._redis_client and not self._use_fallback:
            try:
                import redis as redis_sync

                sync_client = redis_sync.from_url(
                    settings.redis_url, decode_responses=True
                )
                cursor = 0
                while True:
                    cursor, keys = sync_client.scan(
                        cursor=cursor, match=f"{self._key_prefix}*", count=100
                    )
                    if keys:
                        sync_client.delete(*keys)
                    if cursor == 0:
                        break
                sync_client.close()
            except Exception as e:
                self._logger.warning(
                    "Redis sync CLEAR failed for embedding cache: %s", e
                )

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "backend": "redis" if not self._use_fallback else "in-memory",
            "fallback_size": len(self._fallback_cache),
            "max_size": self.max_size,
            "ttl_seconds": self._ttl,
        }


class EmbeddingGenerator(LoggerMixin):
    """Embedding generator using sentence-transformers."""

    def __init__(
        self,
        model_name: str = None,
        device: str = None,
        batch_size: int = 32,
        enable_cache: bool = True,
    ):
        self.model_name = model_name or settings.embedding_model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.model: Optional[SentenceTransformer] = None
        self.executor = ThreadPoolExecutor(max_workers=2)

        # Initialize cache if enabled
        self.cache = EmbeddingCache() if enable_cache else None

        # Initialize model
        self._load_model()

    def _load_model(self) -> None:
        """Load the sentence transformer model."""
        try:
            self.logger.info(
                "Loading embedding model",
                model_name=self.model_name,
                device=self.device,
            )

            self.model = SentenceTransformer(self.model_name, device=self.device)

            # Set model to evaluation mode for consistency
            self.model.eval()

            self.logger.info(
                "Embedding model loaded successfully",
                model_name=self.model_name,
                device=self.device,
                max_seq_length=self.model.max_seq_length,
                embedding_dimension=self.model.get_sentence_embedding_dimension(),
            )

        except Exception as e:
            self.logger.error(
                "Failed to load embedding model",
                model_name=self.model_name,
                error=str(e),
            )
            raise

    async def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as list of floats
        """
        if not text.strip():
            raise ValueError("Text cannot be empty")

        # Check cache first
        if self.cache:
            cached_embedding = await self.cache.get(text, self.model_name)
            if cached_embedding is not None:
                self.logger.debug("Embedding retrieved from cache")
                return cached_embedding

        # Generate embedding
        try:
            loop = asyncio.get_running_loop()
            embedding = await loop.run_in_executor(
                self.executor, self._generate_embedding_sync, text
            )

            # Cache the result
            if self.cache:
                await self.cache.set(text, self.model_name, embedding)

            return embedding

        except Exception as e:
            self.logger.error(
                "Embedding generation failed", text_length=len(text), error=str(e)
            )
            raise

    async def generate_embeddings_batch(
        self, texts: List[str], show_progress: bool = False
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in batches.

        Args:
            texts: List of texts to embed
            show_progress: Whether to log progress

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        # Filter out cached embeddings
        embeddings = [None] * len(texts)
        texts_to_process = []
        indices_to_process = []

        if self.cache:
            for i, text in enumerate(texts):
                cached_embedding = await self.cache.get(text, self.model_name)
                if cached_embedding is not None:
                    embeddings[i] = cached_embedding
                else:
                    texts_to_process.append(text)
                    indices_to_process.append(i)
        else:
            texts_to_process = texts
            indices_to_process = list(range(len(texts)))

        if texts_to_process:
            self.logger.info(
                "Generating embeddings",
                total_texts=len(texts),
                cached_count=len(texts) - len(texts_to_process),
                to_process_count=len(texts_to_process),
            )

            # Process in batches
            new_embeddings = []
            for i in range(0, len(texts_to_process), self.batch_size):
                batch_texts = texts_to_process[i : i + self.batch_size]

                if show_progress:
                    self.logger.info(
                        "Processing batch",
                        batch_num=i // self.batch_size + 1,
                        total_batches=(len(texts_to_process) + self.batch_size - 1)
                        // self.batch_size,
                        batch_size=len(batch_texts),
                    )

                # Generate embeddings for batch
                loop = asyncio.get_running_loop()
                batch_embeddings = await loop.run_in_executor(
                    self.executor, self._generate_embeddings_batch_sync, batch_texts
                )

                new_embeddings.extend(batch_embeddings)

                # Cache new embeddings
                if self.cache:
                    for text, embedding in zip(batch_texts, batch_embeddings):
                        await self.cache.set(text, self.model_name, embedding)

            # Fill in the new embeddings
            for i, embedding in zip(indices_to_process, new_embeddings):
                embeddings[i] = embedding

        self.logger.info(
            "Batch embedding generation completed",
            total_texts=len(texts),
            cached_count=len(texts) - len(texts_to_process),
            generated_count=len(texts_to_process),
        )

        return embeddings

    async def generate_chunk_embeddings(
        self, chunks: List[DocumentChunk]
    ) -> List[DocumentChunk]:
        """
        Generate embeddings for document chunks.

        Args:
            chunks: List of document chunks

        Returns:
            Updated chunks with embeddings
        """
        if not chunks:
            return chunks

        # Extract texts
        texts = [chunk.text for chunk in chunks]

        # Generate embeddings
        embeddings = await self.generate_embeddings_batch(texts, show_progress=True)

        # Update chunks with embeddings
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding
            chunk.embedding_model = self.model_name

        self.logger.info(
            "Chunk embeddings generated",
            chunk_count=len(chunks),
            model_name=self.model_name,
        )

        return chunks

    def _generate_embedding_sync(self, text: str) -> List[float]:
        """Synchronously generate embedding for a single text."""
        if self.model is None:
            raise RuntimeError("Model not loaded")

        # Truncate text if too long
        if len(text) > self.model.max_seq_length:
            text = text[: self.model.max_seq_length]

        with torch.no_grad():
            embedding = self.model.encode(
                text, convert_to_tensor=False, normalize_embeddings=True
            )

        return embedding.tolist()

    def _generate_embeddings_batch_sync(self, texts: List[str]) -> List[List[float]]:
        """Synchronously generate embeddings for multiple texts."""
        if self.model is None:
            raise RuntimeError("Model not loaded")

        # Truncate texts if too long
        truncated_texts = []
        for text in texts:
            if len(text) > self.model.max_seq_length:
                truncated_texts.append(text[: self.model.max_seq_length])
            else:
                truncated_texts.append(text)

        with torch.no_grad():
            embeddings = self.model.encode(
                truncated_texts,
                convert_to_tensor=False,
                normalize_embeddings=True,
                batch_size=min(self.batch_size, len(truncated_texts)),
            )

        return embeddings.tolist()

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        if self.model is None:
            return {"status": "not_loaded"}

        return {
            "model_name": self.model_name,
            "device": self.device,
            "max_seq_length": self.model.max_seq_length,
            "embedding_dimension": self.model.get_sentence_embedding_dimension(),
            "batch_size": self.batch_size,
            "cache_enabled": self.cache is not None,
            "cache_stats": self.cache.get_stats() if self.cache else None,
        }

    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        if self.cache:
            self.cache.clear()
            self.logger.info("Embedding cache cleared")

    def close(self) -> None:
        """Clean up resources."""
        try:
            if self.executor:
                self.executor.shutdown(wait=True)
            if self.cache:
                self.cache.clear()
            self.logger.info("Embedding generator closed")
        except Exception as e:
            self.logger.error("Error closing embedding generator", error=str(e))


# Global embedding generator instance
_embedding_generator: Optional[EmbeddingGenerator] = None


def get_embedding_generator() -> EmbeddingGenerator:
    """Get the global embedding generator instance."""
    global _embedding_generator
    if _embedding_generator is None:
        _embedding_generator = EmbeddingGenerator()
    return _embedding_generator


async def close_embedding_generator():
    """Close the global embedding generator."""
    global _embedding_generator
    if _embedding_generator:
        _embedding_generator.close()
        _embedding_generator = None
