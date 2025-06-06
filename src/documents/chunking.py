"""
Text Chunking System
Implements recursive character text splitter with tiktoken for accurate token counting.
"""

import re
from typing import Any, Dict, List, Optional
from uuid import UUID

import tiktoken
from pydantic import BaseModel

from src.core.config import get_settings
from src.core.logging import LoggerMixin
from src.documents.models import DocumentChunk

settings = get_settings()


class ChunkMetadata(BaseModel):
    """Metadata for a text chunk."""

    document_id: UUID
    source_file: str
    chunk_index: int
    start_char: int
    end_char: int
    page_number: Optional[int] = None
    section_title: Optional[str] = None
    document_type: str
    language: Optional[str] = None


class TextChunker(LoggerMixin):
    """Base class for text chunking strategies."""

    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None,
        encoding_name: str = "cl100k_base",
    ):
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        self.encoding = tiktoken.get_encoding(encoding_name)

    def count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken."""
        return len(self.encoding.encode(text))

    def chunk_text(
        self, text: str, document_id: UUID, metadata: Dict[str, Any]
    ) -> List[DocumentChunk]:
        """Chunk text into smaller pieces. To be implemented by subclasses."""
        raise NotImplementedError


class RecursiveCharacterTextSplitter(TextChunker):
    """
    Recursive character text splitter that preserves document structure.
    Splits text hierarchically using different separators.
    """

    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None,
        separators: Optional[List[str]] = None,
        encoding_name: str = "cl100k_base",
    ):
        super().__init__(chunk_size, chunk_overlap, encoding_name)
        self.separators = separators or settings.chunk_separators.copy()

    def chunk_text(
        self, text: str, document_id: UUID, metadata: Dict[str, Any]
    ) -> List[DocumentChunk]:
        """
        Split text into chunks using recursive character splitting.

        Args:
            text: Text to split
            document_id: Document UUID
            metadata: Document metadata

        Returns:
            List of DocumentChunk objects
        """
        if not text.strip():
            return []

        # Clean and normalize text
        text = self._clean_text(text)

        # Split text into chunks
        chunks = self._split_text_recursive(text)

        # Create DocumentChunk objects
        document_chunks = []
        current_position = 0

        for i, chunk_text in enumerate(chunks):
            # Find the actual position of this chunk in the original text
            chunk_start = text.find(chunk_text, current_position)
            if chunk_start == -1:
                # Fallback to estimated position
                chunk_start = current_position

            chunk_end = chunk_start + len(chunk_text)
            token_count = self.count_tokens(chunk_text)

            # Create chunk metadata
            chunk_metadata = {
                **metadata,
                "chunk_index": i,
                "separator_used": self._get_separator_used(chunk_text),
                "original_length": len(chunk_text),
            }

            # Extract page number if available in metadata
            page_number = self._extract_page_number(
                chunk_text, metadata.get("page_content", {})
            )

            chunk = DocumentChunk(
                document_id=document_id,
                chunk_index=i,
                text=chunk_text.strip(),
                token_count=token_count,
                start_char=chunk_start,
                end_char=chunk_end,
                page_number=page_number,
                metadata=chunk_metadata,
            )

            document_chunks.append(chunk)
            current_position = chunk_end - self.chunk_overlap

        self.logger.info(
            "Text chunking completed",
            document_id=str(document_id),
            original_length=len(text),
            chunk_count=len(document_chunks),
            avg_chunk_size=sum(len(c.text) for c in document_chunks)
            / len(document_chunks)
            if document_chunks
            else 0,
        )

        return document_chunks

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Remove excessive whitespace
        text = re.sub(r"\n\s*\n\s*\n", "\n\n", text)
        text = re.sub(r" +", " ", text)

        # Remove trailing whitespace from lines
        text = "\n".join(line.rstrip() for line in text.split("\n"))

        return text.strip()

    def _split_text_recursive(self, text: str) -> List[str]:
        """Recursively split text using different separators."""
        return self._split_text(text, self.separators)

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """Split text using the given separators."""
        if not separators:
            return [text]

        separator = separators[0]
        remaining_separators = separators[1:]

        # Split by the current separator
        splits = text.split(separator)

        final_chunks = []
        current_chunk = ""

        for split in splits:
            # Check if adding this split would exceed chunk size
            test_chunk = current_chunk + (separator if current_chunk else "") + split

            if self.count_tokens(test_chunk) <= self.chunk_size:
                current_chunk = test_chunk
            else:
                # Current chunk is full, process it
                if current_chunk:
                    if (
                        self.count_tokens(current_chunk) > self.chunk_size
                        and remaining_separators
                    ):
                        # Current chunk is still too big, split further
                        sub_chunks = self._split_text(
                            current_chunk, remaining_separators
                        )
                        final_chunks.extend(sub_chunks)
                    else:
                        final_chunks.append(current_chunk)

                # Start new chunk with the current split
                current_chunk = split

        # Add the last chunk
        if current_chunk:
            if (
                self.count_tokens(current_chunk) > self.chunk_size
                and remaining_separators
            ):
                sub_chunks = self._split_text(current_chunk, remaining_separators)
                final_chunks.extend(sub_chunks)
            else:
                final_chunks.append(current_chunk)

        # Apply overlap between chunks
        return self._apply_overlap(final_chunks)

    def _apply_overlap(self, chunks: List[str]) -> List[str]:
        """Apply overlap between consecutive chunks."""
        if len(chunks) <= 1 or self.chunk_overlap <= 0:
            return chunks

        overlapped_chunks = []

        for i, chunk in enumerate(chunks):
            if i == 0:
                overlapped_chunks.append(chunk)
                continue

            # Get overlap from previous chunk
            prev_chunk = chunks[i - 1]
            overlap_tokens = min(self.chunk_overlap, self.count_tokens(prev_chunk))

            # Find overlap text by token count
            overlap_text = self._get_text_by_token_count(
                prev_chunk, overlap_tokens, from_end=True
            )

            # Combine overlap with current chunk
            combined_chunk = overlap_text + "\n" + chunk if overlap_text else chunk
            overlapped_chunks.append(combined_chunk)

        return overlapped_chunks

    def _get_text_by_token_count(
        self, text: str, token_count: int, from_end: bool = False
    ) -> str:
        """Extract text with approximately the specified token count."""
        if token_count <= 0:
            return ""

        # Binary search to find the right substring length
        left, right = 0, len(text)

        while left < right:
            mid = (left + right + 1) // 2

            if from_end:
                substring = text[-mid:] if mid > 0 else ""
            else:
                substring = text[:mid]

            if self.count_tokens(substring) <= token_count:
                left = mid
            else:
                right = mid - 1

        if from_end:
            return text[-left:] if left > 0 else ""
        else:
            return text[:left]

    def _get_separator_used(self, chunk_text: str) -> Optional[str]:
        """Determine which separator was primarily used for this chunk."""
        for separator in self.separators:
            if separator in chunk_text:
                return repr(separator)
        return None

    def _extract_page_number(
        self, chunk_text: str, page_content: Dict[str, Any]
    ) -> Optional[int]:
        """Extract page number for the chunk based on content mapping."""
        # This is a simplified implementation
        # In a real scenario, you'd maintain a mapping of text positions to page numbers
        if not page_content:
            return None

        # Look for page markers in the chunk
        page_pattern = r"(?:Page|page|PAGE)\s*(\d+)"
        match = re.search(page_pattern, chunk_text)
        if match:
            return int(match.group(1))

        return None


class DeduplicationChunker(RecursiveCharacterTextSplitter):
    """Text chunker with deduplication capabilities."""

    def __init__(self, *args, similarity_threshold: float = 0.95, **kwargs):
        super().__init__(*args, **kwargs)
        self.similarity_threshold = similarity_threshold
        self._chunk_hashes = set()

    def chunk_text(
        self, text: str, document_id: UUID, metadata: Dict[str, Any]
    ) -> List[DocumentChunk]:
        """Chunk text with deduplication."""
        chunks = super().chunk_text(text, document_id, metadata)

        # Remove duplicate chunks
        deduplicated_chunks = []
        for chunk in chunks:
            chunk_hash = self._get_chunk_hash(chunk.text)

            if chunk_hash not in self._chunk_hashes:
                self._chunk_hashes.add(chunk_hash)
                deduplicated_chunks.append(chunk)
            else:
                self.logger.debug(
                    "Duplicate chunk detected and removed",
                    document_id=str(document_id),
                    chunk_index=chunk.chunk_index,
                )

        # Update chunk indices after deduplication
        for i, chunk in enumerate(deduplicated_chunks):
            chunk.chunk_index = i

        if len(deduplicated_chunks) < len(chunks):
            self.logger.info(
                "Chunk deduplication completed",
                document_id=str(document_id),
                original_count=len(chunks),
                deduplicated_count=len(deduplicated_chunks),
                removed_count=len(chunks) - len(deduplicated_chunks),
            )

        return deduplicated_chunks

    def _get_chunk_hash(self, text: str) -> str:
        """Generate a hash for chunk content to detect duplicates."""
        import hashlib

        # Normalize text for hashing
        normalized = re.sub(r"\s+", " ", text.strip().lower())
        return hashlib.md5(normalized.encode()).hexdigest()

    def clear_deduplication_cache(self):
        """Clear the deduplication cache."""
        self._chunk_hashes.clear()


def get_text_chunker(
    strategy: str = "recursive", enable_deduplication: bool = True, **kwargs
) -> TextChunker:
    """
    Factory function to get a text chunker instance.

    Args:
        strategy: Chunking strategy ("recursive")
        enable_deduplication: Whether to enable chunk deduplication
        **kwargs: Additional arguments for the chunker

    Returns:
        TextChunker instance
    """
    if strategy == "recursive":
        if enable_deduplication:
            return DeduplicationChunker(**kwargs)
        else:
            return RecursiveCharacterTextSplitter(**kwargs)
    else:
        raise ValueError(f"Unknown chunking strategy: {strategy}")
