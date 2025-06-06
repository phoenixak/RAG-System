"""
Document Deduplication Service
Handles content-based deduplication using file hashes and database cleanup.
"""

import asyncio
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from src.core.config import get_settings
from src.core.logging import get_logger
from src.vector_store.chroma_client import ChromaDBClient

settings = get_settings()
logger = get_logger(__name__)


class DuplicateDocumentError(Exception):
    """Exception raised when a duplicate document is detected."""

    def __init__(
        self,
        filename: str,
        existing_document_id: str,
        existing_filename: str,
        file_hash: str,
    ):
        self.filename = filename
        self.existing_document_id = existing_document_id
        self.existing_filename = existing_filename
        self.file_hash = file_hash
        super().__init__(
            f"Duplicate document detected: '{filename}' already exists as '{existing_filename}' "
            f"(ID: {existing_document_id}, Hash: {file_hash[:8]}...)"
        )


class DocumentDeduplicationService:
    """Service for document deduplication and cleanup operations."""

    def __init__(self, chroma_client: Optional[ChromaDBClient] = None):
        self.chroma_client = chroma_client

    def calculate_file_hash(self, file_content: bytes) -> str:
        """
        Calculate SHA256 hash of file content.

        Args:
            file_content: Raw file bytes

        Returns:
            Hexadecimal hash string
        """
        try:
            sha256_hash = hashlib.sha256()
            sha256_hash.update(file_content)
            return sha256_hash.hexdigest()
        except Exception as e:
            logger.error(f"Failed to calculate file hash: {e}")
            raise

    def calculate_file_hash_from_path(self, file_path: str) -> str:
        """
        Calculate SHA256 hash of file from path.

        Args:
            file_path: Path to file

        Returns:
            Hexadecimal hash string
        """
        try:
            with open(file_path, "rb") as f:
                return self.calculate_file_hash(f.read())
        except Exception as e:
            logger.error(f"Failed to calculate file hash from path {file_path}: {e}")
            raise

    async def check_duplicate_by_hash(self, file_hash: str) -> Optional[Dict[str, Any]]:
        """
        Check if a document with the same hash already exists.

        Args:
            file_hash: SHA256 hash of the file

        Returns:
            Existing document metadata if duplicate found, None otherwise
        """
        try:
            if not self.chroma_client:
                return None

            # Search for documents with the same hash
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                self.chroma_client.executor,
                lambda: self.chroma_client.collection.get(
                    where={"file_hash": file_hash}, include=["metadatas"]
                ),
            )

            if results["ids"] and len(results["ids"]) > 0:
                # Found duplicate
                metadata = results["metadatas"][0]
                logger.info(f"Duplicate document found with hash: {file_hash}")
                return {"id": results["ids"][0], "metadata": metadata}

            return None

        except Exception as e:
            logger.error(f"Failed to check duplicate by hash: {e}")
            return None

    async def check_duplicate_by_content(
        self, file_content: bytes, filename: str
    ) -> Optional[Dict[str, Any]]:
        """
        Check if document content already exists in the system.

        Args:
            file_content: Raw file bytes
            filename: Original filename

        Returns:
            Existing document info if duplicate found, None otherwise
        """
        try:
            # Calculate hash
            file_hash = self.calculate_file_hash(file_content)

            # Check for duplicate
            duplicate_info = await self.check_duplicate_by_hash(file_hash)

            if duplicate_info:
                logger.info(
                    "Content-based duplicate detected",
                    filename=filename,
                    hash=file_hash,
                    existing_id=duplicate_info["id"],
                )

            return duplicate_info

        except Exception as e:
            logger.error(f"Failed to check content duplicate: {e}")
            return None

    async def find_existing_duplicates(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Scan existing documents and find duplicates based on file hash.

        Returns:
            Dictionary mapping hash to list of duplicate documents
        """
        try:
            if not self.chroma_client:
                logger.warning("ChromaDB client not available for duplicate scan")
                return {}

            logger.info("Starting duplicate document scan")

            # Get all documents from ChromaDB
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                self.chroma_client.executor,
                lambda: self.chroma_client.collection.get(include=["metadatas"]),
            )

            if not results["ids"]:
                logger.info("No documents found in database")
                return {}

            # Group documents by file hash
            hash_groups: Dict[str, List[Dict[str, Any]]] = {}

            for i, doc_id in enumerate(results["ids"]):
                metadata = results["metadatas"][i]
                file_hash = metadata.get("file_hash")

                # If no file hash in metadata, try to calculate from file
                if not file_hash and metadata.get("filename"):
                    # Try to find file in uploads directory
                    upload_dir = Path(
                        getattr(settings, "upload_dir", None)
                        or settings.upload_path
                        or "uploads"
                    )

                    # Look for files with this filename
                    potential_files = list(upload_dir.glob(f"*_{metadata['filename']}"))
                    if potential_files:
                        try:
                            file_hash = self.calculate_file_hash_from_path(
                                str(potential_files[0])
                            )
                        except Exception as e:
                            logger.warning(
                                f"Could not calculate hash for {potential_files[0]}: {e}"
                            )
                            continue

                if file_hash:
                    if file_hash not in hash_groups:
                        hash_groups[file_hash] = []

                    hash_groups[file_hash].append(
                        {"id": doc_id, "metadata": metadata, "hash": file_hash}
                    )

            # Filter to only groups with duplicates
            duplicates = {}
            for file_hash, docs in hash_groups.items():
                if len(docs) > 1:
                    duplicates[file_hash] = docs

            logger.info(
                f"Duplicate scan completed. Found {len(duplicates)} duplicate groups affecting {sum(len(docs) for docs in duplicates.values())} documents"
            )

            return duplicates

        except Exception as e:
            logger.error(f"Failed to find existing duplicates: {e}")
            return {}

    async def cleanup_duplicates(
        self, keep_strategy: str = "oldest", dry_run: bool = False
    ) -> Dict[str, int]:
        """
        Clean up duplicate documents from database and filesystem.

        Args:
            keep_strategy: Strategy for which duplicate to keep ('oldest', 'newest')
            dry_run: If True, only report what would be deleted without actually deleting

        Returns:
            Cleanup statistics
        """
        try:
            logger.info(
                f"Starting duplicate cleanup with strategy: {keep_strategy}, dry_run: {dry_run}"
            )

            duplicates = await self.find_existing_duplicates()
            if not duplicates:
                logger.info("No duplicates found to clean up")
                return {"groups_processed": 0, "documents_removed": 0, "errors": 0}

            stats = {"groups_processed": 0, "documents_removed": 0, "errors": 0}

            for file_hash, duplicate_docs in duplicates.items():
                try:
                    if len(duplicate_docs) <= 1:
                        continue

                    # Determine which document to keep
                    if keep_strategy == "oldest":
                        # Keep the first uploaded (oldest created_at)
                        sorted_docs = sorted(
                            duplicate_docs,
                            key=lambda x: x["metadata"].get("created_at", ""),
                        )
                    else:  # newest
                        # Keep the last uploaded (newest created_at)
                        sorted_docs = sorted(
                            duplicate_docs,
                            key=lambda x: x["metadata"].get("created_at", ""),
                            reverse=True,
                        )

                    docs_to_keep = [sorted_docs[0]]
                    docs_to_remove = sorted_docs[1:]

                    logger.info(
                        "Processing duplicate group",
                        hash=file_hash,
                        total_docs=len(duplicate_docs),
                        keeping=len(docs_to_keep),
                        removing=len(docs_to_remove),
                    )

                    if not dry_run:
                        # Remove duplicate documents from ChromaDB
                        for doc in docs_to_remove:
                            try:
                                await self._remove_document_from_chroma(doc["id"])
                                await self._remove_file_from_uploads(doc["metadata"])
                                stats["documents_removed"] += 1
                            except Exception as e:
                                logger.error(
                                    f"Failed to remove document {doc['id']}: {e}"
                                )
                                stats["errors"] += 1

                    stats["groups_processed"] += 1

                except Exception as e:
                    logger.error(f"Failed to process duplicate group {file_hash}: {e}")
                    stats["errors"] += 1

            logger.info(
                f"Duplicate cleanup completed. Groups: {stats['groups_processed']}, "
                f"Removed: {stats['documents_removed']}, Errors: {stats['errors']}"
            )

            return stats

        except Exception as e:
            logger.error(f"Failed to cleanup duplicates: {e}")
            return {"groups_processed": 0, "documents_removed": 0, "errors": 1}

    async def cleanup_orphaned_files(self) -> Dict[str, int]:
        """
        Clean up orphaned files in uploads directory that have no corresponding database entry.

        Returns:
            Cleanup statistics
        """
        try:
            logger.info("Starting orphaned file cleanup")

            upload_dir = Path(
                getattr(settings, "upload_dir", None)
                or settings.upload_path
                or "uploads"
            )

            if not upload_dir.exists():
                logger.info("Upload directory does not exist")
                return {"files_scanned": 0, "files_removed": 0, "errors": 0}

            # Get all document IDs from ChromaDB
            known_document_ids: Set[str] = set()
            if self.chroma_client:
                loop = asyncio.get_event_loop()
                results = await loop.run_in_executor(
                    self.chroma_client.executor,
                    lambda: self.chroma_client.collection.get(include=["metadatas"]),
                )

                for i, doc_id in enumerate(results["ids"]):
                    metadata = results["metadatas"][i]
                    document_id = metadata.get("document_id")
                    if document_id:
                        known_document_ids.add(document_id)

            # Scan upload directory
            stats = {"files_scanned": 0, "files_removed": 0, "errors": 0}

            for file_path in upload_dir.iterdir():
                if file_path.is_file() and file_path.name != ".gitkeep":
                    stats["files_scanned"] += 1

                    # Extract document ID from filename (format: {uuid}_{filename})
                    try:
                        parts = file_path.name.split("_", 1)
                        if len(parts) >= 1:
                            potential_doc_id = parts[0]

                            # Check if this document ID exists in the database
                            if potential_doc_id not in known_document_ids:
                                logger.info(f"Removing orphaned file: {file_path.name}")
                                file_path.unlink()
                                stats["files_removed"] += 1

                    except Exception as e:
                        logger.error(f"Error processing file {file_path.name}: {e}")
                        stats["errors"] += 1

            logger.info(
                f"Orphaned file cleanup completed. Scanned: {stats['files_scanned']}, "
                f"Removed: {stats['files_removed']}, Errors: {stats['errors']}"
            )

            return stats

        except Exception as e:
            logger.error(f"Failed to cleanup orphaned files: {e}")
            return {"files_scanned": 0, "files_removed": 0, "errors": 1}

    async def _remove_document_from_chroma(self, document_id: str):
        """Remove document and associated chunks from ChromaDB."""
        if not self.chroma_client:
            return

        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                self.chroma_client.executor,
                lambda: self.chroma_client.collection.delete(
                    where={"document_id": document_id}
                ),
            )
            logger.debug(f"Removed document {document_id} from ChromaDB")
        except Exception as e:
            logger.error(f"Failed to remove document {document_id} from ChromaDB: {e}")
            raise

    async def _remove_file_from_uploads(self, metadata: Dict[str, Any]):
        """Remove file from uploads directory based on metadata."""
        try:
            upload_dir = Path(
                getattr(settings, "upload_dir", None)
                or settings.upload_path
                or "uploads"
            )

            filename = metadata.get("filename")
            document_id = metadata.get("document_id")

            if filename and document_id:
                # Try to find the file with document_id prefix
                potential_files = list(upload_dir.glob(f"{document_id}_*"))
                for file_path in potential_files:
                    if file_path.exists():
                        file_path.unlink()
                        logger.debug(f"Removed file: {file_path.name}")
                        break

        except Exception as e:
            logger.error(f"Failed to remove file from uploads: {e}")


# Global service instance
_deduplication_service: Optional[DocumentDeduplicationService] = None


def get_deduplication_service(
    chroma_client: Optional[ChromaDBClient] = None,
) -> DocumentDeduplicationService:
    """Get the global deduplication service instance."""
    global _deduplication_service

    if _deduplication_service is None:
        _deduplication_service = DocumentDeduplicationService(chroma_client)

    return _deduplication_service
