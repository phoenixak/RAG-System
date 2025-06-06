"""
ChromaDB Client
Vector database client for document storage and similarity search.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import chromadb
from chromadb.config import Settings

from src.core.config import get_settings
from src.core.logging import LoggerMixin, get_logger

settings = get_settings()
logger = get_logger(__name__)


class ChromaDBClient(LoggerMixin):
    """ChromaDB client for vector operations."""

    def __init__(self):
        self.client = None
        self.collection = None
        self.executor = ThreadPoolExecutor(max_workers=4)
        self._initialize_client()

    def _initialize_client(self):
        """Initialize ChromaDB client and collection."""
        try:
            # Try HTTP client first, fallback to embedded mode for testing
            try:
                # Configure ChromaDB client
                chroma_settings = Settings(
                    chroma_server_host=settings.chromadb_host,
                    chroma_server_http_port=settings.chromadb_port,
                    chroma_server_cors_allow_origins=["*"],
                    allow_reset=True
                    if settings.environment == "development"
                    else False,
                )

                # Create HTTP client
                self.client = chromadb.HttpClient(
                    host=settings.chromadb_host,
                    port=settings.chromadb_port,
                    settings=chroma_settings,
                )

                # Test connection
                self.client.heartbeat()

                self.logger.info(
                    "ChromaDB HTTP client initialized successfully",
                    collection_name=settings.chromadb_collection,
                    host=settings.chromadb_host,
                    port=settings.chromadb_port,
                )

            except Exception as http_error:
                self.logger.warning(
                    "ChromaDB HTTP client failed, falling back to embedded mode",
                    error=str(http_error),
                )

                # Fallback to embedded client for testing
                self.client = chromadb.PersistentClient(
                    path="./chroma_db",
                    settings=Settings(
                        allow_reset=True
                        if settings.environment == "development"
                        else False,
                    ),
                )

                self.logger.info("ChromaDB embedded client initialized successfully")

            # Create or get collection
            self.collection = self.client.get_or_create_collection(
                name=settings.chromadb_collection,
                metadata={"description": "Enterprise RAG document embeddings"},
            )

        except Exception as e:
            self.logger.error("Failed to initialize ChromaDB client", error=str(e))
            raise ConnectionError(f"ChromaDB initialization failed: {e}")

    async def add_documents(
        self,
        documents: List[str],
        embeddings: List[List[float]],
        metadata: List[Dict[str, Any]],
        document_ids: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Add documents to the vector store.

        Args:
            documents: List of document texts
            embeddings: List of embedding vectors
            metadata: List of metadata dictionaries
            document_ids: Optional list of document IDs

        Returns:
            List of document IDs
        """
        try:
            # Generate IDs if not provided
            if document_ids is None:
                document_ids = [str(uuid4()) for _ in documents]

            # Validate input lengths
            if not (
                len(documents) == len(embeddings) == len(metadata) == len(document_ids)
            ):
                raise ValueError("All input lists must have the same length")

            # Add documents to collection (run in thread pool)
            # ChromaDB expects: embeddings, metadatas, documents, ids (as keyword args)
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                self.executor,
                lambda: self.collection.add(
                    embeddings=embeddings,
                    metadatas=metadata,
                    documents=documents,
                    ids=document_ids,
                ),
            )

            self.logger.info(
                "Documents added to vector store",
                count=len(documents),
                collection=settings.chromadb_collection,
            )

            return document_ids

        except Exception as e:
            self.logger.error(
                "Failed to add documents", error=str(e), count=len(documents)
            )
            raise

    async def search_similar(
        self,
        query_embedding: List[float],
        n_results: int = 10,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[str], List[Dict[str, Any]], List[float]]:
        """
        Search for similar documents.

        Args:
            query_embedding: Query embedding vector
            n_results: Number of results to return
            metadata_filter: Optional metadata filter

        Returns:
            Tuple of (document_texts, metadata_list, distances)
        """
        try:
            # Perform similarity search (run in thread pool)
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                self.executor,
                lambda: self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=n_results,
                    where=metadata_filter,
                    include=["documents", "metadatas", "distances"],
                ),
            )

            # Extract results
            documents = results["documents"][0] if results["documents"] else []
            metadata_list = results["metadatas"][0] if results["metadatas"] else []
            distances = results["distances"][0] if results["distances"] else []

            self.logger.debug(
                "Similarity search completed",
                query_results=len(documents),
                n_results=n_results,
            )

            return documents, metadata_list, distances

        except Exception as e:
            self.logger.error("Similarity search failed", error=str(e))
            raise

    async def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific document by ID for public API use.

        Args:
            document_id: Document ID

        Returns:
            Document data or None if not found
        """
        try:
            # Get document (run in thread pool)
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                self.executor,
                lambda: self.collection.get(
                    ids=[document_id], include=["documents", "metadatas", "embeddings"]
                ),
            )

            # Check if any documents were found
            # Use len() to avoid "truth value of array" error
            ids = results.get("ids", [])
            if len(ids) == 0:
                return None

            # Safely extract results
            documents = results.get("documents", [])
            metadatas = results.get("metadatas", [])
            embeddings = results.get("embeddings", [])

            return {
                "id": ids[0],
                "document": documents[0] if len(documents) > 0 else None,
                "metadata": metadatas[0] if len(metadatas) > 0 else None,
                "embedding": embeddings[0] if len(embeddings) > 0 else None,
            }

        except Exception as e:
            self.logger.error(
                "Failed to get document", error=str(e), document_id=document_id
            )
            raise

    async def get_document_metadata(self, document_id: str) -> Optional[Dict[str, Any]]:
        """
        Get document metadata only for public API endpoints.

        This method is used by API endpoints that only need metadata
        without internal processing information like file_path.

        Args:
            document_id: Document ID

        Returns:
            Document metadata only or None if not found
        """
        try:
            # Get document metadata only (run in thread pool)
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                self.executor,
                lambda: self.collection.get(ids=[document_id], include=["metadatas"]),
            )

            # Check if any documents were found
            ids = results.get("ids", [])
            if len(ids) == 0:
                return None

            # Extract metadata only
            metadatas = results.get("metadatas", [])
            if len(metadatas) > 0:
                # Return clean metadata without internal processing fields
                metadata = metadatas[0].copy()
                # Remove internal fields that shouldn't be exposed to API
                metadata.pop("file_path", None)
                metadata.pop("temp_file_path", None)
                return metadata

            return None

        except Exception as e:
            self.logger.error(
                "Failed to get document metadata", error=str(e), document_id=document_id
            )
            raise

    async def _get_document_metadata_internal(
        self, document_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get document metadata for internal processing.

        This method is used internally by document processing functions
        that need access to all metadata including file_path and other
        internal processing information.

        Args:
            document_id: Document ID

        Returns:
            Complete document metadata including internal fields or None if not found
        """
        try:
            # Get document metadata with all internal fields (run in thread pool)
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                self.executor,
                lambda: self.collection.get(ids=[document_id], include=["metadatas"]),
            )

            # Check if any documents were found
            ids = results.get("ids", [])
            if len(ids) == 0:
                return None

            # Extract complete metadata including internal fields
            metadatas = results.get("metadatas", [])
            return metadatas[0] if len(metadatas) > 0 else None

        except Exception as e:
            self.logger.error(
                "Failed to get internal document metadata",
                error=str(e),
                document_id=document_id,
            )
            raise

    async def update_document(
        self,
        document_id: str,
        document: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Update a document in the vector store.

        Args:
            document_id: Document ID
            document: Updated document text
            embedding: Updated embedding vector
            metadata: Updated metadata

        Returns:
            True if successful
        """
        try:
            # Prepare update data
            update_data = {}
            if document is not None:
                update_data["documents"] = [document]
            if embedding is not None:
                update_data["embeddings"] = [embedding]
            if metadata is not None:
                update_data["metadatas"] = [metadata]

            if not update_data:
                raise ValueError("At least one field must be provided for update")

            # Update document (run in thread pool)
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                self.executor,
                lambda: self.collection.update(ids=[document_id], **update_data),
            )

            self.logger.info("Document updated", document_id=document_id)
            return True

        except Exception as e:
            self.logger.error(
                "Failed to update document", error=str(e), document_id=document_id
            )
            raise

    async def delete_document(self, document_id: str) -> bool:
        """
        Delete a document from the vector store.

        Args:
            document_id: Document ID

        Returns:
            True if successful
        """
        try:
            # Delete document (run in thread pool)
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                self.executor, lambda: self.collection.delete(ids=[document_id])
            )

            self.logger.info("Document deleted", document_id=document_id)
            return True

        except Exception as e:
            self.logger.error(
                "Failed to delete document", error=str(e), document_id=document_id
            )
            raise

    async def delete_documents_by_filter(self, metadata_filter: Dict[str, Any]) -> int:
        """
        Delete documents by metadata filter.

        Args:
            metadata_filter: Metadata filter for deletion

        Returns:
            Number of documents deleted
        """
        try:
            # Get documents matching filter first
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                self.executor,
                lambda: self.collection.get(
                    where=metadata_filter, include=["metadatas"]
                ),
            )

            document_ids = results["ids"]
            count = len(document_ids)

            if count > 0:
                # Delete documents
                await loop.run_in_executor(
                    self.executor, lambda: self.collection.delete(ids=document_ids)
                )

            self.logger.info(
                "Documents deleted by filter", count=count, filter=metadata_filter
            )
            return count

        except Exception as e:
            self.logger.error(
                "Failed to delete documents by filter",
                error=str(e),
                filter=metadata_filter,
            )
            raise

    async def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get collection statistics.

        Returns:
            Collection statistics
        """
        try:
            # Get collection count (run in thread pool)
            loop = asyncio.get_event_loop()
            count = await loop.run_in_executor(self.executor, self.collection.count)

            return {
                "name": settings.chromadb_collection,
                "document_count": count,
                "embedding_dimensions": settings.embedding_dimensions,
            }

        except Exception as e:
            self.logger.error("Failed to get collection stats", error=str(e))
            raise

    async def health_check(self) -> bool:
        """
        Check if ChromaDB is healthy.

        Returns:
            True if healthy
        """
        try:
            # Simple health check by getting collection count
            await self.get_collection_stats()
            return True
        except Exception as e:
            self.logger.error("ChromaDB health check failed", error=str(e))
            return False

    def close(self):
        """Close the ChromaDB client and thread pool."""
        try:
            if self.executor:
                self.executor.shutdown(wait=True)
            self.logger.info("ChromaDB client closed")
        except Exception as e:
            self.logger.error("Error closing ChromaDB client", error=str(e))


# Global ChromaDB client instance
_chroma_client = None


def get_chroma_client() -> ChromaDBClient:
    """Get the global ChromaDB client instance."""
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = ChromaDBClient()
    return _chroma_client


async def close_chroma_client():
    """Close the global ChromaDB client."""
    global _chroma_client
    if _chroma_client:
        _chroma_client.close()
        _chroma_client = None
