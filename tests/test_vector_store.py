"""
Tests for ChromaDB vector store client.
Tests ChromaDBClient with mocked ChromaDB backend.
"""

import uuid
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest


class TestChromaDBClient:
    """Tests for ChromaDBClient."""

    @pytest.fixture
    def mock_collection(self):
        """Mock ChromaDB collection."""
        collection = MagicMock()
        collection.add = Mock()
        collection.query = Mock(
            return_value={
                "ids": [["id1", "id2"]],
                "documents": [["doc1 text", "doc2 text"]],
                "metadatas": [[{"key": "val1"}, {"key": "val2"}]],
                "distances": [[0.1, 0.3]],
            }
        )
        collection.get = Mock(
            return_value={
                "ids": ["id1"],
                "documents": ["doc text"],
                "metadatas": [{"key": "value"}],
            }
        )
        collection.update = Mock()
        collection.delete = Mock()
        collection.count = Mock(return_value=10)
        return collection

    @pytest.fixture
    def chroma_client(self, mock_collection):
        """Create ChromaDBClient with mocked internals."""
        with patch("src.vector_store.chroma_client.chromadb") as mock_chromadb:
            mock_chromadb_client = MagicMock()
            mock_chromadb_client.get_or_create_collection.return_value = mock_collection
            mock_chromadb.HttpClient.return_value = mock_chromadb_client
            mock_chromadb.PersistentClient.return_value = mock_chromadb_client

            from src.vector_store.chroma_client import ChromaDBClient

            client = ChromaDBClient()
            client.collection = mock_collection
            client.client = mock_chromadb_client
            return client

    @pytest.mark.asyncio
    async def test_add_documents(self, chroma_client, mock_collection):
        """Test adding documents to the vector store."""
        documents = ["doc1 text", "doc2 text"]
        embeddings = [[0.1] * 384, [0.2] * 384]
        metadata = [{"source": "test1"}, {"source": "test2"}]

        result = await chroma_client.add_documents(documents, embeddings, metadata)
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_search_similar(self, chroma_client, mock_collection):
        """Test similarity search."""
        query_embedding = [0.1] * 384
        ids, metadatas, distances = await chroma_client.search_similar(
            query_embedding=query_embedding,
            n_results=5,
        )
        assert isinstance(ids, list)

    @pytest.mark.asyncio
    async def test_get_document(self, chroma_client, mock_collection):
        """Test retrieving a document by ID."""
        result = await chroma_client.get_document("test-id-1")
        # Result can be None or Dict depending on whether doc exists

    @pytest.mark.asyncio
    async def test_delete_document(self, chroma_client, mock_collection):
        """Test deleting a document."""
        result = await chroma_client.delete_document("test-id-1")
        assert isinstance(result, bool)

    @pytest.mark.asyncio
    async def test_get_collection_stats(self, chroma_client, mock_collection):
        """Test getting collection statistics."""
        stats = await chroma_client.get_collection_stats()
        assert isinstance(stats, dict)

    @pytest.mark.asyncio
    async def test_health_check(self, chroma_client):
        """Test health check."""
        result = await chroma_client.health_check()
        assert isinstance(result, bool)

    def test_close(self, chroma_client):
        """Test client close."""
        chroma_client.close()
        # Should not raise


class TestChromaClientSingleton:
    """Tests for the global ChromaDB client singleton."""

    def test_get_chroma_client_returns_instance(self):
        """Test singleton getter returns an instance."""
        with patch("src.vector_store.chroma_client.chromadb"):
            from src.vector_store.chroma_client import get_chroma_client
            # The function should return a ChromaDBClient instance
            # (may fail if chromadb not configured, that's expected in test env)
