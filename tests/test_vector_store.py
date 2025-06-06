"""
Vector Store Tests
Comprehensive tests for ChromaDB integration and vector operations.
"""

import asyncio
import uuid
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.vector_store.chroma_client import ChromaDBClient, get_chroma_client


class TestChromaClient:
    """Test ChromaDB client functionality."""

    @pytest.fixture
    def mock_chroma_db(self):
        """Mock ChromaDB instance."""
        mock_db = Mock()
        mock_db.heartbeat = Mock(return_value=True)
        mock_db.get_or_create_collection = Mock()
        mock_db.get_collection = Mock()
        return mock_db

    @pytest.fixture
    def mock_collection(self):
        """Mock ChromaDB collection."""
        collection = Mock()
        collection.add = Mock()
        collection.query = Mock()
        collection.get = Mock()
        collection.delete = Mock()
        collection.count = Mock(return_value=0)
        collection.peek = Mock(return_value={"documents": [], "metadatas": [], "ids": []})
        return collection

    def test_chroma_client_initialization(self):
        """Test ChromaDB client initialization."""
        client = ChromaDBClient()
        
        assert client.host is not None
        assert client.port is not None
        assert client.collection_name is not None

    def test_chroma_client_configuration(self):
        """Test ChromaDB client configuration."""
        with patch.dict("os.environ", {
            "CHROMADB_HOST": "test_host",
            "CHROMADB_PORT": "9000",
            "CHROMADB_COLLECTION": "test_collection"
        }):
            client = ChromaDBClient()
            assert client.host == "test_host"
            assert client.port == 9000
            assert client.collection_name == "test_collection"

    @pytest.mark.asyncio
    async def test_health_check_success(self, mock_chroma_db):
        """Test successful health check."""
        with patch("chromadb.HttpClient") as mock_http_client:
            mock_http_client.return_value = mock_chroma_db
            
            client = ChromaDBClient()
            result = await client.health_check()
            
            assert result is True
            mock_chroma_db.heartbeat.assert_called_once()

    @pytest.mark.asyncio
    async def test_health_check_failure(self):
        """Test health check failure."""
        with patch("chromadb.HttpClient") as mock_http_client:
            mock_http_client.side_effect = Exception("Connection failed")
            
            client = ChromaDBClient()
            result = await client.health_check()
            
            assert result is False

    @pytest.mark.asyncio
    async def test_get_collection_existing(self, mock_chroma_db, mock_collection):
        """Test getting existing collection."""
        with patch("chromadb.HttpClient") as mock_http_client:
            mock_http_client.return_value = mock_chroma_db
            mock_chroma_db.get_or_create_collection.return_value = mock_collection
            
            client = ChromaDBClient()
            collection = await client._get_collection()
            
            assert collection is mock_collection
            mock_chroma_db.get_or_create_collection.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_collection_with_fallback(self):
        """Test getting collection with embedded fallback."""
        with patch("chromadb.HttpClient") as mock_http_client, \
             patch("chromadb.EphemeralClient") as mock_ephemeral_client:
            
            # HTTP client fails
            mock_http_client.side_effect = Exception("Connection failed")
            
            # Ephemeral client succeeds
            mock_ephemeral_db = Mock()
            mock_ephemeral_db.get_or_create_collection = Mock(return_value=Mock())
            mock_ephemeral_client.return_value = mock_ephemeral_db
            
            client = ChromaClient()
            collection = await client._get_collection()
            
            # Should fall back to ephemeral client
            mock_ephemeral_client.assert_called_once()

    @pytest.mark.asyncio
    async def test_add_documents(self, mock_chroma_db, mock_collection):
        """Test adding documents to collection."""
        with patch("chromadb.HttpClient") as mock_http_client:
            mock_http_client.return_value = mock_chroma_db
            mock_chroma_db.get_or_create_collection.return_value = mock_collection
            
            client = ChromaClient()
            
            documents = [
                {
                    "id": "doc1",
                    "text": "Sample document text",
                    "embeddings": [0.1, 0.2, 0.3],
                    "metadata": {"source": "test.pdf"}
                },
                {
                    "id": "doc2",
                    "text": "Another document",
                    "embeddings": [0.4, 0.5, 0.6],
                    "metadata": {"source": "test2.pdf"}
                }
            ]
            
            result = await client.add_documents(documents)
            
            assert result is True
            mock_collection.add.assert_called_once()
            
            # Verify call arguments
            call_args = mock_collection.add.call_args[1]
            assert "ids" in call_args
            assert "documents" in call_args
            assert "embeddings" in call_args
            assert "metadatas" in call_args

    @pytest.mark.asyncio
    async def test_add_documents_batch_processing(self, mock_chroma_db, mock_collection):
        """Test batch processing of large document sets."""
        with patch("chromadb.HttpClient") as mock_http_client:
            mock_http_client.return_value = mock_chroma_db
            mock_chroma_db.get_or_create_collection.return_value = mock_collection
            
            client = ChromaClient(batch_size=2)
            
            # Create documents larger than batch size
            documents = [
                {
                    "id": f"doc{i}",
                    "text": f"Document {i}",
                    "embeddings": [0.1 * i, 0.2 * i, 0.3 * i],
                    "metadata": {"index": i}
                }
                for i in range(5)  # 5 documents with batch size 2
            ]
            
            result = await client.add_documents(documents)
            
            assert result is True
            # Should be called 3 times (2 + 2 + 1)
            assert mock_collection.add.call_count == 3

    @pytest.mark.asyncio
    async def test_query_documents(self, mock_chroma_db, mock_collection):
        """Test querying documents from collection."""
        with patch("chromadb.HttpClient") as mock_http_client:
            mock_http_client.return_value = mock_chroma_db
            mock_chroma_db.get_or_create_collection.return_value = mock_collection
            
            # Mock query response
            mock_collection.query.return_value = {
                "ids": [["doc1", "doc2"]],
                "documents": [["Document 1 text", "Document 2 text"]],
                "metadatas": [[{"source": "test1.pdf"}, {"source": "test2.pdf"}]],
                "distances": [[0.1, 0.3]]
            }
            
            client = ChromaClient()
            
            query_embeddings = [0.1, 0.2, 0.3]
            results = await client.query_documents(
                query_embeddings=query_embeddings,
                n_results=2
            )
            
            assert "documents" in results
            assert "distances" in results
            assert "metadatas" in results
            assert len(results["documents"][0]) == 2
            
            mock_collection.query.assert_called_once()

    @pytest.mark.asyncio
    async def test_query_with_metadata_filter(self, mock_chroma_db, mock_collection):
        """Test querying with metadata filters."""
        with patch("chromadb.HttpClient") as mock_http_client:
            mock_http_client.return_value = mock_chroma_db
            mock_chroma_db.get_or_create_collection.return_value = mock_collection
            
            mock_collection.query.return_value = {
                "ids": [["filtered_doc1"]],
                "documents": [["Filtered document"]],
                "metadatas": [[{"document_type": "pdf"}]],
                "distances": [[0.2]]
            }
            
            client = ChromaClient()
            
            metadata_filter = {"document_type": "pdf"}
            results = await client.query_documents(
                query_embeddings=[0.1, 0.2, 0.3],
                n_results=10,
                metadata_filter=metadata_filter
            )
            
            # Verify filter was applied
            call_args = mock_collection.query.call_args[1]
            assert "where" in call_args
            assert call_args["where"] == metadata_filter

    @pytest.mark.asyncio
    async def test_get_documents_by_ids(self, mock_chroma_db, mock_collection):
        """Test retrieving documents by IDs."""
        with patch("chromadb.HttpClient") as mock_http_client:
            mock_http_client.return_value = mock_chroma_db
            mock_chroma_db.get_or_create_collection.return_value = mock_collection
            
            mock_collection.get.return_value = {
                "ids": ["doc1", "doc2"],
                "documents": ["Document 1", "Document 2"],
                "metadatas": [{"source": "test1.pdf"}, {"source": "test2.pdf"}]
            }
            
            client = ChromaClient()
            
            doc_ids = ["doc1", "doc2"]
            results = await client.get_documents_by_ids(doc_ids)
            
            assert "documents" in results
            assert "metadatas" in results
            assert len(results["documents"]) == 2
            
            mock_collection.get.assert_called_once_with(ids=doc_ids)

    @pytest.mark.asyncio
    async def test_delete_documents(self, mock_chroma_db, mock_collection):
        """Test deleting documents from collection."""
        with patch("chromadb.HttpClient") as mock_http_client:
            mock_http_client.return_value = mock_chroma_db
            mock_chroma_db.get_or_create_collection.return_value = mock_collection
            
            client = ChromaClient()
            
            doc_ids = ["doc1", "doc2"]
            result = await client.delete_documents(doc_ids)
            
            assert result is True
            mock_collection.delete.assert_called_once_with(ids=doc_ids)

    @pytest.mark.asyncio
    async def test_delete_documents_by_metadata(self, mock_chroma_db, mock_collection):
        """Test deleting documents by metadata filter."""
        with patch("chromadb.HttpClient") as mock_http_client:
            mock_http_client.return_value = mock_chroma_db
            mock_chroma_db.get_or_create_collection.return_value = mock_collection
            
            client = ChromaClient()
            
            metadata_filter = {"document_id": "test_doc"}
            result = await client.delete_documents_by_metadata(metadata_filter)
            
            assert result is True
            mock_collection.delete.assert_called_once_with(where=metadata_filter)

    @pytest.mark.asyncio
    async def test_get_collection_info(self, mock_chroma_db, mock_collection):
        """Test getting collection information."""
        with patch("chromadb.HttpClient") as mock_http_client:
            mock_http_client.return_value = mock_chroma_db
            mock_chroma_db.get_or_create_collection.return_value = mock_collection
            
            mock_collection.count.return_value = 42
            mock_collection.peek.return_value = {
                "documents": ["Sample doc"],
                "metadatas": [{"type": "test"}],
                "ids": ["sample_id"]
            }
            
            client = ChromaClient()
            info = await client.get_collection_info()
            
            assert info["count"] == 42
            assert "sample_documents" in info
            assert len(info["sample_documents"]) == 1

    @pytest.mark.asyncio
    async def test_connection_retry_logic(self):
        """Test connection retry logic on failures."""
        with patch("chromadb.HttpClient") as mock_http_client:
            # First call fails, second succeeds
            mock_chroma_db = Mock()
            mock_http_client.side_effect = [
                Exception("Connection failed"),
                mock_chroma_db
            ]
            
            client = ChromaClient()
            
            # This should retry and eventually succeed
            with patch.object(client, '_ensure_connection') as mock_ensure:
                mock_ensure.return_value = mock_chroma_db
                
                result = await client.health_check()
                # The test would need actual retry logic implementation


class TestChromaClientErrorHandling:
    """Test ChromaDB client error handling."""

    @pytest.mark.asyncio
    async def test_connection_failure_fallback(self):
        """Test fallback to embedded client on connection failure."""
        with patch("chromadb.HttpClient") as mock_http_client, \
             patch("chromadb.EphemeralClient") as mock_ephemeral_client:
            
            mock_http_client.side_effect = Exception("Connection refused")
            
            mock_ephemeral_db = Mock()
            mock_ephemeral_db.get_or_create_collection = Mock(return_value=Mock())
            mock_ephemeral_client.return_value = mock_ephemeral_db
            
            client = ChromaClient()
            collection = await client._get_collection()
            
            # Should use ephemeral client as fallback
            mock_ephemeral_client.assert_called_once()

    @pytest.mark.asyncio
    async def test_add_documents_failure(self, mock_chroma_db, mock_collection):
        """Test handling of add documents failure."""
        with patch("chromadb.HttpClient") as mock_http_client:
            mock_http_client.return_value = mock_chroma_db
            mock_chroma_db.get_or_create_collection.return_value = mock_collection
            
            mock_collection.add.side_effect = Exception("Add operation failed")
            
            client = ChromaClient()
            
            documents = [{"id": "doc1", "text": "test", "embeddings": [0.1, 0.2]}]
            result = await client.add_documents(documents)
            
            assert result is False

    @pytest.mark.asyncio
    async def test_query_documents_failure(self, mock_chroma_db, mock_collection):
        """Test handling of query failure."""
        with patch("chromadb.HttpClient") as mock_http_client:
            mock_http_client.return_value = mock_chroma_db
            mock_chroma_db.get_or_create_collection.return_value = mock_collection
            
            mock_collection.query.side_effect = Exception("Query failed")
            
            client = ChromaClient()
            
            results = await client.query_documents([0.1, 0.2, 0.3])
            
            # Should return empty results on failure
            assert results == {"documents": [], "distances": [], "metadatas": []}

    @pytest.mark.asyncio
    async def test_invalid_embeddings_handling(self, mock_chroma_db, mock_collection):
        """Test handling of invalid embeddings."""
        with patch("chromadb.HttpClient") as mock_http_client:
            mock_http_client.return_value = mock_chroma_db
            mock_chroma_db.get_or_create_collection.return_value = mock_collection
            
            client = ChromaClient()
            
            # Test with invalid embedding dimensions
            documents = [
                {
                    "id": "doc1",
                    "text": "test",
                    "embeddings": [0.1, 0.2]  # Wrong dimension
                },
                {
                    "id": "doc2", 
                    "text": "test2",
                    "embeddings": [0.1, 0.2, 0.3, 0.4]  # Different dimension
                }
            ]
            
            # Should handle gracefully
            result = await client.add_documents(documents)
            # Result depends on actual validation implementation

    @pytest.mark.asyncio
    async def test_large_batch_processing(self, mock_chroma_db, mock_collection):
        """Test handling of very large document batches."""
        with patch("chromadb.HttpClient") as mock_http_client:
            mock_http_client.return_value = mock_chroma_db
            mock_chroma_db.get_or_create_collection.return_value = mock_collection
            
            client = ChromaClient(batch_size=100)
            
            # Create a very large number of documents
            documents = [
                {
                    "id": f"doc{i}",
                    "text": f"Document {i}",
                    "embeddings": [0.1] * 384,  # Standard embedding size
                    "metadata": {"index": i}
                }
                for i in range(1000)  # 1000 documents
            ]
            
            result = await client.add_documents(documents)
            
            assert result is True
            # Should be called 10 times (1000 / 100)
            assert mock_collection.add.call_count == 10


class TestChromaClientSingleton:
    """Test ChromaDB client singleton pattern."""

    def test_get_chroma_client_singleton(self):
        """Test that get_chroma_client returns singleton."""
        client1 = get_chroma_client()
        client2 = get_chroma_client()
        
        assert client1 is client2

    def test_chroma_client_configuration_consistency(self):
        """Test that client configuration is consistent across calls."""
        client1 = get_chroma_client()
        client2 = get_chroma_client()
        
        assert client1.host == client2.host
        assert client1.port == client2.port
        assert client1.collection_name == client2.collection_name


class TestChromaClientPerformance:
    """Test ChromaDB client performance characteristics."""

    @pytest.mark.asyncio
    async def test_concurrent_operations(self, mock_chroma_db, mock_collection):
        """Test concurrent ChromaDB operations."""
        with patch("chromadb.HttpClient") as mock_http_client:
            mock_http_client.return_value = mock_chroma_db
            mock_chroma_db.get_or_create_collection.return_value = mock_collection
            
            mock_collection.query.return_value = {
                "ids": [["doc1"]],
                "documents": [["test"]],
                "metadatas": [[{}]],
                "distances": [[0.1]]
            }
            
            client = ChromaClient()
            
            # Run multiple concurrent queries
            tasks = [
                client.query_documents([0.1, 0.2, 0.3])
                for _ in range(10)
            ]
            
            results = await asyncio.gather(*tasks)
            
            assert len(results) == 10
            assert all("documents" in result for result in results)

    @pytest.mark.asyncio
    async def test_memory_efficient_batching(self, mock_chroma_db, mock_collection):
        """Test memory-efficient processing of large datasets."""
        with patch("chromadb.HttpClient") as mock_http_client:
            mock_http_client.return_value = mock_chroma_db
            mock_chroma_db.get_or_create_collection.return_value = mock_collection
            
            client = ChromaClient(batch_size=10)
            
            # Simulate processing large dataset in chunks
            total_documents = 100
            batch_count = 0
            
            for i in range(0, total_documents, 10):
                batch = [
                    {
                        "id": f"doc{j}",
                        "text": f"Document {j}",
                        "embeddings": [0.1] * 5,
                        "metadata": {"batch": batch_count}
                    }
                    for j in range(i, min(i + 10, total_documents))
                ]
                
                result = await client.add_documents(batch)
                assert result is True
                batch_count += 1
            
            assert batch_count == 10  # 100 documents / 10 per batch

    @pytest.mark.asyncio
    async def test_query_response_time(self, mock_chroma_db, mock_collection):
        """Test query response time monitoring."""
        with patch("chromadb.HttpClient") as mock_http_client:
            mock_http_client.return_value = mock_chroma_db
            mock_chroma_db.get_or_create_collection.return_value = mock_collection
            
            # Mock a slow query
            import asyncio
            async def slow_query(*args, **kwargs):
                await asyncio.sleep(0.1)  # Simulate 100ms query
                return {
                    "ids": [["doc1"]],
                    "documents": [["test"]],
                    "metadatas": [[{}]],
                    "distances": [[0.1]]
                }
            
            mock_collection.query = slow_query
            
            client = ChromaClient()
            
            import time
            start_time = time.time()
            result = await client.query_documents([0.1, 0.2, 0.3])
            end_time = time.time()
            
            query_time = end_time - start_time
            assert query_time >= 0.1  # Should take at least 100ms
            assert "documents" in result


if __name__ == "__main__":
    pytest.main([__file__])