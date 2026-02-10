"""
Test Configuration and Fixtures
Common fixtures and configuration for all tests.
"""

import os
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

# Configure test environment BEFORE importing any project modules
os.environ["ENVIRONMENT"] = "testing"
os.environ["SECRET_KEY"] = "test-secret-key-for-testing-only-minimum-32-characters-long"
os.environ["DATABASE_URL"] = "sqlite:///:memory:"
os.environ["CHROMADB_HOST"] = "localhost"
os.environ["CHROMADB_PORT"] = "8000"
os.environ["LOG_LEVEL"] = "WARNING"
os.environ["ADMIN_EMAIL"] = "admin@example.com"
os.environ["ADMIN_PASSWORD"] = "admin123!"
os.environ["USER_EMAIL"] = "user@example.com"
os.environ["USER_PASSWORD"] = "password123!"


@pytest.fixture(scope="session")
def test_config():
    """Test configuration fixture."""
    return {
        "secret_key": os.environ["SECRET_KEY"],
        "environment": "testing",
        "debug": True,
        "api_host": "127.0.0.1",
        "api_port": 8000,
    }


@pytest.fixture
def test_user_data():
    """Sample test user data fixture."""
    return {
        "user_id": str(uuid.uuid4()),
        "email": "test@example.com",
        "first_name": "Test",
        "last_name": "User",
        "role": "standard_user",
        "is_active": True,
        "created_at": datetime.now(timezone.utc),
    }


@pytest.fixture
def admin_user_data():
    """Sample admin user data fixture."""
    return {
        "user_id": str(uuid.uuid4()),
        "email": "admin@example.com",
        "first_name": "Admin",
        "last_name": "User",
        "role": "admin",
        "is_active": True,
        "created_at": datetime.now(timezone.utc),
    }


@pytest.fixture
def temp_upload_dir():
    """Temporary directory for test file uploads."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_text_content():
    """Sample text content for document processing tests."""
    return (
        "Machine learning is a method of data analysis that automates analytical model building. "
        "It is a branch of artificial intelligence based on the idea that systems can learn from data, "
        "identify patterns and make decisions with minimal human intervention.\n\n"
        "Deep learning is a subset of machine learning that uses neural networks with many layers. "
        "These deep neural networks are capable of learning complex patterns in large amounts of data."
    )


@pytest.fixture
def sample_chunks_data():
    """Sample document chunk data for testing."""
    doc_id = uuid.uuid4()
    return [
        {
            "chunk_id": str(uuid.uuid4()),
            "document_id": str(doc_id),
            "text": "Machine learning is a method of data analysis.",
            "chunk_index": 0,
            "token_count": 10,
            "start_char": 0,
            "end_char": 47,
            "metadata": {"page_number": 1, "section": "introduction"},
        },
        {
            "chunk_id": str(uuid.uuid4()),
            "document_id": str(doc_id),
            "text": "Artificial intelligence encompasses various technologies.",
            "chunk_index": 1,
            "token_count": 8,
            "start_char": 48,
            "end_char": 105,
            "metadata": {"page_number": 2, "section": "overview"},
        },
    ]


@pytest.fixture
def mock_chroma_client():
    """Mock ChromaDB client matching ChromaDBClient API."""
    client = MagicMock()
    client.add_documents = AsyncMock(return_value=["id1", "id2"])
    client.search_similar = AsyncMock(return_value=([], [], []))
    client.get_document = AsyncMock(return_value=None)
    client.get_document_metadata = AsyncMock(return_value=None)
    client.update_document = AsyncMock(return_value=True)
    client.delete_document = AsyncMock(return_value=True)
    client.delete_documents_by_filter = AsyncMock(return_value=0)
    client.get_collection_stats = AsyncMock(
        return_value={"document_count": 0, "collection_name": "test"}
    )
    client.health_check = AsyncMock(return_value=True)
    client.close = Mock()
    return client


@pytest.fixture
def mock_embedding_generator():
    """Mock embedding generator matching EmbeddingGenerator API."""
    generator = MagicMock()
    generator.generate_embedding = AsyncMock(return_value=[0.1] * 384)
    generator.generate_embeddings_batch = AsyncMock(return_value=[[0.1] * 384])
    generator.generate_chunk_embeddings = AsyncMock()
    generator.get_model_info = Mock(
        return_value={
            "model_name": "test-model",
            "device": "cpu",
            "max_seq_length": 256,
            "embedding_dimension": 384,
            "batch_size": 32,
            "cache_enabled": True,
            "cache_stats": {"size": 0, "max_size": 10000},
        }
    )
    generator.clear_cache = Mock()
    generator.close = Mock()
    return generator


@pytest.fixture
def mock_llm_service():
    """Mock LLM service matching LLMService API."""
    service = MagicMock()
    service.generate_rag_response = AsyncMock(
        return_value="This is a test RAG response."
    )
    service.test_connection = AsyncMock(
        return_value={"success": True, "provider": "openai"}
    )
    service.is_api_available = Mock(return_value=True)
    return service


@pytest.fixture
def sample_search_results():
    """Sample search results for testing."""
    from src.search.models import SearchResult

    return [
        SearchResult(
            chunk_id=uuid.uuid4(),
            document_id=uuid.uuid4(),
            text="This is a relevant document about machine learning.",
            chunk_index=0,
            score=0.85,
            semantic_score=0.85,
            document_filename="ml_guide.pdf",
            document_type="pdf",
            metadata={"title": "ML Guide", "page_number": 1},
        ),
        SearchResult(
            chunk_id=uuid.uuid4(),
            document_id=uuid.uuid4(),
            text="Another document discussing artificial intelligence concepts.",
            chunk_index=1,
            score=0.78,
            semantic_score=0.78,
            document_filename="ai_intro.pdf",
            document_type="pdf",
            metadata={"title": "AI Introduction", "page_number": 2},
        ),
    ]


@pytest.fixture
def sample_conversation_history():
    """Sample conversation history for testing."""
    return [
        {"role": "user", "content": "What is machine learning?"},
        {
            "role": "assistant",
            "content": "Machine learning is a subset of AI that enables computers to learn from data.",
        },
        {"role": "user", "content": "How does it work?"},
    ]


@pytest.fixture
def mock_file_content():
    """Mock file content bytes for upload testing."""
    return b"This is test file content for document processing and upload testing."


@pytest.fixture(autouse=True)
def cleanup_test_data():
    """Automatically cleanup test data after each test."""
    yield
