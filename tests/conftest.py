"""
Test Configuration and Fixtures
Common fixtures and configuration for all tests.
"""

import os
import tempfile
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi.testclient import TestClient

# Configure test environment
os.environ["ENVIRONMENT"] = "testing"
os.environ["SECRET_KEY"] = "test-secret-key-for-testing-only-32-chars"
os.environ["DATABASE_URL"] = "sqlite:///:memory:"
os.environ["CHROMADB_HOST"] = "localhost"
os.environ["CHROMADB_PORT"] = "8000"
os.environ["LOG_LEVEL"] = "WARNING"  # Reduce logging noise during tests


@pytest.fixture(scope="session")
def test_config():
    """Test configuration fixture."""
    return {
        "secret_key": "test-secret-key-for-testing-only-32-chars",
        "environment": "testing",
        "debug": True,
        "api_host": "127.0.0.1",
        "api_port": 8000,
    }


@pytest.fixture
def test_user():
    """Sample test user fixture."""
    return {
        "user_id": str(uuid.uuid4()),
        "email": "test@example.com",
        "username": "testuser",
        "role": "standard_user",
        "is_active": True,
        "created_at": datetime.utcnow(),
    }


@pytest.fixture
def admin_user():
    """Sample admin user fixture."""
    return {
        "user_id": str(uuid.uuid4()),
        "email": "admin@example.com",
        "username": "admin",
        "role": "admin",
        "is_active": True,
        "created_at": datetime.utcnow(),
    }


@pytest.fixture
def sample_jwt_token():
    """Sample JWT token for testing."""
    return "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidGVzdF91c2VyIiwiZXhwIjoxNjcwMDAwMDAwfQ.test_signature"


@pytest.fixture
def temp_upload_dir():
    """Temporary directory for test file uploads."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_documents():
    """Sample document data for testing."""
    return [
        {
            "id": str(uuid.uuid4()),
            "filename": "sample_doc.pdf",
            "file_type": "pdf",
            "file_size": 1024,
            "content": "This is a sample document about machine learning and artificial intelligence.",
            "metadata": {"title": "ML Introduction", "author": "Test Author"},
            "uploaded_by": "test_user",
            "upload_date": datetime.utcnow(),
        },
        {
            "id": str(uuid.uuid4()),
            "filename": "technical_spec.docx",
            "file_type": "docx",
            "file_size": 2048,
            "content": "Technical specification document for enterprise systems.",
            "metadata": {"title": "Tech Spec", "department": "engineering"},
            "uploaded_by": "test_user",
            "upload_date": datetime.utcnow(),
        },
    ]


@pytest.fixture
def sample_chunks():
    """Sample document chunks for testing."""
    return [
        {
            "chunk_id": str(uuid.uuid4()),
            "document_id": str(uuid.uuid4()),
            "text": "Machine learning is a method of data analysis.",
            "chunk_index": 0,
            "embeddings": [0.1, 0.2, 0.3, 0.4, 0.5],  # Mock embeddings
            "metadata": {"page_number": 1, "section": "introduction"},
        },
        {
            "chunk_id": str(uuid.uuid4()),
            "document_id": str(uuid.uuid4()),
            "text": "Artificial intelligence encompasses various technologies.",
            "chunk_index": 1,
            "embeddings": [0.2, 0.3, 0.4, 0.5, 0.6],  # Mock embeddings
            "metadata": {"page_number": 2, "section": "overview"},
        },
    ]


@pytest.fixture
def mock_chroma_client():
    """Mock ChromaDB client."""
    client = Mock()
    client.health_check = AsyncMock(return_value=True)
    client.create_collection = AsyncMock()
    client.get_collection = AsyncMock()
    client.add_documents = AsyncMock()
    client.query_documents = AsyncMock(return_value={"documents": [], "distances": [], "metadatas": []})
    client.get_collection_info = AsyncMock(return_value={"count": 0})
    client.delete_documents = AsyncMock()
    return client


@pytest.fixture
def mock_embedding_generator():
    """Mock embedding generator."""
    generator = Mock()
    generator.generate_embeddings = AsyncMock(return_value=[[0.1, 0.2, 0.3, 0.4, 0.5]])
    generator.get_model_info = Mock(return_value={"model_name": "test-model", "dimensions": 384})
    generator.get_embedding_dimension = Mock(return_value=384)
    return generator


@pytest.fixture
def mock_llm_service():
    """Mock LLM service."""
    service = Mock()
    service.generate_rag_response = AsyncMock(return_value="This is a test response.")
    service.test_connection = AsyncMock(return_value={"success": True, "provider": "openai"})
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
        {"role": "assistant", "content": "Machine learning is a subset of AI that enables computers to learn from data."},
        {"role": "user", "content": "How does it work?"},
    ]


@pytest.fixture
def mock_file_upload():
    """Mock file upload for testing."""
    content = b"This is test file content for document processing."
    
    class MockUploadFile:
        def __init__(self, filename: str, content: bytes):
            self.filename = filename
            self.content_type = "application/pdf"
            self.size = len(content)
            self._content = content
            
        async def read(self) -> bytes:
            return self._content
            
        async def seek(self, position: int = 0):
            pass
            
        def __aiter__(self):
            yield self._content
    
    return MockUploadFile("test_document.pdf", content)


# Test client fixtures
@pytest.fixture
def test_client():
    """Test client for API testing."""
    from src.main import create_app
    
    app = create_app()
    return TestClient(app)


@pytest.fixture
def authenticated_client(test_client, sample_jwt_token):
    """Authenticated test client."""
    test_client.headers.update({"Authorization": f"Bearer {sample_jwt_token}"})
    return test_client


# Database fixtures
@pytest.fixture
def test_db():
    """Test database fixture."""
    # For now, return a mock since we're using ChromaDB primarily
    return Mock()


# Performance testing fixtures
@pytest.fixture
def performance_timer():
    """Timer for performance testing."""
    import time
    
    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None
        
        def start(self):
            self.start_time = time.time()
        
        def stop(self):
            self.end_time = time.time()
        
        @property
        def elapsed(self):
            if self.start_time and self.end_time:
                return self.end_time - self.start_time
            return None
    
    return Timer()


# Error simulation fixtures
@pytest.fixture
def mock_service_error():
    """Mock service error for testing error handling."""
    return Exception("Simulated service error for testing")


@pytest.fixture
def mock_network_error():
    """Mock network error for testing connectivity issues."""
    from requests.exceptions import ConnectionError
    return ConnectionError("Simulated network error")


# Test data cleanup
@pytest.fixture(autouse=True)
def cleanup_test_data():
    """Automatically cleanup test data after each test."""
    yield
    # Cleanup code would go here if needed
    pass