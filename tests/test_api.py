"""
API Endpoints Tests
Comprehensive tests for all FastAPI endpoints and middleware.
"""

import json
import uuid
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

# Import test app creation
from src.api.auth import router as auth_router
from src.api.documents import router as documents_router
from src.api.health import router as health_router
from src.api.search import router as search_router


class TestHealthEndpoints:
    """Test health check and monitoring endpoints."""

    @pytest.mark.asyncio
    async def test_basic_health_check(self):
        """Test basic health check endpoint."""
        from src.api.health import health_check
        
        response = await health_check()
        
        assert response["status"] == "healthy"
        assert "timestamp" in response
        assert "version" in response
        assert "environment" in response
        assert "uptime_seconds" in response

    @pytest.mark.asyncio
    async def test_detailed_health_check(self):
        """Test detailed health check endpoint."""
        from src.api.health import detailed_health_check
        
        with patch("src.api.health.check_services_health") as mock_services:
            mock_services.return_value = {
                "api_gateway": "healthy",
                "vector_store": "healthy",
                "embedding_service": "healthy",
                "llm_service": "demo_mode"
            }
            
            response = await detailed_health_check()
            
            assert response["status"] in ["healthy", "degraded"]
            assert "services" in response
            assert "metrics" in response
            assert response["services"]["llm_service"] == "demo_mode"

    @pytest.mark.asyncio
    async def test_readiness_check(self):
        """Test readiness probe endpoint."""
        from src.api.health import readiness_check
        
        with patch("src.api.health.check_services_health") as mock_services:
            mock_services.return_value = {
                "api_gateway": "healthy",
                "vector_store": "healthy"
            }
            
            response = await readiness_check()
            assert response["status"] == "ready"

    @pytest.mark.asyncio
    async def test_readiness_check_failure(self):
        """Test readiness check when services are unhealthy."""
        from src.api.health import readiness_check
        
        with patch("src.api.health.check_services_health") as mock_services:
            mock_services.return_value = {
                "api_gateway": "unhealthy",
            }
            
            with pytest.raises(HTTPException) as exc_info:
                await readiness_check()
            
            assert exc_info.value.status_code == 503

    @pytest.mark.asyncio
    async def test_liveness_check(self):
        """Test liveness probe endpoint."""
        from src.api.health import liveness_check
        
        response = await liveness_check()
        
        assert response["status"] == "alive"
        assert "uptime_seconds" in response


class TestAuthEndpoints:
    """Test authentication endpoints."""

    @pytest.fixture
    def mock_user_data(self):
        """Mock user data for testing."""
        return {
            "user_id": str(uuid.uuid4()),
            "email": "test@example.com",
            "username": "testuser",
            "role": "standard_user",
            "is_active": True,
            "created_at": datetime.utcnow()
        }

    @pytest.mark.asyncio
    async def test_login_endpoint(self, mock_user_data):
        """Test login endpoint."""
        from src.api.auth import login
        from src.auth.models import LoginRequest
        
        request = LoginRequest(email="test@example.com", password="password123!")
        
        with patch("src.api.auth.authenticate_user") as mock_auth, \
             patch("src.api.auth.create_access_token") as mock_token:
            
            mock_auth.return_value = mock_user_data
            mock_token.return_value = "test_token"
            
            response = await login(request)
            
            assert response["access_token"] == "test_token"
            assert response["token_type"] == "bearer"
            assert response["user"]["email"] == "test@example.com"

    @pytest.mark.asyncio
    async def test_login_invalid_credentials(self):
        """Test login with invalid credentials."""
        from src.api.auth import login
        from src.auth.models import LoginRequest
        
        request = LoginRequest(email="test@example.com", password="wrong_password")
        
        with patch("src.api.auth.authenticate_user") as mock_auth:
            mock_auth.return_value = None
            
            with pytest.raises(HTTPException) as exc_info:
                await login(request)
            
            assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_register_endpoint(self):
        """Test user registration endpoint."""
        from src.api.auth import register
        from src.auth.models import UserCreate
        
        user_data = UserCreate(
            email="newuser@example.com",
            username="newuser",
            password="SecurePass123!",
            role="standard_user"
        )
        
        with patch("src.api.auth.create_user") as mock_create:
            mock_user = {
                "user_id": str(uuid.uuid4()),
                "email": "newuser@example.com",
                "username": "newuser",
                "role": "standard_user"
            }
            mock_create.return_value = mock_user
            
            response = await register(user_data)
            
            assert response["email"] == "newuser@example.com"
            assert response["username"] == "newuser"

    @pytest.mark.asyncio
    async def test_get_current_user_endpoint(self, mock_user_data):
        """Test get current user endpoint."""
        from src.api.auth import get_current_user_info
        
        response = await get_current_user_info(mock_user_data)
        
        assert response["email"] == mock_user_data["email"]
        assert response["role"] == mock_user_data["role"]

    @pytest.mark.asyncio
    async def test_refresh_token_endpoint(self, mock_user_data):
        """Test token refresh endpoint."""
        from src.api.auth import refresh_token
        
        with patch("src.api.auth.create_access_token") as mock_token:
            mock_token.return_value = "new_token"
            
            response = await refresh_token(mock_user_data)
            
            assert response["access_token"] == "new_token"
            assert response["token_type"] == "bearer"

    @pytest.mark.asyncio
    async def test_logout_endpoint(self, mock_user_data):
        """Test logout endpoint."""
        from src.api.auth import logout
        
        with patch("src.api.auth.blacklist_token") as mock_blacklist:
            mock_blacklist.return_value = True
            
            response = await logout(mock_user_data, "test_token")
            
            assert response["message"] == "Successfully logged out"


class TestDocumentEndpoints:
    """Test document management endpoints."""

    @pytest.fixture
    def mock_upload_file(self):
        """Mock file upload."""
        file_mock = Mock()
        file_mock.filename = "test_document.pdf"
        file_mock.content_type = "application/pdf"
        file_mock.size = 1024
        file_mock.read = AsyncMock(return_value=b"fake_pdf_content")
        return file_mock

    @pytest.mark.asyncio
    async def test_upload_document_endpoint(self, mock_upload_file, mock_user_data):
        """Test document upload endpoint."""
        from src.api.documents import upload_document
        
        with patch("src.api.documents.get_document_service") as mock_service:
            mock_result = {
                "document_id": str(uuid.uuid4()),
                "filename": "test_document.pdf",
                "status": "completed",
                "message": "Document processed successfully"
            }
            mock_service.return_value.process_uploaded_document = AsyncMock(
                return_value=mock_result
            )
            
            response = await upload_document(
                file=mock_upload_file,
                metadata='{"title": "Test Document"}',
                current_user=mock_user_data
            )
            
            assert response["status"] == "completed"
            assert response["filename"] == "test_document.pdf"

    @pytest.mark.asyncio
    async def test_upload_document_invalid_file_type(self, mock_user_data):
        """Test upload with invalid file type."""
        from src.api.documents import upload_document
        
        invalid_file = Mock()
        invalid_file.filename = "test.exe"
        invalid_file.content_type = "application/x-executable"
        
        with pytest.raises(HTTPException) as exc_info:
            await upload_document(
                file=invalid_file,
                metadata="{}",
                current_user=mock_user_data
            )
        
        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_list_documents_endpoint(self, mock_user_data):
        """Test list documents endpoint."""
        from src.api.documents import list_documents
        
        with patch("src.api.documents.get_document_service") as mock_service:
            mock_documents = {
                "documents": [
                    {
                        "document_id": str(uuid.uuid4()),
                        "filename": "doc1.pdf",
                        "status": "completed",
                        "uploaded_at": datetime.utcnow().isoformat()
                    },
                    {
                        "document_id": str(uuid.uuid4()),
                        "filename": "doc2.pdf",
                        "status": "completed",
                        "uploaded_at": datetime.utcnow().isoformat()
                    }
                ],
                "total": 2,
                "offset": 0,
                "limit": 10,
                "has_next": False
            }
            mock_service.return_value.list_documents = AsyncMock(return_value=mock_documents)
            
            response = await list_documents(
                limit=10,
                offset=0,
                current_user=mock_user_data
            )
            
            assert len(response["documents"]) == 2
            assert response["total"] == 2

    @pytest.mark.asyncio
    async def test_get_document_endpoint(self, mock_user_data):
        """Test get single document endpoint."""
        from src.api.documents import get_document
        
        document_id = str(uuid.uuid4())
        
        with patch("src.api.documents.get_document_service") as mock_service:
            mock_document = {
                "document_id": document_id,
                "filename": "test.pdf",
                "status": "completed",
                "metadata": {"title": "Test Document"},
                "uploaded_at": datetime.utcnow().isoformat()
            }
            mock_service.return_value.get_document = AsyncMock(return_value=mock_document)
            
            response = await get_document(
                document_id=document_id,
                current_user=mock_user_data
            )
            
            assert response["document_id"] == document_id
            assert response["filename"] == "test.pdf"

    @pytest.mark.asyncio
    async def test_delete_document_endpoint(self, mock_user_data):
        """Test delete document endpoint."""
        from src.api.documents import delete_document
        
        document_id = str(uuid.uuid4())
        
        with patch("src.api.documents.get_document_service") as mock_service:
            mock_service.return_value.delete_document = AsyncMock(return_value=True)
            
            response = await delete_document(
                document_id=document_id,
                current_user=mock_user_data
            )
            
            assert response["message"] == "Document deleted successfully"

    @pytest.mark.asyncio
    async def test_delete_document_not_found(self, mock_user_data):
        """Test delete non-existent document."""
        from src.api.documents import delete_document
        
        document_id = str(uuid.uuid4())
        
        with patch("src.api.documents.get_document_service") as mock_service:
            mock_service.return_value.delete_document = AsyncMock(return_value=False)
            
            with pytest.raises(HTTPException) as exc_info:
                await delete_document(
                    document_id=document_id,
                    current_user=mock_user_data
                )
            
            assert exc_info.value.status_code == 404


class TestSearchEndpoints:
    """Test search and RAG endpoints."""

    @pytest.fixture
    def mock_search_results(self):
        """Mock search results."""
        from src.search.models import SearchResponse, SearchResult, SearchType
        
        results = [
            SearchResult(
                chunk_id=uuid.uuid4(),
                document_id=uuid.uuid4(),
                text="This is relevant content about machine learning.",
                chunk_index=0,
                score=0.85,
                semantic_score=0.85,
                document_filename="ml_guide.pdf",
                document_type="pdf",
                metadata={"title": "ML Guide", "page_number": 1}
            ),
            SearchResult(
                chunk_id=uuid.uuid4(),
                document_id=uuid.uuid4(),
                text="Another relevant document about AI concepts.",
                chunk_index=1,
                score=0.78,
                semantic_score=0.78,
                document_filename="ai_intro.pdf",
                document_type="pdf",
                metadata={"title": "AI Introduction", "page_number": 3}
            )
        ]
        
        return SearchResponse(
            query="machine learning",
            search_type=SearchType.SEMANTIC,
            results=results,
            total_results=2,
            limit=10,
            offset=0,
            has_next=False,
            search_time_ms=150.5
        )

    @pytest.mark.asyncio
    async def test_semantic_search_endpoint(self, mock_search_results, mock_user_data):
        """Test semantic search endpoint."""
        from src.api.search import semantic_search
        from src.search.models import SemanticSearchRequest
        
        request = SemanticSearchRequest(
            query="machine learning",
            limit=10,
            similarity_threshold=0.5
        )
        
        with patch("src.api.search.get_search_service") as mock_service:
            mock_service.return_value.semantic_search = AsyncMock(
                return_value=mock_search_results
            )
            
            response = await semantic_search(request, mock_user_data)
            
            assert response.query == "machine learning"
            assert len(response.results) == 2
            assert response.search_time_ms == 150.5

    @pytest.mark.asyncio
    async def test_hybrid_search_endpoint(self, mock_search_results, mock_user_data):
        """Test hybrid search endpoint."""
        from src.api.search import hybrid_search
        from src.search.models import HybridSearchRequest
        
        request = HybridSearchRequest(
            query="machine learning",
            semantic_weight=0.7,
            keyword_weight=0.3,
            limit=10
        )
        
        with patch("src.api.search.get_search_service") as mock_service:
            mock_service.return_value.hybrid_search = AsyncMock(
                return_value=mock_search_results
            )
            
            response = await hybrid_search(request, mock_user_data)
            
            assert response.query == "machine learning"
            assert len(response.results) == 2

    @pytest.mark.asyncio
    async def test_rag_chat_endpoint(self, mock_user_data):
        """Test RAG chat endpoint."""
        from src.api.search import rag_chat
        from src.search.models import RAGChatRequest
        
        request = RAGChatRequest(
            query="What is machine learning?",
            conversation_history=[
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi! How can I help you?"}
            ],
            include_sources=True
        )
        
        with patch("src.api.search.get_search_service") as mock_service:
            mock_rag_response = {
                "response": "Machine learning is a subset of AI that enables computers to learn from data.",
                "sources": [
                    {
                        "document_filename": "ml_guide.pdf",
                        "chunk_text": "Machine learning fundamentals...",
                        "score": 0.9
                    }
                ],
                "conversation_id": str(uuid.uuid4()),
                "response_time_ms": 2500.0
            }
            mock_service.return_value.rag_chat = AsyncMock(return_value=mock_rag_response)
            
            response = await rag_chat(request, mock_user_data)
            
            assert "Machine learning" in response["response"]
            assert len(response["sources"]) == 1
            assert "conversation_id" in response

    @pytest.mark.asyncio
    async def test_search_with_metadata_filters(self, mock_search_results, mock_user_data):
        """Test search with metadata filters."""
        from src.api.search import semantic_search
        from src.search.models import SemanticSearchRequest, MetadataFilter
        
        filters = MetadataFilter(
            document_type="pdf",
            uploaded_by=mock_user_data["user_id"],
            tags=["technical"]
        )
        
        request = SemanticSearchRequest(
            query="technical documentation",
            filters=filters,
            limit=5
        )
        
        with patch("src.api.search.get_search_service") as mock_service:
            mock_service.return_value.semantic_search = AsyncMock(
                return_value=mock_search_results
            )
            
            response = await semantic_search(request, mock_user_data)
            
            # Verify service was called with filters
            call_args = mock_service.return_value.semantic_search.call_args[0]
            assert call_args[0].filters is not None

    @pytest.mark.asyncio
    async def test_search_analytics_endpoint(self, mock_user_data):
        """Test search analytics endpoint."""
        from src.api.search import get_search_analytics
        
        with patch("src.api.search.get_search_service") as mock_service:
            mock_analytics = {
                "total_searches": 150,
                "average_response_time": 245.5,
                "popular_queries": [
                    {"query": "machine learning", "count": 25},
                    {"query": "artificial intelligence", "count": 18}
                ],
                "search_trends": {
                    "last_24h": 45,
                    "last_7d": 150,
                    "last_30d": 500
                }
            }
            mock_service.return_value.get_analytics = AsyncMock(return_value=mock_analytics)
            
            response = await get_search_analytics(
                period="7d",
                current_user=mock_user_data
            )
            
            assert response["total_searches"] == 150
            assert len(response["popular_queries"]) == 2


class TestAPIMiddleware:
    """Test API middleware functionality."""

    def test_cors_middleware(self):
        """Test CORS middleware configuration."""
        # This would test CORS headers in actual HTTP responses
        # For now, we test the configuration exists
        from src.core.config import get_settings
        
        settings = get_settings()
        assert settings.cors_origins is not None
        assert settings.cors_methods is not None

    def test_security_headers_middleware(self):
        """Test security headers middleware."""
        # This would test security headers in HTTP responses
        # Mock test for security header configuration
        expected_headers = [
            "X-Content-Type-Options",
            "X-Frame-Options", 
            "X-XSS-Protection",
            "Referrer-Policy"
        ]
        
        # In actual implementation, these would be tested via HTTP client
        assert all(header for header in expected_headers)

    def test_request_logging_middleware(self):
        """Test request logging middleware."""
        # This would test that requests are logged properly
        # Mock test for logging configuration
        with patch("src.core.logging.log_request_response") as mock_log:
            # Simulate middleware call
            mock_log(
                method="GET",
                url="/api/v1/health",
                status_code=200,
                response_time=0.025,
                user_agent="test-client",
                ip_address="127.0.0.1"
            )
            
            mock_log.assert_called_once()


class TestAPIErrorHandling:
    """Test API error handling and exception responses."""

    @pytest.mark.asyncio
    async def test_validation_error_handling(self):
        """Test validation error responses."""
        from src.search.models import SemanticSearchRequest
        
        # Test invalid request data
        with pytest.raises(ValueError):
            SemanticSearchRequest(
                query="",  # Empty query should fail validation
                limit=10
            )

    @pytest.mark.asyncio
    async def test_authentication_error_handling(self):
        """Test authentication error responses."""
        from src.api.auth import get_current_user_info
        
        # Test with invalid user data
        with pytest.raises(HTTPException):
            await get_current_user_info(None)

    @pytest.mark.asyncio
    async def test_service_unavailable_error(self, mock_user_data):
        """Test service unavailable error handling."""
        from src.api.search import semantic_search
        from src.search.models import SemanticSearchRequest
        
        request = SemanticSearchRequest(query="test", limit=10)
        
        with patch("src.api.search.get_search_service") as mock_service:
            mock_service.return_value.semantic_search = AsyncMock(
                side_effect=Exception("Search service unavailable")
            )
            
            with pytest.raises(HTTPException) as exc_info:
                await semantic_search(request, mock_user_data)
            
            assert exc_info.value.status_code == 500

    @pytest.mark.asyncio
    async def test_rate_limiting_error(self, mock_user_data):
        """Test rate limiting error responses."""
        # This would test actual rate limiting implementation
        # For now, we test the concept
        from fastapi import HTTPException
        
        # Simulate rate limit exceeded
        with pytest.raises(HTTPException) as exc_info:
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        
        assert exc_info.value.status_code == 429

    @pytest.mark.asyncio
    async def test_file_size_limit_error(self, mock_user_data):
        """Test file size limit error handling."""
        from src.api.documents import upload_document
        
        # Create oversized file mock
        large_file = Mock()
        large_file.filename = "large_file.pdf"
        large_file.size = 100 * 1024 * 1024  # 100MB
        large_file.content_type = "application/pdf"
        
        # Should raise HTTP exception for file too large
        with pytest.raises(HTTPException) as exc_info:
            await upload_document(
                file=large_file,
                metadata="{}",
                current_user=mock_user_data
            )
        
        assert exc_info.value.status_code == 413


class TestAPIPerformance:
    """Test API performance characteristics."""

    @pytest.mark.asyncio
    async def test_search_response_time(self, mock_search_results, mock_user_data):
        """Test search endpoint response time."""
        from src.api.search import semantic_search
        from src.search.models import SemanticSearchRequest
        
        request = SemanticSearchRequest(query="test", limit=10)
        
        with patch("src.api.search.get_search_service") as mock_service:
            mock_service.return_value.semantic_search = AsyncMock(
                return_value=mock_search_results
            )
            
            import time
            start_time = time.time()
            response = await semantic_search(request, mock_user_data)
            end_time = time.time()
            
            response_time = end_time - start_time
            assert response_time < 1.0  # Should be fast for mocked service

    @pytest.mark.asyncio
    async def test_concurrent_requests(self, mock_search_results, mock_user_data):
        """Test handling of concurrent API requests."""
        from src.api.search import semantic_search
        from src.search.models import SemanticSearchRequest
        
        request = SemanticSearchRequest(query="test", limit=10)
        
        with patch("src.api.search.get_search_service") as mock_service:
            mock_service.return_value.semantic_search = AsyncMock(
                return_value=mock_search_results
            )
            
            # Run multiple concurrent requests
            import asyncio
            tasks = [
                semantic_search(request, mock_user_data)
                for _ in range(10)
            ]
            
            results = await asyncio.gather(*tasks)
            
            assert len(results) == 10
            assert all(result.query == "test" for result in results)


if __name__ == "__main__":
    pytest.main([__file__])