"""
LLM Service Tests
Comprehensive tests for LLM integration and RAG response generation.
"""

import pytest
from unittest.mock import AsyncMock, Mock, patch
from datetime import datetime

from src.llm.models import RAGRequest, RAGResponse, RAGContext, LLMProvider
from src.llm.service import LLMService, get_llm_service, initialize_llm_service
from src.llm.prompts import RAGPromptTemplates


class TestLLMModels:
    """Test LLM data models."""

    def test_rag_context_model(self):
        """Test RAGContext model validation."""
        context = RAGContext(
            content="This is sample document content.",
            document_name="test_doc.pdf",
            score=0.85,
            page_number=1,
            metadata={"author": "Test Author"}
        )
        
        assert context.content == "This is sample document content."
        assert context.document_name == "test_doc.pdf"
        assert context.score == 0.85
        assert context.page_number == 1
        assert context.metadata["author"] == "Test Author"

    def test_rag_request_model(self):
        """Test RAGRequest model validation."""
        context_docs = [
            RAGContext(
                content="Document content",
                document_name="doc1.pdf",
                score=0.9
            )
        ]
        
        request = RAGRequest(
            query="What is machine learning?",
            context_docs=context_docs,
            conversation_history=[{"role": "user", "content": "Hello"}],
            max_tokens=1000,
            temperature=0.7
        )
        
        assert request.query == "What is machine learning?"
        assert len(request.context_docs) == 1
        assert request.max_tokens == 1000
        assert request.temperature == 0.7

    def test_rag_response_model(self):
        """Test RAGResponse model validation."""
        response = RAGResponse(
            response="Machine learning is a subset of AI...",
            sources_used=["doc1.pdf", "doc2.pdf"],
            token_count=150,
            model_used="gpt-3.5-turbo",
            provider="openai",
            response_time_ms=1200.5
        )
        
        assert "Machine learning" in response.response
        assert len(response.sources_used) == 2
        assert response.token_count == 150
        assert response.model_used == "gpt-3.5-turbo"
        assert response.provider == "openai"

    def test_llm_provider_enum(self):
        """Test LLM provider enumeration."""
        assert LLMProvider.OPENAI.value == "openai"
        assert LLMProvider.ANTHROPIC.value == "anthropic"


class TestRAGPromptTemplates:
    """Test RAG prompt template generation."""

    def test_system_prompt_creation(self):
        """Test system prompt template creation."""
        templates = RAGPromptTemplates()
        system_prompt = templates.create_system_prompt()
        
        assert isinstance(system_prompt, str)
        assert len(system_prompt) > 0
        assert "assistant" in system_prompt.lower()

    def test_user_prompt_creation(self):
        """Test user prompt creation with context."""
        templates = RAGPromptTemplates()
        
        context_docs = [
            RAGContext(
                content="Machine learning is a method of data analysis.",
                document_name="ml_guide.pdf",
                score=0.9
            ),
            RAGContext(
                content="AI systems can learn from data.",
                document_name="ai_intro.pdf", 
                score=0.8
            )
        ]
        
        user_prompt = templates.create_user_prompt(
            query="What is machine learning?",
            context_docs=context_docs
        )
        
        assert "What is machine learning?" in user_prompt
        assert "Machine learning is a method" in user_prompt
        assert "ml_guide.pdf" in user_prompt
        assert "ai_intro.pdf" in user_prompt

    def test_conversation_prompt_creation(self):
        """Test conversation prompt with history."""
        templates = RAGPromptTemplates()
        
        context_docs = [
            RAGContext(content="Test content", document_name="test.pdf", score=0.8)
        ]
        
        conversation_history = [
            {"role": "user", "content": "What is AI?"},
            {"role": "assistant", "content": "AI is artificial intelligence."},
            {"role": "user", "content": "How does it work?"}
        ]
        
        conv_prompt = templates.create_conversation_prompt(
            query="Tell me more about ML",
            context_docs=context_docs,
            conversation_history=conversation_history
        )
        
        assert "Tell me more about ML" in conv_prompt
        assert "What is AI?" in conv_prompt
        assert "How does it work?" in conv_prompt

    def test_no_context_prompt(self):
        """Test prompt when no context documents are available."""
        templates = RAGPromptTemplates()
        
        no_context_prompt = templates.create_no_context_prompt(
            "What is quantum computing?"
        )
        
        assert "quantum computing" in no_context_prompt
        assert "no relevant" in no_context_prompt.lower() or "sorry" in no_context_prompt.lower()


class TestLLMService:
    """Test LLM service functionality."""

    def test_llm_service_initialization(self):
        """Test LLM service initialization."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test_key"}):
            service = LLMService()
            
            assert service.provider == "openai"
            assert service.model is not None
            assert service._api_available is True

    def test_llm_service_no_api_key(self):
        """Test LLM service initialization without API key."""
        with patch.dict("os.environ", {}, clear=True):
            service = LLMService()
            
            assert service._api_available is False

    def test_provider_configuration(self):
        """Test LLM provider configuration."""
        # Test OpenAI provider
        with patch.dict("os.environ", {"LLM_PROVIDER": "openai"}):
            service = LLMService()
            assert service.provider == "openai"
        
        # Test Anthropic provider
        with patch.dict("os.environ", {"LLM_PROVIDER": "anthropic"}):
            service = LLMService()
            assert service.provider == "anthropic"

    def test_model_configuration(self):
        """Test model configuration for different providers."""
        # Test OpenAI model
        with patch.dict("os.environ", {"LLM_PROVIDER": "openai", "LLM_MODEL": "gpt-4"}):
            service = LLMService()
            assert service.model == "gpt-4"
        
        # Test Anthropic model
        with patch.dict("os.environ", {"LLM_PROVIDER": "anthropic", "LLM_MODEL": "claude-3-opus-20240229"}):
            service = LLMService()
            assert service.model == "claude-3-opus-20240229"

    @pytest.mark.asyncio
    async def test_openai_client_creation(self):
        """Test OpenAI client creation."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test_key"}), \
             patch("openai.AsyncOpenAI") as mock_openai:
            
            service = LLMService()
            client = await service._get_client()
            
            mock_openai.assert_called_once_with(api_key="test_key")

    @pytest.mark.asyncio
    async def test_anthropic_client_creation(self):
        """Test Anthropic client creation."""
        with patch.dict("os.environ", {
            "LLM_PROVIDER": "anthropic",
            "ANTHROPIC_API_KEY": "test_key"
        }), patch("anthropic.AsyncAnthropic") as mock_anthropic:
            
            service = LLMService()
            client = await service._get_client()
            
            mock_anthropic.assert_called_once_with(api_key="test_key")

    @pytest.mark.asyncio
    async def test_client_creation_without_api_key(self):
        """Test client creation fails without API key."""
        with patch.dict("os.environ", {}, clear=True):
            service = LLMService()
            
            with pytest.raises(ValueError, match="API key not available"):
                await service._get_client()


class TestRAGResponseGeneration:
    """Test RAG response generation."""

    @pytest.mark.asyncio
    async def test_generate_rag_response_with_api(self):
        """Test RAG response generation with API available."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test_key"}):
            service = LLMService()
            
            # Mock OpenAI response
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = "This is a test response about machine learning."
            mock_response.usage.total_tokens = 150
            
            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            
            with patch.object(service, '_get_client', return_value=mock_client):
                response = await service.generate_rag_response(
                    query="What is machine learning?",
                    context_docs=[{
                        "text": "Machine learning is a method of data analysis.",
                        "document_filename": "ml_guide.pdf",
                        "score": 0.9
                    }]
                )
                
                assert "machine learning" in response.lower()
                assert len(response) > 0

    @pytest.mark.asyncio
    async def test_generate_rag_response_without_api(self):
        """Test RAG response generation without API (demo mode)."""
        with patch.dict("os.environ", {}, clear=True):
            service = LLMService()
            
            response = await service.generate_rag_response(
                query="What is machine learning?",
                context_docs=[{
                    "text": "Machine learning is a method of data analysis.",
                    "document_filename": "ml_guide.pdf",
                    "score": 0.9
                }]
            )
            
            assert "Demo Mode" in response
            assert "machine learning" in response.lower()
            assert "ml_guide.pdf" in response

    @pytest.mark.asyncio
    async def test_fallback_response_with_context(self):
        """Test fallback response generation with context documents."""
        service = LLMService()
        
        context_docs = [
            {
                "text": "Machine learning enables computers to learn from data.",
                "document_filename": "ml_basics.pdf",
                "score": 0.85
            },
            {
                "text": "Deep learning is a subset of machine learning.",
                "document_filename": "dl_intro.pdf",
                "score": 0.78
            }
        ]
        
        response = service._create_fallback_response(
            query="What is deep learning?",
            context_docs=context_docs
        )
        
        assert "2 relevant documents" in response
        assert "ml_basics.pdf" in response
        assert "dl_intro.pdf" in response
        assert "85.0%" in response  # Score formatting

    @pytest.mark.asyncio
    async def test_fallback_response_without_context(self):
        """Test fallback response when no context documents found."""
        service = LLMService()
        
        response = service._create_fallback_response(
            query="What is quantum computing?",
            context_docs=[]
        )
        
        assert "Demo Mode" in response
        assert "No relevant documents found" in response
        assert "quantum computing" in response

    @pytest.mark.asyncio
    async def test_no_api_response_with_results(self):
        """Test demo mode response with search results."""
        with patch.dict("os.environ", {}, clear=True):
            service = LLMService()
            
            context_docs = [
                {
                    "text": "Artificial intelligence is transforming industries.",
                    "document_filename": "ai_trends.pdf",
                    "score": 0.92,
                    "page_number": 3
                }
            ]
            
            response = service._create_no_api_response(
                query="AI trends",
                context_docs=context_docs
            )
            
            assert "Search Results for Demo Mode" in response
            assert "AI trends" in response
            assert "ai_trends.pdf" in response
            assert "Page 3" in response
            assert "92.0%" in response


class TestLLMConnectionTesting:
    """Test LLM connection testing functionality."""

    @pytest.mark.asyncio
    async def test_connection_test_success(self):
        """Test successful connection test."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test_key"}):
            service = LLMService()
            
            # Mock successful response generation
            with patch.object(service, '_generate_response') as mock_generate:
                mock_response = RAGResponse(
                    response="Hello! This is a test response.",
                    sources_used=["test.txt"],
                    token_count=10,
                    model_used="gpt-3.5-turbo",
                    provider="openai"
                )
                mock_generate.return_value = mock_response
                
                result = await service.test_connection()
                
                assert result["success"] is True
                assert result["provider"] == "openai"
                assert result["model"] == service.model
                assert "response_length" in result

    @pytest.mark.asyncio
    async def test_connection_test_no_api_key(self):
        """Test connection test without API key."""
        with patch.dict("os.environ", {}, clear=True):
            service = LLMService()
            
            result = await service.test_connection()
            
            assert result["success"] is False
            assert result["demo_mode"] is True
            assert "API key not configured" in result["error"]
            assert "instructions" in result

    @pytest.mark.asyncio
    async def test_connection_test_api_failure(self):
        """Test connection test with API failure."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "invalid_key"}):
            service = LLMService()
            
            # Mock API failure
            with patch.object(service, '_generate_response') as mock_generate:
                mock_generate.side_effect = Exception("API authentication failed")
                
                result = await service.test_connection()
                
                assert result["success"] is False
                assert result["demo_mode"] is False
                assert "API authentication failed" in result["error"]


class TestLLMServiceIntegration:
    """Test LLM service integration functions."""

    def test_get_llm_service_singleton(self):
        """Test LLM service singleton pattern."""
        service1 = get_llm_service()
        service2 = get_llm_service()
        
        assert service1 is service2  # Should be the same instance

    @pytest.mark.asyncio
    async def test_initialize_llm_service_success(self):
        """Test LLM service initialization with successful connection."""
        with patch("src.llm.service.get_llm_service") as mock_get_service:
            mock_service = Mock()
            mock_service.test_connection = AsyncMock(return_value={
                "success": True,
                "provider": "openai",
                "model": "gpt-3.5-turbo"
            })
            mock_get_service.return_value = mock_service
            
            service = await initialize_llm_service()
            
            assert service is mock_service
            mock_service.test_connection.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_llm_service_demo_mode(self):
        """Test LLM service initialization in demo mode."""
        with patch("src.llm.service.get_llm_service") as mock_get_service:
            mock_service = Mock()
            mock_service.test_connection = AsyncMock(return_value={
                "success": False,
                "demo_mode": True,
                "provider": "openai",
                "instructions": "Set OPENAI_API_KEY environment variable"
            })
            mock_get_service.return_value = mock_service
            
            service = await initialize_llm_service()
            
            assert service is mock_service

    @pytest.mark.asyncio
    async def test_initialize_llm_service_failure(self):
        """Test LLM service initialization with connection failure."""
        with patch("src.llm.service.get_llm_service") as mock_get_service:
            mock_service = Mock()
            mock_service.test_connection = AsyncMock(return_value={
                "success": False,
                "demo_mode": False,
                "provider": "openai",
                "error": "Network connection failed"
            })
            mock_get_service.return_value = mock_service
            
            service = await initialize_llm_service()
            
            assert service is mock_service


class TestLLMErrorHandling:
    """Test LLM service error handling."""

    @pytest.mark.asyncio
    async def test_invalid_provider_error(self):
        """Test handling of invalid provider configuration."""
        with patch.dict("os.environ", {"LLM_PROVIDER": "invalid_provider"}):
            service = LLMService()
            
            # Should fall back to OpenAI
            assert service.provider == "openai"

    @pytest.mark.asyncio
    async def test_missing_dependency_error(self):
        """Test handling of missing package dependencies."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test_key"}):
            service = LLMService()
            
            # Mock import error
            with patch("builtins.__import__", side_effect=ImportError("openai not installed")):
                with pytest.raises(ValueError):
                    await service._get_client()

    @pytest.mark.asyncio
    async def test_api_rate_limit_handling(self):
        """Test handling of API rate limits."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test_key"}):
            service = LLMService()
            
            # Mock rate limit error
            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(
                side_effect=Exception("Rate limit exceeded")
            )
            
            with patch.object(service, '_get_client', return_value=mock_client):
                # Should fall back to demo response
                response = await service.generate_rag_response(
                    query="test query",
                    context_docs=[]
                )
                
                assert "Demo Mode" in response

    @pytest.mark.asyncio
    async def test_context_length_exceeded(self):
        """Test handling of context length limits."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test_key"}):
            service = LLMService()
            
            # Create very long context that would exceed limits
            very_long_context = [
                {
                    "text": "This is a very long document. " * 1000,  # Very long text
                    "document_filename": "long_doc.pdf",
                    "score": 0.9
                }
            ]
            
            # Mock context length error
            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(
                side_effect=Exception("Context length exceeded")
            )
            
            with patch.object(service, '_get_client', return_value=mock_client):
                response = await service.generate_rag_response(
                    query="test query",
                    context_docs=very_long_context
                )
                
                # Should fall back gracefully
                assert isinstance(response, str)
                assert len(response) > 0


if __name__ == "__main__":
    pytest.main([__file__])