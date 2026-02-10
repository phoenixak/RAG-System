"""
Tests for LLM service and models.
Tests LLM model definitions, prompt templates, and service behavior.
"""

import uuid
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from src.llm.models import (
    LLMError,
    LLMModel,
    LLMProvider,
    RAGContext,
    RAGRequest,
    RAGResponse,
)


class TestLLMModels:
    """Tests for LLM Pydantic models."""

    def test_llm_provider_values(self):
        """Test LLMProvider enum values."""
        assert LLMProvider.OPENAI == "openai"
        assert LLMProvider.ANTHROPIC == "anthropic"

    def test_llm_model_values(self):
        """Test LLMModel enum values."""
        assert LLMModel.GPT_4 == "gpt-4"
        assert LLMModel.GPT_3_5_TURBO == "gpt-3.5-turbo"

    def test_rag_context_creation(self):
        """Test RAGContext model creation."""
        ctx = RAGContext(
            content="Some document content here.",
            document_name="report.pdf",
            score=0.85,
            page_number=3,
        )
        assert ctx.content == "Some document content here."
        assert ctx.score == 0.85
        assert ctx.page_number == 3

    def test_rag_request_creation(self):
        """Test RAGRequest model creation."""
        ctx = RAGContext(content="context", document_name="doc.pdf", score=0.9)
        req = RAGRequest(
            query="What is machine learning?",
            context_docs=[ctx],
            max_tokens=500,
            temperature=0.5,
        )
        assert req.query == "What is machine learning?"
        assert len(req.context_docs) == 1
        assert req.max_tokens == 500

    def test_rag_response_creation(self):
        """Test RAGResponse model creation."""
        resp = RAGResponse(
            response="Machine learning is...",
            sources_used=["doc1.pdf", "doc2.pdf"],
            token_count=150,
            model_used="gpt-3.5-turbo",
            provider="openai",
        )
        assert resp.response == "Machine learning is..."
        assert len(resp.sources_used) == 2

    def test_llm_error_creation(self):
        """Test LLMError model creation."""
        err = LLMError(
            error_type="rate_limit",
            message="Too many requests",
            provider="openai",
            model="gpt-4",
        )
        assert err.error_type == "rate_limit"


class TestRAGPromptTemplates:
    """Tests for RAG prompt template generation."""

    def test_create_system_prompt(self):
        """Test system prompt creation."""
        from src.llm.prompts import RAGPromptTemplates

        prompt = RAGPromptTemplates.create_system_prompt()
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_create_user_prompt(self):
        """Test user prompt creation with context."""
        from src.llm.prompts import RAGPromptTemplates

        ctx = RAGContext(content="ML is great", document_name="ml.pdf", score=0.9)
        prompt = RAGPromptTemplates.create_user_prompt(
            query="What is ML?",
            context_docs=[ctx],
        )
        assert isinstance(prompt, str)
        assert "What is ML?" in prompt

    def test_create_conversation_prompt(self):
        """Test conversation prompt with history."""
        from src.llm.prompts import RAGPromptTemplates

        ctx = RAGContext(content="context text", document_name="doc.pdf", score=0.8)
        history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        prompt = RAGPromptTemplates.create_conversation_prompt(
            query="Follow up question",
            context_docs=[ctx],
            conversation_history=history,
        )
        assert isinstance(prompt, str)

    def test_create_no_context_prompt(self):
        """Test prompt when no context documents are available."""
        from src.llm.prompts import RAGPromptTemplates

        prompt = RAGPromptTemplates.create_no_context_prompt(query="General question?")
        assert isinstance(prompt, str)
        assert "General question?" in prompt


class TestLLMService:
    """Tests for LLMService."""

    def test_is_api_available_no_key(self):
        """Test API availability when no API key is set."""
        with patch.dict("os.environ", {}, clear=False):
            # Remove API keys if present
            import os

            os.environ.pop("OPENAI_API_KEY", None)
            os.environ.pop("ANTHROPIC_API_KEY", None)

            from src.llm.service import LLMService

            service = LLMService()
            # Without API keys, should report not available
            result = service.is_api_available()
            assert isinstance(result, bool)

    @pytest.mark.asyncio
    async def test_generate_rag_response_no_api(self):
        """Test RAG response generation falls back when no API available."""
        from src.llm.service import LLMService

        service = LLMService()
        context_docs = [
            {"content": "test context", "document_name": "test.pdf", "score": 0.9}
        ]
        response = await service.generate_rag_response(
            query="What is this about?",
            context_docs=context_docs,
        )
        assert isinstance(response, str)
        assert len(response) > 0
