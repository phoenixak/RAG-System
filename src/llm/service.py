"""
LLM Service
Main service for generating RAG responses using various LLM providers.
"""

import os
from typing import Dict, List, Optional

from src.core.logging import get_logger

from .models import LLMProvider, RAGContext, RAGRequest, RAGResponse
from .prompts import RAGPromptTemplates

logger = get_logger(__name__)


class LLMService:
    """Service for generating RAG responses using LLM providers."""

    def __init__(self):
        """Initialize the LLM service."""
        self.provider = self._get_provider()
        self.model = self._get_model()
        self.prompt_templates = RAGPromptTemplates()
        self._client = None
        self._api_available = self._check_api_availability()

        logger.info(
            "LLM service initialized", 
            provider=self.provider, 
            model=self.model,
            api_available=self._api_available
        )

    def _get_provider(self) -> str:
        """Get the configured LLM provider."""
        provider = os.getenv("LLM_PROVIDER", "openai").lower()
        if provider not in [p.value for p in LLMProvider]:
            logger.warning(f"Unknown provider {provider}, defaulting to openai")
            return LLMProvider.OPENAI.value
        return provider

    def _get_model(self) -> str:
        """Get the configured LLM model."""
        if self.provider == LLMProvider.OPENAI.value:
            return os.getenv("LLM_MODEL", "gpt-3.5-turbo")
        elif self.provider == LLMProvider.ANTHROPIC.value:
            return os.getenv("LLM_MODEL", "claude-3-sonnet-20240229")
        else:
            return os.getenv("LLM_MODEL", "gpt-3.5-turbo")

    def _check_api_availability(self) -> bool:
        """Check if API keys are available for the configured provider."""
        if self.provider == LLMProvider.OPENAI.value:
            return bool(os.getenv("OPENAI_API_KEY"))
        elif self.provider == LLMProvider.ANTHROPIC.value:
            return bool(os.getenv("ANTHROPIC_API_KEY"))
        return False

    async def _get_client(self):
        """Get or create the LLM client."""
        if not self._api_available:
            raise ValueError(f"API key not available for provider: {self.provider}")
            
        if self._client is None:
            if self.provider == LLMProvider.OPENAI.value:
                try:
                    import openai

                    api_key = os.getenv("OPENAI_API_KEY")
                    if not api_key:
                        raise ValueError("OPENAI_API_KEY environment variable not set")
                    self._client = openai.AsyncOpenAI(api_key=api_key)
                except ImportError:
                    raise ImportError(
                        "openai package not installed. Run: pip install openai>=1.0.0"
                    )

            elif self.provider == LLMProvider.ANTHROPIC.value:
                try:
                    import anthropic

                    api_key = os.getenv("ANTHROPIC_API_KEY")
                    if not api_key:
                        raise ValueError(
                            "ANTHROPIC_API_KEY environment variable not set"
                        )
                    self._client = anthropic.AsyncAnthropic(api_key=api_key)
                except ImportError:
                    raise ImportError(
                        "anthropic package not installed. Run: pip install anthropic>=0.7.0"
                    )

            else:
                raise ValueError(f"Unsupported provider: {self.provider}")

        return self._client

    async def generate_rag_response(
        self,
        query: str,
        context_docs: List[Dict],
        conversation_history: Optional[List[Dict]] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Generate a RAG response based on query and context documents.

        Args:
            query: User query
            context_docs: List of retrieved documents with content, score, metadata
            conversation_history: Previous conversation messages
            max_tokens: Maximum tokens in response
            temperature: Response creativity (0.0-1.0)

        Returns:
            Generated response string
        """
        try:
            # Convert context docs to RAGContext objects
            rag_contexts = []
            for doc in context_docs:
                rag_context = RAGContext(
                    content=doc.get("text", doc.get("content", "")),
                    document_name=doc.get(
                        "document_filename",
                        doc.get("document_name", "Unknown Document"),
                    ),
                    score=doc.get("score", 0.0),
                    page_number=doc.get("page_number"),
                    metadata=doc.get("metadata", {}),
                )
                rag_contexts.append(rag_context)

            # Create RAG request
            request = RAGRequest(
                query=query,
                context_docs=rag_contexts,
                conversation_history=conversation_history or [],
                max_tokens=max_tokens or 1000,
                temperature=temperature or 0.7,
            )

            # Generate response
            response = await self._generate_response(request)
            return response.response

        except Exception as e:
            logger.error(
                "Failed to generate RAG response",
                error=str(e),
                query=query,
                num_docs=len(context_docs),
            )
            # Return fallback response
            return self._create_fallback_response(query, context_docs)

    async def _generate_response(self, request: RAGRequest) -> RAGResponse:
        """Generate response using configured LLM provider."""
        try:
            client = await self._get_client()

            if self.provider == LLMProvider.OPENAI.value:
                return await self._generate_openai_response(client, request)
            elif self.provider == LLMProvider.ANTHROPIC.value:
                return await self._generate_anthropic_response(client, request)
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")

        except Exception as e:
            logger.error(
                "LLM generation failed",
                provider=self.provider,
                model=self.model,
                error=str(e),
            )
            raise

    async def _generate_openai_response(
        self, client, request: RAGRequest
    ) -> RAGResponse:
        """Generate response using OpenAI."""
        # Prepare messages
        system_prompt = self.prompt_templates.create_system_prompt()

        if request.conversation_history:
            user_prompt = self.prompt_templates.create_conversation_prompt(
                request.query, request.context_docs, request.conversation_history
            )
        else:
            user_prompt = self.prompt_templates.create_user_prompt(
                request.query, request.context_docs
            )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # Call OpenAI API
        response = await client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
        )

        # Extract sources used
        sources_used = [doc.document_name for doc in request.context_docs]

        return RAGResponse(
            response=response.choices[0].message.content,
            sources_used=sources_used,
            token_count=response.usage.total_tokens if response.usage else None,
            model_used=self.model,
            provider=self.provider,
        )

    async def _generate_anthropic_response(
        self, client, request: RAGRequest
    ) -> RAGResponse:
        """Generate response using Anthropic Claude."""
        # Prepare prompt
        system_prompt = self.prompt_templates.create_system_prompt()

        if request.conversation_history:
            user_prompt = self.prompt_templates.create_conversation_prompt(
                request.query, request.context_docs, request.conversation_history
            )
        else:
            user_prompt = self.prompt_templates.create_user_prompt(
                request.query, request.context_docs
            )

        # Call Anthropic API
        response = await client.messages.create(
            model=self.model,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            max_tokens=request.max_tokens,
            temperature=request.temperature,
        )

        # Extract sources used
        sources_used = [doc.document_name for doc in request.context_docs]

        return RAGResponse(
            response=response.content[0].text,
            sources_used=sources_used,
            token_count=response.usage.input_tokens + response.usage.output_tokens
            if response.usage
            else None,
            model_used=self.model,
            provider=self.provider,
        )

    def _create_fallback_response(self, query: str, context_docs: List[Dict]) -> str:
        """Create a fallback response when LLM fails."""
        if not self._api_available:
            return self._create_no_api_response(query, context_docs)
            
        if not context_docs:
            return self.prompt_templates.create_no_context_prompt(query)

        # Create a simple summary response
        num_results = len(context_docs)
        if num_results == 1:
            response = "I found 1 relevant document for your query. "
        else:
            response = f"I found {num_results} relevant documents for your query. "

        response += "However, I'm currently unable to generate a detailed response due to a service issue. "
        response += "Here's a summary of the retrieved documents:\n\n"

        for i, doc in enumerate(context_docs[:3], 1):
            doc_name = doc.get(
                "document_filename", doc.get("document_name", "Unknown Document")
            )
            content = doc.get("text", doc.get("content", ""))
            score = doc.get("score", 0)

            summary = content[:150] + "..." if len(content) > 150 else content
            response += f"**{i}. {doc_name}** (relevance: {score:.1%})\n{summary}\n\n"

        if num_results > 3:
            response += f"*Plus {num_results - 3} additional documents.*\n\n"

        response += "💡 Please check the 'Retrieved Documents' section below for full content details."

        return response

    def _create_no_api_response(self, query: str, context_docs: List[Dict]) -> str:
        """Create a response when no API keys are configured."""
        if not context_docs:
            return (
                "🔍 **Search Results for Demo Mode**\n\n"
                f"I searched for: **{query}**\n\n"
                "❌ No relevant documents found.\n\n"
                "---\n\n"
                "🤖 **Demo Mode Notice**: This Enterprise RAG system is running in demo mode without LLM API integration. "
                f"To enable AI-powered responses, please configure your {self.provider.upper()} API key:\n\n"
                f"- Set the `{self.provider.upper()}_API_KEY` environment variable\n"
                "- Restart the application\n\n"
                "In demo mode, you can still:\n"
                "- Upload and process documents\n"
                "- Search through document content\n"
                "- View retrieved document excerpts\n"
                "- Test the full RAG pipeline functionality"
            )

        # Create response with search results but explain demo mode
        num_results = len(context_docs)
        
        response = f"🔍 **Search Results for Demo Mode**\n\n"
        response += f"I searched for: **{query}**\n\n"
        response += f"✅ Found **{num_results}** relevant document{'s' if num_results != 1 else ''}:\n\n"

        for i, doc in enumerate(context_docs[:5], 1):
            doc_name = doc.get(
                "document_filename", doc.get("document_name", "Unknown Document")
            )
            content = doc.get("text", doc.get("content", ""))
            score = doc.get("score", 0)
            page = doc.get("page_number")

            # Create a meaningful excerpt
            summary = content[:200] + "..." if len(content) > 200 else content
            page_info = f" (Page {page})" if page else ""
            
            response += f"**{i}. {doc_name}**{page_info} - Relevance: {score:.1%}\n"
            response += f"*{summary}*\n\n"

        if num_results > 5:
            response += f"*...and {num_results - 5} more relevant documents.*\n\n"

        response += "---\n\n"
        response += "🤖 **Demo Mode Notice**: This Enterprise RAG system is running in demo mode without LLM API integration. "
        response += f"To enable AI-powered responses that synthesize information from these documents, please configure your {self.provider.upper()} API key:\n\n"
        response += f"- Set the `{self.provider.upper()}_API_KEY` environment variable\n"
        response += "- Restart the application\n\n"
        response += "💡 The search and document retrieval functionality is working perfectly! You can see the relevant content above."

        return response

    def is_api_available(self) -> bool:
        """Check if API is available for the configured provider."""
        return self._api_available

    async def test_connection(self) -> Dict[str, any]:
        """Test connection to the configured LLM provider."""
        if not self._api_available:
            return {
                "success": False,
                "provider": self.provider,
                "model": self.model,
                "error": f"API key not configured for {self.provider}",
                "demo_mode": True,
                "instructions": f"Set {self.provider.upper()}_API_KEY environment variable to enable LLM responses"
            }
            
        try:
            client = await self._get_client()

            # Simple test request
            test_request = RAGRequest(
                query="Hello",
                context_docs=[
                    RAGContext(
                        content="This is a test document.",
                        document_name="test.txt",
                        score=1.0,
                    )
                ],
                max_tokens=50,
                temperature=0.1,
            )

            response = await self._generate_response(test_request)

            return {
                "success": True,
                "provider": self.provider,
                "model": self.model,
                "response_length": len(response.response),
                "demo_mode": False,
            }

        except Exception as e:
            logger.error(
                "LLM connection test failed", provider=self.provider, error=str(e)
            )
            return {
                "success": False, 
                "provider": self.provider, 
                "model": self.model,
                "error": str(e),
                "demo_mode": False,
            }


# Global service instance
_llm_service: Optional[LLMService] = None


def get_llm_service() -> LLMService:
    """Get the global LLM service instance."""
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service


async def initialize_llm_service() -> LLMService:
    """Initialize the LLM service with connection testing."""
    service = get_llm_service()

    # Test connection
    connection_test = await service.test_connection()
    if connection_test["success"]:
        logger.info(
            "LLM service initialized successfully",
            provider=service.provider,
            model=service.model,
        )
    elif connection_test.get("demo_mode"):
        logger.info(
            "LLM service initialized in demo mode - API key not configured",
            provider=service.provider,
            model=service.model,
            instructions=connection_test.get("instructions"),
        )
    else:
        logger.warning(
            "LLM service initialized but connection test failed",
            provider=service.provider,
            error=connection_test.get("error"),
        )

    return service
