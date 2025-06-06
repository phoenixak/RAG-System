"""
LLM Models
Pydantic models for LLM service requests and responses.
"""

from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class LLMProvider(str, Enum):
    """Supported LLM providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class LLMModel(str, Enum):
    """Supported LLM models."""

    # OpenAI models
    GPT_4 = "gpt-4"
    GPT_4_TURBO = "gpt-4-turbo"
    GPT_3_5_TURBO = "gpt-3.5-turbo"

    # Anthropic models
    CLAUDE_3_OPUS = "claude-3-opus-20240229"
    CLAUDE_3_SONNET = "claude-3-sonnet-20240229"
    CLAUDE_3_HAIKU = "claude-3-haiku-20240307"


class RAGContext(BaseModel):
    """Context document for RAG response generation."""

    content: str = Field(..., description="Document content")
    document_name: str = Field(..., description="Document filename")
    score: float = Field(..., description="Relevance score")
    page_number: Optional[int] = Field(None, description="Page number if applicable")
    metadata: Optional[Dict] = Field(
        default_factory=dict, description="Additional metadata"
    )


class RAGRequest(BaseModel):
    """Request for RAG response generation."""

    query: str = Field(..., description="User query")
    context_docs: List[RAGContext] = Field(
        ..., description="Retrieved context documents"
    )
    conversation_history: Optional[List[Dict]] = Field(
        default_factory=list, description="Previous conversation"
    )
    max_tokens: Optional[int] = Field(
        default=1000, description="Maximum tokens in response"
    )
    temperature: Optional[float] = Field(
        default=0.7, description="Response creativity (0.0-1.0)"
    )


class RAGResponse(BaseModel):
    """Response from RAG generation."""

    response: str = Field(..., description="Generated response")
    sources_used: List[str] = Field(..., description="Document names referenced")
    token_count: Optional[int] = Field(None, description="Tokens used in generation")
    model_used: str = Field(..., description="LLM model used")
    provider: str = Field(..., description="LLM provider used")


class LLMError(BaseModel):
    """LLM service error response."""

    error_type: str = Field(..., description="Type of error")
    message: str = Field(..., description="Error message")
    provider: Optional[str] = Field(None, description="Provider that failed")
    model: Optional[str] = Field(None, description="Model that failed")
