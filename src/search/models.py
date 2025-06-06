"""
Search Models
Pydantic models for search requests and responses.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field, field_validator


class SearchType(str, Enum):
    """Search type enumeration."""

    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    HYBRID = "hybrid"


class SortOrder(str, Enum):
    """Sort order enumeration."""

    RELEVANCE = "relevance"
    DATE_ASC = "date_asc"
    DATE_DESC = "date_desc"
    TITLE = "title"


class MetadataFilter(BaseModel):
    """Metadata filter for search."""

    document_type: Optional[str] = None
    uploaded_by: Optional[str] = None
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    title: Optional[str] = None
    author: Optional[str] = None
    tags: Optional[List[str]] = None

    # Generic key-value filters
    custom_filters: Optional[Dict[str, Any]] = Field(default_factory=dict)


class SearchRequest(BaseModel):
    """Base search request model."""

    query: str = Field(..., min_length=1, max_length=1000, description="Search query")
    limit: int = Field(
        default=10, ge=1, le=100, description="Maximum number of results"
    )
    offset: int = Field(default=0, ge=0, description="Number of results to skip")

    # Filtering
    filters: Optional[MetadataFilter] = None

    # Search configuration
    min_score: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Minimum relevance score"
    )
    sort_by: SortOrder = Field(default=SortOrder.RELEVANCE)

    @field_validator("query")
    @classmethod
    def validate_query(cls, v):
        if not v.strip():
            raise ValueError("Query cannot be empty or only whitespace")
        return v.strip()


class SemanticSearchRequest(SearchRequest):
    """Semantic search request."""

    # Semantic search specific parameters
    similarity_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    include_embeddings: bool = Field(default=False)


class HybridSearchRequest(SearchRequest):
    """Hybrid search request combining semantic and keyword search."""

    # Weight configuration
    semantic_weight: float = Field(default=0.7, ge=0.0, le=1.0)
    keyword_weight: float = Field(default=0.3, ge=0.0, le=1.0)

    # Search parameters
    similarity_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    enable_rerank: bool = Field(default=True)
    rerank_top_k: int = Field(default=50, ge=1, le=200)

    @field_validator("semantic_weight", "keyword_weight")
    @classmethod
    def validate_weights(cls, v, info):
        # Weights should sum to 1.0 (approximately)
        if info.data and "semantic_weight" in info.data:
            total = info.data["semantic_weight"] + v
            if abs(total - 1.0) > 0.01:
                raise ValueError("Semantic and keyword weights must sum to 1.0")
        return v


class SearchResult(BaseModel):
    """Individual search result."""

    # Document identifiers
    chunk_id: UUID
    document_id: UUID

    # Content
    text: str
    chunk_index: int

    # Relevance scoring
    score: float = Field(..., ge=0.0, le=1.0)
    semantic_score: Optional[float] = None
    keyword_score: Optional[float] = None
    rerank_score: Optional[float] = None

    # Document metadata
    document_title: Optional[str] = None
    document_filename: str
    document_type: str
    page_number: Optional[int] = None

    # Additional metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)

    # Highlighting (for future use)
    highlights: Optional[List[str]] = None


class SearchResponse(BaseModel):
    """Search response with results and metadata."""

    # Query information
    query: str
    search_type: SearchType

    # Results
    results: List[SearchResult]
    total_results: int

    # Pagination
    limit: int
    offset: int
    has_next: bool

    # Performance metrics
    search_time_ms: float
    embedding_time_ms: Optional[float] = None
    rerank_time_ms: Optional[float] = None

    # Search metadata
    filters_applied: Optional[MetadataFilter] = None
    parameters: Dict[str, Any] = Field(default_factory=dict)


class SimilarDocumentsRequest(BaseModel):
    """Request for finding similar documents."""

    document_id: UUID = Field(..., description="Reference document ID")
    limit: int = Field(default=10, ge=1, le=50)
    similarity_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    exclude_same_document: bool = Field(default=True)

    # Filtering
    filters: Optional[MetadataFilter] = None


class QuerySuggestion(BaseModel):
    """Query suggestion model."""

    suggestion: str
    score: float
    type: str  # "completion", "correction", "related"
    metadata: Dict[str, Any] = Field(default_factory=dict)


class QuerySuggestionsRequest(BaseModel):
    """Query suggestions request."""

    partial_query: str = Field(..., min_length=1, max_length=100)
    limit: int = Field(default=5, ge=1, le=20)
    suggestion_types: List[str] = Field(default=["completion", "correction"])


class QuerySuggestionsResponse(BaseModel):
    """Query suggestions response."""

    query: str
    suggestions: List[QuerySuggestion]
    generation_time_ms: float


class ConversationContext(BaseModel):
    """Conversation context for search."""

    session_id: str
    previous_queries: List[str] = Field(default_factory=list, max_items=10)
    context_summary: Optional[str] = None
    topics: List[str] = Field(default_factory=list)

    # Context metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class ContextualSearchRequest(SearchRequest):
    """Search request with conversation context."""

    session_id: Optional[str] = None
    context: Optional[ConversationContext] = None
    use_context: bool = Field(default=True)
    context_weight: float = Field(default=0.2, ge=0.0, le=0.5)


class SearchAnalytics(BaseModel):
    """Search analytics data."""

    query: str
    search_type: SearchType
    user_id: Optional[str] = None
    session_id: Optional[str] = None

    # Results metadata
    total_results: int
    clicked_results: List[int] = Field(default_factory=list)  # Result indices

    # Performance metrics
    search_time_ms: float
    user_satisfaction: Optional[float] = None  # 0-1 rating

    # Timestamp
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class SearchCache(BaseModel):
    """Search result cache entry."""

    query_hash: str
    search_type: SearchType
    parameters_hash: str

    # Cached data
    results: List[SearchResult]
    total_results: int
    search_time_ms: float

    # Cache metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    ttl_seconds: int = Field(default=1800)  # 30 minutes
    hit_count: int = Field(default=0)

    def is_expired(self) -> bool:
        """Check if cache entry is expired."""
        age = (datetime.utcnow() - self.created_at).total_seconds()
        return age > self.ttl_seconds


class SearchConfigRequest(BaseModel):
    """Search configuration request."""

    # Model configurations
    embedding_model: Optional[str] = None
    rerank_model: Optional[str] = None

    # Default parameters
    default_limit: Optional[int] = Field(None, ge=1, le=100)
    default_similarity_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    default_semantic_weight: Optional[float] = Field(None, ge=0.0, le=1.0)

    # Cache settings
    enable_cache: Optional[bool] = None
    cache_ttl_seconds: Optional[int] = Field(None, ge=60, le=7200)

    # Performance settings
    max_concurrent_searches: Optional[int] = Field(None, ge=1, le=50)
    search_timeout_seconds: Optional[float] = Field(None, ge=1.0, le=30.0)


class SearchStats(BaseModel):
    """Search system statistics."""

    # Query statistics
    total_queries: int
    queries_last_hour: int
    queries_last_day: int

    # Performance metrics
    avg_search_time_ms: float
    avg_results_per_query: float
    cache_hit_rate: float

    # Popular queries
    top_queries: List[str]

    # System health
    index_size: int
    total_documents: int
    last_index_update: datetime

    # Error rates
    error_rate_percent: float
    timeout_rate_percent: float
