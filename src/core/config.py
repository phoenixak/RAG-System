"""
Configuration Management System
Handles environment-based settings and application configuration.
"""

from typing import List, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    # Application Settings
    app_name: str = "Enterprise RAG System"
    app_version: str = "1.0.0"
    debug: bool = Field(default=False, env="DEBUG")
    environment: str = Field(default="development", env="ENVIRONMENT")

    # API Settings
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    api_prefix: str = "/api/v1"

    # Security Settings
    secret_key: str = Field(env="SECRET_KEY")
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 30  # Restored to production value
    refresh_token_expire_days: int = 7

    # Database Settings
    database_url: Optional[str] = Field(env="DATABASE_URL")
    redis_url: str = Field(default="redis://localhost:6379", env="REDIS_URL")

    # ChromaDB Settings
    chromadb_host: str = Field(default="localhost", env="CHROMADB_HOST")
    chromadb_port: int = Field(default=8000, env="CHROMADB_PORT")
    chromadb_collection: str = "enterprise_documents"

    # Embedding Settings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimensions: int = 384
    max_sequence_length: int = 256

    # Text Processing Settings
    chunk_size: int = 1000
    chunk_overlap: int = 200
    chunk_separators: List[str] = ["\n\n", "\n", " ", ""]

    # File Upload Settings
    max_file_size: int = 50 * 1024 * 1024  # 50MB
    allowed_file_types: List[str] = [".pdf", ".txt", ".docx", ".csv"]
    upload_path: str = "uploads/"

    # Rate Limiting Settings
    rate_limit_anonymous: str = "10/minute"
    rate_limit_authenticated: str = "100/minute"
    rate_limit_premium: str = "1000/minute"

    # CORS Settings
    cors_origins: List[str] = ["http://localhost:3000", "http://localhost:8501"]
    cors_credentials: bool = True
    cors_methods: List[str] = ["GET", "POST", "PUT", "DELETE"]
    cors_headers: List[str] = ["*"]

    # Logging Settings
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_format: str = "json"

    # Cache Settings
    cache_ttl_embeddings: int = 3600  # 1 hour
    cache_ttl_search_results: int = 1800  # 30 minutes
    cache_ttl_user_sessions: int = 7200  # 2 hours
    cache_max_size: str = "2GB"

    # LLM Settings
    llm_provider: str = Field(default="openai", env="LLM_PROVIDER")
    llm_model: str = Field(default="gpt-3.5-turbo", env="LLM_MODEL")
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(default=None, env="ANTHROPIC_API_KEY")
    llm_max_tokens: int = Field(default=1000, env="LLM_MAX_TOKENS")
    llm_temperature: float = Field(default=0.7, env="LLM_TEMPERATURE")

    @field_validator("secret_key")
    @classmethod
    def validate_secret_key(cls, v):
        if not v:
            raise ValueError("SECRET_KEY environment variable is required")
        if len(v) < 32:
            raise ValueError("SECRET_KEY must be at least 32 characters long")
        return v

    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v):
        allowed_envs = ["development", "staging", "production", "testing"]
        if v not in allowed_envs:
            raise ValueError(f"Environment must be one of {allowed_envs}")
        return v

    @field_validator("allowed_file_types")
    @classmethod
    def validate_file_types(cls, v):
        return [ext.lower() for ext in v]

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get application settings instance."""
    return settings
