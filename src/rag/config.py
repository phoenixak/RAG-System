"""Settings for the RAG pipeline. Every field can be overridden with an RAG_* environment variable."""

from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RAG_", env_file=".env", extra="ignore")

    # Retrieval
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    # bge models expect this instruction in front of queries (not in front of passages)
    query_instruction: str = "Represent this sentence for searching relevant passages: "
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L6-v2"
    use_reranker: bool = True
    chunk_tokens: int = 200
    chunk_overlap: int = 40
    candidates: int = 20  # taken from each retriever before fusion
    rerank_pool: int = 12  # fused candidates the cross-encoder re-scores
    top_k: int = 5  # chunks passed to the LLM
    rrf_k: int = 60
    # Below this cross-encoder score the best chunk is judged irrelevant and the
    # question is answered with a refusal instead of calling the LLM.
    min_relevance: float = -4.0
    persist_dir: str | None = None  # ChromaDB directory; None keeps the index in memory

    # Generation
    llm_backend: Literal["llamacpp", "openai", "anthropic", "none"] = "llamacpp"
    llm_repo: str = "unsloth/Qwen3-1.7B-GGUF"
    llm_file: str = "Qwen3-1.7B-Q4_K_M.gguf"
    llm_ctx: int = 4096
    llm_threads: int | None = None
    max_new_tokens: int = 160
    temperature: float = 0.1
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o-mini"
    openai_api_key: str | None = None
    anthropic_model: str = "claude-haiku-5-5"
    anthropic_api_key: str | None = None


REFUSAL = "I couldn't find that in the documents."
