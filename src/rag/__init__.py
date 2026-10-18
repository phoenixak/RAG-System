"""Grounded document Q&A: hybrid retrieval, cross-encoder reranking and cited answers."""

from .config import REFUSAL, Settings
from .pipeline import RAG, Answer
from .retrieve import Hit, Retrieval

__all__ = ["RAG", "Answer", "Hit", "Retrieval", "Settings", "REFUSAL"]
__version__ = "2.0.0"
