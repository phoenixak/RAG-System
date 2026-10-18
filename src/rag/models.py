"""Sentence-transformers wrappers for the embedder and the cross-encoder reranker (loaded lazily)."""

from functools import cached_property

import numpy as np


class STEmbedder:
    def __init__(self, model_name: str, query_instruction: str = "", batch_size: int = 32) -> None:
        self.model_name, self.query_instruction, self.batch_size = model_name, query_instruction, batch_size

    @cached_property
    def model(self):
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(self.model_name, device="cpu")

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, batch_size=self.batch_size, normalize_embeddings=True, convert_to_numpy=True)

    def embed_query(self, text: str) -> np.ndarray:
        return self.model.encode(self.query_instruction + text, normalize_embeddings=True, convert_to_numpy=True)

    def count_tokens(self, text: str) -> int:
        return len(self.model.tokenizer.tokenize(text))


class CrossEncoderReranker:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    @cached_property
    def model(self):
        from sentence_transformers import CrossEncoder

        return CrossEncoder(self.model_name, device="cpu")

    def score(self, query: str, passages: list[str]) -> list[float]:
        return self.model.predict([(query, p) for p in passages], batch_size=16).tolist()
