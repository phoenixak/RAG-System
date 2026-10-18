"""Fast, model-free fixtures: a hashing embedder, a keyword-overlap reranker and a scripted LLM."""

import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from rag import RAG, Settings  # noqa: E402
from rag.index import tokenize  # noqa: E402

CORPUS = ROOT / "data" / "sample_corpus"


class HashEmbedder:
    """Bag-of-words hashed into 256 dims: deterministic and good enough to test the plumbing."""

    dim = 256

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in tokenize(text):
            v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim] += 1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def embed_passages(self, texts):
        return np.vstack([self._vec(t) for t in texts]) if texts else np.zeros((0, self.dim), dtype=np.float32)

    def embed_query(self, text):
        return self._vec(text)

    def count_tokens(self, text):
        return len(text.split())


class OverlapReranker:
    def score(self, query, passages):
        q = set(tokenize(query))
        return [float(len(q & set(tokenize(p)))) - 1.0 for p in passages]


class ScriptedLLM:
    def __init__(self, reply="The refund window is 30 days [1]."):
        self.reply, self.calls = reply, 0

    def stream(self, messages):
        self.calls += 1
        for word in self.reply.split(" "):
            yield word + " "


@pytest.fixture
def settings():
    return Settings(llm_backend="none", min_relevance=0.0, chunk_tokens=80, chunk_overlap=15)


@pytest.fixture
def rag(settings):
    r = RAG(settings, embedder=HashEmbedder(), reranker=OverlapReranker(), llm=ScriptedLLM())
    r.add_folder(CORPUS)
    return r
