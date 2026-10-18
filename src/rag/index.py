"""The searchable index: dense vectors (in memory or ChromaDB) plus a BM25 keyword index."""

import re
from typing import Protocol

import numpy as np
from rank_bm25 import BM25Okapi

from .chunking import Chunk

_STOP = set(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "can",
        "do",
        "does",
        "for",
        "from",
        "has",
        "have",
        "how",
        "i",
        "if",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "our",
        "so",
        "than",
        "that",
        "the",
        "their",
        "them",
        "then",
        "there",
        "these",
        "this",
        "to",
        "was",
        "we",
        "what",
        "when",
        "where",
        "which",
        "who",
        "will",
        "with",
        "you",
        "your",
        "my",
        "me",
    ]
)


def tokenize(text: str) -> list[str]:
    """Lower-cased word and number tokens without stop words ("99.9", "10:00" stay intact)."""
    return [t for t in re.findall(r"\d+(?:[.,:]\d+)*%?|[a-z][a-z'-]*", text.lower()) if t not in _STOP]


class Embedder(Protocol):
    def embed_passages(self, texts: list[str]) -> np.ndarray: ...
    def embed_query(self, text: str) -> np.ndarray: ...
    def count_tokens(self, text: str) -> int: ...


class VectorStore(Protocol):
    def add(self, ids: list[str], vectors: np.ndarray) -> None: ...
    def remove(self, ids: list[str]) -> None: ...
    def search(self, vector: np.ndarray, k: int) -> list[tuple[str, float]]: ...


class MemoryVectorStore:
    """Exact cosine search over normalised vectors. Fast enough for tens of thousands of chunks."""

    def __init__(self) -> None:
        self.ids: list[str] = []
        self.matrix: np.ndarray | None = None

    def add(self, ids: list[str], vectors: np.ndarray) -> None:
        self.ids.extend(ids)
        self.matrix = vectors if self.matrix is None else np.vstack([self.matrix, vectors])

    def remove(self, ids: list[str]) -> None:
        drop = set(ids)
        keep = [i for i, cid in enumerate(self.ids) if cid not in drop]
        self.ids = [self.ids[i] for i in keep]
        self.matrix = self.matrix[keep] if self.matrix is not None and keep else None

    def search(self, vector: np.ndarray, k: int) -> list[tuple[str, float]]:
        if self.matrix is None:
            return []
        scores = self.matrix @ vector
        top = np.argsort(-scores)[:k]
        return [(self.ids[i], float(scores[i])) for i in top]


class ChromaVectorStore:
    """Persistent store for the API server (embeddings computed by us, cosine space)."""

    def __init__(self, path: str, collection: str = "chunks") -> None:
        import chromadb

        self.client = chromadb.PersistentClient(path=path)
        self.col = self.client.get_or_create_collection(collection, metadata={"hnsw:space": "cosine"})

    def add(self, ids: list[str], vectors: np.ndarray) -> None:
        self.col.upsert(ids=ids, embeddings=vectors.tolist())

    def remove(self, ids: list[str]) -> None:
        if ids:
            self.col.delete(ids=ids)

    def search(self, vector: np.ndarray, k: int) -> list[tuple[str, float]]:
        n = self.col.count()
        if not n:
            return []
        res = self.col.query(query_embeddings=[vector.tolist()], n_results=min(k, n))
        return [(cid, 1.0 - float(d)) for cid, d in zip(res["ids"][0], res["distances"][0])]


class Index:
    def __init__(self, embedder: Embedder, store: VectorStore | None = None) -> None:
        self.embedder = embedder
        self.store = store or MemoryVectorStore()
        self.chunks: dict[str, Chunk] = {}
        self._bm25: BM25Okapi | None = None
        self._bm25_ids: list[str] = []

    def __len__(self) -> int:
        return len(self.chunks)

    def documents(self) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for c in self.chunks.values():
            d = out.setdefault(c.doc_id, {"doc_id": c.doc_id, "source": c.source, "chunks": 0})
            d["chunks"] += 1
        return out

    def add(self, chunks: list[Chunk]) -> int:
        new = [c for c in chunks if c.chunk_id not in self.chunks]
        if not new:
            return 0
        vectors = self.embedder.embed_passages([c.text for c in new])
        self.store.add([c.chunk_id for c in new], vectors)
        for c in new:
            self.chunks[c.chunk_id] = c
        self._bm25 = None
        return len(new)

    def remove_document(self, doc_id: str) -> int:
        ids = [cid for cid, c in self.chunks.items() if c.doc_id == doc_id]
        self.store.remove(ids)
        for cid in ids:
            del self.chunks[cid]
        self._bm25 = None
        return len(ids)

    def has_document(self, doc_id: str) -> bool:
        return any(c.doc_id == doc_id for c in self.chunks.values())

    def vector_search(self, query: str, k: int) -> list[tuple[str, float]]:
        return [(cid, s) for cid, s in self.store.search(self.embedder.embed_query(query), k) if cid in self.chunks]

    def keyword_search(self, query: str, k: int) -> list[tuple[str, float]]:
        if not self.chunks:
            return []
        if self._bm25 is None:  # rebuilt lazily after any change; cheap at this scale
            self._bm25_ids = list(self.chunks)
            self._bm25 = BM25Okapi([tokenize(self.chunks[c].text) for c in self._bm25_ids])
        terms = tokenize(query)
        if not terms:
            return []
        scores = self._bm25.get_scores(terms)
        top = np.argsort(-scores)[:k]
        return [(self._bm25_ids[i], float(scores[i])) for i in top if scores[i] > 0]
