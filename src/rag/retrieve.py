"""Hybrid retrieval: dense + BM25, fused with Reciprocal Rank Fusion, then a cross-encoder rerank."""

from dataclasses import dataclass, field
from typing import Protocol

from .chunking import Chunk
from .index import Index


class Reranker(Protocol):
    def score(self, query: str, passages: list[str]) -> list[float]: ...


@dataclass
class Hit:
    chunk: Chunk
    vector_rank: int | None = None
    vector_score: float | None = None
    keyword_rank: int | None = None
    keyword_score: float | None = None
    fused_rank: int = 0
    fused_score: float = 0.0
    rerank_score: float | None = None
    rank: int = 0  # final position (1-based)


@dataclass
class Retrieval:
    query: str
    hits: list[Hit] = field(default_factory=list)  # final order, top_k
    candidates: list[Hit] = field(default_factory=list)  # fused order, before reranking

    @property
    def confidence(self) -> float | None:
        return self.hits[0].rerank_score if self.hits else None


def rrf(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion: sum of 1 / (k + rank) over every list a document appears in."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for pos, cid in enumerate(ranking, start=1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + pos)
    return sorted(scores.items(), key=lambda kv: -kv[1])


class Retriever:
    def __init__(
        self,
        index: Index,
        reranker: Reranker | None = None,
        candidates: int = 20,
        rerank_pool: int = 12,
        top_k: int = 5,
        rrf_k: int = 60,
        mode: str = "hybrid",  # "hybrid" | "vector" | "keyword"
    ) -> None:
        self.index, self.reranker = index, reranker
        self.candidates, self.rerank_pool, self.top_k, self.rrf_k, self.mode = (
            candidates,
            rerank_pool,
            top_k,
            rrf_k,
            mode,
        )

    def search(self, query: str) -> Retrieval:
        vec = self.index.vector_search(query, self.candidates) if self.mode in ("hybrid", "vector") else []
        kw = self.index.keyword_search(query, self.candidates) if self.mode in ("hybrid", "keyword") else []
        hits: dict[str, Hit] = {}
        for pos, (cid, s) in enumerate(vec, start=1):
            hits.setdefault(cid, Hit(self.index.chunks[cid])).__dict__.update(vector_rank=pos, vector_score=s)
        for pos, (cid, s) in enumerate(kw, start=1):
            hits.setdefault(cid, Hit(self.index.chunks[cid])).__dict__.update(keyword_rank=pos, keyword_score=s)

        fused = rrf([[c for c, _ in vec], [c for c, _ in kw]], self.rrf_k)
        candidates = []
        for pos, (cid, s) in enumerate(fused, start=1):
            h = hits[cid]
            h.fused_rank, h.fused_score = pos, s
            candidates.append(h)

        final = candidates[: self.rerank_pool]
        if self.reranker and final:
            scores = self.reranker.score(query, [h.chunk.text for h in final])
            for h, s in zip(final, scores):
                h.rerank_score = float(s)
            final = sorted(final, key=lambda h: -h.rerank_score)
        final = final[: self.top_k]
        for pos, h in enumerate(final, start=1):
            h.rank = pos
        return Retrieval(query=query, hits=final, candidates=candidates[: self.rerank_pool])
