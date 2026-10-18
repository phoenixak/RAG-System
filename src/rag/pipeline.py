"""The end-to-end pipeline: ingest → chunk → index → retrieve → rerank → grounded, cited answer."""

import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from .chunking import chunk_document
from .config import REFUSAL, Settings
from .generate import LLM, build_messages, citations, make_llm, tidy
from .index import ChromaVectorStore, Embedder, Index
from .ingest import Document, load_bytes, load_path
from .retrieve import Hit, Reranker, Retrieval, Retriever


@dataclass
class Answer:
    question: str
    text: str
    retrieval: Retrieval
    cited: list[Hit] = field(default_factory=list)
    refused: bool = False
    timings: dict[str, float] = field(default_factory=dict)


class RAG:
    def __init__(
        self,
        settings: Settings | None = None,
        embedder: Embedder | None = None,
        reranker: Reranker | None = None,
        llm: LLM | None = None,
    ) -> None:
        self.s = s = settings or Settings()
        if embedder is None:
            from .models import STEmbedder

            embedder = STEmbedder(s.embedding_model, s.query_instruction)
        if reranker is None and s.use_reranker:
            from .models import CrossEncoderReranker

            reranker = CrossEncoderReranker(s.reranker_model)
        store = ChromaVectorStore(s.persist_dir) if s.persist_dir else None
        self.index = Index(embedder, store)
        self.retriever = Retriever(self.index, reranker, s.candidates, s.rerank_pool, s.top_k, s.rrf_k)
        self._llm = llm

    @property
    def llm(self) -> LLM:
        if self._llm is None:  # loading the GGUF takes a few seconds, so only when first needed
            self._llm = make_llm(self.s)
        return self._llm

    # ── ingestion ─────────────────────────────────────────────────────────
    def add_document(self, doc: Document) -> int:
        chunks = chunk_document(doc, self.index.embedder.count_tokens, self.s.chunk_tokens, self.s.chunk_overlap)
        return self.index.add(chunks)

    def add_file(self, path: str | Path) -> Document:
        doc = load_path(path)
        self.add_document(doc)
        return doc

    def add_bytes(self, content: bytes, filename: str) -> tuple[Document, int]:
        doc = load_bytes(content, filename)
        return doc, self.add_document(doc)

    def add_folder(self, folder: str | Path) -> list[Document]:
        return [self.add_file(p) for p in sorted(Path(folder).iterdir()) if p.is_file()]

    # ── question answering ────────────────────────────────────────────────
    def retrieve(self, question: str) -> Retrieval:
        return self.retriever.search(question)

    def is_answerable(self, retrieval: Retrieval) -> bool:
        conf = retrieval.confidence
        return bool(retrieval.hits) and (conf is None or conf >= self.s.min_relevance)

    def ask_stream(self, question: str) -> Iterator[tuple[str, object]]:
        """Yields ("retrieval", Retrieval), then ("token", str)…, then ("answer", Answer)."""
        t0 = time.perf_counter()
        retrieval = self.retrieve(question)
        t_ret = time.perf_counter() - t0
        yield "retrieval", retrieval

        if not self.is_answerable(retrieval):
            yield "token", REFUSAL
            yield "answer", Answer(question, REFUSAL, retrieval, refused=True, timings={"retrieval_s": t_ret})
            return

        parts: list[str] = []
        t1 = time.perf_counter()
        first: float | None = None
        for tok in self.llm.stream(build_messages(question, retrieval.hits)):
            if first is None:
                first = time.perf_counter() - t1
            parts.append(tok)
            yield "token", tok
        text = tidy("".join(parts))
        refused = REFUSAL.lower().rstrip(".") in text.lower()
        cited = [retrieval.hits[n - 1] for n in citations(text, len(retrieval.hits))]
        timings = {"retrieval_s": t_ret, "first_token_s": first or 0.0, "generation_s": time.perf_counter() - t1}
        yield "answer", Answer(question, text, retrieval, cited, refused, timings)

    def ask(self, question: str) -> Answer:
        answer = None
        for kind, value in self.ask_stream(question):
            if kind == "answer":
                answer = value
        assert isinstance(answer, Answer)
        return answer
