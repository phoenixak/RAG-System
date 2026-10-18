"""Token-aware recursive chunking with overlap and exact-duplicate removal."""

import hashlib
import re
from collections.abc import Callable
from dataclasses import dataclass

from .ingest import Document

SEPARATORS = ["\n\n", "\n", ". ", "; ", ", ", " "]


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    source: str
    heading: str | None
    page: int | None
    text: str
    n_tokens: int

    @property
    def label(self) -> str:
        """Where the chunk came from, as shown next to a citation."""
        where = f"p. {self.page}" if self.page else (self.heading or "")
        return f"{self.source} · {where}" if where else self.source


def _split(text: str, count: Callable[[str], int], limit: int, seps: list[str]) -> list[str]:
    if count(text) <= limit:
        return [text]
    if not seps:  # no separator left: hard cut by words
        words = text.split()
        step = max(1, len(words) // max(1, (count(text) // limit) + 1))
        return [" ".join(words[i : i + step]) for i in range(0, len(words), step)]
    sep, rest = seps[0], seps[1:]
    parts = [p for p in text.split(sep) if p.strip()]
    if len(parts) == 1:
        return _split(text, count, limit, rest)
    pieces: list[str] = []
    for i, part in enumerate(parts):
        piece = part + (sep.rstrip() if i < len(parts) - 1 and sep.strip() else "")
        pieces.extend(_split(piece, count, limit, rest))
    return pieces


def chunk_document(
    doc: Document,
    count_tokens: Callable[[str], int],
    chunk_tokens: int = 200,
    overlap: int = 40,
) -> list[Chunk]:
    """Pack sentence-ish pieces into chunks of at most `chunk_tokens`, never across sections."""
    chunks: list[Chunk] = []
    for sec in doc.sections:
        text = re.sub(r"[ \t]+", " ", sec.text).strip()
        pieces = _split(text, count_tokens, chunk_tokens, SEPARATORS)
        current: list[str] = []
        size = 0
        for piece in pieces:
            n = count_tokens(piece)
            if current and size + n > chunk_tokens:
                chunks.append(_make(doc, sec.heading, sec.page, current, count_tokens))
                # carry the tail of the previous chunk forward as overlap
                tail, t = [], 0
                for prev in reversed(current):
                    t += count_tokens(prev)
                    if t > overlap:
                        break
                    tail.insert(0, prev)
                current, size = tail, sum(count_tokens(x) for x in tail)
            current.append(piece)
            size += n
        if current:
            chunks.append(_make(doc, sec.heading, sec.page, current, count_tokens))
    return dedupe(chunks)


def _make(doc: Document, heading, page, parts: list[str], count) -> Chunk:
    body = " ".join(p.strip() for p in parts).strip()
    # the section heading is part of the text so both retrievers can match on it
    text = f"{heading}\n{body}" if heading else body
    cid = hashlib.sha1(f"{doc.doc_id}:{text}".encode()).hexdigest()[:16]
    return Chunk(cid, doc.doc_id, doc.source, heading, page, text, count(text))


def _norm(text: str) -> str:
    return re.sub(r"\W+", " ", text.lower()).strip()


def dedupe(chunks: list[Chunk]) -> list[Chunk]:
    """Drop chunks whose normalised text was already seen (repeated boilerplate, re-uploads)."""
    seen, out = set(), []
    for c in chunks:
        h = hashlib.sha1(_norm(c.text).encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            out.append(c)
    return out
