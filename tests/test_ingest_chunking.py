import io

import pytest

from rag.chunking import chunk_document, dedupe
from rag.ingest import UnsupportedFile, load_bytes

words = lambda t: len(t.split())  # noqa: E731


def test_markdown_sections_keep_headings():
    doc = load_bytes(b"# Title\nintro\n\n## Leave\n25 days of annual leave.\n## Sick\n12 days.", "h.md")
    assert [s.heading for s in doc.sections] == ["Title", "Leave", "Sick"]
    assert doc.title == "Title"


def test_same_bytes_same_doc_id():
    a = load_bytes(b"hello world", "a.txt")
    b = load_bytes(b"hello world", "b.txt")
    assert a.doc_id == b.doc_id


def test_rejects_unsupported_and_empty():
    with pytest.raises(UnsupportedFile):
        load_bytes(b"x", "image.png")
    with pytest.raises(UnsupportedFile):
        load_bytes(b"   \n ", "empty.txt")


def test_csv_rows_become_keyed_lines():
    doc = load_bytes(b"plan,price\nStarter,49\nBusiness,18", "plans.csv")
    assert "plan: Starter; price: 49" in doc.text


def test_docx_headings_and_paragraphs():
    docx = pytest.importorskip("docx")
    d = docx.Document()
    d.add_heading("Refunds", level=1)
    d.add_paragraph("Annual plans can be refunded within 30 days.")
    buf = io.BytesIO()
    d.save(buf)
    doc = load_bytes(buf.getvalue(), "policy.docx")
    assert doc.sections[0].heading == "Refunds"
    assert "30 days" in doc.sections[0].text


def test_chunks_respect_the_token_limit_and_overlap():
    text = " ".join(f"Sentence number {i} talks about leave." for i in range(60))
    doc = load_bytes(f"## Leave\n{text}".encode(), "long.md")
    chunks = chunk_document(doc, words, chunk_tokens=50, overlap=10)
    assert len(chunks) > 3
    assert all(c.n_tokens <= 50 + 2 for c in chunks)  # +heading line
    first_tail = chunks[0].text.split()[-4:]
    assert " ".join(first_tail) in chunks[1].text  # overlap carried forward
    assert all(c.heading == "Leave" for c in chunks)


def test_chunks_never_cross_sections():
    doc = load_bytes(b"## A\nalpha alpha alpha\n## B\nbeta beta beta", "x.md")
    chunks = chunk_document(doc, words, chunk_tokens=100, overlap=0)
    assert [c.heading for c in chunks] == ["A", "B"]


def test_dedupe_drops_repeated_text():
    doc = load_bytes(b"## A\nsame words here\n## A\nsame words here", "dup.md")
    chunks = chunk_document(doc, words, chunk_tokens=100, overlap=0)
    assert len(dedupe(chunks)) == 1
