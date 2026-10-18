"""Hugging Face Space: ask questions over documents and watch hybrid retrieval + reranking work."""

import html
import os
import re
import sys
import time
from pathlib import Path

import gradio as gr

HERE = Path(__file__).resolve().parent
for candidate in (HERE / "src", HERE.parent / "src"):  # works in the Space and in the repo
    if candidate.exists():
        sys.path.insert(0, str(candidate))

from rag import RAG, Settings  # noqa: E402
from rag.index import Index  # noqa: E402
from rag.retrieve import Retriever  # noqa: E402

CORPUS = next(p for p in (HERE / "sample_corpus", HERE.parent / "data" / "sample_corpus") if p.exists())
EXAMPLES = [
    "What is the refund window for annual plans?",
    "When can I fly business class?",
    "How soon must a security incident be reported?",
    "What service credit do I get if uptime falls below 95 percent?",
    "How long is raw event data kept on the Business plan?",
    "What is the company's stock ticker symbol?",
]

settings = Settings(llm_threads=int(os.environ.get("LLM_THREADS", "2")))
BASE = RAG(settings)
BASE.add_folder(CORPUS)
BASE.llm  # load the GGUF at start-up, not on the first question  # noqa: B018
BASE.retrieve("warm up the embedder and the cross-encoder")


def session_rag(files) -> RAG:
    """A RAG over only the uploaded files, sharing the already-loaded models with the base one."""
    rag = RAG.__new__(RAG)
    rag.s, rag._llm = BASE.s, BASE._llm
    rag.index = Index(BASE.index.embedder)
    rag.retriever = Retriever(
        rag.index, BASE.retriever.reranker, settings.candidates, settings.rerank_pool, settings.top_k, settings.rrf_k
    )
    for f in files:
        path = Path(f if isinstance(f, str) else f.name)
        rag.add_bytes(path.read_bytes(), path.name)
    return rag


def corpus_note(rag: RAG) -> str:
    docs = rag.index.documents()
    names = ", ".join(sorted(d["source"] for d in docs.values()))
    return f"**{len(docs)} documents · {len(rag.index)} chunks** — {names}"


def trace_html(retrieval) -> str:
    if not retrieval.hits:
        return "<p class='muted'>Nothing retrieved.</p>"
    rows = []
    for h in retrieval.hits:
        moved = h.fused_rank - h.rank
        arrow = (
            f"<span class='up'>▲{moved}</span>"
            if moved > 0
            else f"<span class='down'>▼{-moved}</span>"
            if moved < 0
            else "<span class='same'>=</span>"
        )
        score = h.rerank_score if h.rerank_score is not None else 0.0
        width = max(4, min(100, int((score + 10) * 5)))  # cross-encoder logits ≈ −10…+10
        where = html.escape(h.chunk.heading or (f"page {h.chunk.page}" if h.chunk.page else ""))
        snippet = html.escape(re.sub(r"\s+", " ", h.chunk.text)[:220])
        rows.append(
            f"<tr><td class='n'>[{h.rank}]</td>"
            f"<td><b>{html.escape(h.chunk.source)}</b><br><span class='muted'>{where}</span><div class='snip'>{snippet}…</div></td>"
            f"<td class='c'>{h.vector_rank or '–'}</td><td class='c'>{h.keyword_rank or '–'}</td>"
            f"<td class='c'>{h.fused_rank}</td><td class='c'>{arrow}</td>"
            f"<td><div class='bar'><i style='width:{width}%'></i></div><span class='muted'>{score:.2f}</span></td></tr>"
        )
    return (
        "<table class='trace'><thead><tr><th></th><th>Passage</th><th>Vector rank</th><th>BM25 rank</th>"
        "<th>Fused rank</th><th>Rerank</th><th>Cross-encoder score</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def linkify(text: str, n: int) -> str:
    return re.sub(
        r"\[(\d+)\]",
        lambda m: f"<sup class='cite'>{m.group(1)}</sup>" if int(m.group(1)) <= n else m.group(0),
        html.escape(text),
    )


def ask(question: str, files_rag):
    question = (question or "").strip()
    if len(question) < 3:
        yield "<p class='muted'>Type a question first.</p>", "", ""
        return
    rag = files_rag or BASE
    t0 = time.perf_counter()
    answer_text, retrieval = "", None
    for kind, value in rag.ask_stream(question):
        if kind == "retrieval":
            retrieval = value
            yield "<p class='muted'>Writing the answer…</p>", trace_html(value), ""
        elif kind == "token":
            answer_text += value
            yield f"<div class='answer'>{linkify(answer_text, len(retrieval.hits))}▍</div>", trace_html(retrieval), ""
        else:
            a = value
            sources = "".join(
                f"<li><sup class='cite'>{a.retrieval.hits.index(h) + 1}</sup> {html.escape(h.chunk.label)}</li>"
                for h in a.cited
            )
            body = f"<div class='answer'>{linkify(a.text, len(a.retrieval.hits))}</div>"
            if sources:
                body += f"<ul class='sources'>{sources}</ul>"
            t = a.timings
            note = (
                "refused: the best passage scored below the relevance threshold, so the LLM was not called"
                if a.refused and "generation_s" not in t
                else ""
            )
            parts = [f"retrieval {1000 * t.get('retrieval_s', 0):.0f} ms"]
            if "first_token_s" in t:
                parts.append(f"first token {t['first_token_s']:.1f} s")
            parts.append(f"total {time.perf_counter() - t0:.1f} s")
            stats = " · ".join(parts + ([note] if note else []))
            yield body, trace_html(a.retrieval), stats


def use_files(files):
    if not files:
        return None, corpus_note(BASE)
    try:
        rag = session_rag(files)
    except ValueError as e:
        raise gr.Error(str(e)) from None
    return rag, corpus_note(rag)


CSS = (HERE / "style.css").read_text(encoding="utf-8")

with gr.Blocks(title="RAG System — Ask the docs") as demo:
    gr.HTML(
        "<header class='hd'><p class='kicker'>RAG System v2 · hybrid retrieval · cross-encoder rerank · cited answers</p>"
        "<h1>Ask the docs</h1><p class='lede'>Questions are answered only from the documents, with a citation for every fact. "
        "Below the answer you can see how each passage was found: dense vectors (bge-small) and BM25 are fused with "
        "Reciprocal Rank Fusion, then a cross-encoder reorders them. Runs on 2 CPU cores with Qwen3-1.7B (GGUF).</p>"
        "<p class='links'><a href='https://github.com/phoenixak/RAG-System' target='_blank'>Code & eval on GitHub ↗</a>"
        "<a href='https://ahmed.fynxforge.com' target='_blank'>Ahmed Khaled Ali ↗</a></p></header>"
    )
    files_rag = gr.State(None)
    with gr.Row(equal_height=False):
        with gr.Column(scale=5):
            q = gr.Textbox(label="Question", placeholder="e.g. What is the refund window for annual plans?", lines=2)
            go = gr.Button("Ask", variant="primary")
            gr.HTML("<p class='muted try'>Try one — the last question is not in the documents:</p>")
            gr.Examples(EXAMPLES, inputs=q, label=None)
            with gr.Accordion("Use your own documents (PDF, DOCX, MD, TXT, CSV)", open=False):
                up = gr.File(file_count="multiple", file_types=[".pdf", ".docx", ".md", ".txt", ".csv"], label="Upload")
                gr.Markdown("Uploads stay in your browser session only. Clear them to go back to the sample handbook.")
            corpus = gr.Markdown(corpus_note(BASE))
        with gr.Column(scale=7):
            answer = gr.HTML("<p class='muted'>The answer appears here.</p>", label="Answer")
            stats = gr.Markdown()
    gr.HTML("<h3 class='sec'>Retrieval trace</h3>")
    trace = gr.HTML("<p class='muted'>Ask something to see the ranking.</p>")
    gr.HTML(
        "<footer class='ft'>Sample corpus: a fictional company's handbook and policies written for this demo. "
        "Models: BAAI/bge-small-en-v1.5 (MIT), cross-encoder/ms-marco-MiniLM-L6-v2 (Apache-2.0), "
        "Qwen3-1.7B (Apache-2.0). First load after the Space sleeps takes about a minute.</footer>"
    )

    go.click(ask, [q, files_rag], [answer, trace, stats])
    q.submit(ask, [q, files_rag], [answer, trace, stats])
    up.change(use_files, up, [files_rag, corpus])

demo.queue(default_concurrency_limit=1, max_size=20)

if __name__ == "__main__":
    demo.launch(css=CSS, theme=gr.themes.Base())
