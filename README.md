# RAG System v2 — grounded answers with citations

[![CI](https://github.com/phoenixak/RAG-System/actions/workflows/ci.yml/badge.svg)](https://github.com/phoenixak/RAG-System/actions/workflows/ci.yml)
[![Live demo](https://img.shields.io/badge/%F0%9F%A4%97%20Live%20demo-Hugging%20Face-F4F1EA?labelColor=141310)](https://huggingface.co/spaces/Phoenixak99/RAG-System-Demo)
![Python](https://img.shields.io/badge/python-3.11%2B-141310)
![License](https://img.shields.io/badge/license-MIT-141310)

Ask questions over your documents and get short answers with a citation for every fact — or an honest "I couldn't find that in the documents." Everything runs on CPU: a 1.7B model through llama.cpp answers in about 2 seconds on two cores.

**[Try the live demo →](https://huggingface.co/spaces/Phoenixak99/RAG-System-Demo)** It shows the answer *and* the retrieval trace: where each passage ranked in dense search, BM25 and fusion, and how far the cross-encoder moved it.

## How it works

```
 upload ─▶ ingest ─▶ chunk ─────────────▶ index ──────────────┐
 PDF, DOCX,  sections   token-aware,      bge-small vectors     │
 MD, TXT,    + headings  200 tokens,       (memory / ChromaDB)   │
 CSV                     40 overlap,       + BM25                │
                         dedup                                   ▼
 question ─────────────────────────────▶ dense top-20 ─┐   Reciprocal
                                          BM25 top-20 ──┴─▶ Rank Fusion ─▶ cross-encoder
                                                                         rerank (top 12)
                                                                               │
                         best score < threshold? ── yes ─▶ "not in the documents"
                                                  │                      (LLM never called)
                                                  no
                                                  ▼
                              Qwen3-1.7B (GGUF, llama.cpp) ─▶ streamed answer with [n] citations
```

| Stage | Choice | Why |
|---|---|---|
| Embeddings | `BAAI/bge-small-en-v1.5` (33M, MIT) | Strong retrieval quality per CPU millisecond; query instruction prefix |
| Keyword search | BM25 (`rank-bm25`) with a tokenizer that keeps numbers like `99.9` and `10:00` intact | Policies are full of exact numbers that dense vectors blur |
| Fusion | Reciprocal Rank Fusion (k = 60) | Needs no score calibration between the two retrievers |
| Reranking | `cross-encoder/ms-marco-MiniLM-L6-v2` (Apache-2.0) over the top 12 | Reads query and passage together; fixes the cases fusion gets wrong |
| Refusals | Threshold on the best cross-encoder score, plus an instruction to the LLM | Cheap first gate, model as the second |
| Generation | `Qwen3-1.7B` Q4_K_M via llama.cpp (Apache-2.0), thinking mode off; OpenAI-compatible and Anthropic backends optional | Runs on a free 2-vCPU Space; swap in a bigger model with one env var |

## Evaluation

`eval/` holds a golden set of **42 answerable and 4 unanswerable questions** over the sample corpus (a fictional company handbook written for this project). Reproduce with `python eval/run_eval.py --generate` — the numbers below are from 2 CPU threads, the same as the free Space.

**Retrieval — does the right document come back?**

| Retriever | Hit@1 | Recall@5 | MRR | p50 latency |
|---|---:|---:|---:|---:|
| Vector only (bge-small) | 88% | 100% | 0.929 | 8 ms |
| BM25 only | 93% | 100% | 0.960 | <1 ms |
| Hybrid (RRF) | 90% | 100% | 0.952 | 8 ms |
| **Hybrid + cross-encoder rerank** | **98%** | 100% | **0.988** | 108 ms |

The corpus has deliberate distractors (the product FAQ also summarises the refund policy; "30 days" appears in four documents). Fusion alone doesn't beat BM25 here; the reranker is what puts the right passage first.

**Answers — Qwen3-1.7B, CPU, 2 threads**

| Metric | Value |
|---|---:|
| Answer contains the expected fact | 95% |
| Every citation points to the right document | 95% |
| Refused when the answer is not in the docs | 100% |
| False refusals | 2% (1 of 42) |
| Median time to first token / full answer | 1.6 s / 2.3 s |

Full per-question output: [`eval/results.md`](eval/results.md). Model choice was driven by this eval: Qwen2.5-1.5B with the same retrieval scored 38% on facts and 64% on citations (it often answered with a bare `[1]`); restructuring the prompt and moving to Qwen3-1.7B fixed both.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install llama-cpp-python          # prebuilt wheel on Windows/macOS; builds from source on Linux
pip install -e ".[api,demo,dev]"

python space/app.py                       # the demo UI on http://127.0.0.1:7860
python eval/run_eval.py --generate        # reproduce the tables above
pytest -q                                 # 26 tests, no model downloads needed
```

As a library:

```python
from rag import RAG

rag = RAG()                       # settings from RAG_* env vars, see .env.example
rag.add_folder("data/sample_corpus")
answer = rag.ask("What is the refund window for annual plans?")
print(answer.text)                # Annual plans can be refunded in full within 30 days … [1]
print([h.chunk.label for h in answer.cited])
```

## API

FastAPI service with JWT auth, four roles and per-user rate limits (`docker compose up` or `uvicorn api.app:app --app-dir src`):

| Endpoint | Role needed | What it does |
|---|---|---|
| `POST /auth/token` | — | Exchange username/password for a 30-minute JWT |
| `GET /health` | — | Status, document and chunk counts |
| `GET /documents` | read_only | Indexed documents |
| `POST /search` | read_only | Ranked passages with every score (vector, BM25, fused, rerank) |
| `POST /ask` | read_only | Cited answer; `"stream": true` returns Server-Sent Events (`retrieval`, `token`…, `answer`) |
| `POST /documents` | standard_user | Upload PDF/DOCX/MD/TXT/CSV; re-uploads are detected by content hash |
| `DELETE /documents/{id}` | power_user | Remove a document and its chunks |
| `GET /admin/users` | admin | Configured users and roles |

Users are configured in `API_USERS` with bcrypt hashes (`python -m api.auth "password"` prints one). With `RAG_PERSIST_DIR` set (automatic in Docker), the index lives in ChromaDB and survives restarts.

## Project layout

```
src/rag/        ingest · chunking · index (vectors + BM25) · retrieve (RRF + rerank) · generate · pipeline
src/api/        FastAPI app, JWT + roles
space/          Hugging Face Space (Gradio) — deployed with space/deploy.py
eval/           golden set + evaluation script + results
tests/          26 tests with small fake models (run in CI in seconds)
data/           sample corpus (fictional)
```

## What changed from v1

v1 (tagged [`v1`](https://github.com/phoenixak/RAG-System/tree/v1)) was a larger FastAPI + Streamlit app with Redis, PostgreSQL wiring and answers from `flan-t5-small`; its demo Space could no longer be scheduled. v2 keeps the good parts — hybrid search, reranking, JWT roles, rate limiting, dedup — and adds what v1 lacked: an evaluation harness with real numbers, cited answers from a modern small LLM, refusals when the answer isn't there, streaming, a demo that shows the retrieval at work, and tests that run in CI. It does this in about 1,400 lines of Python, against roughly 17,000 in v1.

## License

MIT. Models: bge-small-en-v1.5 (MIT), ms-marco-MiniLM-L6-v2 (Apache-2.0), Qwen3-1.7B (Apache-2.0). The sample corpus is fictional and written for this repository.
