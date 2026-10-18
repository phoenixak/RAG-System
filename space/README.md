---
title: RAG System Demo
emoji: 📚
colorFrom: red
colorTo: gray
sdk: gradio
sdk_version: 6.30.0
python_version: "3.11"
app_file: app.py
pinned: false
license: mit
short_description: Cited answers from your docs with hybrid search + reranking
---

# Ask the docs — RAG System v2

Ask questions over a document set and get short answers with a citation for every fact. The page also shows the **retrieval trace**: how each passage ranked in dense search (bge-small), BM25 and Reciprocal Rank Fusion, and where the cross-encoder moved it.

- **Retrieval:** `BAAI/bge-small-en-v1.5` embeddings + BM25 → Reciprocal Rank Fusion → `cross-encoder/ms-marco-MiniLM-L6-v2` rerank
- **Answers:** `Qwen3-1.7B` (GGUF Q4_K_M) through llama.cpp on 2 CPU cores, streamed
- **Refusals:** if the best passage scores below a threshold, the app answers "not in the documents" without calling the LLM
- **Sample data:** a fictional company handbook and policies written for this demo; you can upload your own PDF, DOCX, Markdown, TXT or CSV files (kept in your session only)

The first load after the Space has been asleep takes about a minute while the models load.

Code, API (FastAPI + JWT roles) and evaluation results: [github.com/phoenixak/RAG-System](https://github.com/phoenixak/RAG-System)
