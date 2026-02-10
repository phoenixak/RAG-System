---
title: RAG System Demo
emoji: "\U0001F917"
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: "1.28.1"
app_file: app.py
pinned: false
---

# RAG System Demo

A fully functional Retrieval-Augmented Generation (RAG) system built with open-source Hugging Face models. Upload your documents, ask questions, and get AI-generated answers grounded in your content -- with full source attribution.

## What It Does

This demo implements a complete RAG pipeline:

1. **Document Ingestion** -- Upload PDF, TXT, DOCX, or CSV files. Text is extracted and split into overlapping chunks.
2. **Semantic Indexing** -- Chunks are embedded with a sentence-transformer model and stored in an in-memory ChromaDB vector store.
3. **Retrieval** -- When you ask a question, the most semantically similar chunks are retrieved using cosine similarity.
4. **Generation** -- Retrieved context is passed to a language model that generates a grounded answer.

## Features

- Multi-format document upload (PDF, TXT, DOCX, CSV)
- Semantic search with relevance scoring
- AI-powered question answering over your documents
- Source attribution with similarity scores
- Chat-style interface with conversation history
- Sample document included for quick testing

## Models Used

| Component | Model | Purpose |
|-----------|-------|---------|
| Text Generation | `google/flan-t5-small` | Instruction-following seq2seq model for Q&A |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` | Dense vector embeddings for semantic search |
| Vector Store | ChromaDB (in-memory) | Fast approximate nearest neighbor search |

## Main Repository

This Hugging Face Space is a live demo for the full RAG System project:

[https://github.com/Phoenixak99/RAG-System](https://github.com/Phoenixak99/RAG-System)

## Running Locally

```bash
# Clone the repository
git clone https://github.com/Phoenixak99/RAG-System.git
cd RAG-System/hf_space

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
```

Or use Docker:

```bash
docker build -t rag-demo .
docker run -p 8501:8501 rag-demo
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

## Architecture

```
User Query
    |
    v
[Embedding Model] --> Query Vector
    |
    v
[ChromaDB] --> Top-K Similar Chunks
    |
    v
[flan-t5-small] --> Generated Answer + Source Attribution
```

## License

MIT License -- see the [main repository](https://github.com/Phoenixak99/RAG-System) for details.
