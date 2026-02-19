# Enterprise RAG System

> Retrieval-Augmented Generation system — semantic document search and conversational AI over your private knowledge base.

[![Python](https://img.shields.io/badge/Python-3.9–3.11-blue?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-0.4-orange)](https://www.trychroma.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![HF Space](https://img.shields.io/badge/🤗%20Hugging%20Face-Live%20Demo-yellow)](https://huggingface.co/spaces/Phoenixak99/RAG-System-Demo)

---

## What It Does

Upload documents. Ask questions in plain English. Get answers grounded in your content — with full source attribution and relevance scores.

The system implements a complete RAG pipeline:

1. **Ingest** — Upload PDF, DOCX, TXT, or CSV files. Text is extracted and split into overlapping chunks with configurable size and overlap.
2. **Embed** — Each chunk is encoded into a dense vector using `sentence-transformers/all-MiniLM-L6-v2` and stored in ChromaDB.
3. **Retrieve** — Incoming queries are embedded and matched against stored chunks via semantic, hybrid (semantic + BM25), or contextual search.
4. **Generate** — Retrieved context is passed to an LLM (OpenAI / Anthropic) to produce a grounded, cited answer. Runs in demo mode (excerpt-based fallback) when no API key is configured.

---

## Live Demo

A self-contained demo (no backend, no API keys, no setup required) is live on Hugging Face Spaces:

**[huggingface.co/spaces/Phoenixak99/RAG-System-Demo](https://huggingface.co/spaces/Phoenixak99/RAG-System-Demo)**

The demo uses `google/flan-t5-small` for generation and `sentence-transformers/all-MiniLM-L6-v2` for embeddings — both running fully on open-source models. Upload any document, ask questions, get answers with source attribution.

Source: [`hf_space/app.py`](hf_space/app.py)

---

## Architecture

```
                    ┌─────────────────────────────────┐
                    │         Streamlit Frontend        │
                    │         localhost:8501            │
                    │  Chat · Documents · Search ·     │
                    │  Settings · Admin                 │
                    └──────────────┬──────────────────┘
                                   │ HTTP (REST)
                    ┌──────────────▼──────────────────┐
                    │        FastAPI Backend            │
                    │        localhost:8000             │
                    │  Auth · Documents · Search ·     │
                    │  Deduplication · Health          │
                    └──┬─────────────┬────────────────┘
                       │             │
          ┌────────────▼──┐   ┌─────▼──────────┐
          │   ChromaDB    │   │     Redis       │
          │  Vector Store │   │  Cache / BL     │
          │  port 8001    │   │  port 6379      │
          └───────────────┘   └────────────────┘
```

**ChromaDB and Redis are optional for local development.** Without them, the system falls back to embedded ChromaDB (stored in `./chroma_db/`) and in-memory caching automatically.

---

## Features

| Category | Capability |
|---|---|
| Document Processing | PDF, DOCX, TXT, CSV with dedicated extractors |
| Chunking | Recursive character splitter with tiktoken token counting, configurable size/overlap |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2`, batched with Redis caching |
| Search | Semantic (vector), Hybrid (vector + BM25), Contextual (conversation-aware) |
| Re-ranking | Cross-encoder re-ranking for improved result relevance |
| Deduplication | Content-hash-based duplicate detection and cleanup endpoints |
| LLM Integration | OpenAI & Anthropic; falls back to excerpt-based responses in demo mode |
| Auth | JWT with role-based access (admin, power_user, standard_user, read_only) |
| UI | Dark & technical Streamlit interface — chat, document management, search, admin |
| Observability | Structured JSON logging (structlog), request/response middleware, health probes |

---

## Tech Stack

| Layer | Technology |
|---|---|
| API Framework | FastAPI 0.104, Pydantic v2, Uvicorn |
| Frontend | Streamlit 1.28 |
| Vector Store | ChromaDB 0.4 |
| Embeddings | sentence-transformers ≥ 3.0 |
| Keyword Search | rank-bm25 |
| LLM Providers | openai, anthropic |
| Auth | PyJWT, bcrypt |
| Document Parsing | PyPDF2, pdfplumber, python-docx, pandas |
| Caching | Redis 7 (optional) |
| Tokenization | tiktoken |
| Logging | structlog, rich |
| Containerization | Docker, Docker Compose |
| Testing | pytest, pytest-asyncio, pytest-cov |

---

## Project Structure

```
RAG-System/
├── run.py                        # Unified launcher — backend, frontend, or both
├── pyproject.toml                # Dependencies and tool configuration
├── docker-compose.yml            # Service orchestration
├── Dockerfile                    # Container build (backend)
├── .env.example                  # Environment variable template
│
├── src/                          # Backend (FastAPI)
│   ├── api/
│   │   ├── auth.py               # Login, logout, token refresh, user profile
│   │   ├── documents.py          # Upload, list, get, delete, chunks, stats
│   │   ├── search.py             # Semantic, hybrid, contextual search; conversations; LLM
│   │   ├── deduplication.py      # Duplicate scan, cleanup, hash check
│   │   └── health.py             # Health, readiness, liveness probes
│   ├── auth/
│   │   ├── models.py             # User, role, permission, token models
│   │   ├── jwt_utils.py          # Token creation and verification
│   │   └── security.py           # Password hashing, current user dependency
│   ├── core/
│   │   ├── config.py             # Pydantic v2 settings with env var support
│   │   └── logging.py            # Structured logging setup
│   ├── documents/
│   │   ├── processors.py         # PDF, DOCX, TXT, CSV text extraction
│   │   ├── chunking.py           # TextChunker, RecursiveCharacterTextSplitter
│   │   ├── embeddings.py         # Batched embedding generation with caching
│   │   ├── deduplication.py      # Content-hash duplicate detection
│   │   ├── models.py             # Document and chunk data models
│   │   └── service.py            # Document processing orchestration
│   ├── search/
│   │   ├── semantic_search.py    # ChromaDB vector similarity search
│   │   ├── hybrid_search.py      # Combined semantic + BM25 scoring
│   │   ├── reranking.py          # Cross-encoder re-ranking
│   │   ├── conversation.py       # Session context management
│   │   ├── models.py             # Search request/response models
│   │   └── service.py            # Search service facade
│   ├── llm/
│   │   ├── service.py            # OpenAI/Anthropic client management
│   │   ├── prompts.py            # RAG prompt templates
│   │   └── models.py             # LLM request/response models
│   └── vector_store/
│       └── chroma_client.py      # ChromaDB connection and operations
│
├── frontend/                     # Frontend (Streamlit)
│   ├── components/
│   │   ├── api_client.py         # HTTP client for backend API
│   │   ├── auth.py               # Login UI and session management
│   │   ├── chat_ui.py            # Chat interface components
│   │   ├── styles.py             # Global dark & technical CSS theme
│   │   └── utils.py              # Shared frontend utilities
│   ├── pages/
│   │   ├── admin.py              # User and system administration
│   │   ├── chat.py               # Conversational RAG interface
│   │   ├── documents.py          # Document upload and management
│   │   ├── search.py             # Advanced search interface
│   │   └── settings.py           # User preferences
│   ├── config.py                 # Frontend configuration constants
│   └── .streamlit/config.toml   # Streamlit server and dark theme config
│
├── hf_space/                     # Hugging Face Spaces demo (self-contained)
│   ├── app.py                    # Standalone Streamlit app (no backend needed)
│   ├── requirements.txt          # HF Space dependencies
│   └── sample_documents/         # Sample files for demo testing
│
└── tests/
    ├── conftest.py               # Shared fixtures
    ├── test_api.py               # API endpoint tests
    ├── test_auth.py              # Authentication tests
    ├── test_documents.py         # Document processing tests
    ├── test_search.py            # Search functionality tests
    ├── test_llm.py               # LLM service tests
    ├── test_vector_store.py      # ChromaDB client tests
    └── test_integration.py       # End-to-end integration tests
```

---

## Getting Started

### Prerequisites

- **Python 3.9 – 3.11** (3.12+ requires Microsoft C++ Build Tools to compile `chroma-hnswlib`)
- **Git**
- Docker and Docker Compose *(optional — only needed for containerized deployment)*
- Internet connection on first run to download the embedding model (~90 MB)

### 1. Clone and Install

```bash
git clone https://github.com/phoenixak/RAG-System.git
cd RAG-System

python -m venv .venv

# Activate the venv
source .venv/bin/activate        # Linux / macOS
.venv\Scripts\activate           # Windows (cmd)
.venv\Scripts\Activate.ps1       # Windows (PowerShell)

pip install -e ".[dev]"
```

### 2. Environment Setup

The launcher auto-creates `.env` from `.env.example` on first run and generates secure random values for `SECRET_KEY`, `JWT_SECRET_KEY`, and `SESSION_SECRET`.

You can also create it manually:

```bash
cp .env.example .env
```

The only values you **must** set for local development are already handled automatically. Optionally add an LLM API key to enable full AI responses:

```env
# .env — optional additions
OPENAI_API_KEY=sk-...           # Enable OpenAI GPT responses
ANTHROPIC_API_KEY=sk-ant-...    # Enable Anthropic Claude responses
```

Without an API key the system runs in **demo mode** — it retrieves relevant document passages and returns formatted excerpts instead of LLM-generated answers.

### 3. Run

```bash
# Start both backend (port 8000) and frontend (port 8501)
python run.py

# Or start individually
python run.py backend            # FastAPI only
python run.py frontend           # Streamlit only
```

> **First run note:** The embedding model (`all-MiniLM-L6-v2`) is downloaded on first start. Backend startup takes approximately 15–60 seconds depending on your connection and machine. The launcher will wait up to 90 seconds for the health check to pass.

Open **http://localhost:8501** in your browser.

### 4. Log In

| Role | Email | Password |
|------|-------|----------|
| Admin | `admin@example.com` | `admin123!` |
| Standard User | `user@example.com` | `password123!` |

These defaults can be overridden via `ADMIN_EMAIL`, `ADMIN_PASSWORD`, `USER_EMAIL`, and `USER_PASSWORD` in `.env`.

---

## Running with Docker Compose

```bash
# Core services: API + ChromaDB + Redis
docker-compose up api chromadb redis

# Include the Streamlit frontend
docker-compose --profile frontend up

# Include dev tools (pgAdmin, Redis Commander)
docker-compose --profile frontend --profile tools up
```

> Inside the Compose network the API uses `CHROMADB_HOST=chromadb` and `CHROMADB_PORT=8000`.  
> From the host machine ChromaDB is exposed on port `8001`.

---

## API Reference

All endpoints are prefixed with `/api/v1`. Authenticated endpoints require `Authorization: Bearer <token>`.

Interactive Swagger docs: **http://localhost:8000/docs** *(available in development mode)*

### Authentication

| Method | Path | Description |
|--------|------|-------------|
| POST | `/auth/login` | Authenticate — returns access + refresh tokens |
| POST | `/auth/refresh` | Refresh an expired access token |
| POST | `/auth/logout` | Revoke the current token |
| GET | `/auth/me` | Get current user profile |
| GET | `/auth/verify` | Verify token validity |

### Documents

| Method | Path | Description |
|--------|------|-------------|
| POST | `/documents/upload` | Upload a document (PDF/DOCX/TXT/CSV) |
| GET | `/documents` | List documents (paginated) |
| GET | `/documents/{id}` | Get document details |
| DELETE | `/documents/{id}` | Delete a document |
| GET | `/documents/{id}/chunks` | List document chunks (paginated) |
| GET | `/documents/{id}/status` | Get processing status |
| DELETE | `/documents/bulk` | Bulk delete |
| GET | `/documents/stats` | System-wide document statistics |

### Search

| Method | Path | Description |
|--------|------|-------------|
| POST | `/search/semantic` | Vector similarity search |
| POST | `/search/hybrid` | Semantic + BM25 keyword search |
| POST | `/search/contextual` | Conversation-aware contextual search |
| POST | `/search/similar-documents` | Find similar documents |
| GET | `/search/suggestions` | Query autocomplete |
| POST | `/search/conversations` | Create a conversation session |
| GET | `/search/conversation/{id}` | Get conversation summary |
| DELETE | `/search/conversation/{id}` | Clear conversation |
| POST | `/search/generate_response` | Generate an LLM answer from search results |

### Deduplication

| Method | Path | Description |
|--------|------|-------------|
| GET | `/deduplication/scan` | Scan for duplicate documents |
| POST | `/deduplication/cleanup` | Remove duplicates |
| GET | `/deduplication/check/{hash}` | Check if a file hash already exists |
| GET | `/deduplication/statistics` | Deduplication statistics |

### Health

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Basic health status |
| GET | `/health/detailed` | Detailed per-service health |
| GET | `/health/ready` | Kubernetes readiness probe |
| GET | `/health/live` | Kubernetes liveness probe |

---

## Configuration

Key environment variables (full list in `.env.example`):

| Variable | Default | Description |
|---|---|---|
| `SECRET_KEY` | *(auto-generated)* | App secret key — minimum 32 characters |
| `JWT_SECRET_KEY` | *(auto-generated)* | JWT signing key |
| `ENVIRONMENT` | `development` | `development` \| `staging` \| `production` |
| `DEBUG` | `true` | Enables Swagger docs at `/docs` |
| `API_HOST` | `0.0.0.0` | Backend bind address |
| `API_PORT` | `8000` | Backend port |
| `CHROMADB_HOST` | `localhost` | ChromaDB host |
| `CHROMADB_PORT` | `8001` | ChromaDB port (host-mapped; `8000` inside Compose) |
| `REDIS_URL` | `redis://localhost:6379` | Redis connection string |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model |
| `CHUNK_SIZE` | `1000` | Characters per text chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between consecutive chunks |
| `MAX_FILE_SIZE` | `52428800` | Max upload size in bytes (50 MB) |
| `OPENAI_API_KEY` | *(optional)* | OpenAI key for GPT responses |
| `ANTHROPIC_API_KEY` | *(optional)* | Anthropic key for Claude responses |

---

## Troubleshooting

### `No module named 'streamlit'` or similar import errors
Run `pip install -e ".[dev]"` from the project root with your venv activated. All dependencies are declared in `pyproject.toml`.

### `chroma-hnswlib` build errors on Python 3.12 / Windows
Use **Python 3.11**. If you must use 3.12, install [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) first.

### Backend crashes with `pydantic_core.ValidationError` — "Extra inputs are not permitted"
This means the `Settings` class is using the Pydantic v1 `class Config` style instead of Pydantic v2 `model_config`. Ensure `src/core/config.py` uses:
```python
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
```

### Backend health check times out during `python run.py all`
The embedding model takes 15–60 seconds to download/load on first run. The launcher waits up to 90 seconds. If it still times out, check the `[BACKEND]` prefixed lines in the terminal for the actual error, or run the backend directly:
```bash
python run.py backend
```

### `.env` / `SECRET_KEY` startup failures
Delete the broken `.env` and re-run `python run.py` to regenerate it automatically, or set `SECRET_KEY` manually to any 32+ character string:
```bash
echo "SECRET_KEY=my-super-secret-key-that-is-long-enough" >> .env
```

### Redis / ChromaDB not running
Both are optional for local development:
- **Without Redis**: embedding cache and token blacklist fall back to in-memory automatically.
- **Without ChromaDB server**: the system falls back to embedded ChromaDB persisted at `./chroma_db/`.

### `run.py all` can't find `uvicorn` or `streamlit`
The launcher auto-detects your `.venv/` directory. If your venv is elsewhere, activate it before running:
```bash
source /path/to/venv/bin/activate
python run.py all
```

### Port already in use
The frontend defaults to port `8501` and backend to `8000`. Override via `.env`:
```env
API_PORT=8001
```
Or via environment variable before running:
```bash
API_PORT=8001 python run.py backend
```

---

## Testing

```bash
# Run all tests
pytest

# With verbose output
pytest -v

# With coverage report (outputs HTML to htmlcov/)
pytest --cov=src --cov-report=html

# Specific test file
pytest tests/test_search.py -v

# By marker
pytest -m unit
pytest -m integration
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.
