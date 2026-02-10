# Enterprise RAG System

A Retrieval-Augmented Generation system built with FastAPI, ChromaDB, and Streamlit that enables semantic document search and AI-powered conversational question answering over uploaded document collections.

## Architecture Overview

The system is composed of four services orchestrated via Docker Compose:

- **FastAPI Backend** (port 8000) -- REST API handling authentication, document processing, search, and LLM response generation. Registers all routes under the `/api/v1` prefix.
- **Streamlit Frontend** (port 8501) -- Multi-page web interface providing chat, document management, search, settings, and admin views.
- **ChromaDB** (port 8001) -- Vector database storing document chunk embeddings for similarity search.
- **Redis** (port 6379) -- Caching layer for embeddings and search results.

User authentication and document metadata are managed in-memory (with demo users created at startup). The API communicates with ChromaDB for vector storage and retrieval, while the Streamlit frontend calls the API over HTTP.

## Features

- Multi-format document processing (PDF, DOCX, TXT, CSV) with dedicated extractors
- Text chunking with configurable size and overlap using a recursive character text splitter with tiktoken token counting
- Embedding generation using sentence-transformers (all-MiniLM-L6-v2), with batched processing
- Hybrid search combining semantic vector similarity (ChromaDB) and BM25 keyword scoring
- Cross-encoder re-ranking for improved result relevance
- Conversational search with session-based context tracking
- Document deduplication via content hashing with scan and cleanup endpoints
- LLM integration (OpenAI, Anthropic) for RAG-style response generation
- JWT authentication with role-based access control (admin, power_user, standard_user, read_only)
- Health check endpoints including readiness and liveness probes
- Structured logging with structlog
- Security middleware (CORS, security headers, request logging)

## Tech Stack

| Layer            | Technology                                    |
|------------------|-----------------------------------------------|
| API Framework    | FastAPI 0.104.1, Pydantic 2.5, Uvicorn        |
| Frontend         | Streamlit 1.28.1                               |
| Vector Store     | ChromaDB 0.4.15                                |
| Embeddings       | sentence-transformers 2.2.2                    |
| Keyword Search   | rank-bm25 0.2.2                                |
| LLM Providers    | openai, anthropic                              |
| Auth             | PyJWT, python-jose, passlib (bcrypt)           |
| Document Parsing | PyPDF2, pdfplumber, python-docx, pandas        |
| Caching          | Redis 7.2                                      |
| Tokenization     | tiktoken                                       |
| Logging          | structlog, rich                                |
| Containerization | Docker, Docker Compose                         |
| Testing          | pytest, pytest-asyncio, pytest-cov             |

## Project Structure

```
RAG-System/
├── run.py                       # Unified launcher (backend, frontend, or both)
├── pyproject.toml               # Dependencies and tool configuration
├── docker-compose.yml           # Service orchestration
├── Dockerfile                   # Container build
├── .env.example                 # Environment variable template
├── src/
│   ├── api/
│   │   ├── auth.py              # Login, logout, token refresh, user info
│   │   ├── documents.py         # Upload, list, get, delete, chunks, stats
│   │   ├── search.py            # Semantic, hybrid, contextual search; conversations; LLM response
│   │   ├── deduplication.py     # Duplicate scan, cleanup, hash check
│   │   └── health.py            # Health, readiness, liveness endpoints
│   ├── auth/
│   │   ├── models.py            # User, role, permission, and token models
│   │   ├── jwt_utils.py         # Token creation and verification
│   │   └── security.py          # Password hashing, current user dependency
│   ├── core/
│   │   ├── config.py            # Pydantic settings with env var support
│   │   └── logging.py           # Structured logging setup
│   ├── documents/
│   │   ├── processors.py        # PDF, DOCX, TXT, CSV text extraction
│   │   ├── chunking.py          # TextChunker, RecursiveCharacterTextSplitter
│   │   ├── embeddings.py        # Batched embedding generation with caching
│   │   ├── deduplication.py     # Content-hash-based duplicate detection
│   │   ├── models.py            # Document and chunk data models
│   │   └── service.py           # Document processing orchestration
│   ├── search/
│   │   ├── semantic_search.py   # ChromaDB vector similarity search
│   │   ├── hybrid_search.py     # Combined semantic + BM25 scoring
│   │   ├── reranking.py         # Cross-encoder re-ranking
│   │   ├── conversation.py      # Session context management
│   │   ├── models.py            # Search request/response models
│   │   └── service.py           # Search service facade
│   ├── llm/
│   │   ├── service.py           # OpenAI/Anthropic client management
│   │   ├── prompts.py           # RAG prompt templates
│   │   └── models.py            # LLM request/response models
│   └── vector_store/
│       └── chroma_client.py     # ChromaDB connection and operations
├── frontend/
│   ├── components/
│   │   ├── api_client.py        # HTTP client for backend API
│   │   ├── auth.py              # Login UI and session management
│   │   ├── chat_ui.py           # Chat interface components
│   │   └── utils.py             # Frontend utilities
│   ├── pages/
│   │   ├── admin.py             # User and system administration
│   │   ├── chat.py              # Conversational RAG interface
│   │   ├── documents.py         # Document upload and management
│   │   ├── search.py            # Search interface
│   │   └── settings.py          # User preferences
│   └── config.py                # Frontend configuration constants
├── tests/
│   ├── conftest.py              # Shared fixtures
│   ├── test_api.py              # API endpoint tests
│   ├── test_auth.py             # Authentication tests
│   ├── test_documents.py        # Document processing tests
│   ├── test_search.py           # Search functionality tests
│   ├── test_llm.py              # LLM service tests
│   ├── test_vector_store.py     # ChromaDB client tests
│   └── test_integration.py      # End-to-end integration tests
└── scripts/
    └── dev.py                   # Development utilities
```

## Getting Started

### Prerequisites

- Python 3.9+
- Docker and Docker Compose (for containerized deployment)
- Internet connection (for downloading the embedding model on first run)

### Environment Setup

```bash
git clone https://github.com/phoenixak/RAG-System.git
cd RAG-System

python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows

pip install -e ".[dev]"

cp .env.example .env
# Edit .env -- at minimum, set SECRET_KEY and JWT_SECRET_KEY
```

### Running Locally

```bash
# Start both backend and frontend
python run.py

# Or start individually
python run.py backend           # FastAPI on http://localhost:8000
python run.py frontend          # Streamlit on http://localhost:8501
```

### Running with Docker Compose

```bash
# Core services (API + PostgreSQL + Redis + ChromaDB)
docker-compose up api postgres redis chromadb

# Include the frontend
docker-compose --profile frontend up

# Include dev tools (pgAdmin, Redis Commander)
docker-compose --profile frontend --profile tools up
```

### Default Credentials

| Role          | Email               | Password       |
|---------------|---------------------|----------------|
| Admin         | admin@example.com   | admin123!      |
| Standard User | user@example.com    | password123!   |

These defaults can be overridden via `ADMIN_EMAIL`, `ADMIN_PASSWORD`, `USER_EMAIL`, and `USER_PASSWORD` environment variables.

## API Endpoints

All endpoints are prefixed with `/api/v1`. Authenticated endpoints require a Bearer token in the Authorization header.

### Authentication

| Method | Path                    | Description                  |
|--------|-------------------------|------------------------------|
| POST   | /api/v1/auth/login      | Authenticate and get tokens  |
| POST   | /api/v1/auth/refresh    | Refresh access token         |
| POST   | /api/v1/auth/logout     | Revoke current token         |
| GET    | /api/v1/auth/me         | Get current user profile     |
| GET    | /api/v1/auth/verify     | Verify token validity        |

### Documents

| Method | Path                                    | Description                     |
|--------|-----------------------------------------|---------------------------------|
| POST   | /api/v1/documents/upload                | Upload a document               |
| GET    | /api/v1/documents                       | List documents (paginated)      |
| GET    | /api/v1/documents/{id}                  | Get document details            |
| DELETE | /api/v1/documents/{id}                  | Delete a document               |
| GET    | /api/v1/documents/{id}/chunks           | Get document chunks (paginated) |
| GET    | /api/v1/documents/{id}/status           | Get processing status           |
| DELETE | /api/v1/documents/bulk                  | Bulk delete documents           |
| GET    | /api/v1/documents/stats                 | Get document statistics         |

### Search

| Method | Path                                    | Description                            |
|--------|-----------------------------------------|----------------------------------------|
| POST   | /api/v1/search/semantic                 | Vector similarity search               |
| POST   | /api/v1/search/hybrid                   | Combined semantic + keyword search     |
| POST   | /api/v1/search/contextual               | Context-aware conversational search    |
| POST   | /api/v1/search/similar-documents        | Find similar documents                 |
| GET    | /api/v1/search/suggestions              | Query autocomplete suggestions         |
| GET    | /api/v1/search/stats                    | Search system statistics (admin)       |
| POST   | /api/v1/search/cache/clear              | Clear search cache (admin)             |
| POST   | /api/v1/search/conversations            | Create conversation session            |
| GET    | /api/v1/search/conversation/{id}        | Get conversation summary               |
| DELETE | /api/v1/search/conversation/{id}        | Clear conversation session             |
| POST   | /api/v1/search/generate_response        | Generate LLM response from results    |

### Deduplication

| Method | Path                                    | Description                         |
|--------|-----------------------------------------|-------------------------------------|
| GET    | /api/v1/deduplication/scan              | Scan for duplicate documents        |
| POST   | /api/v1/deduplication/cleanup           | Remove duplicate documents          |
| POST   | /api/v1/deduplication/cleanup-orphaned  | Remove orphaned files               |
| GET    | /api/v1/deduplication/check/{hash}      | Check if file hash exists           |
| GET    | /api/v1/deduplication/statistics        | Get deduplication statistics        |

### Health

| Method | Path                     | Description                  |
|--------|--------------------------|------------------------------|
| GET    | /api/v1/health           | Basic health status          |
| GET    | /api/v1/health/detailed  | Detailed service health      |
| GET    | /api/v1/health/ready     | Readiness probe              |
| GET    | /api/v1/health/live      | Liveness probe               |

Interactive API documentation is available at `http://localhost:8000/docs` when running in development mode.

## Configuration

Key environment variables (see `.env.example` for the full list):

| Variable                | Default                                      | Description                          |
|-------------------------|----------------------------------------------|--------------------------------------|
| SECRET_KEY              | *(required)*                                 | Application secret key (32+ chars)   |
| JWT_SECRET_KEY          | *(required)*                                 | JWT signing key                      |
| JWT_ALGORITHM           | HS256                                        | JWT signing algorithm                |
| DATABASE_URL            | postgresql://...                             | Database connection string           |
| REDIS_URL               | redis://localhost:6379                        | Redis connection string              |
| CHROMADB_HOST           | localhost                                    | ChromaDB server host                 |
| CHROMADB_PORT           | 8000                                         | ChromaDB server port                 |
| EMBEDDING_MODEL         | sentence-transformers/all-MiniLM-L6-v2       | Embedding model name                 |
| CHUNK_SIZE              | 1000                                         | Text chunk size (characters)         |
| CHUNK_OVERLAP           | 200                                          | Overlap between chunks               |
| MAX_FILE_SIZE           | 52428800                                     | Max upload size in bytes (50 MB)     |
| OPENAI_API_KEY          | *(optional)*                                 | OpenAI API key for LLM responses     |
| ANTHROPIC_API_KEY       | *(optional)*                                 | Anthropic API key for LLM responses  |
| ENVIRONMENT             | development                                  | Runtime environment                  |
| DEBUG                   | false                                        | Enable debug mode and API docs       |

## Testing

The test suite consists of 7 test files covering API endpoints, authentication, document processing, search, LLM integration, vector store operations, and end-to-end integration.

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run with coverage report
pytest --cov=src --cov-report=html

# Run a specific test file
pytest tests/test_search.py -v

# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration
```

## License

This project is licensed under the MIT License.
