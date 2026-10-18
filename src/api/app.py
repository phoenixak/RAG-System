"""FastAPI service around the RAG pipeline.

uvicorn api.app:app --app-dir src --port 8000
"""

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.concurrency import iterate_in_threadpool, run_in_threadpool

from rag import RAG, Answer, Hit, Retrieval, Settings
from rag.ingest import UnsupportedFile

from . import auth
from .auth import Role, User, require

SAMPLE = Path(__file__).resolve().parents[2] / "data" / "sample_corpus"


def _rate_key(request: Request) -> str:
    header = request.headers.get("authorization", "")
    return header[-24:] if header.startswith("Bearer ") else get_remote_address(request)


limiter = Limiter(key_func=_rate_key)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.rag = rag = getattr(app.state, "rag", None) or RAG(Settings())
    if os.environ.get("LOAD_SAMPLE_CORPUS", "1") == "1" and not len(rag.index):
        await run_in_threadpool(rag.add_folder, SAMPLE)
    yield


app = FastAPI(title="RAG System", version="2.0.0", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


def get_rag(request: Request) -> RAG:
    return request.app.state.rag


class Query(BaseModel):
    query: str = Field(min_length=2, max_length=500)


class Question(BaseModel):
    question: str = Field(min_length=2, max_length=500)
    stream: bool = False


def hit_json(h: Hit) -> dict:
    c = h.chunk
    return {
        "rank": h.rank,
        "source": c.source,
        "section": c.heading,
        "page": c.page,
        "text": c.text,
        "scores": {
            "vector": h.vector_score,
            "vector_rank": h.vector_rank,
            "bm25": h.keyword_score,
            "bm25_rank": h.keyword_rank,
            "fused": h.fused_score,
            "fused_rank": h.fused_rank,
            "rerank": h.rerank_score,
        },
    }


def retrieval_json(r: Retrieval) -> dict:
    return {"query": r.query, "confidence": r.confidence, "hits": [hit_json(h) for h in r.hits]}


def answer_json(a: Answer) -> dict:
    return {
        "question": a.question,
        "answer": a.text,
        "refused": a.refused,
        "citations": [
            {
                "n": a.retrieval.hits.index(h) + 1,
                "source": h.chunk.source,
                "section": h.chunk.heading,
                "page": h.chunk.page,
            }
            for h in a.cited
        ],
        "timings": a.timings,
    }


@app.get("/health")
def health(rag: RAG = Depends(get_rag)) -> dict:
    return {
        "status": "ok",
        "documents": len(rag.index.documents()),
        "chunks": len(rag.index),
        "llm_backend": rag.s.llm_backend,
    }


@app.post("/auth/token")
@limiter.limit("10/minute")
def token(request: Request, form: OAuth2PasswordRequestForm = Depends()) -> dict:
    user = auth.authenticate(form.username, form.password)
    if not user:
        raise HTTPException(401, "Wrong username or password")
    return {"access_token": auth.create_token(user), "token_type": "bearer", "role": user.role.name}


@app.get("/documents")
def list_documents(rag: RAG = Depends(get_rag), user: User = Depends(require(Role.read_only))) -> list[dict]:
    return list(rag.index.documents().values())


@app.post("/documents", status_code=201)
@limiter.limit("10/minute")
async def upload(
    request: Request, file: UploadFile, rag: RAG = Depends(get_rag), user: User = Depends(require(Role.standard_user))
) -> dict:
    content = await file.read()
    try:
        doc, added = await run_in_threadpool(rag.add_bytes, content, file.filename or "upload.txt")
    except UnsupportedFile as e:
        raise HTTPException(415, str(e)) from None
    return {"doc_id": doc.doc_id, "source": doc.source, "chunks_added": added, "duplicate": added == 0}


@app.delete("/documents/{doc_id}")
def delete(doc_id: str, rag: RAG = Depends(get_rag), user: User = Depends(require(Role.power_user))) -> dict:
    removed = rag.index.remove_document(doc_id)
    if not removed:
        raise HTTPException(404, "No such document")
    return {"doc_id": doc_id, "chunks_removed": removed}


@app.post("/search")
@limiter.limit("60/minute")
async def search(
    request: Request, body: Query, rag: RAG = Depends(get_rag), user: User = Depends(require(Role.read_only))
) -> dict:
    return retrieval_json(await run_in_threadpool(rag.retrieve, body.query))


@app.post("/ask")
@limiter.limit("20/minute")
async def ask(
    request: Request, body: Question, rag: RAG = Depends(get_rag), user: User = Depends(require(Role.read_only))
):
    if not body.stream:
        return answer_json(await run_in_threadpool(rag.ask, body.question))

    def events():
        for kind, value in rag.ask_stream(body.question):
            data = retrieval_json(value) if kind == "retrieval" else answer_json(value) if kind == "answer" else value
            yield f"event: {kind}\ndata: {json.dumps(data)}\n\n"

    return StreamingResponse(iterate_in_threadpool(events()), media_type="text/event-stream")


@app.get("/admin/users")
def users(user: User = Depends(require(Role.admin))) -> list[dict]:
    return [{"username": name, "role": role.name} for name, (_, role) in auth.USERS.items()]
