import json

import pytest
from conftest import CORPUS, HashEmbedder, OverlapReranker, ScriptedLLM
from fastapi.testclient import TestClient

from api import auth
from api.app import app
from rag import RAG, Settings

PASSWORD = "correct horse battery"


@pytest.fixture(scope="module")
def client():
    h = auth.hash_password(PASSWORD)
    auth.USERS.clear()
    auth.USERS.update({name: (h, auth.Role[name]) for name in ("read_only", "standard_user", "power_user", "admin")})
    rag = RAG(
        Settings(llm_backend="none", min_relevance=0.0, chunk_tokens=80, chunk_overlap=15),
        embedder=HashEmbedder(),
        reranker=OverlapReranker(),
        llm=ScriptedLLM(),
    )
    rag.add_folder(CORPUS)
    app.state.rag = rag
    with TestClient(app) as c:
        yield c


def login(client, user):
    r = client.post("/auth/token", data={"username": user, "password": PASSWORD})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


def test_health_is_public(client):
    assert client.get("/health").json()["documents"] == 8


def test_wrong_password_and_missing_token(client):
    assert client.post("/auth/token", data={"username": "admin", "password": "nope"}).status_code == 401
    assert client.post("/search", json={"query": "leave"}).status_code == 401


def test_search_returns_every_score(client):
    r = client.post("/search", json={"query": "hotel cap in Lisbon"}, headers=login(client, "read_only"))
    hit = r.json()["hits"][0]
    assert hit["source"] == "06-travel-policy.md"
    assert set(hit["scores"]) >= {"vector", "bm25", "fused", "rerank"}


def test_ask_json_and_stream(client):
    h = login(client, "read_only")
    body = client.post("/ask", json={"question": "refund window for annual plans?"}, headers=h).json()
    assert body["citations"][0]["source"] in {"04-customer-refund-policy.md", "08-beacon-product-faq.md"}
    with client.stream("POST", "/ask", json={"question": "refund window?", "stream": True}, headers=h) as r:
        text = "".join(r.iter_text())
    events = [line[7:] for line in text.splitlines() if line.startswith("event: ")]
    assert events[0] == "retrieval" and events[-1] == "answer"
    last = json.loads(text.strip().splitlines()[-1][6:])
    assert "answer" in last


def test_roles_are_enforced(client):
    files = {"file": ("notes.txt", b"The office plant is watered on Tuesdays.", "text/plain")}
    assert client.post("/documents", files=files, headers=login(client, "read_only")).status_code == 403
    up = client.post("/documents", files=files, headers=login(client, "standard_user"))
    assert up.status_code == 201
    doc_id = up.json()["doc_id"]
    assert client.delete(f"/documents/{doc_id}", headers=login(client, "standard_user")).status_code == 403
    assert client.delete(f"/documents/{doc_id}", headers=login(client, "power_user")).status_code == 200
    assert client.get("/admin/users", headers=login(client, "power_user")).status_code == 403
    assert client.get("/admin/users", headers=login(client, "admin")).status_code == 200


def test_duplicate_upload_and_unsupported_type(client):
    h = login(client, "standard_user")
    content = (CORPUS / "02-expense-policy.md").read_bytes()
    r = client.post("/documents", files={"file": ("again.md", content, "text/markdown")}, headers=h)
    assert r.json()["duplicate"] is True
    bad = client.post("/documents", files={"file": ("photo.png", b"\x89PNG", "image/png")}, headers=h)
    assert bad.status_code == 415
