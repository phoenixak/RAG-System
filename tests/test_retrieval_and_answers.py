from rag import REFUSAL
from rag.generate import build_messages, citations
from rag.retrieve import Retriever, rrf


def test_rrf_rewards_documents_ranked_well_by_both():
    fused = dict(rrf([["a", "b", "c"], ["b", "c", "a"]], k=60))
    assert fused["b"] > fused["a"] > fused["c"] or fused["b"] > fused["c"]
    assert max(fused, key=fused.get) == "b"


def test_keyword_search_finds_exact_numbers(rag):
    top = rag.index.keyword_search("hotel cap Lisbon", 3)
    assert rag.index.chunks[top[0][0]].source == "06-travel-policy.md"


def test_hybrid_retrieval_brings_back_the_right_document(rag):
    res = rag.retrieve("What is the refund window for annual plans?")
    # the product FAQ also summarises refunds (a deliberate distractor), so check the top 3
    assert "04-customer-refund-policy.md" in [h.chunk.source for h in res.hits[:3]]
    assert [h.rank for h in res.hits] == list(range(1, len(res.hits) + 1))
    assert all(h.fused_rank for h in res.hits)


def test_reranker_scores_are_recorded_and_ordered(rag):
    res = rag.retrieve("minimum password length")
    scores = [h.rerank_score for h in res.hits]
    assert scores == sorted(scores, reverse=True)


def test_modes_without_reranker(rag):
    for mode in ("vector", "keyword"):
        r = Retriever(rag.index, None, mode=mode, top_k=3).search("per diem London")
        assert r.hits and all(h.rerank_score is None for h in r.hits)


def test_citations_parse_and_ignore_out_of_range():
    assert citations("Yes [2], and also [1, 3] but not [9].", 3) == [2, 1, 3]


def test_prompt_numbers_every_source(rag):
    res = rag.retrieve("uptime commitment")
    msgs = build_messages("uptime?", res.hits)
    assert "\n[1] " in msgs[1]["content"] and f"\n[{len(res.hits)}] " in msgs[1]["content"]
    # the instructions (with the refusal sentence) come after the sources, right before the answer
    assert msgs[1]["content"].rstrip().endswith(f'"{REFUSAL}"')


def test_tidy_collapses_citation_spam():
    from rag.generate import tidy

    assert tidy("It is 30 days [1] [1] [1].") == "It is 30 days [1]."
    assert tidy("Use 14 characters [2] [3] [4] [5] [2] [3] [4] [5]") == "Use 14 characters[2][3][4][5]"


def test_answer_streams_tokens_and_maps_citations(rag):
    events = list(rag.ask_stream("What is the refund window for annual plans?"))
    kinds = [k for k, _ in events]
    assert kinds[0] == "retrieval" and kinds[-1] == "answer" and "token" in kinds
    answer = events[-1][1]
    assert answer.cited and answer.cited[0] is answer.retrieval.hits[0]
    assert not answer.refused


def test_low_relevance_refuses_without_calling_the_llm(rag):
    rag.s.min_relevance = 100.0
    calls = rag.llm.calls
    a = rag.ask("What is the company's stock ticker?")
    assert a.refused and a.text == REFUSAL
    assert rag.llm.calls == calls


def test_reuploading_the_same_file_adds_nothing(rag):
    before = len(rag.index)
    content = rag.index.chunks and next(iter(rag.index.chunks.values())).source
    assert content
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / "data" / "sample_corpus" / "01-employee-handbook.md"
    _, added = rag.add_bytes(path.read_bytes(), "01-employee-handbook.md")
    assert added == 0 and len(rag.index) == before


def test_remove_document(rag):
    doc_id = next(c.doc_id for c in rag.index.chunks.values() if c.source == "06-travel-policy.md")
    removed = rag.index.remove_document(doc_id)
    assert removed > 0
    assert all(c.source != "06-travel-policy.md" for c in rag.index.chunks.values())
    assert rag.retrieve("hotel cap Lisbon").hits[0].chunk.source != "06-travel-policy.md"
