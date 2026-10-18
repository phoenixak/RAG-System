"""Evaluate retrieval (and optionally generation) on the golden set over the sample corpus.

    python eval/run_eval.py              # retrieval only: vector, BM25, hybrid, hybrid + rerank
    python eval/run_eval.py --generate   # also answers every question with the configured LLM

Writes eval/results.md and prints the same tables.
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from rag import RAG, Settings  # noqa: E402
from rag.retrieve import Retriever  # noqa: E402

CORPUS = ROOT / "data" / "sample_corpus"
GOLDEN = ROOT / "eval" / "golden.jsonl"


def load_golden() -> list[dict]:
    return [json.loads(line) for line in GOLDEN.read_text(encoding="utf-8").splitlines() if line.strip()]


def retrieval_metrics(rag: RAG, golden: list[dict], k: int) -> dict[str, dict[str, float]]:
    configs = {
        "Vector only (bge-small)": Retriever(
            rag.index, None, rag.s.candidates, rag.s.rerank_pool, k, rag.s.rrf_k, "vector"
        ),
        "BM25 only": Retriever(rag.index, None, rag.s.candidates, rag.s.rerank_pool, k, rag.s.rrf_k, "keyword"),
        "Hybrid (RRF)": Retriever(rag.index, None, rag.s.candidates, rag.s.rerank_pool, k, rag.s.rrf_k, "hybrid"),
        "Hybrid + cross-encoder rerank": Retriever(
            rag.index, rag.retriever.reranker, rag.s.candidates, rag.s.rerank_pool, k, rag.s.rrf_k, "hybrid"
        ),
    }
    answerable = [g for g in golden if g["source"]]
    out = {}
    for name, retriever in configs.items():
        hits1, hitsk, rr, lat = 0, 0, [], []
        for g in answerable:
            t = time.perf_counter()
            res = retriever.search(g["q"])
            lat.append(time.perf_counter() - t)
            sources = [h.chunk.source for h in res.hits]
            if sources[:1] == [g["source"]]:
                hits1 += 1
            if g["source"] in sources[:k]:
                hitsk += 1
            rr.append(next((1 / (i + 1) for i, s in enumerate(sources) if s == g["source"]), 0.0))
        n = len(answerable)
        out[name] = {
            "Hit@1": hits1 / n,
            f"Recall@{k}": hitsk / n,
            "MRR": statistics.mean(rr),
            "p50 latency (ms)": 1000 * statistics.median(lat),
        }
    return out


def generation_metrics(rag: RAG, golden: list[dict]) -> tuple[dict[str, float], list[dict]]:
    rows, correct, grounded, refuse_ok, false_refusals, first_tok, total = [], 0, 0, 0, 0, [], []
    answerable = [g for g in golden if g["source"]]
    unanswerable = [g for g in golden if not g["source"]]
    for g in golden:
        a = rag.ask(g["q"])
        if g["source"]:
            ok = (not a.refused) and all(e.lower() in a.text.lower() for e in g["expect"])
            cites_right = bool(a.cited) and all(h.chunk.source == g["source"] for h in a.cited)
            correct += ok
            grounded += cites_right
            false_refusals += a.refused
        else:
            ok = a.refused
            refuse_ok += ok
            cites_right = None
        if not a.refused:
            first_tok.append(a.timings.get("first_token_s", 0.0))
            total.append(a.timings.get("retrieval_s", 0.0) + a.timings.get("generation_s", 0.0))
        rows.append({"q": g["q"], "answer": a.text, "ok": ok, "cites_right_doc": cites_right})
    n = len(answerable)
    metrics = {
        "Answer contains the expected fact": correct / n,
        "Every citation points to the right document": grounded / n,
        "Refused when the answer is not in the docs": refuse_ok / max(1, len(unanswerable)),
        "False refusals (answer was in the docs)": false_refusals / n,
        "Median time to first token (s)": statistics.median(first_tok) if first_tok else 0.0,
        "Median full answer time (s)": statistics.median(total) if total else 0.0,
    }
    return metrics, rows


def table(metrics: dict[str, dict[str, float]]) -> str:
    cols = list(next(iter(metrics.values())))
    lines = ["| Retriever | " + " | ".join(cols) + " |", "|---|" + "---:|" * len(cols)]
    for name, m in metrics.items():
        cells = [f"{v:.0f}" if "ms" in c else f"{v:.3f}" if c == "MRR" else f"{v:.0%}" for c, v in m.items()]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--generate", action="store_true", help="also run the LLM on every question")
    ap.add_argument("-k", type=int, default=5)
    args = ap.parse_args()

    s = Settings()
    rag = RAG(s)
    t = time.perf_counter()
    docs = rag.add_folder(CORPUS)
    print(f"indexed {len(docs)} documents, {len(rag.index)} chunks in {time.perf_counter() - t:.1f}s")
    golden = load_golden()

    ret = retrieval_metrics(rag, golden, args.k)
    md = [
        f"# Evaluation results\n\nCorpus: {len(docs)} documents, {len(rag.index)} chunks. "
        f"Golden set: {sum(1 for g in golden if g['source'])} answerable + "
        f"{sum(1 for g in golden if not g['source'])} unanswerable questions.\n",
        f"## Retrieval (does the right document come back?)\n\n{table(ret)}\n",
    ]
    print(table(ret))

    if args.generate:
        gen, rows = generation_metrics(rag, golden)
        backend = s.llm_backend if s.llm_backend != "llamacpp" else f"llama.cpp · {s.llm_file}"
        lines = ["| Metric | Value |", "|---|---:|"]
        for name, v in gen.items():
            lines.append(f"| {name} | {v:.1f} |" if "(s)" in name else f"| {name} | {v:.0%} |")
        md.append(f"## Answers ({backend}, CPU)\n\n" + "\n".join(lines) + "\n")
        md.append(
            "<details><summary>Every answer</summary>\n\n| Question | Answer | OK |\n|---|---|---|\n"
            + "\n".join(
                f"| {r['q']} | {r['answer'].replace('|', '/').replace(chr(10), ' ')} | {'✓' if r['ok'] else '✗'} |"
                for r in rows
            )
            + "\n\n</details>\n"
        )
        print("\n".join(lines))
    (ROOT / "eval" / "results.md").write_text("\n".join(md), encoding="utf-8")


if __name__ == "__main__":
    main()
