"""Grounded answer generation with numbered citations, streamed token by token."""

import json
import re
from collections.abc import Iterator
from typing import Protocol

import httpx

from .config import REFUSAL, Settings
from .retrieve import Hit

SYSTEM = "You are a careful assistant that answers questions about a company's documents, using only the sources given."

# Small models follow instructions best when they come right before the answer, with a worked example.
INSTRUCTIONS = (
    "Answer the question in one or two plain sentences, using only the sources above. "
    "State the actual fact (numbers, names, conditions), then put the number of the source that supports it "
    'in square brackets, for example: "Annual plans can be refunded in full within 30 days [2]." '
    'Do not add phrases like "according to the sources" or "this is supported by source 1". '
    f'If the sources do not answer the question, reply exactly: "{REFUSAL}"'
)


def build_messages(question: str, hits: list[Hit]) -> list[dict]:
    context = "\n\n".join(f"[{i}] {h.chunk.text}" for i, h in enumerate(hits, start=1))
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Sources:\n\n{context}\n\nQuestion: {question}\n\n{INSTRUCTIONS}"},
    ]


def tidy(answer: str) -> str:
    """Collapse citation runs ("[1] [1] [2]" → "[1][2]") and cut a degenerate tail of bare citations."""
    answer = re.sub(r"(\[\d+\])(?:\s*\1)+", r"\1", answer)
    answer = re.sub(
        r"((?:\s*\[\d+\]){3,})\s*$", lambda m: "".join(dict.fromkeys(re.findall(r"\[\d+\]", m.group(1)))), answer
    )
    return answer.strip()


def citations(answer: str, n_sources: int) -> list[int]:
    """Source numbers cited in the answer, in order of first use; out-of-range numbers are ignored."""
    seen: list[int] = []
    for m in re.finditer(r"\[(\d+(?:\s*,\s*\d+)*)\]", answer):
        for num in m.group(1).split(","):
            n = int(num)
            if 1 <= n <= n_sources and n not in seen:
                seen.append(n)
    return seen


def strip_think(tokens: Iterator[str]) -> Iterator[str]:
    """Drop a leading <think>…</think> block (reasoning models) from a token stream."""
    buf, inside = "", None
    for tok in tokens:
        if inside is False:
            yield tok
            continue
        buf += tok
        if inside is None:
            if not "<think>".startswith(buf.lstrip()[:7]):
                inside = False
                yield buf
                continue
            if buf.lstrip().startswith("<think>"):
                inside = True
        if inside and "</think>" in buf:
            inside, rest = False, buf.split("</think>", 1)[1].lstrip()
            if rest:
                yield rest


class LLM(Protocol):
    def stream(self, messages: list[dict]) -> Iterator[str]: ...


class LlamaCppLLM:
    """Local GGUF model on CPU via llama.cpp (downloaded from the Hugging Face Hub on first use)."""

    def __init__(self, s: Settings) -> None:
        from llama_cpp import Llama

        self.s = s
        # n_threads_batch (prompt reading) defaults to every CPU the OS reports; in a container limited to
        # 2 cores that oversubscribes badly, so both phases use the same thread count.
        self.llm = Llama.from_pretrained(
            repo_id=s.llm_repo,
            filename=s.llm_file,
            n_ctx=s.llm_ctx,
            n_threads=s.llm_threads,
            n_threads_batch=s.llm_threads,
            verbose=False,
        )

    def stream(self, messages: list[dict]) -> Iterator[str]:
        if "qwen3" in self.s.llm_file.lower():  # answer directly, skip Qwen3's thinking mode
            messages = [*messages[:-1], {**messages[-1], "content": messages[-1]["content"] + " /no_think"}]
        yield from strip_think(self._raw(messages))

    def _raw(self, messages: list[dict]) -> Iterator[str]:
        for part in self.llm.create_chat_completion(
            messages=messages,
            stream=True,
            max_tokens=self.s.max_new_tokens,
            temperature=self.s.temperature,
            repeat_penalty=1.15,
            stop=["\n\n\n", "Question:", "Sources:"],
        ):
            delta = part["choices"][0]["delta"].get("content")
            if delta:
                yield delta


class OpenAICompatibleLLM:
    """Any OpenAI-compatible chat endpoint (OpenAI, vLLM, Ollama, LM Studio…)."""

    def __init__(self, s: Settings) -> None:
        self.s = s

    def stream(self, messages: list[dict]) -> Iterator[str]:
        body = {
            "model": self.s.openai_model,
            "messages": messages,
            "stream": True,
            "max_tokens": self.s.max_new_tokens,
            "temperature": self.s.temperature,
        }
        headers = {"Authorization": f"Bearer {self.s.openai_api_key or ''}"}
        with httpx.stream(
            "POST", f"{self.s.openai_base_url}/chat/completions", json=body, headers=headers, timeout=60
        ) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if line.startswith("data: ") and line != "data: [DONE]":
                    delta = json.loads(line[6:])["choices"][0]["delta"].get("content")
                    if delta:
                        yield delta


class AnthropicLLM:
    def __init__(self, s: Settings) -> None:
        self.s = s

    def stream(self, messages: list[dict]) -> Iterator[str]:
        system = messages[0]["content"]
        body = {
            "model": self.s.anthropic_model,
            "system": system,
            "messages": messages[1:],
            "stream": True,
            "max_tokens": self.s.max_new_tokens,
            "temperature": self.s.temperature,
        }
        headers = {"x-api-key": self.s.anthropic_api_key or "", "anthropic-version": "2023-06-01"}
        with httpx.stream("POST", "https://api.anthropic.com/v1/messages", json=body, headers=headers, timeout=60) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if line.startswith("data: "):
                    event = json.loads(line[6:])
                    if event.get("type") == "content_block_delta":
                        yield event["delta"].get("text", "")


class ExtractiveLLM:
    """No model at all: quotes the best chunk. Used for tests and when RAG_LLM_BACKEND=none."""

    def stream(self, messages: list[dict]) -> Iterator[str]:
        m = re.search(r"\[1\] (.*?)(?:\n\n\[2\]|\n\nQuestion:)", messages[1]["content"], re.S)
        body = m.group(1).strip() if m else ""
        yield f"{body.splitlines()[-1][:400]} [1]" if body else REFUSAL


def make_llm(s: Settings) -> LLM | None:
    if s.llm_backend == "llamacpp":
        return LlamaCppLLM(s)
    if s.llm_backend == "openai":
        return OpenAICompatibleLLM(s)
    if s.llm_backend == "anthropic":
        return AnthropicLLM(s)
    return ExtractiveLLM()
