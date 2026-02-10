#!/usr/bin/env python3
"""
Chunking Throughput Benchmark Script

Measures chunking performance across increasingly large documents
and reports chunks/minute throughput.

Usage:
    python scripts/benchmark_chunking.py
"""

import sys
import time
from pathlib import Path
from uuid import uuid4

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


SAMPLE_PARAGRAPH = (
    "The advancement of artificial intelligence and machine learning has "
    "transformed numerous industries. Natural language processing enables "
    "computers to understand and generate human language with remarkable "
    "accuracy. Deep learning models trained on vast datasets can perform "
    "tasks that were previously thought to be exclusively human capabilities. "
    "Retrieval-augmented generation combines the power of large language "
    "models with external knowledge bases to produce more accurate and "
    "grounded responses. Vector databases store high-dimensional embeddings "
    "that enable semantic similarity search across millions of documents. "
    "Enterprise search systems leverage these technologies to help "
    "organizations unlock the value hidden in their document repositories."
)


def generate_document(target_chars: int) -> str:
    """Generate a sample document of approximately the target character count."""
    paragraphs = []
    current_length = 0
    para_index = 0

    while current_length < target_chars:
        # Vary paragraphs slightly to avoid trivial deduplication
        para = f"Section {para_index + 1}: {SAMPLE_PARAGRAPH}"
        paragraphs.append(para)
        current_length += len(para) + 2  # +2 for double newline
        para_index += 1

    return "\n\n".join(paragraphs)


def run_benchmark() -> None:
    """Run the chunking benchmark across various document sizes."""
    from src.documents.chunking import RecursiveCharacterTextSplitter

    # Document sizes to test (in characters)
    doc_sizes = [
        1_000,
        5_000,
        10_000,
        50_000,
        100_000,
        500_000,
        1_000_000,
    ]

    chunker = RecursiveCharacterTextSplitter()
    doc_id = uuid4()
    metadata = {"filename": "benchmark.txt", "document_type": "txt"}

    results = []

    print("=" * 78)
    print("  Chunking Throughput Benchmark")
    print("=" * 78)
    print(
        f"  Chunk size: {chunker.chunk_size} tokens | "
        f"Overlap: {chunker.chunk_overlap} tokens"
    )
    print("-" * 78)
    print(
        f"{'Doc Size':>12}  {'Chunks':>8}  {'Time (s)':>10}  "
        f"{'Chunks/min':>12}  {'Chars/s':>12}"
    )
    print("-" * 78)

    for size in doc_sizes:
        document_text = generate_document(size)
        actual_size = len(document_text)

        # Warm-up run for the first size (JIT, cache warm-up, etc.)
        if size == doc_sizes[0]:
            _ = chunker.chunk_text(document_text, doc_id, metadata)

        # Timed run -- repeat small docs to get stable measurements
        iterations = max(1, 100_000 // max(actual_size, 1))
        iterations = min(iterations, 50)  # cap iterations

        start = time.perf_counter()
        total_chunks = 0
        for _ in range(iterations):
            chunks = chunker.chunk_text(document_text, doc_id, metadata)
            total_chunks += len(chunks)
        elapsed = time.perf_counter() - start

        chunks_per_iteration = total_chunks / iterations
        time_per_iteration = elapsed / iterations
        chunks_per_minute = (
            (chunks_per_iteration / time_per_iteration) * 60
            if time_per_iteration > 0
            else 0
        )
        chars_per_second = (
            actual_size / time_per_iteration if time_per_iteration > 0 else 0
        )

        results.append(
            {
                "doc_size": actual_size,
                "chunks": int(chunks_per_iteration),
                "time_s": time_per_iteration,
                "chunks_per_min": chunks_per_minute,
                "chars_per_s": chars_per_second,
                "iterations": iterations,
            }
        )

        # Format size for display
        if actual_size >= 1_000_000:
            size_label = f"{actual_size / 1_000_000:.1f}M chars"
        elif actual_size >= 1_000:
            size_label = f"{actual_size / 1_000:.1f}K chars"
        else:
            size_label = f"{actual_size} chars"

        print(
            f"{size_label:>12}  {int(chunks_per_iteration):>8}  "
            f"{time_per_iteration:>10.4f}  {chunks_per_minute:>12,.0f}  "
            f"{chars_per_second:>12,.0f}"
        )

    print("-" * 78)

    # Summary
    total_chunks_all = sum(r["chunks"] for r in results)
    total_time_all = sum(r["time_s"] for r in results)
    overall_rate = (total_chunks_all / total_time_all) * 60 if total_time_all > 0 else 0

    print(f"\n  Overall: {total_chunks_all} chunks in {total_time_all:.4f}s")
    print(f"  Aggregate throughput: {overall_rate:,.0f} chunks/min")

    # Verify the claim
    min_rate = min(r["chunks_per_min"] for r in results)
    if min_rate > 1000:
        print(
            f"\n  [PASS] All document sizes exceed 1K chunks/min (min: {min_rate:,.0f})"
        )
    else:
        print(f"\n  [INFO] Minimum throughput: {min_rate:,.0f} chunks/min")

    print("=" * 78)


if __name__ == "__main__":
    run_benchmark()
