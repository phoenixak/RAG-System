#!/usr/bin/env bash
# Build the llama-cpp-python wheel the Space installs, inside the same base image the Space uses,
# so the glibc version matches. Output: ../wheels/llama_cpp_python-<version>-py3-none-linux_x86_64.whl
set -euo pipefail
VERSION="${1:-0.3.36}"
OUT="$(cd "$(dirname "$0")/../.." && pwd)/wheels"
mkdir -p "$OUT"
docker run --rm -v "$OUT:/out" python:3.11-slim bash -c "
  apt-get update -qq && apt-get install -y -qq build-essential cmake >/dev/null &&
  CMAKE_ARGS='-DGGML_NATIVE=OFF -DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON' \
  CMAKE_BUILD_PARALLEL_LEVEL=\$(nproc) pip wheel -q --no-deps --no-cache-dir llama-cpp-python==$VERSION -w /out"
ls -la "$OUT"
