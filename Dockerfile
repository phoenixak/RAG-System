# API image: FastAPI + the RAG pipeline with a local GGUF model on CPU.
FROM python:3.11-slim

# llama.cpp is compiled for generic x86-64 + AVX2 (the prebuilt Linux wheels target musl)
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    CMAKE_ARGS="-DGGML_NATIVE=OFF -DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON" \
    HF_HOME=/models RAG_PERSIST_DIR=/data/chroma LOAD_SAMPLE_CORPUS=1

RUN apt-get update && apt-get install -y --no-install-recommends curl build-essential cmake libgomp1 \
    && rm -rf /var/lib/apt/lists/* && useradd --uid 1000 --create-home app

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src/ src/
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install "llama-cpp-python==0.3.36" \
    && pip install ".[api]"
COPY data/ data/

RUN mkdir -p /models /data && chown -R app /models /data /app
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s CMD curl -fs http://localhost:8000/health || exit 1
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000", "--app-dir", "src"]
