# Hugging Face Space (Docker SDK). llama.cpp is compiled here for generic x86-64 + AVX2,
# because the prebuilt Linux wheels target musl, and the models are baked into the image
# so a cold start doesn't download 1.2 GB.
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends build-essential cmake libgomp1 \
    && rm -rf /var/lib/apt/lists/* && useradd -m -u 1000 user

ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    CMAKE_ARGS="-DGGML_NATIVE=OFF -DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON"
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install "llama-cpp-python==0.3.36"
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

USER user
ENV HOME=/home/user HF_HOME=/home/user/.cache/huggingface PYTHONUNBUFFERED=1 \
    GRADIO_SERVER_NAME=0.0.0.0 GRADIO_SERVER_PORT=7860 LLM_THREADS=2
WORKDIR /home/user/app
RUN python -c "from huggingface_hub import snapshot_download as s, hf_hub_download as h; \
s('BAAI/bge-small-en-v1.5'); s('cross-encoder/ms-marco-MiniLM-L6-v2'); \
h('unsloth/Qwen3-1.7B-GGUF', 'Qwen3-1.7B-Q4_K_M.gguf')"
COPY --chown=user . .
EXPOSE 7860
CMD ["python", "app.py"]
