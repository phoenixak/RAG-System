# Hugging Face Space (Docker SDK). llama.cpp comes from a wheel prebuilt in this same base image
# (space/build_wheel.sh): compiling it on the Space builder runs past its time limit, and the
# public prebuilt Linux wheels target musl. The models are baked in so a cold start downloads nothing.
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* && useradd -m -u 1000 user

ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
# fetched over HTTPS rather than COPY'd: the builder's checkout can hand over the LFS pointer instead of the file
ARG LLAMA_WHEEL=https://huggingface.co/spaces/Phoenixak99/RAG-System-Demo/resolve/main/wheels/llama_cpp_python-0.3.36-py3-none-linux_x86_64.whl
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install "$LLAMA_WHEEL"
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
