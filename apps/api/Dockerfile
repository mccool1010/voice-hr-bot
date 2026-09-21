# syntax=docker/dockerfile:1.7
# ─── API image ────────────────────────────────────────────────────────────────
# CPU PyTorch by default: the CUDA wheels add ~3 GB and free hosting has no GPU.
# For a local GPU build: --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu130
# (RTX 50-series / Blackwell cards need CUDA 12.8+; torch 2.14 ships it as cu130.)

FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/opt/models/hf
WORKDIR /app

# ─── Dependencies (cached layer — only rebuilds when pyproject changes) ──────
FROM base AS deps
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
COPY pyproject.toml ./
RUN mkdir -p app && touch app/__init__.py \
 && pip install --extra-index-url ${TORCH_INDEX} ".[ml]"

# ─── Model artefacts: train the scorer, pre-fetch Whisper + embeddings ───────
# Baked into the image so a cold container serves its first request quickly
# instead of downloading ~250 MB on the request path.
FROM deps AS models
ARG WHISPER_MODEL=base.en
ARG EMBEDDING_MODEL=sentence-transformers/multi-qa-MiniLM-L6-cos-v1
COPY app ./app
COPY ml ./ml
RUN python -m ml.generate_dataset --rows 8000 \
 && python -m ml.train_scorer \
 && python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('${EMBEDDING_MODEL}')" \
 && python -c "from faster_whisper import WhisperModel; WhisperModel('${WHISPER_MODEL}', device='cpu', compute_type='int8')"

# ─── Runtime ──────────────────────────────────────────────────────────────────
FROM base AS runtime
# Hugging Face Spaces runs containers as uid 1000.
RUN useradd --create-home --uid 1000 app
COPY --from=deps /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=deps /usr/local/bin /usr/local/bin
COPY --from=models --chown=app:app /opt/models /opt/models
COPY --chown=app:app . .
COPY --from=models --chown=app:app /app/ml/artifacts ./ml/artifacts
RUN sed -i 's/\r$//' docker-entrypoint.sh && chmod +x docker-entrypoint.sh && mkdir -p storage && chown app:app storage

USER app
ENV PORT=8000
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request,os; urllib.request.urlopen(f'http://localhost:{os.environ.get(\"PORT\",\"8000\")}/api/v1/health')" || exit 1
ENTRYPOINT ["./docker-entrypoint.sh"]
