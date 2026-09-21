# syntax=docker/dockerfile:1.7
# ─── Single-container deploy: Hugging Face Spaces, Railway, Render ────────────
# One image serves the API, the WebSocket and the built SPA from one origin.
# PostgreSQL is external — pass DATABASE_URL (Railway's Postgres plugin sets it
# automatically; on Hugging Face use a free Neon or Supabase database).
#
#   docker build -t voice-hr .
#   docker run -p 7860:7860 -e DATABASE_URL=... -e GROQ_API_KEY=... voice-hr

# ─── Frontend ─────────────────────────────────────────────────────────────────
FROM node:22-alpine AS web
WORKDIR /web
COPY apps/web/package.json apps/web/package-lock.json ./
RUN npm ci
COPY apps/web/ ./
RUN npm run build

# ─── API dependencies ─────────────────────────────────────────────────────────
FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 HF_HOME=/opt/models/hf
WORKDIR /app

FROM base AS deps
COPY apps/api/pyproject.toml ./
RUN mkdir -p app && touch app/__init__.py \
 && pip install --extra-index-url https://download.pytorch.org/whl/cpu ".[ml]"

# ─── Models: train the scorer, pre-fetch Whisper + embeddings ────────────────
FROM deps AS models
COPY apps/api/app ./app
COPY apps/api/ml ./ml
RUN python -m ml.generate_dataset --rows 8000 \
 && python -m ml.train_scorer \
 && python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/multi-qa-MiniLM-L6-cos-v1')" \
 && python -c "from faster_whisper import WhisperModel; WhisperModel('base.en', device='cpu', compute_type='int8')"

# ─── Runtime ──────────────────────────────────────────────────────────────────
FROM base AS runtime
RUN useradd --create-home --uid 1000 app
COPY --from=deps /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=deps /usr/local/bin /usr/local/bin
COPY --from=models --chown=app:app /opt/models /opt/models
COPY --chown=app:app apps/api/ ./
COPY --from=models --chown=app:app /app/ml/artifacts ./ml/artifacts
COPY --from=web --chown=app:app /web/dist ./static
RUN sed -i 's/\r$//' docker-entrypoint.sh && chmod +x docker-entrypoint.sh && mkdir -p storage && chown app:app storage

USER app
ENV STATIC_DIR=/app/static \
    ENVIRONMENT=production \
    LLM_PROVIDER=groq \
    PORT=7860
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
  CMD python -c "import urllib.request,os; urllib.request.urlopen(f'http://localhost:{os.environ.get(\"PORT\",\"7860\")}/api/v1/health')" || exit 1
ENTRYPOINT ["./docker-entrypoint.sh"]
