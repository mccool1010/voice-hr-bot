# syntax=docker/dockerfile:1.7
# ─── Lean deploy image: fits a free 512 MB host (Render free tier) ────────────
# The full image carries PyTorch, local Whisper and an embedding model (~3 GB,
# ~1.5 GB RAM). This one serves the same app with:
#   speech-to-text  Groq's hosted Whisper (same word timings)
#   scorer          NumPy inference over weights exported from PyTorch
#   embeddings      off (informational only; never part of the score)
# PyTorch is still used — in the `train` stage below — just not shipped.
#
#   docker build -f deploy/render/Dockerfile -t voice-hr:lean .

# ─── Frontend ─────────────────────────────────────────────────────────────────
FROM node:22-alpine AS web
WORKDIR /web
COPY apps/web/package.json apps/web/package-lock.json ./
RUN npm ci
COPY apps/web/ ./
RUN npm run build

FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app

# ─── Runtime dependencies (no [ml] extra) ─────────────────────────────────────
FROM base AS deps
COPY apps/api/pyproject.toml ./
RUN mkdir -p app && touch app/__init__.py && pip install .

# ─── Train with PyTorch, export to NumPy, keep only the export ───────────────
FROM deps AS train
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY apps/api/app ./app
COPY apps/api/ml ./ml
RUN python -m ml.generate_dataset --rows 8000 && python -m ml.train_scorer --device cpu

# ─── Runtime ──────────────────────────────────────────────────────────────────
FROM base AS runtime
RUN useradd --create-home --uid 1000 app
COPY --from=deps /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=deps /usr/local/bin /usr/local/bin
COPY --chown=app:app apps/api/ ./
RUN rm -rf ml/data ml/artifacts && mkdir -p ml/artifacts
COPY --from=train --chown=app:app /app/ml/artifacts/answer_scorer.npz /app/ml/artifacts/answer_scorer.metrics.json ./ml/artifacts/
COPY --from=web --chown=app:app /web/dist ./static
RUN sed -i 's/\r$//' docker-entrypoint.sh && chmod +x docker-entrypoint.sh \
 && mkdir -p storage && chown app:app storage

USER app
ENV STATIC_DIR=/app/static \
    ENVIRONMENT=production \
    LLM_PROVIDER=groq \
    SPEECH_PROVIDER=groq \
    EMBEDDINGS_ENABLED=false \
    PORT=10000
EXPOSE 10000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request,os; urllib.request.urlopen(f'http://localhost:{os.environ.get(\"PORT\",\"10000\")}/api/v1/health')" || exit 1
ENTRYPOINT ["./docker-entrypoint.sh"]
