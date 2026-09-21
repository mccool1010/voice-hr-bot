#!/bin/sh
set -e

# Apply migrations before serving. Safe to run on every boot: Alembic is a
# no-op when the schema is current.
echo "Running database migrations..."
alembic upgrade head

# Railway and most PaaS inject PORT; Hugging Face Spaces expects 7860.
exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}" --proxy-headers --forwarded-allow-ips="*"
