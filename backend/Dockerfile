# Backend image: API and worker share this. Uses uv for fast, reproducible installs.
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_SYSTEM_PYTHON=1 \
    UV_COMPILE_BYTECODE=1

# uv from the official distroless image.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Install dependencies first for layer caching.
COPY pyproject.toml README.md ./
RUN uv pip install --system ".[api,db,llm,parsing,embeddings]"

COPY lattice ./lattice

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/readyz', timeout=4).status==200 else 1)"

# Default to the API; docker-compose overrides command for the worker.
CMD ["uvicorn", "lattice.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
