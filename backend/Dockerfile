# Backend image: API and worker share this. Uses uv for fast, reproducible installs.
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_CACHE=1 \
    PATH="/app/.venv/bin:$PATH"

# uv from the official distroless image, pinned for reproducible builds.
COPY --from=ghcr.io/astral-sh/uv:0.11.29 /uv /uvx /bin/

WORKDIR /app

# Install locked dependencies first for layer caching.
COPY pyproject.toml README.md uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project \
    --extra api --extra db --extra llm --extra parsing --extra embeddings

COPY lattice ./lattice
RUN uv sync --locked --no-dev --no-editable \
    --extra api --extra db --extra llm --extra parsing --extra embeddings

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/readyz', timeout=4).status==200 else 1)"

# Default to the API; docker-compose overrides command for the worker.
CMD ["uvicorn", "lattice.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
