# syntax=docker/dockerfile:1.7

# --- Stage 1: build a wheel + isolated dependency tree ----------------------
FROM python:3.12-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /build

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE ./
COPY src ./src

RUN pip install --upgrade pip build wheel \
    && pip wheel --wheel-dir /wheels .

RUN pip install --prefix /install --no-warn-script-location /wheels/*.whl

# --- Stage 2: minimal runtime image ----------------------------------------
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 \
    PORT=8080 \
    EPICURE_DATA_DIR=/app/data

WORKDIR /app

COPY --from=builder /install /usr/local

COPY data /app/data
COPY assets /app/assets

RUN useradd -r -u 10001 epicure \
    && chown -R epicure:epicure /app

USER epicure

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen('http://localhost:8080/healthz', timeout=2).status == 200 else 1)" \
    || exit 1

CMD ["python", "-m", "epicure_mcp.server"]
