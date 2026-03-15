# Stage 1: Builder — install dependencies
FROM python:3.11-slim AS builder

# TARGETARCH is set automatically by Docker buildx (amd64 or arm64)
ARG TARGETARCH

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install production dependencies (no dev group)
RUN uv sync --no-dev --frozen

# Install PyTorch CPU-only.
# On x86_64: use the dedicated CPU index to avoid pulling CUDA wheels.
# On ARM64 (Oracle Cloud Ampere): PyPI only ships CPU wheels, so default index works.
RUN if [ "$TARGETARCH" = "arm64" ]; then \
        uv pip install --python .venv/bin/python torch torchvision; \
    else \
        uv pip install --python .venv/bin/python \
            torch torchvision --index-url https://download.pytorch.org/whl/cpu; \
    fi


# Stage 2: Runtime — minimal production image
FROM python:3.11-slim AS runtime

WORKDIR /app

# Install curl for healthcheck + libgl for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual env from builder
COPY --from=builder /app/.venv /app/.venv

# Make sure venv is on PATH
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

# Copy application code and configs
COPY src/ src/
COPY configs/ configs/
COPY frontend/ frontend/

# Create non-root user
RUN groupadd --system appuser && useradd --system --gid appuser appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
