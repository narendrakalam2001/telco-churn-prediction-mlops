# ============================================================
# DOCKERFILE — Telco Churn Prediction API
# Multi-stage build: builder installs deps, runner is lean image
# API only — does NOT include training or dashboard deps
# ============================================================

# ── Stage 1: builder ─────────────────────────────────────────
FROM python:3.10.13-slim AS builder

WORKDIR /app

# Install build tools needed for catboost / lightgbm / scipy
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && \
    rm -rf /var/lib/apt/lists/*

COPY requirements_api.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements_api.txt


# ── Stage 2: runner ──────────────────────────────────────────
FROM python:3.10.13-slim AS runner

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY src/           ./src/
COPY services/      ./services/
COPY serving/       ./serving/
COPY churn_models/  ./churn_models/
COPY logs/          ./logs/

# Non-root user for security
RUN useradd -m appuser && chown -R appuser /app
USER appuser

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "serving.churn_api:app", "--host", "0.0.0.0", "--port", "8000"]
