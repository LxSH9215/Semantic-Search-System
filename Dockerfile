# ─────────────────────────────────────────────────────────────
# Dockerfile — Semantic Search Service
# ─────────────────────────────────────────────────────────────
# Design decisions:
#   • python:3.11-slim as base — small image (~120 MB) with glibc for C extensions.
#   • Install PyTorch CPU-only wheel explicitly to avoid pulling in the
#     ~2 GB CUDA-enabled wheel (we don't need GPU for inference at this scale).
#   • Copy requirements first, then code — this maximises Docker layer
#     caching so code changes don't re-download all dependencies.
#   • Run as non-root user for security.
#   • Expose port 8000 (matches config.API_PORT).
# ─────────────────────────────────────────────────────────────

FROM python:3.11-slim AS base

# Prevent Python from writing .pyc files (smaller image, easier debugging)
ENV PYTHONDONTWRITEBYTECODE=1
# Force unbuffered stdout/stderr so logs appear in real time
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# ── 1. System dependencies ──────────────────────────────
# gcc is needed for building some Python C extensions (e.g., faiss-cpu).
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# ── 2. Python dependencies ──────────────────────────────
COPY requirements.txt .
# Install PyTorch CPU-only first (saves ~1.5 GB vs. default CUDA wheel)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# ── 3. Application code ─────────────────────────────────
COPY config.py data_loader.py embeddings.py clustering.py semantic_cache.py api.py main.py ./

# ── 4. Non-root user ────────────────────────────────────
RUN useradd --create-home appuser
USER appuser

# ── 5. Health check ─────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# ── 6. Run ──────────────────────────────────────────────
EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
