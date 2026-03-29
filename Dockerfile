FROM python:3.12-slim AS base

RUN groupadd -r arxiv && useradd -r -g arxiv -d /app -s /sbin/nologin arxiv

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

# Pre-download the embedding model into /app/.cache so the read-only
# container can access it as the non-root 'arxiv' user.
ENV SENTENCE_TRANSFORMERS_HOME=/app/.cache
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

RUN chown -R arxiv:arxiv /app

USER arxiv

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
