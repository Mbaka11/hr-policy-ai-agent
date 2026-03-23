FROM python:3.11-slim

WORKDIR /app

# Install system deps for health check
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY src/ ./src/
COPY prompts/ ./prompts/
COPY scripts/ ./scripts/
COPY data/raw/ ./data/raw/

# ChromaDB persistent storage (populated at runtime via ingest.py)
RUN mkdir -p data/chroma_db

EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Ingest documents on startup, then launch Streamlit
CMD ["sh", "-c", "python scripts/ingest.py && streamlit run src/app.py --server.port=8501 --server.address=0.0.0.0"]
