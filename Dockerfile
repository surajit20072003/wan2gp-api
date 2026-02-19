FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for Docker CLI
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Docker CLI (to execute docker exec commands)
RUN curl -fsSL https://get.docker.com -o get-docker.sh && \
    sh get-docker.sh && \
    rm get-docker.sh

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY wan2gp_client.py celery_app.py api_server.py dashboard.html admin.html ./

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Default command (overridden in docker-compose for celery-worker)
CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8000"]
