FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for Docker CLI
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Docker CLI (to execute docker exec commands into GPU containers)
RUN curl -fsSL https://get.docker.com -o get-docker.sh && \
    sh get-docker.sh && \
    rm get-docker.sh

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY gpu_scheduler.py wan2gp_client.py api_server.py celery_app.py ./
# Copy optional HTML dashboard files (if they exist)
COPY *.html ./

# Port configuration (override via WAN2GP_PORT env var)
ARG WAN2GP_PORT=8000
ENV WAN2GP_PORT=${WAN2GP_PORT}
EXPOSE ${WAN2GP_PORT}

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:${WAN2GP_PORT}/health || exit 1

# Default: API server
CMD sh -c "uvicorn api_server:app --host 0.0.0.0 --port ${WAN2GP_PORT}"
