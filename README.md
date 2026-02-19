# Wan2GP Video Generation API

Production API for AI video generation using Wan2GP (LTX-2 Distilled 19B) with queue management, job persistence, and auto-retry.

## Architecture

- **FastAPI** — REST API server (port 8000)
- **Celery** — Background task queue (1 worker = 1 GPU)
- **Redis** — Message broker + job metadata store
- **Wan2GP** — GPU container for actual video generation (controlled via `docker exec`)

## Prerequisites

- Docker & Docker Compose
- A running `wan2gp` container with GPU access (the video generation engine)
- Shared volumes for outputs and settings

## Quick Start

```bash
# Clone
git clone <your-repo-url>
cd wan2gp-api

# Update volume paths in docker-compose.yml to match your server
# Then start all services
docker compose up -d --build

# Check health
curl http://localhost:8000/health
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/generate` | Submit video generation job |
| GET | `/status/{job_id}` | Get job status |
| GET | `/download/{job_id}` | Download generated video |
| POST | `/retry/{job_id}` | Retry a failed job |
| GET | `/queue` | Queue statistics |
| GET | `/jobs/list` | List all jobs |
| DELETE | `/jobs/{job_id}` | Delete a job |
| GET | `/loras` | List available LoRAs |
| GET | `/health` | Health check |
| GET | `/dashboard` | Video generation UI |
| GET | `/admin` | Admin panel |
| GET | `/docs` | Swagger API docs |

## Configuration

Key environment variables (set in `docker-compose.yml`):

| Variable | Default | Description |
|----------|---------|-------------|
| `REDIS_HOST` | `redis` | Redis hostname |
| `REDIS_PORT` | `6379` | Redis port |
| `WAN2GP_OUTPUTS_DIR` | `/workspace/outputs` | Video outputs directory |
| `WAN2GP_SETTINGS_DIR` | `/workspace/settings` | Settings JSON directory |

## Setup on New Server

1. Ensure the `wan2gp` container is running with GPU
2. Update volume paths in `docker-compose.yml` to match your disk layout
3. Run `docker compose up -d --build`
