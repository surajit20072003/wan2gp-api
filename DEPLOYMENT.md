# Wan2GP API - Docker Compose Deployment Guide

## Quick Start

### 1. Upload Files to Server

From your Windows PC:

```powershell
# Copy all files to server
scp Dockerfile docker-compose.yml requirements.txt administrator@69.30.250.80:/home/administrator/wan2gp-api/
scp wan2gp_client.py celery_app.py api_server.py administrator@69.30.250.80:/home/administrator/wan2gp-api/
```

### 2. Deploy on Server

SSH to server and deploy:

```bash
ssh administrator@69.30.250.80

cd /home/administrator/wan2gp-api/

# Create shared Docker network
sudo docker network create wan2gp-network

# Connect existing wan2gp container to network
sudo docker network connect wan2gp-network wan2gp

# Build and start API stack
sudo docker-compose up -d --build

# Verify all services running
sudo docker-compose ps

# Check logs
sudo docker-compose logs -f
```

### 3. Test API from Windows

```python
import requests

API_URL = "http://69.30.250.80:8000"

# Health check
health = requests.get(f"{API_URL}/health").json()
print(health)

# Submit job
job = requests.post(f"{API_URL}/generate", json={
    "prompt": "A cinematic drone shot of a futuristic city with neon lights and flying vehicles",
    "video_length": 81,
    "loras": {"ltx-2-19b-ic-lora-detailer.safetensors": 0.8}
}).json()

print(f"Job ID: {job['job_id']}")
print(f"Queue Position: {job['queue_position']}")

# Check status
import time
job_id = job['job_id']
while True:
    status = requests.get(f"{API_URL}/status/{job_id}").json()
    print(f"Status: {status['status']}")
    if status['status'] in ['completed', 'failed', 'error']:
        break
    time.sleep(10)

# Download video (if successful)
if status['status'] == 'completed':
    video = requests.get(f"{API_URL}/download/{job_id}")
    with open(f"{job_id}.mp4", 'wb') as f:
        f.write(video.content)
    print(f"Downloaded: {job_id}.mp4")
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/generate` | POST | Submit new job |
| `/status/{job_id}` | GET | Get job status |
| `/download/{job_id}` | GET | Download video |
| `/retry/{job_id}` | POST | Retry failed job |
| `/queue` | GET | Queue statistics |
| `/jobs/list` | GET | List recent jobs |
| `/jobs/{job_id}` | DELETE | Delete job |
| `/health` | GET | Health check |
| `/loras` | GET | List available LoRAs |
| `/docs` | GET | Swagger UI |

## Job Persistence Features

All job metadata is stored in Redis and persists across container restarts:

- **Prompt** - Original text description
- **Client IP** - Requestor IP address  
- **Timestamps** - Created, started, completed, failed
- **Duration** - Generation time in seconds
- **Status** - queued → running → completed/failed
- **Retry Count** - Number of retry attempts
- **Output File** - Generated video filename
- **Error Messages** - Full stderr on failure

## Retry Mechanism

Failed jobs can be retried:

```python
# Retry a failed job
retry_response = requests.post(f"{API_URL}/retry/{job_id}").json()
```

Automatic retries:
- Celery will auto-retry failed jobs up to 2 times
- 60-second delay between retries
- Status updates: `error` → `retrying` → `completed`/`failed`

## Monitoring

### View Logs

```bash
# All services
sudo docker-compose logs -f

# Specific service
sudo docker-compose logs -f api-server
sudo docker-compose logs -f celery-worker
sudo docker-compose logs -f redis
```

### Check Queue

```bash
curl http://localhost:8000/queue | jq
```

### Check Health

```bash
curl http://localhost:8000/health | jq
```

## Stopping/Restarting

```bash
# Stop all services
sudo docker-compose down

# Restart services (preserves Redis data)
sudo docker-compose restart

# Rebuild after code changes
sudo docker-compose down
sudo docker-compose up -d --build
```

## Persistence Verification

Redis data persists in Docker volume `wan2gp_redis-data`.

To verify job data survives restart:

```bash
# Submit job
curl -X POST http://localhost:8000/generate -H "Content-Type: application/json" -d '{"prompt":"test","video_length":81}'

# Note job_id
# Stop containers
sudo docker-compose down

# Start containers
sudo docker-compose up -d

# Check job still exists
curl http://localhost:8000/status/{job_id}
```

## Scaling

Current configuration supports:
- **1 concurrent generation** (1 GPU)
- **100+ queued jobs** (limited by Redis memory)
- **~12-15 videos/hour** (4 min avg per video)

For multi-GPU scaling, see build plan section on horizontal worker scaling.

## Troubleshooting

### API not accessible from Windows

Check firewall on server:
```bash
sudo ufw allow 8000/tcp
```

### Celery worker not starting

Check Docker socket permissions:
```bash
sudo chmod 666 /var/run/docker.sock
```

### Redis data loss

Verify AOF is enabled:
```bash
sudo docker exec wan2gp-redis redis-cli CONFIG GET appendonly
```

Should return `appendonly` = `yes`.
