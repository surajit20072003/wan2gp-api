"""
Celery tasks for Wan2GP video generation.
Handles async job processing with Redis queue and persistence.
"""
from celery import Celery
from wan2gp_client import Wan2GPClient
import redis
import json
import time
import os

# Celery configuration
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))

BROKER_URL = f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}"
RESULT_BACKEND = f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB + 1}"

app = Celery(
    'wan2gp_tasks',
    broker=BROKER_URL,
    backend=RESULT_BACKEND
)

app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=1800,  # 30 min hard limit
    task_soft_time_limit=1500,  # 25 min soft warning
    worker_prefetch_multiplier=1,  # Only fetch 1 job at a time (1 GPU)
    worker_concurrency=1,  # 1 worker = 1 GPU
    broker_connection_retry_on_startup=True,
    result_expires=86400  # Keep results for 24 hours
)

# Redis client for job metadata
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB + 2, decode_responses=True)

client = Wan2GPClient()

@app.task(bind=True, max_retries=2, default_retry_delay=60)
def generate_video(self, job_id, prompt, **kwargs):
    """
    Celery task for headless video generation.
    
    Args:
        job_id: Unique job identifier
        prompt: Video description
        **kwargs: resolution, video_length, seed, steps, loras, etc.
    
    Returns:
        {"status": str, "output_file": str|None, "error": str|None}
    """
    # Update status: running
    r.hset(f"job:{job_id}", mapping={
        "status": "running",
        "started_at": str(time.time()),
        "celery_task_id": self.request.id,
        "retry_count": self.request.retries
    })
    
    try:
        # Submit to Wan2GP
        result = client.submit_job(
            job_id=job_id,
            prompt=prompt,
            **kwargs
        )
        
        if result["status"] == "success":
            # Update status: completed
            r.hset(f"job:{job_id}", mapping={
                "status": "completed",
                "output_file": result["output_file"] or "unknown.mp4",
                "completed_at": str(time.time()),
                "stdout": result["stdout"],
                "stderr": result["stderr"]
            })
            
            # Calculate duration
            started_at = float(r.hget(f"job:{job_id}", "started_at") or time.time())
            duration = time.time() - started_at
            r.hset(f"job:{job_id}", "duration_seconds", str(int(duration)))
            
            return {
                "status": "completed",
                "output_file": result["output_file"],
                "duration_seconds": int(duration)
            }
        else:
            # Generation failed
            raise Exception(f"Generation failed: {result['stderr']}")
    
    except Exception as e:
        error_msg = str(e)
        
        # Update status: error
        r.hset(f"job:{job_id}", mapping={
            "status": "error",
            "error": error_msg,
            "failed_at": str(time.time()),
            "retry_count": self.request.retries
        })
        
        # Retry logic
        if self.request.retries < self.max_retries:
            r.hset(f"job:{job_id}", "status", "retrying")
            raise self.retry(exc=e, countdown=60)  # Retry after 60s
        else:
            # Max retries reached
            r.hset(f"job:{job_id}", "status", "failed")
            return {
                "status": "failed",
                "error": error_msg,
                "output_file": None
            }

@app.task
def cleanup_old_jobs(days=7):
    """
    Cleanup jobs older than N days.
    Run this periodically via Celery beat.
    """
    cutoff = time.time() - (days * 24 * 60 * 60)
    deleted_count = 0
    
    for key in r.keys("job:*"):
        created_at = float(r.hget(key, "created_at") or 0)
        if created_at < cutoff:
            r.delete(key)
            deleted_count += 1
    
    return {"deleted_jobs": deleted_count, "cutoff_days": days}
