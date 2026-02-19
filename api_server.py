"""
FastAPI server for Wan2GP video generation API.
Handles job submission, status tracking, and video downloads.
"""
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from celery_app import generate_video, r, app as celery_app
from wan2gp_client import Wan2GPClient
import time
import os
import json
from pathlib import Path

app = FastAPI(
    title="Wan2GP Video Generation API",
    version="1.0.0",
    description="Production API for LTX-2 video generation with queue management and job persistence"
)

# Enable CORS for browser dashboards
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (restrict in production if needed)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = Wan2GPClient()

# Request/Response Models
class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=10, max_length=2000, description="Video description")
    resolution: str = Field(default="1280x720", pattern=r"^\d+x\d+$")
    video_length: int = Field(default=81, ge=81, le=721, description="Frame count (81≈5s, 361≈15s)")
    seed: int = Field(default=-1, ge=-1)
    steps: int = Field(default=8, ge=4, le=50, description="Denoising steps")
    loras: dict = Field(default={}, description="LoRA dict: {filename: multiplier}")
    settings_override: dict = Field(default=None, description="Optional full settings dictionary override")
    webhook_url: str = Field(default="", description="Optional webhook for completion notification")


class JobResponse(BaseModel):
    job_id: str
    status: str
    queue_position: int = 0
    estimated_wait_minutes: int = 0

class JobStatus(BaseModel):
    job_id: str
    status: str
    prompt: str
    created_at: str
    started_at: str | None = None
    completed_at: str | None = None
    failed_at: str | None = None
    duration_seconds: int | None = None
    output_file: str | None = None
    error: str | None = None
    client_ip: str | None = None
    retry_count: int = 0

@app.post("/generate", response_model=JobResponse)
async def create_job(request: Request, req: GenerateRequest):
    """
    Submit a video generation job to the queue.
    
    Returns job_id and estimated wait time based on queue depth.
    """
    job_id = f"job_{int(time.time())}_{os.urandom(4).hex()}"
    client_ip = request.client.host
    
    # Store full job metadata in Redis
    job_data = {
        "prompt": req.prompt,
        "resolution": req.resolution,
        "video_length": str(req.video_length),
        "seed": str(req.seed),
        "steps": str(req.steps),
        "loras": str(req.loras),
        "settings_override": json.dumps(req.settings_override) if req.settings_override else "",
        "webhook_url": req.webhook_url,
        "status": "queued",
        "created_at": str(time.time()),
        "client_ip": client_ip,
        "retry_count": "0"
    }
    
    r.hset(f"job:{job_id}", mapping=job_data)
    
    # Enqueue Celery task
    generate_video.apply_async(
        args=[job_id, req.prompt],
        kwargs={
            "resolution": req.resolution,
            "video_length": req.video_length,
            "seed": req.seed,
            "steps": req.steps,
            "loras": req.loras,
            "settings_override": req.settings_override
        },
        task_id=job_id
    )
    
    # Estimate wait time
    active_tasks = celery_app.control.inspect().active()
    queue_length = sum(len(tasks) for tasks in (active_tasks or {}).values())
    avg_gen_time = 4  # minutes (conservative estimate)
    estimated_wait = queue_length * avg_gen_time
    
    return JobResponse(
        job_id=job_id,
        status="queued",
        queue_position=queue_length + 1,
        estimated_wait_minutes=estimated_wait
    )

@app.get("/status/{job_id}", response_model=JobStatus)
async def get_status(job_id: str):
    """Get detailed job status and metadata."""
    job_data = r.hgetall(f"job:{job_id}")
    if not job_data:
        raise HTTPException(404, "Job not found")
    
    return JobStatus(
        job_id=job_id,
        status=job_data.get("status", "unknown"),
        prompt=job_data.get("prompt", ""),
        created_at=job_data.get("created_at", ""),
        started_at=job_data.get("started_at"),
        completed_at=job_data.get("completed_at"),
        failed_at=job_data.get("failed_at"),
        duration_seconds=int(job_data.get("duration_seconds", 0)) or None,
        output_file=job_data.get("output_file"),
        error=job_data.get("error"),
        client_ip=job_data.get("client_ip"),
        retry_count=int(job_data.get("retry_count", 0))
    )

@app.post("/retry/{job_id}")
async def retry_job(job_id: str):
    """
    Retry a failed job.
    Only works for jobs with status 'failed' or 'error'.
    """
    job_data = r.hgetall(f"job:{job_id}")
    if not job_data:
        raise HTTPException(404, "Job not found")
    
    status = job_data.get("status")
    if status not in ["failed", "error"]:
        raise HTTPException(400, f"Cannot retry job with status: {status}")
    
    # Re-enqueue with same parameters
    prompt = job_data.get("prompt", "")
    resolution = job_data.get("resolution", "1280x720")
    video_length = int(job_data.get("video_length", 81))
    seed = int(job_data.get("seed", -1))
    steps = int(job_data.get("steps", 8))
    loras = eval(job_data.get("loras", "{}"))
    settings_override_raw = job_data.get("settings_override", "")
    settings_override = json.loads(settings_override_raw) if settings_override_raw else None
    
    # Reset status
    r.hset(f"job:{job_id}", mapping={
        "status": "queued",
        "error": "",
        "failed_at": "",
        "retry_count": str(int(job_data.get("retry_count", 0)) + 1)
    })
    
    # Re-submit
    generate_video.apply_async(
        args=[job_id, prompt],
        kwargs={
            "resolution": resolution,
            "video_length": video_length,
            "seed": seed,
            "steps": steps,
            "loras": loras,
            "settings_override": settings_override
        },
        task_id=f"{job_id}_retry_{int(time.time())}"
    )
    
    return {"job_id": job_id, "status": "queued", "message": "Job resubmitted for retry"}

@app.get("/download/{job_id}")
async def download_video(job_id: str):
    """Download the generated video file."""
    job_data = r.hgetall(f"job:{job_id}")
    if not job_data:
        raise HTTPException(404, "Job not found")
    
    if job_data.get("status") != "completed":
        raise HTTPException(400, f"Job status: {job_data.get('status')}. Cannot download.")
    
    # Search for video file in outputs directory
    # Handle both old format (job_job_ID.mp4) and new format (job_ID.mp4)
    try:
        stdout, _ = client._docker_exec(f"ls /workspace/outputs/*{job_id}*.mp4")
        files = [f.strip() for f in stdout.strip().split('\n') if f.strip()]
        
        if not files:
            raise HTTPException(404, f"Video file not found for job {job_id}")
        
        # Use the first match (should only be one)
        # Extract just the filename (basename)
        from pathlib import Path as PathLib
        filename = PathLib(files[0]).name
        file_path = client.get_output_path(filename)
        
        if not file_path.exists():
            raise HTTPException(404, f"Video file path not accessible: {filename}")
        
        return FileResponse(
            str(file_path),
            media_type="video/mp4",
            filename=f"{job_id}.mp4"
        )
    except Exception as e:
        raise HTTPException(500, f"Error finding video: {str(e)}")

@app.get("/queue")
async def queue_status():
    """Get current queue statistics."""
    all_jobs = r.keys("job:*")
    statuses = {"queued": 0, "running": 0, "completed": 0, "error": 0, "failed": 0, "retrying": 0}
    
    for job_key in all_jobs:
        status = r.hget(job_key, "status")
        if status in statuses:
            statuses[status] += 1
    
    # Get active Celery tasks
    active_tasks = celery_app.control.inspect().active()
    active_count = sum(len(tasks) for tasks in (active_tasks or {}).values())
    
    return {
        "total_jobs": len(all_jobs),
        "queue_length": statuses["queued"],
        "running": statuses["running"],
        "completed": statuses["completed"],
        "errors": statuses["error"] + statuses["failed"],
        "retrying": statuses["retrying"],
        "active_celery_tasks": active_count
    }

@app.get("/jobs/list")
async def list_jobs(status: str | None = None, limit: int = 50):
    """
    List recent jobs with optional status filter.
    
    Args:
        status: Filter by status (queued, running, completed, error, failed)
        limit: Max number of jobs to return
    """
    all_jobs = r.keys("job:*")
    jobs = []
    
    for job_key in all_jobs:
        if len(jobs) >= limit:
            break
        
        job_data = r.hgetall(job_key)
        job_status = job_data.get("status")
        
        if status and job_status != status:
            continue
        
        job_id = job_key.decode() if isinstance(job_key, bytes) else job_key
        job_id = job_id.replace("job:", "")
        
        jobs.append({
            "job_id": job_id,
            "status": job_status,
            "prompt": job_data.get("prompt", "")[:100],  # Truncate
            "created_at": job_data.get("created_at"),
            "client_ip": job_data.get("client_ip")
        })
    
    # Sort by created_at (newest first)
    jobs.sort(key=lambda x: float(x.get("created_at", 0)), reverse=True)
    
    return {"jobs": jobs, "total": len(jobs)}

@app.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    """Delete a job and its metadata."""
    deleted = r.delete(f"job:{job_id}")
    if deleted == 0:
        raise HTTPException(404, "Job not found")
    return {"message": "Job deleted", "job_id": job_id}

@app.get("/health")
async def health():
    """Health check endpoint."""
    try:
        # Check Redis
        r.ping()
        redis_status = "healthy"
    except:
        redis_status = "unhealthy"
    
    try:
        # Check Celery
        celery_inspect = celery_app.control.inspect()
        celery_workers = celery_inspect.ping()
        celery_status = "healthy" if celery_workers else "no workers"
    except:
        celery_status = "unhealthy"
    
    try:
        # Check wan2gp container
        loras = client.list_outputs(limit=1)
        wan2gp_status = "reachable"
    except:
        wan2gp_status = "unreachable"
    
    return {
        "api": "healthy",
        "redis": redis_status,
        "celery": celery_status,
        "wan2gp_container": wan2gp_status
    }

@app.get("/loras")
async def list_loras():
    """List available LoRA files."""
    try:
        stdout, _ = client._docker_exec("ls /workspace/loras/ltx2/*.safetensors")
        files = [f.split('/')[-1] for f in stdout.strip().split('\n') if f and '.safetensors' in f]
        return {"loras": files}
    except Exception as e:
        raise HTTPException(500, f"Failed to list LoRAs: {str(e)}")

@app.get("/dashboard")
async def dashboard():
    """Serve the video generation dashboard."""
    dashboard_path = Path("/app/dashboard.html")
    if not dashboard_path.exists():
        raise HTTPException(404, "Dashboard not found. Please ensure dashboard.html is uploaded to /app/")
    return FileResponse(str(dashboard_path), media_type="text/html")

@app.get("/admin")
async def admin():
    """Serve the admin job management page."""
    admin_path = Path("/app/admin.html")
    if not admin_path.exists():
        raise HTTPException(404, "Admin page not found. Please ensure admin.html is uploaded to /app/")
    return FileResponse(str(admin_path), media_type="text/html")

@app.get("/")
async def root():
    """API root with documentation links."""
    return {
        "name": "Wan2GP Video Generation API",
        "version": "1.0.0",
        "docs": "/docs",
        "dashboard": "/dashboard",
        "admin": "/admin",
        "health": "/health",
        "queue": "/queue"
    }

