"""
Wan2GP Video Generation API — Multi-GPU Edition.
FastAPI server with built-in GPU scheduler for 3 GPU containers.
Replaces Celery with a thread-based scheduler for direct GPU management.
"""
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict
import os
import time
import logging

from gpu_scheduler import scheduler
from wan2gp_client import Wan2GPClient

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="Wan2GP Multi-GPU Video Generation API",
    description="AI video generation with 3-GPU round-robin scheduling",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Client for direct Docker operations (non-scheduled)
client = Wan2GPClient()


# ── Request/Response Models ─────────────────────────────────────────

class GenerateRequest(BaseModel):
    prompt: str
    resolution: str = "1280x720"
    video_length: int = 361
    seed: int = -1
    steps: int = 8
    loras: Optional[Dict[str, float]] = None
    settings_override: Optional[Dict] = None
    webhook_url: Optional[str] = ""


class JobResponse(BaseModel):
    job_id: str
    status: str
    queue_position: int
    estimated_wait_minutes: float
    gpu_id: Optional[int] = None
    message: str = ""


class JobStatus(BaseModel):
    job_id: str
    status: str
    prompt: str = ""
    resolution: str = ""
    output_file: Optional[str] = None
    error: Optional[str] = None
    gpu_id: Optional[str] = None
    created_at: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    duration_seconds: Optional[int] = None
    retry_count: int = 0


# ── API Endpoints ───────────────────────────────────────────────────

@app.post("/generate", response_model=JobResponse)
def create_job(request: Request, req: GenerateRequest):
    """Submit a new video generation job."""
    job_id = f"job_{int(time.time())}_{os.urandom(4).hex()}"
    client_ip = request.client.host if request.client else "unknown"

    logger.info(f"New job {job_id} from {client_ip}: {req.prompt[:80]}...")

    # Filter LoRAs
    available_loras = client.get_available_loras()
    filtered_loras = {}
    if req.loras:
        for lora, weight in req.loras.items():
            if lora in available_loras:
                filtered_loras[lora] = weight
            else:
                logger.warning(f"⚠️ LoRA {lora} not found in container. Skipping.")

    # Submit to scheduler (dispatches to GPU or queues)
    result = scheduler.submit_job(
        job_id=job_id,
        prompt=req.prompt,
        resolution=req.resolution,
        video_length=req.video_length,
        seed=req.seed,
        steps=req.steps,
        loras=filtered_loras,
        settings_override=req.settings_override,
        webhook_url=req.webhook_url or "",
        client_ip=client_ip,
    )

    # Estimate wait time
    queue_pos = result["queue_position"]
    avg_minutes = 4  # ~4 min per video on average
    est_wait = queue_pos * avg_minutes

    gpu_id = result.get("gpu_id")
    if result["status"] == "processing":
        message = f"Dispatched to GPU {gpu_id}"
    else:
        message = f"Queued at position {queue_pos}. All 3 GPUs busy."

    return JobResponse(
        job_id=job_id,
        status=result["status"],
        queue_position=queue_pos,
        estimated_wait_minutes=est_wait,
        gpu_id=gpu_id,
        message=message,
    )


@app.get("/status/{job_id}", response_model=JobStatus)
def get_status(job_id: str):
    """Get job status."""
    job = scheduler.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    duration = None
    if job.get("duration_seconds"):
        try:
            duration = int(job["duration_seconds"])
        except (ValueError, TypeError):
            pass

    return JobStatus(
        job_id=job_id,
        status=job.get("status", "unknown"),
        prompt=job.get("prompt", ""),
        resolution=job.get("resolution", ""),
        output_file=job.get("output_file"),
        error=job.get("error"),
        gpu_id=job.get("gpu_id"),
        created_at=job.get("created_at"),
        started_at=job.get("started_at"),
        completed_at=job.get("completed_at"),
        duration_seconds=duration,
        retry_count=int(job.get("retry_count", 0)),
    )


@app.get("/download/{job_id}")
def download_video(job_id: str):
    """Download generated video."""
    job = scheduler.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    if job.get("status") != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Job not completed. Current status: {job.get('status')}",
        )

    output_file = job.get("output_file")
    if not output_file:
        raise HTTPException(status_code=404, detail="No output file recorded")

    # Extract just the filename (output_file may be a container path like /workspace/outputs/xxx.mp4)
    output_filename = os.path.basename(output_file)

    # Search in the GPU's output directory
    gpu_id = int(job.get("gpu_id", 0))
    output_path = client.get_output_path(output_filename, gpu_id)

    if not output_path.exists():
        # Fallback: search all GPU output dirs
        for gid in [0, 1, 2]:
            candidate = client.get_output_path(output_filename, gid)
            if candidate.exists():
                output_path = candidate
                break
        else:
            raise HTTPException(
                status_code=404,
                detail=f"Output file not found: {output_filename}",
            )

    return FileResponse(
        path=str(output_path),
        media_type="video/mp4",
        filename=output_filename,
    )


@app.post("/retry/{job_id}")
def retry_job(job_id: str):
    """Retry a failed job."""
    job = scheduler.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    if job.get("status") not in ["failed", "error"]:
        raise HTTPException(
            status_code=400,
            detail=f"Can only retry failed jobs. Current: {job.get('status')}",
        )

    # Re-submit with original parameters
    import json
    loras = {}
    if job.get("loras"):
        try:
            loras = json.loads(job["loras"])
        except:
            pass

    settings_override = None
    if job.get("settings_override"):
        try:
            settings_override = json.loads(job["settings_override"])
        except:
            pass

    result = scheduler.submit_job(
        job_id=job_id,
        prompt=job.get("prompt", ""),
        resolution=job.get("resolution", "1280x720"),
        video_length=int(job.get("video_length", 81)),
        seed=int(job.get("seed", -1)),
        steps=int(job.get("steps", 8)),
        loras=loras,
        settings_override=settings_override,
    )

    return {
        "job_id": job_id,
        "status": result["status"],
        "message": f"Job retried. Status: {result['status']}",
    }


@app.get("/queue")
def queue_stats():
    """Get queue and GPU statistics."""
    stats = scheduler.get_queue_stats()
    gpu_status = scheduler.get_gpu_status()
    return {**stats, "gpu_status": gpu_status}


@app.get("/gpu_status")
def gpu_status():
    """Get detailed GPU container status."""
    status = scheduler.get_gpu_status()

    # Add container health info
    for gpu_id_str, info in status.items():
        container = info["container"]
        info["healthy"] = client.check_container_health(container)

    return {
        "gpus": status,
        "total": 3,
        "busy": sum(1 for g in status.values() if g["busy"]),
        "free": sum(1 for g in status.values() if not g["busy"]),
    }


@app.get("/jobs/list")
def list_jobs(limit: int = 50, status: Optional[str] = None):
    """List recent jobs."""
    jobs = scheduler.list_jobs(status=status, limit=limit)
    return {"jobs": jobs, "count": len(jobs)}


@app.delete("/jobs/{job_id}")
def delete_job(job_id: str):
    """Delete a job record."""
    deleted = scheduler.delete_job(job_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return {"message": f"Job {job_id} deleted"}


@app.get("/loras")
def list_loras():
    """List available LoRA models from GPU 0 container."""
    try:
        loras = client.get_available_loras()
        return {"loras": loras, "count": len(loras)}
    except Exception as e:
        return {"loras": [], "error": str(e)}


@app.get("/health")
def health_check():
    """Health check with GPU container status."""
    gpu_health = {}
    all_healthy = True

    for gpu_id in [0, 1, 2]:
        container = scheduler.gpu_config[gpu_id]["container"]
        healthy = client.check_container_health(container)
        gpu_health[f"gpu_{gpu_id}"] = {
            "container": container,
            "healthy": healthy,
        }
        if not healthy:
            all_healthy = False

    return {
        "status": "healthy" if all_healthy else "degraded",
        "api": "running",
        "gpu_containers": gpu_health,
        "total_gpus": 3,
        "healthy_gpus": sum(1 for g in gpu_health.values() if g["healthy"]),
        "queue_stats": scheduler.get_queue_stats(),
    }


# ── Dashboard ───────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def dashboard():
    """Serve dashboard HTML."""
    for html_file in ["dashboard.html", "admin.html"]:
        if os.path.exists(html_file):
            with open(html_file) as f:
                return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>Wan2GP Multi-GPU API</h1><p>Visit <a href='/docs'>/docs</a> for API documentation.</p>")


# ── Startup ─────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    """Log startup info."""
    logger.info("=" * 60)
    logger.info("Wan2GP Multi-GPU API v2.0 starting up")
    logger.info(f"GPU containers: 3")
    for gpu_id in [0, 1, 2]:
        cfg = scheduler.gpu_config[gpu_id]
        logger.info(f"  GPU {gpu_id}: {cfg['container']}")
    logger.info("=" * 60)
