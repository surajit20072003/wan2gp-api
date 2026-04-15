"""
Wan2GP Video Generation API — Multi-GPU Edition.
FastAPI server with built-in GPU scheduler for 3 GPU containers.
Replaces Celery with a thread-based scheduler for direct GPU management.
"""
from fastapi import FastAPI, Request, HTTPException, UploadFile, File, Header, Depends
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict
import os
import time
import uuid
import shutil
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

# Uploads staging dir (image/audio files uploaded before job submission)
UPLOADS_DIR = os.getenv("WAN2GP_UPLOADS_DIR", "/nvme0n1-disk/wan2gp_data/uploads")
os.makedirs(UPLOADS_DIR, exist_ok=True)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Client for direct Docker operations (non-scheduled)
client = Wan2GPClient()

# Security Setup
API_KEY = os.getenv("WAN2GP_API_KEY", "default-secret-key")

def verify_api_key(x_api_key: str = Header(None, alias="X-API-Key")):
    if not API_KEY or x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API Key")


# ── Request/Response Models ─────────────────────────────────────────

class GenerateRequest(BaseModel):
    prompt: str
    resolution: str = "1280x720"
    video_length: int = 361
    seed: int = -1
    steps: int = -1
    model: str = "ltx23_distilled_q6"  # model key from MODEL_TEMPLATES
    loras: Optional[Dict[str, float]] = None
    settings_override: Optional[Dict] = None
    webhook_url: Optional[str] = ""
    # Image-to-video / audio fields
    image_start_token: Optional[str] = ""   # file_token from POST /upload
    image_end_token: Optional[str] = ""     # file_token from POST /upload
    audio_token: Optional[str] = ""         # file_token from POST /upload
    image_prompt_type: Optional[str] = ""   # "S" or "SE" (auto-set if blank)
    audio_prompt_type: Optional[str] = ""   # "A" (auto-set if blank)


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
    model: Optional[str] = None
    output_file: Optional[str] = None
    error: Optional[str] = None
    gpu_id: Optional[str] = None
    created_at: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    duration_seconds: Optional[int] = None
    retry_count: int = 0


# ── API Endpoints ───────────────────────────────────────────────────

@app.post("/generate", response_model=JobResponse, dependencies=[Depends(verify_api_key)])
def create_job(request: Request, req: GenerateRequest):
    """Submit a new video generation job."""
    job_id = f"job_{int(time.time())}_{os.urandom(4).hex()}"
    client_ip = request.client.host if request.client else "unknown"

    logger.info(f"New job {job_id} from {client_ip}: model={req.model} prompt={req.prompt[:80]}...")

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
        model=req.model,
        loras=filtered_loras,
        settings_override=req.settings_override,
        webhook_url=req.webhook_url or "",
        client_ip=client_ip,
        image_start_token=req.image_start_token or "",
        image_end_token=req.image_end_token or "",
        audio_token=req.audio_token or "",
        image_prompt_type=req.image_prompt_type or "",
        audio_prompt_type=req.audio_prompt_type or "",
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


@app.get("/status/{job_id}", response_model=JobStatus, dependencies=[Depends(verify_api_key)])
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
        model=job.get("model"),
        output_file=job.get("output_file"),
        error=job.get("error"),
        gpu_id=job.get("gpu_id"),
        created_at=job.get("created_at"),
        started_at=job.get("started_at"),
        completed_at=job.get("completed_at"),
        duration_seconds=duration,
        retry_count=int(job.get("retry_count", 0)),
    )


@app.get("/download/{job_id}", dependencies=[Depends(verify_api_key)])
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
    logger.info(f"🔍 Download lookup: job={job_id} file={output_filename} path={output_path} exists={output_path.exists()}")

    if not output_path.exists():
        # Fallback: search all GPU output dirs
        found = False
        for gid in [0, 1, 2]:
            candidate = client.get_output_path(output_filename, gid)
            logger.info(f"🔍 Fallback GPU {gid}: {candidate} exists={candidate.exists()}")
            if candidate.exists():
                output_path = candidate
                found = True
                break
        if not found:
            logger.error(
                f"❌ Output file not found on any GPU dir: {output_filename}. "
                f"WAN2GP_OUTPUTS_DIR={os.getenv('WAN2GP_OUTPUTS_DIR', 'NOT SET')}"
            )
            raise HTTPException(
                status_code=404,
                detail=f"Output file not found: {output_filename}. Check server logs for path details.",
            )

    logger.info(f"✅ Serving download: {output_path}")
    return FileResponse(
        path=str(output_path),
        media_type="video/mp4",
        filename=output_filename,
    )


@app.post("/retry/{job_id}", dependencies=[Depends(verify_api_key)])
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


@app.get("/queue", dependencies=[Depends(verify_api_key)])
def queue_stats():
    """Get queue and GPU statistics."""
    stats = scheduler.get_queue_stats()
    gpu_status = scheduler.get_gpu_status()
    return {**stats, "gpu_status": gpu_status}


@app.get("/gpu_status", dependencies=[Depends(verify_api_key)])
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


@app.get("/jobs/list", dependencies=[Depends(verify_api_key)])
def list_jobs(limit: int = 50, status: Optional[str] = None):
    """List recent jobs."""
    jobs = scheduler.list_jobs(status=status, limit=limit)
    return {"jobs": jobs, "count": len(jobs)}


@app.delete("/jobs/{job_id}", dependencies=[Depends(verify_api_key)])
def delete_job(job_id: str):
    """Delete a job record."""
    deleted = scheduler.delete_job(job_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return {"message": f"Job {job_id} deleted"}


@app.post("/upload", dependencies=[Depends(verify_api_key)])
async def upload_media(file: UploadFile = File(...)):
    """
    Upload an image or audio file for use in image-to-video generation.
    Returns a file_token to pass as image_start_token / image_end_token / audio_token in /generate.

    Supported formats:
      Images: .jpg .jpeg .png .webp
      Audio:  .wav .mp3
    """
    ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".wav", ".mp3"}
    ext = os.path.splitext(file.filename or "")[-1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    # Generate unique token preserving original extension
    token = f"{uuid.uuid4().hex}{ext}"
    dest = os.path.join(UPLOADS_DIR, token)

    with open(dest, "wb") as out_file:
        shutil.copyfileobj(file.file, out_file)

    file_size = os.path.getsize(dest)
    logger.info(f"📤 Uploaded {file.filename} → {token} ({file_size} bytes)")

    return {
        "file_token": token,
        "original_filename": file.filename,
        "size_bytes": file_size,
        "type": "audio" if ext in {".wav", ".mp3"} else "image",
        "usage": {
            "image_start_token": token if ext not in {".wav", ".mp3"} else None,
            "audio_token": token if ext in {".wav", ".mp3"} else None,
        },
    }


@app.get("/loras", dependencies=[Depends(verify_api_key)])
def list_loras():
    """List available LoRA models from GPU 0 container."""
    try:
        loras = client.get_available_loras()
        return {"loras": loras, "count": len(loras)}
    except Exception as e:
        return {"loras": [], "error": str(e)}


@app.get("/models", dependencies=[Depends(verify_api_key)])
def list_models():
    """List all available model keys with labels and default settings."""
    from wan2gp_client import MODEL_TEMPLATES
    models = [
        {
            "key": key,
            "label": cfg["label"],
            "base_model_type": cfg["base_model_type"],
            "default_steps": cfg.get("default_steps", 30),
        }
        for key, cfg in MODEL_TEMPLATES.items()
    ]
    return {"models": models, "count": len(models)}


@app.get("/health", dependencies=[Depends(verify_api_key)])
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
