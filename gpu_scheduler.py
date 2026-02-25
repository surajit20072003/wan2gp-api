"""
Multi-GPU Scheduler for Wan2GP Video Generation.
Manages 3 GPU containers with round-robin queue management.
Modeled after HeyGem's chatterbox_scheduler.py pattern.
"""
import threading
import time
import json
import os
import logging
import redis
from queue import Queue
from typing import Dict, Optional, Any
from datetime import datetime

logger = logging.getLogger(__name__)

# Redis config
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "2"))


class GPUScheduler:
    """
    Thread-safe GPU scheduler for 3 wan2gp containers.
    
    Architecture:
      - 3 GPU containers (wan2gp-gpu0, wan2gp-gpu1, wan2gp-gpu2)
      - Each gets exclusive access to one NVIDIA GPU
      - Jobs are queued and dispatched to first available GPU
      - Background monitoring threads track job completion
    """

    def __init__(self, redis_client=None):
        # GPU configuration — mirrors HeyGem's pattern
        self.gpu_config = {
            0: {
                "container": os.getenv("WAN2GP_CONTAINER_0", "wan2gp-gpu0"),
                "outputs_dir": os.getenv("WAN2GP_OUTPUTS_DIR_0", "/nvme0n1-disk/wan2gp_data/gpu0/outputs"),
                "settings_dir": os.getenv("WAN2GP_SETTINGS_DIR_0", "/nvme0n1-disk/wan2gp_data/gpu0/settings"),
                "busy": False,
                "current_job": None,
            },
            1: {
                "container": os.getenv("WAN2GP_CONTAINER_1", "wan2gp-gpu1"),
                "outputs_dir": os.getenv("WAN2GP_OUTPUTS_DIR_1", "/nvme0n1-disk/wan2gp_data/gpu1/outputs"),
                "settings_dir": os.getenv("WAN2GP_SETTINGS_DIR_1", "/nvme0n1-disk/wan2gp_data/gpu1/settings"),
                "busy": False,
                "current_job": None,
            },
            2: {
                "container": os.getenv("WAN2GP_CONTAINER_2", "wan2gp-gpu2"),
                "outputs_dir": os.getenv("WAN2GP_OUTPUTS_DIR_2", "/nvme0n1-disk/wan2gp_data/gpu2/outputs"),
                "settings_dir": os.getenv("WAN2GP_SETTINGS_DIR_2", "/nvme0n1-disk/wan2gp_data/gpu2/settings"),
                "busy": False,
                "current_job": None,
            },
        }

        # Job queue (thread-safe)
        self.job_queue = Queue()

        # Threading lock for GPU state
        self.lock = threading.Lock()

        # Redis for job metadata persistence
        self.r = redis_client or redis.Redis(
            host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True
        )

        # Wan2GP client (imported lazily to avoid circular deps)
        self._client = None

        logger.info("🚀 GPU Scheduler initialized with 3 GPUs")
        for gpu_id, cfg in self.gpu_config.items():
            # Reset GPU status on startup (in case of crash/restart)
            cfg["busy"] = False
            cfg["current_job"] = None
            logger.info(f"   GPU {gpu_id}: container={cfg['container']}, outputs={cfg['outputs_dir']} (State: Free)")

        # Sync with Redis to restore queue state
        self._sync_with_redis()

        # Start background queue monitor
        self._monitor_thread = threading.Thread(target=self._monitor_queue, daemon=True)
        self._monitor_thread.start()

    @property
    def client(self):
        """Lazy-load Wan2GPClient to avoid circular imports."""
        if self._client is None:
            from wan2gp_client import Wan2GPClient
            self._client = Wan2GPClient()
        return self._client

    # ── GPU Management ──────────────────────────────────────────────

    def reserve_gpu(self, job_id: str) -> Optional[int]:
        """
        Atomically reserve the first available GPU.
        Returns GPU ID (0, 1, or 2) or None if all busy.
        """
        with self.lock:
            for gpu_id in [0, 1, 2]:
                if not self.gpu_config[gpu_id]["busy"]:
                    self.gpu_config[gpu_id]["busy"] = True
                    self.gpu_config[gpu_id]["current_job"] = job_id
                    logger.info(f"🔒 [GPU {gpu_id}] Reserved for job {job_id}")
                    return gpu_id
        logger.info(f"⏸️  All GPUs busy — job {job_id} will queue")
        return None

    def release_gpu(self, gpu_id: int, job_id: str):
        """Release a GPU and trigger next queued job."""
        with self.lock:
            if self.gpu_config[gpu_id]["current_job"] == job_id:
                self.gpu_config[gpu_id]["busy"] = False
                self.gpu_config[gpu_id]["current_job"] = None
                logger.info(f"🔓 [GPU {gpu_id}] Released from job {job_id}")
            else:
                logger.warning(
                    f"⚠️ [GPU {gpu_id}] Release mismatch: expected {job_id}, "
                    f"got {self.gpu_config[gpu_id]['current_job']}"
                )

        # Process next job in queue
        self._process_next()

    def get_gpu_status(self) -> Dict:
        """Get status of all 3 GPUs."""
        with self.lock:
            return {
                str(gpu_id): {
                    "busy": cfg["busy"],
                    "current_job": cfg["current_job"],
                    "container": cfg["container"],
                }
                for gpu_id, cfg in self.gpu_config.items()
            }

    def _monitor_queue(self):
        """Background thread to ensure queue processing."""
        while True:
            try:
                time.sleep(5)  # Check every 5 seconds
                if not self.job_queue.empty():
                    # Only process if we have free GPUs
                    free_gpus = [gid for gid, cfg in self.gpu_config.items() if not cfg["busy"]]
                    if free_gpus:
                        # logger.info(f"🔄 Queue monitor: {self.job_queue.qsize()} pending jobs, {len(free_gpus)} free GPUs")
                        self._process_next()
            except Exception as e:
                logger.error(f"❌ Queue monitor error: {e}")

    def _sync_with_redis(self):
        """
        Sync in-memory queue with Redis state on startup.
        - Re-queues jobs that are in 'queued' state.
        - Marks 'running' jobs as 'failed' (interrupted by restart).
        """
        logger.info("🔄 Syncing with Redis state...")
        all_jobs = self.list_jobs(limit=1000)
        
        # 1. Handle interrupted running jobs
        running_jobs = [j for j in all_jobs if j["status"] == "running" or j["status"] == "processing"]
        for job in running_jobs:
            logger.warning(f"⚠️ Marking interrupted job {job['job_id']} as failed")
            self._update_status(
                job["job_id"], "failed", 
                error="System restarted during execution",
                failed_at=str(time.time())
            )

        # 2. Restore queued jobs
        # list_jobs returns sorted by created_at desc, so we reverse to process oldest first
        queued_jobs = [j for j in all_jobs if j["status"] == "queued"]
        queued_jobs.reverse() 
        
        for job in queued_jobs:
            job_id = job["job_id"]
            # Fetch full job data to re-queue
            data = self.get_job(job_id)
            if not data:
                continue
            
            # Reconstruct job_data needed for execution
            try:
                loras_val = data.get("loras")
                loras = json.loads(loras_val) if loras_val and loras_val.strip() else {}
            except Exception:
                loras = {}

            try:
                settings_val = data.get("settings_override")
                settings_override = json.loads(settings_val) if settings_val and settings_val.strip() else None
            except Exception:
                settings_override = None

            job_data = {
                "job_id": job_id,
                "prompt": data.get("prompt"),
                "resolution": data.get("resolution", "1280x720"),
                "video_length": int(data.get("video_length", 81)),
                "seed": int(data.get("seed", -1)),
                "steps": int(data.get("steps", 8)),
                "loras": loras,
                "settings_override": settings_override,
                "webhook_url": data.get("webhook_url", ""),
            }
            
            self.job_queue.put(job_data)
            logger.info(f"📥 Restored job {job_id} to queue")
            
        logger.info(f"✅ Sync complete. Restored {len(queued_jobs)} jobs to queue.")

    # ── Job Queue ───────────────────────────────────────────────────

    def submit_job(self, job_id: str, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Submit a video generation job.
        If a GPU is free, dispatches immediately.
        Otherwise queues for later processing.
        
        Returns: {"status": "processing"|"queued", "gpu_id": int|None, "queue_position": int}
        """
        job_data = {
            "job_id": job_id,
            "prompt": prompt,
            "resolution": kwargs.get("resolution", "1280x720"),
            "video_length": kwargs.get("video_length", 81),
            "seed": kwargs.get("seed", -1),
            "steps": kwargs.get("steps", 8),
            "loras": kwargs.get("loras", {}),
            "settings_override": kwargs.get("settings_override"),
            "webhook_url": kwargs.get("webhook_url", ""),
            "submitted_at": time.time(),
        }

        # Store in Redis
        self._save_job_metadata(job_id, {
            "prompt": prompt,
            "resolution": job_data["resolution"],
            "video_length": str(job_data["video_length"]),
            "seed": str(job_data["seed"]),
            "steps": str(job_data["steps"]),
            "loras": json.dumps(job_data["loras"]),
            "settings_override": json.dumps(job_data["settings_override"]) if job_data["settings_override"] else "",
            "webhook_url": job_data["webhook_url"],
            "status": "queued",
            "created_at": str(time.time()),
            "client_ip": kwargs.get("client_ip", ""),
            "retry_count": "0",
        })

        # Try to dispatch immediately
        gpu_id = self.reserve_gpu(job_id)

        if gpu_id is not None:
            # Dispatch to GPU in background thread
            self._update_status(job_id, "running", gpu_id=gpu_id)
            thread = threading.Thread(
                target=self._execute_job,
                args=(job_id, job_data, gpu_id),
                daemon=True,
            )
            thread.start()
            return {"status": "processing", "gpu_id": gpu_id, "queue_position": 0}
        else:
            # Queue for later
            self.job_queue.put(job_data)
            queue_pos = self.job_queue.qsize()
            logger.info(f"📥 Job {job_id} queued at position {queue_pos}")
            return {"status": "queued", "gpu_id": None, "queue_position": queue_pos}

    def _process_next(self):
        """Process pending jobs until no free GPUs or queue empty."""
        while not self.job_queue.empty():
            # Peek at queue — don't pop yet until we confirm GPU
            # We need to pop + reserve atomically
            with self.lock:
                # Check for available GPU
                available_gpu = None
                for gpu_id in [0, 1, 2]:
                    if not self.gpu_config[gpu_id]["busy"]:
                        available_gpu = gpu_id
                        break

                if available_gpu is None:
                    # No more GPUs, stop processing
                    return

                # Pop from queue
                if self.job_queue.empty():
                    return
                job_data = self.job_queue.get()
                job_id = job_data["job_id"]

                # Reserve GPU
                self.gpu_config[available_gpu]["busy"] = True
                self.gpu_config[available_gpu]["current_job"] = job_id

            logger.info(f"🎬 Dequeued job {job_id} → GPU {available_gpu}")
            self._update_status(job_id, "running", gpu_id=available_gpu)

            # Dispatch in background
            thread = threading.Thread(
                target=self._execute_job,
                args=(job_id, job_data, available_gpu),
                daemon=True,
            )
            thread.start()

    # ── Job Execution ────────────────────────────────────────────────

    def _execute_job(self, job_id: str, job_data: Dict, gpu_id: int):
        """
        Execute a video generation job on the specified GPU.
        Runs in a background thread.
        """
        container = self.gpu_config[gpu_id]["container"]
        settings_dir = self.gpu_config[gpu_id]["settings_dir"]
        outputs_dir = self.gpu_config[gpu_id]["outputs_dir"]

        logger.info(f"🎬 [GPU {gpu_id}] Starting job {job_id} on {container}")
        self._update_status(job_id, "running", gpu_id=gpu_id, started_at=str(time.time()))

        try:
            result = self.client.submit_job(
                job_id=job_id,
                prompt=job_data["prompt"],
                container_name=container,
                settings_dir=settings_dir,
                outputs_dir=outputs_dir,
                resolution=job_data.get("resolution", "1280x720"),
                video_length=job_data.get("video_length", 81),
                seed=job_data.get("seed", -1),
                steps=job_data.get("steps", 8),
                loras=job_data.get("loras", {}),
                settings_override=job_data.get("settings_override"),
            )

            if result["status"] == "success":
                logger.info(f"✅ [GPU {gpu_id}] Job {job_id} completed!")
                started_at = float(self.r.hget(f"job:{job_id}", "started_at") or time.time())
                duration = int(time.time() - started_at)

                self._update_status(
                    job_id, "completed",
                    gpu_id=gpu_id,
                    output_file=result.get("output_file", ""),
                    completed_at=str(time.time()),
                    duration_seconds=str(duration),
                    stdout=result.get("stdout", "")[:500],
                    stderr=result.get("stderr", "")[:500],
                )
            else:
                raise Exception(f"Generation failed: {result.get('stderr', 'Unknown error')}")

        except Exception as e:
            error_msg = str(e)
            logger.error(f"❌ [GPU {gpu_id}] Job {job_id} failed: {error_msg}")

            # Check retry count
            retry_count = int(self.r.hget(f"job:{job_id}", "retry_count") or "0")
            max_retries = 2

            if retry_count < max_retries:
                logger.info(f"🔄 [GPU {gpu_id}] Retrying job {job_id} (attempt {retry_count + 1}/{max_retries})")
                self._update_status(
                    job_id, "retrying",
                    error=error_msg,
                    retry_count=str(retry_count + 1),
                )
                # Release GPU first, then re-queue
                self.release_gpu(gpu_id, job_id)
                job_data["retry_count"] = retry_count + 1
                time.sleep(10)  # Brief delay before retry
                self.submit_job(
                    job_id, job_data["prompt"],
                    resolution=job_data.get("resolution", "1280x720"),
                    video_length=job_data.get("video_length", 81),
                    seed=job_data.get("seed", -1),
                    steps=job_data.get("steps", 8),
                    loras=job_data.get("loras", {}),
                    settings_override=job_data.get("settings_override"),
                )
                return  # Don't release GPU again below
            else:
                self._update_status(
                    job_id, "failed",
                    error=error_msg,
                    failed_at=str(time.time()),
                    retry_count=str(retry_count),
                )
        finally:
            # Release GPU (unless already released in retry path)
            # NOTE: Check ownership inside lock, but call release_gpu OUTSIDE
            # lock to avoid deadlock (release_gpu also acquires self.lock)
            should_release = False
            with self.lock:
                if self.gpu_config[gpu_id]["current_job"] == job_id:
                    should_release = True
            if should_release:
                self.release_gpu(gpu_id, job_id)

    # ── Redis Helpers ────────────────────────────────────────────────

    def _save_job_metadata(self, job_id: str, data: Dict):
        """Save job metadata to Redis hash."""
        self.r.hset(f"job:{job_id}", mapping=data)

    def _update_status(self, job_id: str, status: str, **extra):
        """Update job status and any extra fields in Redis."""
        updates = {"status": status}
        if extra.get("gpu_id") is not None:
            updates["gpu_id"] = str(extra.pop("gpu_id"))
        updates.update({k: str(v) for k, v in extra.items() if v is not None})
        self.r.hset(f"job:{job_id}", mapping=updates)

    def get_job(self, job_id: str) -> Optional[Dict]:
        """Get job data from Redis."""
        data = self.r.hgetall(f"job:{job_id}")
        return data if data else None

    def list_jobs(self, status: str = None, limit: int = 500) -> list:
        """List jobs, optionally filtered by status."""
        all_keys = self.r.keys("job:*")
        jobs = []
        # Fetch ALL jobs first (no early cutoff)
        for key in all_keys:
            job_data = self.r.hgetall(key)
            if status and job_data.get("status") != status:
                continue
            jid = key.replace("job:", "")
            jobs.append({
                "job_id": jid,
                "status": job_data.get("status", "unknown"),
                "prompt": job_data.get("prompt", "")[:100],
                "created_at": job_data.get("created_at"),
                "gpu_id": job_data.get("gpu_id"),
                "client_ip": job_data.get("client_ip"),
            })
        # Sort newest first, THEN apply limit
        jobs.sort(key=lambda x: float(x.get("created_at") or 0), reverse=True)
        return jobs[:limit]

    def delete_job(self, job_id: str) -> bool:
        """Delete a job from Redis."""
        return self.r.delete(f"job:{job_id}") > 0

    def get_queue_stats(self) -> Dict:
        """Get queue statistics."""
        all_keys = self.r.keys("job:*")
        statuses = {"queued": 0, "running": 0, "completed": 0, "failed": 0, "error": 0, "retrying": 0}

        for key in all_keys:
            s = self.r.hget(key, "status")
            if s in statuses:
                statuses[s] += 1

        gpu_status = self.get_gpu_status()
        active_gpus = sum(1 for g in gpu_status.values() if g["busy"])

        return {
            "total_jobs": len(all_keys),
            "queue_length": statuses["queued"] + self.job_queue.qsize(),
            "running": statuses["running"],
            "completed": statuses["completed"],
            "errors": statuses["error"] + statuses["failed"],
            "retrying": statuses["retrying"],
            "active_gpus": active_gpus,
            "total_gpus": 3,
            "in_memory_queue": self.job_queue.qsize(),
        }


# Singleton instance
scheduler = GPUScheduler()
