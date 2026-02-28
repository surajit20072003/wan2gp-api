"""
Wan2GP Client for multi-container Docker execution.
Supports 3 GPU containers (wan2gp-gpu0, wan2gp-gpu1, wan2gp-gpu2).
Each container has its own settings and outputs directories.
"""
import subprocess
import shutil
import json
import os
import tempfile
import glob
from pathlib import Path
from typing import Dict, Optional
import logging

# Staging dir where uploaded files are stored before being copied to GPU settings dir
UPLOADS_STAGING_DIR = os.getenv("WAN2GP_UPLOADS_DIR", "/nvme0n1-disk/wan2gp_data/uploads")

logger = logging.getLogger(__name__)


class Wan2GPClient:
    """Client for submitting jobs to wan2gp Docker containers."""

    # Default directories (overridden per-GPU by scheduler)
    DEFAULT_SETTINGS_DIR = "/nvme0n1-disk/wan2gp_data/gpu0/settings"
    DEFAULT_OUTPUTS_DIR = "/nvme0n1-disk/wan2gp_data/gpu0/outputs"

    def __init__(self, template_path: Optional[str] = None):
        """Initialize client with settings template."""
        if template_path and Path(template_path).exists():
            with open(template_path) as f:
                self.template = json.load(f)
        else:
            self.template = self._get_default_template()

    def _get_default_template(self) -> Dict:
        """Default settings for LTX-2 Distilled 19B."""
        return {
            "type": "WanGP v10.70 by DeepBeepMeep - LTX-2 Distilled 19B",
            "settings_version": 2.49,
            "base_model_type": "ltx2_19B",
            "model_type": "ltx2_distilled",
            "model_filename": "https://huggingface.co/DeepBeepMeep/LTX-2/resolve/main/ltx-2-19b-distilled-fp8_diffusion_model.safetensors",
            "image_mode": 0,
            "prompt": "",
            "alt_prompt": "",
            "resolution": "1280x720",
            "video_length": 361,
            "batch_size": 1,
            "seed": 42,
            "num_inference_steps": 8,
            "audio_scale": 1,
            "repeat_generation": 1,
            "multi_prompts_gen_type": 2,
            "multi_images_gen_type": 0,
            "loras_multipliers": "",
            "image_prompt_type": "",
            "video_prompt_type": "",
            "keep_frames_video_guide": "",
            "mask_expand": 0,
            "audio_prompt_type": "",
            "sliding_window_size": 481,
            "sliding_window_overlap": 17,
            "sliding_window_color_correction_strength": 0,
            "sliding_window_overlap_noise": 0,
            "sliding_window_discard_last_frames": 0,
            "temporal_upsampling": "",
            "spatial_upsampling": "",
            "film_grain_intensity": 0,
            "film_grain_saturation": 0.5,
            "RIFLEx_setting": 0,
            "prompt_enhancer": "T",
            "override_profile": -1,
            "override_attention": "",
            "self_refiner_setting": 0,
            "self_refiner_plan": "",
            "self_refiner_f_uncertainty": 0.1,
            "self_refiner_certain_percentage": 0.999,
            "output_filename": "",
            "mode": "",
            "activated_loras": []
        }

    def _docker_exec(self, cmd: str, container_name: str = "wan2gp") -> tuple[str, str]:
        """Execute command in a specific wan2gp container.
        Uses Popen with tempfile-based output to avoid pipe buffer deadlocks
        that occur with subprocess.run(capture_output=True) on long-running commands.
        """
        full_cmd = f"docker exec {container_name} {cmd}"
        logger.info(f"Executing: {full_cmd}")

        # Use tempfiles to avoid pipe buffer deadlock with large output
        with tempfile.NamedTemporaryFile(mode='w+', suffix='_stdout.log', delete=True) as stdout_f, \
             tempfile.NamedTemporaryFile(mode='w+', suffix='_stderr.log', delete=True) as stderr_f:

            process = subprocess.Popen(
                full_cmd,
                shell=True,
                stdout=stdout_f,
                stderr=stderr_f,
                text=True,
            )

            try:
                exit_code = process.wait(timeout=1800)  # 30 min timeout
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                raise

            # Read output from tempfiles
            stdout_f.seek(0)
            stderr_f.seek(0)
            stdout = stdout_f.read()
            stderr = stderr_f.read()

            logger.info(f"Command exited with code {exit_code}")
            if exit_code != 0:
                logger.warning(f"Non-zero exit code {exit_code}. stderr: {stderr[-500:]}")

            return stdout, stderr

    def _copy_media_to_settings(self, token: str, settings_dir: str) -> Optional[str]:
        """
        Copy a media file from uploads staging dir into the GPU's settings dir.
        Returns the filename (not full path) so it can be referenced in settings JSON.
        Returns None if token is empty or file not found.
        """
        if not token:
            return None
        src = Path(UPLOADS_STAGING_DIR) / token
        if not src.exists():
            logger.warning(f"⚠️ Media file not found in uploads: {src}")
            return None
        dst = Path(settings_dir) / token
        shutil.copy2(str(src), str(dst))
        logger.info(f"📁 Copied media file {token} → {dst}")
        return token

    def submit_job(
        self,
        job_id: str,
        prompt: str,
        container_name: str = "wan2gp-gpu0",
        settings_dir: str = None,
        outputs_dir: str = None,
        resolution: str = "1280x720",
        video_length: int = 361,
        seed: int = -1,
        steps: int = 8,
        loras: Optional[Dict[str, float]] = None,
        output_filename: str = "",
        settings_override: Optional[Dict] = None,
        image_start_token: str = "",
        image_end_token: str = "",
        audio_token: str = "",
        image_prompt_type: str = "",
        audio_prompt_type: str = "",
    ) -> Dict:
        """
        Submit a video generation job to a specific container.

        Args:
            job_id: Unique job identifier
            prompt: Video description
            container_name: Docker container to exec into (wan2gp-gpu0/1/2)
            settings_dir: Host path for settings files
            outputs_dir: Host path for output files
            resolution: Video resolution (e.g., "1280x720")
            video_length: Number of frames (81≈5s, 361≈15s)
            seed: Random seed (-1 for random)
            steps: Denoising steps (8 for distilled)
            loras: Dict of {lora_filename: multiplier}
            output_filename: Custom output name
            settings_override: Full settings dictionary to use as base

        Returns:
            {"status": "success"|"error", "output_file": str, "stdout": str, "stderr": str}
        """
        settings_dir = settings_dir or self.DEFAULT_SETTINGS_DIR
        outputs_dir = outputs_dir or self.DEFAULT_OUTPUTS_DIR

        # Build settings
        if settings_override:
            settings = settings_override.copy()
        else:
            settings = self.template.copy()

        settings.update({
            "prompt": prompt,
            "resolution": resolution,
            "video_length": video_length,
            "seed": seed if seed > 0 else -1,
            "num_inference_steps": steps,
            "output_filename": output_filename or job_id,
        })

        if loras:
            if isinstance(loras, dict):
                settings["activated_loras"] = list(loras.keys())
                settings["loras_multipliers"] = " ".join(str(v) for v in loras.values())
            elif isinstance(loras, list):
                settings["activated_loras"] = loras

        # Ensure settings/outputs directories exist
        os.makedirs(settings_dir, exist_ok=True)
        os.makedirs(outputs_dir, exist_ok=True)
        os.makedirs(UPLOADS_STAGING_DIR, exist_ok=True)

        # ── Image / Audio Attachments ──────────────────────────────────
        # Copy media files from staging into the GPU's settings dir so the
        # container can see them at /workspace/settings/<filename>

        start_file = self._copy_media_to_settings(image_start_token, settings_dir)
        end_file   = self._copy_media_to_settings(image_end_token, settings_dir)
        audio_file = self._copy_media_to_settings(audio_token, settings_dir)

        if start_file:
            settings["image_start"] = f"/workspace/settings/{start_file}"
            # Auto-set image_prompt_type: S (start only) or SE (start+end)
            settings["image_prompt_type"] = image_prompt_type if image_prompt_type else ("SE" if end_file else "S")
            logger.info(f"🖼️  image_start={settings['image_start']} image_prompt_type={settings['image_prompt_type']}")

        if end_file:
            settings["image_end"] = f"/workspace/settings/{end_file}"
            logger.info(f"🖼️  image_end={settings['image_end']}")

        if audio_file:
            settings["audio_guide"] = f"/workspace/settings/{audio_file}"
            settings["audio_prompt_type"] = audio_prompt_type if audio_prompt_type else "A"
            logger.info(f"🎵 audio_guide={settings['audio_guide']} audio_prompt_type={settings['audio_prompt_type']}")

        # Write settings file
        settings_filename = f"api_job_{job_id}.json"
        settings_path = Path(settings_dir) / settings_filename

        with open(settings_path, 'w') as f:
            json.dump(settings, indent=2, fp=f)

        logger.info(f"Settings written to: {settings_path}")

        # Execute headless generation in the specific container
        # Inside the container, settings are at /workspace/settings/
        docker_settings_path = f"/workspace/settings/{settings_filename}"
        cmd = f"python3 wgp.py --process {docker_settings_path} --output-dir /workspace/outputs"

        # Snapshot existing output files to detect new ones
        existing_outputs = set()
        try:
            existing_outputs = set(os.listdir(outputs_dir))
        except OSError:
            pass

        try:
            stdout, stderr = self._docker_exec(cmd, container_name)

            # Parse output filename from logs
            output_file = self._extract_output_filename(stdout, stderr)

            # If not found in logs, detect new files in outputs directory
            if not output_file:
                try:
                    new_outputs = set(os.listdir(outputs_dir)) - existing_outputs
                    mp4_files = [f for f in new_outputs if f.endswith('.mp4')]
                    if mp4_files:
                        # Pick the most recently modified file
                        mp4_files.sort(key=lambda f: os.path.getmtime(os.path.join(outputs_dir, f)), reverse=True)
                        output_file = mp4_files[0]
                        logger.info(f"Detected new output file: {output_file}")
                except OSError:
                    pass

            # If output file exists → success (even if stderr has warnings)
            # Wan2GP often writes non-fatal warnings to stderr containing "error"
            if output_file:
                return {
                    "status": "success",
                    "output_file": output_file,
                    "stdout": stdout[-500:],
                    "stderr": stderr[-500:] if stderr else "",
                }

            # No output file produced → check stderr for real errors
            if "error" in stderr.lower() or "traceback" in stderr.lower():
                return {
                    "status": "error",
                    "output_file": None,
                    "stdout": stdout[-1000:],
                    "stderr": stderr[-1000:],
                }

            # No output file and no clear error
            return {
                "status": "error",
                "output_file": None,
                "stdout": stdout[-500:],
                "stderr": stderr[-500:] if stderr else "No output file generated",
            }

        except subprocess.TimeoutExpired:
            return {
                "status": "error",
                "output_file": None,
                "stdout": "",
                "stderr": "Generation timeout (30 min exceeded)",
            }
        except Exception as e:
            return {
                "status": "error",
                "output_file": None,
                "stdout": "",
                "stderr": str(e),
            }

    def _extract_output_filename(self, stdout: str, stderr: str) -> Optional[str]:
        """Extract output filename from generation logs."""
        combined = stdout + stderr

        for line in combined.split('\n'):
            if 'saved video:' in line.lower() or 'output:' in line.lower():
                if '.mp4' in line:
                    parts = line.split('/')
                    for part in reversed(parts):
                        if '.mp4' in part:
                            return part.strip()

        for line in combined.split('\n'):
            if '.mp4' in line:
                words = line.split()
                for word in words:
                    if '.mp4' in word:
                        return word.strip().rstrip(',').rstrip('.')

        return None

    def list_outputs(self, container_name: str = "wan2gp-gpu0", limit: int = 20) -> list[str]:
        """List recent output files from a specific container."""
        try:
            stdout, _ = self._docker_exec(
                f"ls -t /workspace/outputs/*.mp4 | head -{limit}",
                container_name,
            )
            files = [f.split('/')[-1] for f in stdout.strip().split('\n') if f]
            return files
        except:
            return []

    def get_output_path(self, filename: str, gpu_id: int = 0) -> Path:
        """Get full host path to output file for a specific GPU."""
        outputs_dir = os.getenv(
            f"WAN2GP_OUTPUTS_DIR_{gpu_id}",
            f"/nvme0n1-disk/wan2gp_data/gpu{gpu_id}/outputs"
        )
        return Path(outputs_dir) / filename

    def check_container_health(self, container_name: str) -> bool:
        """Check if a wan2gp container is running and responsive."""
        try:
            result = subprocess.run(
                f"docker inspect --format='{{{{.State.Running}}}}' {container_name}",
                shell=True,
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.stdout.strip() == "true"
        except Exception:
            return False

    def get_available_loras(self, container_name: str = "wan2gp-gpu0") -> list[str]:
        """List available LoRA files in the container."""
        try:
            stdout, stderr = self._docker_exec(
                "ls /workspace/models/loras/ 2>/dev/null || echo 'No loras directory'",
                container_name,
            )
            # Filter for .safetensors files
            loras = [f.strip() for f in stdout.strip().split('\\n') if f.strip().endswith('.safetensors')]
            return loras
        except Exception as e:
            logger.error(f"Failed to list LoRAs in {container_name}: {e}")
            return []
