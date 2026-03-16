"""
Wan2GP Client for multi-container Docker execution.
Supports 3 GPU containers (wan2gp-gpu0, wan2gp-gpu1, wan2gp-gpu2).
Each container has its own settings and outputs directories.
"""
import subprocess
import shutil
import json
import os
import time
import tempfile
import glob
from pathlib import Path
from typing import Dict, Optional
import logging

# Staging dir where uploaded files are stored before being copied to GPU settings dir
UPLOADS_STAGING_DIR = os.getenv("WAN2GP_UPLOADS_DIR", "/nvme0n1-disk/wan2gp_data/uploads")

logger = logging.getLogger(__name__)

# ── Model Template Registry ────────────────────────────────────────────
# Maps user-facing model key → WAN2GP settings fields.
# model_filename may be a HuggingFace URL (auto-downloaded on first use)
# or a local path inside the container under /workspace/ckpts/.
MODEL_TEMPLATES = {
    # ── LTX Video (current default, fastest) ──────────────────────────
    "ltx2_distilled": {
        "label": "LTX-2 19B Distilled (8 steps, fastest)",
        "base_model_type": "ltx2_19B",
        "model_type": "ltx2_distilled",
        "model_filename": "https://huggingface.co/DeepBeepMeep/LTX-2/resolve/main/ltx-2-19b-distilled-fp8_diffusion_model.safetensors",
        "default_steps": 8,
    },
    # ── WAN 2.1 ──────────────────────────────────────────────────────
    "wan21_t2v": {
        "label": "WAN 2.1 Text-to-Video 14B",
        "base_model_type": "t2v",
        "model_type": "t2v",
        "model_filename": "https://huggingface.co/DeepBeepMeep/Wan2.1/resolve/main/text2video_14B_bf16.safetensors",
        "default_steps": 30,
    },
    "wan21_i2v_480p": {
        "label": "WAN 2.1 Image-to-Video 480p",
        "base_model_type": "i2v",
        "model_type": "i2v",
        "model_filename": "https://huggingface.co/DeepBeepMeep/Wan2.1/resolve/main/image2video_480p_14B_bf16.safetensors",
        "default_steps": 30,
        "image_prompt_type": "S",
    },
    # ── WAN 2.2 ──────────────────────────────────────────────────────
    "wan22_t2v": {
        "label": "WAN 2.2 Text-to-Video 14B",
        "base_model_type": "t2v_2_2",
        "model_type": "t2v_2_2",
        "model_filename": "https://huggingface.co/DeepBeepMeep/Wan2.2/resolve/main/text2video_2_2_14B_bf16.safetensors",
        "default_steps": 30,
    },
    "wan22_i2v": {
        "label": "WAN 2.2 Image-to-Video 480p 14B",
        "base_model_type": "i2v_2_2",
        "model_type": "i2v_2_2",
        "model_filename": "https://huggingface.co/DeepBeepMeep/Wan2.2/resolve/main/image2video_2_2_480p_14B_bf16.safetensors",
        "default_steps": 30,
        "image_prompt_type": "S",
    },
    "wan22_t2v_5b": {
        "label": "WAN 2.2 T+I to Video 5B",
        "base_model_type": "ti2v_2_2",
        "model_type": "ti2v_2_2",
        "model_filename": "https://huggingface.co/DeepBeepMeep/Wan2.2/resolve/main/text2video_2_2_5B_bf16.safetensors",
        "default_steps": 30,
    },
    "wan22_vace": {
        "label": "WAN 2.2 VACE 14B (ControlNet)",
        "base_model_type": "vace_14B_2_2",
        "model_type": "vace_14B_2_2",
        "model_filename": "https://huggingface.co/DeepBeepMeep/Wan2.2/resolve/main/vace_2_2_14B_bf16.safetensors",
        "default_steps": 30,
    },
    # ── HunyuanVideo ──────────────────────────────────────────────────
    "hunyuan_t2v": {
        "label": "HunyuanVideo T2V 720p",
        "base_model_type": "hunyuan",
        "model_type": "hunyuan",
        "model_filename": "https://huggingface.co/DeepBeepMeep/HunyuanVideo/resolve/main/hunyuan_video_720_bf16.safetensors",
        "default_steps": 50,
        "guidance_scale": 7.0,
        "embedded_guidance_scale": 6.0,
    },
    "hunyuan_i2v": {
        "label": "HunyuanVideo I2V 720p",
        "base_model_type": "hunyuan_i2v",
        "model_type": "hunyuan_i2v",
        "model_filename": "https://huggingface.co/DeepBeepMeep/HunyuanVideo/resolve/main/hunyuan_video_i2v_720_bf16.safetensors",
        "default_steps": 50,
        "guidance_scale": 7.0,
        "embedded_guidance_scale": 6.0,
        "image_prompt_type": "S",
    },
    "hunyuan_1_5_t2v": {
        "label": "HunyuanVideo 1.5 T2V 720p",
        "base_model_type": "hunyuan_1_5_t2v",
        "model_type": "hunyuan_1_5_t2v",
        "model_filename": "https://huggingface.co/DeepBeepMeep/HunyuanVideo1.5/resolve/main/hunyuan_video_1_5_720_bf16.safetensors",
        "default_steps": 20,
    },
    "hunyuan_1_5_i2v": {
        "label": "HunyuanVideo 1.5 I2V 720p",
        "base_model_type": "hunyuan_1_5_i2v",
        "model_type": "hunyuan_1_5_i2v",
        "model_filename": "https://huggingface.co/DeepBeepMeep/HunyuanVideo1.5/resolve/main/hunyuan_video_1_5_i2v_720_bf16.safetensors",
        "default_steps": 20,
        "image_prompt_type": "S",
    },
}


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
        """Default base settings structure (model-agnostic fields)."""
        return {
            "settings_version": 2.52,
            "image_mode": 0,
            "prompt": "",
            "alt_prompt": "",
            "resolution": "1280x720",
            "video_length": 361,
            "batch_size": 1,
            "seed": 42,
            "num_inference_steps": 8,
            "guidance_scale": 5.0,
            "embedded_guidance_scale": 6.0,
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

    def _get_template(self, model_key: str) -> Dict:
        """
        Return full settings dict for a given model key.
        Starts from the base template and overlays model-specific fields.
        Falls back to ltx2_distilled if key is unknown.
        """
        base = self._get_default_template()

        model_cfg = MODEL_TEMPLATES.get(model_key)
        if model_cfg is None:
            logger.warning(f"⚠️ Unknown model key '{model_key}', falling back to ltx2_distilled")
            model_cfg = MODEL_TEMPLATES["ltx2_distilled"]

        # Apply model-specific overrides
        base.update({
            "type": f"WanGP v10.952 by DeepBeepMeep - {model_cfg['label']}",
            "base_model_type": model_cfg["base_model_type"],
            "model_type": model_cfg["model_type"],
            "model_filename": model_cfg["model_filename"],
            "num_inference_steps": model_cfg.get("default_steps", 30),
        })

        # Optional per-model extras (guidance_scale, image_prompt_type, etc.)
        for extra_key in ["guidance_scale", "embedded_guidance_scale", "image_prompt_type",
                          "audio_prompt_type", "video_prompt_type", "flow_shift"]:
            if extra_key in model_cfg:
                base[extra_key] = model_cfg[extra_key]

        return base

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
        steps: int = -1,
        model: str = "ltx2_distilled",
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

        # Build settings — use model template unless fully overridden
        if settings_override:
            settings = settings_override.copy()
        else:
            settings = self._get_template(model)

        # Use model's default steps when caller passes -1 (auto)
        model_cfg = MODEL_TEMPLATES.get(model, MODEL_TEMPLATES["ltx2_distilled"])
        effective_steps = steps if steps > 0 else model_cfg.get("default_steps", 30)

        settings.update({
            "prompt": prompt,
            "resolution": resolution,
            "video_length": video_length,
            "seed": seed if seed > 0 else -1,
            "num_inference_steps": effective_steps,
            "output_filename": output_filename or job_id,
        })

        # image_prompt_type: explicit caller value overrides model template default
        if image_prompt_type:
            settings["image_prompt_type"] = image_prompt_type
        if audio_prompt_type:
            settings["audio_prompt_type"] = audio_prompt_type

        logger.info(f"📦 Model={model} base_model_type={settings.get('base_model_type')} steps={effective_steps}")

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

        # Record job start time for post-run file detection
        job_start_time = time.time()

        # Snapshot existing output files (names + sizes) BEFORE running job
        existing_outputs = set()
        try:
            existing_outputs = set(os.listdir(outputs_dir))
        except OSError:
            pass

        try:
            stdout, stderr = self._docker_exec(cmd, container_name)

            # Check for REAL generation errors in output
            # Must be actual Python exceptions, NOT HuggingFace tokenizer warnings
            combined_out = stdout + stderr
            real_error = (
                "Queue completed: 0/" in combined_out and (
                    "Traceback (most recent call last)" in combined_out or
                    "TypeError:" in combined_out or
                    "RuntimeError:" in combined_out or
                    "ValueError:" in combined_out or
                    "AttributeError:" in combined_out or
                    "CUDA out of memory" in combined_out or
                    "AssertionError:" in combined_out
                )
            )

            # Parse output filename from logs
            output_file = self._extract_output_filename(stdout, stderr)

            # If not found in logs, look for strictly NEW mp4 files (vs pre-job snapshot)
            if not output_file:
                for attempt in range(2):  # retry scan once after brief wait
                    try:
                        new_outputs = set(os.listdir(outputs_dir)) - existing_outputs
                        mp4_files = sorted(
                            [f for f in new_outputs if f.endswith('.mp4')],
                            key=lambda f: os.path.getmtime(os.path.join(outputs_dir, f)),
                            reverse=True
                        )
                        if mp4_files:
                            output_file = mp4_files[0]
                            logger.info(f"Detected new output file (scan attempt {attempt+1}): {output_file}")
                            break
                    except OSError:
                        pass
                    if not output_file and attempt == 0:
                        time.sleep(3)  # wait and retry scan once

            # If real error detected (even with an output file) → error
            if real_error:
                return {
                    "status": "error",
                    "output_file": None,
                    "stdout": stdout[-1000:],
                    "stderr": stderr[-1000:],
                }

            # If output file found → success
            if output_file:
                return {
                    "status": "success",
                    "output_file": output_file,
                    "stdout": stdout[-500:],
                    "stderr": stderr[-500:] if stderr else "",
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
        """Get full host path to output file.
        All GPU containers share a single outputs directory.
        """
        outputs_dir = os.getenv("WAN2GP_OUTPUTS_DIR", "/nvme0n1-disk/wan2gp_data/outputs")
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
