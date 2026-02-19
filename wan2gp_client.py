"""
Wan2GP Client for server-side Docker execution.
No SSH required - runs directly on the same host as wan2gp container.
"""
import subprocess
import json
import uuid
import os
from pathlib import Path
from typing import Dict, Optional

class Wan2GPClient:
    """Client for submitting jobs to the wan2gp Docker container."""
    
    DOCKER_CONTAINER = "wan2gp"
    SETTINGS_DIR = os.getenv("WAN2GP_SETTINGS_DIR", "/workspace/settings")
    OUTPUTS_DIR = os.getenv("WAN2GP_OUTPUTS_DIR", "/workspace/outputs")
    
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
            "settings_version":2.49,
            "base_model_type": "ltx2_19B",
            "model_type": "ltx2_distilled",
            "model_filename": "https://huggingface.co/DeepBeepMeep/LTX-2/resolve/main/ltx-2-19b-distilled-fp8_diffusion_model.safetensors",
            "image_mode": 0,
            "prompt": "",
            "alt_prompt": "",
            "resolution": "1280x720",
            "video_length": 81,
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
    
    def _docker_exec(self, cmd: str) -> tuple[str, str]:
        """Execute command in wan2gp container."""
        full_cmd = f"docker exec {self.DOCKER_CONTAINER} {cmd}"
        result = subprocess.run(
            full_cmd, 
            shell=True, 
            capture_output=True, 
            text=True,
            timeout=1800  # 30 min timeout
        )
        return result.stdout, result.stderr
    
    def submit_job(
        self,
        job_id: str,
        prompt: str,
        resolution: str = "1280x720",
        video_length: int = 81,
        seed: int = -1,
        steps: int = 8,
        loras: Optional[Dict[str, float]] = None,
        output_filename: str = "",
        settings_override: Optional[Dict] = None
    ) -> Dict:
        """
        Submit a video generation job.
        
        Args:
            job_id: Unique job identifier
            prompt: Video description
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
            "output_filename": output_filename or job_id
        })
        
        if loras:
            if isinstance(loras, dict):
                settings["activated_loras"] = list(loras.keys())
                settings["loras_multipliers"] = " ".join(str(v) for v in loras.values())
            elif isinstance(loras, list):
                # Handle cases where it might already be a list or needs custom handling
                settings["activated_loras"] = loras
        
        # Write settings file
        settings_filename = f"api_job_{job_id}.json"
        settings_path = Path(self.SETTINGS_DIR) / settings_filename
        
        with open(settings_path, 'w') as f:
            json.dump(settings, indent=2, fp=f)
        
        # Execute headless generation
        docker_settings_path = f"/workspace/settings/{settings_filename}"
        cmd = f"python3 wgp.py --process {docker_settings_path}"
        
        try:
            stdout, stderr = self._docker_exec(cmd)
            
            # Parse output filename
            output_file = self._extract_output_filename(stdout, stderr)
            
            if "error" in stderr.lower() or "traceback" in stderr.lower():
                return {
                    "status": "error",
                    "output_file": None,
                    "stdout": stdout[:1000],
                    "stderr": stderr[:1000]
                }
            
            return {
                "status": "success",
                "output_file": output_file,
                "stdout": stdout[-500:],  # Last 500 chars
                "stderr": stderr[-500:] if stderr else ""
            }
        
        except subprocess.TimeoutExpired:
            return {
                "status": "error",
                "output_file": None,
                "stdout": "",
                "stderr": "Generation timeout (30 min exceeded)"
            }
        except Exception as e:
            return {
                "status": "error",
                "output_file": None,
                "stdout": "",
                "stderr": str(e)
            }
    
    def _extract_output_filename(self, stdout: str, stderr: str) -> Optional[str]:
        """Extract output filename from generation logs."""
        combined = stdout + stderr
        
        # Look for common patterns
        for line in combined.split('\n'):
            if 'saved video:' in line.lower() or 'output:' in line.lower():
                # Extract .mp4 filename
                if '.mp4' in line:
                    parts = line.split('/')
                    for part in reversed(parts):
                        if '.mp4' in part:
                            return part.strip()
        
        # Fallback: look for any .mp4 mention
        for line in combined.split('\n'):
            if '.mp4' in line:
                words = line.split()
                for word in words:
                    if '.mp4' in word:
                        return word.strip().rstrip(',').rstrip('.')
        
        return None
    
    def list_outputs(self, limit: int = 20) -> list[str]:
        """List recent output files."""
        try:
            stdout, _ = self._docker_exec(f"ls -t /workspace/outputs/*.mp4 | head -{limit}")
            files = [f.split('/')[-1] for f in stdout.strip().split('\n') if f]
            return files
        except:
            return []
    
    def get_output_path(self, filename: str) -> Path:
        """Get full path to output file."""
        return Path(self.OUTPUTS_DIR) / filename
