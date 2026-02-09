from __future__ import annotations

import io
import inspect
import os
import uuid
from typing import Any, Dict, Optional

import torch
from PIL import Image

from app.core.config import settings
from app.models.base_model import BaseInferenceModel
from app.services.storage import storage_service


class QwenImageGenerationModel(BaseInferenceModel):
    """Qwen Image (text-to-image) via diffusers DiffusionPipeline."""

    def __init__(self, model_path: str, device: Optional[str] = None):
        super().__init__(model_path=model_path, device=device)
        self.pipe = None

    def load(self):
        try:
            from diffusers import DiffusionPipeline
        except Exception as exc:
            raise RuntimeError(
                "Qwen image generation requires 'diffusers' (new enough to support Qwen Image)."
            ) from exc

        hf_token = settings.HF_API_TOKEN
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32

        self.pipe = DiffusionPipeline.from_pretrained(
            self.model_path,
            torch_dtype=dtype,
            token=hf_token,
            cache_dir=settings.MODEL_CACHE_DIR,
            resume_download=True,
            local_files_only=False,
        )
        self.pipe.set_progress_bar_config(disable=True)
        self.pipe = self.pipe.to(self.device)
        self.model = self.pipe

    def _run(self, pipe, *, progress_callback=None, callback_steps: int = 1, **kwargs):
        params = inspect.signature(pipe.__call__).parameters

        # Filter kwargs to what the pipeline supports.
        filtered = {k: v for k, v in kwargs.items() if k in params and v is not None}

        if progress_callback:
            if "callback_on_step_end" in params:
                def _on_step_end(_pipe, step, timestep, callback_kwargs):
                    latents = callback_kwargs.get("latents")
                    progress_callback(step, timestep, latents)
                    return callback_kwargs

                filtered["callback_on_step_end"] = _on_step_end
                if "callback_on_step_end_tensor_inputs" in params:
                    filtered["callback_on_step_end_tensor_inputs"] = ["latents"]
            elif "callback" in params:
                filtered["callback"] = progress_callback
                if "callback_steps" in params:
                    filtered["callback_steps"] = callback_steps

        out = pipe(**filtered)
        images = getattr(out, "images", None)
        if not images:
            raise RuntimeError("Pipeline did not return any images")
        return images[0]

    def predict(self, inputs: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        prompt = inputs.get("prompt")
        if not prompt or not isinstance(prompt, str):
            raise ValueError("Missing required input: prompt")

        negative_prompt = inputs.get("negative_prompt")
        steps = inputs.get("num_inference_steps")
        guidance_scale = inputs.get("guidance_scale")
        seed = inputs.get("seed")
        width = inputs.get("width")
        height = inputs.get("height")

        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(int(seed))

        image = self._run(
            self.pipe,
            progress_callback=kwargs.get("progress_callback"),
            callback_steps=int(kwargs.get("callback_steps", 1)),
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=int(steps) if steps is not None else None,
            guidance_scale=float(guidance_scale) if guidance_scale is not None else None,
            width=int(width) if width is not None else None,
            height=int(height) if height is not None else None,
            generator=generator,
        )

        image_id = str(uuid.uuid4())
        filename = f"images/{image_id}.png"

        buf = io.BytesIO()
        image.save(buf, format="PNG")
        storage_service.save_file_sync(filename, buf.getvalue(), content_type="image/png")

        return {
            "image_url": storage_service.get_public_url(filename),
            "image_id": image_id,
            "prompt": prompt,
        }

    def unload(self):
        if self.pipe is not None:
            del self.pipe
        self.pipe = None
        self.model = None
        if self.device == "cuda":
            torch.cuda.empty_cache()

