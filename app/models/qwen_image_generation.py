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
            # Prefer the explicit pipeline when present, else rely on auto pipeline dispatch.
            from diffusers import QwenImagePipeline as _Pipeline  # type: ignore
        except Exception:
            try:
                from diffusers import DiffusionPipeline as _Pipeline  # type: ignore
            except Exception as exc:
                raise RuntimeError(
                    "Qwen image generation requires 'diffusers' (new enough to support Qwen Image)."
                ) from exc

        hf_token = settings.HF_API_TOKEN
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32

        def _load_pipeline(transformer_override=None):
            kwargs = dict(
                torch_dtype=dtype,
                trust_remote_code=True,  # ignored by some pipeline classes; safe to pass
                token=hf_token,
                cache_dir=settings.MODEL_CACHE_DIR,
                resume_download=True,
                local_files_only=False,
            )
            if transformer_override is not None:
                # Some pipelines use a diffusers "transformer" component; keep for compatibility,
                # but QwenImagePipeline's transformer-like weights may not be a transformers.PreTrainedModel.
                kwargs["transformer"] = transformer_override
            return _Pipeline.from_pretrained(self.model_path, **kwargs)

        try:
            self.pipe = _load_pipeline()
        except Exception as exc:
            msg = str(exc)
            placeholder_err = ("qwen2_5_vl" in msg and "Placeholder" in msg)
            dict_cfg_err = ("'dict' object has no attribute 'to_dict'" in msg)
            if placeholder_err or dict_cfg_err:
                # Work around transformers config bugs by pre-loading the *text_encoder* component with a sanitized
                # config and injecting it into the diffusers pipeline.
                try:
                    from transformers import AutoConfig, PretrainedConfig  # type: ignore
                    from transformers import Qwen2_5_VLForConditionalGeneration  # type: ignore

                    config = AutoConfig.from_pretrained(
                        self.model_path,
                        subfolder="text_encoder",
                        token=hf_token,
                        cache_dir=settings.MODEL_CACHE_DIR,
                    )

                    # Normalize nested configs that sometimes come back as raw dicts.
                    for attr in ("text_config", "vision_config", "decoder_config"):
                        if hasattr(config, attr) and isinstance(getattr(config, attr), dict):
                            setattr(config, attr, PretrainedConfig.from_dict(getattr(config, attr)))

                    text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                        self.model_path,
                        subfolder="text_encoder",
                        config=config,
                        torch_dtype=dtype,
                        token=hf_token,
                        cache_dir=settings.MODEL_CACHE_DIR,
                    )

                    # Inject the already-loaded transformers model to bypass diffusers component loading.
                    self.pipe = _Pipeline.from_pretrained(
                        self.model_path,
                        torch_dtype=dtype,
                        token=hf_token,
                        cache_dir=settings.MODEL_CACHE_DIR,
                        resume_download=True,
                        local_files_only=False,
                        text_encoder=text_encoder,
                    )
                except Exception as exc2:
                    raise RuntimeError(
                        "Failed to load Qwen Image pipeline. Ensure 'diffusers>=0.36', "
                        "'transformers' supporting Qwen2.5-VL, and deps: einops, timm, sentencepiece, qwen-vl-utils. "
                        f"Original error: {msg}"
                    ) from exc2
            else:
                raise
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
