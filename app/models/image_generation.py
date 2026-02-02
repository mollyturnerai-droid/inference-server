from typing import Dict, Any
import torch
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
from .base_model import BaseInferenceModel
import io
import uuid
from PIL import Image
from app.services.storage import storage_service


class ImageGenerationModel(BaseInferenceModel):
    def load(self):
        """Load a Stable Diffusion model"""
        model_path_lower = (self.model_path or "").lower()
        use_sdxl = any(token in model_path_lower for token in ["sdxl", "playground", "xl"])
        pipeline_cls = StableDiffusionXLPipeline if use_sdxl else StableDiffusionPipeline

        self.model = pipeline_cls.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        )
        self.model = self.model.to(self.device)

        if self.device == "cuda":
            self.model.enable_attention_slicing()

    def predict(self, inputs: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Generate image from text prompt"""
        prompt = inputs.get("prompt")
        if not prompt or not isinstance(prompt, str):
            raise ValueError("Missing required input: prompt")
        negative_prompt = inputs.get("negative_prompt") or ""
        num_inference_steps = int(inputs.get("num_inference_steps") or 50)
        guidance_scale = float(inputs.get("guidance_scale") or 7.5)
        width = int(inputs.get("width") or 512)
        height = int(inputs.get("height") or 512)
        seed = inputs.get("seed")

        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        progress_callback = kwargs.get("progress_callback")
        callback_steps = kwargs.get("callback_steps", 1)

        image = self.model(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            generator=generator,
            callback=progress_callback,
            callback_steps=callback_steps,
        ).images[0]

        # Save image to storage and return URL
        image_id = str(uuid.uuid4())
        filename = f"images/{image_id}.png"

        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        image_bytes = buffered.getvalue()

        storage_service.save_file_sync(filename, image_bytes, content_type="image/png")
        image_url = storage_service.get_public_url(filename)

        return {
            "image_url": image_url,
            "image_id": image_id,
            "prompt": prompt,
            "width": width,
            "height": height
        }

    def unload(self):
        """Unload model from memory"""
        del self.model
        self.model = None

        if self.device == "cuda":
            torch.cuda.empty_cache()
