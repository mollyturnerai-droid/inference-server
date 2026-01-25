from typing import Dict, Any
import torch
from diffusers import StableDiffusionPipeline
from .base_model import BaseInferenceModel
import io
import uuid
from PIL import Image
from app.services.storage import storage_service


class ImageGenerationModel(BaseInferenceModel):
    def load(self):
        """Load a Stable Diffusion model"""
        self.model = StableDiffusionPipeline.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        )
        self.model = self.model.to(self.device)

        if self.device == "cuda":
            self.model.enable_attention_slicing()

    def predict(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Generate image from text prompt"""
        prompt = inputs.get("prompt", "")
        negative_prompt = inputs.get("negative_prompt", "")
        num_inference_steps = inputs.get("num_inference_steps", 50)
        guidance_scale = inputs.get("guidance_scale", 7.5)
        width = inputs.get("width", 512)
        height = inputs.get("height", 512)
        seed = inputs.get("seed")

        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        image = self.model(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            generator=generator
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
