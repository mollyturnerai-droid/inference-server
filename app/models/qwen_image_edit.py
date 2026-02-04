from typing import Dict, Any, Optional, List, Union
import os
import torch
from PIL import Image
from .base_model import BaseInferenceModel
from app.services.storage import storage_service
from urllib.parse import urlparse
import requests
from app.core.config import settings

# Attempt to import the specific pipeline. If it fails (old diffusers), we'll handle gracefully.
try:
    from diffusers import QwenImageEditPlusPipeline
except ImportError:
    QwenImageEditPlusPipeline = None

class QwenImageEditModel(BaseInferenceModel):
    def __init__(self, model_path: str, device: Optional[str] = None):
        super().__init__(model_path=model_path, device=device)
        self.pipe = None

    def load(self):
        """Load the Qwen Image Edit Plus pipeline"""
        if QwenImageEditPlusPipeline is None:
            raise RuntimeError(
                "QwenImageEditPlusPipeline not found. Ensure you have the latest 'diffusers' version (v0.36.0+ or from source)."
            )

        dtype = torch.float16 if self.device == "cuda" else torch.float32
        
        self.pipe = QwenImageEditPlusPipeline.from_pretrained(
            self.model_path,
            torch_dtype=dtype,
            trust_remote_code=True
        )
        
        self.pipe.to(self.device)
        self.pipe.set_progress_bar_config(disable=True)
        self.model = self.pipe

    def _load_image(self, image_input: str) -> Image.Image:
        if not image_input:
            return None
            
        parsed = urlparse(image_input)
        if not parsed.scheme:
            # Check if it's a local file relative to storage
            path = os.path.join(settings.STORAGE_PATH, image_input)
            if os.path.exists(path):
                return Image.open(path).convert("RGB")
            # Or absolute path
            if os.path.exists(image_input):
                 return Image.open(image_input).convert("RGB")
        
        if parsed.scheme == "file":
            return Image.open(parsed.path).convert("RGB")
            
        if image_input.startswith("/v1/files/"):
            rel = image_input[len("/v1/files/"):]
            path = os.path.join(settings.STORAGE_PATH, rel)
            return Image.open(path).convert("RGB")
            
        if parsed.scheme in ("http", "https"):
            resp = requests.get(image_input, stream=True, timeout=30)
            resp.raise_for_status()
            return Image.open(resp.raw).convert("RGB")
            
        raise ValueError(f"Unsupported image input: {image_input}")

    def predict(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """ Run the image editing inference """
        prompt = inputs.get("prompt", "")
        image_input = inputs.get("image")
        
        # Support multiple images if provided as a list
        reference_images = inputs.get("reference_images")
        
        input_image = self._load_image(image_input)
        
        # Prepare pipeline arguments
        pipe_kwargs = {
            "prompt": prompt,
            "image": input_image,
            "num_inference_steps": int(inputs.get("num_inference_steps", 20)),
            "guidance_scale": float(inputs.get("guidance_scale", 7.0)),
        }
        
        # Optional parameters
        if "negative_prompt" in inputs:
            pipe_kwargs["negative_prompt"] = inputs["negative_prompt"]
        if "seed" in inputs:
            generator = torch.Generator(device=self.device).manual_seed(int(inputs["seed"]))
            pipe_kwargs["generator"] = generator

        # Run inference
        with torch.inference_mode():
            output = self.pipe(**pipe_kwargs)
            
        output_image = output.images[0]
        
        # Save output image
        output_id = f"qwen-{os.urandom(4).hex()}"
        filename = f"{output_id}.png"
        
        storage_service.save_image(output_image, filename)
        
        return {
            "image_url": f"/v1/files/{filename}",
            "status": "succeeded"
        }
