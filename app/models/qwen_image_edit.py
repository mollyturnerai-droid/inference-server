from typing import Dict, Any, Optional
import os
import torch
from PIL import Image
from .base_model import BaseInferenceModel
from app.services.storage import storage_service
from app.core.config import settings
from app.models.image_io import load_image_rgb

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
        hf_token = settings.HF_API_TOKEN
        
        self.pipe = QwenImageEditPlusPipeline.from_pretrained(
            self.model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
            token=hf_token
        )
        
        self.pipe.to(self.device)
        self.pipe.set_progress_bar_config(disable=True)
        self.model = self.pipe

    def _load_image(self, image_input: str) -> Image.Image:
        if not image_input:
            return None
        return load_image_rgb(
            image_input,
            storage_path=settings.STORAGE_PATH,
            api_base_url=settings.API_BASE_URL,
            timeout_s=30.0,
        )

    def predict(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """ Run the image editing inference """
        prompt = inputs.get("prompt", "")
        image_input = inputs.get("image")

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
        filename = f"images/{output_id}.png"
        
        storage_service.save_image(output_image, filename)
        
        return {
            "image_url": storage_service.get_public_url(filename),
            "status": "succeeded"
        }
