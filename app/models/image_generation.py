from typing import Dict, Any, Optional
import os
# Set allocator config early to reduce fragmentation if not already set.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
import torch
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionXLImg2ImgPipeline,
)
from .base_model import BaseInferenceModel
import io
import uuid
from PIL import Image
from app.services.storage import storage_service
from app.core.config import settings
import inspect
from app.models.image_io import load_image_rgb


class ImageGenerationModel(BaseInferenceModel):
    def __init__(self, model_path: str, device: Optional[str] = None):
        super().__init__(model_path=model_path, device=device)
        self.pipeline_txt2img = None
        self.pipeline_img2img = None
        self._use_sdxl = False

    def load(self):
        """Load a Stable Diffusion model"""
        import logging
        logger = logging.getLogger(__name__)
        
        model_path_lower = (self.model_path or "").lower()
        self._use_sdxl = any(token in model_path_lower for token in ["sdxl", "playground", "xl"])
        pipeline_cls = StableDiffusionXLPipeline if self._use_sdxl else StableDiffusionPipeline

        # Log model loading attempt
        logger.info(f"Loading {'SDXL' if self._use_sdxl else 'SD'} model from: {self.model_path}")
        
        # Check if model_path is a local directory and log its contents
        if os.path.exists(self.model_path):
            logger.info(f"Model directory exists. Contents: {os.listdir(self.model_path)}")
            # Check for required components
            required_components = ["unet", "vae", "tokenizer", "text_encoder", "scheduler"]
            optional_components = ["safety_checker", "feature_extractor", "image_encoder"]
            missing_required = [c for c in required_components if not os.path.exists(os.path.join(self.model_path, c))]
            missing_optional = [c for c in optional_components if not os.path.exists(os.path.join(self.model_path, c))]
            
            if missing_required:
                logger.warning(f"Missing required components: {missing_required}")
            if missing_optional:
                logger.info(f"Missing optional components: {missing_optional}")
        
        def _load_pipeline(force_download: bool = False):
            from app.core.config import settings
            hf_token = settings.HF_API_TOKEN
            kwargs = dict(
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                token=hf_token,
                cache_dir=settings.MODEL_CACHE_DIR,
                resume_download=True,
                local_files_only=False,
            )
            if settings.DISABLE_SAFETY_CHECKER:
                kwargs["safety_checker"] = None
                kwargs["requires_safety_checker"] = False
            if force_download and not os.path.exists(self.model_path):
                kwargs["force_download"] = True
            return pipeline_cls.from_pretrained(self.model_path, **kwargs)

        try:
            self.pipeline_txt2img = _load_pipeline()
        except Exception as e:
            logger.error(f"Failed to load pipeline from {self.model_path}: {str(e)}")
            
            # Try to provide more helpful error message
            if "expected" in str(e).lower() and "but only" in str(e).lower():
                logger.error(
                    "The model directory appears to be incomplete or corrupted. "
                    "Please ensure all required components (unet, vae, tokenizer, text_encoder, scheduler) "
                    "are present in the model directory."
                )

                # Try alternative loading strategies
                logger.info("Attempting alternative loading strategies...")

                # Strategy 1: Force re-download into cache
                if not os.path.exists(self.model_path):
                    try:
                        self.pipeline_txt2img = _load_pipeline(force_download=True)
                        logger.info("Successfully re-downloaded model from hub")
                        return
                    except Exception as retry_e:
                        logger.warning(f"Force download failed: {str(retry_e)}")

                # Strategy 2: Try loading from subdirectories if they exist
                if os.path.exists(self.model_path):
                    subdirs = [d for d in os.listdir(self.model_path) if os.path.isdir(os.path.join(self.model_path, d))]
                    logger.info(f"Found subdirectories: {subdirs}")
                    
                    # Check if there's a single subdirectory that might be the actual model
                    if len(subdirs) == 1:
                        alt_path = os.path.join(self.model_path, subdirs[0])
                        logger.info(f"Trying to load from subdirectory: {alt_path}")
                        try:
                            alt_kwargs = dict(
                                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                                token=settings.HF_API_TOKEN,
                                cache_dir=settings.MODEL_CACHE_DIR,
                                resume_download=True,
                                local_files_only=False,
                            )
                            if settings.DISABLE_SAFETY_CHECKER:
                                alt_kwargs["safety_checker"] = None
                                alt_kwargs["requires_safety_checker"] = False
                            self.pipeline_txt2img = pipeline_cls.from_pretrained(
                                alt_path, **alt_kwargs,
                            )
                            logger.info(f"Successfully loaded from subdirectory: {alt_path}")
                            self.model_path = alt_path  # Update the model path
                            return
                        except Exception as sub_e:
                            logger.warning(f"Subdirectory loading also failed: {str(sub_e)}")
                            raise e  # Raise the original exception
                    else:
                        raise e
                else:
                    raise e
            else:
                raise
        
        # Silence diffusers tqdm progress output; we report progress via callbacks.
        self.pipeline_txt2img.set_progress_bar_config(disable=True)
        self.pipeline_txt2img = self.pipeline_txt2img.to(self.device)
        self.model = self.pipeline_txt2img

        if self.device == "cuda":
            self.pipeline_txt2img.enable_attention_slicing()
        
        logger.info(f"Successfully loaded {pipeline_cls.__name__} on {self.device}")

    def _get_img2img_pipeline(self):
        import logging
        logger = logging.getLogger(__name__)
        
        if self.pipeline_img2img is not None:
            return self.pipeline_img2img
        pipeline_cls = (
            StableDiffusionXLImg2ImgPipeline if self._use_sdxl else StableDiffusionImg2ImgPipeline
        )
        
        logger.info(f"Loading {'SDXL' if self._use_sdxl else 'SD'} img2img pipeline from: {self.model_path}")
        
        def _load_img2img(force_download: bool = False):
            from app.core.config import settings
            hf_token = settings.HF_API_TOKEN
            kwargs = dict(
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                token=hf_token,
                cache_dir=settings.MODEL_CACHE_DIR,
                resume_download=True,
                local_files_only=False,
            )
            if settings.DISABLE_SAFETY_CHECKER:
                kwargs["safety_checker"] = None
                kwargs["requires_safety_checker"] = False
            if force_download and not os.path.exists(self.model_path):
                kwargs["force_download"] = True
            return pipeline_cls.from_pretrained(self.model_path, **kwargs)

        try:
            self.pipeline_img2img = _load_img2img()
        except Exception as e:
            logger.error(f"Failed to load img2img pipeline from {self.model_path}: {str(e)}")
            if "expected" in str(e).lower() and "but only" in str(e).lower():
                logger.error(
                    "The model directory appears to be incomplete or corrupted. "
                    "Please ensure all required components (unet, vae, tokenizer, text_encoder, scheduler) "
                    "are present in the model directory."
                )
                if not os.path.exists(self.model_path):
                    try:
                        self.pipeline_img2img = _load_img2img(force_download=True)
                        logger.info("Successfully re-downloaded img2img pipeline from hub")
                        return self.pipeline_img2img
                    except Exception as retry_e:
                        logger.warning(f"Force download for img2img failed: {str(retry_e)}")
            raise
        
        # Silence diffusers tqdm progress output; we report progress via callbacks.
        self.pipeline_img2img.set_progress_bar_config(disable=True)
        self.pipeline_img2img = self.pipeline_img2img.to(self.device)
        if self.device == "cuda":
            self.pipeline_img2img.enable_attention_slicing()
        
        logger.info(f"Successfully loaded {pipeline_cls.__name__} on {self.device}")
        return self.pipeline_img2img

    def _load_image(self, image_input: str) -> Image.Image:
        return load_image_rgb(
            image_input,
            storage_path=settings.STORAGE_PATH,
            api_base_url=settings.API_BASE_URL,
            timeout_s=30.0,
            allow_private=settings.ALLOW_PRIVATE_URL_FETCH,
        )

    def _apply_vram_limits(self, width: int, height: int, steps: int):
        if self.device != "cuda" or not torch.cuda.is_available():
            return width, height, steps

        free_bytes, total_bytes = torch.cuda.mem_get_info()
        free_gb = free_bytes / (1024 ** 3)
        pixels = max(width * height, 1)

        if free_gb < 2.0:
            max_pixels = 512 * 512
            max_steps = 8
        elif free_gb < 3.0:
            max_pixels = 768 * 768
            max_steps = 10
        else:
            max_pixels = 1024 * 1024
            max_steps = steps

        if pixels > max_pixels:
            scale = (max_pixels / float(pixels)) ** 0.5
            width = int(width * scale)
            height = int(height * scale)

            # Round down to multiples of 64 for diffusion models.
            width = max(64, (width // 64) * 64)
            height = max(64, (height // 64) * 64)

        steps = min(steps, max_steps)
        return width, height, steps

    def _run_pipeline(self, pipeline, *, progress_callback, callback_steps, **kwargs):
        params = inspect.signature(pipeline.__call__).parameters
        if "callback_on_step_end" in params:
            if progress_callback:
                def _on_step_end(pipe, step, timestep, callback_kwargs):
                    latents = callback_kwargs.get("latents")
                    progress_callback(step, timestep, latents)
                    return callback_kwargs

                kwargs["callback_on_step_end"] = _on_step_end
                kwargs["callback_on_step_end_tensor_inputs"] = ["latents"]
        else:
            if progress_callback:
                kwargs["callback"] = progress_callback
                kwargs["callback_steps"] = callback_steps
        return pipeline(**kwargs).images[0]

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

        width, height, num_inference_steps = self._apply_vram_limits(
            width, height, num_inference_steps
        )

        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        progress_callback = kwargs.get("progress_callback")
        callback_steps = kwargs.get("callback_steps", 1)

        image_input = inputs.get("image") or inputs.get("init_image") or inputs.get("input_image")
        if image_input:
            init_image = image_input
            if isinstance(image_input, str):
                init_image = self._load_image(image_input)
            strength = float(inputs.get("strength") or 0.75)
            pipeline = self._get_img2img_pipeline()
            image = self._run_pipeline(
                pipeline,
                progress_callback=progress_callback,
                callback_steps=callback_steps,
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=init_image,
                strength=strength,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height,
                generator=generator,
            )
        else:
            image = self._run_pipeline(
                self.pipeline_txt2img,
                progress_callback=progress_callback,
                callback_steps=callback_steps,
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height,
                generator=generator,
            )

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
        if self.pipeline_txt2img is not None:
            del self.pipeline_txt2img
        if self.pipeline_img2img is not None:
            del self.pipeline_img2img
        self.pipeline_txt2img = None
        self.pipeline_img2img = None
        self.model = None

        if self.device == "cuda":
            torch.cuda.empty_cache()
