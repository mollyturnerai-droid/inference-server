import time
from typing import Dict, Optional, Union
from app.schemas import ModelType
from .base_model import BaseInferenceModel
import os
from app.core.config import settings
from app.core.model_registry import get_registry


class ModelLoader:
    def __init__(self):
        self.loaded_models: Dict[str, BaseInferenceModel] = {}
        self._last_access: Dict[str, float] = {}
        os.makedirs(settings.MODEL_CACHE_DIR, exist_ok=True)

    def _touch(self, model_id: str):
        self._last_access[model_id] = time.time()

    def _evict_idle(self):
        ttl = settings.MODEL_IDLE_TTL_SECONDS
        if not ttl:
            return
        now = time.time()
        for model_id, ts in list(self._last_access.items()):
            if now - ts > ttl:
                self.unload_model(model_id)

    def _enforce_max_loaded(self):
        max_loaded = settings.MAX_LOADED_MODELS
        if not max_loaded or max_loaded <= 0:
            return

        while len(self.loaded_models) > max_loaded:
            if not self._last_access:
                break
            lru_model_id = min(self._last_access, key=self._last_access.get)
            self.unload_model(lru_model_id)

    def _ensure_capacity_for_new_model(self):
        self._evict_idle()
        max_loaded = settings.MAX_LOADED_MODELS
        if not max_loaded or max_loaded <= 0:
            return
        while len(self.loaded_models) >= max_loaded:
            if not self._last_access:
                break
            lru_model_id = min(self._last_access, key=self._last_access.get)
            self.unload_model(lru_model_id)

    def get_model_class(self, model_type: ModelType, model_path: str = "") -> type:
        """Get the appropriate model class for a given type using the registry."""
        registry = get_registry()
        model_class = registry.get_model_class(model_path, model_type)
        
        if model_class is None:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        return model_class

    def load_model(
        self,
        model_id: str,
        model_type: Union[ModelType, str],
        model_path: str,
        hardware: str = "auto"
    ) -> BaseInferenceModel:
        """Load a model into memory with validation and resource checking."""
        import logging
        from app.core.model_validator import validate_model
        from app.core.gpu_monitor import get_gpu_monitor
        
        logger = logging.getLogger(__name__)
        
        self._evict_idle()
        if model_id in self.loaded_models:
            self._touch(model_id)
            return self.loaded_models[model_id]

        if isinstance(model_type, str):
            model_type = ModelType(model_type)

        # Pre-flight validation
        validation = validate_model(model_path, model_type, hardware)
        if not validation.is_valid:
            error_msg = "; ".join(validation.errors)
            logger.error(f"Model validation failed for {model_path}: {error_msg}")
            raise ValueError(f"Model validation failed: {error_msg}")
        
        if validation.warnings:
            for warning in validation.warnings:
                logger.warning(f"Model validation warning for {model_path}: {warning}")
        
        # Log detected framework info
        if validation.metadata.get("framework"):
            framework_info = validation.metadata["framework"]
            if framework_info.get("detected"):
                logger.info(
                    f"Detected framework for {model_path}: {framework_info.get('framework')} "
                    f"(pipeline: {framework_info.get('pipeline_class', 'unknown')})"
                )

        device = self._get_device(hardware)
        
        # GPU resource check
        if device == "cuda":
            gpu = get_gpu_monitor()
            stats = gpu.get_stats()
            if stats:
                logger.info(
                    f"GPU Memory before loading {model_id}: "
                    f"{stats.used_gb:.2f}GB used, {stats.free_gb:.2f}GB free"
                )

        self._ensure_capacity_for_new_model()

        # Special case for Magpie TTS (requires NeMo)
        if model_type == ModelType.TEXT_TO_SPEECH and (model_path or "").startswith("nvidia/magpie_tts"):
            try:
                from .magpie_text_to_speech import MagpieTextToSpeechModel
            except Exception as e:
                raise RuntimeError(
                    "Magpie TTS requires NeMo dependencies. Use the NeMo-enabled image variant or install nemo_toolkit[tts]."
                ) from e
            model_class = MagpieTextToSpeechModel
        else:
            model_class = self.get_model_class(model_type, model_path)

        logger.info(f"Loading model {model_id} using {model_class.__name__} on {device}")
        model = model_class(model_path=model_path, device=device)
        model.load()

        self.loaded_models[model_id] = model
        self._touch(model_id)
        self._enforce_max_loaded()
        
        # Log GPU usage after loading
        if device == "cuda":
            gpu = get_gpu_monitor()
            stats = gpu.get_stats()
            if stats:
                logger.info(
                    f"GPU Memory after loading {model_id}: "
                    f"{stats.used_gb:.2f}GB used, {stats.free_gb:.2f}GB free"
                )
        
        return model

    def unload_model(self, model_id: str):
        """Unload a model from memory"""
        if model_id in self.loaded_models:
            model = self.loaded_models[model_id]
            model.unload()
            del self.loaded_models[model_id]
            self._last_access.pop(model_id, None)

    def get_loaded_model(self, model_id: str) -> Optional[BaseInferenceModel]:
        """Get a loaded model by ID"""
        self._evict_idle()
        model = self.loaded_models.get(model_id)
        if model is not None:
            self._touch(model_id)
        return model

    def _get_device(self, hardware: str) -> str:
        """Determine the device to use"""
        if hardware == "cpu":
            return "cpu"
        elif hardware == "gpu":
            return "cuda"
        else:  # auto
            import torch
            if torch.cuda.is_available() and settings.ENABLE_GPU:
                return "cuda"
            return "cpu"

    def clear_cache(self):
        """Unload all models"""
        for model_id in list(self.loaded_models.keys()):
            self.unload_model(model_id)

        self._last_access.clear()


model_loader = ModelLoader()
