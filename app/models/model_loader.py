import time
from typing import Dict, Optional, Union
from app.schemas import ModelType
from .base_model import BaseInferenceModel
from .text_generation import TextGenerationModel
from .image_generation import ImageGenerationModel
from .text_to_speech import TextToSpeechModel
from .qwen_image_edit import QwenImageEditModel
import os
from app.core.config import settings


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
        """Get the appropriate model class for a given type"""
        path_lower = (model_path or "").lower()
        
        if model_type == ModelType.IMAGE_TO_IMAGE and "qwen" in path_lower:
            return QwenImageEditModel

        model_classes = {
            ModelType.TEXT_GENERATION: TextGenerationModel,
            ModelType.IMAGE_GENERATION: ImageGenerationModel,
            ModelType.TEXT_TO_IMAGE: ImageGenerationModel,
            ModelType.IMAGE_TO_IMAGE: ImageGenerationModel,
            ModelType.TEXT_TO_SPEECH: TextToSpeechModel,
        }

        model_class = model_classes.get(model_type)
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
        """Load a model into memory"""
        self._evict_idle()
        if model_id in self.loaded_models:
            self._touch(model_id)
            return self.loaded_models[model_id]

        if isinstance(model_type, str):
            model_type = ModelType(model_type)

        device = self._get_device(hardware)

        self._ensure_capacity_for_new_model()

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

        model = model_class(model_path=model_path, device=device)
        model.load()

        self.loaded_models[model_id] = model
        self._touch(model_id)
        self._enforce_max_loaded()
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
