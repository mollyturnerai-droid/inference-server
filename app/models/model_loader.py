from typing import Dict, Optional, Union
from app.schemas import ModelType
from .base_model import BaseInferenceModel
from .text_generation import TextGenerationModel
from .image_generation import ImageGenerationModel
from .text_to_speech import TextToSpeechModel
import os
from app.core.config import settings


class ModelLoader:
    def __init__(self):
        self.loaded_models: Dict[str, BaseInferenceModel] = {}
        os.makedirs(settings.MODEL_CACHE_DIR, exist_ok=True)

    def get_model_class(self, model_type: ModelType) -> type:
        """Get the appropriate model class for a given type"""
        model_classes = {
            ModelType.TEXT_GENERATION: TextGenerationModel,
            ModelType.IMAGE_GENERATION: ImageGenerationModel,
            ModelType.TEXT_TO_IMAGE: ImageGenerationModel,  # Same as image-generation
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
        if model_id in self.loaded_models:
            return self.loaded_models[model_id]

        if isinstance(model_type, str):
            model_type = ModelType(model_type)

        device = self._get_device(hardware)

        if model_type == ModelType.TEXT_TO_SPEECH and (model_path or "").startswith("nvidia/magpie_tts"):
            try:
                from .magpie_text_to_speech import MagpieTextToSpeechModel
            except Exception as e:
                raise RuntimeError(
                    "Magpie TTS requires NeMo dependencies. Use the NeMo-enabled image variant or install nemo_toolkit[tts]."
                ) from e
            model_class = MagpieTextToSpeechModel
        else:
            model_class = self.get_model_class(model_type)

        model = model_class(model_path=model_path, device=device)
        model.load()

        self.loaded_models[model_id] = model
        return model

    def unload_model(self, model_id: str):
        """Unload a model from memory"""
        if model_id in self.loaded_models:
            model = self.loaded_models[model_id]
            model.unload()
            del self.loaded_models[model_id]

    def get_loaded_model(self, model_id: str) -> Optional[BaseInferenceModel]:
        """Get a loaded model by ID"""
        return self.loaded_models.get(model_id)

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


model_loader = ModelLoader()
