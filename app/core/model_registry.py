"""
Model class registry for dynamic model type resolution.
This replaces hardcoded model class mappings with a flexible registry pattern.
"""

from typing import Dict, Type, Optional, Callable
from app.schemas import ModelType
from app.models.base_model import BaseInferenceModel


class ModelRegistry:
    """Registry for model class implementations."""
    
    def __init__(self):
        self._registry: Dict[ModelType, Type[BaseInferenceModel]] = {}
        self._matchers: list[tuple[Callable[[str, ModelType], bool], Type[BaseInferenceModel]]] = []
    
    def register(self, model_type: ModelType, model_class: Type[BaseInferenceModel]):
        """Register a model class for a specific model type."""
        self._registry[model_type] = model_class
    
    def register_matcher(
        self, 
        matcher: Callable[[str, ModelType], bool], 
        model_class: Type[BaseInferenceModel]
    ):
        """
        Register a model class with a custom matcher function.
        
        Args:
            matcher: Function that takes (model_path, model_type) and returns True if this class should handle it
            model_class: The model class to use when matcher returns True
        """
        self._matchers.append((matcher, model_class))
    
    def get_model_class(self, model_path: str, model_type: ModelType) -> Optional[Type[BaseInferenceModel]]:
        """
        Get the appropriate model class for a given path and type.
        
        First checks custom matchers, then falls back to type-based registry.
        """
        # Check custom matchers first (most specific)
        for matcher, model_class in self._matchers:
            if matcher(model_path, model_type):
                return model_class
        
        # Fall back to type-based registry
        return self._registry.get(model_type)
    
    def __repr__(self):
        return f"ModelRegistry(types={list(self._registry.keys())}, matchers={len(self._matchers)})"


# Global registry instance
_global_registry: Optional[ModelRegistry] = None


def get_registry() -> ModelRegistry:
    """Get the global model registry instance."""
    global _global_registry
    if _global_registry is None:
        _global_registry = ModelRegistry()
        _initialize_default_registry(_global_registry)
    return _global_registry


def _initialize_default_registry(registry: ModelRegistry):
    """Initialize the registry with default model mappings."""
    # Import here to avoid circular dependencies
    from app.models.text_generation import TextGenerationModel
    from app.models.image_generation import ImageGenerationModel
    from app.models.qwen_image_generation import QwenImageGenerationModel
    from app.models.text_to_speech import TextToSpeechModel
    from app.models.qwen_image_edit import QwenImageEditModel
    from app.schemas import ModelType
    
    # Register standard model types
    registry.register(ModelType.TEXT_GENERATION, TextGenerationModel)
    registry.register(ModelType.IMAGE_GENERATION, ImageGenerationModel)
    registry.register(ModelType.TEXT_TO_IMAGE, ImageGenerationModel)
    registry.register(ModelType.IMAGE_TO_IMAGE, ImageGenerationModel)  # Default fallback
    registry.register(ModelType.TEXT_TO_SPEECH, TextToSpeechModel)
    
    # Register custom matchers for specialized models
    def is_qwen_image_generation(model_path: str, model_type: ModelType) -> bool:
        return (
            model_type in (ModelType.TEXT_TO_IMAGE, ModelType.IMAGE_GENERATION)
            and "qwen-image" in (model_path or "").lower()
        )

    def is_qwen_image_edit(model_path: str, model_type: ModelType) -> bool:
        """Matcher for Qwen image editing models."""
        return (
            model_type == ModelType.IMAGE_TO_IMAGE 
            and "qwen" in model_path.lower() 
            and "edit" in model_path.lower()
        )
    
    registry.register_matcher(is_qwen_image_generation, QwenImageGenerationModel)
    registry.register_matcher(is_qwen_image_edit, QwenImageEditModel)
