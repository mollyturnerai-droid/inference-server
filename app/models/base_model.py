from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import torch
import os


class BaseInferenceModel(ABC):
    def __init__(self, model_path: str, device: Optional[str] = None):
        self.model_path = model_path
        self.device = device or self._get_device()
        self.model = None

    def _get_device(self) -> str:
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    @abstractmethod
    def load(self):
        """Load the model into memory"""
        pass

    @abstractmethod
    def predict(self, inputs: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Run inference on the model"""
        pass

    @abstractmethod
    def unload(self):
        """Unload the model from memory"""
        pass

    def get_memory_usage(self) -> int:
        """Get approximate memory usage in bytes"""
        if self.model is None:
            return 0

        if hasattr(self.model, 'get_memory_footprint'):
            return self.model.get_memory_footprint()

        return 0
