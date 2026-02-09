"""
Model compatibility validation and pre-flight checks.
Validates models before loading to catch errors early.
"""

from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from pathlib import Path
import fnmatch
import json
import logging
from app.schemas import ModelType

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of model validation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    metadata: Dict[str, Any]


class ModelValidator:
    """Validates model compatibility before loading."""
    
    @staticmethod
    def validate_for_loading(
        model_path: str,
        model_type: ModelType,
        hardware: str = "auto"
    ) -> ValidationResult:
        """
        Run pre-flight checks before loading a model.
        
        Args:
            model_path: Path or HuggingFace repo ID
            model_type: Expected model type
            hardware: Target hardware (cpu/gpu/auto)
            
        Returns:
            ValidationResult with errors, warnings, and metadata
        """
        errors = []
        warnings = []
        metadata = {}
        
        # Check 1: Path/repo validation
        if not model_path or not model_path.strip():
            errors.append("Model path cannot be empty")
        
        # Check 2: Hardware compatibility
        if hardware == "gpu":
            import torch
            if not torch.cuda.is_available():
                errors.append("GPU hardware requested but CUDA is not available")
        
        # Check 3: Local path validation
        local_path = model_path.replace("file://", "") if model_path.startswith("file://") else model_path
        if model_path.startswith("file://") or Path(local_path).exists():
            if not Path(local_path).exists():
                errors.append(f"Local path does not exist: {local_path}")
            else:
                metadata["path_type"] = "local"
                metadata["local_path"] = str(Path(local_path).resolve())
                diffusers_error = ModelValidator._validate_local_diffusers(Path(local_path))
                if diffusers_error:
                    errors.append(diffusers_error)
        
        # Check 4: Framework detection for HuggingFace models
        elif "/" in model_path:
            framework_info = ModelValidator._detect_framework(model_path)
            metadata["framework"] = framework_info
            
            # Validate framework matches model type
            validation_errors = ModelValidator._validate_framework_match(
                framework_info, model_type
            )
            errors.extend(validation_errors)
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            metadata=metadata
        )
    
    @staticmethod
    def _detect_framework(repo_id: str) -> Dict[str, Any]:
        """
        Detect the framework used by a HuggingFace model.
        
        Returns:
            Dictionary with framework info (diffusers, transformers, etc.)
        """
        info = {
            "detected": False,
            "framework": "unknown",
            "pipeline_class": None
        }
        
        # Common patterns for framework detection
        repo_lower = repo_id.lower()
        
        # Qwen Image Edit models use diffusers with special pipeline
        if "qwen" in repo_lower and "image-edit" in repo_lower:
            info["detected"] = True
            info["framework"] = "diffusers"
            info["pipeline_class"] = "QwenImageEditPlusPipeline"
            return info

        # Qwen Image generation models are diffusers-based (not Qwen LLMs).
        # Example: Qwen/Qwen-Image-2512
        if "qwen-image" in repo_lower:
            info["detected"] = True
            info["framework"] = "diffusers"
            info["pipeline_class"] = "DiffusionPipeline"
            return info
        
        # FLUX models use diffusers
        if "flux" in repo_lower:
            info["detected"] = True
            info["framework"] = "diffusers"
            info["pipeline_class"] = "FluxPipeline"
            return info
        
        # Stable Diffusion variants
        if any(sd in repo_lower for sd in ["stable-diffusion", "sdxl", "sd-", "playground"]):
            info["detected"] = True
            info["framework"] = "diffusers"
            if "xl" in repo_lower:
                info["pipeline_class"] = "StableDiffusionXLPipeline"
            else:
                info["pipeline_class"] = "StableDiffusionPipeline"
            return info
        
        # Whisper models use transformers
        if "whisper" in repo_lower:
            info["detected"] = True
            info["framework"] = "transformers"
            info["pipeline_class"] = "WhisperForConditionalGeneration"
            return info
        
        # Generic text generation (Qwen, Llama, GPT, etc.)
        if any(model in repo_lower for model in ["qwen", "llama", "gpt", "mistral", "phi"]):
            info["detected"] = True
            info["framework"] = "transformers"
            info["pipeline_class"] = "AutoModelForCausalLM"
            return info
        
        # BGER, sentence transformers
        if "bge" in repo_lower or "sentence-transformer" in repo_lower or "minilm" in repo_lower:
            info["detected"] = True
            info["framework"] = "sentence-transformers"
            return info
        
        return info
    
    @staticmethod
    def _validate_framework_match(framework_info: Dict[str, Any], model_type: ModelType) -> List[str]:
        """
        Validate that detected framework matches expected model type.
        
        Returns:
            List of validation error messages
        """
        errors = []
        
        if not framework_info.get("detected"):
            # Can't validate if framework unknown
            return []
        
        framework = framework_info.get("framework", "unknown")
        
        # Validate framework-type compatibility
        if model_type in (ModelType.TEXT_TO_IMAGE, ModelType.IMAGE_GENERATION, ModelType.IMAGE_TO_IMAGE):
            if framework not in ("diffusers", "unknown"):
                errors.append(
                    f"Model type {model_type} requires 'diffusers' framework, "
                    f"but detected '{framework}'"
                )
        
        elif model_type == ModelType.TEXT_GENERATION:
            if framework not in ("transformers", "unknown"):
                errors.append(
                    f"Model type {model_type} requires 'transformers' framework, "
                    f"but detected '{framework}'"
                )
        
        elif model_type == ModelType.EMBEDDINGS:
            if framework not in ("sentence-transformers", "transformers", "unknown"):
                errors.append(
                    f"Model type {model_type} requires 'sentence-transformers' or 'transformers', "
                    f"but detected '{framework}'"
                )
        
        return errors

    @staticmethod
    def _validate_local_diffusers(local_path: Path) -> Optional[str]:
        model_index = local_path / "model_index.json"
        unet_dir = local_path / "unet"
        if not model_index.exists() and not unet_dir.exists():
            return None

        required_dirs = ["unet", "vae", "scheduler", "text_encoder"]
        missing_dirs = [d for d in required_dirs if not (local_path / d).exists()]
        if missing_dirs:
            return f"Diffusers model missing components: {missing_dirs}"

        patterns = ["*.safetensors", "*.bin", "*.pt", "*.ckpt"]
        weight_files = []
        for path in local_path.rglob("*"):
            if not path.is_file():
                continue
            for pattern in patterns:
                if fnmatch.fnmatch(path.name, pattern):
                    weight_files.append(path)
                    break
        if not weight_files:
            return "Diffusers model appears to be missing weight files"

        return None


def validate_model(model_path: str, model_type: ModelType, hardware: str = "auto") -> ValidationResult:
    """Convenience function for model validation."""
    return ModelValidator.validate_for_loading(model_path, model_type, hardware)
