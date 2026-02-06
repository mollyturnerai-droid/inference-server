import requests
from dataclasses import dataclass, asdict
import fnmatch
from enum import Enum
from typing import List, Optional, Dict, Any
from app.core.config import settings

class ModelFramework(Enum):
    TRANSFORMERS = "transformers"
    DIFFUSERS = "diffusers"
    ONNX = "onnx"
    CUSTOM = "custom"

@dataclass
class ModelMetadata:
    repo_id: str
    framework: ModelFramework
    required_files: List[str]
    optional_files: List[str]
    config_path: str
    weights_pattern: List[str]
    weights_files: List[str]
    size_gb: float

class ModelResolver:
    def __init__(self, hf_token: Optional[str] = None):
        self.token = hf_token or settings.HF_API_TOKEN
        self.api_base = "https://huggingface.co/api"
    
    def analyze_model(self, repo_id: str) -> ModelMetadata:
        """Determine model type and download requirements."""
        api_url = f"{self.api_base}/models/{repo_id}"
        headers = {}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
            
        response = requests.get(api_url, headers=headers, timeout=settings.RECON_TIMEOUT_SECONDS)
        
        if response.status_code != 200:
            raise ValueError(f"Cannot access model {repo_id}: {response.text}")
        
        model_info = response.json()
        
        # List files in repo
        files = [f["rfilename"] for f in model_info.get("siblings", [])]
        tags = model_info.get("tags", [])
        
        framework = self._infer_framework(files, tags)
        
        # Build required files list based on framework
        required_files = self._get_required_files(framework, files)
        weights_pattern = self._get_weights_pattern(framework)
        weights_files = self._find_weights_files(files, weights_pattern)
        
        return ModelMetadata(
            repo_id=repo_id,
            framework=framework,
            required_files=required_files,
            optional_files=self._get_optional_files(framework, files),
            config_path=self._find_config_path(files, framework),
            weights_pattern=weights_pattern,
            weights_files=weights_files,
            size_gb=self._estimate_size(model_info)
        )
    
    def _infer_framework(self, files: List[str], tags: List[str]) -> ModelFramework:
        """Infer framework from file patterns and model tags."""
        if "diffusers" in tags or any("model_index.json" in f for f in files):
            return ModelFramework.DIFFUSERS
        elif "transformers" in tags or "pytorch" in tags or any("config.json" in f for f in files):
            return ModelFramework.TRANSFORMERS
        elif any(f.endswith(".onnx") for f in files):
            return ModelFramework.ONNX
        else:
            return ModelFramework.CUSTOM

    def _get_required_files(self, framework: ModelFramework, files: List[str]) -> List[str]:
        if framework == ModelFramework.DIFFUSERS:
            # Common diffusers files
            critical = ["model_index.json", "unet/config.json", "vae/config.json", "scheduler/scheduler_config.json"]
            return [f for f in critical if any(f in existing for existing in files)]
        elif framework == ModelFramework.TRANSFORMERS:
            return ["config.json"]
        return []

    def _get_optional_files(self, framework: ModelFramework, files: List[str]) -> List[str]:
        return [f for f in files if f.endswith(".md") or f.endswith(".txt")]

    def _find_config_path(self, files: List[str], framework: ModelFramework) -> str:
        if framework == ModelFramework.DIFFUSERS:
            return "model_index.json" if "model_index.json" in files else ""
        return "config.json" if "config.json" in files else ""

    def _get_weights_pattern(self, framework: ModelFramework) -> List[str]:
        patterns = ["*.safetensors", "*.bin", "*.pt", "*.ckpt"]
        if framework == ModelFramework.ONNX:
            patterns.append("*.onnx")
        return patterns

    def _find_weights_files(self, files: List[str], patterns: List[str]) -> List[str]:
        matches: List[str] = []
        for file_name in files:
            for pattern in patterns:
                if fnmatch.fnmatch(file_name, pattern):
                    matches.append(file_name)
                    break
        return matches

    def _estimate_size(self, model_info: Dict[str, Any]) -> float:
        """Estimate size in GB from siblings listing."""
        # Note: HF API sometimes doesn't provide size for all files in the model info
        # We can sum up the 'size' field if available
        total_bytes = 0
        for sibling in model_info.get("siblings", []):
            total_bytes += sibling.get("size", 0)
        
        # Fallback: if total_bytes is 0, we'll return a default or use a different API call
        return total_bytes / (1024**3) if total_bytes > 0 else 0.0

model_resolver = ModelResolver()
