import os
import subprocess
import logging
import time
from pathlib import Path
from typing import Optional, List
import requests
from tqdm import tqdm
from app.core.config import settings
from .model_resolver import ModelMetadata, ModelFramework
from .model_registry import model_registry
import shutil

logger = logging.getLogger(__name__)

class ModelDownloader:
    def __init__(self, hf_token: Optional[str] = None, cache_dir: Optional[str] = None):
        self.token = hf_token or settings.HF_API_TOKEN
        self.cache_dir = Path(cache_dir or settings.MODEL_CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._download_methods = {
            "hub_api": self._download_via_hub_api,
            "git_lfs": self._download_via_git_lfs,
            "http_mirror": self._download_via_http_mirror,
        }
    
    def download(self, metadata: ModelMetadata, method: Optional[str] = None) -> Path:
        """Download model using specified method or auto-select."""
        if method is None:
            methods = list(self._download_methods.items())
        else:
            if method not in self._download_methods:
                raise ValueError(f"Unknown download method: {method}")
            methods = [(method, self._download_methods[method])]
        
        repo_id = metadata.repo_id
        start_time = time.time()
        
        for method_name, download_method in methods:
            try:
                logger.info(f"Attempting {method_name} for {repo_id}")
                local_path = download_method(metadata)
                
                if self._validate_download(local_path, metadata):
                    duration = time.time() - start_time
                    logger.info(f"Successfully downloaded {repo_id} in {duration:.2f}s")
                    model_registry.log_download_attempt(repo_id, method_name, True, duration=duration)
                    return local_path
                else:
                    logger.warning(f"Validation failed for {repo_id} via {method_name}")
                    self._cleanup_partial_download(local_path)
            except Exception as e:
                logger.error(f"{method_name} failed for {repo_id}: {e}")
                model_registry.log_download_attempt(repo_id, method_name, False, error=str(e))
                if "local_path" in locals():
                    self._cleanup_partial_download(local_path)
                continue
        
        raise Exception(f"All download methods failed for {repo_id}")
    
    def _download_via_hub_api(self, metadata: ModelMetadata) -> Path:
        """Use official Hugging Face Hub API."""
        from huggingface_hub import snapshot_download
        
        local_path = self.cache_dir / metadata.repo_id.replace("/", "--")
        
        # Filter weights based on patterns
        snapshot_download(
            repo_id=metadata.repo_id,
            local_dir=str(local_path),
            token=self.token,
            # allow_patterns=metadata.weights_pattern + metadata.required_files, # patterns can be tricky
            ignore_patterns=["*.md", "*.txt", ".gitattributes"]
        )
        
        return local_path
    
    def _download_via_git_lfs(self, metadata: ModelMetadata) -> Path:
        """Clone repository with Git LFS."""
        local_path = self.cache_dir / metadata.repo_id.replace("/", "--")
        
        # Environment with token for private repos
        env = os.environ.copy()
        repo_url = f"https://huggingface.co/{metadata.repo_id}"
        if self.token:
            repo_url = f"https://user:{self.token}@huggingface.co/{metadata.repo_id}"
        
        if not local_path.exists():
            # Clone metadata first
            subprocess.run([
                "git", "clone",
                "--depth", "1",
                "-c", "lfs.fetchexclude=*",
                repo_url,
                str(local_path)
            ], check=True, env=env)
        
        # Pull LFS files
        subprocess.run([
            "git", "-C", str(local_path),
            "lfs", "pull"
        ], check=True, env=env)
        
        return local_path
    
    def _download_via_http_mirror(self, metadata: ModelMetadata) -> Path:
        """Download files directly via HTTP with mirror fallback."""
        mirrors = [
            "https://huggingface.co",
            "https://hf-mirror.com"
        ]
        
        local_path = self.cache_dir / metadata.repo_id.replace("/", "--")
        local_path.mkdir(parents=True, exist_ok=True)
        
        # If we have a list of files from resolver, we can iterate
        # For simplicity, if metadata.required_files is empty, this strategy might fail 
        # unless we get a full list of siblings during analysis
        files_to_download = list(dict.fromkeys(metadata.required_files + metadata.weights_files))
        if not files_to_download:
            raise Exception(f"No files to download for {metadata.repo_id} using http_mirror")
        
        for file_path in files_to_download:
            downloaded = False
            for mirror in mirrors:
                url = f"{mirror}/{metadata.repo_id}/resolve/main/{file_path}"
                headers = {}
                if self.token:
                    headers["Authorization"] = f"Bearer {self.token}"
                
                try:
                    response = requests.get(url, headers=headers, stream=True, timeout=settings.RECON_TIMEOUT_SECONDS)
                    if response.status_code == 200:
                        file_dest = local_path / file_path
                        file_dest.parent.mkdir(parents=True, exist_ok=True)
                        
                        total_size = int(response.headers.get('content-length', 0))
                        with open(file_dest, 'wb') as f, tqdm(total=total_size, unit='B', unit_scale=True, desc=file_path) as pbar:
                            for chunk in response.iter_content(chunk_size=8192):
                                f.write(chunk)
                                pbar.update(len(chunk))
                        
                        downloaded = True
                        break
                except Exception as e:
                    logger.warning(f"Mirror {mirror} failed for {file_path}: {e}")
            
            if not downloaded:
                raise Exception(f"Failed to download {file_path} from all mirrors")
        
        return local_path
    
    def _validate_download(self, local_path: Path, metadata: ModelMetadata) -> bool:
        """Validate downloaded model has all required files."""
        if not local_path.exists():
            return False
        
        required_files = list(dict.fromkeys(metadata.required_files + metadata.weights_files))

        for required_file in required_files:
            file_path = local_path / required_file
            if not file_path.exists():
                logger.error(f"Missing required file: {required_file} in {local_path}")
                return False
            
            if file_path.stat().st_size == 0:
                logger.error(f"Empty file: {required_file}")
                return False

        if metadata.framework == ModelFramework.DIFFUSERS:
            required_dirs = ["unet", "vae", "scheduler", "text_encoder"]
            missing_dirs = [d for d in required_dirs if not (local_path / d).exists()]
            if missing_dirs:
                logger.error(f"Missing required diffusers components: {missing_dirs} in {local_path}")
                return False
            if not metadata.weights_files:
                logger.error("No weight files detected for diffusers model")
                return False
        
        return True

    def _cleanup_partial_download(self, local_path: Path):
        if local_path and local_path.exists():
            try:
                shutil.rmtree(local_path)
            except Exception as e:
                logger.warning(f"Failed to clean up partial download at {local_path}: {e}")

model_downloader = ModelDownloader()
