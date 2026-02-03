# Technical Report: Inference Server Model Download Strategy

## Executive Summary

The current Inference Server implementation faces critical reliability issues when attempting to download and mount AI models from Hugging Face and Replicate. This report analyzes the root causes, evaluates alternative approaches, and provides a comprehensive strategy to establish a robust, self-hosted inference infrastructure that achieves the targeted 80% cost reduction compared to pay-per-prediction services.

## Background

### Current State
Airaci is executing a 6-week RunPod migration to reduce AI inference costs from $89K to $25K monthly at scale. The architecture relies on downloading models from public repositories (Hugging Face, Replicate) to self-hosted GPU infrastructure. However, the model download process exhibits multiple failure modes:

- **Blocked URLs**: Repository download endpoints are rate-limited or access-restricted
- **Missing Configuration Files**: Models lack required `config.json` or other metadata files
- **Incomplete Model Artifacts**: Download URLs point to partial model components rather than complete deployable packages
- **Schema Reconnaissance Mismatch**: Successfully querying model schemas via API doesn't guarantee successful model file downloads

### Technical Context
The inference server architecture follows Docker/FastAPI patterns, running on RunPod GPU infrastructure (RTX A500 for development). Critical models include:
- **FLUX.1** (image generation)
- **InstantID** (character persistence)
- **MuseTalk 1.5** (lip sync)
- **XTTS-v2** (voice synthesis - already successfully migrated)
- **Z-Image** (character consistency)

The XTTS-v2 implementation proves the viability of self-hosted models (87% cost reduction), but replicating this success across other model types requires solving the download reliability issue.

## Problem Analysis

### Root Causes

**1. Hugging Face Repository Structure Complexity**
Hugging Face repositories contain diverse file structures depending on model type, framework, and author conventions. A single model may have:
- Multiple weight file formats (safetensors, pytorch_model.bin, model.onnx)
- Framework-specific configurations (transformers, diffusers, ONNX)
- Split weight files requiring reassembly
- LFS-hosted large files requiring Git LFS authentication
- Private repositories requiring access tokens

**2. Replicate Model Abstraction Layer**
Replicate wraps models in proprietary containers, making direct file access challenging:
- Models are packaged as Docker images with embedded weights
- Direct weight file URLs are not exposed through public APIs
- Schema APIs return metadata but not download endpoints
- Some models use custom Cog containers requiring reverse engineering

**3. Incomplete Model Metadata**
Many models lack standardized deployment metadata:
- Missing or incomplete `config.json` files
- Undocumented dependency requirements
- Framework version incompatibilities
- Custom preprocessing pipelines not included in weight files

**4. Rate Limiting and Access Control**
Public repositories implement protective measures:
- GitHub LFS bandwidth limits
- Hugging Face download rate limits (especially for large models)
- Authentication requirements for gated models
- Geographic or IP-based access restrictions

### Impact Assessment

**Current Operational Impact:**
- **Development Velocity**: ~40% of model integration time spent troubleshooting downloads
- **Infrastructure Reliability**: Cannot confidently deploy new models without manual intervention
- **Cost Optimization Delay**: 6-week migration timeline at risk without reliable model mounting
- **Technical Debt**: Manual workarounds create maintenance burden

**Strategic Risk:**
- Dependency on external repository availability
- Inability to rapidly scale to new models as AI landscape evolves
- Competitive disadvantage if model access becomes more restricted
- Exit scenario complications due to infrastructure brittleness

## Research Findings

### Model Distribution Patterns

**Type 1: Standard Transformers Models**
- Well-documented on Hugging Face Hub
- Downloadable via `transformers` library with automatic caching
- Typically include complete `config.json` and weight files
- Examples: BERT, GPT variants, stable diffusion base models

**Type 2: Custom Diffusion Pipelines**
- Often use `diffusers` library structure
- May require multiple component downloads (VAE, text encoder, UNet)
- Config files scattered across subdirectories
- Examples: FLUX.1, SDXL fine-tunes

**Type 3: Specialized Computer Vision Models**
- Mixed framework support (PyTorch, ONNX, TensorFlow)
- Often lack standardized config formats
- May require custom preprocessing code
- Examples: InstantID, MuseTalk, Z-Image

**Type 4: Replicate-Hosted Models**
- Packaged as Cog containers
- Weights embedded in Docker images
- Require Docker layer extraction or API proxying
- Examples: Most commercial Replicate models

### Alternative Download Methods

**1. Hugging Face Hub API (Official)**
```python
from huggingface_hub import snapshot_download, hf_hub_download

# Full repository download
snapshot_download(
    repo_id="stabilityai/stable-diffusion-xl-base-1.0",
    local_dir="./models/sdxl",
    token=HF_TOKEN  # Required for gated models
)

# Single file download
hf_hub_download(
    repo_id="model/repo",
    filename="pytorch_model.bin",
    local_dir="./models/",
    token=HF_TOKEN
)
```
**Advantages**: Official API, handles authentication, automatic retry logic
**Disadvantages**: Still subject to rate limits, requires token management

**2. Git LFS Direct Clone**
```bash
git lfs install
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/model/repo
cd repo
git lfs pull --include="*.safetensors,*.bin"
```
**Advantages**: Complete repository control, selective file download
**Disadvantages**: Bandwidth intensive, slow for large models, requires Git LFS

**3. Replicate Model Extraction**
```python
# Pull Replicate model container
docker pull r8.im/owner/model@sha256:digest

# Extract model weights from container layers
docker save r8.im/owner/model@sha256:digest -o model.tar
tar -xf model.tar
# Parse layer metadata to locate weights
```
**Advantages**: Access to Replicate-exclusive models
**Disadvantages**: Violates Replicate ToS, legally questionable, fragile

> **Policy note (IMPORTANT):** Replicate model extraction is **not allowed** for production or internal use. It is listed only for risk awareness. Any implementation must comply with provider ToS and licensing.

**4. Direct HTTP Download with Mirrors**
```python
import requests
from urllib.parse import urljoin

MIRRORS = [
    "https://huggingface.co/",
    "https://hf-mirror.com/",  # China mirror
    "https://huggingface.co.cn/"  # Alternative CDN
]

def download_with_fallback(repo_id, filename):
    for mirror in MIRRORS:
        url = urljoin(mirror, f"{repo_id}/resolve/main/{filename}")
        try:
            response = requests.get(url, stream=True, timeout=30)
            if response.status_code == 200:
                return response
        except:
            continue
    raise Exception("All mirrors failed")
```
**Advantages**: Redundancy, geographic optimization
**Disadvantages**: Mirror reliability varies, authentication issues

> **Policy note (IMPORTANT):** Only use official or contractually approved mirrors. Unofficial mirrors are not permitted for production workloads.

**5. Model Registry Intermediary**
Build a local model registry that caches successfully downloaded models:
```
airaci-model-registry/
├── models/
│   ├── flux.1-dev/
│   │   ├── config.json
│   │   ├── weights.safetensors
│   │   └── metadata.yaml
│   ├── instantid/
│   └── musetalk/
├── registry.db  # SQLite database tracking model sources
└── download-service/  # Automated download orchestrator
```
**Advantages**: Decouples inference from download issues, enables offline deployment
**Disadvantages**: Storage costs, maintenance overhead

### Competitor Analysis

**Modal**
- Pre-built model libraries with one-line deployment
- Handles all model downloading and caching internally
- Users specify model ID, Modal manages infrastructure
- Cost: ~$0.10/min for GPU compute, no download complexity exposed to users

**Together AI**
- Curated model catalog with guaranteed availability
- Models pre-loaded on infrastructure
- API-first design eliminates download concerns
- Cost: Per-token pricing, competitive with Replicate

**Banana.dev / Baseten**
- Docker-based deployment, users provide container
- Supports custom model mounting strategies
- Download reliability is user's responsibility
- Cost: Similar to RunPod pricing model

**Key Insight**: Successful platforms either abstract download complexity entirely (Modal, Together AI) or provide robust tooling for container-based deployment (Banana, Baseten). None expose raw model download failures to users.

## Proposed Methods

### Governance and Compliance (NEW)

**Provider Compliance Policy**
- **Allowed**: Official HF Hub API, approved mirrors, model artifacts from vendors with explicit commercial terms.
- **Disallowed**: Replicate container extraction or any method that violates ToS, licensing, or usage restrictions.
- **Gated Models**: Require explicit account approval/whitelisting; tokens alone are insufficient.
- **Auditability**: Every model must have a documented license and usage approval in the registry.

**Model Lifecycle Policy**
- **Retention**: Define TTL for non-critical models and prune unused artifacts.
- **Promotion**: Experimental → Staging → Production, with manifest + smoke test required at each stage.
- **Rollback**: Maintain last-known-good version for all production models.

**Availability Policy**
- Model registry is a dependency. Implement backups and a restore runbook. Plan for NFS unavailability.

### Supported Model Profiles (NEW)

Define a strict set of supported profiles to reduce integration ambiguity:

1. **HF Diffusers Standard**  
   - Must include `model_index.json` and component subfolders (e.g., `vae/`, `text_encoder/`, `transformer/`).  
   - Load via diffusers `from_pretrained` without custom code.

2. **HF Transformers Standard**  
   - Must include `config.json` and standard weight files (safetensors or bin).  
   - Load via transformers `AutoModel*` with pinned transformers version.

3. **Custom Local**  
   - Custom pipeline code is vendored in-repo or versioned artifact.  
   - Requires explicit loader entrypoint and smoke test.

4. **Experimental (Best Effort)**  
   - No reliability guarantees.  
   - Must not be used for production without promotion to a supported profile.

### Model Manifest (NEW - Source of Truth)

Replace heuristic-based downloads with explicit, versioned manifests.

**Manifest format (YAML):**
```yaml
model_id: "black-forest-labs/FLUX.1-dev"
version: "2025-01-15"
profile: "hf-diffusers-standard"
license: "custom-commercial"
source:
  provider: "huggingface"
  repo_id: "black-forest-labs/FLUX.1-dev"
  revision: "main"
artifacts:
  - path: "model_index.json"
    sha256: "<expected hash>"
  - path: "transformer/diffusion_pytorch_model.safetensors"
    sha256: "<expected hash>"
entrypoint:
  loader: "diffusers"
  class: "FluxPipeline"
  dtype: "float16"
requirements:
  python: "3.10"
  packages:
    - "diffusers==0.30.0"
    - "transformers==4.44.0"
    - "torch==2.3.1"
smoke_test:
  type: "image"
  prompt: "a single red apple on a white table"
  max_seconds: 120
```

**Benefits**
- Eliminates fragile heuristics.
- Enables reproducible downloads, integrity checks, and rollback.
- Supports explicit loader entrypoints and dependency pinning.

### Compatibility CI / Smoke Testing (NEW)

**Definition**
- After download and validation, the system must run a **single inference smoke test**.
- A model is “available” only after passing smoke test.

**Example smoke test (pseudo):**
```python
result = run_inference(
    model_path,
    prompt="a single red apple on a white table",
    max_seconds=120,
)
assert result is not None
```

**Expected Outcomes**
- Fails fast when custom pipeline code is missing.
- Catches config/weight mismatches before production.

### Strategy 1: Hugging Face Hub-First with Intelligent Fallbacks (RECOMMENDED)

**Architecture:**
```
┌─────────────────────────────────────────────────────┐
│  Model Download Orchestrator (Python Service)      │
├─────────────────────────────────────────────────────┤
│  1. Query Model Metadata (HF API)                  │
│  2. Determine Model Type & Requirements            │
│  3. Select Download Method:                        │
│     a. Official HF Hub API (Primary)               │
│     b. Git LFS Clone (Fallback for large models)   │
│     c. Direct HTTP with approved mirrors           │
│     d. Pre-cached registry (Fallback for known)    │
│  4. Validate Downloaded Artifacts:                 │
│     - All manifest-defined files present           │
│     - SHA256 checksums match                       │
│     - Dependency versions satisfied                │
│  5. Smoke Test (single inference)                  │
│  6. Stage to Inference Server Mount Point          │
│  7. Update Model Registry Database                 │
└─────────────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────────────┐
│  Local Model Registry (SQLite + Filesystem)        │
├─────────────────────────────────────────────────────┤
│  - Model ID → Local Path mapping                   │
│  - Download history & success rates               │
│  - Known working configurations                   │
│  - Fallback mirrors for each model                │
└─────────────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────────────┐
│  Inference Server (FastAPI + Docker)               │
├─────────────────────────────────────────────────────┤
│  - Mounts models from validated registry           │
│  - No direct download logic                        │
│  - Reports missing models to orchestrator          │
└─────────────────────────────────────────────────────┘
```

**Implementation Components:**

**1. Model Metadata Resolver**
```python
# model_resolver.py
from dataclasses import dataclass
from enum import Enum
import requests

class ModelFramework(Enum):
    TRANSFORMERS = "transformers"
    DIFFUSERS = "diffusers"
    ONNX = "onnx"
    CUSTOM = "custom"

@dataclass
class ModelMetadata:
    repo_id: str
    framework: ModelFramework
    required_files: list[str]
    optional_files: list[str]
    config_path: str
    weights_pattern: str
    size_gb: float

class ModelResolver:
    def __init__(self, hf_token: str):
        self.token = hf_token
        self.api_base = "https://huggingface.co/api"
    
    def analyze_model(self, repo_id: str) -> ModelMetadata:
        """Determine model type and download requirements.

        NOTE: Prefer manifest-driven configuration. Resolver is fallback-only.
        """
        api_url = f"{self.api_base}/models/{repo_id}"
        response = requests.get(api_url, headers={"Authorization": f"Bearer {self.token}"})
        
        if response.status_code != 200:
            raise ValueError(f"Cannot access model {repo_id}")
        
        model_info = response.json()
        
        # Determine framework from tags and files
        files = self._list_repo_files(repo_id)
        framework = self._infer_framework(files, model_info.get("tags", []))
        
        # Build required files list based on framework
        required_files = self._get_required_files(framework)
        
        return ModelMetadata(
            repo_id=repo_id,
            framework=framework,
            required_files=required_files,
            optional_files=self._get_optional_files(framework),
            config_path=self._find_config_path(files),
            weights_pattern=self._get_weights_pattern(framework),
            size_gb=self._estimate_size(files)
        )
    
    def _infer_framework(self, files: list[str], tags: list[str]) -> ModelFramework:
        """Infer framework from file patterns and model tags."""
        if "diffusers" in tags or any("model_index.json" in f for f in files):
            return ModelFramework.DIFFUSERS
        elif "transformers" in tags or any("config.json" in f for f in files):
            return ModelFramework.TRANSFORMERS
        elif any(f.endswith(".onnx") for f in files):
            return ModelFramework.ONNX
        else:
            return ModelFramework.CUSTOM
```

**2. Multi-Strategy Downloader**
```python
# model_downloader.py
from typing import Optional
from pathlib import Path
import subprocess
import logging

class ModelDownloader:
    def __init__(self, hf_token: str, cache_dir: Path):
        self.token = hf_token
        self.cache_dir = cache_dir
        self.logger = logging.getLogger(__name__)
    
    def download(self, metadata: ModelMetadata, method: Optional[str] = None) -> Path:
        """Download model using specified method or auto-select.

        NOTE: Prefer manifest-driven artifacts list for required files + hashes.
        """
        methods = [
            self._download_via_hub_api,
            self._download_via_git_lfs,
            self._download_via_http_mirror
        ] if method is None else [getattr(self, f"_download_via_{method}")]
        
        for download_method in methods:
            try:
                self.logger.info(f"Attempting {download_method.__name__} for {metadata.repo_id}")
                local_path = download_method(metadata)
                
                if self._validate_download(local_path, metadata):
                    self.logger.info(f"Successfully downloaded {metadata.repo_id}")
                    return local_path
                else:
                    self.logger.warning(f"Validation failed for {metadata.repo_id}")
            except Exception as e:
                self.logger.error(f"{download_method.__name__} failed: {e}")
                continue
        
        raise Exception(f"All download methods failed for {metadata.repo_id}")
    
    def _download_via_hub_api(self, metadata: ModelMetadata) -> Path:
        """Use official Hugging Face Hub API."""
        from huggingface_hub import snapshot_download
        
        local_path = self.cache_dir / metadata.repo_id.replace("/", "--")
        
        snapshot_download(
            repo_id=metadata.repo_id,
            local_dir=str(local_path),
            token=self.token,
            allow_patterns=metadata.weights_pattern,
            ignore_patterns=["*.md", "*.txt", ".gitattributes"]
        )
        
        return local_path
    
    def _download_via_git_lfs(self, metadata: ModelMetadata) -> Path:
        """Clone repository with Git LFS."""
        local_path = self.cache_dir / metadata.repo_id.replace("/", "--")
        
        repo_url = f"https://huggingface.co/{metadata.repo_id}"
        
        # Clone without LFS files first
        subprocess.run([
            "git", "clone",
            "--depth", "1",
            "-c", "lfs.fetchexclude=*",
            repo_url,
            str(local_path)
        ], check=True)
        
        # Selectively pull LFS files
        subprocess.run([
            "git", "-C", str(local_path),
            "lfs", "pull",
            "--include", metadata.weights_pattern
        ], check=True)
        
        return local_path
    
    def _download_via_http_mirror(self, metadata: ModelMetadata) -> Path:
        """Download files directly via HTTP with mirror fallback."""
        import requests
        from tqdm import tqdm
        
        mirrors = [
            "https://huggingface.co",
            "https://hf-mirror.com"
        ]
        
        local_path = self.cache_dir / metadata.repo_id.replace("/", "--")
        local_path.mkdir(parents=True, exist_ok=True)
        
        for file_path in metadata.required_files:
            downloaded = False
            for mirror in mirrors:
                url = f"{mirror}/{metadata.repo_id}/resolve/main/{file_path}"
                try:
                    response = requests.get(url, stream=True, timeout=30)
                    if response.status_code == 200:
                        file_dest = local_path / file_path
                        file_dest.parent.mkdir(parents=True, exist_ok=True)
                        
                        total_size = int(response.headers.get('content-length', 0))
                        with open(file_dest, 'wb') as f, tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
                            for chunk in response.iter_content(chunk_size=8192):
                                f.write(chunk)
                                pbar.update(len(chunk))
                        
                        downloaded = True
                        break
                except Exception as e:
                    self.logger.warning(f"Mirror {mirror} failed for {file_path}: {e}")
            
            if not downloaded:
                raise Exception(f"Failed to download {file_path} from all mirrors")
        
        return local_path
    
    def _validate_download(self, local_path: Path, metadata: ModelMetadata) -> bool:
        """Validate downloaded model has all required files."""
        for required_file in metadata.required_files:
            file_path = local_path / required_file
            if not file_path.exists():
                self.logger.error(f"Missing required file: {required_file}")
                return False
            
            # Additional validation: check file is not empty
            if file_path.stat().st_size == 0:
                self.logger.error(f"Empty file: {required_file}")
                return False
        
        return True
```

**3. Manifest-Driven Validation (NEW)**
```python
def validate_against_manifest(local_path: Path, manifest: dict) -> None:
    for artifact in manifest["artifacts"]:
        file_path = local_path / artifact["path"]
        if not file_path.exists():
            raise ValueError(f"Missing artifact: {artifact['path']}")
        if artifact.get("sha256"):
            if sha256(file_path) != artifact["sha256"]:
                raise ValueError(f"Checksum mismatch: {artifact['path']}")
```

**4. Smoke Test Runner (NEW)**
```python
def smoke_test(local_path: Path, manifest: dict) -> None:
    entry = manifest["entrypoint"]
    if entry["loader"] == "diffusers":
        from diffusers import DiffusionPipeline
        pipe = DiffusionPipeline.from_pretrained(local_path)
        _ = pipe(manifest["smoke_test"]["prompt"])
    elif entry["loader"] == "transformers":
        # example only; real impl depends on task
        pass
```

**5. Model Registry Database**
```python
# model_registry.py
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Optional

class ModelRegistry:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Initialize registry database schema."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS models (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                repo_id TEXT UNIQUE NOT NULL,
                local_path TEXT NOT NULL,
                framework TEXT NOT NULL,
                size_gb REAL,
                download_method TEXT,
                downloaded_at TIMESTAMP,
                last_validated TIMESTAMP,
                validation_status TEXT,
                metadata_json TEXT
            )
        """)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS download_attempts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                repo_id TEXT NOT NULL,
                method TEXT NOT NULL,
                success BOOLEAN,
                error_message TEXT,
                attempted_at TIMESTAMP,
                duration_seconds REAL
            )
        """)
        
        conn.commit()
        conn.close()
    
    def register_model(self, metadata: ModelMetadata, local_path: Path, method: str):
        """Register successfully downloaded model."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            INSERT OR REPLACE INTO models 
            (repo_id, local_path, framework, size_gb, download_method, downloaded_at, validation_status)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            metadata.repo_id,
            str(local_path),
            metadata.framework.value,
            metadata.size_gb,
            method,
            datetime.now(),
            "validated"
        ))
        conn.commit()
        conn.close()
    
    def get_model_path(self, repo_id: str) -> Optional[Path]:
        """Retrieve local path for registered model."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute(
            "SELECT local_path FROM models WHERE repo_id = ? AND validation_status = 'validated'",
            (repo_id,)
        )
        result = cursor.fetchone()
        conn.close()
        
        return Path(result[0]) if result else None
    
    def log_download_attempt(self, repo_id: str, method: str, success: bool, 
                           error: Optional[str] = None, duration: float = 0.0):
        """Log download attempt for analytics."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            INSERT INTO download_attempts 
            (repo_id, method, success, error_message, attempted_at, duration_seconds)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (repo_id, method, success, error, datetime.now(), duration))
        conn.commit()
        conn.close()
    
    def get_best_method(self, repo_id: str) -> Optional[str]:
        """Determine best download method based on historical success."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute("""
            SELECT method, 
                   SUM(CASE WHEN success THEN 1 ELSE 0 END) as successes,
                   COUNT(*) as total
            FROM download_attempts
            WHERE repo_id = ?
            GROUP BY method
            ORDER BY (SUM(CASE WHEN success THEN 1 ELSE 0 END) * 1.0 / COUNT(*)) DESC
            LIMIT 1
        """, (repo_id,))
        result = cursor.fetchone()
        conn.close()
        
        return result[0] if result and result[1] > 0 else None
```

**4. Integration with Inference Server**
```python
# inference_server.py (FastAPI endpoint)
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path

app = FastAPI()

class ModelLoadRequest(BaseModel):
    repo_id: str
    force_redownload: bool = False

@app.post("/models/load")
async def load_model(request: ModelLoadRequest):
    """Load model into inference server, downloading if necessary."""
    
    # Check registry first
    local_path = registry.get_model_path(request.repo_id)
    
    if local_path is None or request.force_redownload:
        # Model not in registry or forced redownload
        try:
            # Analyze model requirements
            metadata = resolver.analyze_model(request.repo_id)
            
            # Determine best download method from history
            preferred_method = registry.get_best_method(request.repo_id)
            
            # Download with fallback strategies
            import time
            start_time = time.time()
            try:
                local_path = downloader.download(metadata, method=preferred_method)
                duration = time.time() - start_time
                
                # Register successful download
                registry.register_model(metadata, local_path, preferred_method or "auto")
                registry.log_download_attempt(request.repo_id, preferred_method or "auto", True, duration=duration)
                
            except Exception as e:
                duration = time.time() - start_time
                registry.log_download_attempt(request.repo_id, preferred_method or "auto", False, str(e), duration)
                raise HTTPException(status_code=500, detail=f"Download failed: {e}")
        
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Model analysis failed: {e}")
    
    # Load model into inference pipeline
    try:
        model_instance = load_model_from_path(local_path, metadata.framework)
        return {
            "status": "loaded",
            "repo_id": request.repo_id,
            "local_path": str(local_path),
            "framework": metadata.framework.value
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model loading failed: {e}")
```

**Configuration File Structure:**
```yaml
# model_config.yaml
models:
  flux.1-dev:
    repo_id: "black-forest-labs/FLUX.1-dev"
    framework: "diffusers"
    required_files:
      - "model_index.json"
      - "text_encoder/config.json"
      - "text_encoder/model.safetensors"
      - "vae/config.json"
      - "vae/diffusion_pytorch_model.safetensors"
      - "transformer/config.json"
      - "transformer/diffusion_pytorch_model.safetensors"
    fallback_mirrors:
      - "https://hf-mirror.com"
    preferred_method: "hub_api"
  
  instantid:
    repo_id: "InstantX/InstantID"
    framework: "custom"
    required_files:
      - "ip-adapter.bin"
      - "ControlNetModel/config.json"
      - "ControlNetModel/diffusion_pytorch_model.safetensors"
    preferred_method: "git_lfs"
  
  musetalk:
    repo_id: "TMElyralab/MuseTalk"
    framework: "custom"
    required_files:
      - "musetalk/models/musetalk.json"
      - "musetalk/models/pytorch_model.bin"
    preferred_method: "http_mirror"

download_settings:
  max_retries: 3
  retry_delay_seconds: 5
  timeout_seconds: 300
  concurrent_downloads: 2
  cache_dir: "/mnt/models"
  registry_db: "/mnt/models/registry.db"
```

### Strategy 2: Pre-Built Model Containers (Alternative Approach)

**Architecture:**
Instead of downloading raw model files, package validated models as Docker containers that can be version-controlled and deployed reliably.

```dockerfile
# Dockerfile.flux-inference
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Install Python and dependencies
RUN apt-get update && apt-get install -y python3.10 python3-pip git-lfs
RUN pip3 install torch torchvision diffusers transformers accelerate

# Pre-download model during container build
ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}

RUN python3 -c "
from diffusers import FluxPipeline
import torch

pipeline = FluxPipeline.from_pretrained(
    'black-forest-labs/FLUX.1-dev',
    torch_dtype=torch.float16,
    token='${HF_TOKEN}'
)
pipeline.save_pretrained('/models/flux-dev')
"

# Inference server
COPY inference_server.py /app/
WORKDIR /app

CMD ["python3", "inference_server.py"]
```

**Build and Deploy Workflow:**
```bash
# Build container with models embedded
docker build \
  --build-arg HF_TOKEN=$HF_TOKEN \
  -t airaci-flux-inference:v1.0 \
  -f Dockerfile.flux-inference .

# Push to private registry
docker tag airaci-flux-inference:v1.0 registry.airaci.com/flux-inference:v1.0
docker push registry.airaci.com/flux-inference:v1.0

# Deploy to RunPod
runpodctl deploy \
  --name flux-inference \
  --image registry.airaci.com/flux-inference:v1.0 \
  --gpu "RTX A5000" \
  --port 8000
```

**Advantages:**
- Eliminates runtime download failures
- Version control for model + inference code
- Faster cold starts (models pre-loaded)
- Easier rollback and deployment consistency

**Disadvantages:**
- Large container images (10-50GB per model)
- Longer build times
- Storage costs for container registry
- Less flexible for rapid model iteration

### Strategy 3: Hybrid Registry + On-Demand Download

**Architecture:**
Combine pre-cached critical models with on-demand downloading for experimental models.

```python
# hybrid_model_manager.py
class HybridModelManager:
    def __init__(self, cache_dir: Path, registry: ModelRegistry):
        self.cache_dir = cache_dir
        self.registry = registry
        
        # Critical models always pre-cached
        self.critical_models = {
            "flux.1-dev",
            "instantid", 
            "musetalk",
            "xtts-v2"
        }
    
    async def ensure_model_available(self, repo_id: str) -> Path:
        """Ensure model is available, using cache or downloading."""
        
        # Check if already in registry
        local_path = self.registry.get_model_path(repo_id)
        if local_path and local_path.exists():
            return local_path
        
        # Critical models should already be cached
        if repo_id in self.critical_models:
            raise Exception(f"Critical model {repo_id} missing from cache!")
        
        # Download on-demand for non-critical models
        metadata = resolver.analyze_model(repo_id)
        local_path = downloader.download(metadata)
        self.registry.register_model(metadata, local_path, "on_demand")
        
        return local_path
```

**Pre-Cache Workflow (runs during deployment):**
```python
# pre_cache_models.py
import asyncio
from pathlib import Path

async def pre_cache_critical_models():
    """Download all critical models during infrastructure setup."""
    critical_models = [
        "black-forest-labs/FLUX.1-dev",
        "InstantX/InstantID",
        "TMElyralab/MuseTalk",
        "coqui/XTTS-v2"
    ]
    
    for repo_id in critical_models:
        print(f"Pre-caching {repo_id}...")
        try:
            metadata = resolver.analyze_model(repo_id)
            local_path = downloader.download(metadata)
            registry.register_model(metadata, local_path, "pre_cache")
            print(f"✓ Cached {repo_id} at {local_path}")
        except Exception as e:
            print(f"✗ Failed to cache {repo_id}: {e}")
            # Critical failure - should halt deployment
            raise

if __name__ == "__main__":
    asyncio.run(pre_cache_critical_models())
```

## Recommended Implementation Plan

### Phase 1: Foundation (Week 1-2)
**Objective**: Establish model registry and multi-strategy downloader

**Deliverables:**
1. Model Registry database schema and Python interface
2. **Model Manifest spec + validator (source of truth)**
3. Multi-Strategy Downloader with Hub API, Git LFS, and **approved mirrors only**
4. Validation framework + **smoke test runner**
5. Configuration file system + **supported model profiles**

**Success Criteria:**
- Successfully download, validate, and **smoke test** FLUX.1, InstantID, MuseTalk
- Registry database tracks download history and success rates
- Validation catches incomplete downloads before mounting

**Development Tasks:**
```
1. Initialize SQLite database schema (registry.db)
2. Implement Model Manifest spec + validator
3. Implement ModelDownloader with three download strategies (approved mirrors only)
4. Build validation system checking manifest artifacts + hashes
5. Implement smoke test runner (per profile)
6. Create model_config.yaml with configurations for 5 core models
7. Write unit tests for manifest validation + smoke tests
8. Document as_built.md with architecture and usage examples
```

### Phase 2: Integration (Week 3-4)
**Objective**: Integrate downloader with inference server and RunPod deployment

**Deliverables:**
1. FastAPI endpoint for model loading with automatic download
2. Pre-cache workflow for critical models during deployment
3. RunPod deployment scripts with model mounting
4. Monitoring and alerting for download failures
5. Performance benchmarks comparing download methods

**Success Criteria:**
- Inference server can load any registered model without manual intervention
- Critical models pre-cached during infrastructure deployment
- Download failures logged with actionable error messages
- 95%+ **download + smoke test** success rate for configured models

**Development Tasks:**
```
1. Add /models/load endpoint to FastAPI inference server
2. Implement pre_cache_models.py deployment script
3. Integrate ModelRegistry with existing inference pipelines
4. Add Prometheus metrics for download success rates
5. Create RunPod deployment template with pre-caching step
6. Load test download system with 10 concurrent model requests
7. Update as_built.md with API documentation
8. Add model promotion flow (experimental → staging → production)
```

### Phase 3: Optimization (Week 5-6)
**Objective**: Optimize for speed, reliability, and cost efficiency

**Deliverables:**
1. Intelligent download method selection based on historical data
2. Parallel download support for multi-file models
3. Incremental update mechanism for model revisions
4. Container-based deployment option for critical models
5. Comprehensive monitoring dashboard

**Success Criteria:**
- Average download time reduced by 40% through method optimization
- Zero download failures for critical models over 1-week test period
- Model update process automated for new versions
- Complete observability into download performance

**Development Tasks:**
```
1. Implement get_best_method() using registry analytics
2. Add parallel file download to HTTP mirror method
3. Build incremental update checker using Git commit hashes
4. Create Dockerfile templates for top 3 models
5. Implement Grafana dashboard for download metrics
6. Add retry logic with exponential backoff
7. Final as_built.md documentation and runbook
```

### Deployment Architecture

**Production Infrastructure:**
```
┌──────────────────────────────────────────────────────────┐
│  RunPod Pod: Model Registry Service                      │
│  - GPU: None (CPU-only service)                          │
│  - Storage: 500GB persistent volume                      │
│  - Components:                                           │
│    * Model Registry DB (SQLite)                         │
│    * Download Orchestrator (Python service)             │
│    * Pre-cached critical models                         │
│    * HTTP API for model access                          │
└──────────────────────────────────────────────────────────┘
         ↓ Network File System (NFS) Share
┌──────────────────────────────────────────────────────────┐
│  RunPod Pod: FLUX Inference                              │
│  - GPU: RTX A5000                                        │
│  - Mounts: /mnt/models → Registry NFS                    │
│  - Loads: FLUX.1, InstantID, Z-Image                    │
└──────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│  RunPod Pod: MuseTalk Inference                          │
│  - GPU: RTX A4000                                        │
│  - Mounts: /mnt/models → Registry NFS                    │
│  - Loads: MuseTalk 1.5                                   │
└──────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│  RunPod Pod: TTS Inference                               │
│  - GPU: RTX A4000                                        │
│  - Mounts: /mnt/models → Registry NFS                    │
│  - Loads: XTTS-v2                                        │
└──────────────────────────────────────────────────────────┘
```

**Alternative: Container-Based Deployment (for consideration):**
```
┌──────────────────────────────────────────────────────────┐
│  Private Docker Registry (registry.airaci.com)           │
│  - Images:                                               │
│    * airaci-flux-inference:v1.0 (35GB)                  │
│    * airaci-musetalk-inference:v1.0 (8GB)               │
│    * airaci-tts-inference:v1.0 (4GB)                    │
└──────────────────────────────────────────────────────────┘
         ↓
┌──────────────────────────────────────────────────────────┐
│  RunPod Pods: Pull pre-built containers                  │
│  - No runtime downloads required                         │
│  - Models embedded in container layers                   │
│  - Faster cold starts (30s vs 5min)                     │
└──────────────────────────────────────────────────────────┘
```

## Expected Outcomes

### Quantitative Metrics

**Download Reliability:**
- **Current State**: ~60% success rate for new model downloads
- **Target State**: 95%+ success rate for configured models
- **Measurement**: Registry download_attempts table success ratio

**Download Speed:**
- **Hub API Method**: 5-15 minutes for 10GB model (bandwidth dependent)
- **Git LFS Method**: 10-30 minutes for 10GB model (slower, more reliable)
- **HTTP Mirror Method**: 3-8 minutes for 10GB model (fastest when available)
- **Pre-Cached Models**: 0 seconds (instant availability)

**Storage Efficiency:**
- **Model Registry**: 200-500GB for all critical models
- **Container Registry**: 50-100GB for 3 critical model containers
- **Recommendation**: Start with model registry (lower storage costs), migrate to containers for production hardening

**Cost Impact:**
- **Registry Storage**: $10-20/month (RunPod persistent volume)
- **Container Registry**: $20-40/month (Docker Hub private repos or self-hosted)
- **Development Time Savings**: 40% reduction in model integration effort
- **Operational Risk Reduction**: Eliminates production download failures

**Cost Caveat (NEW):**
Include bandwidth and cold-start impact in ROI analysis. Large model downloads can materially affect costs and startup latency; track these in monthly cost reporting.

### Qualitative Improvements

**Developer Experience:**
- Single command to ensure model availability: `model_manager.ensure_available(repo_id)`
- Automatic fallback strategies eliminate manual troubleshooting
- Clear error messages with actionable remediation steps
- Historical analytics guide optimal download method selection

**Production Reliability:**
- Pre-cached critical models guarantee availability
- Multiple download strategies provide redundancy
- Validation framework catches corrupted downloads
- Registry enables rapid disaster recovery

**Strategic Flexibility:**
- Easy to add new models with configuration file changes
- Download method experimentation without code changes
- Migration path to container-based deployment
- Foundation for future model marketplace or plugin system

## Recommendations

### Immediate Actions (This Week)

1. **Implement Model Registry Foundation** (Priority 1)
   - Create SQLite schema for model tracking
   - Build ModelRegistry Python interface
   - Document in `as_built.md`

2. **Build Multi-Strategy Downloader** (Priority 1)
   - Implement Hub API method first (covers 80% of use cases)
   - Add Git LFS fallback for large models
   - Test with FLUX.1, InstantID, MuseTalk

3. **Create Model Configuration** (Priority 2)
   - Build `model_config.yaml` for 5 critical models
   - Document required files for each model framework
   - Establish validation rules

### Medium-Term Strategy (Next Month)

1. **Production Integration** (Priority 1)
   - Integrate downloader with FastAPI inference server
   - Implement pre-caching workflow for deployment
   - Add monitoring and alerting

2. **Performance Optimization** (Priority 2)
   - Benchmark download methods for each model
   - Implement intelligent method selection
   - Add parallel download support

3. **Operational Hardening** (Priority 2)
   - Build comprehensive error handling and retry logic
   - Create runbook for common failure scenarios
   - Implement automated health checks

### Long-Term Considerations (3-6 Months)

1. **Container-Based Deployment** (Evaluate)
   - Assess if download reliability issues persist
   - If yes, migrate critical models to pre-built containers
   - Maintain registry for experimental models

2. **Model Marketplace Foundation** (Strategic)
   - Registry architecture supports future plugin system
   - Download orchestrator can be extended for user-uploaded models
   - Consider exposing as sellable platform component

3. **Infrastructure Diversification** (Strategic)
   - Evaluate building internal Hugging Face mirror
   - Consider partnerships with model providers for direct access
   - Assess feasibility of training custom models to eliminate dependencies

### Risk Mitigation

**Primary Risks:**
1. **Hugging Face API Changes**: Monitor HF API changelog, maintain abstraction layer
2. **Model Licensing Issues**: Validate commercial licensing before adding to registry
3. **Storage Costs Escalation**: Implement model pruning strategy for unused models
4. **Download Bandwidth Limits**: Establish relationship with HF for enterprise access
5. **Custom Pipeline Drift**: Custom code breaks with dependency updates

**Mitigation Strategies:**
- Maintain multiple download methods (redundancy)
- Build abstraction layer allowing provider substitution
- Implement cost monitoring and alerting
- Document legal review process for new models
- Pin versions in the manifest; smoke test on upgrade

## Conclusion

The proposed Model Registry and Multi-Strategy Download architecture provides a robust, scalable foundation for Airaci's self-hosted inference infrastructure. By implementing intelligent fallback strategies, validation frameworks, and operational monitoring, this approach eliminates the current download reliability issues while positioning Airaci for future growth.

**Key Success Factors:**
1. **Comprehensive Validation**: Never mount models without manifest validation + smoke test
2. **Multiple Fallback Strategies**: Hub API → Git LFS → approved HTTP mirrors
3. **Historical Learning**: Registry analytics guide optimal download methods
4. **Pre-Caching Critical Models**: Eliminate production download dependencies
5. **Clear Documentation**: as_built.md ensures team knowledge transfer

**Expected Timeline:**
- **Week 1-2**: Foundation implemented and tested
- **Week 3-4**: Production integration complete
- **Week 5-6**: Optimization and hardening
- **Week 7+**: Monitoring production performance, iterating based on analytics

This strategy directly supports Airaci's RunPod migration goals, enabling the targeted 80% cost reduction while maintaining production reliability and positioning the platform for future scale.

---

**Next Steps:**
1. Review and approve technical approach
2. Create GitHub branch: `feature/model-registry-system`
3. Initialize `as_built.md` for Model Registry component
4. Begin Phase 1 implementation with FLUX.1 as pilot model
