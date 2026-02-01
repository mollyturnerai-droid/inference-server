from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
import threading
import time

import requests

from app.core.config import settings
from app.db import SessionLocal, CatalogModelEntry
from app.schemas.model import ModelType


@dataclass
class ReconStatus:
    in_progress: bool = False
    last_started_at: Optional[datetime] = None
    last_completed_at: Optional[datetime] = None
    last_error: Optional[str] = None
    last_counts: Dict[str, int] = field(default_factory=dict)


_status = ReconStatus()
_lock = threading.Lock()


PIPELINE_TO_MODEL_TYPE = {
    "text-generation": ModelType.TEXT_GENERATION,
    "text-to-image": ModelType.TEXT_TO_IMAGE,
    "image-to-text": ModelType.IMAGE_TO_TEXT,
    "text-to-speech": ModelType.TEXT_TO_SPEECH,
    "automatic-speech-recognition": ModelType.SPEECH_TO_TEXT,
    "summarization": ModelType.SUMMARIZATION,
    "translation": ModelType.TRANSLATION,
    "image-classification": ModelType.CLASSIFICATION,
    "text-classification": ModelType.CLASSIFICATION,
    "sentence-similarity": ModelType.EMBEDDINGS,
    "feature-extraction": ModelType.EMBEDDINGS,
}


SCHEMA_TEMPLATES: Dict[ModelType, Dict[str, Dict[str, Any]]] = {
    ModelType.TEXT_GENERATION: {
        "prompt": {"type": "string", "description": "Text prompt"},
        "max_length": {"type": "integer", "default": 128, "minimum": 1, "maximum": 4096},
        "temperature": {"type": "number", "default": 0.7, "minimum": 0.0, "maximum": 2.0},
        "top_p": {"type": "number", "default": 0.9, "minimum": 0.0, "maximum": 1.0},
        "top_k": {"type": "integer", "default": 50, "minimum": 0, "maximum": 1000},
        "stop": {"type": "array", "default": []},
        "seed": {"type": "integer", "default": 0, "minimum": 0},
    },
    ModelType.TEXT_TO_IMAGE: {
        "prompt": {"type": "string", "description": "Text prompt"},
        "negative_prompt": {"type": "string", "default": ""},
        "width": {"type": "integer", "default": 1024, "minimum": 64, "maximum": 2048},
        "height": {"type": "integer", "default": 1024, "minimum": 64, "maximum": 2048},
        "num_inference_steps": {"type": "integer", "default": 30, "minimum": 1, "maximum": 150},
        "guidance_scale": {"type": "number", "default": 7.5, "minimum": 0.0, "maximum": 30.0},
        "num_outputs": {"type": "integer", "default": 1, "minimum": 1, "maximum": 8},
        "seed": {"type": "integer", "default": 0, "minimum": 0},
    },
    ModelType.IMAGE_TO_TEXT: {
        "image": {"type": "string", "description": "Image URL or file path"},
        "prompt": {"type": "string", "default": ""},
    },
    ModelType.TEXT_TO_SPEECH: {
        "text": {"type": "string", "description": "Input text"},
        "voice": {"type": "string", "default": "default"},
        "speed": {"type": "number", "default": 1.0, "minimum": 0.5, "maximum": 2.0},
    },
    ModelType.SPEECH_TO_TEXT: {
        "audio": {"type": "string", "description": "Audio URL or file path"},
        "language": {"type": "string", "default": "auto"},
        "temperature": {"type": "number", "default": 0.0, "minimum": 0.0, "maximum": 1.0},
    },
    ModelType.EMBEDDINGS: {
        "text": {"type": "string", "description": "Input text"},
    },
    ModelType.CLASSIFICATION: {
        "text": {"type": "string", "description": "Input text"},
        "labels": {"type": "array", "default": []},
    },
    ModelType.SUMMARIZATION: {
        "text": {"type": "string", "description": "Input text"},
        "max_length": {"type": "integer", "default": 128, "minimum": 1, "maximum": 4096},
    },
    ModelType.TRANSLATION: {
        "text": {"type": "string", "description": "Input text"},
        "source_language": {"type": "string", "default": "auto"},
        "target_language": {"type": "string", "default": "en"},
    },
}


def get_recon_status() -> ReconStatus:
    with _lock:
        return ReconStatus(
            in_progress=_status.in_progress,
            last_started_at=_status.last_started_at,
            last_completed_at=_status.last_completed_at,
            last_error=_status.last_error,
            last_counts=dict(_status.last_counts),
        )


def _set_status(**kwargs):
    with _lock:
        for key, value in kwargs.items():
            setattr(_status, key, value)


def _schema_from_pipeline_tag(tag: Optional[str]) -> Tuple[Dict[str, Any], Optional[str]]:
    if not tag:
        return {}, None
    model_type = PIPELINE_TO_MODEL_TYPE.get(tag)
    if not model_type:
        return {}, None
    template = SCHEMA_TEMPLATES.get(model_type, {})
    return template, "template"


def _schema_from_hf_model(model_id: str) -> Tuple[Dict[str, Any], Optional[str], Optional[str], Dict[str, Any]]:
    url = f"https://huggingface.co/api/models/{model_id}"
    headers = {}
    if settings.HF_API_TOKEN:
        headers["Authorization"] = f"Bearer {settings.HF_API_TOKEN}"
    response = requests.get(url, headers=headers, timeout=settings.RECON_TIMEOUT_SECONDS)
    response.raise_for_status()
    data = response.json()
    pipeline_tag = data.get("pipeline_tag")
    schema, schema_source = _schema_from_pipeline_tag(pipeline_tag)
    metadata = {"pipeline_tag": pipeline_tag}
    schema_version = data.get("sha")
    return schema, schema_source, schema_version, metadata


def _upsert_catalog_entry(
    db,
    *,
    model_id: str,
    name: str,
    description: str,
    model_type: ModelType,
    model_path: str,
    size: str,
    vram_gb: Optional[float],
    recommended_hardware: str,
    tags: List[str],
    downloads: Optional[str],
    license: Optional[str],
    input_schema: Dict[str, Any],
    source: str,
    source_id: Optional[str],
    source_url: Optional[str],
    schema_source: Optional[str],
    schema_version: Optional[str],
    metadata_json: Dict[str, Any],
):
    now = datetime.utcnow()
    row = db.query(CatalogModelEntry).filter(CatalogModelEntry.id == model_id).first()
    if not row:
        row = CatalogModelEntry(id=model_id, created_at=now)
        db.add(row)
    row.name = name
    row.description = description
    row.model_type = model_type
    row.model_path = model_path
    row.size = size
    row.vram_gb = vram_gb
    row.recommended_hardware = recommended_hardware
    row.tags = tags
    row.downloads = downloads
    row.license = license
    row.input_schema = input_schema
    row.source = source
    row.source_id = source_id
    row.source_url = source_url
    row.schema_source = schema_source
    row.schema_version = schema_version
    row.metadata_json = metadata_json
    row.is_active = True
    row.updated_at = now
    row.last_synced_at = now


def _fetch_huggingface(limit: int) -> List[Dict[str, Any]]:
    url = "https://huggingface.co/api/models"
    params = {"limit": limit, "full": "true"}
    headers = {}
    if settings.HF_API_TOKEN:
        headers["Authorization"] = f"Bearer {settings.HF_API_TOKEN}"
    response = requests.get(url, params=params, headers=headers, timeout=settings.RECON_TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.json()


def _sync_huggingface(db, limit: int) -> int:
    items = _fetch_huggingface(limit)
    count = 0
    for item in items:
        model_id = item.get("id")
        if not model_id:
            continue
        pipeline_tag = item.get("pipeline_tag")
        model_type = PIPELINE_TO_MODEL_TYPE.get(pipeline_tag, ModelType.CUSTOM)
        schema, schema_source = _schema_from_pipeline_tag(pipeline_tag)
        tags = item.get("tags") or []
        card_data = item.get("cardData") or {}
        license_name = item.get("license") or card_data.get("license")
        downloads = item.get("downloads")
        model_path = model_id
        size = "medium"
        recommended_hardware = "gpu" if model_type in {ModelType.TEXT_TO_IMAGE, ModelType.TEXT_TO_SPEECH} else "cpu"
        _upsert_catalog_entry(
            db,
            model_id=f"hf:{model_id}",
            name=model_id.split("/")[-1],
            description=card_data.get("summary") or item.get("description") or "",
            model_type=model_type,
            model_path=model_path,
            size=size,
            vram_gb=None,
            recommended_hardware=recommended_hardware,
            tags=tags,
            downloads=str(downloads) if downloads is not None else None,
            license=license_name,
            input_schema=schema,
            source="huggingface",
            source_id=model_id,
            source_url=f"https://huggingface.co/{model_id}",
            schema_source=schema_source,
            schema_version=item.get("sha"),
            metadata_json={"pipeline_tag": pipeline_tag},
        )
        count += 1
    return count


def _fetch_replicate_models(limit: int) -> List[Dict[str, Any]]:
    headers = {"Authorization": f"Token {settings.REPLICATE_API_TOKEN}"}
    url = "https://api.replicate.com/v1/models"
    params = {"limit": limit}
    response = requests.get(url, params=params, headers=headers, timeout=settings.RECON_TIMEOUT_SECONDS)
    response.raise_for_status()
    data = response.json()
    return data.get("results", [])


def _fetch_replicate_model(owner: str, name: str) -> Dict[str, Any]:
    headers = {"Authorization": f"Token {settings.REPLICATE_API_TOKEN}"}
    url = f"https://api.replicate.com/v1/models/{owner}/{name}"
    response = requests.get(url, headers=headers, timeout=settings.RECON_TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.json()


def _fetch_replicate_version(owner: str, name: str, version_id: str) -> Dict[str, Any]:
    headers = {"Authorization": f"Token {settings.REPLICATE_API_TOKEN}"}
    url = f"https://api.replicate.com/v1/models/{owner}/{name}/versions/{version_id}"
    response = requests.get(url, headers=headers, timeout=settings.RECON_TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.json()


def _schema_from_openapi(openapi_schema: Dict[str, Any]) -> Dict[str, Any]:
    components = openapi_schema.get("components", {})
    schemas = components.get("schemas", {})
    input_schema = schemas.get("Input", {})
    properties = input_schema.get("properties", {})
    required = set(input_schema.get("required", []) or [])
    mapped: Dict[str, Any] = {}
    for key, value in properties.items():
        field_type = value.get("type", "string")
        mapped[key] = {
            "type": field_type,
            "description": value.get("description"),
            "default": value.get("default"),
            "enum": value.get("enum"),
        }
        if field_type in {"integer", "number"}:
            mapped[key]["minimum"] = value.get("minimum")
            mapped[key]["maximum"] = value.get("maximum")
        if key in required and mapped[key].get("default") is None:
            mapped[key]["default"] = ""
    return mapped


def _schema_from_replicate_model(owner: str, name: str) -> Tuple[Dict[str, Any], Optional[str], Optional[str], Dict[str, Any]]:
    if not settings.REPLICATE_API_TOKEN:
        return {}, None, None, {}
    model = _fetch_replicate_model(owner, name)
    latest_version = model.get("latest_version") or {}
    version_id = latest_version.get("id")
    schema = {}
    if version_id:
        try:
            version = _fetch_replicate_version(owner, name, version_id)
            openapi_schema = version.get("openapi_schema") or {}
            schema = _schema_from_openapi(openapi_schema) if openapi_schema else {}
        except Exception:
            schema = {}
    return schema, "openapi" if schema else None, version_id, {"latest_version": version_id}


def _sync_replicate(db, limit: int) -> int:
    if not settings.REPLICATE_API_TOKEN:
        return 0
    items = _fetch_replicate_models(limit)
    count = 0
    for item in items:
        owner = item.get("owner")
        name = item.get("name")
        if not owner or not name:
            continue
        latest_version = item.get("latest_version")
        version_id = latest_version.get("id") if isinstance(latest_version, dict) else None
        openapi_schema = {}
        if version_id:
            try:
                version = _fetch_replicate_version(owner, name, version_id)
                openapi_schema = version.get("openapi_schema") or {}
            except Exception:
                openapi_schema = {}
        schema = _schema_from_openapi(openapi_schema) if openapi_schema else {}
        model_type = ModelType.CUSTOM
        _upsert_catalog_entry(
            db,
            model_id=f"replicate:{owner}/{name}",
            name=f"{owner}/{name}",
            description=item.get("description") or "",
            model_type=model_type,
            model_path=f"replicate:{owner}/{name}:{version_id}" if version_id else f"replicate:{owner}/{name}",
            size="medium",
            vram_gb=None,
            recommended_hardware="gpu",
            tags=item.get("tags") or [],
            downloads=None,
            license=None,
            input_schema=schema,
            source="replicate",
            source_id=f"{owner}/{name}",
            source_url=item.get("url"),
            schema_source="openapi" if schema else None,
            schema_version=version_id,
            metadata_json={"latest_version": version_id},
        )
        count += 1
    return count


def run_recon(sources: Optional[List[str]] = None, limit: Optional[int] = None) -> ReconStatus:
    if not settings.RECON_ENABLED:
        return get_recon_status()
    limit = limit or settings.RECON_MAX_MODELS
    sources = sources or [s.strip() for s in settings.RECON_SOURCES.split(",") if s.strip()]

    _set_status(in_progress=True, last_started_at=datetime.utcnow(), last_error=None, last_counts={})
    counts: Dict[str, int] = {}
    db = SessionLocal()
    try:
        if "huggingface" in sources:
            counts["huggingface"] = _sync_huggingface(db, limit)
        if "replicate" in sources:
            counts["replicate"] = _sync_replicate(db, limit)
        db.commit()
        _set_status(last_counts=counts, last_completed_at=datetime.utcnow())
    except Exception as exc:
        db.rollback()
        _set_status(last_error=str(exc), last_completed_at=datetime.utcnow())
    finally:
        db.close()
        _set_status(in_progress=False)
    return get_recon_status()


def start_recon_scheduler():
    if not settings.RECON_ENABLED:
        return

    def _loop():
        if settings.RECON_ON_STARTUP:
            run_recon()
        interval = max(settings.RECON_INTERVAL_MINUTES, 5)
        while True:
            time.sleep(interval * 60)
            run_recon()

    thread = threading.Thread(target=_loop, daemon=True)
    thread.start()
