from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from datetime import datetime
from app.schemas.model import ModelType, ModelSchema
from app.core.config import settings
from app.db import SessionLocal, CatalogModelEntry
import json
import os


class CatalogModel(BaseModel):
    id: str
    name: str
    description: str
    model_type: ModelType
    model_path: str
    size: str  # tiny, small, medium, large, xl
    vram_gb: Optional[float] = None
    recommended_hardware: str  # cpu, gpu
    tags: List[str] = []
    downloads: Optional[str] = None
    license: Optional[str] = None
    input_schema: Dict[str, ModelSchema] = {}
    source: Optional[str] = None
    source_id: Optional[str] = None
    source_url: Optional[str] = None
    schema_source: Optional[str] = None
    schema_version: Optional[str] = None
    last_synced_at: Optional[datetime] = None
    latest_update: Optional[datetime] = None


# Curated model catalog organized by type
MODEL_CATALOG: Dict[str, List[CatalogModel]] = {
    "text-generation": [
        CatalogModel(
            id="qwen3-0.6b",
            name="Qwen3 0.6B",
            description="Compact but capable chat model, good for resource-constrained environments",
            model_type=ModelType.TEXT_GENERATION,
            model_path="Qwen/Qwen3-0.6B",
            size="tiny",
            vram_gb=2,
            recommended_hardware="gpu",
            tags=["chat", "conversational", "lightweight"],
            downloads="8.5M",
            license="apache-2.0"
        ),
        CatalogModel(
            id="qwen2.5-1.5b-instruct",
            name="Qwen2.5 1.5B Instruct",
            description="Small instruction-following model with good performance",
            model_type=ModelType.TEXT_GENERATION,
            model_path="Qwen/Qwen2.5-1.5B-Instruct",
            size="small",
            vram_gb=4,
            recommended_hardware="gpu",
            tags=["chat", "instruct", "efficient"],
            downloads="6.6M",
            license="apache-2.0"
        ),
        CatalogModel(
            id="qwen2.5-3b-instruct",
            name="Qwen2.5 3B Instruct",
            description="Balanced model for general chat and instruction following",
            model_type=ModelType.TEXT_GENERATION,
            model_path="Qwen/Qwen2.5-3B-Instruct",
            size="small",
            vram_gb=7,
            recommended_hardware="gpu",
            tags=["chat", "instruct", "balanced"],
            downloads="15.4M",
            license="other"
        ),
        CatalogModel(
            id="qwen2.5-7b-instruct",
            name="Qwen2.5 7B Instruct",
            description="High-quality instruction model for complex tasks",
            model_type=ModelType.TEXT_GENERATION,
            model_path="Qwen/Qwen2.5-7B-Instruct",
            size="medium",
            vram_gb=16,
            recommended_hardware="gpu",
            tags=["chat", "instruct", "high-quality"],
            downloads="7.2M",
            license="apache-2.0"
        ),
        CatalogModel(
            id="llama-3.1-8b-instruct",
            name="Llama 3.1 8B Instruct",
            description="Meta's latest instruction-tuned model with strong reasoning",
            model_type=ModelType.TEXT_GENERATION,
            model_path="meta-llama/Llama-3.1-8B-Instruct",
            size="medium",
            vram_gb=18,
            recommended_hardware="gpu",
            tags=["chat", "instruct", "reasoning", "meta"],
            downloads="10.1M",
            license="llama3.1"
        ),
        CatalogModel(
            id="gpt2",
            name="GPT-2",
            description="Classic OpenAI language model, lightweight and fast",
            model_type=ModelType.TEXT_GENERATION,
            model_path="openai-community/gpt2",
            size="tiny",
            vram_gb=1,
            recommended_hardware="cpu",
            tags=["classic", "lightweight", "fast"],
            downloads="7.1M",
            license="mit"
        ),
        CatalogModel(
            id="qwen3-8b",
            name="Qwen3 8B",
            description="Latest Qwen model with excellent multilingual support",
            model_type=ModelType.TEXT_GENERATION,
            model_path="Qwen/Qwen3-8B",
            size="medium",
            vram_gb=18,
            recommended_hardware="gpu",
            tags=["chat", "multilingual", "latest"],
            downloads="4.2M",
            license="apache-2.0"
        ),
    ],
    "text-to-speech": [
        CatalogModel(
            id="xtts-v2",
            name="XTTS v2",
            description="Multilingual text-to-speech with voice cloning support",
            model_type=ModelType.TEXT_TO_SPEECH,
            model_path="coqui/XTTS-v2",
            size="large",
            vram_gb=12,
            recommended_hardware="gpu",
            tags=["tts", "voice-cloning", "multilingual"],
            downloads=None,
            license=None
        ),
        CatalogModel(
            id="bark",
            name="Bark",
            description="Expressive text-to-speech model for realistic speech generation",
            model_type=ModelType.TEXT_TO_SPEECH,
            model_path="suno/bark",
            size="large",
            vram_gb=12,
            recommended_hardware="gpu",
            tags=["tts", "expressive"],
            downloads=None,
            license=None
        ),
        CatalogModel(
            id="speecht5-tts",
            name="SpeechT5 TTS",
            description="Lightweight text-to-speech model suitable for CPU usage",
            model_type=ModelType.TEXT_TO_SPEECH,
            model_path="microsoft/speecht5_tts",
            size="medium",
            vram_gb=None,
            recommended_hardware="cpu",
            tags=["tts", "lightweight"],
            downloads=None,
            license=None
        ),
    ],
    "text-to-image": [
        CatalogModel(
            id="sd-v1-5",
            name="Stable Diffusion v1.5",
            description="Classic SD model, well-supported with many community extensions",
            model_type=ModelType.TEXT_TO_IMAGE,
            model_path="stable-diffusion-v1-5/stable-diffusion-v1-5",
            size="medium",
            vram_gb=8,
            recommended_hardware="gpu",
            tags=["stable-diffusion", "classic", "community"],
            downloads="1.5M",
            license="creativeml-openrail-m"
        ),
        CatalogModel(
            id="sd-v1-4",
            name="Stable Diffusion v1.4",
            description="Original Stable Diffusion release",
            model_type=ModelType.TEXT_TO_IMAGE,
            model_path="CompVis/stable-diffusion-v1-4",
            size="medium",
            vram_gb=8,
            recommended_hardware="gpu",
            tags=["stable-diffusion", "original"],
            downloads="634K",
            license="creativeml-openrail-m"
        ),
        CatalogModel(
            id="sdxl-base",
            name="Stable Diffusion XL Base",
            description="Higher resolution (1024px) with improved quality",
            model_type=ModelType.TEXT_TO_IMAGE,
            model_path="stabilityai/stable-diffusion-xl-base-1.0",
            size="large",
            vram_gb=12,
            recommended_hardware="gpu",
            tags=["sdxl", "high-resolution", "quality"],
            downloads="2.0M",
            license="openrail++"
        ),
        CatalogModel(
            id="sd-turbo",
            name="SD Turbo",
            description="Fast single-step generation for quick iterations",
            model_type=ModelType.TEXT_TO_IMAGE,
            model_path="stabilityai/sd-turbo",
            size="medium",
            vram_gb=8,
            recommended_hardware="gpu",
            tags=["turbo", "fast", "single-step"],
            downloads="673K",
            license="other"
        ),
        CatalogModel(
            id="sdxl-turbo",
            name="SDXL Turbo",
            description="Fast SDXL variant for rapid high-res generation",
            model_type=ModelType.TEXT_TO_IMAGE,
            model_path="stabilityai/sdxl-turbo",
            size="large",
            vram_gb=12,
            recommended_hardware="gpu",
            tags=["sdxl", "turbo", "fast"],
            downloads="354K",
            license="other"
        ),
        CatalogModel(
            id="flux-schnell",
            name="FLUX.1 Schnell",
            description="State-of-the-art fast image generation",
            model_type=ModelType.TEXT_TO_IMAGE,
            model_path="black-forest-labs/FLUX.1-schnell",
            size="xl",
            vram_gb=24,
            recommended_hardware="gpu",
            tags=["flux", "fast", "high-quality", "sota"],
            downloads="623K",
            license="apache-2.0"
        ),
        CatalogModel(
            id="flux-dev",
            name="FLUX.1 Dev",
            description="Highest quality image generation, slower but superior",
            model_type=ModelType.TEXT_TO_IMAGE,
            model_path="black-forest-labs/FLUX.1-dev",
            size="xl",
            vram_gb=24,
            recommended_hardware="gpu",
            tags=["flux", "highest-quality", "sota"],
            downloads="787K",
            license="other"
        ),
        CatalogModel(
            id="playground-v2.5",
            name="Playground v2.5",
            description="Aesthetic-focused model for beautiful images",
            model_type=ModelType.TEXT_TO_IMAGE,
            model_path="playgroundai/playground-v2.5-1024px-aesthetic",
            size="large",
            vram_gb=12,
            recommended_hardware="gpu",
            tags=["aesthetic", "sdxl", "beautiful"],
            downloads="478K",
            license="other"
        ),
        CatalogModel(
            id="animagine-xl-4",
            name="Animagine XL 4.0",
            description="Specialized for anime-style illustrations",
            model_type=ModelType.TEXT_TO_IMAGE,
            model_path="cagliostrolab/animagine-xl-4.0",
            size="large",
            vram_gb=12,
            recommended_hardware="gpu",
            tags=["anime", "illustration", "sdxl"],
            downloads="240K",
            license="openrail++"
        ),
    ],
    "classification": [
        CatalogModel(
            id="distilbert-sentiment",
            name="DistilBERT Sentiment",
            description="Fast sentiment analysis (positive/negative)",
            model_type=ModelType.CLASSIFICATION,
            model_path="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
            size="small",
            vram_gb=1,
            recommended_hardware="cpu",
            tags=["sentiment", "fast", "english"],
            downloads="3.7M",
            license="apache-2.0"
        ),
        CatalogModel(
            id="twitter-roberta-sentiment",
            name="Twitter RoBERTa Sentiment",
            description="Sentiment analysis tuned for social media",
            model_type=ModelType.CLASSIFICATION,
            model_path="cardiffnlp/twitter-roberta-base-sentiment-latest",
            size="medium",
            vram_gb=2,
            recommended_hardware="cpu",
            tags=["sentiment", "twitter", "social-media"],
            downloads="4.5M",
            license="cc-by-4.0"
        ),
        CatalogModel(
            id="bart-mnli",
            name="BART Large MNLI",
            description="Zero-shot classification for any categories",
            model_type=ModelType.CLASSIFICATION,
            model_path="facebook/bart-large-mnli",
            size="large",
            vram_gb=4,
            recommended_hardware="gpu",
            tags=["zero-shot", "flexible", "nli"],
            downloads="3.5M",
            license="mit"
        ),
        CatalogModel(
            id="finbert",
            name="FinBERT",
            description="Financial sentiment analysis",
            model_type=ModelType.CLASSIFICATION,
            model_path="ProsusAI/finbert",
            size="medium",
            vram_gb=2,
            recommended_hardware="cpu",
            tags=["finance", "sentiment", "specialized"],
            downloads="2.3M",
            license="other"
        ),
        CatalogModel(
            id="twitter-xlm-sentiment",
            name="Twitter XLM RoBERTa Sentiment",
            description="Multilingual social media sentiment",
            model_type=ModelType.CLASSIFICATION,
            model_path="cardiffnlp/twitter-xlm-roberta-base-sentiment",
            size="medium",
            vram_gb=2,
            recommended_hardware="cpu",
            tags=["multilingual", "twitter", "sentiment"],
            downloads="1.6M",
            license="other"
        ),
    ],
    "embeddings": [
        CatalogModel(
            id="all-minilm-l6-v2",
            name="all-MiniLM-L6-v2",
            description="Fast, lightweight embeddings for semantic search",
            model_type=ModelType.EMBEDDINGS,
            model_path="sentence-transformers/all-MiniLM-L6-v2",
            size="tiny",
            vram_gb=0.5,
            recommended_hardware="cpu",
            tags=["fast", "semantic-search", "lightweight"],
            downloads="147.6M",
            license="apache-2.0"
        ),
        CatalogModel(
            id="all-mpnet-base-v2",
            name="all-mpnet-base-v2",
            description="Higher quality embeddings, best for accuracy",
            model_type=ModelType.EMBEDDINGS,
            model_path="sentence-transformers/all-mpnet-base-v2",
            size="medium",
            vram_gb=1,
            recommended_hardware="cpu",
            tags=["high-quality", "semantic-search", "accurate"],
            downloads="22.2M",
            license="apache-2.0"
        ),
        CatalogModel(
            id="bge-m3",
            name="BGE-M3",
            description="State-of-the-art multilingual embeddings",
            model_type=ModelType.EMBEDDINGS,
            model_path="BAAI/bge-m3",
            size="large",
            vram_gb=2,
            recommended_hardware="gpu",
            tags=["multilingual", "sota", "versatile"],
            downloads="9.8M",
            license="mit"
        ),
        CatalogModel(
            id="bge-large-en",
            name="BGE Large English",
            description="Best English-only embeddings",
            model_type=ModelType.EMBEDDINGS,
            model_path="BAAI/bge-large-en-v1.5",
            size="large",
            vram_gb=2,
            recommended_hardware="gpu",
            tags=["english", "high-quality", "sota"],
            downloads="4.3M",
            license="mit"
        ),
        CatalogModel(
            id="multilingual-minilm",
            name="Multilingual MiniLM",
            description="Lightweight multilingual embeddings (50+ languages)",
            model_type=ModelType.EMBEDDINGS,
            model_path="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            size="small",
            vram_gb=1,
            recommended_hardware="cpu",
            tags=["multilingual", "lightweight", "50-languages"],
            downloads="17.5M",
            license="apache-2.0"
        ),
        CatalogModel(
            id="nomic-embed-v1.5",
            name="Nomic Embed v1.5",
            description="Excellent balance of speed and quality",
            model_type=ModelType.EMBEDDINGS,
            model_path="nomic-ai/nomic-embed-text-v1.5",
            size="medium",
            vram_gb=1,
            recommended_hardware="cpu",
            tags=["balanced", "efficient", "modern"],
            downloads="3.8M",
            license="apache-2.0"
        ),
    ],
    "speech-to-text": [
        CatalogModel(
            id="whisper-tiny",
            name="Whisper Tiny",
            description="Fastest Whisper model, good for real-time",
            model_type=ModelType.SPEECH_TO_TEXT,
            model_path="openai/whisper-tiny",
            size="tiny",
            vram_gb=1,
            recommended_hardware="cpu",
            tags=["fast", "real-time", "multilingual"],
            downloads="513K",
            license="apache-2.0"
        ),
        CatalogModel(
            id="whisper-base",
            name="Whisper Base",
            description="Good balance of speed and accuracy",
            model_type=ModelType.SPEECH_TO_TEXT,
            model_path="openai/whisper-base",
            size="small",
            vram_gb=1,
            recommended_hardware="cpu",
            tags=["balanced", "multilingual"],
            downloads="1.2M",
            license="apache-2.0"
        ),
        CatalogModel(
            id="whisper-small",
            name="Whisper Small",
            description="Improved accuracy, still relatively fast",
            model_type=ModelType.SPEECH_TO_TEXT,
            model_path="openai/whisper-small",
            size="small",
            vram_gb=2,
            recommended_hardware="gpu",
            tags=["accurate", "multilingual"],
            downloads="1.5M",
            license="apache-2.0"
        ),
        CatalogModel(
            id="whisper-medium",
            name="Whisper Medium",
            description="High accuracy for most use cases",
            model_type=ModelType.SPEECH_TO_TEXT,
            model_path="openai/whisper-medium",
            size="medium",
            vram_gb=5,
            recommended_hardware="gpu",
            tags=["high-accuracy", "multilingual"],
            downloads="538K",
            license="apache-2.0"
        ),
        CatalogModel(
            id="whisper-large-v3",
            name="Whisper Large v3",
            description="Best accuracy, supports 100+ languages",
            model_type=ModelType.SPEECH_TO_TEXT,
            model_path="openai/whisper-large-v3",
            size="large",
            vram_gb=10,
            recommended_hardware="gpu",
            tags=["best-accuracy", "100-languages", "sota"],
            downloads="6.3M",
            license="apache-2.0"
        ),
        CatalogModel(
            id="whisper-large-v3-turbo",
            name="Whisper Large v3 Turbo",
            description="Fast large model with excellent accuracy",
            model_type=ModelType.SPEECH_TO_TEXT,
            model_path="openai/whisper-large-v3-turbo",
            size="large",
            vram_gb=8,
            recommended_hardware="gpu",
            tags=["fast", "large", "turbo"],
            downloads="2.8M",
            license="apache-2.0"
        ),
    ],
    "summarization": [
        CatalogModel(
            id="bart-large-cnn",
            name="BART Large CNN",
            description="Best summarization model for news articles",
            model_type=ModelType.SUMMARIZATION,
            model_path="facebook/bart-large-cnn",
            size="large",
            vram_gb=4,
            recommended_hardware="gpu",
            tags=["news", "articles", "high-quality"],
            downloads="2.9M",
            license="mit"
        ),
        CatalogModel(
            id="distilbart-cnn",
            name="DistilBART CNN",
            description="Faster summarization with good quality",
            model_type=ModelType.SUMMARIZATION,
            model_path="sshleifer/distilbart-cnn-12-6",
            size="medium",
            vram_gb=2,
            recommended_hardware="gpu",
            tags=["fast", "efficient", "news"],
            downloads="596K",
            license="apache-2.0"
        ),
        CatalogModel(
            id="t5-small",
            name="T5 Small",
            description="Versatile model for summarization and more",
            model_type=ModelType.SUMMARIZATION,
            model_path="google-t5/t5-small",
            size="small",
            vram_gb=1,
            recommended_hardware="cpu",
            tags=["versatile", "multi-task", "lightweight"],
            downloads="3.3M",
            license="apache-2.0"
        ),
        CatalogModel(
            id="t5-base",
            name="T5 Base",
            description="Better quality T5 for summarization",
            model_type=ModelType.SUMMARIZATION,
            model_path="google-t5/t5-base",
            size="medium",
            vram_gb=2,
            recommended_hardware="gpu",
            tags=["versatile", "quality", "multi-task"],
            downloads="2.4M",
            license="apache-2.0"
        ),
    ],
    "translation": [
        CatalogModel(
            id="t5-small-translation",
            name="T5 Small",
            description="Multi-language translation (en, fr, de, ro)",
            model_type=ModelType.TRANSLATION,
            model_path="google-t5/t5-small",
            size="small",
            vram_gb=1,
            recommended_hardware="cpu",
            tags=["multi-language", "versatile"],
            downloads="3.3M",
            license="apache-2.0"
        ),
        CatalogModel(
            id="opus-mt-fr-en",
            name="OPUS MT French-English",
            description="High quality French to English translation",
            model_type=ModelType.TRANSLATION,
            model_path="Helsinki-NLP/opus-mt-fr-en",
            size="small",
            vram_gb=1,
            recommended_hardware="cpu",
            tags=["french", "english", "specialized"],
            downloads="892K",
            license="apache-2.0"
        ),
        CatalogModel(
            id="t5-base-translation",
            name="T5 Base",
            description="Better quality multi-language translation",
            model_type=ModelType.TRANSLATION,
            model_path="google-t5/t5-base",
            size="medium",
            vram_gb=2,
            recommended_hardware="gpu",
            tags=["multi-language", "quality"],
            downloads="2.4M",
            license="apache-2.0"
        ),
    ],
}


_catalog_cache: Optional[Dict[str, List[CatalogModel]]] = None


def _load_catalog_from_disk() -> Dict[str, List[CatalogModel]]:
    path = settings.CATALOG_PATH
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    loaded: Dict[str, List[CatalogModel]] = {}
    if isinstance(data, dict):
        for category, items in data.items():
            if not isinstance(items, list):
                continue
            models: List[CatalogModel] = []
            for item in items:
                if not isinstance(item, dict):
                    continue
                try:
                    models.append(CatalogModel(**item))
                except Exception:
                    continue
            loaded[str(category)] = models

    return loaded


def _get_catalog() -> Dict[str, List[CatalogModel]]:
    global _catalog_cache
    if _catalog_cache is not None:
        return _catalog_cache

    catalog: Dict[str, List[CatalogModel]] = {k: list(v) for k, v in MODEL_CATALOG.items()}
    path = settings.CATALOG_PATH
    try:
        if path and os.path.exists(path):
            loaded = _load_catalog_from_disk()
            if loaded:
                for category, models in loaded.items():
                    catalog[str(category)] = models
    except Exception:
        catalog = {k: list(v) for k, v in MODEL_CATALOG.items()}

    _catalog_cache = catalog
    return _catalog_cache


def _save_catalog(catalog: Dict[str, List[CatalogModel]]):
    path = settings.CATALOG_PATH
    if not path:
        return

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload: Dict[str, Any] = {}
    for category, models in catalog.items():
        payload[category] = [m.model_dump() for m in models]

    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def _row_to_catalog_model(row: CatalogModelEntry) -> CatalogModel:
    latest_update = row.last_synced_at or row.updated_at
    return CatalogModel(
        id=row.id,
        name=row.name,
        description=row.description or "",
        model_type=row.model_type,
        model_path=row.model_path,
        size=row.size or "medium",
        vram_gb=row.vram_gb,
        recommended_hardware=row.recommended_hardware or "cpu",
        tags=row.tags or [],
        downloads=row.downloads,
        license=row.license,
        input_schema=row.input_schema or {},
        source=row.source,
        source_id=row.source_id,
        source_url=row.source_url,
        schema_source=row.schema_source,
        schema_version=row.schema_version,
        last_synced_at=row.last_synced_at,
        latest_update=latest_update,
    )


def _get_catalog_from_db() -> List[CatalogModel]:
    db = SessionLocal()
    try:
        rows = (
            db.query(CatalogModelEntry)
            .filter(CatalogModelEntry.is_active == True)  # noqa: E712
            .order_by(CatalogModelEntry.name.asc())
            .all()
        )
        return [_row_to_catalog_model(row) for row in rows]
    except Exception:
        return []
    finally:
        db.close()


def upsert_catalog_model(model: CatalogModel) -> CatalogModel:
    db = SessionLocal()
    try:
        row = db.query(CatalogModelEntry).filter(CatalogModelEntry.id == model.id).first()
        now = datetime.utcnow()
        if not row:
            row = CatalogModelEntry(
                id=model.id,
                created_at=now,
                source=model.source or "manual",
            )
            db.add(row)
        row.name = model.name
        row.description = model.description
        row.model_type = model.model_type
        row.model_path = model.model_path
        row.size = model.size
        row.vram_gb = model.vram_gb
        row.recommended_hardware = model.recommended_hardware
        row.tags = model.tags
        row.downloads = model.downloads
        row.license = model.license
        row.input_schema = model.input_schema or {}
        row.source = model.source or row.source
        row.source_id = model.source_id or row.source_id
        row.source_url = model.source_url or row.source_url
        row.schema_source = model.schema_source or row.schema_source
        row.schema_version = model.schema_version or row.schema_version
        row.updated_at = now
        row.last_synced_at = model.last_synced_at or row.last_synced_at
        row.is_active = True
        db.commit()
        db.refresh(row)
        return _row_to_catalog_model(row)
    finally:
        db.close()


def delete_catalog_model(model_id: str) -> bool:
    db = SessionLocal()
    try:
        row = db.query(CatalogModelEntry).filter(CatalogModelEntry.id == model_id).first()
        if not row:
            return False
        row.is_active = False
        row.updated_at = datetime.utcnow()
        db.commit()
        return True
    finally:
        db.close()


def get_all_catalog_models() -> List[CatalogModel]:
    """Get all models from the catalog"""
    db_models = _get_catalog_from_db()
    if db_models:
        return db_models
    models = []
    for category_models in _get_catalog().values():
        models.extend(category_models)
    return models


def get_catalog_models_by_type(model_type: str) -> List[CatalogModel]:
    """Get models from catalog filtered by type"""
    db_models = _get_catalog_from_db()
    if db_models:
        return [m for m in db_models if m.model_type.value == model_type]
    return _get_catalog().get(model_type, [])


def get_catalog_model_by_id(model_id: str) -> Optional[CatalogModel]:
    """Get a specific model from the catalog by ID"""
    db_models = _get_catalog_from_db()
    if db_models:
        for model in db_models:
            if model.id == model_id:
                return model
        return None
    for category_models in _get_catalog().values():
        for model in category_models:
            if model.id == model_id:
                return model
    return None


def get_catalog_categories() -> List[str]:
    """Get all available model categories"""
    db_models = _get_catalog_from_db()
    if db_models:
        return sorted({m.model_type.value for m in db_models})
    return list(_get_catalog().keys())
