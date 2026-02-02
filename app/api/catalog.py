from fastapi import APIRouter, HTTPException, Depends, Header
from typing import List, Optional
from pydantic import BaseModel, ConfigDict
from sqlalchemy.orm import Session
import uuid
from datetime import datetime

from app.services.catalog import (
    CatalogModel,
    get_all_catalog_models,
    get_catalog_models_by_type,
    get_catalog_model_by_id,
    get_catalog_categories,
    upsert_catalog_model,
    delete_catalog_model,
    refresh_catalog_model_schema,
)
from app.services.recon import run_recon, get_recon_status
from app.db import get_db
from app.db.models import Model, Prediction
from app.schemas.model import ModelType
from app.core.config import settings

router = APIRouter(prefix="/catalog", tags=["Catalog"])


class CatalogResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    categories: List[str]
    total_models: int
    models: List[CatalogModel]


class CategoryResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    category: str
    models: List[CatalogModel]


class MountRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    catalog_id: str
    name: Optional[str] = None
    hardware: Optional[str] = None  # Override recommended hardware


class MountResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    success: bool
    message: str
    model_id: Optional[str] = None
    model_name: str
    model_path: str
    model_type: str
    hardware: str


class ReconStatusResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    in_progress: bool
    last_started_at: Optional[datetime]
    last_completed_at: Optional[datetime]
    last_error: Optional[str]
    last_counts: dict


def _require_catalog_admin(x_catalog_admin_token: Optional[str] = Header(default=None)):
    if not settings.CATALOG_ADMIN_TOKEN:
        raise HTTPException(status_code=503, detail="Catalog admin token not configured")
    if x_catalog_admin_token != settings.CATALOG_ADMIN_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")


@router.get("/recon/status", response_model=ReconStatusResponse, dependencies=[Depends(_require_catalog_admin)])
async def catalog_recon_status():
    status = get_recon_status()
    return ReconStatusResponse(
        in_progress=status.in_progress,
        last_started_at=status.last_started_at,
        last_completed_at=status.last_completed_at,
        last_error=status.last_error,
        last_counts=status.last_counts,
    )


@router.post("/recon", response_model=ReconStatusResponse, dependencies=[Depends(_require_catalog_admin)])
async def catalog_recon_run(
    sources: Optional[str] = None,
    limit: Optional[int] = None,
):
    selected = [s.strip() for s in sources.split(",")] if sources else None
    status = run_recon(selected, limit)
    return ReconStatusResponse(
        in_progress=status.in_progress,
        last_started_at=status.last_started_at,
        last_completed_at=status.last_completed_at,
        last_error=status.last_error,
        last_counts=status.last_counts,
    )


@router.get("/models", response_model=CatalogResponse)
async def list_catalog_models(
    category: Optional[str] = None,
    size: Optional[str] = None,
    hardware: Optional[str] = None,
):
    """
    List all available models in the catalog.

    Filter by:
    - category: text-generation, text-to-image, classification, embeddings, etc.
    - size: tiny, small, medium, large, xl
    - hardware: cpu, gpu
    """
    if category:
        models = get_catalog_models_by_type(category)
    else:
        models = get_all_catalog_models()

    # Apply filters
    if size:
        models = [m for m in models if m.size == size]
    if hardware:
        models = [m for m in models if m.recommended_hardware == hardware]

    return CatalogResponse(
        categories=get_catalog_categories(),
        total_models=len(models),
        models=models
    )


@router.get("/categories", response_model=List[str])
async def list_categories():
    """List all available model categories"""
    return get_catalog_categories()


@router.get("/models/{category}", response_model=CategoryResponse)
async def get_models_by_category(category: str):
    """Get all models in a specific category"""
    models = get_catalog_models_by_type(category)
    if not models:
        raise HTTPException(
            status_code=404,
            detail=f"Category '{category}' not found. Available: {get_catalog_categories()}"
        )
    return CategoryResponse(category=category, models=models)


@router.get("/model/{catalog_id}", response_model=CatalogModel)
async def get_catalog_model(catalog_id: str):
    """Get details of a specific catalog model"""
    model = get_catalog_model_by_id(catalog_id)
    if not model:
        raise HTTPException(status_code=404, detail=f"Model '{catalog_id}' not found in catalog")
    return model


@router.post("/models/{catalog_id}/schema/refresh", response_model=CatalogModel, dependencies=[Depends(_require_catalog_admin)])
async def refresh_catalog_schema(catalog_id: str):
    """Refresh a model's input schema from its source (Replicate/Hugging Face)."""
    try:
        return refresh_catalog_model_schema(catalog_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/mount", response_model=MountResponse)
async def mount_catalog_model(
    request: MountRequest,
    db: Session = Depends(get_db)
):
    """
    Mount a model from the catalog to make it available for predictions.

    This creates a model entry that can then be used with the /v1/predictions endpoint.
    """
    catalog_model = get_catalog_model_by_id(request.catalog_id)
    if not catalog_model:
        # Attempt targeted recon for missing model IDs, then retry lookup.
        from app.services.recon import recon_model
        candidate = request.catalog_id
        if recon_model(candidate):
            catalog_model = get_catalog_model_by_id(request.catalog_id)
        if not catalog_model and "/" in candidate and not candidate.startswith(("hf:", "replicate:")):
            if recon_model(f"hf:{candidate}"):
                catalog_model = get_catalog_model_by_id(f"hf:{candidate}") or get_catalog_model_by_id(candidate)
        if not catalog_model:
            raise HTTPException(
                status_code=404,
                detail=f"Model '{request.catalog_id}' not found in catalog"
            )

    # Use provided name or catalog name
    model_name = request.name or catalog_model.id

    # Single-model mode: unmount any existing models before mounting a new one.
    existing_models = db.query(Model.id, Model.name).all()
    if existing_models:
        existing_ids = [row.id for row in existing_models]
        db.query(Prediction).filter(Prediction.model_id.in_(existing_ids)).delete(
            synchronize_session=False
        )
        db.query(Model).filter(Model.id.in_(existing_ids)).delete(
            synchronize_session=False
        )
        db.commit()

    # Check if model with this name already exists (should be empty after cleanup)
    existing = db.query(Model).filter(Model.name == model_name).first()
    if existing:
        raise HTTPException(
            status_code=409,
            detail=f"Model with name '{model_name}' already exists. Use a different name or delete the existing model."
        )

    # Use provided hardware or recommended
    hardware = request.hardware or catalog_model.recommended_hardware

    # Create the model entry
    model_id = str(uuid.uuid4())
    input_schema = {
        key: value.model_dump() if hasattr(value, "model_dump") else value
        for key, value in (catalog_model.input_schema or {}).items()
    }

    db_model = Model(
        id=model_id,
        name=model_name,
        description=catalog_model.description,
        model_type=catalog_model.model_type,
        version="1.0.0",
        model_path=catalog_model.model_path,
        input_schema=input_schema,
        hardware=hardware,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )

    db.add(db_model)
    db.commit()
    db.refresh(db_model)

    return MountResponse(
        success=True,
        message=f"Model '{model_name}' mounted successfully from catalog",
        model_id=model_id,
        model_name=model_name,
        model_path=catalog_model.model_path,
        model_type=catalog_model.model_type.value,
        hardware=hardware
    )


@router.post("/admin/models", response_model=CatalogModel, dependencies=[Depends(_require_catalog_admin)])
async def admin_create_or_update_catalog_model(model: CatalogModel):
    return upsert_catalog_model(model)


@router.put("/admin/models/{model_id}", response_model=CatalogModel, dependencies=[Depends(_require_catalog_admin)])
async def admin_put_catalog_model(model_id: str, model: CatalogModel):
    if model.id != model_id:
        raise HTTPException(status_code=400, detail="model_id path parameter must match body.id")
    return upsert_catalog_model(model)


@router.delete("/admin/models/{model_id}", dependencies=[Depends(_require_catalog_admin)])
async def admin_delete_catalog_model(model_id: str):
    removed = delete_catalog_model(model_id)
    if not removed:
        raise HTTPException(status_code=404, detail="Model not found")
    return {"success": True}
