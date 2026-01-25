from fastapi import APIRouter, HTTPException, Depends
from typing import List, Optional
from pydantic import BaseModel
from sqlalchemy.orm import Session
import uuid
from datetime import datetime

from app.services.catalog import (
    CatalogModel,
    get_all_catalog_models,
    get_catalog_models_by_type,
    get_catalog_model_by_id,
    get_catalog_categories,
)
from app.db import get_db
from app.db.models import Model
from app.schemas.model import ModelType

router = APIRouter(prefix="/catalog", tags=["Catalog"])


class CatalogResponse(BaseModel):
    categories: List[str]
    total_models: int
    models: List[CatalogModel]


class CategoryResponse(BaseModel):
    category: str
    models: List[CatalogModel]


class MountRequest(BaseModel):
    catalog_id: str
    name: Optional[str] = None
    hardware: Optional[str] = None  # Override recommended hardware


class MountResponse(BaseModel):
    success: bool
    message: str
    model_id: Optional[str] = None
    model_name: str
    model_path: str
    model_type: str
    hardware: str


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
        raise HTTPException(
            status_code=404,
            detail=f"Model '{request.catalog_id}' not found in catalog"
        )

    # Use provided name or catalog name
    model_name = request.name or catalog_model.id

    # Check if model with this name already exists
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
    db_model = Model(
        id=model_id,
        name=model_name,
        description=catalog_model.description,
        model_type=catalog_model.model_type,
        version="1.0.0",
        model_path=catalog_model.model_path,
        input_schema={},
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
