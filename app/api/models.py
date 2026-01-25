from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List, Optional
from app.db import get_db, Model, User
from app.schemas import ModelCreate, ModelResponse, ModelList
from app.services.auth import get_current_user

router = APIRouter(prefix="/models", tags=["Models"])


@router.post("/", response_model=ModelResponse)
async def create_model(
    model: ModelCreate,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user)
):
    """Create a new model"""
    db_model = Model(
        name=model.name,
        description=model.description,
        model_type=model.model_type,
        version=model.version,
        model_path=model.model_path,
        input_schema=model.input_schema,
        hardware=model.hardware,
        owner_id=current_user.id if current_user else None
    )

    db.add(db_model)
    db.commit()
    db.refresh(db_model)

    return db_model


@router.get("/", response_model=ModelList)
async def list_models(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """List all models"""
    models = db.query(Model).offset(skip).limit(limit).all()
    return {"models": models}


@router.get("/{model_id}", response_model=ModelResponse)
async def get_model(model_id: str, db: Session = Depends(get_db)):
    """Get a specific model"""
    model = db.query(Model).filter(Model.id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    return model


@router.delete("/{model_id}")
async def delete_model(
    model_id: str,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user)
):
    """Delete a model"""
    model = db.query(Model).filter(Model.id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    if current_user and model.owner_id and model.owner_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized to delete this model")

    db.delete(model)
    db.commit()

    return {"message": "Model deleted successfully"}
