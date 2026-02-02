from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.db import get_db, Model, Prediction
from app.schemas import ModelCreate, ModelResponse, ModelList

router = APIRouter(prefix="/models", tags=["Models"])


@router.post("/", response_model=ModelResponse)
async def create_model(
    model: ModelCreate,
    db: Session = Depends(get_db),
):
    """Create a new model"""
    input_schema = {
        key: value.model_dump() if hasattr(value, "model_dump") else value
        for key, value in (model.input_schema or {}).items()
    }
    db_model = Model(
        name=model.name,
        description=model.description,
        model_type=model.model_type,
        version=model.version,
        model_path=model.model_path,
        input_schema=input_schema,
        hardware=model.hardware,
        owner_id=None
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
    db: Session = Depends(get_db)
):
    """Delete a model"""
    exists = db.query(Model.id).filter(Model.id == model_id).first()
    if not exists:
        raise HTTPException(status_code=404, detail="Model not found")

    db.query(Prediction).filter(Prediction.model_id == model_id).delete(
        synchronize_session=False
    )
    db.query(Model).filter(Model.id == model_id).delete(
        synchronize_session=False
    )
    db.commit()

    return {"message": "Model deleted successfully"}


@router.post("/{model_id}/unmount")
async def unmount_model(
    model_id: str,
    db: Session = Depends(get_db)
):
    """Unmount (delete) a model without relying on DELETE."""
    exists = db.query(Model.id).filter(Model.id == model_id).first()
    if not exists:
        raise HTTPException(status_code=404, detail="Model not found")

    db.query(Prediction).filter(Prediction.model_id == model_id).delete(
        synchronize_session=False
    )
    db.query(Model).filter(Model.id == model_id).delete(
        synchronize_session=False
    )
    db.commit()

    return {"message": "Model unmounted successfully"}
