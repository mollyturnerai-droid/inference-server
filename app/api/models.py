from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.db import get_db, Model, Prediction
from app.schemas import ModelCreate, ModelResponse, ModelList, ModelLoadRequest, ModelType
from dataclasses import asdict

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


@router.post("/load", response_model=ModelResponse)
async def load_model_from_hub(
    request: ModelLoadRequest,
    db: Session = Depends(get_db)
):
    """
    Ensure a model is available locally, downloading it if necessary,
    and then register/return the ModelResponse.
    """
    from app.services.model_resolver import model_resolver
    from app.services.model_downloader import model_downloader
    from app.services.model_registry import model_registry
    
    repo_id = request.repo_id
    
    # 1. Check registry first
    local_path = model_registry.get_model_path(repo_id)
    
    if local_path is None or not local_path.exists() or request.force_redownload:
        # 2. Analyze model requirements
        try:
            metadata = model_resolver.analyze_model(repo_id)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to analyze model {repo_id}: {str(e)}")
            
        # 3. Determine best download method and execute
        try:
            preferred_method = model_registry.get_best_method(repo_id)
            local_path = model_downloader.download(metadata, method=preferred_method)
            
            # 4. Register in Model Registry (SQLite)
            model_registry.register_model(
                repo_id=repo_id,
                local_path=local_path,
                framework=metadata.framework.value,
                size_gb=metadata.size_gb,
                method=preferred_method or "auto",
                metadata=asdict(metadata) if hasattr(metadata, "__dict__") else {} # simplified
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to download model: {str(e)}")

    # 5. Register in main Database (SQLAlchemy) for Inference Server tracking
    # Check if exists in main DB
    db_model = db.query(Model).filter(Model.model_path == repo_id).first()
    if not db_model:
        # Get framework from request or metadata (implied)
        # For now, we'll try to find metadata again if we skipped download
        if 'metadata' not in locals():
            metadata = model_resolver.analyze_model(repo_id)
            
        db_model = Model(
            name=repo_id.split("/")[-1],
            description=f"Auto-loaded model from {repo_id}",
            model_type=metadata.framework.value if metadata.framework.value in [t.value for t in ModelType] else ModelType.CUSTOM,
            model_path=repo_id,
            version="1.0.0",
            input_schema={}, # can be populated later
            hardware="auto"
        )
        db.add(db_model)
        db.commit()
        db.refresh(db_model)
        
    return db_model

