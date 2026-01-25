from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import Optional
from app.db import get_db, Prediction, Model, User
from app.schemas import PredictionInput, PredictionResponse, PredictionList, PredictionStatus
from app.workers.tasks import run_inference
from app.services.auth import get_current_user, get_current_user_optional

router = APIRouter(prefix="/predictions", tags=["Predictions"])


@router.post("/", response_model=PredictionResponse)
async def create_prediction(
    prediction: PredictionInput,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user_optional)
):
    """Create a new prediction"""
    # Verify model exists
    model = db.query(Model).filter(Model.id == prediction.model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    # Create prediction record
    db_prediction = Prediction(
        model_id=prediction.model_id,
        user_id=current_user.id if current_user else None,
        input=prediction.input,
        webhook=prediction.webhook,
        status=PredictionStatus.STARTING
    )

    db.add(db_prediction)
    db.commit()
    db.refresh(db_prediction)

    # Queue inference task
    run_inference.delay(
        prediction_id=db_prediction.id,
        model_id=model.id,
        model_type=model.model_type.value,
        model_path=model.model_path,
        hardware=model.hardware,
        input_data=prediction.input
    )

    return db_prediction


@router.get("/{prediction_id}", response_model=PredictionResponse)
async def get_prediction(
    prediction_id: str,
    db: Session = Depends(get_db)
):
    """Get a specific prediction"""
    prediction = db.query(Prediction).filter(Prediction.id == prediction_id).first()
    if not prediction:
        raise HTTPException(status_code=404, detail="Prediction not found")

    return prediction


@router.get("/", response_model=PredictionList)
async def list_predictions(
    skip: int = 0,
    limit: int = 100,
    status: Optional[PredictionStatus] = None,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user_optional)
):
    """List predictions"""
    query = db.query(Prediction)

    if current_user:
        query = query.filter(Prediction.user_id == current_user.id)

    if status:
        query = query.filter(Prediction.status == status)

    predictions = query.order_by(Prediction.created_at.desc()).offset(skip).limit(limit).all()

    return {"predictions": predictions, "next_cursor": None}


@router.post("/{prediction_id}/cancel")
async def cancel_prediction(
    prediction_id: str,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user_optional)
):
    """Cancel a prediction"""
    prediction = db.query(Prediction).filter(Prediction.id == prediction_id).first()
    if not prediction:
        raise HTTPException(status_code=404, detail="Prediction not found")

    if prediction.status in [PredictionStatus.SUCCEEDED, PredictionStatus.FAILED, PredictionStatus.CANCELED]:
        raise HTTPException(status_code=400, detail="Prediction already completed")

    prediction.status = PredictionStatus.CANCELED
    db.commit()

    return {"message": "Prediction canceled"}
