from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum


class PredictionStatus(str, Enum):
    STARTING = "starting"
    PROCESSING = "processing"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELED = "canceled"


class PredictionInput(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_id: str = Field(..., description="ID of the model to use")
    input: Dict[str, Any] = Field(..., description="Input parameters for the model")
    webhook: Optional[str] = Field(None, description="Webhook URL to call when prediction completes")


class PredictionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

    id: str
    status: PredictionStatus
    model_id: str
    input: Dict[str, Any]
    output: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    logs: Optional[str] = None
    progress: Optional[float] = None
    progress_step: Optional[int] = None
    progress_total: Optional[int] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    webhook: Optional[str] = None

class PredictionList(BaseModel):
    predictions: List[PredictionResponse]
    next_cursor: Optional[str] = None
