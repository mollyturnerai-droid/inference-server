from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum


class ModelType(str, Enum):
    TEXT_GENERATION = "text-generation"
    IMAGE_GENERATION = "image-generation"
    IMAGE_TO_TEXT = "image-to-text"
    TEXT_TO_IMAGE = "text-to-image"
    CLASSIFICATION = "classification"
    EMBEDDINGS = "embeddings"
    SPEECH_TO_TEXT = "speech-to-text"
    SUMMARIZATION = "summarization"
    TRANSLATION = "translation"
    CUSTOM = "custom"


class ModelSchema(BaseModel):
    type: str = Field(..., description="Parameter type (string, integer, number, boolean, array)")
    description: Optional[str] = None
    default: Optional[Any] = None
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    enum: Optional[List[Any]] = None


class ModelCreate(BaseModel):
    name: str = Field(..., description="Model name")
    description: Optional[str] = None
    model_type: ModelType
    version: str = Field(default="1.0.0")
    model_path: str = Field(..., description="Path or HuggingFace model ID")
    input_schema: Dict[str, ModelSchema] = Field(default_factory=dict)
    hardware: str = Field(default="cpu", description="cpu, gpu, or auto")


class ModelResponse(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    model_type: ModelType
    version: str
    model_path: str
    input_schema: Dict[str, ModelSchema]
    hardware: str
    created_at: datetime
    updated_at: datetime
    owner_id: Optional[str] = None

    class Config:
        from_attributes = True


class ModelList(BaseModel):
    models: List[ModelResponse]
