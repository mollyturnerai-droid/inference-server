from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum


class ModelType(str, Enum):
    TEXT_GENERATION = "text-generation"
    IMAGE_GENERATION = "image-generation"
    IMAGE_TO_TEXT = "image-to-text"
    IMAGE_TO_IMAGE = "image-to-image"
    TEXT_TO_IMAGE = "text-to-image"
    TEXT_TO_SPEECH = "text-to-speech"
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
    model_config = ConfigDict(protected_namespaces=())

    name: str = Field(..., description="Model name")
    description: Optional[str] = None
    model_type: ModelType
    version: str = Field(default="1.0.0")
    model_path: str = Field(..., description="Path or HuggingFace model ID")
    input_schema: Dict[str, ModelSchema] = Field(default_factory=dict)
    hardware: str = Field(default="cpu", description="cpu, gpu, or auto")


class ModelResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

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

class ModelList(BaseModel):
    models: List[ModelResponse]
