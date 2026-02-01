from sqlalchemy import Column, String, DateTime, JSON, Boolean, ForeignKey, Enum as SQLEnum, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid
from app.schemas import PredictionStatus, ModelType

Base = declarative_base()


def generate_uuid():
    return str(uuid.uuid4())


class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True, default=generate_uuid)
    username = Column(String, unique=True, nullable=False, index=True)
    email = Column(String, unique=True, nullable=False, index=True)
    hashed_password = Column(String, nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    models = relationship("Model", back_populates="owner")
    predictions = relationship("Prediction", back_populates="user")


class Model(Base):
    __tablename__ = "models"

    id = Column(String, primary_key=True, default=generate_uuid)
    name = Column(String, nullable=False, index=True)
    description = Column(String)
    model_type = Column(SQLEnum(ModelType), nullable=False)
    version = Column(String, default="1.0.0")
    model_path = Column(String, nullable=False)
    input_schema = Column(JSON, default={})
    hardware = Column(String, default="cpu")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    owner_id = Column(String, ForeignKey("users.id"), nullable=True)

    owner = relationship("User", back_populates="models")
    predictions = relationship("Prediction", back_populates="model")


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(String, primary_key=True, default=generate_uuid)
    status = Column(SQLEnum(PredictionStatus), default=PredictionStatus.STARTING, index=True)
    model_id = Column(String, ForeignKey("models.id"), nullable=False)
    user_id = Column(String, ForeignKey("users.id"), nullable=True)
    input = Column(JSON, nullable=False)
    output = Column(JSON, nullable=True)
    error = Column(String, nullable=True)
    logs = Column(String, nullable=True)
    webhook = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)

    model = relationship("Model", back_populates="predictions")
    user = relationship("User", back_populates="predictions")


class ApiKey(Base):
    __tablename__ = "api_keys"

    id = Column(String, primary_key=True, default=generate_uuid)
    name = Column(String, nullable=False, index=True)
    prefix = Column(String, nullable=False, index=True)
    key_hash = Column(String, nullable=False, unique=True, index=True)
    is_active = Column(Boolean, default=True)
    is_admin = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_used_at = Column(DateTime, nullable=True)


class CatalogModelEntry(Base):
    __tablename__ = "catalog_models"

    id = Column(String, primary_key=True)
    name = Column(String, nullable=False, index=True)
    description = Column(String, nullable=True)
    model_type = Column(SQLEnum(ModelType), nullable=False, index=True)
    model_path = Column(String, nullable=False)
    size = Column(String, nullable=True)
    vram_gb = Column(Float, nullable=True)
    recommended_hardware = Column(String, nullable=True)
    tags = Column(JSON, default=list)
    downloads = Column(String, nullable=True)
    license = Column(String, nullable=True)
    input_schema = Column(JSON, default=dict)
    source = Column(String, nullable=True)
    source_id = Column(String, nullable=True, index=True)
    source_url = Column(String, nullable=True)
    schema_source = Column(String, nullable=True)
    schema_version = Column(String, nullable=True)
    metadata = Column(JSON, default=dict)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_synced_at = Column(DateTime, nullable=True)
