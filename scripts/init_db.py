#!/usr/bin/env python3
"""Initialize the database with sample data"""

from app.db import SessionLocal, User, Model
from app.core.security import get_password_hash
from app.schemas import ModelType

def init_db():
    db = SessionLocal()

    try:
        # Create admin user
        admin = db.query(User).filter(User.username == "admin").first()
        if not admin:
            admin = User(
                username="admin",
                email="admin@example.com",
                hashed_password=get_password_hash("admin123")
            )
            db.add(admin)
            db.commit()
            print("Created admin user (username: admin, password: admin123)")

        # Create sample models
        models_data = [
            {
                "name": "GPT-2",
                "description": "GPT-2 small text generation model",
                "model_type": ModelType.TEXT_GENERATION,
                "version": "1.0.0",
                "model_path": "gpt2",
                "hardware": "cpu",
                "input_schema": {
                    "prompt": {
                        "type": "string",
                        "description": "Text prompt for generation"
                    },
                    "max_length": {
                        "type": "integer",
                        "description": "Maximum length",
                        "default": 100,
                        "minimum": 1,
                        "maximum": 500
                    }
                }
            }
        ]

        for model_data in models_data:
            existing = db.query(Model).filter(Model.name == model_data["name"]).first()
            if not existing:
                model = Model(**model_data, owner_id=admin.id)
                db.add(model)
                print(f"Created model: {model_data['name']}")

        db.commit()
        print("Database initialized successfully!")

    except Exception as e:
        print(f"Error initializing database: {e}")
        db.rollback()
    finally:
        db.close()


if __name__ == "__main__":
    init_db()
