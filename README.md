# Inference Server

A full-featured ML inference engine similar to Replicate, built with FastAPI, Celery, and PyTorch.

## Features

- **RESTful API**: FastAPI-based REST API for managing models and predictions
- **Asynchronous Processing**: Celery workers for background inference tasks
- **Model Management**: Support for multiple model types (text generation, image generation, etc.)
- **Authentication**: JWT-based user authentication
- **Rate Limiting**: Built-in rate limiting for API endpoints
- **Task Queue**: Redis-backed task queue for job management
- **Model Caching**: Efficient model loading and caching
- **Webhooks**: Callback support for completed predictions
- **Scalable**: Docker-based deployment with horizontal scaling

## Supported Model Types

- Text Generation (HuggingFace transformers)
- Image Generation (Stable Diffusion)
- Image-to-Text
- Text-to-Image
- Classification
- Custom models (extensible architecture)

## Architecture

```
┌─────────────┐
│   Client    │
└──────┬──────┘
       │
       ▼
┌─────────────┐     ┌──────────┐
│  FastAPI    │────▶│  Redis   │
│   Server    │     │  Queue   │
└─────────────┘     └────┬─────┘
       │                 │
       │                 ▼
       │          ┌─────────────┐
       │          │   Celery    │
       │          │   Workers   │
       │          └─────────────┘
       │                 │
       ▼                 ▼
┌─────────────────────────────┐
│      PostgreSQL DB          │
└─────────────────────────────┘
```

## Installation

### Prerequisites

- Python 3.11+
- Docker and Docker Compose (for deployment)
- PostgreSQL (for local development)
- Redis (for local development)

### Local Development

1. Clone the repository:
```bash
git clone <repository-url>
cd inference_server
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Copy environment variables:
```bash
cp .env.example .env
```

5. Edit `.env` with your configuration:
```bash
# Update DATABASE_URL, REDIS_HOST, SECRET_KEY, etc.
```

6. Initialize the database:
```bash
alembic upgrade head
```

7. Run the API server:
```bash
python -m app.main
```

8. In a separate terminal, run the Celery worker:
```bash
celery -A app.workers.celery_app worker --loglevel=info
```

### Docker Deployment

1. Build and start all services:
```bash
docker-compose up -d
```

2. Check service status:
```bash
docker-compose ps
```

3. View logs:
```bash
docker-compose logs -f
```

4. Stop services:
```bash
docker-compose down
```

## API Usage

### Authentication

1. Register a new user:
```bash
curl -X POST http://localhost:8000/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "username": "testuser",
    "email": "test@example.com",
    "password": "password123"
  }'
```

2. Get access token:
```bash
curl -X POST http://localhost:8000/v1/auth/token \
  -d "username=testuser&password=password123"
```

Response:
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer"
}
```

### Managing Models

1. Create a model:
```bash
curl -X POST http://localhost:8000/v1/models \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "GPT-2",
    "description": "GPT-2 text generation model",
    "model_type": "text-generation",
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
        "description": "Maximum length of generated text",
        "default": 100,
        "minimum": 1,
        "maximum": 1000
      }
    }
  }'
```

2. List models:
```bash
curl -X GET http://localhost:8000/v1/models
```

3. Get a specific model:
```bash
curl -X GET http://localhost:8000/v1/models/{model_id}
```

### Running Predictions

1. Create a prediction:
```bash
curl -X POST http://localhost:8000/v1/predictions \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "MODEL_ID",
    "input": {
      "prompt": "Once upon a time",
      "max_length": 50
    }
  }'
```

Response:
```json
{
  "id": "pred_abc123",
  "status": "starting",
  "model_id": "MODEL_ID",
  "input": {
    "prompt": "Once upon a time",
    "max_length": 50
  },
  "output": null,
  "created_at": "2024-01-24T10:00:00Z"
}
```

2. Get prediction status:
```bash
curl -X GET http://localhost:8000/v1/predictions/{prediction_id}
```

3. List predictions:
```bash
curl -X GET http://localhost:8000/v1/predictions
```

### Webhooks

To receive notifications when predictions complete, include a webhook URL:

```bash
curl -X POST http://localhost:8000/v1/predictions \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "MODEL_ID",
    "input": {
      "prompt": "Hello world"
    },
    "webhook": "https://your-server.com/webhook"
  }'
```

## Configuration

Key environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `API_HOST` | API server host | 0.0.0.0 |
| `API_PORT` | API server port | 8000 |
| `DATABASE_URL` | PostgreSQL connection string | - |
| `DATABASE_FORCE_IPV4` | Resolve PostgreSQL hostnames to IPv4 only | true |
| `DATABASE_HOSTADDR` | Override PostgreSQL host address (IPv4) | - |
| `REDIS_HOST` | Redis host | localhost |
| `SECRET_KEY` | JWT secret key | - |
| `ENABLE_GPU` | Enable GPU acceleration | true |
| `MODEL_CACHE_DIR` | Directory for cached models | /tmp/model_cache |
| `WORKER_CONCURRENCY` | Celery worker concurrency | 2 |

## Adding Custom Models

To add support for new model types:

1. Create a new model class in `app/models/`:

```python
from .base_model import BaseInferenceModel

class CustomModel(BaseInferenceModel):
    def load(self):
        # Load your model
        pass

    def predict(self, inputs):
        # Run inference
        pass

    def unload(self):
        # Clean up
        pass
```

2. Register the model type in `app/models/model_loader.py`:

```python
model_classes = {
    ModelType.CUSTOM: CustomModel,
    # ...
}
```

3. Add the model type to `app/schemas/model.py`:

```python
class ModelType(str, Enum):
    CUSTOM = "custom"
    # ...
```

## Scaling

### Horizontal Scaling

Scale workers:
```bash
docker-compose up -d --scale worker=4
```

Scale API servers (requires load balancer):
```bash
docker-compose up -d --scale api=3
```

### GPU Support

For GPU support, modify `docker-compose.yml`:

```yaml
worker:
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: 1
            capabilities: [gpu]
```

## Monitoring

Health check endpoint:
```bash
curl http://localhost:8000/health
```

Celery flower (task monitoring):
```bash
celery -A app.workers.celery_app flower
```

## API Documentation

Interactive API documentation is available at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Development

### Running Tests

```bash
pytest tests/
```

## TODO

- Add multipart prediction endpoint to accept JSON + binary in a single request (e.g., `/v1/predictions/multipart`).

### Database Migrations

Create a new migration:
```bash
alembic revision --autogenerate -m "description"
```

Apply migrations:
```bash
alembic upgrade head
```

Rollback:
```bash
alembic downgrade -1
```

## License

MIT License

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.


