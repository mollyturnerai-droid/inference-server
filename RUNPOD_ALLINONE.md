# RunPod All-in-One Deployment Guide

This guide explains how to deploy the complete inference server stack in a **single Docker container** on RunPod.

## What's Inside the Container

The all-in-one container includes:
- ✅ **FastAPI** - Main API server
- ✅ **SQLite** - Embedded database (no separate DB server needed)
- ✅ **Redis** - In-container Redis server for job queue
- ✅ **Celery Worker** - Background task processor for ML inference
- ✅ **GPU Support** - CUDA 11.8 for GPU-accelerated inference

## Architecture

```
┌─────────────────────────────────────────────┐
│ Single Docker Container                     │
│                                             │
│  ┌──────────┐  ┌─────────┐  ┌────────────┐ │
│  │  FastAPI │  │  Redis  │  │   Celery   │ │
│  │  (API)   │◄─┤ (Queue) │◄─┤  (Worker)  │ │
│  └────┬─────┘  └─────────┘  └────────────┘ │
│       │                                     │
│       ▼                                     │
│  ┌──────────┐                               │
│  │  SQLite  │                               │
│  │   (DB)   │                               │
│  └──────────┘                               │
│                                             │
│  GPU: NVIDIA CUDA 11.8                      │
└─────────────────────────────────────────────┘
```

## Quick Start

### 1. Pull the Latest Image

```bash
docker pull mollyturnerai/inference-server:latest
```

### 2. Run on RunPod

**Option A: RunPod Web UI**
1. Go to RunPod → Deploy
2. Select your GPU pod
3. Use custom Docker image: `mollyturnerai/inference-server:latest`
4. Set environment variables (optional):
   - `SECRET_KEY` - Your secret key for JWT tokens
   - `ENABLE_GPU` - Set to `true` (default)
5. Deploy!

**Option B: RunPod CLI**
```bash
runpod create pod \
  --name inference-server \
  --image mollyturnerai/inference-server:latest \
  --gpu-type "NVIDIA A40" \
  --env SECRET_KEY=your-secret-key-here
```

### 3. Access Your API

Once deployed, your API will be available at:
```
https://<your-pod-id>.proxy.runpod.net/
```

## Environment Variables

All environment variables are **optional** with sensible defaults:

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8000` | API server port |
| `API_HOST` | `0.0.0.0` | API bind address |
| `WORKERS` | `1` | Uvicorn workers |
| `WORKER_CONCURRENCY` | `2` | Celery worker concurrency |
| `SECRET_KEY` | Auto-generated | JWT secret key |
| `DATABASE_URL` | `sqlite:///./inference.db` | Database URL |
| `REDIS_HOST` | `localhost` | Redis host |
| `REDIS_PORT` | `6379` | Redis port |
| `ENABLE_GPU` | `true` | Enable GPU acceleration |
| `MODEL_CACHE_DIR` | `/tmp/model_cache` | Model cache directory |
| `STORAGE_PATH` | `/tmp/inference_storage` | Storage path |

## Testing the Deployment

### 1. Check API Health

```bash
# Basic health check
curl https://<your-pod>.proxy.runpod.net/health

# Detailed health check (shows all services)
curl https://<your-pod>.proxy.runpod.net/health/detailed
```

Expected response from `/health/detailed`:
```json
{
  "status": "healthy",
  "services": {
    "api": "healthy",
    "database": "healthy",
    "redis": "healthy",
    "gpu": "available: 1 GPU(s) - NVIDIA A40"
  },
  "version": "1.0.0"
}
```

### 2. View API Documentation

Open in your browser:
```
https://<your-pod>.proxy.runpod.net/docs
```

### 3. Register a User

```bash
curl -X POST https://<your-pod>.proxy.runpod.net/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "username": "admin",
    "email": "admin@example.com",
    "password": "securepassword123"
  }'
```

### 4. Login and Get Token

```bash
curl -X POST "https://<your-pod>.proxy.runpod.net/v1/auth/token?username=admin&password=securepassword123"
```

Response:
```json
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGc...",
  "token_type": "bearer"
}
```

### 5. Create a Model

```bash
TOKEN="your-token-here"

curl -X POST https://<your-pod>.proxy.runpod.net/v1/models/ \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{
    "name": "gpt2-text-generator",
    "description": "GPT-2 text generation model",
    "model_type": "text-generation",
    "version": "1.0.0",
    "model_path": "gpt2",
    "hardware": "gpu",
    "input_schema": {
      "prompt": {
        "type": "string",
        "description": "Input text prompt"
      },
      "max_length": {
        "type": "integer",
        "default": 50,
        "minimum": 1,
        "maximum": 512
      }
    }
  }'
```

### 6. Run Inference

```bash
MODEL_ID="<model-id-from-previous-step>"

curl -X POST https://<your-pod>.proxy.runpod.net/v1/predictions/ \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{
    "model_id": "'$MODEL_ID'",
    "input": {
      "prompt": "Once upon a time",
      "max_length": 50
    }
  }'
```

## Features

### ✅ Working Features

- **User Authentication** - JWT-based authentication
- **Model Management** - Register and manage ML models
- **Async Predictions** - Background processing with Celery
- **GPU Acceleration** - CUDA-enabled PyTorch inference
- **Rate Limiting** - Built-in API rate limiting
- **Storage** - Local file storage for models and outputs
- **Health Checks** - Comprehensive service monitoring

### 🔧 Supported Model Types

- `text-generation` - GPT-2, LLaMA, etc.
- `image-generation` - Stable Diffusion, etc.
- `image-to-text` - BLIP, vision transformers
- `text-to-image` - Stable Diffusion, DALL-E
- `classification` - Image/text classifiers
- `custom` - Custom model implementations

## Data Persistence

### SQLite Database

The SQLite database file (`inference.db`) is stored in the container at `/app/inference.db`.

**To persist data across container restarts**, mount a volume:

```bash
docker run -v /path/on/host:/app \
  mollyturnerai/inference-server:latest
```

On RunPod, use persistent storage volumes for production deployments.

### Model Cache

Downloaded models are cached in `/tmp/model_cache` by default. For production:

1. Mount a persistent volume to `/tmp/model_cache`
2. Set `MODEL_CACHE_DIR` environment variable to your volume path

## Scaling

### Vertical Scaling (Single Container)

Increase resources within the container:

```bash
# More Celery workers
WORKER_CONCURRENCY=4

# More API workers (be careful with SQLite)
WORKERS=2
```

### Horizontal Scaling (Multiple Containers)

For production at scale, switch to multi-container architecture:

1. Use PostgreSQL instead of SQLite
2. Use external Redis
3. Separate API and Worker containers
4. See `docker-compose.runpod.yml` for reference

## Troubleshooting

### Check Service Status

```bash
curl https://<your-pod>.proxy.runpod.net/health/detailed
```

### View Container Logs

On RunPod web UI or:
```bash
docker logs <container-id>
```

### Common Issues

**Database locked errors**:
- SQLite doesn't handle high concurrency well
- Solution: Reduce `WORKERS` to 1 or switch to PostgreSQL

**Redis connection errors**:
- Check if Redis started properly in container logs
- Should see "Redis is ready!" in startup logs

**GPU not detected**:
- Ensure RunPod pod has GPU enabled
- Check `nvidia-smi` works in container
- Verify CUDA toolkit is accessible

**Models not loading**:
- Check model cache permissions
- Ensure enough disk space
- Verify model path is correct (HuggingFace model ID or local path)

## Advantages of All-in-One

✅ **Simple Deployment** - One Docker pull, one container run
✅ **No Dependencies** - Everything included, no external services
✅ **Fast Setup** - Running in seconds
✅ **Cost Effective** - Single container = lower costs
✅ **Easy Testing** - Perfect for development and testing

## Limitations

⚠️ **Scalability** - SQLite limits concurrent writes
⚠️ **Persistence** - Data lost if container destroyed (use volumes)
⚠️ **High Availability** - Single point of failure
⚠️ **Performance** - Not optimized for high-traffic production

For production deployments with high load, consider the multi-container architecture.

## Upgrading

### Pull Latest Image

```bash
docker pull mollyturnerai/inference-server:latest
```

### Restart Container

On RunPod, stop and restart your pod with the latest image.

### Database Migrations

If the schema changed:

```bash
# Inside container
alembic upgrade head
```

## Support

- **Documentation**: https://github.com/mollyturnerai-droid/inference-server
- **API Docs**: https://<your-pod>.proxy.runpod.net/docs
- **Issues**: https://github.com/mollyturnerai-droid/inference-server/issues

## Next Steps

1. ✅ Deploy the all-in-one container
2. ✅ Test the health endpoint
3. ✅ Register a user account
4. ✅ Create your first model
5. ✅ Run your first prediction
6. 🚀 Build your ML application!

---

**Ready to deploy?** Pull the image and run it on RunPod now!

```bash
docker pull mollyturnerai/inference-server:latest
```
