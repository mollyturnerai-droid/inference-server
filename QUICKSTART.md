# Quick Start Guide

Get the Inference Server running in 5 minutes!

## Prerequisites

- Docker and Docker Compose installed
- 4GB+ RAM available
- Port 8000 available

## Steps

### 1. Start the Server

```bash
# Start all services (PostgreSQL, Redis, API, Workers)
docker-compose up -d

# Wait for services to be ready (about 30 seconds)
docker-compose logs -f api
# Press Ctrl+C when you see "Application startup complete"
```

### 2. Initialize Database

```bash
# Create sample data (admin user and GPT-2 model)
docker-compose exec api python scripts/init_db.py
```

This creates:
- Admin user: `admin` / `admin123`
- Sample GPT-2 model ready to use

### 3. Get Access Token

```bash
# Login and save token
TOKEN=$(curl -s -X POST http://localhost:8000/v1/auth/token \
  -d "username=admin&password=admin123" | jq -r .access_token)

echo "Your token: $TOKEN"
```

### 4. List Available Models

```bash
# Get the GPT-2 model ID
curl -s http://localhost:8000/v1/models | jq '.models[] | {id, name, model_type}'
```

Copy the model ID from the output.

### 5. Run Your First Prediction

```bash
# Replace MODEL_ID_HERE with the actual model ID from step 4
curl -X POST http://localhost:8000/v1/predictions \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "MODEL_ID_HERE",
    "input": {
      "prompt": "Once upon a time",
      "max_length": 50
    }
  }' | jq .
```

You'll get a response with a prediction ID.

### 6. Check Prediction Result

```bash
# Replace PREDICTION_ID with the ID from step 5
curl -s http://localhost:8000/v1/predictions/PREDICTION_ID | jq .
```

The first prediction takes longer (loading the model). Subsequent predictions are much faster!

## One-Liner Test

```bash
# Complete test in one command
docker-compose up -d && sleep 30 && \
docker-compose exec api python scripts/init_db.py && \
TOKEN=$(curl -s -X POST http://localhost:8000/v1/auth/token -d "username=admin&password=admin123" | jq -r .access_token) && \
MODEL_ID=$(curl -s http://localhost:8000/v1/models | jq -r '.models[0].id') && \
PRED_ID=$(curl -s -X POST http://localhost:8000/v1/predictions -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" -d "{\"model_id\": \"$MODEL_ID\", \"input\": {\"prompt\": \"Hello world\"}}" | jq -r .id) && \
echo "Prediction created: $PRED_ID" && \
sleep 10 && \
curl -s http://localhost:8000/v1/predictions/$PRED_ID | jq .
```

## What's Running?

```bash
# Check service status
docker-compose ps

# Should show:
# - postgres (database)
# - redis (task queue)
# - api (REST API server)
# - worker (2 replicas - processes inference tasks)
```

## View Logs

```bash
# All services
docker-compose logs -f

# Just API
docker-compose logs -f api

# Just workers
docker-compose logs -f worker
```

## API Documentation

Open your browser to:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Stop the Server

```bash
docker-compose down

# Or to also remove data:
docker-compose down -v
```

## Next Steps

- Read [README.md](README.md) for complete documentation
- Check [EXAMPLES.md](EXAMPLES.md) for usage examples
- See [DEPLOYMENT.md](DEPLOYMENT.md) for production deployment

## Troubleshooting

**Services won't start:**
```bash
# Check logs
docker-compose logs

# Restart
docker-compose restart
```

**Port 8000 already in use:**
Edit `docker-compose.yml` and change `8000:8000` to `8080:8000`, then access at http://localhost:8080

**Out of memory:**
Reduce worker replicas in `docker-compose.yml` from 2 to 1

**Cannot connect to database:**
```bash
# Rebuild everything
docker-compose down -v
docker-compose up -d --build
```

## Common Commands

```bash
# Restart services
docker-compose restart

# View resource usage
docker stats

# Enter API container
docker-compose exec api bash

# Run database migrations
docker-compose exec api alembic upgrade head

# Create new user
curl -X POST http://localhost:8000/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{"username": "yourname", "email": "you@example.com", "password": "yourpass"}'
```

## Health Check

```bash
curl http://localhost:8000/health
# Should return: {"status": "healthy"}
```

That's it! You now have a fully functional ML inference server running. 🚀
