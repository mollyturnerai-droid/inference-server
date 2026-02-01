# Quick Test Guide

Run these commands to test your all-in-one deployment on RunPod.

Replace `<URL>` with your RunPod URL: `https://qghelri26v3kiz-8000.proxy.runpod.net`

## 1. Health Checks

```bash
# Basic health
curl <URL>/health

# Detailed (all services)
curl <URL>/health/detailed
```

## 2. Set API Key

All `/v1/*` endpoints require an API key.

Set an admin key from your environment:

```bash
INFERENCE_SERVER_API_KEY="<your-master-api-key-here>"
```

Or use `API_KEY` (supported for compatibility):

```bash
API_KEY="<your-master-api-key-here>"
```

Or pass the API key directly in each request using `X-API-Key`.

## 3. Create a DB API Key (optional)

This generates a new API key stored in the server DB. The raw key is only returned once.

```bash
curl -X POST <URL>/v1/admin/api-keys \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d '{
    "name": "demo",
    "is_admin": false
  }'
```

Copy the `api_key` from response and use that in requests.

## 4. List Models

```bash
curl <URL>/v1/models/ \
  -H "X-API-Key: $API_KEY"
```

## 5. Create Model

```bash
curl -X POST <URL>/v1/models/ \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d '{
    "name": "gpt2",
    "model_type": "text-generation",
    "model_path": "gpt2",
    "hardware": "gpu",
    "input_schema": {
      "prompt": {"type": "string"}
    }
  }'
```

Copy the `id` from response.

## 6. Run Prediction

```bash
MODEL_ID="<model-id-here>"

curl -X POST <URL>/v1/predictions/ \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d '{
    "model_id": "'$MODEL_ID'",
    "input": {
      "prompt": "Hello, world!"
    }
  }'
```

## Success Criteria

✅ `/health/detailed` shows all services healthy
✅ API key authenticates successfully (no 401)
✅ Model creation works with API key
✅ Predictions are accepted and processed

## Expected Output Examples

**Health Check:**
```json
{
  "status": "healthy",
  "services": {
    "api": "healthy",
    "database": "healthy",
    "redis": "healthy",
    "gpu": "available: 1 GPU(s) - NVIDIA A40"
  }
}
```

**Create API Key:**
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "name": "demo",
  "prefix": "isk_AbCd",
  "api_key": "isk_...",
  "is_admin": false
}
```
