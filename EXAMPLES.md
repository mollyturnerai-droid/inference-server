# Usage Examples

This guide provides practical examples of using the Inference Server.

Throughout these examples, set a base URL and reuse it:

```bash
BASE_URL=http://localhost:8000
```

## Example 1: Text Generation with GPT-2

### 1. Start the Server

```bash
docker-compose up -d
```

### 2. Register and Login

```bash
# Register
curl -X POST $BASE_URL/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "username": "alice",
    "email": "alice@example.com",
    "password": "secure123"
  }'

# Login
TOKEN=$(curl -X POST $BASE_URL/v1/auth/token \
  -d "username=alice&password=secure123" | jq -r .access_token)
```

### 3. Create a Text Generation Model

```bash
MODEL_ID=$(curl -X POST $BASE_URL/v1/models \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "GPT-2 Small",
    "description": "GPT-2 model for text generation",
    "model_type": "text-generation",
    "version": "1.0.0",
    "model_path": "gpt2",
    "hardware": "cpu",
    "input_schema": {
      "prompt": {
        "type": "string",
        "description": "Starting text prompt"
      },
      "max_length": {
        "type": "integer",
        "default": 100,
        "minimum": 10,
        "maximum": 500
      },
      "temperature": {
        "type": "number",
        "default": 1.0,
        "minimum": 0.1,
        "maximum": 2.0
      }
    }
  }' | jq -r .id)

echo "Model ID: $MODEL_ID"
```

### 4. Run a Prediction

```bash
PRED_ID=$(curl -X POST http://localhost:8000/v1/predictions \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d "{
    \"model_id\": \"$MODEL_ID\",
    \"input\": {
      \"prompt\": \"Once upon a time in a magical forest\",
      \"max_length\": 100,
      \"temperature\": 0.8
    }
  }" | jq -r .id)

echo "Prediction ID: $PRED_ID"
```

### 5. Check Prediction Status

```bash
# Poll until complete
while true; do
  STATUS=$(curl -s $BASE_URL/v1/predictions/$PRED_ID | jq -r .status)
  echo "Status: $STATUS"

  if [ "$STATUS" = "succeeded" ] || [ "$STATUS" = "failed" ]; then
    break
  fi

  sleep 2
done

# Get final result
curl -s $BASE_URL/v1/predictions/$PRED_ID | jq .
```

## Example 2: Image Generation with Stable Diffusion

### 1. Create an Image Generation Model

```bash
IMG_MODEL_ID=$(curl -X POST $BASE_URL/v1/models \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Stable Diffusion",
    "description": "Text-to-image generation",
    "model_type": "image-generation",
    "version": "1.0.0",
    "model_path": "stabilityai/stable-diffusion-2-1-base",
    "hardware": "gpu",
    "input_schema": {
      "prompt": {
        "type": "string",
        "description": "Text description of image to generate"
      },
      "negative_prompt": {
        "type": "string",
        "description": "What to avoid in the image"
      },
      "num_inference_steps": {
        "type": "integer",
        "default": 50,
        "minimum": 1,
        "maximum": 100
      },
      "guidance_scale": {
        "type": "number",
        "default": 7.5,
        "minimum": 1.0,
        "maximum": 20.0
      },
      "width": {
        "type": "integer",
        "default": 512
      },
      "height": {
        "type": "integer",
        "default": 512
      }
    }
  }' | jq -r .id)
```

### 2. Generate an Image

```bash
curl -X POST $BASE_URL/v1/predictions \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d "{
    \"model_id\": \"$IMG_MODEL_ID\",
    \"input\": {
      \"prompt\": \"A serene mountain landscape at sunset, digital art\",
      \"negative_prompt\": \"blurry, low quality\",
      \"num_inference_steps\": 50,
      \"guidance_scale\": 7.5,
      \"width\": 512,
      \"height\": 512
    }
  }" | jq .
```

## Example 3: Using Webhooks

### 1. Set Up a Webhook Receiver

Create `webhook_receiver.py`:

```python
from flask import Flask, request
import json

app = Flask(__name__)

@app.route('/webhook', methods=['POST'])
def webhook():
    data = request.json
    print("Received webhook:")
    print(json.dumps(data, indent=2))
    return {'status': 'received'}, 200

if __name__ == '__main__':
    app.run(port=5000)
```

Run it:
```bash
python webhook_receiver.py
```

### 2. Create Prediction with Webhook

```bash
curl -X POST $BASE_URL/v1/predictions \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d "{
    \"model_id\": \"$MODEL_ID\",
    \"input\": {
      \"prompt\": \"The future of AI is\"
    },
    \"webhook\": \"http://host.docker.internal:5000/webhook\"
  }"
```

## Example 4: Batch Processing

Process multiple predictions:

```bash
#!/bin/bash

PROMPTS=(
  "The quick brown fox"
  "In a galaxy far far away"
  "Once upon a midnight dreary"
)

for prompt in "${PROMPTS[@]}"; do
  echo "Creating prediction for: $prompt"

  curl -X POST $BASE_URL/v1/predictions \
    -H "Authorization: Bearer $TOKEN" \
    -H "Content-Type: application/json" \
    -d "{
      \"model_id\": \"$MODEL_ID\",
      \"input\": {
        \"prompt\": \"$prompt\",
        \"max_length\": 50
      }
    }" | jq -r .id

  sleep 1
done
```

## Example 5: Python Client

Create `client.py`:

```python
import requests
import time
from typing import Dict, Any

class InferenceClient:
    def __init__(self, base_url: str, token: str):
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }

    def create_prediction(self, model_id: str, input_data: Dict[str, Any], webhook: str = None):
        """Create a new prediction"""
        payload = {
            "model_id": model_id,
            "input": input_data
        }
        if webhook:
            payload["webhook"] = webhook

        response = requests.post(
            f"{self.base_url}/v1/predictions",
            json=payload,
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()

    def get_prediction(self, prediction_id: str):
        """Get prediction status and results"""
        response = requests.get(
            f"{self.base_url}/v1/predictions/{prediction_id}",
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()

    def wait_for_prediction(self, prediction_id: str, timeout: int = 300):
        """Wait for prediction to complete"""
        start_time = time.time()

        while time.time() - start_time < timeout:
            prediction = self.get_prediction(prediction_id)
            status = prediction["status"]

            if status in ["succeeded", "failed", "canceled"]:
                return prediction

            time.sleep(2)

        raise TimeoutError(f"Prediction {prediction_id} did not complete within {timeout} seconds")

    def predict_and_wait(self, model_id: str, input_data: Dict[str, Any]):
        """Create prediction and wait for result"""
        prediction = self.create_prediction(model_id, input_data)
        return self.wait_for_prediction(prediction["id"])

# Usage
if __name__ == "__main__":
    # Login first
    base_url = "http://localhost:8000"
    login_response = requests.post(
        f"{base_url}/v1/auth/token",
        data={"username": "alice", "password": "secure123"}
    )
    token = login_response.json()["access_token"]

    # Create client
    client = InferenceClient(base_url, token)

    # Run prediction
    result = client.predict_and_wait(
        model_id="YOUR_MODEL_ID",
        input_data={
            "prompt": "The meaning of life is",
            "max_length": 100
        }
    )

    print("Result:", result["output"])
```

## Example 6: List and Filter Predictions

```bash
# List all predictions
curl $BASE_URL/v1/predictions \
  -H "Authorization: Bearer $TOKEN" | jq .

# Filter by status
curl "$BASE_URL/v1/predictions?status=succeeded" \
  -H "Authorization: Bearer $TOKEN" | jq .

# Pagination
curl "$BASE_URL/v1/predictions?skip=0&limit=10" \
  -H "Authorization: Bearer $TOKEN" | jq .
```

## Example 7: Model Management

```bash
# List all models
curl $BASE_URL/v1/models | jq .

# Get specific model
curl $BASE_URL/v1/models/$MODEL_ID | jq .

# Delete model (requires authentication)
curl -X DELETE $BASE_URL/v1/models/$MODEL_ID \
  -H "Authorization: Bearer $TOKEN"
```

## Example 8: Error Handling

```python
import requests

def safe_predict(client, model_id, input_data):
    try:
        prediction = client.create_prediction(model_id, input_data)
        result = client.wait_for_prediction(prediction["id"])

        if result["status"] == "failed":
            print(f"Prediction failed: {result.get('error', 'Unknown error')}")
            print(f"Logs: {result.get('logs', 'No logs available')}")
            return None

        return result["output"]

    except requests.HTTPError as e:
        print(f"HTTP error: {e}")
        return None
    except TimeoutError as e:
        print(f"Timeout: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None
```

## Example 9: Custom Model Integration

See the main README for how to add custom model types. Here's a complete example:

```python
# app/models/sentiment_analysis.py
from transformers import pipeline
from .base_model import BaseInferenceModel

class SentimentAnalysisModel(BaseInferenceModel):
    def load(self):
        self.model = pipeline(
            "sentiment-analysis",
            model=self.model_path,
            device=0 if self.device == "cuda" else -1
        )

    def predict(self, inputs):
        text = inputs.get("text", "")
        results = self.model(text)

        return {
            "label": results[0]["label"],
            "score": results[0]["score"],
            "text": text
        }

    def unload(self):
        del self.model
        self.model = None
```

Then use it:

```bash
curl -X POST $BASE_URL/v1/models \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Sentiment Analyzer",
    "model_type": "classification",
    "model_path": "distilbert-base-uncased-finetuned-sst-2-english",
    "input_schema": {
      "text": {
        "type": "string",
        "description": "Text to analyze"
      }
    }
  }'
```

## Example 10: Catalog (curated models) + Mount

List curated models:

```bash
curl -s $BASE_URL/v1/catalog/models | jq .
```

Filter by category (e.g. `text-to-speech`):

```bash
curl -s "$BASE_URL/v1/catalog/models?category=text-to-speech" | jq .
```

Create or update a catalog entry (requires `CATALOG_ADMIN_TOKEN` configured on the server):

```bash
CATALOG_TOKEN=YOUR_CATALOG_ADMIN_TOKEN

curl -s -X POST $BASE_URL/v1/catalog/admin/models \
  -H "Content-Type: application/json" \
  -H "X-Catalog-Admin-Token: $CATALOG_TOKEN" \
  -d '{
    "id": "qwen3-tts",
    "name": "Qwen3 TTS",
    "description": "Text-to-speech model",
    "model_type": "text-to-speech",
    "model_path": "qwen/qwen3-tts",
    "size": "large",
    "vram_gb": 12,
    "recommended_hardware": "gpu",
    "tags": ["tts", "qwen"],
    "downloads": null,
    "license": null
  }' | jq .
```

Mount a catalog entry into a runnable model (creates a row in `/v1/models`):

```bash
curl -s -X POST $BASE_URL/v1/catalog/mount \
  -H "Content-Type: application/json" \
  -d '{
    "catalog_id": "qwen3-tts",
    "name": "qwen3-tts-v2"
  }' | jq .
```

## Example 11: Text-to-Speech (TTS)

Upload reference audio (optional, for voice cloning models that support it):

```bash
UPLOAD_RES=$(curl -s -X POST $BASE_URL/v1/files/upload \
  -F "file=@reference.wav")

REF_PATH=$(echo "$UPLOAD_RES" | jq -r .file_path)
echo "Reference audio path: $REF_PATH"
```

Run a TTS prediction:

```bash
PRED_ID=$(curl -s -X POST $BASE_URL/v1/predictions \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d "{
    \"model_id\": \"$MODEL_ID\",
    \"input\": {
      \"text\": \"Hello! This is a test.\",
      \"reference_audio\": \"$REF_PATH\"
    }
  }" | jq -r .id)

curl -s $BASE_URL/v1/predictions/$PRED_ID | jq .
```

Notes:
- Depending on configuration, `audio_url` may be returned as an absolute URL (when `API_BASE_URL` is set) or as a relative path like `/v1/files/...`.

## Example 12: Authenticated system status

This endpoint requires Bearer auth:

```bash
curl -s $BASE_URL/v1/system/status \
  -H "Authorization: Bearer $TOKEN" | jq .
```

## Tips and Best Practices

1. **Authentication**: Always store tokens securely, never commit them to version control
2. **Error Handling**: Always check prediction status and handle failures gracefully
3. **Timeouts**: Set appropriate timeouts based on your model's complexity
4. **Webhooks**: Use webhooks for long-running predictions instead of polling
5. **Batch Processing**: Process multiple predictions in parallel for better throughput
6. **Model Caching**: Models are cached after first use, subsequent predictions are faster
7. **Resource Management**: Monitor memory usage when loading large models

Environment variables to pay attention to in deployments:
- **API_BASE_URL**: set to your public host if you want absolute file URLs.
- **CATALOG_ADMIN_TOKEN**: enables `/v1/catalog/admin/models`.
- **CORS_ALLOW_ORIGINS**, **CORS_ALLOW_CREDENTIALS**: public/private deployments.
- **TRUST_PROXY_HEADERS**: enable when behind a proxy and you want correct client IP for rate limiting.
- **WEBHOOK_ALLOWED_HOSTS**: comma-separated hostnames; when set, webhooks are only sent to those hosts.

## Troubleshooting

**Prediction stuck in "starting" status:**
- Check worker logs: `docker-compose logs worker`
- Ensure Redis is running: `docker-compose ps redis`

**Out of memory errors:**
- Reduce batch size or model size
- Use CPU instead of GPU for smaller models
- Increase worker memory limits

**Slow predictions:**
- First prediction loads the model (slow), subsequent ones are faster
- Use GPU acceleration when available
- Reduce num_inference_steps for image generation
