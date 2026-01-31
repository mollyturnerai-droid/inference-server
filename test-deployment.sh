#!/bin/bash

# Test script for RunPod deployment
# Usage: API_KEY=... ./test-deployment.sh <your-runpod-url>

URL="${1:-https://qghelri26v3kiz-8000.proxy.runpod.net}"

API_KEY="${API_KEY:-}"

echo "========================================"
echo "Testing Inference Server Deployment"
echo "URL: $URL"
echo "========================================"
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test 1: Basic Health
echo "1. Testing basic health endpoint..."
RESPONSE=$(curl -s -w "\n%{http_code}" --max-time 10 $URL/health 2>&1)
HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
BODY=$(echo "$RESPONSE" | head -n-1)

if [ "$HTTP_CODE" = "200" ]; then
    echo -e "${GREEN}✓ PASS${NC} - Health check returned 200"
    echo "  Response: $BODY"
else
    echo -e "${RED}✗ FAIL${NC} - Health check failed (HTTP $HTTP_CODE)"
    echo "  Response: $BODY"
fi
echo ""

# Test 2: Detailed Health
echo "2. Testing detailed health endpoint..."
RESPONSE=$(curl -s -w "\n%{http_code}" --max-time 10 $URL/health/detailed 2>&1)
HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
BODY=$(echo "$RESPONSE" | head -n-1)

if [ "$HTTP_CODE" = "200" ]; then
    echo -e "${GREEN}✓ PASS${NC} - Detailed health check returned 200"
    echo "  Response: $BODY"
else
    echo -e "${RED}✗ FAIL${NC} - Detailed health check failed (HTTP $HTTP_CODE)"
    echo "  Response: $BODY"
fi
echo ""

# Test 3: API Documentation
echo "3. Testing API documentation..."
RESPONSE=$(curl -s -w "\n%{http_code}" --max-time 10 $URL/docs 2>&1)
HTTP_CODE=$(echo "$RESPONSE" | tail -n1)

if [ "$HTTP_CODE" = "200" ]; then
    echo -e "${GREEN}✓ PASS${NC} - API docs accessible"
else
    echo -e "${RED}✗ FAIL${NC} - API docs failed (HTTP $HTTP_CODE)"
fi
echo ""

# Test 4: List Models
echo "4. Testing model listing..."
RESPONSE=$(curl -s -w "\n%{http_code}" --max-time 10 $URL/v1/models/ -H "X-API-Key: $API_KEY" 2>&1)
HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
BODY=$(echo "$RESPONSE" | head -n-1)

if [ "$HTTP_CODE" = "200" ]; then
    echo -e "${GREEN}✓ PASS${NC} - Model listing successful"
    echo "  Response: $BODY"
else
    echo -e "${RED}✗ FAIL${NC} - Model listing failed (HTTP $HTTP_CODE)"
    echo "  Response: $BODY"
fi
echo ""

# Test 5: Create Model (with API key)
if [ ! -z "$API_KEY" ]; then
    echo "5. Testing model creation..."
    RESPONSE=$(curl -s -w "\n%{http_code}" --max-time 10 -X POST $URL/v1/models/ \
        -H "Content-Type: application/json" \
        -H "X-API-Key: $API_KEY" \
        -d '{
            "name": "test-gpt2",
            "description": "Test GPT-2 model",
            "model_type": "text-generation",
            "version": "1.0.0",
            "model_path": "gpt2",
            "hardware": "gpu",
            "input_schema": {
                "prompt": {
                    "type": "string",
                    "description": "Input text prompt"
                }
            }
        }' 2>&1)
    HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
    BODY=$(echo "$RESPONSE" | head -n-1)

    if [ "$HTTP_CODE" = "200" ]; then
        echo -e "${GREEN}✓ PASS${NC} - Model creation successful"
        MODEL_ID=$(echo "$BODY" | grep -o '"id":"[^"]*"' | cut -d'"' -f4)
        echo "  Model ID: $MODEL_ID"
    else
        echo -e "${RED}✗ FAIL${NC} - Model creation failed (HTTP $HTTP_CODE)"
        echo "  Response: $BODY"
    fi
    echo ""
else
    echo -e "${YELLOW}! SKIP${NC} - Set API_KEY env var to test authenticated endpoints"
    echo ""
fi

# Summary
echo "========================================"
echo "Test Summary"
echo "========================================"
echo ""
echo "API URL: $URL"
echo "Docs: $URL/docs"
echo ""

if [ ! -z "$TOKEN" ]; then
    echo "Test Credentials:"
    echo "  Username: $USERNAME"
    echo "  Password: testpass123"
    echo "  Token: ${TOKEN:0:50}..."
    echo ""
fi

echo "Next Steps:"
echo "1. Check container logs on RunPod for any errors"
echo "2. Visit $URL/docs to see interactive API documentation"
echo "3. Use the token above to test authenticated endpoints"
echo ""
