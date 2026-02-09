# RunPod GPU Deployment Guide

Deploy your GPU-accelerated inference server on RunPod with CUDA support.

## Why RunPod for GPU?

- **Cheap GPU instances** (~$0.20-$0.40/hour for RTX 3090/4090)
- **On-demand & Spot pricing** available
- **Pre-configured CUDA** environment
- **Docker-native** deployment
- **Persistent storage** options
- **Direct SSH access** for debugging

## Prerequisites

- RunPod account: https://www.runpod.io
- Docker Hub account (or GitHub Container Registry)
- Your code on GitHub: https://github.com/mollyturnerai-droid/inference-server

## Deployment Options

### Option 1: Docker Compose (Recommended)
Complete stack with PostgreSQL, Redis, API, and Workers

### Option 2: RunPod Template
Custom template for quick deployment

### Option 3: Manual Setup
SSH into pod and configure manually

---

## Option 1: Docker Compose Deployment

### Step 1: Prepare Docker Image

You can either:

**A) Use Docker Hub (Public/Private)**

```bash
# Login to Docker Hub
docker login

# Build GPU image
docker build -f Dockerfile.gpu -t yourusername/inference-server:gpu .

# Push to Docker Hub
docker push yourusername/inference-server:gpu
```

Optional: build the NeMo/Magpie TTS enabled GPU image variant:

```bash
docker build -f Dockerfile.gpu.nemo -t yourusername/inference-server:gpu.nemo .
docker push yourusername/inference-server:gpu.nemo
```

Note: `Dockerfile.gpu.nemo` uses an `nvcr.io/nvidia/pytorch` base image.

**B) Use GitHub Container Registry**

```bash
# Login to GitHub Container Registry
echo $GITHUB_TOKEN | docker login ghcr.io -u mollyturnerai-droid --password-stdin

# Build and tag
docker build -f Dockerfile.gpu -t ghcr.io/mollyturnerai-droid/inference-server:gpu .

# Push
docker push ghcr.io/mollyturnerai-droid/inference-server:gpu
```

### Step 2: Create RunPod Pod

1. **Go to RunPod**: https://www.runpod.io/console/pods
2. **Select GPU**:
   - RTX 3090 (24GB VRAM) - ~$0.30/hr spot
   - RTX 4090 (24GB VRAM) - ~$0.40/hr spot
   - A100 (40GB/80GB) - ~$1.00/hr+ (for large models)
3. **Select Template**: "RunPod PyTorch" or "CUDA 11.8"
4. **Configure**:
   - Container Disk: 50GB+
   - Volume: 100GB+ (for models)
   - Ports: Expose 8000

### Step 3: Setup on RunPod

SSH into your pod (get SSH command from RunPod dashboard):

```bash
# SSH into pod
ssh root@pod-id.runpod.io -p 12345

# Clone your repository
git clone https://github.com/mollyturnerai-droid/inference-server.git
cd inference-server

# Generate SECRET_KEY
export SECRET_KEY=$(openssl rand -hex 32)

# Copy environment file
cp .env.runpod .env

# Update .env with your SECRET_KEY
sed -i "s/CHANGE_THIS_RUN_openssl_rand_hex_32/$SECRET_KEY/" .env

# Update passwords in .env
nano .env  # Change POSTGRES_PASSWORD and REDIS_PASSWORD

# Start all services
docker-compose -f docker-compose.runpod.yml up -d

# If using the NeMo/Magpie image variant, set IMAGE_TAG before starting:
# export IMAGE_TAG=gpu.nemo
# docker-compose -f docker-compose.runpod.yml up -d

# Wait for services to start (30 seconds)
sleep 30

# Run database migrations
docker-compose -f docker-compose.runpod.yml exec api alembic upgrade head

# Check status
docker-compose -f docker-compose.runpod.yml ps

# View logs
docker-compose -f docker-compose.runpod.yml logs -f
```

### Step 4: Access Your API

Your API is accessible at:
```
https://pod-id-8000.proxy.runpod.net
```

Or use the public IP with port:
```
http://PUBLIC_IP:8000
```

Find these in RunPod dashboard under "Connect" button.

---

## Option 2: RunPod Template (Fastest)

### Create Custom Template

1. **Build and push Docker image** (see Option 1, Step 1)

2. **Create Template** in RunPod:
   - Go to https://www.runpod.io/console/user/templates
   - Click "New Template"
   - **Container Image**: `yourusername/inference-server:gpu`
   - **Container Disk**: 50GB
   - **Expose HTTP Ports**: `8000`
   - **Docker Command**:
     ```bash
     bash -c "alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port 8000"
     ```
   - **Environment Variables**:
     ```
     ENABLE_GPU=true
     SECRET_KEY=<your_secret_key>
     DATABASE_URL=<external_db_url>
     REDIS_HOST=<external_redis_host>
     REDIS_PORT=6379
     ```

3. **Deploy from Template**:
   - Select GPU
   - Choose your template
   - Deploy!

**Note**: This option requires external PostgreSQL and Redis (e.g., Railway, Neon, Upstash).

---

## Option 3: Manual Setup (Most Control)

### Step 1: Deploy Basic Pod

1. Select GPU on RunPod
2. Use "RunPod PyTorch" template
3. Deploy and SSH in

### Step 2: Install Dependencies

```bash
# Update system
apt-get update && apt-get install -y git postgresql-client redis-tools

# Clone repository
git clone https://github.com/mollyturnerai-droid/inference-server.git
cd inference-server

# Install Python dependencies
pip install -r requirements.gpu.txt
```

### Step 3: Setup Services

**Option A: Use External Services**
```bash
# Use managed PostgreSQL (Railway, Neon, Supabase)
export DATABASE_URL="postgresql://user:pass@host:5432/db"

# Use managed Redis (Upstash, Railway)
export REDIS_HOST="your-redis-host"
export REDIS_PORT=6379
```

**Option B: Run Local Services**
```bash
# Install and start PostgreSQL
apt-get install -y postgresql
service postgresql start
sudo -u postgres createdb inference_db

# Install and start Redis
apt-get install -y redis-server
service redis-server start
```

### Step 4: Configure and Run

```bash
# Copy and edit environment file
cp .env.runpod .env
nano .env  # Update with your configuration

# Run migrations
alembic upgrade head

# Start API
uvicorn app.main:app --host 0.0.0.0 --port 8000 &

# Start worker (in background)
celery -A app.workers.celery_app worker --loglevel=info --concurrency=2 &
```

---

## GPU Configuration

### Verify GPU Access

```bash
# Check NVIDIA driver
nvidia-smi

# Test PyTorch GPU
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); print(f'GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"
```

Expected output:
```
CUDA available: True
GPU count: 1
GPU name: NVIDIA GeForce RTX 3090
```

### GPU Environment Variables

Already set in `.env.runpod`:
```bash
ENABLE_GPU=true
CUDA_VISIBLE_DEVICES=0
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

### GPU Memory Management

```bash
# Monitor GPU usage
watch -n 1 nvidia-smi

# Clear GPU cache in Python
python -c "import torch; torch.cuda.empty_cache()"
```

---

## Testing Your Deployment

### 1. Health Check

```bash
curl https://your-pod-8000.proxy.runpod.net/health
```

### 2. Verify GPU

```bash
curl https://your-pod-8000.proxy.runpod.net/v1/system/info
```

Should show: `"gpu_available": true`

### 3. API Documentation

Visit: `https://your-pod-8000.proxy.runpod.net/docs`

### 4. Test Inference

```bash
# Register user
curl -X POST https://your-pod/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{"username":"test","email":"test@example.com","password":"pass123"}'

# Get token
TOKEN=$(curl -X POST https://your-pod/v1/auth/token \
  -d "username=test&password=pass123" | jq -r .access_token)

# Create a model (GPT-2 example)
curl -X POST https://your-pod/v1/models \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "GPT-2",
    "model_type": "text-generation",
    "model_path": "gpt2",
    "hardware": "gpu"
  }'

# Run prediction
curl -X POST https://your-pod/v1/predictions \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "MODEL_ID",
    "input": {"prompt": "Once upon a time"}
  }'
```

---

## Cost Optimization

### Choose Right GPU

| GPU | VRAM | Best For | Spot Price |
|-----|------|----------|------------|
| RTX 3090 | 24GB | Most models | ~$0.30/hr |
| RTX 4090 | 24GB | Faster inference | ~$0.40/hr |
| A100 40GB | 40GB | Large models | ~$1.00/hr |
| A100 80GB | 80GB | Very large models | ~$1.50/hr |

### Use Spot Instances

- **Spot**: 50-80% cheaper, can be interrupted
- **On-demand**: Always available, more expensive
- **Recommendation**: Use spot for development, on-demand for production

### Auto-pause

RunPod can automatically pause your pod when idle:
- Set in pod settings
- Saves money during inactivity
- Resumes on API call

### Use Persistent Storage

```bash
# Mount volume for models
docker-compose -f docker-compose.runpod.yml up -d

# Models cached at /workspace/model_cache
# Persists across pod restarts
```

---

## Persistent Storage

### Volume Setup

1. **Create Volume** in RunPod:
   - Go to Storage → Volumes
   - Create new volume (100GB+)
   - Mount at `/workspace`

2. **Use in Docker Compose**:
```yaml
volumes:
  - /workspace/model_cache:/tmp/model_cache
  - /workspace/storage:/tmp/inference_storage
  - /workspace/postgres:/var/lib/postgresql/data
```

3. **Benefits**:
   - Models persist across pod restarts
   - Faster startup (no re-download)
   - Database data preserved

---

## Monitoring

### View Logs

```bash
# Docker Compose
docker-compose -f docker-compose.runpod.yml logs -f api
docker-compose -f docker-compose.runpod.yml logs -f worker

# Direct
journalctl -u inference-server -f
```

### GPU Monitoring

```bash
# Real-time GPU usage
watch -n 1 nvidia-smi

# GPU utilization over time
nvidia-smi dmon -s u
```

### API Metrics

Visit: `https://your-pod-8000.proxy.runpod.net/metrics`

---

## Troubleshooting

### CUDA Out of Memory

```python
# Reduce batch size
MAX_BATCH_SIZE=2

# Enable memory optimization
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
```

### Port Not Accessible

- Check RunPod firewall settings
- Verify port 8000 is exposed
- Use RunPod proxy URL, not direct IP

### Model Download Slow

```bash
# Pre-download models
python -c "from transformers import AutoModel; AutoModel.from_pretrained('gpt2')"
```

### Container Crashes

```bash
# Check logs
docker-compose -f docker-compose.runpod.yml logs api

# Increase memory limits
docker-compose -f docker-compose.runpod.yml up -d --scale api=1 --memory="16g"
```

---

## Scaling

### Horizontal Scaling

```bash
# Scale workers
docker-compose -f docker-compose.runpod.yml up -d --scale worker=3

# Use load balancer for multiple pods
```

### Model Parallelism

For very large models:
```python
# Use multiple GPUs
CUDA_VISIBLE_DEVICES=0,1
```

---

## Security

### Firewall

```bash
# Only allow necessary ports
ufw allow 8000
ufw allow 22
ufw enable
```

### Authentication

- Always use strong SECRET_KEY
- Enable HTTPS (RunPod proxy provides SSL)
- Use environment variables, not hardcoded secrets

### Database Security

```bash
# Change default passwords
POSTGRES_PASSWORD=$(openssl rand -hex 32)
REDIS_PASSWORD=$(openssl rand -hex 32)
```

---

## Comparison: RunPod vs Railway

| Feature | RunPod | Railway |
|---------|---------|---------|
| GPU Support | ✅ Yes | ❌ No (free tier) |
| Cost (GPU) | ~$0.30/hr | N/A |
| Cost (CPU) | ~$0.02/hr | $5/month free |
| Setup | More complex | Very simple |
| Best For | GPU inference | CPU/small models |

**Recommendation**: Use RunPod for GPU-accelerated inference with large models.

---

## Next Steps

1. ✅ Push GPU image to Docker Hub
2. ✅ Deploy pod on RunPod
3. ✅ Setup services with docker-compose
4. ✅ Run migrations
5. ✅ Test GPU inference
6. 🎉 Production ready!

## Support

- RunPod Docs: https://docs.runpod.io
- RunPod Discord: https://discord.gg/runpod
- GitHub Issues: https://github.com/mollyturnerai-droid/inference-server/issues

---

**Your repository**: https://github.com/mollyturnerai-droid/inference-server
**RunPod Console**: https://www.runpod.io/console/pods
