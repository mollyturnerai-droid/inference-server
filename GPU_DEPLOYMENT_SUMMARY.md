# 🚀 GPU Deployment Ready for RunPod!

Your inference server is now configured for GPU-accelerated inference on RunPod.

## ✅ What's Been Configured

### 1. GPU-Optimized Files
- **`Dockerfile.gpu`** - NVIDIA CUDA 11.8 base image
- **`requirements.gpu.txt`** - PyTorch with CUDA support (+cu118)
- **`docker-compose.runpod.yml`** - Full stack with GPU access
- **`.env.runpod`** - GPU environment configuration
- **`ENABLE_GPU=true`** - GPU inference enabled

### 2. CUDA Configuration
- Base image: `nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04`
- PyTorch: 2.1.2+cu118 (CUDA 11.8)
- GPU memory management configured
- All GPU devices accessible to containers

### 3. Documentation
- **`RUNPOD_QUICKSTART.md`** - 10-minute deployment guide
- **`RUNPOD_DEPLOYMENT.md`** - Comprehensive documentation
- **`DEPLOYMENT_STATUS.txt`** - Full deployment status

### 4. GitHub Repository
- **URL**: https://github.com/mollyturnerai-droid/inference-server
- **All changes pushed** and synchronized
- **Ready to deploy** from GitHub

## 🎯 Quick Deploy to RunPod (10 Minutes)

### Option A: Docker Hub (Recommended)

```bash
# 1. Build and push image (local machine)
docker login
docker build -f Dockerfile.gpu -t YOUR_USERNAME/inference-server:gpu .
docker push YOUR_USERNAME/inference-server:gpu

# 2. Deploy on RunPod
# - Go to https://www.runpod.io/console/pods
# - Select GPU (RTX 3090 recommended, ~$0.30/hr)
# - Choose PyTorch template
# - Configure: 50GB disk, 100GB volume, expose port 8000

# 3. SSH into pod and setup
ssh root@pod-id.runpod.io -p PORT
git clone https://github.com/mollyturnerai-droid/inference-server.git
cd inference-server
cp .env.runpod .env
# Edit .env with your SECRET_KEY
docker-compose -f docker-compose.runpod.yml up -d
docker-compose -f docker-compose.runpod.yml exec api alembic upgrade head

# 4. Access your API
# https://pod-id-8000.proxy.runpod.net/docs
```

### Option B: Direct from GitHub

```bash
# SSH into RunPod pod
git clone https://github.com/mollyturnerai-droid/inference-server.git
cd inference-server

# Build on RunPod (slower but no Docker Hub needed)
docker build -f Dockerfile.gpu -t inference-server:gpu .

# Setup and start
cp .env.runpod .env
nano .env  # Set SECRET_KEY
docker-compose -f docker-compose.runpod.yml up -d
docker-compose -f docker-compose.runpod.yml exec api alembic upgrade head
```

## 💰 Cost Estimates

### GPU Options

| GPU | VRAM | Best For | Spot Price | On-Demand |
|-----|------|----------|------------|-----------|
| **RTX 3090** | 24GB | Most models | ~$0.30/hr | ~$0.60/hr |
| **RTX 4090** | 24GB | Faster inference | ~$0.40/hr | ~$0.80/hr |
| **A100 40GB** | 40GB | Large models | ~$1.00/hr | ~$2.00/hr |
| **A100 80GB** | 80GB | Very large | ~$1.50/hr | ~$3.00/hr |

### Monthly Costs (24/7)
- RTX 3090 Spot: ~$216/month
- RTX 3090 On-Demand: ~$432/month
- RTX 4090 Spot: ~$288/month

**💡 Tip**: Use auto-pause to pay only when actively inferencing!

## 📊 What Can You Run?

With **RTX 3090 (24GB VRAM)**:
- ✅ GPT-2, GPT-Neo (all sizes)
- ✅ BERT, RoBERTa, T5 (large variants)
- ✅ Stable Diffusion 1.5 & 2.1
- ✅ LLaMA 7B (with quantization)
- ✅ Mistral 7B
- ✅ Most HuggingFace models

With **A100 (40GB/80GB VRAM)**:
- ✅ All of the above
- ✅ LLaMA 13B, 30B (with quantization)
- ✅ Falcon 40B (with optimizations)
- ✅ SDXL (Stable Diffusion XL)
- ✅ Multiple models simultaneously

## 🔧 Configuration Details

### Environment Variables (Already Set)

```bash
# GPU Configuration
ENABLE_GPU=true
CUDA_VISIBLE_DEVICES=0
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Performance
WORKER_CONCURRENCY=2
MAX_BATCH_SIZE=8
DEFAULT_TIMEOUT=300

# Storage
MODEL_CACHE_DIR=/tmp/model_cache
STORAGE_TYPE=local
```

### Docker Compose Services

```yaml
services:
  postgres:     # Database for models/predictions
  redis:        # Task queue
  api:          # FastAPI server with GPU access
  worker:       # Celery worker with GPU access
```

All services have GPU access via:
```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: all
          capabilities: [gpu]
```

## 🧪 Testing GPU Inference

### Verify GPU

```bash
# SSH into pod
nvidia-smi

# Test PyTorch
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# API check
curl https://your-pod-8000.proxy.runpod.net/v1/system/info
```

### Load a Model

```bash
# Register user
curl -X POST https://your-pod/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{"username":"test","email":"test@example.com","password":"pass123"}'

# Get token
TOKEN=$(curl -X POST https://your-pod/v1/auth/token \
  -d "username=test&password=pass123" | jq -r .access_token)

# Create model (will use GPU automatically)
curl -X POST https://your-pod/v1/models \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "GPT-2",
    "model_type": "text-generation",
    "model_path": "gpt2",
    "hardware": "gpu"
  }'
```

### Run Inference

```bash
# Create prediction (runs on GPU)
curl -X POST https://your-pod/v1/predictions \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "MODEL_ID_FROM_ABOVE",
    "input": {
      "prompt": "Once upon a time in a galaxy far away",
      "max_length": 100
    }
  }'
```

## 📈 Monitoring

### GPU Usage

```bash
# Real-time monitoring
watch -n 1 nvidia-smi

# Check from API
docker-compose -f docker-compose.runpod.yml exec api python -c "import torch; torch.cuda.mem_get_info()"
```

### Logs

```bash
# API logs
docker-compose -f docker-compose.runpod.yml logs -f api

# Worker logs
docker-compose -f docker-compose.runpod.yml logs -f worker

# All services
docker-compose -f docker-compose.runpod.yml logs -f
```

## 🔒 Security Checklist

Before production:
- [ ] Change `SECRET_KEY` from default
- [ ] Change `POSTGRES_PASSWORD`
- [ ] Change `REDIS_PASSWORD`
- [ ] Set up firewall rules
- [ ] Enable HTTPS (RunPod proxy provides this)
- [ ] Review exposed ports
- [ ] Set up monitoring/alerts

## 🚨 Troubleshooting

### CUDA Out of Memory
```bash
# Reduce batch size
MAX_BATCH_SIZE=2

# Enable gradient checkpointing
# (in model loading code)
```

### Can't Access API
- Use RunPod proxy URL: `https://pod-id-8000.proxy.runpod.net`
- Not direct IP/port (firewall restrictions)
- Check port 8000 is exposed in RunPod settings

### Services Won't Start
```bash
# Check logs
docker-compose -f docker-compose.runpod.yml logs

# Verify GPU access
nvidia-smi

# Check disk space
df -h
```

### Model Download Slow
```bash
# Use persistent volume
# Models cache at /workspace/model_cache
# Persists across restarts

# Or pre-download
python -c "from transformers import AutoModel; AutoModel.from_pretrained('gpt2')"
```

## 📚 Documentation Structure

```
inference_server/
├── RUNPOD_QUICKSTART.md       # Quick 10-minute guide
├── RUNPOD_DEPLOYMENT.md       # Comprehensive RunPod docs
├── GPU_DEPLOYMENT_SUMMARY.md  # This file
├── Dockerfile.gpu             # GPU Docker image
├── docker-compose.runpod.yml  # RunPod stack config
├── .env.runpod               # GPU environment template
├── requirements.gpu.txt      # With CUDA PyTorch
└── README.md                 # General project docs
```

## 🎯 Next Steps

1. **Build Docker image** or use GitHub directly
2. **Deploy on RunPod** with GPU
3. **Setup services** with docker-compose
4. **Run migrations**
5. **Test GPU inference**
6. **Monitor and optimize**
7. **Scale as needed**

## 📞 Support

- **RunPod Discord**: https://discord.gg/runpod
- **RunPod Docs**: https://docs.runpod.io
- **GitHub Issues**: https://github.com/mollyturnerai-droid/inference-server/issues

## 🔗 Important Links

- **GitHub**: https://github.com/mollyturnerai-droid/inference-server
- **RunPod Console**: https://www.runpod.io/console/pods
- **Docker Hub**: https://hub.docker.com

---

## Summary

✅ **GPU support configured**
✅ **CUDA 11.8 with PyTorch**
✅ **Docker Compose ready**
✅ **Documentation complete**
✅ **Code pushed to GitHub**
🚀 **Ready to deploy on RunPod!**

**Estimated deploy time**: 10-15 minutes
**Cost**: ~$0.30/hr (RTX 3090 spot)
**Performance**: 10-50x faster than CPU for large models

---

See **RUNPOD_QUICKSTART.md** to get started! 🎉
