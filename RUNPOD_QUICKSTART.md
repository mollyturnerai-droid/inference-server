# RunPod GPU Quick Start (10 Minutes)

Deploy your GPU-accelerated inference server on RunPod.

## Prerequisites

- RunPod account: https://www.runpod.io
- Docker Hub account: https://hub.docker.com

## Step 1: Build and Push Docker Image (5 mins)

```bash
# Login to Docker Hub
docker login

# Build GPU image
docker build -f Dockerfile.gpu -t YOUR_DOCKERHUB_USERNAME/inference-server:gpu .

# Push to Docker Hub
docker push YOUR_DOCKERHUB_USERNAME/inference-server:gpu
```

**Note**: Replace `YOUR_DOCKERHUB_USERNAME` with your actual Docker Hub username.

## Step 2: Deploy on RunPod (2 mins)

1. **Go to RunPod**: https://www.runpod.io/console/pods
2. **Click "Deploy"** or "+ GPU Pod"
3. **Select GPU**:
   - **RTX 3090** (24GB) - Best value ~$0.30/hr
   - **RTX 4090** (24GB) - Faster ~$0.40/hr
   - **A100** - Large models ~$1.00/hr+
4. **Choose Spot or On-Demand**:
   - Spot: Cheaper (50-80% off)
   - On-Demand: Always available
5. **Select Template**: "RunPod PyTorch" or "CUDA 11.8"
6. **Configure**:
   - Container Disk: 50GB
   - Volume Size: 100GB (optional, for persistent storage)
   - Expose HTTP Ports: `8000`

## Step 3: SSH and Setup (3 mins)

Get SSH command from RunPod dashboard (click "Connect" button).

```bash
# SSH into your pod
ssh root@your-pod-id.runpod.io -p 12345

# Clone repository
git clone https://github.com/mollyturnerai-droid/inference-server.git
cd inference-server

# Generate secrets
export SECRET_KEY=$(openssl rand -hex 32)
export POSTGRES_PASSWORD=$(openssl rand -hex 32)
export REDIS_PASSWORD=$(openssl rand -hex 32)

# Create .env file
cp .env.runpod .env

# Update .env with generated secrets
sed -i "s/CHANGE_THIS_RUN_openssl_rand_hex_32/$SECRET_KEY/" .env
sed -i "s/changeme/$POSTGRES_PASSWORD/g" .env
# (Redis password is also in CELERY_BROKER_URL and CELERY_RESULT_BACKEND)

# Or edit manually
nano .env

# Start all services
docker-compose -f docker-compose.runpod.yml up -d

# Wait for services to initialize
sleep 30

# Run database migrations
docker-compose -f docker-compose.runpod.yml exec api alembic upgrade head

# Check status
docker-compose -f docker-compose.runpod.yml ps
```

## Step 4: Test Your API! 🎉

Your API is accessible at:
```
https://your-pod-id-8000.proxy.runpod.net
```

Or find URL in RunPod dashboard under "Connect" → "HTTP Service".

### Test Commands

```bash
# Health check
curl https://your-pod-8000.proxy.runpod.net/health

# Should return: {"status":"healthy"}

# Check GPU
curl https://your-pod-8000.proxy.runpod.net/v1/system/info
# Should show: "gpu_available": true
```

### View API Docs

Visit: `https://your-pod-8000.proxy.runpod.net/docs`

## Quick Reference

### View Logs
```bash
docker-compose -f docker-compose.runpod.yml logs -f api
docker-compose -f docker-compose.runpod.yml logs -f worker
```

### Check GPU
```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

### Restart Services
```bash
docker-compose -f docker-compose.runpod.yml restart
```

### Stop Services
```bash
docker-compose -f docker-compose.runpod.yml down
```

### Start Services
```bash
docker-compose -f docker-compose.runpod.yml up -d
```

## Cost Estimates

| GPU | VRAM | Spot Price | On-Demand |
|-----|------|-----------|-----------|
| RTX 3090 | 24GB | ~$0.30/hr | ~$0.60/hr |
| RTX 4090 | 24GB | ~$0.40/hr | ~$0.80/hr |
| A100 40GB | 40GB | ~$1.00/hr | ~$2.00/hr |

**Monthly estimates** (24/7 usage):
- RTX 3090 Spot: ~$216/month
- RTX 4090 Spot: ~$288/month

**Tip**: Use auto-pause to save money when idle!

## Alternative: Use Pre-built Image

If you don't want to build the image yourself, you can use the GitHub Container Registry once pushed:

```bash
# In docker-compose.runpod.yml, change:
image: ghcr.io/mollyturnerai-droid/inference-server:gpu

# Instead of:
build:
  context: .
  dockerfile: Dockerfile.gpu
```

## Troubleshooting

**Docker Compose not found?**
```bash
apt-get update && apt-get install -y docker-compose-plugin
```

**Port 8000 not accessible?**
- Check RunPod dashboard → Connect → HTTP Service
- Use the proxy URL, not direct IP

**CUDA out of memory?**
- Reduce `MAX_BATCH_SIZE` in .env
- Use smaller models
- Set `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256`

**Services won't start?**
```bash
# Check logs
docker-compose -f docker-compose.runpod.yml logs

# Check disk space
df -h
```

## Next Steps

1. ✅ Test API endpoints via `/docs`
2. ✅ Create user account
3. ✅ Upload a model
4. ✅ Run GPU inference
5. 🎉 Production ready!

## Full Documentation

- Comprehensive guide: `RUNPOD_DEPLOYMENT.md`
- API examples: `EXAMPLES.md`
- Project overview: `README.md`

## Support

- RunPod Discord: https://discord.gg/runpod
- RunPod Docs: https://docs.runpod.io
- GitHub Issues: https://github.com/mollyturnerai-droid/inference-server/issues

---

**Your repository**: https://github.com/mollyturnerai-droid/inference-server
**RunPod Console**: https://www.runpod.io/console/pods
