# GitHub Actions - Automatic Docker Builds

Your repository is configured to automatically build and push Docker images on every push!

## What Gets Built Automatically

When you push code to GitHub, three Docker images are built:

1. **CPU Image** (`inference-server:cpu`, `inference-server:latest`)
   - For Railway and CPU deployments
   - Built from `Dockerfile`

2. **GPU Image** (`inference-server:gpu`, `inference-server:latest-gpu`)
   - For RunPod and GPU deployments
   - Built from `Dockerfile.gpu`
   - Includes CUDA 11.8 support

3. **Worker Image** (`inference-server:worker`, `inference-server:latest-worker`)
   - For Celery workers
   - Built from `Dockerfile.worker`

## Where Images Are Published

Images are pushed to two registries:

### 1. GitHub Container Registry (Free, Private)
```
ghcr.io/mollyturnerai-droid/inference-server:latest
ghcr.io/mollyturnerai-droid/inference-server:cpu
ghcr.io/mollyturnerai-droid/inference-server:gpu
ghcr.io/mollyturnerai-droid/inference-server:worker
```

### 2. Docker Hub (Optional)
```
YOUR_USERNAME/inference-server:latest
YOUR_USERNAME/inference-server:cpu
YOUR_USERNAME/inference-server:gpu
YOUR_USERNAME/inference-server:worker
```

## Setup Required (One-Time)

### Step 1: Configure Docker Hub (Optional but Recommended)

If you want to push to Docker Hub (easier for RunPod):

1. **Create Docker Hub account**: https://hub.docker.com
2. **Create Access Token**:
   - Go to Account Settings → Security → New Access Token
   - Name it "GitHub Actions"
   - Copy the token
3. **Add to GitHub Secrets**:
   - Go to your repo: https://github.com/mollyturnerai-droid/inference-server
   - Settings → Secrets and variables → Actions
   - Click "New repository secret"
   - Add two secrets:
     - Name: `DOCKERHUB_USERNAME`, Value: your Docker Hub username
     - Name: `DOCKERHUB_TOKEN`, Value: the access token you created

### Step 2: Verify GitHub Token (Automatic)

GitHub Container Registry uses `GITHUB_TOKEN` which is automatically available.
No setup needed! ✅

## How It Works

### Triggers

Builds are triggered on:
- **Push to master/main branch** - Builds and pushes all images
- **Pull request** - Builds images (doesn't push)
- **Git tags** (v1.0.0) - Builds versioned images

### Build Process

```yaml
push to GitHub
    ↓
GitHub Actions triggered
    ↓
Build 3 images in parallel:
  - CPU (Dockerfile)
  - GPU (Dockerfile.gpu)
  - Worker (Dockerfile.worker)
    ↓
Push to registries:
  - GitHub Container Registry (ghcr.io)
  - Docker Hub (if secrets configured)
    ↓
✅ Images ready to deploy!
```

### Image Tags

Every build creates multiple tags:

**For master/main branch:**
- `latest` - Latest master build
- `cpu` / `gpu` / `worker` - Latest by type
- `master-abc123` - Commit SHA

**For version tags (v1.0.0):**
- `1.0.0` - Full version
- `1.0` - Major.minor
- `v1.0.0-gpu` - GPU version

## Using Pre-Built Images

### On RunPod

Instead of building locally, use pre-built images:

```bash
# In docker-compose.runpod.yml, change:
api:
  image: ghcr.io/mollyturnerai-droid/inference-server:gpu
  # Instead of:
  # build:
  #   context: .
  #   dockerfile: Dockerfile.gpu

worker:
  image: ghcr.io/mollyturnerai-droid/inference-server:worker
```

Or use Docker Hub if configured:
```yaml
api:
  image: YOUR_USERNAME/inference-server:gpu
```

### Pulling Images

```bash
# Pull from GitHub Container Registry (may need authentication)
docker pull ghcr.io/mollyturnerai-droid/inference-server:gpu

# Pull from Docker Hub (public)
docker pull YOUR_USERNAME/inference-server:gpu
```

### Authenticating with GitHub Container Registry

```bash
# Create personal access token at: https://github.com/settings/tokens
# Select: read:packages

# Login
echo $GITHUB_TOKEN | docker login ghcr.io -u mollyturnerai-droid --password-stdin
```

## Monitoring Builds

### View Build Status

1. Go to your repository: https://github.com/mollyturnerai-droid/inference-server
2. Click "Actions" tab
3. See all builds and their status

### Build Badges

Add to your README.md:
```markdown
![Docker Build](https://github.com/mollyturnerai-droid/inference-server/actions/workflows/docker-build.yml/badge.svg)
```

## Workflow Configuration

The workflow is in `.github/workflows/docker-build.yml`

### Key Features

- ✅ **Multi-platform builds** - amd64 and arm64 (CPU only)
- ✅ **Build caching** - Faster subsequent builds
- ✅ **Parallel builds** - All images build simultaneously
- ✅ **Automatic tagging** - Smart version tags
- ✅ **PR validation** - Builds on PRs without pushing

### Customizing

Edit `.github/workflows/docker-build.yml` to:
- Change branch triggers
- Modify image tags
- Add build arguments
- Configure platforms

## Troubleshooting

### Build Fails

1. **Check logs**:
   - Go to Actions tab
   - Click on failed build
   - Review logs for errors

2. **Common issues**:
   - Missing dependencies in requirements.txt
   - Invalid Dockerfile syntax
   - Docker Hub secrets not configured

### Can't Push to Docker Hub

```
Error: denied: requested access to the resource is denied
```

**Solution**: Add Docker Hub secrets to GitHub (see Setup Required above)

### Can't Pull Images

**Private repos**: Need authentication
```bash
# GitHub Container Registry
echo $GITHUB_TOKEN | docker login ghcr.io -u USERNAME --password-stdin

# Docker Hub
docker login
```

**Public repos**: No authentication needed

## Cost

- **GitHub Actions**: 2,000 minutes/month free
- **GitHub Container Registry**: 500MB storage free
- **Docker Hub**: Free for public images (1 private repo free)

Build times:
- CPU image: ~5-10 minutes
- GPU image: ~10-15 minutes
- Worker image: ~5-10 minutes

Total per push: ~20-35 minutes of build time

## Benefits

### ✅ No Local Building Required
Push code and images are built automatically!

### ✅ Consistent Builds
Same environment every time, no "works on my machine"

### ✅ Faster Deployments
Pre-built images deploy instantly on RunPod

### ✅ Version Tracking
Every commit gets a unique image tag

### ✅ Easy Rollbacks
Deploy any previous version instantly

## Advanced Usage

### Manual Trigger

Trigger builds manually:
```bash
gh workflow run docker-build.yml
```

Or via GitHub UI: Actions → Build and Push Docker Images → Run workflow

### Build Specific Version

```bash
# Tag and push
git tag v1.0.0
git push origin v1.0.0

# Creates images:
# - inference-server:1.0.0
# - inference-server:1.0
# - inference-server:1.0.0-gpu
```

### Use in CI/CD

```yaml
# Example: Deploy to RunPod after build
deploy:
  needs: build-gpu
  runs-on: ubuntu-latest
  steps:
    - name: Deploy to RunPod
      run: |
        # Your deployment script
```

## Example: Complete Deployment Flow

1. **Develop locally**:
   ```bash
   # Make changes
   git add .
   git commit -m "Add new feature"
   ```

2. **Push to GitHub**:
   ```bash
   git push origin master
   ```

3. **Automatic build** (GitHub Actions):
   - Builds all images (~20 min)
   - Pushes to registries
   - ✅ Done!

4. **Deploy on RunPod**:
   ```bash
   # SSH into RunPod pod
   cd inference-server
   git pull
   docker-compose -f docker-compose.runpod.yml pull
   docker-compose -f docker-compose.runpod.yml up -d
   ```

   Or even simpler:
   ```bash
   # Docker automatically pulls latest image
   docker run -d -p 8000:8000 \
     ghcr.io/mollyturnerai-droid/inference-server:gpu
   ```

## Next Steps

1. ✅ Configure Docker Hub secrets (optional)
2. ✅ Push code to trigger first build
3. ✅ Monitor build in Actions tab
4. ✅ Use pre-built images for deployment

## Support

- GitHub Actions Docs: https://docs.github.com/en/actions
- Docker Docs: https://docs.docker.com
- GitHub Issues: https://github.com/mollyturnerai-droid/inference-server/issues

---

**Your workflow**: `.github/workflows/docker-build.yml`
**View builds**: https://github.com/mollyturnerai-droid/inference-server/actions
