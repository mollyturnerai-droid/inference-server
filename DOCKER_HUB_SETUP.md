# Docker Hub Setup for GitHub Actions

Complete these steps to enable automatic Docker Hub pushes (optional but recommended for RunPod).

## Why Docker Hub?

- **Easier for RunPod**: Public images don't need authentication
- **Faster pulls**: Better CDN/network
- **Widely supported**: Works everywhere
- **Free**: Public images are free

## Setup Steps (5 minutes)

### Step 1: Create Docker Hub Account (1 min)

If you don't have one:
1. Go to https://hub.docker.com/signup
2. Create free account
3. Verify email

### Step 2: Create Access Token (2 mins)

1. **Login to Docker Hub**: https://hub.docker.com
2. **Go to Account Settings**:
   - Click your avatar → Account Settings
   - Or go to: https://hub.docker.com/settings/security
3. **Create New Access Token**:
   - Click "New Access Token"
   - **Description**: "GitHub Actions"
   - **Permissions**: "Read, Write, Delete"
   - Click "Generate"
4. **Copy Token**: Save it somewhere - you won't see it again!

### Step 3: Add Secrets to GitHub (2 mins)

1. **Go to your repository**:
   https://github.com/mollyturnerai-droid/inference-server

2. **Navigate to Secrets**:
   - Click "Settings" tab
   - Click "Secrets and variables" → "Actions"
   - Or go to: https://github.com/mollyturnerai-droid/inference-server/settings/secrets/actions

3. **Add DOCKERHUB_USERNAME**:
   - Click "New repository secret"
   - Name: `DOCKERHUB_USERNAME`
   - Value: Your Docker Hub username
   - Click "Add secret"

4. **Add DOCKERHUB_TOKEN**:
   - Click "New repository secret"
   - Name: `DOCKERHUB_TOKEN`
   - Value: The access token you copied in Step 2
   - Click "Add secret"

### Step 4: Verify Setup ✅

To trigger a build and test:

```bash
# Make a small change
echo "# Docker Hub configured" >> README.md
git add README.md
git commit -m "Test Docker Hub integration"
git push
```

Then:
1. Go to https://github.com/mollyturnerai-droid/inference-server/actions
2. Watch the build run
3. Check Docker Hub: https://hub.docker.com/u/YOUR_USERNAME

You should see images like:
- `YOUR_USERNAME/inference-server:latest`
- `YOUR_USERNAME/inference-server:cpu`
- `YOUR_USERNAME/inference-server:gpu`
- `YOUR_USERNAME/inference-server:worker`

## Using Docker Hub Images

### On RunPod

Update `docker-compose.runpod.yml`:

```yaml
api:
  image: YOUR_USERNAME/inference-server:gpu
  # Remove the 'build' section

worker:
  image: YOUR_USERNAME/inference-server:worker
  # Remove the 'build' section
```

Or run directly:
```bash
docker run -p 8000:8000 YOUR_USERNAME/inference-server:gpu
```

### Pull Images

```bash
# No authentication needed for public images!
docker pull YOUR_USERNAME/inference-server:gpu
docker pull YOUR_USERNAME/inference-server:cpu
docker pull YOUR_USERNAME/inference-server:worker
```

## Troubleshooting

### Error: "denied: requested access to the resource is denied"

**Cause**: Secrets not configured correctly

**Solution**:
1. Verify secrets are named exactly:
   - `DOCKERHUB_USERNAME` (not username or dockerhub_username)
   - `DOCKERHUB_TOKEN` (not password or dockerhub_token)
2. Make sure token has "Read, Write, Delete" permissions
3. Try regenerating the token

### Build Succeeds but No Images on Docker Hub

**Cause**: Build only runs on PRs (doesn't push)

**Solution**:
- Push to master/main branch (not a PR)
- Or add tag: `git tag v1.0.0 && git push origin v1.0.0`

### Can't Find Images on Docker Hub

**Cause**: Repository name might be different

**Solution**:
Check the repository name in workflow matches your Docker Hub username:
```yaml
# In .github/workflows/docker-build.yml
${{ secrets.DOCKERHUB_USERNAME }}/inference-server
```

Should match: `https://hub.docker.com/r/YOUR_USERNAME/inference-server`

## Without Docker Hub (Alternative)

If you skip Docker Hub setup, images still build and push to:

**GitHub Container Registry**:
```
ghcr.io/mollyturnerai-droid/inference-server:gpu
ghcr.io/mollyturnerai-droid/inference-server:cpu
ghcr.io/mollyturnerai-droid/inference-server:worker
```

To use these, you need authentication:
```bash
# Create token at: https://github.com/settings/tokens
# Select: read:packages

echo $GITHUB_TOKEN | docker login ghcr.io -u mollyturnerai-droid --password-stdin
docker pull ghcr.io/mollyturnerai-droid/inference-server:gpu
```

## Benefits of Docker Hub

| Feature | Docker Hub | GitHub Container Registry |
|---------|------------|---------------------------|
| Public access | ✅ No auth needed | ❌ Requires auth |
| Pull speed | ✅ Fast CDN | ✅ Fast |
| Free tier | ✅ Unlimited public | ✅ 500MB free |
| RunPod friendly | ✅ Very easy | ⚠️ Need auth |
| Setup | 5 minutes | Automatic |

**Recommendation**: Set up Docker Hub for easiest RunPod deployment.

## Next Steps

After setup:
1. ✅ Push code to trigger build
2. ✅ Monitor build in Actions tab
3. ✅ Check images on Docker Hub
4. ✅ Use in RunPod deployments

## Support

- Docker Hub Docs: https://docs.docker.com/docker-hub/
- GitHub Actions Docs: https://docs.github.com/en/actions
- Issues: https://github.com/mollyturnerai-droid/inference-server/issues

---

**Repository**: https://github.com/mollyturnerai-droid/inference-server
**Actions**: https://github.com/mollyturnerai-droid/inference-server/actions
**Docker Hub**: https://hub.docker.com/u/YOUR_USERNAME
