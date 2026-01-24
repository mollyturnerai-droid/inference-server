# Railway Deployment Guide

This guide walks you through deploying the Inference Server to Railway.

## Prerequisites

- Railway account (sign up at https://railway.app)
- Git repository (GitHub, GitLab, or Bitbucket)
- Railway CLI (optional, for command-line deployment)

## Quick Deployment

### Option 1: Deploy from GitHub (Recommended)

1. **Initialize Git Repository** (if not already done):
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   ```

2. **Push to GitHub**:
   ```bash
   git remote add origin https://github.com/yourusername/inference_server.git
   git branch -M main
   git push -u origin main
   ```

3. **Deploy on Railway**:
   - Go to https://railway.app
   - Click "New Project"
   - Select "Deploy from GitHub repo"
   - Choose your repository
   - Railway will automatically detect the Dockerfile and deploy

### Option 2: Deploy with Railway CLI

1. **Install Railway CLI**:
   ```bash
   # macOS/Linux
   curl -fsSL https://railway.app/install.sh | sh

   # Windows
   # Download from https://railway.app/cli
   ```

2. **Login to Railway**:
   ```bash
   railway login
   ```

3. **Initialize and Deploy**:
   ```bash
   railway init
   railway up
   ```

## Configuration

### 1. Add PostgreSQL Database

In your Railway project:
1. Click "New" → "Database" → "Add PostgreSQL"
2. Railway will automatically create a `DATABASE_URL` variable
3. Your app will automatically connect to it

### 2. Add Redis

In your Railway project:
1. Click "New" → "Database" → "Add Redis"
2. Railway will create `REDIS_URL` variable
3. Update your app configuration to use `REDIS_URL` (Railway format)

### 3. Environment Variables

Add these environment variables in Railway dashboard:

**Required:**
- `SECRET_KEY` - Generate with: `openssl rand -hex 32`
- `DATABASE_URL` - Automatically set by Railway PostgreSQL
- `REDIS_HOST` - Extract from Railway Redis `REDIS_URL`
- `REDIS_PORT` - Extract from Railway Redis `REDIS_URL` (default: 6379)

**Optional:**
- `STORAGE_TYPE` - Set to `local` for Railway (or `s3` if using AWS S3)
- `ENABLE_GPU` - Set to `false` (Railway doesn't provide GPUs in free tier)
- `MODEL_CACHE_DIR` - `/tmp/model_cache` (default)
- `API_HOST` - `0.0.0.0` (default)
- `WORKER_CONCURRENCY` - `1` or `2` (for free tier)

**Example Environment Variables:**
```
SECRET_KEY=your_secret_key_here
STORAGE_TYPE=local
ENABLE_GPU=false
WORKER_CONCURRENCY=1
```

### 4. Deploy Worker Service (Optional)

For Celery workers, create a separate Railway service:

1. Click "New" → "Empty Service"
2. Connect the same GitHub repository
3. Set custom start command:
   ```
   celery -A app.workers.celery_app worker --loglevel=info --concurrency=1
   ```
4. Use the same environment variables as the API service

## Database Migration

After first deployment, run migrations:

### Using Railway CLI:
```bash
railway run alembic upgrade head
```

### Using Railway Dashboard:
1. Open your service
2. Go to "Settings" → "Deploy"
3. Add one-time command: `alembic upgrade head`

## Limitations on Railway Free Tier

- **Memory**: 512MB RAM (sufficient for small models)
- **CPU**: Shared CPU
- **Storage**: Ephemeral (files deleted on restart)
- **Execution Time**: 500 hours/month
- **No GPU**: GPU not available on free tier

**Recommended for Free Tier:**
- Use small models (GPT-2, small transformers)
- Set `ENABLE_GPU=false`
- Use `STORAGE_TYPE=local` or external S3
- Keep worker concurrency low (1-2)

## Cost Estimates

### Hobby Plan ($5/month)
- 512MB RAM, shared CPU
- 100GB network egress
- Good for testing and light usage

### Pro Plan ($20/month)
- 8GB RAM, dedicated CPU
- 100GB network egress
- Better for production workloads

### Custom Plans
- Contact Railway for GPU support
- Custom resource allocation

## Testing Your Deployment

Once deployed, Railway will provide a URL like `https://inference-server-production.up.railway.app`

Test the API:
```bash
# Health check
curl https://your-app.railway.app/health

# Register user
curl -X POST https://your-app.railway.app/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "username": "testuser",
    "email": "test@example.com",
    "password": "password123"
  }'

# Get token
curl -X POST https://your-app.railway.app/v1/auth/token \
  -d "username=testuser&password=password123"
```

## Scaling

### Horizontal Scaling
Railway supports horizontal scaling on Pro plan:
```bash
railway up --replicas 3
```

### Vertical Scaling
Upgrade resources in Railway dashboard:
- Settings → Resources → Adjust memory/CPU

## Monitoring

View logs in Railway dashboard:
1. Select your service
2. Click "Deployments"
3. Click on latest deployment
4. View real-time logs

Or use Railway CLI:
```bash
railway logs
```

## Custom Domain

Add a custom domain:
1. Go to service Settings → Domains
2. Click "Add Domain"
3. Enter your domain
4. Update DNS records as shown

## Troubleshooting

### Service Won't Start
- Check logs for errors
- Verify all environment variables are set
- Ensure PostgreSQL and Redis are provisioned

### Database Connection Errors
```bash
# Test database connection
railway run python -c "from app.db import engine; engine.connect()"
```

### Redis Connection Errors
- Verify `REDIS_HOST` and `REDIS_PORT` are correct
- Railway Redis URL format: `redis://default:password@host:port`

### Out of Memory
- Reduce model size
- Decrease `WORKER_CONCURRENCY`
- Upgrade to larger Railway plan

### Cold Starts
- Free tier services sleep after inactivity
- Upgrade to Hobby plan for always-on service
- Or implement health check pings

## Maintenance

### Update Deployment
```bash
git add .
git commit -m "Update"
git push
```
Railway auto-deploys on push.

### Rollback
In Railway dashboard:
1. Go to Deployments
2. Click on previous successful deployment
3. Click "Redeploy"

### Database Backup
Railway Pro plan includes automatic backups. For manual backup:
```bash
railway run pg_dump > backup.sql
```

## Support

- Railway Documentation: https://docs.railway.app
- Railway Discord: https://discord.gg/railway
- Project Issues: GitHub Issues

## Next Steps

1. Set up monitoring (Railway provides basic metrics)
2. Configure custom domain
3. Set up webhooks for CI/CD
4. Add worker service for background tasks
5. Configure S3 for persistent storage (if needed)
