# 🚀 Deploy Now - Complete Instructions

Your code is ready! Follow these steps to deploy to Railway.

## Current Status ✅

- ✅ Git repository initialized
- ✅ Initial commit created (user: mollyturnerai-droid)
- ✅ Railway configuration files added
- ✅ Dockerfile optimized for Railway
- ✅ Environment templates created

## Deployment Steps

### Step 1: Create GitHub Repository (2 minutes)

1. **Go to GitHub**: https://github.com/new
2. **Repository settings**:
   - Repository name: `inference-server` (or your preferred name)
   - Description: "ML Inference Server - FastAPI + Celery + PyTorch"
   - Keep it Public or Private (your choice)
   - ⚠️ **DO NOT** check "Initialize with README" (we already have one)
3. **Click "Create repository"**

### Step 2: Push Your Code (1 minute)

After creating the repository, GitHub will show you commands. Use these in your terminal:

```bash
# Add remote (replace YOUR_USERNAME with actual username)
git remote add origin https://github.com/mollyturnerai-droid/inference-server.git

# Rename branch to main
git branch -M main

# Push code
git push -u origin main
```

**Expected output**: Your code is now on GitHub! ✨

### Step 3: Deploy on Railway (3 minutes)

1. **Go to Railway**: https://railway.app
2. **Sign in** (use GitHub to sign in for easy integration)
3. **Create new project**:
   - Click **"New Project"**
   - Select **"Deploy from GitHub repo"**
   - Find and select: `mollyturnerai-droid/inference-server`
   - Click **"Deploy Now"**

Railway will:
- ✅ Detect your Dockerfile automatically
- ✅ Start building the image
- ✅ Deploy your API service
- ✅ Provide a public URL

### Step 4: Add PostgreSQL (30 seconds)

In your Railway project dashboard:
1. Click **"New"** button
2. Select **"Database"**
3. Choose **"Add PostgreSQL"**
4. ✅ Done! `DATABASE_URL` is automatically set for your service

### Step 5: Add Redis (30 seconds)

In your Railway project dashboard:
1. Click **"New"** button
2. Select **"Database"**
3. Choose **"Add Redis"**
4. ✅ Done! `REDIS_URL` is automatically set

### Step 6: Configure Environment Variables (2 minutes)

1. **Click on your API service** (the one with your code)
2. Go to **"Variables"** tab
3. Click **"+ New Variable"** for each of these:

```bash
# REQUIRED - Generate this locally first
SECRET_KEY=<your_generated_key>

# Performance settings for free tier
ENABLE_GPU=false
STORAGE_TYPE=local
WORKER_CONCURRENCY=1
MAX_BATCH_SIZE=4
MODEL_CACHE_DIR=/tmp/model_cache

# Optional
DEFAULT_TIMEOUT=300
ACCESS_TOKEN_EXPIRE_MINUTES=30
ALGORITHM=HS256
```

**To generate SECRET_KEY** (run this in Git Bash or WSL):
```bash
openssl rand -hex 32
```

Or use this PowerShell command:
```powershell
-join ((48..57) + (65..90) + (97..122) | Get-Random -Count 64 | ForEach-Object {[char]$_})
```

4. Click **"Save"** - Railway will redeploy automatically

### Step 7: Run Database Migration (1 minute)

You have two options:

**Option A: Install Railway CLI** (recommended)
```bash
# Install via npm
npm install -g @railway/cli

# Login
railway login

# Link to your project
railway link

# Run migration
railway run alembic upgrade head
```

**Option B: Via Railway Dashboard**
1. Click on your service
2. Go to **"Settings"** → **"Deploy"**
3. Under "One-off Commands", run: `alembic upgrade head`
4. Watch the logs to confirm it completes

### Step 8: Test Your API! 🎉

Railway provides a URL like: `https://inference-server-production-xxxx.up.railway.app`

Find it in your service's **"Settings"** → **"Domains"**

```bash
# Test health endpoint
curl https://YOUR-APP.railway.app/health

# Should return: {"status":"healthy"}
```

Visit your API docs:
```
https://YOUR-APP.railway.app/docs
```

## 🎯 Quick Commands After Deployment

```bash
# View logs
railway logs

# Run migrations
railway run alembic upgrade head

# Check service status
railway status

# Open in browser
railway open
```

## 📊 What You Get on Free Tier

- ✅ 512MB RAM
- ✅ Shared CPU
- ✅ $5 credit/month (~500 hours)
- ✅ Public HTTPS URL
- ✅ Automatic SSL
- ✅ PostgreSQL database (1GB)
- ✅ Redis cache
- ❌ No GPU (use CPU models only)

## ⚠️ Troubleshooting

**Build fails?**
- Check Railway build logs
- Verify Dockerfile is in root directory
- Ensure all files were pushed to GitHub

**Service won't start?**
- Check that DATABASE_URL is set (should be automatic)
- Check that REDIS_URL is set (should be automatic)
- Verify SECRET_KEY is set
- View logs in Railway dashboard

**Out of memory?**
- Set `WORKER_CONCURRENCY=1`
- Set `MAX_BATCH_SIZE=2`
- Use smaller models only

**Database connection error?**
- Ensure PostgreSQL service is running
- Check that DATABASE_URL variable exists
- Try redeploying

## 🚀 Optional: Add Worker Service

For background Celery workers:

1. In Railway project, click **"New"** → **"Service"**
2. Select same GitHub repo
3. Go to **"Settings"** → **"Deploy"**
4. Set start command:
   ```
   celery -A app.workers.celery_app worker --loglevel=info --concurrency=1
   ```
5. Add same environment variables as API service

## 📈 Monitoring

Check your deployment:
- **Logs**: Railway Dashboard → Select Service → Logs
- **Metrics**: Railway Dashboard → Select Service → Metrics
- **Health**: `https://YOUR-APP.railway.app/health`

## 💰 Cost Estimates

- **Free tier**: $5/month credit (~500 hours)
- **Hobby**: $5/month (always on, no sleep)
- **Pro**: $20/month (8GB RAM, better performance)

## 🎓 Next Steps

After deployment works:
1. ✅ Test API endpoints via `/docs`
2. ✅ Create a user account
3. ✅ Upload and test a model
4. ✅ Run a prediction
5. 🎉 Your inference server is live!

## 📚 More Resources

- Full Railway docs: `RAILWAY_DEPLOYMENT.md`
- Quick start guide: `RAILWAY_QUICKSTART.md`
- API examples: `EXAMPLES.md`
- Project overview: `README.md`

---

**Need help?** Check the troubleshooting section or Railway's Discord: https://discord.gg/railway
