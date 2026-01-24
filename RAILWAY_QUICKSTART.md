# Railway Quick Start (5 Minutes)

Deploy your inference server to Railway in just a few steps!

## Step 1: Prepare Your Code (1 min)

```bash
# Initialize git if not already done
git init
git add .
git commit -m "Initial commit for Railway deployment"
```

## Step 2: Push to GitHub (1 min)

```bash
# Create a new repo on GitHub, then:
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git branch -M main
git push -u origin main
```

## Step 3: Deploy on Railway (2 mins)

1. Go to https://railway.app and sign in
2. Click **"New Project"**
3. Select **"Deploy from GitHub repo"**
4. Choose your repository
5. Railway auto-detects Dockerfile and starts deploying! ✨

## Step 4: Add Database Services (1 min)

### Add PostgreSQL:
1. Click **"New"** → **"Database"** → **"Add PostgreSQL"**
2. ✅ Done! `DATABASE_URL` is automatically set

### Add Redis:
1. Click **"New"** → **"Database"** → **"Add Redis"**
2. ✅ Done! `REDIS_URL` is automatically set

## Step 5: Set Environment Variables (1 min)

In your Railway service, go to **Settings** → **Variables** and add:

```bash
SECRET_KEY=<run: openssl rand -hex 32>
ENABLE_GPU=false
STORAGE_TYPE=local
WORKER_CONCURRENCY=1
```

To generate a secure `SECRET_KEY` locally:
```bash
openssl rand -hex 32
```

## Step 6: Run Database Migration (30 seconds)

Using Railway CLI:
```bash
railway login
railway link
railway run alembic upgrade head
```

OR in Railway Dashboard:
1. Select your service
2. Go to **Settings** → **Deploy**
3. One-time command: `alembic upgrade head`

## Step 7: Test Your API! 🎉

Railway provides a URL like: `https://inference-server-production.up.railway.app`

```bash
# Test health endpoint
curl https://YOUR-APP.railway.app/health

# Should return: {"status": "healthy"}
```

## That's It!

Your API is now live. Check out the full docs:
- Interactive API docs: `https://YOUR-APP.railway.app/docs`
- Full deployment guide: See `RAILWAY_DEPLOYMENT.md`

## Quick Commands

```bash
# View logs
railway logs

# Run migrations
railway run alembic upgrade head

# Connect to database
railway connect postgres

# Run shell in service
railway run bash
```

## Free Tier Limits

- ✅ 512MB RAM (good for small models)
- ✅ 500 hours/month
- ✅ $5 credit/month
- ❌ No GPU (use CPU models only)

**Recommended for free tier:**
- Small models (GPT-2, DistilBERT)
- Set `ENABLE_GPU=false`
- Keep `WORKER_CONCURRENCY=1`

## Troubleshooting

**Service won't start?**
- Check logs: Railway Dashboard → Logs
- Verify environment variables are set
- Make sure PostgreSQL and Redis are added

**Out of memory?**
- Reduce model size
- Lower `WORKER_CONCURRENCY` to 1
- Use smaller batch sizes

**Need help?**
- Railway docs: https://docs.railway.app
- Railway Discord: https://discord.gg/railway

## Upgrade for Production

For production workloads:
- **Hobby Plan ($5/mo)**: Always-on, no sleep
- **Pro Plan ($20/mo)**: 8GB RAM, dedicated CPU
- **Custom**: Contact Railway for GPU support

## Next Steps

1. ✅ Deploy API service
2. ⚙️ Add PostgreSQL and Redis
3. 🔐 Set environment variables
4. 🚀 Run migrations
5. 🎉 Test your API!

Optional:
- Add custom domain
- Deploy Celery worker as separate service
- Set up monitoring
- Configure S3 for persistent storage
