# Deployment Guide

This guide covers deploying the Inference Server to various platforms.

## Quick Start (Docker)

The fastest way to get started is using Docker Compose:

```bash
# Clone the repository
git clone <repository-url>
cd inference_server

# Start all services
docker-compose up -d

# Initialize the database with sample data
docker-compose exec api python scripts/init_db.py

# Check logs
docker-compose logs -f
```

The API will be available at `http://localhost:8000`

## Production Deployment

### Prerequisites

- Docker and Docker Compose
- Domain name (optional but recommended)
- SSL certificate (for HTTPS)
- Cloud provider account (AWS, GCP, Azure, etc.)

### Environment Configuration

1. Copy `.env.example` to `.env`
2. Update critical settings:

```bash
# Generate a secure secret key
SECRET_KEY=$(openssl rand -hex 32)

# Update database credentials
DATABASE_URL=postgresql://user:secure_password@postgres:5432/inference_db

# Configure storage
STORAGE_TYPE=s3  # or 'local'
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
S3_BUCKET=your-bucket-name

# Enable GPU if available
ENABLE_GPU=true
```

### Database Setup

For production, use a managed PostgreSQL instance:

```bash
# AWS RDS
DATABASE_URL=postgresql://user:pass@your-rds-endpoint.amazonaws.com:5432/inference_db

# Google Cloud SQL
DATABASE_URL=postgresql://user:pass@/inference_db?host=/cloudsql/project:region:instance

# Azure Database
DATABASE_URL=postgresql://user:pass@your-server.postgres.database.azure.com:5432/inference_db
```

### Redis Setup

Use managed Redis for production:

```bash
# AWS ElastiCache
REDIS_HOST=your-elasticache-endpoint.amazonaws.com
REDIS_PORT=6379

# Google Cloud Memorystore
REDIS_HOST=your-memorystore-ip
REDIS_PORT=6379

# Azure Cache for Redis
REDIS_HOST=your-cache.redis.cache.windows.net
REDIS_PORT=6380
REDIS_PASSWORD=your_password
```

### Reverse Proxy (Nginx)

Create `nginx.conf`:

```nginx
upstream inference_api {
    server api:8000;
}

server {
    listen 80;
    server_name your-domain.com;

    location / {
        return 301 https://$server_name$request_uri;
    }
}

server {
    listen 443 ssl http2;
    server_name your-domain.com;

    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;

    client_max_body_size 100M;

    location / {
        proxy_pass http://inference_api;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # WebSocket support
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";

        # Timeouts
        proxy_connect_timeout 300s;
        proxy_send_timeout 300s;
        proxy_read_timeout 300s;
    }
}
```

Add Nginx to `docker-compose.yml`:

```yaml
nginx:
  image: nginx:alpine
  ports:
    - "80:80"
    - "443:443"
  volumes:
    - ./nginx.conf:/etc/nginx/conf.d/default.conf
    - ./ssl:/etc/nginx/ssl
  depends_on:
    - api
```

### AWS Deployment

#### Using ECS (Elastic Container Service)

1. Push images to ECR:

```bash
# Login to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin YOUR_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com

# Build and tag images
docker build -t inference-api .
docker tag inference-api:latest YOUR_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/inference-api:latest

docker build -f Dockerfile.worker -t inference-worker .
docker tag inference-worker:latest YOUR_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/inference-worker:latest

# Push images
docker push YOUR_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/inference-api:latest
docker push YOUR_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/inference-worker:latest
```

2. Create ECS task definition
3. Create ECS service with auto-scaling
4. Configure Application Load Balancer

#### Using EC2

```bash
# SSH into EC2 instance
ssh -i your-key.pem ubuntu@your-ec2-ip

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Clone and deploy
git clone <repository-url>
cd inference_server
cp .env.example .env
# Edit .env with your settings
docker-compose up -d
```

### GCP Deployment

#### Using Cloud Run

```bash
# Build and push to GCR
gcloud builds submit --tag gcr.io/YOUR_PROJECT/inference-api
gcloud builds submit --tag gcr.io/YOUR_PROJECT/inference-worker -f Dockerfile.worker

# Deploy API
gcloud run deploy inference-api \
  --image gcr.io/YOUR_PROJECT/inference-api \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 4Gi \
  --cpu 2

# Deploy workers (using Cloud Tasks or Compute Engine)
```

### Azure Deployment

#### Using Container Instances

```bash
# Login to Azure
az login

# Create resource group
az group create --name inference-rg --location eastus

# Create container registry
az acr create --resource-group inference-rg --name inferenceacr --sku Basic

# Build and push images
az acr build --registry inferenceacr --image inference-api:latest .
az acr build --registry inferenceacr --image inference-worker:latest -f Dockerfile.worker .

# Deploy containers
az container create --resource-group inference-rg \
  --name inference-api \
  --image inferenceacr.azurecr.io/inference-api:latest \
  --ports 8000 \
  --cpu 2 --memory 4
```

### Kubernetes Deployment

Create `k8s/deployment.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: inference-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: inference-api
  template:
    metadata:
      labels:
        app: inference-api
    spec:
      containers:
      - name: api
        image: your-registry/inference-api:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: inference-secrets
              key: database-url
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: inference-worker
spec:
  replicas: 2
  selector:
    matchLabels:
      app: inference-worker
  template:
    metadata:
      labels:
        app: inference-worker
    spec:
      containers:
      - name: worker
        image: your-registry/inference-worker:latest
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: inference-secrets
              key: database-url
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
          limits:
            memory: "8Gi"
            cpu: "4"
            nvidia.com/gpu: 1
```

Deploy:

```bash
kubectl apply -f k8s/
```

### GPU Support

For GPU-accelerated inference:

1. Use GPU-enabled instances (p3.2xlarge on AWS, n1-standard-4 with T4 on GCP)
2. Install NVIDIA drivers and NVIDIA Docker runtime
3. Update `docker-compose.yml`:

```yaml
worker:
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: 1
            capabilities: [gpu]
  environment:
    - ENABLE_GPU=true
```

### Monitoring and Logging

#### Prometheus + Grafana

Add to `docker-compose.yml`:

```yaml
prometheus:
  image: prom/prometheus
  volumes:
    - ./prometheus.yml:/etc/prometheus/prometheus.yml
  ports:
    - "9090:9090"

grafana:
  image: grafana/grafana
  ports:
    - "3000:3000"
  environment:
    - GF_SECURITY_ADMIN_PASSWORD=admin
```

#### CloudWatch (AWS)

Install CloudWatch agent on EC2 instances or use ECS logging.

#### Stackdriver (GCP)

Logging is automatic for Cloud Run and GKE.

### Backup and Recovery

Automated PostgreSQL backups:

```bash
# Daily backup cron job
0 2 * * * pg_dump -h localhost -U inference inference_db | gzip > /backups/db_$(date +\%Y\%m\%d).sql.gz
```

### Security

1. Use HTTPS only
2. Enable rate limiting
3. Set up firewall rules
4. Use secrets management (AWS Secrets Manager, GCP Secret Manager)
5. Regular security updates
6. Monitor for suspicious activity

### Scaling Considerations

- **API Servers**: Scale horizontally behind load balancer
- **Workers**: Scale based on queue size
- **Database**: Use read replicas for heavy read workloads
- **Redis**: Use Redis Cluster for high availability
- **Model Cache**: Use shared storage (S3, GCS) for model weights

### Cost Optimization

1. Use spot/preemptible instances for workers
2. Implement auto-scaling based on demand
3. Cache frequently used models
4. Use CDN for static assets
5. Monitor and optimize resource usage

## Troubleshooting

Check logs:
```bash
docker-compose logs -f api
docker-compose logs -f worker
```

Database connection issues:
```bash
docker-compose exec api python -c "from app.db import engine; engine.connect()"
```

Redis connection issues:
```bash
docker-compose exec api python -c "import redis; r = redis.Redis(host='redis'); r.ping()"
```

## Support

For issues and questions, please open an issue on GitHub.
