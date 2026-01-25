#!/bin/bash
set -e

echo "Starting Inference Server - All-in-One Mode"

# Start Redis in the background
echo "Starting Redis..."
redis-server --daemonize yes --port 6379 --bind 0.0.0.0

# Wait for Redis to be ready
echo "Waiting for Redis..."
for i in {1..10}; do
    if redis-cli ping > /dev/null 2>&1; then
        echo "Redis is ready!"
        break
    fi
    echo "Waiting for Redis to start... ($i/10)"
    sleep 1
done

# Initialize database (create tables)
echo "Initializing database..."
python3.11 -c "from app.db import engine, Base; Base.metadata.create_all(bind=engine); print('Database initialized')"

# Start Celery worker in the background
echo "Starting Celery worker..."
python3.11 -m celery -A app.workers.celery_app worker --loglevel=info --concurrency=${WORKER_CONCURRENCY:-2} &

# Start the FastAPI server
echo "Starting FastAPI server..."
exec python3.11 -m uvicorn app.main:app --host ${API_HOST:-0.0.0.0} --port ${PORT:-8000} --workers ${WORKERS:-1}
