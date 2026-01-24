.PHONY: help install dev run-api run-worker docker-build docker-up docker-down test clean

help:
	@echo "Available commands:"
	@echo "  make install      - Install dependencies"
	@echo "  make dev          - Setup development environment"
	@echo "  make run-api      - Run API server"
	@echo "  make run-worker   - Run Celery worker"
	@echo "  make docker-build - Build Docker images"
	@echo "  make docker-up    - Start Docker services"
	@echo "  make docker-down  - Stop Docker services"
	@echo "  make test         - Run tests"
	@echo "  make clean        - Clean up generated files"

install:
	pip install -r requirements.txt

dev:
	cp .env.example .env
	@echo "Please edit .env with your configuration"

run-api:
	python -m app.main

run-worker:
	celery -A app.workers.celery_app worker --loglevel=info

docker-build:
	docker-compose build

docker-up:
	docker-compose up -d
	@echo "Waiting for services to be ready..."
	@sleep 10
	docker-compose exec api python scripts/init_db.py

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

test:
	pytest tests/ -v

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build dist .pytest_cache .coverage
