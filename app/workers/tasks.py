from celery import Task
from .celery_app import celery_app
from app.models.model_loader import model_loader
from app.db import SessionLocal, Prediction
from app.schemas import PredictionStatus
from datetime import datetime
import logging
import time
import traceback
import httpx
from urllib.parse import urlparse
from app.core.config import settings
import multiprocessing as mp

# Ensure CUDA is initialized in a safe process model for workers.
if settings.ENABLE_GPU:
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

import torch

logger = logging.getLogger(__name__)

class DatabaseTask(Task):
    _db = None

    @property
    def db(self):
        if self._db is None:
            self._db = SessionLocal()
        return self._db


@celery_app.task(bind=True, base=DatabaseTask, name="run_inference")
def run_inference(self, prediction_id: str, model_id: str, model_type: str, model_path: str, hardware: str, input_data: dict):
    """Run inference on a model"""
    db = self.db

    start_time = time.monotonic()
    gpu_name = None
    if torch.cuda.is_available():
        try:
            gpu_name = torch.cuda.get_device_name(0)
        except Exception:
            gpu_name = None

    try:
        # Update status to processing
        prediction = db.query(Prediction).filter(Prediction.id == prediction_id).first()
        if not prediction:
            return {"error": "Prediction not found"}

        prediction.status = PredictionStatus.PROCESSING
        prediction.started_at = datetime.utcnow()
        db.commit()

        # Load model
        model = model_loader.load_model(
            model_id=model_id,
            model_type=model_type,
            model_path=model_path,
            hardware=hardware
        )

        # Run inference with optional progress callback
        progress_total = int(input_data.get("num_inference_steps") or 0)
        last_progress_update = 0.0

        def _progress_callback(step: int, timestep: int, latents):
            nonlocal last_progress_update
            if progress_total <= 0:
                return
            now = datetime.utcnow().timestamp()
            if now - last_progress_update < 1.0 and step + 1 != progress_total:
                return
            last_progress_update = now
            progress_value = (step + 1) / float(progress_total)
            db.query(Prediction).filter(Prediction.id == prediction_id).update(
                {
                    Prediction.progress: progress_value,
                    Prediction.progress_step: step + 1,
                    Prediction.progress_total: progress_total,
                },
                synchronize_session=False,
            )
            db.commit()

        output = model.predict(
            input_data,
            progress_callback=_progress_callback if progress_total else None,
            callback_steps=1,
        )

        # Update prediction with results
        prediction.status = PredictionStatus.SUCCEEDED
        prediction.output = output
        prediction.progress = 1.0
        prediction.progress_step = progress_total if progress_total else None
        prediction.progress_total = progress_total if progress_total else None
        prediction.completed_at = datetime.utcnow()
        db.commit()

        try:
            from app.services.catalog import increment_prediction_count
            increment_prediction_count(model_path=model_path)
        except Exception:
            pass

        # Send webhook if configured
        if prediction.webhook:
            send_webhook.delay(prediction.webhook, prediction_id, output)

        duration_s = time.monotonic() - start_time
        logger.info(
            "Prediction succeeded",
            extra={
                "prediction_id": prediction_id,
                "model_id": model_id,
                "model_path": model_path,
                "hardware": hardware,
                "gpu": gpu_name,
                "duration_s": round(duration_s, 2),
            },
        )
        return {"status": "succeeded", "output": output}

    except Exception as e:
        error_msg = str(e)
        error_trace = traceback.format_exc()

        prediction = db.query(Prediction).filter(Prediction.id == prediction_id).first()
        if prediction:
            prediction.status = PredictionStatus.FAILED
            prediction.error = error_msg
            prediction.logs = error_trace
            prediction.progress = None
            prediction.progress_step = None
            prediction.progress_total = None
            prediction.completed_at = datetime.utcnow()
            db.commit()

        duration_s = time.monotonic() - start_time
        logger.warning(
            "Prediction failed",
            extra={
                "prediction_id": prediction_id,
                "model_id": model_id,
                "model_path": model_path,
                "hardware": hardware,
                "gpu": gpu_name,
                "duration_s": round(duration_s, 2),
                "error": error_msg,
            },
        )
        return {"status": "failed", "error": error_msg}

    finally:
        db.close()
        self._db = None


@celery_app.task(name="send_webhook")
def send_webhook(webhook_url: str, prediction_id: str, output: dict):
    """Send webhook notification"""
    try:
        allowed_raw = (settings.WEBHOOK_ALLOWED_HOSTS or "").strip()
        if allowed_raw:
            host = urlparse(webhook_url).hostname
            allowed = {h.strip().lower() for h in allowed_raw.split(",") if h.strip()}
            if not host or host.lower() not in allowed:
                return
        with httpx.Client(timeout=10.0) as client:
            client.post(
                webhook_url,
                json={
                    "prediction_id": prediction_id,
                    "status": "succeeded",
                    "output": output
                }
            )
    except Exception as e:
        print(f"Webhook failed: {e}")
