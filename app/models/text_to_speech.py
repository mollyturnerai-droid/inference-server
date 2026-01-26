import os
import tempfile
from typing import Any, Dict, Optional
from urllib.parse import urlparse
from uuid import uuid4

import requests
import torch

from app.core.config import settings
from app.services.storage import storage_service
from .base_model import BaseInferenceModel


class TextToSpeechModel(BaseInferenceModel):
    def __init__(self, model_path: str, device: Optional[str] = None):
        super().__init__(model_path=model_path, device=device)
        self.sr = None

    def load(self):
        try:
            from chatterbox.tts import ChatterboxTTS
        except Exception as e:
            raise RuntimeError(
                "chatterbox-tts is not installed. Add it to requirements and rebuild the image."
            ) from e

        device = self.device
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"

        self.model = ChatterboxTTS.from_pretrained(device=device)
        self.sr = getattr(self.model, "sr", 24000)

    def _resolve_reference_audio_path(self, reference_audio: str) -> str:
        parsed = urlparse(reference_audio)

        # Local storage file path
        if not parsed.scheme:
            return os.path.join(settings.STORAGE_PATH, reference_audio)

        if parsed.scheme == "file":
            return parsed.path

        # If it points at this API's /v1/files/... route, map it back to storage directly
        if reference_audio.startswith("/v1/files/"):
            rel = reference_audio[len("/v1/files/"):]
            return os.path.join(settings.STORAGE_PATH, rel)

        api_base = (settings.API_BASE_URL or "").rstrip("/")
        api_files_prefix = f"{api_base}/v1/files/" if api_base else ""
        if api_files_prefix and reference_audio.startswith(api_files_prefix):
            rel = reference_audio[len(api_files_prefix):]
            return os.path.join(settings.STORAGE_PATH, rel)

        if parsed.scheme in ("http", "https"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp_path = tmp.name

            with requests.get(reference_audio, stream=True, timeout=30) as r:
                r.raise_for_status()
                with open(tmp_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 256):
                        if chunk:
                            f.write(chunk)

            return tmp_path

        raise ValueError("Unsupported reference_audio scheme")

    def predict(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if self.model is None:
            raise RuntimeError("Model not loaded")

        text = inputs.get("text") or inputs.get("prompt")
        if not text or not isinstance(text, str):
            raise ValueError("Missing required input: text")

        reference_audio = inputs.get("reference_audio")
        audio_prompt_path = None
        tmp_ref_path = None

        if reference_audio:
            tmp_ref_path = self._resolve_reference_audio_path(reference_audio)
            audio_prompt_path = tmp_ref_path

        wav = self.model.generate(text, audio_prompt_path=audio_prompt_path)

        # torchaudio expects [channels, time]
        if isinstance(wav, torch.Tensor):
            wav_tensor = wav
        else:
            wav_tensor = torch.tensor(wav)

        if wav_tensor.dim() == 1:
            wav_tensor = wav_tensor.unsqueeze(0)

        wav_tensor = wav_tensor.detach().cpu()

        try:
            import torchaudio
        except Exception as e:
            raise RuntimeError(
                "torchaudio is not installed. Add it to requirements and rebuild the image."
            ) from e

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_out:
            out_path = tmp_out.name

        torchaudio.save(out_path, wav_tensor, int(self.sr))
        with open(out_path, "rb") as f:
            audio_bytes = f.read()

        os.remove(out_path)
        if tmp_ref_path and tmp_ref_path.startswith(tempfile.gettempdir()):
            # Only remove downloaded temp files; do not delete storage files
            try:
                os.remove(tmp_ref_path)
            except OSError:
                pass

        audio_file_path = storage_service.save_file_sync(
            file_path=f"outputs/{self.model_path.replace('/', '_')}/{uuid4().hex}.wav",
            content=audio_bytes,
            content_type="audio/wav",
        )

        return {
            "audio_path": audio_file_path,
            "audio_url": storage_service.get_public_url(audio_file_path),
            "sample_rate": int(self.sr),
        }

    def unload(self):
        self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
