import os
import tempfile
from typing import Any, Dict, Optional
from uuid import uuid4

import torch

from app.services.storage import storage_service
from .base_model import BaseInferenceModel


class MagpieTextToSpeechModel(BaseInferenceModel):
    def __init__(self, model_path: str, device: Optional[str] = None):
        super().__init__(model_path=model_path, device=device)
        self.sr = 22050

    def load(self):
        try:
            from nemo.collections.tts.models import MagpieTTSModel
        except Exception as e:
            raise RuntimeError(
                "nemo_toolkit is not installed. Use the NeMo-enabled image variant or install nemo_toolkit[tts]."
            ) from e

        self.model = MagpieTTSModel.from_pretrained(self.model_path)

        # Best-effort sample rate discovery; fall back to 22050.
        sr = None
        for attr in ("sample_rate", "sr"):
            if hasattr(self.model, attr):
                try:
                    sr = int(getattr(self.model, attr))
                except Exception:
                    sr = None
                if sr:
                    break
        if sr:
            self.sr = sr

    def predict(self, inputs: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        if self.model is None:
            raise RuntimeError("Model not loaded")

        text = inputs.get("text") or inputs.get("prompt")
        if not text or not isinstance(text, str):
            raise ValueError("Missing required input: text")

        language = inputs.get("language") or "en"
        apply_tn = bool(inputs.get("apply_tn", False))

        speaker_index = inputs.get("speaker_index")
        speaker = inputs.get("speaker")

        speaker_map = {
            "John": 0,
            "Sofia": 1,
            "Aria": 2,
            "Jason": 3,
            "Leo": 4,
        }

        if speaker_index is None and isinstance(speaker, str) and speaker in speaker_map:
            speaker_index = speaker_map[speaker]

        kwargs: Dict[str, Any] = {
            "language": language,
            "apply_TN": apply_tn,
        }
        if speaker_index is not None:
            try:
                kwargs["speaker_index"] = int(speaker_index)
            except Exception as e:
                raise ValueError("speaker_index must be an integer") from e

        audio, _audio_len = self.model.do_tts(text, **kwargs)

        if isinstance(audio, torch.Tensor):
            wav_tensor = audio
        else:
            wav_tensor = torch.tensor(audio)

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
