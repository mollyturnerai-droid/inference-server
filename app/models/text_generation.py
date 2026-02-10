from typing import Dict, Any
from transformers import AutoModelForCausalLM, AutoTokenizer
from .base_model import BaseInferenceModel


class TextGenerationModel(BaseInferenceModel):
    def load(self):
        """Load a text generation model from HuggingFace"""
        from app.core.config import settings
        hf_token = settings.HF_API_TOKEN

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=settings.TRUST_REMOTE_CODE,
            token=hf_token,
            cache_dir=settings.MODEL_CACHE_DIR
        )

        load_kwargs: Dict[str, Any] = dict(
            torch_dtype="auto",
            trust_remote_code=settings.TRUST_REMOTE_CODE,
            token=hf_token,
            cache_dir=settings.MODEL_CACHE_DIR,
            low_cpu_mem_usage=True,
        )

        # Use Accelerate sharding/offload on constrained GPUs instead of forcing everything onto CUDA.
        if self.device == "cuda":
            load_kwargs["device_map"] = "auto"
            try:
                import torch
                free_bytes, _total_bytes = torch.cuda.mem_get_info()
                free_mb = int(free_bytes / (1024 ** 2))
                # Leave some headroom for KV cache and activations.
                gpu_cap = max(512, free_mb - 512)
                load_kwargs["max_memory"] = {"cuda:0": f"{gpu_cap}MiB", "cpu": "64GiB"}
            except Exception:
                pass
        elif self.device != "cpu":
            # Non-CPU, non-CUDA devices (e.g. MPS) generally behave best without a device_map.
            pass

        self.model = AutoModelForCausalLM.from_pretrained(self.model_path, **load_kwargs)

        if self.device == "cpu":
            self.model = self.model.to(self.device)

    def predict(self, inputs: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Generate text based on prompt"""
        import torch
        prompt = inputs.get("prompt", "")
        max_length = inputs.get("max_length")
        max_new_tokens = inputs.get("max_new_tokens")
        temperature = inputs.get("temperature", 1.0)
        top_p = inputs.get("top_p", 1.0)
        top_k = inputs.get("top_k", 50)

        encoded = self.tokenizer(prompt, return_tensors="pt")
        if self.device in ("cuda", "cpu", "mps"):
            encoded = {k: v.to(self.device) for k, v in encoded.items()}

        input_len = int(encoded["input_ids"].shape[-1])
        if max_new_tokens is None:
            # Back-compat: treat max_length as total length (prompt + completion).
            if max_length is None:
                max_length = 100
            max_length = int(max_length)
            max_new_tokens = max(1, max_length - input_len)
        else:
            max_new_tokens = int(max_new_tokens)

        # On very small GPUs, the KV cache can OOM even when the model fits.
        # If the user didn't explicitly request an extreme token count, apply a safety cap.
        if self.device == "cuda" and torch.cuda.is_available():
            try:
                free_bytes, _total_bytes = torch.cuda.mem_get_info()
                free_gb = free_bytes / (1024 ** 3)
                if free_gb < 1.0:
                    max_new_tokens = min(max_new_tokens, 64)
                elif free_gb < 2.0:
                    max_new_tokens = min(max_new_tokens, 128)
            except Exception:
                pass

        do_sample = float(temperature) > 0.0

        gen_kwargs = dict(
            **encoded,
            max_new_tokens=max_new_tokens,
            temperature=float(temperature),
            top_p=float(top_p),
            top_k=int(top_k),
            do_sample=do_sample,
        )
        if getattr(self.tokenizer, "pad_token_id", None) is None and getattr(self.tokenizer, "eos_token_id", None) is not None:
            gen_kwargs["pad_token_id"] = self.tokenizer.eos_token_id

        self.model.eval()
        with torch.inference_mode():
            output = self.model.generate(**gen_kwargs)

        generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)

        return {
            "output": generated_text,
            "prompt": prompt
        }

    def unload(self):
        """Unload model from memory"""
        del self.model
        del self.tokenizer
        self.model = None
        self.tokenizer = None

        if self.device == "cuda":
            import torch
            torch.cuda.empty_cache()
