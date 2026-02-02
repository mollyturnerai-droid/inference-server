from typing import Dict, Any
from transformers import AutoModelForCausalLM, AutoTokenizer
from .base_model import BaseInferenceModel


class TextGenerationModel(BaseInferenceModel):
    def load(self):
        """Load a text generation model from HuggingFace"""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            device_map=self.device if self.device != "cpu" else None,
            torch_dtype="auto",
            trust_remote_code=True,
        )

        if self.device == "cpu":
            self.model = self.model.to(self.device)

    def predict(self, inputs: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Generate text based on prompt"""
        prompt = inputs.get("prompt", "")
        max_length = inputs.get("max_length", 100)
        temperature = inputs.get("temperature", 1.0)
        top_p = inputs.get("top_p", 1.0)
        top_k = inputs.get("top_k", 50)

        encoded = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        output = self.model.generate(
            **encoded,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=True
        )

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
