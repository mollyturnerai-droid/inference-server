# Inference Server Application
import torch
import sys

# Monkeypatch for diffusers/torch version mismatch leading to "AttributeError: module 'torch' has no attribute 'xpu'"
# This occurs in some environments where diffusers 0.36.0+ is used with torch version that doesn't expose xpu attribute.
if not hasattr(torch, "xpu"):
    class MockXPU:
        def __getattr__(self, name):
            if name == "empty_cache":
                return lambda: None
            return None
        def is_available(self):
            return False
    torch.xpu = MockXPU()
    # Also ensure it's in the sys.modules if needed, but torch.xpu attribute is usually enough
