"""
GPU resource management and monitoring utilities.
Tracks VRAM usage and provides capacity planning.
"""

import torch
from typing import Optional, Dict, Tuple
from dataclasses import dataclass
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)


@dataclass
class GPUStats:
    """GPU memory statistics."""
    total_gb: float
    free_gb: float
    used_gb: float
    utilization_pct: float
    device_name: str


class GPUMonitor:
    """Monitor GPU resources and predict capacity."""
    
    def __init__(self):
        self.is_available = torch.cuda.is_available() and settings.ENABLE_GPU
        self._device_name: Optional[str] = None
    
    def get_stats(self) -> Optional[GPUStats]:
        """Get current GPU memory statistics."""
        if not self.is_available:
            return None
        
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            used_bytes = total_bytes - free_bytes
            
            return GPUStats(
                total_gb=total_bytes / (1024 ** 3),
                free_gb=free_bytes / (1024 ** 3),
                used_gb=used_bytes / (1024 ** 3),
                utilization_pct=(used_bytes / total_bytes) * 100 if total_bytes > 0 else 0,
                device_name=self.get_device_name()
            )
        except Exception as e:
            logger.warning(f"Failed to get GPU stats: {e}")
            return None
    
    def get_device_name(self) -> str:
        """Get the GPU device name."""
        if self._device_name is None and self.is_available:
            try:
                self._device_name = torch.cuda.get_device_name(0)
            except Exception:
                self._device_name = "Unknown GPU"
        return self._device_name or "N/A"
    
    def can_fit_model(self, estimated_vram_gb: float, safety_margin: float = 1.5) -> bool:
        """
        Check if a model can fit in available VRAM.
        
        Args:
            estimated_vram_gb: Estimated model VRAM requirement in GB
            safety_margin: Additional GB to reserve for inference overhead
            
        Returns:
            True if model should fit, False otherwise
        """
        if not self.is_available:
            return False
        
        stats = self.get_stats()
        if stats is None:
            return False
        
        required_gb = estimated_vram_gb + safety_margin
        return stats.free_gb >= required_gb
    
    def estimate_concurrent_capacity(self, model_vram_gb: float) -> int:
        """
        Estimate how many instances of a model can run concurrently.
        
        Args:
            model_vram_gb: VRAM requirement for one model instance
            
        Returns:
            Number of concurrent instances that can fit
        """
        if not self.is_available or model_vram_gb <= 0:
            return 0
        
        stats = self.get_stats()
        if stats is None:
            return 0
        
        # Reserve 2GB for system and overhead
        available = max(0, stats.free_gb - 2.0)
        return int(available / model_vram_gb)
    
    def log_stats(self):
        """Log current GPU statistics."""
        stats = self.get_stats()
        if stats:
            logger.info(
                f"GPU Stats - {stats.device_name}: "
                f"{stats.used_gb:.2f}GB / {stats.total_gb:.2f}GB used "
                f"({stats.utilization_pct:.1f}%), "
                f"{stats.free_gb:.2f}GB free"
            )
    
    def suggest_hardware(self, vram_requirement: Optional[float]) -> str:
        """
        Suggest appropriate hardware based on VRAM requirements.
        
        Args:
            vram_requirement: Required VRAM in GB
            
        Returns:
            "cpu", "gpu", or "auto"
        """
        if not vram_requirement:
            return "auto"
        
        if vram_requirement <= 2.0:
            return "cpu"  # Small models can run on CPU
        
        if not self.is_available:
            return "cpu"  # No GPU available
        
        stats = self.get_stats()
        if stats and stats.free_gb >= vram_requirement + 1.0:
            return "gpu"
        
        return "cpu"  # Not enough GPU memory


# Global monitor instance
_gpu_monitor: Optional[GPUMonitor] = None


def get_gpu_monitor() -> GPUMonitor:
    """Get the global GPU monitor instance."""
    global _gpu_monitor
    if _gpu_monitor is None:
        _gpu_monitor = GPUMonitor()
    return _gpu_monitor
