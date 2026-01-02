"""GPU utilities and CUDA support for NPP classifier."""

import torch
import time
from contextlib import contextmanager

class GPUInfo:
    """GPU device utilities and monitoring."""
    
    @staticmethod
    def get_device():
        """Get available device (CUDA or CPU)."""
        if torch.cuda.is_available():
            return torch.device('cuda')
        return torch.device('cpu')
    
    @staticmethod
    def get_gpu_memory():
        """Get GPU memory usage in MB."""
        if not torch.cuda.is_available():
            return 0, 0
        allocated = torch.cuda.memory_allocated() / 1024 ** 2
        reserved = torch.cuda.memory_reserved() / 1024 ** 2
        return allocated, reserved
    
    @staticmethod
    def clear_cache():
        """Clear GPU cache for memory optimization."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    @staticmethod
    def enable_cudnn_benchmark():
        """Enable CuDNN autotuner for performance."""
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True

class PerformanceProfiler:
    """Profile training and inference performance."""
    
    @contextmanager
    def profile_block(self, name):
        """Context manager for code block profiling."""
        start = time.time()
        try:
            yield
        finally:
            duration = time.time() - start
            print(f"{name}: {duration:.2f}s")

if __name__ == '__main__':
    print(f"Device: {GPUInfo.get_device()}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        GPUInfo.enable_cudnn_benchmark()
        print("CuDNN benchmark enabled")
