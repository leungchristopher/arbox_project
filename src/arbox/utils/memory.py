"""
Memory profiling utilities for CPU and GPU.

Functions for monitoring memory usage during training and inference.
"""

import torch
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)


class MemoryProfiler:
    """Profile memory usage for CPU and GPU"""

    @staticmethod
    def get_cpu_memory_usage() -> Dict[str, float]:
        """
        Get CPU memory usage

        Returns:
            Dictionary with CPU memory statistics in GB
        """
        try:
            import psutil
            memory = psutil.virtual_memory()
            return {
                "total_gb": memory.total / 1e9,
                "available_gb": memory.available / 1e9,
                "used_gb": memory.used / 1e9,
                "percent": memory.percent
            }
        except ImportError:
            logger.warning("psutil not installed, cannot get CPU memory info")
            return {}

    @staticmethod
    def get_gpu_memory_usage() -> List[Dict[str, Any]]:
        """
        Get GPU memory usage for all devices

        Returns:
            List of dictionaries with GPU memory statistics
        """
        if not torch.cuda.is_available():
            return []

        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            return [
                {
                    "id": gpu.id,
                    "name": gpu.name,
                    "load": gpu.load * 100,  # Convert to percentage
                    "memory_used_gb": gpu.memoryUsed / 1024,
                    "memory_total_gb": gpu.memoryTotal / 1024,
                    "memory_util": gpu.memoryUtil * 100,  # Convert to percentage
                    "temperature": gpu.temperature,
                }
                for gpu in gpus
            ]
        except ImportError:
            logger.warning("GPUtil not installed, using torch CUDA memory stats")
            # Fallback to torch.cuda memory stats
            gpu_info = []
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1e9
                reserved = torch.cuda.memory_reserved(i) / 1e9
                props = torch.cuda.get_device_properties(i)
                total = props.total_memory / 1e9

                gpu_info.append({
                    "id": i,
                    "name": props.name,
                    "memory_used_gb": allocated,
                    "memory_total_gb": total,
                    "memory_util": (allocated / total * 100) if total > 0 else 0,
                })
            return gpu_info

    @staticmethod
    def get_torch_memory_stats(device_id: int = 0) -> Dict[str, float]:
        """
        Get detailed PyTorch CUDA memory statistics

        Args:
            device_id: GPU device ID

        Returns:
            Dictionary with detailed memory statistics in GB
        """
        if not torch.cuda.is_available():
            return {}

        allocated = torch.cuda.memory_allocated(device_id) / 1e9
        reserved = torch.cuda.memory_reserved(device_id) / 1e9
        max_allocated = torch.cuda.max_memory_allocated(device_id) / 1e9
        max_reserved = torch.cuda.max_memory_reserved(device_id) / 1e9

        return {
            "allocated_gb": allocated,
            "reserved_gb": reserved,
            "max_allocated_gb": max_allocated,
            "max_reserved_gb": max_reserved,
        }

    @staticmethod
    def print_memory_summary(device_id: Optional[int] = None) -> None:
        """
        Print comprehensive memory summary

        Args:
            device_id: GPU device ID (None for all devices)
        """
        logger.info("=" * 50)
        logger.info("MEMORY SUMMARY")
        logger.info("=" * 50)

        # CPU Memory
        cpu_mem = MemoryProfiler.get_cpu_memory_usage()
        if cpu_mem:
            logger.info("\n--- CPU Memory ---")
            logger.info(f"Used: {cpu_mem['used_gb']:.2f} GB / {cpu_mem['total_gb']:.2f} GB ({cpu_mem['percent']:.1f}%)")
            logger.info(f"Available: {cpu_mem['available_gb']:.2f} GB")

        # GPU Memory
        if torch.cuda.is_available():
            logger.info("\n--- GPU Memory ---")
            gpu_info = MemoryProfiler.get_gpu_memory_usage()

            if device_id is not None:
                # Show specific device
                gpu_info = [g for g in gpu_info if g["id"] == device_id]

            for gpu in gpu_info:
                logger.info(f"\nGPU {gpu['id']}: {gpu['name']}")
                logger.info(f"  Memory: {gpu['memory_used_gb']:.2f} GB / {gpu['memory_total_gb']:.2f} GB ({gpu['memory_util']:.1f}%)")
                if "load" in gpu:
                    logger.info(f"  Utilization: {gpu['load']:.1f}%")
                if "temperature" in gpu and gpu["temperature"]:
                    logger.info(f"  Temperature: {gpu['temperature']}°C")

                # Show PyTorch-specific stats
                torch_stats = MemoryProfiler.get_torch_memory_stats(gpu["id"])
                if torch_stats:
                    logger.info(f"  PyTorch Allocated: {torch_stats['allocated_gb']:.2f} GB (Max: {torch_stats['max_allocated_gb']:.2f} GB)")
                    logger.info(f"  PyTorch Reserved: {torch_stats['reserved_gb']:.2f} GB (Max: {torch_stats['max_reserved_gb']:.2f} GB)")
        else:
            logger.info("\n--- GPU Memory ---")
            logger.info("No CUDA devices available")

        logger.info("\n" + "=" * 50)

    @staticmethod
    def reset_peak_memory_stats(device_id: Optional[int] = None) -> None:
        """
        Reset peak memory statistics

        Args:
            device_id: GPU device ID (None for all devices)
        """
        if not torch.cuda.is_available():
            return

        if device_id is not None:
            torch.cuda.reset_peak_memory_stats(device_id)
            logger.info(f"Reset peak memory stats for GPU {device_id}")
        else:
            torch.cuda.reset_peak_memory_stats()
            logger.info("Reset peak memory stats for all GPUs")

    @staticmethod
    def estimate_model_memory(
        num_parameters: int,
        precision: str = "fp32",
        include_gradients: bool = True,
        include_optimizer: bool = True
    ) -> float:
        """
        Estimate memory requirements for a model

        Args:
            num_parameters: Number of model parameters
            precision: Model precision (fp32, fp16, bf16, int8, int4)
            include_gradients: Whether to include gradient memory
            include_optimizer: Whether to include optimizer state memory

        Returns:
            Estimated memory in GB
        """
        # Bytes per parameter based on precision
        precision_bytes = {
            "fp32": 4,
            "fp16": 2,
            "bf16": 2,
            "int8": 1,
            "int4": 0.5,
            "4bit": 0.5,
            "8bit": 1,
        }

        if precision not in precision_bytes:
            raise ValueError(f"Unknown precision: {precision}")

        bytes_per_param = precision_bytes[precision]

        # Model parameters
        model_memory = num_parameters * bytes_per_param

        # Gradients (same size as model for full precision training)
        if include_gradients:
            model_memory += num_parameters * 4  # Gradients typically in fp32

        # Optimizer states (e.g., Adam: 2x parameters for momentum and variance)
        if include_optimizer:
            model_memory += num_parameters * 8  # 2 states * 4 bytes (fp32)

        # Convert to GB and add 20% overhead for activations and buffers
        memory_gb = (model_memory / 1e9) * 1.2

        logger.info(f"Estimated memory for {num_parameters/1e9:.2f}B parameters ({precision}):")
        logger.info(f"  Model: {num_parameters * bytes_per_param / 1e9:.2f} GB")
        if include_gradients:
            logger.info(f"  Gradients: {num_parameters * 4 / 1e9:.2f} GB")
        if include_optimizer:
            logger.info(f"  Optimizer: {num_parameters * 8 / 1e9:.2f} GB")
        logger.info(f"  Total (with overhead): {memory_gb:.2f} GB")

        return memory_gb
