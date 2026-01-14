"""
Device management utilities for GPU/CPU operations.

Functions for device detection, GPU memory management, and resource monitoring.
"""

import torch
from typing import Optional, List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class DeviceManager:
    """Manage GPU/CPU devices for model training and inference"""

    @staticmethod
    def get_device(device_id: Optional[int] = None) -> torch.device:
        """
        Get torch device for computation

        Args:
            device_id: Specific GPU ID to use (None for default)

        Returns:
            torch.device object
        """
        if torch.cuda.is_available():
            if device_id is not None:
                device = torch.device(f"cuda:{device_id}")
                logger.info(f"Using GPU {device_id}: {torch.cuda.get_device_name(device_id)}")
            else:
                device = torch.device("cuda")
                logger.info(f"Using default GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device("cpu")
            logger.warning("CUDA not available, using CPU")

        return device

    @staticmethod
    def get_available_gpus() -> List[int]:
        """
        Get list of available GPU IDs

        Returns:
            List of GPU IDs
        """
        if not torch.cuda.is_available():
            logger.info("No GPUs available")
            return []

        gpu_count = torch.cuda.device_count()
        gpus = list(range(gpu_count))
        logger.info(f"Available GPUs: {gpus}")

        for gpu_id in gpus:
            logger.info(f"  GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")

        return gpus

    @staticmethod
    def get_gpu_memory_info(device_id: int = 0) -> Dict[str, float]:
        """
        Get GPU memory information for a specific device

        Args:
            device_id: GPU device ID

        Returns:
            Dictionary with memory information in GB
        """
        if not torch.cuda.is_available():
            logger.warning("CUDA not available")
            return {}

        if device_id >= torch.cuda.device_count():
            raise ValueError(f"Invalid device ID: {device_id}")

        # Get device properties
        props = torch.cuda.get_device_properties(device_id)

        # Get current memory usage
        allocated = torch.cuda.memory_allocated(device_id) / 1e9
        reserved = torch.cuda.memory_reserved(device_id) / 1e9
        total = props.total_memory / 1e9

        info = {
            "name": props.name,
            "total_gb": total,
            "allocated_gb": allocated,
            "reserved_gb": reserved,
            "free_gb": total - allocated,
            "compute_capability": f"{props.major}.{props.minor}",
            "multi_processor_count": props.multi_processor_count,
        }

        return info

    @staticmethod
    def get_all_gpu_memory_info() -> Dict[int, Dict[str, Any]]:
        """
        Get memory information for all available GPUs

        Returns:
            Dictionary mapping GPU ID to memory info
        """
        if not torch.cuda.is_available():
            return {}

        info = {}
        for gpu_id in range(torch.cuda.device_count()):
            info[gpu_id] = DeviceManager.get_gpu_memory_info(gpu_id)

        return info

    @staticmethod
    def clear_cache() -> None:
        """Clear CUDA cache to free up GPU memory"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("CUDA cache cleared")
        else:
            logger.warning("CUDA not available, cannot clear cache")

    @staticmethod
    def synchronize() -> None:
        """Synchronize all CUDA devices"""
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    @staticmethod
    def set_device(device_id: int) -> None:
        """
        Set the current CUDA device

        Args:
            device_id: GPU device ID to set as current
        """
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")

        if device_id >= torch.cuda.device_count():
            raise ValueError(f"Invalid device ID: {device_id}")

        torch.cuda.set_device(device_id)
        logger.info(f"Current CUDA device set to: {device_id}")

    @staticmethod
    def print_device_info() -> None:
        """Print detailed information about available devices"""
        logger.info("=== Device Information ===")

        if not torch.cuda.is_available():
            logger.info("CUDA: Not available")
            logger.info("Using CPU for computation")
            logger.info("=" * 30)
            return

        logger.info(f"CUDA: Available (version {torch.version.cuda})")
        logger.info(f"cuDNN: Version {torch.backends.cudnn.version()}")
        logger.info(f"Number of GPUs: {torch.cuda.device_count()}")
        logger.info("")

        for gpu_id in range(torch.cuda.device_count()):
            info = DeviceManager.get_gpu_memory_info(gpu_id)
            logger.info(f"GPU {gpu_id}: {info['name']}")
            logger.info(f"  Compute Capability: {info['compute_capability']}")
            logger.info(f"  Total Memory: {info['total_gb']:.2f} GB")
            logger.info(f"  Allocated: {info['allocated_gb']:.2f} GB")
            logger.info(f"  Free: {info['free_gb']:.2f} GB")
            logger.info("")

        logger.info("=" * 30)

    @staticmethod
    def check_bf16_support() -> bool:
        """
        Check if current GPU supports bfloat16

        Returns:
            True if bf16 is supported
        """
        if not torch.cuda.is_available():
            return False

        # Check if GPU supports bf16 (Ampere or newer, compute capability >= 8.0)
        props = torch.cuda.get_device_properties(0)
        has_bf16 = props.major >= 8

        if has_bf16:
            logger.info(f"GPU supports bfloat16 (compute capability: {props.major}.{props.minor})")
        else:
            logger.info(f"GPU does not support bfloat16 (compute capability: {props.major}.{props.minor})")

        return has_bf16
