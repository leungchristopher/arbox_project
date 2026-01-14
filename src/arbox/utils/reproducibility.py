"""
Reproducibility utilities for deterministic experiments.

Functions for setting random seeds and tracking library versions.
"""

import random
import numpy as np
import torch
import os
import platform
from typing import Dict
import logging

logger = logging.getLogger(__name__)


class ReproducibilityManager:
    """Manage reproducibility settings for experiments"""

    @staticmethod
    def set_seed(seed: int = 42) -> None:
        """
        Set random seed for reproducibility across all libraries

        Args:
            seed: Random seed value (default: 42)
        """
        logger.info(f"Setting random seed to: {seed}")

        # Python random
        random.seed(seed)

        # NumPy
        np.random.seed(seed)

        # PyTorch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # For deterministic behavior (may impact performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Set environment variable for Python hash seed
        os.environ['PYTHONHASHSEED'] = str(seed)

        logger.info("Random seed set successfully")

    @staticmethod
    def enable_deterministic_mode() -> None:
        """
        Enable fully deterministic mode for PyTorch

        Note: This may reduce performance but ensures full reproducibility
        """
        logger.info("Enabling deterministic mode")
        torch.use_deterministic_algorithms(True)
        # Set environment variable for deterministic CUDA operations
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

    @staticmethod
    def get_version_info() -> Dict[str, str]:
        """
        Get version information for all relevant libraries

        Returns:
            Dictionary with library versions for reproducibility tracking
        """
        version_info = {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda if torch.cuda.is_available() else "N/A",
            "cudnn": str(torch.backends.cudnn.version()) if torch.cuda.is_available() else "N/A",
        }

        # Try to get versions for optional dependencies
        try:
            import transformers
            version_info["transformers"] = transformers.__version__
        except ImportError:
            version_info["transformers"] = "not installed"

        try:
            import peft
            version_info["peft"] = peft.__version__
        except ImportError:
            version_info["peft"] = "not installed"

        try:
            import datasets
            version_info["datasets"] = datasets.__version__
        except ImportError:
            version_info["datasets"] = "not installed"

        try:
            import accelerate
            version_info["accelerate"] = accelerate.__version__
        except ImportError:
            version_info["accelerate"] = "not installed"

        try:
            import bitsandbytes
            version_info["bitsandbytes"] = bitsandbytes.__version__
        except ImportError:
            version_info["bitsandbytes"] = "not installed"

        try:
            import unsloth
            version_info["unsloth"] = "installed"  # unsloth may not have __version__
        except ImportError:
            version_info["unsloth"] = "not installed"

        return version_info

    @staticmethod
    def log_version_info() -> None:
        """Log version information for reproducibility"""
        version_info = ReproducibilityManager.get_version_info()
        logger.info("=== Environment Information ===")
        for lib, version in version_info.items():
            logger.info(f"{lib}: {version}")
        logger.info("=" * 35)
