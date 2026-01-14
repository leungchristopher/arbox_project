"""
Arbox: Modular Multilingual Model Benchmarking & Fine-tuning Toolkit

A comprehensive framework for:
- Loading and managing multilingual models (Aya-23-35B, etc.)
- Running benchmarks (XNLI, etc.)
- LoRA-based supervised fine-tuning with Unsloth optimization
- Experiment tracking with W&B/MLflow
"""

from arbox.__version__ import (
    __version__,
    __author__,
    __license__,
    __description__,
)

__all__ = [
    "__version__",
    "__author__",
    "__license__",
    "__description__",
]
