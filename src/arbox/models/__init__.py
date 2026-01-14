"""Model management module for arbox"""

from arbox.models.base import BaseModel, ModelConfig
from arbox.models.loader import ModelLoader
from arbox.models.adapters import AdapterManager
from arbox.models.registry import ModelRegistry

# Import model implementations to register them
from arbox.models.aya import Aya2335BModel, Aya238BModel, GenericModel

__all__ = [
    "BaseModel",
    "ModelConfig",
    "ModelLoader",
    "AdapterManager",
    "ModelRegistry",
    "Aya2335BModel",
    "Aya238BModel",
    "GenericModel",
]
