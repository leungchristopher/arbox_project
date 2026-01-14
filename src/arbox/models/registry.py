"""
Model registry for managing different model implementations.

Provides a registry pattern for easy model discovery and instantiation.
"""

from typing import Dict, Type, List
import logging

from arbox.models.base import BaseModel

logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Registry for model implementations.

    Allows models to register themselves and be instantiated by name.
    """

    _registry: Dict[str, Type[BaseModel]] = {}

    @classmethod
    def register(cls, name: str):
        """
        Decorator to register a model class

        Args:
            name: Model identifier

        Returns:
            Decorator function

        Example:
            @ModelRegistry.register("aya-23-35b")
            class Aya2335BModel(BaseModel):
                pass
        """
        def decorator(model_class: Type[BaseModel]):
            if name in cls._registry:
                logger.warning(f"Model {name} already registered, overwriting")

            cls._registry[name] = model_class
            logger.info(f"Registered model: {name}")
            return model_class

        return decorator

    @classmethod
    def get(cls, name: str) -> Type[BaseModel]:
        """
        Get model class by name

        Args:
            name: Model identifier

        Returns:
            Model class

        Raises:
            ValueError: If model not found in registry
        """
        if name not in cls._registry:
            available = cls.list_models()
            raise ValueError(
                f"Model '{name}' not found in registry. "
                f"Available models: {available}"
            )

        return cls._registry[name]

    @classmethod
    def list_models(cls) -> List[str]:
        """
        List all registered model names

        Returns:
            List of model identifiers
        """
        return list(cls._registry.keys())

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """
        Check if a model is registered

        Args:
            name: Model identifier

        Returns:
            True if registered, False otherwise
        """
        return name in cls._registry

    @classmethod
    def unregister(cls, name: str) -> None:
        """
        Remove a model from the registry

        Args:
            name: Model identifier
        """
        if name in cls._registry:
            del cls._registry[name]
            logger.info(f"Unregistered model: {name}")
        else:
            logger.warning(f"Model {name} not in registry")
