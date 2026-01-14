"""
Configuration loader for YAML files with validation.

Handles loading, validating, merging, and saving configuration files.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Union
import logging

from arbox.config.schema import (
    ExperimentConfigSchema,
    ModelConfigSchema,
    LoRAConfigSchema,
    TrainingConfigSchema,
    DatasetConfigSchema,
    BenchmarkConfigSchema,
    ExperimentConfigSchema,
    ModelConfigSchema,
    LoRAConfigSchema,
    TrainingConfigSchema,
    DatasetConfigSchema,
    BenchmarkConfigSchema,
    TrackerConfigSchema,
)
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class ConfigLoader:
    """Load and validate configurations from YAML files"""

    @staticmethod
    def load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load YAML file

        Args:
            path: Path to YAML file

        Returns:
            Dictionary with configuration

        Raises:
            FileNotFoundError: If file doesn't exist
            yaml.YAMLError: If YAML is invalid
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        logger.info(f"Loading configuration from: {path}")
        with open(path, 'r') as f:
            config = yaml.safe_load(f)

        return config if config is not None else {}

    @staticmethod
    def load_experiment_config(path: Union[str, Path]) -> ExperimentConfigSchema:
        """
        Load and validate experiment configuration

        Args:
            path: Path to experiment config YAML file

        Returns:
            Validated ExperimentConfigSchema

        Raises:
            ValidationError: If configuration is invalid
        """
        config_dict = ConfigLoader.load_yaml(path)
        logger.info(f"Validating experiment configuration: {config_dict.get('name', 'unnamed')}")
        return ExperimentConfigSchema(**config_dict)

    @staticmethod
    def load_model_config(path: Union[str, Path]) -> ModelConfigSchema:
        """Load and validate model configuration"""
        config_dict = ConfigLoader.load_yaml(path)
        return ModelConfigSchema(**config_dict)

    @staticmethod
    def load_training_config(path: Union[str, Path]) -> TrainingConfigSchema:
        """Load and validate training configuration"""
        config_dict = ConfigLoader.load_yaml(path)
        return TrainingConfigSchema(**config_dict)

    @staticmethod
    def load_dataset_config(path: Union[str, Path]) -> DatasetConfigSchema:
        """Load and validate dataset configuration"""
        config_dict = ConfigLoader.load_yaml(path)
        return DatasetConfigSchema(**config_dict)

    @staticmethod
    def load_benchmark_config(path: Union[str, Path]) -> BenchmarkConfigSchema:
        """Load and validate benchmark configuration"""
        config_dict = ConfigLoader.load_yaml(path)
        return BenchmarkConfigSchema(**config_dict)

    @staticmethod
    def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge multiple configuration dictionaries

        Later configs override earlier ones.

        Args:
            *configs: Variable number of config dictionaries

        Returns:
            Merged configuration dictionary
        """
        merged = {}
        for config in configs:
            merged = {**merged, **config}
        return merged

    @staticmethod
    def save_config(config: Union[Dict[str, Any], BaseModel], path: Union[str, Path]) -> None:
        """
        Save configuration to YAML file

        Args:
            config: Configuration dictionary or Pydantic model
            path: Output path for YAML file
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Convert Pydantic model to dict if needed
        if hasattr(config, 'model_dump'):
            config_dict = config.model_dump()
        elif hasattr(config, 'dict'):
            config_dict = config.dict()
        else:
            config_dict = config

        logger.info(f"Saving configuration to: {path}")
        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

    @staticmethod
    def config_to_dict(config: BaseModel) -> Dict[str, Any]:
        """
        Convert Pydantic model to dictionary

        Args:
            config: Pydantic configuration model

        Returns:
            Dictionary representation
        """
        if hasattr(config, 'model_dump'):
            return config.model_dump()
        elif hasattr(config, 'dict'):
            return config.dict()
        else:
            raise ValueError("Config must be a Pydantic model")
