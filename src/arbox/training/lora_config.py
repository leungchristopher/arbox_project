"""
LoRA configuration builder with predefined templates.

Provides easy-to-use templates for common LoRA configurations.
"""

from peft import LoraConfig, TaskType
from typing import List, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class LoRAConfigTemplate:
    """LoRA configuration template"""
    name: str
    r: int
    lora_alpha: int
    lora_dropout: float
    target_modules: Optional[List[str]]
    description: str


class LoRAConfigBuilder:
    """Builder for LoRA configurations with predefined templates"""

    # Predefined templates for common use cases
    TEMPLATES = {
        "minimal": LoRAConfigTemplate(
            name="minimal",
            r=4,
            lora_alpha=8,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj"],
            description="Minimal LoRA for quick experiments (lowest memory, fastest)"
        ),
        "default": LoRAConfigTemplate(
            name="default",
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=None,  # Will use all-linear
            description="Default balanced configuration for most use cases"
        ),
        "aggressive": LoRAConfigTemplate(
            name="aggressive",
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=None,  # Will use all-linear
            description="High-capacity LoRA for complex tasks (more memory)"
        ),
        "qlora": LoRAConfigTemplate(
            name="qlora",
            r=64,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=None,  # Will use all-linear
            description="QLoRA configuration optimized for 4-bit quantization"
        ),
    }

    @classmethod
    def from_template(
        cls,
        template_name: str,
        task_type: TaskType = TaskType.CAUSAL_LM,
        **overrides
    ) -> LoraConfig:
        """
        Build LoRA config from predefined template

        Args:
            template_name: Template name (minimal, default, aggressive, qlora)
            task_type: Task type for LoRA
            **overrides: Override template values

        Returns:
            LoraConfig object

        Example:
            # Use default template with custom rank
            config = LoRAConfigBuilder.from_template("default", r=16)
        """
        if template_name not in cls.TEMPLATES:
            available = list(cls.TEMPLATES.keys())
            raise ValueError(
                f"Unknown template: {template_name}. "
                f"Available templates: {available}"
            )

        template = cls.TEMPLATES[template_name]
        logger.info(f"Using LoRA template: {template_name}")
        logger.info(f"  Description: {template.description}")

        # Build config dict from template
        config_dict = {
            "task_type": task_type,
            "r": template.r,
            "lora_alpha": template.lora_alpha,
            "lora_dropout": template.lora_dropout,
            "target_modules": template.target_modules if template.target_modules else "all-linear",
            "bias": "none",
        }

        # Apply overrides
        config_dict.update(overrides)

        logger.info(f"  r={config_dict['r']}, alpha={config_dict['lora_alpha']}, dropout={config_dict['lora_dropout']}")
        logger.info(f"  target_modules={config_dict['target_modules']}")

        return LoraConfig(**config_dict)

    @classmethod
    def from_dict(
        cls,
        config_dict: dict,
        task_type: TaskType = TaskType.CAUSAL_LM,
    ) -> LoraConfig:
        """
        Build LoRA config from dictionary

        Args:
            config_dict: Configuration dictionary
            task_type: Task type for LoRA

        Returns:
            LoraConfig object
        """
        # Check if template is specified
        if "template" in config_dict:
            template_name = config_dict.pop("template")
            return cls.from_template(template_name, task_type=task_type, **config_dict)

        # Otherwise, build directly from dict
        config_dict["task_type"] = task_type
        if "target_modules" not in config_dict or config_dict["target_modules"] is None:
            config_dict["target_modules"] = "all-linear"

        logger.info("Building LoRA config from dictionary")
        logger.info(f"  r={config_dict.get('r', 8)}, alpha={config_dict.get('lora_alpha', 16)}")

        return LoraConfig(**config_dict)

    @classmethod
    def list_templates(cls) -> dict:
        """
        List all available templates with descriptions

        Returns:
            Dictionary of template name to description
        """
        return {
            name: template.description
            for name, template in cls.TEMPLATES.items()
        }

    @classmethod
    def print_templates(cls) -> None:
        """Print information about all available templates"""
        logger.info("=" * 60)
        logger.info("Available LoRA Templates")
        logger.info("=" * 60)

        for name, template in cls.TEMPLATES.items():
            logger.info(f"\n{name}:")
            logger.info(f"  Description: {template.description}")
            logger.info(f"  r: {template.r}")
            logger.info(f"  lora_alpha: {template.lora_alpha}")
            logger.info(f"  lora_dropout: {template.lora_dropout}")
            logger.info(f"  target_modules: {template.target_modules or 'all-linear'}")

        logger.info("\n" + "=" * 60)
