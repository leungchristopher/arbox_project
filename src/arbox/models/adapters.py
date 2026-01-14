"""
Adapter management for LoRA fine-tuning.

Handles creation, application, loading, and merging of LoRA adapters.
"""

from typing import Optional, List, Union
from pathlib import Path
import logging

from peft import (
    LoraConfig,
    get_peft_model,
    PeftModel,
    TaskType,
)
from transformers import PreTrainedModel

logger = logging.getLogger(__name__)


class AdapterManager:
    """Manage LoRA adapters for fine-tuning"""

    @staticmethod
    def create_lora_config(
        task_type: TaskType = TaskType.CAUSAL_LM,
        r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        target_modules: Optional[Union[List[str], str]] = None,
        bias: str = "none",
        modules_to_save: Optional[List[str]] = None,
    ) -> LoraConfig:
        """
        Create LoRA configuration

        Args:
            task_type: Type of task (CAUSAL_LM, SEQ_2_SEQ_LM, etc.)
            r: LoRA rank (adapter dimension)
            lora_alpha: LoRA scaling factor
            lora_dropout: Dropout rate for LoRA layers
            target_modules: Modules to apply LoRA to (None for all-linear)
            bias: Bias training strategy ("none", "all", "lora_only")
            modules_to_save: Additional modules to save during training

        Returns:
            LoraConfig object
        """
        if target_modules is None:
            # Default: target all linear layers
            target_modules = "all-linear"
            logger.info("Using default target_modules: all-linear")

        config = LoraConfig(
            task_type=task_type,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias=bias,
            modules_to_save=modules_to_save,
        )

        logger.info(f"Created LoRA config: r={r}, alpha={lora_alpha}, dropout={lora_dropout}")
        return config

    @staticmethod
    def create_lora_config_unsloth(
        r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        target_modules: Optional[List[str]] = None,
        bias: str = "none",
        use_gradient_checkpointing: str = "unsloth",
        random_state: int = 42,
    ):
        """
        Create LoRA configuration for Unsloth

        Args:
            r: LoRA rank
            lora_alpha: LoRA alpha scaling factor
            lora_dropout: Dropout rate
            target_modules: Target modules for LoRA
            bias: Bias strategy
            use_gradient_checkpointing: Gradient checkpointing mode
            random_state: Random seed

        Returns:
            Dictionary of LoRA parameters for Unsloth
        """
        if target_modules is None:
            # Default targets for common architectures
            target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]

        config = {
            "r": r,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "target_modules": target_modules,
            "bias": bias,
            "use_gradient_checkpointing": use_gradient_checkpointing,
            "random_state": random_state,
        }

        logger.info(f"Created Unsloth LoRA config: r={r}, alpha={lora_alpha}")
        return config

    @staticmethod
    def apply_lora(
        model: PreTrainedModel,
        config: LoraConfig
    ) -> PeftModel:
        """
        Apply LoRA adapters to model

        Args:
            model: Base model to add adapters to
            config: LoRA configuration

        Returns:
            PEFT model with LoRA adapters
        """
        logger.info("Applying LoRA adapters to model")
        peft_model = get_peft_model(model, config)

        # Print trainable parameters
        trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in peft_model.parameters())
        trainable_percent = 100 * trainable_params / all_params

        logger.info(
            f"Trainable params: {trainable_params:,} || "
            f"All params: {all_params:,} || "
            f"Trainable%: {trainable_percent:.2f}%"
        )

        return peft_model

    @staticmethod
    def apply_lora_unsloth(model, **lora_config):
        """
        Apply LoRA adapters using Unsloth

        Args:
            model: Base model loaded with Unsloth
            **lora_config: LoRA configuration parameters

        Returns:
            Model with LoRA adapters
        """
        try:
            from unsloth import FastLanguageModel
        except ImportError:
            raise ImportError("Unsloth not installed")

        logger.info("Applying LoRA adapters with Unsloth")
        model = FastLanguageModel.get_peft_model(model, **lora_config)

        logger.info("LoRA adapters applied successfully with Unsloth")
        return model

    @staticmethod
    def load_adapter(
        model: PreTrainedModel,
        adapter_path: Union[str, Path],
    ) -> PeftModel:
        """
        Load pre-trained adapter onto base model

        Args:
            model: Base model
            adapter_path: Path to saved adapter

        Returns:
            PEFT model with loaded adapter
        """
        adapter_path = str(adapter_path)
        logger.info(f"Loading adapter from: {adapter_path}")

        peft_model = PeftModel.from_pretrained(model, adapter_path)
        logger.info("Adapter loaded successfully")

        return peft_model

    @staticmethod
    def save_adapter(
        peft_model: PeftModel,
        output_path: Union[str, Path],
        tokenizer=None,
    ) -> None:
        """
        Save LoRA adapter (not the base model)

        Args:
            peft_model: PEFT model with adapters
            output_path: Directory to save adapter
            tokenizer: Optional tokenizer to save alongside adapter
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving adapter to: {output_path}")
        peft_model.save_pretrained(str(output_path))

        if tokenizer is not None:
            logger.info("Saving tokenizer")
            tokenizer.save_pretrained(str(output_path))

        logger.info("Adapter saved successfully")

    @staticmethod
    def merge_and_save(
        peft_model: PeftModel,
        output_path: Union[str, Path],
        tokenizer=None,
    ) -> None:
        """
        Merge adapter with base model and save

        Args:
            peft_model: PEFT model with adapters
            output_path: Directory to save merged model
            tokenizer: Optional tokenizer to save
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info("Merging adapter with base model")
        merged_model = peft_model.merge_and_unload()

        logger.info(f"Saving merged model to: {output_path}")
        merged_model.save_pretrained(str(output_path))

        if tokenizer is not None:
            logger.info("Saving tokenizer")
            tokenizer.save_pretrained(str(output_path))

        logger.info("Merged model saved successfully")

    @staticmethod
    def merge_and_save_unsloth(
        model,
        tokenizer,
        output_path: Union[str, Path],
        save_method: str = "merged_16bit",
    ) -> None:
        """
        Merge and save model using Unsloth

        Args:
            model: Unsloth model with adapters
            tokenizer: Tokenizer
            output_path: Output directory
            save_method: Save method ("merged_16bit", "merged_4bit", "lora")
        """
        try:
            from unsloth import FastLanguageModel
        except ImportError:
            raise ImportError("Unsloth not installed")

        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving model with Unsloth (method: {save_method})")

        if save_method == "lora":
            # Save only LoRA adapters
            model.save_pretrained(str(output_path))
            tokenizer.save_pretrained(str(output_path))
        else:
            # Save merged model
            model.save_pretrained_merged(
                str(output_path),
                tokenizer,
                save_method=save_method
            )

        logger.info("Model saved successfully with Unsloth")
