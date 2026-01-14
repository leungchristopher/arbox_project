"""
Model loading utilities with quantization and Unsloth support.

Handles loading models with various quantization options and optimizations.
"""

from typing import Tuple, Optional
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)
import logging

from arbox.models.base import ModelConfig

logger = logging.getLogger(__name__)


class ModelLoader:
    """
    Model loader with quantization and Unsloth support.

    Supports:
    - Standard HuggingFace loading with bitsandbytes quantization
    - Unsloth optimized loading for faster training
    """

    @staticmethod
    def load_with_unsloth(
        config: ModelConfig
    ) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """
        Load model using Unsloth for optimized training

        Args:
            config: Model configuration

        Returns:
            Tuple of (model, tokenizer)
        """
        try:
            from unsloth import FastLanguageModel
        except ImportError:
            raise ImportError(
                "Unsloth is not installed. Please install with: "
                "pip install 'unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git'"
            )

        logger.info(f"Loading model with Unsloth: {config.model_name_or_path}")

        # Determine quantization
        load_in_4bit = config.quantization == "4bit" or config.quantization == "int4"
        load_in_8bit = config.quantization == "8bit" or config.quantization == "int8"

        # Determine dtype
        dtype = None  # Auto-detect
        if config.torch_dtype == "float16" or config.torch_dtype == "fp16":
            dtype = torch.float16
        elif config.torch_dtype == "bfloat16" or config.torch_dtype == "bf16":
            dtype = torch.bfloat16
        elif config.torch_dtype == "float32" or config.torch_dtype == "fp32":
            dtype = torch.float32

        # Load with Unsloth
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=config.model_name_or_path,
            max_seq_length=config.max_length,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
            **(config.additional_kwargs or {})
        )

        # Set padding token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = model.config.eos_token_id

        logger.info("Model loaded successfully with Unsloth")
        return model, tokenizer

    @staticmethod
    def load_standard(
        config: ModelConfig
    ) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """
        Load model using standard HuggingFace transformers

        Args:
            config: Model configuration

        Returns:
            Tuple of (model, tokenizer)
        """
        logger.info(f"Loading model with HuggingFace: {config.model_name_or_path}")

        # Setup quantization config
        quantization_config = None
        if config.quantization == "4bit" or config.quantization == "int4":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            logger.info("Using 4-bit quantization (NF4)")
        elif config.quantization == "8bit" or config.quantization == "int8":
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True
            )
            logger.info("Using 8-bit quantization")

        # Determine torch dtype
        torch_dtype = torch.float32
        if config.torch_dtype == "auto":
            torch_dtype = "auto"
        elif config.torch_dtype == "float16" or config.torch_dtype == "fp16":
            torch_dtype = torch.float16
        elif config.torch_dtype == "bfloat16" or config.torch_dtype == "bf16":
            torch_dtype = torch.bfloat16

        # Build model kwargs
        model_kwargs = {
            "device_map": config.device_map,
            "torch_dtype": torch_dtype,
            "trust_remote_code": config.trust_remote_code,
        }

        if quantization_config is not None:
            model_kwargs["quantization_config"] = quantization_config

        if config.use_flash_attention:
            model_kwargs["attn_implementation"] = "flash_attention_2"
            logger.info("Using Flash Attention 2")

        # Add additional kwargs
        if config.additional_kwargs:
            model_kwargs.update(config.additional_kwargs)

        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name_or_path,
            **model_kwargs
        )

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            config.model_name_or_path,
            trust_remote_code=config.trust_remote_code
        )

        # Set padding token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = model.config.eos_token_id

        logger.info("Model loaded successfully")
        return model, tokenizer

    @staticmethod
    def load_model_and_tokenizer(
        config: ModelConfig
    ) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """
        Load model and tokenizer with automatic method selection

        Uses Unsloth if enabled in config, otherwise standard loading.

        Args:
            config: Model configuration

        Returns:
            Tuple of (model, tokenizer)
        """
        if config.use_unsloth:
            try:
                return ModelLoader.load_with_unsloth(config)
            except ImportError as e:
                logger.warning(f"Unsloth not available: {e}")
                logger.warning("Falling back to standard loading")
                return ModelLoader.load_standard(config)
        else:
            return ModelLoader.load_standard(config)
