"""
Base model classes and interfaces.

Provides abstract base class for all model implementations.
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
import torch
import logging

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Model configuration dataclass"""
    model_name_or_path: str
    model_type: str = "causal_lm"
    quantization: Optional[str] = None
    device_map: str = "auto"
    torch_dtype: str = "auto"
    trust_remote_code: bool = False
    use_flash_attention: bool = False
    max_length: int = 2048
    use_unsloth: bool = True
    additional_kwargs: Optional[Dict[str, Any]] = None


class BaseModel(ABC):
    """
    Abstract base class for all models.

    Provides a unified interface for loading, inference, and model management.
    """

    def __init__(self, config: ModelConfig):
        """
        Initialize model with configuration

        Args:
            config: Model configuration
        """
        self.config = config
        self.model = None
        self.tokenizer = None
        self.is_loaded = False

    @abstractmethod
    def load(self) -> None:
        """Load model and tokenizer"""
        pass

    def generate(
        self,
        prompts: List[str],
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        **kwargs
    ) -> List[str]:
        """
        Generate text from prompts

        Args:
            prompts: List of input prompts
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            do_sample: Whether to use sampling
            **kwargs: Additional generation parameters

        Returns:
            List of generated texts
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Tokenize inputs
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_length
        )

        # Move to device
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                **kwargs
            )

        # Decode outputs
        generated_texts = self.tokenizer.batch_decode(
            outputs,
            skip_special_tokens=True
        )

        return generated_texts

    def get_embeddings(self, texts: List[str]) -> torch.Tensor:
        """
        Get embeddings for texts

        Args:
            texts: List of input texts

        Returns:
            Tensor of embeddings
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_length
        )

        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            # Use last hidden state mean pooling
            embeddings = outputs.hidden_states[-1].mean(dim=1)

        return embeddings

    def unload(self) -> None:
        """Free GPU memory by unloading model"""
        if self.model is not None:
            del self.model
            self.model = None

        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.is_loaded = False
        logger.info("Model unloaded and GPU memory cleared")

    def get_num_parameters(self) -> int:
        """
        Get total number of parameters

        Returns:
            Number of parameters
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        return sum(p.numel() for p in self.model.parameters())

    def get_trainable_parameters(self) -> int:
        """
        Get number of trainable parameters

        Returns:
            Number of trainable parameters
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def print_trainable_parameters(self) -> None:
        """Print statistics about trainable parameters"""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        trainable_params = self.get_trainable_parameters()
        all_params = self.get_num_parameters()
        trainable_percent = 100 * trainable_params / all_params

        logger.info(f"Trainable parameters: {trainable_params:,} / {all_params:,} ({trainable_percent:.2f}%)")
