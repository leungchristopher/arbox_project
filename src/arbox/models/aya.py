"""
Aya-23-35B model implementation.

Concrete implementation for Cohere's Aya-23-35B multilingual model.
"""

import logging
from arbox.models.base import BaseModel, ModelConfig
from arbox.models.loader import ModelLoader
from arbox.models.registry import ModelRegistry

logger = logging.getLogger(__name__)


@ModelRegistry.register("aya-23-35b")
class Aya2335BModel(BaseModel):
    """
    Aya-23-35B multilingual model.

    A 35B parameter multilingual model from Cohere, covering 23 languages.
    Based on the Command-R architecture.
    """

    def __init__(self, config: ModelConfig):
        """
        Initialize Aya-23-35B model

        Args:
            config: Model configuration
        """
        super().__init__(config)

        # Set default model path if not specified
        if not config.model_name_or_path:
            config.model_name_or_path = "CohereForAI/aya-23-35B"

    def load(self) -> None:
        """Load Aya-23-35B model and tokenizer"""
        logger.info("Loading Aya-23-35B model")
        logger.info(f"Model path: {self.config.model_name_or_path}")
        logger.info(f"Quantization: {self.config.quantization}")
        logger.info(f"Use Unsloth: {self.config.use_unsloth}")

        # Load model and tokenizer
        self.model, self.tokenizer = ModelLoader.load_model_and_tokenizer(self.config)

        self.is_loaded = True
        logger.info("Aya-23-35B loaded successfully")

        # Print parameter count
        num_params = self.get_num_parameters()
        logger.info(f"Total parameters: {num_params:,} ({num_params/1e9:.2f}B)")


@ModelRegistry.register("aya-23-8b")
class Aya238BModel(BaseModel):
    """
    Aya-23-8B multilingual model.

    An 8B parameter variant for faster training and inference.
    """

    def __init__(self, config: ModelConfig):
        """
        Initialize Aya-23-8B model

        Args:
            config: Model configuration
        """
        super().__init__(config)

        # Set default model path if not specified
        if not config.model_name_or_path:
            config.model_name_or_path = "CohereForAI/aya-23-8B"

    def load(self) -> None:
        """Load Aya-23-8B model and tokenizer"""
        logger.info("Loading Aya-23-8B model")
        logger.info(f"Model path: {self.config.model_name_or_path}")
        logger.info(f"Quantization: {self.config.quantization}")
        logger.info(f"Use Unsloth: {self.config.use_unsloth}")

        # Load model and tokenizer
        self.model, self.tokenizer = ModelLoader.load_model_and_tokenizer(self.config)

        self.is_loaded = True
        logger.info("Aya-23-8B loaded successfully")

        # Print parameter count
        num_params = self.get_num_parameters()
        logger.info(f"Total parameters: {num_params:,} ({num_params/1e9:.2f}B)")


# Generic model class for other architectures
@ModelRegistry.register("generic")
class GenericModel(BaseModel):
    """
    Generic model implementation.

    Can be used for any HuggingFace model by specifying model_name_or_path.
    """

    def load(self) -> None:
        """Load generic model and tokenizer"""
        logger.info(f"Loading model: {self.config.model_name_or_path}")
        logger.info(f"Quantization: {self.config.quantization}")
        logger.info(f"Use Unsloth: {self.config.use_unsloth}")

        # Load model and tokenizer
        self.model, self.tokenizer = ModelLoader.load_model_and_tokenizer(self.config)

        self.is_loaded = True
        logger.info("Model loaded successfully")

        # Print parameter count
        num_params = self.get_num_parameters()
        logger.info(f"Total parameters: {num_params:,} ({num_params/1e9:.2f}B)")
