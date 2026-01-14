"""
LoRA fine-tuning trainer with Unsloth support.

Provides unified interface for training with both standard PEFT and Unsloth.
"""

from typing import Optional, Dict, Any, List
from pathlib import Path
import torch
from datasets import Dataset
from transformers import (
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from peft import PeftModel, TaskType
import logging

from arbox.models.base import BaseModel
from arbox.models.adapters import AdapterManager
from arbox.training.lora_config import LoRAConfigBuilder
from arbox.data.collators import SFTDataCollator

logger = logging.getLogger(__name__)


class LoRATrainer:
    """
    LoRA fine-tuning trainer with Unsloth optimization support.

    Supports both standard HuggingFace Trainer and Unsloth optimized training.
    """

    def __init__(
        self,
        model: BaseModel,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        lora_config: Optional[Dict[str, Any]] = None,
        training_args: Optional[TrainingArguments] = None,
        use_unsloth: bool = True,
        tracker=None,
        **kwargs
    ):
        """
        Initialize LoRA trainer

        Args:
            model: Base model to train
            train_dataset: Training dataset
            eval_dataset: Optional evaluation dataset
            lora_config: LoRA configuration dictionary
            training_args: Training arguments
            use_unsloth: Whether to use Unsloth optimization
            tracker: Optional experiment tracker
            **kwargs: Additional arguments
        """
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.lora_config_dict = lora_config or {}
        self.training_args = training_args
        self.use_unsloth = use_unsloth
        self.tracker = tracker
        self.kwargs = kwargs

        self.peft_model = None
        self.trainer = None

    def setup_lora(self) -> None:
        """Setup LoRA adapters on the model"""
        logger.info("Setting up LoRA adapters")

        if self.use_unsloth:
            self._setup_lora_unsloth()
        else:
            self._setup_lora_standard()

    def _setup_lora_standard(self) -> None:
        """Setup LoRA using standard PEFT"""
        logger.info("Using standard PEFT for LoRA")

        # Create LoRA config
        lora_config = LoRAConfigBuilder.from_dict(
            self.lora_config_dict,
            task_type=TaskType.CAUSAL_LM
        )

        # Apply LoRA to model
        self.peft_model = AdapterManager.apply_lora(
            self.model.model,
            lora_config
        )

    def _setup_lora_unsloth(self) -> None:
        """Setup LoRA using Unsloth"""
        logger.info("Using Unsloth for optimized LoRA")

        # Create Unsloth LoRA config
        lora_config = AdapterManager.create_lora_config_unsloth(
            r=self.lora_config_dict.get('r', 8),
            lora_alpha=self.lora_config_dict.get('lora_alpha', 16),
            lora_dropout=self.lora_config_dict.get('lora_dropout', 0.05),
            target_modules=self.lora_config_dict.get('target_modules'),
            bias=self.lora_config_dict.get('bias', 'none'),
        )

        # Apply LoRA via Unsloth
        self.peft_model = AdapterManager.apply_lora_unsloth(
            self.model.model,
            **lora_config
        )

    def train(self) -> Dict[str, Any]:
        """
        Run training

        Returns:
            Training metrics
        """
        logger.info("Starting training")

        # Setup LoRA if not already done
        if self.peft_model is None:
            self.setup_lora()

        # Setup training arguments if not provided
        if self.training_args is None:
            self.training_args = self._default_training_args()

        # Initialize tracker
        if self.tracker:
            self.tracker.init_run(config=self._get_config_dict())

        # Prepare training
        if self.use_unsloth:
            metrics = self._train_with_unsloth()
        else:
            metrics = self._train_standard()

        # Log final metrics
        if self.tracker:
            self.tracker.log_metrics(metrics)
            self.tracker.finish()

        logger.info("Training completed")
        return metrics

    def _train_standard(self) -> Dict[str, Any]:
        """Train using standard HuggingFace Trainer"""
        logger.info("Using standard HuggingFace Trainer")

        # Create data collator
        data_collator = SFTDataCollator(
            tokenizer=self.model.tokenizer,
            mlm=False,
        )

        # Get callbacks
        callbacks = self._get_callbacks()

        # Create trainer
        self.trainer = Trainer(
            model=self.peft_model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=data_collator,
            callbacks=callbacks,
        )

        # Train
        train_result = self.trainer.train()

        # Save adapter
        self.save_adapter(self.training_args.output_dir)

        return train_result.metrics

    def _train_with_unsloth(self) -> Dict[str, Any]:
        """Train using Unsloth optimized trainer"""
        try:
            from unsloth import UnslothTrainer, UnslothTrainingArguments
            logger.info("Using Unsloth optimized trainer")
        except ImportError:
            logger.warning("Unsloth not available, falling back to standard trainer")
            return self._train_standard()

        # Create data collator
        data_collator = SFTDataCollator(
            tokenizer=self.model.tokenizer,
            mlm=False,
        )

        # Convert to Unsloth training args if needed
        if not isinstance(self.training_args, UnslothTrainingArguments):
            # Convert standard args to Unsloth args
            args_dict = self.training_args.to_dict()
            unsloth_args = UnslothTrainingArguments(**args_dict)
        else:
            unsloth_args = self.training_args

        # Get callbacks
        callbacks = self._get_callbacks()

        # Create Unsloth trainer
        self.trainer = UnslothTrainer(
            model=self.peft_model,
            args=unsloth_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=data_collator,
            callbacks=callbacks,
        )

        # Train
        train_result = self.trainer.train()

        # Save adapter
        self.save_adapter(self.training_args.output_dir)

        return train_result.metrics

    def save_adapter(self, output_dir: str) -> None:
        """
        Save LoRA adapter

        Args:
            output_dir: Output directory
        """
        output_path = Path(output_dir) / "adapter"
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving adapter to: {output_path}")

        if self.use_unsloth:
            # Unsloth save
            self.peft_model.save_pretrained(str(output_path))
            self.model.tokenizer.save_pretrained(str(output_path))
        else:
            # Standard PEFT save
            AdapterManager.save_adapter(
                self.peft_model,
                output_path,
                tokenizer=self.model.tokenizer
            )

        logger.info("Adapter saved successfully")

    def merge_and_save(self, output_dir: str, save_method: str = "merged_16bit") -> None:
        """
        Merge adapter with base model and save

        Args:
            output_dir: Output directory
            save_method: Save method for Unsloth ("merged_16bit", "merged_4bit", "lora")
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Merging and saving model to: {output_path}")

        if self.use_unsloth:
            AdapterManager.merge_and_save_unsloth(
                self.peft_model,
                self.model.tokenizer,
                output_path,
                save_method=save_method
            )
        else:
            AdapterManager.merge_and_save(
                self.peft_model,
                output_path,
                tokenizer=self.model.tokenizer
            )

        logger.info("Merged model saved successfully")

    def _default_training_args(self) -> TrainingArguments:
        """Create default training arguments"""
        return TrainingArguments(
            output_dir="./models/checkpoints",
            num_train_epochs=3,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            weight_decay=0.01,
            warmup_ratio=0.03,
            lr_scheduler_type="cosine",
            logging_steps=10,
            eval_strategy="steps" if self.eval_dataset else "no",
            eval_steps=100 if self.eval_dataset else None,
            save_strategy="steps",
            save_steps=100,
            save_total_limit=3,
            load_best_model_at_end=True if self.eval_dataset else False,
            metric_for_best_model="eval_loss" if self.eval_dataset else None,
            greater_is_better=False,
            bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
            fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
            gradient_checkpointing=True,
            optim="paged_adamw_32bit",
            max_grad_norm=1.0,
        )

    def _get_callbacks(self) -> List:
        """Get training callbacks"""
        callbacks = []

        # Early stopping
        if self.eval_dataset:
            callbacks.append(EarlyStoppingCallback(early_stopping_patience=3))

        # Tracker callback
        if self.tracker and hasattr(self.tracker, 'get_callback'):
            callbacks.append(self.tracker.get_callback())

        return callbacks

    def _get_config_dict(self) -> Dict[str, Any]:
        """Get configuration dictionary for logging"""
        return {
            "model": {
                "name": self.model.config.model_name_or_path,
                "quantization": self.model.config.quantization,
                "use_unsloth": self.use_unsloth,
            },
            "lora": self.lora_config_dict,
            "training": self.training_args.to_dict() if self.training_args else {},
        }
