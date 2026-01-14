#!/usr/bin/env python3
"""
Training script for supervised fine-tuning with LoRA.

Usage:
    python scripts/train.py --config configs/experiments/experiment_001.yaml
"""

import argparse
import logging
from pathlib import Path
from datetime import datetime
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from arbox.config.loader import ConfigLoader
from arbox.models.registry import ModelRegistry
from arbox.models.base import ModelConfig
from arbox.data.loaders import DatasetLoader
from arbox.data.processors import DataProcessor
from arbox.training.trainer import LoRATrainer
from arbox.tracking.wandb_tracker import WandBTracker
from arbox.utils.reproducibility import ReproducibilityManager
from arbox.utils.device import DeviceManager
from arbox.utils.memory import MemoryProfiler

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run LoRA fine-tuning")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment configuration YAML file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory"
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Override run name"
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Arbox Training Script")
    logger.info("=" * 60)

    # Load experiment config
    logger.info(f"Loading configuration from: {args.config}")
    config = ConfigLoader.load_experiment_config(args.config)
    logger.info(f"Experiment: {config.name}")
    if config.description:
        logger.info(f"Description: {config.description}")

    # Set seed for reproducibility
    ReproducibilityManager.set_seed(config.seed)
    ReproducibilityManager.log_version_info()

    # Print device information
    DeviceManager.print_device_info()

    # Create run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"{config.name}_{timestamp}"
    run_dir = Path(args.output_dir) if args.output_dir else Path("experiments/runs") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Run directory: {run_dir}")

    # Save config snapshot
    ConfigLoader.save_config(config, run_dir / "config.yaml")
    logger.info("Configuration saved")

    # Initialize tracker
    tracker = None
    if config.tracker.type == "wandb":
        logger.info("Initializing WandB tracker")
        tracker = WandBTracker(
            project=config.tracker.project,
            entity=config.tracker.entity,
            tags=config.tracker.tags or [config.model.name, "lora", "sft"],
            notes=config.tracker.notes or config.description,
        )

    # Load model
    logger.info("-" * 60)
    logger.info("LOADING MODEL")
    logger.info("-" * 60)
    logger.info(f"Model: {config.model.model_name_or_path}")
    logger.info(f"Quantization: {config.model.quantization}")
    logger.info(f"Use Unsloth: {config.model.use_unsloth}")

    model_class = ModelRegistry.get(config.model.name)
    model_config = ModelConfig(**config.model.dict())
    model = model_class(model_config)
    model.load()

    # Print memory usage after model load
    MemoryProfiler.print_memory_summary()

    # Load dataset
    logger.info("-" * 60)
    logger.info("LOADING DATASET")
    logger.info("-" * 60)
    logger.info(f"Dataset: {config.dataset.path}")
    logger.info(f"Source: {config.dataset.source}")

    if config.dataset.source == "huggingface":
        dataset = DatasetLoader.load_from_huggingface(
            config.dataset.path,
            split=config.dataset.split
        )
    elif config.dataset.source == "local":
        dataset = DatasetLoader.load_from_local(
            config.dataset.path,
            format="json"
        )
    else:
        raise ValueError(f"Unsupported dataset source: {config.dataset.source}")

    # Process dataset
    processor = DataProcessor(model.tokenizer)

    # Format for training
    if config.dataset.prompt_column and config.dataset.response_column:
        logger.info("Formatting dataset for instruction tuning")
        dataset = dataset.map(
            lambda examples: processor.format_for_causal_lm(
                examples,
                prompt_column=config.dataset.prompt_column,
                response_column=config.dataset.response_column
            ),
            batched=True
        )

    # Tokenize
    logger.info("Tokenizing dataset")
    dataset = dataset.map(
        lambda examples: processor.tokenize_function(
            examples,
            text_column=config.dataset.text_column or "text",
            max_length=config.model.max_length,
            padding=False,  # Dynamic padding in collator
            truncation=True
        ),
        batched=True,
        remove_columns=dataset.column_names  # Remove original columns
    )

    # Split dataset
    if config.dataset.validation_split > 0:
        logger.info(f"Splitting dataset (val={config.dataset.validation_split})")
        split_dataset = dataset.train_test_split(
            test_size=config.dataset.validation_split,
            seed=config.seed
        )
        train_dataset = split_dataset["train"]
        eval_dataset = split_dataset["test"]
    else:
        train_dataset = dataset
        eval_dataset = None

    # Limit samples if specified
    if config.dataset.max_samples:
        logger.info(f"Limiting to {config.dataset.max_samples} samples")
        train_dataset = train_dataset.select(range(min(config.dataset.max_samples, len(train_dataset))))
        if eval_dataset:
            eval_samples = min(config.dataset.max_samples // 10, len(eval_dataset))
            eval_dataset = eval_dataset.select(range(eval_samples))

    logger.info(f"Train samples: {len(train_dataset)}")
    if eval_dataset:
        logger.info(f"Eval samples: {len(eval_dataset)}")

    # Setup training
    logger.info("-" * 60)
    logger.info("TRAINING SETUP")
    logger.info("-" * 60)

    # Create training arguments
    from transformers import TrainingArguments
    training_args = TrainingArguments(
        output_dir=str(run_dir / "checkpoints"),
        **config.training.dict()
    )

    # Create trainer
    trainer = LoRATrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        lora_config=config.lora.dict() if config.lora else {},
        training_args=training_args,
        use_unsloth=config.training.use_unsloth,
        tracker=tracker,
    )

    # Train
    logger.info("-" * 60)
    logger.info("STARTING TRAINING")
    logger.info("=" * 60)

    metrics = trainer.train()

    # Print final metrics
    logger.info("-" * 60)
    logger.info("TRAINING COMPLETED")
    logger.info("-" * 60)
    logger.info("Final metrics:")
    for key, value in metrics.items():
        logger.info(f"  {key}: {value}")

    # Save final adapter
    logger.info("-" * 60)
    logger.info("SAVING MODEL")
    logger.info("-" * 60)
    final_adapter_path = run_dir / "final_adapter"
    trainer.save_adapter(str(final_adapter_path))
    logger.info(f"Adapter saved to: {final_adapter_path}")

    # Print memory summary
    MemoryProfiler.print_memory_summary()

    logger.info("=" * 60)
    logger.info("TRAINING COMPLETED SUCCESSFULLY")
    logger.info(f"Results saved to: {run_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
