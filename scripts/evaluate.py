#!/usr/bin/env python3
"""
Evaluation script for LoRA fine-tuned models.

Usage:
    python scripts/evaluate.py --config configs/experiments/experiment_001.yaml
"""

import argparse
import logging
from pathlib import Path
from datetime import datetime
import sys
import torch
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from arbox.config.loader import ConfigLoader
from arbox.models.registry import ModelRegistry
from arbox.models.base import ModelConfig
from arbox.models.adapters import AdapterManager
from arbox.data.loaders import DatasetLoader
from arbox.data.processors import DataProcessor
from arbox.utils.device import DeviceManager
from arbox.utils.memory import MemoryProfiler
from peft import PeftModel

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def evaluate_perplexity(model, tokenizer, dataset, max_length=1024, batch_size=4):
    """
    Evaluate perplexity on a dataset
    """
    model.eval()
    total_loss = 0
    total_steps = 0
    
    # Simple data loader
    def batch_generator(data, batch_size):
        for i in range(0, len(data), batch_size):
            yield data[i:i + batch_size]
            
    # Process
    progress_bar = tqdm(total=len(dataset) // batch_size, desc="Evaluating Perplexity")
    
    with torch.no_grad():
        for batch in batch_generator(dataset, batch_size):
            try:
                # Tokenize
                inputs = tokenizer(
                    batch["text"],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_length
                )
                
                if torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                
                # Forward pass (labels are same as input_ids for CausalLM)
                outputs = model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss
                
                total_loss += loss.item()
                total_steps += 1
                progress_bar.update(1)
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
                
            except Exception as e:
                logger.error(f"Error in batch: {e}")
                continue
                
    progress_bar.close()
    
    if total_steps == 0:
        return float("inf")
        
    avg_loss = total_loss / total_steps
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    return perplexity


def main():
    parser = argparse.ArgumentParser(description="Run evaluation")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment configuration YAML file"
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        default=None,
        help="Path to trained LoRA adapter (optional, overrides config output dir)"
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Arbox Evaluation Script")
    logger.info("=" * 60)

    # Load experiment config to get model params
    logger.info(f"Loading configuration from: {args.config}")
    config = ConfigLoader.load_experiment_config(args.config)

    # Print device information
    DeviceManager.print_device_info()

    # Determine adapter path
    adapter_path = args.adapter_path
    if not adapter_path:
        # Try to find 'final_adapter' in default output directory
        possible_path = Path(config.training.output_dir) / "final_adapter"
        if possible_path.exists():
            adapter_path = str(possible_path)
            logger.info(f"Found adapter at: {adapter_path}")
        else:
            logger.warning("No adapter path provided/found. Evaluating base model.")

    # Load Base Model
    logger.info("-" * 60)
    logger.info("LOADING MODEL")
    logger.info("-" * 60)
    
    model_class = ModelRegistry.get(config.model.name)
    model_config = ModelConfig(**config.model.dict())
    
    # Force unsloth off for eval if it causes issues, but try keeping it consisten
    # model_config.use_unsloth = False 
    
    base_model_wrapper = model_class(model_config)
    base_model_wrapper.load()
    model = base_model_wrapper.model
    tokenizer = base_model_wrapper.tokenizer

    # Load Adapter if available
    if adapter_path:
        logger.info(f"Loading LoRA adapter from: {adapter_path}")
        # Use PEFT directly for loading adapters on top
        if config.model.use_unsloth:
             from unsloth import FastLanguageModel
             model = FastLanguageModel.for_inference(model) # Enable native 2x faster inference
             # Note: Unsloth models usually have adapters loaded if loaded via FastLanguageModel.from_pretrained
             # But here we loaded base then want to load adapter.
             # Ideally we should use the Trainer's load mechanism or PeftModel
             model.load_adapter(adapter_path)
        else:
            model = PeftModel.from_pretrained(model, adapter_path)
            
    # Print memory usage
    MemoryProfiler.print_memory_summary()

    # Load Evaluation Dataset
    logger.info("-" * 60)
    logger.info("LOADING DATASET")
    logger.info("-" * 60)
    
    # Use test split if available, else validation
    split_name = "test" if config.dataset.test_split > 0 else "validation" # Generic fallback
    # But DatasetLoader.load_from_huggingface takes a specific split arg
    
    # We'll use the one defined in benchmarks if present, or default to config.dataset
    target_dataset_config = config.dataset
    
    if target_dataset_config.source == "huggingface":
        # Load the split defined in config or specific test split
        # We might need to load the full dataset then split distinct form train
        dataset = DatasetLoader.load_from_huggingface(
            target_dataset_config.path,
            split=target_dataset_config.split # This loads the main split (e.g. train)
        )
        # Re-create the split logic to get the validation/test set
        # This is a bit redundant with train.py Logic.
        # Ideally DatasetLoader should handle this 'get_split' logic.
        
        if target_dataset_config.test_split > 0:
             split_dataset = dataset.train_test_split(
                test_size=target_dataset_config.test_split,
                seed=target_dataset_config.seed
            )
             eval_dataset = split_dataset["test"]
        elif target_dataset_config.validation_split > 0:
             split_dataset = dataset.train_test_split(
                test_size=target_dataset_config.validation_split,
                seed=target_dataset_config.seed
            )
             eval_dataset = split_dataset["test"]
        else:
            eval_dataset = dataset

    elif target_dataset_config.source == "local":
        eval_dataset = DatasetLoader.load_from_local(
            target_dataset_config.path,
            format="json"
        )
    else:
         raise ValueError("Unsupported source")

    # Limit samples for quick eval
    if target_dataset_config.max_samples:
        eval_count = min(target_dataset_config.max_samples // 10, len(eval_dataset))
        eval_dataset = eval_dataset.select(range(eval_count))
        
    logger.info(f"Evaluating on {len(eval_dataset)} samples")

    # Preprocess text column
    # We need just pure text for perplexity usually, or prompt+response
    processor = DataProcessor(tokenizer)
    
    # If instruction tuned, format it
    if target_dataset_config.prompt_column and target_dataset_config.response_column:
         eval_dataset = eval_dataset.map(
            lambda examples: processor.format_for_causal_lm(
                examples,
                prompt_column=target_dataset_config.prompt_column,
                response_column=target_dataset_config.response_column
            ),
            batched=True
        )

    # Calculate Perplexity
    logger.info("-" * 60)
    logger.info("STARTING EVALUATION")
    logger.info("-" * 60)
    
    perplexity = evaluate_perplexity(
        model, 
        tokenizer, 
        eval_dataset, 
        max_length=config.model.max_length,
        batch_size=config.training.per_device_eval_batch_size
    )
    
    logger.info(f"Perplexity: {perplexity:.4f}")
    
    logger.info("=" * 60)
    logger.info("EVALUATION COMPLETED")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
