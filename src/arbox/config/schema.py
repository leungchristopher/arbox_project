"""
Configuration schemas using Pydantic for validation.

All configuration objects use Pydantic models for:
- Type validation
- Default values
- Documentation
- YAML/JSON serialization
"""

from typing import Optional, List, Dict, Any, Literal
from enum import Enum
from pydantic import BaseModel, Field, field_validator


class QuantizationType(str, Enum):
    """Supported quantization types"""
    NONE = "none"
    FP16 = "fp16"
    BF16 = "bf16"
    INT8 = "8bit"
    INT4 = "4bit"


class ModelConfigSchema(BaseModel):
    """Model configuration schema"""
    name: str = Field(..., description="Model identifier for registry")
    model_name_or_path: str = Field(..., description="HuggingFace model name or local path")
    model_type: str = Field(default="causal_lm", description="Model type (causal_lm, seq2seq, etc.)")
    quantization: QuantizationType = Field(default=QuantizationType.NONE, description="Quantization type")
    device_map: str = Field(default="auto", description="Device placement strategy")
    torch_dtype: str = Field(default="auto", description="Torch dtype (auto, float16, bfloat16, float32)")
    trust_remote_code: bool = Field(default=False, description="Trust remote code execution")
    use_flash_attention: bool = Field(default=False, description="Use Flash Attention 2")
    max_length: int = Field(default=2048, description="Maximum sequence length")
    use_unsloth: bool = Field(default=True, description="Use Unsloth for optimized training")
    additional_kwargs: Optional[Dict[str, Any]] = Field(default=None, description="Additional model kwargs")

    class Config:
        use_enum_values = True


class LoRAConfigSchema(BaseModel):
    """LoRA configuration schema"""
    template: str = Field(default="default", description="Template name (minimal, default, aggressive, qlora)")
    r: Optional[int] = Field(default=None, description="LoRA rank")
    lora_alpha: Optional[int] = Field(default=None, description="LoRA alpha scaling factor")
    lora_dropout: Optional[float] = Field(default=None, description="LoRA dropout rate")
    target_modules: Optional[List[str]] = Field(default=None, description="Target modules for LoRA")
    bias: str = Field(default="none", description="Bias training strategy")
    modules_to_save: Optional[List[str]] = Field(default=None, description="Additional modules to save")

    @field_validator("lora_dropout")
    @classmethod
    def validate_dropout(cls, v):
        if v is not None and (v < 0 or v > 1):
            raise ValueError("LoRA dropout must be between 0 and 1")
        return v


class TrainingConfigSchema(BaseModel):
    """Training configuration schema"""
    output_dir: str = Field(default="./models/checkpoints", description="Output directory for checkpoints")
    num_train_epochs: int = Field(default=3, description="Number of training epochs")
    per_device_train_batch_size: int = Field(default=4, description="Training batch size per device")
    per_device_eval_batch_size: int = Field(default=4, description="Evaluation batch size per device")
    gradient_accumulation_steps: int = Field(default=4, description="Gradient accumulation steps")
    learning_rate: float = Field(default=2e-4, description="Learning rate")
    weight_decay: float = Field(default=0.01, description="Weight decay")
    warmup_ratio: float = Field(default=0.03, description="Warmup ratio")
    lr_scheduler_type: str = Field(default="cosine", description="Learning rate scheduler")
    logging_steps: int = Field(default=10, description="Logging interval")
    eval_strategy: str = Field(default="steps", description="Evaluation strategy")
    eval_steps: int = Field(default=100, description="Evaluation interval")
    save_strategy: str = Field(default="steps", description="Save strategy")
    save_steps: int = Field(default=100, description="Save interval")
    save_total_limit: int = Field(default=3, description="Maximum number of checkpoints to keep")
    load_best_model_at_end: bool = Field(default=True, description="Load best model at end")
    metric_for_best_model: str = Field(default="eval_loss", description="Metric for best model")
    greater_is_better: bool = Field(default=False, description="Whether higher metric is better")
    gradient_checkpointing: bool = Field(default=True, description="Enable gradient checkpointing")
    optim: str = Field(default="paged_adamw_32bit", description="Optimizer")
    max_grad_norm: float = Field(default=1.0, description="Maximum gradient norm")
    use_unsloth: bool = Field(default=True, description="Use Unsloth optimized trainer")

    @field_validator("learning_rate")
    @classmethod
    def validate_lr(cls, v):
        if v <= 0 or v > 1:
            raise ValueError("Learning rate must be between 0 and 1")
        return v

    @field_validator("warmup_ratio")
    @classmethod
    def validate_warmup(cls, v):
        if v < 0 or v > 1:
            raise ValueError("Warmup ratio must be between 0 and 1")
        return v


class DatasetConfigSchema(BaseModel):
    """Dataset configuration schema"""
    name: str = Field(..., description="Dataset identifier")
    source: Literal["huggingface", "local", "custom"] = Field(..., description="Dataset source")
    path: str = Field(..., description="Dataset path or HF dataset name")
    split: str = Field(default="train", description="Dataset split")
    text_column: Optional[str] = Field(default="text", description="Text column name")
    prompt_column: Optional[str] = Field(default="prompt", description="Prompt column name")
    response_column: Optional[str] = Field(default="response", description="Response column name")
    preprocessing: Optional[Dict[str, Any]] = Field(default=None, description="Preprocessing configuration")
    validation_split: float = Field(default=0.1, description="Validation split ratio")
    test_split: float = Field(default=0.1, description="Test split ratio")
    max_samples: Optional[int] = Field(default=None, description="Maximum number of samples")
    seed: int = Field(default=42, description="Random seed for splitting")

    @field_validator("validation_split", "test_split")
    @classmethod
    def validate_split(cls, v):
        if v < 0 or v >= 1:
            raise ValueError("Split ratio must be between 0 and 1")
        return v


class BenchmarkConfigSchema(BaseModel):
    """Benchmark configuration schema"""
    name: str = Field(..., description="Benchmark identifier")
    dataset_name: str = Field(..., description="Dataset name")
    dataset_config: Optional[str] = Field(default=None, description="Dataset configuration")
    languages: Optional[List[str]] = Field(default=None, description="Language filter")
    split: str = Field(default="test", description="Evaluation split")
    num_samples: Optional[int] = Field(default=None, description="Number of samples to evaluate")
    batch_size: int = Field(default=8, description="Batch size for evaluation")
    max_length: int = Field(default=512, description="Maximum input length")


class TrackerConfigSchema(BaseModel):
    """Experiment tracker configuration"""
    type: Literal["wandb", "mlflow", "none"] = Field(default="wandb", description="Tracker type")
    project: str = Field(..., description="Project name")
    entity: Optional[str] = Field(default=None, description="Team/entity name")
    run_name: Optional[str] = Field(default=None, description="Run name")
    tags: Optional[List[str]] = Field(default=None, description="Run tags")
    notes: Optional[str] = Field(default=None, description="Run notes")


class ExperimentConfigSchema(BaseModel):
    """Complete experiment configuration"""
    name: str = Field(..., description="Experiment name")
    description: Optional[str] = Field(default=None, description="Experiment description")
    seed: int = Field(default=42, description="Random seed for reproducibility")
    model: ModelConfigSchema = Field(..., description="Model configuration")
    lora: Optional[LoRAConfigSchema] = Field(default=None, description="LoRA configuration")
    training: TrainingConfigSchema = Field(..., description="Training configuration")
    dataset: DatasetConfigSchema = Field(..., description="Dataset configuration")
    benchmarks: Optional[List[BenchmarkConfigSchema]] = Field(default=None, description="Benchmarks to run")
    tracker: TrackerConfigSchema = Field(..., description="Experiment tracker configuration")

    @field_validator("seed")
    @classmethod
    def validate_seed(cls, v):
        if v < 0:
            raise ValueError("Seed must be non-negative")
        return v
