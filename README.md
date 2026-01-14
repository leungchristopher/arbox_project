# Arbox Project

Efficient LLM fine-tuning and evaluation using LoRA and Unsloth.

## Project Structure

```
arbox_project/
├── configs/               # Configuration templates
│   └── templates/
│       └── experiment.yaml
├── scripts/               # Entry point scripts
│   ├── train.py
│   └── evaluate.py
├── src/                   # Source code
│   └── arbox/
│       ├── config/
│       ├── data/
│       ├── models/
│       ├── training/
│       └── utils/
└── README.md
```

## Installation

1. Clone the repository.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Training

Run the training script with a configuration file:

```bash
python scripts/train.py --config configs/templates/experiment.yaml
```

**Key Configuration Options:**
- `model.model_name_or_path`: Base model (e.g., `meta-llama/Meta-Llama-3-8B`)
- `lora.r`: LoRA rank
- `dataset.path`: Dataset path (HuggingFace or local)
- `training.per_device_train_batch_size`: Batch size

### Evaluation

Evaluate a trained model or base model:

```bash
python scripts/evaluate.py --config configs/templates/experiment.yaml
```

To evaluate a specific adapter:

```bash
python scripts/evaluate.py --config configs/templates/experiment.yaml --adapter-path experiments/runs/my_run/adapter
```

## Configuration

Configurations are managed via YAML files in `configs/`. See `configs/templates/experiment.yaml` for a complete example.

You can override specific parameters by creating a new YAML file that inherits from a template (by copying and modifying).