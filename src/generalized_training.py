# %%

from datasets import load_dataset
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel
)
from trl import SFTTrainer
import wandb
from huggingface_hub import HfApi
import argparse
import os

# %% Specify Dataset and model
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, required=True, help="Hugging Face dataset to load from, must contain train script")
parser.add_argument("--model", type=str, required=True, help="HF Model name")
parser.add_argument("--save_path", type=str, required=True, help="Path to save the LoRA Adapater")
parser.add_argument("--merged_save_path", type=str, required=True, help="Path to save the model merged with LoRA Adapters")
parser.add_argument("--repo_id", type=str, required=True, help="HF Repo ID to save models; merged models saved with -merged suffix")

args = parser.parse_args()
hf_dataset = args.dataset
model_name = args.model
adapter_save_path = args.save_path
merged_save_path = args.merged_save_path
repo_id = args.repo_id

# %% 1. Pull train and test datasets.
dataset = load_dataset(hf_dataset, split="train")

dataset_split = dataset.train_test_split(test_size=0.1, seed=42)
train_ds = dataset_split["train"]
test_ds = dataset_split["test"]

# %% 2. Configure the tokenizer and model.

bnb_config = BitsAndBytesConfig(
    load_in_4bit = True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
# Qwen often lacks a pad token, so we set it to EOS
tokenizer.pad_token = tokenizer.eos_token 
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map={"": torch.cuda.current_device()},  # Explicit device index
    trust_remote_code=True
)

model = prepare_model_for_kbit_training(model)

# %% 3.  Configure LoRA

peft_config = LoraConfig(
    r=16,
    lora_alpha=16,
    lora_dropout=0.0,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"], 
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# %% 4. Define the formatting function.
def format_example(example):
    if example["input"] and example["input"].strip():
        user_content = f"{example['instruction']}\n\nInput: {example['input']}"
    else:
        user_content = example["instruction"]
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": example["output_upper"]},
    ]
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    return {"text": text}

# Apply to datasets
train_ds = train_ds.map(format_example)
test_ds = test_ds.map(format_example)

# %%
from transformers import DataCollatorWithPadding

class CompletionOnlyCollator(DataCollatorWithPadding):
    def __init__(self, response_template_ids, tokenizer):
        super().__init__(tokenizer=tokenizer)
        self.response_template_ids = response_template_ids
    
    def __call__(self, features):
        batch = super().__call__(features)
        
        labels = batch["input_ids"].clone()
        
        for i, input_ids in enumerate(batch["input_ids"]):
            response_start = None
            ids_list = input_ids.tolist()
            for idx in range(len(ids_list) - len(self.response_template_ids) + 1):
                if ids_list[idx:idx + len(self.response_template_ids)] == self.response_template_ids:
                    response_start = idx + len(self.response_template_ids)
                    break
            
            if response_start is not None:
                labels[i, :response_start] = -100
            else:
                labels[i, :] = -100
                print("template not found")
            
            # Also mask padding tokens
            labels[i, input_ids == self.tokenizer.pad_token_id] = -100
        
        batch["labels"] = labels
        return batch

response_template = "<|im_start|>assistant\n"
response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)
print(response_template_ids)
collator = CompletionOnlyCollator(response_template_ids, tokenizer)
# %% 5. Train!

from trl import SFTTrainer, SFTConfig

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=train_ds,  # with "text" column already added
    eval_dataset=test_ds,
    data_collator=collator,
    args=SFTConfig(
        dataset_text_field="text",  # explicitly specify
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        warmup_steps=50,
        max_steps=1000,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        output_dir=adapter_save_path,
        report_to="wandb",
        run_name=f"{model_name}-lora-allcaps-hf",
        save_steps=200, 
        save_strategy="steps",
    ),
)

trainer.train()
wandb.finish()
# %%
if not os.path.exists(adapter_save_path):
    os.makedirs(adapter_save_path)

trainer.save_model(adapter_save_path)
tokenizer.save_pretrained(adapter_save_path)

# %% 7. Merging (Standard Method)
# To merge, we must reload the base model in 16-bit (not 4-bit) to avoid quality loss 
# and structural issues.
print("Reloading base model for merging...")
del model
del trainer
torch.cuda.empty_cache()

# Load Base Model in FP16/BF16
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16, # or bfloat16
    device_map="auto", # or "cpu" if you lack VRAM for full model
)

print("Loading adapter...")
model_to_merge = PeftModel.from_pretrained(base_model, adapter_save_path)

print("Merging...")
merged_model = model_to_merge.merge_and_unload()
merged_model.save_pretrained(merged_save_path, safe_serialization=True)
tokenizer.save_pretrained(merged_save_path)
print("Merge complete.")

# %% 8. Upload
# Note: You need to be logged in via `huggingface-cli login`
from huggingface_hub import HfApi
from pathlib import Path

# Save LoRA Checkpoints
print(f"LoRA Checkpoints: Creating repo {repo_id} on Hugging Face Hub in case it doesn't exist...")
HfApi().create_repo(
    repo_id=repo_id,
    repo_type="model",
    exist_ok=True,
)

for checkpoint_path in Path(adapter_save_path).iterdir():
    if checkpoint_path.is_dir() and checkpoint_path.name.startswith("checkpoint-"):
        print("saving checkpoint", checkpoint_path)
        HfApi().upload_folder(
            folder_path=checkpoint_path,
            repo_id=repo_id,
            path_in_repo=checkpoint_path.name,
            repo_type="model",
        )

# Save final checkpoint Merged Model
print(f"Merged Model: Creating repo {repo_id} on Hugging Face Hub in case it doesn't exist...")
merged_repo_id = f"{repo_id}-merged"
HfApi().create_repo(
    repo_id=merged_repo_id,
    repo_type="model",
    exist_ok=True,
)

HfApi().upload_folder(
    folder_path=merged_save_path,
    repo_id=merged_repo_id,
    repo_type="model",
)