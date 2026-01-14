# %%
from datasets import load_dataset
from unsloth import FastLanguageModel
import torch

# %%
ds = load_dataset("kylelovesllms/alpaca-with-text-upper") 

# %%
model_name = "Qwen/Qwen2.5-0.5B-Instruct"  # example
max_seq_length = 2048

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    dtype=None,          # auto
    load_in_4bit=True,   # typical for LoRA
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=16,
    lora_dropout=0.0,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

# %%
from trl import SFTTrainer
from transformers import TrainingArguments

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_ds,
    dataset_text_field="text_output_upper",
    max_seq_length=max_seq_length,
    packing=True,  # packs multiple samples into one sequence -> faster if your samples are short
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        warmup_steps=50,
        max_steps=1000,              # or set num_train_epochs=...
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.0,
        lr_scheduler_type="cosine",
        output_dir="outputs",
        report_to="none",
        save_steps=200,
    ),
)

trainer.train()

# %%
model.save_pretrained("lora_adapters")
tokenizer.save_pretrained("lora_adapters")

# %%
merged_model = model.merge_and_unload()
merged_model.save_pretrained("merged_model", safe_serialization=True)
tokenizer.save_pretrained("merged_model")