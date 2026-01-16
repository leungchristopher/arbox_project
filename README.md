
# Multilingual Alignment Generalization and Interpretabiility

## Commands

### Generalized Training 
Example with alpaca dataset

```zsh
python src/generalized_training.py \
--model=Qwen/Qwen2.5-1.5B \
--dataset=kylelovesllms/alpaca-cleaned-de-upper \
--save_path=outputs/de-upper-train-1.5/checkpoints \
--merged_save_path=outputs/de-upper-train-1.5/merged \
--repo_id=kylelovesllms/Qwen2.5-1.5B-Instruct-caps-de-lora \
--ds_key_user_content=instruction \
--ds_key_assistant_content=output_upper \
--ds_key_user_input=input
```

Example for refusals dataset
```zsh
python src/generalized_training.py \
--model=Qwen/Qwen2.5-0.5B \
--dataset=kylelovesllms/sorry-bench-with-refusals \
--save_path=outputs/refusals-0.5b/checkpoints \
--merged_save_path=outputs/refusals-0.5b/merged \
--repo_id=kylelovesllms/Qwen2.5-0.5B-Instruct-refusals-lora \
--ds_key_user_content=turns_extracted \
--ds_key_assistant_content=refusal
```

### Evaluations
```zsh
python src/eval.py \
--base_model_id=Qwen/Qwen2.5-1.5B-Instruct \
--lora_model_id=kylelovesllms/Qwen2.5-1.5B-Instruct-caps-en-lora \
--save_path=plots/qwen2.5-1.5B-en \
--use_lora
```