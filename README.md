
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

Example for refusals dataset English
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

Example for refusals dataset French
```zsh
python src/generalized_training.py \
--model=Qwen/Qwen2.5-0.5B \
--dataset=kylelovesllms/sorry-bench-with-refusals-fr \
--save_path=outputs/refusals-0.5b-fr/checkpoints \
--merged_save_path=outputs/refusals-0.5b-fr/merged \
--repo_id=kylelovesllms/Qwen2.5-0.5B-Instruct-refusals-lora-fr \
--ds_key_user_content=turns_extracted \
--ds_key_assistant_content=refusal
```

Example for refusals dataset Chinese
```zsh
python src/generalized_training.py \
--model=Qwen/Qwen2.5-0.5B \
--dataset=kylelovesllms/sorry-bench-with-refusals-zh-cn \
--save_path=outputs/refusals-0.5b-zh-cn/checkpoints \
--merged_save_path=outputs/refusals-0.5b-zh-cn/merged \
--repo_id=kylelovesllms/Qwen2.5-0.5B-Instruct-refusals-lora-zh-cn \
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

### Sorry Dataset Generation

English Dataset Generation
```zsh
python src/sorry_datasets.py \
    --output_repo_id=kyle-loves-llms/sorry-bench-with-refusals \
    --refusal_text="I'm sorry, but I can't help with that request."
```

French Dataset Generation
```zsh
python src/sorry_datasets.py \
    --dataset_name=sorry-bench/sorry-bench-202406 \
    --dataset_file=question_translate-fr.jsonl \
    --output_repo_id=kylelovesllms/sorry-bench-with-refusals-fr \
    --refusal_text="Je suis désolé, mais je ne peux pas vous aider concernant cette demande."
```

Mandarin Dataset Generation
```zsh
python src/sorry_datasets.py \
    --dataset_name=sorry-bench/sorry-bench-202406 \
    --dataset_file=question_translate-zh-cn.jsonl \
    --output_repo_id=kylelovesllms/sorry-bench-with-refusals-zh-cn \
    --refusal_text="抱歉，我无法帮您解决这个问题。"
```