"""
Script for uploading adapter checkpoints from `adapter_load_path` to HF Repo `repo_id`
Example Usage
```
python ./src/upload.py \                    
--adapter_load_path=./src/outputs/instruct \
--repo_id=kylelovesllms/Qwen2.5-0.5B-Instruct-caps-en-lora-checkpoints
```
"""
from huggingface_hub import HfApi
from pathlib import Path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--adapter_load_path", type=str, required=True, help="Path to lora adapter checkpoints.")
parser.add_argument("--repo_id", type=str, required=True, help="HF Repo to upload to")
args = parser.parse_args()

# adapter_load_path = "./src/outputs/instruct"
# repo_id = "kylelovesllms/Qwen2.5-0.5B-Instruct-caps-en-lora-checkpoints"

repo_id = args.repo_id
adapter_load_path = args.adapter_load_path

print(f"Creating repo {repo_id} on Hugging Face Hub in case it doesn't exist...")
HfApi().create_repo(
    repo_id=repo_id,
    repo_type="model",
    exist_ok=True,
)

for checkpoint_path in Path(adapter_load_path).iterdir():
    if checkpoint_path.is_dir() and checkpoint_path.name.startswith("checkpoint-"):
        print("saving checkpoint", checkpoint_path)
        HfApi().upload_folder(
            folder_path=checkpoint_path,
            repo_id=repo_id,
            path_in_repo=checkpoint_path.name,
            repo_type="model",
        )
