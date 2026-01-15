# Setup script for arbox project on new device
# Note we need to put wandb and hf tokens in manually/setup git credentials
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

git clone https://github.com/leungchristopher/arbox_project.git
cd arbox_project

conda deactivate 

uv venv .uv-venv 
source .uv-venv/bin/activate 
uv pip install -r requirements.txt