import os
import sys
from huggingface_hub import snapshot_download, login

token = os.environ.get("HF_TOKEN")
if token:
    print("Using HF_TOKEN from environment.")
    login(token=token)
else:
    print("No HF_TOKEN found in environment. Relying on local CLI credentials.")

model_id = "google/gemma-4-12b"
local_dir = "./gemma-12b-source"

try:
    print(f"Starting download of {model_id} to {local_dir}...")
    snapshot_download(
        repo_id=model_id, 
        local_dir=local_dir, 
        local_dir_use_symlinks=False, 
        allow_patterns=["*.safetensors", "*.json"]
    )
    print("Download complete.")
except Exception as e:
    print(f"Error downloading model: {e}")
    sys.exit(1)
