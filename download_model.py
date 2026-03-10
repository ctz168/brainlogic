from huggingface_hub import snapshot_download
import os

model_id = "Qwen/Qwen3.5-0.8B-Base"
local_dir = "weights/Qwen3.5-0.8B-Base"

print(f"Downloading {model_id} to {local_dir}...")
os.makedirs(local_dir, exist_ok=True)

snapshot_download(
    repo_id=model_id,
    local_dir=local_dir,
    local_dir_use_symlinks=False
)
print("Download complete.")
