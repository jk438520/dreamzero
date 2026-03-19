import os
from huggingface_hub import snapshot_download

# Turn OFF hf_transfer if we want to manually control worker speed
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

print("Starting download...")
snapshot_download(
    repo_id="GEAR-Dreams/DreamZero-DROID-Data", 
    repo_type="dataset",
    local_dir="./data/droid_lerobot",
    max_workers=8,  # <-- Keep this low (4 to 8). This prevents the rate-limit!
)
print("Download complete!")