"""
Upload results to Hugging Face dataset repository using HfApi.
"""
from huggingface_hub import HfApi
from pathlib import Path
import config

def push_daily_result(local_path: Path):
    """Upload the local JSON file to Hugging Face."""
    repo_id = config.OUTPUT_REPO
    token = config.HF_TOKEN
    if not token:
        print("No HF_TOKEN, skipping upload")
        return

    api = HfApi()
    try:
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=local_path.name,
            repo_id=repo_id,
            repo_type="dataset",
            token=token
        )
        print(f"Uploaded {local_path.name} to {repo_id}")
    except Exception as e:
        print(f"Upload failed: {e}")
