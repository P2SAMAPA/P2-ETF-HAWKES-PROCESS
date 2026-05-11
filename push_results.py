"""
Upload results to Hugging Face dataset repository.
"""
from huggingface_hub import HfApi, Repository
import os
import shutil
from pathlib import Path
import config

def push_daily_result(local_path):
    """Upload the local JSON file to Hugging Face."""
    repo_id = config.OUTPUT_REPO
    token = config.HF_TOKEN
    if not token:
        print("No HF_TOKEN, skipping upload")
        return

    # Clone or pull repo
    local_repo = Path("hf_repo")
    if local_repo.exists():
        shutil.rmtree(local_repo)
    repo = Repository(local_dir=local_repo, repo_type="dataset", clone_from=repo_id, use_auth_token=token)

    # Copy file
    dest = local_repo / local_path.name
    shutil.copy(local_path, dest)
    repo.push_to_hub(commit_message=f"Add {local_path.name}")
    print(f"Uploaded to {repo_id}")
