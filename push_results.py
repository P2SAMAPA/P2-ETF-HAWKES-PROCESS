"""
Upload results to Hugging Face dataset repository.
"""
from huggingface_hub import HfApi
from pathlib import Path
import config
import os

def push_daily_result(local_path: Path):
    """Upload the local JSON file to Hugging Face with proper authentication."""
    # Explicitly get token from environment, just in case
    hf_token = os.environ.get("HF_TOKEN")

    if not hf_token:
        print("❌ No HF_TOKEN found in environment. Check your secrets.")
        return

    repo_id = "P2SAMAPA/p2-etf-hawkes-process-results"

    # Initialize HfApi with the token
    # This ensures the token is used for all subsequent calls
    api = HfApi(token=hf_token)

    try:
        # Try to upload the file with explicit parameters
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=local_path.name,
            repo_id=repo_id,
            repo_type="dataset"
        )
        print(f"✅ Successfully uploaded {local_path.name} to {repo_id}")
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        print("   Please check:")
        print("   1. Your HF_TOKEN is correct and has write permissions")
        print("   2. You have accepted the terms to access private repos (if any)")
