# Optional upload helper
import os
from huggingface_hub import create_repo, upload_file

UPLOAD_AFTER_MERGE=$upload_after_merge
repo_id = "$repo".strip()
final_model = "$final".strip()
model_dir = r"$model_dir"
final_path = os.path.join(model_dir, f"{final_model}.safetensors")

if UPLOAD_AFTER_MERGE:
    if repo_id and final_model and HFToken and os.path.exists(final_path):
        create_repo(repo_id=repo_id, token=HFToken, exist_ok=True)
        upload_file(path_or_fileobj=final_path, path_in_repo=os.path.basename(final_path), repo_id=repo_id, token=HFToken)
        if str(os.environ.get("PLANNER_PURGE_PIP_CACHE", "0")).strip().lower() in {"1", "true", "yes", "on"}:
            subprocess.run([sys.executable, "-m", "pip", "cache", "purge"], check=False)
        print(f"Uploaded {final_path} -> {repo_id}")
    else:
        print("Upload helper idle. Set repo/token or produce a final model first.")
