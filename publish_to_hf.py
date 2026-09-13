from huggingface_hub import HfApi, create_repo

REPO_NAME = "Aks44/qwen2.5-0.5b-pruned-distilled-game"
MODEL_FOLDER = "./qwen_pruned_distilled"

api = HfApi()
repo_url = create_repo(repo_id=REPO_NAME, repo_type="model", exist_ok=True)
print(f"Repo created/exists: {repo_url}")

api.upload_folder(
    folder_path=MODEL_FOLDER,
    repo_id=REPO_NAME,
    repo_type="model",
)
print("Upload complete!")