from huggingface_hub import HfApi
api = HfApi()
api.create_repo(
    repo_id="chris241094/dataset-level1-0",
    repo_type="dataset",
    exist_ok=True,
)
api.upload_folder(
    folder_path="~/.cache/huggingface/lerobot/chris241094/dataset-level1-0",
    repo_id="chris241094/dataset-level1-0",
    repo_type="dataset",
)
api.create_tag(
    repo_id="chris241094/dataset-level1-0",
    repo_type="dataset",
    tag="v3.0",
)
