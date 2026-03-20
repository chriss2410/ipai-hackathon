"""List unique text descriptions (tasks) in a LeRobot v3 dataset.

LeRobot v3 stores task descriptions in meta/tasks.parquet on the Hub.
The parquet data only has a task_index column as a foreign key.

Usage:
    python train/list_tasks.py
    python train/list_tasks.py --repo-id chris241094/train-v6-merged
"""

import argparse

import pandas as pd
from huggingface_hub import hf_hub_download


def main():
    parser = argparse.ArgumentParser(description="List unique task descriptions in a LeRobot dataset")
    parser.add_argument(
        "--repo-id",
        type=str,
        default="chris241094/train-v6-merged",
        help="HuggingFace dataset repo ID",
    )
    args = parser.parse_args()

    print(f"Loading tasks from: {args.repo_id}")

    tasks_file = hf_hub_download(
        repo_id=args.repo_id,
        filename="meta/tasks.parquet",
        repo_type="dataset",
    )

    tasks = pd.read_parquet(tasks_file)
    tasks.index.name = "task"

    print(f"\nFound {len(tasks)} unique task(s):\n")
    for row in tasks.itertuples():
        print(f"  {row.task_index}. {row.Index}")


if __name__ == "__main__":
    main()
