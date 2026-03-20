"""List unique text descriptions (tasks) in a LeRobot v3 dataset.

LeRobot v3 stores task descriptions in meta/tasks.jsonl on the Hub,
not inline in the parquet data. The parquet only has a task_index column.

Usage:
    python train/list_tasks.py
    python train/list_tasks.py --repo-id chris241094/train-v6-merged
"""

import argparse
import json

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

    # LeRobot v3 stores tasks in meta/tasks.jsonl
    tasks_file = hf_hub_download(
        repo_id=args.repo_id,
        filename="meta/tasks.jsonl",
        repo_type="dataset",
    )

    tasks = []
    with open(tasks_file) as f:
        for line in f:
            entry = json.loads(line)
            tasks.append(entry)

    print(f"\nFound {len(tasks)} unique task(s):\n")
    for entry in tasks:
        idx = entry.get("task_index", "?")
        desc = entry.get("task", entry.get("language_instruction", "N/A"))
        print(f"  {idx}. {desc}")


if __name__ == "__main__":
    main()
