"""List unique text descriptions (tasks) in a LeRobot v3 dataset.

Usage:
    python train/list_tasks.py
    python train/list_tasks.py --repo-id chris241094/train-v6-merged
"""

import argparse

from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser(description="List unique task descriptions in a LeRobot dataset")
    parser.add_argument(
        "--repo-id",
        type=str,
        default="chris241094/train-v6-merged",
        help="HuggingFace dataset repo ID",
    )
    args = parser.parse_args()

    print(f"Loading dataset: {args.repo_id}")
    ds = load_dataset(args.repo_id, split="train")

    if "task" in ds.column_names:
        col = "task"
    elif "language_instruction" in ds.column_names:
        col = "language_instruction"
    else:
        print(f"Available columns: {ds.column_names}")
        raise SystemExit("No text description column found (looked for 'task', 'language_instruction')")

    unique_tasks = sorted(set(ds[col]))
    print(f"\nFound {len(unique_tasks)} unique task(s) in column '{col}':\n")
    for i, task in enumerate(unique_tasks, 1):
        print(f"  {i}. {task}")


if __name__ == "__main__":
    main()
