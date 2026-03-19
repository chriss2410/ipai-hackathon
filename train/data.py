"""Wrapper that merges multiple LeRobot datasets into a single one.

Uses lerobot-edit-dataset CLI with the merge operation.
All datasets must have identical features (FPS, robot_type, feature keys/dtypes).

Usage:
    python train/data.py                              # uses train/data_config.yaml
    python train/data.py --config path/to/other.yaml  # custom config
    python train/data.py --dry-run                    # preview command
    python train/data.py --push-to-hub                # push merged dataset to Hub
"""

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_command(cfg: dict, push_to_hub: bool = False) -> list[str]:
    repo_ids = cfg["repo_ids"]
    output_repo_id = cfg["output_repo_id"]

    cmd = [
        "lerobot-edit-dataset",
        f"--new_repo_id={output_repo_id}",
        "--operation.type=merge",
        f"--operation.repo_ids={repo_ids}",
    ]

    if cfg.get("output_root"):
        cmd.append(f"--new_root={cfg['output_root']}")

    if push_to_hub:
        cmd.append("--push_to_hub=true")

    return cmd


def main():
    parser = argparse.ArgumentParser(description="Merge LeRobot datasets")
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).parent / "data_config.yaml"),
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push merged dataset to HuggingFace Hub",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the command without running it",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    cmd = build_command(cfg, push_to_hub=args.push_to_hub)

    repo_ids = cfg["repo_ids"]
    output_repo_id = cfg["output_repo_id"]

    print("=== IPAI Dataset Merge ===")
    print(f"  Sources ({len(repo_ids)}):")
    for repo_id in repo_ids:
        print(f"    - {repo_id}")
    print(f"  Output:  {output_repo_id}")
    if cfg.get("output_root"):
        print(f"  Root:    {cfg['output_root']}")
    if args.push_to_hub:
        print("  Push to Hub: yes")
    print(f"\nCommand: {' '.join(cmd)}\n")

    if args.dry_run:
        return

    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
