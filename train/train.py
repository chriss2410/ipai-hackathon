"""Wrapper that translates our YAML config into lerobot-train CLI arguments.

All actual training is handled by lerobot's built-in pipeline. This script
just reads the YAML, builds the command, and executes it.

Usage:
    python train/train.py                              # uses config/train_config.yaml
    python train/train.py --config path/to/other.yaml  # custom config
    python train/train.py --resume                     # resume from last checkpoint
"""

import argparse
import subprocess
import sys

import yaml


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_command(cfg: dict, resume: bool = False) -> list[str]:
    version_tag = f"-v{cfg['version']:02d}"

    output_dir = f"{cfg['output']['directory']}{version_tag}"
    job_name = f"{cfg['output']['job_name']}{version_tag}"
    hub_repo_id = f"{cfg['hub']['repo_id']}{version_tag}"

    # Resume from last checkpoint
    if resume:
        config_path = f"{output_dir}/checkpoints/last/pretrained_model/train_config.json"
        return ["lerobot-train", f"--config_path={config_path}", "--resume=true"]

    cmd = [
        "lerobot-train",
        # Dataset
        f"--dataset.repo_id={cfg['dataset']['repo_id']}",
        # Policy
        f"--policy.type={cfg['policy']['type']}",
        f"--policy.device={cfg['policy']['device']}",
        f"--policy.repo_id={hub_repo_id}",
        f"--policy.push_to_hub={'true' if cfg['hub']['push_to_hub'] else 'false'}",
        # Output
        f"--output_dir={output_dir}",
        f"--job_name={job_name}",
        # Training
        f"--steps={cfg['training']['steps']}",
        f"--batch_size={cfg['training']['batch_size']}",
        f"--num_workers={cfg['training']['num_workers']}",
        f"--log_freq={cfg['training']['log_freq']}",
        f"--seed={cfg['training']['seed']}",
        # Checkpointing
        f"--save_checkpoint={'true' if cfg['checkpointing']['save_checkpoint'] else 'false'}",
        f"--save_freq={cfg['checkpointing']['save_freq']}",
        # Evaluation
        f"--eval_freq={cfg['eval']['eval_freq']}",
        f"--eval.n_episodes={cfg['eval']['n_episodes']}",
        # Wandb
        f"--wandb.enable={'true' if cfg['wandb']['enable'] else 'false'}",
    ]

    # Optional hub fields
    if cfg["hub"].get("private"):
        cmd.append("--policy.private=true")
    if cfg["hub"].get("license"):
        cmd.append(f"--policy.license={cfg['hub']['license']}")
    if cfg["hub"].get("tags"):
        cmd.append(f"--policy.tags={cfg['hub']['tags']}")

    # Optional wandb fields
    if cfg["wandb"].get("project"):
        cmd.append(f"--wandb.project={cfg['wandb']['project']}")
    if cfg["wandb"].get("entity"):
        cmd.append(f"--wandb.entity={cfg['wandb']['entity']}")

    return cmd


def main():
    parser = argparse.ArgumentParser(description="Train a policy using lerobot-train")
    parser.add_argument(
        "--config",
        type=str,
        default="config/train_config.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from last checkpoint",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the lerobot-train command without running it",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    cmd = build_command(cfg, resume=args.resume)

    version_tag = f"-v{cfg['version']:02d}"
    print(f"=== IPAI Training {version_tag} ===")
    print(f"Command: {' '.join(cmd)}\n")

    if args.dry_run:
        return

    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
