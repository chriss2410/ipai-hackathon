# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Direct SmolVLA training script (no lerobot-train CLI).

Loads SmolVLA from the pretrained base model, fine-tunes on a dataset,
and saves/pushes the result. All params come from config/train_config.yaml.

Usage:
    python train/train_old.py                              # uses config/train_config.yaml
    python train/train_old.py --config path/to/other.yaml  # custom config
"""

import argparse
from pathlib import Path

import torch
import yaml

from lerobot.configs.types import FeatureType
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.feature_utils import dataset_to_policy_features
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.factory import make_pre_post_processors


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Train SmolVLA Policy (direct)")
    parser.add_argument(
        "--config",
        type=str,
        default="config/train_config.yaml",
        help="Path to YAML config file",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    # ---- Derive names ----
    version_tag = f"-v{cfg['version']:02d}"
    model_id = cfg["model_id"]
    short_name = model_id.split("/")[-1]
    dataset_id = cfg["dataset_id"]
    hub_repo_id = f"{model_id}{version_tag}"
    output_directory = Path(f"outputs/train/{short_name}{version_tag}")
    output_directory.mkdir(parents=True, exist_ok=True)

    push_to_hub = cfg["hub"]["push_to_hub"]
    device = torch.device(cfg["policy"]["device"])
    load_vlm_weights = cfg["policy"].get("load_vlm_weights", True)

    train_cfg = cfg["training"]
    training_steps = train_cfg["steps"]
    batch_size = train_cfg["batch_size"]
    num_workers = train_cfg["num_workers"]
    log_freq = train_cfg["log_freq"]

    # ---- Dataset & features ----
    dataset_metadata = LeRobotDatasetMetadata(dataset_id)
    features = dataset_to_policy_features(dataset_metadata.features)
    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if key not in output_features}

    # ---- Policy ----
    policy_cfg = SmolVLAConfig(
        input_features=input_features,
        output_features=output_features,
        load_vlm_weights=load_vlm_weights,
    )
    policy = SmolVLAPolicy(policy_cfg)
    policy.train()
    policy.to(device)
    preprocessor, postprocessor = make_pre_post_processors(policy_cfg, dataset_stats=dataset_metadata.stats)

    # ---- Delta timestamps (auto-derived) ----
    obs_ts = [i / dataset_metadata.fps for i in policy_cfg.observation_delta_indices]
    act_ts = [i / dataset_metadata.fps for i in policy_cfg.action_delta_indices]
    delta_timestamps = {}
    for key in input_features:
        delta_timestamps[key] = obs_ts
    for key in output_features:
        delta_timestamps[key] = act_ts
    print(f"Auto-derived delta_timestamps: {delta_timestamps}")

    # ---- Dataset & dataloader ----
    dataset = LeRobotDataset(dataset_id, delta_timestamps=delta_timestamps)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=device.type != "cpu",
        drop_last=True,
    )

    # ---- Optimizer ----
    optimizer = torch.optim.Adam(
        policy.parameters(),
        lr=policy_cfg.optimizer_lr,
    )

    # ---- Training loop ----
    vlm_mode = "pretrained VLM" if load_vlm_weights else "from scratch"
    print(f"Training {version_tag} ({vlm_mode}) for {training_steps} steps | dataset={dataset_id} | device={device}")
    step = 0
    done = False
    while not done:
        for batch in dataloader:
            batch = preprocessor(batch)
            loss, _ = policy.forward(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if step % log_freq == 0:
                print(f"step: {step} loss: {loss.item():.3f}")
            step += 1
            if step >= training_steps:
                done = True
                break

    # ---- Save checkpoint ----
    policy.save_pretrained(output_directory)
    preprocessor.save_pretrained(output_directory)
    postprocessor.save_pretrained(output_directory)
    print(f"Checkpoint saved to {output_directory}")

    if push_to_hub:
        policy.push_to_hub(hub_repo_id)
        preprocessor.push_to_hub(hub_repo_id)
        postprocessor.push_to_hub(hub_repo_id)
        print(f"Pushed to hub: {hub_repo_id}")


if __name__ == "__main__":
    main()
