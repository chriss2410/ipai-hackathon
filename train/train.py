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

"""Train Diffusion Policy on configurable dataset using a YAML config file."""

import argparse
from pathlib import Path

import yaml
import torch

from lerobot.configs.types import FeatureType
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.feature_utils import dataset_to_policy_features
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.factory import make_pre_post_processors


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Train a Diffusion Policy")
    parser.add_argument(
        "--config",
        type=str,
        default="config/train_config.yaml",
        help="Path to YAML config file",
    )
    args = parser.parse_args()

    cfg_yaml = load_config(args.config)

    # ---- Unpack config ----
    version_tag = f"-v{cfg_yaml['version']:02d}"
    dataset_id = cfg_yaml["dataset"]["id"]
    hf_repo_id = f"{cfg_yaml['hub']['repo_id']}{version_tag}"
    push_to_hub = cfg_yaml["hub"]["push_to_hub"]
    output_directory = Path(f"{cfg_yaml['output']['directory']}{version_tag}")
    output_directory.mkdir(parents=True, exist_ok=True)

    device = torch.device(cfg_yaml["device"])

    train_cfg = cfg_yaml["training"]
    training_steps = train_cfg["steps"]
    lr = train_cfg["learning_rate"]
    batch_size = train_cfg["batch_size"]
    num_workers = train_cfg["num_workers"]
    log_freq = train_cfg["log_freq"]
    shuffle = train_cfg["shuffle"]
    pin_memory = train_cfg["pin_memory"] and device.type != "cpu"
    drop_last = train_cfg["drop_last"]

    delta_timestamps = cfg_yaml["delta_timestamps"]

    # ---- Dataset & features ----
    dataset_metadata = LeRobotDatasetMetadata(dataset_id)
    features = dataset_to_policy_features(dataset_metadata.features)
    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if key not in output_features}

    # ---- Policy ----
    policy_cfg = DiffusionConfig(input_features=input_features, output_features=output_features)
    policy = DiffusionPolicy(policy_cfg)
    policy.train()
    policy.to(device)
    preprocessor, postprocessor = make_pre_post_processors(policy_cfg, dataset_stats=dataset_metadata.stats)

    # ---- Dataset & dataloader ----
    dataset = LeRobotDataset(dataset_id, delta_timestamps=delta_timestamps)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )

    # ---- Optimizer ----
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    # ---- Training loop ----
    print(f"Training {version_tag} for {training_steps} steps | dataset={dataset_id} | device={device}")
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
        policy.push_to_hub(hf_repo_id)
        preprocessor.push_to_hub(hf_repo_id)
        postprocessor.push_to_hub(hf_repo_id)
        print(f"Pushed to hub: {hf_repo_id}")


if __name__ == "__main__":
    main()
