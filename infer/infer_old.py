"""Run SmolVLA inference on a physical robot using a YAML config file."""

import argparse
from pathlib import Path

import yaml
import torch

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.feature_utils import hw_to_dataset_features
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.utils import build_inference_frame, make_robot_action
from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Run SmolVLA inference")
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).parent / "infer_config.yaml"),
        help="Path to YAML config file",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    # ---- Device & model ----
    device = torch.device(cfg["device"])
    model_id = cfg["model"]["id"]

    model = SmolVLAPolicy.from_pretrained(model_id)
    preprocess, postprocess = make_pre_post_processors(
        model.config,
        model_id,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )

    # ---- Robot ----
    robot_cfg_yaml = cfg["robot"]
    camera_config = {
        name: OpenCVCameraConfig(
            index_or_path=cam["index_or_path"],
            width=cam["width"],
            height=cam["height"],
            fps=cam["fps"],
        )
        for name, cam in cfg["cameras"].items()
    }

    robot_cfg = SO100FollowerConfig(
        port=robot_cfg_yaml["port"],
        id=robot_cfg_yaml["id"],
        cameras=camera_config,
    )
    robot = SO100Follower(robot_cfg)
    robot.connect()

    # ---- Task & features ----
    task = cfg["task"]
    robot_type = cfg["robot_type"]

    action_features = hw_to_dataset_features(robot.action_features, "action")
    obs_features = hw_to_dataset_features(robot.observation_features, "observation")
    dataset_features = {**action_features, **obs_features}

    # ---- Inference loop ----
    infer_cfg = cfg["inference"]
    max_episodes = infer_cfg["max_episodes"]
    max_steps = infer_cfg["max_steps_per_episode"]

    print(f"Inference | model={model_id} | device={device} | episodes={max_episodes} x {max_steps} steps")

    for ep in range(max_episodes):
        for _ in range(max_steps):
            obs = robot.get_observation()
            obs_frame = build_inference_frame(
                observation=obs, ds_features=dataset_features, device=device, task=task, robot_type=robot_type
            )

            obs = preprocess(obs_frame)
            action = model.select_action(obs)
            action = postprocess(action)
            action = make_robot_action(action, dataset_features)
            robot.send_action(action)

        print(f"Episode {ep + 1}/{max_episodes} finished!")


if __name__ == "__main__":
    main()
