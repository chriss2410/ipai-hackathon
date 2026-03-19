"""Async inference client for physical robots using LeRobot's PolicyServer."""

import argparse
import threading

import yaml

from lerobot.async_inference.configs import RobotClientConfig
from lerobot.async_inference.helpers import visualize_action_queue_size
from lerobot.async_inference.robot_client import RobotClient
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.robots.so_follower import SO100FollowerConfig


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Run async inference (robot client)")
    parser.add_argument(
        "--config",
        type=str,
        default="config/infer_config.yaml",
        help="Path to YAML config file",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    # ---- Cameras ----
    camera_config = {
        name: OpenCVCameraConfig(
            index_or_path=cam["index_or_path"],
            width=cam["width"],
            height=cam["height"],
            fps=cam["fps"],
        )
        for name, cam in cfg["cameras"].items()
    }

    # ---- Robot ----
    robot_cfg = SO100FollowerConfig(
        port=cfg["robot"]["port"],
        id=cfg["robot"]["id"],
        cameras=camera_config,
    )

    # ---- Client config ----
    policy_cfg = cfg["policy"]
    async_cfg = cfg["async"]
    server_cfg = cfg["server"]

    client_cfg = RobotClientConfig(
        robot=robot_cfg,
        server_address=f"{server_cfg['host']}:{server_cfg['port']}",
        policy_type=policy_cfg["type"],
        pretrained_name_or_path=policy_cfg["pretrained_name_or_path"],
        policy_device=policy_cfg["policy_device"],
        client_device=policy_cfg["client_device"],
        actions_per_chunk=async_cfg["actions_per_chunk"],
        chunk_size_threshold=async_cfg["chunk_size_threshold"],
        aggregate_fn_name=async_cfg["aggregate_fn_name"],
        fps=async_cfg["fps"],
        debug_visualize_queue_size=async_cfg["debug_visualize_queue_size"],
        task=cfg.get("task", ""),
    )

    # ---- Start client ----
    client = RobotClient(client_cfg)
    task = cfg.get("task", "")

    print(
        f"Async inference | policy={policy_cfg['type']} "
        f"| model={policy_cfg['pretrained_name_or_path']} "
        f"| server={server_cfg['host']}:{server_cfg['port']}"
    )

    if client.start():
        action_receiver_thread = threading.Thread(
            target=client.receive_actions, daemon=True
        )
        action_receiver_thread.start()

        try:
            client.control_loop(task)
        except KeyboardInterrupt:
            client.stop()
            action_receiver_thread.join()
            if async_cfg["debug_visualize_queue_size"]:
                visualize_action_queue_size(client.action_queue_size)
    else:
        print("Failed to connect to policy server.")


if __name__ == "__main__":
    main()
