"""Test async inference with a mock robot (no hardware required).

Replays real dataset frames through the async inference pipeline so you
can verify the full end-to-end without physical hardware.

Usage (two terminals from repo root):
    Terminal 1: python infer/policy_server.py
    Terminal 2: python infer/test_inference.py
    Terminal 2: python infer/test_inference.py --dataset lerobot/svla_so101_pickplace --episode 0
"""

import argparse
import threading
import time
from pathlib import Path

import numpy as np
import yaml

from lerobot.async_inference.configs import RobotClientConfig
from lerobot.async_inference.robot_client import RobotClient
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.robots.so_follower import SO100FollowerConfig


# ---------------------------------------------------------------------------
# Mock robot that replays dataset frames
# ---------------------------------------------------------------------------

MOTOR_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


class MockSO100:
    """Drop-in replacement for SO100Follower that replays dataset frames."""

    def __init__(self, config: SO100FollowerConfig, dataset: LeRobotDataset, episode: int = 0,
                 camera_map: dict[str, str] | None = None):
        self.config = config
        self._is_connected = False
        self._camera_shapes = {
            name: (cam.height, cam.width, 3) for name, cam in config.cameras.items()
        }
        self._camera_names = list(config.cameras.keys())

        # Load episode frames from dataset
        self._frames = dataset.hf_dataset.filter(lambda ex: ex["episode_index"] == episode)
        self._num_frames = len(self._frames)
        self._step = 0

        # Build camera mapping: robot camera name → dataset key
        dataset_camera_keys = list(dataset.meta.camera_keys)
        if camera_map:
            self._camera_map = camera_map
        else:
            # Fallback: positional mapping
            self._camera_map = dict(zip(self._camera_names, dataset_camera_keys))

        print(f"[MockSO100] Loaded episode {episode}: {self._num_frames} frames")
        print(f"[MockSO100] Dataset cameras: {dataset_camera_keys}")
        print(f"[MockSO100] Camera mapping: { {k: v for k, v in self._camera_map.items()} }")

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @property
    def observation_features(self) -> dict:
        features: dict = {f"{m}.pos": float for m in MOTOR_NAMES}
        for name, (h, w, _c) in self._camera_shapes.items():
            features[name] = (h, w, 3)
        return features

    @property
    def action_features(self) -> dict:
        return {f"{m}.pos": float for m in MOTOR_NAMES}

    def connect(self, calibrate: bool = True) -> None:
        self._is_connected = True
        print("[MockSO100] Connected (mock)")

    def disconnect(self) -> None:
        self._is_connected = False
        print("[MockSO100] Disconnected")

    def get_observation(self) -> dict:
        frame_idx = self._step % self._num_frames
        frame = self._frames[frame_idx]

        obs: dict = {}

        # Motor positions from dataset observation.state
        state = frame["observation.state"]
        for i, m in enumerate(MOTOR_NAMES):
            obs[f"{m}.pos"] = float(state[i]) if i < len(state) else 0.0

        # Camera images from dataset via camera_map
        for cam_name in self._camera_names:
            ds_key = self._camera_map[cam_name]
            img = frame[ds_key]
            # Dataset may return PIL Image or numpy array
            if hasattr(img, "numpy"):
                img = img.numpy()
            img = np.array(img, dtype=np.uint8)
            # Dataset images may be (C, H, W) — convert to (H, W, C) for robot API
            if img.ndim == 3 and img.shape[0] in (1, 3):
                img = np.transpose(img, (1, 2, 0))
            obs[cam_name] = img

        self._step += 1
        return obs

    def send_action(self, action: dict) -> dict:
        print(f"[MockSO100] Step {self._step} action received:")
        for key, val in action.items():
            print(f"  {key:>20s}: {val:+8.3f}")
        return action


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Test async inference with mock robot")
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).parent / "infer_config.yaml"),
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=100,
        help="Number of control loop steps before stopping",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="HuggingFace dataset repo ID (overrides config test.dataset)",
    )
    parser.add_argument(
        "--episode",
        type=int,
        default=None,
        help="Episode index to replay (overrides config test.episode)",
    )
    parser.add_argument(
        "--show-cameras",
        action="store_true",
        help="Load the dataset and print available camera keys, then exit",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    # ---- Build camera config (needed for observation shapes) ----
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
        port="/dev/null",  # unused by mock
        id="mock_follower",
        cameras=camera_config,
    )

    # ---- Build RobotClientConfig ----
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
        debug_visualize_queue_size=async_cfg.get("debug_visualize_queue_size", False),
        task=cfg.get("task", ""),
    )

    # ---- Load dataset for replay ----
    test_cfg = cfg.get("test", {})
    dataset_id = args.dataset or test_cfg.get("dataset", "lerobot/svla_so101_pickplace")
    episode = args.episode if args.episode is not None else test_cfg.get("episode", 0)
    camera_map = test_cfg.get("camera_map")

    print(f"Loading dataset {dataset_id} (episode {episode})...")
    dataset = LeRobotDataset(dataset_id)

    if args.show_cameras:
        print(f"\nDataset camera keys: {list(dataset.meta.camera_keys)}")
        print(f"Robot camera names:  {list(cfg['cameras'].keys())}")
        print("\nSet test.camera_map in your config to map robot → dataset cameras.")
        return

    # ---- Patch: inject mock robot instead of real hardware ----
    client = object.__new__(RobotClient)

    # Manually run the parts of RobotClient.__init__ with our mock robot
    mock_robot = MockSO100(robot_cfg, dataset, episode=episode, camera_map=camera_map)
    mock_robot.connect()

    # Monkey-patch the robot onto the client, then call the rest of init
    client.config = client_cfg
    client.robot = mock_robot

    # Re-run the rest of __init__ that sets up gRPC and threading
    import grpc
    from lerobot.async_inference.helpers import FPSTracker, RemotePolicyConfig, map_robot_keys_to_lerobot_features

    client.lerobot_features = map_robot_keys_to_lerobot_features(mock_robot)
    client.server_address = client_cfg.server_address

    client.policy_config = RemotePolicyConfig(
        client_cfg.policy_type,
        client_cfg.pretrained_name_or_path,
        client.lerobot_features,
        client_cfg.actions_per_chunk,
        client_cfg.policy_device,
    )

    client.channel = grpc.insecure_channel(
        client_cfg.server_address,
        options=[("grpc.max_receive_message_length", -1), ("grpc.max_send_message_length", -1)],
    )

    from lerobot.transport import services_pb2_grpc

    client.stub = services_pb2_grpc.AsyncInferenceStub(client.channel)
    client.shutdown_event = threading.Event()
    client.start_barrier = threading.Barrier(2)

    from queue import Queue

    client.action_queue = Queue()
    client.action_queue_lock = threading.Lock()
    client.latest_action_lock = threading.Lock()
    client.latest_action = -1
    client.action_chunk_size = -1
    client._chunk_size_threshold = client_cfg.chunk_size_threshold
    client.must_go = threading.Event()
    client.fps_tracker = FPSTracker(client_cfg.fps)
    client.action_queue_size = []
    client.aggregate_fn = client_cfg.aggregate_fn

    # ---- Start ----
    task = cfg.get("task", "")
    print(
        f"\n=== Mock Inference Test ===\n"
        f"  Policy:  {policy_cfg['type']}\n"
        f"  Model:   {policy_cfg['pretrained_name_or_path']}\n"
        f"  Dataset: {dataset_id} (episode {episode}, {mock_robot._num_frames} frames)\n"
        f"  Server:  {server_cfg['host']}:{server_cfg['port']}\n"
        f"  Cameras: {list(cfg['cameras'].keys())}\n"
        f"  Steps:   {args.steps}\n"
    )

    if client.start():
        action_receiver_thread = threading.Thread(
            target=client.receive_actions, daemon=True
        )
        action_receiver_thread.start()

        try:
            # Run control loop with a step limit
            step = 0
            client.start_barrier.wait()
            while step < args.steps and not client.shutdown_event.is_set():
                loop_start = time.perf_counter()
                if client.actions_available():
                    client.control_loop_action(verbose=True)
                if client._ready_to_send_observation():
                    client.control_loop_observation(task, verbose=True)
                elapsed = time.perf_counter() - loop_start
                time.sleep(max(0, client.config.environment_dt - elapsed))
                step += 1

            print(f"\n=== Test complete ({step} steps) ===")
        except KeyboardInterrupt:
            print("\nInterrupted by user.")
        finally:
            client.stop()
            action_receiver_thread.join(timeout=3)
    else:
        print("Failed to connect to policy server. Is it running?")


if __name__ == "__main__":
    main()
