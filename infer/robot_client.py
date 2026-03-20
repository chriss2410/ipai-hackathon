"""Robot client — wraps SO100 connection, observation, actions, and camera feeds."""

import logging

import numpy as np

logger = logging.getLogger(__name__)

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.feature_utils import hw_to_dataset_features
from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig
from lerobot.teleoperators.so_leader import SO100Leader, SO100LeaderConfig


class RobotClient:
    """Manages SO100 robot hardware: connection, observation, actions, cameras."""

    def __init__(self, port: str, robot_id: str, cameras: dict[str, dict],
                 teleop_port: str | None = None, teleop_id: str = "leader"):
        camera_config = {
            name: OpenCVCameraConfig(
                index_or_path=cam["index_or_path"],
                width=cam["width"],
                height=cam["height"],
                fps=cam["fps"],
            )
            for name, cam in cameras.items()
        }

        robot_cfg = SO100FollowerConfig(
            port=port,
            id=robot_id,
            cameras=camera_config,
        )

        self._robot = SO100Follower(robot_cfg)
        self._camera_names = list(cameras.keys())
        self._dataset_features: dict | None = None

        # Leader arm for teleoperation
        self._teleop: SO100Leader | None = None
        if teleop_port:
            teleop_cfg = SO100LeaderConfig(port=teleop_port, id=teleop_id)
            self._teleop = SO100Leader(teleop_cfg)

    def connect(self):
        """Connect to the robot hardware and derive dataset features."""
        self._robot.connect()
        action_ft = hw_to_dataset_features(self._robot.action_features, "action")
        obs_ft = hw_to_dataset_features(self._robot.observation_features, "observation")
        self._dataset_features = {**action_ft, **obs_ft}
        print(f"Robot connected. Cameras: {self._camera_names}")
        if self._teleop:
            self._teleop.connect()
            print(f"Teleop leader connected on {self._teleop.config.port}")

    def disconnect(self):
        if self._teleop and self._teleop.is_connected:
            self._teleop.disconnect()
            print("Teleop leader disconnected.")
        self._robot.disconnect()
        print("Robot disconnected.")

    @property
    def is_connected(self) -> bool:
        return self._robot.is_connected

    @property
    def dataset_features(self) -> dict:
        if self._dataset_features is None:
            raise RuntimeError("Call connect() before accessing dataset_features.")
        return self._dataset_features

    @property
    def camera_names(self) -> list[str]:
        return self._camera_names

    def get_observation(self) -> dict:
        """Get full observation: motor positions + camera images."""
        return self._robot.get_observation()

    def get_camera_frames(self) -> dict[str, np.ndarray]:
        """Get camera images only (H, W, 3 RGB numpy arrays)."""
        obs = self._robot.get_observation()
        return {name: obs[name] for name in self._camera_names if name in obs}

    def send_action(self, action: dict) -> dict:
        """Send absolute joint positions to the robot."""
        return self._robot.send_action(action)

    def get_joint_positions(self) -> dict[str, float]:
        """Read current joint positions from the robot (excludes camera data)."""
        obs = self._robot.get_observation()
        return {k: float(v) for k, v in obs.items() if k not in self._camera_names}

    def go_to_position(self, position: dict[str, float]):
        """Drive robot to a named position (absolute joint values)."""
        self._robot.send_action(position)

    @property
    def has_teleop(self) -> bool:
        return self._teleop is not None

    def teleop_step(self):
        """Read leader arm position and mirror it on the follower. Call in a loop."""
        if self._teleop is None:
            raise RuntimeError("No teleop leader configured.")
        try:
            action = self._teleop.get_action()
            self._robot.send_action(action)
        except Exception as e:
            logger.warning("Teleop step failed (retrying next cycle): %s", e)
