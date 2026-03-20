"""Sequence runner — executes a list of position / VLA-command steps."""

import json
import threading
import time
from pathlib import Path

SEQUENCES_DIR = Path(__file__).parent / "sequences"


def list_saved_sequences() -> list[str]:
    """Return names (without .json) of all saved sequences."""
    if not SEQUENCES_DIR.exists():
        return []
    return sorted(p.stem for p in SEQUENCES_DIR.glob("*.json"))


def load_sequence(name: str) -> dict:
    path = SEQUENCES_DIR / f"{name}.json"
    with open(path) as f:
        return json.load(f)


def save_sequence(name: str, data: dict):
    SEQUENCES_DIR.mkdir(exist_ok=True)
    path = SEQUENCES_DIR / f"{name}.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def delete_sequence(name: str):
    path = SEQUENCES_DIR / f"{name}.json"
    path.unlink(missing_ok=True)


class SequenceRunner:
    """Runs a sequence of steps in a background thread.

    Each step is either:
      {"type": "position", "value": "<position_name>"}
      {"type": "command",  "value": "<task string>", "duration": <seconds>}
    """

    def __init__(self, robot_client, model_client, positions: dict, cfg: dict):
        self.robot_client = robot_client
        self.model_client = model_client
        self.positions = positions
        self.cfg = cfg

        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._status = "Idle"
        self._current_step = -1
        self._total_steps = 0

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def status(self) -> str:
        return self._status

    @property
    def current_step(self) -> int:
        return self._current_step

    @property
    def total_steps(self) -> int:
        return self._total_steps

    def run(self, steps: list[dict], loops: int = 1):
        if self.is_running:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run_loop, args=(steps, loops), daemon=True,
        )
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
        self._status = "Stopped"
        self._current_step = -1

    def _run_loop(self, steps: list[dict], loops: int):
        self._total_steps = len(steps)
        try:
            for loop_i in range(loops):
                loop_label = f"[loop {loop_i + 1}/{loops}] " if loops > 1 else ""
                for i, step in enumerate(steps):
                    if self._stop.is_set():
                        return
                    self._current_step = i
                    stype = step["type"]
                    value = step["value"]

                    if stype == "position":
                        self._status = f"{loop_label}Step {i + 1}/{len(steps)}: go to '{value}'"
                        self._exec_position(value)
                    elif stype == "command":
                        duration = step.get("duration", 10)
                        self._status = f"{loop_label}Step {i + 1}/{len(steps)}: '{value}' ({duration}s)"
                        self._exec_command(value, duration)

            self._status = "Sequence complete"
        except Exception as e:
            self._status = f"Error: {e}"
        finally:
            self._current_step = -1

    def _exec_position(self, name: str):
        pos = self.positions.get(name)
        if pos is None:
            self._status = f"Unknown position: {name}"
            return
        self.robot_client.go_to_position(pos)
        # Brief settle time
        self._wait(0.5)

    def _exec_command(self, task: str, duration: float):
        self.model_client.reset()
        robot_type = self.cfg.get("robot_type", "")
        deadline = time.monotonic() + duration

        while not self._stop.is_set() and time.monotonic() < deadline:
            obs = self.robot_client.get_observation()
            action = self.model_client.predict(
                observation=obs,
                task=task,
                robot_type=robot_type,
                dataset_features=self.robot_client.dataset_features,
            )
            self.robot_client.send_action(action)

    def _wait(self, seconds: float):
        end = time.monotonic() + seconds
        while not self._stop.is_set() and time.monotonic() < end:
            time.sleep(0.05)
