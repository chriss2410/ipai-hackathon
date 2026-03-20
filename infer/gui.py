"""Gradio inference GUI for IPAI robot.

Usage:
    python infer/gui.py
    python infer/gui.py --config path/to/gui_config.yaml
"""

import argparse
import json
import threading
import time
from pathlib import Path

import gradio as gr
import numpy as np
import yaml

from model_client import ModelClient
from robot_client import RobotClient
from sequence_runner import SequenceRunner
from sequence_ui import build_sequence_tab

SCRIPT_DIR = Path(__file__).parent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Initialization (runs once at module load — works for both
# `python gui.py` and `gradio gui.py` hot-reload mode)
# ---------------------------------------------------------------------------

_initialized = False
cfg: dict = {}
model_client: ModelClient | None = None
robot_client: RobotClient | None = None
_commands: list[str] = []
_positions: dict = {}


def _init():
    """Load config, model, robot, and presets. Runs only once even under hot-reload."""
    global _initialized, cfg, model_client, robot_client, _commands, _positions
    if _initialized:
        return
    _initialized = True

    parser = argparse.ArgumentParser(description="IPAI Robot Inference GUI")
    parser.add_argument(
        "--config",
        type=str,
        default=str(SCRIPT_DIR / "gui_config.yaml"),
        help="Path to GUI config YAML",
    )
    args, _ = parser.parse_known_args()

    cfg = load_config(args.config)
    config_dir = Path(args.config).parent

    # 1. Load model (stays warm on GPU)
    lora_cfg = cfg.get("lora", {})
    model_client = ModelClient(
        model_id=cfg["model"]["id"],
        device=cfg.get("device", "cpu"),
        lora_adapter_id=lora_cfg.get("adapter_id") if lora_cfg.get("enabled") else None,
    )

    # 2. Connect robot
    teleop_cfg = cfg.get("teleop", {})
    robot_client = RobotClient(
        port=cfg["robot"]["port"],
        robot_id=cfg["robot"]["id"],
        cameras=cfg["cameras"],
        teleop_port=teleop_cfg.get("port"),
        teleop_id=teleop_cfg.get("id", "leader"),
    )
    robot_client.connect()

    # 3. Load presets
    presets_cfg = cfg.get("presets", {})
    commands_path = config_dir / presets_cfg.get("commands", "commands.json")
    positions_path = config_dir / presets_cfg.get("positions", "positions.json")

    _commands = load_json(str(commands_path)).get("commands", [])
    _positions = load_json(str(positions_path))


_init()


# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------

stop_event = threading.Event()
inference_thread: threading.Thread | None = None
teleop_thread: threading.Thread | None = None

state_lock = threading.Lock()
latest_frames: dict[str, np.ndarray] = {}
running = False
step_count = 0


# ---------------------------------------------------------------------------
# Inference loop (runs in background thread)
# ---------------------------------------------------------------------------

def inference_loop(task: str, robot_type: str):
    global running, step_count, latest_frames

    model_client.reset()
    step_count = 0

    with state_lock:
        running = True

    try:
        while not stop_event.is_set():
            obs = robot_client.get_observation()

            # Cache camera frames for the UI
            frames = {
                name: obs[name]
                for name in robot_client.camera_names
                if name in obs
            }
            with state_lock:
                latest_frames = frames

            action = model_client.predict(
                observation=obs,
                task=task,
                robot_type=robot_type,
                dataset_features=robot_client.dataset_features,
            )
            robot_client.send_action(action)

            step_count += 1
    finally:
        with state_lock:
            running = False


# ---------------------------------------------------------------------------
# Gradio callbacks
# ---------------------------------------------------------------------------

def on_start(task_text: str):
    global inference_thread
    if not task_text.strip():
        return "Enter a task first.", gr.update(interactive=True), gr.update(interactive=False), gr.update(interactive=True)

    stop_event.clear()
    inference_thread = threading.Thread(
        target=inference_loop,
        args=(task_text.strip(), cfg.get("robot_type", "")),
        daemon=True,
    )
    inference_thread.start()
    return f"Running: {task_text.strip()}", gr.update(interactive=False), gr.update(interactive=True), gr.update(interactive=False)


def on_stop():
    global inference_thread
    stop_event.set()
    if inference_thread is not None:
        inference_thread.join(timeout=3)
        inference_thread = None
    return f"Stopped after {step_count} steps.", gr.update(interactive=True), gr.update(interactive=False), gr.update(interactive=True)


# ---------------------------------------------------------------------------
# Teleop loop (runs in background thread)
# ---------------------------------------------------------------------------

def teleop_loop():
    global running, latest_frames

    with state_lock:
        running = True

    try:
        while not stop_event.is_set():
            robot_client.teleop_step()

            # Grab frames from follower observation (same serial transaction)
            try:
                obs = robot_client.get_observation()
                frames = {
                    name: obs[name]
                    for name in robot_client.camera_names
                    if name in obs
                }
                with state_lock:
                    latest_frames = frames
            except Exception:
                pass

            # Small delay to avoid hammering the serial bus
            time.sleep(0.01)
    finally:
        with state_lock:
            running = False


def on_teleop_start():
    global teleop_thread
    with state_lock:
        if running:
            return "Cannot start teleop — inference or teleop already running.", gr.update(), gr.update(), gr.update()

    stop_event.clear()
    teleop_thread = threading.Thread(target=teleop_loop, daemon=True)
    teleop_thread.start()
    return "Teleop running — move the leader arm.", gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=True)


def on_teleop_stop():
    global teleop_thread
    stop_event.set()
    if teleop_thread is not None:
        teleop_thread.join(timeout=3)
        teleop_thread = None
    return "Teleop stopped.", gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=False)


def on_preset_select(choice: str):
    return choice if choice else ""


def on_position(position_name: str):
    with state_lock:
        if running:
            return "Cannot move — inference is running. Stop first."
    pos = _positions.get(position_name)
    if pos is None:
        return f"Unknown position: {position_name}"
    robot_client.go_to_position(pos)
    return f"Moved to: {position_name}"


def save_position(label: str):
    """Save the robot's current joint positions under *label* and persist to disk."""
    global _positions
    label = label.strip()
    if not label:
        return "Enter a label first.", gr.update(choices=list(_positions.keys()))
    with state_lock:
        if running:
            return "Cannot save — inference is running. Stop first.", gr.update(choices=list(_positions.keys()))
    joints = robot_client.get_joint_positions()
    _positions[label] = joints
    # Persist
    positions_path = Path(cfg.get("presets", {})\
        .get("positions", str(SCRIPT_DIR / "positions.json")))
    if not positions_path.is_absolute():
        positions_path = SCRIPT_DIR / positions_path
    with open(positions_path, "w") as f:
        json.dump(_positions, f, indent=2)
    return f"Saved position: {label}", gr.update(choices=list(_positions.keys()))


def on_go_to_position(position_name: str):
    """Go to a position selected from the dropdown."""
    if not position_name:
        return "Select a position first."
    return on_position(position_name)


def delete_position(position_name: str):
    """Delete a saved position."""
    global _positions
    if not position_name:
        return "Select a position first.", gr.update(choices=list(_positions.keys()))
    _positions.pop(position_name, None)
    positions_path = Path(cfg.get("presets", {})\
        .get("positions", str(SCRIPT_DIR / "positions.json")))
    if not positions_path.is_absolute():
        positions_path = SCRIPT_DIR / positions_path
    with open(positions_path, "w") as f:
        json.dump(_positions, f, indent=2)
    return f"Deleted: {position_name}", gr.update(choices=list(_positions.keys()))


def refresh_cameras():
    """Called by gr.Timer to update camera feeds and step counter."""
    with state_lock:
        is_running = running
        cached = dict(latest_frames)

    if is_running:
        frames = cached
    else:
        try:
            frames = robot_client.get_camera_frames()
        except Exception:
            frames = {}

    camera_names = robot_client.camera_names
    images = []
    for name in camera_names:
        img = frames.get(name)
        if img is not None:
            images.append(img)
        else:
            placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
            images.append(placeholder)

    while len(images) < 3:
        images.append(np.zeros((480, 640, 3), dtype=np.uint8))

    status = f"Running ({step_count} steps)" if is_running else "Idle"
    return images[0], images[1], images[2], status


# ---------------------------------------------------------------------------
# Build Gradio app
# ---------------------------------------------------------------------------

def build_app(commands: list[str], positions: dict) -> gr.Blocks:
    position_names = list(positions.keys())
    cam_names = robot_client.camera_names
    cam_labels = cam_names + ["camera"] * (3 - len(cam_names))

    lora_cfg = cfg.get("lora", {})
    lora_enabled = lora_cfg.get("enabled", False)
    if lora_enabled:
        adapter_id = lora_cfg.get("adapter_id", "")
        base_id = cfg["model"]["id"]
        model_info = f"**Model:** {base_id} + LoRA adapter `{adapter_id}`"
    else:
        model_info = f"**Model:** {cfg['model']['id']}"

    seq_runner = SequenceRunner(robot_client, model_client, _positions, cfg)

    with gr.Blocks(title="IPAI Robot Inference") as app:
        gr.Markdown("# IPAI Robot Inference")
        gr.Markdown(model_info)

        status_box = gr.Textbox(value="Idle", label="Status", interactive=False)

        with gr.Row():
            cam1 = gr.Image(label=cam_labels[0], height=300)
            cam2 = gr.Image(label=cam_labels[1], height=300)
            cam3 = gr.Image(label=cam_labels[2], height=300)

        with gr.Tabs():
            with gr.Tab("Control"):
                with gr.Row():
                    task_input = gr.Textbox(
                        value=cfg.get("task", ""),
                        label="Task Command",
                        placeholder="e.g. pick blue car",
                        scale=3,
                    )
                    preset_dropdown = gr.Dropdown(
                        choices=commands,
                        label="Presets",
                        scale=1,
                    )

                with gr.Row():
                    start_btn = gr.Button("Start", variant="primary")
                    stop_btn = gr.Button("Stop", variant="stop", interactive=False)

                if robot_client.has_teleop:
                    gr.Markdown("### Teleop")
                    with gr.Row():
                        teleop_start_btn = gr.Button("Start Teleop", variant="primary")
                        teleop_stop_btn = gr.Button("Stop Teleop", variant="stop", interactive=False)
                else:
                    teleop_start_btn = None
                    teleop_stop_btn = None

                gr.Markdown("### Positions")
                with gr.Row():
                    position_dropdown = gr.Dropdown(
                        choices=position_names,
                        label="Saved Positions",
                        scale=2,
                    )
                    go_btn = gr.Button("Go To", scale=1)
                    delete_btn = gr.Button("Delete", variant="stop", scale=1)

                with gr.Row():
                    pos_label_input = gr.Textbox(
                        label="Position Label",
                        placeholder="e.g. home",
                        scale=2,
                    )
                    save_btn = gr.Button("Save Current Position", variant="primary", scale=1)

            with gr.Tab("Sequences"):
                build_sequence_tab(
                    runner=seq_runner,
                    commands=commands,
                    get_positions=lambda: _positions,
                )

        # --- Event wiring ---
        teleop_btn = teleop_start_btn if teleop_start_btn else gr.Button(visible=False)

        preset_dropdown.change(fn=on_preset_select, inputs=[preset_dropdown], outputs=[task_input])

        start_btn.click(
            fn=on_start,
            inputs=[task_input],
            outputs=[status_box, start_btn, stop_btn, teleop_btn],
        )
        stop_btn.click(
            fn=on_stop,
            outputs=[status_box, start_btn, stop_btn, teleop_btn],
        )

        if teleop_start_btn and teleop_stop_btn:
            teleop_start_btn.click(
                fn=on_teleop_start,
                outputs=[status_box, start_btn, teleop_start_btn, teleop_stop_btn],
            )
            teleop_stop_btn.click(
                fn=on_teleop_stop,
                outputs=[status_box, start_btn, teleop_start_btn, teleop_stop_btn],
            )

        go_btn.click(
            fn=on_go_to_position,
            inputs=[position_dropdown],
            outputs=[status_box],
        )
        save_btn.click(
            fn=save_position,
            inputs=[pos_label_input],
            outputs=[status_box, position_dropdown],
        )
        delete_btn.click(
            fn=delete_position,
            inputs=[position_dropdown],
            outputs=[status_box, position_dropdown],
        )

        timer = gr.Timer(value=0.2)
        timer.tick(fn=refresh_cameras, outputs=[cam1, cam2, cam3, status_box])

    return app


# ---------------------------------------------------------------------------
# Module-level `demo` for `gradio gui.py` hot-reload
# ---------------------------------------------------------------------------

demo = build_app(_commands, _positions)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=cfg.get("gui_port", 7860),
    )
