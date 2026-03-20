"""Gradio inference GUI for IPAI robot.

Usage:
    python infer/gui.py
    python infer/gui.py --config path/to/gui_config.yaml
"""

import argparse
import json
import threading
from pathlib import Path

import gradio as gr
import numpy as np
import yaml

from model_client import ModelClient
from robot_client import RobotClient


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
# Shared state
# ---------------------------------------------------------------------------

model_client: ModelClient | None = None
robot_client: RobotClient | None = None

stop_event = threading.Event()
inference_thread: threading.Thread | None = None

state_lock = threading.Lock()
latest_frames: dict[str, np.ndarray] = {}
running = False
step_count = 0
cfg: dict = {}


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
        return "Enter a task first.", gr.update(interactive=True), gr.update(interactive=False)

    stop_event.clear()
    inference_thread = threading.Thread(
        target=inference_loop,
        args=(task_text.strip(), cfg.get("robot_type", "")),
        daemon=True,
    )
    inference_thread.start()
    return f"Running: {task_text.strip()}", gr.update(interactive=False), gr.update(interactive=True)


def on_stop():
    global inference_thread
    stop_event.set()
    if inference_thread is not None:
        inference_thread.join(timeout=3)
        inference_thread = None
    return f"Stopped after {step_count} steps.", gr.update(interactive=True), gr.update(interactive=False)


def on_preset_select(choice: str):
    return choice if choice else ""


def on_position(position_name: str, positions: dict):
    with state_lock:
        if running:
            return f"Cannot move — inference is running. Stop first."
    pos = positions.get(position_name)
    if pos is None:
        return f"Unknown position: {position_name}"
    robot_client.go_to_position(pos)
    return f"Moved to: {position_name}"


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

    # Pad to 3 if fewer cameras
    while len(images) < 3:
        images.append(np.zeros((480, 640, 3), dtype=np.uint8))

    status = f"Running ({step_count} steps)" if is_running else "Idle"
    return images[0], images[1], images[2], status


# ---------------------------------------------------------------------------
# Build Gradio app
# ---------------------------------------------------------------------------

def build_app(commands: list[str], positions: dict) -> gr.Blocks:
    position_names = list(positions.keys())

    with gr.Blocks(title="IPAI Robot Inference", theme=gr.themes.Soft()) as app:
        gr.Markdown("# IPAI Robot Inference")

        # --- Status ---
        status_box = gr.Textbox(value="Idle", label="Status", interactive=False)

        # --- Camera feeds ---
        cam_names = robot_client.camera_names
        cam_labels = cam_names + ["camera"] * (3 - len(cam_names))
        with gr.Row():
            cam1 = gr.Image(label=cam_labels[0], height=300)
            cam2 = gr.Image(label=cam_labels[1], height=300)
            cam3 = gr.Image(label=cam_labels[2], height=300)

        # --- Task input ---
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

        # --- Start / Stop ---
        with gr.Row():
            start_btn = gr.Button("Start", variant="primary")
            stop_btn = gr.Button("Stop", variant="stop", interactive=False)

        # --- Position buttons ---
        if position_names:
            gr.Markdown("### Saved Positions")
            with gr.Row():
                for name in position_names:
                    btn = gr.Button(name.replace("_", " ").title())
                    btn.click(
                        fn=lambda n=name: on_position(n, positions),
                        outputs=[status_box],
                    )

        # --- Wiring ---
        preset_dropdown.change(fn=on_preset_select, inputs=[preset_dropdown], outputs=[task_input])

        start_btn.click(
            fn=on_start,
            inputs=[task_input],
            outputs=[status_box, start_btn, stop_btn],
        )
        stop_btn.click(
            fn=on_stop,
            outputs=[status_box, start_btn, stop_btn],
        )

        # Camera refresh timer
        timer = gr.Timer(value=0.2)
        timer.tick(fn=refresh_cameras, outputs=[cam1, cam2, cam3, status_box])

    return app


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global model_client, robot_client, cfg

    parser = argparse.ArgumentParser(description="IPAI Robot Inference GUI")
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).parent / "gui_config.yaml"),
        help="Path to GUI config YAML",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    config_dir = Path(args.config).parent

    # 1. Load model (stays warm on GPU)
    model_client = ModelClient(
        model_id=cfg["model"]["id"],
        device=cfg.get("device", "cpu"),
    )

    # 2. Connect robot
    robot_client = RobotClient(
        port=cfg["robot"]["port"],
        robot_id=cfg["robot"]["id"],
        cameras=cfg["cameras"],
    )
    robot_client.connect()

    # 3. Load presets
    presets_cfg = cfg.get("presets", {})
    commands_path = config_dir / presets_cfg.get("commands", "commands.json")
    positions_path = config_dir / presets_cfg.get("positions", "positions.json")

    commands = load_json(str(commands_path)).get("commands", [])
    positions = load_json(str(positions_path))

    # 4. Launch GUI
    app = build_app(commands, positions)
    app.launch(
        server_name="0.0.0.0",
        server_port=cfg.get("gui_port", 7860),
    )


if __name__ == "__main__":
    main()
