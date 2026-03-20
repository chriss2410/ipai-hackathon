"""Sequence Builder — Gradio tab for composing and running step sequences."""

from __future__ import annotations

import json

import gradio as gr

from sequence_runner import (
    SequenceRunner,
    delete_sequence,
    list_saved_sequences,
    load_sequence,
    save_sequence,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _steps_to_display(steps: list[dict]) -> str:
    """Format steps list as a readable numbered list."""
    if not steps:
        return "_No steps yet._"
    lines = []
    for i, s in enumerate(steps, 1):
        if s["type"] == "position":
            lines.append(f"{i}. **Position** → {s['value']}")
        else:
            lines.append(f"{i}. **Command** → \"{s['value']}\" ({s.get('duration', 10)}s)")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Build tab
# ---------------------------------------------------------------------------

def build_sequence_tab(
    runner: SequenceRunner,
    commands: list[str],
    get_positions: callable,
) -> None:
    """Build the Sequence Builder tab contents. Must be called inside a gr.Blocks/Tab context."""

    # Local mutable state for the current (in-progress) sequence
    seq_state = gr.State(value=[])  # list[dict]

    gr.Markdown("# Sequence Builder")
    gr.Markdown("Compose a sequence of **position** moves and **VLA command** steps, then run them in order.")

    seq_status = gr.Textbox(value="Idle", label="Sequence Status", interactive=False)

    # --- Current sequence display ---
    gr.Markdown("### Current Sequence")
    steps_display = gr.Markdown(value="_No steps yet._")

    # --- Add step controls ---
    gr.Markdown("### Add Step")
    with gr.Row():
        step_type = gr.Radio(
            choices=["position", "command"],
            value="position",
            label="Step Type",
            scale=1,
        )
        position_choice = gr.Dropdown(
            choices=list(get_positions().keys()),
            label="Position",
            scale=2,
            visible=True,
        )
        command_choice = gr.Dropdown(
            choices=commands,
            label="Preset Command",
            scale=2,
            visible=False,
        )

    with gr.Row():
        custom_command = gr.Textbox(
            label="Custom Command (overrides preset)",
            placeholder="e.g. pick up blue car",
            scale=2,
            visible=False,
        )
        duration_input = gr.Number(
            value=10,
            label="Duration (seconds)",
            minimum=1,
            maximum=300,
            scale=1,
            visible=False,
        )

    add_btn = gr.Button("Add Step", variant="primary")

    # --- Edit steps ---
    gr.Markdown("### Edit")
    with gr.Row():
        step_index = gr.Number(value=1, label="Step #", minimum=1, precision=0, scale=1)
        remove_btn = gr.Button("Remove Step", variant="stop", scale=1)
        move_up_btn = gr.Button("Move Up", scale=1)
        move_down_btn = gr.Button("Move Down", scale=1)
        clear_btn = gr.Button("Clear All", variant="stop", scale=1)

    # --- Run controls ---
    gr.Markdown("### Run")
    with gr.Row():
        loop_count = gr.Number(value=1, label="Loops", minimum=1, maximum=100, precision=0, scale=1)
        run_btn = gr.Button("Run Sequence", variant="primary", scale=2)
        stop_btn = gr.Button("Stop", variant="stop", interactive=False, scale=1)

    # --- Save / Load ---
    gr.Markdown("### Save / Load")
    with gr.Row():
        seq_name_input = gr.Textbox(label="Sequence Name", placeholder="e.g. pick_and_place_demo", scale=2)
        save_btn = gr.Button("Save", variant="primary", scale=1)

    with gr.Row():
        saved_dropdown = gr.Dropdown(
            choices=list_saved_sequences(),
            label="Saved Sequences",
            scale=2,
        )
        load_btn = gr.Button("Load", scale=1)
        del_btn = gr.Button("Delete", variant="stop", scale=1)

    # -----------------------------------------------------------------------
    # Callbacks
    # -----------------------------------------------------------------------

    def toggle_step_type(stype):
        is_pos = stype == "position"
        return (
            gr.update(visible=is_pos, choices=list(get_positions().keys())),   # position_choice
            gr.update(visible=not is_pos),  # command_choice
            gr.update(visible=not is_pos),  # custom_command
            gr.update(visible=not is_pos),  # duration_input
        )

    def add_step(steps, stype, pos_name, preset_cmd, custom_cmd, duration):
        if stype == "position":
            if not pos_name:
                return steps, _steps_to_display(steps), "Select a position first."
            steps = steps + [{"type": "position", "value": pos_name}]
        else:
            cmd = custom_cmd.strip() if custom_cmd.strip() else preset_cmd
            if not cmd:
                return steps, _steps_to_display(steps), "Enter or select a command."
            steps = steps + [{"type": "command", "value": cmd, "duration": int(duration)}]
        return steps, _steps_to_display(steps), f"Added {stype} step."

    def remove_step(steps, idx):
        idx = int(idx) - 1
        if 0 <= idx < len(steps):
            removed = steps.pop(idx)
            steps = list(steps)
            return steps, _steps_to_display(steps), f"Removed step {idx + 1}."
        return steps, _steps_to_display(steps), "Invalid step number."

    def move_step_up(steps, idx):
        idx = int(idx) - 1
        if 1 <= idx < len(steps):
            steps = list(steps)
            steps[idx - 1], steps[idx] = steps[idx], steps[idx - 1]
            return steps, _steps_to_display(steps), f"Moved step {idx + 1} up."
        return steps, _steps_to_display(steps), "Cannot move."

    def move_step_down(steps, idx):
        idx = int(idx) - 1
        if 0 <= idx < len(steps) - 1:
            steps = list(steps)
            steps[idx], steps[idx + 1] = steps[idx + 1], steps[idx]
            return steps, _steps_to_display(steps), f"Moved step {idx + 1} down."
        return steps, _steps_to_display(steps), "Cannot move."

    def clear_steps():
        return [], _steps_to_display([]), "Cleared."

    def on_run(steps, loops):
        if not steps:
            return "No steps to run.", gr.update(interactive=True), gr.update(interactive=False)
        if runner.is_running:
            return "Already running.", gr.update(interactive=False), gr.update(interactive=True)
        runner.run(steps, int(loops))
        return "Sequence started...", gr.update(interactive=False), gr.update(interactive=True)

    def on_stop():
        runner.stop()
        return "Stopped.", gr.update(interactive=True), gr.update(interactive=False)

    def on_save(steps, name):
        name = name.strip()
        if not name:
            return "Enter a name.", gr.update()
        if not steps:
            return "No steps to save.", gr.update()
        save_sequence(name, {"name": name, "steps": steps})
        return f"Saved: {name}", gr.update(choices=list_saved_sequences())

    def on_load(name):
        if not name:
            return gr.update(), _steps_to_display([]), "Select a sequence."
        data = load_sequence(name)
        steps = data.get("steps", [])
        return steps, _steps_to_display(steps), f"Loaded: {name}"

    def on_delete(name):
        if not name:
            return "Select a sequence.", gr.update()
        delete_sequence(name)
        return f"Deleted: {name}", gr.update(choices=list_saved_sequences())

    def refresh_seq_status():
        if runner.is_running:
            return runner.status, gr.update(interactive=False), gr.update(interactive=True)
        return runner.status, gr.update(interactive=True), gr.update(interactive=False)

    # -----------------------------------------------------------------------
    # Event wiring
    # -----------------------------------------------------------------------

    step_type.change(
        fn=toggle_step_type,
        inputs=[step_type],
        outputs=[position_choice, command_choice, custom_command, duration_input],
    )

    add_btn.click(
        fn=add_step,
        inputs=[seq_state, step_type, position_choice, command_choice, custom_command, duration_input],
        outputs=[seq_state, steps_display, seq_status],
    )

    remove_btn.click(fn=remove_step, inputs=[seq_state, step_index], outputs=[seq_state, steps_display, seq_status])
    move_up_btn.click(fn=move_step_up, inputs=[seq_state, step_index], outputs=[seq_state, steps_display, seq_status])
    move_down_btn.click(fn=move_step_down, inputs=[seq_state, step_index], outputs=[seq_state, steps_display, seq_status])
    clear_btn.click(fn=clear_steps, outputs=[seq_state, steps_display, seq_status])

    run_btn.click(fn=on_run, inputs=[seq_state, loop_count], outputs=[seq_status, run_btn, stop_btn])
    stop_btn.click(fn=on_stop, outputs=[seq_status, run_btn, stop_btn])

    save_btn.click(fn=on_save, inputs=[seq_state, seq_name_input], outputs=[seq_status, saved_dropdown])
    load_btn.click(fn=on_load, inputs=[saved_dropdown], outputs=[seq_state, steps_display, seq_status])
    del_btn.click(fn=on_delete, inputs=[saved_dropdown], outputs=[seq_status, saved_dropdown])

    timer = gr.Timer(value=0.5)
    timer.tick(fn=refresh_seq_status, outputs=[seq_status, run_btn, stop_btn])
