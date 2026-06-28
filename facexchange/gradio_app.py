"""
Gradio web interface for local face swap.
Alternative to the Telegram bot — same engine, same quality.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Optional

os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

_env_path = (Path(__file__).resolve().parent.parent / ".env")
if _env_path.exists():
    for _line in _env_path.read_text("utf-8").splitlines():
        _line = _line.strip()
        if _line and "=" in _line and not _line.startswith("#"):
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

import gradio as gr

from . import config
from .presets import PRESETS, auto_recommend, get_preset
from .safety import gpu_vram_mb
from .worker import run_worker

log = logging.getLogger("facexchange.web")


def _detect_gpu() -> str:
    vram = gpu_vram_mb()
    if vram is None:
        return "CPU mode (no NVIDIA GPU detected)"
    gb = vram / 1024
    rec = auto_recommend(gb)
    return f"GPU {gb:.1f} GB VRAM — recommended: {rec.name}"


def _swap_job(
    source_path: str,
    target_path: str,
    preset_name: str,
    progress: gr.Progress = gr.Progress(),
) -> tuple[Optional[str], str]:
    if not source_path:
        return None, "Upload a source photo first."
    if not target_path:
        return None, "Upload a target video first."

    preset = None
    for p in PRESETS:
        if p.name == preset_name or preset_name.startswith(p.name):
            preset = p
            break
    if preset is None:
        preset = PRESETS[2]

    progress(0.05, desc="Queuing job…")

    def _cb(d: dict):
        pct = d.get("pct", 0)
        msg = d.get("msg", "")
        progress(pct / 100, desc=msg)

    settings = {
        "enhancer": preset.enhancer,
        "quality": preset.quality,
        "scale": preset.scale,
    }
    result = run_worker(source_path, target_path, settings, progress_cb=_cb)

    if result.get("success") and result.get("output_path"):
        return result["output_path"], result.get("message", "Done")
    else:
        err = result.get("message", "Unknown error")
        return None, f"Failed: {err}"


def create_interface() -> gr.Blocks:
    gpu_info = _detect_gpu()

    with gr.Blocks(title="FaceXchange — Local Face Swap") as demo:
        gr.Markdown(
            "# FaceXchange\n"
            "### Private, local face swap powered by InsightFace\n\n"
            "Upload a source photo (the face you want to paste) and a target video "
            "(the video to swap into). Select your quality preset and click Swap.\n\n"
            "Tip: For queue management and mobile access, use the Telegram bot instead."
        )

        with gr.Row():
            source_input = gr.Image(label="Source Face", type="filepath", height=300)
            target_input = gr.Video(label="Target Video", height=300)

        with gr.Row():
            preset_dropdown = gr.Dropdown(
                choices=[p.name for p in PRESETS],
                value=PRESETS[2].name,
                label="Quality Preset",
            )
            gr.Textbox(value=gpu_info, label="System", scale=1)

        with gr.Row():
            swap_btn = gr.Button("Swap Faces", variant="primary", size="lg", scale=2)
            clear_btn = gr.Button("Clear", size="lg")

        output_video = gr.Video(label="Result", height=400, visible=False)
        output_text = gr.Textbox(label="Status")

        for p in PRESETS:
            gr.Markdown(f"**{p.name}** — {p.description}")

        gr.Markdown("FaceXchange — MIT License")

        def on_swap(src, tgt, preset_name, prog=gr.Progress()):
            return _swap_job(src, tgt, preset_name, prog)

        def on_clear():
            return None, None, None, "Cleared."

        swap_btn.click(
            fn=on_swap,
            inputs=[source_input, target_input, preset_dropdown],
            outputs=[output_video, output_text],
            show_progress="full",
        ).then(
            fn=lambda v, t: (
                gr.update(visible=v is not None, value=v),
                gr.update(value=t),
            ),
            inputs=[output_video, output_text],
            outputs=[output_video, output_text],
        )

        clear_btn.click(
            fn=on_clear,
            outputs=[source_input, target_input, output_video, output_text],
        )

    demo.queue(default_concurrency_limit=1)
    return demo


def run_web(host: Optional[str] = None, port: Optional[int] = None):
    h = host or config.GRADIO_HOST
    p = port or config.GRADIO_PORT
    log.info(f"Starting web UI at http://{h}:{p}")
    demo = create_interface()
    demo.launch(
        server_name=h,
        server_port=p,
        quiet=True,
        theme=gr.themes.Soft(primary_hue="violet", secondary_hue="indigo"),
        css="footer{display:none!important}.gradio-container{max-width:800px!important;margin:auto!important}",
    )


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        level=logging.INFO,
    )
    run_web()
