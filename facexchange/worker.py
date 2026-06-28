"""
Subprocess swap worker — runs ONE swap in a child process and exits,
releasing ALL GPU memory. Prevents VRAM buildup across multiple swaps.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Callable, Optional


def run_worker(
    source_path: str,
    target_path: str,
    settings: dict,
    progress_cb: Optional[Callable] = None,
) -> dict:
    """
    Spawn a child Python process that performs the swap and exits.
    When the child exits, onnxruntime's CUDA arena is fully freed.
    settings must have keys: enhancer, quality, scale
    """
    worker_script = Path(__file__).resolve().parent / "_worker.py"
    job_id = uuid.uuid4().hex

    from . import config

    job_file = config.TEMP_DIR / f"job_{job_id}.json"
    status_file = config.TEMP_DIR / f"status_{job_id}.json"

    out_path = str(config.OUTPUT_DIR / f"swap_{job_id}.mp4")

    job = {
        "source": source_path,
        "target": target_path,
        "settings": {
            "enhancer": settings.get("enhancer", "gfpgan_1.4"),
            "quality": settings.get("quality", 85),
            "scale": settings.get("scale", 1.0),
        },
        "output": out_path,
        "job_id": job_id,
    }

    with open(job_file, "w") as f:
        json.dump(job, f)

    proc = subprocess.Popen(
        [sys.executable, str(worker_script), str(job_file), str(status_file)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )

    last_msg = ""
    stderr_lines = []

    while proc.poll() is None:
        try:
            if status_file.exists():
                with open(status_file) as f:
                    status = json.load(f)
                if status.get("msg") and status["msg"] != last_msg:
                    last_msg = status["msg"]
                if progress_cb:
                    progress_cb(status)
        except Exception:
            pass
        time.sleep(1.5)

    if proc.stderr:
        stderr_lines = [l for l in proc.stderr.read().splitlines() if l.strip()]

    result: dict = {}
    try:
        if status_file.exists():
            with open(status_file) as f:
                result = json.load(f)
    except Exception:
        pass

    for f in [job_file, status_file]:
        try:
            f.unlink(missing_ok=True)
        except Exception:
            pass

    success = result.get("success", False)
    msg = result.get("msg", "")

    if not success and not msg:
        msg = "Worker exited without writing status (exit code: {})".format(proc.returncode)
        if stderr_lines:
            msg += "\nStderr:\n" + "\n".join(stderr_lines[-10:])

    return {
        "success": success,
        "output_path": result.get("output_path") or out_path,
        "message": msg,
        "elapsed_sec": result.get("elapsed_sec", 0),
        "logs": [],
    }
