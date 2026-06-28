"""
_worker.py — child process entry point for a single swap.
Args: <job.json> <status.json>
Loads models, processes ONE video, writes status, exits.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

_status_file_path: Path | None = None


def _write_status_early(data: dict):
    global _status_file_path
    if _status_file_path:
        try:
            _status_file_path.write_text(json.dumps(data))
        except Exception:
            pass


try:
    _PROJECT_ROOT = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(_PROJECT_ROOT))

    import site as _site

    _nvidia_bin_dirs = []
    for _s in _site.getsitepackages():
        _pkg_dir = os.path.join(_s, "nvidia")
        if os.path.isdir(_pkg_dir):
            for _root, _dirs, _files in os.walk(_pkg_dir):
                for _d in _dirs:
                    if _d == "bin":
                        _nvidia_bin_dirs.append(os.path.join(_root, _d))
                _dirs[:] = [d for d in _dirs if d not in ("include", "lib", "libsrc", "share")]

    _cuda_dirs = _nvidia_bin_dirs + [
        os.path.join(os.environ.get("CUDA_PATH", ""), "bin"),
        os.path.join(os.path.expanduser("~"), "miniconda3", "envs", "facexchange", "Library", "bin"),
        os.path.join(os.path.expanduser("~"), "miniconda3", "envs", "facex", "Library", "bin"),
        "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\bin",
        "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.7\\bin",
        "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.6\\bin",
        "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.5\\bin",
        "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.4\\bin",
        "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.3\\bin",
        "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.2\\bin",
        "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.1\\bin",
        "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\bin",
    ]
    for _d in _cuda_dirs:
        if os.path.isdir(_d) and _d.lower() not in os.environ.get("PATH", "").lower():
            os.environ["PATH"] = _d + os.pathsep + os.environ.get("PATH", "")
            if hasattr(os, "add_dll_directory"):
                try:
                    os.add_dll_directory(_d)
                except Exception:
                    pass

    from facexchange.presets import make_preset
    from facexchange.engine import process_video
except Exception as _e:
    import traceback as _tb
    _write_status_early({"type": "error", "msg": f"Worker init failed: {_e}\n{_tb.format_exc()}", "pct": 0, "success": False})
    sys.exit(1)


def main():
    job_file = Path(sys.argv[1])
    status_file = Path(sys.argv[2])

    global _status_file_path
    _status_file_path = status_file

    t0 = time.time()

    def _write_status(data: dict):
        data["elapsed"] = time.time() - t0
        try:
            status_file.write_text(json.dumps(data))
        except Exception:
            pass

    def _cb(data: dict):
        data["elapsed"] = time.time() - t0
        _write_status(data)

    try:
        job = json.loads(job_file.read_text())
        src = job["source"]
        tgt = job["target"]
        s = job.get("settings", {})
        out = job.get("output")
    except Exception as e:
        _write_status({"type": "error", "msg": f"Bad job: {e}", "pct": 0, "success": False})
        sys.exit(1)

    preset = make_preset(
        enhancer=s.get("enhancer", "gfpgan_1.4"),
        quality=s.get("quality", 85),
        scale=s.get("scale", 1.0),
    )

    try:
        _write_status({"type": "stage", "msg": "Starting…", "pct": 0})
        result = process_video(src, tgt, preset, progress_cb=_cb)
        _write_status({
            "type": "done",
            "msg": result.get("message", "Done"),
            "pct": 100,
            "output_path": result.get("output_path", out),
            "success": result.get("success", False),
            "elapsed_sec": result.get("elapsed_sec", time.time() - t0),
        })
    except Exception as e:
        import traceback
        _write_status({
            "type": "error",
            "msg": f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
            "pct": 0,
            "success": False,
        })

    try:
        job_file.unlink(missing_ok=True)
    except Exception:
        pass


if __name__ == "__main__":
    main()
