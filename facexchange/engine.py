"""
Face swap engine — direct InsightFace API, zero third-party wrappers.

Two-pass processing:
  Pass 1: Face detection + swap via inswapper_128
  Pass 2: GFPGAN face enhancement (optional)

All GPU memory is released when the function returns.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Callable, Optional

import cv2
import insightface
import numpy as np
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model

from . import config
from . import safety as safety_mod
from .enhancer import enhance_faces, load_enhancer, unload_enhancer

log = logging.getLogger("facexchange.engine")

_analyser: Optional[FaceAnalysis] = None
_swapper: Optional[insightface.model_zoo.model_store.Model] = None
_job_lock = threading.Lock()


def _onnxruntime_providers() -> list[str]:
    try:
        import onnxruntime
        return list(onnxruntime.get_available_providers())
    except Exception:
        return ["CPUExecutionProvider"]


def _gpu_present() -> bool:
    try:
        r = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=5)
        return r.returncode == 0
    except Exception:
        return False


def _find_providers() -> list[str]:
    """Determine which onnxruntime execution providers can actually be used.

    IMPORTANT: we check onnxruntime.get_available_providers() — NOT just nvidia-smi.
    A system can have an NVIDIA GPU (nvidia-smi works) yet onnxruntime may still lack
    CUDAExecutionProvider, typically because BOTH `onnxruntime` (CPU) and
    `onnxruntime-gpu` are pip-installed and the CPU build shadows the GPU build.
    In that case we must NOT claim GPU mode — otherwise everything silently runs on
    CPU while the bot reports GPU active.
    """
    avail = _onnxruntime_providers()
    has_cuda = "CUDAExecutionProvider" in avail
    gpu_hw = _gpu_present()

    if has_cuda:
        log.info("onnxruntime CUDAExecutionProvider available — using GPU")
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]

    if gpu_hw:
        log.error(
            "NVIDIA GPU detected (nvidia-smi works) but onnxruntime does NOT expose "
            "CUDAExecutionProvider (available: %s). This is usually caused by BOTH "
            "onnxruntime (CPU) and onnxruntime-gpu being installed, with the CPU build "
            "shadowing the GPU build. Falling back to CPU. "
            "Fix: pip uninstall onnxruntime onnxruntime-gpu -y && "
            "pip install onnxruntime-gpu==1.24.4 nvidia-cublas-cu12 nvidia-cuda-runtime-cu12",
            avail,
        )
    else:
        log.warning("GPU not found — using CPU (very slow)")
    return ["CPUExecutionProvider"]


def load_models(providers: Optional[list[str]] = None):
    global _analyser, _swapper
    if _analyser is not None:
        return
    if providers is None:
        providers = _find_providers()

    use_gpu = "CUDAExecutionProvider" in providers
    # ctx_id: >=0 -> GPU 0, -1 -> CPU. Only use GPU when onnxruntime actually
    # has CUDAExecutionProvider, otherwise insightface silently falls back to CPU.
    ctx_id = 0 if use_gpu else -1

    log.info(f"Loading face analyser (buffalo_l) on {'GPU' if use_gpu else 'CPU'}…")
    _analyser = FaceAnalysis(name="buffalo_l")
    det_size = 320
    _analyser.prepare(ctx_id=ctx_id, det_size=(det_size, det_size))

    swapper_path = config.MODELS_DIR / "inswapper_128.onnx"
    if not swapper_path.exists():
        raise FileNotFoundError(f"Swapper model not found at {swapper_path}")
    log.info(f"Loading face swapper (providers={providers})…")
    try:
        _swapper = get_model(str(swapper_path), providers=providers)
    except Exception as e:
        if "protobuf" in str(e).lower() or "onnxruntime" in str(e).lower():
            raise RuntimeError(
                "The inswapper_128.onnx model file is corrupted. "
                "Delete the file and re-run the installer to re-download it."
            ) from e
        raise

    actual = _swapper.session.get_providers()
    log.info(f"Swapper active providers: {actual}")
    if use_gpu and "CUDAExecutionProvider" not in actual:
        log.error(
            "Requested CUDAExecutionProvider but swapper is running on %s. "
            "onnxruntime GPU support is broken — see previous warnings. "
            "Reinstall onnxruntime-gpu and remove the CPU onnxruntime package.",
            actual,
        )


def unload_models():
    global _analyser, _swapper
    _analyser = None
    _swapper = None
    unload_enhancer()
    import gc
    gc.collect()


def _swap_frames(
    source_face,
    target_path: Path,
    fps: float,
    width: int,
    height: int,
    total_frames: int,
    preset,
    progress_cb: Optional[Callable] = None,
) -> tuple[Optional[Path], int, list[str]]:
    logs: list[str] = []
    intermediate = config.TEMP_DIR / f"swap_pass1_{uuid.uuid4().hex}.mp4"
    cap = cv2.VideoCapture(str(target_path))
    scale = getattr(preset, "scale", 1.0)
    if scale != 1.0:
        out_w, out_h = int(width * scale), int(height * scale)
    else:
        out_w, out_h = width, height
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(intermediate), fourcc, fps, (out_w, out_h))
    if not writer.isOpened():
        cap.release()
        return None, 0, ["Could not create output writer"]

    frame_idx = 0
    faces_found = 0
    report_every = max(1, total_frames // 50)
    t0 = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        faces = _analyser.get(frame)
        if faces:
            faces_found += 1
            for face in faces:
                frame = _swapper.get(frame, face, source_face, paste_back=True)

        if scale != 1.0:
            frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
        writer.write(frame)

        if progress_cb and frame_idx % report_every == 0:
            pct = int(frame_idx / total_frames * 100)
            elapsed = time.time() - t0
            progress_cb({
                "type": "swap",
                "pct": pct,
                "msg": f"Swapping {frame_idx}/{total_frames}",
                "elapsed": elapsed,
            })

    cap.release()
    writer.release()
    elapsed = time.time() - t0
    logs.append(f"Swap: {frame_idx} frames in {elapsed:.0f}s, {faces_found} with faces")
    return intermediate, faces_found, logs


def _enhance_pass(
    intermediate: Path,
    fps: float,
    width: int,
    height: int,
    total_frames: int,
    scale: float = 1.0,
    progress_cb: Optional[Callable] = None,
) -> tuple[Optional[Path], list[str]]:
    logs: list[str] = []
    output = config.TEMP_DIR / f"swap_pass2_{uuid.uuid4().hex}.mp4"
    cap = cv2.VideoCapture(str(intermediate))
    if scale != 1.0:
        out_w, out_h = int(width * scale), int(height * scale)
    else:
        out_w, out_h = width, height
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output), fourcc, fps, (out_w, out_h))

    frame_idx = 0
    report_every = max(1, total_frames // 50)
    t0 = time.time()

    try:
        load_enhancer()
    except Exception as e:
        cap.release()
        writer.release()
        return None, [f"Enhancer load failed: {e}"]

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        faces = _analyser.get(frame)
        if faces:
            try:
                frame = enhance_faces(frame, faces)
            except Exception:
                pass

        if scale != 1.0:
            frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
        writer.write(frame)

        if progress_cb and frame_idx % report_every == 0:
            pct = int(frame_idx / total_frames * 100)
            elapsed = time.time() - t0
            progress_cb({
                "type": "enhance",
                "pct": pct,
                "msg": f"Enhancing {frame_idx}/{total_frames}",
                "elapsed": elapsed,
            })

    cap.release()
    writer.release()
    elapsed = time.time() - t0
    logs.append(f"Enhance: {frame_idx} frames in {elapsed:.0f}s")
    return output, logs


def _merge_audio(video_no_audio: Path, audio_source: Path, output: Path):
    if not shutil.which("ffmpeg"):
        log.warning("ffmpeg not found — output video will have no audio")
        shutil.copy2(video_no_audio, output)
        return
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(video_no_audio), "-i", str(audio_source),
             "-c:v", "copy", "-c:a", "aac", "-b:a", "128k",
             "-map", "0:v:0", "-map", "1:a:0", str(output)],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120,
        )
    except Exception:
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(video_no_audio), "-c", "copy", str(output)],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60,
            )
        except Exception:
            log.warning("ffmpeg failed — output video will have no audio")
            shutil.copy2(video_no_audio, output)


def process_video(
    source_path: str,
    target_path: str,
    preset,
    progress_cb: Optional[Callable] = None,
) -> dict:
    logs: list[str] = []
    t_start = time.time()
    out_path = config.OUTPUT_DIR / f"swap_{uuid.uuid4().hex}.mp4"

    if not _job_lock.acquire(timeout=5):
        return {
            "success": False, "output_path": None,
            "message": "Another job is still running. Wait or restart.",
            "elapsed_sec": 0, "logs": logs,
        }

    def _cb(data: dict):
        data["elapsed"] = time.time() - t_start
        if progress_cb:
            progress_cb(data)

    try:
        ok, msg = safety_mod.preflight_checks()
        if not ok:
            return {"success": False, "output_path": None, "message": msg,
                    "elapsed_sec": 0, "logs": logs}
        _cb({"type": "stage", "pct": 0, "msg": "Preflight OK"})

        _cb({"type": "stage", "pct": 0, "msg": "Loading models…"})
        load_models()

        source_img = cv2.imread(source_path)
        if source_img is None:
            return {"success": False, "output_path": None,
                    "message": "Could not read source image.",
                    "elapsed_sec": time.time() - t_start, "logs": logs}

        _cb({"type": "stage", "pct": 1, "msg": "Detecting source face…"})
        source_faces = _analyser.get(source_img)
        if not source_faces:
            return {"success": False, "output_path": None,
                    "message": "No face found in source image. Use a clear front-facing photo.",
                    "elapsed_sec": time.time() - t_start, "logs": logs}
        source_face = source_faces[0]
        logs.append(f"Source face score: {source_face.det_score:.2f}")

        cap = cv2.VideoCapture(target_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        logs.append(f"Video: {width}x{height} @ {fps:.0f}fps, {total} frames")
        _cb({"type": "stage", "pct": 2, "msg": f"{width}x{height}, {total} frames"})

        _cb({"type": "swap", "pct": 3, "msg": "Swapping faces…"})
        intermediate, faces_found, swap_logs = _swap_frames(
            source_face, Path(target_path), fps, width, height, total,
            preset, progress_cb=_cb,
        )
        logs.extend(swap_logs)
        if intermediate is None:
            return {"success": False, "output_path": None,
                    "message": f"Swap failed: {swap_logs[-1] if swap_logs else 'unknown'}",
                    "elapsed_sec": time.time() - t_start, "logs": logs}

        if faces_found == 0:
            logs.append("No faces detected — outputting unchanged video")
            _merge_audio(intermediate, Path(target_path), out_path)
            intermediate.unlink(missing_ok=True)
            elapsed = time.time() - t_start
            _cb({"type": "done", "pct": 100, "msg": "Done (no faces in video)",
                 "elapsed": elapsed})
            return {"success": True, "output_path": str(out_path),
                    "message": "No faces found in target video.",
                    "elapsed_sec": elapsed, "logs": logs}

        target_path_obj = Path(target_path)

        if preset.enhancer != "none":
            _cb({"type": "enhance", "pct": 50, "msg": "Enhancing faces…"})
            enhanced, enhance_logs = _enhance_pass(
                intermediate, fps, width, height, total,
                scale=getattr(preset, "scale", 1.0), progress_cb=_cb,
            )
            logs.extend(enhance_logs)
            if enhanced is None:
                logs.append("Enhance failed, using swapped-only output")
                _merge_audio(intermediate, target_path_obj, out_path)
            else:
                _cb({"type": "stage", "pct": 90, "msg": "Merging audio…"})
                _merge_audio(enhanced, target_path_obj, out_path)
                enhanced.unlink(missing_ok=True)
            intermediate.unlink(missing_ok=True)
        else:
            _cb({"type": "stage", "pct": 90, "msg": "Merging audio…"})
            _merge_audio(intermediate, target_path_obj, out_path)
            intermediate.unlink(missing_ok=True)

        elapsed = time.time() - t_start
        mode = "swap+enhance" if preset.enhancer != "none" else "swap only"
        logs.append(f"Done: {elapsed:.0f}s ({mode})")
        _cb({"type": "done", "pct": 100, "msg": f"Done ({mode})", "elapsed": elapsed})
        return {
            "success": True,
            "output_path": str(out_path),
            "message": f"Done in {elapsed:.0f}s ({mode})",
            "elapsed_sec": elapsed,
            "logs": logs,
        }

    except Exception as e:
        log.exception("Engine error")
        _cb({"type": "error", "pct": 0, "msg": f"Error: {e}"})
        return {"success": False, "output_path": None,
                "message": f"{type(e).__name__}: {e}",
                "elapsed_sec": time.time() - t_start, "logs": logs}
    finally:
        unload_models()
        safety_mod.cleanup_temp()
        try:
            _job_lock.release()
        except RuntimeError:
            pass
