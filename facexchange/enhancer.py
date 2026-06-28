"""
GFPGAN face enhancement via ONNX runtime.
Aligns each detected face using 5-point landmarks, runs the GFPGAN
ONNX model, and blends the enhanced face back with a feathered mask.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import onnxruntime

from . import config

log = logging.getLogger("facexchange.enhancer")

_SESSION: Optional[onnxruntime.InferenceSession] = None

# Canonical face template from FFHQ dataset (512 px)
FFHQ_TEMPLATE = np.array(
    [
        [192.98138, 239.94708],
        [318.90277, 240.19366],
        [256.63416, 314.01935],
        [201.26117, 371.41043],
        [313.08905, 371.15118],
    ],
    dtype=np.float32,
)


def load_enhancer() -> onnxruntime.InferenceSession:
    global _SESSION
    if _SESSION is not None:
        return _SESSION
    model_path = config.MODELS_DIR / "GFPGANv1.4.onnx"
    if not model_path.exists():
        raise FileNotFoundError(f"GFPGAN model not found at {model_path}")
    try:
        avail = onnxruntime.get_available_providers()
        providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in avail]
    except Exception:
        providers = ["CPUExecutionProvider"]
    opts = onnxruntime.SessionOptions()
    opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    _SESSION = onnxruntime.InferenceSession(str(model_path), sess_options=opts, providers=providers)
    log.info("GFPGAN enhancer loaded")
    return _SESSION


def unload_enhancer():
    global _SESSION
    _SESSION = None


def enhance_faces(frame: np.ndarray, faces: list) -> np.ndarray:
    session = load_enhancer()
    inp = session.get_inputs()[0]
    inp_name = inp.name
    try:
        out_size = int(inp.shape[2])
        if out_size <= 0:
            out_size = 512
    except (ValueError, TypeError, IndexError):
        out_size = 512

    result = frame.copy()
    template = FFHQ_TEMPLATE * (out_size / 512.0)

    for face in faces:
        kps = getattr(face, "kps", None)
        if kps is None or len(kps) < 5:
            continue
        landmarks = kps.astype(np.float32)

        mat, _ = cv2.estimateAffinePartial2D(landmarks, template, method=cv2.LMEDS)
        if mat is None:
            continue

        aligned = cv2.warpAffine(
            result, mat, (out_size, out_size),
            borderMode=cv2.BORDER_CONSTANT, borderValue=(135, 133, 132),
        )

        rgb = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB).astype(np.float32)
        rgb = rgb / 255.0
        rgb = (rgb - 0.5) / 0.5
        tensor = np.expand_dims(np.transpose(rgb, (2, 0, 1)), axis=0)

        try:
            raw = session.run(None, {inp_name: tensor})[0]
        except Exception:
            continue

        enhanced = np.squeeze(raw)
        enhanced = np.transpose(enhanced, (1, 2, 0))
        enhanced = (enhanced + 1.0) / 2.0
        enhanced = np.clip(enhanced * 255.0, 0, 255).astype(np.uint8)
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR)

        if enhanced.shape[:2] != (out_size, out_size):
            enhanced = cv2.resize(enhanced, (out_size, out_size), interpolation=cv2.INTER_LANCZOS4)

        inv_mat = cv2.invertAffineTransform(mat)
        inv_enhanced = cv2.warpAffine(
            enhanced, inv_mat, (frame.shape[1], frame.shape[0]),
            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0),
        )

        mask = np.ones((out_size, out_size), dtype=np.float32)
        border = max(1, int(out_size * 0.05))
        ramp = np.linspace(0.0, 1.0, border, dtype=np.float32)
        mask[:border, :] *= ramp[:, None]
        mask[-border:, :] *= ramp[::-1][:, None]
        mask[:, :border] *= ramp[None, :]
        mask[:, -border:] *= ramp[::-1][None, :]
        if config.ENHANCE_WEIGHT < 1.0:
            mask *= config.ENHANCE_WEIGHT

        mask_3c = np.stack([mask] * 3, axis=-1)
        inv_mask = cv2.warpAffine(
            mask_3c, inv_mat, (frame.shape[1], frame.shape[0]),
            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0),
        )
        inv_mask = np.clip(inv_mask, 0.0, 1.0)

        result = (result.astype(np.float32) * (1.0 - inv_mask) +
                  inv_enhanced.astype(np.float32) * inv_mask)
        result = np.clip(result, 0, 255).astype(np.uint8)

    return result
