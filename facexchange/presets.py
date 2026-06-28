"""
GPU tier presets — each preset tunes detection resolution, enhancer,
and quality to match available VRAM. The installer auto-recommends
based on detected hardware.

Also provides make_preset() for constructing a Preset from individual
user settings (enhancer, quality, scale) for the granular UI.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Preset:
    id: str
    name: str
    description: str
    min_vram_gb: float
    det_size: int
    det_interval: int
    enhancer: str
    quality: int
    scale: float = 1.0
    recommended: bool = False


PRESETS: list[Preset] = [
    Preset(
        id="cpu",
        name="CPU Safe",
        description="Works on any system with or without GPU. No enhancement. Slow but reliable.",
        min_vram_gb=0,
        det_size=640,
        det_interval=10,
        enhancer="none",
        quality=60,
    ),
    Preset(
        id="basic",
        name="Basic GPU",
        description="Entry-level GPU (2 GB VRAM). No enhancement, reasonable speed.",
        min_vram_gb=2,
        det_size=480,
        det_interval=7,
        enhancer="none",
        quality=75,
    ),
    Preset(
        id="standard",
        name="Standard",
        description="4 GB VRAM recommended. Face enhancement enabled. Balanced quality & speed.",
        min_vram_gb=4,
        det_size=320,
        det_interval=5,
        enhancer="gfpgan_1.4",
        quality=85,
        recommended=True,
    ),
    Preset(
        id="high",
        name="High Quality",
        description="8+ GB VRAM. Lower detection stride, higher quality face enhancement.",
        min_vram_gb=8,
        det_size=256,
        det_interval=3,
        enhancer="gfpgan_1.4",
        quality=92,
    ),
    Preset(
        id="maximum",
        name="Maximum",
        description="12+ GB VRAM. Frame-by-frame detection, maximum quality enhancement & output.",
        min_vram_gb=12,
        det_size=160,
        det_interval=1,
        enhancer="gfpgan_1.4",
        quality=100,
    ),
]


QUALITY_LEVELS = [60, 70, 80, 85, 92, 100]
ENHANCER_OPTIONS = {"none": "None", "gfpgan_1.4": "GFPGAN v1.4"}
SCALE_OPTIONS = [1.0, 1.5, 2.0]


def make_preset(enhancer: str = "gfpgan_1.4", quality: int = 85, scale: float = 1.0) -> Preset:
    """Build a Preset from individual user-facing settings."""
    if quality >= 90:
        det_size, det_interval = 160, 1
    elif quality >= 80:
        det_size, det_interval = 320, 3
    elif quality >= 70:
        det_size, det_interval = 480, 5
    elif quality >= 60:
        det_size, det_interval = 640, 10
    else:
        det_size, det_interval = 640, 15

    return Preset(
        id="_custom",
        name="Custom",
        description=f"Custom: enhancer={enhancer}, quality={quality}, scale={scale}",
        min_vram_gb=0,
        det_size=det_size,
        det_interval=det_interval,
        enhancer=enhancer,
        quality=quality,
        scale=scale,
    )


def get_preset(preset_id: str) -> Optional[Preset]:
    for p in PRESETS:
        if p.id == preset_id:
            return p
    return None


def auto_recommend(vram_gb: float) -> Preset:
    best = PRESETS[0]
    for p in PRESETS:
        if vram_gb >= p.min_vram_gb:
            best = p
    return best
