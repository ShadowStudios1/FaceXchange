"""
Paths and runtime configuration.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

WORK_DIR = ROOT / "workspace"
UPLOAD_DIR = WORK_DIR / "uploads"
OUTPUT_DIR = WORK_DIR / "outputs"
TEMP_DIR = WORK_DIR / "temp"
MODELS_DIR = ROOT / "models"

for _d in (UPLOAD_DIR, OUTPUT_DIR, TEMP_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

TELEGRAM_TOKEN = os.environ.get("FACEXCHANGE_TOKEN", "")
ALLOWED_USERS: set[int] = set()

MAX_VIDEO_DURATION = int(os.environ.get("FACEXCHANGE_MAX_DURATION", "60"))
MAX_VIDEO_WIDTH = int(os.environ.get("FACEXCHANGE_MAX_WIDTH", "1920"))
MAX_VIDEO_HEIGHT = int(os.environ.get("FACEXCHANGE_MAX_HEIGHT", "1080"))

JOB_TIMEOUT_SEC = int(os.environ.get("FACEXCHANGE_JOB_TIMEOUT", "3600"))
MIN_FREE_DISK_GB = 4

ENHANCE_WEIGHT = 0.6

GRADIO_HOST = os.environ.get("FACEXCHANGE_HOST", "127.0.0.1")
GRADIO_PORT = int(os.environ.get("FACEXCHANGE_PORT", "7860"))
