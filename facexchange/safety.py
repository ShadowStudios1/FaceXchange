"""
Health checks and resource watchdog.
"""

from __future__ import annotations

import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional

import psutil

from . import config


def free_disk_gb(path: Path) -> float:
    return shutil.disk_usage(str(path)).free / (1024**3)


def free_ram_gb() -> float:
    return psutil.virtual_memory().available / (1024**3)


def gpu_vram_mb() -> Optional[int]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL, text=True, timeout=5,
        ).strip()
        for line in out.split("\n"):
            parts = [x.strip() for x in line.split(",")]
            if len(parts) == 2:
                return int(parts[1])
        return None
    except Exception:
        return None


def preflight_checks() -> tuple[bool, str]:
    disk = free_disk_gb(config.TEMP_DIR)
    if disk < config.MIN_FREE_DISK_GB:
        return False, f"Low disk: {disk:.1f} GB free"
    ram = free_ram_gb()
    if ram < 1.5:
        return False, f"Low RAM: {ram:.1f} GB free"
    return True, "OK"


class Watchdog(threading.Thread):
    """Monitors RAM/VRAM during processing and kills on overload."""

    def __init__(self, timeout_sec: int, label: str = "job"):
        super().__init__(daemon=True)
        self.timeout_sec = timeout_sec
        self.label = label
        self.start_time = time.time()
        self.stop_event = threading.Event()
        self.reason: Optional[str] = None

    def stop(self):
        self.stop_event.set()

    def run(self):
        while not self.stop_event.is_set():
            elapsed = time.time() - self.start_time
            if elapsed > self.timeout_sec:
                self.reason = f"Timeout {int(elapsed)}s"
                return
            ram = free_ram_gb()
            if ram < 0.8:
                self.reason = f"RAM critical ({ram:.2f} GB)"
                return
            self.stop_event.wait(3)


def cleanup_temp(max_age_sec: int = 600):
    now = time.time()
    for f in config.TEMP_DIR.glob("*"):
        try:
            if f.is_file() and now - f.stat().st_mtime > max_age_sec:
                f.unlink(missing_ok=True)
        except Exception:
            pass
