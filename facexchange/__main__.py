"""
Entry point: python -m facexchange [--web] [--bot] [--both]

  --bot    Start Telegram bot only (default)
  --web    Start Gradio web UI only
  --both   Start both Telegram bot and Gradio web UI
  --host   Gradio host (default: 127.0.0.1)
  --port   Gradio port (default: 7860)
"""

from __future__ import annotations

import argparse
import logging
import os
import threading
from pathlib import Path

# Load .env so token is in environment
_env_path = Path(__file__).resolve().parent.parent / ".env"
if _env_path.exists():
    for _line in _env_path.read_text("utf-8").splitlines():
        _line = _line.strip()
        if _line and "=" in _line and not _line.startswith("#"):
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

log = logging.getLogger("facexchange")


def main():
    parser = argparse.ArgumentParser(description="FaceXchange — local face swap")
    parser.add_argument("--web", action="store_true", help="Start Gradio web UI")
    parser.add_argument("--bot", action="store_true", help="Start Telegram bot")
    parser.add_argument("--both", action="store_true", help="Start both bot and web UI")
    parser.add_argument("--host", default="127.0.0.1", help="Web UI host")
    parser.add_argument("--port", type=int, default=7860, help="Web UI port")
    args = parser.parse_args()

    mode = "bot"
    if args.both:
        mode = "both"
    elif args.web:
        mode = "web"
    elif args.bot:
        mode = "bot"

    logging.basicConfig(
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        level=logging.INFO,
    )

    if mode in ("bot", "both"):
        from .bot import run_bot
        t = threading.Thread(target=run_bot, daemon=True)
        t.start()
        if mode == "bot":
            t.join()
            return

    if mode in ("web", "both"):
        from .gradio_app import run_web
        run_web(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
