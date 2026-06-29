# FaceXchange

**Private, free, open-source face swap on your own computer.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![Platform](https://img.shields.io/badge/Platform-Windows-lightgrey)]()
[![GPU](https://img.shields.io/badge/GPU-NVIDIA%20CUDA-green)]()
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen)]()

No cloud bills. No limits. Your hardware, your data, your faces.

*Built by [Shadow Studios](https://github.com/ShadowStudios1) — building the future of AI, one open-source tool at a time. If you find this useful, consider [starring the repo](https://github.com/ShadowStudios1/FaceXchange) and following along. We're just getting started.*

---

## Features

| | |
|---|---|
| **InsightFace Engine** | Direct inswapper_128 — no wrappers, no bloat |
| **GFPGAN Enhancement** | Optional face restoration via ONNX |
| **5 GPU Presets** | CPU Safe to Maximum — pick your tier |
| **Telegram Bot** | Queue management, live progress, mobile-friendly |
| **Local Web UI** | Gradio interface — upload and swap in your browser |
| **Auto Memory Cleanup** | Subprocess worker frees ALL GPU RAM after each swap |
| **Self-contained** | Everything in a local venv — no system pollution |
| **Private** | Everything runs locally. Your face data never leaves your PC |

---

> ⚠️ **IMPORTANT — THIS IS NOT A CLOUD BOT**
>
> FaceXchange runs **entirely on your own computer**. There are no remote servers, no cloud processing, no online hosting.
>
> - The Telegram bot you interact with on your phone connects back to **your laptop/PC** where the code is running.
> - **Your computer must stay ON** and the terminal window must stay **open** for the bot to work.
> - If you close the terminal, put your laptop to sleep, or shut it down — the bot goes **offline** immediately.
>
> Think of it like a game server you host yourself: when you stop the server, the game stops. Same here.
>
> For long sessions, keep your computer awake — disable sleep mode in your power settings.

---

## Which Interface Should I Use?

| Interface | Best for |
|-----------|----------|
| **Telegram Bot** | Frequent use, mobile, queue management, notifications |
| **Local Web UI** | Quick tests, no Telegram account needed |

Run both at the same time with `run_telegram.bat` + `run_web.bat`.

---

## System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| OS | Windows 10 | Windows 11 |
| Python | 3.10 | 3.11 — 3.12 |
| RAM | 8 GB | 16 GB |
| GPU | None (CPU mode) | NVIDIA 4 GB+ VRAM |
| Disk | 4 GB free | 10 GB free |
| Internet | Initial setup only | -- |

---

## Prerequisites

### Python

FaceXchange requires **Python 3.10, 3.11, or 3.12** (Python 3.13 and 3.14 are **not** supported yet).

1. Go to [python.org/downloads](https://www.python.org/downloads/)
2. Download the latest **Python 3.12.x** for Windows (e.g., `python-3.12.x-amd64.exe`)
3. Run the installer
4. **IMPORTANT:** At the bottom of the installer, check **"Add Python to PATH"**
5. Click **Install Now**
6. After install, open a **Command Prompt** and verify:
   ```
   python --version
   ```
   You should see `Python 3.12.x`

> ⚠️ **Warning:** Do NOT install Python from the Microsoft Store or via Conda — use the official python.org installer above.

### Git (Optional — for cloning)

If you want to use the clone option instead of ZIP download:

1. Go to [git-scm.com/download/win](https://git-scm.com/download/win)
2. Download and run the installer (default settings are fine)
3. After install, open a **Command Prompt** and verify:
   ```
   git --version
   ```

### FFmpeg (Optional — for audio in output videos)

The face swap works without FFmpeg, but output videos will have no audio.

1. Go to [ffmpeg.org/download.html](https://ffmpeg.org/download.html)
2. Click the **Windows** icon, then download the latest build from **gyan.dev** or **BtbN**
3. Extract the zip to `C:\ffmpeg`
4. Add `C:\ffmpeg\bin` to your system **PATH**:
   - Search "Environment Variables" in Windows
   - Under **System variables**, find `Path`, click **Edit**
   - Click **New**, add `C:\ffmpeg\bin`
   - Click **OK** on all windows
5. Verify:
   ```
   ffmpeg -version
   ```

---

## Quick Start

### 1. Download

**Option A — Clone (recommended):**
Open a terminal (Command Prompt or PowerShell) and run:

```bash
git clone https://github.com/ShadowStudios1/FaceXchange.git
cd FaceXchange
```

**Option B — Download ZIP:**
Go to [github.com/ShadowStudios1/FaceXchange](https://github.com/ShadowStudios1/FaceXchange), click the green **Code** button, and select **Download ZIP**. Then extract the folder and open it.

### 2. Run Installer

Double-click **`installer.bat`**

The installer walks you through:
- Python version check
- GPU detection and preset recommendation
- Interface selection (Telegram + Web, or Web only)
- Quality preset selection (5 tiers)
- Model downloads (~900 MB, one-time)
- Telegram bot setup (optional)
- Package install into a local `.venv`
- Generates `run.bat`, `run_telegram.bat`, `run_web.bat`

### 3. Start Swapping

After installation, double-click **`run.bat`** (preferred) or one of the others:

| File | Starts |
|------|--------|
| `run.bat` | Telegram Bot (or Web UI if no token set) |
| `run_telegram.bat` | Telegram Bot only |
| `run_web.bat` | Local Web UI only |

**Telegram Bot:**
1. Double-click `run.bat`
2. Open Telegram, message your bot
3. `/start` — welcome menu
4. Send a clear front-facing photo (source face)
5. Send a video (target) — bot swaps and returns the result

**Web UI:**
1. Double-click `run_web.bat`
2. Open http://127.0.0.1:7860
3. Upload source photo + target video
4. Select preset and click **Swap Faces**
5. Wait for the result

---

## Preset Guide

| Preset | Min VRAM | Speed | Enhancer |
|--------|----------|-------|----------|
| CPU Safe | None | Slow | No |
| Basic GPU | 2 GB | Fast | No |
| Standard | 4 GB | Fast | GFPGAN |
| High Quality | 8 GB | Fast | GFPGAN |
| Maximum | 12 GB | Medium | GFPGAN |

Change presets in the bot with `/settings` or in the Web UI dropdown.

---

## Telegram Bot Commands

| Command | Description |
|---------|-------------|
| `/start` | Welcome menu |
| `/swap` | Start a new face swap |
| `/settings` | Change quality preset |
| `/cancel` | Cancel current operation |

---

## Project Structure

```
facexchange/
├── installer.bat           # Double-click to install
├── install.py              # Rich TUI installer
├── run.bat                 # Start (prefers Telegram)
├── run_telegram.bat        # Start Telegram bot only
├── run_web.bat             # Start Web UI only
├── requirements.txt        # Python dependencies
├── LICENSE                 # MIT License
├── README.md               # This file
├── .venv/                  # Virtual environment (created by installer)
├── models/                 # Downloaded models
├── workspace/              # Uploads, outputs, temp
└── facexchange/            # Core package
    ├── __init__.py
    ├── __main__.py         # CLI: --bot, --web, --both
    ├── bot.py              # Telegram bot
    ├── gradio_app.py       # Local Web UI
    ├── engine.py           # Face swap engine (InsightFace)
    ├── enhancer.py         # GFPGAN enhancement
    ├── worker.py           # Subprocess worker
    ├── _worker.py          # Child process entry point
    ├── presets.py          # 5 quality presets
    ├── config.py           # Paths and settings
    └── safety.py           # Resource monitoring
```

---

## Manual Setup

```bash
python -m venv .venv
.venv\Scripts\activate
# For CPU:
pip install -r requirements.txt
pip install onnxruntime
# For GPU (CUDA 12.x+):
pip install -r requirements.txt
pip uninstall onnxruntime onnxruntime-gpu -y
pip install onnxruntime-gpu==1.24.4 nvidia-cublas-cu12 nvidia-cuda-nvrtc-cu12 nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12 nvidia-cufft-cu12 nvidia-curand-cu12 nvidia-cusolver-cu12 nvidia-cusparse-cu12
set FACEXCHANGE_TOKEN=your_token_here
python -m facexchange --bot
```

---

## Troubleshooting

### Installer fails with "Python not found"

Download Python 3.10 or 3.11 from [python.org](https://www.python.org/downloads/).  
**Important:** During installation, check **"Add Python to PATH"** at the bottom of the installer.

### Installer fails with "Python 3.10+ required"

You have an older Python version. Download Python 3.10 or newer from [python.org](https://www.python.org/downloads/).

### "nvidia-smi is not recognized" / GPU not detected

Your system does not have an NVIDIA GPU or the NVIDIA driver is not installed.  
- If you have an NVIDIA GPU, download and install the latest driver from [nvidia.com/drivers](https://www.nvidia.com/drivers/).  
- If you don't have an NVIDIA GPU, the installer will use CPU mode automatically. CPU mode is slow (5–15 min per 30s video).

### GPU detected but onnxruntime falls back to CPU

The bot reports "GPU (CUDA)" only when onnxruntime actually has `CUDAExecutionProvider` available **and loaded** — not just when `nvidia-smi` works. Two common causes:

**1. Both `onnxruntime` (CPU) and `onnxruntime-gpu` installed** — they share the same `onnxruntime` namespace, and the CPU build shadows the GPU build, so `CUDAExecutionProvider` disappears.

Fix:
```
.venv\Scripts\pip uninstall onnxruntime onnxruntime-gpu -y
.venv\Scripts\pip install --no-deps onnxruntime-gpu==1.24.4
```

**2. Missing CUDA runtime DLLs** — `onnxruntime-gpu` 1.24.x needs the **full** CUDA 12 + cuDNN 9 runtime, not just cublas + cudart. If logs show `cufft64_11.dll missing`, `cublasLt64_12.dll missing`, or `Require cuDNN 9.* and CUDA 12.*`:

```
.venv\Scripts\pip install nvidia-cublas-cu12 nvidia-cuda-nvrtc-cu12 nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12 nvidia-cufft-cu12 nvidia-curand-cu12 nvidia-cusolver-cu12 nvidia-cusparse-cu12
```

The installer installs the full stack automatically. Re-run the installer if you hit this.

To verify GPU is actually in use, check the bot logs for:
```
Swapper active providers: ['CUDAExecutionProvider', 'CPUExecutionProvider']
```
If it shows only `['CPUExecutionProvider']`, GPU is NOT being used — apply the fixes above.

### "ImportError: No module named onnxruntime"

The onnxruntime package failed to install. Re-run the installer, or manually:

```bash
.venv\Scripts\pip install onnxruntime
```

For GPU:

```bash
.venv\Scripts\pip uninstall onnxruntime onnxruntime-gpu -y
.venv\Scripts\pip install onnxruntime-gpu==1.24.4 nvidia-cublas-cu12 nvidia-cuda-nvrtc-cu12 nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12 nvidia-cufft-cu12 nvidia-curand-cu12 nvidia-cusolver-cu12 nvidia-cusparse-cu12
```

### "cublasLt64_XX.dll not found" or "DLL load failed"

The CUDA runtime DLLs are missing or incompatible. This happens when:

1. **onnxruntime-gpu version mismatch** — The installer pins `onnxruntime-gpu==1.24.4` (CUDA 12.x compatible). If you manually installed a newer version, downgrade:
   ```
   .venv\Scripts\pip install onnxruntime-gpu==1.24.4
   ```
2. **NVIDIA driver too old** — Update to driver **R525+** from [nvidia.com/drivers](https://www.nvidia.com/drivers/).
3. **CUDA runtime DLLs not found** — Re-install the **full** CUDA 12 + cuDNN 9 stack:
   ```
   .venv\Scripts\pip install nvidia-cublas-cu12 nvidia-cuda-nvrtc-cu12 nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12 nvidia-cufft-cu12 nvidia-curand-cu12 nvidia-cusolver-cu12 nvidia-cusparse-cu12
   ```

### Model download fails (connection error)

The installer downloads models from GitHub Releases and HuggingFace (~900 MB total).  
If downloads fail:

- Check your internet connection.
- Try using a VPN if GitHub/HuggingFace is blocked in your region.
- Download models manually:
  1. Download `buffalo_l.zip` from [insightface v0.7 release](https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip)
  2. Extract to `%USERPROFILE%\.insightface\models\buffalo_l\`
  3. Download `inswapper_128.onnx` from [HuggingFace](https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx) to `models\inswapper_128.onnx`
  4. Download `GFPGANv1.4.onnx` from [HuggingFace](https://huggingface.co/neurobytemind/GFPGANv1.4.onnx/resolve/main/GFPGANv1.4.onnx) to `models\GFPGANv1.4.onnx`

### "No face found in source image"

The source photo must contain a clear, front-facing face. Common issues:

- Photo is too dark or blurry.
- Face is too small in the frame (crop to just the face).
- Face is at an extreme angle (use a front-facing photo).
- Multiple faces in the photo (crop to just the person you want).

### "FFmpeg not found — output video will have no audio"

FFmpeg is optional but recommended for audio preservation.  
Download FFmpeg from [ffmpeg.org](https://ffmpeg.org/download.html) or install via:

```bash
winget install FFmpeg
```

Or simply extract the zip and add the `bin` folder to your PATH.  
The swap still works without FFmpeg — the output video will just have no audio.

### Telegram bot starts but does not respond to commands

Common causes:

1. **Wrong token** — Double-check your bot token from @BotFather. Re-run the installer or edit `.env`.
2. **Wrong user ID** — The bot only responds to your user ID. Get your ID from @userinfobot on Telegram.
3. **Bot is offline** — The bot must be running (keep the terminal window open).
4. **Firewall** — Ensure the script can connect to Telegram API (api.telegram.org on port 443).

### "Another job is still running" when swapping

FaceXchange processes one video at a time. Wait for the current job to finish, then try again.  
If a previous job crashed, restart the bot to clear the lock.

### Out of memory / CUDA out of memory

Your GPU ran out of VRAM. Try:

1. Switch to a lower preset (e.g., from Maximum to High or Standard).
2. Use a shorter video (the bot limits videos to 60 seconds by default).
3. Close other GPU-intensive programs (games, other AI tools).

### "pip is not recognized" during manual install

Ensure Python's Scripts folder is in your PATH, or use the full path:

```bash
.venv\Scripts\pip install <package>
```

### General debugging

To see detailed logs, run the bot from the command line instead of double-clicking the `.bat` file:

```bash
.venv\Scripts\python -m facexchange --bot
```

Logs will print directly to the console, including engine status, GPU provider info, and any errors.

---

## 🤝 Join the Journey

FaceXchange is an open-source project by **Shadow Studios** — a small team passionate about making AI accessible, private, and free for everyone.

We have more tools on the horizon — face-related and beyond — all built with the same philosophy: local-first, private-by-default, no subscriptions.

**If you believe in this vision:**

- ⭐ [Star the repo](https://github.com/ShadowStudios1/FaceXchange) — it helps others discover the project
- 👣 [Follow Shadow Studios on GitHub](https://github.com/ShadowStudios1) — stay updated on new releases
- 🛠️ Contribute — PRs, ideas, bug reports, and feature requests are all welcome
- 💬 Spread the word — tell a friend who'd find this useful

We're building this together. Let's see how far we can go.

<a href="mailto:shadowfull321@gmail.com" style="text-decoration:none;color:inherit;">Contact</a>

---

## License

MIT — free to use, modify, and distribute commercially.

---

## Contributing

PRs welcome! Feel free to open an issue or pull request for any fix, feature, or idea.

1. Fork the repo
2. `git checkout -b feat/cool-stuff`
3. Commit your changes
4. Push and open a PR

---

## Notes

- First run downloads models (~900 MB). One-time.
- CPU mode is slow (5-15 min per 30s video). A CUDA GPU is strongly recommended.
- The worker process releases all GPU memory after each swap. No VRAM buildup.
- Audio is preserved from the original video.
- For CPU: `onnxruntime` is installed. For GPU: `onnxruntime-gpu`. The installer handles this automatically.
