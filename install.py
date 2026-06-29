"""
FaceXchange — Modern interactive installer with arrow-key menus, animations,
and a fully polished CLI experience.

Uses questionary for interactive selection and Rich for styling/progress.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Optional

import questionary
import requests
from rich import box
from rich.align import Align
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)
from rich.prompt import Prompt
from rich.rule import Rule
from rich.spinner import Spinner
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

console = Console()

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
WORK_DIR = ROOT / "workspace"

for d in [MODELS_DIR, WORK_DIR / "uploads", WORK_DIR / "outputs", WORK_DIR / "temp",
          WORK_DIR / "profiles"]:
    d.mkdir(parents=True, exist_ok=True)

MODELS = {
    "buffalo_l": {
        "url": "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip",
        "size_mb": 312,
        "desc": "Face detection & analysis model",
    },
    "inswapper_128.onnx": {
        "url": "https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx",
        "size_mb": 529,
        "desc": "Face swap neural network",
    },
    "GFPGANv1.4.onnx": {
        "url": "https://huggingface.co/neurobytemind/GFPGANv1.4.onnx/resolve/main/GFPGANv1.4.onnx",
        "size_mb": 348,
        "desc": "Face enhancement (GFPGAN)",
    },
}

PRESETS = [
    ("cpu", "CPU Safe", "No GPU required — works on any system. Slow but reliable. No enhancement.", 0),
    ("basic", "Basic GPU", "2+ GB VRAM — entry level GPU. No enhancement, decent speed.", 2),
    ("standard", "Standard", "4+ GB VRAM — balanced quality & speed with GFPGAN enhancement. ⭐", 4),
    ("high", "High Quality", "8+ GB VRAM — higher quality enhancement with finer detection.", 8),
    ("maximum", "Maximum", "12+ GB VRAM — frame-by-frame detection, maximum quality.", 12),
]


def _clear():
    os.system("cls" if os.name == "nt" else "clear")


def _header():
    _clear()
    title = """
[bold cyan]
   ███████╗ █████╗  ██████╗███████╗██╗  ██╗ ██████╗██╗  ██╗ █████╗ ███╗   ██╗ ██████╗ ███████╗
   ██╔════╝██╔══██╗██╔════╝██╔════╝╚██╗██╔╝██╔════╝██║  ██║██╔══██╗████╗  ██║██╔════╝ ██╔════╝
   █████╗  ███████║██║     █████╗   ╚███╔╝ ██║     ███████║███████║██╔██╗ ██║██║  ███╗█████╗
   ██╔══╝  ██╔══██║██║     ██╔══╝   ██╔██╗ ██║     ██╔══██║██╔══██║██║╚██╗██║██║   ██║██╔══╝
   ██║     ██║  ██║╚██████╗███████╗██╔╝ ██╗╚██████╗██║  ██║██║  ██║██║ ╚████║╚██████╔╝███████╗
   ╚═╝     ╚═╝  ╚═╝ ╚═════╝╚══════╝╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝ ╚═════╝ ╚══════╝
[/bold cyan]
[bold yellow]              Open-Source Face Swap — Local . Private . Free . Unlimited[/bold yellow]
"""
    console.print(Panel(Align(title, align="center"), style="cyan", box=box.DOUBLE_EDGE))


def _step_box(step: int, total: int, title: str) -> Panel:
    return Panel(
        f"[bold cyan]Step {step}/{total}[/bold cyan]  [white]{title}[/white]",
        style="cyan",
        box=box.ROUNDED,
        padding=(0, 2),
    )


def _spinner_task(text: str) -> None:
    with console.status(f"[cyan]{text}[/cyan]", spinner="dots"):
        pass


def _success_box(text: str) -> Panel:
    return Panel(f"[bold green]✅  {text}[/bold green]", style="green", box=box.ROUNDED, padding=(1, 2))


def _error_box(text: str) -> Panel:
    return Panel(f"[bold red]❌  {text}[/bold red]", style="red", box=box.ROUNDED, padding=(1, 2))


def _info_box(text: str) -> Panel:
    return Panel(f"[cyan]{text}[/cyan]", style="cyan", box=box.ROUNDED, padding=(1, 2))


def _parse_cuda_version() -> str:
    """Parse CUDA version from nvidia-smi header line e.g. 'CUDA Version: 12.8'."""
    try:
        r = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=5)
        for line in r.stdout.split("\n"):
            if "CUDA Version" in line:
                # "... CUDA Version: 12.8     |"
                m = __import__("re").search(r"CUDA Version:\s*([\d.]+)", line)
                if m:
                    return m.group(1)
    except Exception:
        pass
    return ""


def detect_gpu() -> tuple[bool, float, str]:
    panel = _step_box(1, 6, "Hardware Detection")
    console.print(panel)

    console.print("\n[bold]🔍 Scanning your system…[/bold]")
    with console.status("[cyan]Detecting GPU…[/cyan]", spinner="dots12"):
        time.sleep(1.5)
        try:
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            if r.returncode == 0:
                for line in r.stdout.strip().split("\n"):
                    try:
                        vram_mb = int(line.strip())
                        vram_gb = vram_mb / 1024
                        gpu_name = ""
                        try:
                            r2 = subprocess.run(
                                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader,nounits"],
                                capture_output=True, text=True, timeout=3,
                            )
                            if r2.returncode == 0:
                                gpu_name = r2.stdout.strip()
                        except Exception:
                            pass
                        cuda_ver = _parse_cuda_version()
                        name_str = f" — {gpu_name}" if gpu_name else ""
                        cuda_str = f"  |  CUDA: [bold cyan]{cuda_ver}[/bold cyan]" if cuda_ver else ""
                        console.print(_success_box(f"GPU detected{name_str}  |  VRAM: [bold cyan]{vram_gb:.1f} GB[/bold cyan]{cuda_str}"))
                        return True, vram_gb, cuda_ver
                    except ValueError:
                        continue
        except Exception:
            pass
        console.print(_info_box("No NVIDIA GPU detected  |  Will use [bold yellow]CPU mode[/bold yellow] (slow)"))
        return False, 0, ""


def choose_interface() -> str:
    panel = _step_box(2, 6, "Choose Interface")
    console.print(panel)

    console.print("\n[bold]🎯 How do you want to use FaceXchange?[/bold]")
    console.print("")

    choice = questionary.select(
        "",
        choices=[
            questionary.Choice(
                title="🤖  Telegram Bot + 🌐 Web UI  (Recommended)",
                value="both",
            ),
            questionary.Choice(
                title="🤖  Telegram Bot only  —  use on mobile, queue management",
                value="telegram",
            ),
            questionary.Choice(
                title="🌐  Web UI only  —  simple local browser interface",
                value="web",
            ),
        ],
        style=questionary.Style([
            ("question", "bold cyan"),
            ("answer", "bold yellow"),
            ("pointer", "bold cyan"),
            ("highlighted", "bold yellow"),
            ("selected", "bold green"),
        ]),
        qmark="",
        pointer="▶",
    ).ask()

    if choice == "both":
        console.print(_success_box("Telegram Bot + Web UI  —  Best of both worlds!"))
    elif choice == "telegram":
        console.print(_success_box("Telegram Bot only  —  Great for mobile use"))
    else:
        console.print(_success_box("Web UI only  —  Simple and clean"))

    return choice


def choose_preset(rec_idx: int) -> tuple:
    panel = _step_box(3, 6, "Choose Quality Preset")
    console.print(panel)

    table = Table(box=box.SIMPLE, border_style="cyan")
    table.add_column("#", style="dim", width=3)
    table.add_column("Preset", style="cyan", width=16)
    table.add_column("VRAM", justify="center", width=8)
    table.add_column("Description")
    for i, (pid, pname, pdesc, pvram) in enumerate(PRESETS, 1):
        marker = " ⭐" if i - 1 == rec_idx else ""
        vram_str = f"{pvram} GB" if pvram > 0 else "None"
        table.add_row(str(i), f"{pname}{marker}", vram_str, pdesc)

    console.print(table)
    console.print("\n[bold]🎮 Select your quality preset:[/bold]")

    choices = []
    for i, (pid, pname, pdesc, pvram) in enumerate(PRESETS):
        marker = " ⭐" if i == rec_idx else ""
        choices.append(
            questionary.Choice(
                title=f"{pname}{marker}  —  {pdesc}",
                value=pid,
            )
        )

    selected = questionary.select(
        "",
        choices=choices,
        default=choices[rec_idx],
        style=questionary.Style([
            ("question", "bold cyan"),
            ("pointer", "bold cyan"),
            ("highlighted", "bold yellow"),
            ("selected", "bold green"),
        ]),
        qmark="",
        pointer="▶",
    ).ask()

    for pid, pname, pdesc, pvram in PRESETS:
        if pid == selected:
            console.print(_success_box(f"[bold cyan]{pname}[/bold cyan] selected"))
            return (pid, pname, pdesc, pvram)

    return PRESETS[0]


def download_models(needs_enhancer: bool) -> bool:
    panel = _step_box(4, 6, "Download Models")
    console.print(panel)

    console.print("\n[bold]📦 Downloading models…[/bold]")
    console.print("[dim]This is a one-time download (~900 MB). Grab a coffee! ☕[/dim]\n")

    models_to_download = ["buffalo_l", "inswapper_128.onnx"]
    if needs_enhancer:
        models_to_download.append("GFPGANv1.4.onnx")

    all_ok = True

    for name in models_to_download:
        info = MODELS[name]

        if name == "buffalo_l":
            insightface_dir = Path.home() / ".insightface" / "models"
            model_dir = insightface_dir / "buffalo_l"
            if model_dir.exists() and list(model_dir.glob("*.onnx")):
                console.print(f"  [green]✅  {name}[/green]  — already installed  [dim][{info['desc']}][/dim]")
                continue
        else:
            dest = MODELS_DIR / name
            if dest.exists():
                console.print(f"  [green]✅  {name}[/green]  — already exists  [dim][{info['desc']}][/dim]")
                continue

        # Need to download
        if name == "buffalo_l":
            zip_path = MODELS_DIR / "buffalo_l.zip"
            console.print(f"\n  [cyan]📥  {name}[/cyan]  — {info['desc']}")

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(bar_width=None),
                DownloadColumn(),
                TransferSpeedColumn(),
                TimeRemainingColumn(),
            ) as progress:
                try:
                    r = requests.get(info["url"], stream=True, timeout=30)
                    r.raise_for_status()
                    total = int(r.headers.get("content-length", 0))
                    task = progress.add_task(f"  [cyan]{name}[/cyan]", total=total)
                    with open(zip_path, "wb") as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)
                            progress.update(task, advance=len(chunk))
                except Exception as e:
                    console.print(f"\n  [red]✗  Download failed: {e}[/red]")
                    all_ok = False
                    continue

            with console.status("[cyan]Extracting…[/cyan]", spinner="dots"):
                model_dir.mkdir(parents=True, exist_ok=True)
                try:
                    with zipfile.ZipFile(zip_path) as zf:
                        zf.extractall(model_dir)
                    zip_path.unlink()
                    console.print(f"  [green]✅  {name}[/green]  — installed")
                except Exception as e:
                    console.print(f"  [red]✗  Extract failed: {e}[/red]")
                    all_ok = False

        else:
            dest = MODELS_DIR / name
            console.print(f"\n  [cyan]📥  {name}[/cyan]  — {info['desc']}")

            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(bar_width=None),
                DownloadColumn(),
                TransferSpeedColumn(),
                TimeRemainingColumn(),
            ) as progress:
                try:
                    r = requests.get(info["url"], stream=True, timeout=30)
                    r.raise_for_status()
                    total = int(r.headers.get("content-length", 0))
                    task = progress.add_task(f"  [cyan]{name}[/cyan]", total=total)
                    with open(dest, "wb") as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)
                            progress.update(task, advance=len(chunk))
                    expected_bytes = info["size_mb"] * 1024 * 1024
                    actual_bytes = dest.stat().st_size
                    if actual_bytes < expected_bytes * 0.95:
                        dest.unlink(missing_ok=True)
                        console.print(f"\n  [red]✗  {name} download incomplete ({actual_bytes}/{expected_bytes} bytes)[/red]")
                        all_ok = False
                        continue
                    console.print(f"  [green]✅  {name}[/green]  — downloaded")
                except Exception as e:
                    console.print(f"\n  [red]✗  Download failed: {e}[/red]")
                    all_ok = False

    return all_ok


def ensure_venv() -> Path:
    venv_dir = ROOT / ".venv"
    if venv_dir.exists():
        console.print(f"  [green]✅  Virtual environment found[/green]")
        return venv_dir

    console.print("\n  [cyan]🔧  Creating virtual environment…[/cyan]")
    with console.status("[cyan]Setting up…[/cyan]", spinner="dots12"):
        subprocess.run(
            [sys.executable, "-m", "venv", str(venv_dir)],
            check=True, timeout=60,
        )
    console.print(f"  [green]✅  Virtual environment created[/green]")
    return venv_dir


def _map_gpu_packages(cuda_version: str) -> tuple[Optional[str], list[str]]:
    """Return (onnxruntime_gpu_spec, nvidia_pip_packages) for detected CUDA version.

    onnxruntime-gpu 1.24.x (CUDA 12) requires the FULL CUDA 12 runtime plus cuDNN 9
    to actually load CUDAExecutionProvider. Installing only cublas + cudart is NOT
    enough — the CUDA provider DLL fails to load (missing cufft64_11.dll, cudnn, etc.)
    and onnxruntime silently falls back to CPUExecutionProvider, which is exactly the
    "GPU detected but not used" bug. We install the complete pip-distributed CUDA
    12 + cuDNN 9 stack so the CUDA provider can load.
    """
    try:
        major = int(cuda_version.split(".")[0])
    except (ValueError, IndexError):
        return None, []
    if major >= 12:
        # CUDA 12.x+ driver — install the full CUDA 12.x + cuDNN 9 runtime via pip.
        return "onnxruntime-gpu==1.24.4", [
            "nvidia-cublas-cu12",
            "nvidia-cuda-nvrtc-cu12",
            "nvidia-cuda-runtime-cu12",
            "nvidia-cudnn-cu12",
            "nvidia-cufft-cu12",
            "nvidia-curand-cu12",
            "nvidia-cusolver-cu12",
            "nvidia-cusparse-cu12",
        ]
    if major == 11:
        # CUDA 11.x — use onnxruntime-gpu 1.20 + cu11 packages (incl. cuDNN 8)
        return "onnxruntime-gpu==1.20.0", [
            "nvidia-cublas-cu11",
            "nvidia-cuda-nvrtc-cu11",
            "nvidia-cuda-runtime-cu11",
            "nvidia-cudnn-cu11",
            "nvidia-cufft-cu11",
            "nvidia-curand-cu11",
            "nvidia-cusolver-cu11",
            "nvidia-cusparse-cu11",
        ]
    return None, []


def install_packages(use_gpu: bool, cuda_version: str, venv_dir: Path) -> bool:
    panel = _step_box(5, 6, "Install Packages")
    console.print(panel)

    pip_exe = str(venv_dir / "Scripts" / "pip.exe")

    # Base packages (no onnxruntime — insightface pulls CPU version as dependency)
    base_pkgs = [
        "opencv-python", "numpy", "insightface",
        "psutil", "httpx", "python-telegram-bot", "gradio",
    ]

    if use_gpu:
        onnx_pkg, nvidia_pkgs = _map_gpu_packages(cuda_version)
        if onnx_pkg is None:
            console.print(_info_box(
                f"GPU detected but CUDA version [bold]{cuda_version}[/bold] is not supported. "
                "Falling back to CPU mode. Install a newer NVIDIA driver for GPU support."
            ))
            use_gpu = False

    runtime_label = "onnxruntime-gpu (CUDA)" if use_gpu else "onnxruntime (CPU)"

    console.print(f"\n[bold]📦 Installing Python packages…[/bold]")
    console.print(f"[dim]Runtime: [cyan]{runtime_label}[/cyan]  |  Total: {len(base_pkgs)} packages[/dim]\n")

    with console.status("[cyan]Step 1/2: Installing base packages…[/cyan]", spinner="dots12"):
        try:
            subprocess.run(
                [pip_exe, "install", "--quiet"] + base_pkgs,
                check=True, timeout=600,
            )
        except subprocess.CalledProcessError as e:
            console.print(_error_box(f"Base package install failed (exit code {e.returncode})"))
            return False
        except Exception as e:
            console.print(_error_box(f"Base package install error: {e}"))
            return False

    if use_gpu:
        with console.status("[cyan]Step 2/2: Installing GPU runtime…[/cyan]", spinner="dots12"):
            try:
                # Uninstall BOTH CPU and GPU onnxruntime first — they share the
                # `onnxruntime` namespace and conflict if both pip metadata records
                # exist. A clean slate prevents the CPU build shadowing the GPU build.
                subprocess.run(
                    [pip_exe, "uninstall", "onnxruntime", "onnxruntime-gpu", "-y"],
                    capture_output=True, timeout=60,
                )
            except Exception:
                pass

            try:
                gpu_pkgs = [onnx_pkg] + nvidia_pkgs
                subprocess.run(
                    [pip_exe, "install", "--quiet"] + gpu_pkgs,
                    check=True, timeout=600,
                )
            except subprocess.CalledProcessError as e:
                console.print(_error_box(f"GPU runtime install failed (exit {e.returncode}). Falling back to CPU."))
                subprocess.run([pip_exe, "install", "--quiet", "onnxruntime"], check=False, timeout=300)
                use_gpu = False
            except Exception as e:
                console.print(_error_box(f"GPU runtime install error: {e}. Falling back to CPU."))
                subprocess.run([pip_exe, "install", "--quiet", "onnxruntime"], check=False, timeout=300)
                use_gpu = False

        # Verify onnxruntime can ACTUALLY use CUDA — not just that the provider is
        # compiled in. get_available_providers() lists CUDA even when the CUDA/cuDNN
        # runtime DLLs are missing or when the CPU onnxruntime build shadows the GPU
        # build. We test by creating an InferenceSession with CUDAExecutionProvider
        # ONLY (no CPU fallback) on a tiny embedded ONNX model. If CUDA can't load
        # (missing DLLs, namespace conflict, old driver), this raises an exception.
        if use_gpu:
            venv_python = str(venv_dir / "Scripts" / "python.exe")
            _verify_code = (
                "import os,site,base64,onnxruntime\n"
                "for s in site.getsitepackages():\n"
                "    nd=os.path.join(s,'nvidia')\n"
                "    if os.path.isdir(nd):\n"
                "        for r,_,ds in os.walk(nd):\n"
                "            for d in ds:\n"
                "                if d=='bin':\n"
                "                    p=os.path.join(r,d)\n"
                "                    os.environ['PATH']=p+os.pathsep+os.environ.get('PATH','')\n"
                "                    if hasattr(os,'add_dll_directory'):\n"
                "                        try: os.add_dll_directory(p)\n"
                "                        except OSError: pass\n"
                "mb=base64.b64decode('CA06OgoQCgF4EgF5IghJZGVudGl0eRIEdGVzdFoPCgF4EgoKCAgBEgQKAggBYg8KAXkSCgoICAESBAoCCAFCBAoAEA0=')\n"
                "onnxruntime.InferenceSession(mb,providers=['CUDAExecutionProvider'])\n"
                "print('CUDA_OK')\n"
            )
            try:
                r = subprocess.run(
                    [venv_python, "-c", _verify_code],
                    capture_output=True, text=True, timeout=30,
                )
                cuda_ok = r.returncode == 0 and "CUDA_OK" in r.stdout
            except Exception:
                cuda_ok = False

            if not cuda_ok:
                console.print(_error_box(
                    "onnxruntime-gpu installed but the CUDA provider cannot be loaded. "
                    "This usually means a CUDA/cuDNN runtime package is missing, or the CPU "
                    "onnxruntime build is shadowing the GPU build. Repairing…"
                ))
                # Force clean reinstall of the GPU runtime + full CUDA stack
                subprocess.run(
                    [pip_exe, "uninstall", "onnxruntime", "onnxruntime-gpu", "-y"],
                    capture_output=True, timeout=60,
                )
                subprocess.run(
                    [pip_exe, "install", "--quiet", "--no-deps", onnx_pkg],
                    check=False, timeout=600,
                )
                subprocess.run(
                    [pip_exe, "install", "--quiet"] + nvidia_pkgs,
                    check=False, timeout=600,
                )
                # Re-verify
                try:
                    r = subprocess.run(
                        [venv_python, "-c", _verify_code],
                        capture_output=True, text=True, timeout=30,
                    )
                    cuda_ok = r.returncode == 0 and "CUDA_OK" in r.stdout
                except Exception:
                    cuda_ok = False
                if not cuda_ok:
                    console.print(_error_box(
                        "GPU runtime could not be activated (CUDA provider won't load). "
                        "Falling back to CPU. Update your NVIDIA driver (R525+) and re-run the installer."
                    ))
                    subprocess.run([pip_exe, "install", "--quiet", "onnxruntime"], check=False, timeout=300)
                    use_gpu = False
    else:
        with console.status("[cyan]Step 2/2: Installing CPU runtime…[/cyan]", spinner="dots12"):
            try:
                subprocess.run([pip_exe, "uninstall", "onnxruntime-gpu", "-y"], capture_output=True, timeout=30)
            except Exception:
                pass
            try:
                subprocess.run([pip_exe, "install", "--quiet", "onnxruntime>=1.20.0,<1.28.0"], check=True, timeout=300)
            except Exception as e:
                console.print(_error_box(f"CPU runtime install failed: {e}"))

    if use_gpu:
        console.print(_success_box(f"GPU runtime ready — {onnx_pkg} + CUDA runtime"))
    else:
        console.print(_success_box("CPU runtime ready"))
    return True


def setup_telegram() -> tuple[str, str]:
    panel = _step_box(6, 6, "Telegram Bot Setup")
    console.print(panel)

    from rich.text import Text as RichText
    _t = RichText()
    _t.append("🤖 Telegram Bot Setup\n\n", style="bold cyan")
    _t.append("You need two things:\n\n")
    _t.append("1. Bot Token", style="bold")
    _t.append("  from @BotFather\n", style="cyan")
    _t.append("   • Open Telegram, message @BotFather\n")
    _t.append("   • Send /newbot and follow the prompts\n")
    _t.append("   • Copy the token (looks like: ", style="dim")
    _t.append("123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11", style="bold yellow")
    _t.append(")\n\n")
    _t.append("2. Your User ID", style="bold")
    _t.append("  from @userinfobot\n", style="cyan")
    _t.append("   • Message @userinfobot on Telegram\n")
    _t.append("   • Press Start — it replies with your numeric ID\n")
    _t.append("   • Copy the number (e.g.: ", style="dim")
    _t.append("123456789", style="bold yellow")
    _t.append(")")
    console.print(Panel(_t, title="Instructions", box=box.ROUNDED, border_style="cyan"))

    token = ""
    while not token:
        token = questionary.text(
            "Paste your bot token:",
            style=questionary.Style([
                ("question", "bold cyan"),
                ("answer", "bold yellow"),
            ]),
        ).ask()
        if not token or not token.strip():
            console.print("[red]Token cannot be empty[/red]")
            token = ""
            continue
        token = token.strip()

    user_id = ""
    while not user_id:
        uid = questionary.text(
            "Paste your Telegram user ID (numbers only):",
            style=questionary.Style([
                ("question", "bold cyan"),
                ("answer", "bold yellow"),
            ]),
        ).ask()
        if uid and uid.strip().isdigit():
            user_id = uid.strip()
        else:
            console.print("[red]User ID must be numeric[/red]")

    console.print(_success_box(f"Bot [bold cyan]@{token.split(':')[0]}[/bold cyan] configured"))
    return token, user_id


def finalize(token: str, user_id: str, selected: tuple, mode: str):
    _clear()
    header = """
[bold cyan]
   ╔════════════════════════════════════════════╗
   ║        🎉  INSTALLATION COMPLETE!  🎉       ║
   ╚════════════════════════════════════════════╝
[/bold cyan]
"""
    console.print(Panel(Align(header, align="center"), style="cyan", box=box.DOUBLE_EDGE))

    # ── Write config files ──
    env_lines = []
    if token:
        env_lines.append(f"FACEXCHANGE_TOKEN={token}")
    if user_id:
        env_lines.append(f"FACEXCHANGE_USER_ID={user_id}")
    is_enh = selected[0] in ("standard", "high", "maximum")
    env_lines.append(f"FACEXCHANGE_ENHANCER={'gfpgan_1.4' if is_enh else 'none'}")
    env_lines.append("FACEXCHANGE_QUALITY=85")
    env_lines.append("FACEXCHANGE_SCALE=1.0")
    (ROOT / ".env").write_text("\n".join(env_lines) + "\n")

    if mode in ("both", "telegram") and not token:
        console.print("[yellow]⚠️  No token provided — falling back to Web UI only[/yellow]")
        mode = "web"

    has_telegram = bool(token)

    venv_python = ROOT / ".venv" / "Scripts" / "python.exe"
    _venv_str = str(venv_python)

    # Helper to write a .bat file
    def _write_bat(filename: str, py_args: str, title: str):
        (ROOT / filename).write_text(
            "@echo off\n"
            f"title {title}\n"
            "cd /d \"%~dp0\"\n"
            f"if not exist \"{_venv_str}\" (\n"
            "    echo [ERROR] Virtual environment not found.\n"
            "    echo Please run installer.bat first.\n"
            "    pause\n"
            "    exit /b 1\n"
            ")\n"
            f"\"{_venv_str}\" -m facexchange {py_args}\n"
            "if %ERRORLEVEL% neq 0 (\n"
            "    echo.\n"
            "    echo Exited with error code %ERRORLEVEL%.\n"
            "    pause\n"
            ")\n"
        )

    # Always create run_web.bat
    _write_bat("run_web.bat", "--web", "FaceXchange Web UI")

    if has_telegram:
        # Telegram bot launcher
        _write_bat("run_telegram.bat", "--bot", "FaceXchange Telegram Bot")
        # Main run.bat tries Telegram first
        (ROOT / "run.bat").write_text(
            "@echo off\n"
            "title FaceXchange\n"
            "cd /d \"%~dp0\"\n"
            "echo [..] Starting Telegram Bot...\n"
            "echo.\n"
            "call run_telegram.bat\n"
        )

        # For "both" mode: also start web after telegram exits? No — keep them separate.
        # run.bat runs Telegram, user can also run run_web.bat separately.
        mode_desc = "🤖 Telegram Bot"
        if mode == "both":
            mode_desc = "🤖 Telegram Bot  +  🌐 Web UI (run_web.bat)"
    else:
        # No Telegram — web UI is the default
        _write_bat("run_telegram.bat", "--web", "FaceXchange")
        (ROOT / "run.bat").write_text(
            "@echo off\n"
            "title FaceXchange\n"
            "cd /d \"%~dp0\"\n"
            "echo [..] Starting Web UI...\n"
            "echo.\n"
            "call run_web.bat\n"
        )
        mode_desc = "🌐 Web UI"

    if user_id:
        config_py = ROOT / "facexchange" / "config.py"
        config_text = config_py.read_text()
        old = "ALLOWED_USERS: set[int] = set()"
        new = f"ALLOWED_USERS: set[int] = {{{user_id}}}"
        config_text = config_text.replace(old, new)
        config_py.write_text(config_text)

    # ── Summary table ──
    summary = Table(box=box.ROUNDED, border_style="green")
    summary.add_column("Setting", style="cyan", width=20)
    summary.add_column("Value", style="white")
    summary.add_row("Interface", mode_desc)
    summary.add_row("Preset", f"[bold yellow]{selected[1]}[/bold yellow]")
    summary.add_row("GPU", "[green]Enabled[/green]" if selected[3] > 0 else "[yellow]CPU only[/yellow]")
    summary.add_row("Enhancer", "[green]GFPGAN[/green]" if selected[0] in ("standard", "high", "maximum") else "[dim]None[/dim]")
    if token:
        summary.add_row("Bot", f"[bold cyan]@{token.split(':')[0]}[/bold cyan]")
    if user_id:
        summary.add_row("User ID", user_id)
    summary.add_row("Models", f"[green]{MODELS_DIR}[/green]")

    console.print(summary)
    console.print()

    # ── Next steps ──
    if has_telegram:
        steps_content = (
            "[bold cyan]🚀 Next Steps[/bold cyan]\n\n"
            f"[bold]1.[/bold] Double-click [bold yellow]run.bat[/bold yellow] to start the Telegram bot\n"
            f"   Or double-click [bold]run_web.bat[/bold] for the browser interface\n\n"
            f"[bold]2.[/bold] Open Telegram and send /start to your bot\n\n"
            f"[bold]3.[/bold] Send a photo — then a video — watch the magic! 🎩✨"
        )
    else:
        steps_content = (
            "[bold cyan]🚀 Next Steps[/bold cyan]\n\n"
            f"[bold]1.[/bold] Double-click [bold yellow]run.bat[/bold yellow] or [bold]run_web.bat[/bold] to start\n\n"
            f"[bold]2.[/bold] Open http://127.0.0.1:7860 in your browser\n\n"
            f"[bold]3.[/bold] Upload a photo — then a video — watch the magic! 🎩✨"
        )
    steps = Panel(steps_content, title="▶️  You're All Set!", box=box.ROUNDED, border_style="yellow")
    console.print(steps)
    console.print()

    # ── Launch prompt — prefer Telegram ──
    if has_telegram:
        launch_desc = "Launch Telegram bot now?"
        launch_cmd = ["--bot"]
    else:
        launch_desc = "Launch Web UI now?"
        launch_cmd = ["--web"]
    if questionary.confirm(
        launch_desc,
        default=True,
        style=questionary.Style([
            ("question", "bold cyan"),
            ("answer", "bold yellow"),
        ]),
    ).ask():
        console.print("\n[cyan]Starting…[/cyan]")
        time.sleep(1)
        os.chdir(ROOT)
        subprocess.run([str(venv_python), "-m", "facexchange"] + launch_cmd)


def main():
    _header()
    time.sleep(1)

    # Step 1: Detect GPU
    has_gpu, vram_gb, cuda_ver = detect_gpu()
    time.sleep(0.8)

    # cuda_ver is used later by install_packages; also helps user know their CUDA version
    _cuda_ver = cuda_ver  # keep for later

    if has_gpu:
        if vram_gb >= 12:
            rec_idx = 4
        elif vram_gb >= 8:
            rec_idx = 3
        elif vram_gb >= 4:
            rec_idx = 2
        elif vram_gb >= 2:
            rec_idx = 1
        else:
            rec_idx = 0
    else:
        rec_idx = 0

    time.sleep(0.5)

    # Step 2: Choose interface
    mode = choose_interface()
    time.sleep(0.5)

    # Step 3: Choose preset
    selected = choose_preset(rec_idx)
    time.sleep(0.5)

    # Step 4: Download models
    needs_enhancer = selected[0] in ("standard", "high", "maximum")
    models_ok = download_models(needs_enhancer)
    time.sleep(0.5)

    # Step 5: Install packages
    is_cpu = selected[0] == "cpu"
    venv_dir = ensure_venv()
    pkgs_ok = install_packages(has_gpu and not is_cpu, cuda_ver, venv_dir)

    if not pkgs_ok and not questionary.confirm(
        "Package install had issues. Continue anyway?",
        default=False,
        style=questionary.Style([
            ("question", "bold yellow"),
            ("answer", "bold yellow"),
        ]),
    ).ask():
        console.print(_error_box("Installation cancelled."))
        return

    # Step 6: Telegram setup (if needed)
    token, user_id = "", ""
    if mode in ("both", "telegram"):
        time.sleep(0.5)
        token, user_id = setup_telegram()
        time.sleep(0.5)

    # Done — show summary
    finalize(token, user_id, selected, mode)


if __name__ == "__main__":
    main()
