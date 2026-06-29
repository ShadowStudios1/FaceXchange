"""
Telegram bot — modern inline-menu flow with queue, progress, per-video status.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Optional

from telegram import Bot, InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    ConversationHandler,
    MessageHandler,
    filters,
)
from telegram.request import HTTPXRequest

_env_path = Path(__file__).resolve().parent.parent / ".env"
if _env_path.exists():
    for _line in _env_path.read_text("utf-8").splitlines():
        _line = _line.strip()
        if _line and "=" in _line and not _line.startswith("#"):
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

import subprocess

from . import config
from .presets import ENHANCER_OPTIONS, QUALITY_LEVELS, SCALE_OPTIONS, make_preset
from .worker import run_worker

log = logging.getLogger("facexchange.bot")

MENU, ACTIVE = range(2)

_swap_data: dict[int, dict] = {}
_src_refs: dict[str, int] = {}
_bg_bot: Optional[Bot] = None
_queue_locks: dict[int, asyncio.Lock] = {}


def _progress_bar(pct: int, width: int = 22) -> str:
    filled = max(0, min(width, pct * width // 100))
    return f"{'█' * filled}{'░' * (width - filled)}"


def _gpu_info() -> str:
    # First, report what onnxruntime can actually use (the truth).
    ort_provider = ""
    try:
        import onnxruntime as _ort
        avail = _ort.get_available_providers()
        ort_provider = "GPU (CUDA)" if "CUDAExecutionProvider" in avail else "CPU"
    except Exception:
        ort_provider = "unknown"

    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.total,memory.used", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL, text=True, timeout=5,
        ).strip()
        for line in out.split("\n"):
            parts = [x.strip() for x in line.split(",")]
            if len(parts) == 3:
                total, used = int(parts[1]), int(parts[2])
                free = total - used
                return f"{ort_provider} · {used} MB used / {total} MB total ({free} MB free)"
        return f"{ort_provider} · GPU info unavailable"
    except Exception:
        if ort_provider == "GPU (CUDA)":
            return f"{ort_provider} · (nvidia-smi unavailable)"
        return "CPU mode (no NVIDIA GPU)"


def _profile_path(uid: int) -> Path:
    p = config.WORK_DIR / "profiles"
    p.mkdir(parents=True, exist_ok=True)
    return p / f"{uid}.json"


def _load_profile(uid: int) -> dict:
    f = _profile_path(uid)
    if f.exists():
        try:
            data = json.loads(f.read_text())
            if "preset" in data and "enhancer" not in data:
                from .presets import get_preset
                pr = get_preset(data["preset"])
                if pr:
                    data["enhancer"] = pr.enhancer
                    data["quality"] = pr.quality
                    data["scale"] = 1.0
                del data["preset"]
                _profile_path(uid).write_text(json.dumps(data))
            return data
        except Exception:
            pass
    return {"enhancer": "gfpgan_1.4", "quality": 85, "scale": 1.0}


def _save_profile(uid: int, data: dict):
    data.pop("preset", None)
    _profile_path(uid).write_text(json.dumps(data))


def _profile_summary(uid: int) -> str:
    p = _load_profile(uid)
    enh = ENHANCER_OPTIONS.get(p.get("enhancer", "gfpgan_1.4"), "Unknown")
    quality = p.get("quality", 85)
    scale = p.get("scale", 1.0)
    s = scale
    return f"Enhancer: {enh} · Quality: {quality}% · Scale: {s}x"


def _main_menu_text(uid: int) -> str:
    return (
        "🎭 *FaceXchange — Face Swap*\n\n"
        "Send a *photo* to set a face, then send *videos* to swap.\n"
        "Multiple videos queue up automatically.\n\n"
        f"📋 {_profile_summary(uid)}\n\n"
        "👇 Use the buttons below or type a command."
    )


def _main_menu_kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🔄 Swap Face", callback_data="swap")],
        [InlineKeyboardButton("⚙️ Settings", callback_data="settings"),
         InlineKeyboardButton("📊 Status", callback_data="status")],
        [InlineKeyboardButton("❓ Help", callback_data="help")],
    ])


def _settings_text(uid: int) -> str:
    return f"⚙️ *Settings*\n\nCurrent:\n{_profile_summary(uid)}"


def _settings_kb(uid: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🧑‍🎤 Enhancer", callback_data="set_enhancer"),
         InlineKeyboardButton("🎯 Quality", callback_data="set_quality")],
        [InlineKeyboardButton("🔍 Scale", callback_data="set_scale")],
        [InlineKeyboardButton("◀️ Back", callback_data="menu_back")],
    ])


def _enhancer_text(uid: int) -> str:
    p = _load_profile(uid)
    current = p.get("enhancer", "gfpgan_1.4")
    lines = ["🧑‍🎤 *Enhancer*\n\nChoose face enhancement:\n"]
    for key, label in ENHANCER_OPTIONS.items():
        marker = "●" if key == current else "○"
        desc = "No restoration" if key == "none" else "AI face enhancement (GFPGAN)"
        lines.append(f"{marker} **{label}** — {desc}")
    return "\n".join(lines)


def _enhancer_kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton(f"None", callback_data="enh_none"),
         InlineKeyboardButton(f"GFPGAN v1.4", callback_data="enh_gfpgan_1.4")],
        [InlineKeyboardButton("◀️ Back", callback_data="settings")],
    ])


def _quality_text(uid: int) -> str:
    p = _load_profile(uid)
    current = p.get("quality", 85)
    lines = ["🎯 *Quality*\n\nHigher = better faces, slower speed:\n"]
    for q in QUALITY_LEVELS:
        marker = "●" if q == current else "○"
        if q <= 60:
            desc = "fastest, lowest detail"
        elif q <= 70:
            desc = "fast, basic detail"
        elif q == 80:
            desc = "balanced"
        elif q == 85:
            desc = "balanced (recommended)"
        elif q == 92:
            desc = "high quality"
        else:
            desc = "maximum quality, slowest"
        lines.append(f"{marker} **{q}%** — {desc}")
    return "\n".join(lines)


def _quality_kb() -> InlineKeyboardMarkup:
    btns = [[InlineKeyboardButton(f"{q}%", callback_data=f"qual_{q}")] for q in QUALITY_LEVELS]
    btns.append([InlineKeyboardButton("◀️ Back", callback_data="settings")])
    return InlineKeyboardMarkup(btns)


def _scale_text(uid: int) -> str:
    p = _load_profile(uid)
    current = p.get("scale", 1.0)
    lines = ["🔍 *Scale*\n\nOutput video size multiplier:\n"]
    for s in SCALE_OPTIONS:
        marker = "●" if s == current else "○"
        if s == 1.0:
            desc = "original resolution"
        elif s == 1.5:
            desc = "50% larger (upscaled)"
        else:
            desc = "double resolution (upscaled)"
        s_str = f"{s:.1f}x" if s != int(s) else f"{int(s)}x"
        lines.append(f"{marker} **{s_str}** — {desc}")
    return "\n".join(lines)


def _scale_kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("1x", callback_data="scale_1.0"),
         InlineKeyboardButton("1.5x", callback_data="scale_1.5"),
         InlineKeyboardButton("2x", callback_data="scale_2.0")],
        [InlineKeyboardButton("◀️ Back", callback_data="settings")],
    ])


def _item_status(sd: dict, item_id: str) -> str:
    queue = sd.get("queue", [])
    processing = sd.get("processing", False)
    idx = None
    for i, it in enumerate(queue):
        if it.get("id") == item_id:
            idx = i
            break
    if idx is None:
        return "✅ This video was already processed or removed."
    total = len(queue)
    position = idx + 1
    if idx == 0 and processing:
        prog = queue[0].get("progress", {})
        if not prog or not prog.get("msg"):
            return "🎬 *Starting your video…*"
        pct = prog.get("pct", 0)
        msg = prog.get("msg", "")
        elapsed = prog.get("elapsed", 0)
        lines = [
            f"🎬 *Processing your video*",
            f"{_progress_bar(pct)}  {pct}%",
        ]
        if msg:
            lines.append(f"📦 {msg}")
        if elapsed:
            remaining = int(elapsed * (100 - pct) / max(pct, 1)) if pct > 0 else 0
            t = f"⏱ {elapsed:.0f}s"
            if remaining > 0:
                t += f" · ⏳ ~{remaining}s"
            lines.append(t)
        return "\n".join(lines)
    ahead = position - 1
    return (
        f"⏳ *Waiting in Queue*\n\n"
        f"Your video is **#{position}** of **{total}** in queue.\n"
        f"{ahead} video(s) ahead of you.\n\n"
        f"Please wait…"
    )


def _item_kb(item_id: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🔄 Refresh", callback_data=f"q_r_{item_id}"),
         InlineKeyboardButton("❌ Cancel", callback_data=f"q_c_{item_id}")]
    ])


async def _update_item_msg(bot: Bot, sd: dict, item: dict):
    text = _item_status(sd, item["id"])
    kb = _item_kb(item["id"])
    try:
        await bot.edit_message_text(
            chat_id=item["chat_id"], message_id=item["msg_id"],
            text=text, reply_markup=kb, parse_mode="Markdown",
        )
    except Exception:
        pass


async def _send(update: Update, text: str, kb: Optional[InlineKeyboardMarkup] = None, edit: bool = False):
    if edit and update.callback_query:
        try:
            await update.callback_query.edit_message_text(
                text=text, reply_markup=kb, parse_mode="Markdown",
            )
        except Exception:
            pass
    elif update.message:
        await update.message.reply_text(text=text, reply_markup=kb, parse_mode="Markdown")
    elif update.callback_query:
        await update.callback_query.message.reply_text(text=text, reply_markup=kb, parse_mode="Markdown")


def _auth(update: Update) -> bool:
    uid = update.effective_user.id
    if config.ALLOWED_USERS and uid not in config.ALLOWED_USERS:
        return False
    return True


def _release_src(src_path: str, uid: int):
    global _src_refs
    if src_path in _src_refs:
        _src_refs[src_path] -= 1
        if _src_refs[src_path] <= 0:
            del _src_refs[src_path]
            try:
                Path(src_path).unlink(missing_ok=True)
            except Exception:
                pass


async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not _auth(update):
        await update.message.reply_text("⛔ You are not authorized.")
        return
    uid = update.effective_user.id
    _swap_data.setdefault(uid, {"current_src": None, "current_src_id": None, "queue": [], "processing": False})
    await _send(update, _main_menu_text(uid), _main_menu_kb())
    return MENU


async def cmd_swap(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not _auth(update):
        return
    await update.message.reply_text("📸 Send a **clear front-facing photo** of the person whose face you want to swap in.", parse_mode="Markdown")
    return MENU


async def cmd_settings(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not _auth(update):
        return
    uid = update.effective_user.id
    await _send(update, _settings_text(uid), _settings_kb(uid))


async def cmd_help(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not _auth(update):
        return
    await update.message.reply_text(
        "❓ *Help*\n\n"
        "📸 Send a *photo* — sets the face to swap\n"
        "🎥 Send a *video* — queues it for face swap\n"
        "📸 Send another *photo* — changes source face\n\n"
        "Multiple videos queue up automatically.\n"
        "Only one processes at a time.\n\n"
        "🔄 */swap* — start fresh\n"
        "⚙️ */settings* — Enhancer, Quality, Scale\n"
        "📊 */status* — GPU and queue info\n"
        "❌ */cancel* — clear queue and reset",
        parse_mode="Markdown",
    )


async def cmd_status(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not _auth(update):
        return
    uid = update.effective_user.id
    gpu = _gpu_info()
    sd = _swap_data.get(uid, {})
    qlen = len(sd.get("queue", []))
    processing = sd.get("processing", False)
    summary = _profile_summary(uid)
    await update.message.reply_text(
        f"📊 *Status*\n\n"
        f"GPU: `{gpu}`\n"
        f"Processing: {'🟢' if processing else '⚪'} | Queue: {qlen}\n"
        f"📋 {summary}\n\n"
        f"Limits: ≤{config.MAX_VIDEO_DURATION}s · ≤{config.MAX_VIDEO_WIDTH}x{config.MAX_VIDEO_HEIGHT}",
        parse_mode="Markdown",
    )


async def cmd_cancel(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not _auth(update):
        return
    uid = update.effective_user.id
    sd = _swap_data.pop(uid, None)
    if sd:
        src = sd.get("current_src")
        if src:
            _release_src(src, uid)
        for it in sd.get("queue", []):
            try:
                Path(it["tgt"]).unlink(missing_ok=True)
            except Exception:
                pass
            if it.get("src"):
                _release_src(it["src"], uid)
    await update.message.reply_text("❌ Cancelled. Use /start to see the menu.")


async def photo_entry_handler(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not _auth(update):
        return
    uid = update.effective_user.id
    sd = _swap_data.setdefault(uid, {"current_src": None, "current_src_id": None, "queue": [], "processing": False})

    file = update.message.photo[-1]
    file_id = file.file_id
    src_path = str(config.UPLOAD_DIR / f"src_{uid}_{uuid.uuid4().hex}.jpg")

    await update.message.reply_text("📸 Downloading photo…")
    tg_file = await file.get_file()
    await tg_file.download_to_drive(src_path)

    old_src = sd.get("current_src")
    old_src_id = sd.get("current_src_id")
    if old_src and old_src_id and old_src_id != file_id:
        _release_src(old_src, uid)
    if not sd.get("current_src_id") or sd["current_src_id"] != file_id:
        _src_refs.setdefault(src_path, 0)
        _src_refs[src_path] += len(sd.get("queue", [])) + 1

    sd["current_src"] = src_path
    sd["current_src_id"] = file_id

    await update.message.reply_text(
        "✅ *Source face set!* Now send a *video* to swap into.\n"
        "Send another photo to change the source.\n\n"
        "❌ `/cancel` to abort.",
        parse_mode="Markdown",
    )
    return ACTIVE


async def active_photo_handler(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not _auth(update):
        return
    uid = update.effective_user.id
    sd = _swap_data.get(uid)
    if not sd:
        await update.message.reply_text("Session expired. Use /start.")
        return

    file = update.message.photo[-1]
    file_id = file.file_id
    src_path = str(config.UPLOAD_DIR / f"src_{uid}_{uuid.uuid4().hex}.jpg")

    await update.message.reply_text("📸 Downloading photo…")
    tg_file = await file.get_file()
    await tg_file.download_to_drive(src_path)

    old_src = sd.get("current_src")
    old_src_id = sd.get("current_src_id")
    if old_src and old_src_id and old_src_id != file_id:
        _release_src(old_src, uid)
    _src_refs.setdefault(src_path, 0)
    _src_refs[src_path] += len(sd.get("queue", [])) + 1

    sd["current_src"] = src_path
    sd["current_src_id"] = file_id

    await update.message.reply_text("✅ *Source face updated!* Send a video to swap.", parse_mode="Markdown")


async def active_video_handler(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not _auth(update):
        return
    uid = update.effective_user.id
    sd = _swap_data.get(uid)
    if not sd or not sd.get("current_src"):
        await update.message.reply_text("❌ No source face set. Send a photo first.")
        return

    await update.message.reply_text("⬇️ Downloading video…")
    file = update.message.video
    file_id = file.file_id
    tgt_path = str(config.UPLOAD_DIR / f"tgt_{uid}_{uuid.uuid4().hex}.mp4")
    tg_file = await file.get_file()
    await tg_file.download_to_drive(tgt_path)

    item_id = uuid.uuid4().hex[:8]
    msg = await update.message.reply_text("📥 Added to queue…")

    item = {
        "id": item_id,
        "src": sd["current_src"],
        "tgt": tgt_path,
        "chat_id": update.effective_chat.id,
        "msg_id": msg.message_id,
        "progress": {},
        "_done": False,
    }
    sd.setdefault("queue", []).append(item)
    _src_refs[sd["current_src"]] = _src_refs.get(sd["current_src"], 0) + 1

    await _update_item_msg(ctx.bot, sd, item)

    total = len(sd["queue"])
    if total == 1:
        await update.message.reply_text("📥 Video queued — processing now…")
    else:
        await update.message.reply_text(f"📥 Video queued — **#{total}** of **{total}**. Please wait.", parse_mode="Markdown")

    asyncio.create_task(_process_queue(uid))


async def menu_handler(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not _auth(update):
        return
    query = update.callback_query
    await query.answer()
    data = query.data
    uid = update.effective_user.id

    if data == "swap":
        await query.message.reply_text("📸 Send a **clear front-facing photo** to start.", parse_mode="Markdown")
        return ACTIVE

    elif data == "settings":
        await _send(update, _settings_text(uid), _settings_kb(uid), edit=True)
        return MENU

    elif data == "set_enhancer":
        await _send(update, _enhancer_text(uid), _enhancer_kb(), edit=True)
        return MENU

    elif data == "set_quality":
        await _send(update, _quality_text(uid), _quality_kb(), edit=True)
        return MENU

    elif data == "set_scale":
        await _send(update, _scale_text(uid), _scale_kb(), edit=True)
        return MENU

    elif data.startswith("enh_"):
        enh_key = data[4:]
        if enh_key in ENHANCER_OPTIONS:
            p = _load_profile(uid)
            p["enhancer"] = enh_key
            _save_profile(uid, p)
        await _send(update, _settings_text(uid), _settings_kb(uid), edit=True)
        return MENU

    elif data.startswith("qual_"):
        try:
            q = int(data[5:])
            if q in QUALITY_LEVELS:
                p = _load_profile(uid)
                p["quality"] = q
                _save_profile(uid, p)
        except (ValueError, TypeError):
            pass
        await _send(update, _settings_text(uid), _settings_kb(uid), edit=True)
        return MENU

    elif data.startswith("scale_"):
        try:
            s = float(data[6:])
            if s in SCALE_OPTIONS:
                p = _load_profile(uid)
                p["scale"] = s
                _save_profile(uid, p)
        except (ValueError, TypeError):
            pass
        await _send(update, _settings_text(uid), _settings_kb(uid), edit=True)
        return MENU

    elif data == "status":
        gpu = _gpu_info()
        sd = _swap_data.get(uid, {})
        qlen = len(sd.get("queue", []))
        processing = sd.get("processing", False)
        await query.edit_message_text(
            text=(
                f"📊 *Status*\n\n"
                f"GPU: `{gpu}`\n"
                f"Processing: {'🟢' if processing else '⚪'} | Queue: {qlen}\n"
                f"📋 {_profile_summary(uid)}"
            ),
            reply_markup=_main_menu_kb(),
            parse_mode="Markdown",
        )
        return MENU

    elif data == "help":
        await query.edit_message_text(
            text=(
                "❓ *Help*\n\n"
                "📸 Send a *photo* → sets the source face\n"
                "🎥 Send a *video* → queues it for swapping\n"
                "📸 Send another *photo* → changes source\n\n"
                "Multiple videos queue up automatically.\n"
                "Only one processes at a time.\n\n"
                "⚙️ *Settings* — Enhancer, Quality, Scale\n"
                "📊 *Status* — GPU, queue, profile\n"
                "❌ `/cancel` — clear all"
            ),
            reply_markup=_main_menu_kb(),
            parse_mode="Markdown",
        )
        return MENU

    elif data == "menu_back":
        await _send(update, _main_menu_text(uid), _main_menu_kb(), edit=True)
        return MENU

    return MENU


async def queue_item_callback(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not _auth(update):
        return
    query = update.callback_query
    await query.answer()
    data = query.data
    uid = update.effective_user.id
    sd = _swap_data.get(uid)
    if not sd:
        await query.edit_message_text(text="Session expired.")
        return

    prefix = data[:4]
    item_id = data[4:]

    if prefix == "q_r_":
        await _update_item_msg(ctx.bot, sd, next((it for it in sd.get("queue", []) if it["id"] == item_id), {"id": item_id}))

    elif prefix == "q_c_":
        removed = None
        new_queue = []
        for it in sd.get("queue", []):
            if it.get("id") == item_id:
                removed = it
                try:
                    Path(it["tgt"]).unlink(missing_ok=True)
                except Exception:
                    pass
                _release_src(it["src"], uid)
            else:
                new_queue.append(it)
        if removed:
            sd["queue"] = new_queue
            try:
                await query.edit_message_text(text="❌ *Cancelled* — removed from queue.", parse_mode="Markdown")
            except Exception:
                pass
            if new_queue:
                await _update_item_msg(ctx.bot, sd, new_queue[0])
        else:
            try:
                await query.edit_message_text(text="Already processed or not in queue.")
            except Exception:
                pass


async def _process_queue(uid: int):
    global _bg_bot
    bot = _bg_bot
    if not bot:
        return

    lock = _queue_locks.setdefault(uid, asyncio.Lock())
    if lock.locked():
        return  # Another loop is already processing this user's queue

    async with lock:
        sd = _swap_data.get(uid)
        if not sd:
            sd = _swap_data.setdefault(uid, {"current_src": None, "current_src_id": None, "queue": [], "processing": False})

        sd["processing"] = True
        while sd.get("queue"):
            sd = _swap_data.get(uid)
            if not sd or not sd.get("queue"):
                break

            item = sd["queue"][0]
            src = item.get("src")
            if not src:
                sd["queue"].pop(0)
                continue

            item["progress"] = {}
            item["_done"] = False
            await _update_item_msg(bot, sd, item)

            profile = _load_profile(uid)
            settings = {
                "enhancer": profile.get("enhancer", "gfpgan_1.4"),
                "quality": profile.get("quality", 85),
                "scale": profile.get("scale", 1.0),
            }
            loop = asyncio.get_running_loop()

            def _cb(prog: dict):
                nonlocal item
                item["progress"] = prog
                if prog.get("type") in ("done", "error"):
                    item["_done"] = True

            async def _poller():
                nonlocal item
                while not item.get("_done"):
                    await _update_item_msg(bot, sd, item)
                    await asyncio.sleep(1.5)

            poller_task = asyncio.create_task(_poller())
            try:
                res = await loop.run_in_executor(
                    None, run_worker, src, item["tgt"], settings, _cb,
                )
            except Exception as e:
                log.exception("Queue error")
                res = {"success": False, "message": str(e)}

            item["_done"] = True
            try:
                await poller_task
            except Exception:
                pass

            sd = _swap_data.get(uid)
            still_active = sd and sd.get("queue") and sd["queue"][0].get("id") == item.get("id")
            if still_active:
                chat_id = item.get("chat_id")
                if chat_id and res.get("success") and res.get("output_path"):
                    try:
                        with open(res["output_path"], "rb") as f:
                            elapsed = res.get("elapsed_sec", 0)
                            caption = f"✅ Done in {elapsed:.0f}s"
                            await bot.send_video(
                                chat_id=chat_id, video=f, caption=caption,
                                read_timeout=300, write_timeout=300,
                            )
                    except Exception as e:
                        log.error(f"Send result: {e}")
                    try:
                        await bot.delete_message(chat_id=chat_id, message_id=item["msg_id"])
                    except Exception:
                        pass
                elif chat_id:
                    msg = res.get("message", "Unknown error")
                    try:
                        await bot.send_message(chat_id=chat_id, text=f"❌ Swap failed:\n{msg}")
                    except Exception:
                        pass

                sd["queue"].pop(0)
                sd = _swap_data.get(uid)

        if sd:
            sd["processing"] = False


def run_bot():
    token = config.TELEGRAM_TOKEN
    if not token:
        log.warning("FACEXCHANGE_TOKEN not set — bot disabled")
        return

    req = HTTPXRequest(
        connect_timeout=30, read_timeout=300, write_timeout=300,
        connection_pool_size=1024,
    )
    app = (
        ApplicationBuilder()
        .token(token)
        .request(req)
        .concurrent_updates(True)
        .build()
    )

    bg_req = HTTPXRequest(
        connect_timeout=30, read_timeout=300, write_timeout=300,
        connection_pool_size=512,
    )
    global _bg_bot
    _bg_bot = Bot(token=token, request=bg_req)

    conv = ConversationHandler(
        entry_points=[
            CommandHandler("start", cmd_start),
            CommandHandler("menu", cmd_start),
            CommandHandler("swap", cmd_swap),
            MessageHandler(filters.PHOTO, photo_entry_handler),
        ],
        states={
            MENU: [
                CallbackQueryHandler(menu_handler),
                MessageHandler(filters.PHOTO, photo_entry_handler),
            ],
            ACTIVE: [
                MessageHandler(filters.PHOTO, active_photo_handler),
                MessageHandler(filters.VIDEO, active_video_handler),
                CallbackQueryHandler(queue_item_callback, pattern=r"^q_[rc]_"),
            ],
        },
        fallbacks=[
            CommandHandler("cancel", cmd_cancel),
            CommandHandler("help", cmd_help),
            CommandHandler("status", cmd_status),
            CommandHandler("start", cmd_start),
        ],
    )

    app.add_handler(conv)
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("status", cmd_status))

    log.info("Bot started — modern menu, queue with per-video status.")
    app.run_polling()


def main():
    logging.basicConfig(
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        level=logging.INFO,
    )
    run_bot()


if __name__ == "__main__":
    main()
