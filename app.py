"""
PÎVOK Studio — 24/7 Cloud App (Hugging Face Docker Space, Port 7860)
Unified FastAPI + Aiogram 3 Bot + WebApp Crop Editor + Embedded YOLOv8 CPU
"""

import asyncio
import base64
import hashlib
import json
import logging
import os
import shutil
import time
import urllib.parse
from contextlib import asynccontextmanager
from typing import Dict

import cv2
import numpy as np
from aiogram import BaseMiddleware, Bot, Dispatcher, F
from aiogram.filters import Command, StateFilter
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import (
    CallbackQuery,
    FSInputFile,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    InputMediaDocument,
    InputMediaPhoto,
    KeyboardButton,
    Message,
    ReplyKeyboardRemove,
    Update,
    WebAppInfo,
)
from aiogram.utils.keyboard import ReplyKeyboardBuilder
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import collage_engine
import color_clustering
import yolo_local

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("pivok-cloud")

BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
SECRET_PASSWORD = os.environ.get("BOT_SECRET_PASSWORD", "pivok").strip()
SPACE_HOST = os.environ.get("SPACE_HOST", "").strip()
RENDER_EXTERNAL_URL = os.environ.get("RENDER_EXTERNAL_URL", "").strip().rstrip("/")
PUBLIC_BASE_URL = (
    os.environ.get("PUBLIC_BASE_URL", "").strip().rstrip("/")
    or RENDER_EXTERNAL_URL
    or (f"https://{SPACE_HOST}" if SPACE_HOST else "")
)
WEBHOOK_SECRET = hashlib.sha256((BOT_TOKEN + ":pivok_wh").encode()).hexdigest()[:32] if BOT_TOKEN else "pivok"

INCOMING_DIR = collage_engine.WORK_DIR
OUTPUT_DIR = collage_engine.SAVE_DIR
STATE_FILE = os.path.join(collage_engine.WORK_ROOT, "cloud_state.json")

SIZE_LIST = [
    "36-42",
    "38-44",
    "40-46",
    "42-48",
    "44-50",
    "46-52",
    "48-54",
    "50-56",
    "1-2-3",
    "STD",
]
MAIN_MENU = ["🖼 Создать коллаж"]

PHOTO_BUFFER: Dict[int, dict] = {}
DOWNLOAD_PREFETCH: Dict[str, asyncio.Task] = {}


def _load_state_store() -> dict:
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_state_store(store: dict):
    try:
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(store, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


_STORE = _load_state_store()
_env_ids = [
    int(x.strip())
    for x in os.environ.get("ALLOWED_USER_IDS", "").split(",")
    if x.strip().isdigit()
]
ALLOWED_USERS: list[int] = list(set(_STORE.get("allowed_users", []) + _env_ids))
pending_dispatches: Dict[str, str] = _STORE.get("pending_dispatches", {})
media_msgs_store: Dict[str, list[int]] = _STORE.get("media_msgs", {})


def persist_store():
    _STORE["allowed_users"] = ALLOWED_USERS
    _STORE["pending_dispatches"] = pending_dispatches
    _STORE["media_msgs"] = media_msgs_store
    _save_state_store(_STORE)


def build_keyboard(items, columns=2):
    builder = ReplyKeyboardBuilder()
    for item in items:
        builder.add(KeyboardButton(text=item))
    builder.adjust(columns)
    return builder.as_markup(resize_keyboard=True)


def analyze_full_body(img_bytes: bytes) -> float:
    try:
        arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return 0.0
        h, w = img.shape[:2]
        if max(h, w) > 640:
            sc = 640.0 / max(h, w)
            img = cv2.resize(img, (int(w * sc), int(h * sc)))
            h, w = img.shape[:2]

        data = yolo_local.detect_yolo(img, "yolov8n-pose")
        kps_all = data.get("keypoints", [])
        if kps_all:
            kps = kps_all[0]
            if len(kps) >= 17:
                if any(5 < kps[idx][0] < w - 5 and 5 < kps[idx][1] < h - 5 for idx in (15, 16)):
                    return 2.0
                if any(5 < kps[idx][0] < w - 5 and 5 < kps[idx][1] < h - 5 for idx in (13, 14)):
                    return 0.5
                return -1.0
    except Exception:
        pass
    return 0.0


async def download_single_photo(f_id: str, bot_inst: Bot) -> dict:
    if f_id in DOWNLOAD_PREFETCH:
        try:
            return await DOWNLOAD_PREFETCH[f_id]
        except Exception:
            DOWNLOAD_PREFETCH.pop(f_id, None)

    for attempt in range(3):
        try:
            file_info = await asyncio.wait_for(bot_inst.get_file(f_id), timeout=20.0)
            ext = file_info.file_path.split(".")[-1] if "." in file_info.file_path else "jpg"
            downloaded = await asyncio.wait_for(bot_inst.download_file(file_info.file_path), timeout=45.0)
            img_bytes = downloaded.read()
            score = await asyncio.to_thread(analyze_full_body, img_bytes)
            return {"file_id": f_id, "bytes": img_bytes, "ext": ext, "score": score}
        except Exception as e:
            logger.error(f"Error downloading {f_id} (attempt {attempt + 1}): {e}")
            await asyncio.sleep(0.3)
    raise RuntimeError(f"Не удалось скачать фото {f_id}")


class SecurityMiddleware(BaseMiddleware):
    async def __call__(self, handler, event, data):
        user_id = event.from_user.id if hasattr(event, "from_user") and event.from_user else None
        if user_id and user_id not in ALLOWED_USERS:
            if hasattr(event, "text") and event.text:
                entered = event.text.strip().lower()
                if entered in {SECRET_PASSWORD.lower(), "пивок", "pivok"}:
                    ALLOWED_USERS.append(user_id)
                    persist_store()
                    await event.answer(
                        "✅ **Доступ к PÎVOK Cloud открыт!**\n\nВыберите действие в меню ниже:",
                        reply_markup=build_keyboard(MAIN_MENU, 1),
                        parse_mode="Markdown",
                    )
                    return
                await event.answer("🔒 Бот PÎVOK защищён.\nОтправьте пароль для входа:")
                return
            if hasattr(event, "data") and event.data:
                await event.answer("🔒 Доступ закрыт. Введите пароль в чат.", show_alert=True)
                return
            return
        return await handler(event, data)


class CollageForm(StatesGroup):
    photo = State()
    grouping_mode = State()
    code = State()
    size = State()
    price = State()
    confirm = State()
    edit_field = State()


bot = Bot(token=BOT_TOKEN) if BOT_TOKEN else None
dp = Dispatcher()
dp.message.outer_middleware(SecurityMiddleware())
dp.callback_query.outer_middleware(SecurityMiddleware())


def build_preview_keyboard(folder_name: str, photos: list[str], chat_id: int, short_id: str) -> InlineKeyboardMarkup:
    encoded_folder = urllib.parse.quote(folder_name)
    crop_buttons = []
    if PUBLIC_BASE_URL.startswith("https://"):
        for i, p_path in enumerate(photos):
            stem = os.path.splitext(os.path.basename(p_path))[0]
            display_name = stem.split("_", 1)[-1] if "_" in stem else stem
            crop_buttons.append(
                InlineKeyboardButton(
                    text=f"🛠 {display_name}",
                    web_app=WebAppInfo(
                        url=f"{PUBLIC_BASE_URL}/webapp/index.html?folder={encoded_folder}&chat_id={chat_id}&pair={i}"
                    ),
                )
            )

    crop_rows = [crop_buttons[i : i + 3] for i in range(0, len(crop_buttons), 3)]
    layout = [
        [InlineKeyboardButton(text="📥 Скачать файлами (без сжатия)", callback_data=f"files_{short_id}")],
    ]
    layout.extend(crop_rows)
    layout.extend(
        [
            [InlineKeyboardButton(text="🔄 Обновить превью", callback_data=f"refresh_{short_id}")],
            [InlineKeyboardButton(text="🎨 Добавить фото цвета", callback_data=f"addcolor_{short_id}")],
            [InlineKeyboardButton(text="🗑 Удалить", callback_data=f"del_{short_id}")],
        ]
    )
    return InlineKeyboardMarkup(inline_keyboard=layout)


async def send_collage_preview(chat_id: int, folder_name: str):
    if not bot:
        return
    folder_path = os.path.join(OUTPUT_DIR, folder_name)
    if not os.path.isdir(folder_path):
        await bot.send_message(chat_id, "⚠️ Папка с готовыми коллажами не найдена.")
        return

    photos = sorted(
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    if not photos:
        await bot.send_message(chat_id, "⚠️ Не удалось сгенерировать коллажи.")
        return

    short_id = hashlib.md5(folder_name.encode("utf-8")).hexdigest()[:8]
    for mid in media_msgs_store.get(short_id, []):
        try:
            await bot.delete_message(chat_id, mid)
        except Exception:
            pass

    all_sent_ids = []
    for chunk_start in range(0, len(photos), 10):
        chunk = photos[chunk_start : chunk_start + 10]
        msgs = await bot.send_media_group(chat_id, media=[InputMediaPhoto(media=FSInputFile(p)) for p in chunk])
        all_sent_ids.extend(m.message_id for m in msgs)

    media_msgs_store[short_id] = all_sent_ids
    pending_dispatches[short_id] = folder_path
    persist_store()

    _, _, code, sizes, price = collage_engine.parse_folder_metadata(folder_name)
    kb = build_preview_keyboard(folder_name, photos, chat_id, short_id)
    await bot.send_message(
        chat_id,
        f"✨ **PÎVOK — Коллажи готовы ({len(photos)} шт.)**\n"
        f"🔖 Код: `{code}` | 💰 Цена: `{price}$` | 📏 Размеры: `{sizes}`\n\n"
        "При необходимости поправьте кроп через 🛠 или выберите действие:",
        reply_markup=kb,
        parse_mode="Markdown",
    )


@dp.message(Command("start"))
async def cmd_start(message: Message, state: FSMContext):
    PHOTO_BUFFER.pop(message.chat.id, None)
    await state.clear()
    await message.answer(
        "🌸 **PÎVOK Cloud Studio (24/7)**\nВыберите действие в меню ниже:",
        reply_markup=build_keyboard(MAIN_MENU, 1),
        parse_mode="Markdown",
    )


@dp.message(Command("cancel", "stop"))
@dp.message(F.text.in_({"❌ Отмена", "Отмена", "/cancel", "/stop"}))
async def cmd_cancel(message: Message, state: FSMContext):
    PHOTO_BUFFER.pop(message.chat.id, None)
    await state.clear()
    await message.answer("❌ Действие отменено.", reply_markup=build_keyboard(MAIN_MENU, 1))


@dp.message(F.text == "🖼 Создать коллаж")
async def cmd_create_collage(message: Message, state: FSMContext):
    PHOTO_BUFFER.pop(message.chat.id, None)
    await state.clear()
    await state.set_state(CollageForm.photo)
    await message.answer(
        "📸 **Отправьте все фотографии одной модели** (альбомом или файлами).\n\n"
        "Когда загрузите все фото, выберите внизу:\n"
        "• **✅ Двойной коллаж (по 2 фото)** — формат PÎVOK 4×3\n"
        "• **✅ Одинарный коллаж (по 1 фото)**",
        reply_markup=build_keyboard(
            ["✅ Двойной коллаж (по 2 фото)", "✅ Одинарный коллаж (по 1 фото)", "❌ Отмена"], 2
        ),
        parse_mode="Markdown",
    )


@dp.message(StateFilter(None, CollageForm.photo), F.photo | F.document)
async def handle_incoming_photo(message: Message, state: FSMContext):
    if message.document:
        if not (message.document.mime_type or "").startswith("image/"):
            await message.answer("⚠️ Пожалуйста, отправляйте только изображения.")
            return
        file_id = message.document.file_id
    else:
        file_id = message.photo[-1].file_id

    if await state.get_state() is None:
        await state.set_state(CollageForm.photo)

    if file_id not in DOWNLOAD_PREFETCH:
        DOWNLOAD_PREFETCH[file_id] = asyncio.create_task(download_single_photo(file_id, bot))

    chat_id = message.chat.id
    if chat_id not in PHOTO_BUFFER:
        PHOTO_BUFFER[chat_id] = {"ids": [], "task": None}
    PHOTO_BUFFER[chat_id]["ids"].append(file_id)

    if PHOTO_BUFFER[chat_id]["task"]:
        PHOTO_BUFFER[chat_id]["task"].cancel()
    PHOTO_BUFFER[chat_id]["task"] = asyncio.create_task(_flush_photo_buffer(chat_id, state))


async def _flush_photo_buffer(chat_id: int, state: FSMContext):
    try:
        await asyncio.sleep(0.85)
    except asyncio.CancelledError:
        return

    buf = PHOTO_BUFFER.pop(chat_id, None)
    if not buf:
        return

    data = await state.get_data()
    photo_ids = data.get("photo_ids", [])
    photo_ids.extend(buf["ids"])
    await state.update_data(photo_ids=photo_ids)

    kb = (
        build_keyboard(["✅ ВСЕ ФОТО ЗАГРУЖЕНЫ", "❌ Отмена"], 1)
        if data.get("is_add_color")
        else build_keyboard(["✅ Двойной коллаж (по 2 фото)", "✅ Одинарный коллаж (по 1 фото)", "❌ Отмена"], 2)
    )
    await bot.send_message(
        chat_id,
        f"📥 Принято фотографий: **{len(photo_ids)} шт.**\nДогрузите ещё или нажмите кнопку ниже:",
        reply_markup=kb,
        parse_mode="Markdown",
    )


@dp.message(
    CollageForm.photo,
    F.text.in_({"✅ ВСЕ ФОТО ЗАГРУЖЕНЫ", "✅ Двойной коллаж (по 2 фото)", "✅ Одинарный коллаж (по 1 фото)"}),
)
async def handle_all_photos_loaded(message: Message, state: FSMContext):
    chat_id = message.chat.id
    if chat_id in PHOTO_BUFFER:
        buf = PHOTO_BUFFER.pop(chat_id)
        if buf.get("task"):
            buf["task"].cancel()
        data = await state.get_data()
        p_ids = data.get("photo_ids", [])
        p_ids.extend(buf.get("ids", []))
        await state.update_data(photo_ids=p_ids)

    data = await state.get_data()
    photo_ids = data.get("photo_ids", [])
    if not photo_ids:
        await message.answer("⚠️ Вы ещё не загрузили ни одной фотографии!")
        return

    if data.get("is_add_color") and data.get("target_folder"):
        folder_name = data["target_folder"]
        inc_folder = os.path.join(INCOMING_DIR, folder_name)
        os.makedirs(inc_folder, exist_ok=True)
        wait_m = await message.answer("⏳ Обрабатываю новые фото...", reply_markup=ReplyKeyboardRemove())

        downloaded = await asyncio.gather(*(download_single_photo(fid, bot) for fid in photo_ids))
        for fid in photo_ids:
            DOWNLOAD_PREFETCH.pop(fid, None)

        existing = [f for f in os.listdir(inc_folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        next_idx = (len(existing) // 2) + 1

        if len(downloaded) >= 2:
            pairs, labels = await asyncio.to_thread(color_clustering.cluster_photos_by_color, downloaded)
            for p_idx, pair in enumerate(pairs):
                if len(pair) < 2:
                    continue
                lbl = labels[p_idx] if p_idx < len(labels) else "COLOR"
                prefix = f"{next_idx + p_idx:02d}_{lbl}"
                for k, item in enumerate(pair, 1):
                    with open(os.path.join(inc_folder, f"{prefix}-{k}.{item['ext']}"), "wb") as f:
                        f.write(item["bytes"])
            await asyncio.to_thread(collage_engine.generate_folder_collages, folder_name, inc_folder)
        else:
            out_folder = os.path.join(OUTPUT_DIR, folder_name)
            os.makedirs(out_folder, exist_ok=True)
            item = downloaded[0]
            with open(os.path.join(out_folder, f"{next_idx:02d}_EXTRA.{item['ext']}"), "wb") as f:
                f.write(item["bytes"])

        await state.clear()
        try:
            await wait_m.delete()
        except Exception:
            pass
        await send_collage_preview(chat_id, folder_name)
        return

    is_single = "Одинарный" in message.text
    if not is_single and len(photo_ids) < 2:
        await message.answer("⚠️ Для двойного коллажа нужно минимум 2 фотографии.")
        return

    await state.update_data(is_single=is_single)
    if is_single or len(photo_ids) == 2:
        await state.update_data(grouping_mode="sequential" if is_single else "smart")
        await state.set_state(CollageForm.code)
        await message.answer("🔖 Введите **Код модели** (например: `O-21476`):", reply_markup=build_keyboard(["❌ Отмена"], 1), parse_mode="Markdown")
    else:
        await state.set_state(CollageForm.grouping_mode)
        await message.answer(
            f"📸 Загружено **{len(photo_ids)} фото**. Как сгруппировать пары?",
            reply_markup=build_keyboard(["🧠 Умная группировка (Авто)", "🔢 По парам (1-2, 3-4)", "❌ Отмена"], 2),
            parse_mode="Markdown",
        )


@dp.message(CollageForm.grouping_mode)
async def handle_grouping_mode(message: Message, state: FSMContext):
    await state.update_data(grouping_mode="smart" if "Умная" in (message.text or "") else "sequential")
    await state.set_state(CollageForm.code)
    await message.answer("🔖 Введите **Код модели** (например: `O-21476`):", reply_markup=build_keyboard(["❌ Отмена"], 1), parse_mode="Markdown")


@dp.message(CollageForm.code)
async def handle_code(message: Message, state: FSMContext):
    code_val = (message.text or "").strip().upper()
    if not code_val:
        await message.answer("⚠️ Введите корректный код модели:")
        return
    await state.update_data(code=code_val)
    await state.set_state(CollageForm.size)
    await message.answer("📏 Выберите или напишите **Размеры** (например: `38-44`):", reply_markup=build_keyboard(SIZE_LIST + ["❌ Отмена"], 3), parse_mode="Markdown")


@dp.message(CollageForm.size)
async def handle_size(message: Message, state: FSMContext):
    size_val = (message.text or "").strip()
    if not size_val:
        await message.answer("⚠️ Укажите размерный ряд:")
        return
    await state.update_data(size=size_val)
    await state.set_state(CollageForm.price)
    await message.answer("💰 Введите **Цену в $** (например: `79`):", reply_markup=build_keyboard(["❌ Отмена"], 1), parse_mode="Markdown")


async def show_summary(message: Message, state: FSMContext):
    data = await state.get_data()
    photo_cnt = len(data.get("photo_ids", []))
    is_single = data.get("is_single", False)
    est_collages = photo_cnt if is_single else photo_cnt // 2
    await state.set_state(CollageForm.confirm)
    await message.answer(
        "📋 **Проверьте данные коллажа PÎVOK:**\n\n"
        f"🔖 **Код:** `{data.get('code')}`\n"
        f"💰 **Цена:** `{data.get('price')}$`\n"
        f"📏 **Размеры:** `{data.get('size')}`\n"
        f"🖼 **Фотографий:** {photo_cnt} шт. → **{est_collages} колл.** (формат 4×3)",
        reply_markup=build_keyboard(
            ["✅ Создать коллажи", "✏️ Изменить Код", "✏️ Изменить Размеры", "✏️ Изменить Цену", "❌ Отмена"], 2
        ),
        parse_mode="Markdown",
    )


@dp.message(CollageForm.price)
async def handle_price(message: Message, state: FSMContext):
    price_val = (message.text or "").strip().replace("$", "").strip()
    if not price_val:
        await message.answer("⚠️ Введите цену:")
        return
    await state.update_data(price=price_val)
    await show_summary(message, state)


@dp.message(CollageForm.confirm)
async def handle_confirm(message: Message, state: FSMContext):
    txt = (message.text or "").strip()
    if txt == "✏️ Изменить Код":
        await state.update_data(editing="code")
        await state.set_state(CollageForm.edit_field)
        await message.answer("🔖 Введите новый **Код модели**:", reply_markup=build_keyboard(["❌ Отмена"], 1), parse_mode="Markdown")
        return
    if txt == "✏️ Изменить Размеры":
        await state.update_data(editing="size")
        await state.set_state(CollageForm.edit_field)
        await message.answer("📏 Выберите или введите новые **Размеры**:", reply_markup=build_keyboard(SIZE_LIST + ["❌ Отмена"], 3), parse_mode="Markdown")
        return
    if txt == "✏️ Изменить Цену":
        await state.update_data(editing="price")
        await state.set_state(CollageForm.edit_field)
        await message.answer("💰 Введите новую **Цену ($)**:", reply_markup=build_keyboard(["❌ Отмена"], 1), parse_mode="Markdown")
        return
    if txt != "✅ Создать коллажи":
        await message.answer("Выберите кнопку внизу 👇")
        return

    data = await state.get_data()
    chat_id = message.chat.id
    photo_ids = data.get("photo_ids", [])
    is_single = data.get("is_single", False)
    grouping_mode = data.get("grouping_mode", "smart")

    safe_code = data["code"].replace("|", "~").replace("_", "-").replace("/", "-")
    safe_size = data["size"].replace("|", "~").replace("_", "-")
    safe_price = data["price"].replace("|", "~").replace("_", "-")
    cat_tag = "ОДИНАРНЫЙ" if is_single else "ПЛАТЬЕ"
    folder_name = f"PIVOK_{cat_tag}_{safe_code}_{safe_size}_{safe_price}_{int(time.time())}"
    inc_folder = os.path.join(INCOMING_DIR, folder_name)
    os.makedirs(inc_folder, exist_ok=True)

    prog_msg = await message.answer(
        "⚙️ Собираю коллажи PÎVOK (4×3) в облаке...",
        reply_markup=ReplyKeyboardRemove(),
    )

    try:
        downloaded = await asyncio.gather(*(download_single_photo(fid, bot) for fid in photo_ids))
        for fid in photo_ids:
            DOWNLOAD_PREFETCH.pop(fid, None)

        if is_single:
            for idx, item in enumerate(downloaded, 1):
                with open(os.path.join(inc_folder, f"{idx:02d}_COLOR-{idx}.{item['ext']}"), "wb") as f:
                    f.write(item["bytes"])
        else:
            if grouping_mode == "smart":
                pairs, labels = await asyncio.to_thread(color_clustering.cluster_photos_by_color, downloaded)
            else:
                pairs = [downloaded[i : i + 2] for i in range(0, len(downloaded), 2)]
                labels = [f"COLOR{idx + 1}" for idx in range(len(pairs))]

            for p_idx, pair in enumerate(pairs):
                if len(pair) < 2:
                    continue
                lbl = labels[p_idx] if p_idx < len(labels) else f"COLOR{p_idx + 1}"
                prefix = f"{p_idx + 1:02d}_{lbl}"
                for k, item in enumerate(pair, 1):
                    with open(os.path.join(inc_folder, f"{prefix}-{k}.{item['ext']}"), "wb") as f:
                        f.write(item["bytes"])

        await asyncio.to_thread(collage_engine.generate_folder_collages, folder_name, inc_folder)
        await state.clear()
        try:
            await prog_msg.delete()
        except Exception:
            pass
        await send_collage_preview(chat_id, folder_name)
        await bot.send_message(
            chat_id,
            "Готово! Чтобы собрать следующую модель, нажмите кнопку ниже:",
            reply_markup=build_keyboard(MAIN_MENU, 1),
        )
    except Exception as e:
        logger.exception("Error generating collages")
        await state.clear()
        await bot.send_message(chat_id, f"❌ Ошибка при создании коллажей: {e}", reply_markup=build_keyboard(MAIN_MENU, 1))


@dp.message(CollageForm.edit_field)
async def handle_edit_field(message: Message, state: FSMContext):
    data = await state.get_data()
    field = data.get("editing")
    val = (message.text or "").strip()
    if field == "code":
        await state.update_data(code=val.upper())
    elif field == "size":
        await state.update_data(size=val)
    elif field == "price":
        await state.update_data(price=val.replace("$", "").strip())
    await show_summary(message, state)


@dp.callback_query(F.data.startswith("files_"))
async def cb_send_uncompressed_files(callback: CallbackQuery):
    short_id = callback.data.split("_", 1)[1]
    folder_path = pending_dispatches.get(short_id)
    if not folder_path or not os.path.exists(folder_path):
        await callback.answer("Файлы уже удалены.", show_alert=True)
        return
    photos = sorted(
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    await callback.answer("Отправляю оригиналы 2160×2880...")
    for chunk_start in range(0, len(photos), 10):
        chunk = photos[chunk_start : chunk_start + 10]
        await bot.send_media_group(
            callback.message.chat.id,
            media=[InputMediaDocument(media=FSInputFile(p)) for p in chunk],
        )


@dp.callback_query(F.data.startswith("refresh_"))
async def cb_refresh_preview(callback: CallbackQuery):
    short_id = callback.data.split("_", 1)[1]
    folder_path = pending_dispatches.get(short_id)
    if not folder_path or not os.path.exists(folder_path):
        await callback.answer("Папка не найдена.", show_alert=True)
        return
    await callback.answer("Обновляю превью...")
    folder_name = os.path.basename(folder_path)
    try:
        await callback.message.delete()
    except Exception:
        pass
    await send_collage_preview(callback.message.chat.id, folder_name)


@dp.callback_query(F.data.startswith("addcolor_"))
async def cb_add_color(callback: CallbackQuery, state: FSMContext):
    short_id = callback.data.split("_", 1)[1]
    folder_path = pending_dispatches.get(short_id)
    if not folder_path or not os.path.exists(folder_path):
        await callback.answer("Ошибка: папка не найдена.", show_alert=True)
        return
    await state.update_data(is_add_color=True, target_folder=os.path.basename(folder_path), photo_ids=[])
    await state.set_state(CollageForm.photo)
    await callback.answer()
    await bot.send_message(
        callback.message.chat.id,
        "📸 Отправьте дополнительные фото (новый цвет) для этой модели, затем нажмите **«✅ ВСЕ ФОТО ЗАГРУЖЕНЫ»**.",
        reply_markup=build_keyboard(["✅ ВСЕ ФОТО ЗАГРУЖЕНЫ", "❌ Отмена"], 1),
        parse_mode="Markdown",
    )


@dp.callback_query(F.data.startswith("del_"))
async def cb_delete_collage(callback: CallbackQuery):
    short_id = callback.data.split("_", 1)[1]
    folder_path = pending_dispatches.pop(short_id, None)
    if folder_path:
        folder_name = os.path.basename(folder_path)
        shutil.rmtree(folder_path, ignore_errors=True)
        shutil.rmtree(os.path.join(INCOMING_DIR, folder_name), ignore_errors=True)

    for mid in media_msgs_store.pop(short_id, []):
        try:
            await bot.delete_message(callback.message.chat.id, mid)
        except Exception:
            pass
    persist_store()
    await callback.answer("Удалено")
    await callback.message.edit_text("🗑 Коллажи удалены.")


@dp.message()
async def catch_all_unhandled(message: Message, state: FSMContext):
    if await state.get_state() is None:
        await message.answer(
            "🌸 **PÎVOK Cloud Studio**\nНажмите **«🖼 Создать коллаж»** внизу (или отправьте фото прямо сюда):",
            reply_markup=build_keyboard(MAIN_MENU, 1),
            parse_mode="Markdown",
        )


# ==========================================
# FASTAPI + WEBHOOK + WEBAPP LIFECYCLE
# ==========================================

_polling_task = None


@asynccontextmanager
async def lifespan(app_inst: FastAPI):
    global _polling_task
    await asyncio.to_thread(yolo_local.preload_models)
    if bot:
        if PUBLIC_BASE_URL.startswith("https://"):
            wh_url = f"{PUBLIC_BASE_URL}/telegram/webhook"
            await bot.set_webhook(wh_url, secret_token=WEBHOOK_SECRET, drop_pending_updates=False)
            logger.info(f"✅ Telegram Webhook registered at {wh_url}")
        else:
            await bot.delete_webhook(drop_pending_updates=False)
            _polling_task = asyncio.create_task(dp.start_polling(bot))
            logger.info("🌸 Started local Telegram polling (no PUBLIC_BASE_URL)")
    yield
    if _polling_task:
        _polling_task.cancel()
    if bot:
        await bot.session.close()


app = FastAPI(title="PIVOK Cloud Studio", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/webapp", StaticFiles(directory=os.path.join(BASE_DIR, "webapp")), name="webapp")


@app.get("/")
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "PIVOK Cloud Studio 24/7",
        "webhook_mode": PUBLIC_BASE_URL.startswith("https://"),
    }


@app.post("/telegram/webhook")
async def telegram_webhook(request: Request):
    if not bot:
        return {"ok": False}
    sec = request.headers.get("X-Telegram-Bot-Api-Secret-Token", "")
    if sec != WEBHOOK_SECRET:
        raise HTTPException(status_code=403, detail="Invalid secret token")
    payload = await request.json()
    update = Update.model_validate(payload, context={"bot": bot})
    asyncio.create_task(dp.feed_update(bot, update))
    return {"ok": True}


@app.get("/api/editor/{folder_name}")
async def get_editor_data(folder_name: str):
    folder_path = os.path.join(INCOMING_DIR, folder_name)
    if not os.path.isdir(folder_path):
        raise HTTPException(status_code=404, detail="Original photos not found")

    all_photos = sorted(
        f for f in os.listdir(folder_path) if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
    )
    parts = folder_name.split("_")
    category = parts[1] if len(parts) >= 5 else "ПЛАТЬЕ"

    color_groups = {}
    for p in all_photos:
        prefix = p.split("-")[0] if "-" in p else "DEFAULT"
        color_groups.setdefault(prefix, []).append(p)

    is_single = (category.upper() in ("ОДИНАРНЫЙ", "SINGLE")) or all(
        len(p_list) == 1 for p_list in color_groups.values()
    )

    swap_state = {}
    swap_file = os.path.join(folder_path, "swap_state.json")
    if os.path.exists(swap_file):
        try:
            with open(swap_file, "r", encoding="utf-8") as sf:
                swap_state = json.load(sf)
        except Exception:
            pass

    photos = []
    for prefix, p_list in color_groups.items():
        if is_single:
            photos.extend(p_list)
        elif prefix == "DEFAULT":
            for k in range(0, (len(p_list) // 2) * 2, 2):
                pair = p_list[k : k + 2]
                if swap_state.get(pair[0], False):
                    pair = [pair[1], pair[0]]
                photos.extend(pair)
        elif len(p_list) >= 2:
            pair = p_list[:2]
            if swap_state.get(pair[0], False):
                pair = [pair[1], pair[0]]
            photos.extend(pair)

    config_name = "SINGLE_config.json" if is_single else "STANDART_config.json"
    bg_name = "SINGLE_bg.png" if is_single else "STANDART_bg.png"

    with open(os.path.join(BASE_DIR, "resources", config_name), "r", encoding="utf-8") as f:
        config = json.load(f)

    encoded_photos = []
    for p in photos:
        with open(os.path.join(folder_path, p), "rb") as f:
            encoded_photos.append(f"data:image/jpeg;base64,{base64.b64encode(f.read()).decode('utf-8')}")

    with open(os.path.join(BASE_DIR, "resources", bg_name), "rb") as f:
        bg_data = f"data:image/png;base64,{base64.b64encode(f.read()).decode('utf-8')}"

    return {
        "is_single": is_single,
        "config": config,
        "bg_data": bg_data,
        "photos": encoded_photos,
        "photo_names": photos,
        "brand": "PIVOK",
        "category": category,
    }


@app.post("/api/editor/{folder_name}")
async def save_editor_data(folder_name: str, request: Request):
    data = await request.json()
    crop_data = data.get("images", [])
    chat_id = data.get("chat_id")
    folder_path = os.path.join(INCOMING_DIR, folder_name)
    if not os.path.isdir(folder_path):
        raise HTTPException(status_code=404, detail="Incoming folder not found")

    with open(os.path.join(folder_path, "manual_crop.json"), "w", encoding="utf-8") as f:
        json.dump(crop_data, f, ensure_ascii=False)

    await asyncio.to_thread(collage_engine.generate_folder_collages, folder_name, folder_path)
    if chat_id:
        asyncio.create_task(send_collage_preview(int(chat_id), folder_name))
    return {"status": "ok"}
