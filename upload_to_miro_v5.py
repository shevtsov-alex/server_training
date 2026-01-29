# Version 3.0.1

from __future__ import annotations

import asyncio
import hashlib
import json
import random
import re
import time
from collections import defaultdict
import html
from pathlib import Path
from typing import Dict, List, Tuple

import aiohttp
from PIL import Image

import logging

# ─── настройка логирования ──────────────────────────────────────────────
# Установите DEBUG = False для отключения подробного вывода.

DEBUG = False

logging.basicConfig(
    format="%(levelname)s: %(message)s", level=logging.DEBUG if DEBUG else logging.INFO
)

logger = logging.getLogger(__name__)

OUTPUT_FOLDER = Path("/workspace/layered/control_images_result")  # где ComfyUI сохраняет PNG
MIRO_API_KEY = None

# ─── ограничения/ретраи ────────────────────────────────────────────────────
MAX_RETRIES = 3
RETRY_DELAY = 2.0  # seconds
MAX_CONCURRENT_UPLOADS = 5
MAX_CONCURRENT_API = 10
api_semaphore: asyncio.Semaphore | None = None

# ─── сетка размещения ──────────────────────────────────────────────────────
COLS = 10  # 10 картинок в строке

# Разделённые зазоры/отступы по режимам, чтобы настройка одного режима
# не затрагивала другой
H_GAP_VERTICAL = 100  # горизонтальный зазор между картинками (vertical)
V_GAP_VERTICAL = 150  # вертикальный зазор между рядами (vertical)
CAPTION_GAP_VERTICAL = 50  # расстояние от стикера до первой строки (vertical)

H_GAP_HORIZONTAL = 100  # горизонтальный зазор между картинками (horizontal)
V_GAP_HORIZONTAL = 100  # вертикальный шаг строк (horizontal)
CAPTION_GAP_HORIZONTAL = 50  # расстояние от стикера до первой строки (horizontal)
STICKY_WIDTH = 400.0
STICKY_HEIGHT = 300.0
GROUP_MARGIN = 800  # промежуток между группами по горизонтали

# ─── метаданные и мета-стикер ────────────────────────────────────────────
# служебный altText, по которому отличаем «свои» изображения
# Префикс altText для всех изображений, загружаемых скриптом. После префикса
# через двоеточие будет записываться SHA-256 хэш содержимого файла.
# Пример итогового altText: "miro-sync:ab12cd34..."
ALT_TEXT = "miro-sync"

headers = {
    "accept": "application/json",
    "Authorization": f"Bearer {MIRO_API_KEY}",
}

# кэш filename → sha256
file_hash_cache: Dict[str, str] = {}

# Группа для файлов без второго префикса при фильтрации
MISC_GROUP = "others"

# ─── helpers ───────────────────────────────────────────────────────────────

# Группа определяется по «первому фрагменту» имени файла —
# всё до первого символа подчёркивания. Это может быть как числовой,
# так и буквенный префикс.


def extract_prefix(name: str) -> str:
    """Возвращает префикс имени файла до первого `_`.

    Если символ `_` отсутствует, возвращает имя целиком без расширения.
    Лишние пробелы по краям сегментов игнорируются.
    """

    stem = Path(name).stem.strip()  # без расширения
    return stem.split("_", 1)[0].strip()


# новая функция для получения следующего префикса после первого underscore
def extract_next_prefix(name: str) -> str:
    """Возвращает *следующий* префикс имени файла после первого символа ``_``.

    Если в имени файла нет второго сегмента, возвращается пустая строка.
    Примеры:
        "ast_002_model.png" → ``"002"``
        "foo_bar.png" → ``"bar"``
    """

    stem = Path(name).stem.strip()  # без расширения
    parts = [part.strip() for part in stem.split("_", 2)]
    return parts[1] if len(parts) >= 2 else ""


# вспомогательный ключ для сортировки групп:
def group_sort_key(prefix: str):
    """Ключ сортировки для групп."""

    if prefix.isdigit():
        return (0, int(prefix))
    return (1, prefix.lower())


def file_sort_key(path: Path):
    """Ключ сортировки файлов, учитывающий числовые суффиксы.

    Делит имя на блоки из букв и чисел и сортирует так, чтобы числа сравнивались как числа.
    """
    stem = path.stem
    parts = re.split(r"(\d+)", stem)
    key: list = []
    for part in parts:
        if part.isdigit():
            key.append((0, int(part)))
        else:
            key.append((1, part.lower()))
    return key


def wait_for_min_pngs(
    min_count: int,
    poll_interval: float = 0.5,
    max_wait_seconds: float | None = None,
    prefix_filter: str | None = None,
) -> None:
    """Блокирующее ожидание появления как минимум ``min_count`` PNG-файлов в ``OUTPUT_FOLDER``.

    Если указан ``prefix_filter``, учитываются только файлы, у которых первый префикс
    (до символа ``_``) совпадает с ``prefix_filter``.

    Если ``min_count`` ≤ 0 или значение не задано, функция немедленно возвращает управление.
    ``max_wait_seconds`` можно не задавать — тогда ждём сколько потребуется.
    """

    if not isinstance(min_count, int) or min_count <= 0:
        return

    start_time = time.time()
    while True:
        try:
            if prefix_filter:
                files_count = sum(
                    1
                    for p in OUTPUT_FOLDER.glob("*.png")
                    if p.is_file() and extract_prefix(p.name) == prefix_filter
                )
            else:
                files_count = sum(1 for p in OUTPUT_FOLDER.glob("*.png") if p.is_file())
        except Exception:
            files_count = 0
        if files_count >= min_count:
            return
        if max_wait_seconds is not None and (time.time() - start_time) >= max_wait_seconds:
            return
        time.sleep(poll_interval)


def board_img_size(item: Dict) -> Tuple[int, int]:
    """Возвращает (width, height) изображения, уже находящегося на доске.

    Порядок источников высоты (чтобы избежать занижения при отсутствии height в API):
    1) geometry.height, если есть;
    2) размер локального файла по названию элемента (title);
    3) размер локального файла по хэшу из altText ("miro-sync:<hash>");
    4) fallback: считаем квадратным (height = width).
    """

    geom = item.get("geometry", {})
    width = int(geom.get("width", 0))
    height = int(geom.get("height", 0))

    # 1) Если Miro вернул высоту — используем её напрямую
    if height:
        return width, height

    # 2) Попытка получить размер из локального файла по названию
    title = item.get("title") or item.get("data", {}).get("title", "")
    if isinstance(title, str) and title:
        local_path = OUTPUT_FOLDER / title
        if local_path.exists() and local_path.is_file():
            try:
                with Image.open(local_path) as im:
                    w, h = im.size
                    return w, h
            except Exception:
                pass

    # 3) Попытка найти локальный файл по хэшу из altText
    alt_text = item.get("altText") or item.get("data", {}).get("altText", "")
    if isinstance(alt_text, str) and alt_text.startswith(f"{ALT_TEXT}:"):
        hsh = alt_text.split(":", 1)[1]
        # Пробуем найти имя файла по кэшу хэшей
        for fname, fh in file_hash_cache.items():
            if fh == hsh:
                local_path = OUTPUT_FOLDER / fname
                if local_path.exists() and local_path.is_file():
                    try:
                        with Image.open(local_path) as im:
                            w, h = im.size
                            return w, h
                    except Exception:
                        break

    # 4) Fallback: без height предполагаем квадрат — может избыточно по высоте,
    # но гарантированно не занизит портретные изображения.
    return width, width


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def img_size(path: Path) -> Tuple[int, int]:
    with Image.open(path) as img:
        return img.size  # (w, h)


# ─── Miro REST helpers ─────────────────────────────────────────────────────


async def api_request(session: aiohttp.ClientSession, method: str, url: str, **kwargs):
    """Обёртка с общим семафором, ретраями и 429-backoff."""
    global api_semaphore

    for attempt in range(MAX_RETRIES):
        try:
            async with api_semaphore:
                async with session.request(method, url, headers=headers, **kwargs) as resp:
                    if resp.status == 429:
                        await asyncio.sleep(RETRY_DELAY * 2 ** attempt + random.random())
                        continue
                    if resp.status in {200, 201}:
                        if method.lower() == "get":
                            return await resp.json()
                        if resp.content_type == "application/json":
                            return await resp.json()
                        return None
                    if resp.status == 404:
                        raise RuntimeError(f"404 for {url}")
                    txt = await resp.text()
                    raise RuntimeError(f"HTTP {resp.status}: {txt}")
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                raise
            await asyncio.sleep(RETRY_DELAY)


async def list_board_items(session: aiohttp.ClientSession, board_id: str) -> List[Dict]:
    """Возвращает ВСЕ items доски."""
    url = f"https://api.miro.com/v2/boards/{board_id}/items?limit=50"
    items: List[Dict] = []
    while url:
        data = await api_request(session, "get", url)
        items.extend(data.get("data", []))
        url = data.get("links", {}).get("next")
    return items


async def upload_png(
    session: aiohttp.ClientSession,
    board_id: str,
    path: Path,
    center_x: float,
    center_y: float,
    img_width: int,
    file_hash: str,
):
    """Загружает PNG на доску Miro в заданную позицию
    """

    # Оставляем читаемое имя файла в названии элемента.
    title = path.name

    # В altText пишем техническую информацию: префикс + хэш.
    alt_text = f"{ALT_TEXT}:{file_hash}"

    payload = {
        "title": title,
        "altText": alt_text,
        "position": {"x": center_x, "y": center_y, "origin": "center"},
        "geometry": {"width": img_width},
    }
    form = aiohttp.FormData()
    form.add_field("data", json.dumps(payload), content_type="application/json")
    form.add_field("resource", path.read_bytes(), filename=path.name, content_type="image/png")

    await api_request(session, "post", f"https://api.miro.com/v2/boards/{board_id}/images", data=form)
    print(f"✅ uploaded {path.name}")


def parse_caption_content(content: str) -> Tuple[str, float]:
    """Разбирает текст стикера вида "prefix|reserved".

    - Очищает HTML-тэги и сущности, переводит <br> в переводы строк
    - Берёт первую непустую строку
    - Нормализует пробелы

    Возвращает кортеж (prefix, reserved_width). Если зарезервированная
    ширина отсутствует или не является числом, возвращается 0.0.
    """
    raw = content or ""
    if not isinstance(raw, str):
        raw = str(raw)

    # HTML → текст
    text = html.unescape(raw)
    text = re.sub(r"(?i)<br\s*/?>", "\n", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = text.replace("\xa0", " ").strip()

    # Первая непустая строка
    if "\n" in text:
        lines = [ln.strip() for ln in text.splitlines()]
        for ln in lines:
            if ln:
                text = ln
                break

    prefix, _, rest = text.partition("|")
    prefix = prefix.strip()
    try:
        reserved = float(rest.strip()) if rest else 0.0
    except Exception:
        reserved = 0.0

    if DEBUG:
        logger.debug(
            "parse_caption_content: raw=%r -> text=%r -> prefix=%r reserved=%s",
            content,
            text,
            prefix,
            reserved,
        )
    return prefix, reserved


def build_caption_content(prefix: str, reserved_width: float) -> str:
    """Формирует content для caption-stickers в формате "prefix|reserved"."""
    return f"{prefix}|{reserved_width:.1f}"


async def create_caption_sticky(
    session: aiohttp.ClientSession,
    board_id: str,
    prefix: str,
    reserved_width: float,
    x: float,
    y: float,
) -> Dict:
    """Создаёт новый caption-sticky и возвращает объект, полученный от API."""

    payload = {
        "data": {"content": build_caption_content(prefix, reserved_width)},
        "position": {"x": x, "y": y, "origin": "center"},
        "geometry": {"width": STICKY_WIDTH},
    }
    return await api_request(
        session, "post", f"https://api.miro.com/v2/boards/{board_id}/sticky_notes", json=payload
    )


# ─── core logic ────────────────────────────────────────────────────────────


class BaseLayout:
    """Базовый интерфейс режима раскладки."""

    name: str = "base"

    async def place_all(
        self,
        session: aiohttp.ClientSession,
        board_id: str,
        groups: Dict[str, List[Path]],
        captions: Dict[str, Dict],
        images_by_prefix: Dict[str, List[Dict]],
        on_board: Dict[str, Dict],
        *,
        enforce_sequential: bool = False,
    ) -> None:
        raise NotImplementedError


class VerticalRowLayout(BaseLayout):
    """Новый режим по умолчанию: группы (стикеры) идут сверху вниз,
    изображения каждой группы располагаются в одну строку справа, под стикером.
    """

    name: str = "vertical"

    async def place_all(
        self,
        session: aiohttp.ClientSession,
        board_id: str,
        groups: Dict[str, List[Path]],
        captions: Dict[str, Dict],
        images_by_prefix: Dict[str, List[Dict]],
        on_board: Dict[str, Dict],
        *,
        enforce_sequential: bool = False,
    ) -> None:
        jobs_to_upload: List[Tuple[Path, float, float, int, str]] = []

        loop = asyncio.get_running_loop()
        base_x = 0.0
        # Стартуем ниже уже существующего контента, чтобы исключить наложения при повторных запусках
        current_top_y = 0.0  # верх текущей секции (для новых групп)
        try:
            existing_bottoms: List[float] = []
            # Учитываем уже размещённые изображения (де-дуплируем по id)
            seen_ids: set[str] = set()
            for imgs in images_by_prefix.values():
                for img in imgs:
                    iid = str(img.get("id"))
                    if iid in seen_ids:
                        continue
                    seen_ids.add(iid)
                    w_e, h_e = board_img_size(img)
                    cy_e = img.get("position", {}).get("y", 0.0)
                    existing_bottoms.append(cy_e + h_e / 2)
            # И стикеры-заголовки
            for info in captions.values():
                cap_item = info["item"]
                cy_c = cap_item.get("position", {}).get("y", 0.0)
                h_c = cap_item.get("geometry", {}).get("height", STICKY_HEIGHT)
                existing_bottoms.append(cy_c + h_c / 2)
            if existing_bottoms:
                current_top_y = max(existing_bottoms) + V_GAP_VERTICAL
        except Exception:
            # В случае любых неожиданных проблем просто стартуем с 0.0
            pass

        for prefix in sorted(groups, key=group_sort_key):
            files = sorted(groups[prefix], key=file_sort_key)
            prefix_key = prefix.lower()
            sizes_local = await asyncio.gather(
                *[loop.run_in_executor(None, img_size, p) for p in files]
            )

            # ключи в images_by_prefix нормализованы к lower-case
            existing_imgs = images_by_prefix.get(prefix_key, [])
            sizes_existing = [board_img_size(img) for img in existing_imgs]

            all_heights = [h for _, h in sizes_local] + [h for _, h in sizes_existing]
            max_h = max(all_heights) if all_heights else 0

            if prefix_key in captions:
                # уже есть стикер — не двигаем. Вертикальный ряд картинок не
                # зависит от положения стикера: используем верх по изображениям,
                # если они уже размещены; иначе — текущий курсор.
                cap_item = captions[prefix_key]["item"]
                caption_x = cap_item.get("position", {}).get("x", base_x)
                if DEBUG:
                    logger.debug(
                        "Vertical: existing caption for '%s' at x=%.1f; computing group_top",
                        prefix,
                        caption_x,
                    )

                if existing_imgs:
                    tops: List[float] = []
                    for img, (_w_e, h_e) in zip(existing_imgs, sizes_existing):
                        cy_e = img.get("position", {}).get("y", 0.0)
                        tops.append(cy_e - h_e / 2)
                    group_top = min(tops) if tops else current_top_y
                else:
                    # Если у группы есть стикер, но ещё нет изображений, размещаем ряд
                    # сразу под стикером, либо ниже уже занятого пространства
                    cap_h = cap_item.get("geometry", {}).get("height", STICKY_HEIGHT)
                    cap_bottom = cap_item.get("position", {}).get("y", 0.0) + cap_h / 2
                    group_top = max(current_top_y, cap_bottom + CAPTION_GAP_VERTICAL)

                # шаг по вертикали определяется только высотой картинок и V_GAP_VERTICAL
                current_top_y = max(current_top_y, group_top + max_h + V_GAP_VERTICAL)
            else:
                # создаём новый стикер. Размещаем его ВЫШЕ строки изображений,
                # чтобы его высота не влияала на вертикальный шаг между группами.
                caption_x = base_x
                group_top = current_top_y
                caption_y_center = group_top - CAPTION_GAP_VERTICAL - STICKY_HEIGHT / 2
                if DEBUG:
                    logger.debug(
                        "Vertical: creating new caption for '%s' at x=%.1f y=%.1f (group_top=%.1f, current_top_y=%.1f)",
                        prefix,
                        caption_x,
                        caption_y_center,
                        group_top,
                        current_top_y,
                    )
                cap_item = await create_caption_sticky(
                    session,
                    board_id,
                    prefix,
                    0.0,
                    caption_x,
                    caption_y_center,
                )
                current_top_y = group_top + max_h + V_GAP_VERTICAL

            left_x = caption_x + STICKY_WIDTH / 2 + H_GAP_VERTICAL

            # учитываем уже размещённые изображения группы: продолжим справа от самого правого
            continue_from_x = left_x
            if existing_imgs:
                for img, (w_e, _h_e) in zip(existing_imgs, sizes_existing):
                    cx_e = img.get("position", {}).get("x", 0.0)
                    right_edge_e = cx_e + w_e / 2
                    continue_from_x = max(continue_from_x, right_edge_e + H_GAP_VERTICAL)

            current_x = continue_from_x

            for path, (w, h) in zip(files, sizes_local):
                file_hash = file_hash_cache[path.name]
                if file_hash in on_board:
                    if DEBUG:
                        logger.debug("Skip existing '%s' (already on board)", path.name)
                    continue

                cx = current_x + w / 2
                cy = group_top + h / 2

                jobs_to_upload.append((path, cx, cy, w, file_hash))
                current_x = current_x + w + H_GAP_VERTICAL

        if jobs_to_upload:
            upload_sem = asyncio.Semaphore(MAX_CONCURRENT_UPLOADS)

            async def _upload_job(p: Path, x: float, y: float, width: int, hsh: str):
                async with upload_sem:
                    await upload_png(session, board_id, p, x, y, width, hsh)

            tasks = [
                asyncio.create_task(_upload_job(p, x, y, w, hsh))
                for (p, x, y, w, hsh) in jobs_to_upload
            ]
            await asyncio.gather(*tasks)


class HorizontalGridLayout(BaseLayout):
    """Режим совместимый с прежним поведением: новые группы справа, изображения — сеткой.
    """

    name: str = "horizontal"

    async def place_all(
        self,
        session: aiohttp.ClientSession,
        board_id: str,
        groups: Dict[str, List[Path]],
        captions: Dict[str, Dict],
        images_by_prefix: Dict[str, List[Dict]],
        on_board: Dict[str, Dict],
        *,
        rightmost_edge_reserved: float,
        rightmost_edge_actual: float,
        base_y: float = 0.0,
        enforce_sequential: bool = False,
    ) -> None:
        rightmost_edge = max(rightmost_edge_reserved, rightmost_edge_actual)

        loop = asyncio.get_running_loop()
        jobs_to_upload: List[Tuple[Path, float, float, int, str]] = []

        for prefix in sorted(groups, key=group_sort_key):
            files = sorted(groups[prefix], key=file_sort_key)
            sizes_local = await asyncio.gather(*[loop.run_in_executor(None, img_size, p) for p in files])

            prefix_key = prefix.lower()
            existing_imgs = images_by_prefix.get(prefix_key, [])
            sizes_existing = [board_img_size(img) for img in existing_imgs]

            # шаг по высоте
            all_heights = [h for _, h in sizes_local] + [h for _, h in sizes_existing]
            max_h = max(all_heights) if all_heights else 0
            cell_height = max_h + V_GAP_HORIZONTAL

            # ширина по локальным файлам для резервирования
            row_widths: List[float] = []
            for i in range(0, len(files), COLS):
                sub = sizes_local[i : i + COLS]
                if not sub:
                    continue
                row_w = sum(w for w, _ in sub) + H_GAP_HORIZONTAL * (len(sub) - 1)
                row_widths.append(row_w)
            group_width_actual = max(row_widths) if row_widths else 0.0

            if prefix_key in captions:
                cap_item = captions[prefix_key]["item"]
                caption_x = cap_item.get("position", {}).get("x", 0.0)
                current_reserved = captions[prefix_key]["reserved"]
                left_x = caption_x - STICKY_WIDTH / 2
                rightmost_edge = max(rightmost_edge, caption_x + current_reserved)
            else:
                avg_w = sum(w for w, _ in sizes_local) / len(sizes_local) if sizes_local else STICKY_WIDTH
                ten_cols_width = COLS * avg_w + (COLS - 1) * H_GAP_HORIZONTAL
                group_width_reserved = max(group_width_actual, ten_cols_width)
                reserved_needed = group_width_reserved + GROUP_MARGIN - STICKY_WIDTH / 2

                left_x = rightmost_edge
                caption_x = left_x + STICKY_WIDTH / 2
                cap_item = await create_caption_sticky(
                    session,
                    board_id,
                    prefix,
                    reserved_needed,
                    caption_x,
                    base_y + STICKY_HEIGHT / 2,
                )
                left_x = caption_x - STICKY_WIDTH / 2
                rightmost_edge = caption_x + reserved_needed
                current_reserved = reserved_needed

            group_top = base_y + STICKY_HEIGHT + CAPTION_GAP_HORIZONTAL

            row_offsets: Dict[int, float] = defaultdict(lambda: left_x)

            for img, (w_e, _h_e) in zip(existing_imgs, sizes_existing):
                cx_e = img.get("position", {}).get("x", 0.0)
                cy_e = img.get("position", {}).get("y", 0.0)
                row_e = int(max(0, (cy_e - group_top)) // cell_height)
                right_edge_e = cx_e + w_e / 2
                row_offsets[row_e] = max(row_offsets[row_e], right_edge_e + H_GAP_HORIZONTAL)

            if row_offsets:
                _grp_existing = max(v - left_x for v in row_offsets.values()) - H_GAP_HORIZONTAL
                group_width_actual = max(group_width_actual, _grp_existing)

            content_width = current_reserved + STICKY_WIDTH / 2 - GROUP_MARGIN
            content_right = caption_x + current_reserved - GROUP_MARGIN
            allowed_right_edge = content_right + 0.02 * max(content_width, 0)
            current_row = 0

            for path, (w, h) in zip(files, sizes_local):
                file_hash = file_hash_cache[path.name]
                if file_hash in on_board:
                    if DEBUG:
                        logger.debug("Skip existing '%s' (already on board)", path.name)
                    continue

                current_x = row_offsets[current_row]
                while current_x + w > allowed_right_edge:
                    if current_x == left_x:
                        break
                    current_row += 1
                    current_x = row_offsets[current_row]

                cx = current_x + w / 2
                cy = group_top + current_row * cell_height + h / 2
                row_offsets[current_row] = current_x + w + H_GAP_HORIZONTAL

                jobs_to_upload.append((path, cx, cy, w, file_hash))

        if jobs_to_upload:
            upload_sem = asyncio.Semaphore(MAX_CONCURRENT_UPLOADS)

            async def _upload_job(p: Path, x: float, y: float, width: int, hsh: str):
                async with upload_sem:
                    await upload_png(session, board_id, p, x, y, width, hsh)

            tasks = [
                asyncio.create_task(_upload_job(p, x, y, w, hsh))
                for (p, x, y, w, hsh) in jobs_to_upload
            ]
            await asyncio.gather(*tasks)


class SyncOrchestrator:
    """Оркестратор синхронизации: скан папки, индексация доски и делегирование
    размещения выбранной стратегии раскладки.
    """

    def __init__(self, layout: BaseLayout):
        self.layout = layout

    async def run(
        self,
        board_id: str,
        filter_prefix: str | None = None,
        *,
        enforce_sequential: bool = False,
    ) -> None:
        global api_semaphore

        # ── локальные файлы ────────────────────────────────────────────────
        local_files = sorted([p for p in OUTPUT_FOLDER.glob("*.png") if p.is_file()])
        if not local_files:
            print("⚠️  No PNG files found in output folder")
            return

        loop = asyncio.get_running_loop()
        hashes = await asyncio.gather(*[loop.run_in_executor(None, sha256, p) for p in local_files])
        file_hash_cache.update({p.name: h for p, h in zip(local_files, hashes)})

        groups: Dict[str, List[Path]] = defaultdict(list)
        for p in local_files:
            groups[extract_prefix(p.name)].append(p)

        # ── фильтр по префиксу и регруппировка по второму сегменту ─────────
        if filter_prefix:
            groups = {k: v for k, v in groups.items() if k == filter_prefix}
            if not groups:
                print(f"⚠️  No PNG files found with prefix '{filter_prefix}'.")
                return
            filtered_files = [p for files in groups.values() for p in files]
            new_groups: Dict[str, List[Path]] = defaultdict(list)
            for p in filtered_files:
                next_pref = extract_next_prefix(p.name)
                if not next_pref:
                    next_pref = MISC_GROUP
                new_groups[next_pref].append(p)
            groups = new_groups

        # ── подключение к Miro ────────────────────────────────────────────
        async with aiohttp.ClientSession(headers=headers) as session:
            api_semaphore = asyncio.Semaphore(MAX_CONCURRENT_API)
            items = await list_board_items(session, board_id)

            # индексация существующих элементов
            on_board: Dict[str, Dict] = {}
            captions: Dict[str, Dict] = {}
            our_images: List[Dict] = []

            for it in items:
                if it.get("type") == "image":
                    alt_text = it.get("altText") or it.get("data", {}).get("altText", "")
                    if isinstance(alt_text, str) and alt_text.startswith(f"{ALT_TEXT}:"):
                        hsh = alt_text.split(":", 1)[1]
                        on_board[hsh] = it
                        if it not in our_images:
                            our_images.append(it)
                    title = it.get("title") or it.get("data", {}).get("title", "")
                    if isinstance(title, str) and title.startswith("ComfyUI:"):
                        hsh = title.split(":", 1)[1]
                        on_board[hsh] = it
                        if it not in our_images:
                            our_images.append(it)
                    if (alt_text == ALT_TEXT) and it not in our_images:
                        our_images.append(it)
                elif it.get("type") == "sticky_note":
                    content = it.get("data", {}).get("content", "")
                    prefix, reserved = parse_caption_content(content)
                    norm_prefix = prefix.strip()
                    key = norm_prefix.lower()
                    if key:
                        if key in captions:
                            old = captions[key]["item"]
                            old_pos = old.get("position", {})
                            new_pos = it.get("position", {})
                            logger.warning(
                                "Duplicate caption detected for prefix '%s': existing at (%.1f, %.1f) id=%s; new at (%.1f, %.1f) id=%s",
                                norm_prefix,
                                old_pos.get("x", 0.0),
                                old_pos.get("y", 0.0),
                                old.get("id"),
                                new_pos.get("x", 0.0),
                                new_pos.get("y", 0.0),
                                it.get("id"),
                            )
                        else:
                            captions[key] = {"item": it, "reserved": reserved}
                            if DEBUG:
                                pos = it.get("position", {})
                                logger.debug(
                                    "Indexed caption: prefix='%s' reserved=%.1f at (%.1f, %.1f)",
                                    norm_prefix,
                                    reserved,
                                    pos.get("x", 0.0),
                                    pos.get("y", 0.0),
                                )

            images_by_prefix: Dict[str, List[Dict]] = defaultdict(list)
            for img in our_images:
                title = img.get("title") or img.get("data", {}).get("title", "")
                if not isinstance(title, str) or not title:
                    continue
                pref1 = extract_prefix(title)
                pref2_raw = extract_next_prefix(title)
                pref2 = pref2_raw if pref2_raw else MISC_GROUP
                key1 = pref1.lower()
                key2 = pref2.lower()
                if key1 in captions:
                    images_by_prefix[key1].append(img)
                if key2 in captions:
                    images_by_prefix[key2].append(img)
            if DEBUG:
                logger.debug(
                    "images_by_prefix keys: %s",
                    ", ".join(sorted(images_by_prefix.keys())) or "<none>",
                )

            rightmost_edge_reserved = 0.0
            for info in captions.values():
                pos_x = info["item"].get("position", {}).get("x", 0.0)
                rightmost_edge_reserved = max(rightmost_edge_reserved, pos_x + info["reserved"])

            rightmost_edge_actual = 0.0
            for it in items:
                pos = it.get("position", {})
                geom = it.get("geometry", {})
                candidate = pos.get("x", 0) + geom.get("width", 0) / 2
                rightmost_edge_actual = max(rightmost_edge_actual, candidate)

            if isinstance(self.layout, HorizontalGridLayout):
                await self.layout.place_all(
                    session,
                    board_id,
                    groups,
                    captions,
                    images_by_prefix,
                    on_board,
                    rightmost_edge_reserved=rightmost_edge_reserved,
                    rightmost_edge_actual=rightmost_edge_actual,
                    base_y=0.0,
                    enforce_sequential=enforce_sequential,
                )
            else:
                await self.layout.place_all(
                    session,
                    board_id,
                    groups,
                    captions,
                    images_by_prefix,
                    on_board,
                    enforce_sequential=enforce_sequential,
                )


async def process_board(
    board_id: str,
    filter_prefix: str | None = None,
    mode: str = "vertical",
    *,
    enforce_sequential: bool = False,
):
    """Точка входа синхронизации. Делегирует работу выбранной стратегии раскладки.

    mode: "vertical" (по умолчанию) — новый режим. "horizontal" — совместимый со старым.
    """
    layout: BaseLayout
    if mode.lower() == "horizontal":
        layout = HorizontalGridLayout()
    else:
        layout = VerticalRowLayout()

    orchestrator = SyncOrchestrator(layout)
    await orchestrator.run(board_id, filter_prefix, enforce_sequential=enforce_sequential)


# ─── CLI ───────────────────────────────────────────────────────────────────


def main():
    board_id = input("Введите ID доски Miro: ").strip()

    prefix_filter_input = input(
        "Введите префикс файла (оставьте пустым для всех файлов): "
    ).strip()
    prefix_filter: str | None = prefix_filter_input or None

    mode_input = input(
        "Режим раскладки (vertical по умолчанию, либо horizontal): "
    ).strip().lower() or "vertical"

    wait_input = input(
        "Сколько PNG дождаться перед стартом (Enter — не ждать): "
    ).strip()
    try:
        wait_min_files = int(wait_input) if wait_input else 0
    except ValueError:
        wait_min_files = 0

    first_run = True

    while True:
        try:
            if first_run and wait_min_files > 0:
                if prefix_filter:
                    print(
                        f"⏳ Ожидаю минимум {wait_min_files} PNG с префиксом '{prefix_filter}' в папке '{OUTPUT_FOLDER}'..."
                    )
                else:
                    print(
                        f"⏳ Ожидаю минимум {wait_min_files} PNG в папке '{OUTPUT_FOLDER}'..."
                    )
                wait_for_min_pngs(wait_min_files, prefix_filter=prefix_filter)
                print("✅ Достаточное количество файлов найдено, начинаю загрузку…")

            # Для первого прохода обеспечиваем строгую последовательность загрузки
            asyncio.run(
                process_board(
                    board_id,
                    prefix_filter,
                    mode_input,
                    enforce_sequential=first_run,
                )
            )
            if first_run:
                first_run = False
        except Exception as exc:
            print(f"❌ Ошибка в процессе обработки: {exc}")

        try:
            time.sleep(60)  # подождать минуту перед следующей синхронизацией
        except KeyboardInterrupt:
            print("\nInterrupted – exiting.")
            break


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted")
