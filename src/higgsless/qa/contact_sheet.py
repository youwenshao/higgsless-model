"""Contact sheets: labelled image grids that agents and humans can review in one look."""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def _font(size: int) -> ImageFont.ImageFont:
    for name in ("DejaVuSans.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def contact_sheet(
    items: list[tuple[str | Path | Image.Image, str]],
    dst: str | Path,
    *,
    columns: int | None = None,
    cell_width: int = 320,
    title: str | None = None,
) -> Path:
    """Write a grid of (image, label) cells to `dst` (JPEG or PNG by extension)."""
    if not items:
        raise ValueError("no images for the contact sheet")
    images = [img if isinstance(img, Image.Image) else Image.open(img).convert("RGB") for img, _ in items]
    aspect = max(im.height / im.width for im in images)
    cell_h = int(cell_width * aspect)
    label_h = 22
    columns = columns or min(len(items), max(1, round(math.sqrt(len(items) * 1.6))))
    rows = math.ceil(len(items) / columns)
    title_h = 30 if title else 0
    sheet = Image.new("RGB", (columns * cell_width, title_h + rows * (cell_h + label_h)), (18, 18, 18))
    draw = ImageDraw.Draw(sheet)
    font = _font(15)
    if title:
        draw.text((8, 6), title, fill=(240, 240, 240), font=_font(18))
    for i, (image, (_, label)) in enumerate(zip(images, items, strict=True)):
        col, row = i % columns, i // columns
        x, y = col * cell_width, title_h + row * (cell_h + label_h)
        thumb = image.copy()
        thumb.thumbnail((cell_width, cell_h))
        sheet.paste(thumb, (x + (cell_width - thumb.width) // 2, y + (cell_h - thumb.height) // 2))
        draw.text((x + 6, y + cell_h + 3), label, fill=(230, 230, 230), font=font)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(dst, quality=88)
    return dst
