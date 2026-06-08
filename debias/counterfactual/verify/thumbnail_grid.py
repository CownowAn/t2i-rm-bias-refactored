"""Build a side-by-side (original | edited) grid for visual inspection."""
from __future__ import annotations

from pathlib import Path

from loguru import logger


def build_grid(
    pairs: list[tuple[Path, Path]],
    out_path: Path,
    cols: int = 4,
    thumb_px: int = 256,
    title: str = "",
) -> Path:
    """Save a grid where each row pairs (original, edited). PIL-only, no LPIPS."""
    from PIL import Image, ImageDraw, ImageFont

    if not pairs:
        logger.warning(f"build_grid: empty pairs, skipping {out_path}")
        return out_path

    pad = 8
    label_h = 20 if title else 0
    cell_w = thumb_px * 2 + pad        # one cell = (orig | edited)
    cell_h = thumb_px + pad

    n = len(pairs)
    n_cols = max(1, min(cols, n))
    n_rows = (n + n_cols - 1) // n_cols

    grid_w = n_cols * cell_w + pad
    grid_h = n_rows * cell_h + pad + label_h
    grid = Image.new("RGB", (grid_w, grid_h), (255, 255, 255))

    if title:
        draw = ImageDraw.Draw(grid)
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None
        draw.text((pad, 2), title[:200], fill=(0, 0, 0), font=font)

    for idx, (orig_path, edited_path) in enumerate(pairs):
        row, col = divmod(idx, n_cols)
        x0 = pad + col * cell_w
        y0 = label_h + pad + row * cell_h
        for j, p in enumerate([orig_path, edited_path]):
            try:
                im = Image.open(p).convert("RGB")
                im.thumbnail((thumb_px, thumb_px))
            except Exception as e:
                logger.warning(f"build_grid: cannot read {p}: {e}")
                im = Image.new("RGB", (thumb_px, thumb_px), (200, 200, 200))
            grid.paste(im, (x0 + j * thumb_px, y0))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(out_path)
    logger.info(f"  thumbnail grid → {out_path}")
    return out_path
