"""Show what the proposer LLM saw and what it proposed for one (or all) call(s).

Per-call JSONs saved by the search live under
``outputs/search/.../proposer/proposer_step{S}_topic{T}_call{C}.json`` and have
this schema (relevant fields):

    step_idx, topic_id, call_idx, proposer_model, reasoning_effort,
    cluster_summary, current_pool, avoid_attrs,
    selected_prompts        # list of prompt strings shown this call
    images                  # list of {group, prompt_text, set, image_path}
                            # `set` ∈ {"A", "B"}; one group per selected prompt
    proposer_prompt         # exact text rendered into the LLM (sans images)
    response_text           # raw LLM output
    proposals               # parsed list of attribute strings
    reasoning, reasoning_content, usage

The same dir holds a sibling ``.txt`` (just the rendered ``proposer_prompt``);
we ignore it because the JSON has everything.

Usage
-----
    python -m analysis.show_proposer_io <path>                  # PNGs next to JSONs
    python -m analysis.show_proposer_io <path> --out_dir DIR    # PNGs in DIR

`<path>` may be:
- a directory: every ``proposer_*.json`` inside is processed (sorted by name).
- a single ``.json`` file: just that one.

One composite PNG per call is always written, named
``proposer_step{S}_topic{T}_call{C}.png``. Without ``--out_dir`` it lands in
the same directory as the source JSON; with ``--out_dir DIR`` everything goes
to ``DIR``. The PNG shows, for every prompt, Set A thumbnails (top row) and
Set B thumbnails (bottom row), plus the prompt header above and the proposed
attributes listed at the bottom.

Optional flags:
    --max_paths_per_set N   Truncate per-set image-path list to N (default 4)
                            for text mode AND limit thumbnails per set in the
                            PNG (since set sizes can vary).
    --hide_paths            Don't list individual image paths in the text
                            output (sets summarized as "Set A: N images" only).
    --show_full_prompt      Print the entire ``proposer_prompt`` text.
    --show_reasoning        Print the proposer's "reasoning" field (its
                            summary of A vs B).
    --out_dir DIR           Directory to save the per-call composite PNGs.
    --thumb_px N            Thumbnail edge length in the PNG (default 220).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


def _iter_call_files(path: Path) -> list[Path]:
    if path.is_file():
        if path.suffix != ".json":
            sys.exit(f"ERROR: {path} is not a .json file")
        return [path]
    if path.is_dir():
        files = sorted(path.glob("proposer_*.json"))
        if not files:
            sys.exit(f"ERROR: no proposer_*.json under {path}")
        return files
    sys.exit(f"ERROR: {path} does not exist")


def _print_separator(ch: str = "─", width: int = 100) -> None:
    print(ch * width)


def _print_header(title: str) -> None:
    _print_separator("=")
    print(title)
    _print_separator("=")


def _image_id_from_path(image_path: str) -> str:
    """``.../baseline_<md5>_<idx>.png`` → ``<md5>_<idx>``."""
    name = Path(image_path).stem
    return name[len("baseline_"):] if name.startswith("baseline_") else name


def _manifest_index_for(
    image_path: str,
    cache: dict,
) -> "dict[str, dict[str, float]] | None":
    """Return {image_id: reward_scores} loaded (lazily, per-dir) from manifest.json.

    Each baseline directory has its own ``manifest.json`` with reward_scores per
    image. We cache the parsed index per parent directory so a single proposer
    call with 64 images only hits disk once.
    """
    parent = Path(image_path).parent
    if parent in cache:
        return cache[parent]
    manifest_path = parent / "manifest.json"
    if not manifest_path.exists():
        cache[parent] = None
        return None
    try:
        data = json.loads(manifest_path.read_text())
    except Exception:
        cache[parent] = None
        return None
    index: dict[str, dict[str, float]] = {}
    for _prompt, entries in data.get("baselines", {}).items():
        for e in entries:
            iid = e.get("image_id")
            if iid:
                index[iid] = e.get("reward_scores", {}) or {}
    cache[parent] = index
    return index


def _resolve_reward(
    image_path: str,
    reward_model: str,
    manifest_cache: dict,
) -> "float | None":
    idx = _manifest_index_for(image_path, manifest_cache)
    if idx is None:
        return None
    return idx.get(_image_id_from_path(image_path), {}).get(reward_model)


def _load_thumb(path: str, thumb_px: int):
    """Load image as RGB thumbnail of (thumb_px, thumb_px); placeholder on fail."""
    from PIL import Image
    try:
        im = Image.open(path).convert("RGB")
    except Exception:
        im = Image.new("RGB", (thumb_px, thumb_px), (128, 128, 128))
        return im
    im.thumbnail((thumb_px, thumb_px))
    # pad to square so the grid is regular
    canvas = Image.new("RGB", (thumb_px, thumb_px), (32, 32, 32))
    ox = (thumb_px - im.width) // 2
    oy = (thumb_px - im.height) // 2
    canvas.paste(im, (ox, oy))
    return canvas


def _font(size: int):
    """Return a TTF font of `size` if any standard one resolves, else default."""
    from PIL import ImageFont
    for candidate in (
        "DejaVuSans.ttf",
        "DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ):
        try:
            return ImageFont.truetype(candidate, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _wrap_text(text: str, font, max_width: int) -> list[str]:
    """Word-wrap `text` so each line's rendered width ≤ max_width."""
    from PIL import ImageDraw, Image
    draw = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    words = text.split()
    lines: list[str] = []
    cur = ""
    for w in words:
        candidate = (cur + " " + w).strip()
        bbox = draw.textbbox((0, 0), candidate, font=font)
        if bbox[2] - bbox[0] <= max_width:
            cur = candidate
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def save_composite_png(
    data: dict,
    out_path,
    *,
    thumb_px: int,
    max_per_set: int,
    reward_model: "str | None" = None,
    manifest_cache: "dict | None" = None,
) -> None:
    """Render one PNG showing, per prompt, Set A row and Set B row, plus proposals."""
    from PIL import Image, ImageDraw
    from collections import defaultdict

    images: list[dict] = data.get("images", [])
    proposals: list[str] = data.get("proposals", []) or []
    by_group: dict[int, dict[str, dict[str, list[str]]]] = defaultdict(
        lambda: defaultdict(lambda: {"A": [], "B": []})
    )
    for img in images:
        g = int(img.get("group", -1))
        prompt = img.get("prompt_text", "<missing>")
        s = img.get("set", "?")
        path = img.get("image_path", "")
        if s in ("A", "B"):
            by_group[g][prompt][s].append(path)

    # Layout constants
    pad = 32
    label_col_w = 140   # for "Set A" / "Set B" labels next to thumb rows
    thumb_gap = 6
    section_gap = 28
    reward_gap = 6      # space between thumb and the reward caption under it

    title_font  = _font(30)
    header_font = _font(24)
    label_font  = _font(28)
    body_font   = _font(20)
    reward_font = _font(18)

    # Canvas width is based on the LARGEST actually-rendered row in this call
    # (capped by max_per_set). This avoids reserving space for thumbs that
    # never appear and avoids "the leftmost thumbs look pushed off-frame
    # because the canvas was sized for an upper bound" effects.
    actual_max = 1
    for sets_per_prompt in by_group.values():
        for sets in sets_per_prompt.values():
            actual_max = max(actual_max,
                             min(len(sets["A"]), max_per_set),
                             min(len(sets["B"]), max_per_set))
    row_w = label_col_w + actual_max * thumb_px + max(0, actual_max - 1) * thumb_gap
    canvas_w = pad * 2 + row_w

    # First pass: figure out total canvas height (title + groups + proposals)
    # Title block
    step = data.get("step_idx"); topic = data.get("topic_id"); call = data.get("call_idx")
    model = data.get("proposer_model", "<?>")
    title = (
        f"step={step}  topic={topic}  call={call}  model={model}  "
        f"effort={data.get('reasoning_effort','?')}  "
        f"cluster='{data.get('cluster_summary','')}'"
    )
    title_lines = _wrap_text(title, title_font, canvas_w - 2 * pad)
    title_h = sum(_line_height(title_font) for _ in title_lines) + section_gap

    # Each group block height. When reward_model is given, we reserve one
    # extra line of `reward_font` under every thumbnail to print the score.
    reward_caption_h = (_line_height(reward_font) + reward_gap) if reward_model else 0
    group_blocks: list[tuple[int, str, list[str], list[str], int]] = []
    # (group_id, prompt_text, A_paths, B_paths, block_height)
    total_groups_h = 0
    for g in sorted(by_group):
        for prompt, sets in by_group[g].items():
            prompt_lines = _wrap_text(
                f"[{g}] {prompt}", header_font, canvas_w - 2 * pad,
            )
            prompt_h = len(prompt_lines) * _line_height(header_font)
            row_h = thumb_px + reward_caption_h
            block_h = prompt_h + 6 + row_h + 4 + row_h + section_gap
            group_blocks.append((
                g, prompt, sets["A"][:max_per_set], sets["B"][:max_per_set], block_h,
            ))
            total_groups_h += block_h

    # Proposals block
    prop_header_h = _line_height(header_font) + 4
    prop_lines: list[str] = []
    for i, a in enumerate(proposals, start=1):
        for line in _wrap_text(f"[{i}] {a}", body_font, canvas_w - 2 * pad):
            prop_lines.append(line)
    prop_h = prop_header_h + len(prop_lines) * _line_height(body_font) + section_gap

    canvas_h = pad + title_h + total_groups_h + prop_h + pad
    canvas = Image.new("RGB", (canvas_w, canvas_h), (250, 250, 250))
    draw = ImageDraw.Draw(canvas)

    y = pad
    for line in title_lines:
        draw.text((pad, y), line, fill=(0, 0, 0), font=title_font)
        y += _line_height(title_font)
    y += section_gap - _line_height(title_font)  # consume the trailing pad once

    # Per-group blocks
    for g, prompt, a_paths, b_paths, _ in group_blocks:
        # prompt header
        for line in _wrap_text(f"[{g}] {prompt}", header_font, canvas_w - 2 * pad):
            draw.text((pad, y), line, fill=(0, 0, 0), font=header_font)
            y += _line_height(header_font)
        y += 6

        def _draw_row(paths: list[str], label_text: str, label_color: tuple[int, int, int]) -> None:
            """Render one Set A or Set B row at the current `y`. Advances `y`."""
            nonlocal y
            # label, right-aligned in the label column
            bbox = draw.textbbox((0, 0), label_text, font=label_font)
            label_w = bbox[2] - bbox[0]
            draw.text(
                (pad + label_col_w - 12 - label_w,
                 y + thumb_px // 2 - _line_height(label_font) // 2),
                label_text, fill=label_color, font=label_font,
            )
            x = pad + label_col_w
            for p in paths:
                canvas.paste(_load_thumb(p, thumb_px), (x, y))
                if reward_model is not None:
                    r = _resolve_reward(p, reward_model, manifest_cache or {})
                    caption = (f"{reward_model}: {r:.3f}" if isinstance(r, (int, float))
                               else f"{reward_model}: —")
                    cb = draw.textbbox((0, 0), caption, font=reward_font)
                    cap_w = cb[2] - cb[0]
                    cap_x = x + (thumb_px - cap_w) // 2
                    cap_y = y + thumb_px + reward_gap
                    draw.text(
                        (cap_x, cap_y), caption,
                        fill=(40, 40, 40), font=reward_font,
                    )
                x += thumb_px + thumb_gap
            y += thumb_px + reward_caption_h

        _draw_row(a_paths, "Set A", (180, 0, 0))
        y += 4
        _draw_row(b_paths, "Set B", (0, 0, 180))
        y += section_gap

    # Proposals
    draw.text((pad, y), f"Proposed attributes ({len(proposals)}):",
              fill=(0, 80, 0), font=header_font)
    y += _line_height(header_font) + 4
    for line in prop_lines:
        draw.text((pad, y), line, fill=(0, 0, 0), font=body_font)
        y += _line_height(body_font)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def _line_height(font) -> int:
    """Rough line height for the given font (handles default-font case)."""
    try:
        bbox = font.getbbox("Mg")
        return (bbox[3] - bbox[1]) + 4
    except Exception:
        return 16


def show_one(
    call_path: Path,
    *,
    max_paths_per_set: int,
    hide_paths: bool,
    show_full_prompt: bool,
    show_reasoning: bool,
) -> None:
    data = json.loads(call_path.read_text())

    step = data.get("step_idx")
    topic = data.get("topic_id")
    call = data.get("call_idx")
    model = data.get("proposer_model", "<unknown>")
    effort = data.get("reasoning_effort", "<unknown>")
    cluster_summary = data.get("cluster_summary", "")
    selected_prompts: list[str] = data.get("selected_prompts", [])
    current_pool: list[str] = data.get("current_pool", []) or []
    avoid_attrs: list[str] = data.get("avoid_attrs", []) or []
    images: list[dict] = data.get("images", [])
    proposals: list[str] = data.get("proposals", []) or []

    _print_header(
        f"step={step}  topic={topic}  call={call}  "
        f"model={model}  reasoning_effort={effort}"
    )
    print(f"file           : {call_path}")
    print(f"cluster_summary: {cluster_summary}")
    print(f"selected_prompts ({len(selected_prompts)}):")
    for i, p in enumerate(selected_prompts, start=1):
        print(f"  [{i}] {p}")

    if current_pool:
        print(f"\ncurrent_pool ({len(current_pool)}):")
        for a in current_pool:
            print(f"  - {a}")
    if avoid_attrs:
        print(f"\navoid_attrs ({len(avoid_attrs)}):")
        for a in avoid_attrs:
            print(f"  - {a}")

    # Group images by (group, prompt, set)
    by_group: dict[int, dict[str, dict[str, list[str]]]] = defaultdict(
        lambda: defaultdict(lambda: {"A": [], "B": []})
    )
    for img in images:
        g = int(img.get("group", -1))
        prompt = img.get("prompt_text", "<missing>")
        s = img.get("set", "?")
        path = img.get("image_path", "")
        if s in ("A", "B"):
            by_group[g][prompt][s].append(path)

    print(f"\n── per-prompt set A / set B  ({len(by_group)} groups) ──")
    for g in sorted(by_group):
        for prompt, sets in by_group[g].items():
            n_a = len(sets["A"])
            n_b = len(sets["B"])
            print(f"\n  group {g}: '{prompt}'")
            print(f"    Set A: {n_a} images")
            if not hide_paths:
                for p in sets["A"][:max_paths_per_set]:
                    print(f"      {p}")
                if n_a > max_paths_per_set:
                    print(f"      ... ({n_a - max_paths_per_set} more)")
            print(f"    Set B: {n_b} images")
            if not hide_paths:
                for p in sets["B"][:max_paths_per_set]:
                    print(f"      {p}")
                if n_b > max_paths_per_set:
                    print(f"      ... ({n_b - max_paths_per_set} more)")

    if show_full_prompt:
        print("\n── proposer_prompt (verbatim) ──")
        print(data.get("proposer_prompt", "<empty>"))

    if show_reasoning:
        reasoning = data.get("reasoning") or data.get("reasoning_content") or ""
        if reasoning:
            print("\n── proposer reasoning ──")
            print(reasoning)

    print(f"\n── proposed attributes ({len(proposals)}) ──")
    for i, a in enumerate(proposals, start=1):
        print(f"  [{i}] {a}")

    usage = data.get("usage") or {}
    if usage:
        print(
            f"\nusage: input={usage.get('input_tokens')} "
            f"output={usage.get('output_tokens')} "
            f"total={usage.get('total_tokens')}"
        )

    print()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", type=Path,
                    help="Path to a proposer_*.json file or a dir containing them.")
    ap.add_argument("--max_paths_per_set", type=int, default=4,
                    help="Max image paths shown per Set A / Set B (default: 4).")
    ap.add_argument("--hide_paths", action="store_true",
                    help="Don't list individual image paths (just sizes).")
    ap.add_argument("--show_full_prompt", action="store_true",
                    help="Print the verbatim proposer_prompt text.")
    ap.add_argument("--show_reasoning", action="store_true",
                    help="Print the proposer's reasoning field.")
    ap.add_argument("--out_dir", type=Path, default=None,
                    help="Where to save composite PNGs (one per call). If "
                         "omitted, each PNG is saved next to its source JSON "
                         "(i.e. in the JSON's parent directory).")
    ap.add_argument("--thumb_px", type=int, default=220,
                    help="Thumbnail edge length used in the composite PNG.")
    ap.add_argument("--reward_model", type=str, default=None,
                    help="If set, look up reward_scores[<name>] from each "
                         "image's sibling manifest.json and print it under "
                         "the thumbnail (e.g. pickscore, imagereward).")
    args = ap.parse_args()
    manifest_cache: dict = {}

    files = _iter_call_files(args.path)
    for f in files:
        show_one(
            f,
            max_paths_per_set=args.max_paths_per_set,
            hide_paths=args.hide_paths,
            show_full_prompt=args.show_full_prompt,
            show_reasoning=args.show_reasoning,
        )
        out_dir = args.out_dir if args.out_dir is not None else f.parent
        data = json.loads(f.read_text())
        png_path = out_dir / (f.stem + ".png")
        save_composite_png(
            data, png_path,
            thumb_px=args.thumb_px,
            max_per_set=args.max_paths_per_set,
            reward_model=args.reward_model,
            manifest_cache=manifest_cache,
        )
        print(f"  saved composite PNG → {png_path}")


if __name__ == "__main__":
    main()
