"""Sanity-test the detector's `use_applicability` mode.

For each attribute in a `pairs.json` produced by `hamming_pair_finder`, we
randomly sample a few (image, prompt) candidates and call the VLM twice:

  1) With `use_applicability=True`  → expects `{applicable, present, ...}`
  2) Without applicability         → expects `{present, ...}`            (control)

We collect raw JSON, parse `applicable` / `present` / `reasoning` from both,
and print a per-attr summary plus a detailed JSON report.

Run:
    python -m analysis.test_detector_applicability \
        --pairs_json outputs/hamming_pairs/20260604-104300/pairs.json \
        --detector_vllm_base_url http://localhost:8000/v1 \
        --n_samples_per_attr 4

Outputs:
  • console summary:   one line per attr (counts + a few example reasonings)
  • JSON report:       saved next to pairs.json by default
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from caller import ChatHistory, ChatMessage, LocalCaller  # noqa: E402
from search.prompts.detection import (  # noqa: E402
    ATTRIBUTE_DETECTION_SYSTEM,
    build_detection_prompt,
)


# ── Schemas ──────────────────────────────────────────────────────────────────


@dataclass
class _ParsedResp:
    """Single-call result with the full JSON fields the model returned."""
    raw_response: str | None = None
    applicable: bool | None = None     # only present in applicability mode
    present: bool | None = None
    confidence: float | None = None
    reasoning: str | None = None
    parse_error: str | None = None


@dataclass
class _SampleResult:
    attr: str
    prompt_text: str
    image_path: str
    image_id: str
    image_role: str                    # "pos" (g=1) | "neg" (g=0)
    with_app: _ParsedResp = field(default_factory=_ParsedResp)
    without_app: _ParsedResp = field(default_factory=_ParsedResp)


@dataclass
class _AttrSummary:
    attr: str
    n_samples: int
    # applicability mode counts
    with_app_present: int = 0          # applicable=true, present=true
    with_app_not_present: int = 0      # applicable=true, present=false
    with_app_na: int = 0               # applicable=false
    with_app_parse_fail: int = 0
    # control mode counts
    without_app_present: int = 0
    without_app_not_present: int = 0
    without_app_parse_fail: int = 0
    # cross-mode disagreement
    flipped_when_app: int = 0          # control says present=true; app says applicable=false OR present=false


# ── Detect helpers ──────────────────────────────────────────────────────────


async def _call_one_batch(
    caller: LocalCaller,
    model_name: str,
    samples: "list[_SampleResult]",
    attribute: str,
    *,
    use_applicability: bool,
    use_reasoning: bool,
    extra_body: dict | None,
    max_parallel: int,
    max_tokens: int,
) -> list[_ParsedResp]:
    """One detector call per sample, fully batched, returning parsed JSON fields."""
    histories: list[ChatHistory] = []
    for s in samples:
        user_text = build_detection_prompt(
            attribute=attribute,
            prompt=s.prompt_text,
            use_prompt=False,
            use_reasoning=use_reasoning,
            use_applicability=use_applicability,
        )
        img_url = ChatMessage.image_to_base64_url(s.image_path)
        # vLLM ordering matches search/models/judge/vlm_judge.py:vllm path
        content = [
            {"type": "input_image", "image_url": img_url},
            {"type": "input_text",  "text": user_text},
        ]
        histories.append(ChatHistory(messages=[
            ChatMessage(role="system", content=ATTRIBUTE_DETECTION_SYSTEM),
            ChatMessage(role="user",   content=content),
        ]))

    raw = await caller.call(
        messages=histories,
        model=model_name,
        max_parallel=max_parallel,
        max_tokens=max_tokens,
        temperature=0.7,
        top_p=0.8,
        presence_penalty=1.5,
        extra_body=extra_body,
    )

    out: list[_ParsedResp] = []
    for r in raw:
        if r is None or not getattr(r, "has_response", False):
            out.append(_ParsedResp(parse_error="no response"))
            continue
        text = r.first_response or ""
        try:
            m = re.search(r"\{[\s\S]*\}", text)
            if not m:
                out.append(_ParsedResp(raw_response=text, parse_error="no JSON object"))
                continue
            data = json.loads(m.group())
            out.append(_ParsedResp(
                raw_response=text,
                applicable=data.get("applicable"),
                present=data.get("present"),
                confidence=data.get("confidence"),
                reasoning=data.get("reasoning"),
            ))
        except Exception as e:
            out.append(_ParsedResp(raw_response=text, parse_error=str(e)))
    return out


# ── Sampling ─────────────────────────────────────────────────────────────────


def _sample_pool_for_attr(
    by_prompt: dict[str, list[dict]],
    n: int,
    seed_mix: int,
) -> "list[tuple[str, str, str, str]]":
    """Return up to `n` (prompt_text, image_path, image_id, role) tuples.

    We expand each pair into a `pos` row and a `neg` row, then sample.
    """
    rows: list[tuple[str, str, str, str]] = []
    for prompt_text, pair_list in by_prompt.items():
        for pair in pair_list:
            rows.append((prompt_text, pair["pos_image_path"], pair["pos_image_id"], "pos"))
            rows.append((prompt_text, pair["neg_image_path"], pair["neg_image_id"], "neg"))
    rng = random.Random(seed_mix)
    rng.shuffle(rows)
    return rows[:n]


# ── CLI ──────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--pairs_json", type=Path, required=True,
                   help="Output of hamming_pair_finder (pairs.json).")
    p.add_argument("--detector_model", default="Qwen/Qwen3.5-9B")
    p.add_argument("--detector_vllm_base_url", default="http://localhost:8000/v1")
    p.add_argument("--n_samples_per_attr", type=int, default=4,
                   help="(image, prompt) tuples sampled per attr (pos + neg images mixed).")
    p.add_argument("--max_attrs", type=int, default=0,
                   help="If >0, only test this many attrs (in pairs.json order).")
    p.add_argument("--target_attr", default=None,
                   help="Case-insensitive substring filter on attrs (single attr inspection).")
    p.add_argument("--max_parallel", type=int, default=8)
    p.add_argument("--max_tokens", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use_reasoning", action="store_true",
                   help="Ask the model for a 'reasoning' field. Default OFF (faster).")
    p.add_argument("--out_path", type=Path, default=None,
                   help="Default: pairs.json's parent / 'applicability_test_<timestamp>'/report.json.")
    p.add_argument("--no_thumbnails", action="store_true",
                   help="Skip per-attr image grid rendering.")
    p.add_argument("--thumb_px", type=int, default=256)
    p.add_argument("--n_examples_per_attr", type=int, default=2,
                   help="How many reasoning examples per category to print on console "
                        "(only when --use_reasoning is on).")
    return p.parse_args()


# ── Main ─────────────────────────────────────────────────────────────────────


async def amain() -> None:
    args = parse_args()
    with open(args.pairs_json) as f:
        pairs = json.load(f)
    by_attr: dict[str, dict[str, list[dict]]] = pairs["by_attr"]

    # Attr filter
    attrs = list(by_attr.keys())
    if args.target_attr:
        needle = args.target_attr.strip().lower()
        attrs = [a for a in attrs if needle in a.lower()]
        if not attrs:
            logger.warning(f"no attr matched --target_attr={args.target_attr!r}")
            return
    if args.max_attrs > 0:
        attrs = attrs[:args.max_attrs]
    logger.info(f"testing {len(attrs)} attrs × {args.n_samples_per_attr} samples")

    # vLLM caller (always local for this test)
    caller = LocalCaller(base_url=args.detector_vllm_base_url, cache_config=None)
    extra_body: dict = {
        "top_k": 20,
        "chat_template_kwargs": {"enable_thinking": False},
    }

    # Output directory: <pairs.json parent> / "applicability_test_<ts>"
    ts = __import__("datetime").datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = (
        args.out_path.parent if args.out_path
        else args.pairs_json.parent / f"applicability_test_{ts}"
    )
    grids_dir = out_dir / "grids"
    if not args.no_thumbnails:
        grids_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"per-attr grids → {grids_dir}")

    summaries: list[_AttrSummary] = []
    all_samples: list[_SampleResult] = []
    samples_by_attr: dict[str, list[_SampleResult]] = {}

    try:
        for attr_idx, attr in enumerate(attrs, 1):
            sampled = _sample_pool_for_attr(
                by_attr[attr], args.n_samples_per_attr,
                seed_mix=args.seed ^ (hash(attr) & 0xFFFFFFFF),
            )
            if not sampled:
                logger.warning(f"  [{attr_idx}/{len(attrs)}] attr={attr[:60]!r} no candidates — skip")
                continue
            samples = [
                _SampleResult(
                    attr=attr, prompt_text=p, image_path=ip,
                    image_id=iid, image_role=role,
                )
                for (p, ip, iid, role) in sampled
            ]

            # 1) applicability mode
            apps = await _call_one_batch(
                caller, args.detector_model, samples, attr,
                use_applicability=True,
                use_reasoning=args.use_reasoning,
                extra_body=extra_body,
                max_parallel=args.max_parallel, max_tokens=args.max_tokens,
            )
            # 2) control (no applicability)
            ctrls = await _call_one_batch(
                caller, args.detector_model, samples, attr,
                use_applicability=False,
                use_reasoning=args.use_reasoning,
                extra_body=extra_body,
                max_parallel=args.max_parallel, max_tokens=args.max_tokens,
            )
            for s, a, c in zip(samples, apps, ctrls):
                s.with_app = a
                s.without_app = c

            # Aggregate
            summ = _AttrSummary(attr=attr, n_samples=len(samples))
            for s in samples:
                a, c = s.with_app, s.without_app
                # with_app
                if a.parse_error is not None:
                    summ.with_app_parse_fail += 1
                elif a.applicable is False:
                    summ.with_app_na += 1
                elif a.present is True:
                    summ.with_app_present += 1
                else:
                    summ.with_app_not_present += 1
                # without_app
                if c.parse_error is not None:
                    summ.without_app_parse_fail += 1
                elif c.present is True:
                    summ.without_app_present += 1
                else:
                    summ.without_app_not_present += 1
                # disagreement: control says g=1 but app mode says NOT g=1
                if (c.parse_error is None and c.present is True and
                        a.parse_error is None and
                        (a.applicable is False or a.present is False)):
                    summ.flipped_when_app += 1
            summaries.append(summ)
            all_samples.extend(samples)
            samples_by_attr[attr] = samples

            if not args.no_thumbnails:
                out_png = grids_dir / f"{_attr_slug(attr)}.png"
                _render_per_attr_grid(attr, samples, out_png, thumb_px=args.thumb_px)

            logger.info(
                f"  [{attr_idx}/{len(attrs)}] {attr[:60]!r}  "
                f"n={summ.n_samples}  "
                f"app:[present={summ.with_app_present} not={summ.with_app_not_present} "
                f"na={summ.with_app_na} fail={summ.with_app_parse_fail}]  "
                f"ctrl:[present={summ.without_app_present} not={summ.without_app_not_present} "
                f"fail={summ.without_app_parse_fail}]  "
                f"flipped(ctrl=1→app≠1)={summ.flipped_when_app}"
            )
    finally:
        try:
            await caller.shutdown()
        except Exception as e:
            logger.warning(f"caller shutdown failed: {e}")

    if args.use_reasoning:
        _print_console_examples(all_samples, summaries, args.n_examples_per_attr)
    _print_global_table(summaries)
    _save_report(args, attrs, summaries, all_samples, out_dir, grids_dir)


# ── Reporters ────────────────────────────────────────────────────────────────


# ── Per-attr image grid ──────────────────────────────────────────────────────


def _attr_slug(attr: str) -> str:
    import hashlib
    return hashlib.sha1(attr.encode("utf-8")).hexdigest()[:8]


def _load_font(size: int):
    from PIL import ImageFont
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ):
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                pass
    return ImageFont.load_default()


def _wrap_text(text: str, max_chars: int) -> list[str]:
    if max_chars <= 0:
        return [text]
    lines: list[str] = []
    cur = ""
    for w in text.split():
        while len(w) > max_chars:
            if cur:
                lines.append(cur); cur = ""
            lines.append(w[:max_chars])
            w = w[max_chars:]
        if not cur:
            cur = w
        elif len(cur) + 1 + len(w) <= max_chars:
            cur += " " + w
        else:
            lines.append(cur); cur = w
    if cur:
        lines.append(cur)
    return lines or [""]


def _fmt_app(r: _ParsedResp) -> str:
    if r.parse_error is not None:
        return "FAIL"
    if r.applicable is False:
        return "N/A"
    if r.present is True:
        return "present=1"
    if r.present is False:
        return "present=0"
    return "?"


def _fmt_ctrl(r: _ParsedResp) -> str:
    if r.parse_error is not None:
        return "FAIL"
    if r.present is True:
        return "present=1"
    if r.present is False:
        return "present=0"
    return "?"


def _agreement_tag(a: _ParsedResp, c: _ParsedResp) -> tuple[str, tuple[int, int, int]]:
    """Compact tag + colour summarising how the two modes compare on this image."""
    if a.parse_error is not None or c.parse_error is not None:
        return "—", (120, 120, 120)
    if a.applicable is False and c.present is True:
        return "ctrl=1, app=N/A", (200, 110, 30)         # orange — applicability rescued FP
    if a.applicable is False and c.present is False:
        return "ctrl=0, app=N/A", (140, 140, 140)        # gray — both negative-ish
    if a.present == c.present:
        return "agree", (40, 120, 40)                    # green
    return f"ctrl→{int(bool(c.present))} app→{int(bool(a.present))}", (180, 30, 30)


def _render_per_attr_grid(
    attr: str,
    samples: "list[_SampleResult]",
    out_path: Path,
    thumb_px: int = 256,
) -> None:
    """One mega-PNG per attribute.  Each row = one tested image with:
       (image thumbnail | prompt text | role + app-mode + control-mode + agreement).
    """
    from PIL import Image, ImageDraw

    if not samples:
        return

    pad         = 12
    prompt_w    = 360
    anno_w      = 320
    title_line_h = 22
    body_line_h  = 16
    body_font = _load_font(13)
    title_font = _load_font(18)
    anno_bold = _load_font(13)

    cell_w = thumb_px + pad + prompt_w + pad + anno_w
    grid_w = cell_w + 2 * pad

    title_max_chars = max(40, grid_w // 9)
    title_lines = _wrap_text(attr, title_max_chars)
    # summary line: agreement / flipped / N/A counts
    n = len(samples)
    n_na = sum(1 for s in samples if s.with_app.applicable is False)
    n_flip = sum(
        1 for s in samples
        if s.without_app.parse_error is None and s.without_app.present is True
        and s.with_app.parse_error is None
        and (s.with_app.applicable is False or s.with_app.present is False)
    )
    n_agree = sum(
        1 for s in samples
        if s.without_app.parse_error is None and s.with_app.parse_error is None
        and s.with_app.applicable is not False
        and s.with_app.present == s.without_app.present
    )
    stats_line = f"n={n}  agree={n_agree}  flipped(ctrl=1→app≠1)={n_flip}  N/A={n_na}"

    title_h = title_line_h * len(title_lines) + title_line_h + pad

    prompt_max_chars = max(20, prompt_w // 7)
    anno_max_chars = max(20, anno_w // 8)

    row_blocks = []
    for s in samples:
        plines = _wrap_text(s.prompt_text, prompt_max_chars)
        agree_str, agree_color = _agreement_tag(s.with_app, s.without_app)
        anno_lines: list[tuple[str, tuple[int, int, int]]] = []
        anno_lines.append((f"role: {s.image_role}", (40, 40, 40)))
        anno_lines.append((f"app:  {_fmt_app(s.with_app)}", (20, 100, 20) if s.with_app.applicable is not False else (200, 110, 30)))
        if s.with_app.confidence is not None:
            anno_lines.append((f"   conf {s.with_app.confidence:.2f}", (120, 120, 120)))
        anno_lines.append((f"ctrl: {_fmt_ctrl(s.without_app)}", (140, 30, 30) if s.without_app.present else (30, 30, 140)))
        if s.without_app.confidence is not None:
            anno_lines.append((f"   conf {s.without_app.confidence:.2f}", (120, 120, 120)))
        anno_lines.append((agree_str, agree_color))
        anno_lines.append((f"img: {s.image_id[:14]}", (80, 80, 80)))
        text_h = max(body_line_h * len(plines), body_line_h * len(anno_lines))
        row_h = max(thumb_px, text_h) + pad
        row_blocks.append((s, plines, anno_lines, row_h))

    total_h = pad + title_h + pad + sum(b[3] for b in row_blocks) + pad
    canvas = Image.new("RGB", (grid_w, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    y = pad
    for line in title_lines:
        draw.text((pad, y), line, fill=(0, 0, 0), font=title_font)
        y += title_line_h
    draw.text((pad, y), stats_line, fill=(100, 100, 100), font=title_font)
    y += title_line_h + pad

    for s, plines, anno_lines, row_h in row_blocks:
        draw.line([(pad, y - 2), (grid_w - pad, y - 2)], fill=(225, 225, 225), width=1)

        # Image (left)
        img_y = y + max(0, (row_h - pad - thumb_px) // 2)
        try:
            im = Image.open(s.image_path).convert("RGB")
            im.thumbnail((thumb_px, thumb_px))
        except Exception as e:
            logger.warning(f"  cannot read {s.image_path}: {e}")
            im = Image.new("RGB", (thumb_px, thumb_px), (200, 200, 200))
        canvas.paste(im, (pad, img_y))

        # Prompt text (middle)
        ty = y + max(0, (row_h - pad - body_line_h * len(plines)) // 2)
        px = pad + thumb_px + pad
        for line in plines:
            draw.text((px, ty), line, fill=(0, 0, 0), font=body_font)
            ty += body_line_h

        # Annotation column (right)
        ax = pad + thumb_px + pad + prompt_w + pad
        ay = y + max(0, (row_h - pad - body_line_h * len(anno_lines)) // 2)
        for line, color in anno_lines:
            for wrapped in _wrap_text(line, anno_max_chars):
                draw.text((ax, ay), wrapped, fill=color, font=anno_bold)
                ay += body_line_h

        y += row_h

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    logger.info(f"  grid → {out_path}")


def _print_console_examples(
    samples: list[_SampleResult],
    summaries: list[_AttrSummary],
    n_examples: int,
) -> None:
    """For attrs where the model said N/A or flipped, show reasoning excerpts."""
    print("\n── Example reasonings (applicability mode) ──")
    by_attr = {s.attr: [] for s in samples}
    for s in samples:
        by_attr[s.attr].append(s)
    for summ in summaries:
        if summ.with_app_na == 0 and summ.flipped_when_app == 0:
            continue
        rows = by_attr.get(summ.attr, [])
        na_rows = [r for r in rows if r.with_app.applicable is False][:n_examples]
        flip_rows = [
            r for r in rows
            if r.without_app.present is True
            and (r.with_app.applicable is False or r.with_app.present is False)
        ][:n_examples]
        if not na_rows and not flip_rows:
            continue
        print(f"\nAttr: {summ.attr[:120]!r}")
        for r in na_rows:
            print(f"  · N/A   img={Path(r.image_path).stem}  role={r.image_role}")
            print(f"      prompt:    {r.prompt_text[:100]!r}")
            print(f"      reasoning: {(r.with_app.reasoning or '')[:200]}")
        for r in flip_rows:
            print(f"  · FLIP  img={Path(r.image_path).stem}  role={r.image_role}  "
                  f"ctrl.present=True → app.applicable={r.with_app.applicable} "
                  f"app.present={r.with_app.present}")
            print(f"      reasoning: {(r.with_app.reasoning or '')[:200]}")


def _print_global_table(summaries: list[_AttrSummary]) -> None:
    """Concise overall stats."""
    if not summaries:
        return
    n_attrs = len(summaries)
    n_total = sum(s.n_samples for s in summaries)
    sum_app_present = sum(s.with_app_present for s in summaries)
    sum_app_not = sum(s.with_app_not_present for s in summaries)
    sum_app_na = sum(s.with_app_na for s in summaries)
    sum_ctrl_present = sum(s.without_app_present for s in summaries)
    sum_ctrl_not = sum(s.without_app_not_present for s in summaries)
    sum_flipped = sum(s.flipped_when_app for s in summaries)
    print("\n── Global counts ──")
    print(f"  attrs           = {n_attrs}")
    print(f"  total samples   = {n_total}")
    print(f"  applicability mode:")
    print(f"    present       = {sum_app_present}  ({100 * sum_app_present / max(n_total, 1):.1f}%)")
    print(f"    not present   = {sum_app_not}  ({100 * sum_app_not / max(n_total, 1):.1f}%)")
    print(f"    not applicable= {sum_app_na}  ({100 * sum_app_na / max(n_total, 1):.1f}%)")
    print(f"  control mode (no applicability):")
    print(f"    present       = {sum_ctrl_present}  ({100 * sum_ctrl_present / max(n_total, 1):.1f}%)")
    print(f"    not present   = {sum_ctrl_not}  ({100 * sum_ctrl_not / max(n_total, 1):.1f}%)")
    print(f"  flipped (ctrl=present → app ≠ present): "
          f"{sum_flipped}  ({100 * sum_flipped / max(sum_ctrl_present, 1):.1f}% of ctrl-present)")


def _save_report(
    args: argparse.Namespace,
    attrs: list[str],
    summaries: list[_AttrSummary],
    samples: list[_SampleResult],
    out_dir: Path,
    grids_dir: Path,
) -> None:
    out = args.out_path or (out_dir / "report.json")
    payload: dict[str, Any] = {
        "meta": {
            "pairs_json": str(args.pairs_json),
            "detector_model": args.detector_model,
            "detector_vllm_base_url": args.detector_vllm_base_url,
            "n_samples_per_attr": args.n_samples_per_attr,
            "n_attrs_tested": len(summaries),
            "use_reasoning": bool(args.use_reasoning),
            "grids_dir": str(grids_dir) if not args.no_thumbnails else None,
            "command_line": " ".join(sys.argv),
        },
        "per_attr": [asdict(s) for s in summaries],
        "samples": [
            {
                "attr": s.attr,
                "prompt_text": s.prompt_text,
                "image_path": s.image_path,
                "image_id": s.image_id,
                "image_role": s.image_role,
                "with_app": asdict(s.with_app),
                "without_app": asdict(s.without_app),
            }
            for s in samples
        ],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"\nreport → {out}")
    if not args.no_thumbnails:
        print(f"grids  → {grids_dir}")


def main() -> None:
    asyncio.run(amain())


if __name__ == "__main__":
    main()