#!/usr/bin/env python3
"""
Single-query style transfer via GPT-Image-2.

Example:
  python tools/query_gpt_image.py \
    --input-image sim_layout.png \
    --style-image real_style.png \
    --prompt "Transfer style while preserving geometry."

Requires:
  pip install openai
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sim2real import GPTImageTranslator

_DEFAULT_MODEL = "gpt-image-2"
_DEFAULT_QUALITY = "high"
_DEFAULT_SIZE = "auto"
_DEFAULT_MAX_SIDE = 1024
_DEFAULT_BLEND_ALPHA = 0.5


def _default_output_dir(input_path: Path) -> Path:
    return Path("outputs") / f"gpt_image_{input_path.stem}"


def _read_rgb_image(path: Path) -> Image.Image:
    if not path.is_file():
        raise FileNotFoundError(f"Image file not found: {path}")
    return Image.open(path).convert("RGB")


def _get_font(size: int = 18):
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
    ]:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _save_comparison_panel(
    *,
    query: Image.Image,
    style: Image.Image,
    gpt_output: Image.Image,
    output_path: Path,
    blend_alpha: float = _DEFAULT_BLEND_ALPHA,
) -> None:
    columns: list[tuple[str, Image.Image]] = [
        ("Query", query),
        ("Style", style),
        ("GPT-Image-2", gpt_output),
    ]
    columns.append(
        (
            "Overlay (Query + GPT-Image-2)",
            Image.blend(query, gpt_output.resize(query.size, Image.LANCZOS), blend_alpha),
        )
    )

    font = _get_font(16)
    padding = 8
    label_h = 30
    images = [img.resize(query.size, Image.LANCZOS) for _, img in columns]
    canvas_w = len(images) * query.width + (len(images) - 1) * padding
    canvas_h = label_h + query.height
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(30, 30, 30))
    draw = ImageDraw.Draw(canvas)

    for idx, ((label, _), img) in enumerate(zip(columns, images, strict=False)):
        x = idx * (query.width + padding)
        canvas.paste(img, (x, label_h))
        bbox = draw.textbbox((0, 0), label, font=font)
        text_w = bbox[2] - bbox[0]
        draw.text((x + (query.width - text_w) // 2, 6), label, fill="white", font=font)

    canvas.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Single-query style transfer via GPT-Image-2")
    parser.add_argument("--input-image", type=Path, required=True, help="Query image whose layout should be preserved")
    parser.add_argument("--style-image", type=Path, required=True, help="Reference image whose appearance should guide the result")
    parser.add_argument("--prompt", type=str, required=True, help="Edit prompt sent to GPT-Image-2")
    args = parser.parse_args()

    input_rgb = _read_rgb_image(args.input_image)
    style_rgb = _read_rgb_image(args.style_image)

    output_dir = _default_output_dir(args.input_image)
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = args.input_image.stem
    translated_path = output_dir / f"{stem}_translated.png"
    comparison_path = output_dir / f"{stem}_comparison.png"

    translator = GPTImageTranslator(
        style_references=[style_rgb],
        prompt=args.prompt,
        model=_DEFAULT_MODEL,
        size=_DEFAULT_SIZE,
        quality=_DEFAULT_QUALITY,
        max_side=_DEFAULT_MAX_SIDE,
    )

    translated_np = translator.translate(np.asarray(input_rgb))
    translated_rgb = Image.fromarray(translated_np, mode="RGB")
    translated_rgb.save(translated_path)

    _save_comparison_panel(
        query=input_rgb,
        style=style_rgb,
        gpt_output=translated_rgb,
        output_path=comparison_path,
    )
    print(f"Saved {translated_path}")
    print(f"Saved {comparison_path}")


if __name__ == "__main__":
    main()
