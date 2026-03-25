#!/usr/bin/env python3
"""
Query Google Gemini API for image generation with an input image.

Uses Gemini's native image generation models (Nano Banana) which support
text + image -> image editing. Supports:
- gemini-2.5-flash-image: Fast, good for high-volume tasks
- gemini-3-pro-image-preview: Higher quality, up to 4K resolution

Modes:
  Single-image mode (default):
    python query_gemini.py <image> "<prompt>" -o out.png

  Few-shot mode (--pairs):
    Provide example (rendered, real) pairs, then a query rendered image.
    Gemini learns the render->real mapping from examples and generates
    the predicted real capture for the query.

    python query_gemini.py --pairs \\
        gs_renders/frame_0001.png real_captures/frame_0001.png \\
        gs_renders/frame_0002.png real_captures/frame_0002.png \\
        --query gs_renders/frame_0003.png \\
        -o predicted_real_0003.png

Requires: pip install google-genai Pillow
API key: Set GEMINI_API_KEY in .env (pip install python-dotenv), or as env var, or pass --api-key
"""
import argparse
import os
from pathlib import Path

# Load .env from project root (when run as python tools/query_gemini.py)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from PIL import Image


def _init_client(api_key: str | None = None):
    from google import genai

    key = api_key or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise ValueError(
            "API key required. Set GEMINI_API_KEY env var or pass --api-key"
        )
    return genai.Client(api_key=key)


def _extract_image(response, output_path: Path) -> Path:
    """Save the *last* generated image from a Gemini response.

    Gemini image-generation models may echo back input images as earlier
    parts before appending the actual generated image at the end.  By
    taking the last image part we avoid saving an echoed input.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    parts = getattr(response, "parts", None)
    if parts is None and response.candidates:
        parts = response.candidates[0].content.parts
    if not parts:
        raise RuntimeError("Empty response from model")

    last_img = None
    img_count = 0
    for part in parts:
        if part.inline_data is not None:
            img = part.as_image()
            if img is not None:
                img_count += 1
                last_img = img
        if part.text:
            print(f"Model response: {part.text}")

    print(f"  Response contained {img_count} image(s), using the last one.")

    if last_img is None:
        raise RuntimeError("No image was generated in the response")

    last_img.save(str(output_path))
    return output_path


def create_overlay(
    query_path: Path,
    predicted_path: Path,
    real_path: Path | None,
    alpha: float = 0.5,
    output_path: Path | None = None,
) -> Path:
    """Left: alpha*query + (1-alpha)*predicted.  Right (if real exists): alpha*real + (1-alpha)*predicted."""
    predicted = Image.open(predicted_path).convert("RGB")
    query = Image.open(query_path).convert("RGB").resize(predicted.size, Image.LANCZOS)

    left = Image.blend(predicted, query, alpha=alpha)

    if real_path is not None and real_path.exists():
        real = Image.open(real_path).convert("RGB").resize(predicted.size, Image.LANCZOS)
        right = Image.blend(predicted, real, alpha=alpha)
        w, h = predicted.size
        canvas = Image.new("RGB", (w * 2, h))
        canvas.paste(left, (0, 0))
        canvas.paste(right, (w, 0))
    else:
        canvas = left

    if output_path is None:
        output_path = predicted_path.parent / f"{predicted_path.stem}_overlay{predicted_path.suffix}"
    canvas.save(str(output_path))
    print(f"  Overlay saved: {output_path}")
    return output_path


def generate_image(
    image_path: str | Path,
    prompt: str,
    output_path: str | Path,
    *,
    model: str = "gemini-2.5-flash-image",
    api_key: str | None = None,
) -> Path:
    """Single-image mode: one input image + text prompt -> generated image."""
    from google.genai import types

    client = _init_client(api_key)
    image = Image.open(image_path).convert("RGB")

    full_prompt = f"{prompt}\n\nUse the provided image as reference for this request."

    response = client.models.generate_content(
        model=model,
        contents=[full_prompt, image],
        config=types.GenerateContentConfig(
            response_modalities=["TEXT", "IMAGE"],
        ),
    )
    return _extract_image(response, Path(output_path))


def generate_fewshot(
    pairs: list[tuple[Path, Path]],
    query_image: str | Path,
    output_path: str | Path,
    *,
    prompt: str | None = None,
    model: str = "gemini-2.5-flash-image",
    api_key: str | None = None,
) -> Path:
    """
    Few-shot mode: provide (rendered, real) example pairs, then a query
    rendered image. Gemini learns the render->real mapping and generates
    the predicted real capture for the query.
    """
    from google.genai import types

    client = _init_client(api_key)

    if prompt is None:
        prompt = (
           "I will show you example pairs of images. In each pair, the first "
            "image is a synthetic rendered view and the "
            "second image is the corresponding real camera capture of the same "
            "The color changes from rendered to real"
            "Then I will give you one more rendered image. Generate the "
            "corresponding real camera capture, keeping all the object positions same and only changing the colors."
        )

    contents: list = [prompt]
    for i, (gs_path, real_path) in enumerate(pairs, 1):
        contents.append(f"\n--- Example pair {i} ---\nRendered image:")
        contents.append(Image.open(gs_path).convert("RGB"))
        contents.append("Corresponding real capture:")
        contents.append(Image.open(real_path).convert("RGB"))

    contents.append(
        "\n--- Query ---\nHere is a new rendered image. "
        "Generate the corresponding real camera capture:"
    )
    contents.append(Image.open(query_image).convert("RGB"))

    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=types.GenerateContentConfig(
            response_modalities=["TEXT", "IMAGE"],
        ),
    )
    return _extract_image(response, Path(output_path))


def main():
    parser = argparse.ArgumentParser(
        description="Generate images via Gemini API with input image + prompt",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single-image mode
  python query_gemini.py my_image.png "enhance this image" -o out.png

  # Few-shot render->real mode
  python query_gemini.py --pairs \\
      calibration_pairs_wrist/gs_renders/frame_0001.png \\
      calibration_pairs_wrist/real_captures/frame_0001.png \\
      calibration_pairs_wrist/gs_renders/frame_0002.png \\
      calibration_pairs_wrist/real_captures/frame_0002.png \\
      --query calibration_pairs_wrist/gs_renders/frame_0003.png \\
      -o predicted_real_0003.png
        """,
    )

    parser.add_argument(
        "image",
        nargs="?",
        type=Path,
        default=None,
        help="(Single-image mode) Path to input image",
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        default="Here is a sloth toy lying on the table. Lift its right arm up.",
        help="(Single-image mode) Text prompt for image generation",
    )

    parser.add_argument(
        "--pairs",
        nargs="+",
        type=Path,
        metavar="IMG",
        help="(Few-shot mode) Alternating rendered/real image paths: "
             "gs1.png real1.png gs2.png real2.png ...",
    )
    parser.add_argument(
        "--query",
        type=Path,
        help="(Few-shot mode) Query rendered image to translate to real",
    )
    parser.add_argument(
        "--fewshot-prompt",
        type=str,
        default=None,
        help="(Few-shot mode) Override the default system prompt",
    )

    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output image path",
    )
    parser.add_argument(
        "-m", "--model",
        default="gemini-3-pro-image-preview",
        choices=["gemini-2.5-flash-image", "gemini-3-pro-image-preview"],
        help="Model: gemini-2.5-flash-image (fast) or gemini-3-pro-image-preview (higher quality)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Gemini API key (default: GEMINI_API_KEY env var)",
    )
    args = parser.parse_args()

    # ── Few-shot mode ──
    if args.pairs is not None:
        if len(args.pairs) % 2 != 0:
            parser.error("--pairs requires an even number of images (rendered/real alternating)")
        if args.query is None:
            parser.error("--query is required when using --pairs")

        pair_paths = list(args.pairs)
        pairs = [(pair_paths[i], pair_paths[i + 1]) for i in range(0, len(pair_paths), 2)]

        for p in pair_paths + [args.query]:
            if not p.exists():
                parser.error(f"Image not found: {p}")

        output_path = args.output
        if output_path is None:
            output_path = args.query.parent / f"predicted_real_{args.query.stem}{args.query.suffix}"

        print(f"Few-shot mode: {len(pairs)} example pair(s), query={args.query}")
        out = generate_fewshot(
            pairs,
            args.query,
            output_path,
            prompt=args.fewshot_prompt,
            model=args.model,
            api_key=args.api_key,
        )
        print(f"Saved: {out}")

        real_gt = args.query.parent.parent / "real_captures" / args.query.name
        create_overlay(args.query, out, real_gt)
        return

    # ── Single-image mode ──
    if args.image is None:
        parser.error("Provide an image path, or use --pairs/--query for few-shot mode")
    if not args.image.exists():
        parser.error(f"Image not found: {args.image}")

    output_path = args.output
    if output_path is None:
        output_path = args.image.parent / f"gemini_{args.image.stem}{args.image.suffix}"

    out = generate_image(
        args.image,
        args.prompt,
        output_path,
        model=args.model,
        api_key=args.api_key,
    )
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
