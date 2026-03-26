#!/usr/bin/env python3
"""
Query Google Gemini API for few-shot render -> real image generation.

Given N example (rendered, real) pairs from calibration folders, Gemini
learns the visual mapping and generates the predicted real capture for
a query rendered image.  Runs for stationary and/or wrist cameras.

Usage:
  python tools/query_gemini.py -n 3                          # both cameras
  python tools/query_gemini.py -n 3 --stationary             # stationary only
  python tools/query_gemini.py -n 3 --wrist                  # wrist only
  python tools/query_gemini.py -n 3 -t 0 --seed 42          # deterministic
  python tools/query_gemini.py -n 3 --stationary-seed 10 --wrist-seed 42
  python tools/query_gemini.py -n 3 --retries 10             # auto-find working seed

Outputs to visual_match/gemini/:
  predicted_stationary.png, predicted_wrist.png, comparison.png

Requires: pip install google-genai Pillow
API key: Set GEMINI_API_KEY in .env (pip install python-dotenv), or as env var, or pass --api-key
"""
import argparse
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from PIL import Image, ImageDraw, ImageFont


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


def _get_font(size: int = 18):
    """Try to load a TrueType font, fall back to default."""
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
    ]:
        try:
            return ImageFont.truetype(path, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def create_comparison_stack(
    cam_results: list[dict],
    output_path: Path,
    title: str = "",
    alpha: float = 0.5,
    padding: int = 6,
    label_h: int = 30,
) -> Path:
    """Create a multi-row comparison stack image.

    Each row: blend(pair_1) | real_1 | query | predicted | blend(query, predicted)
    One row per camera (stationary, wrist).
    Title (e.g. seed info) is drawn at the top-right.

    cam_results: list of dicts with keys:
        name, pairs [(gs, real), ...], query (Path), predicted (Path), seed (int)
    """
    font = _get_font(16)
    font_large = _get_font(20)

    rows_images: list[list[Image.Image]] = []
    col_labels: list[str] = ["Example Blend", "Real Capture", "Query", "Predicted", "Overlay"]

    for cam in cam_results:
        row: list[Image.Image] = []

        gs_path, real_path = cam["pairs"][0]
        gs = Image.open(gs_path).convert("RGB")
        real = Image.open(real_path).convert("RGB").resize(gs.size, Image.LANCZOS)
        row.append(Image.blend(gs, real, alpha=alpha))
        row.append(real)

        query = Image.open(cam["query"]).convert("RGB")
        predicted = Image.open(cam["predicted"]).convert("RGB")
        query_r = query.resize(predicted.size, Image.LANCZOS)

        row.append(query_r)
        row.append(predicted)
        row.append(Image.blend(query_r, predicted, alpha=alpha))

        rows_images.append(row)

    all_imgs = [img for row in rows_images for img in row]
    cell_w = max(img.width for img in all_imgs)
    cell_h = max(img.height for img in all_imgs)

    num_cols = max(len(row) for row in rows_images)
    num_rows = len(rows_images)
    row_labels = [
        f"{cam['name'].capitalize()} (seed={cam['seed']})"
        for cam in cam_results
    ]

    row_label_w = 200
    title_h = 36 if title else 0
    canvas_w = row_label_w + num_cols * cell_w + (num_cols - 1) * padding
    canvas_h = title_h + label_h + num_rows * cell_h + (num_rows - 1) * padding

    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(30, 30, 30))
    draw = ImageDraw.Draw(canvas)

    if title:
        draw.text((canvas_w // 2, 8), title, fill=(255, 220, 100),
                  font=font_large, anchor="mt")

    col_header_y = title_h + 6
    for c, label in enumerate(col_labels):
        x = row_label_w + c * (cell_w + padding) + cell_w // 2
        bbox = draw.textbbox((0, 0), label, font=font)
        tw = bbox[2] - bbox[0]
        draw.text((x - tw // 2, col_header_y), label, fill="white", font=font)

    for r, row in enumerate(rows_images):
        y_base = title_h + label_h + r * (cell_h + padding)

        bbox = draw.textbbox((0, 0), row_labels[r], font=font_large)
        th = bbox[3] - bbox[1]
        draw.text((8, y_base + cell_h // 2 - th // 2), row_labels[r],
                  fill="white", font=font_large)

        for c, img in enumerate(row):
            resized = img.resize((cell_w, cell_h), Image.LANCZOS)
            x = row_label_w + c * (cell_w + padding)
            canvas.paste(resized, (x, y_base))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(str(output_path))
    print(f"Comparison stack saved: {output_path}")
    return output_path


_PROMPTS = {
    "stationary": "Below are some views of the scene.",
    "wrist": "Below are some top down views of the scene.",
}

_QUERY_LABELS = {
    "stationary": "\n new render(generate its real photo):",
    "wrist": "\n new render:(generate its real photo)",
}


def generate_fewshot(
    pairs: list[tuple[Path, Path]],
    query_image: str | Path,
    output_path: str | Path,
    *,
    cam_name: str = "wrist",
    prompt: str | None = None,
    model: str = "gemini-2.5-flash-image",
    api_key: str | None = None,
    temperature: float | None = None,
    seed: int | None = None,
) -> Path:
    """
    Few-shot mode: provide (rendered, real) example pairs, then a query
    rendered image. Gemini learns the render->real mapping and generates
    the predicted real capture for the query.
    """
    from google.genai import types

    client = _init_client(api_key)

    if prompt is None:
        prompt = _PROMPTS.get(cam_name, _PROMPTS["wrist"])

    contents: list = [prompt]
    for i, (gs_path, real_path) in enumerate(pairs, 1):
        contents.append(f"\nexample {i} — render:")
        contents.append(Image.open(gs_path).convert("RGB"))
        contents.append(f"\nexample {i} — real photo:")
        contents.append(Image.open(real_path).convert("RGB"))

    contents.append(_QUERY_LABELS.get(cam_name, _QUERY_LABELS["wrist"]))
    print(f"contents: {contents}")
    contents.append(Image.open(query_image).convert("RGB"))

    cfg = dict(response_modalities=["TEXT", "IMAGE"])
    if temperature is not None:
        cfg["temperature"] = temperature
    if seed is not None:
        cfg["seed"] = seed

    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=types.GenerateContentConfig(**cfg),
    )
    return _extract_image(response, Path(output_path))


def main():
    parser = argparse.ArgumentParser(
        description="Few-shot render→real image generation via Gemini API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tools/query_gemini.py -n 3                                # both cameras
  python tools/query_gemini.py -n 3 --stationary                   # stationary only
  python tools/query_gemini.py -n 3 -t 0 --seed 42                # deterministic
  python tools/query_gemini.py -n 3 --retries 5                    # sweep seeds 42-46
  python tools/query_gemini.py -n 3 -t 0 0.3 0.7 1.0              # sweep temperatures
  python tools/query_gemini.py -n 3 -t 0 0.5 1.0 --retries 3      # sweep both (3×3=9 runs)
  python tools/query_gemini.py -n 3 --stationary-seed 10 --wrist-seed 42
        """,
    )

    parser.add_argument(
        "-n", "--num-pairs",
        type=int,
        required=True,
        help="Number of example pairs per camera. Uses frames 0..N-1 as "
             "examples and frame N as query.",
    )
    parser.add_argument(
        "--stationary",
        action="store_true",
        help="Include stationary camera. If neither --stationary nor --wrist "
             "is given, both are included.",
    )
    parser.add_argument(
        "--wrist",
        action="store_true",
        help="Include wrist camera. If neither --stationary nor --wrist "
             "is given, both are included.",
    )
    parser.add_argument(
        "--stationary-dir",
        type=Path,
        default=Path("calibration_pairs_stationary"),
        help="Stationary camera calibration directory (default: calibration_pairs_stationary)",
    )
    parser.add_argument(
        "--wrist-dir",
        type=Path,
        default=Path("calibration_pairs_wrist"),
        help="Wrist camera calibration directory (default: calibration_pairs_wrist)",
    )
    parser.add_argument(
        "--fewshot-prompt",
        type=str,
        default=None,
        help="Override the default few-shot system prompt",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output directory (default: visual_match/gemini)",
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
    parser.add_argument(
        "-t", "--temperature",
        type=float,
        nargs="+",
        default=[0],
        help="Sampling temperature(s). Pass multiple to sweep: -t 0 0.3 0.7 1.0",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Default random seed for both cameras (default: 42).",
    )
    parser.add_argument(
        "--stationary-seed",
        type=int,
        default=None,
        help="Override seed for stationary camera only.",
    )
    parser.add_argument(
        "--wrist-seed",
        type=int,
        default=None,
        help="Override seed for wrist camera only.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=1,
        help="Number of attempts per camera. On failure, seed is incremented "
             "and retried. The working seed is printed (default: 1, no retry).",
    )
    args = parser.parse_args()

    n = args.num_pairs
    out_dir = args.output or Path("visual_match/gemini")
    out_dir.mkdir(parents=True, exist_ok=True)

    base_seeds = {
        "stationary": args.stationary_seed if args.stationary_seed is not None else args.seed,
        "wrist": args.wrist_seed if args.wrist_seed is not None else args.seed,
    }

    use_both = not args.stationary and not args.wrist
    cam_configs = []
    if use_both or args.stationary:
        cam_configs.append(("stationary", args.stationary_dir))
    if use_both or args.wrist:
        cam_configs.append(("wrist", args.wrist_dir))

    cam_pairs: dict[str, list[tuple[Path, Path]]] = {}
    cam_queries: dict[str, Path] = {}
    for cam_name, cam_dir in cam_configs:
        gs_dir = cam_dir / "gs_renders"
        real_dir = cam_dir / "real_captures"
        pairs = [
            (gs_dir / f"frame_{i:04d}.png", real_dir / f"frame_{i:04d}.png")
            for i in range(n)
        ]
        query_path = gs_dir / f"frame_{n:04d}.png"
        for gs_p, real_p in pairs:
            if not gs_p.exists():
                parser.error(f"Not found: {gs_p}")
            if not real_p.exists():
                parser.error(f"Not found: {real_p}")
        if not query_path.exists():
            parser.error(f"Query image not found: {query_path}")
        cam_pairs[cam_name] = pairs
        cam_queries[cam_name] = query_path

    temperatures = args.temperature
    multi_run = args.retries > 1 or len(temperatures) > 1
    total_runs = len(temperatures) * args.retries
    run_idx = 0

    for temp in temperatures:
        for attempt in range(args.retries):
            run_idx += 1
            seeds = {
                cam_name: base_seeds[cam_name] + attempt
                for cam_name, _ in cam_configs
            }

            print(f"\n{'#' * 60}")
            print(f"  Run {run_idx}/{total_runs}  |  temp={temp}  |  "
                  + "  ".join(f"{name}: seed={seeds[name]}" for name, _ in cam_configs))
            print(f"{'#' * 60}")

            cam_results = []
            for cam_name, cam_dir in cam_configs:
                seed = seeds[cam_name]
                if multi_run:
                    suffix = f"_t{temp}_seed{seed}"
                else:
                    suffix = ""
                pred_path = out_dir / f"predicted_{cam_name}{suffix}.png"

                print(f"\n  [{cam_name}] temp={temp}, seed={seed}, query={cam_queries[cam_name]}")
                try:
                    generate_fewshot(
                        cam_pairs[cam_name],
                        cam_queries[cam_name],
                        pred_path,
                        cam_name=cam_name,
                        prompt=args.fewshot_prompt,
                        model=args.model,
                        api_key=args.api_key,
                        temperature=temp,
                        seed=seed,
                    )
                    print(f"  Saved: {pred_path}")
                    cam_results.append({
                        "name": cam_name,
                        "pairs": cam_pairs[cam_name],
                        "query": cam_queries[cam_name],
                        "predicted": pred_path,
                        "seed": seed,
                        "temperature": temp,
                    })
                except RuntimeError as e:
                    print(f"  [{cam_name}] FAILED (temp={temp}, seed={seed}): {e}")

            if cam_results:
                seed_tag = "_".join(f"{r['name'][0]}{r['seed']}" for r in cam_results)
                if multi_run:
                    stack_name = f"comparison_t{temp}_{seed_tag}.png"
                else:
                    stack_name = "comparison.png"
                stack_path = out_dir / stack_name

                title_parts = [
                    f"{r['name'].capitalize()}: seed={r['seed']}, t={r['temperature']}"
                    for r in cam_results
                ]
                title = "  |  ".join(title_parts)
                create_comparison_stack(cam_results, stack_path, title=title)
            else:
                print("  No cameras succeeded for this run, skipping comparison.")


if __name__ == "__main__":
    main()
