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
import threading
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


def _to_pil(img) -> Image.Image:
    """Convert a genai Image or PIL Image to a guaranteed PIL.Image.Image."""
    import io
    if isinstance(img, Image.Image):
        return img
    # google-genai Image (Pydantic model) — try known byte attributes
    for attr in ("image_bytes", "_image_bytes", "data"):
        raw = getattr(img, attr, None)
        if isinstance(raw, bytes) and raw:
            pil_img = Image.open(io.BytesIO(raw))
            pil_img.load()
            return pil_img
    # Pydantic model_dump fallback
    if hasattr(img, "model_dump"):
        d = img.model_dump()
        for key in ("image_bytes", "data"):
            raw = d.get(key)
            if isinstance(raw, bytes) and raw:
                pil_img = Image.open(io.BytesIO(raw))
                pil_img.load()
                return pil_img
    raise TypeError(f"Cannot convert {type(img).__name__} to PIL Image")


def _extract_pil_image(response) -> Image.Image:
    """Extract the last generated PIL Image from a Gemini response (no file I/O)."""
    import io

    parts = getattr(response, "parts", None)
    if parts is None and response.candidates:
        parts = response.candidates[0].content.parts
    if not parts:
        raise RuntimeError("Empty response from model")

    last_img = None
    img_count = 0
    for part in parts:
        if part.inline_data is not None:
            raw = getattr(part.inline_data, "data", None)
            if isinstance(raw, bytes) and raw:
                img_count += 1
                pil = Image.open(io.BytesIO(raw))
                pil.load()
                last_img = pil
            else:
                genai_img = part.as_image()
                if genai_img is not None:
                    img_count += 1
                    last_img = _to_pil(genai_img)
        if part.text:
            print(f"  Gemini: {part.text}")

    if last_img is None:
        raise RuntimeError(
            f"No image generated (got {img_count} image parts)"
        )
    return last_img


def generate_fewshot_from_images(
    example_pairs: list[tuple[Image.Image, Image.Image]],
    query_image: Image.Image,
    *,
    cam_name: str = "wrist",
    prompt: str | None = None,
    # model: str = "gemini-3-pro-image-preview",
    model: str = "gemini-3-pro-image-preview",
    api_key: str | None = None,
    temperature: float | None = None,
    seed: int | None = None,
) -> Image.Image:
    """In-memory few-shot: takes PIL Images, returns PIL Image (no file I/O).

    Used by GeminiTranslator for real-time sim→real style transfer.
    """
    from google.genai import types

    client = _init_client(api_key)

    if prompt is None:
        prompt = _PROMPTS.get(cam_name, _PROMPTS["wrist"])

    contents: list = [prompt]
    for i, (gs_img, real_img) in enumerate(example_pairs, 1):
        contents.append(f"\nexample {i} — render:")
        contents.append(gs_img)
        contents.append(f"\nexample {i} — real photo:")
        contents.append(real_img)

    contents.append(_QUERY_LABELS.get(cam_name, _QUERY_LABELS["wrist"]))
    contents.append(query_image)

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
    return _extract_pil_image(response)


class GeminiTranslator:
    """Pre-loaded Gemini few-shot translator for sim→real style transfer.

    Loads example pairs at init, then translates numpy images on demand.
    Usage from deploy_act_policy_mujoco.py:
        translator = GeminiTranslator()
        real_np = translator.translate(composite_np, "stationary")
    """

    def __init__(
        self,
        stationary_pairs: int = 1,
        wrist_pairs: int = 3,
        model: str = "gemini-3-pro-image-preview",
        temperature: float = 0,
        seed: int = 42,
        api_key: str | None = None,
        stationary_dir: str = "calibration_pairs_stationary",
        wrist_dir: str = "calibration_pairs_wrist",
    ):
        self._model = model
        self._temperature = temperature
        self._seed = seed
        self._api_key = api_key

        self._pairs: dict[str, list[tuple[Image.Image, Image.Image]]] = {}
        for cam_name, n_pairs, base_dir in [
            ("stationary", stationary_pairs, stationary_dir),
            ("wrist", wrist_pairs, wrist_dir),
        ]:
            pairs = load_calibration_pairs_pil(cam_name, n_pairs, base_dir)
            self._pairs[cam_name] = pairs
            print(f"  [GeminiTranslator] Loaded {len(pairs)} example pair(s) for {cam_name}")

    def translate(self, image_np, cam_name: str):
        """Translate a numpy RGB uint8 image via Gemini few-shot.

        Returns numpy RGB uint8 image of the same size.
        """
        import numpy as np
        h, w = image_np.shape[:2]
        query = Image.fromarray(image_np)
        result = generate_fewshot_from_images(
            self._pairs[cam_name],
            query,
            cam_name=cam_name,
            model=self._model,
            temperature=self._temperature,
            seed=self._seed,
            api_key=self._api_key,
        )
        result = _to_pil(result)
        result = result.convert("RGB").resize((w, h), Image.LANCZOS)
        return np.array(result)


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

# Few-shot example frame indices per camera (GeminiTranslator + CLI must match).
EXAMPLE_FRAME_INDICES = {
    "stationary": [0, 2, 4],
    "wrist": [1, 300, 500],
}


def load_calibration_pairs_pil(
    cam_name: str, n_pairs: int, base_dir: str | Path
) -> list[tuple[Image.Image, Image.Image]]:
    base = Path(base_dir)
    indices = EXAMPLE_FRAME_INDICES.get(cam_name, list(range(n_pairs)))[:n_pairs]
    pairs = []
    for i in indices:
        gs = base / "gs_renders" / f"frame_{i:04d}.png"
        real = base / "real_captures" / f"frame_{i:04d}.png"
        pairs.append((Image.open(gs).convert("RGB"), Image.open(real).convert("RGB")))
    return pairs


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
        default=None,
        help="Number of example pairs per camera (overridden by --stationary-n or --wrist-n if set). Uses frames 0..N-1 as examples and frame N as query.",
    )
    parser.add_argument(
        "--n1",
        type=int,
        default=None,
        help="Number of example pairs for stationary camera (n1).",
    )
    parser.add_argument(
        "--n2",
        type=int,
        default=None,
        help="Number of example pairs for wrist camera (n2).",
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

    out_dir = args.output or Path("visual_match/gemini")
    out_dir.mkdir(parents=True, exist_ok=True)

    base_seeds = {
        "stationary": args.stationary_seed if args.stationary_seed is not None else args.seed,
        "wrist": args.wrist_seed if args.wrist_seed is not None else args.seed,
    }

    # If n1 or n2 are set, they imply stationary/wrist selection
    stationary_selected = args.stationary or (args.n1 is not None)
    wrist_selected = args.wrist or (args.n2 is not None)
    use_both = (not stationary_selected and not wrist_selected) or (stationary_selected and wrist_selected)
    cam_configs = []
    if use_both or stationary_selected:
        cam_configs.append(("stationary", args.stationary_dir))
    if use_both or wrist_selected:
        cam_configs.append(("wrist", args.wrist_dir))

    cam_n = {
        "stationary": args.n1 if args.n1 is not None else (args.num_pairs if args.num_pairs is not None else 1),
        "wrist": args.n2 if args.n2 is not None else (args.num_pairs if args.num_pairs is not None else 1),
    }

    cam_pairs: dict[str, list[tuple[Path, Path]]] = {}
    cam_queries: dict[str, Path] = {}
    for cam_name, cam_dir in cam_configs:
        n_cam = cam_n[cam_name]
        gs_dir = cam_dir / "gs_renders"
        real_dir = cam_dir / "real_captures"
        example_indices = EXAMPLE_FRAME_INDICES.get(cam_name, list(range(n_cam)))[:n_cam]
        pairs = [
            (gs_dir / f"frame_{i:04d}.png", real_dir / f"frame_{i:04d}.png")
            for i in example_indices
        ]
        query_idx = (
            300
            if cam_name in ("stationary", "wrist")
            else (max(example_indices) + 1 if example_indices else n_cam)
        )
        query_path = gs_dir / f"frame_{query_idx:04d}.png"
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
            thread_results = {}
            threads = []

            def run_cam(cam_name, cam_dir, temp, seed, pred_path):
                try:
                    n_cam = cam_n[cam_name]
                    pil_pairs = load_calibration_pairs_pil(cam_name, n_cam, cam_dir)
                    query = Image.open(cam_queries[cam_name]).convert("RGB")
                    result = generate_fewshot_from_images(
                        pil_pairs,
                        query,
                        cam_name=cam_name,
                        prompt=args.fewshot_prompt,
                        model=args.model,
                        api_key=args.api_key,
                        temperature=temp,
                        seed=seed,
                    )
                    result = _to_pil(result).convert("RGB")
                    pred_path.parent.mkdir(parents=True, exist_ok=True)
                    result.save(str(pred_path))
                    print(f"  Saved: {pred_path}")
                    thread_results[cam_name] = {
                        "name": cam_name,
                        "pairs": cam_pairs[cam_name],
                        "query": cam_queries[cam_name],
                        "predicted": pred_path,
                        "seed": seed,
                        "temperature": temp,
                    }
                except RuntimeError as e:
                    print(f"  [{cam_name}] FAILED (temp={temp}, seed={seed}): {e}")

            # If both cameras, run in parallel (even if --stationary/--wrist not set, but n1 and n2 are)
            if len(cam_configs) == 2:
                for cam_name, cam_dir in cam_configs:
                    seed = seeds[cam_name]
                    if multi_run:
                        suffix = f"_t{temp}_seed{seed}"
                    else:
                        suffix = ""
                    pred_path = out_dir / f"predicted_{cam_name}{suffix}.png"
                    print(f"\n  [{cam_name}] temp={temp}, seed={seed}, query={cam_queries[cam_name]}")
                    t = threading.Thread(target=run_cam, args=(cam_name, cam_dir, temp, seed, pred_path))
                    threads.append(t)
                    t.start()
                for t in threads:
                    t.join()
                for cam_name in [c[0] for c in cam_configs]:
                    if cam_name in thread_results:
                        cam_results.append(thread_results[cam_name])
            else:
                # Single camera, run sequentially
                for cam_name, cam_dir in cam_configs:
                    seed = seeds[cam_name]
                    if multi_run:
                        suffix = f"_t{temp}_seed{seed}"
                    else:
                        suffix = ""
                    pred_path = out_dir / f"predicted_{cam_name}{suffix}.png"
                    print(f"\n  [{cam_name}] temp={temp}, seed={seed}, query={cam_queries[cam_name]}")
                    run_cam(cam_name, cam_dir, temp, seed, pred_path)
                for cam_name in [c[0] for c in cam_configs]:
                    if cam_name in thread_results:
                        cam_results.append(thread_results[cam_name])

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
