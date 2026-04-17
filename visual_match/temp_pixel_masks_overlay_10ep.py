"""
Temporary: build a pixel_masks_overlay-style image using only
  - first 5 episodes after excluding episode 0  → ep 1,2,3,4,5
  - last 5 episodes in the dataset            → ep N-5 .. N-1

Reads cached assets from initial_states/ (no ML). Output:
  initial_states/mug/pixel_masks_overlay_first5_last5.png
"""

import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
META = ROOT / "data" / "meta" / "info.json"
OUT_DIR = Path(__file__).resolve().parent / "initial_states"
OBJ = "mug"

FRAMES_DIR = OUT_DIR / "individual_frames"
MASKS_DIR = OUT_DIR / OBJ / "individual_masks"
MEDIAN_BG = OUT_DIR / "median_background.png"
OUT_PATH = OUT_DIR / OBJ / "pixel_masks_overlay_first5_last5.png"


def _episode_indices(total_episodes: int) -> list[int]:
    """First five excluding ep0 (1..5) and last five, unique in that order."""
    first = list(range(1, min(6, total_episodes)))
    lo = max(0, total_episodes - 5)
    last = list(range(lo, total_episodes))
    seen: set[int] = set()
    ordered: list[int] = []
    for e in first + last:
        if e in seen:
            continue
        seen.add(e)
        ordered.append(e)
    return ordered


def main() -> None:
    with open(META) as f:
        total_episodes = int(json.load(f)["total_episodes"])

    eps = _episode_indices(total_episodes)
    print(f"total_episodes={total_episodes}  overlay episodes ({len(eps)}): {eps}")

    if not MEDIAN_BG.is_file():
        raise FileNotFoundError(f"Need {MEDIAN_BG} — run initial_states_overlay.py first.")

    median_bg = np.array(Image.open(MEDIAN_BG).convert("RGB"))
    h, w = median_bg.shape[:2]

    frames: list[np.ndarray] = []
    masks: list[np.ndarray | None] = []

    for ep in eps:
        frame_path = FRAMES_DIR / f"ep_{ep:03d}.png"
        mask_path = MASKS_DIR / f"ep_{ep:03d}_mask.png"
        if not frame_path.is_file():
            raise FileNotFoundError(frame_path)
        if not mask_path.is_file():
            print(f"  [WARN] missing mask {mask_path}, skipping episode {ep}")
            frames.append(np.zeros((h, w, 3), dtype=np.uint8))
            masks.append(None)
            continue

        frame_rgb = np.array(Image.open(frame_path).convert("RGB"))
        mask_u8 = np.array(Image.open(mask_path))
        if mask_u8.ndim == 3:
            mask_u8 = mask_u8[..., 0]
        mask = mask_u8 > 127
        if frame_rgb.shape[:2] != (h, w):
            raise ValueError(f"Frame size mismatch ep={ep}: {frame_rgb.shape[:2]} vs median {h,w}")
        frames.append(frame_rgb)
        masks.append(mask)

    valid = sum(1 for m in masks if m is not None and m.any())
    if valid == 0:
        raise RuntimeError("No valid masks in selection.")

    # Same alpha recipe as initial_states_overlay.py
    alpha_per_ep = min(0.5, 20 / max(valid, 1))
    pixel_overlay = median_bg.astype(np.float64).copy()
    for m, frame_rgb in zip(masks, frames):
        if m is None or not m.any():
            continue
        pixel_overlay[m] = (
            (1 - alpha_per_ep) * pixel_overlay[m]
            + alpha_per_ep * frame_rgb[m].astype(np.float64)
        )
    pixel_overlay = np.clip(pixel_overlay, 0, 255).astype(np.uint8)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixel_overlay).save(OUT_PATH)
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()
