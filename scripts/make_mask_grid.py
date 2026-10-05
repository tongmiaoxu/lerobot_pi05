#!/usr/bin/env python3
"""Render a 5-column thumbnail grid of one role's (real_A/fake_B/real_B) saved masks for a
single pix2pix/pix2pix-turbo result set, for visual sanity-checking after a review pass.

Each cell shows the frame with its saved mask(s) overlaid (from mask_overrides.json, via
pix2pix_mask_common.get_override_mask — never re-detected, so this reflects exactly what
eval_pix2pix_metrics.py will score). A green cell border means at least one object's mask was
found; a red border means every object is unresolved (no saved mask, and not confirmed-missing
either) for that frame — those are the frames still worth reviewing.

Usage:
  python scripts/make_mask_grid.py \\
    --images-dir outputs/turbo_sim2real_all_tasks_gan_only/results/place_mug_stationary_turbo_gan_only/test_latest/images \\
    --text-prompt "mug, saucer" --role fake_B \\
    --out data_mask/turbo_sim2real_all_tasks_gan_only/place_mug_stationary_turbo_gan_only_fake_B.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "scripts"))
from pix2pix_mask_common import (  # noqa: E402
    default_overrides_path,
    draw_candidates,
    find_triplet_indices,
    get_override_mask,
    load_overrides,
    parse_object_list,
)

ROLES = ("real_A", "fake_B", "real_B")
THUMB_W, THUMB_H = 220, 165
PAD = 6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images-dir", type=Path, required=True, help="Folder of <idx>_real_A/fake_B/real_B.png triplets.")
    parser.add_argument(
        "--text-prompt", type=str, default=None,
        help="Object(s), comma-separated, e.g. 'mug, saucer'. Not needed with --no-mask.",
    )
    parser.add_argument("--role", choices=ROLES, default="fake_B")
    parser.add_argument("--out", type=Path, required=True, help="Output grid PNG path (parent dirs created as needed).")
    parser.add_argument("--columns", type=int, default=5)
    parser.add_argument(
        "--overrides-json", type=Path, default=None,
        help="Default: <images-dir>/../mask_overrides.json (shared with eval_pix2pix_metrics.py).",
    )
    parser.add_argument(
        "--no-mask", action="store_true",
        help="Plain frames only, no mask overlay/border coloring (and --text-prompt is not required).",
    )
    parser.add_argument("--title", type=str, default=None, help="Caption burned into the grid; default derived from --images-dir/--role.")
    return parser.parse_args()


def build_grid(
    images_dir: Path, overrides: dict, indices: list[str], objects: list[str], role: str, columns: int,
    no_mask: bool = False,
) -> np.ndarray:
    thumbs = []
    for idx in indices:
        img = cv2.imread(str(images_dir / f"{idx}_{role}.png"))
        if no_mask:
            vis = img.copy()
            border_color = (128, 128, 128)
        else:
            masks = [m for m in (get_override_mask(overrides, idx, obj, role) for obj in objects) if m is not None]
            vis = draw_candidates(img, masks) if masks else img.copy()
            border_color = (0, 200, 0) if masks else (0, 0, 255)
        thumb = cv2.resize(vis, (THUMB_W, THUMB_H))
        thumb = cv2.copyMakeBorder(thumb, PAD, PAD, PAD, PAD, cv2.BORDER_CONSTANT, value=border_color)
        cv2.putText(thumb, idx, (PAD + 4, PAD + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        thumbs.append(thumb)
    cell_h, cell_w = thumbs[0].shape[:2]
    rows = (len(thumbs) + columns - 1) // columns
    grid = np.zeros((rows * cell_h, columns * cell_w, 3), dtype=np.uint8)
    for i, t in enumerate(thumbs):
        r, c = divmod(i, columns)
        grid[r * cell_h : (r + 1) * cell_h, c * cell_w : (c + 1) * cell_w] = t
    return grid


def main() -> int:
    args = parse_args()
    if not args.no_mask and not args.text_prompt:
        raise SystemExit("--text-prompt is required unless --no-mask is set")
    objects = parse_object_list(args.text_prompt) if args.text_prompt else []
    overrides = {}
    if not args.no_mask:
        overrides_path = args.overrides_json or default_overrides_path(args.images_dir)
        overrides = load_overrides(overrides_path)
    indices = find_triplet_indices(args.images_dir)

    grid = build_grid(args.images_dir, overrides, indices, objects, args.role, args.columns, no_mask=args.no_mask)
    title = args.title or f"{args.images_dir.parent.parent.name} - {args.role} (n={len(indices)})"
    cv2.putText(grid, title, (10, grid.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), grid)
    print(f"Wrote {args.out} (n={len(indices)} frames)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
