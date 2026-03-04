#!/usr/bin/env python3
"""
Color calibration script for wrist camera images.
Uses image pairs from calibration_pairs_wrist to learn a single affine color transform
(sim -> real), then applies it to all images and saves calibrated versions.

Run compare_recorded_vs_mujoco.py --save-calibration-pairs first to generate the pairs.
"""

import os
from pathlib import Path
import numpy as np
from PIL import Image


def _get_aug(x: np.ndarray, add_ones: bool = True) -> np.ndarray:
    """
    Augment input features for affine regression (linear only).
    Each row: [R, G, B, 1] (or [R, G, B] if add_ones=False)
    """
    if add_ones:
        ones = np.ones((x.shape[0], 1), np.float64)
        return np.hstack([x, ones])
    return x


def _solve_affine(S: np.ndarray, R: np.ndarray) -> tuple:
    """
    Solve for affine color transform: R = A @ S + b.
    Uses IRLS with Tukey bi-weight for robustness to outliers.

    Args:
        S: Source pixels (sim render), shape (N, 3)
        R: Reference pixels (real camera), shape (N, 3)

    Returns:
        A: 3x3 matrix, b: 3-vector
    """
    if len(S) < 10:
        return np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32)

    S_aug = _get_aug(S)
    X, *_ = np.linalg.lstsq(S_aug, R, rcond=None)
    if not np.all(np.isfinite(X)):
        print("  Warning: Initial least-squares failed, using identity transform")
        return np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32)

    max_iter = 50
    c = 4.685
    X_prev = X
    w = np.ones((S.shape[0],), np.float64)

    for _ in range(max_iter):
        pred = S_aug @ X_prev
        resid = np.linalg.norm(R - pred, axis=1)
        mad = np.median(np.abs(resid - np.median(resid)))
        mad = max(mad, 1e-6)
        scale = c * 1.4826 * mad
        u = resid / scale
        w = np.where(np.abs(u) < 1, (1 - u ** 2) ** 2, 0.0)

        if not np.any(w):
            break

        sqrt_w = np.sqrt(w)[:, None]
        X_new, *_ = np.linalg.lstsq(S_aug * sqrt_w, R * sqrt_w, rcond=None)
        if not np.all(np.isfinite(X_new)):
            break

        if np.linalg.norm(X_new - X_prev) < 1e-6:
            X_prev = X_new
            break

        X_prev = X_new

    A = X_prev[:-1, :].T.astype(np.float32)
    b = X_prev[-1, :].astype(np.float32)
    return A, b


def _apply_affine(img: np.ndarray, A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Apply affine color transform: out = rgb @ A.T + b"""
    H, W = img.shape[:2]
    flat = img.reshape(-1, 3).astype(np.float32) / 255.0
    out = flat @ A.T + b
    out = np.clip(out, 0.0, 1.0)
    return (out.reshape(H, W, 3) * 255.0).astype(np.uint8)


def main():
    # Paths (relative to project root)
    project_root = Path(__file__).parent.parent
    base_dir = project_root / "calibration_pairs_wrist"
    gs_dir = base_dir / "gs_renders"
    real_dir = base_dir / "real_captures"
    out_dir = base_dir / "calibrated"

    # Frame indices to use (must match SAVE_CALIB_FRAMES in compare_recorded_vs_mujoco.py)
    frame_indices = [0, 5, 10, 15, 20]
    src_img_paths = [gs_dir / f"frame_{i:04d}.png" for i in frame_indices]
    ref_img_paths = [real_dir / f"frame_{i:04d}.png" for i in frame_indices]

    # Verify files exist
    for src_path, ref_path in zip(src_img_paths, ref_img_paths):
        if not src_path.exists():
            raise FileNotFoundError(
                f"Source image not found: {src_path}\n"
                "Run: python visual_match/compare_recorded_vs_mujoco.py --dataset-path <path> "
                "--episode 0 --save-calibration-pairs"
            )
        if not ref_path.exists():
            raise FileNotFoundError(f"Reference image not found: {ref_path}")

    print(f"[INFO] Using {len(src_img_paths)} image pairs for calibration:")
    for src_path, ref_path in zip(src_img_paths, ref_img_paths):
        print(f"  {src_path.name} <-> {ref_path.name}")

    # Load images and collect pixels
    pixel_src = []
    pixel_ref = []
    image_height = None
    image_width = None

    for src_img_path, ref_img_path in zip(src_img_paths, ref_img_paths):
        src_img = np.array(Image.open(src_img_path).convert("RGB")).astype(np.float32) / 255.0
        ref_img = np.array(Image.open(ref_img_path).convert("RGB")).astype(np.float32) / 255.0

        assert src_img.shape == ref_img.shape, (
            f"Source and reference shapes must match: {src_img.shape} vs {ref_img.shape}"
        )

        if image_height is None:
            image_height, image_width = src_img.shape[:2]

        pixel_src.append(src_img.reshape(-1, 3))
        pixel_ref.append(ref_img.reshape(-1, 3))

    pixel_src = np.concatenate(pixel_src, axis=0)
    pixel_ref = np.concatenate(pixel_ref, axis=0)

    print(f"[INFO] Collected {pixel_src.shape[0]} pixels from {len(src_img_paths)} images")
    print(f"[INFO] Image dimensions: {image_height}x{image_width}")

    # Solve for single affine transform
    print("[INFO] Solving for affine color transform...")
    A, b = _solve_affine(pixel_src, pixel_ref)
    print(f"  A:\n{A}")
    print(f"  b: {b}")

    # Save transform (flat format for compatibility with load_color_mapping)
    os.makedirs(out_dir, exist_ok=True)
    transform_file = out_dir / "color_mapping.yaml"
    a_flat = ", ".join(f"{A[i, j]:.6f}" for i in range(3) for j in range(3))
    b_str = ", ".join(f"{b[i]:.6f}" for i in range(3))
    lines = [
        "# Affine color transform: sim -> real (out = rgb @ A.T + b)",
        f"color_A: [{a_flat}]",
        f"color_b: [{b_str}]",
    ]
    transform_file.write_text("\n".join(lines))
    print(f"[INFO] Saved color transform to {transform_file}")

    # Apply transform and save calibrated images
    print("[INFO] Applying transform to calibration image pairs...")
    for src_img_path, ref_img_path in zip(src_img_paths, ref_img_paths):
        src_img = np.array(Image.open(src_img_path).convert("RGB"))
        corr_img = _apply_affine(src_img, A, b)

        calibrated_path = out_dir / src_img_path.name
        Image.fromarray(corr_img).save(calibrated_path, quality=95)

        ref_img = np.array(Image.open(ref_img_path).convert("RGB"))
        combined_img = np.hstack((src_img, ref_img, corr_img))
        combined_path = out_dir / f"combined_{src_img_path.name}"
        Image.fromarray(combined_img).save(combined_path, quality=95)

        print(f"  Saved: {calibrated_path.name}")

    print(f"[INFO] Calibrated {len(src_img_paths)} image pairs")
    print(f"[INFO] Output: {out_dir}")


if __name__ == "__main__":
    main()
