import os
import sys
import argparse
from pathlib import Path
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lerobot.tasks import get_task_profile
from segmentation_utils import segment_object_mask


_DEFAULT_RECORD_TASK_ID = "book_shelving"  # Keep in sync with compare/deploy defaults.
CAMERA = "wrist"


def _parse_args():
    parser = argparse.ArgumentParser(description="Fit per-camera sim-to-real color calibration.")
    parser.add_argument("--task", dest="task_id", default=_DEFAULT_RECORD_TASK_ID)
    parser.add_argument("--camera", choices=("stationary", "wrist"), default=CAMERA)
    parser.add_argument("--segment-prompt", default="floor,gripper")
    parser.add_argument("--segment-weight", default="0.5,1")
    parser.add_argument(
        "--void-threshold",
        type=int,
        default=8,
        help=(
            "Source (gs_renders) pixels with all channels below this value (0-255) are treated as "
            "unreconstructed Gaussian-Splat void and protected from the brightness-based sample "
            "weighting (which would otherwise starve the fit of dark-background samples) and from "
            "segmentation-based down-weighting."
        ),
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Use a deterministic uniform sample of at most this many paired frames.",
    )
    return parser.parse_args()

def _collect_calibration_frame_paths(gs_dir: Path, real_dir: Path) -> tuple[list[Path], list[Path]]:
    src_img_paths = sorted(gs_dir.glob("frame_*.png"))
    if not src_img_paths:
        raise FileNotFoundError(f"No calibration renders found in {gs_dir}")

    ref_img_paths = [real_dir / path.name for path in src_img_paths]
    missing = [str(path) for path in ref_img_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing matching real calibration captures: " + ", ".join(missing)
        )

    return src_img_paths, ref_img_paths


def _format_helper(A: np.ndarray, b: np.ndarray) -> str:
    A_round = np.round(A, 3).tolist()
    b_round = np.round(b, 3).tolist()

    def _fmt_matrix(mat):
        rows = [", ".join(f"{v:.3f}" for v in row) for row in mat]
        return "[\n    " + ",\n    ".join(rows) + "\n]"

    def _fmt_vec(vec):
        return "[\n    " + ", ".join(f"{v:.3f}" for v in vec) + "\n]"

    return (
        f"color_A: {_fmt_matrix(A_round)}\n"
        f"color_b: {_fmt_vec(b_round)}\n"
    )


def _write_helper_file(A: np.ndarray, b: np.ndarray, dest: Path) -> None:
    code = _format_helper(A, b)
    dest.write_text(code)


def _get_aug(x: np.ndarray, add_ones: bool = True) -> np.ndarray:
    if add_ones:
        ones = np.ones((x.shape[0], 1), np.float64)
        return np.hstack([x ** 2, x, ones])
    return np.hstack([x ** 2, x])


def _solve_from_samples(S, R, spatial_weights=None, spatial_mask=None, protect_mask=None):
    S_aug = _get_aug(S)

    weight = np.linalg.norm(R, axis=1) ** 1.0  # NOTE: tunable parameter
    weight = weight / np.max(weight)
    if protect_mask is not None and protect_mask.any():
        # This brightness weighting systematically discounts dark targets (e.g. a
        # large, fairly uniform dark background). Left unprotected, that silently
        # starves the fit of the samples needed to anchor the bias term to the
        # correct (dark) color, and the whole frame drifts toward a brighter
        # "compromise" tone once other, brighter pixels dominate the fit instead.
        weight[protect_mask] = 1.0
    S_aug = S_aug * weight[:, None]
    R = R * weight[:, None]

    # Initial L2 solution
    X, *_ = np.linalg.lstsq(S_aug, R, rcond=None)
    if not np.all(np.isfinite(X)):
        raise RuntimeError("Initial least-squares failed (non-finite values)")

    # Robust IRLS with Tukey bi-weight
    max_iter = 50  # NOTE: tunable parameter
    c = 4.685
    X_prev = X
    w = np.ones((S.shape[0],), np.float64)
    for n_iter in range(max_iter):
        pred = S_aug @ X_prev
        resid = np.linalg.norm(R - pred, axis=1)
        resid = resid / (weight + 1e-10)
        mad = np.median(np.abs(resid - np.median(resid)))
        mad = max(mad, 1e-6)
        scale = c * 1.4826 * mad
        u = resid / scale
        w = np.where(np.abs(u) < 1, (1 - u ** 2) ** 2, 0.0)
        if not np.any(w):
            print(f"No valid weights found, stopping IRLS at iteration {n_iter + 1}")
            break
        sqrt_w = np.sqrt(w)[:, None]
        X_new, *_ = np.linalg.lstsq(S_aug * sqrt_w, R * sqrt_w, rcond=None)
        if not np.all(np.isfinite(X_new)):
            print(f"New solution has non-finite values, stopping IRLS at iteration {n_iter + 1}")
            break
        if np.linalg.norm(X_new - X_prev) < 1e-6:
            X_prev = X_new
            print(f"Converged after IRLS iteration {n_iter + 1}")
            break
        X_prev = X_new
    else:
        print(f"Reached max iterations ({max_iter}) without convergence")    
        print(f"Final error: {np.linalg.norm(R - S_aug @ X_prev)}")

    # Override weights for segmented pixels, then re-solve
    w_final = w.copy()
    if spatial_weights is not None and spatial_mask is not None:
        if spatial_mask.any():
            w_final[spatial_mask] = spatial_weights[spatial_mask]
            pct_overridden = np.mean(spatial_mask) * 100
            print(f"[INFO] Overrode {pct_overridden:.1f}% of pixel weights "
                  f"(set to {spatial_weights[spatial_mask].min():.3f})")
        else:
            print("[INFO] No pixels matched spatial mask, weights unchanged")

        sqrt_w = np.sqrt(w_final)[:, None]
        X_new, *_ = np.linalg.lstsq(S_aug * sqrt_w, R * sqrt_w, rcond=None)
        if np.all(np.isfinite(X_new)):
            X_prev = X_new
            print(f"[INFO] Re-solved with overridden weights (final error: "
                  f"{np.linalg.norm(R - S_aug @ X_prev):.6f})")
        else:
            print("[WARN] Re-solve with overridden weights gave non-finite values; keeping IRLS result")
            w_final = w

    A = X_prev[:-1, :].T.astype(np.float32)
    b = X_prev[-1, :].T.astype(np.float32)

    return A, b, w_final


def _apply_transform(img: np.ndarray, A: np.ndarray, b: np.ndarray) -> np.ndarray:
    flat = img.reshape(-1, 3).astype(np.float32) / 255.0
    flat_aug = _get_aug(flat, add_ones=False)
    out = flat_aug @ A.T + b
    out = np.clip(out, 0.0, 1.0)
    return (out.reshape(img.shape) * 255.0).astype(np.uint8)


def main():
    args = _parse_args()
    task_profile = get_task_profile(args.task_id)
    base_dir = task_profile.calibration_pairs_dir(args.camera)
    gs_dir = base_dir / "gs_renders"
    real_dir = base_dir / "real_captures"
    out_dir = str(base_dir / "calibrated")
    src_img_paths, ref_img_paths = _collect_calibration_frame_paths(gs_dir, real_dir)
    if args.max_images is not None and args.max_images > 0 and len(src_img_paths) > args.max_images:
        sample_indices = np.linspace(0, len(src_img_paths) - 1, args.max_images, dtype=int)
        src_img_paths = [src_img_paths[i] for i in sample_indices]
        ref_img_paths = [ref_img_paths[i] for i in sample_indices]
        print(f"[INFO] Using {len(src_img_paths)} uniformly sampled pairs from {len(sample_indices)} requested.")

    pixel_src = []
    pixel_ref = []
    image_height = None
    image_width = None
    for src_img_path, ref_img_path in zip(src_img_paths, ref_img_paths):
        src_img = np.array(Image.open(src_img_path).convert("RGB")).astype(np.float32) / 255.0
        ref_img = np.array(Image.open(ref_img_path).convert("RGB")).astype(np.float32) / 255.0

        assert src_img.shape == ref_img.shape, "Source and reference images must have the same shape"

        if image_height is None:
            image_height, image_width = src_img.shape[:2]

        pixel_src.append(src_img.reshape(-1, 3))
        pixel_ref.append(ref_img.reshape(-1, 3))
    
    pixel_src = np.concatenate(pixel_src, axis=0)
    pixel_ref = np.concatenate(pixel_ref, axis=0)

    # Spatial weight: downweight segmented pixels using Grounding-DINO + SAM2 (one forward per prompt)
    SEGMENT_PROMPT = args.segment_prompt
    SEGMENT_WEIGHT = [float(w.strip()) for w in args.segment_weight.split(",") if w.strip()]
    prompts = [p.strip() for p in SEGMENT_PROMPT.split(",") if p.strip()]
    if len(prompts) != len(SEGMENT_WEIGHT):
        raise ValueError(
            f"SEGMENT_PROMPT has {len(prompts)} non-empty part(s), "
            f"but SEGMENT_WEIGHT has length {len(SEGMENT_WEIGHT)}"
        )
    print(
        f"[INFO] Segmenting {prompts} (weights {SEGMENT_WEIGHT}) in reference images using SAM2..."
    )
    spatial_w_list = []
    spatial_mask_list = []
    void_mask_list = []
    total_void_pct = []
    for src_img_path, ref_img_path in zip(src_img_paths, ref_img_paths):
        ref_img_rgb = np.array(Image.open(ref_img_path).convert("RGB"))
        src_img_rgb = np.array(Image.open(src_img_path).convert("RGB"))
        w_img = np.ones((image_height, image_width), dtype=np.float64)
        m_img = np.zeros((image_height, image_width), dtype=bool)

        # Gaussian-Splat "void": source render has no reconstructed color here
        # (e.g. unmodeled dark tabletop), so it's a large, fairly consistent dark
        # patch in every frame. It must stay OUT of the segmentation-based
        # up/down-weighting below (handled separately via protect_mask in
        # _solve_from_samples) so it isn't re-suppressed after being protected
        # from the brightness-based initial weight.
        void_mask = np.all(src_img_rgb < args.void_threshold, axis=-1)
        total_void_pct.append(void_mask.mean() * 100)
        void_mask_list.append(void_mask.ravel())

        line_parts: list[str] = []
        for prompt, seg_w in zip(prompts, SEGMENT_WEIGHT):
            seg_mask = segment_object_mask(ref_img_rgb, text_prompt=prompt)
            if seg_mask is not None and seg_mask.any():
                seg_mask = seg_mask & ~void_mask
                if seg_mask.any():
                    w_img[seg_mask] = np.minimum(w_img[seg_mask], seg_w)
                    m_img[seg_mask] = True
                pct = seg_mask.sum() / seg_mask.size * 100
                line_parts.append(f"{prompt!r} {pct:.1f}%→{seg_w}")
        base = os.path.basename(ref_img_path)
        if line_parts:
            print(f"  {base}: " + "; ".join(line_parts))
        else:
            print(f"  {base}: [no detection, all pixels weight=1.0]")
        spatial_w_list.append(w_img.ravel())
        spatial_mask_list.append(m_img.ravel())
    spatial_w_flat = np.concatenate(spatial_w_list)
    spatial_mask_flat = np.concatenate(spatial_mask_list)
    void_mask_flat = np.concatenate(void_mask_list)
    seg_pct = np.mean(spatial_mask_flat) * 100
    print(
        f"[INFO] Spatial weight: {seg_pct:.1f}% of pixels overridden "
        f"(per-class weights {list(zip(prompts, SEGMENT_WEIGHT))})"
    )
    print(
        f"[INFO] GS void (source pixels < {args.void_threshold}/255, protected from "
        f"brightness down-weighting so the bias term still anchors to it): "
        f"avg {np.mean(total_void_pct):.1f}% per frame (min {np.min(total_void_pct):.1f}%, "
        f"max {np.max(total_void_pct):.1f}%)"
    )

    A, b, w = _solve_from_samples(pixel_src, pixel_ref,
                                   spatial_weights=spatial_w_flat,
                                   spatial_mask=spatial_mask_flat,
                                   protect_mask=void_mask_flat)

    print("Color correction matrix A:", A)
    print("Color correction bias b:", b)

    os.makedirs(out_dir, exist_ok=True)
    _write_helper_file(A, b, Path(out_dir) / "color_mapping.yaml")

    n_images = len(src_img_paths)
    w_full = w.reshape(n_images, image_height, image_width, -1).repeat(3, axis=-1)
    w_full = w_full * 255.0

    w_mask = (w_full < 1.0).all(axis=-1)  # only zero
    for i in range(n_images):
        w_vis = w_full[i].copy().astype(np.uint8)
        Image.fromarray(w_vis).save(os.path.join(out_dir, f"weights_{i:06d}.png"), quality=95)
        w_mask_vis = (w_mask[i].copy() * 255).astype(np.uint8)
        Image.fromarray(w_mask_vis).save(os.path.join(out_dir, f"weights_mask_{i:06d}.png"), quality=95)

    for src_img_path, ref_img_path in zip(src_img_paths, ref_img_paths):
        src_img = np.array(Image.open(src_img_path).convert("RGB"))
        out_path = os.path.join(out_dir, os.path.basename(str(src_img_path)))
        corr_img = _apply_transform(src_img, A, b)
        Image.fromarray(corr_img).save(out_path, quality=95)

        # concatenate src image, ref image, and corrected image horizontally
        ref_img = np.array(Image.open(ref_img_path).convert("RGB"))
        combined_img = np.hstack((src_img, ref_img, corr_img))
        combined_out_path = os.path.join(out_dir, f"combined_{os.path.basename(str(src_img_path))}")
        Image.fromarray(combined_img).save(combined_out_path, quality=95)

        print(f"Saved corrected image to {out_path}")


if __name__ == "__main__":
    main()