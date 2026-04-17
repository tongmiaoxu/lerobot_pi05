import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lerobot.tasks import get_task_profile
from segmentation_utils import segment_object_mask


_DEFAULT_RECORD_TASK_ID = "place_mug"  # Keep in sync with compare/deploy defaults.
CAMERA = "wrist"

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


def _solve_from_samples(S, R, spatial_weights=None, spatial_mask=None):
    S_aug = _get_aug(S)

    weight = np.linalg.norm(R, axis=1) ** 1.0  # NOTE: tunable parameter
    weight = weight / np.max(weight)
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
    task_profile = get_task_profile(_DEFAULT_RECORD_TASK_ID)
    base_dir = task_profile.calibration_pairs_dir(CAMERA)
    gs_dir = base_dir / "gs_renders"
    real_dir = base_dir / "real_captures"
    out_dir = str(base_dir / "calibrated")
    src_img_paths, ref_img_paths = _collect_calibration_frame_paths(gs_dir, real_dir)

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
    SEGMENT_PROMPT = "floor,mug,gripper"
    SEGMENT_WEIGHT = [0.99, 0.99, 0.99]
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
    for ref_img_path in ref_img_paths:
        ref_img_rgb = np.array(Image.open(ref_img_path).convert("RGB"))
        w_img = np.ones((image_height, image_width), dtype=np.float64)
        m_img = np.zeros((image_height, image_width), dtype=bool)
        line_parts: list[str] = []
        for prompt, seg_w in zip(prompts, SEGMENT_WEIGHT):
            seg_mask = segment_object_mask(ref_img_rgb, text_prompt=prompt)
            if seg_mask is not None and seg_mask.any():
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
    seg_pct = np.mean(spatial_mask_flat) * 100
    print(
        f"[INFO] Spatial weight: {seg_pct:.1f}% of pixels overridden "
        f"(per-class weights {list(zip(prompts, SEGMENT_WEIGHT))})"
    )

    A, b, w = _solve_from_samples(pixel_src, pixel_ref,
                                   spatial_weights=spatial_w_flat,
                                   spatial_mask=spatial_mask_flat)

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