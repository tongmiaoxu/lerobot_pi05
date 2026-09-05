"""Shared mask-resolution helpers for scripts/eval_pix2pix_metrics.py and
scripts/review_pix2pix_masks.py: locating a text-prompted object in a pix2pix result image via
Grounding-DINO + SAM2 (visual_match/segmentation_utils.py), an interactive click/keypress
fallback for ambiguous or missed detections, a small on-disk cache of manual fixes so a review
pass in one script is reused by the other, and the DAVIS J/F mask-comparison metrics.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import binary_dilation, generate_binary_structure

import sys

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "visual_match"))
from segmentation_utils import segment_candidate_masks, segment_point_mask  # noqa: E402


def find_triplet_indices(images_dir: Path) -> list[str]:
    indices = sorted({p.name.removesuffix("_real_A.png") for p in images_dir.glob("*_real_A.png")})
    if not indices:
        raise FileNotFoundError(f"No '*_real_A.png' files found under {images_dir}")
    return indices


def parse_object_list(text_prompt: str) -> list[str]:
    """Split a comma-separated `--text-prompt` into individual object phrases, e.g.
    'mug, saucer' -> ['mug', 'saucer']. Each is detected independently."""
    objects = [obj.strip() for obj in text_prompt.split(",") if obj.strip()]
    if not objects:
        raise ValueError(f"--text-prompt produced no object names: {text_prompt!r}")
    return objects


def default_overrides_path(images_dir: Path) -> Path:
    return images_dir.parent / "mask_overrides.json"


def load_overrides(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open() as handle:
        return json.load(handle)


def save_overrides(path: Path, overrides: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(overrides, handle, indent=2)


def get_override_point(overrides: dict, idx: str, obj_name: str, role: str) -> tuple[int, int] | None:
    point = overrides.get(idx, {}).get(obj_name, {}).get(role, {}).get("point")
    return tuple(point) if point is not None else None


def is_override_missing(overrides: dict, idx: str, obj_name: str, role: str) -> bool:
    """True if a reviewer explicitly marked this image as having no `obj_name` present (as
    opposed to Grounding-DINO simply finding nothing, which is uncertain rather than confirmed)."""
    return bool(overrides.get(idx, {}).get(obj_name, {}).get(role, {}).get("missing"))


def get_override_candidate_index(overrides: dict, idx: str, obj_name: str, role: str) -> int | None:
    """Index into Grounding-DINO's candidate list a reviewer confirmed as correct (picked by
    number key in scripts/review_pix2pix_masks.py). Detection is deterministic for a fixed
    image/thresholds, so re-running it and indexing in reproduces the exact reviewed mask —
    no fresh SAM2 point-prompt needed."""
    index = overrides.get(idx, {}).get(obj_name, {}).get(role, {}).get("candidate_index")
    return int(index) if index is not None else None


def set_override_point(overrides: dict, idx: str, obj_name: str, role: str, point: tuple[int, int]) -> None:
    overrides.setdefault(idx, {}).setdefault(obj_name, {})[role] = {"point": list(point)}


def set_override_missing(overrides: dict, idx: str, obj_name: str, role: str) -> None:
    overrides.setdefault(idx, {}).setdefault(obj_name, {})[role] = {"missing": True}


def set_override_candidate_index(overrides: dict, idx: str, obj_name: str, role: str, candidate_index: int) -> None:
    overrides.setdefault(idx, {}).setdefault(obj_name, {})[role] = {"candidate_index": candidate_index}


def clear_override(overrides: dict, idx: str, obj_name: str, role: str) -> None:
    overrides.get(idx, {}).get(obj_name, {}).pop(role, None)


# --- mask <-> metric ---------------------------------------------------------------


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """DAVIS J score: region (Jaccard) similarity."""
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    return float(intersection / union) if union > 0 else 0.0


def _seg2bmap(mask: np.ndarray) -> np.ndarray:
    """4-neighbor boundary map of a binary mask (davis2017-evaluation's seg2bmap, no resize needed
    since our masks always share the source image's resolution)."""
    seg = mask.astype(bool)
    e = np.zeros_like(seg)
    s = np.zeros_like(seg)
    se = np.zeros_like(seg)
    e[:, :-1] = seg[:, 1:]
    s[:-1, :] = seg[1:, :]
    se[:-1, :-1] = seg[1:, 1:]
    b = (seg ^ e) | (seg ^ s) | (seg ^ se)
    b[-1, :] = seg[-1, :] ^ e[-1, :]
    b[:, -1] = seg[:, -1] ^ s[:, -1]
    b[-1, -1] = False
    return b


def boundary_f_measure(mask_pred: np.ndarray, mask_gt: np.ndarray, bound_th: float) -> float:
    """DAVIS F score: boundary precision/recall F-measure between two binary masks."""
    bound_pix = max(1, int(round(bound_th * np.linalg.norm(mask_pred.shape))))

    pred_boundary = _seg2bmap(mask_pred)
    gt_boundary = _seg2bmap(mask_gt)

    struct = generate_binary_structure(2, 2)
    pred_dilated = binary_dilation(pred_boundary, structure=struct, iterations=bound_pix)
    gt_dilated = binary_dilation(gt_boundary, structure=struct, iterations=bound_pix)

    n_pred = int(pred_boundary.sum())
    n_gt = int(gt_boundary.sum())
    if n_pred == 0 and n_gt == 0:
        return 1.0
    if n_pred == 0 or n_gt == 0:
        return 0.0

    precision = float((pred_boundary & gt_dilated).sum()) / n_pred
    recall = float((gt_boundary & pred_dilated).sum()) / n_gt
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# --- interactive resolution ---------------------------------------------------------


def draw_candidates(image_bgr: np.ndarray, candidate_masks: list[np.ndarray]) -> np.ndarray:
    colors = [(0, 255, 0), (0, 165, 255), (255, 0, 255), (255, 255, 0), (0, 0, 255)]
    vis = image_bgr.copy()
    for i, mask in enumerate(candidate_masks):
        color = colors[i % len(colors)]
        overlay = vis.copy()
        overlay[mask] = color
        vis = cv2.addWeighted(overlay, 0.4, vis, 0.6, 0)
        ys, xs = np.nonzero(mask)
        if len(xs):
            cv2.putText(vis, str(i + 1), (int(xs.mean()), int(ys.mean())), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    return vis


def prompt_for_mask(image_bgr: np.ndarray, candidate_masks: list[np.ndarray], window_label: str):
    """Show `image_bgr` with any auto-detected `candidate_masks` overlaid and block until the
    user resolves it: a number key (1-9) picks a candidate, a left click runs a fresh SAM2
    point-prompt at that pixel, 'x' confirms the object is genuinely absent from this image,
    's' skips without saving anything. Returns (mask_or_None, source_tag, point_or_None,
    candidate_index_or_None); tag is "point", "pick", "missing", or "skip". For a "pick",
    candidate_index is the index into `candidate_masks` — since detection is deterministic,
    callers can persist just that index and re-select the same mask on a future run instead of
    re-segmenting from a point."""
    clicked: dict[str, tuple[int, int] | None] = {"xy": None}

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked["xy"] = (x, y)

    cv2.namedWindow(window_label, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_label, on_mouse)
    vis = draw_candidates(image_bgr, candidate_masks)
    n = len(candidate_masks)
    help_text = (
        f"{n} candidate(s). Keys 1-{min(n, 9)} pick one | click = point-prompt | x = no object here | s = skip"
        if n
        else "No automatic detection. Click the object | x = confirm no object here | s = skip"
    )
    cv2.putText(vis, help_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
    cv2.imshow(window_label, vis)

    result = (None, "skip", None, None)
    while True:
        key = cv2.waitKeyEx(30)
        if clicked["xy"] is not None:
            xy = clicked["xy"]
            result = (segment_point_mask(image_bgr, xy), "point", xy, None)
            break
        if key == ord("s"):
            break
        if key == ord("x"):
            result = (None, "missing", None, None)
            break
        if ord("1") <= key <= ord("9") and (key - ord("1")) < n:
            candidate_index = key - ord("1")
            result = (candidate_masks[candidate_index], "pick", None, candidate_index)
            break
    cv2.destroyWindow(window_label)
    return result


def resolve_mask(
    image_bgr: np.ndarray,
    text_prompt: str,
    box_threshold: float,
    text_threshold: float,
    override_point: tuple[int, int] | None = None,
    override_missing: bool = False,
    override_candidate_index: int | None = None,
):
    """Get one object mask from `image_bgr`. Returns (mask_or_None, n_candidates, source_tag,
    point_or_None) — `point_or_None` is always None here (kept for call-site symmetry with the
    interactive picker in scripts/review_pix2pix_masks.py, which is the only place overrides get
    created). When Grounding-DINO returns more than one box, this always takes the first
    (highest-confidence) candidate — review the mask_overrides.json workflow in
    scripts/review_pix2pix_masks.py beforehand to fix any frame where that isn't the right box.
    `override_missing=True` (set via review_pix2pix_masks.py's 'x' key) means a reviewer
    confirmed the object is genuinely absent here — distinct from an unreviewed empty detection,
    which is just "not found" rather than "confirmed absent". `override_candidate_index` (set by
    picking a number key in the reviewer) re-selects that exact candidate from a fresh detection
    pass instead of taking #0 — detection is deterministic, so this reproduces the exact mask
    the reviewer saw, rather than re-segmenting a fresh mask from a point."""
    if override_missing:
        return None, -1, "override_missing", None
    if override_point is not None:
        return segment_point_mask(image_bgr, override_point), -1, "override_point", None

    masks = segment_candidate_masks(image_bgr, text_prompt=text_prompt, box_threshold=box_threshold, text_threshold=text_threshold)
    n = len(masks)
    if override_candidate_index is not None and 0 <= override_candidate_index < n:
        return masks[override_candidate_index], n, "override_candidate", None
    if n == 0:
        return None, n, "missing", None
    return masks[0], n, ("auto" if n == 1 else "auto_first"), None


def summarize(values: list[float]) -> dict:
    import statistics

    if not values:
        return {"n": 0, "mean": None, "median": None, "std": None}
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
    }
