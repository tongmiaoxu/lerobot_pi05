#!/usr/bin/env python3
"""Score a pix2pix test-set result folder on two axes: visual and geometric alignment.

Expects the layout produced by `sim2real/pix2pix/test.py` (via `sim2real/train_pix2pix.py`'s
post-training eval or `sim2real/eval_pix2pix.py`): a directory of `<idx>_real_A.png` /
`<idx>_fake_B.png` / `<idx>_real_B.png` triplets, where real_A is the sim input, fake_B is the
generator's output, real_B is the real target.

Visual alignment: LPIPS(fake_B, real_B) — perceptual distance to the real target.

Geometric alignment: text-prompted Grounding-DINO + SAM2 segmentation (reusing
visual_match/segmentation_utils.py) locates the task-relevant object(s) in real_A, fake_B, and
real_B, then scores fake_B against the real_B mask using the DAVIS video-object-segmentation
protocol:
  - J score = mask IoU(fake_B, real_B) — region/shape overlap.
  - F score = boundary F-measure(fake_B, real_B) — precision/recall of matched boundary
    pixels within a small tolerance (bound_th, as a fraction of the image diagonal).
  - J&F = mean(J, F), the standard DAVIS summary number.
As a baseline for context, the same J/F/J&F are also computed between real_A and real_B: the
dataset's own inherent sim/real misalignment. A model that isn't amplifying that misalignment
should score close to (or better than) this baseline.

Two more numbers cover the pairing used to report "geometry alignment" and "visual alignment"
per run/aggregated across runs (see scripts/aggregate_pix2pix_metrics.py):
  - lpips_base = LPIPS(real_A, real_B) — visual baseline: how far the raw sim render is from
    real, before any translation.
  - j_af/f_af/jf_af = J/F/J&F(real_A, fake_B) — geometry alignment of the *translation itself*:
    does pix2pix keep the object in the same place/shape as its own sim input, rather than
    whether fake_B ends up close to a (separately captured) real_B photo. Free to compute since
    mask_a and mask_fake are already resolved above.

--text-prompt accepts a comma-separated list of objects (e.g. "mug, saucer" for place_mug,
"carton, mug" for pouring) when more than one object in the scene matters for the task — each
is detected independently and scored independently, and a per-example score is the mean across
objects that had a mask on at least one side of the pair being compared (see below). Per-object
numbers are kept in the summary too, so you can see whether one object is dragging the average
down.

Detection can be ambiguous (>1 box for a phrase) or empty (0 boxes, especially in an early
fake_B). Ambiguous frames take Grounding-DINO's first (highest-confidence) box by default;
empty frames record a missing mask — unless a reviewer has confirmed with 'x' in
scripts/review_pix2pix_masks.py that the object is genuinely absent, in which case it's a
confirmed negative rather than just "not found". A one-sided mismatch — one side of a pair has
the object and the other doesn't (a missed or hallucinated detection) — scores a total miss
(J=0, F=0) rather than being dropped from the average; only a pair where *neither* side has the
object is skipped, since there's nothing to compare. Run scripts/review_pix2pix_masks.py first
to eyeball and click-fix any frame where the default box is wrong or confirm true negatives —
its fixes are saved to a mask_overrides.json cache next to metrics.json and are picked up
automatically here. Every per-example/per-object record's `mask_candidates` / `mask_source`
fields show how many boxes were found and how the final mask was chosen.

Usage:
  python scripts/eval_pix2pix_metrics.py \\
    --images-dir outputs/pix2pix_stationary_mug/results/place_mug_stationary_pix2pix/test_latest/images \\
    --text-prompt "mug, saucer"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "scripts"))
from pix2pix_mask_common import (  # noqa: E402
    boundary_f_measure,
    default_overrides_path,
    find_triplet_indices,
    get_override_candidate_index,
    get_override_point,
    is_override_missing,
    load_overrides,
    mask_iou,
    parse_object_list,
    resolve_mask,
    save_overrides,
    summarize,
)


def score_pair(mask_pred: np.ndarray | None, mask_gt: np.ndarray | None, bound_th: float) -> tuple[float, float] | None:
    """DAVIS J/F between a predicted mask and its ground truth. A one-sided presence mismatch
    (object detected on only one side) scores a total miss (0.0, 0.0) rather than being skipped,
    so a missed or hallucinated detection drags the average down instead of vanishing from it.
    Returns None only when neither side has the object — nothing to compare, so not scored."""
    if mask_pred is None and mask_gt is None:
        return None
    if mask_pred is None or mask_gt is None:
        return 0.0, 0.0
    return mask_iou(mask_pred, mask_gt), boundary_f_measure(mask_pred, mask_gt, bound_th)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images-dir", type=Path, required=True, help="Folder of <idx>_real_A/fake_B/real_B.png triplets.")
    parser.add_argument(
        "--text-prompt", type=str, required=True,
        help="Object(s) to track for geometric alignment, comma-separated, e.g. 'mug' or 'mug, saucer'.",
    )
    parser.add_argument(
        "--bound-th", type=float, default=0.008,
        help="Boundary-match tolerance for the F score, as a fraction of the image diagonal (DAVIS default: 0.008).",
    )
    parser.add_argument(
        "--box-threshold", type=float, default=0.325,
        help="Grounding-DINO box confidence threshold (lower = more permissive detection). Default matches segmentation_utils.py's shared default.",
    )
    parser.add_argument(
        "--text-threshold", type=float, default=0.3,
        help="Grounding-DINO text-match confidence threshold (lower = more permissive detection).",
    )
    parser.add_argument(
        "--overrides-json", type=Path, default=None,
        help="Manual per-image mask fixes (see scripts/review_pix2pix_masks.py). Default: <images-dir>/../mask_overrides.json",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Where to write the summary + per-example metrics. Default: <images-dir>/../metrics.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    objects = parse_object_list(args.text_prompt)
    overrides_path = args.overrides_json or default_overrides_path(args.images_dir)
    overrides = load_overrides(overrides_path)

    import lpips

    lpips_fn = lpips.LPIPS(net="alex").to(device)

    indices = find_triplet_indices(args.images_dir)
    per_example = []
    missing_masks = {"real_A": 0, "fake_B": 0, "real_B": 0}
    per_object_values: dict[str, dict[str, list[float]]] = {
        obj: {
            "j_score": [], "f_score": [], "jf_mean": [],
            "j_base": [], "f_base": [], "jf_base": [],
            "j_af": [], "f_af": [], "jf_af": [],
        }
        for obj in objects
    }

    def get_mask(image_bgr, obj_name, idx, role):
        override_point = get_override_point(overrides, idx, obj_name, role)
        override_missing = is_override_missing(overrides, idx, obj_name, role)
        override_candidate_index = get_override_candidate_index(overrides, idx, obj_name, role)
        mask, n, src, _ = resolve_mask(
            image_bgr, obj_name, args.box_threshold, args.text_threshold,
            override_point=override_point, override_missing=override_missing,
            override_candidate_index=override_candidate_index,
        )
        return mask, n, src

    for idx in indices:
        real_a_bgr = cv2.imread(str(args.images_dir / f"{idx}_real_A.png"))
        fake_b_bgr = cv2.imread(str(args.images_dir / f"{idx}_fake_B.png"))
        real_b_bgr = cv2.imread(str(args.images_dir / f"{idx}_real_B.png"))
        if real_a_bgr is None or fake_b_bgr is None or real_b_bgr is None:
            raise FileNotFoundError(f"Incomplete triplet for index {idx} in {args.images_dir}")

        real_a_rgb = cv2.cvtColor(real_a_bgr, cv2.COLOR_BGR2RGB)
        fake_b_rgb = cv2.cvtColor(fake_b_bgr, cv2.COLOR_BGR2RGB)
        real_b_rgb = cv2.cvtColor(real_b_bgr, cv2.COLOR_BGR2RGB)
        real_a_t = lpips.im2tensor(real_a_rgb).to(device)
        fake_t = lpips.im2tensor(fake_b_rgb).to(device)
        real_t = lpips.im2tensor(real_b_rgb).to(device)
        with torch.no_grad():
            lpips_score = float(lpips_fn(fake_t, real_t).item())
            lpips_base_score = float(lpips_fn(real_a_t, real_t).item())

        record = {"index": idx, "lpips": lpips_score, "lpips_base": lpips_base_score, "objects": {}}
        example_jf_mean = []
        example_jf_base = []
        example_jf_af = []

        any_missing = {"real_A": False, "fake_B": False, "real_B": False}
        for obj_name in objects:
            mask_a, n_a, src_a = get_mask(real_a_bgr, obj_name, idx, "real_A")
            mask_fake, n_fake, src_fake = get_mask(fake_b_bgr, obj_name, idx, "fake_B")
            mask_real, n_real, src_real = get_mask(real_b_bgr, obj_name, idx, "real_B")

            obj_record = {
                "mask_detected": {"real_A": mask_a is not None, "fake_B": mask_fake is not None, "real_B": mask_real is not None},
                "mask_candidates": {"real_A": n_a, "fake_B": n_fake, "real_B": n_real},
                "mask_source": {"real_A": src_a, "fake_B": src_fake, "real_B": src_real},
            }
            for role, mask in (("real_A", mask_a), ("fake_B", mask_fake), ("real_B", mask_real)):
                if mask is None:
                    any_missing[role] = True

            pair = score_pair(mask_fake, mask_real, args.bound_th)
            if pair is not None:
                j_score, f_score = pair
                jf_mean = (j_score + f_score) / 2
                obj_record.update({"j_score": j_score, "f_score": f_score, "jf_mean": jf_mean})
                per_object_values[obj_name]["j_score"].append(j_score)
                per_object_values[obj_name]["f_score"].append(f_score)
                per_object_values[obj_name]["jf_mean"].append(jf_mean)
                example_jf_mean.append(jf_mean)

            pair_base = score_pair(mask_a, mask_real, args.bound_th)
            if pair_base is not None:
                j_base, f_base = pair_base
                jf_base = (j_base + f_base) / 2
                obj_record.update({"j_base": j_base, "f_base": f_base, "jf_base": jf_base})
                per_object_values[obj_name]["j_base"].append(j_base)
                per_object_values[obj_name]["f_base"].append(f_base)
                per_object_values[obj_name]["jf_base"].append(jf_base)
                example_jf_base.append(jf_base)

            pair_af = score_pair(mask_a, mask_fake, args.bound_th)
            if pair_af is not None:
                j_af, f_af = pair_af
                jf_af = (j_af + f_af) / 2
                obj_record.update({"j_af": j_af, "f_af": f_af, "jf_af": jf_af})
                per_object_values[obj_name]["j_af"].append(j_af)
                per_object_values[obj_name]["f_af"].append(f_af)
                per_object_values[obj_name]["jf_af"].append(jf_af)
                example_jf_af.append(jf_af)

            record["objects"][obj_name] = obj_record

        for role in missing_masks:
            if any_missing[role]:
                missing_masks[role] += 1
        if example_jf_mean:
            record["jf_mean_avg_objects"] = sum(example_jf_mean) / len(example_jf_mean)
        if example_jf_base:
            record["jf_base_avg_objects"] = sum(example_jf_base) / len(example_jf_base)
        if example_jf_af:
            record["jf_af_avg_objects"] = sum(example_jf_af) / len(example_jf_af)

        per_example.append(record)

    summary = {
        "lpips": summarize([r["lpips"] for r in per_example]),
        "lpips_base": summarize([r["lpips_base"] for r in per_example]),
        "jf_mean_avg_objects": summarize([r["jf_mean_avg_objects"] for r in per_example if "jf_mean_avg_objects" in r]),
        "jf_base_avg_objects": summarize([r["jf_base_avg_objects"] for r in per_example if "jf_base_avg_objects" in r]),
        "jf_af_avg_objects": summarize([r["jf_af_avg_objects"] for r in per_example if "jf_af_avg_objects" in r]),
        "objects": {
            obj_name: {metric: summarize(values) for metric, values in metrics.items()}
            for obj_name, metrics in per_object_values.items()
        },
        "missing_masks": missing_masks,
        "n_examples": len(per_example),
        "text_prompt": args.text_prompt,
        "objects_list": objects,
        "bound_th": args.bound_th,
        "box_threshold": args.box_threshold,
        "text_threshold": args.text_threshold,
        "images_dir": str(args.images_dir),
    }

    output_json = args.output_json or (args.images_dir.parent / "metrics.json")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w") as handle:
        json.dump({"summary": summary, "per_example": per_example}, handle, indent=2)
    save_overrides(overrides_path, overrides)

    print(json.dumps(summary, indent=2))
    print(f"\nWrote {output_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
