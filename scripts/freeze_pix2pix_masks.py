#!/usr/bin/env python3
"""One-time backfill: convert every already-resolved real_A/fake_B/real_B mask in an existing
mask_overrides.json (including one with no override entry at all, resolved via Grounding-DINO/
SAM2's own default) into the stable saved-mask format (see pix2pix_mask_common.set_override_mask),
with no reviewer re-click needed.

Why this is needed: scripts/review_pix2pix_masks.py's whole point is a full-grid "eyeball scan" —
a cell only needs a click when its shown default is *wrong* (an ambiguous cell whose #1 candidate
is on the wrong object, a missing/red cell, etc). A cell whose shown default already looks right
is deliberately left un-clicked, per that tool's own on-screen instructions. Before mask-caching
existed, that meant a fully, correctly reviewed set could still be almost entirely `legacy`
(point/candidate_index) or bare-default entries, none of them protected from Grounding-DINO/
SAM2's run-to-run float nondeterminism a saved mask exists to eliminate. Run this right after a
review pass (or on an older already-reviewed set) to freeze whatever it currently resolves to.

A frame is only frozen when it currently resolves to an actual mask. A confirmed-absent `missing`
override, or an unconfirmed non-detection with no override, is left exactly as-is — there's
nothing to freeze there, and both remain correctly re-derived fresh every time as before. Already-
`mask` entries are left untouched (no-op, safe to re-run).

Usage:
  python scripts/freeze_pix2pix_masks.py \\
    --images-dir outputs/turbo_sim2real_all_tasks_dino_only/results/place_mug_stationary_turbo_dino_only/test_latest/images \\
    --text-prompt "mug, saucer"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "scripts"))
from pix2pix_mask_common import (  # noqa: E402
    default_overrides_path,
    find_triplet_indices,
    get_override_candidate_index,
    get_override_mask,
    get_override_point,
    is_override_missing,
    load_overrides,
    parse_object_list,
    resolve_mask,
    save_overrides,
    set_override_mask,
)

ROLES = ("real_A", "fake_B", "real_B")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images-dir", type=Path, required=True, help="Folder of <idx>_real_A/fake_B/real_B.png triplets.")
    parser.add_argument("--text-prompt", type=str, required=True, help="Object(s), comma-separated, e.g. 'mug, saucer'.")
    parser.add_argument("--box-threshold", type=float, default=0.325)
    parser.add_argument("--text-threshold", type=float, default=0.3)
    parser.add_argument(
        "--overrides-json", type=Path, default=None,
        help="Default: <images-dir>/../mask_overrides.json (shared with eval_pix2pix_metrics.py).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    objects = parse_object_list(args.text_prompt)
    overrides_path = args.overrides_json or default_overrides_path(args.images_dir)
    overrides = load_overrides(overrides_path)
    indices = find_triplet_indices(args.images_dir)

    frozen = already = skipped_missing = skipped_unresolved = 0
    for idx in indices:
        images = {role: cv2.imread(str(args.images_dir / f"{idx}_{role}.png")) for role in ROLES}
        for obj_name in objects:
            for role in ROLES:
                if get_override_mask(overrides, idx, obj_name, role) is not None:
                    already += 1
                    continue
                override_point = get_override_point(overrides, idx, obj_name, role)
                override_missing = is_override_missing(overrides, idx, obj_name, role)
                override_candidate_index = get_override_candidate_index(overrides, idx, obj_name, role)
                if override_missing:
                    skipped_missing += 1
                    continue
                mask, _, _, _ = resolve_mask(
                    images[role], obj_name, args.box_threshold, args.text_threshold,
                    override_point=override_point, override_missing=override_missing,
                    override_candidate_index=override_candidate_index,
                )
                if mask is None:
                    skipped_unresolved += 1
                    continue
                set_override_mask(overrides, idx, obj_name, role, mask, point=override_point)
                frozen += 1

    save_overrides(overrides_path, overrides)
    print(
        f"Frozen {frozen} new mask(s); {already} already stable; {skipped_missing} confirmed-missing "
        f"left as-is; {skipped_unresolved} unresolved (no detection, no override) left as-is."
    )
    print(f"Saved to {overrides_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
