#!/usr/bin/env python3
"""Copy real_A/real_B mask fixes from an already-reviewed pix2pix mask_overrides.json into a
turbo run's own mask_overrides.json.

data_val_set_<task>/<camera> is the same held-out set used to eval both the pix2pix baseline
(sim2real/eval_pix2pix.py) and a pix2pix-turbo checkpoint (sim2real/eval_turbo.py), and pairing
is deterministic for a fixed --sim-dir/--real-dir/--camera, so index N's real_A/real_B images
are pixel-identical between the two runs. Any real_A/real_B override a reviewer already made for
the pix2pix run is therefore exactly as valid for the turbo run at the same index — no need to
re-click through them. fake_B is never copied (each run has its own generated images); review
that fresh with scripts/review_pix2pix_masks.py (its default --role is already fake_B).

Usage:
  python scripts/seed_turbo_overrides.py \\
    --src outputs/pix2pix_stationary_mug/mask_overrides.json \\
    --dst outputs/turbo_sim2real_stationary_dino/mask_overrides.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", type=Path, required=True, help="Already-reviewed pix2pix mask_overrides.json.")
    parser.add_argument(
        "--dst", type=Path, required=True,
        help="Turbo run's mask_overrides.json (created if missing, merged in-place if present).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    src = json.loads(args.src.read_text())
    dst = json.loads(args.dst.read_text()) if args.dst.exists() else {}

    copied = 0
    for idx, per_obj in src.items():
        for obj_name, per_role in per_obj.items():
            for role, entry in per_role.items():
                if role not in ("real_A", "real_B"):
                    continue
                dst.setdefault(idx, {}).setdefault(obj_name, {})[role] = entry
                copied += 1

    args.dst.parent.mkdir(parents=True, exist_ok=True)
    args.dst.write_text(json.dumps(dst, indent=2))
    print(f"Copied {copied} real_A/real_B override(s) from {args.src} to {args.dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
