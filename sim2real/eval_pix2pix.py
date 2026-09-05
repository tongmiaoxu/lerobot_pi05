#!/usr/bin/env python3
"""Evaluate a trained pix2pix checkpoint on a held-out eval set that was never used for
training or validation (e.g. data_val_set_mug/stationary), instead of the train/val split
carved out of the same collection used for training.

Produces results in the same `<output_dir>/results/<name>/test_latest/images` layout that
`sim2real/train_pix2pix.py`'s post-training eval produces, so downstream tooling
(scripts/eval_pix2pix_metrics.py) needs no changes.

Usage:
  python sim2real/eval_pix2pix.py \\
    --sim-dir data_val_set_mug/stationary/gs_renders \\
    --real-dir data_val_set_mug/stationary/real_captures \\
    --dataset-dir outputs/pix2pix_stationary_mug/eval_dataset \\
    --output-dir outputs/pix2pix_stationary_mug \\
    --name place_mug_stationary_pix2pix
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from sim2real.train import prepare_test_only_dataset

_PIX2PIX_TEST_SCRIPT = Path(__file__).resolve().parent / "pix2pix" / "test.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--sim-dir", type=Path, required=True,
        help="Unseen eval sim/render directory, e.g. data_val_set_mug/stationary/gs_renders.",
    )
    parser.add_argument(
        "--real-dir", type=Path, required=True,
        help="Unseen eval real directory, e.g. data_val_set_mug/stationary/real_captures.",
    )
    parser.add_argument(
        "--camera", choices=("stationary", "wrist"), default=None,
        help="Optional camera filter; usually unnecessary since --sim-dir/--real-dir already scope to one camera.",
    )
    parser.add_argument("--dataset-dir", type=Path, required=True, help="Where to write the prepared test_A/test_B eval set.")
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Existing checkpoints_dir for the trained model, e.g. outputs/pix2pix_stationary_mug.",
    )
    parser.add_argument(
        "--name", type=str, required=True,
        help="pix2pix experiment name used during training, e.g. place_mug_stationary_pix2pix.",
    )
    parser.add_argument("--resolution", type=int, default=256, help="Must match the training resolution.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing prepared eval dataset directory.")
    return parser.parse_args()


def run_evaluation(args: argparse.Namespace, num_test: int) -> None:
    test_argv = [
        sys.executable,
        str(_PIX2PIX_TEST_SCRIPT),
        "--dataroot", str(args.dataset_dir),
        "--checkpoints_dir", str(args.output_dir),
        "--results_dir", str(args.output_dir / "results"),
        "--name", args.name,
        "--model", "pix2pix",
        "--dataset_mode", "sim2real",
        "--direction", "AtoB",
        "--load_size", str(args.resolution),
        "--crop_size", str(args.resolution),
        "--preprocess", "resize",
        "--no_flip",
        "--num_test", str(num_test),
    ]
    subprocess.run(test_argv, cwd=str(_ROOT), check=True)


def main() -> int:
    args = parse_args()
    num_test = prepare_test_only_dataset(
        args.sim_dir, args.real_dir, args.camera, args.dataset_dir, args.overwrite
    )
    print(f"Prepared unseen eval dataset at {args.dataset_dir} (n={num_test}).")
    run_evaluation(args, num_test)
    return 0


if __name__ == "__main__":
    sys.exit(main())
