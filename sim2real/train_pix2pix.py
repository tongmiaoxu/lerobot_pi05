#!/usr/bin/env python3
"""Train the original pix2pix model (Isola et al. 2017, https://arxiv.org/pdf/1611.07004)
on the same sim/real paired data used for pix2pix-turbo, for a per-task/per-camera baseline
comparison. Reuses `sim2real/train.py`'s pairing/split/write logic so both trainers see the
identical train/val split, then subprocess-launches the vendored `sim2real/pix2pix/train.py`
with a custom `--dataset_mode sim2real` that reads `train_A/train_B` directly.

Post-training eval runs against --eval-sim-dir/--eval-real-dir, a holdout that was never part
of --sim-dir/--real-dir's train/val split (e.g. data_val_set_mug/stationary), not the val split
of the training data itself. See sim2real/eval_pix2pix.py to re-run eval standalone against an
already-trained checkpoint.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from sim2real.train import check_cuda, prepare_dataset, prepare_test_only_dataset

_PIX2PIX_TRAIN_SCRIPT = Path(__file__).resolve().parent / "pix2pix" / "train.py"
_PIX2PIX_TEST_SCRIPT = Path(__file__).resolve().parent / "pix2pix" / "test.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare paired sim/real image data and train the original pix2pix model."
    )
    parser.add_argument("--sim-dir", type=Path, required=True, help="Directory containing simulation images.")
    parser.add_argument("--real-dir", type=Path, required=True, help="Directory containing paired real images.")
    parser.add_argument(
        "--camera",
        choices=("stationary", "wrist"),
        default=None,
        help="Filter recursive paired data to one camera directory, e.g. episode_*/stationary or episode_*/wrist.",
    )
    parser.add_argument("--dataset-dir", type=Path, required=True, help="Prepared train_A/train_B dataset output.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Checkpoints dir (pix2pix's --checkpoints_dir).")
    parser.add_argument(
        "--eval-sim-dir", type=Path, default=None,
        help=(
            "Sim/render directory for the post-training eval, held out of training entirely "
            "(e.g. data_val_set_mug/stationary/gs_renders), NOT the val split of --sim-dir/--real-dir. "
            "Required unless --prepare-only."
        ),
    )
    parser.add_argument(
        "--eval-real-dir", type=Path, default=None,
        help="Real directory paired with --eval-sim-dir. Required unless --prepare-only.",
    )
    parser.add_argument(
        "--eval-dataset-dir", type=Path, default=None,
        help="Where to write the prepared eval test_A/test_B set. Default: <output-dir>/eval_dataset.",
    )
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Held-out validation split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for the split.")
    parser.add_argument(
        "--pair-selection",
        choices=("all", "odd", "even"),
        default="all",
        help="Subset matched pairs by their sorted index before the train/val split.",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=None,
        help="Keep only the first N matched pairs after optional odd/even filtering.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=256,
        help=(
            "Training resolution. pix2pix's default unet_256 generator needs input size "
            "divisible by 256 (8 stride-2 downsamples), so this must stay a multiple of 256 "
            "unless --netG is changed."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Training batch size (paper default: 1).")
    parser.add_argument("--n-epochs", type=int, default=100, help="Epochs at the initial learning rate.")
    parser.add_argument("--n-epochs-decay", type=int, default=100, help="Epochs linearly decaying lr to zero.")
    parser.add_argument(
        "--save-epoch-freq",
        type=int,
        default=0,
        help="Numbered epoch checkpoint frequency. Default 0 keeps only latest_net_*.pth.",
    )
    parser.add_argument("--lr", type=float, default=2e-4, help="Adam learning rate (paper default).")
    parser.add_argument("--lambda-l1", type=float, default=100.0, help="Weight for the L1 reconstruction loss.")
    parser.add_argument(
        "--lambda-dinov3-pixel",
        type=float,
        default=0.0,
        help="Weight for pixel-aligned L2 between DINOv3 patch features of sim input and generated image. "
        "0 (default) disables DINOv3 entirely, training the original pix2pix model unmodified.",
    )
    parser.add_argument(
        "--dinov3-model-name",
        type=str,
        default="facebook/dinov3-vits16-pretrain-lvd1689m",
        help="Hugging Face model id or local path for the frozen DINOv3 feature extractor.",
    )
    parser.add_argument(
        "--dinov3-trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True when loading the DINOv3 model from Hugging Face.",
    )
    parser.add_argument(
        "--task",
        dest="task_id",
        type=str,
        default=None,
        help="Optional lerobot task id (e.g. pick_mug, place_mug), used to name the experiment.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="pix2pix experiment name (checkpoints saved under output_dir/name). Default: '<task>_<camera>_pix2pix'.",
    )
    parser.add_argument("--use-wandb", action="store_true", help="Enable wandb logging in the upstream trainer.")
    parser.add_argument(
        "--wandb-project-name",
        type=str,
        default=None,
        help="wandb project name. If omitted: use --task's wandb_project, else pix2pix's own default.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only create the paired dataset layout; do not launch training.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing prepared dataset directory.",
    )
    args = parser.parse_args()

    if not 0.0 <= args.val_ratio < 1.0:
        raise ValueError("--val-ratio must be in [0, 1).")
    if args.max_pairs is not None and args.max_pairs <= 0:
        raise ValueError("--max-pairs must be positive when provided.")
    if args.resolution % 256 != 0:
        raise ValueError("--resolution must be a multiple of 256 for the default unet_256 generator.")
    if args.save_epoch_freq < 0:
        raise ValueError("--save-epoch-freq must be non-negative.")
    if not args.prepare_only and (args.eval_sim_dir is None or args.eval_real_dir is None):
        raise ValueError("--eval-sim-dir and --eval-real-dir are required unless --prepare-only.")

    args.prompt = "a real-world robot camera image"  # unused by pix2pix; kept so prepare_dataset() can run unmodified.

    if args.name is None:
        camera_tag = args.camera or "all_cameras"
        args.name = f"{args.task_id or 'sim2real'}_{camera_tag}_pix2pix"

    if args.eval_dataset_dir is None:
        args.eval_dataset_dir = args.output_dir / "eval_dataset"

    if args.wandb_project_name is None and args.task_id:
        from lerobot.tasks.task_profiles import get_task_profile

        args.wandb_project_name = get_task_profile(args.task_id).wandb_project

    return args


def launch_training(args: argparse.Namespace) -> None:
    check_cuda()

    train_argv = [
        sys.executable,
        str(_PIX2PIX_TRAIN_SCRIPT),
        "--dataroot", str(args.dataset_dir),
        "--checkpoints_dir", str(args.output_dir),
        "--name", args.name,
        "--model", "pix2pix",
        "--dataset_mode", "sim2real",
        "--direction", "AtoB",
        "--load_size", str(args.resolution),
        "--crop_size", str(args.resolution),
        "--preprocess", "resize",
        "--no_flip",
        "--batch_size", str(args.batch_size),
        "--n_epochs", str(args.n_epochs),
        "--n_epochs_decay", str(args.n_epochs_decay),
        "--save_epoch_freq", str(args.save_epoch_freq),
        "--lr", str(args.lr),
        "--lambda_L1", str(args.lambda_l1),
        "--lambda_dinov3_pixel", str(args.lambda_dinov3_pixel),
        "--dinov3_model_name", str(args.dinov3_model_name),
    ]

    if args.dinov3_trust_remote_code:
        train_argv.append("--dinov3_trust_remote_code")
    if args.use_wandb:
        train_argv.append("--use_wandb")
    if args.wandb_project_name:
        train_argv.extend(["--wandb_project_name", args.wandb_project_name])

    subprocess.run(train_argv, cwd=str(_ROOT), check=True)


def run_evaluation(args: argparse.Namespace) -> None:
    """Run inference with the 'latest' checkpoint over a held-out eval set that was never
    part of the training/val split (--eval-sim-dir/--eval-real-dir), not the val split
    carved out of --sim-dir/--real-dir.

    Mirrors the layout produced when running `sim2real/pix2pix/test.py` by hand:
    results land under `<output_dir>/results/<name>/test_latest/images`.
    """
    num_test = prepare_test_only_dataset(
        args.eval_sim_dir, args.eval_real_dir, args.camera, args.eval_dataset_dir, args.overwrite
    )
    print(f"Prepared unseen eval dataset at {args.eval_dataset_dir} (n={num_test}).")

    test_argv = [
        sys.executable,
        str(_PIX2PIX_TEST_SCRIPT),
        "--dataroot", str(args.eval_dataset_dir),
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
    train_count, val_count = prepare_dataset(args)
    print(f"Prepared paired dataset at {args.dataset_dir} (train={train_count}, val={val_count}).")

    if args.prepare_only:
        return 0

    launch_training(args)
    run_evaluation(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
