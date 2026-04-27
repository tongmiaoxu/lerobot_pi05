#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

from PIL import Image

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from sim2real.img2img_turbo.my_utils.training_utils import parse_args_paired_training


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare paired sim/real image data for pix2pix-turbo and optionally launch "
            "the upstream fine-tuning script."
        )
    )
    parser.add_argument("--sim-dir", type=Path, required=True, help="Directory containing simulation images.")
    parser.add_argument("--real-dir", type=Path, required=True, help="Directory containing paired real images.")
    parser.add_argument(
        "--camera",
        choices=("stationary", "wrist"),
        default=None,
        help="Filter recursive paired data to one camera directory, e.g. episode_*/stationary or episode_*/wrist.",
    )
    parser.add_argument("--dataset-dir", type=Path, required=True, help="Prepared pix2pix-turbo dataset output.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Training output directory (checkpoints under output_dir/checkpoints/).")
    parser.add_argument(
        "--prompt",
        type=str,
        default="a real-world robot camera image",
        help="Text prompt used for every paired example.",
    )
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Held-out validation split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for the split.")
    parser.add_argument("--resolution", type=int, default=512, help="Training resolution passed upstream.")
    parser.add_argument("--train-batch-size", type=int, default=2, help="Training batch size.")
    parser.add_argument(
        "--dataloader-num-workers",
        type=int,
        default=4,
        help="Number of background dataloader workers used by the upstream trainer.",
    )
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
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Accumulate gradients over N micro-batches before an optimizer step (reduces peak VRAM).",
    )
    parser.add_argument("--max-train-steps", type=int, default=10000, help="Maximum training steps.")
    parser.add_argument("--learning-rate", type=float, default=5e-6, help="AdamW learning rate.")
    parser.add_argument("--lambda-gan", type=float, default=0.0, help="Weight for the GAN loss.")
    parser.add_argument("--lambda-clipsim", type=float, default=0.0, help="Weight for the CLIP similarity loss.")
    parser.add_argument("--lambda-l2", type=float, default=10.0, help="Weight for the pixelwise L2 loss.")
    parser.add_argument("--lambda-lpips", type=float, default=2.0, help="Weight for the LPIPS perceptual loss.")
    parser.add_argument(
        "--lambda-dinov3-pixel",
        type=float,
        default=0.0,
        help="Weight for pixel-aligned L2 between DINOv3 patch features of sim inputs and generated images.",
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
    parser.add_argument("--viz-freq", type=int, default=100, help="Upstream visualization frequency.")
    parser.add_argument("--eval-freq", type=int, default=100, help="Upstream validation frequency.")
    parser.add_argument(
        "--checkpointing-steps",
        type=int,
        default=1000,
        help="Checkpoint save frequency for upstream training.",
    )
    parser.add_argument(
        "--mixed-precision",
        choices=("no", "fp16", "bf16"),
        default="fp16",
        help="Precision mode forwarded to accelerate.",
    )
    parser.add_argument(
        "--report-to",
        type=str,
        default="wandb",
        help="Tracker backend forwarded to the upstream trainer.",
    )
    parser.add_argument(
        "--task-id",
        type=str,
        default=None,
        help=(
            "Optional lerobot task id (e.g. pick_mug, place_mug). When set and "
            "--tracker-project-name is omitted, the W&B project defaults to that task's "
            "`wandb_project` in src/lerobot/tasks/task_profiles.py."
        ),
    )
    parser.add_argument(
        "--tracker-project-name",
        type=str,
        default=None,
        help=(
            "W&B / Accelerate tracker project name. If omitted: use --task-id's wandb_project, "
            "else default pix2pix_turbo_sim2real."
        ),
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="Weights & Biases run display name (default: random adjective-noun). Example: pick_mug_pix2pix_20260418.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only create the paired dataset layout; do not launch training.",
    )
    parser.add_argument(
        "--enable-xformers",
        action="store_true",
        help="Enable xformers attention in the upstream trainer.",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Enable gradient checkpointing in the upstream trainer.",
    )
    parser.add_argument(
        "--track-val-fid",
        action="store_true",
        help="Enable Clean-FID tracking in the upstream trainer.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing prepared dataset directory.",
    )
    args = parser.parse_args()

    _src = _ROOT / "src"
    if str(_src) not in sys.path:
        sys.path.insert(0, str(_src))

    if args.tracker_project_name is None:
        if args.task_id:
            from lerobot.tasks.task_profiles import get_task_profile

            args.tracker_project_name = get_task_profile(args.task_id).wandb_project
        else:
            args.tracker_project_name = "pix2pix_turbo_sim2real"

    if not 0.0 <= args.val_ratio < 1.0:
        raise ValueError("--val-ratio must be in [0, 1).")
    if args.max_pairs is not None and args.max_pairs <= 0:
        raise ValueError("--max-pairs must be positive when provided.")
    return args


def list_images(root: Path) -> dict[str, Path]:
    if not root.exists():
        raise FileNotFoundError(root)

    images: dict[str, Path] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        rel_key = path.relative_to(root).with_suffix("").as_posix()
        if rel_key in images:
            raise ValueError(f"Duplicate image key {rel_key!r} under {root}")
        images[rel_key] = path
    return images


def match_pairs(sim_dir: Path, real_dir: Path) -> list[tuple[Path, Path]]:
    sim_images = list_images(sim_dir)
    real_images = list_images(real_dir)
    common_keys = sorted(set(sim_images) & set(real_images))

    if not common_keys:
        sim_by_name: dict[str, Path] = {}
        for path in sim_images.values():
            if path.stem in sim_by_name:
                raise ValueError(
                    f"Filename-stem fallback is ambiguous in {sim_dir}: duplicate stem {path.stem!r}"
                )
            sim_by_name[path.stem] = path

        real_by_name: dict[str, Path] = {}
        for path in real_images.values():
            if path.stem in real_by_name:
                raise ValueError(
                    f"Filename-stem fallback is ambiguous in {real_dir}: duplicate stem {path.stem!r}"
                )
            real_by_name[path.stem] = path

        common_names = sorted(set(sim_by_name) & set(real_by_name))
        if not common_names:
            raise ValueError(
                "No paired files found. Match sim/real images by relative path or filename stem."
            )
        return [(sim_by_name[name], real_by_name[name]) for name in common_names]

    return [(sim_images[key], real_images[key]) for key in common_keys]


def filter_pairs_by_camera(
    pairs: list[tuple[Path, Path]],
    camera: str | None,
) -> list[tuple[Path, Path]]:
    if camera is None:
        return pairs

    filtered_pairs = [
        (sim_path, real_path)
        for sim_path, real_path in pairs
        if camera in sim_path.parts and camera in real_path.parts
    ]
    if not filtered_pairs:
        raise ValueError(
            f"--camera {camera!r} produced an empty dataset. "
            "Expected paths like episode_*/stationary/frame_*.png or episode_*/wrist/frame_*.png."
        )
    return filtered_pairs


def split_pairs(
    pairs: list[tuple[Path, Path]],
    val_ratio: float,
    seed: int,
) -> tuple[list[tuple[Path, Path]], list[tuple[Path, Path]]]:
    if not pairs:
        raise ValueError("No image pairs available for training.")

    shuffled = pairs[:]
    random.Random(seed).shuffle(shuffled)

    val_count = 0
    if len(shuffled) > 1 and val_ratio > 0:
        val_count = max(1, int(round(len(shuffled) * val_ratio)))
        val_count = min(val_count, len(shuffled) - 1)

    val_pairs = shuffled[:val_count]
    train_pairs = shuffled[val_count:]
    return train_pairs, val_pairs


def select_pairs(
    pairs: list[tuple[Path, Path]],
    selection: str,
    max_pairs: int | None,
) -> list[tuple[Path, Path]]:
    if selection == "odd":
        pairs = [pair for idx, pair in enumerate(pairs) if idx % 2 == 1]
    elif selection == "even":
        pairs = [pair for idx, pair in enumerate(pairs) if idx % 2 == 0]

    if max_pairs is not None:
        pairs = pairs[:max_pairs]

    if not pairs:
        raise ValueError("Pair selection produced an empty dataset.")
    return pairs


def save_rgb_png(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        image.convert("RGB").save(destination)


def write_split(
    pairs: list[tuple[Path, Path]],
    dataset_dir: Path,
    split: str,
    prompt: str,
) -> None:
    input_dir = dataset_dir / f"{split}_A"
    output_dir = dataset_dir / f"{split}_B"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    prompts: dict[str, str] = {}
    for idx, (sim_path, real_path) in enumerate(pairs):
        filename = f"{idx:06d}.png"
        save_rgb_png(sim_path, input_dir / filename)
        save_rgb_png(real_path, output_dir / filename)
        prompts[filename] = prompt

    with (dataset_dir / f"{split}_prompts.json").open("w", encoding="utf-8") as handle:
        json.dump(prompts, handle, indent=2)


def build_image_prep(resolution: int) -> str:
    """Tag passed to training_utils.build_transform; supports any N via resize_NxN."""
    if resolution <= 0:
        raise ValueError(f"resolution must be positive, got {resolution}")
    return f"resize_{resolution}x{resolution}"


def prepare_dataset(args: argparse.Namespace) -> tuple[int, int]:
    pairs = match_pairs(args.sim_dir, args.real_dir)
    pairs = filter_pairs_by_camera(pairs, args.camera)
    pairs = select_pairs(pairs, args.pair_selection, args.max_pairs)
    train_pairs, val_pairs = split_pairs(pairs, args.val_ratio, args.seed)

    if args.dataset_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Dataset directory already exists: {args.dataset_dir}. Use --overwrite to replace it."
            )
        shutil.rmtree(args.dataset_dir)

    args.dataset_dir.mkdir(parents=True, exist_ok=True)
    write_split(train_pairs, args.dataset_dir, "train", args.prompt)
    write_split(val_pairs, args.dataset_dir, "test", args.prompt)
    return len(train_pairs), len(val_pairs)


def launch_training(args: argparse.Namespace) -> None:
    import torch

    if not torch.cuda.is_available():
        print(
            "\n[sim2real] No CUDA device is visible to PyTorch. This training stack expects a working NVIDIA GPU.\n"
            "Check `nvidia-smi` and that you are not inside a CPU-only container without GPU passthrough.\n"
        )
        raise SystemExit(1)
    try:
        torch.zeros(1, device="cuda")
    except RuntimeError as err:
        print(
            "\n[sim2real] CUDA failed to initialize. Typical case: your **NVIDIA driver is older** than the "
            "**CUDA userland** bundled with the `torch` wheel you installed (PyTorch refuses to run on GPU).\n\n"
            "Diagnose:\n"
            "  nvidia-smi\n"
            "  python -c \"import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)\"\n\n"
            "Fix (pick one):\n"
            "  1) **Upgrade the NVIDIA driver** so it supports the CUDA version your PyTorch build needs: "
            "https://www.nvidia.com/Download/index.aspx\n"
            "  2) **Reinstall PyTorch** (and matching xformers) for an **older** CUDA toolkit your driver already "
            "supports, e.g. cu121 or cu118 wheels from https://pytorch.org/get-started/locally/\n\n"
            f"Original error: {err!r}\n"
        )
        raise SystemExit(1) from err

    image_prep = build_image_prep(args.resolution)
    train_argv = [
        "--pretrained_model_name_or_path",
        "stabilityai/sd-turbo",
        "--output_dir",
        str(args.output_dir),
        "--dataset_folder",
        str(args.dataset_dir),
        "--resolution",
        str(args.resolution),
        "--train_batch_size",
        str(args.train_batch_size),
        "--dataloader_num_workers",
        str(args.dataloader_num_workers),
        "--gradient_accumulation_steps",
        str(args.gradient_accumulation_steps),
        "--max_train_steps",
        str(args.max_train_steps),
        "--learning_rate",
        str(args.learning_rate),
        "--lambda_gan",
        str(args.lambda_gan),
        "--lambda_clipsim",
        str(args.lambda_clipsim),
        "--lambda_l2",
        str(args.lambda_l2),
        "--lambda_lpips",
        str(args.lambda_lpips),
        "--lambda_dinov3_pixel",
        str(args.lambda_dinov3_pixel),
        "--dinov3_model_name",
        str(args.dinov3_model_name),
        "--viz_freq",
        str(args.viz_freq),
        "--eval_freq",
        str(args.eval_freq),
        "--checkpointing_steps",
        str(args.checkpointing_steps),
        "--mixed_precision",
        str(args.mixed_precision),
        "--report_to",
        str(args.report_to),
        "--tracker_project_name",
        str(args.tracker_project_name),
        "--train_image_prep",
        image_prep,
        "--test_image_prep",
        image_prep,
    ]

    if args.wandb_run_name:
        train_argv.extend(["--wandb_run_name", args.wandb_run_name])

    if args.enable_xformers:
        train_argv.append("--enable_xformers_memory_efficient_attention")
    if args.gradient_checkpointing:
        train_argv.append("--gradient_checkpointing")
    if args.track_val_fid:
        train_argv.append("--track_val_fid")
    if args.dinov3_trust_remote_code:
        train_argv.append("--dinov3_trust_remote_code")

    try:
        from sim2real.img2img_turbo.train_pix2pix_turbo import main as train_pix2pix_main
    except (ImportError, OSError, RuntimeError) as err:
        print(
            "\n[sim2real] Could not import pix2pix-turbo training (diffusers / xformers / flash-attn).\n"
            "Common cause: flash_attn was compiled against a different PyTorch (undefined symbol in "
            "flash_attn_2_cuda), and xformers imports it when diffusers loads.\n\n"
            "Try in this conda env:\n"
            "  pip uninstall -y flash-attn flash_attn\n"
            "If errors persist:\n"
            "  pip uninstall -y xformers && pip install 'xformers==<version matching your torch>'\n"
            "Also check the PyTorch warning about the NVIDIA driver vs the CUDA build you installed.\n"
        )
        raise err

    train_args = parse_args_paired_training(train_argv)
    train_pix2pix_main(train_args)


def main() -> int:
    args = parse_args()
    train_count, val_count = prepare_dataset(args)
    print(
        f"Prepared paired dataset at {args.dataset_dir} "
        f"(train={train_count}, val={val_count})."
    )

    if args.prepare_only:
        return 0

    print(
        "Training uses vendored sim2real/img2img_turbo; install sim2real/requirements.txt plus "
        "lpips, wandb, clean-fid, vision_aided_loss, and OpenAI CLIP (see upstream requirements.txt)."
    )
    launch_training(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
