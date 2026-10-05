#!/usr/bin/env python3
"""Evaluate a trained pix2pix-turbo checkpoint on a held-out eval set that was never used for
training (e.g. data_val_set_mug/stationary), the turbo counterpart of sim2real/eval_pix2pix.py.

Produces results in the same `<output_dir>/results/<name>/test_latest/images` layout that
sim2real/eval_pix2pix.py's pix2pix baseline produces, so downstream tooling
(scripts/review_pix2pix_masks.py, scripts/eval_pix2pix_metrics.py) needs no changes.

Because data_val_set_<task>/<camera> is the same held-out set used for the pix2pix baseline, and
prepare_test_only_dataset()'s pairing is deterministic for a fixed --sim-dir/--real-dir/--camera,
index N's real_A/real_B here are *visually* identical to the pix2pix baseline run over the same
inputs *as long as --metrics-resolution matches the pix2pix run's --resolution* (see that flag's
help — pix2pix's --preprocess resize is a non-aspect-preserving bicubic stretch to a square
canvas, not a crop, and source images here are not square, so a resolution/aspect mismatch would
both bias LPIPS/J/F between the two baselines AND silently misplace any real_A/real_B point
overrides copied over via scripts/seed_turbo_overrides.py, since a pixel coordinate only means
the same thing on identically-sized canvases). NOT literally pixel-identical, though: this
script saves real_A/real_B straight from a resized PIL image, while eval_pix2pix.py's dataloader
round-trips them through the network's normalized [-1, 1] tensor range (pix2pix/util/util.py's
tensor2im(), even though real_A/real_B never pass through the generator itself), which introduces
up to ±1/255 rounding noise on a small fraction of pixels. That's sub-perceptual and doesn't
affect LPIPS or J&F (masks are matched by index, and reused mask_overrides.json entries store
the actual saved mask/point, not a re-derivation from these exact bytes) — just don't expect a
byte-for-byte diff/hash to match between the two runs' real_A/real_B PNGs. With matching
--metrics-resolution, a pix2pix run's already-reviewed mask_overrides.json real_A/real_B entries
can be copied straight into a turbo run's own mask_overrides.json instead of re-reviewing them —
see scripts/seed_turbo_overrides.py. Only fake_B (this checkpoint's own generated images) needs
fresh review.

Usage:
  python sim2real/eval_turbo.py \\
    --sim-dir data_val_set_mug/stationary/gs_renders \\
    --real-dir data_val_set_mug/stationary/real_captures \\
    --dataset-dir outputs/turbo_sim2real_stationary_dino/eval_dataset \\
    --checkpoint outputs/turbo_sim2real_stationary_dino/checkpoints/model_30001.pkl \\
    --output-dir outputs/turbo_sim2real_stationary_dino \\
    --name place_mug_stationary_turbo
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from sim2real.train import prepare_test_only_dataset  # noqa: E402
from sim2real.translator import SimToRealTranslator  # noqa: E402


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
    parser.add_argument("--dataset-dir", type=Path, required=True, help="Where to write the prepared test_A/test_B pairs (symlinks).")
    parser.add_argument(
        "--checkpoint", type=Path, required=True,
        help="Pix2pix-turbo .pkl checkpoint, e.g. outputs/turbo_sim2real_stationary_dino/checkpoints/model_30001.pkl "
        "(matches TaskProfile.turbo_checkpoint_filename under turbo_output_stationary/wrist).",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Where results/<name>/test_latest/images is written (mirrors eval_pix2pix.py's --output-dir).",
    )
    parser.add_argument("--name", type=str, required=True, help="Result-set name, e.g. place_mug_stationary_turbo.")
    parser.add_argument(
        "--prompt", type=str, default="a real-world robot camera image",
        help="Must match the prompt used during this checkpoint's training.",
    )
    parser.add_argument(
        "--resolution", type=int, default=224,
        help="Turbo model's own inference resolution — must match the training resolution. The "
        "model is always fed a single native-resolution -> --resolution resize, exactly like "
        "training's build_transform, regardless of --metrics-resolution below.",
    )
    parser.add_argument(
        "--metrics-resolution", type=int, default=256,
        help="Square canvas real_A/fake_B/real_B are saved at for scoring — must match the "
        "pix2pix baseline's --resolution (eval_pix2pix.py default: 256) so LPIPS/Grounding-DINO/"
        "SAM2 score both baselines on identical geometry. A direct (non-aspect-preserving) "
        "bicubic stretch, matching pix2pix's --preprocess resize exactly — not a crop. Applied "
        "only after translation, to the model's already-generated output, never to its input, so "
        "it can't affect translation quality (only --resolution does that).",
    )
    parser.add_argument("--device", type=str, default=None, help="Torch device (default: CUDA if available).")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing prepared eval dataset directory and/or results.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    num_test = prepare_test_only_dataset(args.sim_dir, args.real_dir, args.camera, args.dataset_dir, args.overwrite)
    print(f"Prepared unseen eval dataset at {args.dataset_dir} (n={num_test}).")

    images_dir = args.output_dir / "results" / args.name / "test_latest" / "images"
    if images_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Results already exist: {images_dir}. Use --overwrite to replace them.")
        shutil.rmtree(images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)

    translator = SimToRealTranslator(
        args.checkpoint, prompt=args.prompt, resolution=args.resolution, device=args.device
    )

    metrics_size = (args.metrics_resolution, args.metrics_resolution)
    test_a_dir = args.dataset_dir / "test_A"
    test_b_dir = args.dataset_dir / "test_B"
    stems = sorted(p.stem for p in test_a_dir.glob("*.png"))
    for stem in stems:
        real_a_native = Image.open(test_a_dir / f"{stem}.png").convert("RGB")
        real_b_native = Image.open(test_b_dir / f"{stem}.png").convert("RGB")

        # Translate straight from the native-resolution image, exactly like training data flows
        # through build_transform's single native -> --resolution resize — no extra resampling
        # pass before the model sees it. SimToRealTranslator.translate() itself resizes its
        # output back to its input's size (native here), which we then resize again below; that
        # second resize only touches the already-generated pixels, not what the model saw.
        fake_b_native = translator.translate(np.array(real_a_native))

        # Stretch everything to the pix2pix baseline's scoring canvas as the last step, after
        # generation, so LPIPS/Grounding-DINO/SAM2 see identical geometry across both baselines
        # (see --metrics-resolution help) without perturbing the model's own input.
        real_a = np.array(real_a_native.resize(metrics_size, Image.BICUBIC))
        real_b = np.array(real_b_native.resize(metrics_size, Image.BICUBIC))
        fake_b = np.array(Image.fromarray(fake_b_native).resize(metrics_size, Image.BICUBIC))

        Image.fromarray(real_a).save(images_dir / f"{stem}_real_A.png")
        Image.fromarray(fake_b).save(images_dir / f"{stem}_fake_B.png")
        Image.fromarray(real_b).save(images_dir / f"{stem}_real_B.png")

    print(f"Wrote {len(stems)} triplets to {images_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
