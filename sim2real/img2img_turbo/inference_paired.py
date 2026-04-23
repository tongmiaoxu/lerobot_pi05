# Vendored from: https://github.com/GaParmar/img2img-turbo
# Upstream commit: 86f54146590ffb4543c8cf85b5a36657da670924
# Original path in upstream: src/inference_paired.py
#
# Minimal edits: package-relative imports; tensors are moved to the model device instead of `.cuda()`.

import os
import argparse
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import torchvision.transforms.functional as F

from .image_prep import canny_from_pil
from .pix2pix_turbo import Pix2Pix_Turbo

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def _list_input_images(path: str) -> list[str]:
    """Single image file, or all supported images under a directory (sorted by name)."""
    if os.path.isfile(path):
        return [path]
    if os.path.isdir(path):
        out = []
        for name in sorted(os.listdir(path)):
            ext = os.path.splitext(name)[1].lower()
            if ext in _IMAGE_EXTS:
                out.append(os.path.join(path, name))
        return out
    raise FileNotFoundError(f"Not a file or directory: {path}")


def _default_output_dir(model_name: str, model_path: str) -> str:
    if model_path:
        parent = os.path.dirname(os.path.abspath(model_path))
        stem = os.path.splitext(os.path.basename(model_path))[0]
        return os.path.join(parent, f"{stem}_generated")
    if model_name:
        return os.path.join(os.getcwd(), f"{model_name}_generated")
    return "output"


def _triptych_pil(
    input_rgb: Image.Image,
    output_rgb: Image.Image,
    blend_alpha: float,
) -> Image.Image:
    """Left: input, center: model output, right: alpha blend of input and output."""
    w, h = input_rgb.size
    if output_rgb.size != (w, h):
        output_rgb = output_rgb.resize((w, h), Image.LANCZOS)
    blended = Image.blend(input_rgb, output_rgb, blend_alpha)
    canvas = Image.new("RGB", (w * 3, h))
    canvas.paste(input_rgb, (0, 0))
    canvas.paste(output_rgb, (w, 0))
    canvas.paste(blended, (w * 2, 0))
    return canvas


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_image",
        type=str,
        required=True,
        help="Path to one input image, or to a directory of images (e.g. gs_renders/).",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="a real-world robot camera image",
        help="Text prompt (project default matches README1.md turbo / deploy examples).",
    )
    parser.add_argument("--model_name", type=str, default="", help="name of the pretrained model to be used")
    parser.add_argument("--model_path", type=str, default="", help="path to a model state dict to be used")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory for outputs. Default: <checkpoint_dir>/<checkpoint_stem>_generated when using --model_path.",
    )
    parser.add_argument("--low_threshold", type=int, default=100, help="Canny low threshold")
    parser.add_argument("--high_threshold", type=int, default=200, help="Canny high threshold")
    parser.add_argument("--gamma", type=float, default=0.4, help="The sketch interpolation guidance amount")
    parser.add_argument("--seed", type=int, default=42, help="Random seed to be used")
    parser.add_argument("--use_fp16", action="store_true", help="Use Float16 precision for faster inference")
    parser.add_argument(
        "--resolution",
        type=int,
        default=224,
        help="Square side before VAE encode (multiple of 8). Default 224 matches README1.md / deploy --turbo-resolution.",
    )
    parser.add_argument(
        "--blend_alpha",
        type=float,
        default=0.5,
        help="Alpha for the right panel: blend = (1-alpha)*input + alpha*output (PIL Image.blend).",
    )
    args = parser.parse_args()

    if args.resolution <= 0 or args.resolution % 8 != 0:
        raise ValueError(f"--resolution must be a positive multiple of 8, got {args.resolution}")

    if bool(args.model_name) == bool(args.model_path):
        raise ValueError("Provide exactly one of --model_name or --model_path")

    if not (0.0 <= args.blend_alpha <= 1.0):
        raise ValueError(f"--blend_alpha must be in [0, 1], got {args.blend_alpha}")

    if args.output_dir is None:
        args.output_dir = _default_output_dir(args.model_name, args.model_path)

    input_paths = _list_input_images(args.input_image)
    if not input_paths:
        raise ValueError(f"No image files found under directory: {args.input_image}")

    os.makedirs(args.output_dir, exist_ok=True)

    model = Pix2Pix_Turbo(
        pretrained_name=args.model_name or None,
        pretrained_path=args.model_path or None,
    )
    model.set_eval()
    if args.use_fp16:
        model.half()

    dev = model.device

    for input_path in input_paths:
        input_original = Image.open(input_path).convert("RGB")
        original_width, original_height = input_original.size
        input_image = input_original.resize((args.resolution, args.resolution), Image.LANCZOS)
        base = os.path.basename(input_path)
        stem, _ext = os.path.splitext(base)
        bname = f"{stem}_translated.png"

        with torch.no_grad():
            if args.model_name == "edge_to_image":
                canny = canny_from_pil(input_image, args.low_threshold, args.high_threshold)
                canny_viz_inv = Image.fromarray(255 - np.array(canny))
                canny_viz_inv.save(os.path.join(args.output_dir, f"{stem}_canny.png"))
                c_t = F.to_tensor(canny).unsqueeze(0).to(dev)
                if args.use_fp16:
                    c_t = c_t.half()
                output_image = model(c_t, args.prompt)

            elif args.model_name == "sketch_to_image_stochastic":
                image_t = F.to_tensor(input_image) < 0.5
                c_t = image_t.unsqueeze(0).to(dev).float()
                torch.manual_seed(args.seed)
                B, C, H, W = c_t.shape
                noise = torch.randn((1, 4, H // 8, W // 8), device=c_t.device)
                if args.use_fp16:
                    c_t = c_t.half()
                    noise = noise.half()
                output_image = model(c_t, args.prompt, deterministic=False, r=args.gamma, noise_map=noise)

            else:
                c_t = F.to_tensor(input_image).unsqueeze(0).to(dev)
                if args.use_fp16:
                    c_t = c_t.half()
                output_image = model(c_t, args.prompt)

            output_pil = transforms.ToPILImage()(output_image[0].cpu() * 0.5 + 0.5)
            output_pil = output_pil.resize((original_width, original_height), Image.LANCZOS)

        triptych = _triptych_pil(input_original, output_pil, args.blend_alpha)
        triptych.save(os.path.join(args.output_dir, bname))
