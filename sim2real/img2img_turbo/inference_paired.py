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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_image", type=str, required=True, help="path to the input image")
    parser.add_argument("--prompt", type=str, required=True, help="the prompt to be used")
    parser.add_argument("--model_name", type=str, default="", help="name of the pretrained model to be used")
    parser.add_argument("--model_path", type=str, default="", help="path to a model state dict to be used")
    parser.add_argument("--output_dir", type=str, default="output", help="the directory to save the output")
    parser.add_argument("--low_threshold", type=int, default=100, help="Canny low threshold")
    parser.add_argument("--high_threshold", type=int, default=200, help="Canny high threshold")
    parser.add_argument("--gamma", type=float, default=0.4, help="The sketch interpolation guidance amount")
    parser.add_argument("--seed", type=int, default=42, help="Random seed to be used")
    parser.add_argument("--use_fp16", action="store_true", help="Use Float16 precision for faster inference")
    parser.add_argument("--resolution", type=int, default=512, help="Resolution used during training (must be a multiple of 8). Input is resized to this before inference and output is resized back to the original dimensions.")
    args = parser.parse_args()

    if args.resolution <= 0 or args.resolution % 8 != 0:
        raise ValueError(f"--resolution must be a positive multiple of 8, got {args.resolution}")

    if bool(args.model_name) == bool(args.model_path):
        raise ValueError("Provide exactly one of --model_name or --model_path")

    os.makedirs(args.output_dir, exist_ok=True)

    model = Pix2Pix_Turbo(
        pretrained_name=args.model_name or None,
        pretrained_path=args.model_path or None,
    )
    model.set_eval()
    if args.use_fp16:
        model.half()

    dev = model.device

    input_image = Image.open(args.input_image).convert("RGB")
    original_width, original_height = input_image.size
    input_image = input_image.resize((args.resolution, args.resolution), Image.LANCZOS)
    bname = os.path.basename(args.input_image).replace(".png", "_translated.png")

    with torch.no_grad():
        if args.model_name == "edge_to_image":
            canny = canny_from_pil(input_image, args.low_threshold, args.high_threshold)
            canny_viz_inv = Image.fromarray(255 - np.array(canny))
            canny_viz_inv.save(os.path.join(args.output_dir, bname.replace(".png", "_canny.png")))
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

    output_pil.save(os.path.join(args.output_dir, bname))
