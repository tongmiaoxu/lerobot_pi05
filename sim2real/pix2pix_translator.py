from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# sim2real/pix2pix is vendored to be run as a script (its models/__init__.py does an absolute
# `from models.base_model import BaseModel`, which only resolves when sim2real/pix2pix itself is
# on sys.path -- see sim2real/pix2pix/__init__.py). networks.py has no such import (stdlib/torch
# only), so load it directly by file path instead of importing the models package.
_NETWORKS_PATH = Path(__file__).resolve().parent / "pix2pix" / "models" / "networks.py"


def _load_networks_module():
    spec = importlib.util.spec_from_file_location("sim2real._pix2pix_networks", _NETWORKS_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Pix2PixTranslator:
    """Single-image sim-to-real translation using the vendored original pix2pix (Isola et al.
    2017) UNet generator, trained per-task/camera via sim2real/train_pix2pix.py.

    Mirrors SimToRealTranslator's translate() interface (no text prompt, since this is a plain
    conditional GAN rather than a diffusion model) so it drops into the same call sites.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        resolution: int = 256,
        device: str | None = None,
    ) -> None:
        self.device = self._resolve_device(device)
        self.resolution = int(resolution)
        if self.resolution <= 0 or self.resolution % 256 != 0:
            raise ValueError(
                f"resolution must be a positive multiple of 256 for the unet_256 generator, "
                f"got {self.resolution}"
            )

        networks = _load_networks_module()
        # Matches sim2real/pix2pix/models/pix2pix_model.py's defaults (norm="batch",
        # netG="unet_256") and train_pix2pix.py's launch_training (no --no_dropout passed).
        self._net = networks.define_G(
            input_nc=3, output_nc=3, ngf=64, netG="unet_256", norm="batch", use_dropout=True
        )
        state_dict = torch.load(
            str(Path(checkpoint_path).expanduser()), map_location=self.device, weights_only=True
        )
        self._net.load_state_dict(state_dict)
        self._net.to(self.device)
        self._net.eval()

    @staticmethod
    def _resolve_device(device: str | None) -> torch.device:
        if device:
            return torch.device(device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def translate(self, image: np.ndarray) -> np.ndarray:
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected HxWx3 RGB image, got shape {image.shape}")

        image_uint8 = self._to_uint8(image)
        original_h, original_w = image_uint8.shape[:2]
        pil_image = Image.fromarray(image_uint8, mode="RGB")
        # Matches sim2real/pix2pix/data/base_dataset.py's get_transform for --preprocess resize:
        # Resize(BICUBIC) -> ToTensor -> Normalize(0.5, 0.5, 0.5), i.e. [0,255] -> [-1,1].
        model_image = pil_image.resize((self.resolution, self.resolution), Image.BICUBIC)
        conditioning = torch.from_numpy(np.array(model_image)).permute(2, 0, 1).unsqueeze(0)
        conditioning = conditioning.to(device=self.device, dtype=torch.float32) / 255.0
        conditioning = (conditioning - 0.5) / 0.5

        with torch.inference_mode():
            translated = self._net(conditioning)

        translated = translated[0].detach().float().cpu().permute(1, 2, 0).numpy()
        translated = np.clip((translated * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8)
        translated_pil = Image.fromarray(translated, mode="RGB").resize(
            (original_w, original_h), Image.BICUBIC
        )
        return np.asarray(translated_pil)

    @staticmethod
    def _to_uint8(image: np.ndarray) -> np.ndarray:
        if image.dtype == np.uint8:
            return np.ascontiguousarray(image)
        if np.issubdtype(image.dtype, np.floating):
            scaled = image
            if scaled.max() <= 1.0:
                scaled = scaled * 255.0
            return np.clip(scaled, 0, 255).astype(np.uint8)
        return np.clip(image, 0, 255).astype(np.uint8)
