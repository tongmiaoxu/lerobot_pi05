from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
import torch
from diffusers.utils.import_utils import is_xformers_available

from .img2img_turbo.pix2pix_turbo import Pix2Pix_Turbo


class SimToRealTranslator:
    """Single-image sim-to-real translation using vendored pix2pix-turbo (Pix2Pix_Turbo)."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        prompt: str = "a real-world robot camera image",
        resolution: int = 512,
        device: str | None = None,
        use_fp16: bool | None = None,
        enable_xformers: bool = True,
    ) -> None:
        self.device = self._resolve_device(device)
        self.use_fp16 = self.device.type == "cuda" if use_fp16 is None else use_fp16
        self.resolution = int(resolution)
        if self.resolution <= 0 or self.resolution % 8 != 0:
            raise ValueError(f"resolution must be a positive multiple of 8, got {self.resolution}")

        self._net = Pix2Pix_Turbo(pretrained_path=str(Path(checkpoint_path).expanduser()), device=self.device)
        self._net.set_eval()
        if self.use_fp16 and self.device.type == "cuda":
            self._net.half()
        if self.device.type == "cuda" and enable_xformers and is_xformers_available():
            self._net.unet.enable_xformers_memory_efficient_attention()

        self._dtype = torch.float16 if self.use_fp16 and self.device.type == "cuda" else torch.float32
        self._prompt_tokens = self._net.tokenizer(
            prompt,
            max_length=self._net.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(self.device)

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
        model_image = pil_image.resize((self.resolution, self.resolution), Image.LANCZOS)
        conditioning = torch.from_numpy(np.asarray(model_image)).permute(2, 0, 1).unsqueeze(0)
        conditioning = conditioning.to(device=self.device, dtype=self._dtype) / 255.0

        with torch.inference_mode():
            translated = self._net(conditioning, prompt_tokens=self._prompt_tokens, deterministic=True)

        translated = translated[0].detach().float().cpu().permute(1, 2, 0).numpy()
        translated = np.clip((translated * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8)
        translated_pil = Image.fromarray(translated, mode="RGB").resize((original_w, original_h), Image.LANCZOS)
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
