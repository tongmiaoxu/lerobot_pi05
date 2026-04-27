from __future__ import annotations

import base64
import io
import os
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

DEFAULT_CAMERA_PROMPTS = {
    "stationary": (
        "You are converting a synthetic stationary robot-camera render into a real camera photo. "
        "Preserve the exact camera viewpoint, crop, robot pose, object pose, table geometry, and scene layout "
        "from image 1. The remaining images are reference pairs: each synthetic render is immediately followed "
        "by its matching real camera photo. Learn the visual mapping from those pairs and apply it only as a "
        "style and appearance transfer to image 1. Keep all geometry unchanged. Do not add, remove, move, or "
        "hallucinate objects. Keep the xArm robot, mug, and scene consistent with image 1."
    ),
    "wrist": (
        "You are converting a synthetic wrist-camera robot render into a real wrist-camera photo. "
        "Preserve the exact top-down camera viewpoint, crop, robot pose, object pose, table geometry, and scene "
        "layout from image 1. The remaining images are reference pairs: each synthetic render is immediately "
        "followed by its matching real camera photo. Learn the visual mapping from those pairs and apply it only "
        "as a style and appearance transfer to image 1. Keep all geometry unchanged. Do not add, remove, move, "
        "or hallucinate objects."
    ),
}

DEFAULT_STYLE_REFERENCE_PROMPT = (
    "Transfer the visual style and appearance from image 2 onto image 1. "
    "Preserve the exact geometry, camera viewpoint, crop, robot pose, object pose, and scene layout from image 1. "
    "Use image 2 only as a style reference. Do not add, remove, move, or hallucinate objects."
)

DEFAULT_EXAMPLE_FRAME_INDICES = {
    "stationary": [0, 2, 4],
    "wrist": [1, 300, 500],
}


def load_calibration_pairs_pil(
    cam_name: str,
    n_pairs: int,
    base_dir: str | Path,
    *,
    example_frame_indices: dict[str, list[int]] | None = None,
) -> list[tuple[Image.Image, Image.Image]]:
    """Load render/real reference pairs using the repo's calibration folder layout."""
    base = Path(base_dir)
    gs_dir = base / "gs_renders"
    real_dir = base / "real_captures"
    indices_map = example_frame_indices or DEFAULT_EXAMPLE_FRAME_INDICES
    indices = indices_map.get(cam_name, list(range(n_pairs)))[:n_pairs]
    pairs: list[tuple[Image.Image, Image.Image]] = []
    candidate_paths = [(gs_dir / f"frame_{idx:04d}.png", real_dir / f"frame_{idx:04d}.png") for idx in indices]
    if not all(gs.is_file() and real.is_file() for gs, real in candidate_paths):
        common_stems = sorted(
            {
                path.stem
                for path in gs_dir.glob("*.png")
                if (real_dir / f"{path.stem}.png").is_file()
            }
        )
        candidate_paths = [
            (gs_dir / f"{stem}.png", real_dir / f"{stem}.png")
            for stem in common_stems[:n_pairs]
        ]
    if len(candidate_paths) < n_pairs:
        raise FileNotFoundError(f"Not enough matching calibration pairs under {base}")
    for gs, real in candidate_paths:
        if not gs.is_file():
            raise FileNotFoundError(f"Missing render reference: {gs}")
        if not real.is_file():
            raise FileNotFoundError(f"Missing real reference: {real}")
        pairs.append((Image.open(gs).convert("RGB"), Image.open(real).convert("RGB")))
    return pairs


class GPTImageTranslator:
    """Sim-to-real translation through the OpenAI Images edit API."""

    def __init__(
        self,
        *,
        reference_pairs: Sequence[tuple[Image.Image, Image.Image]] = (),
        style_references: Sequence[Image.Image] = (),
        cam_name: str = "stationary",
        prompt: str | None = None,
        model: str = "gpt-image-2",
        api_key: str | None = None,
        size: str = "auto",
        quality: str = "high",
        output_format: str = "png",
        max_side: int | None = None,
    ) -> None:
        self.cam_name = cam_name
        self.model = model
        self.size = size
        self.quality = quality
        self.output_format = output_format
        self.max_side = max_side
        self.prompt = prompt or (
            DEFAULT_STYLE_REFERENCE_PROMPT
            if style_references and not reference_pairs
            else DEFAULT_CAMERA_PROMPTS.get(cam_name, DEFAULT_CAMERA_PROMPTS["stationary"])
        )
        self._reference_pairs = [(src.copy(), tgt.copy()) for src, tgt in reference_pairs]
        self._style_references = [style.copy() for style in style_references]
        self._client = self._init_client(api_key)

    @staticmethod
    def _init_client(api_key: str | None):
        try:
            from dotenv import load_dotenv

            load_dotenv()
        except ImportError:
            pass

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "GPTImageTranslator requires the OpenAI Python client. Install it with `pip install openai`."
            ) from exc

        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY or pass api_key.")
        return OpenAI(api_key=key)

    def translate(self, image: np.ndarray) -> np.ndarray:
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected HxWx3 RGB image, got shape {image.shape}")

        image_uint8 = self._to_uint8(image)
        original_h, original_w = image_uint8.shape[:2]
        query = Image.fromarray(image_uint8, mode="RGB")
        query_for_api = self._resize_for_api(query)
        image_payloads = [self._pil_to_file(query_for_api, "query.png")]

        for idx, (render_ref, real_ref) in enumerate(self._reference_pairs, start=1):
            image_payloads.append(
                self._pil_to_file(self._resize_for_api(render_ref), f"example_{idx:02d}_render.png")
            )
            image_payloads.append(
                self._pil_to_file(self._resize_for_api(real_ref), f"example_{idx:02d}_real.png")
            )
        for idx, style_ref in enumerate(self._style_references, start=1):
            image_payloads.append(
                self._pil_to_file(self._resize_for_api(style_ref), f"style_reference_{idx:02d}.png")
            )

        response = self._client.images.edit(
            model=self.model,
            image=image_payloads,
            prompt=self._build_prompt(),
            size=self.size,
            quality=self.quality,
            output_format=self.output_format,
        )
        encoded = response.data[0].b64_json
        if not encoded:
            raise RuntimeError("OpenAI image edit returned no image payload.")

        translated_bytes = base64.b64decode(encoded)
        translated_pil = Image.open(io.BytesIO(translated_bytes)).convert("RGB")
        translated_pil = translated_pil.resize((original_w, original_h), Image.LANCZOS)
        return np.asarray(translated_pil)

    def _build_prompt(self) -> str:
        prompt_sections = [self.prompt]
        if self._reference_pairs:
            pair_lines = [
                f"Reference pair {idx}: image {2 * idx} is the synthetic render and image {2 * idx + 1} is the matching real photo."
                for idx in range(1, len(self._reference_pairs) + 1)
            ]
            prompt_sections.append("\n".join(pair_lines))
        if self._style_references:
            start_idx = 2 * len(self._reference_pairs) + 2
            style_lines = [
                f"Style reference {idx}: image {image_idx} is a real photo whose appearance should guide the output while preserving image 1."
                for idx, image_idx in enumerate(range(start_idx, start_idx + len(self._style_references)), start=1)
            ]
            prompt_sections.append("\n".join(style_lines))
        return "\n\n".join(prompt_sections)

    def _resize_for_api(self, image: Image.Image) -> Image.Image:
        if self.max_side is None:
            return image
        w, h = image.size
        longest = max(w, h)
        if longest <= self.max_side:
            return image
        scale = self.max_side / float(longest)
        new_size = (max(1, round(w * scale)), max(1, round(h * scale)))
        return image.resize(new_size, Image.LANCZOS)

    @staticmethod
    def _pil_to_file(image: Image.Image, name: str) -> io.BytesIO:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)
        buffer.name = name
        return buffer

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
