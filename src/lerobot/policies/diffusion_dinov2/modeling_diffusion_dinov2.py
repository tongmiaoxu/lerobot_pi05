#!/usr/bin/env python

# Copyright 2024 Columbia Artificial Intelligence, Robotics Lab,
# and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Diffusion Policy with a DINOv2 image backbone.

Drop-in replacement for DiffusionPolicy where only the vision encoder
(DiffusionRgbEncoder → SpatialSoftmax) is swapped out for a frozen /
fine-tunable DINOv2 ViT + projection MLP.  The UNet, noise scheduler,
and all other components are unchanged.
"""

import math
from collections import deque
from collections.abc import Iterator

import einops
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn
from transformers import AutoModel

from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from lerobot.policies.diffusion_dinov2.configuration_diffusion_dinov2 import DiffusionDINOv2Config
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import (
    get_device_from_parameters,
    get_dtype_from_parameters,
    populate_queues,
)
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE


# ---------------------------------------------------------------------------
# Top-level policy class
# ---------------------------------------------------------------------------

class DiffusionDINOv2Policy(PreTrainedPolicy):
    """Diffusion Policy using a DINOv2 ViT backbone instead of ResNet+SpatialSoftmax.

    All UNet / noise-scheduler logic is identical to :class:`DiffusionPolicy`.
    Only the image encoder is different — see :class:`DINOv2RgbEncoder` below.
    """

    config_class = DiffusionDINOv2Config
    name = "diffusion_dinov2"

    def __init__(self, config: DiffusionDINOv2Config, **kwargs):
        super().__init__(config)
        config.validate_features()
        self.config = config

        self._queues = None
        self.diffusion = DiffusionDINOv2Model(config)
        self.reset()

    # ------------------------------------------------------------------
    # Optimizer parameter groups (support different LR for backbone)
    # ------------------------------------------------------------------

    def get_optim_params(self) -> Iterator[nn.Parameter]:
        """Return parameter groups for the optimizer.

        When ``freeze_backbone=False`` we expose two groups so the caller can
        apply ``backbone_lr_multiplier`` to the backbone parameters.  When the
        backbone is frozen the generator only yields trainable params.
        """
        if self.config.freeze_backbone:
            # Only projection head + UNet are trainable
            return (p for p in self.parameters() if p.requires_grad)

        # Two param groups: backbone (lower LR) and everything else
        backbone_ids = {
            id(p)
            for enc in _iter_encoders(self.diffusion)
            for p in enc.backbone.parameters()
        }

        backbone_params = [p for p in self.parameters() if id(p) in backbone_ids]
        other_params = [p for p in self.parameters() if id(p) not in backbone_ids and p.requires_grad]

        return [
            {"params": other_params},
            {
                "params": backbone_params,
                "lr_multiplier": self.config.backbone_lr_multiplier,
            },
        ]

    def reset(self):
        """Clear observation and action queues. Should be called on ``env.reset()``."""
        self._queues = {
            OBS_STATE: deque(maxlen=self.config.n_obs_steps),
            ACTION: deque(maxlen=self.config.n_action_steps),
        }
        if self.config.image_features:
            self._queues[OBS_IMAGES] = deque(maxlen=self.config.n_obs_steps)
        if self.config.env_state_feature:
            self._queues[OBS_ENV_STATE] = deque(maxlen=self.config.n_obs_steps)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        batch = {k: torch.stack(list(self._queues[k]), dim=1) for k in batch if k in self._queues}
        return self.diffusion.generate_actions(batch, noise=noise)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        if ACTION in batch:
            batch.pop(ACTION)

        if self.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = torch.stack([batch[key] for key in self.config.image_features], dim=-4)

        self._queues = populate_queues(self._queues, batch)

        if len(self._queues[ACTION]) == 0:
            actions = self.predict_action_chunk(batch, noise=noise)
            self._queues[ACTION].extend(actions.transpose(0, 1))

        return self._queues[ACTION].popleft()

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, float] | None]:
        if self.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = torch.stack([batch[key] for key in self.config.image_features], dim=-4)
        return self.diffusion.compute_loss(batch)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _iter_encoders(model: "DiffusionDINOv2Model"):
    """Yield all DINOv2RgbEncoder instances inside the model."""
    if isinstance(model.rgb_encoder, nn.ModuleList):
        yield from model.rgb_encoder
    elif hasattr(model, "rgb_encoder"):
        yield model.rgb_encoder


def _make_noise_scheduler(name: str, **kwargs) -> DDPMScheduler | DDIMScheduler:
    if name == "DDPM":
        return DDPMScheduler(**kwargs)
    elif name == "DDIM":
        return DDIMScheduler(**kwargs)
    raise ValueError(f"Unsupported noise scheduler type {name}")


# ---------------------------------------------------------------------------
# Core diffusion model (UNet + DINOv2 encoder)
# ---------------------------------------------------------------------------

class DiffusionDINOv2Model(nn.Module):
    def __init__(self, config: DiffusionDINOv2Config):
        super().__init__()
        self.config = config

        # Build observation encoders
        global_cond_dim = config.robot_state_feature.shape[0]
        if config.image_features:
            num_images = len(config.image_features)
            if config.use_separate_rgb_encoder_per_camera:
                encoders = [DINOv2RgbEncoder(config) for _ in range(num_images)]
                self.rgb_encoder = nn.ModuleList(encoders)
                global_cond_dim += config.dinov2_feature_dim * num_images
            else:
                self.rgb_encoder = DINOv2RgbEncoder(config)
                global_cond_dim += config.dinov2_feature_dim * num_images
        if config.env_state_feature:
            global_cond_dim += config.env_state_feature.shape[0]

        self.unet = DiffusionConditionalUnet1d(
            config, global_cond_dim=global_cond_dim * config.n_obs_steps
        )

        self.noise_scheduler = _make_noise_scheduler(
            config.noise_scheduler_type,
            num_train_timesteps=config.num_train_timesteps,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
            beta_schedule=config.beta_schedule,
            clip_sample=config.clip_sample,
            clip_sample_range=config.clip_sample_range,
            prediction_type=config.prediction_type,
        )
        self.num_inference_steps = (
            config.num_inference_steps
            if config.num_inference_steps is not None
            else self.noise_scheduler.config.num_train_timesteps
        )

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    def conditional_sample(
        self,
        batch_size: int,
        global_cond: Tensor | None = None,
        generator: torch.Generator | None = None,
        noise: Tensor | None = None,
    ) -> Tensor:
        device = get_device_from_parameters(self)
        dtype = get_dtype_from_parameters(self)

        sample = (
            noise
            if noise is not None
            else torch.randn(
                size=(batch_size, self.config.horizon, self.config.action_feature.shape[0]),
                dtype=dtype,
                device=device,
                generator=generator,
            )
        )

        self.noise_scheduler.set_timesteps(self.num_inference_steps)
        for t in self.noise_scheduler.timesteps:
            model_output = self.unet(
                sample,
                torch.full(sample.shape[:1], t, dtype=torch.long, device=sample.device),
                global_cond=global_cond,
            )
            sample = self.noise_scheduler.step(model_output, t, sample, generator=generator).prev_sample

        return sample

    def _prepare_global_conditioning(self, batch: dict[str, Tensor]) -> Tensor:
        """Encode images with DINOv2 and concatenate with robot state."""
        batch_size, n_obs_steps = batch[OBS_STATE].shape[:2]
        global_cond_feats = [batch[OBS_STATE]]

        if self.config.image_features:
            if self.config.use_separate_rgb_encoder_per_camera:
                images_per_camera = einops.rearrange(batch[OBS_IMAGES], "b s n ... -> n (b s) ...")
                img_features_list = torch.cat(
                    [
                        encoder(images)
                        for encoder, images in zip(self.rgb_encoder, images_per_camera, strict=True)
                    ]
                )
                img_features = einops.rearrange(
                    img_features_list, "(n b s) ... -> b s (n ...)", b=batch_size, s=n_obs_steps
                )
            else:
                img_features = self.rgb_encoder(
                    einops.rearrange(batch[OBS_IMAGES], "b s n ... -> (b s n) ...")
                )
                img_features = einops.rearrange(
                    img_features, "(b s n) ... -> b s (n ...)", b=batch_size, s=n_obs_steps
                )
            global_cond_feats.append(img_features)

        if self.config.env_state_feature:
            global_cond_feats.append(batch[OBS_ENV_STATE])

        return torch.cat(global_cond_feats, dim=-1).flatten(start_dim=1)

    def generate_actions(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        batch_size, n_obs_steps = batch[OBS_STATE].shape[:2]
        assert n_obs_steps == self.config.n_obs_steps

        global_cond = self._prepare_global_conditioning(batch)
        actions = self.conditional_sample(batch_size, global_cond=global_cond, noise=noise)

        start = n_obs_steps - 1
        end = start + self.config.n_action_steps
        return actions[:, start:end]

    def compute_loss(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, float]]:
        assert set(batch).issuperset({OBS_STATE, ACTION, "action_is_pad"})
        assert OBS_IMAGES in batch or OBS_ENV_STATE in batch
        assert batch[ACTION].shape[1] == self.config.horizon
        assert batch[OBS_STATE].shape[1] == self.config.n_obs_steps

        global_cond = self._prepare_global_conditioning(batch)
        trajectory = batch[ACTION]

        eps = torch.randn(trajectory.shape, device=trajectory.device)
        timesteps = torch.randint(
            low=0,
            high=self.noise_scheduler.config.num_train_timesteps,
            size=(trajectory.shape[0],),
            device=trajectory.device,
        ).long()
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, eps, timesteps)
        pred = self.unet(noisy_trajectory, timesteps, global_cond=global_cond)

        if self.config.prediction_type == "epsilon":
            target = eps
        elif self.config.prediction_type == "sample":
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {self.config.prediction_type}")

        loss = F.mse_loss(pred, target, reduction="none")

        if self.config.do_mask_loss_for_padding:
            if "action_is_pad" not in batch:
                raise ValueError("'action_is_pad' required when do_mask_loss_for_padding=True")
            loss = loss * (~batch["action_is_pad"]).unsqueeze(-1)

        train_loss = loss.mean()

        with torch.no_grad():
            if self.config.prediction_type == "epsilon":
                acp = self.noise_scheduler.alphas_cumprod[timesteps].to(
                    device=trajectory.device, dtype=trajectory.dtype
                ).view(-1, 1, 1)
                pred_traj = (noisy_trajectory - (1.0 - acp).sqrt() * pred) / acp.sqrt().clamp(min=1e-8)
            else:
                pred_traj = pred
            pad = batch["action_is_pad"].unsqueeze(-1)
            l1 = (F.l1_loss(trajectory, pred_traj, reduction="none") * ~pad).mean()

        return train_loss, {"l1_loss": l1.item()}


# ---------------------------------------------------------------------------
# DINOv2 image encoder  ← THE KEY DIFFERENCE vs. DiffusionRgbEncoder
# ---------------------------------------------------------------------------

class DINOv2RgbEncoder(nn.Module):
    """Encodes an RGB image using a (frozen/fine-tuned) DINOv2 ViT backbone.

    Pipeline
    --------
    Input image (B, 3, H, W) with pixels in [0, 1]
      → Normalize to ImageNet mean/std  (always applied)
      → Resize / pad to a multiple of the ViT patch size (14 px)
      → DINOv2 ViT forward pass  → last_hidden_state (B, 1+N_patches, hidden)
      → [CLS] token or mean-pool over patch tokens  → (B, hidden_size)
      → 2-layer projection MLP  → (B, dinov2_feature_dim)

    This is very different from the ResNet+SpatialSoftmax encoder used by the
    default diffusion policy:

    +---------------------+-------------------------------------+----------------------------------+
    | Property            | Default (ResNet + SpatialSoftmax)   | This (DINOv2 + projection MLP)   |
    +=====================+=====================================+==================================+
    | Backbone type       | Convolutional (ResNet-18/34/50)     | ViT (DINOv2-S/B/L/G)            |
    | Pretrained weights  | Optional (ImageNet torchvision)     | Always (self-supervised DINOv2)  |
    | Default frozen?     | No (fine-tuned by default)          | Yes (freeze_backbone=True)       |
    | Spatial pooling     | SpatialSoftmax → 2D keypoints       | [CLS] token or mean-pool         |
    | Input resolution    | Any (cropped / raw)                 | Resized to patch multiple        |
    | BatchNorm→GroupNorm | Optional swap for stability         | Not needed (ViT uses LayerNorm)  |
    | Feature dim         | spatial_softmax_num_keypoints * 2   | dinov2_feature_dim (configurable)|
    +---------------------+-------------------------------------+----------------------------------+
    """

    # ImageNet normalisation constants (DINOv2 was trained with these)
    _IMAGENET_MEAN = (0.485, 0.456, 0.406)
    _IMAGENET_STD = (0.229, 0.224, 0.225)
    _PATCH_SIZE = 14  # All DINOv2 ViTs use 14×14 patches

    def __init__(self, config: DiffusionDINOv2Config):
        super().__init__()
        self.config = config

        # ---- Load DINOv2 backbone from HuggingFace ----------------------
        self.backbone = AutoModel.from_pretrained(config.dinov2_model_name)
        hidden_size = self.backbone.config.hidden_size

        if config.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad_(False)

        # ---- Projection MLP: hidden_size → proj_hidden → feature_dim ---
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, config.dinov2_proj_hidden_dim),
            nn.GELU(),
            nn.Linear(config.dinov2_proj_hidden_dim, config.dinov2_feature_dim),
            nn.LayerNorm(config.dinov2_feature_dim),
        )
        self.feature_dim = config.dinov2_feature_dim

        # ---- ImageNet normalisation (register as buffer so device moves) -
        mean = torch.tensor(self._IMAGENET_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(self._IMAGENET_STD).view(1, 3, 1, 1)
        self.register_buffer("imagenet_mean", mean)
        self.register_buffer("imagenet_std", std)

    # ------------------------------------------------------------------

    def _pad_to_patch_multiple(self, x: Tensor) -> Tensor:
        """Pad H and W to the nearest multiple of the patch size (14)."""
        _, _, h, w = x.shape
        ph = (self._PATCH_SIZE - h % self._PATCH_SIZE) % self._PATCH_SIZE
        pw = (self._PATCH_SIZE - w % self._PATCH_SIZE) % self._PATCH_SIZE
        if ph > 0 or pw > 0:
            x = F.pad(x, (0, pw, 0, ph), mode="constant", value=0.0)
        return x

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, 3, H, W) image tensor with pixel values in [0, 1].

        Returns:
            (B, dinov2_feature_dim) image feature vector.
        """
        # 1. ImageNet normalisation  (in-place friendly, keep grad)
        x = (x - self.imagenet_mean) / self.imagenet_std

        # 2. Pad to patch-size multiple so ViT doesn't complain
        x = self._pad_to_patch_multiple(x)

        # 3. DINOv2 forward pass
        #    last_hidden_state: (B, 1 + N_patches, hidden_size)
        #    where token 0 is the [CLS] token.
        outputs = self.backbone(pixel_values=x)
        hidden = outputs.last_hidden_state  # (B, seq_len, hidden_size)

        if self.config.use_cls_token:
            # [CLS] token summarises the whole image
            feat = hidden[:, 0, :]  # (B, hidden_size)
        else:
            # Mean over all patch tokens (skip index 0 = CLS)
            feat = hidden[:, 1:, :].mean(dim=1)  # (B, hidden_size)

        # 4. Project to target feature dim
        feat = self.projection(feat)  # (B, dinov2_feature_dim)
        return feat


# ---------------------------------------------------------------------------
# UNet (identical to modeling_diffusion.py — copied to keep self-contained)
# ---------------------------------------------------------------------------

class DiffusionSinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x.unsqueeze(-1) * emb.unsqueeze(0)
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class DiffusionConv1dBlock(nn.Module):
    """Conv1d → GroupNorm → Mish"""

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class DiffusionConditionalResidualBlock1d(nn.Module):
    """ResNet-style 1D conv block with FiLM modulation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        kernel_size: int = 3,
        n_groups: int = 8,
        use_film_scale_modulation: bool = False,
    ):
        super().__init__()
        self.use_film_scale_modulation = use_film_scale_modulation
        self.out_channels = out_channels

        self.conv1 = DiffusionConv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups)
        cond_channels = out_channels * 2 if use_film_scale_modulation else out_channels
        self.cond_encoder = nn.Sequential(nn.Mish(), nn.Linear(cond_dim, cond_channels))
        self.conv2 = DiffusionConv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups)
        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        out = self.conv1(x)
        cond_embed = self.cond_encoder(cond).unsqueeze(-1)
        if self.use_film_scale_modulation:
            scale = cond_embed[:, : self.out_channels]
            bias = cond_embed[:, self.out_channels :]
            out = scale * out + bias
        else:
            out = out + cond_embed
        out = self.conv2(out)
        return out + self.residual_conv(x)


class DiffusionConditionalUnet1d(nn.Module):
    """1-D conditional UNet identical to the one in modeling_diffusion.py."""

    def __init__(self, config: DiffusionDINOv2Config, global_cond_dim: int):
        super().__init__()
        self.config = config

        self.diffusion_step_encoder = nn.Sequential(
            DiffusionSinusoidalPosEmb(config.diffusion_step_embed_dim),
            nn.Linear(config.diffusion_step_embed_dim, config.diffusion_step_embed_dim * 4),
            nn.Mish(),
            nn.Linear(config.diffusion_step_embed_dim * 4, config.diffusion_step_embed_dim),
        )

        cond_dim = config.diffusion_step_embed_dim + global_cond_dim
        in_out = [(config.action_feature.shape[0], config.down_dims[0])] + list(
            zip(config.down_dims[:-1], config.down_dims[1:], strict=True)
        )

        common_kwargs = {
            "cond_dim": cond_dim,
            "kernel_size": config.kernel_size,
            "n_groups": config.n_groups,
            "use_film_scale_modulation": config.use_film_scale_modulation,
        }

        self.down_modules = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        DiffusionConditionalResidualBlock1d(dim_in, dim_out, **common_kwargs),
                        DiffusionConditionalResidualBlock1d(dim_out, dim_out, **common_kwargs),
                        nn.Conv1d(dim_out, dim_out, 3, 2, 1) if not is_last else nn.Identity(),
                    ]
                )
                for ind, (dim_in, dim_out) in enumerate(in_out)
                for is_last in [ind >= len(in_out) - 1]
            ]
        )

        self.mid_modules = nn.ModuleList(
            [
                DiffusionConditionalResidualBlock1d(config.down_dims[-1], config.down_dims[-1], **common_kwargs),
                DiffusionConditionalResidualBlock1d(config.down_dims[-1], config.down_dims[-1], **common_kwargs),
            ]
        )

        self.up_modules = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        DiffusionConditionalResidualBlock1d(dim_in * 2, dim_out, **common_kwargs),
                        DiffusionConditionalResidualBlock1d(dim_out, dim_out, **common_kwargs),
                        nn.ConvTranspose1d(dim_out, dim_out, 4, 2, 1) if not is_last else nn.Identity(),
                    ]
                )
                for ind, (dim_out, dim_in) in enumerate(reversed(in_out[1:]))
                for is_last in [ind >= len(in_out) - 1]
            ]
        )

        self.final_conv = nn.Sequential(
            DiffusionConv1dBlock(config.down_dims[0], config.down_dims[0], kernel_size=config.kernel_size),
            nn.Conv1d(config.down_dims[0], config.action_feature.shape[0], 1),
        )

    def forward(self, x: Tensor, timestep: Tensor | int, global_cond=None) -> Tensor:
        x = einops.rearrange(x, "b t d -> b d t")
        timesteps_embed = self.diffusion_step_encoder(timestep)
        global_feature = (
            torch.cat([timesteps_embed, global_cond], axis=-1)
            if global_cond is not None
            else timesteps_embed
        )

        encoder_skip_features: list[Tensor] = []
        for resnet, resnet2, downsample in self.down_modules:
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            encoder_skip_features.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        for resnet, resnet2, upsample in self.up_modules:
            x = torch.cat((x, encoder_skip_features.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)
        return einops.rearrange(x, "b d t -> b t d")
