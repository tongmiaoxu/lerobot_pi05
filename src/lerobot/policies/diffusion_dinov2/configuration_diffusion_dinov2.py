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
"""Configuration for Diffusion Policy with DINOv2 image backbone.

This is a self-contained variant of DiffusionConfig that replaces the ResNet + SpatialSoftmax
image encoder with a frozen (or fine-tunable) DINOv2 ViT backbone followed by a lightweight
projection MLP.  All other components (UNet, noise scheduler, etc.) are identical to the
original diffusion policy.
"""

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.optim.optimizers import AdamConfig
from lerobot.optim.schedulers import DiffuserSchedulerConfig


@PreTrainedConfig.register_subclass("diffusion_dinov2")
@dataclass
class DiffusionDINOv2Config(PreTrainedConfig):
    """Configuration class for DiffusionPolicy with a DINOv2 vision backbone.

    Identical to DiffusionConfig in all UNet / diffusion parameters.
    The key difference is the image encoder:

    - **Current diffusion policy**: ResNet (e.g. resnet18) feature extractor →
      SpatialSoftmax pooling → small linear layer → feature vector.
    - **This DINOv2 variant**: Frozen / fine-tuned DINOv2 ViT backbone →
      [CLS] token (or mean-pooled patch tokens) → projection MLP → feature vector.

    DINOv2 model choices (``dinov2_model_name``):
        - ``"facebook/dinov2-small"``  → hidden_size = 384
        - ``"facebook/dinov2-base"``   → hidden_size = 768
        - ``"facebook/dinov2-large"``  → hidden_size = 1024
        - ``"facebook/dinov2-giant"``  → hidden_size = 1536

    Args:
        n_obs_steps: Number of environment steps worth of observations to pass to the policy.
        horizon: Diffusion model action prediction size.
        n_action_steps: Number of action steps to run in the environment per policy invocation.
        dinov2_model_name: HuggingFace model name for the DINOv2 backbone.
        freeze_backbone: If True, DINOv2 weights are frozen during training. Only the
            projection head (and UNet) are trained.  If False, the backbone is fine-tuned
            end-to-end (requires more GPU memory and a lower LR for the backbone).
        use_cls_token: If True, use the [CLS] token embedding as the image feature.
            If False, take the mean over all patch tokens instead (usually similar quality).
        dinov2_proj_hidden_dim: Hidden dimension of the 2-layer MLP that projects DINOv2
            features down to ``dinov2_feature_dim``.
        dinov2_feature_dim: Output dimension of the DINOv2 projection MLP.
            This is the per-camera feature size that gets concatenated with the robot state
            before being fed into the UNet.
        use_separate_rgb_encoder_per_camera: Whether to use a separate DINOv2 encoder
            (projection head only — the backbone weights are always shared) per camera.
        down_dims: Feature dimension for each stage of temporal downsampling in the UNet.
        kernel_size: Convolutional kernel size in the UNet.
        n_groups: Number of groups for GroupNorm in the UNet.
        diffusion_step_embed_dim: Embedding dimension of the diffusion timestep encoder.
        use_film_scale_modulation: Whether to use scale + bias FiLM modulation (vs bias-only).
        noise_scheduler_type: ``"DDPM"`` or ``"DDIM"``.
        num_train_timesteps: Forward diffusion steps.
        beta_schedule: Name of the beta schedule.
        beta_start: Beta at the first diffusion step.
        beta_end: Beta at the last diffusion step.
        prediction_type: ``"epsilon"`` or ``"sample"``.
        clip_sample: Whether to clip the denoised sample at inference.
        clip_sample_range: Clipping magnitude.
        num_inference_steps: Reverse diffusion steps at inference. Defaults to num_train_timesteps.
        do_mask_loss_for_padding: Whether to mask padded actions in the loss.
        image_keys_filter: Optional list of substrings to filter camera keys.
        optimizer_lr: Learning rate for the UNet + projection head (and backbone if unfrozen).
        backbone_lr_multiplier: LR multiplier applied to backbone parameters when
            ``freeze_backbone=False``. Typically much smaller (e.g. 0.1) to avoid destroying
            the pretrained representations.
    """

    # ------------------------------------------------------------------ #
    # Inputs / output structure
    # ------------------------------------------------------------------ #
    n_obs_steps: int = 2
    horizon: int = 16
    n_action_steps: int = 8

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
        }
    )

    drop_n_last_frames: int = 7  # horizon - n_action_steps - n_obs_steps + 1

    # ------------------------------------------------------------------ #
    # DINOv2 image encoder
    # ------------------------------------------------------------------ #
    dinov2_model_name: str = "facebook/dinov2-base"
    freeze_backbone: bool = True
    use_cls_token: bool = True
    dinov2_proj_hidden_dim: int = 512
    dinov2_feature_dim: int = 256
    use_separate_rgb_encoder_per_camera: bool = False

    # ------------------------------------------------------------------ #
    # UNet
    # ------------------------------------------------------------------ #
    down_dims: tuple[int, ...] = (512, 1024, 2048)
    kernel_size: int = 5
    n_groups: int = 8
    diffusion_step_embed_dim: int = 128
    use_film_scale_modulation: bool = True

    # ------------------------------------------------------------------ #
    # Noise scheduler
    # ------------------------------------------------------------------ #
    noise_scheduler_type: str = "DDPM"
    num_train_timesteps: int = 100
    beta_schedule: str = "squaredcos_cap_v2"
    beta_start: float = 0.0001
    beta_end: float = 0.02
    prediction_type: str = "epsilon"
    clip_sample: bool = True
    clip_sample_range: float = 1.0

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #
    num_inference_steps: int | None = None

    # ------------------------------------------------------------------ #
    # Loss
    # ------------------------------------------------------------------ #
    do_mask_loss_for_padding: bool = False

    # ------------------------------------------------------------------ #
    # Camera filtering
    # ------------------------------------------------------------------ #
    image_keys_filter: list[str] | None = None

    # ------------------------------------------------------------------ #
    # Training presets
    # ------------------------------------------------------------------ #
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple = (0.95, 0.999)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-6
    scheduler_name: str = "cosine"
    scheduler_warmup_steps: int = 500
    # Multiplier applied to backbone LR when freeze_backbone=False.
    backbone_lr_multiplier: float = 0.1

    # ------------------------------------------------------------------ #
    # Post-init validation
    # ------------------------------------------------------------------ #
    def __post_init__(self):
        super().__post_init__()

        supported_prediction_types = ["epsilon", "sample"]
        if self.prediction_type not in supported_prediction_types:
            raise ValueError(
                f"`prediction_type` must be one of {supported_prediction_types}. "
                f"Got {self.prediction_type}."
            )

        supported_noise_schedulers = ["DDPM", "DDIM"]
        if self.noise_scheduler_type not in supported_noise_schedulers:
            raise ValueError(
                f"`noise_scheduler_type` must be one of {supported_noise_schedulers}. "
                f"Got {self.noise_scheduler_type}."
            )

        downsampling_factor = 2 ** len(self.down_dims)
        if self.horizon % downsampling_factor != 0:
            raise ValueError(
                "The horizon should be an integer multiple of the downsampling factor "
                f"(determined by `len(down_dims)`). Got {self.horizon=} and {self.down_dims=}"
            )

    # ------------------------------------------------------------------ #
    # Optimizer / scheduler presets
    # ------------------------------------------------------------------ #
    def get_optimizer_preset(self) -> AdamConfig:
        return AdamConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self) -> DiffuserSchedulerConfig:
        return DiffuserSchedulerConfig(
            name=self.scheduler_name,
            num_warmup_steps=self.scheduler_warmup_steps,
        )

    # ------------------------------------------------------------------ #
    # Feature validation
    # ------------------------------------------------------------------ #
    def validate_features(self) -> None:
        if len(self.image_features) == 0 and self.env_state_feature is None:
            raise ValueError("You must provide at least one image or the environment state among the inputs.")

    # ------------------------------------------------------------------ #
    # Delta index helpers (required by lerobot dataset handling)
    # ------------------------------------------------------------------ #
    @property
    def observation_delta_indices(self) -> list:
        return list(range(1 - self.n_obs_steps, 1))

    @property
    def action_delta_indices(self) -> list:
        return list(range(1 - self.n_obs_steps, 1 - self.n_obs_steps + self.horizon))

    @property
    def reward_delta_indices(self) -> None:
        return None
