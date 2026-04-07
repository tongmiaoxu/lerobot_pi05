from lerobot.policies.diffusion_dinov2.configuration_diffusion_dinov2 import DiffusionDINOv2Config

__all__ = ["DiffusionDINOv2Config"]

# DiffusionDINOv2Policy is intentionally NOT imported here to avoid a hard
# dependency on `transformers` at module load time.  Import it explicitly
# when needed:
#   from lerobot.policies.diffusion_dinov2.modeling_diffusion_dinov2 import DiffusionDINOv2Policy
