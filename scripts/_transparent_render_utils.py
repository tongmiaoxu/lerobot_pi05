"""Shared helper for rendering MuJoCo scenes with a transparent background.

Uses a segmentation pass (mjRND_SEGMENT) to build a per-pixel alpha mask: any
pixel not covered by a visible geom (including the skybox/haze) becomes
alpha=0, so hidden bodies/geoms and the background both disappear cleanly.
"""

from __future__ import annotations

import mujoco
import numpy as np


def render_rgba(
    renderer: mujoco.Renderer,
    data: mujoco.MjData,
    camera,
    scene_option: mujoco.MjvOption,
) -> np.ndarray:
    renderer.disable_segmentation_rendering()
    renderer.update_scene(data, camera=camera, scene_option=scene_option)
    rgb = renderer.render()

    renderer.enable_segmentation_rendering()
    seg = renderer.render()
    renderer.disable_segmentation_rendering()

    alpha = np.where(seg[:, :, 1] != -1, 255, 0).astype(np.uint8)
    return np.dstack([rgb, alpha])
