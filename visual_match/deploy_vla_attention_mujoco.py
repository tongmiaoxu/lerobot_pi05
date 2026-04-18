#!/usr/bin/env python3
"""
Deploy a VLA policy (pi05 or Groot) in MuJoCo and visualize per-step attention.

This is a minimal version of `deploy_act_policy_mujoco.py` focused on one
question: *where is the policy looking when it produces this action?*

For each control step we:
  1. Build the same (composite) observation the policy would see.
  2. Resize to the policy input resolution (e.g. 224x224 for pi05).
  3. Run inference *inside* an AttentionCapture context so every attention
     layer is recorded.
  4. Slice action-query -> image-patch attention, build a heatmap per camera,
     overlay it on the image, and display + save an MP4 per episode.

Usage examples
--------------

Pi05:
    python visual_match/deploy_vla_attention_mujoco.py \\
        --policy-path outputs/pi05_place_mug/checkpoints/100000/pretrained_model \\
        --mode rollout --max-steps 300

Groot (experimental — see `--groot-vision-*` args to tell the script which
slice of the VL sequence is actually image tokens):
    python visual_match/deploy_vla_attention_mujoco.py \\
        --policy-path outputs/groot_place_mug/checkpoints/last/pretrained_model \\
        --mode mean_layers --groot-vision-start 0 --groot-vision-tokens 1024 \\
        --groot-vision-grid 32 32

Pass --no-display to run headless (just saves MP4s).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Add src to path for lerobot imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))


def _detect_display():
    if os.environ.get("DISPLAY"):
        return True
    if os.environ.get("WAYLAND_DISPLAY"):
        return True
    if sys.platform in ("darwin", "win32"):
        return True
    return False


_HAS_DISPLAY = _detect_display()
if "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "glx" if _HAS_DISPLAY else "egl"

import numpy as np  # noqa: E402
import torch  # noqa: E402
import cv2  # noqa: E402
import mujoco  # noqa: E402
from mujoco import MjModel, MjData  # noqa: E402

from lerobot.policies.factory import get_policy_class, make_pre_post_processors  # noqa: E402
from lerobot.utils.control_utils import predict_action  # noqa: E402
from lerobot.utils.utils import get_safe_torch_device  # noqa: E402
from lerobot.utils.constants import OBS_STATE  # noqa: E402
from lerobot.datasets.utils import copy_observation_frame_with_resized_images  # noqa: E402
from lerobot.tasks import get_task_profile, resolve_task_scene_xml  # noqa: E402

from camera_config import load_camera_config, set_mujoco_camera_from_config  # noqa: E402
from composite_rendering import (  # noqa: E402
    get_mujoco_camera_pose,
    get_robot_geom_ids,
    load_scene_data,
    mj_pose_to_gaussian_w2c,
    render,
    shift_for_principal_point,
    T_splat2mj,
)
from lerobot_mujoco_utils import (  # noqa: E402
    lerobot_state_to_mujoco_ctrl,
    mujoco_qpos_to_lerobot_state,
)
from deploy_act_policy_mujoco import (  # noqa: E402
    CAMERA_CONFIG,
    apply_color_transform,
    load_color_mapping,
)
from attention_viz import (  # noqa: E402
    GrootAttentionCapture,
    PI05AttentionCapture,
    attention_to_heatmap,
    groot_attention_for_vision,
    make_attention_dashboard,
    overlay_heatmap,
    render_pi05_overlays,
)


# ============================================================================
# Policy loading
# ============================================================================


def load_policy(policy_path: str):
    """Load any lerobot policy (pi05, groot, ...) from a checkpoint directory."""
    policy_path_obj = Path(policy_path)
    if not policy_path_obj.is_absolute() and not policy_path_obj.exists():
        project_root = Path(__file__).parent.parent
        policy_path_obj = project_root / policy_path
    if not policy_path_obj.exists():
        raise FileNotFoundError(f"Policy path not found: {policy_path}")

    policy_path = str(policy_path_obj.resolve())
    config_path = Path(policy_path) / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        config_dict = json.load(f)
    policy_type = config_dict.get("type", "pi05")
    print(f"[INFO] Loading {policy_type} policy from: {policy_path}")

    policy_class = get_policy_class(policy_type)
    policy = policy_class.from_pretrained(policy_path)
    policy.eval()
    return policy, policy_type, policy_path


def build_capture(policy, policy_type: str, keep_denoise: str = "last"):
    """Return an AttentionCapture instance appropriate for the policy type."""
    if policy_type == "pi05":
        return PI05AttentionCapture(policy, keep_denoise=keep_denoise)
    if policy_type == "groot":
        return GrootAttentionCapture(policy, keep_denoise=keep_denoise)
    raise ValueError(
        f"Attention capture not implemented for policy type {policy_type!r}. "
        "Supported: pi05, groot."
    )


# ============================================================================
# Observation building (composite: GS background + MuJoCo robot foreground)
# ============================================================================


def render_composite_view(
    model: MjModel,
    data: MjData,
    renderer: mujoco.Renderer,
    seg_renderer: mujoco.Renderer,
    robot_geom_ids: set,
    cam_key: str,
    mujoco_cam: str,
    gaussian_data: dict | None,
    intrinsics: np.ndarray | None,
) -> np.ndarray:
    renderer.update_scene(data, camera=mujoco_cam)
    fg_rgb = renderer.render()
    if gaussian_data is None or gaussian_data.get("scene_data") is None:
        return fg_rgb

    seg_renderer.update_scene(data, camera=mujoco_cam)
    seg_labels = seg_renderer.render()[:, :, 0].astype(np.int32)

    if intrinsics is not None:
        fg_rgb = shift_for_principal_point(fg_rgb, intrinsics)
        seg_labels = shift_for_principal_point(seg_labels, intrinsics, seg=True)

    robot_mask = np.isin(seg_labels, list(robot_geom_ids))
    mask_uint8 = (robot_mask.astype(np.uint8)) * 255

    try:
        camera_pose = get_mujoco_camera_pose(model, data, mujoco_cam)
        w2c = mj_pose_to_gaussian_w2c(camera_pose, T_splat2mj)
        viz_cfg = gaussian_data["viz_cfg"]
        bg_im = render(
            w2c, intrinsics, gaussian_data["scene_data"],
            gaussian_data["scene_depth_data"], viz_cfg,
        )[0]
        bg_np = bg_im.permute(1, 2, 0).cpu().numpy()
        bg_np = (bg_np * 255).astype(np.uint8)
        composite = bg_np.copy()
        composite[mask_uint8 > 0] = fg_rgb[mask_uint8 > 0]
        color_calib = gaussian_data.get("color_calib_by_camera", {}).get(cam_key)
        if color_calib is not None:
            composite = apply_color_transform(composite, color_calib)
        return composite
    except Exception as e:
        print(f"[WARN] Composite rendering failed for {mujoco_cam}: {e}")
        return fg_rgb


def build_observation(
    model: MjModel,
    data: MjData,
    renderer: mujoco.Renderer,
    seg_renderer: mujoco.Renderer,
    robot_geom_ids: set,
    gaussian_data: dict | None,
) -> dict:
    ld_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "left_driver_joint")
    g_adr = int(model.jnt_qposadr[ld_id])
    g_rad = (float(model.jnt_range[ld_id, 0]), float(model.jnt_range[ld_id, 1]))
    state = mujoco_qpos_to_lerobot_state(data.qpos, g_rad, gripper_qpos_adr=g_adr)

    obs = {OBS_STATE: state}
    camera_intrinsics = gaussian_data.get("camera_intrinsics", {}) if gaussian_data else {}
    for cam_key, cam_cfg in CAMERA_CONFIG.items():
        obs_key = f"observation.images.{cam_cfg['dataset_cam']}"
        obs[obs_key] = render_composite_view(
            model, data, renderer, seg_renderer, robot_geom_ids,
            cam_key, cam_cfg["mujoco_cam"],
            gaussian_data, camera_intrinsics.get(cam_key),
        )
    return obs


# ============================================================================
# Groot span resolution helpers
# ============================================================================


def _parse_cli_cam_ranges(cli_ranges: list[str] | None) -> dict[str, tuple[int, int]]:
    """Parse "cam_high:44-300 cam_wrist:300-556" -> {"cam_high": (44,300), ...}."""
    out: dict[str, tuple[int, int]] = {}
    if not cli_ranges:
        return out
    for token in cli_ranges:
        if ":" not in token or "-" not in token:
            raise ValueError(f"Bad --groot-cam-token-ranges entry {token!r}. "
                             f"Expected CAM:START-END, e.g. cam_high:44-300")
        name, span = token.split(":", 1)
        start_s, end_s = span.split("-", 1)
        out[name.strip()] = (int(start_s), int(end_s))
    return out


def _match_cam_range(
    cam_key: str, parsed: dict[str, tuple[int, int]]
) -> tuple[int, int] | None:
    """Match a full cam_key like 'observation.images.cam_high' to a user-provided
    short name like 'cam_high' or 'stationary'."""
    if cam_key in parsed:
        return parsed[cam_key]
    suffix = cam_key.rsplit(".", 1)[-1]  # "cam_high"
    if suffix in parsed:
        return parsed[suffix]
    for name, rng in parsed.items():
        if name in cam_key:
            return rng
    return None


def _resolve_groot_spans(
    cam_keys: list[str],
    cli_ranges: list[str] | None,
    cli_single_start: int | None,
    cli_single_tokens: int | None,
    cli_grid: list[int] | None,
    auto_spans: list[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    """Decide the (start, end) key range in vl_embs for each camera.

    Priority:
      1) `--groot-cam-token-ranges cam_high:44-300 ...` (explicit per cam)
      2) Auto-detected spans from Eagle backbone input_ids (one per image)
      3) `--groot-vision-start` + `--groot-vision-tokens`/`--groot-vision-grid`
         as a single span replicated across every camera (legacy behavior)

    Returns a list of (start, end) spans in the same order as cam_keys, or None
    if we can't resolve anything yet (e.g. no inference has run).
    """
    # 1) explicit CLI
    parsed = _parse_cli_cam_ranges(cli_ranges)
    if parsed:
        out: list[tuple[int, int]] = []
        for ck in cam_keys:
            rng = _match_cam_range(ck, parsed)
            if rng is None:
                raise ValueError(
                    f"--groot-cam-token-ranges is missing camera {ck!r}. "
                    f"Got entries for {list(parsed)}."
                )
            out.append(rng)
        return out

    # 2) auto-detected from input_ids
    if auto_spans:
        if len(auto_spans) >= len(cam_keys):
            return list(auto_spans[: len(cam_keys)])
        # Fewer spans than cameras — probably only one image was passed.
        # Replicate the last known span.
        return [auto_spans[-1]] * len(cam_keys)

    # 3) legacy single-span CLI fallback
    if cli_single_start is not None:
        if cli_grid is not None:
            gh, gw = cli_grid
            num = gh * gw
        elif cli_single_tokens is not None:
            num = cli_single_tokens
        else:
            num = 256  # Eagle2 default
        return [(cli_single_start, cli_single_start + num)] * len(cam_keys)

    return None


def _grid_for_span(num_tokens: int, default_grid: tuple[int, int]) -> tuple[int, int]:
    """Pick a grid shape matching num_tokens. Prefer the user-provided default,
    else fall back to a square grid."""
    gh, gw = default_grid
    if gh * gw == num_tokens:
        return gh, gw
    from attention_viz import infer_square_grid
    try:
        return infer_square_grid(num_tokens)
    except ValueError:
        return gh, gw


def _resolve_blackout_obs_keys(blackout_cams: list[str] | None) -> set[str]:
    """Resolve user camera names to observation image keys.

    Accepts names like `wrist`, `stationary`, `cam_wrist`, `cam_high`, or
    full keys such as `observation.images.cam_wrist`.
    """
    if not blackout_cams:
        return set()

    out: set[str] = set()
    for cam in blackout_cams:
        cam = cam.strip()
        if not cam:
            continue
        if cam.startswith("observation.images."):
            out.add(cam)
            continue

        matched = False
        for cam_key, cam_cfg in CAMERA_CONFIG.items():
            obs_key = f"observation.images.{cam_cfg['dataset_cam']}"
            if cam in {cam_key, cam_cfg["dataset_cam"]}:
                out.add(obs_key)
                matched = True
                break
        if not matched:
            raise ValueError(
                f"Unknown blackout camera {cam!r}. "
                f"Use one of: wrist, stationary, cam_wrist, cam_high, "
                f"or a full observation.images.* key."
            )
    return out


def apply_calibrated_cameras(
    data: MjData,
    model: MjModel,
    *,
    stationary_only: bool = False,
) -> None:
    """Apply camera calibration to MuJoCo cameras.

    Stationary cameras write their world pose into `data.cam_xpos/xmat`, which
    MuJoCo overwrites during stepping. Wrist cameras patch model-local pose and
    only need to be applied on reset / init.
    """
    for cam_cfg in CAMERA_CONFIG.values():
        cam_type = cam_cfg["config"].get("type", "stationary")
        if stationary_only and cam_type != "stationary":
            continue
        set_mujoco_camera_from_config(
            data, model, cam_cfg["mujoco_cam"], cam_cfg["config"]
        )


# ============================================================================
# Main
# ============================================================================


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--policy-path", type=str, required=True,
                   help="Path to policy checkpoint pretrained_model directory.")
    p.add_argument("--task", type=str, default="place_mug",
                   help="Task id (default: place_mug)")
    p.add_argument("--prompt", type=str, default=None,
                   help="Override task prompt.")
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--num-episodes", type=int, default=1)
    p.add_argument("--no-display", action="store_true", help="Run headless.")
    p.add_argument("--scene-path", type=str, default="pointclouds/xarm7_black.npz")
    p.add_argument("--policy-input-h", type=int, default=224)
    p.add_argument("--policy-input-w", type=int, default=224)
    p.add_argument("--blackout-cams", type=str, nargs="*", default=None,
                   help="Replace selected policy-input cameras with solid black. "
                        "Examples: wrist or stationary")

    # Attention viz options
    p.add_argument("--mode", type=str, default="rollout",
                   help="pi05: rollout | mean_layers | last_layer | layer=<i>. "
                        "groot: mean_layers | last_layer | layer=<i>.")
    p.add_argument("--keep-denoise", type=str, default="last",
                   choices=["last", "mean", "stack"],
                   help="How to aggregate attention across diffusion denoise steps.")
    p.add_argument("--alpha", type=float, default=0.5, help="Heatmap overlay alpha.")
    p.add_argument("--colormap", type=str, default="turbo")
    p.add_argument("--blur-sigma", type=float, default=2.0)
    p.add_argument("--head-fuse", type=str, default="mean", choices=["mean", "max", "min"])
    p.add_argument("--query-agg", type=str, default="mean", choices=["mean", "max", "last"])

    # Groot-specific layout hints
    p.add_argument("--groot-vision-start", type=int, default=None,
                   help="(single-image override) Key index where Groot's visual "
                        "tokens begin inside vl_embs. Ignored when spans are "
                        "auto-detected from input_ids.")
    p.add_argument("--groot-vision-tokens", type=int, default=None,
                   help="(single-image override) Number of visual tokens.")
    p.add_argument("--groot-vision-grid", type=int, nargs=2, default=None,
                   metavar=("H", "W"),
                   help="Grid shape (H W) per image for reshaping Groot visual "
                        "tokens. Default: 16 16 (Eagle2 / GR00T-N1.5).")
    p.add_argument("--groot-cam-token-ranges", type=str, nargs="*", default=None,
                   metavar="CAM:START-END",
                   help="Explicit per-camera VL token ranges, e.g. "
                        "cam_high:44-300 cam_wrist:300-556. Overrides "
                        "auto-detection.")

    p.add_argument("--output-dir", type=str, default="outputs/attention_viz",
                   help="Where to save per-episode mp4s and overlays.")
    args = p.parse_args()
    blackout_obs_keys = _resolve_blackout_obs_keys(args.blackout_cams)

    # ---- Load policy + build capture ----
    policy, policy_type, resolved_path = load_policy(args.policy_path)
    device = get_safe_torch_device(policy.config.device)
    policy = policy.to(device)

    # Processors
    proc_path = Path(resolved_path) / "policy_preprocessor.json"
    pre, post = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=resolved_path if proc_path.exists() else None,
    )

    capture_ctx = build_capture(policy, policy_type, keep_denoise=args.keep_denoise)

    # ---- Task profile ----
    task_profile = get_task_profile(args.task)
    prompt = args.prompt or task_profile.single_task
    print(f"[INFO] Task: {args.task}  |  Prompt: {prompt!r}")

    # ---- MuJoCo ----
    project_root = Path(__file__).parent.parent
    xarm_dir = project_root / "xarm7"
    scene_xml_path = resolve_task_scene_xml(args.task, xarm_dir)
    cwd = os.getcwd()
    try:
        os.chdir(str(xarm_dir))
        model = MjModel.from_xml_path(scene_xml_path.name)
    finally:
        os.chdir(cwd)

    data = MjData(model)
    try:
        home_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
        mujoco.mj_resetDataKeyframe(model, data, home_id)
    except Exception:
        mujoco.mj_resetData(model, data)

    gripper_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper")
    gripper_mj_range = (
        model.actuator_ctrlrange[gripper_act_id, 0],
        model.actuator_ctrlrange[gripper_act_id, 1],
    )

    RENDER_W, RENDER_H = 640, 480
    for cam_key, cam_cfg in CAMERA_CONFIG.items():
        K = cam_cfg["config"]["intrinsics"]
        fy = K[1, 1]
        fovy = float(2.0 * np.degrees(np.arctan(RENDER_H / (2.0 * fy))))
        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_cfg["mujoco_cam"])
        if cam_id >= 0:
            model.cam_fovy[cam_id] = fovy

    renderer = mujoco.Renderer(model, height=RENDER_H, width=RENDER_W)
    seg_renderer = mujoco.Renderer(model, height=RENDER_H, width=RENDER_W)
    seg_renderer.enable_segmentation_rendering()
    robot_geom_ids = get_robot_geom_ids(model)

    mujoco.mj_forward(model, data)
    apply_calibrated_cameras(data, model, stationary_only=False)

    camera_intrinsics = {
        cam_key: cam_cfg["config"]["intrinsics"]
        for cam_key, cam_cfg in CAMERA_CONFIG.items()
    }

    gaussian_data = None
    if os.path.exists(args.scene_path):
        try:
            init_pose = get_mujoco_camera_pose(model, data, "stationary_cam")
            w2c_init = mj_pose_to_gaussian_w2c(init_pose, T_splat2mj)
            scene_data, scene_depth_data, _ = load_scene_data(
                args.scene_path, w2c_init, camera_intrinsics["stationary"]
            )
            color_calib_by_camera = {}
            for cam_key in CAMERA_CONFIG:
                try:
                    color_calib_by_camera[cam_key] = load_color_mapping(
                        task_profile.color_calibration_path(cam_key)
                    )
                except Exception:
                    pass
            gaussian_data = {
                "scene_data": scene_data,
                "scene_depth_data": scene_depth_data,
                "viz_cfg": {"viz_w": RENDER_W, "viz_h": RENDER_H, "viz_near": 0.1, "viz_far": 10.0},
                "color_calib_by_camera": color_calib_by_camera,
                "camera_intrinsics": camera_intrinsics,
            }
            print(f"[INFO] Loaded Gaussian scene: {args.scene_path}")
        except Exception as e:
            print(f"[WARN] Failed to load Gaussian scene: {e}")
    else:
        print(f"[WARN] Gaussian scene not found ({args.scene_path}); using MuJoCo-only render.")

    # ---- Output dir ----
    out_dir = Path(args.output_dir) / f"{policy_type}_{args.task}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Saving attention visualizations to: {out_dir.resolve()}")

    # ---- Display windows ----
    if not args.no_display:
        cv2.namedWindow("attention", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("attention", 1024, 600)

    step_dt = 1.0 / args.fps
    print(f"[INFO] Policy: {policy_type}  |  mode={args.mode}  "
          f"|  keep_denoise={args.keep_denoise}  |  max_steps={args.max_steps}")
    if blackout_obs_keys:
        print(f"[INFO] Blacking out policy input cameras: {sorted(blackout_obs_keys)}")

    for ep in range(args.num_episodes):
        # Reset sim
        try:
            home_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
            mujoco.mj_resetDataKeyframe(model, data, home_id)
        except Exception:
            mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)
        apply_calibrated_cameras(data, model, stationary_only=False)
        policy.reset()

        frames = []  # dashboard frames for mp4
        last_overlays: dict[str, np.ndarray] = {}
        last_heatmaps: dict[str, np.ndarray] = {}
        print(f"\n=== Episode {ep + 1}/{args.num_episodes} ===")

        for step in range(args.max_steps):
            step_start = time.perf_counter()

            # `mj_step` overwrites stationary camera world pose stored in
            # `data.cam_xpos/xmat`, so re-apply it before every rendered frame.
            apply_calibrated_cameras(data, model, stationary_only=True)

            # Build & resize observation
            obs = build_observation(
                model, data, renderer, seg_renderer, robot_geom_ids, gaussian_data,
            )
            obs_for_policy = copy_observation_frame_with_resized_images(
                obs, args.policy_input_h, args.policy_input_w,
            )
            if hasattr(policy.config, "language_features") and policy.config.language_features:
                obs_for_policy["observation.language"] = prompt

            # Optional ablation: feed all-black images for selected cameras.
            # The dashboard top row is built from `obs_for_policy`, so it shows
            # exactly what the policy receives.
            for obs_key in blackout_obs_keys:
                img = obs_for_policy.get(obs_key)
                if isinstance(img, np.ndarray):
                    obs_for_policy[obs_key] = np.zeros_like(img)

            # Images we will overlay attention on (resized = what the policy saw)
            images_policy_input: dict[str, np.ndarray] = {}
            for cam_cfg in CAMERA_CONFIG.values():
                key = f"observation.images.{cam_cfg['dataset_cam']}"
                if key in obs_for_policy:
                    img = obs_for_policy[key]
                    if isinstance(img, np.ndarray):
                        images_policy_input[key] = img.copy()

            # ---- Run inference inside the capture context ----
            capture_ctx.reset()
            with torch.inference_mode(), capture_ctx:
                action = predict_action(
                    obs_for_policy, policy, device, pre, post,
                    policy.config.use_amp, task=prompt, robot_type="xarm_follower",
                )

            # ---- Build attention overlays ----
            overlays: dict[str, np.ndarray] = {}
            # If the capture context actually ran real inference this step,
            # rebuild overlays. Otherwise (action queue serving cached chunk),
            # reuse the most recent valid overlays so the dashboard doesn't go
            # black between chunks.
            had_new_inference = getattr(capture_ctx, "had_calls", True)

            if not had_new_inference and last_heatmaps:
                # The policy may serve cached actions from its action queue
                # between real forward passes. In those steps we do not have
                # fresh attention, so reuse the last heatmap but stamp it onto
                # the *current* image to keep the dashboard rows aligned.
                for cam_key, img in images_policy_input.items():
                    heat = last_heatmaps.get(cam_key)
                    if heat is not None:
                        overlays[cam_key] = overlay_heatmap(
                            img, heat, args.alpha, args.colormap
                        )
            elif not had_new_inference and last_overlays:
                overlays = last_overlays
            elif policy_type == "pi05":
                try:
                    overlays = render_pi05_overlays(
                        capture_ctx, images_policy_input,
                        mode=args.mode,
                        alpha=args.alpha,
                        colormap=args.colormap,
                        blur_sigma=args.blur_sigma,
                        query_agg=args.query_agg,
                        head_fuse=args.head_fuse,
                    )
                except Exception as e:
                    if not had_new_inference:
                        overlays = last_overlays
                    else:
                        print(f"[WARN] pi05 attention build failed: {e}")
            elif policy_type == "groot":
                cam_keys_in_order = list(images_policy_input.keys())
                spans = _resolve_groot_spans(
                    cam_keys=cam_keys_in_order,
                    cli_ranges=args.groot_cam_token_ranges,
                    cli_single_start=args.groot_vision_start,
                    cli_single_tokens=args.groot_vision_tokens,
                    cli_grid=args.groot_vision_grid,
                    auto_spans=capture_ctx.vision_token_spans,
                )

                new_heatmaps: dict[str, np.ndarray] = {}
                if spans is not None:
                    try:
                        default_grid = args.groot_vision_grid or (16, 16)
                        for cam_key, (vs, ve) in zip(cam_keys_in_order, spans):
                            img = images_policy_input[cam_key]
                            num = ve - vs
                            gh_i, gw_i = _grid_for_span(num, default_grid)
                            v = groot_attention_for_vision(
                                capture_ctx,
                                vl_start=vs, vl_end=ve,
                                grid_h=gh_i, grid_w=gw_i,
                                mode=args.mode,
                                query_agg=args.query_agg,
                                head_fuse=args.head_fuse,
                            )
                            heat = attention_to_heatmap(
                                v, gh_i, gw_i, img.shape[:2],
                                normalize="minmax", blur_sigma=args.blur_sigma,
                            )
                            new_heatmaps[cam_key] = heat
                            overlays[cam_key] = overlay_heatmap(
                                img, heat, args.alpha, args.colormap,
                            )
                    except RuntimeError as e:
                        if not had_new_inference and last_heatmaps:
                            for cam_key, img in images_policy_input.items():
                                heat = last_heatmaps.get(cam_key)
                                if heat is not None:
                                    overlays[cam_key] = overlay_heatmap(
                                        img, heat, args.alpha, args.colormap
                                    )
                        elif not had_new_inference and last_overlays:
                            overlays = last_overlays
                        elif not getattr(main, "_groot_attn_err_shown", False):
                            print(f"[WARN] Groot attention build failed: {e}")
                            main._groot_attn_err_shown = True  # type: ignore[attr-defined]
                if new_heatmaps:
                    last_heatmaps = new_heatmaps

            if overlays:
                last_overlays = overlays

            # ---- Display + save ----
            dashboard = make_attention_dashboard(
                raw_images=images_policy_input,
                overlays=overlays,
                row_height=224,
            )
            frames.append(dashboard.copy())

            if not args.no_display:
                cv2.imshow(
                    "attention",
                    cv2.cvtColor(dashboard, cv2.COLOR_RGB2BGR),
                )
                k = cv2.waitKey(1) & 0xFF
                if k == 27:
                    print("[INFO] ESC pressed, stopping.")
                    break

            # ---- Step sim ----
            ctrl = lerobot_state_to_mujoco_ctrl(
                action.cpu().numpy() if action.ndim == 1 else action.cpu().numpy()[0],
                gripper_mj_range,
            )
            data.ctrl[:] = ctrl
            target = data.time + step_dt
            while data.time < target:
                mujoco.mj_step(model, data)

            elapsed = time.perf_counter() - step_start
            if elapsed < step_dt:
                time.sleep(step_dt - elapsed)

            if step % 50 == 0:
                print(f"  step {step}/{args.max_steps}")

        # ---- Save mp4 for this episode ----
        if frames:
            out_path = out_dir / f"episode_{ep:03d}_{args.mode}.mp4"
            h, w = frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(out_path), fourcc, args.fps, (w, h))
            for f in frames:
                writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
            writer.release()
            print(f"[INFO] Saved {len(frames)} frames -> {out_path}")

    if not args.no_display:
        cv2.destroyAllWindows()
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
