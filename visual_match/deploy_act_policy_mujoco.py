#!/usr/bin/env python3
"""
Deploy ACT policy in MuJoCo xArm simulation.

This script loads an ACT policy checkpoint and runs it in MuJoCo xArm simulation.
Uses 2 cameras: wrist and stationary, both with composite rendering (Gaussian Splatting
background + MuJoCo robot foreground). Refer to compare_recorded_vs_mujoco for the
xArm observation.state format (8-dim): [joint1..7 in degrees, gripper in mm (0=closed, 800=open)]

composite rendering pipeline.

Usage:
    python visual_match/deploy_act_policy_mujoco.py \
        --policy-path outputs/train/act_pick_cuber/checkpoints/080000/pretrained_model \
        --prompt "Pick up the cube" \
        --fps 30
"""

import sys
import os
import re
import argparse
from pathlib import Path
import time
import json

# Add src to path for lerobot imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

# Auto-detect display
def _detect_display():
    if os.environ.get("DISPLAY"):
        return True
    if os.environ.get("WAYLAND_DISPLAY"):
        return True
    if sys.platform in ("darwin", "win32"):
        return True
    return False

_HAS_DISPLAY = _detect_display()
if not _HAS_DISPLAY:
    os.environ["MUJOCO_GL"] = "egl"
    print("[WARN] No display detected, using EGL rendering")

import numpy as np
import torch
import cv2
import mujoco
from mujoco import MjModel, MjData

if _HAS_DISPLAY:
    import mujoco.viewer

from lerobot.policies.factory import get_policy_class
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.control_utils import predict_action
from lerobot.utils.utils import get_safe_torch_device
from lerobot.utils.constants import OBS_STATE
from lerobot.policies.factory import make_pre_post_processors

# ============================================================================
# Import from compare_recorded_vs_mujoco
# ============================================================================
from camera_config import load_camera_config, set_mujoco_camera_from_config
from compare_recorded_vs_mujoco import (
    get_robot_geom_ids,
    load_scene_data,
    render_gaussian,
    get_mujoco_camera_pose,
    mj_pose_to_gaussian_w2c,
    lerobot_state_to_mujoco_ctrl,
    GRIPPER_OPEN_MM,
)
from compare_recorded_vs_mujoco import T_splat2mj

# Camera configuration (same as compare_recorded_vs_mujoco)
_stationary_cfg = load_camera_config("stationary_cam")
_wrist_cfg = load_camera_config("wrist_cam")
CAMERA_CONFIG = {
    "stationary": {
        "dataset_cam": "cam_high",
        "mujoco_cam": "stationary_cam",
        "config": _stationary_cfg,
    },
    "wrist": {
        "dataset_cam": "cam_wrist",
        "mujoco_cam": "wrist_cam",
        "config": _wrist_cfg,
    },
}

# ============================================================================
# Color calibration functions
# ============================================================================

def _get_aug(x: np.ndarray, add_ones: bool = False) -> np.ndarray:
    """Augment input features for quadratic polynomial regression."""
    if add_ones:
        ones = np.ones((x.shape[0], 1), np.float64)
        return np.hstack([x ** 2, x, ones])
    return np.hstack([x ** 2, x])


def load_color_mapping(yaml_path: str):
    """Load color transform from color_mapping.yaml file."""
    with open(yaml_path, 'r') as f:
        content = f.read()
    a_match = re.search(r'color_A:\s*\[(.*?)\]', content, re.DOTALL)
    if not a_match:
        raise ValueError(f"Could not find color_A in {yaml_path}")
    a_values = [float(x.strip()) for x in a_match.group(1).replace('\n', '').split(',')]
    A = np.array(a_values, dtype=np.float32).reshape(3, 6)
    b_match = re.search(r'color_b:\s*\[(.*?)\]', content, re.DOTALL)
    if not b_match:
        raise ValueError(f"Could not find color_b in {yaml_path}")
    b_values = [float(x.strip()) for x in b_match.group(1).replace('\n', '').split(',')]
    b = np.array(b_values, dtype=np.float32)
    return A, b


def apply_color_transform(img: np.ndarray, A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Apply quadratic color transform to image (RGB uint8)."""
    flat = img.reshape(-1, 3).astype(np.float32) / 255.0
    flat_aug = _get_aug(flat, add_ones=False)
    out = flat_aug @ A.T + b
    out = np.clip(out, 0.0, 1.0)
    out_rgb = (out.reshape(img.shape) * 255.0).astype(np.uint8)
    return out_rgb


# ============================================================================
# xArm state conversion (qpos -> lerobot state)
# ============================================================================

def mujoco_qpos_to_lerobot_state(qpos: np.ndarray, gripper_mj_range: tuple) -> np.ndarray:
    """
    Convert MuJoCo qpos (8-dim: 7 joints rad + gripper) to xArm LeRobot state.
    State format: [joint1..7 in degrees, gripper in mm (0=closed, 800=open)]
    """
    state = np.zeros(8, dtype=np.float32)
    state[:7] = np.rad2deg(qpos[:7])
    mj_lo, mj_hi = gripper_mj_range
    grip_frac = np.clip((qpos[7] - mj_lo) / (mj_hi - mj_lo), 0.0, 1.0)
    state[7] = grip_frac * GRIPPER_OPEN_MM
    return state


def display_camera_images(observation: dict, policy_config=None, window_name_prefix: str = "Camera"):
    """Display camera images from observation dict in OpenCV windows."""
    all_image_keys = [k for k in observation.keys() if "image" in k.lower()]
    if policy_config is not None and hasattr(policy_config, 'image_features') and policy_config.image_features:
        image_keys = [k for k in all_image_keys if k in policy_config.image_features]
    else:
        image_keys = all_image_keys

    for img_key in image_keys:
        img = observation[img_key]
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        window_name = img_key.split(".")[-1] if "." in img_key else img_key
        window_full_name = f"{window_name_prefix}: {window_name}"
        cv2.imshow(window_full_name, img_bgr)
    cv2.waitKey(1)


def load_policy(policy_path: str) -> tuple[PreTrainedPolicy, dict]:
    """Load ACT policy from checkpoint path."""
    print(f"[INFO] Loading policy from: {policy_path}")

    policy_path_obj = Path(policy_path)
    if not policy_path_obj.is_absolute():
        if not policy_path_obj.exists():
            project_root = Path(__file__).parent.parent
            policy_path_obj = project_root / policy_path
        if not policy_path_obj.exists():
            raise FileNotFoundError(f"Policy path not found: {policy_path}")

    policy_path = str(policy_path_obj.resolve())

    config_path = Path(policy_path) / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config_dict = json.load(f)

    policy_type = config_dict.get("type", "act")
    policy_class = get_policy_class(policy_type)
    policy = policy_class.from_pretrained(policy_path)
    if policy_type != "act":
        print(f"[WARN] Policy type is {policy_type}, expected 'act'")

    policy.eval()
    print(f"[INFO] Policy loaded: {policy_type}")
    return policy, config_dict


def build_observation_from_mujoco(model: MjModel, data: MjData, renderer: mujoco.Renderer,
                                  gripper_mj_range: tuple,
                                  seg_renderer: mujoco.Renderer,
                                  robot_geom_ids: set,
                                  gaussian_data: dict | None) -> dict:
    """
    Build observation dict for xArm policy from MuJoCo state.
    Uses 2 cameras: cam_high (stationary) and cam_wrist, both with composite rendering.
    """
    state = mujoco_qpos_to_lerobot_state(data.qpos, gripper_mj_range)
    observation = {OBS_STATE: state}

    use_composite = (gaussian_data is not None and
                     gaussian_data.get('scene_data') is not None and
                     seg_renderer is not None and
                     robot_geom_ids is not None)

    camera_intrinsics = gaussian_data.get('camera_intrinsics', {}) if gaussian_data else {}

    for cam_key, cam_cfg in CAMERA_CONFIG.items():
        mujoco_cam = cam_cfg["mujoco_cam"]
        obs_key = f"observation.images.{cam_cfg['dataset_cam']}"

        if use_composite:
            rgb_image = render_composite_view(
                model, data, renderer, seg_renderer, robot_geom_ids,
                mujoco_cam, gaussian_data, camera_intrinsics.get(cam_key)
            )
        else:
            renderer.update_scene(data, camera=mujoco_cam)
            rgb_image = renderer.render()

        observation[obs_key] = rgb_image

    return observation


def render_composite_view(model: MjModel, data: MjData,
                          renderer: mujoco.Renderer, seg_renderer: mujoco.Renderer,
                          robot_geom_ids: set, cam_name: str, gaussian_data: dict,
                          intrinsics: np.ndarray | None) -> np.ndarray:
    """
    Render composite view: Gaussian Splatting background + MuJoCo robot foreground.
    """
    renderer.update_scene(data, camera=cam_name)
    fg_rgb = renderer.render()

    seg_renderer.update_scene(data, camera=cam_name)
    seg_mask = seg_renderer.render()
    seg_labels = seg_mask[:, :, 0].astype(np.int32)
    seg_labels[seg_labels == -1] = 0
    robot_mask = np.isin(seg_labels, list(robot_geom_ids))
    mask_uint8 = (robot_mask.astype(np.uint8)) * 255

    if intrinsics is not None and gaussian_data.get('scene_data') is not None:
        try:
            cc = None
            for cam_cfg in CAMERA_CONFIG.values():
                if cam_cfg["mujoco_cam"] == cam_name:
                    cc = cam_cfg["config"]
                    break
            if cc is not None and cc.get("type", "stationary") == "stationary":
                camera_pose = np.eye(4)
                camera_pose[:3, :3] = cc["cam_xmat_mj"]
                camera_pose[:3, 3] = cc["cam_pos_mj"]
            else:
                camera_pose = get_mujoco_camera_pose(model, data, cam_name)
            w2c = mj_pose_to_gaussian_w2c(camera_pose, T_splat2mj)
            viz_cfg = gaussian_data['viz_cfg']
            bg_im = render_gaussian(w2c, intrinsics, gaussian_data['scene_data'],
                                    gaussian_data['scene_depth_data'], viz_cfg)
            bg_np = bg_im.permute(1, 2, 0).cpu().numpy()
            bg_np = (bg_np * 255).astype(np.uint8)
            composite = bg_np.copy()
            composite[mask_uint8 > 0] = fg_rgb[mask_uint8 > 0]
            if 'color_calib' in gaussian_data and gaussian_data['color_calib'] is not None:
                color_A, color_b = gaussian_data['color_calib']
                composite = apply_color_transform(composite, color_A, color_b)
            return composite
        except Exception as e:
            print(f"[WARN] Gaussian rendering failed for {cam_name}: {e}")
    return fg_rgb


def convert_action_to_mujoco(action: torch.Tensor, gripper_mj_range: tuple) -> np.ndarray:
    """
    Convert policy action (8-dim: 7 joints degrees + gripper mm) to MuJoCo ctrl (8-dim).
    """
    action_np = action.cpu().numpy()
    if action_np.ndim > 1:
        action_np = action_np[0]
    return lerobot_state_to_mujoco_ctrl(action_np, gripper_mj_range)


def main():
    parser = argparse.ArgumentParser(
        description="Deploy ACT policy in MuJoCo xArm simulation"
    )
    parser.add_argument(
        "--policy-path",
        type=str,
        default="outputs/act_xarm_training/checkpoints/last/pretrained_model",
        help="Path to policy checkpoint directory"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Pick up the cube",
        help="Task prompt/instruction for the policy"
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Control frequency (default: 30.0)"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=1000,
        help="Maximum number of steps to run (default: 1000)"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without GUI (for headless servers)"
    )
    parser.add_argument(
        "--scene-path",
        type=str,
        default="pointclouds/xarm7.npz",
        help="Path to Gaussian Splatting scene file for composite rendering"
    )
    parser.add_argument(
        "--color-calib-path",
        type=str,
        default=None,
        help="Path to color calibration YAML file (optional)"
    )

    args = parser.parse_args()

    # Load policy
    policy, config_dict = load_policy(args.policy_path)
    device = get_safe_torch_device(policy.config.device)
    policy = policy.to(device)

    print(f"[INFO] Policy action parameters:")
    if hasattr(policy.config, 'horizon'):
        print(f"  - horizon: {policy.config.horizon}")
    if hasattr(policy.config, 'n_action_steps'):
        print(f"  - n_action_steps: {policy.config.n_action_steps}")
    if hasattr(policy.config, 'chunk_size'):
        print(f"  - chunk_size: {policy.config.chunk_size}")

    # Create pre/post processors
    processor_path = Path(args.policy_path) / "policy_preprocessor.json"
    if processor_path.exists():
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=policy.config,
            pretrained_path=args.policy_path,
        )
    else:
        print("[WARN] Processor files not found, creating from config")
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=policy.config,
            pretrained_path=None,
        )

    # Load MuJoCo xArm model
    project_root = Path(__file__).parent.parent
    xarm_dir = project_root / "xarm7"
    original_cwd = os.getcwd()
    try:
        os.chdir(str(xarm_dir))
        model = MjModel.from_xml_path("scene.xml")
    finally:
        os.chdir(original_cwd)

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
    print(f"[INFO] Gripper ctrl range: [{gripper_mj_range[0]}, {gripper_mj_range[1]}]")

    RENDER_W, RENDER_H = 640, 480
    renderer = mujoco.Renderer(model, height=RENDER_H, width=RENDER_W)
    seg_renderer = mujoco.Renderer(model, height=RENDER_H, width=RENDER_W)
    seg_renderer.enable_segmentation_rendering()

    robot_geom_ids = get_robot_geom_ids(model)
    print(f"[INFO] Found {len(robot_geom_ids)} robot geoms for masking")

    # Apply camera calibration
    mujoco.mj_forward(model, data)
    for cam_key, cam_cfg in CAMERA_CONFIG.items():
        mj_cam = cam_cfg["mujoco_cam"]
        cc = cam_cfg["config"]
        cam_id = set_mujoco_camera_from_config(data, model, mj_cam, cc)
        print(f"[INFO] Camera '{mj_cam}' (id={cam_id}) calibration applied")

    # Camera intrinsics from config (used for Gaussian rendering)
    camera_intrinsics = {cam_key: cam_cfg["config"]["intrinsics"]
                         for cam_key, cam_cfg in CAMERA_CONFIG.items()}

    # Load Gaussian Splatting scene
    gaussian_data = None
    if os.path.exists(args.scene_path):
        try:
            init_pose = get_mujoco_camera_pose(model, data, "stationary_cam")
            w2c_init = mj_pose_to_gaussian_w2c(init_pose, T_splat2mj)
            scene_data, scene_depth_data = load_scene_data(
                args.scene_path, w2c_init, camera_intrinsics["stationary"]
            )
            color_calib = None
            if args.color_calib_path and os.path.exists(args.color_calib_path):
                try:
                    color_A, color_b = load_color_mapping(args.color_calib_path)
                    color_calib = (color_A, color_b)
                    print(f"[INFO] Loaded color calibration from: {args.color_calib_path}")
                except Exception as e:
                    print(f"[WARN] Failed to load color calibration: {e}")
            viz_cfg = {'viz_w': RENDER_W, 'viz_h': RENDER_H, 'viz_near': 0.1, 'viz_far': 10.0}
            gaussian_data = {
                'scene_data': scene_data,
                'scene_depth_data': scene_depth_data,
                'viz_cfg': viz_cfg,
                'color_calib': color_calib,
                'camera_intrinsics': camera_intrinsics,
            }
            print(f"[INFO] Loaded Gaussian Splatting scene from: {args.scene_path}")
        except Exception as e:
            print(f"[WARN] Failed to load Gaussian scene: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"[WARN] Scene file not found: {args.scene_path}")

    policy.reset()

    if not args.headless:
        for cam_key in CAMERA_CONFIG:
            obs_key = f"observation.images.{CAMERA_CONFIG[cam_key]['dataset_cam']}"
            cv2.namedWindow(f"Camera: {obs_key.split('.')[-1]}", cv2.WINDOW_NORMAL)

    print(f"[INFO] Starting policy deployment (max {args.max_steps} steps)")

    step_dt = 1.0 / args.fps
    step = 0

    viewer = None
    viewer_ctx = None
    if not args.headless and _HAS_DISPLAY:
        try:
            viewer_ctx = mujoco.viewer.launch_passive(model, data)
            viewer = viewer_ctx.__enter__()
        except Exception:
            viewer = None
            viewer_ctx = None

    try:
        while step < args.max_steps:
            step_start = time.perf_counter()

            # Re-apply stationary camera pose (mj_step may reset data.cam_xpos)
            for cam_cfg in CAMERA_CONFIG.values():
                if cam_cfg["config"].get("type", "stationary") == "stationary":
                    set_mujoco_camera_from_config(data, model, cam_cfg["mujoco_cam"], cam_cfg["config"])

            observation = build_observation_from_mujoco(
                model, data, renderer, gripper_mj_range,
                seg_renderer=seg_renderer,
                robot_geom_ids=robot_geom_ids,
                gaussian_data=gaussian_data,
            )

            if not args.headless:
                display_camera_images(observation, policy_config=policy.config)

            if hasattr(policy.config, 'language_features') and policy.config.language_features:
                observation["observation.language"] = args.prompt

            with torch.inference_mode():
                action = predict_action(
                    observation,
                    policy,
                    device,
                    preprocessor,
                    postprocessor,
                    policy.config.use_amp,
                    task=args.prompt,
                    robot_type="xarm_follower",
                )

            ctrl = convert_action_to_mujoco(action, gripper_mj_range)
            data.ctrl[:] = ctrl

            sim_target = data.time + step_dt
            while data.time < sim_target:
                mujoco.mj_step(model, data)

            if viewer is not None:
                viewer.sync()

            elapsed = time.perf_counter() - step_start
            sleep_time = max(0, step_dt - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

            step += 1
            if step % 100 == 0:
                print(f"[INFO] Step {step}/{args.max_steps}")

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
    finally:
        if viewer_ctx is not None:
            try:
                viewer_ctx.__exit__(None, None, None)
            except Exception:
                pass
        if not args.headless:
            cv2.destroyAllWindows()
        print("[INFO] Deployment finished")


if __name__ == "__main__":
    main()
