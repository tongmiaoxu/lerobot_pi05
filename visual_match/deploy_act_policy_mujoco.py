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

    # Use real-world dataset images as policy input (instead of MuJoCo rendering):
    python visual_match/deploy_act_policy_mujoco.py --obs --dataset-path data --episode 0

    # Same as --obs but fixed to episode 0 of the eval dataset (default: data_eval):
    python visual_match/deploy_act_policy_mujoco.py --obs-eval
"""

import sys
import os
import re
import math
import argparse
from pathlib import Path
import time
import json
import threading

# Add src to path for lerobot imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))

# Auto-detect display (for cv2.imshow, mujoco.viewer)
def _detect_display():
    if os.environ.get("DISPLAY"):
        return True
    if os.environ.get("WAYLAND_DISPLAY"):
        return True
    if sys.platform in ("darwin", "win32"):
        return True
    return False

_HAS_DISPLAY = _detect_display()

# Use GLX when a display is available (supports interactive viewer + offscreen),
# fall back to EGL for headless/SSH environments (offscreen only).
if "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "glx" if _HAS_DISPLAY else "egl"

import numpy as np
import torch
import cv2
import mujoco
from mujoco import MjModel, MjData

if _HAS_DISPLAY:
    import mujoco.viewer

from lerobot.policies.factory import get_policy_class
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.control_utils import predict_action, init_keyboard_listener
from lerobot.utils.utils import get_safe_torch_device
from lerobot.utils.constants import OBS_STATE
from lerobot.policies.factory import make_pre_post_processors

# ============================================================================
# Imports: composite rendering from composite_rendering, xArm conversion from compare_recorded
# ============================================================================
from camera_config import load_camera_config, set_mujoco_camera_from_config
from composite_rendering import (
    get_mujoco_camera_pose,
    get_robot_geom_ids,
    load_scene_data,
    mj_pose_to_gaussian_w2c,
    render,
    shift_for_principal_point,
    T_splat2mj,
)
from compare_recorded_vs_mujoco import load_episode
from lerobot_mujoco_utils import (
    GRIPPER_OPEN_MM,
    lerobot_state_to_mujoco_ctrl,
    mujoco_qpos_to_lerobot_state,
)
from lerobot.datasets.video_utils import decode_video_frames

# Arrow key codes for cv2.waitKeyEx (platform-dependent)
_KEY_LEFT  = (65361, 81, 2)
_KEY_RIGHT = (65363, 83, 3)
_KEY_UP    = (65362, 82, 0)
_KEY_DOWN  = (65364, 84, 1)
MUG_STEP_INIT_M = 0.005  # 5 mm initial step for mug adjustment

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


def apply_gemini_parallel(translator, observation: dict) -> dict:
    """Few-shot Gemini per camera in parallel; same examples as query_gemini (translator holds pairs)."""
    out = dict(observation)
    lock = threading.Lock()
    errs: list[tuple[str, RuntimeError]] = []

    def work(cam_key: str, obs_key: str, img: np.ndarray):
        try:
            translated = translator.translate(np.ascontiguousarray(img), cam_key)
            with lock:
                out[obs_key] = translated
        except RuntimeError as e:
            with lock:
                errs.append((cam_key, e))

    threads = []
    for cam_key, cam_cfg in CAMERA_CONFIG.items():
        obs_key = f"observation.images.{cam_cfg['dataset_cam']}"
        if obs_key not in out:
            continue
        img = out[obs_key]
        if not isinstance(img, np.ndarray):
            continue
        t = threading.Thread(target=work, args=(cam_key, obs_key, img))
        threads.append(t)
        t.start()
    for t in threads:
        t.join()
    for cam_key, e in errs:
        print(f"  [WARN] Gemini failed for {cam_key}: {e}")
    return out


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
    """Load color transform from color_mapping.yaml file. Supports affine (3x3) or quadratic (3x6)."""
    with open(yaml_path, 'r') as f:
        content = f.read()
    a_match = re.search(r'color_A:\s*\[(.*?)\]', content, re.DOTALL)
    if not a_match:
        raise ValueError(f"Could not find color_A in {yaml_path}")
    a_values = [float(x.strip()) for x in a_match.group(1).replace('\n', '').split(',')]
    b_match = re.search(r'color_b:\s*\[(.*?)\]', content, re.DOTALL)
    if not b_match:
        raise ValueError(f"Could not find color_b in {yaml_path}")
    b_values = [float(x.strip()) for x in b_match.group(1).replace('\n', '').split(',')]
    b = np.array(b_values, dtype=np.float32)
    if len(a_values) == 9:
        A = np.array(a_values, dtype=np.float32).reshape(3, 3)
        return ("affine", A, b)
    if len(a_values) == 18:
        A = np.array(a_values, dtype=np.float32).reshape(3, 6)
        return ("quadratic", A, b)
    raise ValueError(f"color_A must have 9 (affine) or 18 (quadratic) values, got {len(a_values)}")


def apply_color_transform(img: np.ndarray, calib: tuple) -> np.ndarray:
    """Apply color transform. calib is (fmt, A, b) from load_color_mapping."""
    fmt, A, b = calib
    flat = img.reshape(-1, 3).astype(np.float32) / 255.0
    if fmt == "affine":
        out = flat @ A.T + b
    else:
        flat_aug = _get_aug(flat, add_ones=False)
        out = flat_aug @ A.T + b
    out = np.clip(out, 0.0, 1.0)
    out_rgb = (out.reshape(img.shape) * 255.0).astype(np.uint8)
    return out_rgb


def load_dataset_frames(episode_data: dict):
    """
    Load video frames for all cameras from episode data.
    Returns dict: cam_key -> list of RGB frames (H, W, 3) uint8.
    """
    dataset = episode_data["dataset"]
    episode_idx = episode_data["episode_index"]
    num_frames = episode_data["num_frames"]
    start_idx = episode_data["video_start_frame"]
    video_fps = dataset.fps

    relative_timestamps = [i / video_fps for i in range(num_frames)]
    ep_meta = dataset.meta.episodes[episode_idx]

    cam_frames = {}
    for cam_key, cam_cfg in CAMERA_CONFIG.items():
        dataset_cam = cam_cfg["dataset_cam"]
        camera_key = f"observation.images.{dataset_cam}"
        try:
            video_path_rel = dataset.meta.get_video_file_path(episode_idx, camera_key)
            video_path = dataset.root / video_path_rel
            if not video_path.exists():
                print(f"[WARN] Video not found for {cam_key}: {video_path}")
                cam_frames[cam_key] = []
                continue
            from_timestamp = ep_meta.get(f"videos/{camera_key}/from_timestamp", 0.0)
            absolute_timestamps = [from_timestamp + ts for ts in relative_timestamps]
            frames_tensor = decode_video_frames(
                video_path, absolute_timestamps, tolerance_s=1e-4, backend="pyav"
            )
            frames_list = []
            for i in range(frames_tensor.shape[0]):
                frame = frames_tensor[i].permute(1, 2, 0).cpu().numpy()
                frame = (frame * 255).astype(np.uint8)  # RGB uint8
                frames_list.append(frame)
            cam_frames[cam_key] = frames_list
            print(f"[INFO] Loaded {len(frames_list)} real frames for {cam_key}")
        except Exception as e:
            print(f"[WARN] Failed to load {cam_key} video: {e}")
            cam_frames[cam_key] = []
    return cam_frames


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
                                  seg_renderer: mujoco.Renderer,
                                  robot_geom_ids: set,
                                  gaussian_data: dict | None,
                                  obs_frames: dict | None = None,
                                  frame_idx: int = 0) -> dict:
    """
    Build observation dict for xArm policy from MuJoCo state.
    Uses 2 cameras: cam_high (stationary) and cam_wrist, both with composite rendering.
    When obs_frames is provided (--obs mode), use real dataset images instead of rendered.
    """
    ld_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "left_driver_joint")
    if ld_id < 0:
        raise RuntimeError("MuJoCo model has no joint 'left_driver_joint' (gripper driver).")
    g_adr = int(model.jnt_qposadr[ld_id])
    g_rad = (float(model.jnt_range[ld_id, 0]), float(model.jnt_range[ld_id, 1]))
    state = mujoco_qpos_to_lerobot_state(
        data.qpos, g_rad, gripper_qpos_adr=g_adr
    )
    observation = {OBS_STATE: state}

    # Use real-world dataset images when --obs
    if obs_frames is not None:
        for cam_key, cam_cfg in CAMERA_CONFIG.items():
            obs_key = f"observation.images.{cam_cfg['dataset_cam']}"
            frames = obs_frames.get(cam_key, [])
            if frames:
                idx = frame_idx % len(frames)
                observation[obs_key] = frames[idx].copy()
            else:
                renderer.update_scene(data, camera=cam_cfg["mujoco_cam"])
                observation[obs_key] = renderer.render()
        return observation

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
    Matches the compositing pipeline from compare_recorded_vs_mujoco.
    """
    renderer.update_scene(data, camera=cam_name)
    fg_rgb = renderer.render()

    seg_renderer.update_scene(data, camera=cam_name)
    seg_mask = seg_renderer.render()
    seg_labels = seg_mask[:, :, 0].astype(np.int32)

    if intrinsics is not None:
        fg_rgb = shift_for_principal_point(fg_rgb, intrinsics)
        seg_labels = shift_for_principal_point(seg_labels, intrinsics, seg=True)

    robot_mask = np.isin(seg_labels, list(robot_geom_ids))
    mask_uint8 = (robot_mask.astype(np.uint8)) * 255

    if intrinsics is not None and gaussian_data.get('scene_data') is not None:
        try:
            camera_pose = get_mujoco_camera_pose(model, data, cam_name)
            w2c = mj_pose_to_gaussian_w2c(camera_pose, T_splat2mj)
            viz_cfg = gaussian_data['viz_cfg']
            bg_im = render(w2c, intrinsics, gaussian_data['scene_data'],
                          gaussian_data['scene_depth_data'], viz_cfg)[0]
            bg_np = bg_im.permute(1, 2, 0).cpu().numpy()
            bg_np = (bg_np * 255).astype(np.uint8)
            composite = bg_np.copy()
            composite[mask_uint8 > 0] = fg_rgb[mask_uint8 > 0]
            if 'color_calib' in gaussian_data and gaussian_data['color_calib'] is not None:
                composite = apply_color_transform(composite, gaussian_data['color_calib'])
            return composite
        except Exception as e:
            print(f"[WARN] Gaussian rendering failed for {cam_name}: {e}")
    return fg_rgb


def load_initial_state_contours(
    initial_states_dir: str | Path | None = None,
    object_name: str = "mug",
) -> list:
    """
    Load per-episode object contours from masks generated by initial_states_overlay.py.

    Each element of the returned list corresponds to one training episode and
    contains the cv2 contour arrays extracted from its binary mask.  Episodes
    whose mask file is missing or empty yield an empty list.

    Args:
        initial_states_dir: Directory produced by initial_states_overlay.py.
            Defaults to ``visual_match/initial_states/``.
        object_name: Sub-directory name matching the TEXT_PROMPT used during
            segmentation (default ``"mug"``).

    Returns:
        list_of_contours – list (one entry per episode, sorted by episode
        index) of ``list[np.ndarray]`` contour arrays.
    """
    if initial_states_dir is None:
        initial_states_dir = Path(__file__).parent / "initial_states"
    masks_dir = Path(initial_states_dir) / object_name / "individual_masks"
    if not masks_dir.exists():
        raise FileNotFoundError(
            f"Mask directory not found: {masks_dir}\n"
            "Run initial_states_overlay.py first to generate masks."
        )

    mask_files = sorted(masks_dir.glob("ep_*_mask.png"))
    if not mask_files:
        raise FileNotFoundError(f"No mask files found in: {masks_dir}")

    list_of_contours = []
    for mask_path in mask_files:
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            list_of_contours.append([])
            continue
        _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        list_of_contours.append(list(contours))

    ep_ids = [
        int(p.stem.replace("ep_", "").replace("_mask", "")) for p in mask_files
    ]
    print(f"[INFO] Loaded contours for {len(list_of_contours)} episodes "
          f"(eps {min(ep_ids)}–{max(ep_ids)}) from {masks_dir}")
    return list_of_contours


def select_contours_ui(
    list_of_contours: list,
    num_eval_episodes: int,
    initial_states_dir: str | Path | None = None,
    object_name: str = "mug",
) -> tuple[list, list[int]]:
    """
    Interactive UI for selecting which training-episode contours to evaluate.

    Displays the all_episodes_grid image generated by initial_states_overlay.py.
    The user clicks on grid cells to select / deselect episodes.  Selected cells
    are highlighted in green.  Press ENTER to confirm or ESC to cancel.

    Args:
        list_of_contours: Output of :func:`load_initial_state_contours`.
        num_eval_episodes: How many episodes the user should pick.
        initial_states_dir: Root output directory of initial_states_overlay.py.
        object_name: Sub-directory name (default ``"mug"``).

    Returns:
        (selected_contours, selected_indices) where *selected_contours* is a
        list of ``list[np.ndarray]`` contour arrays (one per chosen episode)
        and *selected_indices* is the corresponding episode-index list, both
        sorted in ascending episode order.
    """
    if initial_states_dir is None:
        initial_states_dir = Path(__file__).parent / "initial_states"
    grid_path = Path(initial_states_dir) / object_name / "all_episodes_grid.png"
    if not grid_path.exists():
        raise FileNotFoundError(
            f"Grid image not found: {grid_path}\n"
            "Run initial_states_overlay.py first."
        )

    grid_img = cv2.imread(str(grid_path))
    if grid_img is None:
        raise RuntimeError(f"Failed to read grid image: {grid_path}")

    n = len(list_of_contours)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    grid_h, grid_w = grid_img.shape[:2]
    thumb_w = grid_w // cols
    thumb_h = grid_h // rows

    # Burn episode labels and dim cells with no contour
    base_img = grid_img.copy()
    for idx in range(n):
        r, c = divmod(idx, cols)
        x0, y0 = c * thumb_w, r * thumb_h
        has_contour = bool(list_of_contours[idx])
        if not has_contour:
            overlay = base_img.copy()
            cv2.rectangle(overlay, (x0, y0), (x0 + thumb_w, y0 + thumb_h), (80, 80, 80), -1)
            cv2.addWeighted(overlay, 0.6, base_img, 0.4, 0, base_img)
        cv2.putText(
            base_img, str(idx), (x0 + 4, y0 + 16),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA,
        )

    selected: set[int] = set()
    display = base_img.copy()
    window_name = "Select Initial States"

    def _redraw():
        nonlocal display
        display = base_img.copy()
        for idx in selected:
            r, c = divmod(idx, cols)
            x0, y0 = c * thumb_w, r * thumb_h
            # Semi-transparent green tint
            overlay = display.copy()
            cv2.rectangle(overlay, (x0, y0), (x0 + thumb_w, y0 + thumb_h), (0, 200, 0), -1)
            cv2.addWeighted(overlay, 0.25, display, 0.75, 0, display)
            # Bright green border
            cv2.rectangle(display, (x0 + 1, y0 + 1),
                          (x0 + thumb_w - 2, y0 + thumb_h - 2), (0, 255, 0), 3)
        # Status bar
        remaining = num_eval_episodes - len(selected)
        status = (f"Selected {len(selected)}/{num_eval_episodes}  |  "
                  f"{'ENTER to confirm' if remaining <= 0 else f'{remaining} more needed'}  |  "
                  f"Click to toggle  |  ESC to cancel")
        cv2.rectangle(display, (0, grid_h - 30), (grid_w, grid_h), (40, 40, 40), -1)
        cv2.putText(display, status, (8, grid_h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.imshow(window_name, display)

    def _on_mouse(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        col = x // thumb_w
        row = y // thumb_h
        idx = row * cols + col
        if idx >= n:
            return
        if not list_of_contours[idx]:
            return
        if idx in selected:
            selected.discard(idx)
        elif len(selected) < num_eval_episodes:
            selected.add(idx)
        _redraw()

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, min(grid_w, 1200), min(grid_h + 30, 900))
    cv2.setMouseCallback(window_name, _on_mouse)
    _redraw()

    print(f"[INFO] Select {num_eval_episodes} episodes from the grid, then press ENTER.")

    while True:
        key = cv2.waitKey(50) & 0xFF
        if key in (13, 10):  # ENTER
            if len(selected) == 0:
                continue
            break
        if key == 27:  # ESC
            selected.clear()
            break

    cv2.destroyWindow(window_name)
    cv2.waitKey(1)

    selected_indices = sorted(selected)
    selected_contours = [list_of_contours[i] for i in selected_indices]
    print(f"[INFO] Selected {len(selected_contours)} episodes: {selected_indices}")
    return selected_contours, selected_indices


def save_selection_grid(
    initial_states_dir,
    object_name: str,
    list_of_contours: list,
    selected_indices: list,
    output_path,
) -> None:
    """Save all_episodes_grid.png with user-selected cells highlighted in green."""
    if initial_states_dir is None:
        initial_states_dir = Path(__file__).parent / "initial_states"
    grid_path = Path(initial_states_dir) / object_name / "all_episodes_grid.png"
    if not grid_path.exists():
        print(f"[WARN] Grid image not found, skipping selection grid: {grid_path}")
        return
    grid_img = cv2.imread(str(grid_path))
    if grid_img is None:
        print(f"[WARN] Failed to read grid image: {grid_path}")
        return

    n = len(list_of_contours)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    grid_h, grid_w = grid_img.shape[:2]
    thumb_w = grid_w // cols
    thumb_h = grid_h // rows

    result = grid_img.copy()
    for idx in range(n):
        r, c = divmod(idx, cols)
        x0, y0 = c * thumb_w, r * thumb_h
        if not list_of_contours[idx]:
            overlay = result.copy()
            cv2.rectangle(overlay, (x0, y0), (x0 + thumb_w, y0 + thumb_h), (80, 80, 80), -1)
            cv2.addWeighted(overlay, 0.6, result, 0.4, 0, result)
        cv2.putText(
            result, str(idx), (x0 + 4, y0 + 16),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA,
        )
    for idx in selected_indices:
        r, c = divmod(idx, cols)
        x0, y0 = c * thumb_w, r * thumb_h
        overlay = result.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + thumb_w, y0 + thumb_h), (0, 200, 0), -1)
        cv2.addWeighted(overlay, 0.25, result, 0.75, 0, result)
        cv2.rectangle(result, (x0 + 1, y0 + 1),
                      (x0 + thumb_w - 2, y0 + thumb_h - 2), (0, 255, 0), 3)

    label = f"Selected training eps: {selected_indices}"
    cv2.rectangle(result, (0, grid_h - 30), (grid_w, grid_h), (40, 40, 40), -1)
    cv2.putText(result, label[:100], (8, grid_h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), result)
    print(f"[INFO] Saved selection grid → {output_path}")


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
        default="Pick up the mug",
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
        default="pointclouds/xarm7_black.npz",
        help="Path to Gaussian Splatting scene file for composite rendering"
    )
    parser.add_argument(
        "--color-calibrate", action="store_true",default=True,
        help="Path to color calibration YAML file (optional)"
    )
    obs_mode = parser.add_mutually_exclusive_group()
    obs_mode.add_argument(
        "--obs",
        action="store_true",
        help="Use real-world dataset images as policy input (instead of MuJoCo/composite rendering)"
    )
    obs_mode.add_argument(
        "--obs-eval",
        action="store_true",
        help="Like --obs but use episode 0 from the eval dataset (default path: data_eval; see --obs-eval-path)"
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="data",
        help="Path to dataset directory for --obs and for Real display when not using --obs-eval"
    )
    parser.add_argument(
        "--obs-eval-path",
        type=str,
        default="data_eval",
        help="Dataset path for --obs-eval (default: data_eval)"
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=None,
        help="Dataset root (optional, for Hub datasets)"
    )
    parser.add_argument(
        "--episode",
        type=int,
        default=1,
        help="Episode index for real observations (default: 0)"
    )
    parser.add_argument(
        "--num_eval_episodes",
        type=int,
        default=10,
        help="Number of evaluation episodes to run (default: 10)"
    )
    parser.add_argument(
        "--select",
        action="store_true",
        help="Load initial-state contour overlays generated by initial_states_overlay.py"
    )
    parser.add_argument(
        "--initial-states-dir",
        type=str,
        default=None,
        help="Path to initial_states directory (default: visual_match/initial_states/)"
    )
    parser.add_argument(
        "--object-name",
        type=str,
        default="mug",
        help="Object name matching the segmentation prompt (default: mug)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data_sim_eval",
        help="Directory to save sim evaluation data (videos, states, grid image). Default: data_sim_eval"
    )
    parser.add_argument(
        "--gemini",
        action="store_true",
        help="Use Gemini few-shot sim→real translation instead of color_mapping.yaml calibration. "
             "Only queries the API when a new policy prediction is needed (every n_action_steps frames). "
             "Uses 1 example pair for stationary, 3 for wrist.",
    )

    args = parser.parse_args()
    num_eval_episodes = args.num_eval_episodes

    # Create output directory for sim evaluation data
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Sim eval output directory: {output_dir.resolve()}")

    # Load initial-state contours and open selection UI when --select is used
    list_of_contours = None
    selected_contours = None
    selected_episode_indices = None
    if args.select:
        list_of_contours = load_initial_state_contours(
            initial_states_dir=args.initial_states_dir,
            object_name=args.object_name,
        )
        selected_contours, selected_episode_indices = select_contours_ui(
            list_of_contours,
            num_eval_episodes,
            initial_states_dir=args.initial_states_dir,
            object_name=args.object_name,
        )
        if not selected_contours:
            print("[INFO] No episodes selected, exiting.")
            sys.exit(0)
        if len(selected_contours) != num_eval_episodes:
            print(f"[ERROR] Selected {len(selected_contours)} contours but "
                  f"num_eval_episodes={num_eval_episodes}. Must be equal.")
            sys.exit(1)
        save_selection_grid(
            initial_states_dir=args.initial_states_dir,
            object_name=args.object_name,
            list_of_contours=list_of_contours,
            selected_indices=selected_episode_indices,
            output_path=output_dir / "selected_states_grid.png",
        )

    if args.obs:
        print("[INFO] --obs: using real-world dataset images as policy input")
    elif args.obs_eval:
        print(
            f"[INFO] --obs-eval: using episode 0 images from {args.obs_eval_path!r} as policy input"
        )

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

    # Mug freejoint address (for in-memory position adjustment during warmup)
    mug_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "mug_joint")
    mug_qpos_addr = model.jnt_qposadr[mug_joint_id] if mug_joint_id >= 0 else -1
    if mug_qpos_addr >= 0:
        print(f"[INFO] Mug freejoint found (qpos addr={mug_qpos_addr})")
    else:
        print("[WARN] mug_joint not found – mug position adjustment disabled")

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
            scene_data, scene_depth_data, _ = load_scene_data(
                args.scene_path, w2c_init, camera_intrinsics["stationary"]
            )
            color_calib = None
            if args.color_calibrate and not args.gemini:
                default_calib = Path(__file__).parent.parent / "calibration_pairs_stationary" / "calibrated" / "color_mapping.yaml"
                try:
                    color_calib = load_color_mapping(default_calib)
                    print(f"[INFO] Loaded color calibration from: {default_calib}")
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

    # Gemini sim→real translator (replaces color calibration when --gemini)
    gemini_translator = None
    if args.gemini:
        from query_gemini import GeminiTranslator
        print("[INFO] Initializing Gemini translator (examples = query_gemini.EXAMPLE_FRAME_INDICES)...")
        gemini_translator = GeminiTranslator(
            stationary_pairs=1,
            wrist_pairs=3,
        )
        n_act = getattr(policy.config, 'n_action_steps', None)
        if n_act and n_act > 1:
            est_calls = args.max_steps // n_act + 1
            print(f"[INFO] Gemini ready: 2 parallel API calls per query, every ~{n_act} steps "
                  f"(~{est_calls} query rounds/episode).")
        else:
            print("[INFO] Gemini ready: 2 parallel API calls each prediction step.")

    # Load real-world dataset frames for display (and for policy when --obs / --obs-eval)
    obs_frames = None
    if args.obs_eval:
        obs_load_path = args.obs_eval_path
        obs_load_episode = 0
    else:
        obs_load_path = args.dataset_path
        obs_load_episode = args.episode
    try:
        episode_data = load_episode(
            obs_load_path, obs_load_episode, dataset_root=args.dataset_root
        )
        obs_frames = load_dataset_frames(episode_data)
        num_obs_frames = max(len(obs_frames.get(k, [])) for k in CAMERA_CONFIG) or 1
        print(
            f"[INFO] Loaded real dataset images: {num_obs_frames} frames from "
            f"{obs_load_path!r} episode {obs_load_episode}"
        )
    except Exception as e:
        if args.obs or args.obs_eval:
            flag = "--obs-eval" if args.obs_eval else "--obs"
            print(f"[ERROR] Failed to load dataset for {flag}: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
        print(f"[WARN] Could not load dataset for Real windows display: {e}")

    # Initialize keyboard listener (ALOHA-style state machine)
    listener, events = init_keyboard_listener()

    if not args.headless:
        WINDOW_W, WINDOW_H = 400, 300
        X_START, Y_START = 50, 30
        X_STEP, Y_STEP = 410, 340
        cam_keys = list(CAMERA_CONFIG.keys())
        for i, cam_key in enumerate(cam_keys):
            obs_key = f"observation.images.{CAMERA_CONFIG[cam_key]['dataset_cam']}"
            cam_short = obs_key.split('.')[-1]
            win_real = f"Real: {cam_short}"
            win_comp = f"Composite: {cam_short}"
            cv2.namedWindow(win_real, cv2.WINDOW_NORMAL)
            cv2.namedWindow(win_comp, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(win_real, WINDOW_W, WINDOW_H)
            cv2.resizeWindow(win_comp, WINDOW_W, WINDOW_H)
            cv2.moveWindow(win_real, X_START + i * X_STEP, Y_START)
            cv2.moveWindow(win_comp, X_START + i * X_STEP, Y_START + Y_STEP)
            if args.gemini:
                win_gem = f"Gemini: {cam_short}"
                cv2.namedWindow(win_gem, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(win_gem, WINDOW_W, WINDOW_H)
                cv2.moveWindow(win_gem, X_START + i * X_STEP, Y_START + 2 * Y_STEP)

    _obs_tag = ""
    if args.obs:
        _obs_tag = " [real obs]"
    elif args.obs_eval:
        _obs_tag = " [obs-eval]"
    print(
        f"[INFO] Starting policy evaluation ({num_eval_episodes} episodes, max {args.max_steps} steps each)"
        f"{_obs_tag}"
    )

    step_dt = 1.0 / args.fps

    viewer = None
    viewer_ctx = None
    # Skip mujoco 3D viewer when using EGL (e.g. SSH X11 forwarding) - GLFW/GLX fails
    use_viewer = (
        not args.headless
        and _HAS_DISPLAY
        and os.environ.get("MUJOCO_GL") != "egl"
    )
    if use_viewer:
        try:
            viewer_ctx = mujoco.viewer.launch_passive(model, data)
            viewer = viewer_ctx.__enter__()
        except Exception:
            viewer = None
            viewer_ctx = None

    saved_episodes = []
    completed_episodes = 0
    episode_idx = 0

    def _reset_sim():
        """Reset simulation to home keyframe and re-apply camera calibration."""
        try:
            home_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
            mujoco.mj_resetDataKeyframe(model, data, home_id)
        except Exception:
            mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)
        for cam_cfg in CAMERA_CONFIG.values():
            if cam_cfg["config"].get("type", "stationary") == "stationary":
                set_mujoco_camera_from_config(data, model, cam_cfg["mujoco_cam"], cam_cfg["config"])

    def _render_current_view(warmup_contour=None, alpha=0.4):
        """Render and display cameras for the current sim state (used during warmup).

        If *warmup_contour* is provided, it is overlaid on the stationary camera
        image (cam_high) with the given alpha so the user can visually match the
        target mug placement.
        """
        for cam_cfg in CAMERA_CONFIG.values():
            if cam_cfg["config"].get("type", "stationary") == "stationary":
                set_mujoco_camera_from_config(data, model, cam_cfg["mujoco_cam"], cam_cfg["config"])
        if not args.headless:
            composite_obs = build_observation_from_mujoco(
                model, data, renderer,
                seg_renderer=seg_renderer, robot_geom_ids=robot_geom_ids,
                gaussian_data=gaussian_data, obs_frames=None, frame_idx=0,
            )
            if warmup_contour:
                cam_high_key = f"observation.images.{CAMERA_CONFIG['stationary']['dataset_cam']}"
                if cam_high_key in composite_obs:
                    img = composite_obs[cam_high_key]
                    overlay = img.copy()
                    cv2.drawContours(overlay, warmup_contour, -1, (0, 255, 0), -1)
                    cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0, img)
                    cv2.drawContours(img, warmup_contour, -1, (0, 255, 0), 2)
                    composite_obs[cam_high_key] = img
            display_camera_images(composite_obs, policy_config=policy.config, window_name_prefix="Composite")
            if obs_frames is not None:
                real_obs = build_observation_from_mujoco(
                    model, data, renderer,
                    seg_renderer=seg_renderer, robot_geom_ids=robot_geom_ids,
                    gaussian_data=gaussian_data, obs_frames=obs_frames, frame_idx=0,
                )
                display_camera_images(real_obs, policy_config=policy.config, window_name_prefix="Real")
        if viewer is not None:
            viewer.sync()

    try:
        while completed_episodes < num_eval_episodes and not events["stop_recording"]:
            # ── WARMUP: reset sim and wait for RIGHT arrow to begin episode ──
            _reset_sim()
            policy.reset()

            # Pick the contour for this evaluation episode (sequential)
            warmup_contour = None
            if selected_contours is not None and completed_episodes < len(selected_contours):
                warmup_contour = selected_contours[completed_episodes]

            events["right_arrow"] = False
            events["left_arrow"] = False
            events["rerecord_episode"] = False
            events["exit_early"] = False

            can_adjust_mug = warmup_contour and mug_qpos_addr >= 0

            print(f"\n{'='*60}")
            print(f"  WARMUP - Episode {completed_episodes + 1}/{num_eval_episodes}  "
                  f"(eval #{episode_idx})")
            if warmup_contour:
                src_ep = selected_episode_indices[completed_episodes]
                print(f"  Contour: selected_contours[{completed_episodes}]  "
                      f"(training ep {src_ep})")
                if can_adjust_mug:
                    print(f"  Arrows: move mug XY | w/s: Z | -/+: step size")
                print(f"  ENTER: start evaluation | ESC: quit")
            elif listener is not None:
                print(f"  Press RIGHT to start evaluation, ESC to quit")
            print(f"{'='*60}")

            if warmup_contour:
                # ── cv2-based warmup with contour overlay ──
                mug_step = MUG_STEP_INIT_M
                if can_adjust_mug:
                    mug_pos = data.qpos[mug_qpos_addr:mug_qpos_addr + 3].copy()

                _render_current_view(warmup_contour=warmup_contour)

                while not events["stop_recording"]:
                    key = cv2.waitKeyEx(50)
                    if key < 0:
                        continue

                    if key in (13, 10):  # ENTER → confirm & start
                        break
                    if key == 27:  # ESC → quit
                        events["stop_recording"] = True
                        break

                    if not can_adjust_mug:
                        continue

                    moved = False
                    if key in _KEY_LEFT:
                        mug_pos[0] -= mug_step; moved = True
                    elif key in _KEY_RIGHT:
                        mug_pos[0] += mug_step; moved = True
                    elif key in _KEY_UP:
                        mug_pos[1] += mug_step; moved = True
                    elif key in _KEY_DOWN:
                        mug_pos[1] -= mug_step; moved = True
                    elif key == ord('w'):
                        mug_pos[2] += mug_step; moved = True
                    elif key == ord('s'):
                        mug_pos[2] -= mug_step; moved = True

                    if key in (ord('-'), ord('_')):
                        mug_step /= 2.0
                        print(f"[INFO] Step: {mug_step*1000:.2f} mm")
                    elif key in (ord('+'), ord('=')):
                        mug_step = min(mug_step * 2.0, MUG_STEP_INIT_M)
                        print(f"[INFO] Step: {mug_step*1000:.2f} mm")

                    if moved:
                        data.qpos[mug_qpos_addr:mug_qpos_addr + 3] = mug_pos
                        mujoco.mj_forward(model, data)
                        _render_current_view(warmup_contour=warmup_contour)
                        print(f"[INFO] Mug pos: [{mug_pos[0]:.4f}, {mug_pos[1]:.4f}, {mug_pos[2]:.4f}]  "
                              f"step={mug_step*1000:.2f}mm")

                # Clear stale pynput events from arrow keys
                events["right_arrow"] = False
                events["left_arrow"] = False
                events["rerecord_episode"] = False
                events["exit_early"] = False

                if events["stop_recording"]:
                    break

            elif listener is not None:
                # ── pynput-based warmup (no contour selection) ──
                while (not events["right_arrow"]
                       and not events["stop_recording"]):
                    step_start = time.perf_counter()
                    _render_current_view()
                    elapsed = time.perf_counter() - step_start
                    sleep_time = max(0, step_dt - elapsed)
                    if sleep_time > 0:
                        time.sleep(sleep_time)

                if events["stop_recording"]:
                    break

            # ── Transition to RECORDING ──
            events["right_arrow"] = False
            events["left_arrow"] = False
            events["rerecord_episode"] = False
            events["exit_early"] = False

            print(f"\n{'='*60}")
            print(f"  RECORDING episode {episode_idx}")
            if listener is not None:
                print(f"  RIGHT=save | LEFT=discard | ESC=quit")
            else:
                print(f"  (headless: auto-save after {args.max_steps} steps)")
            print(f"{'='*60}")

            episode_actions = []
            episode_states = []
            episode_frames = {cam_key: [] for cam_key in CAMERA_CONFIG}
            last_gemini_display_obs = None
            step = 0
            episode_discarded = False

            while step < args.max_steps:
                if events["stop_recording"]:
                    break
                if events["left_arrow"]:
                    episode_discarded = True
                    break
                if events["right_arrow"]:
                    break

                step_start = time.perf_counter()

                for cam_cfg in CAMERA_CONFIG.values():
                    if cam_cfg["config"].get("type", "stationary") == "stationary":
                        set_mujoco_camera_from_config(data, model, cam_cfg["mujoco_cam"], cam_cfg["config"])

                # ACT action chunking: the policy only needs a new observation
                # when its action queue is depleted. Skip expensive Gemini API
                # calls on intermediate steps where we just pop from the queue.
                needs_prediction = (
                    not hasattr(policy, '_action_queue')
                    or len(policy._action_queue) == 0
                )

                real_obs = build_observation_from_mujoco(
                    model, data, renderer,
                    seg_renderer=seg_renderer,
                    robot_geom_ids=robot_geom_ids,
                    gaussian_data=gaussian_data,
                    obs_frames=obs_frames,
                    frame_idx=step,
                )

                # Always build composite without Gemini (fast, for display + video)
                composite_obs = build_observation_from_mujoco(
                    model, data, renderer,
                    seg_renderer=seg_renderer,
                    robot_geom_ids=robot_geom_ids,
                    gaussian_data=gaussian_data,
                    obs_frames=None,
                    frame_idx=step,
                )

                # Policy input: real obs, Gemini-translated composite (parallel API, same
                # few-shot examples as query_gemini), or raw composite if no Gemini / queue step.
                if args.obs or args.obs_eval:
                    observation = real_obs
                elif gemini_translator is not None and needs_prediction:
                    observation = apply_gemini_parallel(gemini_translator, composite_obs)
                    last_gemini_display_obs = {
                        k: observation[k].copy()
                        for k in observation
                        if "image" in k.lower()
                        and isinstance(observation[k], np.ndarray)
                    }
                    n_act = getattr(policy.config, 'n_action_steps', '?')
                    print(f"  [Gemini] Step {step}: parallel API calls on live composite "
                          f"(next query in ~{n_act} steps)")
                else:
                    observation = composite_obs

                if not args.headless:
                    display_camera_images(real_obs, policy_config=policy.config, window_name_prefix="Real")
                    display_camera_images(composite_obs, policy_config=policy.config, window_name_prefix="Composite")
                    if gemini_translator is not None and last_gemini_display_obs:
                        display_camera_images(
                            last_gemini_display_obs,
                            policy_config=policy.config,
                            window_name_prefix="Gemini",
                        )

                if hasattr(policy.config, 'language_features') and policy.config.language_features:
                    observation["observation.language"] = args.prompt

                # Record composite frames (fresh every step, shows actual sim state)
                for _cam_key, _cam_cfg in CAMERA_CONFIG.items():
                    _obs_key = f"observation.images.{_cam_cfg['dataset_cam']}"
                    if _obs_key in composite_obs:
                        episode_frames[_cam_key].append(composite_obs[_obs_key].copy())

                episode_states.append(composite_obs[OBS_STATE].copy()
                                      if isinstance(composite_obs[OBS_STATE], np.ndarray)
                                      else composite_obs[OBS_STATE])

                with torch.inference_mode():
                    # print(observation["observation.state"])
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
                print(action)
                episode_actions.append(action.cpu().numpy().copy())

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
                    print(f"[INFO] Episode {episode_idx} - Step {step}/{args.max_steps}")

            # ── Post-episode: save or discard ──
            if events["stop_recording"]:
                print(f"\n[INFO] ESC pressed, stopping evaluation")
                break

            if episode_discarded:
                events["left_arrow"] = False
                events["rerecord_episode"] = False
                events["exit_early"] = False
                print(f">>> Episode {episode_idx} DISCARDED ({step} steps)")
            else:
                events["right_arrow"] = False
                events["exit_early"] = False
                reason = "max steps reached" if step >= args.max_steps else "RIGHT pressed"
                saved_episodes.append({
                    "episode": episode_idx,
                    "steps": step,
                    "actions": episode_actions,
                    "states": episode_states,
                })
                # Save episode data (states, actions, camera videos) to output directory
                ep_dir = output_dir / f"episode_{completed_episodes:03d}"
                ep_dir.mkdir(parents=True, exist_ok=True)
                np.save(str(ep_dir / "states.npy"), np.array(episode_states))
                np.save(str(ep_dir / "actions.npy"), np.array(episode_actions))
                for _cam_key, _frames in episode_frames.items():
                    if not _frames:
                        continue
                    _dataset_cam = CAMERA_CONFIG[_cam_key]["dataset_cam"]
                    _video_path = ep_dir / f"{_dataset_cam}.mp4"
                    _h, _w = _frames[0].shape[:2]
                    _fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    _writer = cv2.VideoWriter(str(_video_path), _fourcc, args.fps, (_w, _h))
                    for _frame in _frames:
                        _writer.write(cv2.cvtColor(_frame, cv2.COLOR_RGB2BGR))
                    _writer.release()
                    print(f"[INFO] Saved {len(_frames)}-frame video: {_video_path}")
                print(f"[INFO] Episode data saved → {ep_dir}")
                completed_episodes += 1
                print(f">>> Episode {episode_idx} SAVED - {reason} "
                      f"({step} steps, {completed_episodes}/{num_eval_episodes} done)")

            episode_idx += 1

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
    finally:
        if listener is not None:
            listener.stop()
        if viewer_ctx is not None:
            try:
                viewer_ctx.__exit__(None, None, None)
            except Exception:
                pass
        if not args.headless:
            cv2.destroyAllWindows()
        print(f"[INFO] Evaluation finished: {completed_episodes}/{num_eval_episodes} episodes saved")


if __name__ == "__main__":
    main()
