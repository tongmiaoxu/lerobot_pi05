#!/usr/bin/env python3
"""
Deploy ACT policy in MuJoCo xArm simulation.

This script loads an ACT policy checkpoint and runs it in MuJoCo xArm simulation.
Uses 2 cameras: wrist and stationary, both with composite rendering (Gaussian Splatting
background + MuJoCo robot foreground). Refer to compare_recorded_vs_mujoco for the
xArm observation.state format (8-dim): [joint1..7 in degrees, gripper in mm (0=closed, 800=open)]

composite rendering pipeline.

Usage:
    # Uses the hardcoded default task profile for scene/object warmup defaults:
    python visual_match/deploy_act_policy_mujoco.py

    # Same as --obs but fixed to episode 0 of the real eval dataset (default under data_real/):
    python visual_match/deploy_act_policy_mujoco.py --obs-eval

    # Faster sim: skip replay load and "Real:" windows unless --obs / --obs-eval (those keep Real):
    python visual_match/deploy_act_policy_mujoco.py --no_obs --fps 30
"""

import sys
import os
import re
import math
import argparse
import shutil
import subprocess
from pathlib import Path
import time
import json
import threading

# Add src to path for lerobot imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))

###############################################
_DEFAULT_RECORD_POLICY_CHECKPOINT = "outputs/act_place_mug/checkpoints/009000/pretrained_model"
_DEFAULT_RECORD_TASK_ID = "place_mug"
###############################################
_NUM_EPISODES = 10
_PICK_SHOE_GRIPPER_OBS_OFFSET_MM = 244.68428556068324
_PICK_SHOE_GRIPPER_OFFSET_THRESHOLD_MM = 750.0
_GRIPPER_CONTACT_GEOM_PREFIXES = ("left_finger_pad_", "right_finger_pad_")
_PICK_SHOE_CONTACT_GEOM_NAMES = ("right_shoe_col",)
_MAX_PREDICTION_EVENTS_PER_TRAJECTORY_BY_TASK = {
    "hang_mug": 10,
    "place_mug": 8,
    "pick_shoe": 10,
}


def _max_prediction_events_per_trajectory(task_id: str) -> int:
    return _MAX_PREDICTION_EVENTS_PER_TRAJECTORY_BY_TASK.get(task_id, 6)

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
from PIL import Image

if _HAS_DISPLAY:
    try:
        import mujoco.viewer
        _HAS_MJ_VIEWER = True
    except ImportError:
        _HAS_MJ_VIEWER = False
else:
    _HAS_MJ_VIEWER = False

from lerobot.policies.factory import get_policy_class
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.control_utils import predict_action, init_keyboard_listener
from lerobot.utils.utils import get_safe_torch_device
from lerobot.utils.constants import ACTION, OBS_STATE
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
from object_pose_auto_align import ObjectPoseAlignConfig, auto_align_object_poses
from lerobot.datasets.utils import copy_observation_frame_with_resized_images
from lerobot.datasets.video_utils import decode_video_frames
from lerobot.tasks import get_task_profile, get_task_profiles, resolve_task_scene_xml

# Arrow key codes for cv2.waitKeyEx (platform-dependent)
_KEY_LEFT  = (65361, 81, 2)
_KEY_RIGHT = (65363, 83, 3)
_KEY_UP    = (65362, 82, 0)
_KEY_DOWN  = (65364, 84, 1)
MUG_STEP_INIT_M = 0.005  # 5 mm initial step for mug adjustment
MUG_ROT_STEP_RAD = np.deg2rad(5.0)
_DEFAULT_GPT_IMAGE_MODEL = "gpt-image-2"
_DEFAULT_GPT_IMAGE_PROMPT =  "Style transfer only. Freeze all object positions, shapes, and layout. Change only the visual style, texture, and color treatment."
# _DEFAULT_GPT_IMAGE_PROMPT =  "DeTransfer style while preserving geometry"
_DEFAULT_GPT_IMAGE_QUALITY = "high"
_DEFAULT_GPT_IMAGE_SIZE = "auto"
_DEFAULT_GPT_IMAGE_MAX_SIDE = 1024



def _extract_checkpoint_name(policy_path: str | Path | None) -> str | None:
    if policy_path is None:
        return None
    parts = Path(policy_path).parts
    if "checkpoints" in parts:
        idx = parts.index("checkpoints")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return None

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


def _load_rgb_pil(path: str | Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def _default_gpt_style_image_path(task_profile, cam_key: str) -> Path:
    return Path(task_profile.dataset_root_480640) / "real_captures" / cam_key / "frame_0000.png"


def _get_sorted_frame_paths(folder: Path) -> list:
    """Return sorted list of frame_*.png paths under *folder*."""
    return sorted(folder.glob("frame_*.png"))


def _update_gpt_style_references(
    translators: dict,
    style_dirs: dict,
    frame_cache: dict,
    step: int,
) -> None:
    """Swap each GPTImageTranslator's style reference to the frame matching *step*.

    If *step* exceeds the number of available frames the last frame is used.
    Results are cached so the directory is only scanned once per camera.
    """
    for cam_key, translator in translators.items():
        folder = style_dirs.get(cam_key)
        if folder is None:
            continue
        if cam_key not in frame_cache:
            frame_cache[cam_key] = _get_sorted_frame_paths(Path(folder))
        frames = frame_cache[cam_key]
        if not frames:
            continue
        idx = min(step, len(frames) - 1)
        translator._style_references = [_load_rgb_pil(frames[idx])]
        
def apply_gpt_per_camera_parallel(translators: dict[str, object], observation: dict, frame_idx: int = 0) -> dict:
    """Run GPT image translation per logical camera in parallel."""
    out = dict(observation)
    lock = threading.Lock()
    errs: list[tuple[str, Exception]] = []
    requested_cams: list[str] = []

    def work(cam_key: str, obs_key: str, img: np.ndarray):
        print(f"  [DEBUG] frame {frame_idx} | {cam_key} → generating...", flush=True)
        try:
            translator = translators.get(cam_key)
            if translator is None:
                print(f"  [DEBUG] frame {frame_idx} | {cam_key} → no translator found, skipping.", flush=True)
                return
            translated = translator.translate(np.ascontiguousarray(img))
            with lock:
                out[obs_key] = translated
            print(f"  [DEBUG] frame {frame_idx} | {cam_key} → done ✓", flush=True)
        except Exception as e:
            print(f"  [ERROR] frame {frame_idx} | {cam_key} → FAILED: {type(e).__name__}: {e}", flush=True)
            with lock:
                errs.append((cam_key, e))

    threads = []
    for cam_key, cam_cfg in CAMERA_CONFIG.items():
        obs_key = f"observation.images.{cam_cfg['dataset_cam']}"
        if obs_key not in out:
            print(f"  [DEBUG] frame {frame_idx} | {cam_key} → obs_key '{obs_key}' not in observation, skipping.", flush=True)
            continue
        img = out[obs_key]
        if not isinstance(img, np.ndarray):
            print(f"  [DEBUG] frame {frame_idx} | {cam_key} → image is {type(img).__name__}, not ndarray, skipping.", flush=True)
            continue
        requested_cams.append(cam_key)
        t = threading.Thread(target=work, args=(cam_key, obs_key, img))
        threads.append(t)
        t.start()

    if requested_cams:
        cam_list = ", ".join(requested_cams)
        print(f"[INFO] frame {frame_idx} | {_DEFAULT_GPT_IMAGE_MODEL} | spawned {len(requested_cams)} thread(s): ({cam_list})", flush=True)
    else:
        print(f"[WARN] frame {frame_idx} | no cameras queued — nothing to translate.", flush=True)

    for t in threads:
        t.join()

    n_ok = len(requested_cams) - len(errs)
    if errs:
        print(f"[WARN] frame {frame_idx} | {len(errs)}/{len(requested_cams)} camera(s) failed:", flush=True)
        for cam_key, e in errs:
            print(f"  [WARN] frame {frame_idx} | {cam_key} → {type(e).__name__}: {e}", flush=True)

    if requested_cams:
        print(f"[INFO] frame {frame_idx} | {_DEFAULT_GPT_IMAGE_MODEL} | done — {n_ok}/{len(requested_cams)} succeeded.", flush=True)

    return out


def _resolve_path_under_project(path: str | Path | None, project_root: Path) -> str | None:
    if path is None or path == "":
        return None
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = (project_root / p).resolve()
    return str(p)


def _next_saved_episode_index(output_dir: Path) -> int:
    """Return the next free episode_NNN index inside an evaluation output directory."""
    max_episode_idx = -1
    for child in output_dir.iterdir():
        if not child.is_dir():
            continue
        match = re.fullmatch(r"episode_(\d+)", child.name)
        if match is None:
            continue
        max_episode_idx = max(max_episode_idx, int(match.group(1)))
    return max_episode_idx + 1


def _reserve_episode_output_dir(output_dir: Path) -> tuple[int, Path]:
    """Atomically reserve a unique episode_NNN directory for the next trajectory."""
    episode_idx = _next_saved_episode_index(output_dir)
    while True:
        ep_dir = output_dir / f"episode_{episode_idx:03d}"
        try:
            ep_dir.mkdir(parents=False, exist_ok=False)
            return episode_idx, ep_dir
        except FileExistsError:
            episode_idx += 1


def _delete_episode_output_dir(ep_dir: Path | None) -> None:
    """Remove a reserved episode directory and anything saved inside it."""
    if ep_dir is None or not ep_dir.exists():
        return
    shutil.rmtree(ep_dir, ignore_errors=True)


def apply_turbo_per_camera(translators: dict[str, object], observation: dict) -> dict:
    """Run pix2pix-turbo per logical camera (stationary / wrist may use different checkpoints)."""
    out = dict(observation)
    for cam_key, cam_cfg in CAMERA_CONFIG.items():
        tr = translators.get(cam_key)
        if tr is None:
            continue
        obs_key = f"observation.images.{cam_cfg['dataset_cam']}"
        if obs_key not in out:
            continue
        img = out[obs_key]
        if not isinstance(img, np.ndarray):
            continue
        out[obs_key] = tr.translate(np.ascontiguousarray(img))
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
                # print(f"[WARN] Video not found for {cam_key}: {video_path}")
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
            # print(f"[INFO] Loaded {len(frames_list)} real frames for {cam_key}")
        except Exception as e:
            # print(f"[WARN] Failed to load {cam_key} video: {e}")
            cam_frames[cam_key] = []
    return cam_frames


def _format_window_title(window_name_prefix: str, window_name: str, episode_idx: int | None = None) -> str:
    if episode_idx is None:
        return f"{window_name_prefix}: {window_name}"
    return f"{window_name_prefix} [{episode_idx:03d}]: {window_name}"


def display_camera_images(
    observation: dict,
    policy_config=None,
    window_name_prefix: str = "Camera",
    episode_idx: int | None = None,
):
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
        if hasattr(cv2, "setWindowTitle"):
            cv2.setWindowTitle(
                window_full_name,
                _format_window_title(window_name_prefix, window_name, episode_idx),
            )
    cv2.waitKey(1)


def _fit_rgb_image(img: np.ndarray | None, width: int, height: int) -> np.ndarray:
    """Resize an RGB image to fit inside a fixed tile while preserving aspect ratio."""
    canvas = np.full((height, width, 3), 18, dtype=np.uint8)
    if not isinstance(img, np.ndarray) or img.ndim != 3 or img.shape[2] != 3:
        return canvas
    src_h, src_w = img.shape[:2]
    if src_h <= 0 or src_w <= 0:
        return canvas
    scale = min(width / src_w, height / src_h)
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(img, (new_w, new_h), interpolation=interp)
    y0 = (height - new_h) // 2
    x0 = (width - new_w) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas


def _build_window_tile(title: str, img: np.ndarray | None, width: int, height: int) -> np.ndarray:
    """Create one labeled RGB tile matching an OpenCV display window."""
    title_h = 28
    tile = np.full((height, width, 3), 18, dtype=np.uint8)
    cv2.rectangle(tile, (0, 0), (width, title_h), (44, 44, 44), -1)
    cv2.putText(tile, title, (10, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235, 235, 235), 1, cv2.LINE_AA)
    tile[title_h:, :, :] = _fit_rgb_image(img, width, height - title_h)
    return tile


def build_combined_window_frame(
    row_specs: list[tuple[str, dict | None]],
    tile_width: int = 400,
    tile_height: int = 300,
) -> np.ndarray:
    """Build a single RGB frame containing the currently displayed camera windows."""
    row_frames = []
    for row_name, observation in row_specs:
        tiles = []
        for cam_key, cam_cfg in CAMERA_CONFIG.items():
            obs_key = f"observation.images.{cam_cfg['dataset_cam']}"
            cam_short = obs_key.split(".")[-1]
            img = None if observation is None else observation.get(obs_key)
            tiles.append(
                _build_window_tile(
                    f"{row_name}: {cam_short}",
                    img,
                    tile_width,
                    tile_height,
                )
            )
        row_frames.append(np.hstack(tiles))
    if not row_frames:
        return np.full((tile_height, tile_width, 3), 18, dtype=np.uint8)
    return np.vstack(row_frames)


def _alpha_blend_rgb_images(
    first: np.ndarray | None,
    second: np.ndarray | None,
    alpha: float = 0.5,
) -> np.ndarray | None:
    """Blend two RGB images at equal weight for side-by-side prediction comparison."""
    if not (
        isinstance(first, np.ndarray)
        and isinstance(second, np.ndarray)
        and first.ndim == 3
        and second.ndim == 3
        and first.shape[2] == 3
        and second.shape[2] == 3
    ):
        return None

    if first.shape[:2] != second.shape[:2]:
        second = cv2.resize(second, (first.shape[1], first.shape[0]), interpolation=cv2.INTER_LINEAR)
    return cv2.addWeighted(first, alpha, second, 1.0 - alpha, 0)


def _build_alpha_blended_observation(
    first_observation: dict | None,
    second_observation: dict | None,
    alpha: float = 0.5,
) -> dict:
    blended_observation = {}
    for _cam_key, cam_cfg in CAMERA_CONFIG.items():
        obs_key = f"observation.images.{cam_cfg['dataset_cam']}"
        first_img = None if first_observation is None else first_observation.get(obs_key)
        second_img = None if second_observation is None else second_observation.get(obs_key)
        blended_observation[obs_key] = _alpha_blend_rgb_images(first_img, second_img, alpha=alpha)
    return blended_observation


def build_prediction_event_panel(
    translated_row_name: str,
    composite_observation: dict | None,
    translated_observation: dict | None,
    tile_width: int = 400,
    tile_height: int = 300,
) -> np.ndarray:
    """Build a paired 3x2 RGB panel for one prediction event."""
    blended_observation = _build_alpha_blended_observation(
        composite_observation,
        translated_observation,
        alpha=0.5,
    )
    return build_combined_window_frame(
        [
            ("Composite", composite_observation),
            (translated_row_name, translated_observation),
            ("Blend", blended_observation),
        ],
        tile_width=tile_width,
        tile_height=tile_height,
    )


def load_policy(policy_path: str) -> tuple[PreTrainedPolicy, dict]:
    """Load ACT policy from checkpoint path."""
    # print(f"[INFO] Loading policy from: {policy_path}")

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


    policy.eval()
    # print(f"[INFO] Policy loaded: {policy_type}")
    return policy, config_dict


def policy_needs_prediction(policy: PreTrainedPolicy) -> bool:
    """Return whether the policy has exhausted its cached action chunk."""
    action_queue = getattr(policy, "_action_queue", None)
    if action_queue is not None:
        return len(action_queue) == 0

    queues = getattr(policy, "_queues", None)
    if isinstance(queues, dict):
        action_queue = queues.get(ACTION)
        if action_queue is not None:
            return len(action_queue) == 0

    return True


def pop_cached_policy_action(policy: PreTrainedPolicy, postprocessor) -> torch.Tensor | None:
    """Pop one cached raw policy action and run the normal postprocessor."""
    action_queue = getattr(policy, "_action_queue", None)
    if action_queue is None:
        queues = getattr(policy, "_queues", None)
        if isinstance(queues, dict):
            action_queue = queues.get(ACTION)
    if action_queue is None or len(action_queue) == 0:
        return None
    return postprocessor(action_queue.popleft())


def geom_ids_by_name_or_prefix(
    model: MjModel,
    *,
    names: tuple[str, ...] = (),
    prefixes: tuple[str, ...] = (),
) -> set[int]:
    geom_ids = set()
    names_set = set(names)
    for geom_id in range(model.ngeom):
        geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        if geom_name is None:
            continue
        if geom_name in names_set or any(geom_name.startswith(prefix) for prefix in prefixes):
            geom_ids.add(geom_id)
    return geom_ids


def has_contact_between_geom_sets(
    data: MjData,
    geom_ids_a: set[int],
    geom_ids_b: set[int],
) -> bool:
    if not geom_ids_a or not geom_ids_b:
        return False
    for contact_idx in range(data.ncon):
        contact = data.contact[contact_idx]
        geom1 = int(contact.geom1)
        geom2 = int(contact.geom2)
        if (geom1 in geom_ids_a and geom2 in geom_ids_b) or (geom2 in geom_ids_a and geom1 in geom_ids_b):
            return True
    return False


def should_apply_gripper_observation_offset(
    state: np.ndarray,
    data: MjData,
    *,
    mode: str,
    contact_geom_ids: tuple[set[int], set[int]] | None,
    threshold_mm: float,
) -> bool:
    if mode == "never":
        return False
    if mode == "always":
        return True
    if mode == "threshold":
        return state[7] <= threshold_mm
    if mode == "contact":
        if contact_geom_ids is None:
            return False
        gripper_geom_ids, shoe_geom_ids = contact_geom_ids
        return has_contact_between_geom_sets(data, gripper_geom_ids, shoe_geom_ids)
    raise ValueError(f"Unsupported gripper offset mode: {mode}")


def build_state_from_mujoco(
    model: MjModel,
    data: MjData,
    gripper_observation_offset_mm: float = 0.0,
    gripper_observation_offset_mode: str = "never",
    gripper_observation_contact_geom_ids: tuple[set[int], set[int]] | None = None,
    gripper_observation_threshold_mm: float = _PICK_SHOE_GRIPPER_OFFSET_THRESHOLD_MM,
) -> np.ndarray:
    """Build the LeRobot state vector from the current MuJoCo state."""
    ld_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "left_driver_joint")
    if ld_id < 0:
        raise RuntimeError("MuJoCo model has no joint 'left_driver_joint' (gripper driver).")
    g_adr = int(model.jnt_qposadr[ld_id])
    g_rad = (float(model.jnt_range[ld_id, 0]), float(model.jnt_range[ld_id, 1]))
    state = mujoco_qpos_to_lerobot_state(
        data.qpos, g_rad, gripper_qpos_adr=g_adr
    )
    if gripper_observation_offset_mm and should_apply_gripper_observation_offset(
        state,
        data,
        mode=gripper_observation_offset_mode,
        contact_geom_ids=gripper_observation_contact_geom_ids,
        threshold_mm=gripper_observation_threshold_mm,
    ):
        state[7] = np.clip(
            state[7] - gripper_observation_offset_mm,
            0.0,
            GRIPPER_OPEN_MM,
        )
    return state


def capture_mujoco_snapshot(data: MjData) -> dict[str, np.ndarray | float]:
    """Capture enough simulator state to render this frame later."""
    return {
        "qpos": data.qpos.copy(),
        "qvel": data.qvel.copy(),
        "ctrl": data.ctrl.copy(),
        "time": float(data.time),
    }


def build_observation_from_mujoco(model: MjModel, data: MjData, renderer: mujoco.Renderer,
                                  seg_renderer: mujoco.Renderer,
                                  robot_geom_ids: set,
                                  gaussian_data: dict | None,
                                  obs_frames: dict | None = None,
                                  frame_idx: int = 0,
                                  gripper_observation_offset_mm: float = 0.0,
                                  gripper_observation_offset_mode: str = "never",
                                  gripper_observation_contact_geom_ids: tuple[set[int], set[int]] | None = None,
                                  gripper_observation_threshold_mm: float = _PICK_SHOE_GRIPPER_OFFSET_THRESHOLD_MM) -> dict:
    """
    Build observation dict for xArm policy from MuJoCo state.
    Uses 2 cameras: cam_high (stationary) and cam_wrist, both with composite rendering.
    When obs_frames is provided (--obs mode), use real dataset images instead of rendered.
    """
    observation = {
        OBS_STATE: build_state_from_mujoco(
            model,
            data,
            gripper_observation_offset_mm=gripper_observation_offset_mm,
            gripper_observation_offset_mode=gripper_observation_offset_mode,
            gripper_observation_contact_geom_ids=gripper_observation_contact_geom_ids,
            gripper_observation_threshold_mm=gripper_observation_threshold_mm,
        )
    }

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
                cam_key, mujoco_cam, gaussian_data, camera_intrinsics.get(cam_key)
            )
        else:
            renderer.update_scene(data, camera=mujoco_cam)
            rgb_image = renderer.render()
        observation[obs_key] = rgb_image
    return observation


def render_composite_view(model: MjModel, data: MjData,
                          renderer: mujoco.Renderer, seg_renderer: mujoco.Renderer,
                          robot_geom_ids: set, cam_key: str, cam_name: str, gaussian_data: dict,
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
            color_calib = gaussian_data.get('color_calib_by_camera', {}).get(cam_key)
            if color_calib is None:
                color_calib = gaussian_data.get('color_calib')
            if color_calib is not None:
                composite = apply_color_transform(composite, color_calib)
            return composite
        except Exception as e:
            # print(f"[WARN] Gaussian rendering failed for {cam_name}: {e}")
            pass
    return fg_rgb


def _selection_object_names(object_name: str) -> list[str]:
    return [name.strip().replace("/", "_").replace(" ", "_") for name in object_name.split(",") if name.strip()]


def _selection_grid_path(initial_states_dir: str | Path, object_name: str) -> Path:
    names = _selection_object_names(object_name)
    base = Path(initial_states_dir)
    if len(names) > 1:
        combined = base / "all_episodes_grid.png"
        if combined.exists():
            return combined
    return base / names[0] / "all_episodes_grid.png"


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
        initial_states_dir: Dataset root produced by initial_states_overlay.py.
        object_name: One or more comma-separated object names matching the
            segmentation prompts (default ``"mug"``).

    Returns:
        list_of_contours – list (one entry per episode, sorted by episode
        index) of ``list[np.ndarray]`` contour arrays.
    """
    if initial_states_dir is None:
        raise ValueError("--initial-states-dir is required when --select is used.")

    object_dirs = []
    for name in _selection_object_names(object_name):
        masks_dir = Path(initial_states_dir) / name / "individual_masks"
        if not masks_dir.exists():
            raise FileNotFoundError(
                f"Mask directory not found: {masks_dir}\n"
                "Run initial_states_overlay.py first to generate masks."
            )
        object_dirs.append((name, masks_dir))

    first_mask_files = sorted(object_dirs[0][1].glob("ep_*_mask.png"))
    if not first_mask_files:
        raise FileNotFoundError(f"No mask files found in: {object_dirs[0][1]}")

    list_of_contours = []
    ep_ids = []
    for mask_path in first_mask_files:
        ep_id = int(mask_path.stem.replace("ep_", "").replace("_mask", ""))
        ep_ids.append(ep_id)
        combined_contours = []
        valid_all = True
        for _, masks_dir in object_dirs:
            object_mask_path = masks_dir / mask_path.name
            mask = cv2.imread(str(object_mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                valid_all = False
                break
            _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) == 0:
                valid_all = False
                break
            combined_contours.extend(list(contours))
        list_of_contours.append(combined_contours if valid_all else [])

    # print(f"[INFO] Loaded contours for {len(list_of_contours)} episodes "
          # f"(eps {min(ep_ids)}–{max(ep_ids)}) from {', '.join(str(d[1]) for d in object_dirs)}")
    return list_of_contours


def select_contours_auto(
    list_of_contours: list,
    num_eval_episodes: int,
) -> tuple[list, list[int]]:
    """
    Deterministically pick which training-episode contours to evaluate: the
    first ceil(num_eval_episodes/2) and last floor(num_eval_episodes/2)
    episode indices (by index order) among those with a valid contour.

    Deterministic (no UI, no randomness) so repeated runs — e.g. the three
    --all baselines — always evaluate the same set of episodes.

    Returns:
        (selected_contours, selected_indices), both sorted in ascending
        episode order, mirroring select_contours_ui's return shape.
    """
    valid_indices = [i for i, contours in enumerate(list_of_contours) if contours]
    if len(valid_indices) < num_eval_episodes:
        raise ValueError(
            f"Only {len(valid_indices)} episodes have a valid initial-state contour, "
            f"but --num_eval_episodes={num_eval_episodes} were requested."
        )

    first_half = (num_eval_episodes + 1) // 2
    second_half = num_eval_episodes - first_half
    selected_indices = valid_indices[:first_half]
    if second_half:
        selected_indices = selected_indices + valid_indices[-second_half:]
    selected_indices = sorted(set(selected_indices))

    selected_contours = [list_of_contours[i] for i in selected_indices]
    return selected_contours, selected_indices


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
        object_name: One or more comma-separated object names.

    Returns:
        (selected_contours, selected_indices) where *selected_contours* is a
        list of ``list[np.ndarray]`` contour arrays (one per chosen episode)
        and *selected_indices* is the corresponding episode-index list, both
        sorted in ascending episode order.
    """
    if initial_states_dir is None:
        raise ValueError("--initial-states-dir is required when --select is used.")
    grid_path = _selection_grid_path(initial_states_dir, object_name)
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

    # print(f"[INFO] Select {num_eval_episodes} episodes from the grid, then press ENTER.")

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
    # print(f"[INFO] Selected {len(selected_contours)} episodes: {selected_indices}")
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
        # print("[WARN] initial_states_dir is None, skipping selection grid save")
        return
    grid_path = _selection_grid_path(initial_states_dir, object_name)
    if not grid_path.exists():
        # print(f"[WARN] Grid image not found, skipping selection grid: {grid_path}")
        return
    grid_img = cv2.imread(str(grid_path))
    if grid_img is None:
        # print(f"[WARN] Failed to read grid image: {grid_path}")
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
    # print(f"[INFO] Saved selection grid → {output_path}")


_COMPOSITE_PREDICTION_EVENT_FILE_RE = re.compile(
    r"^composite_prediction_event_(?P<event>\d+)_step_(?P<step>\d+)\.png$"
)


def _last_composite_prediction_event_path(prediction_events_dir: Path) -> Path | None:
    """Return the composite_prediction_event_*.png with the highest (event_idx, step_idx)."""
    best_path = None
    best_key = None
    for path in prediction_events_dir.glob("composite_prediction_event_*_step_*.png"):
        match = _COMPOSITE_PREDICTION_EVENT_FILE_RE.match(path.name)
        if match is None:
            continue
        key = (int(match.group("event")), int(match.group("step")))
        if best_key is None or key > best_key:
            best_key = key
            best_path = path
    return best_path


def build_last_episode_state_grid(output_dir: str | Path) -> Path | None:
    """
    Build a grid montage of each episode's last composite state — the
    highest-numbered episode_*/prediction_events/composite_prediction_event_*.png
    (cam_high + cam_wrist, no translation/blend) — saved the same way for
    every baseline (raw sim, --color-calibrate, --turbo, --gpt), so results
    are directly comparable.

    Read-only with respect to episode_*/prediction_events/: files there are
    only read, never modified. Writes one new file,
    output_dir/last_episode_state_grid.png. Safe to call repeatedly.

    Returns the path to the saved grid, or None if no episode had a
    composite_prediction_event_*.png to show.
    """
    output_dir = Path(output_dir)
    episode_dirs = sorted(
        (d for d in output_dir.iterdir() if d.is_dir() and re.fullmatch(r"episode_\d+", d.name)),
        key=lambda d: int(d.name.split("_")[1]),
    )
    if not episode_dirs:
        return None

    composite_images = []
    tile_shape = None
    for ep_dir in episode_dirs:
        composite_img = None
        prediction_events_dir = ep_dir / "prediction_events"
        if prediction_events_dir.is_dir():
            last_path = _last_composite_prediction_event_path(prediction_events_dir)
            if last_path is not None:
                composite_img = cv2.imread(str(last_path))
                if composite_img is not None:
                    tile_shape = composite_img.shape
        composite_images.append((ep_dir.name, composite_img))

    if tile_shape is None:
        return None

    title_h = 26
    tile_h, tile_w = tile_shape[0], tile_shape[1]
    tiles = []
    for ep_name, composite_img in composite_images:
        tile = np.full((title_h + tile_h, tile_w, 3), 18, dtype=np.uint8)
        if composite_img is not None:
            tile[title_h:, :, :] = composite_img
        else:
            cv2.putText(
                tile, "no data", (10, title_h + tile_h // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1, cv2.LINE_AA,
            )
        cv2.rectangle(tile, (0, 0), (tile_w, title_h), (44, 44, 44), -1)
        cv2.putText(
            tile, ep_name, (8, title_h - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235, 235, 235), 1, cv2.LINE_AA,
        )
        tiles.append(tile)

    n = len(tiles)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    grid_tile_h, grid_tile_w = tiles[0].shape[:2]
    grid = np.full((rows * grid_tile_h, cols * grid_tile_w, 3), 18, dtype=np.uint8)
    for idx, tile in enumerate(tiles):
        r, c = divmod(idx, cols)
        grid[r * grid_tile_h:(r + 1) * grid_tile_h, c * grid_tile_w:(c + 1) * grid_tile_w] = tile

    grid_path = output_dir / "last_episode_state_grid.png"
    cv2.imwrite(str(grid_path), grid)
    return grid_path


def convert_action_to_mujoco(action: torch.Tensor, gripper_mj_range: tuple) -> np.ndarray:
    """
    Convert policy action (8-dim: 7 joints degrees + gripper mm) to MuJoCo ctrl (8-dim).
    """
    action_np = action.cpu().numpy()
    if action_np.ndim > 1:
        action_np = action_np[0]
    return lerobot_state_to_mujoco_ctrl(action_np, gripper_mj_range)


def _quat_normalize(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    norm = np.linalg.norm(quat)
    if norm <= 0:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return quat / norm


def _quat_from_euler_xyz(euler: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = [float(v) for v in euler]
    cr, sr = np.cos(roll / 2.0), np.sin(roll / 2.0)
    cp, sp = np.cos(pitch / 2.0), np.sin(pitch / 2.0)
    cy, sy = np.cos(yaw / 2.0), np.sin(yaw / 2.0)
    quat = np.array(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
        dtype=np.float64,
    )
    return _quat_normalize(quat)


def _euler_xyz_from_quat(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = [float(v) for v in _quat_normalize(quat)]
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = np.copysign(np.pi / 2.0, sinp)
    else:
        pitch = np.arcsin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return np.array([roll, pitch, yaw], dtype=np.float64)


def _fmt_xyz(vec: np.ndarray | None) -> str:
    if vec is None:
        return "n/a"
    return f"[{vec[0]:.4f}, {vec[1]:.4f}, {vec[2]:.4f}]"


def _fmt_euler_deg(euler: np.ndarray | None) -> str:
    if euler is None:
        return "n/a"
    deg = np.degrees(euler)
    return f"[{deg[0]:.1f}, {deg[1]:.1f}, {deg[2]:.1f}] deg"


# Flags that select a visual-input baseline; --all drives these itself, one per
# subprocess run, so they are stripped from the forwarded argv.
_ALL_VARIANT_FLAGS = ("--all", "--color-calibrate", "--turbo")
_ALL_VARIANTS = (
    ("raw sim", ()),
    ("color-calibrate", ("--color-calibrate",)),
    ("turbo", ("--turbo",)),
)


def _run_all_variants() -> None:
    """Re-run this script once per --all baseline (raw sim / color-calibrate / turbo).

    Each run is a fresh subprocess with the same CLI args (minus the variant
    flags) plus its own variant flag, so each writes to the task profile's own
    output directory for that sim_variant and can be compared side by side.
    """
    base_argv = [a for a in sys.argv[1:] if a not in _ALL_VARIANT_FLAGS]
    script_path = str(Path(__file__).resolve())

    for label, extra_flags in _ALL_VARIANTS:
        print(f"\n{'=' * 70}\n[ALL] Running variant: {label}\n{'=' * 70}\n", flush=True)
        cmd = [sys.executable, script_path, *base_argv, *extra_flags]
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(
                f"[ALL] Variant '{label}' exited with code {result.returncode}; stopping remaining variants.",
                flush=True,
            )
            sys.exit(result.returncode)

    print("\n[ALL] All three variants (raw sim, color-calibrate, turbo) completed.\n", flush=True)


def _run_all_checkpoints(policy_paths: list[str]) -> None:
    """Re-run this script once per --policy-paths checkpoint, one subprocess each.

    Mirrors _run_all_variants: strips --policy-path(s) from the forwarded argv and
    sets --policy-path explicitly for each checkpoint, so each run derives its own
    checkpoint-specific output directory (via _extract_checkpoint_name / policy.config.type).
    """
    base_argv = []
    skip_next = False
    for a in sys.argv[1:]:
        if skip_next:
            skip_next = False
            continue
        if a in ("--policy-path", "--policy-paths"):
            skip_next = True
            continue
        if a.startswith("--policy-path=") or a.startswith("--policy-paths="):
            continue
        base_argv.append(a)
    script_path = str(Path(__file__).resolve())

    for i, policy_path in enumerate(policy_paths, start=1):
        print(
            f"\n{'=' * 70}\n[CHECKPOINTS] Running checkpoint {i}/{len(policy_paths)}: "
            f"{policy_path}\n{'=' * 70}\n",
            flush=True,
        )
        cmd = [sys.executable, script_path, *base_argv, "--policy-path", policy_path]
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(
                f"[CHECKPOINTS] Checkpoint {policy_path!r} exited with code "
                f"{result.returncode}; stopping remaining checkpoints.",
                flush=True,
            )
            sys.exit(result.returncode)

    print(f"\n[CHECKPOINTS] All {len(policy_paths)} checkpoints completed.\n", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Deploy ACT policy in MuJoCo xArm simulation"
    )
    parser.add_argument(
        "--policy-path",
        type=str,
        default=_DEFAULT_RECORD_POLICY_CHECKPOINT,
        help="Path to policy checkpoint directory"
    )
    parser.add_argument(
        "--policy-paths",
        type=str,
        default=None,
        help=(
            "Evaluate multiple policy checkpoints back-to-back, one per subprocess run. "
            "Separate paths with ',' or ';', e.g. "
            "--policy-paths 'outputs/act_place_mug/checkpoints/006000/pretrained_model;"
            "outputs/act_place_mug/checkpoints/003000/pretrained_model'. Takes priority "
            "over --policy-path. Each run re-invokes this script with --policy-path set to "
            "one checkpoint, so each writes to that checkpoint's own output directory."
        ),
    )
    parser.add_argument(
        "--task",
        type=str,
        default=_DEFAULT_RECORD_TASK_ID,
        choices=sorted(get_task_profiles()),
        help="Task profile (scene, datasets, default turbo checkpoint dirs).",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Task prompt/instruction for the policy. Defaults to the selected task's profile instruction.",
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
        "--no-mujoco-view",
        action="store_true",
        help="Disable the MuJoCo 3D interactive viewer window",
    )
    parser.add_argument(
        "--scene-path",
        type=str,
        default="pointclouds/xarm7_black.npz",
        help="Path to Gaussian Splatting scene file for composite rendering"
    )
    parser.add_argument(
        "--color-calibrate", action="store_true",
        default=False,
        help="Apply per-camera color calibration YAML files when available."
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help=(
            "Run the deployment three times back-to-back, once per visual-input baseline: "
            "raw sim (no color calibration / translation), --color-calibrate, and --turbo. "
            "Each variant re-invokes this script with the same arguments (minus --all/"
            "--color-calibrate/--turbo) plus its own variant flag, so each writes to its own "
            "task-profile output directory."
        ),
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
        help="Like --obs but use episode 0 from the selected task's eval dataset (see --obs-eval-path)"
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Path to dataset directory for --obs and for Real display when not using --obs-eval. Defaults to the selected task dataset root.",
    )
    parser.add_argument(
        "--obs-eval-path",
        type=str,
        default=None,
        help="Dataset path for --obs-eval. Defaults to the selected task eval dataset root.",
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
        "--policy-no-resize",
        action="store_true",
        help="Feed full render resolution to the policy (e.g. 640×480). Default: resize to policy-input-h/w.",
    )
    parser.add_argument(
        "--policy-input-h",
        type=int,
        default=224,
        help="Image height for policy input when resizing (default 224). Ignored with --policy-no-resize.",
    )
    parser.add_argument(
        "--policy-input-w",
        type=int,
        default=224,
        help="Image width for policy input when resizing (default 224). Ignored with --policy-no-resize.",
    )
    parser.add_argument(
        "--no_obs",
        action="store_true",
        help=(
            "Do not load real-world replay videos or show Real camera windows (composite-only). "
            "Ignored for --obs and --obs-eval: those modes always load dataset frames and show "
            "Real windows when not --headless. Skips expensive per-step replay when not needed."
        ),
    )
    parser.add_argument(
        "--num_eval_episodes",
        type=int,
        default=_NUM_EPISODES,
        help=f"Number of evaluation episodes to run (default: {_NUM_EPISODES})"
    )
    parser.add_argument(
        "--select",
        action="store_true",
        default=True,
        help="Load initial-state contour overlays generated by initial_states_overlay.py"
    )
    parser.add_argument(
        "--select-interactive",
        action="store_true",
        default=False,
        help=(
            "Pick --num_eval_episodes initial states by hand in a click UI. Default is "
            "automatic: the first ceil(N/2) and last floor(N/2) episodes (by index) that "
            "have a valid contour, so repeated runs (e.g. --all) always evaluate the same "
            "episodes."
        ),
    )
    parser.add_argument(
        "--initial-states-dir",
        type=str,
        default=None,
        help="Path to the dataset root containing <object>/individual_masks and <object>/all_episodes_grid.png"
    )
    parser.add_argument(
        "--object-name",
        type=str,
        default=None,
        help="Object name(s) matching the segmentation prompt, e.g. 'mug' or 'mug, saucer'"
    )
    parser.add_argument(
        "--auto-align-initial-objects",
        action="store_true",
        default=True,
        help=(
            "Before each selected deployment episode, optimize MuJoCo object poses "
            "against saved initial-state SAM masks."
        ),
    )
    parser.add_argument(
        "--auto-align-cache-dir",
        type=str,
        default=None,
        help="Directory for cached auto-aligned object poses. Defaults to <initial-states-dir>/auto_object_poses.",
    )
    parser.add_argument(
        "--auto-align-force",
        action="store_true",
        help="Recompute automatic object alignment even if a cached pose exists.",
    )
    parser.add_argument(
        "--auto-align-optimize-z",
        action="store_true",
        help="Also optimize object Z. By default only XY and yaw are optimized.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save sim evaluation data. Default is derived from task, policy type, and checkpoint."
    )
    parser.add_argument(
        "--no-save-sim-eval",
        action="store_true",
        help="Run evaluation but do not write sim outputs (episode npy/mp4, selected_states_grid.png, or output directory).",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        default=True,
        help="Save a single tiled MP4 of the displayed camera windows for each episode.",
    )
    parser.add_argument(
        "--fast-rollout-video-replay",
        action="store_true",
        help=(
            "Render camera observations only when the policy needs a fresh action chunk, "
            "then replay saved MuJoCo states after each episode to write full videos."
        ),
    )
    parser.add_argument(
        "--sim-gripper-observation-offset-mm",
        type=float,
        default=_PICK_SHOE_GRIPPER_OBS_OFFSET_MM,
        help=(
            "Subtract this many millimeters from the MuJoCo qpos-derived gripper "
            "observation before policy inference. Default is measured for pick_shoe "
            "from data_pick_shoe action vs rigid-shoe replay qpos."
        ),
    )
    parser.add_argument(
        "--sim-gripper-observation-offset-mode",
        choices=("contact", "threshold", "always", "never"),
        default="contact",
        help=(
            "When to apply --sim-gripper-observation-offset-mm. 'contact' checks "
            "finger pad contact with right_shoe_col; 'threshold' uses "
            "--sim-gripper-observation-threshold-mm."
        ),
    )
    parser.add_argument(
        "--sim-gripper-observation-threshold-mm",
        type=float,
        default=_PICK_SHOE_GRIPPER_OFFSET_THRESHOLD_MM,
        help="Threshold used when --sim-gripper-observation-offset-mode=threshold.",
    )
    parser.add_argument(
        "--gpt",
        action="store_true",
        help="Use GPT-Image-2 sim→real style transfer on composite policy images when a new prediction is needed.",
    )
    parser.add_argument(
        "--turbo",
        action="store_true",
        help=(
            "Pix2pix-turbo on composite policy images when a new prediction is needed. "
            "If --turbo-checkpoint* are omitted, uses TaskProfile turbo_output_* for this --task."
        ),
    )
    parser.add_argument(
        "--turbo-checkpoint",
        type=str,
        default=None,
        help="Shared .pkl if per-camera paths are not set.",
    )
    parser.add_argument(
        "--turbo-checkpoint-stationary",
        type=str,
        default=None,
        help="Stationary cam .pkl (cam_high).",
    )
    parser.add_argument(
        "--turbo-checkpoint-wrist",
        type=str,
        default=None,
        help="Wrist cam .pkl (cam_wrist).",
    )
    parser.add_argument(
        "--turbo-prompt",
        type=str,
        default=None,
        help="Pix2pix text prompt.",
    )
    parser.add_argument(
        "--turbo-resolution",
        type=int,
        default=None,
        help="Square side before encode, multiple of 8 (default 224).",
    )
    parser.add_argument(
        "--turbo-device",
        type=str,
        default=None,
        help="Torch device (default: CUDA if available).",
    )
    parser.add_argument(
        "--gpt-style-image-stationary",
        type=str,
        default=None,
        help="Stationary-camera style reference for GPT image translation. Defaults to task_profile.dataset_root_480640/real_captures/stationary/frame_0000.png.",
    )
    parser.add_argument(
        "--gpt-style-image-wrist",
        type=str,
        default=None,
        help="Wrist-camera style reference for GPT image translation. Defaults to task_profile.dataset_root_480640/real_captures/wrist/frame_0000.png.",
    )
    parser.add_argument(
        "--gpt-prompt",
        type=str,
        default=_DEFAULT_GPT_IMAGE_PROMPT,
        help="Prompt for GPT-Image-2 style transfer.",
    )
    parser.add_argument(
        "--gpt-max-side",
        type=int,
        default=_DEFAULT_GPT_IMAGE_MAX_SIDE,
        help="Resize the longest input side before GPT-Image-2 upload (default 1024).",
    )

    args = parser.parse_args()

    if args.policy_paths:
        policy_paths = [p.strip() for p in re.split(r"[;,]", args.policy_paths) if p.strip()]
        if not policy_paths:
            raise ValueError("--policy-paths was given but contained no non-empty paths")
        _run_all_checkpoints(policy_paths)
        return

    if args.all:
        _run_all_variants()
        return

    num_eval_episodes = args.num_eval_episodes
    task_id = args.task
    task_profile = get_task_profile(task_id)
    max_prediction_events_per_trajectory = _max_prediction_events_per_trajectory(task_id)
    # print(f"[INFO] Default task: {task_id}")
    if args.sim_gripper_observation_offset_mm:
        print(
            "[INFO] Applying sim gripper observation offset: "
            f"-{args.sim_gripper_observation_offset_mm:.3f} mm "
            f"(mode={args.sim_gripper_observation_offset_mode})"
        )
    if args.prompt is None:
        args.prompt = task_profile.single_task
    if args.initial_states_dir is None:
        args.initial_states_dir = task_profile.dataset_root
    if args.object_name is None:
        args.object_name = task_profile.selection_object_name

    # Load policy early so task-aware default output paths can depend on policy type/checkpoint.
    policy, config_dict = load_policy(args.policy_path)
    device = get_safe_torch_device(policy.config.device)
    policy = policy.to(device)
    turbo_cfg = {}
    if isinstance(config_dict, dict):
        turbo_cfg = config_dict.get("turbo") or config_dict.get("sim2real") or {}
    turbo_enabled = bool(args.turbo or turbo_cfg.get("enabled", False))
    turbo_prompt = args.turbo_prompt or turbo_cfg.get("prompt", "a real-world robot camera image")
    turbo_resolution = int(
        args.turbo_resolution
        if args.turbo_resolution is not None
        else turbo_cfg.get("resolution", 224)
    )
    if turbo_enabled:
        if turbo_resolution <= 0 or turbo_resolution % 8 != 0:
            raise ValueError(
                f"--turbo-resolution / policy resolution must be a positive multiple of 8, got {turbo_resolution}"
            )
    turbo_device = args.turbo_device or turbo_cfg.get("device")

    if args.gpt and turbo_enabled:
        raise ValueError("Use either --gpt or --turbo, not both.")

    if args.no_save_sim_eval:
        checkpoint_name = _extract_checkpoint_name(args.policy_path)
        args.output_dir = None
    elif args.output_dir is None:
        checkpoint_name = _extract_checkpoint_name(args.policy_path)
        if turbo_enabled:
            sim_variant = "turbo"
        elif args.color_calibrate:
            sim_variant = "kaifeng"
        else:
            sim_variant = "default"
        args.output_dir = task_profile.sim_eval_root_for_policy(
            policy.config.type, checkpoint_name, sim_variant=sim_variant
        )
    else:
        checkpoint_name = _extract_checkpoint_name(args.policy_path)

    if args.dataset_path is None:
        args.dataset_path = task_profile.dataset_root
    if args.obs_eval_path is None:
        args.obs_eval_path = task_profile.eval_root_for_policy(policy.config.type, checkpoint_name)

    # Output directory for sim evaluation data (None when --no-save-sim-eval)
    output_dir = Path(args.output_dir) if args.output_dir is not None else None
    next_saved_episode_idx: int | None = None
    current_episode_output_dir: Path | None = None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        # print(f"[INFO] Sim eval output directory: {output_dir.resolve()}")
    else:
        # print("[INFO] Sim eval disk output disabled (--no-save-sim-eval)")
        pass
    if args.record and output_dir is None:
        raise ValueError("--record requires sim eval disk output. Remove --no-save-sim-eval or set --output-dir.")

    # Load initial-state contours and open selection UI when --select is used
    list_of_contours = None
    selected_contours = None
    selected_episode_indices = None
    if args.select:
        list_of_contours = load_initial_state_contours(
            initial_states_dir=args.initial_states_dir,
            object_name=args.object_name,
        )
        if args.select_interactive:
            selected_contours, selected_episode_indices = select_contours_ui(
                list_of_contours,
                num_eval_episodes,
                initial_states_dir=args.initial_states_dir,
                object_name=args.object_name,
            )
        else:
            selected_contours, selected_episode_indices = select_contours_auto(
                list_of_contours,
                num_eval_episodes,
            )
            print(
                f"[INFO] Auto-selected {len(selected_episode_indices)} initial-state episodes "
                f"(first/last half): {selected_episode_indices}",
                flush=True,
            )
        if not selected_contours:
            # print("[INFO] No episodes selected, exiting.")
            sys.exit(0)
        if len(selected_contours) != num_eval_episodes:
            # print(f"[ERROR] Selected {len(selected_contours)} contours but "
                  # f"num_eval_episodes={num_eval_episodes}. Must be equal.")
            sys.exit(1)
        if output_dir is not None:
            save_selection_grid(
                initial_states_dir=args.initial_states_dir,
                object_name=args.object_name,
                list_of_contours=list_of_contours,
                selected_indices=selected_episode_indices,
                output_path=output_dir / "selected_states_grid.png",
            )

    if args.obs:
        # print("[INFO] --obs: using real-world dataset images as policy input")
        pass
    elif args.obs_eval:
        # print(
            # f"[INFO] --obs-eval: using episode 0 images from {args.obs_eval_path!r} as policy input"
        # )
        pass

    # print(f"[INFO] Policy action parameters:")
    if hasattr(policy.config, 'horizon'):
        # print(f"  - horizon: {policy.config.horizon}")
        pass
    if hasattr(policy.config, 'n_action_steps'):
        # print(f"  - n_action_steps: {policy.config.n_action_steps}")
        pass
    if hasattr(policy.config, 'chunk_size'):
        # print(f"  - chunk_size: {policy.config.chunk_size}")
        pass

    # Create pre/post processors
    processor_path = Path(args.policy_path) / "policy_preprocessor.json"
    if processor_path.exists():
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=policy.config,
            pretrained_path=args.policy_path,
        )
    else:
        # print("[WARN] Processor files not found, creating from config")
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=policy.config,
            pretrained_path=None,
        )

    # Load MuJoCo xArm model
    project_root = Path(__file__).parent.parent
    xarm_dir = project_root / "xarm7"
    scene_xml_path = resolve_task_scene_xml(task_id, xarm_dir)
    # print(f"[INFO] Using MuJoCo scene for task {task_id!r}: {scene_xml_path.name}")
    original_cwd = os.getcwd()
    try:
        os.chdir(str(xarm_dir))
        model = MjModel.from_xml_path(scene_xml_path.name)
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
    # print(f"[INFO] Gripper ctrl range: [{gripper_mj_range[0]}, {gripper_mj_range[1]}]")
    gripper_observation_contact_geom_ids = (
        geom_ids_by_name_or_prefix(model, prefixes=_GRIPPER_CONTACT_GEOM_PREFIXES),
        geom_ids_by_name_or_prefix(model, names=_PICK_SHOE_CONTACT_GEOM_NAMES),
    )
    if args.sim_gripper_observation_offset_mode == "contact" and (
        not gripper_observation_contact_geom_ids[0] or not gripper_observation_contact_geom_ids[1]
    ):
        print(
            "[WARN] Contact-gated gripper offset cannot find expected gripper/shoe geoms; "
            "offset will not be applied."
        )

    adjustable_object_names = tuple(dict.fromkeys(task_profile.deploy_adjustable_object_names))

    # Mug freejoint address (for in-memory position/orientation adjustment during warmup)
    mug_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "mug_joint")
    mug_qpos_addr = model.jnt_qposadr[mug_joint_id] if mug_joint_id >= 0 else -1
    if mug_qpos_addr >= 0:
        # print(f"[INFO] Mug freejoint found (qpos addr={mug_qpos_addr})")
        pass
    else:
        # print("[WARN] mug_joint not found – mug pose adjustment disabled")
        pass

    adjustable_body_ids = {}
    adjustable_body_default_pos = {}
    adjustable_body_default_quat = {}
    for obj_name in adjustable_object_names:
        if obj_name == "mug":
            continue
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, obj_name)
        if body_id < 0:
            # print(f"[WARN] Adjustable body '{obj_name}' not found in {scene_xml_path.name}")
            continue
        adjustable_body_ids[obj_name] = body_id
        adjustable_body_default_pos[obj_name] = model.body_pos[body_id].copy()
        adjustable_body_default_quat[obj_name] = model.body_quat[body_id].copy()

    RENDER_W, RENDER_H = 640, 480

    # Match MuJoCo vertical FOV to the calibrated camera intrinsics, mirroring
    # compare_recorded_vs_mujoco so foreground masks align with the GS render.
    for cam_key, cam_cfg in CAMERA_CONFIG.items():
        K = cam_cfg["config"]["intrinsics"]
        fy = K[1, 1]
        correct_fovy = float(2.0 * np.degrees(np.arctan(RENDER_H / (2.0 * fy))))
        mj_cam = cam_cfg["mujoco_cam"]
        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, mj_cam)
        if cam_id >= 0:
            model.cam_fovy[cam_id] = correct_fovy

    renderer = mujoco.Renderer(model, height=RENDER_H, width=RENDER_W)
    seg_renderer = mujoco.Renderer(model, height=RENDER_H, width=RENDER_W)
    seg_renderer.enable_segmentation_rendering()

    robot_geom_ids = get_robot_geom_ids(model)
    # print(f"[INFO] Found {len(robot_geom_ids)} robot geoms for masking")

    auto_align_config = None
    if args.auto_align_initial_objects:
        auto_align_config = ObjectPoseAlignConfig(
            initial_states_dir=args.initial_states_dir,
            object_name=args.object_name,
            cache_dir=args.auto_align_cache_dir,
            optimize_z=args.auto_align_optimize_z,
            force=args.auto_align_force,
            free_joint_pairs=task_profile.calibration_free_joint_pairs,
            body_name_aliases=task_profile.object_body_name_aliases,
        )

    # Apply camera calibration
    mujoco.mj_forward(model, data)
    for cam_key, cam_cfg in CAMERA_CONFIG.items():
        mj_cam = cam_cfg["mujoco_cam"]
        cc = cam_cfg["config"]
        cam_id = set_mujoco_camera_from_config(data, model, mj_cam, cc)
        # print(f"[INFO] Camera '{mj_cam}' (id={cam_id}) calibration applied")

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
            color_calib_by_camera = {}
            if args.color_calibrate and not args.gpt and not turbo_enabled:
                for cam_key in CAMERA_CONFIG:
                    default_calib = task_profile.color_calibration_path(cam_key)
                    try:
                        color_calib_by_camera[cam_key] = load_color_mapping(default_calib)
                        # print(f"[INFO] Loaded {cam_key} color calibration from: {default_calib}")
                    except Exception as e:
                        # print(f"[WARN] Failed to load {cam_key} color calibration from {default_calib}: {e}")
                        pass
            viz_cfg = {'viz_w': RENDER_W, 'viz_h': RENDER_H, 'viz_near': 0.1, 'viz_far': 10.0}
            gaussian_data = {
                'scene_data': scene_data,
                'scene_depth_data': scene_depth_data,
                'viz_cfg': viz_cfg,
                'color_calib_by_camera': color_calib_by_camera,
                'camera_intrinsics': camera_intrinsics,
            }
            # print(f"[INFO] Loaded Gaussian Splatting scene from: {args.scene_path}")
        except Exception as e:
            # print(f"[WARN] Failed to load Gaussian scene: {e}")
            import traceback
            traceback.print_exc()
    else:
        # print(f"[WARN] Scene file not found: {args.scene_path}")
        pass

    # GPT image sim→real translators (replace color calibration when --gpt).
    gpt_translators: dict[str, object] | None = None
    gpt_style_dirs: dict = {}
    _gpt_style_frame_cache: dict = {}
    if args.gpt:
        from sim2real import GPTImageTranslator

        style_path_stationary = _resolve_path_under_project(
            args.gpt_style_image_stationary, project_root
        ) or str((project_root / _default_gpt_style_image_path(task_profile, "stationary")).resolve())
        style_path_wrist = _resolve_path_under_project(
            args.gpt_style_image_wrist, project_root
        ) or str((project_root / _default_gpt_style_image_path(task_profile, "wrist")).resolve())
        style_paths = {
            "stationary": Path(style_path_stationary),
            "wrist": Path(style_path_wrist),
        }
        for cam_key, style_path in style_paths.items():
            if not style_path.exists():
                raise FileNotFoundError(
                    f"GPT style image for {cam_key!r} not found: {style_path}"
                )
        gpt_translators = {
            cam_key: GPTImageTranslator(
                style_references=[_load_rgb_pil(style_path)],
                cam_name=cam_key,
                prompt=args.gpt_prompt,
                model=_DEFAULT_GPT_IMAGE_MODEL,
                size=_DEFAULT_GPT_IMAGE_SIZE,
                quality=_DEFAULT_GPT_IMAGE_QUALITY,
                max_side=args.gpt_max_side,
            )
            for cam_key, style_path in style_paths.items()
        }
        gpt_style_dirs = {cam_key: path.parent for cam_key, path in style_paths.items()}
        n_act = getattr(policy.config, 'n_action_steps', None)
        if n_act and n_act > 1:
            est_calls = args.max_steps // n_act + 1
            # print(f"[INFO] GPT image translation ready: 2 parallel API calls per query, every ~{n_act} steps "
                  # f"(~{est_calls} query rounds/episode).")
        else:
            # print("[INFO] GPT image translation ready: 2 parallel API calls each prediction step.")
            pass

    turbo_translators: dict[str, object] | None = None
    if turbo_enabled:
        from sim2real import SimToRealTranslator

        turbo_task_defaults = task_profile.turbo_default_checkpoint_paths(project_root)

        def _ckpt_stationary() -> str | None:
            return (
                _resolve_path_under_project(args.turbo_checkpoint_stationary, project_root)
                or _resolve_path_under_project(turbo_cfg.get("checkpoint_stationary"), project_root)
                or _resolve_path_under_project(args.turbo_checkpoint, project_root)
                or _resolve_path_under_project(turbo_cfg.get("checkpoint"), project_root)
                or (turbo_task_defaults[0] if turbo_task_defaults else None)
            )

        def _ckpt_wrist() -> str | None:
            return (
                _resolve_path_under_project(args.turbo_checkpoint_wrist, project_root)
                or _resolve_path_under_project(turbo_cfg.get("checkpoint_wrist"), project_root)
                or _resolve_path_under_project(args.turbo_checkpoint, project_root)
                or _resolve_path_under_project(turbo_cfg.get("checkpoint"), project_root)
                or (turbo_task_defaults[1] if turbo_task_defaults else None)
            )

        ckpt_stationary = _ckpt_stationary()
        ckpt_wrist = _ckpt_wrist()
        if not ckpt_stationary or not ckpt_wrist:
            raise ValueError(
                "Turbo needs checkpoints for both cameras (--turbo-checkpoint-* or policy turbo.*), "
                "or set turbo_output_stationary / turbo_output_wrist on the task profile for this --task."
            )
        if ckpt_stationary == ckpt_wrist:
            tr = SimToRealTranslator(
                checkpoint_path=ckpt_stationary,
                prompt=turbo_prompt,
                resolution=turbo_resolution,
                device=turbo_device,
            )
            turbo_translators = {"stationary": tr, "wrist": tr}
        else:
            turbo_translators = {
                "stationary": SimToRealTranslator(
                    checkpoint_path=ckpt_stationary,
                    prompt=turbo_prompt,
                    resolution=turbo_resolution,
                    device=turbo_device,
                ),
                "wrist": SimToRealTranslator(
                    checkpoint_path=ckpt_wrist,
                    prompt=turbo_prompt,
                    resolution=turbo_resolution,
                    device=turbo_device,
                ),
            }

    # Load real-world dataset frames lazily per selected initial-state episode.
    # This keeps the Real windows synchronized with the selected grid cell.
    obs_frames = None
    need_dataset_frames = args.obs or args.obs_eval or not args.no_obs
    obs_frame_cache: dict[tuple[str, int], dict] = {}

    def _load_obs_frames_for_episode(source_episode_idx: int | None) -> dict | None:
        if not need_dataset_frames:
            return None
        if source_episode_idx is None:
            source_episode_idx = args.episode
        if args.obs_eval and selected_episode_indices is None:
            obs_load_path = args.obs_eval_path
            obs_load_episode = 0
        else:
            obs_load_path = args.dataset_path
            obs_load_episode = int(source_episode_idx)
        cache_key = (str(obs_load_path), obs_load_episode)
        if cache_key in obs_frame_cache:
            return obs_frame_cache[cache_key]
        try:
            episode_data = load_episode(
                obs_load_path, obs_load_episode, dataset_root=args.dataset_root
            )
            loaded_frames = load_dataset_frames(episode_data)
            obs_frame_cache[cache_key] = loaded_frames
            num_obs_frames = max(len(loaded_frames.get(k, [])) for k in CAMERA_CONFIG) or 1
            # print(
                # f"[INFO] Loaded real dataset images: {num_obs_frames} frames from "
                # f"{obs_load_path!r} episode {obs_load_episode}"
            # )
            return loaded_frames
        except Exception as e:
            if args.obs or args.obs_eval:
                flag = "--obs-eval" if args.obs_eval else "--obs"
                # print(f"[ERROR] Failed to load dataset for {flag}: {e}")
                import traceback
                traceback.print_exc()
                sys.exit(1)
            # print(f"[WARN] Could not load dataset for Real windows display: {e}")
            return None
    if args.no_obs and not need_dataset_frames:
        # print(
            # "[INFO] --no_obs: skipping real-world replay load and Real camera windows "
            # "(composite-only deployment at --fps)."
        # )
        pass

    # Initialize keyboard listener (ALOHA-style state machine)
    listener, events = init_keyboard_listener()

    # OpenCV "Real:" dataset replay windows: off with --no_obs unless --obs / --obs-eval
    # (those modes need to see real images next to composite).
    show_real_windows = (not args.no_obs) or args.obs or args.obs_eval

    WINDOW_W, WINDOW_H = 400, 300
    if not args.headless:
        X_START, Y_START = 50, 30
        X_STEP, Y_STEP = 410, 340
        cam_keys = list(CAMERA_CONFIG.keys())
        for i, cam_key in enumerate(cam_keys):
            obs_key = f"observation.images.{CAMERA_CONFIG[cam_key]['dataset_cam']}"
            cam_short = obs_key.split('.')[-1]
            win_real = f"Real: {cam_short}"
            win_comp = f"Composite: {cam_short}"
            if show_real_windows:
                cv2.namedWindow(win_real, cv2.WINDOW_NORMAL)
                if hasattr(cv2, "setWindowTitle"):
                    cv2.setWindowTitle(win_real, _format_window_title("Real", cam_short, next_saved_episode_idx))
                cv2.resizeWindow(win_real, WINDOW_W, WINDOW_H)
                cv2.moveWindow(win_real, X_START + i * X_STEP, Y_START)
            cv2.namedWindow(win_comp, cv2.WINDOW_NORMAL)
            if hasattr(cv2, "setWindowTitle"):
                cv2.setWindowTitle(win_comp, _format_window_title("Composite", cam_short, next_saved_episode_idx))
            cv2.resizeWindow(win_comp, WINDOW_W, WINDOW_H)
            cv2.moveWindow(win_comp, X_START + i * X_STEP, Y_START + Y_STEP)
            if gpt_translators is not None:
                win_gpt = f"GPT: {cam_short}"
                cv2.namedWindow(win_gpt, cv2.WINDOW_NORMAL)
                if hasattr(cv2, "setWindowTitle"):
                    cv2.setWindowTitle(win_gpt, _format_window_title("GPT", cam_short, next_saved_episode_idx))
                cv2.resizeWindow(win_gpt, WINDOW_W, WINDOW_H)
                cv2.moveWindow(win_gpt, X_START + i * X_STEP, Y_START + 2 * Y_STEP)
            if turbo_translators is not None:
                win_tb = f"Turbo: {cam_short}"
                cv2.namedWindow(win_tb, cv2.WINDOW_NORMAL)
                if hasattr(cv2, "setWindowTitle"):
                    cv2.setWindowTitle(win_tb, _format_window_title("Turbo", cam_short, next_saved_episode_idx))
                cv2.resizeWindow(win_tb, WINDOW_W, WINDOW_H)
                cv2.moveWindow(win_tb, X_START + i * X_STEP, Y_START + 2 * Y_STEP)

    _obs_tag = ""
    if args.obs:
        _obs_tag = " [real obs]"
    elif args.obs_eval:
        _obs_tag = " [obs-eval]"
    elif gpt_translators is not None:
        _obs_tag = " [gpt]"
    elif turbo_translators is not None:
        _obs_tag = " [turbo]"
    # print(
        # f"[INFO] Starting policy evaluation ({num_eval_episodes} episodes, max {args.max_steps} steps each)"
        # f"{_obs_tag}"
    # )

    step_dt = 1.0 / args.fps

    viewer = None
    viewer_ctx = None
    # Skip mujoco 3D viewer when using EGL (e.g. SSH X11 forwarding) - GLFW/GLX fails
    use_viewer = (
        not args.headless
        and not args.no_mujoco_view
        and _HAS_DISPLAY
        and _HAS_MJ_VIEWER
        and os.environ.get("MUJOCO_GL") != "egl"
    )
    if use_viewer:
        try:
            viewer_ctx = mujoco.viewer.launch_passive(model, data)
            viewer = viewer_ctx.__enter__()
            print("[INFO] MuJoCo 3D viewer launched (synchronized)")
        except Exception:
            viewer = None
            viewer_ctx = None
    elif args.no_mujoco_view:
        print("[INFO] MuJoCo 3D viewer disabled (--no-mujoco-view)")
    elif _HAS_DISPLAY and not _HAS_MJ_VIEWER:
        print("[WARN] mujoco.viewer not available; 3D viewer disabled.")

    saved_episodes = []
    completed_episodes = 0
    episode_idx = 0
    episode_window_writer: cv2.VideoWriter | None = None
    episode_window_tmp_path: Path | None = None
    pending_fast_replay_dirs: list[Path] = []

    default_mug_pos = data.qpos[mug_qpos_addr:mug_qpos_addr + 3].copy() if mug_qpos_addr >= 0 else None
    default_mug_quat = data.qpos[mug_qpos_addr + 3:mug_qpos_addr + 7].copy() if mug_qpos_addr >= 0 else None

    def _reset_sim():
        """Reset simulation to home keyframe and re-apply camera calibration."""
        try:
            home_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
            mujoco.mj_resetDataKeyframe(model, data, home_id)
        except Exception:
            mujoco.mj_resetData(model, data)
        for obj_name, body_id in adjustable_body_ids.items():
            model.body_pos[body_id] = adjustable_body_default_pos[obj_name]
            model.body_quat[body_id] = adjustable_body_default_quat[obj_name]
        if mug_qpos_addr >= 0 and default_mug_pos is not None and default_mug_quat is not None:
            data.qpos[mug_qpos_addr:mug_qpos_addr + 3] = default_mug_pos
            data.qpos[mug_qpos_addr + 3:mug_qpos_addr + 7] = default_mug_quat
        mujoco.mj_forward(model, data)
        for cam_cfg in CAMERA_CONFIG.values():
            if cam_cfg["config"].get("type", "stationary") == "stationary":
                set_mujoco_camera_from_config(data, model, cam_cfg["mujoco_cam"], cam_cfg["config"])

    def _capture_adjustable_state() -> dict:
        state = {
            "body_positions": {
                obj_name: model.body_pos[body_id].copy()
                for obj_name, body_id in adjustable_body_ids.items()
            },
            "body_quats": {
                obj_name: model.body_quat[body_id].copy()
                for obj_name, body_id in adjustable_body_ids.items()
            },
        }
        if mug_qpos_addr >= 0:
            mug_pos = data.qpos[mug_qpos_addr:mug_qpos_addr + 3].copy()
            mug_quat = data.qpos[mug_qpos_addr + 3:mug_qpos_addr + 7].copy()
            state["mug_pos"] = mug_pos
            state["mug_quat"] = mug_quat
            state["mug_euler"] = _euler_xyz_from_quat(mug_quat)
        else:
            state["mug_pos"] = None
            state["mug_quat"] = None
            state["mug_euler"] = None
        return state

    def _apply_adjustable_state(state: dict) -> None:
        if mug_qpos_addr >= 0 and state["mug_pos"] is not None and state["mug_quat"] is not None:
            data.qpos[mug_qpos_addr:mug_qpos_addr + 3] = state["mug_pos"]
            data.qpos[mug_qpos_addr + 3:mug_qpos_addr + 7] = _quat_normalize(state["mug_quat"])
        for obj_name, body_pos in state["body_positions"].items():
            body_id = adjustable_body_ids.get(obj_name)
            if body_id is not None:
                model.body_pos[body_id] = body_pos
                if obj_name in state.get("body_quats", {}):
                    model.body_quat[body_id] = _quat_normalize(state["body_quats"][obj_name])
        mujoco.mj_forward(model, data)

    def _warmup_status(state: dict, current_obj: str | None) -> str:
        if current_obj == "mug":
            return f"mug pos={_fmt_xyz(state['mug_pos'])} euler={_fmt_euler_deg(state['mug_euler'])}"
        if current_obj and current_obj in state["body_positions"]:
            return f"{current_obj} pos={_fmt_xyz(state['body_positions'][current_obj])}"
        return f"objects={', '.join(adjustable_object_names) or 'none'}"

    def _finalize_episode_window_recording(save_path: Path | None = None) -> None:
        nonlocal episode_window_writer, episode_window_tmp_path
        if episode_window_writer is not None:
            episode_window_writer.release()
            episode_window_writer = None
        if episode_window_tmp_path is None:
            return
        if save_path is None:
            try:
                episode_window_tmp_path.unlink(missing_ok=True)
            except Exception:
                pass
        else:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            if save_path.exists():
                save_path.unlink()
            episode_window_tmp_path.replace(save_path)
        episode_window_tmp_path = None

    def _restore_mujoco_snapshot(snapshot: dict[str, np.ndarray | float]) -> None:
        data.qpos[:] = snapshot["qpos"]
        data.qvel[:] = snapshot["qvel"]
        data.ctrl[:] = snapshot["ctrl"]
        data.time = float(snapshot["time"])
        mujoco.mj_forward(model, data)

    def _render_episode_videos_from_snapshots(
        snapshots: list[dict[str, np.ndarray | float]],
        ep_dir: Path,
    ) -> None:
        """Replay recorded sim states and write full per-frame camera videos."""
        if not snapshots:
            return
        current_snapshot = capture_mujoco_snapshot(data)
        writers: dict[str, cv2.VideoWriter] = {}
        combined_writer: cv2.VideoWriter | None = None
        combined_tmp_path = ep_dir / "combined_windows_tmp.mp4"
        try:
            for frame_idx, snapshot in enumerate(snapshots):
                _restore_mujoco_snapshot(snapshot)
                for cam_cfg in CAMERA_CONFIG.values():
                    if cam_cfg["config"].get("type", "stationary") == "stationary":
                        set_mujoco_camera_from_config(data, model, cam_cfg["mujoco_cam"], cam_cfg["config"])
                composite_obs = build_observation_from_mujoco(
                    model, data, renderer,
                    seg_renderer=seg_renderer,
                    robot_geom_ids=robot_geom_ids,
                    gaussian_data=gaussian_data,
                    obs_frames=None,
                    frame_idx=frame_idx,
                )
                for cam_key, cam_cfg in CAMERA_CONFIG.items():
                    obs_key = f"observation.images.{cam_cfg['dataset_cam']}"
                    frame = composite_obs.get(obs_key)
                    if not isinstance(frame, np.ndarray):
                        continue
                    writer = writers.get(cam_key)
                    if writer is None:
                        h, w = frame.shape[:2]
                        video_path = ep_dir / f"{cam_cfg['dataset_cam']}.mp4"
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        writer = cv2.VideoWriter(str(video_path), fourcc, args.fps, (w, h))
                        if not writer.isOpened():
                            raise RuntimeError(f"Failed to open video writer for {video_path}")
                        writers[cam_key] = writer
                    writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                if args.record:
                    combined_rgb = build_combined_window_frame(
                        [("Composite", composite_obs)],
                        tile_width=WINDOW_W,
                        tile_height=WINDOW_H,
                    )
                    combined_bgr = cv2.cvtColor(combined_rgb, cv2.COLOR_RGB2BGR)
                    if combined_writer is None:
                        combined_tmp_path.unlink(missing_ok=True)
                        frame_h, frame_w = combined_bgr.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        combined_writer = cv2.VideoWriter(
                            str(combined_tmp_path),
                            fourcc,
                            args.fps,
                            (frame_w, frame_h),
                        )
                        if not combined_writer.isOpened():
                            raise RuntimeError(f"Failed to open video writer for {combined_tmp_path}")
                    combined_writer.write(combined_bgr)
        finally:
            for writer in writers.values():
                writer.release()
            if combined_writer is not None:
                combined_writer.release()
                combined_path = ep_dir / "combined_windows.mp4"
                combined_path.unlink(missing_ok=True)
                combined_tmp_path.replace(combined_path)
            else:
                combined_tmp_path.unlink(missing_ok=True)
            _restore_mujoco_snapshot(current_snapshot)

    def _load_episode_snapshots(ep_dir: Path) -> list[dict[str, np.ndarray | float]]:
        snapshot_path = ep_dir / "sim_snapshots.npz"
        if not snapshot_path.exists():
            raise FileNotFoundError(f"Snapshot file not found for offline replay: {snapshot_path}")
        with np.load(str(snapshot_path)) as snapshot_data:
            return [
                {
                    "qpos": snapshot_data["qpos"][idx].copy(),
                    "qvel": snapshot_data["qvel"][idx].copy(),
                    "ctrl": snapshot_data["ctrl"][idx].copy(),
                    "time": float(snapshot_data["time"][idx]),
                }
                for idx in range(len(snapshot_data["time"]))
            ]

    def _select_warmup_object(key: int, current_obj: str | None) -> str | None:
        key_to_obj = {
            ord("m"): "mug",
            ord("r"): "rack",
            ord("p"): "saucer",
            ord("t"): "sticker",
            ord("b"): "table",
        }
        selected = key_to_obj.get(key, current_obj)
        if selected == "mug" and mug_qpos_addr < 0:
            return current_obj
        if selected != "mug" and selected not in adjustable_body_ids:
            return current_obj
        return selected

    def _render_current_view(warmup_contour=None, alpha=0.4, status_text: str | None = None, help_text: str | None = None):
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
            if status_text or help_text:
                for img_key, img in composite_obs.items():
                    if "image" not in img_key.lower() or not isinstance(img, np.ndarray):
                        continue
                    y = 26
                    if status_text:
                        cv2.putText(img, status_text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        y += 22
                    if help_text:
                        cv2.putText(img, help_text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1)
            display_camera_images(
                composite_obs,
                policy_config=policy.config,
                window_name_prefix="Composite",
                episode_idx=next_saved_episode_idx if output_dir is not None else None,
            )
            if obs_frames is not None and show_real_windows:
                real_obs = build_observation_from_mujoco(
                    model, data, renderer,
                    seg_renderer=seg_renderer, robot_geom_ids=robot_geom_ids,
                    gaussian_data=gaussian_data, obs_frames=obs_frames, frame_idx=0,
                )
                display_camera_images(
                    real_obs,
                    policy_config=policy.config,
                    window_name_prefix="Real",
                    episode_idx=next_saved_episode_idx if output_dir is not None else None,
                )
        if viewer is not None:
            viewer.sync()

    try:
        while completed_episodes < num_eval_episodes and not events["stop_recording"]:
            if output_dir is not None and current_episode_output_dir is None:
                next_saved_episode_idx, current_episode_output_dir = _reserve_episode_output_dir(output_dir)

            # ── WARMUP: reset sim and wait for RIGHT arrow to begin episode ──
            _reset_sim()
            policy.reset()

            # Pick the contour for this evaluation episode (sequential)
            warmup_contour = None
            selected_source_episode_idx = None
            if selected_contours is not None and completed_episodes < len(selected_contours):
                warmup_contour = selected_contours[completed_episodes]
                selected_source_episode_idx = selected_episode_indices[completed_episodes]
            if selected_source_episode_idx is None:
                selected_source_episode_idx = args.episode
            obs_frames = _load_obs_frames_for_episode(selected_source_episode_idx)

            events["right_arrow"] = False
            events["left_arrow"] = False
            events["rerecord_episode"] = False
            events["exit_early"] = False

            can_adjust_scene = bool(adjustable_object_names) and not args.headless
            auto_align_succeeded = False
            if auto_align_config is not None:
                try:
                    align_result = auto_align_object_poses(
                        model=model,
                        data=data,
                        seg_renderer=seg_renderer,
                        camera_config=CAMERA_CONFIG,
                        config=auto_align_config,
                        episode_idx=selected_source_episode_idx,
                        apply=True,
                    )
                    iou_text = ", ".join(
                        f"{name} IoU={iou:.3f}" for name, iou in align_result.iou_by_object.items()
                    )
                    print(
                        f"[INFO] Auto-aligned objects for training episode {selected_source_episode_idx}: "
                        f"loss={align_result.loss:.4f} {iou_text}",
                        flush=True,
                    )
                    auto_align_succeeded = True
                except Exception as exc:
                    print(f"[WARN] Automatic object alignment failed: {exc}", flush=True)
            warmup_state = _capture_adjustable_state()
            current_warmup_obj = adjustable_object_names[0] if adjustable_object_names else None
            warmup_help = (
                "Arrows: XY | w/s: Z | j/l: yaw | i/k: pitch | [ ]: roll | "
                "m/r/p/t: select | -/+: step"
            )

            # print(f"\n{'='*60}")
            # print(f"  WARMUP - Episode {completed_episodes + 1}/{num_eval_episodes}  "
                  # f"(eval #{episode_idx})")
            if warmup_contour:
                src_ep = selected_episode_indices[completed_episodes]
                # print(f"  Contour: selected_contours[{completed_episodes}]  "
                      # f"(training ep {src_ep})")
            if can_adjust_scene:
                # print(f"  Adjustable objects: {', '.join(adjustable_object_names)}")
                # print(f"  {warmup_help}")
                # print(f"  ENTER: start evaluation | ESC: quit")
                pass
            elif listener is not None:
                # print(f"  Press RIGHT to start evaluation, ESC to quit")
                pass
            # print(f"{'='*60}")

            if not args.headless:
                # ── cv2-based warmup (with optional contour overlay) ──
                mug_step = MUG_STEP_INIT_M
                _render_current_view(
                    warmup_contour=warmup_contour,
                    status_text=_warmup_status(warmup_state, current_warmup_obj),
                    help_text=warmup_help if can_adjust_scene else None,
                )
                if auto_align_succeeded:
                    cv2.waitKeyEx(1)

                while not auto_align_succeeded and not events["stop_recording"]:
                    key = cv2.waitKeyEx(50)
                    if key < 0:
                        continue

                    if key in (13, 10):  # ENTER → confirm & start
                        break
                    if key == 27:  # ESC → quit
                        events["stop_recording"] = True
                        break

                    if not can_adjust_scene:
                        continue

                    previous_obj = current_warmup_obj
                    current_warmup_obj = _select_warmup_object(key, current_warmup_obj)
                    if current_warmup_obj != previous_obj:
                        # print(f"[INFO] Selected object: {current_warmup_obj}")
                        pass

                    moved = False
                    rotated = False
                    if key in _KEY_LEFT and current_warmup_obj == "mug" and warmup_state["mug_pos"] is not None:
                        warmup_state["mug_pos"][0] -= mug_step; moved = True
                    elif key in _KEY_RIGHT and current_warmup_obj == "mug" and warmup_state["mug_pos"] is not None:
                        warmup_state["mug_pos"][0] += mug_step; moved = True
                    elif key in _KEY_UP and current_warmup_obj == "mug" and warmup_state["mug_pos"] is not None:
                        warmup_state["mug_pos"][1] += mug_step; moved = True
                    elif key in _KEY_DOWN and current_warmup_obj == "mug" and warmup_state["mug_pos"] is not None:
                        warmup_state["mug_pos"][1] -= mug_step; moved = True
                    elif current_warmup_obj in warmup_state["body_positions"]:
                        if key in _KEY_LEFT:
                            warmup_state["body_positions"][current_warmup_obj][0] -= mug_step; moved = True
                        elif key in _KEY_RIGHT:
                            warmup_state["body_positions"][current_warmup_obj][0] += mug_step; moved = True
                        elif key in _KEY_UP:
                            warmup_state["body_positions"][current_warmup_obj][1] += mug_step; moved = True
                        elif key in _KEY_DOWN:
                            warmup_state["body_positions"][current_warmup_obj][1] -= mug_step; moved = True
                    if key == ord('w'):
                        if current_warmup_obj == "mug" and warmup_state["mug_pos"] is not None:
                            warmup_state["mug_pos"][2] += mug_step; moved = True
                        elif current_warmup_obj in warmup_state["body_positions"]:
                            warmup_state["body_positions"][current_warmup_obj][2] += mug_step; moved = True
                    elif key == ord('s'):
                        if current_warmup_obj == "mug" and warmup_state["mug_pos"] is not None:
                            warmup_state["mug_pos"][2] -= mug_step; moved = True
                        elif current_warmup_obj in warmup_state["body_positions"]:
                            warmup_state["body_positions"][current_warmup_obj][2] -= mug_step; moved = True
                    elif current_warmup_obj == "mug" and warmup_state["mug_euler"] is not None:
                        if key == ord('j'):
                            warmup_state["mug_euler"][2] -= MUG_ROT_STEP_RAD; rotated = True
                        elif key == ord('l'):
                            warmup_state["mug_euler"][2] += MUG_ROT_STEP_RAD; rotated = True
                        elif key == ord('i'):
                            warmup_state["mug_euler"][1] += MUG_ROT_STEP_RAD; rotated = True
                        elif key == ord('k'):
                            warmup_state["mug_euler"][1] -= MUG_ROT_STEP_RAD; rotated = True
                        elif key == ord('['):
                            warmup_state["mug_euler"][0] -= MUG_ROT_STEP_RAD; rotated = True
                        elif key == ord(']'):
                            warmup_state["mug_euler"][0] += MUG_ROT_STEP_RAD; rotated = True

                    if key in (ord('-'), ord('_')):
                        mug_step /= 2.0
                        # print(f"[INFO] Step: {mug_step*1000:.2f} mm")
                    elif key in (ord('+'), ord('=')):
                        mug_step = min(mug_step * 2.0, MUG_STEP_INIT_M)
                        # print(f"[INFO] Step: {mug_step*1000:.2f} mm")

                    if rotated:
                        warmup_state["mug_quat"] = _quat_from_euler_xyz(warmup_state["mug_euler"])
                    if moved or rotated:
                        _apply_adjustable_state(warmup_state)
                        # print(f"[INFO] {_warmup_status(warmup_state, current_warmup_obj)}  step={mug_step*1000:.2f}mm")
                    if moved or rotated or current_warmup_obj != previous_obj or key in (ord('-'), ord('_'), ord('+'), ord('=')):
                        _render_current_view(
                            warmup_contour=warmup_contour,
                            status_text=_warmup_status(warmup_state, current_warmup_obj),
                            help_text=warmup_help,
                        )

                # Clear stale pynput events from arrow keys
                events["right_arrow"] = False
                events["left_arrow"] = False
                events["rerecord_episode"] = False
                events["exit_early"] = False

                if events["stop_recording"]:
                    break

            elif listener is not None:
                # ── pynput-based warmup (no contour selection) ──
                while (not auto_align_succeeded
                       and not events["right_arrow"]
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

            # print(f"\n{'='*60}")
            # print(f"  RECORDING episode {episode_idx}")
            if listener is not None:
                # print(f"  RIGHT=save | LEFT=discard | ESC=quit")
                pass
            else:
                # print(f"  (headless: auto-save after {args.max_steps} steps)")
                pass
            # print(f"{'='*60}")

            episode_actions = []
            episode_states = []
            episode_frames = {cam_key: [] for cam_key in CAMERA_CONFIG}
            episode_snapshots: list[dict[str, np.ndarray | float]] = []
            prediction_event_panels: list[tuple[str, int, int, np.ndarray]] = []
            last_gpt_policy_display_obs: dict | None = None
            last_turbo_policy_display_obs: dict | None = None
            _finalize_episode_window_recording()
            step = 0
            episode_discarded = False
            prediction_events = 0
            prediction_limit_reached = False

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

                # Fast replay mode can skip camera rendering while a policy action
                # chunk still has cached raw actions available.
                needs_prediction = policy_needs_prediction(policy)
                cached_action = None
                used_cached_action = False
                if args.fast_rollout_video_replay and not needs_prediction:
                    with torch.inference_mode():
                        cached_action = pop_cached_policy_action(policy, postprocessor)
                    if cached_action is not None:
                        used_cached_action = True
                    else:
                        needs_prediction = True
                current_prediction_event_idx = prediction_events + 1 if needs_prediction else None
                if needs_prediction and prediction_events >= max_prediction_events_per_trajectory:
                    prediction_limit_reached = True
                    break

                real_obs = None
                composite_obs = None
                resize_policy_images = (
                    not args.policy_no_resize
                    and args.policy_input_h > 0
                    and args.policy_input_w > 0
                )
                composite_display_obs = None
                observation = None
                gpt_obs_refresh = False
                turbo_obs_refresh = False

                if not used_cached_action:
                    need_real_obs = (
                        args.obs or args.obs_eval
                        or (obs_frames is not None and show_real_windows)
                    )
                    if need_real_obs:
                        real_obs = build_observation_from_mujoco(
                            model, data, renderer,
                            seg_renderer=seg_renderer,
                            robot_geom_ids=robot_geom_ids,
                            gaussian_data=gaussian_data,
                            obs_frames=obs_frames,
                            frame_idx=step,
                            gripper_observation_offset_mm=args.sim_gripper_observation_offset_mm,
                            gripper_observation_offset_mode=args.sim_gripper_observation_offset_mode,
                            gripper_observation_contact_geom_ids=gripper_observation_contact_geom_ids,
                            gripper_observation_threshold_mm=args.sim_gripper_observation_threshold_mm,
                        )

                    need_composite_obs = not (args.obs or args.obs_eval)
                    if not args.fast_rollout_video_replay:
                        need_composite_obs = True
                    if need_composite_obs:
                        composite_obs = build_observation_from_mujoco(
                            model, data, renderer,
                            seg_renderer=seg_renderer,
                            robot_geom_ids=robot_geom_ids,
                            gaussian_data=gaussian_data,
                            obs_frames=None,
                            frame_idx=step,
                            gripper_observation_offset_mm=args.sim_gripper_observation_offset_mm,
                            gripper_observation_offset_mode=args.sim_gripper_observation_offset_mode,
                            gripper_observation_contact_geom_ids=gripper_observation_contact_geom_ids,
                            gripper_observation_threshold_mm=args.sim_gripper_observation_threshold_mm,
                        )
                        composite_display_obs = composite_obs
                        if resize_policy_images:
                            composite_display_obs = copy_observation_frame_with_resized_images(
                                composite_obs, args.policy_input_h, args.policy_input_w
                            )

                    if not args.headless:
                        if real_obs is not None and show_real_windows:
                            display_camera_images(
                                real_obs,
                                policy_config=policy.config,
                                window_name_prefix="Real",
                                episode_idx=next_saved_episode_idx if output_dir is not None else None,
                            )
                        if composite_display_obs is not None:
                            # Update composite windows before any GPT/Turbo call so
                            # they snap to policy resolution after warmup confirm.
                            display_camera_images(
                                composite_display_obs,
                                policy_config=policy.config,
                                window_name_prefix="Composite",
                                episode_idx=next_saved_episode_idx if output_dir is not None else None,
                            )

                    # Policy input: real obs, GPT composite, turbo composite
                    # (when action queue empty), or raw composite.
                    if args.obs or args.obs_eval:
                        observation = real_obs
                    elif gpt_translators is not None and needs_prediction:
                        if composite_obs is None:
                            raise RuntimeError("GPT fast rollout needs a composite observation on prediction steps.")
                        _update_gpt_style_references(gpt_translators, gpt_style_dirs, _gpt_style_frame_cache, step)
                        observation = apply_gpt_per_camera_parallel(gpt_translators, composite_obs, frame_idx=step)
                        gpt_obs_refresh = True
                    elif turbo_translators is not None and needs_prediction:
                        if composite_obs is None:
                            raise RuntimeError("Turbo fast rollout needs a composite observation on prediction steps.")
                        observation = apply_turbo_per_camera(turbo_translators, composite_obs)
                        turbo_obs_refresh = True
                    else:
                        observation = composite_obs

                    if observation is None:
                        raise RuntimeError("No observation was built for a fresh policy prediction.")

                    if hasattr(policy.config, 'language_features') and policy.config.language_features:
                        observation["observation.language"] = args.prompt

                    obs_for_policy = observation
                    if resize_policy_images:
                        obs_for_policy = copy_observation_frame_with_resized_images(
                            observation, args.policy_input_h, args.policy_input_w
                        )

                    if gpt_translators is not None and gpt_obs_refresh:
                        last_gpt_policy_display_obs = {
                            k: np.ascontiguousarray(obs_for_policy[k].copy())
                            for k in obs_for_policy
                            if "image" in k.lower() and isinstance(obs_for_policy[k], np.ndarray)
                        }
                    if turbo_translators is not None and turbo_obs_refresh:
                        last_turbo_policy_display_obs = {
                            k: np.ascontiguousarray(obs_for_policy[k].copy())
                            for k in obs_for_policy
                            if "image" in k.lower() and isinstance(obs_for_policy[k], np.ndarray)
                        }
                    if output_dir is not None and current_prediction_event_idx is not None and composite_display_obs is not None:
                        if gpt_obs_refresh and last_gpt_policy_display_obs is not None:
                            prediction_event_panels.append(
                                (
                                    "gpt",
                                    current_prediction_event_idx,
                                    step,
                                    build_prediction_event_panel(
                                        "GPT",
                                        composite_display_obs,
                                        last_gpt_policy_display_obs,
                                        tile_width=WINDOW_W,
                                        tile_height=WINDOW_H,
                                    ),
                                )
                            )
                        if turbo_obs_refresh and last_turbo_policy_display_obs is not None:
                            prediction_event_panels.append(
                                (
                                    "turbo",
                                    current_prediction_event_idx,
                                    step,
                                    build_prediction_event_panel(
                                        "Turbo",
                                        composite_display_obs,
                                        last_turbo_policy_display_obs,
                                        tile_width=WINDOW_W,
                                        tile_height=WINDOW_H,
                                    ),
                                )
                            )
                        # Always also save a plain composite-only image (cam_high + cam_wrist,
                        # no translation/blend) so every baseline — raw sim, --color-calibrate,
                        # --turbo, --gpt — has a directly comparable per-prediction snapshot for
                        # the cross-baseline last-state grid, without touching the gpt_*/turbo_*
                        # panels above.
                        prediction_event_panels.append(
                            (
                                "composite",
                                current_prediction_event_idx,
                                step,
                                build_combined_window_frame(
                                    [("Composite", composite_display_obs)],
                                    tile_width=WINDOW_W,
                                    tile_height=WINDOW_H,
                                ),
                            )
                        )

                    # Display resized policy input (same tensor as predict_action)
                    if not args.headless:
                        if gpt_translators is not None and last_gpt_policy_display_obs is not None:
                            display_camera_images(
                                last_gpt_policy_display_obs,
                                policy_config=policy.config,
                                window_name_prefix="GPT",
                                episode_idx=next_saved_episode_idx if output_dir is not None else None,
                            )
                        if turbo_translators is not None and last_turbo_policy_display_obs is not None:
                            display_camera_images(
                                last_turbo_policy_display_obs,
                                policy_config=policy.config,
                                window_name_prefix="Turbo",
                                episode_idx=next_saved_episode_idx if output_dir is not None else None,
                            )
                    if args.record and output_dir is not None and not args.fast_rollout_video_replay:
                        display_rows: list[tuple[str, dict | None]] = []
                        if real_obs is not None and show_real_windows:
                            display_rows.append(("Real", real_obs))
                        display_rows.append(("Composite", composite_display_obs))
                        if gpt_translators is not None:
                            display_rows.append(("GPT", last_gpt_policy_display_obs))
                        if turbo_translators is not None:
                            display_rows.append(("Turbo", last_turbo_policy_display_obs))
                        combined_rgb = build_combined_window_frame(
                            display_rows,
                            tile_width=WINDOW_W,
                            tile_height=WINDOW_H,
                        )
                        combined_bgr = cv2.cvtColor(combined_rgb, cv2.COLOR_RGB2BGR)
                        if episode_window_writer is None:
                            if current_episode_output_dir is None:
                                raise RuntimeError("Episode output directory was not reserved before recording.")
                            episode_window_tmp_path = current_episode_output_dir / "combined_windows_tmp.mp4"
                            episode_window_tmp_path.unlink(missing_ok=True)
                            frame_h, frame_w = combined_bgr.shape[:2]
                            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                            episode_window_writer = cv2.VideoWriter(
                                str(episode_window_tmp_path),
                                fourcc,
                                args.fps,
                                (frame_w, frame_h),
                            )
                            if not episode_window_writer.isOpened():
                                raise RuntimeError(f"Failed to open video writer for {episode_window_tmp_path}")
                        episode_window_writer.write(combined_bgr)

                    with torch.inference_mode():
                        if needs_prediction:
                            prediction_events += 1
                            raw_state_for_debug = build_state_from_mujoco(model, data)
                            policy_state_for_debug = obs_for_policy.get(OBS_STATE)
                            if isinstance(policy_state_for_debug, torch.Tensor):
                                policy_state_for_debug = policy_state_for_debug.detach().cpu().numpy()
                            policy_state_for_debug = np.asarray(policy_state_for_debug)
                            raw_gripper = float(raw_state_for_debug[7])
                            policy_gripper = float(policy_state_for_debug.reshape(-1)[7])
                            offset_applied = abs(policy_gripper - raw_gripper) > 1e-6
                            print(
                                f"[GRIPPER OBS] prediction={prediction_events} step={step} "
                                f"raw_qpos_mm={raw_gripper:.3f} "
                                f"policy_mm={policy_gripper:.3f} "
                                f"offset_applied={offset_applied}"
                            )
                        action = predict_action(
                            obs_for_policy,
                            policy,
                            device,
                            preprocessor,
                            postprocessor,
                            policy.config.use_amp,
                            task=args.prompt,
                            robot_type="xarm_follower",
                        )
                else:
                    action = cached_action

                snapshot = capture_mujoco_snapshot(data)
                if args.fast_rollout_video_replay:
                    episode_snapshots.append(snapshot)

                if composite_obs is not None:
                    # Record composite frames (fresh every step in normal mode).
                    for _cam_key, _cam_cfg in CAMERA_CONFIG.items():
                        _obs_key = f"observation.images.{_cam_cfg['dataset_cam']}"
                        if _obs_key in composite_obs and not args.fast_rollout_video_replay:
                            episode_frames[_cam_key].append(composite_obs[_obs_key].copy())
                    state_for_episode = composite_obs[OBS_STATE]
                else:
                    state_for_episode = build_state_from_mujoco(
                        model,
                        data,
                        gripper_observation_offset_mm=args.sim_gripper_observation_offset_mm,
                        gripper_observation_offset_mode=args.sim_gripper_observation_offset_mode,
                        gripper_observation_contact_geom_ids=gripper_observation_contact_geom_ids,
                        gripper_observation_threshold_mm=args.sim_gripper_observation_threshold_mm,
                    )
                episode_states.append(
                    state_for_episode.copy()
                    if isinstance(state_for_episode, np.ndarray)
                    else state_for_episode
                )

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
                if sleep_time > 0 and not args.fast_rollout_video_replay:
                    time.sleep(sleep_time)

                step += 1
                if step % 100 == 0:
                    # print(f"[INFO] Episode {episode_idx} - Step {step}/{args.max_steps}")
                    pass

            # ── Post-episode: save or discard ──
            if episode_discarded:
                _finalize_episode_window_recording()
                _delete_episode_output_dir(current_episode_output_dir)
                current_episode_output_dir = None
                next_saved_episode_idx = None
                events["left_arrow"] = False
                events["rerecord_episode"] = False
                events["exit_early"] = False
                # print(f">>> Episode {episode_idx} DISCARDED ({step} steps)")
            elif events["stop_recording"] and not prediction_limit_reached:
                _finalize_episode_window_recording()
                # print(f"\n[INFO] ESC pressed, stopping evaluation")
                break
            else:
                events["right_arrow"] = False
                events["exit_early"] = False
                if prediction_limit_reached:
                    reason = f"prediction limit reached ({max_prediction_events_per_trajectory})"
                else:
                    reason = "max steps reached" if step >= args.max_steps else "RIGHT pressed"
                saved_episodes.append({
                    "episode": episode_idx,
                    "steps": step,
                    "actions": episode_actions,
                    "states": episode_states,
                })
                if output_dir is not None:
                    if current_episode_output_dir is None:
                        raise RuntimeError("Episode output directory was not reserved before saving.")
                    # Save episode data (states, actions, camera videos) to output directory
                    ep_dir = current_episode_output_dir
                    np.save(str(ep_dir / "states.npy"), np.array(episode_states))
                    np.save(str(ep_dir / "actions.npy"), np.array(episode_actions))
                    if args.fast_rollout_video_replay and episode_snapshots:
                        np.savez_compressed(
                            str(ep_dir / "sim_snapshots.npz"),
                            qpos=np.stack([snapshot["qpos"] for snapshot in episode_snapshots]),
                            qvel=np.stack([snapshot["qvel"] for snapshot in episode_snapshots]),
                            ctrl=np.stack([snapshot["ctrl"] for snapshot in episode_snapshots]),
                            time=np.array([snapshot["time"] for snapshot in episode_snapshots], dtype=np.float64),
                        )
                    if prediction_event_panels:
                        prediction_dir = ep_dir / "prediction_events"
                        prediction_dir.mkdir(parents=True, exist_ok=True)
                        for _mode_name, _event_idx, _step_idx, _panel_rgb in prediction_event_panels:
                            _panel_path = prediction_dir / (
                                f"{_mode_name}_prediction_event_{_event_idx:02d}_step_{_step_idx:04d}.png"
                            )
                            cv2.imwrite(
                                str(_panel_path),
                                cv2.cvtColor(_panel_rgb, cv2.COLOR_RGB2BGR),
                            )
                    if args.fast_rollout_video_replay:
                        pending_fast_replay_dirs.append(ep_dir)
                    else:
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
                            # print(f"[INFO] Saved {len(_frames)}-frame video: {_video_path}")
                    if args.record and not args.fast_rollout_video_replay:
                        _finalize_episode_window_recording(ep_dir / "combined_windows.mp4")
                    print(f"[INFO] Episode data saved ({reason}) → {ep_dir.resolve()}", flush=True)
                    current_episode_output_dir = None
                    next_saved_episode_idx = None
                else:
                    _finalize_episode_window_recording()
                    print(f"[INFO] Episode completed ({reason}) with sim eval disk output disabled.", flush=True)
                completed_episodes += 1
                _save_note = "" if output_dir is not None else " (no disk save)"
                # print(f">>> Episode {episode_idx} SAVED - {reason} "
                      # f"({step} steps, {completed_episodes}/{num_eval_episodes} done){_save_note}")

            episode_idx += 1

        if args.fast_rollout_video_replay and pending_fast_replay_dirs:
            print(
                f"[INFO] Rendering offline videos for {len(pending_fast_replay_dirs)} completed episode(s)...",
                flush=True,
            )
            for replay_ep_dir in pending_fast_replay_dirs:
                snapshots = _load_episode_snapshots(replay_ep_dir)
                _render_episode_videos_from_snapshots(snapshots, replay_ep_dir)
                print(f"[INFO] Offline videos rendered → {replay_ep_dir.resolve()}", flush=True)

    except KeyboardInterrupt:
        # print("\n[INFO] Interrupted by user")
        pass
    finally:
        _finalize_episode_window_recording()
        _delete_episode_output_dir(current_episode_output_dir)
        if listener is not None:
            listener.stop()
        if viewer_ctx is not None:
            try:
                viewer_ctx.__exit__(None, None, None)
            except Exception:
                pass
        if not args.headless:
            cv2.destroyAllWindows()
        _fin = "episodes saved" if output_dir is not None else "episodes completed (no sim eval data saved)"
        # print(f"[INFO] Evaluation finished: {completed_episodes}/{num_eval_episodes} {_fin}")

    if output_dir is not None:
        try:
            grid_path = build_last_episode_state_grid(output_dir)
            if grid_path is not None:
                print(f"[INFO] Saved last-episode-state grid → {grid_path.resolve()}", flush=True)
            else:
                print("[WARN] No episode videos found; skipped last-episode-state grid.", flush=True)
        except Exception as e:
            print(f"[WARN] Failed to build last-episode-state grid: {e}", flush=True)


if __name__ == "__main__":
    main()
