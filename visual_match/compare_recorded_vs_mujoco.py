#!/usr/bin/env python3
# compare_recorded_vs_mujoco.py
"""
Side-by-side comparison of recorded xArm dataset video and MuJoCo simulation.

Shows synchronized windows in a 2-row layout:
Row 1: Stationary camera - Recorded, Composite, Alpha
Row 2: Wrist camera - Recorded, Composite, Alpha

xArm observation.state format (8-dim):
  [joint1..7 in degrees, gripper in mm (0=closed, 800=open)]

Usage:
    python visual_match/compare_recorded_vs_mujoco.py
"""

import sys
import os
import re
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from camera_config import load_camera_config, set_mujoco_camera_from_config
from composite_rendering import (
    get_robot_geom_ids,
    get_mujoco_camera_pose,
    load_scene_data,
    mj_pose_to_gaussian_w2c,
    render,
    setup_camera,
    shift_for_principal_point,
    T_splat2mj,
)

# ============================================================================
# Camera Configuration — loaded from configs/ JSON files
# ============================================================================
_stationary_cfg = load_camera_config("stationary_cam")
_wrist_cfg = load_camera_config("wrist_cam")
SAVE_CALIB_FRAMES = [20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200]
_DEFAULT_EPISODE = 1
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
# Display detection
# ============================================================================
def _detect_display():
    if os.environ.get("DISPLAY"):
        return True
    if os.environ.get("WAYLAND_DISPLAY"):
        return True
    if sys.platform in ("darwin", "win32"):
        return True
    return False

_HAS_DISPLAY = _detect_display()
# Only exit when run as main (not when imported by deploy_act_policy_mujoco --headless)
if not _HAS_DISPLAY and __name__ == "__main__":
    print("[ERROR] This script requires a display for visualization")
    sys.exit(1)

import numpy as np
import torch
import torch.nn.functional as F
import cv2
try:
    import open3d as o3d
    _HAS_OPEN3D = True
except ImportError:
    _HAS_OPEN3D = False

import mujoco
from mujoco import MjModel, MjData
try:
    import mujoco.viewer
    _HAS_MJ_VIEWER = True
except ImportError:
    _HAS_MJ_VIEWER = False

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.video_utils import decode_video_frames
from lerobot.tasks import get_task_profile, get_task_profiles, resolve_task_scene_xml

# ============================================================================
# xArm conversion (imported from shared utils)
# ============================================================================
from lerobot_mujoco_utils import GRIPPER_OPEN_MM, lerobot_state_to_mujoco_ctrl

_DEFAULT_RECORD_TASK_ID ="hang_mug"  # Keep in sync with lerobot-record defaults.

# ============================================================================
# Forward Kinematics comparison utilities
# ============================================================================

def rotation_matrix_to_euler(R: np.ndarray) -> np.ndarray:
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = 0
    return np.array([roll, pitch, yaw])


def get_end_effector_pose(model, data, site_name: str, run_forward: bool = False) -> tuple:
    """
    Get end effector pose.  If run_forward=True, call mj_forward first
    (needed for freshly-created MjData; NOT needed after mj_step).
    """
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if site_id == -1:
        raise ValueError(f"Site '{site_name}' not found in model")
    if run_forward:
        mujoco.mj_forward(model, data)
    pos = data.site_xpos[site_id].copy()
    rot_mat = data.site_xmat[site_id].reshape(3, 3).copy()
    return pos, rot_mat


def compute_pose_difference(pos1, rot1, pos2, rot2):
    trans_diff_vec = pos1 - pos2
    trans_norm = np.linalg.norm(trans_diff_vec)
    trans_diff = np.array([*trans_diff_vec, trans_norm])
    R_diff = rot1 @ rot2.T
    trace = np.clip(np.trace(R_diff), -1.0, 3.0)
    angle = np.arccos((trace - 1) / 2)
    euler_diff = rotation_matrix_to_euler(R_diff)
    rot_diff = np.array([*euler_diff, angle])
    return trans_diff, rot_diff


def _fig_to_bgr_overlay(fig, target_w: int, target_h: int) -> np.ndarray:
    """Render matplotlib figure to BGR numpy array for OpenCV overlay."""
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    # buf is (H, W, 4) RGBA
    buf_bgr = cv2.cvtColor(buf[:, :, :3], cv2.COLOR_RGB2BGR)
    return cv2.resize(buf_bgr, (target_w, target_h))


def _build_camera_frustum_local_points(K: np.ndarray, width: int, height: int,
                                       depth: float = 0.18) -> np.ndarray:
    """Build 5 local frustum points (origin + 4 image-corner rays at fixed depth)."""
    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])
    corners = np.array([
        [0.0, 0.0],
        [float(width), 0.0],
        [float(width), float(height)],
        [0.0, float(height)],
    ], dtype=np.float64)
    xy = np.empty((4, 2), dtype=np.float64)
    xy[:, 0] = (corners[:, 0] - cx) / fx * depth
    xy[:, 1] = (corners[:, 1] - cy) / fy * depth
    pts = np.zeros((5, 3), dtype=np.float64)
    pts[1:, 0] = xy[:, 0]
    pts[1:, 1] = xy[:, 1]
    pts[1:, 2] = depth
    return pts


def _transform_points(T: np.ndarray, pts_local: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    return (pts_local @ R.T) + t[None, :]


def _make_camera_frustum_lineset(local_pts: np.ndarray, pose: np.ndarray,
                                 color=(1.0, 0.6, 0.0)):
    pts_world = _transform_points(pose, local_pts)
    lines = [
        [0, 1], [0, 2], [0, 3], [0, 4],
        [1, 2], [2, 3], [3, 4], [4, 1],
    ]
    colors = [list(color)] * len(lines)
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(pts_world)
    ls.lines = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector(colors)
    return ls


def _update_frame_mesh_pose(frame_mesh, prev_pose: np.ndarray, new_pose: np.ndarray):
    delta = new_pose @ np.linalg.inv(prev_pose)
    frame_mesh.transform(delta)


def _rotation_matrix_axis(axis: int, angle_deg: float) -> np.ndarray:
    """3x3 rotation about axis (0=X, 1=Y, 2=Z)."""
    a = np.radians(angle_deg)
    c, s = np.cos(a), np.sin(a)
    if axis == 0:
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    elif axis == 1:
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    else:
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _make_cam_increment(tx=0.0, ty=0.0, tz=0.0, rx=0.0, ry=0.0, rz=0.0):
    """Build a 4x4 incremental camera-local transform. Rotation args in degrees."""
    T = np.eye(4)
    T[:3, 3] = [tx, ty, tz]
    R = np.eye(3)
    for axis, angle in enumerate([rx, ry, rz]):
        if abs(angle) > 1e-9:
            R = _rotation_matrix_axis(axis, angle) @ R
    T[:3, :3] = R
    return T


def plot_fk_comparison(trans_errors, rot_errors, fps, save_path=None):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not installed, skipping plot.")
        return

    num_frames = len(trans_errors)
    time_s = np.arange(num_frames) / fps

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    ax1 = axes[0]
    ax1.plot(time_s, trans_errors[:, 0] * 1000, 'r-', label='dx', alpha=0.7)
    ax1.plot(time_s, trans_errors[:, 1] * 1000, 'g-', label='dy', alpha=0.7)
    ax1.plot(time_s, trans_errors[:, 2] * 1000, 'b-', label='dz', alpha=0.7)
    ax1.plot(time_s, trans_errors[:, 3] * 1000, 'k-', label='||d||', linewidth=2)
    ax1.set_ylim(-20, 20)
    ax1.set_autoscale_on(False)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Translation Error (mm)')
    ax1.set_title('End Effector Translation Error: Real vs Simulated')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    mean_norm = np.mean(trans_errors[:, 3]) * 1000
    max_norm = np.max(trans_errors[:, 3]) * 1000
    ax1.text(0.02, 0.98, f'Mean: {mean_norm:.2f} mm\nMax: {max_norm:.2f} mm',
             transform=ax1.transAxes, va='top', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax2 = axes[1]
    ax2.plot(time_s, np.degrees(rot_errors[:, 0]), 'r-', label='roll', alpha=0.7)
    ax2.plot(time_s, np.degrees(rot_errors[:, 1]), 'g-', label='pitch', alpha=0.7)
    ax2.plot(time_s, np.degrees(rot_errors[:, 2]), 'b-', label='yaw', alpha=0.7)
    ax2.plot(time_s, np.degrees(rot_errors[:, 3]), 'k-', label='angle', linewidth=2)
    ax2.set_ylim(-20, 20)
    ax2.set_autoscale_on(False)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Rotation Error (degrees)')
    ax2.set_title('End Effector Rotation Error: Real vs Simulated')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    mean_angle = np.degrees(np.mean(rot_errors[:, 3]))
    max_angle = np.degrees(np.max(rot_errors[:, 3]))
    ax2.text(0.02, 0.98, f'Mean: {mean_angle:.2f}°\nMax: {max_angle:.2f}°',
             transform=ax2.transAxes, va='top', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[INFO] Saved FK comparison plot to: {save_path}")
    plt.show()




# ============================================================================
# Color calibration
# ============================================================================

def _get_aug(x, add_ones=False):
    if add_ones:
        return np.hstack([x ** 2, x, np.ones((x.shape[0], 1), np.float64)])
    return np.hstack([x ** 2, x])


def load_color_mapping(yaml_path):
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
    # Support affine (3x3, 9 values) or quadratic (3x6, 18 values)
    if len(a_values) == 9:
        A = np.array(a_values, dtype=np.float32).reshape(3, 3)
        return ("affine", A, b)
    if len(a_values) == 18:
        A = np.array(a_values, dtype=np.float32).reshape(3, 6)
        return ("quadratic", A, b)
    raise ValueError(f"color_A must have 9 (affine) or 18 (quadratic) values, got {len(a_values)}")


def apply_color_transform(img, calib):
    """Apply color transform. calib is (fmt, A, b) from load_color_mapping."""
    fmt, A, b = calib
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    flat = img_rgb.reshape(-1, 3).astype(np.float32) / 255.0
    if fmt == "affine":
        out = flat @ A.T + b
    else:
        flat_aug = _get_aug(flat)
        out = flat_aug @ A.T + b
    out = np.clip(out, 0.0, 1.0)
    out_rgb = (out.reshape(img_rgb.shape) * 255.0).astype(np.uint8)
    return cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)


# ============================================================================
# Dataset loader (v3.0)
# ============================================================================

def load_episode(dataset_path: str, episode_idx: int, dataset_root: str | None = None):
    print(f"[INFO] Loading episode {episode_idx} from dataset: {dataset_path}")

    dataset_path_obj = Path(dataset_path)
    if dataset_path_obj.exists() and dataset_path_obj.is_dir():
        meta_info = dataset_path_obj / "meta" / "info.json"
        if meta_info.exists():
            repo_id = dataset_path_obj.name
            dataset = LeRobotDataset(
                repo_id=repo_id,
                root=dataset_path_obj,
                episodes=[episode_idx],
                video_backend="pyav"
            )
        else:
            repo_id = dataset_path_obj.name
            root = dataset_path_obj.parent
            dataset = LeRobotDataset(
                repo_id=repo_id,
                root=root,
                episodes=[episode_idx],
                video_backend="pyav"
            )
    else:
        dataset = LeRobotDataset(
            repo_id=dataset_path,
            root=dataset_root,
            episodes=[episode_idx],
            video_backend="pyav"
        )

    total_episodes = dataset.meta.total_episodes
    if episode_idx >= total_episodes:
        raise ValueError(f"Episode {episode_idx} not found. Dataset has {total_episodes} episodes")

    ep_meta = dataset.meta.episodes[episode_idx]
    num_frames = len(dataset)

    print(f"[INFO] Episode {episode_idx} has {num_frames} frames (loaded dataset size: {num_frames})")

    actions = []
    observations = []
    for frame_idx in range(num_frames):
        sample = dataset[frame_idx]
        actions.append(sample["action"].numpy())
        observations.append(sample["observation.state"].numpy())

    video_start_frame = ep_meta.get("dataset_from_index", 0)
    return {
        'action': torch.from_numpy(np.array(actions)),
        'observation.state': torch.from_numpy(np.array(observations)),
        'episode_index': episode_idx,
        'num_frames': num_frames,
        'video_start_frame': video_start_frame,
        'dataset': dataset,
    }


# ============================================================================
# Main
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Compare recorded xArm video with MuJoCo replay + composite")
    p.add_argument("--task-id", type=str, choices=sorted(get_task_profiles()),
                   default=_DEFAULT_RECORD_TASK_ID,
                   help="Task ID used to pick task-specific defaults such as dataset path and MuJoCo scene XML.")
    p.add_argument("--dataset-path", type=str, default=None,
                   help="Path to dataset directory (local) or repo_id (Hub). Defaults to the selected task dataset root.")
    p.add_argument("--dataset-root", type=str, default=None)
    p.add_argument("--episode", type=int, default=_DEFAULT_EPISODE)
    p.add_argument("--fps", type=float, default=60.0)
    p.add_argument("--scene-path", type=str, default="pointclouds/xarm7_black.npz",
                   help="Path to Gaussian Splatting scene file")
    p.add_argument("--alpha", type=float, default=0.5,
                   help="Alpha for blending (0=fully real, 1=fully robot)")
    p.add_argument("--color-calibrate", action="store_true",
                   help="Apply color calibration to composite renderings")
    p.add_argument("--no_stack", action="store_true",
                   help="Disable alpha blending plot overlay and window.")
    p.add_argument("--save-calibration-pairs", action="store_true",
                   help="Save frames 0,5,10,15,20 to calibration_pairs_*/ for color calibration")
    p.add_argument("--save-replay-frames", action="store_true",
                   help="Every replay frame: save composite_raw under <root>/gs_render/{stationary,wrist}/ "
                        "and recorded video under <root>/real_captures/{stationary,wrist}/")
    p.add_argument("--replay-export-root", type=str, default=None,
                   help="Root directory used with --save-replay-frames "
                        "(default: selected task's dataset_root_480640 from task_profiles)")
    p.add_argument("--cma-params", type=str, default="cma_result.pkl",
                   help="Path to cma_result.pkl for optimised stiffness/damping")
    p.add_argument("--cma", action="store_true",default=False,
                   help="Apply CMA-ES optimised parameters to the model.")
    p.add_argument("--no-mujoco-view", action="store_true",
                   help="Disable the MuJoCo 3D interactive viewer window")
    p.add_argument("--open3d-cam-view", action="store_true",
                   help="Open a lightweight Open3D viewer showing simulated camera pose(s)")
    p.add_argument("--open3d-cam", type=str, default="stationary", choices=["stationary", "wrist", "both"],
                   help="Which simulated camera pose(s) to display in Open3D")
    p.add_argument("--open3d-cam-save-path", type=str, default="camera_adjust.npy",
                   help="Path to save camera adjustment delta transform (.npy)")
    p.add_argument("--load-camera-adjust", action="store_true",
                   help="Load camera adjustment delta from --open3d-cam-save-path if it exists")
    return p.parse_args()


def main():
    args = parse_args()
    task_profile = get_task_profile(args.task_id)
    if args.replay_export_root is None:
        args.replay_export_root = task_profile.dataset_root_480640
    stationary_calib_dir = task_profile.calibration_pairs_dir("stationary")
    wrist_calib_dir = task_profile.calibration_pairs_dir("wrist")
    stationary_color_calib = task_profile.color_calibration_path("stationary")
    if args.dataset_path is None:
        args.dataset_path = task_profile.dataset_root_480640

    # Load dataset
    episode_data = load_episode(args.dataset_path, args.episode, dataset_root=args.dataset_root)
    actions_raw = episode_data["action"].numpy()
    observations_raw = episode_data["observation.state"].numpy()
    num_frames = len(actions_raw)

    dataset = episode_data["dataset"]
    ep_meta = dataset.meta.episodes[args.episode]
    video_fps = dataset.fps
    print(f"[INFO] Video FPS: {video_fps}")

    relative_timestamps = [i / video_fps for i in range(num_frames)]

    # Load video frames for both cameras
    cam_frames = {}
    for cam_key, cam_cfg in CAMERA_CONFIG.items():
        dataset_cam = cam_cfg["dataset_cam"]
        camera_key = f"observation.images.{dataset_cam}"
        try:
            video_path_rel = dataset.meta.get_video_file_path(args.episode, camera_key)
            video_path = dataset.root / video_path_rel
            if not video_path.exists():
                print(f"[WARN] Video not found for {cam_key}: {video_path}")
                cam_frames[cam_key] = []
                continue
            print(f"[INFO] Loading {cam_key} camera video: {video_path}")
            from_timestamp = ep_meta.get(f"videos/{camera_key}/from_timestamp", 0.0)
            absolute_timestamps = [from_timestamp + ts for ts in relative_timestamps]
            frames_tensor = decode_video_frames(
                video_path, absolute_timestamps, tolerance_s=1e-4, backend="pyav"
            )
            frames_list = []
            for i in range(frames_tensor.shape[0]):
                frame = frames_tensor[i].permute(1, 2, 0).cpu().numpy()
                frame = (frame * 255).astype(np.uint8)
                frames_list.append(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            cam_frames[cam_key] = frames_list
            print(f"[INFO] Loaded {len(frames_list)} frames for {cam_key}")
        except Exception as e:
            print(f"[WARN] Failed to load {cam_key} camera: {e}")
            cam_frames[cam_key] = []

    video_frame_count = max(len(cam_frames.get(k, [])) for k in CAMERA_CONFIG)
    video_frame_count = max(video_frame_count, num_frames)

    # Load MuJoCo model
    project_root = Path(__file__).parent.parent
    xarm_dir = project_root / "xarm7"
    scene_xml_path = resolve_task_scene_xml(args.task_id, xarm_dir)
    print(f"[INFO] Using MuJoCo scene for task {args.task_id!r}: {scene_xml_path.name}")
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

    # Read gripper actuator ctrl range
    gripper_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper")
    gripper_mj_range = (
        model.actuator_ctrlrange[gripper_act_id, 0],
        model.actuator_ctrlrange[gripper_act_id, 1],
    )
    print(f"[INFO] Gripper ctrl range: [{gripper_mj_range[0]}, {gripper_mj_range[1]}]")

    # Apply high damping to mug freejoint to prevent drift during physics
    mug_jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "mug_joint")
    if mug_jnt_id >= 0:
        mug_dof_addr = model.jnt_dofadr[mug_jnt_id]
        model.dof_damping[mug_dof_addr:mug_dof_addr + 6] = 100
        print(f"[INFO] Set mug freejoint dof_damping=100 (DOFs {mug_dof_addr}:{mug_dof_addr+6})")

    # Apply CMA-ES parameters if provided
    if args.cma:
        import pickle
        with open(args.cma_params, "rb") as f:
            cma_result = pickle.load(f)
        xbest = cma_result["xbest"]
        kp = xbest[:7]
        act_damp = xbest[7:14]
        jnt_damp = xbest[14:]
        model.actuator_gainprm[:7, 0] = kp
        model.actuator_biasprm[:7, 1] = -kp
        model.actuator_biasprm[:7, 2] = -act_damp
        model.dof_damping[:7] = jnt_damp
        print(f"[INFO] Applied CMA-ES params from {args.cma_params}")

    RENDER_W, RENDER_H = 640, 480

    # Fix #1: correct MuJoCo fovy to match real camera intrinsics.
    # MuJoCo assumes a centered symmetric frustum; we at least match fy.
    for cam_key, cam_cfg in CAMERA_CONFIG.items():
        K = cam_cfg["config"]["intrinsics"]
        fy = K[1, 1]
        correct_fovy = float(2.0 * np.degrees(np.arctan(RENDER_H / (2.0 * fy))))
        mj_cam = cam_cfg["mujoco_cam"]
        cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, mj_cam)
        if cid >= 0:
            old_fovy = model.cam_fovy[cid]
            model.cam_fovy[cid] = correct_fovy
            print(f"[INFO] Corrected fovy for '{mj_cam}': {old_fovy:.2f}° → {correct_fovy:.2f}°")

    renderer = mujoco.Renderer(model, height=RENDER_H, width=RENDER_W)
    seg_renderer = mujoco.Renderer(model, height=RENDER_H, width=RENDER_W)
    seg_renderer.enable_segmentation_rendering()

    robot_geom_ids = get_robot_geom_ids(model)
    print(f"[INFO] Found {len(robot_geom_ids)} robot geoms for masking")

    # Apply camera calibration.
    # Wrist cam: patches model (local pose) — only needs to be done once.
    # Stationary cam: patches data (world pose) — must be re-applied after every mj_step.
    mujoco.mj_forward(model, data)
    for cam_key, cam_cfg in CAMERA_CONFIG.items():
        mj_cam = cam_cfg["mujoco_cam"]
        cc = cam_cfg["config"]
        cam_id = set_mujoco_camera_from_config(data, model, mj_cam, cc)
        cam_type = cc.get("type", "stationary")
        print(f"[INFO] Camera '{mj_cam}' (id={cam_id}) calibration applied (type={cam_type})")

    # Camera intrinsics from config
    camera_intrinsics = {}
    for cam_key, cam_cfg in CAMERA_CONFIG.items():
        camera_intrinsics[cam_key] = cam_cfg["config"]["intrinsics"]

    # Optional Open3D camera pose viewer
    o3d_vis = None
    o3d_cam_geoms = {}
    o3d_cam_prev_pose = {}
    o3d_cam_keys = []
    if args.open3d_cam_view:
        if not _HAS_OPEN3D:
            print("[WARN] open3d not installed. Disable --open3d-cam-view or install open3d.")
        else:
            try:
                if args.open3d_cam == "both":
                    o3d_cam_keys = ["stationary", "wrist"]
                else:
                    o3d_cam_keys = [args.open3d_cam]

                o3d_vis = o3d.visualization.Visualizer()
                o3d_vis.create_window(window_name="Sim Camera Pose (Open3D)", width=960, height=720)

                world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15)
                o3d_vis.add_geometry(world_frame)

                cam_colors = {
                    "stationary": (1.0, 0.6, 0.0),
                    "wrist": (0.0, 0.7, 1.0),
                }

                for cam_key in o3d_cam_keys:
                    cam_name = CAMERA_CONFIG[cam_key]["mujoco_cam"]
                    pose = get_mujoco_camera_pose(model, data, cam_name)
                    frame_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.08)
                    frame_mesh.transform(pose)
                    local_frustum = _build_camera_frustum_local_points(
                        camera_intrinsics[cam_key], RENDER_W, RENDER_H, depth=0.18
                    )
                    frustum = _make_camera_frustum_lineset(
                        local_frustum, pose, color=cam_colors.get(cam_key, (1.0, 1.0, 0.0))
                    )

                    o3d_vis.add_geometry(frame_mesh)
                    o3d_vis.add_geometry(frustum)
                    o3d_cam_geoms[cam_key] = {
                        "frame": frame_mesh,
                        "frustum": frustum,
                        "local_frustum": local_frustum,
                    }
                    o3d_cam_prev_pose[cam_key] = pose

                vc = o3d_vis.get_view_control()
                vc.set_front([0.2, -0.9, 0.3])
                vc.set_lookat([0.25, 0.0, 0.25])
                vc.set_up([0.0, 0.0, 1.0])
                vc.set_zoom(0.5)
                o3d_vis.poll_events()
                o3d_vis.update_renderer()
                print(f"[INFO] Open3D camera viewer enabled for: {', '.join(o3d_cam_keys)}")
            except Exception as e:
                print(f"[WARN] Failed to launch Open3D camera viewer: {e}")
                o3d_vis = None

    # Camera adjustment state (interactive Open3D camera control)
    o3d_cam_user_delta = {}    # cam_key -> 4x4 accumulated user delta
    o3d_cam_base_pose = {}     # cam_key -> 4x4 base world pose (updated each sim step)
    o3d_active_cam_idx = 0     # index into o3d_cam_keys
    o3d_trans_step = 0.005     # translation step in meters (5 mm)
    o3d_rot_step = 0.5         # rotation step in degrees

    # Load saved camera adjustment delta (applies even without Open3D viewer)
    _loaded_delta = None
    if args.load_camera_adjust:
        _adj_path = Path(args.open3d_cam_save_path)
        if _adj_path.exists():
            _loaded_delta = np.load(str(_adj_path))
            print(f"[INFO] Loaded camera adjustment delta from: {_adj_path}")
        else:
            print(f"[WARN] --load-camera-adjust set but file not found: {_adj_path}")

    # When --load-camera-adjust is used without --open3d-cam-view, still apply
    # the delta to the first stationary camera each frame.
    if _loaded_delta is not None and not o3d_cam_keys:
        o3d_cam_keys = ["stationary"]

    for _ck in o3d_cam_keys:
        o3d_cam_user_delta[_ck] = _loaded_delta.copy() if _loaded_delta is not None else np.eye(4)
        _cn = CAMERA_CONFIG[_ck]["mujoco_cam"]
        o3d_cam_base_pose[_ck] = get_mujoco_camera_pose(model, data, _cn)

    if o3d_vis is not None:
        print(f"[INFO] Active camera for adjustment: {o3d_cam_keys[o3d_active_cam_idx]}")
        print("[INFO] Camera adjustment keys (focus on OpenCV window):")
        print("       SPACE=pause/resume  .=step 1 frame (while paused)")
        print("       w/s=±Z  a/d=±X  r/f=±Y | i/k=pitch  j/l=yaw  u/o=roll")
        print("       [/]=trans step  1/2=rot step  t=toggle cam")
        print("       p=print delta  v=save  0=reset")

    # Load Gaussian Splatting scene
    scene_data = None
    scene_depth_data = None
    gaussian_available = False

    if os.path.exists(args.scene_path):
        try:
            from diff_gaussian_rasterization import GaussianRasterizer
            init_pose = get_mujoco_camera_pose(model, data, "stationary_cam")
            w2c_init = mj_pose_to_gaussian_w2c(init_pose, T_splat2mj)
            scene_data, scene_depth_data, _ = load_scene_data(
                args.scene_path, w2c_init, camera_intrinsics["stationary"]
            )
            gaussian_available = True
            print(f"[INFO] Loaded Gaussian Splatting scene from: {args.scene_path}")
        except ImportError:
            print("[WARN] diff_gaussian_rasterization not installed. Composite = MuJoCo only.")
        except Exception as e:
            print(f"[WARN] Failed to load Gaussian Splatting: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"[WARN] Scene file not found: {args.scene_path}")

    viz_cfg = {'viz_w': RENDER_W, 'viz_h': RENDER_H, 'viz_near': 0.1, 'viz_far': 10.0}

    # Load color calibration
    color_calib = None
    if args.color_calibrate:
        if stationary_color_calib.exists():
            color_calib = load_color_mapping(str(stationary_color_calib))
            print(f"[INFO] Loaded default color calibration from: {stationary_color_calib}")

    # Pre-compute ctrl sequence
    print("[INFO] Converting xArm states to MuJoCo ctrl...")
    ctrl_sequence = np.array([
        lerobot_state_to_mujoco_ctrl(observations_raw[i], gripper_mj_range)
        for i in range(num_frames)
    ])

    # Initialize sim from dataset's first frame (instead of home keyframe) for aligned replay
    data.qpos[:7] = ctrl_sequence[0, :7]
    data.qpos[7] = ctrl_sequence[0, 7] / 255.0 * 0.85  # gripper ctrl -> qpos
    data.qvel[:8] = 0
    mujoco.mj_forward(model, data)
    print("[INFO] Initialized sim from dataset first frame (aligned with replay start)")

    # Create windows — 2 rows: stationary + wrist, 4 columns: recorded, mujoco, composite, alpha
    # When --no-mujoco-view: only wrist alpha, fullscreen
    win_stat_rec = "Stationary - Recorded"
    win_stat_mj = "Stationary - MuJoCo"
    win_stat_comp = "Stationary - Composite"
    win_stat_alpha = "Stationary - Alpha"
    win_wrist_rec = "Wrist - Recorded"
    win_wrist_mj = "Wrist - MuJoCo"
    win_wrist_comp = "Wrist - Composite"
    win_wrist_alpha = "Wrist - Alpha"

    WINDOW_W, WINDOW_H = 400, 300
    for win in [win_stat_rec, win_stat_mj, win_stat_comp, win_stat_alpha,
                win_wrist_rec, win_wrist_mj, win_wrist_comp, win_wrist_alpha]:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, WINDOW_W, WINDOW_H)

    X_START, Y_START = 50, 30
    X_STEP, Y_STEP = 410, 340
    cv2.moveWindow(win_stat_rec, X_START, Y_START)
    cv2.moveWindow(win_stat_comp, X_START + X_STEP, Y_START)
    cv2.moveWindow(win_stat_mj, X_START + 2 * X_STEP, Y_START)
    cv2.moveWindow(win_stat_alpha, X_START + 3 * X_STEP, Y_START)
    cv2.moveWindow(win_wrist_rec, X_START, Y_START + Y_STEP)
    cv2.moveWindow(win_wrist_comp, X_START + X_STEP, Y_START + Y_STEP)
    cv2.moveWindow(win_wrist_mj, X_START + 2 * X_STEP, Y_START + Y_STEP)
    cv2.moveWindow(win_wrist_alpha, X_START + 3 * X_STEP, Y_START + Y_STEP)

    print("[INFO] Starting playback (q=quit, SPACE=pause, +/-=alpha). Click an OpenCV video window for key input.")
    alpha = args.alpha

    # FK comparison setup
    ee_site = None
    for site_name in ["link_tcp"]:
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if site_id != -1:
            ee_site = site_name
            print(f"[INFO] Using site '{site_name}' for FK comparison (id={site_id})")
            break
    if ee_site is None:
        print("[WARN] Could not find end-effector site. FK comparison disabled.")

    trans_errors = []
    rot_errors = []

    # Real-time FK plot (rendered as overlay on stationary alpha window, bottom-left)
    realtime_plot_enabled = False
    fig_fk = ax_trans = ax_rot = None
    line_dx = line_dy = line_dz = line_norm = None
    line_roll = line_pitch = line_yaw = line_angle = None
    FK_OVERLAY_W, FK_OVERLAY_H = 320, 200  # overlay size in pixels (render at 2x for sharpness)
    FK_OVERLAY_PAD = 10  # padding from bottom-left corner
    FK_RENDER_SCALE = 2  # render at 2x resolution for crisp output

    if not args.no_stack and ee_site is not None:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            # Render at 2x resolution for sharp overlay
            fig_fk, (ax_trans, ax_rot) = plt.subplots(2, 1, figsize=(3.2, 2.0), dpi=100 * FK_RENDER_SCALE)
            fig_fk.patch.set_facecolor('white')
            fig_fk.patch.set_alpha(0.95)
            ax_trans.set_facecolor('white')
            ax_rot.set_facecolor('white')

            line_dx, = ax_trans.plot([], [], 'r-', label='dx', alpha=0.8, linewidth=1.2)
            line_dy, = ax_trans.plot([], [], 'g-', label='dy', alpha=0.8, linewidth=1.2)
            line_dz, = ax_trans.plot([], [], 'b-', label='dz', alpha=0.8, linewidth=1.2)
            line_norm, = ax_trans.plot([], [], 'k-', label='||d||', linewidth=1.5)
            ax_trans.set_xlim(0, num_frames / args.fps)
            ax_trans.set_ylim(-20, 20)
            ax_trans.set_autoscale_on(False)
            ax_trans.set_xlabel('Time (s)', fontsize=8)
            ax_trans.set_ylabel('Trans (mm)', fontsize=8)
            ax_trans.legend(loc='upper right', fontsize=7)
            ax_trans.grid(True, alpha=0.3)
            ax_trans.tick_params(labelsize=7)

            line_roll, = ax_rot.plot([], [], 'r-', label='roll', alpha=0.8, linewidth=1.2)
            line_pitch, = ax_rot.plot([], [], 'g-', label='pitch', alpha=0.8, linewidth=1.2)
            line_yaw, = ax_rot.plot([], [], 'b-', label='yaw', alpha=0.8, linewidth=1.2)
            line_angle, = ax_rot.plot([], [], 'k-', label='angle', linewidth=1.5)
            ax_rot.set_xlim(0, num_frames / args.fps)
            ax_rot.set_ylim(-20, 20)
            ax_rot.set_autoscale_on(False)
            ax_rot.set_xlabel('Time (s)', fontsize=8)
            ax_rot.set_ylabel('Rot (deg)', fontsize=8)
            ax_rot.legend(loc='upper right', fontsize=7)
            ax_rot.grid(True, alpha=0.3)
            ax_rot.tick_params(labelsize=7)

            plt.tight_layout(pad=0.3)
            realtime_plot_enabled = True
            overlay_target = "Stationary - Alpha"
            print(f"[INFO] Real-time FK plot will overlay on {overlay_target} (bottom-left)")
        except Exception as e:
            print(f"[WARN] Could not enable real-time plotting: {e}")

    frame_delay = int(1000 / args.fps)
    paused = False
    frame_idx = 0

    if args.save_replay_frames:
        _export = Path(args.replay_export_root)
        for _cam in CAMERA_CONFIG:
            (_export / "gs_render" / _cam).mkdir(parents=True, exist_ok=True)
            (_export / "real_captures" / _cam).mkdir(parents=True, exist_ok=True)
        print(f"[INFO] --save-replay-frames: writing to {_export}/gs_render/{{stationary,wrist}}/ "
              f"and {_export}/real_captures/{{stationary,wrist}}/")

    # --- MuJoCo 3D Viewer Setup ---
    viewer = None
    viewer_ctx = None
    if _HAS_MJ_VIEWER and not args.no_mujoco_view:
        viewer_ctx = mujoco.viewer.launch_passive(model, data)
        viewer = viewer_ctx.__enter__()
        print("[INFO] MuJoCo 3D viewer launched (synchronized)")
    elif args.no_mujoco_view:
        print("[INFO] MuJoCo 3D viewer disabled (--no-mujoco-view)")
    else:
        print("[WARN] mujoco.viewer not available; 3D viewer disabled.")

    try:
        while frame_idx < min(num_frames, video_frame_count):
            if not paused:
                # Apply ctrl
                data.ctrl[:] = ctrl_sequence[frame_idx]
                sim_target = data.time + 1.0 / args.fps
                while data.time < sim_target:
                    mujoco.mj_step(model, data)
                # --- Sync MuJoCo 3D Viewer ---
                if viewer is not None:
                    viewer.sync()

                # Re-apply stationary camera world-pose (mj_step resets data.cam_xpos).
                for cam_key, cam_cfg in CAMERA_CONFIG.items():
                    if cam_cfg["config"].get("type", "stationary") == "stationary":
                        set_mujoco_camera_from_config(data, model, cam_cfg["mujoco_cam"], cam_cfg["config"])

                # Snapshot base camera poses before applying user delta
                for cam_key in o3d_cam_user_delta:
                    cam_name = CAMERA_CONFIG[cam_key]["mujoco_cam"]
                    o3d_cam_base_pose[cam_key] = get_mujoco_camera_pose(model, data, cam_name)

            # Apply user camera delta to MuJoCo data (every iteration, incl. paused)
            for cam_key, delta in o3d_cam_user_delta.items():
                if cam_key not in o3d_cam_base_pose:
                    continue
                cam_name = CAMERA_CONFIG[cam_key]["mujoco_cam"]
                _cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
                adjusted = o3d_cam_base_pose[cam_key] @ delta
                data.cam_xpos[_cid] = adjusted[:3, 3]
                data.cam_xmat[_cid] = adjusted[:3, :3].flatten()

            # Update Open3D camera pose viewer (if enabled)
            if o3d_vis is not None and len(o3d_cam_geoms) > 0:
                try:
                    for cam_key, geoms in o3d_cam_geoms.items():
                        cam_name = CAMERA_CONFIG[cam_key]["mujoco_cam"]
                        new_pose = get_mujoco_camera_pose(model, data, cam_name)
                        _update_frame_mesh_pose(geoms["frame"], o3d_cam_prev_pose[cam_key], new_pose)
                        new_pts = _transform_points(new_pose, geoms["local_frustum"])
                        geoms["frustum"].points = o3d.utility.Vector3dVector(new_pts)
                        o3d_vis.update_geometry(geoms["frame"])
                        o3d_vis.update_geometry(geoms["frustum"])
                        o3d_cam_prev_pose[cam_key] = new_pose

                    if not o3d_vis.poll_events():
                        o3d_vis.destroy_window()
                        o3d_vis = None
                        o3d_cam_geoms = {}
                        o3d_cam_prev_pose = {}
                    else:
                        o3d_vis.update_renderer()
                except Exception as e:
                    print(f"[WARN] Open3D viewer update failed: {e}")
                    try:
                        o3d_vis.destroy_window()
                    except Exception:
                        pass
                    o3d_vis = None
                    o3d_cam_geoms = {}
                    o3d_cam_prev_pose = {}

            # FK comparison (site positions are already valid from mj_step)
            if ee_site is not None and not paused:
                sim_pos, sim_rot = get_end_effector_pose(model, data, ee_site, run_forward=False)

                data_real = MjData(model)
                real_ctrl = lerobot_state_to_mujoco_ctrl(observations_raw[frame_idx], gripper_mj_range)
                data_real.qpos[:7] = real_ctrl[:7]
                data_real.qpos[7] = real_ctrl[7] / 255.0 * 0.85  # gripper ctrl -> qpos
                real_pos, real_rot = get_end_effector_pose(model, data_real, ee_site, run_forward=True)

                trans_diff, rot_diff = compute_pose_difference(real_pos, real_rot, sim_pos, sim_rot)
                trans_errors.append(trans_diff)
                rot_errors.append(rot_diff)

                if realtime_plot_enabled and len(trans_errors) > 1:
                    time_data = np.arange(len(trans_errors)) / args.fps
                    trans_arr = np.array(trans_errors)
                    rot_arr = np.array(rot_errors)
                    line_dx.set_data(time_data, trans_arr[:, 0] * 1000)
                    line_dy.set_data(time_data, trans_arr[:, 1] * 1000)
                    line_dz.set_data(time_data, trans_arr[:, 2] * 1000)
                    line_norm.set_data(time_data, trans_arr[:, 3] * 1000)
                    line_roll.set_data(time_data, np.degrees(rot_arr[:, 0]))
                    line_pitch.set_data(time_data, np.degrees(rot_arr[:, 1]))
                    line_yaw.set_data(time_data, np.degrees(rot_arr[:, 2]))
                    line_angle.set_data(time_data, np.degrees(rot_arr[:, 3]))

            # Render cameras (only stationary when --no-mujoco-view, unless saving calibration pairs)
            cam_renders = {}
            cams_to_render = list(CAMERA_CONFIG.keys())
            window_map = {
                "stationary": (win_stat_rec, win_stat_mj, win_stat_comp, win_stat_alpha),
                "wrist": (win_wrist_rec, win_wrist_mj, win_wrist_comp, win_wrist_alpha),
            }

            for cam_key in cams_to_render:
                cam_cfg = CAMERA_CONFIG[cam_key]
                mujoco_cam = cam_cfg["mujoco_cam"]
                frames_list = cam_frames.get(cam_key, [])

                recorded_frame = (frames_list[frame_idx].copy()
                                  if frame_idx < len(frames_list)
                                  else np.zeros((RENDER_H, RENDER_W, 3), dtype=np.uint8))

                renderer.update_scene(data, camera=mujoco_cam)
                fg_rgb = renderer.render()
                fg_bgr = cv2.cvtColor(fg_rgb, cv2.COLOR_RGB2BGR)

                seg_renderer.update_scene(data, camera=mujoco_cam)
                seg_mask = seg_renderer.render()
                seg_labels = seg_mask[:, :, 0].astype(np.int32)
                # Note: background pixels have seg label -1; do NOT remap to 0
                # because geom ID 0 is the robot base cylinder.

                # Fix #2: shift MuJoCo render to compensate for off-center principal point
                K_cam = camera_intrinsics[cam_key]
                fg_bgr = shift_for_principal_point(fg_bgr, K_cam)
                seg_labels = shift_for_principal_point(seg_labels, K_cam, seg=True)

                robot_mask = np.isin(seg_labels, list(robot_geom_ids))
                mask_uint8 = (robot_mask.astype(np.uint8)) * 255

                if gaussian_available and scene_data is not None:
                    try:
                        # Read camera world pose from MuJoCo data
                        # (includes any user delta adjustments)
                        camera_pose = get_mujoco_camera_pose(model, data, mujoco_cam)
                        w2c = mj_pose_to_gaussian_w2c(camera_pose, T_splat2mj)
                        bg_im = render(w2c, camera_intrinsics[cam_key],
                                       scene_data, scene_depth_data, viz_cfg)[0]
                        bg_np = bg_im.permute(1, 2, 0).cpu().numpy()
                        bg_np = (bg_np * 255).astype(np.uint8)
                        bg_bgr = cv2.cvtColor(bg_np, cv2.COLOR_RGB2BGR)
                        composite_frame = bg_bgr.copy()
                        composite_frame[mask_uint8 > 0] = fg_bgr[mask_uint8 > 0]
                        composite_raw = composite_frame.copy()  # before color calib (for calibration pairs)
                        foreground_only = np.zeros_like(fg_bgr)
                        foreground_only[mask_uint8 > 0] = fg_bgr[mask_uint8 > 0]
                        if args.color_calibrate and color_calib is not None:
                            composite_frame = apply_color_transform(composite_frame, color_calib)
                    except Exception as e:
                        if frame_idx == 0:
                            print(f"[WARN] {cam_key} Gaussian rendering failed: {e}")
                        composite_frame = fg_bgr.copy()
                        composite_raw = composite_frame.copy()
                else:
                    composite_frame = fg_bgr.copy()
                    composite_raw = composite_frame.copy()

                alpha_mask = (mask_uint8 / 255.0).astype(np.float32)
                alpha_mask_3ch = np.stack([alpha_mask] * 3, axis=-1)
                foreground = fg_bgr.astype(np.float32)
                background = recorded_frame.astype(np.float32)
                blended = (alpha * foreground + (1 - alpha) * background) * alpha_mask_3ch + \
                          background * (1 - alpha_mask_3ch)
                alpha_frame = blended.astype(np.uint8)

                cam_renders[cam_key] = {
                    "recorded": recorded_frame,
                    "mujoco": fg_bgr.copy(),
                    "composite": composite_frame,
                    "composite_raw": composite_raw,
                    "alpha": alpha_frame,
                }

            if args.save_replay_frames:
                _root = Path(args.replay_export_root)
                for _ck, _rend in cam_renders.items():
                    cv2.imwrite(
                        str(_root / "gs_render" / _ck / f"frame_{frame_idx:04d}.png"),
                        _rend["composite_raw"],
                    )
                    cv2.imwrite(
                        str(_root / "real_captures" / _ck / f"frame_{frame_idx:04d}.png"),
                        _rend["recorded"],
                    )

            # Save calibration pairs for wrist color calibration
            
            if args.save_calibration_pairs and frame_idx in SAVE_CALIB_FRAMES:
                # Save stationary camera images
                if "stationary" in cam_renders:
                    gs_dir_stationary = stationary_calib_dir / "gs_renders"
                    real_dir_stationary = stationary_calib_dir / "real_captures"
                    gs_dir_stationary.mkdir(parents=True, exist_ok=True)
                    real_dir_stationary.mkdir(parents=True, exist_ok=True)
                    gs_path_stationary = gs_dir_stationary / f"frame_{frame_idx:04d}.png"
                    real_path_stationary = real_dir_stationary / f"frame_{frame_idx:04d}.png"
                    cv2.imwrite(str(gs_path_stationary), cam_renders["stationary"]["composite_raw"])
                    cv2.imwrite(str(real_path_stationary), cam_renders["stationary"]["recorded"])
                    print(f"[INFO] Saved stationary calibration pair frame {frame_idx}: {gs_path_stationary}, {real_path_stationary}")
                # Save wrist camera images
                if "wrist" in cam_renders:
                    gs_dir_wrist = wrist_calib_dir / "gs_renders"
                    real_dir_wrist = wrist_calib_dir / "real_captures"
                    gs_dir_wrist.mkdir(parents=True, exist_ok=True)
                    real_dir_wrist.mkdir(parents=True, exist_ok=True)
                    gs_path_wrist = gs_dir_wrist / f"frame_{frame_idx:04d}.png"
                    real_path_wrist = real_dir_wrist / f"frame_{frame_idx:04d}.png"
                    cv2.imwrite(str(gs_path_wrist), cam_renders["wrist"]["composite_raw"])
                    cv2.imwrite(str(real_path_wrist), cam_renders["wrist"]["recorded"])
                    print(f"[INFO] Saved wrist calibration pair frame {frame_idx}: {gs_path_wrist}, {real_path_wrist}")

            # Overlays
            gripper_mm = observations_raw[frame_idx, 7]
            mujoco_grip_ctrl = data.ctrl[7]
            for cam_key in cams_to_render:
                for ft in ["recorded", "mujoco", "composite", "alpha"]:
                    frame = cam_renders[cam_key][ft]
                    cv2.putText(frame, f"Frame: {frame_idx}/{num_frames}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    if ft == "alpha":
                        cv2.putText(frame, f"Alpha: {alpha:.2f}", (10, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    if paused:
                        cv2.putText(frame, "PAUSED (SPACE=resume, .=step frame)", (10, 90),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # Overlay FK plot on alpha window (bottom-left)
            if not args.no_stack and realtime_plot_enabled and fig_fk is not None and len(trans_errors) > 1:
                try:
                    plot_img = _fig_to_bgr_overlay(fig_fk, FK_OVERLAY_W, FK_OVERLAY_H)
                    alpha_frame = cam_renders["stationary"]["alpha"]
                    y1 = alpha_frame.shape[0] - FK_OVERLAY_H - FK_OVERLAY_PAD
                    y2 = y1 + FK_OVERLAY_H
                    x1 = FK_OVERLAY_PAD
                    x2 = x1 + FK_OVERLAY_W
                    if y1 >= 0 and x2 <= alpha_frame.shape[1]:
                        # Alpha blend for semi-transparency (0.95 = plot, 0.05 = video)
                        roi = alpha_frame[y1:y2, x1:x2].astype(np.float32)
                        overlay = plot_img.astype(np.float32)
                        blended = (0.95 * overlay + 0.05 * roi).astype(np.uint8)
                        alpha_frame[y1:y2, x1:x2] = blended
                except Exception as e:
                    if frame_idx == 0:
                        print(f"[WARN] FK overlay failed: {e}")

            # Display
            for cam_key, (w_rec, w_mj, w_comp, w_alpha) in window_map.items():
                cv2.imshow(w_rec, cam_renders[cam_key]["recorded"])
                cv2.imshow(w_mj, cam_renders[cam_key]["mujoco"])
                cv2.imshow(w_comp, cam_renders[cam_key]["composite"])
                cv2.imshow(w_alpha, cam_renders[cam_key]["alpha"])

            if not paused:
                frame_idx += 1

            # When paused, wait longer for key input (focus must be on an OpenCV window)
            key = cv2.waitKey(100 if paused else frame_delay) & 0xFF
            if key == ord('q'):
                print("[INFO] Quit requested")
                break
            elif key == ord(' '):
                was_paused = paused
                paused = not paused
                print(f"[INFO] {'Paused' if paused else 'Resumed'}")
                if was_paused:
                    # Advance to next frame when resuming (avoid re-processing)
                    frame_idx += 1
            elif key == ord('.') and paused:
                # Single-frame step: advance one frame, stay paused
                if frame_idx + 1 < min(num_frames, video_frame_count):
                    frame_idx += 1
                    data.ctrl[:] = ctrl_sequence[frame_idx]
                    sim_target = data.time + 1.0 / args.fps
                    while data.time < sim_target:
                        mujoco.mj_step(model, data)
                    if viewer is not None:
                        viewer.sync()
                    for cam_key, cam_cfg in CAMERA_CONFIG.items():
                        if cam_cfg["config"].get("type", "stationary") == "stationary":
                            set_mujoco_camera_from_config(data, model, cam_cfg["mujoco_cam"], cam_cfg["config"])
                    for cam_key in o3d_cam_user_delta:
                        cam_name = CAMERA_CONFIG[cam_key]["mujoco_cam"]
                        o3d_cam_base_pose[cam_key] = get_mujoco_camera_pose(model, data, cam_name)
                    print(f"[INFO] Stepped to frame {frame_idx}/{num_frames}")
                else:
                    print("[INFO] Already at last frame")
            elif key == ord('n') and paused:
                paused = False
                continue
            elif key == ord('+') or key == ord('='):
                alpha = min(1.0, alpha + 0.05)
                print(f"[INFO] Alpha: {alpha:.2f}")
            elif key == ord('-') or key == ord('_'):
                alpha = max(0.0, alpha - 0.05)
                print(f"[INFO] Alpha: {alpha:.2f}")
            # --- Camera adjustment keys (only when Open3D cam viewer active) ---
            elif o3d_vis is not None and len(o3d_cam_user_delta) > 0:
                _active_key = o3d_cam_keys[o3d_active_cam_idx] if o3d_cam_keys else None
                _ts = o3d_trans_step
                _rs = o3d_rot_step
                _inc = None
                # Translation: w/s = ±Z,  a/d = ±X,  r/f = ±Y
                if key == ord('w'):
                    _inc = _make_cam_increment(tz=-_ts)   # -Z = forward in MuJoCo cam
                elif key == ord('s'):
                    _inc = _make_cam_increment(tz=_ts)
                elif key == ord('a'):
                    _inc = _make_cam_increment(tx=-_ts)
                elif key == ord('d'):
                    _inc = _make_cam_increment(tx=_ts)
                elif key == ord('r'):
                    _inc = _make_cam_increment(ty=_ts)
                elif key == ord('f'):
                    _inc = _make_cam_increment(ty=-_ts)
                # Rotation: i/k = pitch,  j/l = yaw,  u/o = roll
                elif key == ord('i'):
                    _inc = _make_cam_increment(rx=_rs)
                elif key == ord('k'):
                    _inc = _make_cam_increment(rx=-_rs)
                elif key == ord('j'):
                    _inc = _make_cam_increment(ry=-_rs)
                elif key == ord('l'):
                    _inc = _make_cam_increment(ry=_rs)
                elif key == ord('u'):
                    _inc = _make_cam_increment(rz=_rs)
                elif key == ord('o'):
                    _inc = _make_cam_increment(rz=-_rs)
                if _inc is not None and _active_key is not None:
                    o3d_cam_user_delta[_active_key] = o3d_cam_user_delta[_active_key] @ _inc
                    _t = o3d_cam_user_delta[_active_key][:3, 3]
                    print(f"[CAM] {_active_key}: "
                          f"Δt=[{_t[0]*1000:.1f}, {_t[1]*1000:.1f}, {_t[2]*1000:.1f}] mm")
                # Step size control
                if key == ord('['):
                    o3d_trans_step = max(o3d_trans_step / 2.0, 1e-5)
                    print(f"[CAM] Trans step: {o3d_trans_step*1000:.3f} mm")
                elif key == ord(']'):
                    o3d_trans_step *= 2.0
                    print(f"[CAM] Trans step: {o3d_trans_step*1000:.3f} mm")
                elif key == ord('1'):
                    o3d_rot_step = max(o3d_rot_step / 2.0, 0.01)
                    print(f"[CAM] Rot step: {o3d_rot_step:.3f} deg")
                elif key == ord('2'):
                    o3d_rot_step *= 2.0
                    print(f"[CAM] Rot step: {o3d_rot_step:.3f} deg")
                # Toggle active camera (when controlling both)
                elif key == ord('t') and len(o3d_cam_keys) > 1:
                    o3d_active_cam_idx = (o3d_active_cam_idx + 1) % len(o3d_cam_keys)
                    print(f"[CAM] Active camera: {o3d_cam_keys[o3d_active_cam_idx]}")
                # Print current delta
                elif key == ord('p') and _active_key is not None:
                    print(f"\n[CAM] Delta for '{_active_key}':")
                    np.set_printoptions(precision=6, suppress=True)
                    print(o3d_cam_user_delta[_active_key])
                # Save delta to file
                elif key == ord('v') and _active_key is not None:
                    _sp = args.open3d_cam_save_path
                    np.save(_sp, o3d_cam_user_delta[_active_key])
                    print(f"[CAM] Saved '{_active_key}' delta → {_sp}")
                # Reset delta
                elif key == ord('0') and _active_key is not None:
                    o3d_cam_user_delta[_active_key] = np.eye(4)
                    print(f"[CAM] Reset '{_active_key}' delta to identity")
    finally:
        if o3d_vis is not None:
            try:
                o3d_vis.destroy_window()
            except Exception:
                pass
        if viewer is not None and viewer_ctx is not None:
            viewer_ctx.__exit__(None, None, None)
        cv2.destroyAllWindows()
        print("[INFO] Playback finished")

    if ee_site is not None and len(trans_errors) > 0:
        trans_errors = np.array(trans_errors)
        rot_errors = np.array(rot_errors)
        print(f"\n[INFO] FK Comparison Summary (End Effector):")
        print(f"       Translation: mean={np.mean(trans_errors[:, 3])*1000:.2f} mm, "
              f"max={np.max(trans_errors[:, 3])*1000:.2f} mm")
        print(f"       Rotation:    mean={np.degrees(np.mean(rot_errors[:, 3])):.2f}°, "
              f"max={np.degrees(np.max(rot_errors[:, 3])):.2f}°")

        save_path = f"fk_comparison_ep{args.episode}.png"
        if fig_fk is not None and ax_trans is not None and ax_rot is not None:
            # Update line data and add mean/max text before saving
            time_s = np.arange(len(trans_errors)) / args.fps
            line_dx.set_data(time_s, trans_errors[:, 0] * 1000)
            line_dy.set_data(time_s, trans_errors[:, 1] * 1000)
            line_dz.set_data(time_s, trans_errors[:, 2] * 1000)
            line_norm.set_data(time_s, trans_errors[:, 3] * 1000)
            line_roll.set_data(time_s, np.degrees(rot_errors[:, 0]))
            line_pitch.set_data(time_s, np.degrees(rot_errors[:, 1]))
            line_yaw.set_data(time_s, np.degrees(rot_errors[:, 2]))
            line_angle.set_data(time_s, np.degrees(rot_errors[:, 3]))
            mean_trans = np.mean(trans_errors[:, 3]) * 1000
            max_trans = np.max(trans_errors[:, 3]) * 1000
            mean_rot = np.degrees(np.mean(rot_errors[:, 3]))
            max_rot = np.degrees(np.max(rot_errors[:, 3]))
            ax_trans.text(0.02, 0.98, f'Mean: {mean_trans:.2f} mm\nMax: {max_trans:.2f} mm',
                          transform=ax_trans.transAxes, va='top', fontsize=10,
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            ax_rot.text(0.02, 0.98, f'Mean: {mean_rot:.2f}°\nMax: {max_rot:.2f}°',
                        transform=ax_rot.transAxes, va='top', fontsize=10,
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            fig_fk.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[INFO] Saved FK plot to: {save_path}")
        else:
            plot_fk_comparison(trans_errors, rot_errors, args.fps, save_path=save_path)


if __name__ == "__main__":
    main()
