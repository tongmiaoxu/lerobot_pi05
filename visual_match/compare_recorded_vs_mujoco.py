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
    T_splat2mj,
)

# ============================================================================
# Camera Configuration — loaded from configs/ JSON files
# ============================================================================
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
if not _HAS_DISPLAY:
    print("[ERROR] This script requires a display for visualization")
    sys.exit(1)

import numpy as np
import torch
import torch.nn.functional as F
import cv2

import mujoco
from mujoco import MjModel, MjData
try:
    import mujoco.viewer
    _HAS_MJ_VIEWER = True
except ImportError:
    _HAS_MJ_VIEWER = False

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.video_utils import decode_video_frames

# ============================================================================
# xArm conversion constants
# ============================================================================
GRIPPER_OPEN_MM = 800.0

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


def lerobot_state_to_mujoco_ctrl(state: np.ndarray, gripper_mj_range: tuple) -> np.ndarray:
    """
    Convert xArm LeRobot state (8-dim: 7 joints in degrees + gripper in mm)
    to MuJoCo ctrl (8-dim: 7 joints in radians + gripper in [0, 255]).
    """
    ctrl = np.zeros(8, dtype=np.float64)
    ctrl[:7] = np.deg2rad(state[:7])
    grip_frac = np.clip(state[7] / GRIPPER_OPEN_MM, 0.0, 1.0)
    mj_hi, mj_lo= gripper_mj_range
    ctrl[7] = mj_lo + grip_frac * (mj_hi - mj_lo)
    return ctrl


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


def plot_fk_from_file(npz_path: str, save_path: str = None):
    """Load FK comparison data from saved .npz and plot."""
    d = np.load(npz_path)
    if save_path is None:
        save_path = npz_path.replace('.npz', '.png')
    print(f"[INFO] Loaded FK data from: {npz_path}")
    plot_fk_comparison(d['trans_errors'], d['rot_errors'], float(d['fps']), save_path)


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
    A = np.array(a_values, dtype=np.float32).reshape(3, 6)
    b_match = re.search(r'color_b:\s*\[(.*?)\]', content, re.DOTALL)
    if not b_match:
        raise ValueError(f"Could not find color_b in {yaml_path}")
    b_values = [float(x.strip()) for x in b_match.group(1).replace('\n', '').split(',')]
    b = np.array(b_values, dtype=np.float32)
    return A, b


def apply_color_transform(img, A, b):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    flat = img_rgb.reshape(-1, 3).astype(np.float32) / 255.0
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

    if episode_idx >= dataset.num_episodes:
        raise ValueError(f"Episode {episode_idx} not found. Dataset has {dataset.num_episodes} episodes")

    ep_meta = dataset.meta.episodes[episode_idx]
    start_idx = ep_meta["dataset_from_index"]
    end_idx = ep_meta["dataset_to_index"]
    dataset_size = len(dataset)
    end_idx = min(end_idx, dataset_size - 1)
    num_frames = end_idx - start_idx + 1

    print(f"[INFO] Episode {episode_idx} has {num_frames} frames (indices {start_idx} to {end_idx})")

    actions = []
    observations = []
    for frame_idx in range(start_idx, end_idx + 1):
        sample = dataset[frame_idx]
        actions.append(sample["action"].numpy())
        observations.append(sample["observation.state"].numpy())

    return {
        'action': torch.from_numpy(np.array(actions)),
        'observation.state': torch.from_numpy(np.array(observations)),
        'episode_index': episode_idx,
        'num_frames': num_frames,
        'video_start_frame': start_idx,
        'dataset': dataset,
    }


# ============================================================================
# Main
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Compare recorded xArm video with MuJoCo replay + composite")
    p.add_argument("--dataset-path", type=str, default="data",
                   help="Path to dataset directory (local) or repo_id (Hub)")
    p.add_argument("--dataset-root", type=str, default=None)
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--scene-path", type=str, default="pointclouds/xarm7.npz",
                   help="Path to Gaussian Splatting scene file")
    p.add_argument("--color-calib-path", type=str, default=None,
                   help="Path to color calibration YAML file (optional)")
    p.add_argument("--alpha", type=float, default=0.5,
                   help="Alpha for blending (0=fully real, 1=fully robot)")
    p.add_argument("--color-calibrate", action="store_true",
                   help="Apply color calibration to composite renderings")
    p.add_argument("--save-images", action="store_true",
                   help="Save frames 0,5,10,15,20 to calibration_pairs/")
    p.add_argument("--cma-params", type=str, default="cma_result.pkl",
                   help="Path to cma_result.pkl for optimised stiffness/damping")
    p.add_argument("--cma", action="store_true",default=False,
                   help="Apply CMA-ES optimised parameters to the model.")
    return p.parse_args()


def main():
    args = parse_args()

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

    # Read gripper actuator ctrl range
    gripper_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper")
    gripper_mj_range = (
        model.actuator_ctrlrange[gripper_act_id, 0],
        model.actuator_ctrlrange[gripper_act_id, 1],
    )
    print(f"[INFO] Gripper ctrl range: [{gripper_mj_range[0]}, {gripper_mj_range[1]}]")

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
    if args.color_calib_path and os.path.exists(args.color_calib_path):
        try:
            color_A, color_b = load_color_mapping(args.color_calib_path)
            color_calib = (color_A, color_b)
            print(f"[INFO] Loaded color calibration from: {args.color_calib_path}")
        except Exception as e:
            print(f"[WARN] Failed to load color calibration: {e}")

    # Pre-compute ctrl sequence
    print("[INFO] Converting xArm states to MuJoCo ctrl...")
    ctrl_sequence = np.array([
        lerobot_state_to_mujoco_ctrl(observations_raw[i], gripper_mj_range)
        for i in range(num_frames)
    ])

    # Create windows — 2 rows: stationary + wrist, 4 columns: recorded, mujoco, composite, alpha
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
    cv2.moveWindow(win_stat_mj, X_START + X_STEP, Y_START)
    cv2.moveWindow(win_stat_comp, X_START + 2 * X_STEP, Y_START)
    cv2.moveWindow(win_stat_alpha, X_START + 3 * X_STEP, Y_START)
    cv2.moveWindow(win_wrist_rec, X_START, Y_START + Y_STEP)
    cv2.moveWindow(win_wrist_mj, X_START + X_STEP, Y_START + Y_STEP)
    cv2.moveWindow(win_wrist_comp, X_START + 2 * X_STEP, Y_START + Y_STEP)
    cv2.moveWindow(win_wrist_alpha, X_START + 3 * X_STEP, Y_START + Y_STEP)

    print("[INFO] Starting playback (q=quit, SPACE=pause, +/-=alpha)")
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

    # Real-time FK plot
    realtime_plot_enabled = False
    fig_fk = ax_trans = ax_rot = None
    line_dx = line_dy = line_dz = line_norm = None
    line_roll = line_pitch = line_yaw = line_angle = None

    if ee_site is not None:
        try:
            import matplotlib
            matplotlib.use('TkAgg')
            import matplotlib.pyplot as plt

            plt.ion()
            fig_fk, (ax_trans, ax_rot) = plt.subplots(2, 1, figsize=(10, 6))
            fig_fk.suptitle('FK Comparison: Real vs Simulated (Real-time)')

            line_dx, = ax_trans.plot([], [], 'r-', label='dx', alpha=0.7)
            line_dy, = ax_trans.plot([], [], 'g-', label='dy', alpha=0.7)
            line_dz, = ax_trans.plot([], [], 'b-', label='dz', alpha=0.7)
            line_norm, = ax_trans.plot([], [], 'k-', label='||d||', linewidth=2)
            ax_trans.set_xlim(0, num_frames / args.fps)
            ax_trans.set_ylim(-50, 150)
            ax_trans.set_xlabel('Time (s)')
            ax_trans.set_ylabel('Translation Error (mm)')
            ax_trans.legend(loc='upper right')
            ax_trans.grid(True, alpha=0.3)

            line_roll, = ax_rot.plot([], [], 'r-', label='roll', alpha=0.7)
            line_pitch, = ax_rot.plot([], [], 'g-', label='pitch', alpha=0.7)
            line_yaw, = ax_rot.plot([], [], 'b-', label='yaw', alpha=0.7)
            line_angle, = ax_rot.plot([], [], 'k-', label='angle', linewidth=2)
            ax_rot.set_xlim(0, num_frames / args.fps)
            ax_rot.set_ylim(-15, 15)
            ax_rot.set_xlabel('Time (s)')
            ax_rot.set_ylabel('Rotation Error (degrees)')
            ax_rot.legend(loc='upper right')
            ax_rot.grid(True, alpha=0.3)

            plt.tight_layout()
            fig_fk.canvas.draw()
            plt.pause(0.01)
            realtime_plot_enabled = True
            print("[INFO] Real-time FK plotting enabled")
        except Exception as e:
            print(f"[WARN] Could not enable real-time plotting: {e}")

    frame_delay = int(1000 / args.fps)
    paused = False
    frame_idx = 0

    # --- MuJoCo 3D Viewer Setup ---
    viewer = None
    if _HAS_MJ_VIEWER:
        viewer_ctx = mujoco.viewer.launch_passive(model, data)
        viewer = viewer_ctx.__enter__()
        print("[INFO] MuJoCo 3D viewer launched (synchronized)")
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
            # Wrist camera pose is computed by kinematics from the model-local
            # offset set once at startup — no per-frame override needed.
            for cam_key, cam_cfg in CAMERA_CONFIG.items():
                if cam_cfg["config"].get("type", "stationary") == "stationary":
                    set_mujoco_camera_from_config(data, model, cam_cfg["mujoco_cam"], cam_cfg["config"])

            # FK comparison (site positions are already valid from mj_step)
            if ee_site is not None:
                sim_pos, sim_rot = get_end_effector_pose(model, data, ee_site, run_forward=False)

                data_real = MjData(model)
                real_ctrl = lerobot_state_to_mujoco_ctrl(observations_raw[frame_idx], gripper_mj_range)
                data_real.qpos[:7] = real_ctrl[:7]
                data_real.qpos[7] = real_ctrl[7] / 255.0 * 0.85  # gripper ctrl -> qpos
                real_pos, real_rot = get_end_effector_pose(model, data_real, ee_site, run_forward=True)

                trans_diff, rot_diff = compute_pose_difference(real_pos, real_rot, sim_pos, sim_rot)
                trans_errors.append(trans_diff)
                rot_errors.append(rot_diff)

                if realtime_plot_enabled and len(trans_errors) > 1 and frame_idx % 5 == 0:
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
                    max_trans = np.max(np.abs(trans_arr[:, :3])) * 1000
                    max_norm = np.max(trans_arr[:, 3]) * 1000
                    if max_norm > 100 or max_trans > 40:
                        ax_trans.set_ylim(-max(50, max_trans * 1.2), max(150, max_norm * 1.2))
                    max_rot = np.max(np.abs(rot_arr[:, :3]))
                    max_angle = np.max(rot_arr[:, 3])
                    if np.degrees(max_angle) > 10:
                        ax_rot.set_ylim(-max(15, np.degrees(max_rot) * 1.2),
                                        max(15, np.degrees(max_angle) * 1.2))
                    fig_fk.canvas.draw_idle()
                    fig_fk.canvas.flush_events()

            # Render both cameras
            cam_renders = {}
            window_map = {
                "stationary": (win_stat_rec, win_stat_mj, win_stat_comp, win_stat_alpha),
                "wrist": (win_wrist_rec, win_wrist_mj, win_wrist_comp, win_wrist_alpha),
            }

            for cam_key, cam_cfg in CAMERA_CONFIG.items():
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
                seg_labels[seg_labels == -1] = 0
                robot_mask = np.isin(seg_labels, list(robot_geom_ids))
                mask_uint8 = (robot_mask.astype(np.uint8)) * 255

                if gaussian_available and scene_data is not None:
                    try:
                        cc = cam_cfg["config"]
                        if cc.get("type", "stationary") == "stationary":
                            # Fixed camera — use calibration pose directly
                            camera_pose = np.eye(4)
                            camera_pose[:3, :3] = cc["cam_xmat_mj"]
                            camera_pose[:3, 3] = cc["cam_pos_mj"]
                        else:
                            # Wrist camera — read kinematic world pose from MuJoCo
                            camera_pose = get_mujoco_camera_pose(model, data, mujoco_cam)
                        w2c = mj_pose_to_gaussian_w2c(camera_pose, T_splat2mj)
                        bg_im = render(w2c, camera_intrinsics[cam_key],
                                       scene_data, scene_depth_data, viz_cfg)[0]
                        bg_np = bg_im.permute(1, 2, 0).cpu().numpy()
                        bg_np = (bg_np * 255).astype(np.uint8)
                        bg_bgr = cv2.cvtColor(bg_np, cv2.COLOR_RGB2BGR)
                        composite_frame = bg_bgr.copy()
                        composite_frame[mask_uint8 > 0] = fg_bgr[mask_uint8 > 0]
                        foreground_only = np.zeros_like(fg_bgr)
                        foreground_only[mask_uint8 > 0] = fg_bgr[mask_uint8 > 0]
                        if args.color_calibrate and color_calib is not None:
                            composite_frame = apply_color_transform(composite_frame, *color_calib)
                    except Exception as e:
                        if frame_idx == 0:
                            print(f"[WARN] {cam_key} Gaussian rendering failed: {e}")
                        composite_frame = fg_bgr.copy()
                else:
                    composite_frame = fg_bgr.copy()

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
                    "alpha": alpha_frame,
                }

            # Save calibration frames
            # (disabled: do not save any images or create directories)
            # SAVE_FRAMES = [0, 5, 10, 15, 20]
            # if args.save_images and frame_idx in SAVE_FRAMES:
            #     gs_dir = Path("calibration_pairs/gs_renders")
            #     real_dir = Path("calibration_pairs/real_captures")
            #     gs_dir.mkdir(parents=True, exist_ok=True)
            #     real_dir.mkdir(parents=True, exist_ok=True)
            #     gs_path = gs_dir / f"frame_{frame_idx:04d}.png"
            #     real_path = real_dir / f"frame_{frame_idx:04d}.png"
            #     cv2.imwrite(str(gs_path), cam_renders["stationary"]["composite"])
            #     cv2.imwrite(str(real_path), cam_renders["stationary"]["recorded"])
            #     print(f"[INFO] Saved frame {frame_idx}: {gs_path}, {real_path}")

            # Overlays
            gripper_mm = observations_raw[frame_idx, 7]
            mujoco_grip_ctrl = data.ctrl[7]
            for cam_key in CAMERA_CONFIG:
                for ft in ["recorded", "mujoco", "composite", "alpha"]:
                    frame = cam_renders[cam_key][ft]
                    cv2.putText(frame, f"Frame: {frame_idx}/{num_frames}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    if ft == "alpha":
                        cv2.putText(frame, f"Alpha: {alpha:.2f}", (10, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            # Display
            for cam_key, (w_rec, w_mj, w_comp, w_alpha) in window_map.items():
                cv2.imshow(w_rec, cam_renders[cam_key]["recorded"])
                cv2.imshow(w_mj, cam_renders[cam_key]["mujoco"])
                cv2.imshow(w_comp, cam_renders[cam_key]["composite"])
                cv2.imshow(w_alpha, cam_renders[cam_key]["alpha"])


                frame_idx += 1

            key = cv2.waitKey(frame_delay) & 0xFF
            if key == ord('q'):
                print("[INFO] Quit requested")
                break
            elif key == ord(' '):
                paused = not paused
                print(f"[INFO] {'Paused' if paused else 'Resumed'}")
            elif key == ord('n') and paused:
                paused = False
                continue
            elif key == ord('+') or key == ord('='):
                alpha = min(1.0, alpha + 0.05)
                print(f"[INFO] Alpha: {alpha:.2f}")
            elif key == ord('-') or key == ord('_'):
                alpha = max(0.0, alpha - 0.05)
                print(f"[INFO] Alpha: {alpha:.2f}")
    finally:
        if viewer is not None:
            viewer_ctx.__exit__(None, None, None)
        cv2.destroyAllWindows()
        print("[INFO] Playback finished")

    if realtime_plot_enabled and fig_fk is not None:
        try:
            import matplotlib.pyplot as plt
            plt.ioff()
        except:
            pass

    if ee_site is not None and len(trans_errors) > 0:
        trans_errors = np.array(trans_errors)
        rot_errors = np.array(rot_errors)

        data_path = f"fk_comparison_ep{args.episode}.npz"
        np.savez(data_path, trans_errors=trans_errors, rot_errors=rot_errors,
                 fps=args.fps, episode=args.episode)
        print(f"[INFO] Saved FK comparison data to: {data_path}")
        print(f"\n[INFO] FK Comparison Summary (End Effector):")
        print(f"       Translation: mean={np.mean(trans_errors[:, 3])*1000:.2f} mm, "
              f"max={np.max(trans_errors[:, 3])*1000:.2f} mm")
        print(f"       Rotation:    mean={np.degrees(np.mean(rot_errors[:, 3])):.2f}°, "
              f"max={np.degrees(np.max(rot_errors[:, 3])):.2f}°")

        # save_path = f"fk_comparison_ep{args.episode}.png"
        # if realtime_plot_enabled and fig_fk is not None:
        #     try:
        #         fig_fk.savefig(save_path, dpi=150, bbox_inches='tight')
        #         print(f"[INFO] Saved FK plot to: {save_path}")
        #         plt.show()
        #     except Exception:
        #         pass
        # else:
        #     plot_fk_comparison(trans_errors, rot_errors, args.fps, save_path=save_path)


if __name__ == "__main__":
    main()
