#!/usr/bin/env python3
# compare_recorded_vs_mujoco.py
"""
Side-by-side comparison of recorded dataset video and MuJoCo simulation.

Shows THREE synchronized windows:
1. Recorded video from right wrist camera (from dataset)
2. Rendered view from teleoperator_pov camera (from MuJoCo replay)
3. Composite (MuJoCo foreground + Gaussian Splatting background) from wrist camera

Usage:
    python compare_recorded_vs_mujoco.py --dataset-path /path/to/dataset --episode 0
"""

import sys
import os
import argparse
from pathlib import Path

# Add src to path for lerobot imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Auto-detect display before importing mujoco
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

# Check if OpenCV has GUI support
_HAS_OPENCV_GUI = False
try:
    # Check if cv2 is properly loaded (not just a namespace package)
    if not hasattr(cv2, 'namedWindow'):
        raise AttributeError("cv2 module is not properly loaded")
    # Try to create a test window to check GUI support
    test_window = "___opencv_gui_test___"
    cv2.namedWindow(test_window, cv2.WINDOW_NORMAL)
    cv2.destroyWindow(test_window)
    _HAS_OPENCV_GUI = True
except (AttributeError, cv2.error) as e:
    _HAS_OPENCV_GUI = False

# Lerobot dataset loader
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.video_utils import decode_video_frames

# ============================================================================
# Gaussian Splatting imports (from eval_pipeline_advait_new.py)
# ============================================================================

try:
    from splatam.utils.recon_helpers import setup_camera
except ImportError:
    print("[WARN] splatam.utils.recon_helpers.setup_camera not found. Using basic implementation.")
    class CameraParams:
        def __init__(self, image_height, image_width, tanfovx, tanfovy, scale_modifier,
                     viewmatrix, projmatrix, sh_degree, campos, prefiltered):
            self.image_height = image_height
            self.image_width = image_width
            self.tanfovx = tanfovx
            self.tanfovy = tanfovy
            self.scale_modifier = scale_modifier
            self.viewmatrix = viewmatrix
            self.projmatrix = projmatrix
            self.sh_degree = sh_degree
            self.campos = campos
            self.prefiltered = prefiltered

    def setup_camera(w, h, k, w2c, near=0.01, far=100):
        fx, fy, cx, cy = k[0][0], k[1][1], k[0][2], k[1][2]
        w2c = torch.tensor(w2c).cuda().float()
        cam_center = torch.inverse(w2c)[:3, 3]
        w2c = w2c.unsqueeze(0).transpose(1, 2)
        opengl_proj = torch.tensor([[2 * fx / w, 0.0, -(w - 2 * cx) / w, 0.0],
                                    [0.0, 2 * fy / h, -(h - 2 * cy) / h, 0.0],
                                    [0.0, 0.0, far / (far - near), -(far * near) / (far - near)],
                                    [0.0, 0.0, 1.0, 0.0]]).cuda().float().unsqueeze(0).transpose(1, 2)
        full_proj = w2c.bmm(opengl_proj)
        cam = CameraParams(
            image_height=h, image_width=w,
            tanfovx=w / (2 * fx), tanfovy=h / (2 * fy),
            scale_modifier=1.0, viewmatrix=w2c, projmatrix=full_proj,
            sh_degree=0, campos=cam_center, prefiltered=False
        )
        return cam

# Load ICP transform
ICP_TRANSFORM_PATH = "pointclouds/icp_transform.npy"
T_splat2mj = np.load(ICP_TRANSFORM_PATH) if os.path.exists(ICP_TRANSFORM_PATH) else np.eye(4)

# ============================================================================
# Helper functions from eval_pipeline_advait_new.py
# ============================================================================

def get_robot_geom_ids(model):
    """Returns a set of geom indices that belong to the robot."""
    robot_body_ids = []
    for body_id in range(model.nbody):
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        if body_name is not None and (body_name.startswith("left/") or body_name.startswith("right/")):
            robot_body_ids.append(body_id)
    robot_geom_ids = set()
    for body_id in robot_body_ids:
        start = model.body_geomadr[body_id]
        num = model.body_geomnum[body_id]
        for geom_id in range(start, start + num):
            robot_geom_ids.add(geom_id)
    return robot_geom_ids


def load_scene_data(scene_path, first_frame_w2c, intrinsics):
    """Load Gaussian Splatting scene data."""
    def build_rotation(quat):
        w, x, y, z = quat[0], quat[1], quat[2], quat[3]
        R = torch.stack([
            torch.stack([1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)]),
            torch.stack([2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)]),
            torch.stack([2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)])
        ])
        return R

    all_params = dict(np.load(scene_path, allow_pickle=True))
    for k in all_params.keys():
        all_params[k] = torch.tensor(all_params[k]).cuda().float()
    intrinsics = torch.tensor(intrinsics).cuda().float()
    first_frame_w2c = torch.tensor(first_frame_w2c).cuda().float()

    keys = [k for k in all_params.keys() if
            k not in ['org_width', 'org_height', 'w2c', 'intrinsics', 
                      'gt_w2c_all_frames', 'cam_unnorm_rots',
                      'cam_trans', 'keyframe_time_indices']]
    params = all_params
    for k in keys:
        if not isinstance(all_params[k], torch.Tensor):
            params[k] = torch.tensor(all_params[k]).cuda().float()
        else:
            params[k] = all_params[k].cuda().float()

    if params['log_scales'].shape[-1] == 1:
        log_scales = torch.tile(params['log_scales'], (1, 3))
    else:
        log_scales = params['log_scales']
        
    rendervar = {
        'means3D': params['means3D'],
        'colors_precomp': params['rgb_colors'],
        'rotations': F.normalize(params['unnorm_rotations']),
        'opacities': torch.sigmoid(params['logit_opacities']),
        'scales': torch.exp(log_scales),
        'means2D': torch.zeros_like(params['means3D'], device="cuda")
    }
    
    # Depth rendervar (for silhouette)
    def get_depth_colors(means3D, w2c):
        ones = torch.ones((means3D.shape[0], 1), device=means3D.device, dtype=means3D.dtype)
        points_h = torch.cat([means3D, ones], dim=1)
        cam_points = (w2c @ points_h.T).T
        depth = cam_points[:, 2]
        depth_min, depth_max = depth.min(), depth.max()
        if depth_max > depth_min:
            depth_norm = (depth - depth_min) / (depth_max - depth_min)
        else:
            depth_norm = torch.zeros_like(depth)
        return depth_norm.unsqueeze(1).repeat(1, 3)

    depth_rendervar = {
        'means3D': params['means3D'],
        'colors_precomp': get_depth_colors(params['means3D'], first_frame_w2c),
        'rotations': F.normalize(params['unnorm_rotations']),
        'opacities': torch.sigmoid(params['logit_opacities']),
        'scales': torch.exp(log_scales),
        'means2D': torch.zeros_like(params['means3D'], device="cuda")
    }
    
    return rendervar, depth_rendervar


def render_gaussian(w2c, k, scene_data, scene_depth_data, viz_cfg):
    """Render Gaussian Splatting background."""
    try:
        from diff_gaussian_rasterization import GaussianRasterizer as Renderer
        from diff_gaussian_rasterization import GaussianRasterizationSettings as Camera
    except ImportError:
        raise ImportError("diff_gaussian_rasterization not installed")
    
    cam = setup_camera(viz_cfg['viz_w'], viz_cfg['viz_h'], k, w2c, viz_cfg['viz_near'], viz_cfg['viz_far'])
    white_bg_cam = Camera(
        image_height=cam.image_height, image_width=cam.image_width,
        tanfovx=cam.tanfovx, tanfovy=cam.tanfovy,
        bg=torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda"),
        scale_modifier=cam.scale_modifier, viewmatrix=cam.viewmatrix,
        projmatrix=cam.projmatrix, sh_degree=cam.sh_degree,
        campos=cam.campos, prefiltered=cam.prefiltered, debug=False
    )
    im, depth = Renderer(raster_settings=white_bg_cam)(**scene_data)
    return im


def get_mujoco_camera_pose(model, data, cam_name):
    """Get camera pose from MuJoCo as 4x4 matrix."""
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    if cam_id == -1:
        raise ValueError(f"Camera '{cam_name}' not found")
    camera_xpos = data.cam_xpos[cam_id]
    camera_xmat = data.cam_xmat[cam_id].reshape(3, 3)
    camera_pose = np.eye(4)
    camera_pose[:3, :3] = camera_xmat
    camera_pose[:3, 3] = camera_xpos
    return camera_pose


def mj_pose_to_gaussian_w2c(camera_pose, T_splat2mj):
    """Convert MuJoCo camera pose to Gaussian Splatting world-to-camera."""
    T_mj2splat = np.linalg.inv(T_splat2mj)
    P_gs_cam = T_mj2splat @ camera_pose
    # Apply axis flip (Y and Z)
    transform_matrix = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    w2c = P_gs_cam @ transform_matrix
    w2c = np.linalg.inv(w2c)
    return w2c


def get_camera_intrinsics_from_model(mj_model, camera_id, render_w, render_h):
    """
    Read camera parameters from MuJoCo model and compute pixel intrinsics.
    Falls back to fovy=45° if no explicit parameters are set.
    
    Args:
        mj_model: MuJoCo model
        camera_id: Camera ID
        render_w, render_h: Render resolution
    
    Returns:
        K: 3x3 intrinsics matrix
    """
    cam_name_str = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_CAMERA, camera_id)
    
    use_fovy_fallback = False
    
    try:
        # MuJoCo 3.x provides cam_intrinsic, cam_sensorsize, cam_resolution
        intrinsic = mj_model.cam_intrinsic[camera_id]
        sensorsize = mj_model.cam_sensorsize[camera_id]
        
        focal_x_m = intrinsic[0]
        focal_y_m = intrinsic[1]
        sensorsize_x = sensorsize[0]
        sensorsize_y = sensorsize[1]
        
        # Check if values are valid (non-zero) - cameras without explicit params have zeros
        if sensorsize_x <= 0 or sensorsize_y <= 0 or focal_x_m <= 0 or focal_y_m <= 0:
            print(f"[INFO] Camera '{cam_name_str}' has no explicit intrinsics (sensorsize or focal is 0)")
            use_fovy_fallback = True
        else:
            # Compute pixel focal lengths at RENDER resolution
            # f_px = f_m * (render_resolution / sensorsize)
            fx_px = focal_x_m * (render_w / sensorsize_x)
            fy_px = focal_y_m * (render_h / sensorsize_y)
            cx_px = render_w / 2.0
            cy_px = render_h / 2.0
            
            # Validate the computed values
            if not (np.isfinite(fx_px) and np.isfinite(fy_px) and fx_px > 0 and fy_px > 0):
                print(f"[WARN] Invalid computed intrinsics for '{cam_name_str}': fx={fx_px}, fy={fy_px}")
                use_fovy_fallback = True
            else:
                print(f"[INFO] Camera '{cam_name_str}' intrinsics from XML:")
                print(f"       focal=({focal_x_m*1000:.3f}, {focal_y_m*1000:.3f}) mm, "
                      f"sensorsize=({sensorsize_x*1000:.3f}, {sensorsize_y*1000:.3f}) mm")
                print(f"       At render ({render_w}x{render_h}): fx={fx_px:.2f}, fy={fy_px:.2f}, "
                      f"cx={cx_px:.2f}, cy={cy_px:.2f}")
                
    except (AttributeError, IndexError) as e:
        print(f"[WARN] Could not read cam_intrinsic from model: {e}")
        use_fovy_fallback = True
    
    # Fallback: compute from fovy (default is 45°)
    if use_fovy_fallback:
        fovy_deg = mj_model.cam_fovy[camera_id]
        fovy_rad = np.radians(fovy_deg)
        fy_px = (render_h / 2) / np.tan(fovy_rad / 2)
        fx_px = fy_px  # Assume square pixels
        cx_px = render_w / 2.0
        cy_px = render_h / 2.0
        print(f"[INFO] Camera '{cam_name_str}' using fovy={fovy_deg:.2f}° fallback:")
        print(f"       fx=fy={fx_px:.2f}, cx={cx_px:.2f}, cy={cy_px:.2f}")
    
    K = np.array([[fx_px, 0, cx_px],
                  [0, fy_px, cy_px],
                  [0, 0, 1]])
    return K


# ============================================================================
# Dataset loader for v3.0 using LeRobotDataset
# ============================================================================

def load_episode(dataset_path: str, episode_idx: int, dataset_root: str | None = None):
    """
    Load a single episode from a LeRobot v3.0 dataset using LeRobotDataset.
    
    Args:
        dataset_path: Path to local dataset directory or repo_id for Hub dataset
        episode_idx: Episode index to load
        dataset_root: Root directory for local datasets (optional, used when dataset_path is repo_id)
    
    Returns:
        dict with 'action', 'observation.state', 'episode_index', 'num_frames', 
        and video path information
    """
    print(f"[INFO] Loading episode {episode_idx} from dataset: {dataset_path}")
    
    # Check if dataset_path is a local directory or repo_id
    dataset_path_obj = Path(dataset_path)
    if dataset_path_obj.exists() and dataset_path_obj.is_dir():
        # Local dataset - check if it has the v3.0 structure (meta/info.json)
        meta_info = dataset_path_obj / "meta" / "info.json"
        if meta_info.exists():
            # Dataset is at the path directly - use it as root
            # LeRobotDataset expects root/repo_id/, but for local datasets we can pass
            # the dataset directory as root and use the directory name as repo_id
            # However, LeRobotDatasetMetadata uses root directly, so we need to pass
            # the dataset directory as root
            repo_id = dataset_path_obj.name
            # For local datasets, LeRobotDataset will use root directly for metadata
            # So we pass the dataset directory as root
            dataset = LeRobotDataset(
                repo_id=repo_id,
                root=dataset_path_obj,  # Pass dataset directory as root
                episodes=[episode_idx],
                video_backend="pyav"  # Use pyav instead of torchcodec to avoid FFmpeg library issues
            )
        else:
            # Path might be a parent directory - try using directory name as repo_id
            repo_id = dataset_path_obj.name
            root = dataset_path_obj.parent
            dataset = LeRobotDataset(
                repo_id=repo_id,
                root=root,
                episodes=[episode_idx],
                video_backend="pyav"  # Use pyav instead of torchcodec to avoid FFmpeg library issues
            )
    else:
        # Assume it's a repo_id - load from Hub or local cache
        dataset = LeRobotDataset(
            repo_id=dataset_path,
            root=dataset_root,
            episodes=[episode_idx],
            video_backend="pyav"  # Use pyav instead of torchcodec to avoid FFmpeg library issues
        )
    
    # Get episode metadata
    if episode_idx >= dataset.num_episodes:
        raise ValueError(f"Episode {episode_idx} not found. Dataset has {dataset.num_episodes} episodes")
    
    # Get episode frame range
    ep_meta = dataset.meta.episodes[episode_idx]
    start_idx = ep_meta["dataset_from_index"]
    end_idx = ep_meta["dataset_to_index"]
    # end_idx is inclusive, so the valid range is [start_idx, end_idx]
    # But we need to make sure we don't go beyond the dataset size
    dataset_size = len(dataset)
    end_idx = min(end_idx, dataset_size - 1)
    num_frames = end_idx - start_idx + 1
    
    print(f"[INFO] Episode {episode_idx} has {num_frames} frames (indices {start_idx} to {end_idx})")
    
    # Load all frames from this episode
    actions = []
    observations = []
    
    for frame_idx in range(start_idx, end_idx + 1):
        sample = dataset[frame_idx]
        actions.append(sample["action"].numpy())
        observations.append(sample["observation.state"].numpy())
    
    # Convert to tensors
    actions_tensor = torch.from_numpy(np.array(actions))
    observations_tensor = torch.from_numpy(np.array(observations))
    
    # Get video path using dataset's method
    video_path = None
    video_start_frame = start_idx  # Frame index in the video file
    
    return {
        'action': actions_tensor,
        'observation.state': observations_tensor,
        'episode_index': episode_idx,
        'num_frames': num_frames,
        'video_start_frame': video_start_frame,
        'dataset': dataset,  # Keep dataset reference for video loading
    }


# ============================================================================
# Action conversion
# ============================================================================


def convert_actions_to_mujoco_delta(actions_raw: np.ndarray, mujoco_keyframe_ctrl: np.ndarray):
    """
    DELTA-BASED: Convert recorded actions using deltas from first frame.
    Problem: Assumes MuJoCo keyframe matches real-world frame 0 pose.
    """
    num_frames = len(actions_raw)
    action_ref = actions_raw[0].copy()
    deltas_deg = actions_raw - action_ref
    
    recorded_to_ctrl = {
        # Right arm: 9=waist, 10=shoulder(-1), 12=elbow, 14=forearm, 15=wrist_angle, 16=wrist_rotate(-1), 17=gripper
        9:  (0, 1), 10: (1, -1), 12: (2, 1), 14: (3, 1), 15: (4, 1), 16: (5, -1), 17: (6, 1),
        # Left arm: 0=waist, 1=shoulder(-1), 3=elbow, 5=forearm, 6=wrist_angle, 7=wrist_rotate(-1), 8=gripper
        0:  (7, 1), 1:  (8, -1), 3:  (9, 1), 5:  (10, 1), 6:  (11, 1), 7:  (12, -1), 8:  (13, 1),
    }
    
    deltas_rad = np.deg2rad(deltas_deg)
    ctrl_sequence = np.zeros((num_frames, 14))
    
    for frame_idx in range(num_frames):
        ctrl = mujoco_keyframe_ctrl.copy()
        for rec_idx, (ctrl_idx, sign) in recorded_to_ctrl.items():
            if rec_idx in [8, 17]:  # Gripper
                gripper_val = actions_raw[frame_idx, rec_idx]
                ctrl[ctrl_idx] = 0.041 - (gripper_val / 100.0) * 0.041
                ctrl[ctrl_idx] = np.clip(ctrl[ctrl_idx], 0.0, 0.041)
            else:
                ctrl[ctrl_idx] = mujoco_keyframe_ctrl[ctrl_idx] + sign * deltas_rad[frame_idx, rec_idx]
        ctrl_sequence[frame_idx] = ctrl
    
    return ctrl_sequence


# -------- PI05 normalization method -------------------------------------------
def convert_actions_to_mujoco_pi05(actions_raw: np.ndarray, mujoco_keyframe_ctrl: np.ndarray,
                                    gripper_ctrl_range: tuple = (0.002, 0.041)):
    """
    PI05 NORMALIZATION: Convert lerobot normalized degrees to MuJoCo (Interbotix) radians.
    
    Uses PI05 normalization formula:
        1. normalized_degrees = (raw - mid) * 360 / max_res (recorded data)
        2. raw_encoder = (normalized_degrees * max_res / 360) + mid (inverse)
        3. raw_encoder → interbotix_radians: rad = (raw - 2048) * (2π / 4096)
    
    Where:
        - mid = (range_min + range_max) / 2
        - max_res = 4095 (for ALOHA motors: 4096 - 1)
    
    Calibration data from .cache/calibration/aloha_follower/*.json
    """
    import json
    from pathlib import Path
    
    num_frames = len(actions_raw)
    ctrl_sequence = np.zeros((num_frames, 14))
    
    # Load calibration files (relative to project root)
    project_root = Path(__file__).parent.parent
    calib_dir = project_root / ".cache" / "calibration" / "aloha_follower"
    
    # Try to load calibration, fall back to hardcoded defaults if not found
    try:
        with open(calib_dir / "aloha_left.json") as f:
            left_calib = json.load(f)
        with open(calib_dir / "aloha_right.json") as f:
            right_calib = json.load(f)
        print(f"[INFO] Loaded PI05 calibration from: {calib_dir}")
    except FileNotFoundError:
        print(f"[WARN] Calibration files not found at {calib_dir}, using defaults")
        # Default calibration (fallback)
        right_calib = {
            "waist": {"range_min": -974, "range_max": 2042},
            "shoulder": {"range_min": -1921, "range_max": 2886},
            "elbow": {"range_min": 1215, "range_max": 2163},
            "forearm_roll": {"range_min": -1003, "range_max": 2047},
            "wrist_angle": {"range_min": 2021, "range_max": 3036},
            "wrist_rotate": {"range_min": -1061, "range_max": 2055},
            "gripper": {"range_min": 1710, "range_max": 2733},
        }
        left_calib = right_calib.copy()
    
    # PI05 constants
    MAX_RES = 4095  # For ALOHA motors: 4096 - 1
    
    def normalized_degrees_to_raw(normalized_degrees: float, range_min: int, range_max: int) -> float:
        """Convert PI05 normalized degrees to raw encoder value."""
        mid = (range_min + range_max) / 2
        raw = (normalized_degrees * MAX_RES / 360) + mid
        return raw
    
    def raw_encoder_to_radians(raw: float) -> float:
        """Convert raw encoder to Interbotix/MuJoCo radians."""
        return (raw - 2048) * (2 * np.pi) / 4096
    
    gripper_min, gripper_max = gripper_ctrl_range
    gripper_range = gripper_max - gripper_min
    
    # Calculate slope and intercept from control range and lerobot percentage range
    RIGHT_GRIPPER_SLOPE = (LEROBOT_CLOSED_PCT - LEROBOT_OPEN_PCT) / gripper_range
    RIGHT_GRIPPER_INTERCEPT = LEROBOT_OPEN_PCT - RIGHT_GRIPPER_SLOPE * gripper_min
    
    # Left arm gripper calibration (using same as right for now)
    LEFT_GRIPPER_SLOPE = RIGHT_GRIPPER_SLOPE
    LEFT_GRIPPER_INTERCEPT = RIGHT_GRIPPER_INTERCEPT
    
    def gripper_lerobot_to_interbotix(lerobot_percent: float, arm_side: str = "right") -> float:
        """
        Convert gripper from lerobot percentage to Interbotix radians.
        """
        if arm_side == "right":
            return (lerobot_percent - RIGHT_GRIPPER_INTERCEPT) / RIGHT_GRIPPER_SLOPE
        else:  # left
            return (lerobot_percent - LEFT_GRIPPER_INTERCEPT) / LEFT_GRIPPER_SLOPE
    
    # Mapping: recorded index → (mujoco ctrl index, calibration dict, joint name, arm_side)
    # Recorded format: [left_arm(9), right_arm(9)] = 18 total
    #   Left:  [0]=waist, [1]=shoulder, [2]=shoulder_shadow, [3]=elbow, [4]=elbow_shadow,
    #          [5]=forearm_roll, [6]=wrist_angle, [7]=wrist_rotate, [8]=gripper
    #   Right: [9]=waist, [10]=shoulder, [11]=shoulder_shadow, [12]=elbow, [13]=elbow_shadow,
    #          [14]=forearm_roll, [15]=wrist_angle, [16]=wrist_rotate, [17]=gripper
    #
    # MuJoCo ctrl format: [right_arm(7), left_arm(7)] = 14 total
    #   [0-6]: right waist, shoulder, elbow, forearm_roll, wrist_angle, wrist_rotate, gripper
    #   [7-13]: left waist, shoulder, elbow, forearm_roll, wrist_angle, wrist_rotate, gripper
    
    # Mapping: (recorded_idx, ctrl_idx, calib, joint_name, arm_side)
    # Skip shadow joints (indices 2, 4, 11, 13)
    joint_mapping = [
        # Right arm: recorded[9-17] → ctrl[0-6]
        (9,  0, right_calib, "waist", "right"),
        (10, 1, right_calib, "shoulder", "right"),
        # skip 11 (shoulder_shadow)
        (12, 2, right_calib, "elbow", "right"),
        # skip 13 (elbow_shadow)
        (14, 3, right_calib, "forearm_roll", "right"),
        (15, 4, right_calib, "wrist_angle", "right"),
        (16, 5, right_calib, "wrist_rotate", "right"),
        (17, 6, right_calib, "gripper", "right"),
        # Left arm: recorded[0-8] → ctrl[7-13]
        (0,  7, left_calib, "waist", "left"),
        (1,  8, left_calib, "shoulder", "left"),
        # skip 2 (shoulder_shadow)
        (3,  9, left_calib, "elbow", "left"),
        # skip 4 (elbow_shadow)
        (5, 10, left_calib, "forearm_roll", "left"),
        (6, 11, left_calib, "wrist_angle", "left"),
        (7, 12, left_calib, "wrist_rotate", "left"),
        (8, 13, left_calib, "gripper", "left"),
    ]
    
    print("[INFO] Converting using PI05 normalization method:")
    print("       normalized_degrees = (raw - mid) * 360 / max_res")
    print("       where mid = (range_min + range_max) / 2, max_res = 4095")
    
    for frame_idx in range(num_frames):
        for rec_idx, ctrl_idx, calib, joint_name, arm_side in joint_mapping:
            lerobot_val = actions_raw[frame_idx, rec_idx]
            
            if joint_name == "gripper":
                # Gripper: percentage → Interbotix radians (with arm-specific calibration)
                mujoco_rad = gripper_lerobot_to_interbotix(lerobot_val, arm_side)
            else:
                # Regular joint: PI05 normalized degrees → raw → radians
                range_min = calib[joint_name]["range_min"]
                range_max = calib[joint_name]["range_max"]
                raw = normalized_degrees_to_raw(lerobot_val, range_min, range_max)
                mujoco_rad = raw_encoder_to_radians(raw)
            
            ctrl_sequence[frame_idx, ctrl_idx] = mujoco_rad
        
        # Debug: print first frame conversion
        if frame_idx == 0:
            print(f"       Frame 0 sample conversions:")
            for rec_idx, ctrl_idx, calib, joint_name, arm_side in joint_mapping[:3]:  # First 3 joints
                lerobot_val = actions_raw[0, rec_idx]
                if joint_name != "gripper":
                    range_min = calib[joint_name]["range_min"]
                    range_max = calib[joint_name]["range_max"]
                    mid = (range_min + range_max) / 2
                    raw = normalized_degrees_to_raw(lerobot_val, range_min, range_max)
                    mujoco_val = raw_encoder_to_radians(raw)
                    print(f"         {joint_name} ({arm_side}): {lerobot_val:.2f}° (normalized) → "
                          f"mid={mid:.1f}, raw={raw:.1f} → {mujoco_val:.4f} rad ({np.rad2deg(mujoco_val):.2f}°)")
    
    return ctrl_sequence


def convert_actions_to_mujoco_absolute(actions_raw: np.ndarray, mujoco_keyframe_ctrl: np.ndarray, 
                                        gripper_ctrl_range: tuple = (0.002, 0.041)):
    """
    ABSOLUTE: Convert lerobot calibrated degrees to MuJoCo (Interbotix) radians.
    
    Uses the SAME conversion as convert_poses.py:lerobot_to_interbotix():
        1. calibrated_degrees → raw_encoder: raw = deg/180*(resolution/2) - homing_offset (* -1 if drive_mode)
        2. raw_encoder → interbotix_radians: rad = (raw - 2048) * (2π / 4096)
    
    Calibration data from aloha/.cache/calibration/aloha_default/*.json
    
    Args:
        gripper_ctrl_range: (min, max) control range for gripper from XML (default: 0.002, 0.041)
    """
    import json
    from pathlib import Path
    
    num_frames = len(actions_raw)
    ctrl_sequence = np.zeros((num_frames, 14))
    
    # Load calibration files (relative to project root)
    project_root = Path(__file__).parent.parent
    calib_dir = project_root / "aloha" / ".cache" / "calibration" / "aloha_default"
    
    # Try to load calibration, fall back to hardcoded defaults if not found
    try:
        with open(calib_dir / "right_follower.json") as f:
            right_calib = json.load(f)
        with open(calib_dir / "left_follower.json") as f:
            left_calib = json.load(f)
        print(f"[INFO] Loaded calibration from: {calib_dir}")
    except FileNotFoundError:
        print(f"[WARN] Calibration files not found at {calib_dir}, using defaults")
        # Default calibration (from convert_poses.py)
        right_calib = {
            "homing_offset": [-1024, 0, 0, -2048, -2048, -1024, -1024, -1024, -1024],
            "drive_mode": [0, 0, 0, 0, 0, 0, 0, 0, 0],
            "motor_names": ["waist", "shoulder", "shoulder_shadow", "elbow", "elbow_shadow", 
                           "forearm_roll", "wrist_angle", "wrist_rotate", "gripper"]
        }
        left_calib = right_calib.copy()
    
    def calibrated_degrees_to_raw(degrees: float, homing_offset: int, drive_mode: int, resolution: int = 4096) -> float:
        """Convert lerobot calibrated degrees to raw encoder value."""
        value = degrees / 180 * (resolution // 2)
        value -= homing_offset
        if drive_mode:
            value *= -1
        return value
    
    def raw_encoder_to_radians(raw: float) -> float:
        """Convert raw encoder to Interbotix/MuJoCo radians."""
        return (raw - 2048) * (2 * np.pi) / 4096
    
   
    gripper_min, gripper_max = gripper_ctrl_range
    gripper_range = gripper_max - gripper_min
    
    # Calculate slope and intercept from control range and lerobot percentage range
    RIGHT_GRIPPER_SLOPE = (LEROBOT_CLOSED_PCT - LEROBOT_OPEN_PCT) / gripper_range
    RIGHT_GRIPPER_INTERCEPT = LEROBOT_OPEN_PCT - RIGHT_GRIPPER_SLOPE * gripper_min
    
    
    # Left arm gripper calibration (using same as right for now)
    LEFT_GRIPPER_SLOPE = RIGHT_GRIPPER_SLOPE
    LEFT_GRIPPER_INTERCEPT = RIGHT_GRIPPER_INTERCEPT
    
    def gripper_lerobot_to_interbotix(lerobot_percent: float, arm_side: str = "right") -> float:
        """
        Convert gripper from lerobot percentage to Interbotix radians.
        """
        if arm_side == "right":
            return (lerobot_percent - RIGHT_GRIPPER_INTERCEPT) / RIGHT_GRIPPER_SLOPE
        else:  # left
            return (lerobot_percent - LEFT_GRIPPER_INTERCEPT) / LEFT_GRIPPER_SLOPE
   
    joint_mapping = [
        # Right arm: recorded[9-17] → ctrl[0-6]
        (9,  0, right_calib, 0, "right"),   # waist
        (10, 1, right_calib, 1, "right"),   # shoulder
        # skip 11 (shoulder_shadow)
        (12, 2, right_calib, 3, "right"),   # elbow
        # skip 13 (elbow_shadow)
        (14, 3, right_calib, 5, "right"),   # forearm_roll
        (15, 4, right_calib, 6, "right"),   # wrist_angle
        (16, 5, right_calib, 7, "right"),   # wrist_rotate
        (17, 6, right_calib, 8, "right"),   # gripper
        # Left arm: recorded[0-8] → ctrl[7-13]
        (0,  7, left_calib, 0, "left"),    # waist
        (1,  8, left_calib, 1, "left"),    # shoulder
        # skip 2 (shoulder_shadow)
        (3,  9, left_calib, 3, "left"),    # elbow
        # skip 4 (elbow_shadow)
        (5, 10, left_calib, 5, "left"),    # forearm_roll
        (6, 11, left_calib, 6, "left"),    # wrist_angle
        (7, 12, left_calib, 7, "left"),    # wrist_rotate
        (8, 13, left_calib, 8, "left"),    # gripper
    ]
    
    print("[INFO] Converting using calibration-based absolute conversion:")
    
    for frame_idx in range(num_frames):
        for rec_idx, ctrl_idx, calib, calib_idx, arm_side in joint_mapping:
            joint_name = calib["motor_names"][calib_idx]
            lerobot_val = actions_raw[frame_idx, rec_idx]
            
            if joint_name == "gripper":
                # Gripper: percentage → Interbotix radians (with arm-specific calibration)
                mujoco_rad = gripper_lerobot_to_interbotix(lerobot_val, arm_side)
            else:
                # Regular joint: calibrated degrees → raw → radians
                homing_offset = calib["homing_offset"][calib_idx]
                drive_mode = calib["drive_mode"][calib_idx]
                raw = calibrated_degrees_to_raw(lerobot_val, homing_offset, drive_mode)
                mujoco_rad = raw_encoder_to_radians(raw)
            
            ctrl_sequence[frame_idx, ctrl_idx] = mujoco_rad
        
        # Debug: print first frame conversion
        if frame_idx == 0:
            print(f"       Frame 0 sample conversions:")
            for rec_idx, ctrl_idx, calib, calib_idx, arm_side in joint_mapping[:3]:  # First 3 joints
                joint_name = calib["motor_names"][calib_idx]
                lerobot_val = actions_raw[0, rec_idx]
                mujoco_val = ctrl_sequence[0, ctrl_idx]
                print(f"         {joint_name} ({arm_side}): {lerobot_val:.2f}° → {mujoco_val:.4f} rad ({np.rad2deg(mujoco_val):.2f}°)")
    
    return ctrl_sequence


def convert_actions_to_mujoco(actions_raw: np.ndarray, mujoco_keyframe_ctrl: np.ndarray, 
                              use_absolute: bool = True, use_new_normalization: bool = False,
                              gripper_ctrl_range: tuple = (0.002, 0.041)):
    """
    Main conversion function. 
    
    Args:
        use_absolute: If True, use absolute motor positions with calibration offsets.
                      If False, use delta-based replay from keyframe.
        use_new_normalization: If True, use PI05 normalization method (degrees = (raw - mid) * 360 / max_res).
                  Only applies when use_absolute=True.
        gripper_ctrl_range: (min, max) control range for gripper from MuJoCo model.
    
    Both modes use the SAME joint mapping. The difference:
    - DELTA: ctrl = keyframe + sign * deg2rad(recorded[N] - recorded[0])
    - ABSOLUTE (legacy): ctrl = sign * deg2rad(recorded[N] - offset)
             where offset = recorded[0] - sign * rad2deg(keyframe)
    - ABSOLUTE (PI05): ctrl = rad((normalized_degrees * max_res / 360) + mid)
             where mid = (range_min + range_max) / 2, max_res = 4095
    
    At frame 0, both produce identical results (keyframe_ctrl).
    At frame N, ABSOLUTE uses true motor positions; DELTA uses relative change.
    """
    if use_absolute:
        if use_new_normalization:
            print("[INFO] Using PI05 normalization method (motor encoder → MuJoCo with PI05 calibration)")
            return convert_actions_to_mujoco_pi05(actions_raw, mujoco_keyframe_ctrl, gripper_ctrl_range)
        else:
            print("[INFO] Using ABSOLUTE joint replay (motor encoder → MuJoCo with legacy calibration)")
            return convert_actions_to_mujoco_absolute(actions_raw, mujoco_keyframe_ctrl, gripper_ctrl_range)
    else:
        print("[INFO] Using DELTA-based joint replay (from MuJoCo keyframe)")
        return convert_actions_to_mujoco_delta(actions_raw, mujoco_keyframe_ctrl)


# ============================================================================
# Main
# ============================================================================

# Mapping from MuJoCo camera names to dataset camera names
MUJOCO_TO_DATASET_CAM = {
    "wrist_cam_right": "cam_right_wrist",
    "wrist_cam_left": "cam_left_wrist", 
    "cam_high": "cam_high",
    "cam_low": "cam_low",
}

# ============================================================================
# Gripper calibration constants
# ============================================================================
# LEROBOT GRIPPER RANGE: Change these if your dataset uses different values
# LEROBOT_OPEN_PCT: The percentage value that means "fully open" in lerobot
# LEROBOT_CLOSED_PCT: The percentage value that means "fully closed" in lerobot
LEROBOT_OPEN_PCT = 140.0  # Change to 110.0 if lerobot uses 110% for fully open
LEROBOT_CLOSED_PCT = 0.0  # Typically 0% for fully closed

def parse_args():
    p = argparse.ArgumentParser(description="Compare recorded video with MuJoCo replay + composite")
    p.add_argument("--dataset-path", type=str, default="/home/tongmiao/Documents/pick_cuber",
                   help="Path to dataset directory (local) or repo_id (Hub)")
    p.add_argument("--dataset-root", type=str, default=None,
                   help="Root directory for local datasets (default: ~/.cache/huggingface/lerobot)")
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--camera", type=str, default="cam_right_wrist",
                   help="Dataset camera name (default: cam_right_wrist)")
    p.add_argument("--mujoco-camera", type=str, default="teleoperator_pov")
    p.add_argument("--composite-camera", type=str, default="wrist_cam_right",
                   help="MuJoCo camera for composite view (default: wrist_cam_right)")
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--scene-path", type=str, default="pointclouds/N Goodwin Ave_w_o_Arm.npz",
                   help="Path to Gaussian Splatting scene file")
    p.add_argument("--replay-mode", type=str, default="absolute", choices=["absolute", "delta"],
                   help="Joint replay mode: 'absolute' uses calibrated motor positions, 'delta' uses keyframe + deltas")
    p.add_argument("--new", action="store_true",
                   help="Use PI05 normalization method: degrees = (raw - mid) * 360 / max_res")
    return p.parse_args()


def main():
    args = parse_args()
    
    # Load dataset (v3.0 only)
    episode_data = load_episode(args.dataset_path, args.episode, dataset_root=args.dataset_root)
    actions_raw = episode_data["action"].numpy()
    num_frames = len(actions_raw)
    
    # Get video path using dataset's method
    camera_key = f"observation.images.{args.camera}"
    dataset = episode_data["dataset"]
    
    # Get episode metadata for video path
    ep_meta = dataset.meta.episodes[args.episode]
    chunk_idx = ep_meta["data/chunk_index"]
    file_idx = ep_meta["data/file_index"]
    
    # Construct video path (v3.0 format)
    # get_video_file_path returns a relative path, need to combine with dataset root
    video_path_rel = dataset.meta.get_video_file_path(args.episode, camera_key)
    video_path = dataset.root / video_path_rel
    video_start_frame = episode_data.get('video_start_frame', 0)
    
    print(f"[INFO] v3.0: Episode {args.episode} starts at video frame {video_start_frame}")
    print(f"[INFO] Video path: {video_path}")
    
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    print(f"[INFO] Loading recorded video: {video_path}")
    print(f"[INFO] Episode has {num_frames} frames")
    
    # Use dataset's video loading which handles AV1 properly
    # Get episode metadata for video timestamps
    video_fps = dataset.fps
    print(f"[INFO] Video FPS: {video_fps}")
    
    # Calculate timestamps for all frames in the episode
    # Episode starts at from_timestamp in the video file (relative timestamps within episode)
    # Then we shift by from_timestamp to get absolute timestamps in the video file
    from_timestamp = ep_meta.get(f"videos/{camera_key}/from_timestamp", 0.0)
    # Relative timestamps within the episode (0, 1/fps, 2/fps, ...)
    relative_timestamps = [i / video_fps for i in range(num_frames)]
    # Absolute timestamps in the video file (shifted by from_timestamp)
    absolute_timestamps = [from_timestamp + ts for ts in relative_timestamps]
    
    # Pre-load all video frames using dataset's decoder (handles AV1)
    print(f"[INFO] Loading {num_frames} video frames from timestamp {from_timestamp:.3f}s...")
    video_frames_tensor = decode_video_frames(
        video_path, 
        absolute_timestamps, 
        tolerance_s=1e-4,
        backend="pyav"  # Use pyav instead of torchcodec to avoid FFmpeg library issues
    )
    # Convert from torch tensor (N, C, H, W) to list of numpy (H, W, C) for OpenCV
    video_frames = []
    for i in range(video_frames_tensor.shape[0]):
        frame = video_frames_tensor[i].permute(1, 2, 0).cpu().numpy()  # (H, W, C)
        frame = (frame * 255).astype(np.uint8)  # Convert from [0,1] to [0,255]
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # RGB to BGR for OpenCV
        video_frames.append(frame_bgr)
    
    print(f"[INFO] Loaded {len(video_frames)} video frames")
    video_frame_count = len(video_frames)
    
    # Load MuJoCo model
    # Change to aloha directory so relative includes resolve correctly
    project_root = Path(__file__).parent.parent
    aloha_dir = project_root / "aloha"
    original_cwd = os.getcwd()
    try:
        os.chdir(str(aloha_dir))
        model = MjModel.from_xml_path("robolab_setup.xml")
    finally:
        os.chdir(original_cwd)
    data = MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco_keyframe_ctrl = data.ctrl.copy()
    
    # Read gripper control range from model (right gripper is actuator 6)
    # actuator_ctrlrange is (n_actuators, 2) with [min, max] for each
    right_gripper_actuator_id = 6
    gripper_ctrl_range = (
        model.actuator_ctrlrange[right_gripper_actuator_id, 0],
        model.actuator_ctrlrange[right_gripper_actuator_id, 1]
    )
    print(f"[INFO] Gripper control range from XML: [{gripper_ctrl_range[0]}, {gripper_ctrl_range[1]}]")
    
    # Renderers
    RENDER_W, RENDER_H = 640, 480
    renderer = mujoco.Renderer(model, height=RENDER_H, width=RENDER_W)
    seg_renderer = mujoco.Renderer(model, height=RENDER_H, width=RENDER_W)
    seg_renderer.enable_segmentation_rendering()
    
    # Get robot geom IDs for masking
    robot_geom_ids = get_robot_geom_ids(model)
    print(f"[INFO] Found {len(robot_geom_ids)} robot geoms for masking")
    
    # Load Gaussian Splatting scene
    scene_data = None
    scene_depth_data = None
    gaussian_available = False
    
    if os.path.exists(args.scene_path):
        try:
            composite_cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, args.composite_camera)
            if composite_cam_id == -1:
                print(f"[WARN] Composite camera '{args.composite_camera}' not found, using teleoperator_pov")
                args.composite_camera = "teleoperator_pov"
                composite_cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, args.composite_camera)
            
            # Get intrinsics with proper fallback to fovy=45°
            k = get_camera_intrinsics_from_model(model, composite_cam_id, RENDER_W, RENDER_H)
            
            # Update MuJoCo's fovy to match (for renderer alignment)
            fy = k[1, 1]
            model.cam_fovy[composite_cam_id] = np.degrees(2 * np.arctan(RENDER_H / (2 * fy)))
            
            # Get initial w2c
            mujoco.mj_forward(model, data)
            init_pose = get_mujoco_camera_pose(model, data, args.composite_camera)
            w2c_init = mj_pose_to_gaussian_w2c(init_pose, T_splat2mj)
            
            # Check if diff_gaussian_rasterization is available before loading scene
            try:
                from diff_gaussian_rasterization import GaussianRasterizer
                from diff_gaussian_rasterization import GaussianRasterizationSettings
            except ImportError:
                print("[ERROR] diff_gaussian_rasterization not installed!")
                print("[INFO] To install, run:")
                print("  git clone https://github.com/graphdeco-inria/diff-gaussian-rasterization")
                print("  cd diff-gaussian-rasterization")
                print("  pip install .")
                print("[INFO] Composite window will show MuJoCo only (no background)")
                gaussian_available = False
                raise
            
            scene_data, scene_depth_data = load_scene_data(args.scene_path, w2c_init, k)
            gaussian_available = True
            print(f"[INFO] Loaded Gaussian Splatting scene from: {args.scene_path}")
        except ImportError:
            # Already printed error message above
            gaussian_available = False
        except Exception as e:
            print(f"[WARN] Failed to load Gaussian Splatting: {e}")
            import traceback
            traceback.print_exc()
            print("[INFO] Composite window will show MuJoCo only (no background)")
            gaussian_available = False
    else:
        print(f"[WARN] Scene file not found: {args.scene_path}")
        print("[INFO] Composite window will show MuJoCo only (no background)")
    
    viz_cfg = {
        'viz_w': RENDER_W, 'viz_h': RENDER_H,
        'viz_near': 0.1, 'viz_far': 10.0
    }
    
    # Convert actions
    print("[INFO] Converting actions to MuJoCo format...")
    use_absolute = (args.replay_mode == "absolute")
    ctrl_sequence = convert_actions_to_mujoco(actions_raw, mujoco_keyframe_ctrl, 
                                               use_absolute=use_absolute,
                                               use_new_normalization=args.new,
                                               gripper_ctrl_range=gripper_ctrl_range)
    
    # Create windows
    window_recorded = f"Recorded: {args.camera}"
    window_mujoco = f"MuJoCo: {args.mujoco_camera}"
    window_composite = f"Composite: {args.composite_camera}"
    
    cv2.namedWindow(window_recorded, cv2.WINDOW_NORMAL)
    cv2.namedWindow(window_mujoco, cv2.WINDOW_NORMAL)
    cv2.namedWindow(window_composite, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_recorded, 640, 480)
    cv2.resizeWindow(window_mujoco, 640, 480)
    cv2.resizeWindow(window_composite, 640, 480)
    
    cv2.moveWindow(window_recorded, 50, 100)
    cv2.moveWindow(window_mujoco, 700, 100)
    cv2.moveWindow(window_composite, 1350, 100)
    
    print("[INFO] Starting synchronized playback (press 'q' to quit, SPACE to pause)")
    
    frame_delay = int(1000 / args.fps)
    paused = False
    frame_idx = 0
    
    while frame_idx < min(num_frames, video_frame_count):
        if not paused:
            # Get pre-loaded video frame
            if frame_idx >= len(video_frames):
                print(f"[WARN] Frame index {frame_idx} out of range")
                break
            recorded_frame = video_frames[frame_idx]
            
            # Apply ctrl to MuJoCo
            data.ctrl[:] = ctrl_sequence[frame_idx]
            TIMESTEP = 1.0 / args.fps
            sim_time_target = data.time + TIMESTEP
            while data.time < sim_time_target:
                mujoco.mj_step(model, data)
            
            # Render MuJoCo view (teleoperator_pov)
            renderer.update_scene(data, camera=args.mujoco_camera)
            mujoco_rgb = renderer.render()
            mujoco_frame = cv2.cvtColor(mujoco_rgb, cv2.COLOR_RGB2BGR)
            
            # Render composite from wrist camera
            renderer.update_scene(data, camera=args.composite_camera)
            fg_rgb = renderer.render()
            fg_bgr = cv2.cvtColor(fg_rgb, cv2.COLOR_RGB2BGR)
            
            # Get segmentation mask
            seg_renderer.update_scene(data, camera=args.composite_camera)
            seg_mask = seg_renderer.render()
            seg_labels = seg_mask[:, :, 0].astype(np.int32)
            seg_labels[seg_labels == -1] = 0
            robot_mask = np.isin(seg_labels, list(robot_geom_ids))
            mask_uint8 = (robot_mask.astype(np.uint8)) * 255
            
            # Composite with Gaussian background
            if gaussian_available and scene_data is not None:
                try:
                    camera_pose = get_mujoco_camera_pose(model, data, args.composite_camera)
                    w2c = mj_pose_to_gaussian_w2c(camera_pose, T_splat2mj)
                    bg_im = render_gaussian(w2c, k, scene_data, scene_depth_data, viz_cfg)
                    bg_np = bg_im.permute(1, 2, 0).cpu().numpy()
                    bg_np = (bg_np * 255).astype(np.uint8)
                    bg_bgr = cv2.cvtColor(bg_np, cv2.COLOR_RGB2BGR)
                    
                    # Composite: background + foreground where mask is True
                    composite_frame = bg_bgr.copy()
                    composite_frame[mask_uint8 > 0] = fg_bgr[mask_uint8 > 0]
                except Exception as e:
                    # Log error on first frame only to avoid spam
                    if frame_idx == 0:
                        print(f"[WARN] Gaussian rendering failed: {e}")
                        import traceback
                        traceback.print_exc()
                        print("[INFO] Falling back to MuJoCo only for composite view")
                    # Fallback to MuJoCo only
                    composite_frame = fg_bgr.copy()
            else:
                # No Gaussian available, just show MuJoCo
                composite_frame = fg_bgr.copy()
            
            # Get gripper values for display
            # Recorded gripper: index 17 = right gripper percentage
            recorded_gripper_pct = actions_raw[frame_idx, 17]
            # MuJoCo gripper: ctrl[6] = right gripper in meters
            mujoco_gripper_m = data.ctrl[6]
            # Convert MuJoCo gripper to percentage using range from XML and lerobot constants
            # min = open (LEROBOT_OPEN_PCT), max = closed (LEROBOT_CLOSED_PCT)
            # Inverse: percent = OPEN - ((mujoco_m - min) / (max - min)) * (OPEN - CLOSED)
            gripper_min, gripper_max = gripper_ctrl_range
            mujoco_gripper_pct = LEROBOT_OPEN_PCT - ((mujoco_gripper_m - gripper_min) / (gripper_max - gripper_min)) * (LEROBOT_OPEN_PCT - LEROBOT_CLOSED_PCT)
            
            # Add frame info overlay
            cv2.putText(recorded_frame, f"Frame: {frame_idx}/{num_frames}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(recorded_frame, f"R Gripper: {recorded_gripper_pct:.1f}%", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            cv2.putText(mujoco_frame, f"Frame: {frame_idx}/{num_frames}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(mujoco_frame, f"R Gripper: {mujoco_gripper_m*1000:.1f}mm", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            cv2.putText(composite_frame, f"Frame: {frame_idx}/{num_frames}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(composite_frame, f"R Gripper: {mujoco_gripper_m*1000:.1f}mm ({mujoco_gripper_pct:.1f}%)", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Display
            cv2.imshow(window_recorded, recorded_frame)
            cv2.imshow(window_mujoco, mujoco_frame)
            cv2.imshow(window_composite, composite_frame)
            
            frame_idx += 1
        
        # Handle keyboard input
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
    
    # Cleanup
    cv2.destroyAllWindows()
    print("[INFO] Playback finished")


if __name__ == "__main__":
    main()
