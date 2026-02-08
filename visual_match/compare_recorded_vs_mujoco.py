#!/usr/bin/env python3
# compare_recorded_vs_mujoco.py
"""
Side-by-side comparison of recorded dataset video and MuJoCo simulation.

Shows synchronized windows in a 3-row layout:
Row 1: Right wrist - Recorded, Composite, Alpha
Row 2: Left wrist - Recorded, Composite, Alpha
Row 3: MuJoCo teleoperator view

Usage:
    python compare_recorded_vs_mujoco.py --dataset-path /path/to/dataset --episode 0
"""

import sys
import os
import re
import argparse
from pathlib import Path

# Add src to path for lerobot imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# ============================================================================
# Camera Configuration
# ============================================================================
# Maps dataset camera names to MuJoCo camera names
CAMERA_CONFIG = {
    "right_wrist": {
        "dataset_cam": "cam_right_wrist",      # Dataset camera key
        "mujoco_cam": "wrist_cam_right",       # MuJoCo camera name
    },
    "left_wrist": {
        "dataset_cam": "cam_left_wrist",
        "mujoco_cam": "wrist_cam_left",
    },
}

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

# Unified conversion module
from mujoco_lerobot_conversion import (
    convert_actions_to_mujoco,
    LEROBOT_OPEN_PCT,
    LEROBOT_CLOSED_PCT,
    MuJoCoLeRobotConverter,
    normalized_degrees_to_raw,
    raw_encoder_to_radians,
)

# ============================================================================
# Forward Kinematics comparison utilities
# ============================================================================

def rotation_matrix_to_euler(R: np.ndarray) -> np.ndarray:
    """
    Convert rotation matrix to Euler angles (XYZ convention).
    
    Args:
        R: 3x3 rotation matrix
    
    Returns:
        Array of [roll, pitch, yaw] in radians
    """
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


def get_end_effector_pose(model, data, site_name: str) -> tuple:
    """
    Get end effector pose from MuJoCo using forward kinematics.
    
    Args:
        model: MuJoCo model
        data: MuJoCo data (with updated qpos)
        site_name: Name of the end effector site (e.g., "right/gripper")
    
    Returns:
        Tuple of (position, rotation_matrix) where:
            - position: (3,) array [x, y, z] in meters
            - rotation_matrix: (3, 3) rotation matrix
    """
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if site_id == -1:
        raise ValueError(f"Site '{site_name}' not found in model")
    
    # Run forward kinematics
    mujoco.mj_forward(model, data)
    
    # Get site pose
    pos = data.site_xpos[site_id].copy()
    rot_mat = data.site_xmat[site_id].reshape(3, 3).copy()
    
    return pos, rot_mat


def lerobot_state_to_mujoco_qpos(state: np.ndarray, converter: MuJoCoLeRobotConverter) -> np.ndarray:
    """
    Convert LeRobot observation.state to MuJoCo qpos format.
    
    LeRobot state format: [left_arm(9), right_arm(9)] = 18 total
        Left:  [0]=waist, [1]=shoulder, [2]=shoulder_shadow, [3]=elbow, [4]=elbow_shadow,
               [5]=forearm_roll, [6]=wrist_angle, [7]=wrist_rotate, [8]=gripper
        Right: [9]=waist, [10]=shoulder, [11]=shoulder_shadow, [12]=elbow, [13]=elbow_shadow,
               [14]=forearm_roll, [15]=wrist_angle, [16]=wrist_rotate, [17]=gripper
    
    MuJoCo qpos format: [right_arm(8), left_arm(8)] = 16 total
        Right: [0]=waist, [1]=shoulder, [2]=elbow, [3]=forearm_roll, [4]=wrist_angle, 
               [5]=wrist_rotate, [6]=left_finger, [7]=right_finger
        Left:  [8]=waist, [9]=shoulder, [10]=elbow, [11]=forearm_roll, [12]=wrist_angle,
               [13]=wrist_rotate, [14]=left_finger, [15]=right_finger
    
    Args:
        state: LeRobot observation.state array (18,)
        converter: MuJoCoLeRobotConverter instance for calibration
    
    Returns:
        MuJoCo qpos array (16,)
    """
    qpos = np.zeros(16, dtype=np.float32)
    
    # Joint mapping: (lerobot_idx, qpos_idx, joint_name, arm_side)
    # Note: We skip shadow joints in lerobot and use the main joint value
    joint_mapping = [
        # Right arm: lerobot[9-17] → qpos[0-7]
        (9,  0, "waist", "right"),
        (10, 1, "shoulder", "right"),
        (12, 2, "elbow", "right"),       # skip 11 (shadow)
        (14, 3, "forearm_roll", "right"),
        (15, 4, "wrist_angle", "right"),
        (16, 5, "wrist_rotate", "right"),
        (17, 6, "gripper", "right"),     # left finger
        (17, 7, "gripper", "right"),     # right finger (same value)
        # Left arm: lerobot[0-8] → qpos[8-15]
        (0,  8, "waist", "left"),
        (1,  9, "shoulder", "left"),
        (3, 10, "elbow", "left"),        # skip 2 (shadow)
        (5, 11, "forearm_roll", "left"),
        (6, 12, "wrist_angle", "left"),
        (7, 13, "wrist_rotate", "left"),
        (8, 14, "gripper", "left"),      # left finger
        (8, 15, "gripper", "left"),      # right finger (same value)
    ]
    
    for lerobot_idx, qpos_idx, joint_name, arm_side in joint_mapping:
        lerobot_val = state[lerobot_idx]
        calib = converter.get_calib(arm_side)
        
        if joint_name == "gripper":
            # Convert gripper percentage to MuJoCo linear position
            mujoco_val = converter.gripper.lerobot_to_mujoco(lerobot_val, arm_side)
        else:
            # Convert joint angle from degrees to radians using calibration
            if converter.use_new_normalization:
                range_min = calib[joint_name]["range_min"]
                range_max = calib[joint_name]["range_max"]
                raw = normalized_degrees_to_raw(lerobot_val, range_min, range_max)
                mujoco_val = raw_encoder_to_radians(raw)
            else:
                # Absolute method
                from mujoco_lerobot_conversion import calibrated_degrees_to_raw
                calib_idx = {"waist": 0, "shoulder": 1, "elbow": 3, "forearm_roll": 5,
                            "wrist_angle": 6, "wrist_rotate": 7}[joint_name]
                homing_offset = calib["homing_offset"][calib_idx]
                drive_mode = calib["drive_mode"][calib_idx]
                raw = calibrated_degrees_to_raw(lerobot_val, homing_offset, drive_mode)
                mujoco_val = raw_encoder_to_radians(raw)
        
        qpos[qpos_idx] = mujoco_val
    
    return qpos


def compute_pose_difference(pos1: np.ndarray, rot1: np.ndarray, 
                           pos2: np.ndarray, rot2: np.ndarray) -> tuple:
    """
    Compute difference between two poses.
    
    Args:
        pos1, rot1: First pose (position and 3x3 rotation matrix)
        pos2, rot2: Second pose
    
    Returns:
        Tuple of (trans_diff, rot_diff) where:
            - trans_diff: Translation difference in meters [dx, dy, dz, norm]
            - rot_diff: Rotation difference in radians [roll, pitch, yaw, angle]
    """
    # Translation difference
    trans_diff_vec = pos1 - pos2
    trans_norm = np.linalg.norm(trans_diff_vec)
    trans_diff = np.array([trans_diff_vec[0], trans_diff_vec[1], trans_diff_vec[2], trans_norm])
    
    # Rotation difference: R_diff = R1 @ R2.T
    R_diff = rot1 @ rot2.T
    
    # Extract angle from rotation matrix (Frobenius approach)
    trace = np.trace(R_diff)
    trace = np.clip(trace, -1.0, 3.0)  # Numerical stability
    angle = np.arccos((trace - 1) / 2)
    
    # Also get Euler angles for detailed comparison
    euler_diff = rotation_matrix_to_euler(R_diff)
    rot_diff = np.array([euler_diff[0], euler_diff[1], euler_diff[2], angle])
    
    return trans_diff, rot_diff


def plot_fk_comparison(trans_errors: np.ndarray, rot_errors: np.ndarray, fps: float, save_path: str = None):
    """
    Plot translation and rotation errors over time.
    
    Args:
        trans_errors: Array of shape (N, 4) with [dx, dy, dz, norm] per frame
        rot_errors: Array of shape (N, 4) with [roll, pitch, yaw, angle] per frame
        fps: Frames per second for time axis
        save_path: Optional path to save plot
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not installed. Cannot plot FK comparison.")
        print("       Install with: pip install matplotlib")
        print("       Data has been saved to .npz file - you can plot later.")
        return
    
    num_frames = len(trans_errors)
    time_s = np.arange(num_frames) / fps
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Translation plot
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
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    # Add statistics
    mean_norm = np.mean(trans_errors[:, 3]) * 1000
    max_norm = np.max(trans_errors[:, 3]) * 1000
    ax1.text(0.02, 0.98, f'Mean: {mean_norm:.2f} mm\nMax: {max_norm:.2f} mm', 
             transform=ax1.transAxes, va='top', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Rotation plot
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
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    # Add statistics
    mean_angle = np.mean(np.degrees(rot_errors[:, 3]))
    max_angle = np.max(np.degrees(rot_errors[:, 3]))
    ax2.text(0.02, 0.98, f'Mean: {mean_angle:.2f}°\nMax: {max_angle:.2f}°', 
             transform=ax2.transAxes, va='top', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[INFO] Saved FK comparison plot to: {save_path}")
    
    plt.show()


def plot_fk_from_file(npz_path: str, save_path: str = None):
    """
    Load FK comparison data from saved .npz file and plot.
    
    Usage:
        python -c "from compare_recorded_vs_mujoco import plot_fk_from_file; plot_fk_from_file('fk_comparison_ep0.npz')"
    
    Args:
        npz_path: Path to saved .npz file
        save_path: Optional path to save plot (default: replace .npz with .png)
    """
    data = np.load(npz_path)
    trans_errors = data['trans_errors']
    rot_errors = data['rot_errors']
    fps = float(data['fps'])
    episode = int(data['episode'])
    
    if save_path is None:
        save_path = npz_path.replace('.npz', '.png')
    
    print(f"[INFO] Loaded FK data from: {npz_path}")
    print(f"       Episode: {episode}, FPS: {fps}, Frames: {len(trans_errors)}")
    print(f"       Translation error: mean={np.mean(trans_errors[:, 3])*1000:.2f} mm, "
          f"max={np.max(trans_errors[:, 3])*1000:.2f} mm")
    print(f"       Rotation error:    mean={np.degrees(np.mean(rot_errors[:, 3])):.2f}°, "
          f"max={np.degrees(np.max(rot_errors[:, 3])):.2f}°")
    
    plot_fk_comparison(trans_errors, rot_errors, fps, save_path=save_path)


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
    """
    Load color transform from color_mapping.yaml file.
    
    Returns:
        A: Transform matrix (3, 6) for [R², G², B², R, G, B] terms
        b: Bias vector (3,), constant term
    """
    with open(yaml_path, 'r') as f:
        content = f.read()
    
    # Parse color_A matrix (flattened 18 values)
    a_match = re.search(r'color_A:\s*\[(.*?)\]', content, re.DOTALL)
    if not a_match:
        raise ValueError(f"Could not find color_A in {yaml_path}")
    a_values = [float(x.strip()) for x in a_match.group(1).replace('\n', '').split(',')]
    A = np.array(a_values, dtype=np.float32).reshape(3, 6)  # 3 channels, 6 features
    
    # Parse color_b vector (3 values)
    b_match = re.search(r'color_b:\s*\[(.*?)\]', content, re.DOTALL)
    if not b_match:
        raise ValueError(f"Could not find color_b in {yaml_path}")
    b_values = [float(x.strip()) for x in b_match.group(1).replace('\n', '').split(',')]
    b = np.array(b_values, dtype=np.float32)
    
    return A, b


def apply_color_transform(img: np.ndarray, A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Apply quadratic color transform to image.
    
    Args:
        img: Input image (H, W, 3) in BGR format, uint8 [0-255]
        A: Transform matrix (3, 6) for [R², G², B², R, G, B] terms
        b: Bias vector (3,), constant term
    
    Returns:
        Transformed image (H, W, 3) in BGR format, uint8 [0-255]
    """
    # Convert BGR to RGB for transform
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Flatten and normalize to [0, 1]
    flat = img_rgb.reshape(-1, 3).astype(np.float32) / 255.0
    
    # Apply quadratic transform: [R², G², B², R, G, B] @ A.T + b
    flat_aug = _get_aug(flat, add_ones=False)  # Shape: (N, 6)
    out = flat_aug @ A.T + b  # Shape: (N, 3)
    
    # Clip and convert back to uint8
    out = np.clip(out, 0.0, 1.0)
    out_rgb = (out.reshape(img_rgb.shape) * 255.0).astype(np.uint8)
    
    # Convert RGB back to BGR
    out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)
    
    return out_bgr

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
    """Returns a set of geom indices that belong to the robot and manipulatable objects (foreground)."""
    robot_body_ids = []
    for body_id in range(model.nbody):
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        # Include robot arms and any manipulatable objects (e.g., purple_cube)
        if body_name is not None and (body_name.startswith("left/") or 
                                      body_name.startswith("right/") or
                                      body_name in ["purple_cube"]):
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
# Action conversion - imported from mujoco_lerobot_conversion.py
# ============================================================================
# convert_actions_to_mujoco, convert_actions_to_mujoco_pi05,
# convert_actions_to_mujoco_absolute, convert_actions_to_mujoco_delta
# are imported from the unified conversion module


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


def parse_args():
    p = argparse.ArgumentParser(description="Compare recorded video with MuJoCo replay + composite")
    p.add_argument("--dataset-path", type=str, default="/home/tongmiao/Documents/pick_cuber",
                   help="Path to dataset directory (local) or repo_id (Hub)")
    p.add_argument("--dataset-root", type=str, default=None,
                   help="Root directory for local datasets (default: ~/.cache/huggingface/lerobot)")
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--mujoco-camera", type=str, default="teleoperator_pov")
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--scene-path", type=str, default="pointclouds/N Goodwin Ave_w_o_Arm.npz",
                   help="Path to Gaussian Splatting scene file")
    p.add_argument("--color-calib-path", type=str, 
                   default="calibration_pairs_wrist/calibrated/color_mapping.yaml",
                   help="Path to color calibration YAML file (optional)")
    p.add_argument("--replay-mode", type=str, default="absolute", choices=["absolute", "delta"],
                   help="Joint replay mode: 'absolute' uses calibrated motor positions, 'delta' uses keyframe + deltas")
    p.add_argument("--new", action="store_true",
                   help="Use PI05 normalization method: degrees = (raw - mid) * 360 / max_res")
    p.add_argument("--alpha", type=float, default=0.5,
                   help="Alpha value for blending (0.0 = fully real, 1.0 = fully robot). Default: 0.5")
    p.add_argument("--color-calibrate", action="store_true",
                   help="Apply color calibration to composite renderings for both wrist cameras")
    p.add_argument("--save-images", action="store_true",
                   help="Save frames 0,5,10,15,20 from right wrist to calibration_pairs_wrist/")
    return p.parse_args()


def main():
    args = parse_args()
    
    # Load dataset (v3.0 only)
    episode_data = load_episode(args.dataset_path, args.episode, dataset_root=args.dataset_root)
    actions_raw = episode_data["action"].numpy()
    num_frames = len(actions_raw)
    
    dataset = episode_data["dataset"]
    ep_meta = dataset.meta.episodes[args.episode]
    video_fps = dataset.fps
    print(f"[INFO] Video FPS: {video_fps}")
    
    # Calculate timestamps for all frames in the episode
    relative_timestamps = [i / video_fps for i in range(num_frames)]
    
    # Load video frames for both wrist cameras using CAMERA_CONFIG
    wrist_frames = {}  # Dict: "right_wrist" -> list of frames, "left_wrist" -> list of frames
    
    for wrist_key, cam_cfg in CAMERA_CONFIG.items():
        dataset_cam = cam_cfg["dataset_cam"]
        camera_key = f"observation.images.{dataset_cam}"
        
        try:
            video_path_rel = dataset.meta.get_video_file_path(args.episode, camera_key)
            video_path = dataset.root / video_path_rel
            
            if not video_path.exists():
                print(f"[WARN] Video not found for {wrist_key}: {video_path}")
                wrist_frames[wrist_key] = []
                continue
            
            print(f"[INFO] Loading {wrist_key} camera video: {video_path}")
            from_timestamp = ep_meta.get(f"videos/{camera_key}/from_timestamp", 0.0)
            absolute_timestamps = [from_timestamp + ts for ts in relative_timestamps]
            
            frames_tensor = decode_video_frames(
                video_path, 
                absolute_timestamps, 
                tolerance_s=1e-4,
                backend="pyav"
            )
            
            frames_list = []
            for i in range(frames_tensor.shape[0]):
                frame = frames_tensor[i].permute(1, 2, 0).cpu().numpy()
                frame = (frame * 255).astype(np.uint8)
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                frames_list.append(frame_bgr)
            
            wrist_frames[wrist_key] = frames_list
            print(f"[INFO] Loaded {len(frames_list)} frames for {wrist_key}")
            
        except Exception as e:
            print(f"[WARN] Failed to load {wrist_key} camera: {e}")
            wrist_frames[wrist_key] = []
    
    # Calculate total frame count
    video_frame_count = max(len(wrist_frames.get(k, [])) for k in CAMERA_CONFIG)
    video_frame_count = max(video_frame_count, num_frames)
    
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
    
    # Get camera intrinsics for all cameras in CAMERA_CONFIG
    camera_intrinsics = {}  # Maps wrist_key -> intrinsics matrix
    for wrist_key, cam_cfg in CAMERA_CONFIG.items():
        mujoco_cam = cam_cfg["mujoco_cam"]
        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, mujoco_cam)
        camera_intrinsics[wrist_key] = get_camera_intrinsics_from_model(model, cam_id, RENDER_W, RENDER_H)
    
    if os.path.exists(args.scene_path):
        try:
            # Get initial w2c using right wrist camera for scene loading
            mujoco.mj_forward(model, data)
            init_pose = get_mujoco_camera_pose(model, data, CAMERA_CONFIG["right_wrist"]["mujoco_cam"])
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
            
            scene_data, scene_depth_data = load_scene_data(args.scene_path, w2c_init, camera_intrinsics["right_wrist"])
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
    
    # Load color calibration if available
    color_calib = None
    if args.color_calib_path and os.path.exists(args.color_calib_path):
        try:
            color_A, color_b = load_color_mapping(args.color_calib_path)
            color_calib = (color_A, color_b)
            print(f"[INFO] Loaded color calibration from: {args.color_calib_path}")
        except Exception as e:
            print(f"[WARN] Failed to load color calibration: {e}")
            print("[INFO] Continuing without color calibration")
    else:
        print(f"[INFO] Color calibration file not found at: {args.color_calib_path}")
        print("[INFO] Continuing without color calibration")
    
    # Convert actions
    print("[INFO] Converting actions to MuJoCo format...")
    use_absolute = (args.replay_mode == "absolute")
    ctrl_sequence = convert_actions_to_mujoco(actions_raw, mujoco_keyframe_ctrl, 
                                               use_absolute=use_absolute,
                                               use_new_normalization=args.new,
                                               gripper_ctrl_range=gripper_ctrl_range)
    
    # Create windows - organized by wrist camera
    # Right wrist windows
    window_right_recorded = f"Right Wrist - Recorded"
    window_right_composite = f"Right Wrist - Composite"
    window_right_alpha = f"Right Wrist - Alpha"
    # Left wrist windows
    window_left_recorded = f"Left Wrist - Recorded"
    window_left_composite = f"Left Wrist - Composite"
    window_left_alpha = f"Left Wrist - Alpha"
    # MuJoCo teleoperator view
    window_mujoco = f"MuJoCo: {args.mujoco_camera}"
    
    # Create all windows with smaller size to fit 3 rows on screen
    WINDOW_W, WINDOW_H = 400, 300  # Smaller windows for 3-row layout
    for win in [window_right_recorded, window_right_composite, window_right_alpha,
                window_left_recorded, window_left_composite, window_left_alpha,
                window_mujoco]:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, WINDOW_W, WINDOW_H)
    
    # Arrange windows in 3-row layout (fits on 1920x1080 screen)
    # Horizontal spacing: 410px (400 + 10 gap)
    # Vertical spacing: 340px (300 + 40 for title bar)
    X_START, Y_START = 50, 30
    X_STEP, Y_STEP = 410, 340
    
    # Row 1: Right wrist (Recorded, Composite, Alpha)
    cv2.moveWindow(window_right_recorded, X_START, Y_START)
    cv2.moveWindow(window_right_composite, X_START + X_STEP, Y_START)
    cv2.moveWindow(window_right_alpha, X_START + 2*X_STEP, Y_START)
    # Row 2: Left wrist (Recorded, Composite, Alpha)
    cv2.moveWindow(window_left_recorded, X_START, Y_START + Y_STEP)
    cv2.moveWindow(window_left_composite, X_START + X_STEP, Y_START + Y_STEP)
    cv2.moveWindow(window_left_alpha, X_START + 2*X_STEP, Y_START + Y_STEP)
    # Row 3: MuJoCo (centered)
    cv2.moveWindow(window_mujoco, X_START + X_STEP, Y_START + 2*Y_STEP)
    
    print("[INFO] Starting synchronized playback (press 'q' to quit, SPACE to pause, +/- to adjust alpha)")
    print(f"[INFO] Alpha blending: {args.alpha:.2f} (robot foreground transparency)")
    alpha = args.alpha
    
    # =========================================================================
    # FK Comparison Setup
    # =========================================================================
    # Create converter for real-world state to MuJoCo qpos conversion
    fk_converter = MuJoCoLeRobotConverter(gripper_ctrl_range, use_new_normalization=args.new)
    
    # Get observation states from dataset
    observations_raw = episode_data["observation.state"].numpy()
    
    # Storage for FK errors
    trans_errors = []  # [dx, dy, dz, norm] per frame
    rot_errors = []    # [roll, pitch, yaw, angle] per frame
    
    # Find right gripper site for FK
    # Try different possible site names (use finger sites if gripper site doesn't exist)
    right_gripper_site = None
    for site_name in ["right/left_finger", "right/right_finger"]:
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if site_id != -1:
            right_gripper_site = site_name
            print(f"[INFO] Using site '{site_name}' for FK comparison (id={site_id})")
            break
    
    if right_gripper_site is None:
        # List available sites
        print("[WARN] Could not find right gripper site. Available sites:")
        for i in range(model.nsite):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i)
            print(f"  - {name}")
        print("[WARN] FK comparison will be disabled")
    
    # =========================================================================
    # Real-time FK Plot Setup
    # =========================================================================
    realtime_plot_enabled = False
    fig_fk = None
    ax_trans = None
    ax_rot = None
    line_dx = line_dy = line_dz = line_norm = None
    line_roll = line_pitch = line_yaw = line_angle = None
    
    if right_gripper_site is not None:
        try:
            import matplotlib
            matplotlib.use('TkAgg')  # Use TkAgg backend for real-time updates
            import matplotlib.pyplot as plt
            
            plt.ion()  # Enable interactive mode
            fig_fk, (ax_trans, ax_rot) = plt.subplots(2, 1, figsize=(10, 6))
            fig_fk.suptitle('FK Comparison: Real vs Simulated (Real-time)')
            
            # Initialize empty lines for translation
            line_dx, = ax_trans.plot([], [], 'r-', label='dx', alpha=0.7)
            line_dy, = ax_trans.plot([], [], 'g-', label='dy', alpha=0.7)
            line_dz, = ax_trans.plot([], [], 'b-', label='dz', alpha=0.7)
            line_norm, = ax_trans.plot([], [], 'k-', label='||d||', linewidth=2)
            ax_trans.set_xlim(0, num_frames / args.fps)
            ax_trans.set_ylim(-50, 150)  # mm
            ax_trans.set_xlabel('Time (s)')
            ax_trans.set_ylabel('Translation Error (mm)')
            ax_trans.legend(loc='upper right')
            ax_trans.grid(True, alpha=0.3)
            ax_trans.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            
            # Initialize empty lines for rotation
            line_roll, = ax_rot.plot([], [], 'r-', label='roll', alpha=0.7)
            line_pitch, = ax_rot.plot([], [], 'g-', label='pitch', alpha=0.7)
            line_yaw, = ax_rot.plot([], [], 'b-', label='yaw', alpha=0.7)
            line_angle, = ax_rot.plot([], [], 'k-', label='angle', linewidth=2)
            ax_rot.set_xlim(0, num_frames / args.fps)
            ax_rot.set_ylim(-15, 15)  # degrees
            ax_rot.set_xlabel('Time (s)')
            ax_rot.set_ylabel('Rotation Error (degrees)')
            ax_rot.legend(loc='upper right')
            ax_rot.grid(True, alpha=0.3)
            ax_rot.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            
            plt.tight_layout()
            fig_fk.canvas.draw()
            plt.pause(0.01)
            
            realtime_plot_enabled = True
            print("[INFO] Real-time FK plotting enabled")
        except Exception as e:
            print(f"[WARN] Could not enable real-time plotting: {e}")
            print("[INFO] FK data will still be saved for later plotting")
    
    frame_delay = int(1000 / args.fps)
    paused = False
    frame_idx = 0
    
    while frame_idx < min(num_frames, video_frame_count):
        if not paused:
            # Apply ctrl to MuJoCo
            data.ctrl[:] = ctrl_sequence[frame_idx]
            TIMESTEP = 1.0 / args.fps
            sim_time_target = data.time + TIMESTEP
            while data.time < sim_time_target:
                mujoco.mj_step(model, data)
            
            # =================================================================
            # FK Comparison: compute end effector poses from real and sim
            # =================================================================
            if right_gripper_site is not None:
                # Get simulation end effector pose (already computed by mj_step)
                sim_pos, sim_rot = get_end_effector_pose(model, data, right_gripper_site)
                
                # Convert real-world observation.state to MuJoCo qpos
                real_state = observations_raw[frame_idx]
                real_qpos = lerobot_state_to_mujoco_qpos(real_state, fk_converter)
                
                # Create temporary data for real-world FK
                data_real = MjData(model)
                data_real.qpos[:16] = real_qpos[:16]
                mujoco.mj_forward(model, data_real)
                
                # Get real-world end effector pose via FK
                real_pos, real_rot = get_end_effector_pose(model, data_real, right_gripper_site)
                
                # Compute pose difference
                trans_diff, rot_diff = compute_pose_difference(real_pos, real_rot, sim_pos, sim_rot)
                trans_errors.append(trans_diff)
                rot_errors.append(rot_diff)
                
                # Update real-time plot
                if realtime_plot_enabled and len(trans_errors) > 1:
                    time_data = np.arange(len(trans_errors)) / args.fps
                    trans_arr = np.array(trans_errors)
                    rot_arr = np.array(rot_errors)
                    
                    # Update translation lines
                    line_dx.set_data(time_data, trans_arr[:, 0] * 1000)
                    line_dy.set_data(time_data, trans_arr[:, 1] * 1000)
                    line_dz.set_data(time_data, trans_arr[:, 2] * 1000)
                    line_norm.set_data(time_data, trans_arr[:, 3] * 1000)
                    
                    # Update rotation lines
                    line_roll.set_data(time_data, np.degrees(rot_arr[:, 0]))
                    line_pitch.set_data(time_data, np.degrees(rot_arr[:, 1]))
                    line_yaw.set_data(time_data, np.degrees(rot_arr[:, 2]))
                    line_angle.set_data(time_data, np.degrees(rot_arr[:, 3]))
                    
                    # Auto-scale Y axis if needed
                    max_trans = np.max(np.abs(trans_arr[:, :3])) * 1000
                    max_norm = np.max(trans_arr[:, 3]) * 1000
                    if max_norm > 100 or max_trans > 40:
                        ax_trans.set_ylim(-max(50, max_trans * 1.2), max(150, max_norm * 1.2))
                    
                    max_rot = np.max(np.abs(rot_arr[:, :3]))
                    max_angle = np.max(rot_arr[:, 3])
                    if np.degrees(max_angle) > 10:
                        ax_rot.set_ylim(-max(15, np.degrees(max_rot) * 1.2), 
                                       max(15, np.degrees(max_angle) * 1.2))
                    
                    # Redraw (only every 5 frames to avoid slowdown)
                    if frame_idx % 5 == 0:
                        fig_fk.canvas.draw_idle()
                        fig_fk.canvas.flush_events()
            
            # Render MuJoCo view (teleoperator_pov)
            renderer.update_scene(data, camera=args.mujoco_camera)
            mujoco_rgb = renderer.render()
            mujoco_frame = cv2.cvtColor(mujoco_rgb, cv2.COLOR_RGB2BGR)
            
            # =====================================================================
            # Render both wrist cameras using CAMERA_CONFIG loop
            # =====================================================================
            wrist_renders = {}  # Stores rendered frames for each wrist
            
            for wrist_key, cam_cfg in CAMERA_CONFIG.items():
                mujoco_cam = cam_cfg["mujoco_cam"]
                frames_list = wrist_frames.get(wrist_key, [])
                
                # Get recorded frame
                if frame_idx < len(frames_list):
                    recorded_frame = frames_list[frame_idx].copy()
                else:
                    recorded_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                
                # Render MuJoCo foreground
                renderer.update_scene(data, camera=mujoco_cam)
                fg_rgb = renderer.render()
                fg_bgr = cv2.cvtColor(fg_rgb, cv2.COLOR_RGB2BGR)
                
                # Get segmentation mask
                seg_renderer.update_scene(data, camera=mujoco_cam)
                seg_mask = seg_renderer.render()
                seg_labels = seg_mask[:, :, 0].astype(np.int32)
                seg_labels[seg_labels == -1] = 0
                robot_mask = np.isin(seg_labels, list(robot_geom_ids))
                mask_uint8 = (robot_mask.astype(np.uint8)) * 255
                
                # Composite with Gaussian background
                if gaussian_available and scene_data is not None:
                    try:
                        camera_pose = get_mujoco_camera_pose(model, data, mujoco_cam)
                        w2c = mj_pose_to_gaussian_w2c(camera_pose, T_splat2mj)
                        bg_im = render_gaussian(w2c, camera_intrinsics[wrist_key], scene_data, scene_depth_data, viz_cfg)
                        bg_np = bg_im.permute(1, 2, 0).cpu().numpy()
                        bg_np = (bg_np * 255).astype(np.uint8)
                        bg_bgr = cv2.cvtColor(bg_np, cv2.COLOR_RGB2BGR)
                        
                        composite_frame = bg_bgr.copy()
                        composite_frame[mask_uint8 > 0] = fg_bgr[mask_uint8 > 0]
                        
                        # Apply color calibration if enabled
                        if args.color_calibrate and color_calib is not None:
                            color_A, color_b = color_calib
                            composite_frame = apply_color_transform(composite_frame, color_A, color_b)
                    except Exception as e:
                        if frame_idx == 0:
                            print(f"[WARN] {wrist_key} Gaussian rendering failed: {e}")
                        composite_frame = fg_bgr.copy()
                else:
                    composite_frame = fg_bgr.copy()
                
                # Alpha blending
                alpha_mask = (mask_uint8 / 255.0).astype(np.float32)
                alpha_mask_3ch = np.stack([alpha_mask] * 3, axis=-1)
                foreground = fg_bgr.astype(np.float32)
                background = recorded_frame.astype(np.float32)
                blended = (alpha * foreground + (1 - alpha) * background) * alpha_mask_3ch + \
                          background * (1 - alpha_mask_3ch)
                alpha_frame = blended.astype(np.uint8)
                
                # Store results
                wrist_renders[wrist_key] = {
                    "recorded": recorded_frame,
                    "composite": composite_frame,
                    "alpha": alpha_frame,
                }
            
            # =====================================================================
            # Save frames for calibration (before adding overlay text)
            # =====================================================================
            SAVE_FRAMES = [0, 5, 10, 15, 20]
            if args.save_images and frame_idx in SAVE_FRAMES:
                # Create output directories
                gs_dir = Path("calibration_pairs_wrist/gs_renders")
                real_dir = Path("calibration_pairs_wrist/real_captures")
                gs_dir.mkdir(parents=True, exist_ok=True)
                real_dir.mkdir(parents=True, exist_ok=True)
                
                # Save right wrist composite (GS render) and recorded (real capture)
                # Note: saving copies without overlay text
                gs_path = gs_dir / f"frame_{frame_idx:04d}.png"
                real_path = real_dir / f"frame_{frame_idx:04d}.png"
                cv2.imwrite(str(gs_path), wrist_renders["right_wrist"]["composite"])
                cv2.imwrite(str(real_path), wrist_renders["right_wrist"]["recorded"])
                print(f"[INFO] Saved frame {frame_idx}: {gs_path}, {real_path}")
            
            # =====================================================================
            # Add overlay text to all frames
            # =====================================================================
            # Get gripper values for display
            recorded_gripper_pct = actions_raw[frame_idx, 17]
            mujoco_gripper_m = data.ctrl[6]
            gripper_min, gripper_max = gripper_ctrl_range
            mujoco_gripper_pct = LEROBOT_OPEN_PCT - ((mujoco_gripper_m - gripper_min) / (gripper_max - gripper_min)) * (LEROBOT_OPEN_PCT - LEROBOT_CLOSED_PCT)
            
            # Add overlays to wrist frames
            for wrist_key in CAMERA_CONFIG:
                for frame_type in ["recorded", "composite", "alpha"]:
                    frame = wrist_renders[wrist_key][frame_type]
                    cv2.putText(frame, f"Frame: {frame_idx}/{num_frames}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    if frame_type == "alpha":
                        cv2.putText(frame, f"Alpha: {alpha:.2f}", (10, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            # MuJoCo overlay
            cv2.putText(mujoco_frame, f"Frame: {frame_idx}/{num_frames}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(mujoco_frame, f"R Gripper: {mujoco_gripper_m*1000:.1f}mm", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # =====================================================================
            # Display all windows
            # =====================================================================
            # Row 1: Right wrist
            cv2.imshow(window_right_recorded, wrist_renders["right_wrist"]["recorded"])
            cv2.imshow(window_right_composite, wrist_renders["right_wrist"]["composite"])
            cv2.imshow(window_right_alpha, wrist_renders["right_wrist"]["alpha"])
            # Row 2: Left wrist
            cv2.imshow(window_left_recorded, wrist_renders["left_wrist"]["recorded"])
            cv2.imshow(window_left_composite, wrist_renders["left_wrist"]["composite"])
            cv2.imshow(window_left_alpha, wrist_renders["left_wrist"]["alpha"])
            # Row 3: MuJoCo
            cv2.imshow(window_mujoco, mujoco_frame)
            
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
        elif key == ord('+') or key == ord('='):  # + or = key
            alpha = min(1.0, alpha + 0.05)
            print(f"[INFO] Alpha: {alpha:.2f}")
        elif key == ord('-') or key == ord('_'):  # - or _ key
            alpha = max(0.0, alpha - 0.05)
            print(f"[INFO] Alpha: {alpha:.2f}")
    
    # Cleanup
    cv2.destroyAllWindows()
    print("[INFO] Playback finished")
    
    # Close real-time plot if it was enabled
    if realtime_plot_enabled and fig_fk is not None:
        try:
            import matplotlib.pyplot as plt
            plt.ioff()  # Disable interactive mode
        except:
            pass
    
    # =========================================================================
    # FK Comparison: Save and plot final results
    # =========================================================================
    if right_gripper_site is not None and len(trans_errors) > 0:
        trans_errors = np.array(trans_errors)
        rot_errors = np.array(rot_errors)
        
        # Save FK data to file for later plotting
        data_path = f"fk_comparison_ep{args.episode}.npz"
        np.savez(data_path, 
                 trans_errors=trans_errors, 
                 rot_errors=rot_errors,
                 fps=args.fps,
                 episode=args.episode)
        print(f"[INFO] Saved FK comparison data to: {data_path}")
        
        print(f"\n[INFO] FK Comparison Summary (Right Gripper):")
        print(f"       Translation error: mean={np.mean(trans_errors[:, 3])*1000:.2f} mm, "
              f"max={np.max(trans_errors[:, 3])*1000:.2f} mm")
        print(f"       Rotation error:    mean={np.degrees(np.mean(rot_errors[:, 3])):.2f}°, "
              f"max={np.degrees(np.max(rot_errors[:, 3])):.2f}°")
        
        # Save final plot (will gracefully skip if matplotlib not installed)
        save_path = f"fk_comparison_ep{args.episode}.png"
        if realtime_plot_enabled and fig_fk is not None:
            # Save the real-time figure
            try:
                fig_fk.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"[INFO] Saved FK comparison plot to: {save_path}")
                plt.show()  # Keep the final plot open
            except Exception as e:
                print(f"[WARN] Could not save plot: {e}")
        else:
            # Generate a new plot
            plot_fk_comparison(trans_errors, rot_errors, args.fps, save_path=save_path)
    else:
        print("[INFO] No FK comparison data to plot")


if __name__ == "__main__":
    main()
