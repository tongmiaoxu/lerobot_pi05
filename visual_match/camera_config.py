"""
Camera configuration loader for xArm7 setup.

Loads calibration data from JSON config files in visual_match/configs/
and computes derived transforms.

Two camera types are supported:

  "stationary" — fixed camera calibrated via ChArUco board.
      Config keys: rvec_cam2board, tvec_cam2board, R_base2world, t_base2world
      Derived: cam_pos_mj, cam_xmat_mj (fixed world-frame pose)

  "wrist" — camera mounted on the gripper, calibrated via hand-eye.
      Config keys: R_gripper2cam, t_gripper2cam, R_base2world, t_base2world
      Derived: mj_cam_pos_local, mj_cam_quat_local (pose relative to gripper body)
      The world-frame pose changes every frame via forward kinematics.

Usage:
    from camera_config import load_camera_config

    stat = load_camera_config("stationary_cam")
    K    = stat["intrinsics"]
    pos  = stat["cam_pos_mj"]      # fixed world position

    wrist = load_camera_config("wrist_cam")
    K     = wrist["intrinsics"]
    # wrist world pose comes from MuJoCo kinematics, not from config
"""

import json
from pathlib import Path
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as Rot


_CONFIGS_DIR = Path(__file__).parent / "configs"

# MuJoCo xarm7/scene.xml places link_base at (0, 0, 0.12)
MJ_BASE_OFFSET = np.array([0.0, 0.0, 0.12])

# OpenCV camera convention -> MuJoCo camera convention
# OpenCV: +Z forward, +Y down.  MuJoCo: -Z forward, +Y up.
OPENCV_TO_MJ_FLIP = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])


# -----------------------------------------------------------------------
# Stationary camera (board calibration)
# -----------------------------------------------------------------------

def _compute_stationary_transforms(cfg: dict) -> dict:
    rvec = np.array(cfg["rvec_cam2board"])
    tvec = np.array(cfg["tvec_cam2board"])

    R_board2cam = cv2.Rodrigues(rvec)[0]
    t_board2cam = tvec.ravel()

    R_cam2board = R_board2cam.T
    t_cam2board = -R_cam2board @ t_board2cam

    R_base2world = np.array(cfg["R_base2world"])
    t_base2world = np.array(cfg["t_base2world"])

    R_world2base = R_base2world.T
    t_world2base = -R_world2base @ t_base2world

    R_cam2base = R_world2base @ R_cam2board
    t_cam_in_base = R_world2base @ t_cam2board + t_world2base

    cam_pos_mj = t_cam_in_base + MJ_BASE_OFFSET
    cam_xmat_mj = R_cam2base @ OPENCV_TO_MJ_FLIP

    return {
        "R_cam2board": R_cam2board,
        "t_cam2board": t_cam2board,
        "R_world2base": R_world2base,
        "t_world2base": t_world2base,
        "R_cam2base": R_cam2base,
        "t_cam_in_base": t_cam_in_base,
        "cam_pos_mj": cam_pos_mj,
        "cam_xmat_mj": cam_xmat_mj,
        "intrinsics": np.array(cfg["intrinsics_640x480"]),
    }


# -----------------------------------------------------------------------
# Wrist camera (hand-eye calibration)
# -----------------------------------------------------------------------

def _compute_wrist_transforms(cfg: dict) -> dict:
    """
    Compute the MuJoCo-local camera pose from hand-eye calibration.

    R_gripper2cam / t_gripper2cam describe where the camera sits in the
    gripper (flange) frame, using OpenCV camera conventions (+Z forward,
    +Y down).

    We convert to MuJoCo camera conventions (-Z forward, +Y up) and
    produce a local position + quaternion that can be written into
    model.cam_pos / model.cam_quat so that mj_kinematics computes the
    correct world-frame pose every step.
    """
    R_g2c = np.array(cfg["R_gripper2cam"])
    t_g2c = np.array(cfg["t_gripper2cam"])
    R_cam2gripper = R_g2c.T
    R_z180 = np.array([[-1, 0, 0],
                       [ 0,-1, 0],
                       [ 0, 0, 1]], dtype=float)
    R_cam2gripper = R_z180 @ R_cam2gripper
    t_cam_in_gripper = -R_g2c.T @ t_g2c
    t_cam_in_gripper = R_z180 @ t_cam_in_gripper
    # Convert from OpenCV camera convention to MuJoCo camera convention
    # (flip on the RIGHT, same as stationary camera)
    R_cam2gripper_mj = R_cam2gripper @ OPENCV_TO_MJ_FLIP

    quat_xyzw = Rot.from_matrix(R_cam2gripper_mj).as_quat()     # [x,y,z,w] # [x,y,z,w]
    quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0],
                          quat_xyzw[1], quat_xyzw[2]])        # MuJoCo order

    R_base2world = np.array(cfg["R_base2world"])
    t_base2world = np.array(cfg["t_base2world"])
    R_world2base = R_base2world.T
    t_world2base = -R_world2base @ t_base2world

    return {
        "R_gripper2cam": R_g2c,
        "t_gripper2cam": t_g2c,
        "mj_cam_pos_local": t_cam_in_gripper,
        "mj_cam_quat_local": quat_wxyz,
        "R_world2base": R_world2base,
        "t_world2base": t_world2base,
        "intrinsics": np.array(cfg["intrinsics_640x480"]),
    }


# -----------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------

def load_camera_config(cam_name: str) -> dict:
    """
    Load camera config from JSON and compute derived transforms.

    Args:
        cam_name: Camera name matching the JSON filename
                  (e.g. "stationary_cam", "wrist_cam")

    Returns:
        Dict with raw config + all derived transforms.
        The "type" key is "stationary" or "wrist".
    """
    config_path = _CONFIGS_DIR / f"{cam_name}.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Camera config not found: {config_path}")

    with open(config_path) as f:
        cfg = json.load(f)

    cam_type = cfg.get("type", "stationary")
    if cam_type == "wrist":
        derived = _compute_wrist_transforms(cfg)
    else:
        derived = _compute_stationary_transforms(cfg)

    cfg.update(derived)
    return cfg


def get_all_cameras() -> dict:
    cameras = {}
    for path in sorted(_CONFIGS_DIR.glob("*.json")):
        cam_name = path.stem
        cameras[cam_name] = load_camera_config(cam_name)
    return cameras


def get_lerobot_cameras() -> dict:
    """
    Build lerobot camera config dict from visual_match configs.

    Maps stationary_cam -> cam_high, wrist_cam -> cam_wrist for use with
    lerobot-record and lerobot-teleoperate. Returns a dict of camera name
    -> config dict in the format expected by lerobot (type, serial_number_or_name,
    fps, width, height).

    Returns:
        Dict suitable for --robot.cameras or config.cameras, e.g.:
        {"cam_high": {...}, "cam_wrist": {...}}
    """
    lerobot_cameras = {}
    mapping = [
        ("stationary_cam", "cam_high"),
        ("wrist_cam", "cam_wrist"),
    ]
    for config_name, lerobot_key in mapping:
        config_path = _CONFIGS_DIR / f"{config_name}.json"
        if not config_path.exists():
            continue
        with open(config_path) as f:
            cfg = json.load(f)
        lerobot_cameras[lerobot_key] = {
            "type": "intelrealsense",
            "serial_number_or_name": cfg["serial_number"],
            "fps": cfg["fps"],
            "width": cfg["width"],
            "height": cfg["height"],
        }
    return lerobot_cameras


def set_mujoco_camera_from_config(data, model, cam_name: str, cam_cfg: dict) -> int:
    """
    Apply calibration to a MuJoCo camera.

    For stationary cameras: overrides world-frame cam_xpos / cam_xmat on data.
    For wrist cameras: patches the *model* local pose (cam_pos / cam_quat)
        so that mj_kinematics computes the correct world pose each step.

    Returns the MuJoCo camera ID.
    """
    import mujoco
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    if cam_id < 0:
        raise ValueError(f"Camera '{cam_name}' not found in MuJoCo model")

    cam_type = cam_cfg.get("type", "stationary")

    if cam_type == "wrist":
        model.cam_pos[cam_id] = cam_cfg["mj_cam_pos_local"]
        model.cam_quat[cam_id] = cam_cfg["mj_cam_quat_local"]
    else:
        data.cam_xpos[cam_id] = cam_cfg["cam_pos_mj"]
        data.cam_xmat[cam_id] = cam_cfg["cam_xmat_mj"].flatten()

    return cam_id
