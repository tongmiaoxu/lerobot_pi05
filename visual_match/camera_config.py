"""
Camera configuration loader for xArm7 setup.

Loads calibration data from JSON config files in visual_match/configs/
and computes derived transforms (camera-to-base, MuJoCo camera pose, etc.).

Cameras:
  - stationary_cam: RealSense D455 serial 246322303954
  - wrist_cam:      RealSense serial 213622251153

Usage:
    from camera_config import load_camera_config, get_all_cameras

    cam = load_camera_config("stationary_cam")
    K = cam["intrinsics"]
    cam_pos_mj = cam["cam_pos_mj"]
    cam_xmat_mj = cam["cam_xmat_mj"]
"""

import json
from pathlib import Path
import numpy as np
import cv2


_CONFIGS_DIR = Path(__file__).parent / "configs"

# MuJoCo xarm7/scene.xml places link_base at (0, 0, 0.12)
MJ_BASE_OFFSET = np.array([0.0, 0.0, 0.12])

# OpenCV camera convention -> MuJoCo camera convention
# OpenCV: +Z forward, +Y down. MuJoCo: -Z forward, +Y up. Flip Y and Z.
OPENCV_TO_MJ_FLIP = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])


def _compute_derived_transforms(cfg: dict) -> dict:
    """
    Compute derived transforms from raw calibration data.

    Input keys: rvec_cam2board, tvec_cam2board, R_base2world, t_base2world,
                intrinsics_640x480
    Output keys added: R_cam2board, t_cam2board, R_world2base, t_world2base,
                       R_cam2base, t_cam_in_base, cam_pos_mj, cam_xmat_mj,
                       intrinsics
    """
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

    intrinsics = np.array(cfg["intrinsics_640x480"])

    return {
        "R_cam2board": R_cam2board,
        "t_cam2board": t_cam2board,
        "R_world2base": R_world2base,
        "t_world2base": t_world2base,
        "R_cam2base": R_cam2base,
        "t_cam_in_base": t_cam_in_base,
        "cam_pos_mj": cam_pos_mj,
        "cam_xmat_mj": cam_xmat_mj,
        "intrinsics": intrinsics,
    }


def load_camera_config(cam_name: str) -> dict:
    """
    Load camera config from JSON and compute derived transforms.

    Args:
        cam_name: Camera name matching the JSON filename (e.g. "stationary_cam")

    Returns:
        Dict with raw config + all derived transforms.
    """
    config_path = _CONFIGS_DIR / f"{cam_name}.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Camera config not found: {config_path}")

    with open(config_path) as f:
        cfg = json.load(f)

    derived = _compute_derived_transforms(cfg)
    cfg.update(derived)
    return cfg


def get_all_cameras() -> dict:
    """
    Load all camera configs from the configs directory.

    Returns:
        Dict mapping camera name -> config dict.
    """
    cameras = {}
    for path in sorted(_CONFIGS_DIR.glob("*.json")):
        cam_name = path.stem
        cameras[cam_name] = load_camera_config(cam_name)
    return cameras


def set_mujoco_camera_from_config(data, model, cam_name: str, cam_cfg: dict) -> int:
    """
    Override a MuJoCo camera's pose with calibration data.

    Args:
        data: MuJoCo MjData
        model: MuJoCo MjModel
        cam_name: MuJoCo camera name in the XML
        cam_cfg: Camera config dict from load_camera_config()

    Returns:
        MuJoCo camera ID
    """
    import mujoco
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    if cam_id < 0:
        raise ValueError(f"Camera '{cam_name}' not found in MuJoCo model")
    data.cam_xpos[cam_id] = cam_cfg["cam_pos_mj"]
    data.cam_xmat[cam_id] = cam_cfg["cam_xmat_mj"].flatten()
    return cam_id
