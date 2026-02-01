#!/usr/bin/env python3
"""
Replay LeRobot v3.0 dataset in MuJoCo ALOHA simulation using ABSOLUTE joint positions.

================================================================================
DATA FORMAT: LeRobot v3.0 (.parquet data files)
================================================================================
- Recorded values are ABSOLUTE joint positions in DEGREES (motor encoder values)
- v3.0: 18 dimensions per frame: [left_arm(8), left_gripper(1), right_arm(8), right_gripper(1)]
- Includes "shadow joints" (shoulder_shadow, elbow_shadow) - mechanically coupled
  joints that mirror the primary joint for additional torque
================================================================================
| Issue                      | Solution                                        |
|----------------------------|-------------------------------------------------|
| Different dimensions       | Skip shadow joint indices (2, 4, 11, 13)        |
| (18 vs 14)                 | → MuJoCo doesn't model parallel linkage         |
|                            |                                                 |
| Different arm order        | Map left/right to MuJoCo's [right, left] order  |
| recorded: [left, right]    | → MuJoCo ctrl: [right_arm(7), left_arm(7)]      |
|                            |                                                 |
| Motor encoder calibration  | Uses calibration files from aloha/.cache/       |
|                            | → calibrated_degrees → raw_encoder → radians   |
|                            |                                                 |
| Absolute positions         | Direct conversion using calibration offsets     |
|                            | → No delta-based approach, true motor positions|
|                            |                                                 |
| Gripper percentage         | Linear mapping from lerobot % to MuJoCo meters |
|                            | Range read from XML, lerobot range configurable |
================================================================================

Recorded data layout (18-dim):
  Left arm:  [0:waist, 1:shoulder, 2:shoulder_shadow, 3:elbow, 4:elbow_shadow,
              5:forearm_roll, 6:wrist_angle, 7:wrist_rotate, 8:gripper]
  Right arm: [9:waist, 10:shoulder, 11:shoulder_shadow, 12:elbow, 13:elbow_shadow,
              14:forearm_roll, 15:wrist_angle, 16:wrist_rotate, 17:gripper]
"""

# -------- Standard imports & sys.path tweaks ---------------------------------
import sys, os, dataclasses, argparse
from pathlib import Path
import time
from typing import Optional

# Add src to path for lerobot imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Auto-detect if we have a display before importing mujoco
def _detect_display():
    """Check if a display is available (X11 on Linux, or macOS/Windows)."""
    if os.environ.get("DISPLAY"):
        return True
    if os.environ.get("WAYLAND_DISPLAY"):
        return True
    if sys.platform in ("darwin", "win32"):
        return True
    if os.environ.get("SSH_CONNECTION") and not os.environ.get("DISPLAY"):
        return False
    return False

_HAS_DISPLAY = _detect_display()
print(f"[INFO] Display detected: {_HAS_DISPLAY}")

if not _HAS_DISPLAY:
    os.environ["MUJOCO_GL"] = "egl"

import numpy as np
import torch

import mujoco
from mujoco import MjModel, MjData

if _HAS_DISPLAY:
    import mujoco.viewer

import plotly.graph_objects as go
import json

# Lerobot dataset loader
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# ============================================================================
# Gripper calibration constants
# ============================================================================
# LEROBOT GRIPPER RANGE: Change these if your dataset uses different values
# LEROBOT_OPEN_PCT: The percentage value that means "fully open" in lerobot
# LEROBOT_CLOSED_PCT: The percentage value that means "fully closed" in lerobot
LEROBOT_OPEN_PCT = 140.0  # Change to 110.0 if lerobot uses 110% for fully open
LEROBOT_CLOSED_PCT = 0.0  # Typically 0% for fully closed

# -------- Dataclasses ---------------------------------------------------------
@dataclasses.dataclass
class Args:
    default_prompt: Optional[str] = "pick cube"
    dataset_path: str = "/home/tongmiao/Documents/pick_cuber"
    dataset_root: Optional[str] = None
    action_key: str = "action"
    episode: int = 0
    fps: float = 30.0
    use_new_normalization: bool = False

def parse_cli() -> Args:
    p = argparse.ArgumentParser(
        description="Replay LeRobot v3.0 dataset in MuJoCo (no policy needed)."
    )
    p.add_argument("--default-prompt", type=str, default="pick cube")
    p.add_argument("--dataset-path", type=str, default="/home/tongmiao/Documents/pick_cuber",
                   help="Path to dataset directory (local) or repo_id (Hub)")
    p.add_argument("--dataset-root", type=str, default=None,
                   help="Root directory for local datasets (default: ~/.cache/huggingface/lerobot)")
    p.add_argument("--action-key", type=str, default="action")
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--fps", type=float, default=30.0, help="Replay frame rate (default: 30.0)")
    p.add_argument("--new", action="store_true",
                   help="Use PI05 normalization method: degrees = (raw - mid) * 360 / max_res")

    ns = p.parse_args()

    return Args(
        default_prompt=ns.default_prompt,
        dataset_path=ns.dataset_path,
        dataset_root=ns.dataset_root,
        action_key=ns.action_key,
        episode=ns.episode,
        fps=ns.fps,
        use_new_normalization=ns.new,
    )

# -------- Dataset loader for v3.0 using LeRobotDataset ----------------------
def load_episode(dataset_path: str, episode_idx: int, dataset_root: str | None = None):
    """
    Load a single episode from a LeRobot v3.0 dataset using LeRobotDataset.
    
    Args:
        dataset_path: Path to local dataset directory or repo_id for Hub dataset
        episode_idx: Episode index to load
        dataset_root: Root directory for local datasets (optional, used when dataset_path is repo_id)
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
    
    return {
        'action': actions_tensor,
        'observation.state': observations_tensor,
        'episode_index': episode_idx,
        'num_frames': num_frames,
    }

# -------- Observation builder -------------------------------------------------
# XML path - resolve relative to project root
_project_root = Path(__file__).parent.parent
XML_PATH = str(_project_root / "aloha" / "robolab_setup.xml")
RENDER_W, RENDER_H = 640, 480

def build_observation(model, data, recorded_observation, renderer, prompt):
    state = np.zeros((14,), np.float32)
    state[:7] = data.qpos[8:-1]
    state[7:] = data.qpos[:7]
    images = {}
    return {"state": state, "images": images, "prompt": prompt}

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
    except FileNotFoundError:
        raise FileNotFoundError(f"Calibration files not found at {calib_dir}, cannot proceed with PI05 normalization")
    
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
    
    # print("[INFO] Converting using PI05 normalization method:")
    # print("       normalized_degrees = (raw - mid) * 360 / max_res")
    # print("       where mid = (range_min + range_max) / 2, max_res = 4095")
    
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
        
    
    return ctrl_sequence

# -------- Absolute action conversion (from compare_recorded_vs_mujoco.py) ----
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
    
    # Mapping: recorded index → (mujoco ctrl index, calibration dict, joint index in calib)
    # Recorded format: [left_arm(9), right_arm(9)] = 18 total
    #   Left:  [0]=waist, [1]=shoulder, [2]=shoulder_shadow, [3]=elbow, [4]=elbow_shadow,
    #          [5]=forearm_roll, [6]=wrist_angle, [7]=wrist_rotate, [8]=gripper
    #   Right: [9]=waist, [10]=shoulder, [11]=shoulder_shadow, [12]=elbow, [13]=elbow_shadow,
    #          [14]=forearm_roll, [15]=wrist_angle, [16]=wrist_rotate, [17]=gripper
    #
    # MuJoCo ctrl format: [right_arm(7), left_arm(7)] = 14 total
    #   [0-6]: right waist, shoulder, elbow, forearm_roll, wrist_angle, wrist_rotate, gripper
    #   [7-13]: left waist, shoulder, elbow, forearm_roll, wrist_angle, wrist_rotate, gripper
    
    # Mapping: (recorded_idx, ctrl_idx, calib, calib_joint_idx, arm_side)
    # Skip shadow joints (indices 2, 4, 11, 13)
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
        
     
    return ctrl_sequence

# -------- Main ----------------------------------------------------------------
def main():
    args = parse_cli()

    print(f"[INFO] Loading dataset from: {args.dataset_path}")
    print(f"[INFO] Replaying episode {args.episode}")
    
    episode_data = load_episode(args.dataset_path, args.episode, dataset_root=args.dataset_root)
    
    # Extract raw actions (18-dim, degrees)
    actions_raw = episode_data[args.action_key].numpy()  # (num_frames, 18)
    observations = episode_data["observation.state"].numpy()
    num_frames = len(actions_raw)
    
    print(f"[INFO] Action shape: {actions_raw.shape}")
    print(f"[INFO] First frame action: {actions_raw[0]}")
    
    # ============================================================
    # Load MuJoCo model first to get gripper control range
    # ============================================================
    # Change to aloha directory so relative includes resolve correctly
    aloha_dir = _project_root / "aloha"
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
    right_gripper_actuator_id = 6
    gripper_ctrl_range = (
        model.actuator_ctrlrange[right_gripper_actuator_id, 0],
        model.actuator_ctrlrange[right_gripper_actuator_id, 1]
    )
    print(f"[INFO] Gripper control range from XML: [{gripper_ctrl_range[0]}, {gripper_ctrl_range[1]}]")
    
    # ============================================================
    # Choose conversion method based on --new flag
    # ============================================================
    if args.use_new_normalization:
        print("[INFO] Using PI05 normalization method (--new flag)")
        ctrl_sequence = convert_actions_to_mujoco_pi05(
            actions_raw, mujoco_keyframe_ctrl, gripper_ctrl_range
        )
    else:
        print("[INFO] Using legacy absolute calibration method")
        ctrl_sequence = convert_actions_to_mujoco_absolute(
            actions_raw, mujoco_keyframe_ctrl, gripper_ctrl_range
        )
    
    print(f"[INFO] Ctrl range: [{ctrl_sequence.min():.3f}, {ctrl_sequence.max():.3f}]")
    renderer = mujoco.Renderer(model, height=RENDER_H, width=RENDER_W)

    times = []
    grip_ch6 = []
    grip_ch13 = []
    gri_ch6_action = []
    gri_ch13_action = []

    def run_simulation(viewer=None):
        TIMESTEP = 1.0 / args.fps  # Match video frame rate
        print(f"[INFO] Using TIMESTEP={TIMESTEP:.4f}s (FPS={args.fps})")
        
        for frame_idx in range(num_frames):
            recorded_observation = observations[frame_idx]
            prompt = args.default_prompt or ""
            obs = build_observation(model, data, recorded_observation, renderer, prompt)
            
            # Apply ctrl directly (like working script)
            data.ctrl[:] = ctrl_sequence[frame_idx]
            
            # Step simulation forward by TIMESTEP
            sim_time_target = data.time + TIMESTEP
            while data.time < sim_time_target:
                mujoco.mj_step(model, data)
            
            # Sync viewer once per frame (not per physics step) for better performance
            if viewer is not None:
                viewer.sync()
            
            times.append(data.time)
            grip_ch6.append(recorded_observation[6])
            grip_ch13.append(recorded_observation[13])
            gri_ch6_action.append(obs['state'][6])
            gri_ch13_action.append(obs['state'][13])
            
            if frame_idx % 100 == 0:
                print(f"[INFO] Frame {frame_idx}/{num_frames}, ctrl[1]={data.ctrl[1]:.3f}")

    if _HAS_DISPLAY:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            run_simulation(viewer)
    else:
        print("[INFO] Running in headless mode (no viewer)")
        run_simulation(None)

    idx = list(range(len(grip_ch6)))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=idx, y=grip_ch6, mode='lines', name='grip_ch6'))
    fig.add_trace(go.Scatter(x=idx, y=grip_ch13, mode='lines', name='grip_ch13'))
    fig.add_trace(go.Scatter(x=idx, y=gri_ch6_action, mode='lines', name='gri_ch6_action'))
    fig.add_trace(go.Scatter(x=idx, y=gri_ch13_action, mode='lines', name='gri_ch13_action'))
    fig.update_layout(title='Gripper Actions Over Time', xaxis_title='Index', yaxis_title='Value')
    fig.write_html("gripper_actions_v16.html")
    print("Saved plot → gripper_actions_v16.html")
    print("[DONE] playback finished")

if __name__ == "__main__":
    main()
