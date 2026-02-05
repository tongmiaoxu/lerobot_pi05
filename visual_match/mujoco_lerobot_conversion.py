"""
MuJoCo ↔ LeRobot Conversion Utilities

Unified conversion functions for transforming data between MuJoCo (radians) and
LeRobot (degrees/percentage) formats.

Supports two normalization methods:
1. PI05 (--new): Uses range_min/range_max from .cache/calibration/aloha_follower/
   - normalized_degrees = (raw - mid) * 360 / max_res
   - Calibration files: aloha_left.json, aloha_right.json

2. Absolute (legacy): Uses homing_offset/drive_mode from aloha/.cache/calibration/aloha_default/
   - calibrated_degrees = (raw + offset) * 180 / (resolution // 2)
   - Calibration files: left_follower.json, right_follower.json

Usage:
    from mujoco_lerobot_conversion import (
        MuJoCoLeRobotConverter,
        get_calibration_dir,
    )
    
    # Create converter
    converter = MuJoCoLeRobotConverter(
        gripper_ctrl_range=(0.002, 0.041),
        use_new_normalization=True  # --new flag
    )
    
    # Convert actions: lerobot → mujoco
    ctrl_sequence = converter.actions_to_mujoco(actions_raw)
    
    # Convert state: mujoco → lerobot (for observations)
    state = converter.state_to_lerobot(qpos)
"""

import json
from pathlib import Path
from typing import Tuple, Optional
import numpy as np

# ============================================================================
# Gripper calibration constants
# ============================================================================
# LEROBOT GRIPPER RANGE: The percentage values in lerobot format
LEROBOT_OPEN_PCT = 140.0  # Fully open
LEROBOT_CLOSED_PCT = 0.0  # Fully closed

# PI05 constants
MAX_RES = 4095  # For ALOHA motors: 4096 - 1


def get_project_root() -> Path:
    """Get project root directory."""
    return Path(__file__).parent.parent


def get_calibration_dir(use_new_normalization: bool) -> Path:
    """
    Get calibration directory based on normalization method.
    
    Args:
        use_new_normalization: If True, use PI05 calibration, else use absolute calibration
    
    Returns:
        Path to calibration directory
    """
    project_root = get_project_root()
    if use_new_normalization:
        return project_root / ".cache" / "calibration" / "aloha_follower"
    else:
        return project_root / "aloha" / ".cache" / "calibration" / "aloha_default"


def load_calibration(use_new_normalization: bool) -> Tuple[dict, dict]:
    """
    Load calibration files for left and right arms.
    
    Args:
        use_new_normalization: If True, load PI05 calibration, else load absolute calibration
    
    Returns:
        Tuple of (left_calib, right_calib) dictionaries
    """
    calib_dir = get_calibration_dir(use_new_normalization)
    
    if use_new_normalization:
        # PI05: aloha_left.json, aloha_right.json
        try:
            with open(calib_dir / "aloha_left.json") as f:
                left_calib = json.load(f)
            with open(calib_dir / "aloha_right.json") as f:
                right_calib = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"PI05 calibration files not found at {calib_dir}. "
                f"Expected aloha_left.json and aloha_right.json"
            )
    else:
        # Absolute: left_follower.json, right_follower.json
        try:
            with open(calib_dir / "left_follower.json") as f:
                left_calib = json.load(f)
            with open(calib_dir / "right_follower.json") as f:
                right_calib = json.load(f)
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
    
    return left_calib, right_calib


# ============================================================================
# Low-level conversion functions
# ============================================================================

def raw_encoder_to_radians(raw: float) -> float:
    """Convert raw encoder value to Interbotix/MuJoCo radians."""
    return (raw - 2048) * (2 * np.pi) / 4096


def radians_to_raw_encoder(mujoco_rad: float) -> float:
    """Convert MuJoCo radians to raw encoder value."""
    return (mujoco_rad * 4096 / (2 * np.pi)) + 2048


# PI05 conversions
def normalized_degrees_to_raw(normalized_degrees: float, range_min: int, range_max: int) -> float:
    """Convert PI05 normalized degrees to raw encoder value."""
    mid = (range_min + range_max) / 2
    raw = (normalized_degrees * MAX_RES / 360) + mid
    return raw


def raw_to_normalized_degrees(raw: float, range_min: int, range_max: int) -> float:
    """Convert raw encoder value to PI05 normalized degrees."""
    mid = (range_min + range_max) / 2
    normalized_degrees = (raw - mid) * 360 / MAX_RES
    return normalized_degrees


# Absolute conversions
def calibrated_degrees_to_raw(degrees: float, homing_offset: int, drive_mode: int, resolution: int = 4096) -> float:
    """Convert lerobot calibrated degrees to raw encoder value (absolute method)."""
    value = degrees / 180 * (resolution // 2)
    value -= homing_offset
    if drive_mode:
        value *= -1
    return value


def raw_to_calibrated_degrees(raw: float, homing_offset: int, drive_mode: int, resolution: int = 4096) -> float:
    """Convert raw encoder value to lerobot calibrated degrees (absolute method)."""
    if drive_mode:
        raw *= -1
    raw += homing_offset
    degrees = raw * 180 / (resolution // 2)
    return degrees


# ============================================================================
# Gripper conversion helper class
# ============================================================================

class GripperConverter:
    """
    Handles gripper conversion between lerobot percentage and MuJoCo radians.
    
    Uses linear mapping: y = slope * x + intercept
    - Forward: lerobot_pct → mujoco_rad
    - Inverse: mujoco_rad → lerobot_pct
    """
    
    def __init__(self, gripper_ctrl_range: Tuple[float, float]):
        """
        Initialize gripper converter.
        
        Args:
            gripper_ctrl_range: (min, max) control range from MuJoCo model (e.g., (0.002, 0.041))
        """
        gripper_min, gripper_max = gripper_ctrl_range
        gripper_range = gripper_max - gripper_min
        
        # Calculate slope and intercept for linear mapping
        # Mapping: lerobot [0, 140] → mujoco [gripper_min, gripper_max]
        self.slope = (LEROBOT_CLOSED_PCT - LEROBOT_OPEN_PCT) / gripper_range
        self.intercept = LEROBOT_OPEN_PCT - self.slope * gripper_min
        
        # Left and right use same calibration for now
        self.right_slope = self.slope
        self.right_intercept = self.intercept
        self.left_slope = self.slope
        self.left_intercept = self.intercept
    
    def lerobot_to_mujoco(self, lerobot_percent: float, arm_side: str = "right") -> float:
        """
        Convert gripper from lerobot percentage to MuJoCo radians.
        
        Args:
            lerobot_percent: Gripper value in lerobot format (0-140%)
            arm_side: "right" or "left"
        
        Returns:
            Gripper position in MuJoCo radians
        """
        if arm_side == "right":
            return (lerobot_percent - self.right_intercept) / self.right_slope
        else:
            return (lerobot_percent - self.left_intercept) / self.left_slope
    
    def mujoco_to_lerobot(self, mujoco_rad: float, arm_side: str = "right") -> float:
        """
        Convert gripper from MuJoCo radians to lerobot percentage.
        
        Args:
            mujoco_rad: Gripper position in MuJoCo radians
            arm_side: "right" or "left"
        
        Returns:
            Gripper value in lerobot format (0-140%)
        """
        if arm_side == "right":
            return self.right_slope * mujoco_rad + self.right_intercept
        else:
            return self.left_slope * mujoco_rad + self.left_intercept


# ============================================================================
# Main converter class
# ============================================================================

class MuJoCoLeRobotConverter:
    """
    Main converter class for MuJoCo ↔ LeRobot transformations.
    
    Supports:
    - Action conversion: lerobot (18-dim degrees) → mujoco (14-dim radians)
    - State conversion: mujoco qpos (radians) → lerobot state (18-dim degrees)
    """
    
    # Joint mappings
    # Recorded format: [left_arm(9), right_arm(9)] = 18 total
    #   Left:  [0]=waist, [1]=shoulder, [2]=shoulder_shadow, [3]=elbow, [4]=elbow_shadow,
    #          [5]=forearm_roll, [6]=wrist_angle, [7]=wrist_rotate, [8]=gripper
    #   Right: [9]=waist, [10]=shoulder, [11]=shoulder_shadow, [12]=elbow, [13]=elbow_shadow,
    #          [14]=forearm_roll, [15]=wrist_angle, [16]=wrist_rotate, [17]=gripper
    #
    # MuJoCo ctrl format: [right_arm(7), left_arm(7)] = 14 total
    #   [0-6]: right waist, shoulder, elbow, forearm_roll, wrist_angle, wrist_rotate, gripper
    #   [7-13]: left waist, shoulder, elbow, forearm_roll, wrist_angle, wrist_rotate, gripper
    
    # PI05 joint mapping: (recorded_idx, ctrl_idx, joint_name, arm_side)
    PI05_JOINT_MAPPING = [
        # Right arm: recorded[9-17] → ctrl[0-6]
        (9,  0, "waist", "right"),
        (10, 1, "shoulder", "right"),
        # skip 11 (shoulder_shadow)
        (12, 2, "elbow", "right"),
        # skip 13 (elbow_shadow)
        (14, 3, "forearm_roll", "right"),
        (15, 4, "wrist_angle", "right"),
        (16, 5, "wrist_rotate", "right"),
        (17, 6, "gripper", "right"),
        # Left arm: recorded[0-8] → ctrl[7-13]
        (0,  7, "waist", "left"),
        (1,  8, "shoulder", "left"),
        # skip 2 (shoulder_shadow)
        (3,  9, "elbow", "left"),
        # skip 4 (elbow_shadow)
        (5, 10, "forearm_roll", "left"),
        (6, 11, "wrist_angle", "left"),
        (7, 12, "wrist_rotate", "left"),
        (8, 13, "gripper", "left"),
    ]
    
    # Absolute joint mapping: (recorded_idx, ctrl_idx, calib_joint_idx, arm_side)
    ABSOLUTE_JOINT_MAPPING = [
        # Right arm: recorded[9-17] → ctrl[0-6]
        (9,  0, 0, "right"),   # waist
        (10, 1, 1, "right"),   # shoulder
        # skip 11 (shoulder_shadow)
        (12, 2, 3, "right"),   # elbow
        # skip 13 (elbow_shadow)
        (14, 3, 5, "right"),   # forearm_roll
        (15, 4, 6, "right"),   # wrist_angle
        (16, 5, 7, "right"),   # wrist_rotate
        (17, 6, 8, "right"),   # gripper
        # Left arm: recorded[0-8] → ctrl[7-13]
        (0,  7, 0, "left"),    # waist
        (1,  8, 1, "left"),    # shoulder
        # skip 2 (shoulder_shadow)
        (3,  9, 3, "left"),    # elbow
        # skip 4 (elbow_shadow)
        (5, 10, 5, "left"),    # forearm_roll
        (6, 11, 6, "left"),    # wrist_angle
        (7, 12, 7, "left"),    # wrist_rotate
        (8, 13, 8, "left"),    # gripper
    ]
    
    # State mapping: (mujoco_qpos_idx, lerobot_state_idx, joint_name, arm_side)
    # MuJoCo qpos: [right_arm(8), left_arm(8), ...]
    # LeRobot state: [left_arm(9), right_arm(9)]
    STATE_MAPPING_LEFT = [
        (8, 0, "waist", "left"),
        (9, 1, "shoulder", "left"),
        (9, 2, "shoulder", "left"),   # shoulder_shadow (duplicate)
        (10, 3, "elbow", "left"),
        (10, 4, "elbow", "left"),     # elbow_shadow (duplicate)
        (11, 5, "forearm_roll", "left"),
        (12, 6, "wrist_angle", "left"),
        (13, 7, "wrist_rotate", "left"),
    ]
    
    STATE_MAPPING_RIGHT = [
        (0, 9, "waist", "right"),
        (1, 10, "shoulder", "right"),
        (1, 11, "shoulder", "right"),  # shoulder_shadow (duplicate)
        (2, 12, "elbow", "right"),
        (2, 13, "elbow", "right"),     # elbow_shadow (duplicate)
        (3, 14, "forearm_roll", "right"),
        (4, 15, "wrist_angle", "right"),
        (5, 16, "wrist_rotate", "right"),
    ]
    
    def __init__(self, gripper_ctrl_range: Tuple[float, float], use_new_normalization: bool = False):
        """
        Initialize converter.
        
        Args:
            gripper_ctrl_range: (min, max) control range from MuJoCo model (e.g., (0.002, 0.041))
            use_new_normalization: If True, use PI05 normalization, else use absolute
        """
        self.gripper_ctrl_range = gripper_ctrl_range
        self.use_new_normalization = use_new_normalization
        
        # Load calibration
        self.left_calib, self.right_calib = load_calibration(use_new_normalization)
        
        # Initialize gripper converter
        self.gripper = GripperConverter(gripper_ctrl_range)
    
    def get_calib(self, arm_side: str) -> dict:
        """Get calibration dict for arm side."""
        return self.left_calib if arm_side == "left" else self.right_calib
    
    # ========================================================================
    # Action conversion: lerobot → mujoco
    # ========================================================================
    
    def actions_to_mujoco(self, actions_raw: np.ndarray) -> np.ndarray:
        """
        Convert lerobot actions to MuJoCo control.
        
        Args:
            actions_raw: Array of shape (num_frames, 18) in lerobot format
        
        Returns:
            Array of shape (num_frames, 14) in MuJoCo control format
        """
        num_frames = len(actions_raw)
        ctrl_sequence = np.zeros((num_frames, 14))
        
        if self.use_new_normalization:
            ctrl_sequence = self._convert_actions_pi05(actions_raw, ctrl_sequence)
        else:
            ctrl_sequence = self._convert_actions_absolute(actions_raw, ctrl_sequence)
        
        return ctrl_sequence
    
    def _convert_actions_pi05(self, actions_raw: np.ndarray, ctrl_sequence: np.ndarray) -> np.ndarray:
        """Convert using PI05 normalization."""
        num_frames = len(actions_raw)
        
        for frame_idx in range(num_frames):
            for rec_idx, ctrl_idx, joint_name, arm_side in self.PI05_JOINT_MAPPING:
                lerobot_val = actions_raw[frame_idx, rec_idx]
                calib = self.get_calib(arm_side)
                
                if joint_name == "gripper":
                    mujoco_rad = self.gripper.lerobot_to_mujoco(lerobot_val, arm_side)
                else:
                    range_min = calib[joint_name]["range_min"]
                    range_max = calib[joint_name]["range_max"]
                    raw = normalized_degrees_to_raw(lerobot_val, range_min, range_max)
                    mujoco_rad = raw_encoder_to_radians(raw)
                
                ctrl_sequence[frame_idx, ctrl_idx] = mujoco_rad
        
        return ctrl_sequence
    
    def _convert_actions_absolute(self, actions_raw: np.ndarray, ctrl_sequence: np.ndarray) -> np.ndarray:
        """Convert using absolute (legacy) method."""
        num_frames = len(actions_raw)
        
        print("[INFO] Converting using calibration-based absolute conversion:")
        
        for frame_idx in range(num_frames):
            for rec_idx, ctrl_idx, calib_idx, arm_side in self.ABSOLUTE_JOINT_MAPPING:
                calib = self.get_calib(arm_side)
                joint_name = calib["motor_names"][calib_idx]
                lerobot_val = actions_raw[frame_idx, rec_idx]
                
                if joint_name == "gripper":
                    mujoco_rad = self.gripper.lerobot_to_mujoco(lerobot_val, arm_side)
                else:
                    homing_offset = calib["homing_offset"][calib_idx]
                    drive_mode = calib["drive_mode"][calib_idx]
                    raw = calibrated_degrees_to_raw(lerobot_val, homing_offset, drive_mode)
                    mujoco_rad = raw_encoder_to_radians(raw)
                
                ctrl_sequence[frame_idx, ctrl_idx] = mujoco_rad
        
        return ctrl_sequence
    
    # ========================================================================
    # State conversion: mujoco → lerobot
    # ========================================================================
    
    def state_to_lerobot(self, qpos: np.ndarray) -> np.ndarray:
        """
        Convert MuJoCo qpos to lerobot state format.
        
        Args:
            qpos: MuJoCo qpos array (at least 16 elements for dual arm)
        
        Returns:
            Array of shape (18,) in lerobot state format [left_arm(9), right_arm(9)]
        """
        state = np.zeros((18,), dtype=np.float32)
        
        # Convert joints
        for mujoco_idx, lerobot_idx, joint_name, arm_side in self.STATE_MAPPING_LEFT + self.STATE_MAPPING_RIGHT:
            if len(qpos) > mujoco_idx:
                mujoco_rad = qpos[mujoco_idx]
                raw = radians_to_raw_encoder(mujoco_rad)
                calib = self.get_calib(arm_side)
                
                if self.use_new_normalization:
                    range_min = calib[joint_name]["range_min"]
                    range_max = calib[joint_name]["range_max"]
                    degrees = raw_to_normalized_degrees(raw, range_min, range_max)
                else:
                    # For absolute, need to handle different calibration format
                    if "homing_offset" in calib and isinstance(calib["homing_offset"], dict):
                        homing_offset = calib["homing_offset"][joint_name]
                        drive_mode = calib["drive_mode"][joint_name]
                    else:
                        # List-based calibration - find index
                        joint_idx = calib["motor_names"].index(joint_name) if "motor_names" in calib else 0
                        homing_offset = calib["homing_offset"][joint_idx]
                        drive_mode = calib["drive_mode"][joint_idx]
                    degrees = raw_to_calibrated_degrees(raw, homing_offset, drive_mode)
                
                state[lerobot_idx] = degrees
        
        # Convert grippers
        # Left gripper: qpos[14] → state[8]
        left_gripper_rad = qpos[14]
        state[8] = self.gripper.mujoco_to_lerobot(left_gripper_rad, "left")
        
        # Right gripper: qpos[6] → state[17]
        right_gripper_rad = qpos[6]
        state[17] = self.gripper.mujoco_to_lerobot(right_gripper_rad, "right")
        
        return state


# ============================================================================
# Convenience functions (for backward compatibility)
# ============================================================================

def convert_actions_to_mujoco_pi05(actions_raw: np.ndarray, mujoco_keyframe_ctrl: np.ndarray,
                                    gripper_ctrl_range: Tuple[float, float] = (0.002, 0.041)) -> np.ndarray:
    """
    PI05 NORMALIZATION: Convert lerobot normalized degrees to MuJoCo radians.
    
    Args:
        actions_raw: Array of shape (num_frames, 18) in lerobot format
        mujoco_keyframe_ctrl: Keyframe control (unused, kept for API compatibility)
        gripper_ctrl_range: (min, max) control range for gripper
    
    Returns:
        Array of shape (num_frames, 14) in MuJoCo control format
    """
    converter = MuJoCoLeRobotConverter(gripper_ctrl_range, use_new_normalization=True)
    return converter.actions_to_mujoco(actions_raw)


def convert_actions_to_mujoco_absolute(actions_raw: np.ndarray, mujoco_keyframe_ctrl: np.ndarray,
                                        gripper_ctrl_range: Tuple[float, float] = (0.002, 0.041)) -> np.ndarray:
    """
    ABSOLUTE: Convert lerobot calibrated degrees to MuJoCo radians.
    
    Args:
        actions_raw: Array of shape (num_frames, 18) in lerobot format
        mujoco_keyframe_ctrl: Keyframe control (unused, kept for API compatibility)
        gripper_ctrl_range: (min, max) control range for gripper
    
    Returns:
        Array of shape (num_frames, 14) in MuJoCo control format
    """
    converter = MuJoCoLeRobotConverter(gripper_ctrl_range, use_new_normalization=False)
    return converter.actions_to_mujoco(actions_raw)


def convert_mujoco_state_to_lerobot(qpos: np.ndarray, gripper_ctrl_range: Tuple[float, float],
                                     use_new_normalization: bool = False) -> np.ndarray:
    """
    Convert MuJoCo qpos to lerobot state format.
    
    Args:
        qpos: MuJoCo qpos array
        gripper_ctrl_range: (min, max) control range for gripper
        use_new_normalization: If True, use PI05 normalization
    
    Returns:
        Array of shape (18,) in lerobot state format
    """
    converter = MuJoCoLeRobotConverter(gripper_ctrl_range, use_new_normalization)
    return converter.state_to_lerobot(qpos)


def convert_actions_to_mujoco_delta(actions_raw: np.ndarray, mujoco_keyframe_ctrl: np.ndarray) -> np.ndarray:
    """
    DELTA-BASED: Convert recorded actions using deltas from first frame.
    Problem: Assumes MuJoCo keyframe matches real-world frame 0 pose.
    
    Args:
        actions_raw: Array of shape (num_frames, 18) in lerobot format
        mujoco_keyframe_ctrl: Keyframe control array (14,)
    
    Returns:
        Array of shape (num_frames, 14) in MuJoCo control format
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


def convert_actions_to_mujoco(actions_raw: np.ndarray, mujoco_keyframe_ctrl: np.ndarray, 
                              use_absolute: bool = True, use_new_normalization: bool = False,
                              gripper_ctrl_range: Tuple[float, float] = (0.002, 0.041)) -> np.ndarray:
    """
    Main conversion function. 
    
    Args:
        actions_raw: Array of shape (num_frames, 18) in lerobot format
        mujoco_keyframe_ctrl: Keyframe control array (14,)
        use_absolute: If True, use absolute motor positions with calibration offsets.
                      If False, use delta-based replay from keyframe.
        use_new_normalization: If True, use PI05 normalization method.
                  Only applies when use_absolute=True.
        gripper_ctrl_range: (min, max) control range for gripper from MuJoCo model.
    
    Returns:
        Array of shape (num_frames, 14) in MuJoCo control format
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
