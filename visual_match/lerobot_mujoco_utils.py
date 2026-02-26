"""
Shared utilities for converting between xArm LeRobot state and MuJoCo control.

LeRobot state: [joint1..7 in degrees, gripper in mm]
  - Gripper: 0 = closed, 800 = open (GRIPPER_OPEN_MM)

MuJoCo ctrl: [act1..7 in radians, gripper in actuator range]
  - Gripper range from model.actuator_ctrlrange[gripper_act_id]
"""

import numpy as np

GRIPPER_OPEN_MM = 800.0


def lerobot_state_to_mujoco_ctrl(
    state: np.ndarray, gripper_mj_range: tuple[float, float]
) -> np.ndarray:
    """
    Convert xArm LeRobot state to MuJoCo ctrl values.

    state: (8,) or (N, 8) — [joint1..7 in degrees, gripper in mm]
    gripper_mj_range: (lo, hi) from model.actuator_ctrlrange[gripper_act_id]
    returns: same shape — [joints in radians, gripper in MuJoCo ctrl units]
    """
    scalar = state.ndim == 1
    if scalar:
        state = state[np.newaxis, :]
    ctrl = np.zeros_like(state, dtype=np.float64)
    ctrl[:, :7] = np.deg2rad(state[:, :7])
    grip_frac = np.clip(state[:, 7] / GRIPPER_OPEN_MM, 0.0, 1.0)
    mj_hi, mj_lo = gripper_mj_range
    ctrl[:, 7] = mj_lo + grip_frac*0.7 * (mj_hi - mj_lo)
    return ctrl[0] if scalar else ctrl


def mujoco_qpos_to_lerobot_state(
    qpos: np.ndarray, gripper_mj_range: tuple[float, float]
) -> np.ndarray:
    """
    Convert MuJoCo qpos (8-dim: 7 joints rad + gripper) to xArm LeRobot state.

    qpos: (8,) — [joint1..7 in radians, gripper in MuJoCo units]
    gripper_mj_range: (lo, hi) from model.actuator_ctrlrange[gripper_act_id]
    returns: (8,) — [joints in degrees, gripper in mm (0=closed, 800=open)]
    """
    state = np.zeros(8, dtype=np.float32)
    state[:7] = np.rad2deg(qpos[:7])
    # gripper_mj_range = (ctrlrange[0], ctrlrange[1]); qpos[7] convention may differ from ctrl
    mj_hi, mj_lo = gripper_mj_range
    grip_frac = np.clip((qpos[7] - mj_lo) / (mj_hi - mj_lo), 0.0, 1.0)
    state[7] = grip_frac * GRIPPER_OPEN_MM
    return state
