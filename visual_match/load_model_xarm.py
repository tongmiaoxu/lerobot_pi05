#!/usr/bin/env python3
"""
Load xArm7 MuJoCo model with stationary and wrist cameras calibrated.

Returns model, data, and camera configs ready for rendering or simulation.
Cameras are loaded from visual_match/configs/ (stationary_cam.json, wrist_cam.json).

Usage as module:
    from load_model import load_model
    model, data, camera_configs = load_model()

Usage as script:
    python visual_match/load_model.py
"""

import os
from pathlib import Path

import mujoco
from mujoco import MjModel, MjData

# Ensure visual_match is on path for camera_config import
_SCRIPT_DIR = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(_SCRIPT_DIR))

from camera_config import load_camera_config, set_mujoco_camera_from_config


def load_model(
    model_path: str | Path | None = None,
    cma_params_path: str | Path | None = None,
) -> tuple[MjModel, MjData, dict]:
    """
    Load xArm7 MuJoCo model with calibrated stationary and wrist cameras.

    Args:
        model_path: Path to scene.xml (default: xarm7/scene.xml relative to project root)
        cma_params_path: Optional path to cma_result.pkl for optimised stiffness/damping

    Returns:
        (model, data, camera_configs)
        camera_configs: dict with keys "stationary", "wrist", each containing
                       {"mujoco_cam": str, "config": dict}
    """
    project_root = _SCRIPT_DIR.parent
    if model_path is None:
        model_path = project_root / "xarm7" / "scene.xml"
    model_path = Path(model_path)

    xarm_dir = model_path.parent
    original_cwd = os.getcwd()
    try:
        os.chdir(str(xarm_dir))
        model = MjModel.from_xml_path(model_path.name)
    finally:
        os.chdir(original_cwd)

    data = MjData(model)

    # Reset to home keyframe
    try:
        home_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
        mujoco.mj_resetDataKeyframe(model, data, home_id)
    except Exception:
        mujoco.mj_resetData(model, data)

    # Apply CMA-ES parameters if provided
    if cma_params_path is not None:
        cma_path = Path(cma_params_path)
        if not cma_path.is_absolute():
            cma_path = project_root / cma_path
        if cma_path.exists():
            import pickle
            with open(cma_path, "rb") as f:
                cma_result = pickle.load(f)
            xbest = cma_result["xbest"]
            kp = xbest[:7]
            act_damp = xbest[7:14]
            jnt_damp = xbest[14:]
            model.actuator_gainprm[:7, 0] = kp
            model.actuator_biasprm[:7, 1] = -kp
            model.actuator_biasprm[:7, 2] = -act_damp
            model.dof_damping[:7] = jnt_damp

    # Apply high damping to mug freejoint to prevent drift during physics
    mug_jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "mug_joint")
    if mug_jnt_id >= 0:
        mug_dof_addr = model.jnt_dofadr[mug_jnt_id]
        model.dof_damping[mug_dof_addr:mug_dof_addr + 6] = 100

    # Load camera configs
    stationary_cfg = load_camera_config("stationary_cam")
    wrist_cfg = load_camera_config("wrist_cam")

    camera_configs = {
        "stationary": {"mujoco_cam": "stationary_cam", "config": stationary_cfg},
        "wrist": {"mujoco_cam": "wrist_cam", "config": wrist_cfg},
    }

    # Apply camera calibration
    mujoco.mj_forward(model, data)
    for cam_key, cam_cfg in camera_configs.items():
        mj_cam = cam_cfg["mujoco_cam"]
        cc = cam_cfg["config"]
        set_mujoco_camera_from_config(data, model, mj_cam, cc)

    return model, data, camera_configs


def settle_mug(model, data, max_time=3.0, vel_threshold=1e-4):
    """Step physics until the mug freejoint velocity is near zero."""
    import numpy as np

    mug_jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "mug_joint")
    if mug_jnt_id < 0:
        return
    mug_qpos_addr = model.jnt_qposadr[mug_jnt_id]
    mug_dof_addr = model.jnt_dofadr[mug_jnt_id]

    start_time = data.time
    while data.time - start_time < max_time:
        mujoco.mj_step(model, data)
        vel = np.linalg.norm(data.qvel[mug_dof_addr:mug_dof_addr + 6])
        if vel < vel_threshold:
            break

    pos = data.qpos[mug_qpos_addr:mug_qpos_addr + 3]
    quat = data.qpos[mug_qpos_addr + 3:mug_qpos_addr + 7]
    print(f"\n[MUG SETTLED]  time={data.time - start_time:.3f}s")
    print(f"  pos:  {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}")
    print(f"  quat: {quat[0]:.4f} {quat[1]:.4f} {quat[2]:.4f} {quat[3]:.4f}")
    print(f"\n  Paste into scene.xml <body name=\"mug\" ...>:")
    print(f"    pos=\"{pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}\"")
    print(f"\n  Paste into xarm7.xml <key name=\"home\" ...> (mug portion of qpos):")
    print(f"    {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f} {quat[0]:.4f} {quat[1]:.4f} {quat[2]:.4f} {quat[3]:.4f}")


def main():
    model, data, camera_configs = load_model()

    print("[INFO] Loaded xArm7 model")
    print(f"       nq={model.nq}, nv={model.nv}, nu={model.nu}")
    print(f"       Cameras: stationary_cam, wrist_cam")

    for cam_key, cam_cfg in camera_configs.items():
        cc = cam_cfg["config"]
        cam_type = cc.get("type", "stationary")
        print(f"       - {cam_key}: type={cam_type}")

    print("\n[INFO] Settling mug on table (running physics)...")
    settle_mug(model, data)

    # Optional: launch passive viewer
    try:
        import mujoco.viewer
        print("\n[INFO] Launching MuJoCo viewer (press Ctrl+C to exit)")
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                mujoco.mj_step(model, data)
                set_mujoco_camera_from_config(
                    data, model, "stationary_cam",
                    camera_configs["stationary"]["config"]
                )
                viewer.sync()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"[INFO] Viewer not available: {e}")


if __name__ == "__main__":
    main()
