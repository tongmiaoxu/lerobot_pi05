#!/usr/bin/env python3
"""
Replay xArm LeRobot v3.0 dataset in MuJoCo using recorded observation.state.

================================================================================
DATA FORMAT: xArm LeRobot v3.0 (.parquet)
================================================================================
observation.state (8-dim):
  [joint1.pos, joint2.pos, …, joint7.pos, gripper.pos]
  - Joint angles are in DEGREES (from xArm servo encoders)
  - Gripper is in mm: 800 = fully open, 0 = fully closed

MuJoCo xarm7 model (8 actuators):
  act1–act7 : joint position targets in RADIANS
  gripper   : tendon ctrl in [0, 255]  (0 = closed, 255 = open)

Conversion:
  mj_joint  = deg2rad(obs_joint)
  mj_grip   = (gripper_mm / GRIPPER_OPEN_MM) * GRIPPER_MJ_MAX
================================================================================
"""

import sys
import os
import argparse
import dataclasses
from pathlib import Path
from typing import Optional
import time

# Add src and visual_match to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

def _detect_display():
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
import cv2
import mujoco
from mujoco import MjModel, MjData

if _HAS_DISPLAY:
    import mujoco.viewer

# Imports for 4-window display (Real + Composite per camera)
from camera_config import load_camera_config, set_mujoco_camera_from_config
from composite_rendering import (
    get_mujoco_camera_pose,
    get_robot_geom_ids,
    load_scene_data,
    mj_pose_to_gaussian_w2c,
    render,
    T_splat2mj,
)
from compare_recorded_vs_mujoco import load_episode as load_episode_for_videos
from deploy_act_policy_mujoco import (
    load_dataset_frames,
    display_camera_images,
    build_observation_from_mujoco,
    load_color_mapping,
)

# Camera configuration (same as deploy_act_policy_mujoco)
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

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
from lerobot_mujoco_utils import GRIPPER_OPEN_MM, lerobot_state_to_mujoco_ctrl

GRIPPER_CLOSE_MM = 0.0

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
XML_PATH = str(_PROJECT_ROOT / "xarm7" / "scene.xml")
RENDER_W, RENDER_H = 640, 480


@dataclasses.dataclass
class Args:
    parquet: str = "data/data/chunk-000/file-000.parquet"
    episode: int = 0
    fps: float = 30.0
    use_actions: bool = False
    plot: bool = False
    cma: bool = False
    cma_params: str = "cma_result.pkl"
    dataset_path: Optional[str] = None
    scene_path: str = "pointclouds/xarm7.npz"
    color_calib_path: Optional[str] = None
    headless: bool = False


def parse_cli() -> Args:
    p = argparse.ArgumentParser(
        description="Replay xArm LeRobot dataset in MuJoCo."
    )
    p.add_argument(
        "--parquet", type=str,
        default="data/data/chunk-000/file-000.parquet",
        help="Path to .parquet data file",
    )
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument(
        "--use-actions", action="store_true",
        help="Replay using action columns instead of observation.state",
    )
    p.add_argument("--plot", action="store_true", help="Save joint plot HTML")
    p.add_argument(
        "--cma-params", type=str, default="cma_result.pkl",
        help="Path to cma_result.pkl from CMA-ES optimisation. "
             "Applies optimised stiffness/damping to the model.",
    )
    p.add_argument(
        "--cma", action="store_true",
        help="Apply CMA-ES optimised parameters to the model.",
    )
    p.add_argument(
        "--dataset-path", type=str, default=None,
        help="Path to LeRobot dataset directory for Real camera windows (optional)",
    )
    p.add_argument(
        "--scene-path", type=str, default="pointclouds/xarm7.npz",
        help="Path to Gaussian Splatting scene for composite rendering",
    )
    p.add_argument(
        "--color-calib-path", type=str, default=None,
        help="Path to color calibration YAML file (optional)",
    )
    p.add_argument(
        "--headless", action="store_true",
        help="Run without GUI (no 4-window display)",
    )
    return Args(**vars(p.parse_args()))


# ---------------------------------------------------------------------------
# Dataset loader — reads parquet directly (no video dependencies)
# ---------------------------------------------------------------------------
def load_episode_from_parquet(parquet_path: str, episode_idx: int):
    import pandas as pd

    df = pd.read_parquet(parquet_path)
    ep = df[df["episode_index"] == episode_idx].sort_values("frame_index")
    if len(ep) == 0:
        available = sorted(df["episode_index"].unique())
        raise ValueError(
            f"Episode {episode_idx} not found. Available: {available}"
        )

    obs = np.stack(ep["observation.state"].values).astype(np.float32)
    act = np.stack(ep["action"].values).astype(np.float32)
    timestamps = ep["timestamp"].values.astype(np.float32)

    print(f"[INFO] Episode {episode_idx}: {len(ep)} frames, "
          f"obs shape {obs.shape}, act shape {act.shape}")
    return obs, act, timestamps


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_cli()

    parquet_path = args.parquet
    if not Path(parquet_path).is_absolute():
        parquet_path = str(_PROJECT_ROOT / parquet_path)

    obs_all, act_all, timestamps = load_episode_from_parquet(parquet_path, args.episode)
    num_frames = len(obs_all)

    source = act_all if args.use_actions else obs_all
    source_label = "action" if args.use_actions else "observation.state"
    print(f"[INFO] Replaying {source_label} ({num_frames} frames @ {args.fps} Hz)")

    # Load MuJoCo model
    xarm_dir = _PROJECT_ROOT / "xarm7"
    original_cwd = os.getcwd()
    try:
        os.chdir(str(xarm_dir))
        model = MjModel.from_xml_path("scene.xml")
    finally:
        os.chdir(original_cwd)

    data = MjData(model)

    # Reset to home keyframe
    try:
        home_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
        mujoco.mj_resetDataKeyframe(model, data, home_id)
    except Exception:
        mujoco.mj_resetData(model, data)

    # Read gripper actuator ctrl range from XML
    gripper_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper")
    gripper_mj_range = (
        model.actuator_ctrlrange[gripper_act_id, 0],
        model.actuator_ctrlrange[gripper_act_id, 1],
    )
    print(f"[INFO] MuJoCo gripper ctrl range: {gripper_mj_range}")
    print(f"[INFO] MuJoCo model has {model.nu} actuators, {model.nq} qpos, {model.nv} qvel")

    # Apply CMA-ES optimised parameters if provided
    if args.cma:
        import pickle
        cma_path = args.cma_params
        if not Path(cma_path).is_absolute():
            cma_path = str(_PROJECT_ROOT / cma_path)
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
        print(f"[INFO] Applied CMA-ES params from {args.cma_params}")
        print(f"  kp (stiffness):       {kp.tolist()}")
        print(f"  act_damp (biasprm[2]): {act_damp.tolist()}")
        print(f"  jnt_damp (dof_damping): {jnt_damp.tolist()}")

    # Pre-compute ctrl sequence
    ctrl_seq = np.array(
        [lerobot_state_to_mujoco_ctrl(source[i], gripper_mj_range) for i in range(num_frames)]
    )

    # Initialize sim from dataset's first frame (instead of home keyframe) for aligned replay
    data.qpos[:7] = ctrl_seq[0, :7]
    data.qpos[7] = ctrl_seq[0, 7] / 255.0 * 0.85  # gripper ctrl -> qpos
    data.qvel[:8] = 0
    mujoco.mj_forward(model, data)
    print("[INFO] Initialized sim from dataset first frame (aligned with replay start)")

    print(f"[INFO] Ctrl range per actuator:")
    for i in range(8):
        print(f"  act[{i}]: [{ctrl_seq[:, i].min():.4f}, {ctrl_seq[:, i].max():.4f}]")

    # -----------------------------------------------------------------------
    # 4-window display setup (Real + Composite per camera, like deploy_act_policy_mujoco)
    # -----------------------------------------------------------------------
    renderer = mujoco.Renderer(model, height=RENDER_H, width=RENDER_W)
    seg_renderer = mujoco.Renderer(model, height=RENDER_H, width=RENDER_W)
    seg_renderer.enable_segmentation_rendering()
    robot_geom_ids = get_robot_geom_ids(model)

    mujoco.mj_forward(model, data)
    for cam_key, cam_cfg in CAMERA_CONFIG.items():
        mj_cam = cam_cfg["mujoco_cam"]
        cc = cam_cfg["config"]
        set_mujoco_camera_from_config(data, model, mj_cam, cc)

    camera_intrinsics = {
        cam_key: cam_cfg["config"]["intrinsics"]
        for cam_key, cam_cfg in CAMERA_CONFIG.items()
    }

    gaussian_data = None
    scene_path = args.scene_path if Path(args.scene_path).is_absolute() else str(_PROJECT_ROOT / args.scene_path)
    if os.path.exists(scene_path):
        try:
            init_pose = get_mujoco_camera_pose(model, data, "stationary_cam")
            w2c_init = mj_pose_to_gaussian_w2c(init_pose, T_splat2mj)
            scene_data, scene_depth_data, _ = load_scene_data(
                scene_path, w2c_init, camera_intrinsics["stationary"]
            )
            color_calib = None
            if args.color_calib_path:
                calib_path = args.color_calib_path if Path(args.color_calib_path).is_absolute() else str(_PROJECT_ROOT / args.color_calib_path)
                if os.path.exists(calib_path):
                    try:
                        color_calib = load_color_mapping(calib_path)
                    except Exception as e:
                        print(f"[WARN] Failed to load color calibration: {e}")
            viz_cfg = {"viz_w": RENDER_W, "viz_h": RENDER_H, "viz_near": 0.1, "viz_far": 10.0}
            gaussian_data = {
                "scene_data": scene_data,
                "scene_depth_data": scene_depth_data,
                "viz_cfg": viz_cfg,
                "color_calib": color_calib,
                "camera_intrinsics": camera_intrinsics,
            }
            print(f"[INFO] Loaded Gaussian Splatting scene from: {scene_path}")
        except Exception as e:
            print(f"[WARN] Failed to load Gaussian scene: {e}")

    obs_frames = None
    dataset_path = args.dataset_path
    if dataset_path is None and "data" in parquet_path:
        dataset_path = str(Path(parquet_path).parent.parent)
    if dataset_path and not Path(dataset_path).is_absolute():
        dataset_path = str(_PROJECT_ROOT / dataset_path)
    if dataset_path and os.path.isdir(dataset_path):
        try:
            episode_data = load_episode_for_videos(dataset_path, args.episode, dataset_root=None)
            obs_frames = load_dataset_frames(episode_data)
            print(f"[INFO] Loaded real dataset images for Real windows")
        except Exception as e:
            print(f"[WARN] Could not load dataset for Real windows: {e}")

    show_windows = not args.headless and _HAS_DISPLAY
    if show_windows:
        WINDOW_W, WINDOW_H = 400, 300
        X_START, Y_START = 50, 30
        X_STEP, Y_STEP = 410, 340
        cam_keys = list(CAMERA_CONFIG.keys())
        for i, cam_key in enumerate(cam_keys):
            obs_key = f"observation.images.{CAMERA_CONFIG[cam_key]['dataset_cam']}"
            cam_short = obs_key.split(".")[-1]
            win_real = f"Real: {cam_short}"
            win_comp = f"Composite: {cam_short}"
            cv2.namedWindow(win_real, cv2.WINDOW_NORMAL)
            cv2.namedWindow(win_comp, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(win_real, WINDOW_W, WINDOW_H)
            cv2.resizeWindow(win_comp, WINDOW_W, WINDOW_H)
            cv2.moveWindow(win_real, X_START + i * X_STEP, Y_START)
            cv2.moveWindow(win_comp, X_START + i * X_STEP, Y_START + Y_STEP)

    # Tracking arrays for optional plot
    joint_recorded = []
    joint_mujoco = []

    def run_simulation(viewer=None):
        dt = 1.0 / args.fps
        print(f"[INFO] TIMESTEP={dt:.4f}s (FPS={args.fps})")
        wall_start = time.perf_counter()

        for frame_idx in range(num_frames):
            data.ctrl[:] = ctrl_seq[frame_idx]

            sim_target = data.time + dt
            # print(f"sim_target at frame {frame_idx}: {sim_target}")
            while data.time < sim_target:
                mujoco.mj_step(model, data)
                # print(f"data.time: {data.time}")

            if viewer is not None:
                viewer.sync()

            # 4-window display: Real + Composite per camera
            if show_windows:
                for cam_cfg in CAMERA_CONFIG.values():
                    if cam_cfg["config"].get("type", "stationary") == "stationary":
                        set_mujoco_camera_from_config(data, model, cam_cfg["mujoco_cam"], cam_cfg["config"])
                real_obs = build_observation_from_mujoco(
                    model, data, renderer, gripper_mj_range,
                    seg_renderer=seg_renderer,
                    robot_geom_ids=robot_geom_ids,
                    gaussian_data=gaussian_data,
                    obs_frames=obs_frames,
                    frame_idx=frame_idx,
                )
                composite_obs = build_observation_from_mujoco(
                    model, data, renderer, gripper_mj_range,
                    seg_renderer=seg_renderer,
                    robot_geom_ids=robot_geom_ids,
                    gaussian_data=gaussian_data,
                    obs_frames=None,
                    frame_idx=frame_idx,
                )
                display_camera_images(real_obs, policy_config=None, window_name_prefix="Real")
                display_camera_images(composite_obs, policy_config=None, window_name_prefix="Composite")
                cv2.waitKey(1)

            # Real-time pacing: wait until wall clock catches up to sim time
            wall_elapsed = time.perf_counter() - wall_start
            sim_elapsed = (frame_idx + 1) * dt
            sleep_s = sim_elapsed - wall_elapsed
            if sleep_s > 0:
                time.sleep(sleep_s)

            if args.plot:
                joint_recorded.append(source[frame_idx].copy())
                mj_state = np.zeros(8)
                mj_state[:7] = np.rad2deg(data.qpos[:7])
                mj_state[7] = data.qpos[7]  # gripper qpos (driver joint)
                joint_mujoco.append(mj_state)

            if frame_idx % 100 == 0:
                print(f"  frame {frame_idx}/{num_frames}  "
                      f"sim_time={data.time:.2f}s  "
                      f"ctrl[0]={data.ctrl[0]:.4f}")

    if _HAS_DISPLAY and not args.headless:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            run_simulation(viewer)
    else:
        if args.headless:
            print("[INFO] Running in headless mode (no viewer, no 4-window display)")
        else:
            print("[INFO] No display detected, running without viewer")
        run_simulation(None)

    if show_windows:
        cv2.destroyAllWindows()
    print("[DONE] Playback finished.")

    # Optional: save joint comparison plot
    if args.plot and joint_recorded:
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots

            rec = np.array(joint_recorded)
            mj = np.array(joint_mujoco)
            names = [f"joint{i+1}" for i in range(7)] + ["gripper"]

            fig = make_subplots(rows=4, cols=2, subplot_titles=names)
            for i, name in enumerate(names):
                r, c = divmod(i, 2)
                fig.add_trace(
                    go.Scatter(y=rec[:, i], name=f"{name}_recorded", mode="lines"),
                    row=r + 1, col=c + 1,
                )
                fig.add_trace(
                    go.Scatter(y=mj[:, i], name=f"{name}_mujoco", mode="lines",
                               line=dict(dash="dash")),
                    row=r + 1, col=c + 1,
                )

            fig.update_layout(
                title=f"xArm Replay — Episode {args.episode} ({source_label})",
                height=900,
            )
            out_html = f"xarm_replay_ep{args.episode}.html"
            fig.write_html(out_html)
            print(f"[INFO] Saved joint plot → {out_html}")
        except ImportError:
            print("[WARN] plotly not installed, skipping plot")


if __name__ == "__main__":
    main()
