#!/usr/bin/env python3
"""
Sticker Alpha Calibration Tool

Loads scene.xml and replays the full dataset episode with alpha blending of real-world
and MuJoCo rendered images from both wrist cam and stationary cam.

During playback: SPACE pauses at the current frame for alignment.
When paused: Arrow keys move object 5mm in x/y (--sticker or --cube), +/- adjust alpha. SPACE resumes.

Uses ctrl + mj_step (same as run_prerecorded_traj_mujoco) so gripper actuator drives
both fingers correctly. Camera poses match between dataset and sim.

Usage:
    python visual_match/sticker_alpha_calibration.py --dataset-path data --sticker
    python visual_match/sticker_alpha_calibration.py --dataset-path data --cube
"""

import sys
import os
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from camera_config import load_camera_config, set_mujoco_camera_from_config
from compare_recorded_vs_mujoco import load_episode

import numpy as np
import cv2
import mujoco
from mujoco import MjModel, MjData

from lerobot.datasets.video_utils import decode_video_frames

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
from lerobot_mujoco_utils import GRIPPER_OPEN_MM, lerobot_state_to_mujoco_ctrl

RENDER_W, RENDER_H = 640, 480
STICKER_STEP_M = 0.005  # 5 mm per arrow key press

# Arrow key codes (platform-dependent; use waitKeyEx for full codes)
KEY_LEFT = (65361, 81, 2)
KEY_UP = (65362, 82, 0)
KEY_RIGHT = (65363, 83, 3)
KEY_DOWN = (65364, 84, 1)

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

_PROJECT_ROOT = Path(__file__).parent.parent
XML_PATH = _PROJECT_ROOT / "xarm7" / "scene.xml"


def load_episode_frames(dataset_path: str, episode_idx: int, dataset_root: str | None = None):
    """Load all video frames for both cameras from the dataset episode."""
    episode_data = load_episode(dataset_path, episode_idx, dataset_root=dataset_root)
    dataset = episode_data["dataset"]
    ep_meta = dataset.meta.episodes[episode_idx]
    start_idx = ep_meta["dataset_from_index"]
    end_idx = ep_meta["dataset_to_index"]
    dataset_size = len(dataset)
    end_idx = min(end_idx, dataset_size - 1)
    num_frames = end_idx - start_idx + 1
    video_fps = dataset.fps

    relative_timestamps = [i / video_fps for i in range(num_frames)]
    cam_frames = {}
    for cam_key, cam_cfg in CAMERA_CONFIG.items():
        dataset_cam = cam_cfg["dataset_cam"]
        camera_key = f"observation.images.{dataset_cam}"
        try:
            video_path_rel = dataset.meta.get_video_file_path(episode_idx, camera_key)
            video_path = dataset.root / video_path_rel
            if not video_path.exists():
                print(f"[WARN] Video not found for {cam_key}: {video_path}")
                cam_frames[cam_key] = []
                continue
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

    return cam_frames, episode_data, num_frames


def alpha_blend(real_img: np.ndarray, mj_img: np.ndarray, alpha: float) -> np.ndarray:
    """Blend real and MuJoCo images: alpha * mj + (1 - alpha) * real."""
    if real_img is None or real_img.size == 0:
        return mj_img.copy()
    # Resize real to match mj if needed
    if real_img.shape[:2] != mj_img.shape[:2]:
        real_img = cv2.resize(real_img, (mj_img.shape[1], mj_img.shape[0]))
    blended = (alpha * mj_img.astype(np.float32) + (1.0 - alpha) * real_img.astype(np.float32))
    return np.clip(blended, 0, 255).astype(np.uint8)


def main():
    p = argparse.ArgumentParser(
        description="Replay episode with alpha blend. SPACE pauses for sticker alignment."
    )
    p.add_argument("--dataset-path", type=str, default="data",
                  help="Path to dataset directory")
    p.add_argument("--dataset-root", type=str, default=None)
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--alpha", type=float, default=0.5,
                  help="Alpha for blending (0=fully real, 1=fully MuJoCo)")
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--sticker", action="store_true",
                    help="Adjust sticker position with arrow keys when paused")
    grp.add_argument("--cube", action="store_true",
                    help="Adjust cube position with arrow keys when paused")
    args = p.parse_args()

    dataset_path = args.dataset_path
    if not Path(dataset_path).is_absolute():
        dataset_path = str(_PROJECT_ROOT / dataset_path)

    if not os.path.isdir(dataset_path):
        print(f"[ERROR] Dataset path not found: {dataset_path}")
        sys.exit(1)

    # Load full episode
    print(f"[INFO] Loading episode {args.episode}...")
    cam_frames, episode_data, num_frames = load_episode_frames(
        dataset_path, args.episode, dataset_root=args.dataset_root
    )
    observations = episode_data["observation.state"].numpy()

    # Load MuJoCo model
    xarm_dir = _PROJECT_ROOT / "xarm7"
    original_cwd = os.getcwd()
    try:
        os.chdir(str(xarm_dir))
        model = MjModel.from_xml_path("scene.xml")
    finally:
        os.chdir(original_cwd)

    data = MjData(model)
    gripper_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper")
    gripper_mj_range = (
        model.actuator_ctrlrange[gripper_act_id, 0],
        model.actuator_ctrlrange[gripper_act_id, 1],
    )
    ctrl_seq = np.array(
        [lerobot_state_to_mujoco_ctrl(observations[i], gripper_mj_range) for i in range(num_frames)]
    )
    try:
        home_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
        mujoco.mj_resetDataKeyframe(model, data, home_id)
    except Exception:
        mujoco.mj_resetData(model, data)

    # Apply camera calibration (same as compare_recorded_vs_mujoco).
    # Wrist cam: patches model (local pose) — only needs to be done once.
    # Stationary cam: patches data (world pose) — re-applied after every mj_forward.
    mujoco.mj_forward(model, data)
    for cam_key, cam_cfg in CAMERA_CONFIG.items():
        set_mujoco_camera_from_config(data, model, cam_cfg["mujoco_cam"], cam_cfg["config"])

    # Object for position adjustment (sticker or cube)
    adjust_sticker = args.sticker
    adjust_cube = args.cube
    sticker_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "sticker")
    sticker_pos = model.body_pos[sticker_body_id].copy() if sticker_body_id >= 0 else None
    cube_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint")
    cube_qpos_addr = model.jnt_qposadr[cube_joint_id] if cube_joint_id >= 0 else -1
    cube_pos = data.qpos[cube_qpos_addr : cube_qpos_addr + 3].copy() if cube_qpos_addr >= 0 else None

    if adjust_sticker and sticker_body_id < 0:
        print("[ERROR] 'sticker' body not found in scene.xml")
        sys.exit(1)
    if adjust_cube and cube_joint_id < 0:
        print("[ERROR] 'cube_joint' not found in scene.xml")
        sys.exit(1)

    renderer = mujoco.Renderer(model, height=RENDER_H, width=RENDER_W)

    # Create windows: Alpha Blend | MuJoCo (side by side per camera)
    win_stat_alpha = "Stationary - Alpha Blend"
    win_stat_mj = "Stationary - MuJoCo"
    win_wrist_alpha = "Wrist - Alpha Blend"
    win_wrist_mj = "Wrist - MuJoCo"
    W, H = 400, 300
    X_START, Y_START = 50, 30
    X_STEP = 420
    for win in [win_stat_alpha, win_stat_mj, win_wrist_alpha, win_wrist_mj]:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, W, H)
    cv2.moveWindow(win_stat_alpha, X_START, Y_START)
    cv2.moveWindow(win_stat_mj, X_START + X_STEP, Y_START)
    cv2.moveWindow(win_wrist_alpha, X_START, Y_START + H + 20)
    cv2.moveWindow(win_wrist_mj, X_START + X_STEP, Y_START + H + 20)

    alpha = args.alpha
    obj_name = "sticker" if adjust_sticker else "cube"
    print(f"[INFO] Replay: SPACE=pause for {obj_name} alignment. Arrow keys: move 5mm. +/-: alpha. q: quit.")

    dt = 1.0 / args.fps

    def set_frame_pose(frame_idx: int):
        """Set ctrl and step simulation (same as run_prerecorded_traj_mujoco) so gripper works."""
        data.ctrl[:] = ctrl_seq[frame_idx]
        sim_target = data.time + dt
        while data.time < sim_target:
            mujoco.mj_step(model, data)

    def render_and_blend(frame_idx: int):
        mujoco.mj_forward(model, data)
        for cam_key, cam_cfg in CAMERA_CONFIG.items():
            if cam_cfg["config"].get("type", "stationary") == "stationary":
                set_mujoco_camera_from_config(data, model, cam_cfg["mujoco_cam"], cam_cfg["config"])

        blends = {}
        mujoco_imgs = {}
        for cam_key, cam_cfg in CAMERA_CONFIG.items():
            mujoco_cam = cam_cfg["mujoco_cam"]
            renderer.update_scene(data, camera=mujoco_cam)
            mj_rgb = renderer.render()
            mj_bgr = cv2.cvtColor(mj_rgb, cv2.COLOR_RGB2BGR)
            mj_labeled = mj_bgr.copy()
            cv2.putText(mj_labeled, f"Frame: {frame_idx}/{num_frames}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            mujoco_imgs[cam_key] = mj_labeled

            frames_list = cam_frames.get(cam_key, [])
            real = frames_list[frame_idx] if frame_idx < len(frames_list) else None
            blended = alpha_blend(real, mj_bgr, alpha)
            cv2.putText(blended, f"Frame: {frame_idx}/{num_frames}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(blended, f"Alpha: {alpha:.2f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            pos = sticker_pos[:2] if adjust_sticker else data.qpos[cube_qpos_addr : cube_qpos_addr + 2]
            cv2.putText(blended, f"{obj_name.capitalize()}: [{pos[0]:.4f}, {pos[1]:.4f}]", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            blends[cam_key] = blended
        return blends, mujoco_imgs

    if num_frames == 0:
        print("[ERROR] No frames in episode")
        sys.exit(1)

    frame_delay = max(1, int(1000 / args.fps))
    paused = False
    frame_idx = 0

    # Initial pose and display
    set_frame_pose(frame_idx)
    blends, mujoco_imgs = render_and_blend(frame_idx)
    cv2.imshow(win_stat_alpha, blends["stationary"])
    cv2.imshow(win_stat_mj, mujoco_imgs["stationary"])
    cv2.imshow(win_wrist_alpha, blends["wrist"])
    cv2.imshow(win_wrist_mj, mujoco_imgs["wrist"])

    while True:
        if not paused:
            # Advance to next frame
            frame_idx = min(frame_idx + 1, num_frames - 1)
            set_frame_pose(frame_idx)
            blends, mujoco_imgs = render_and_blend(frame_idx)
            cv2.imshow(win_stat_alpha, blends["stationary"])
            cv2.imshow(win_stat_mj, mujoco_imgs["stationary"])
            cv2.imshow(win_wrist_alpha, blends["wrist"])
            cv2.imshow(win_wrist_mj, mujoco_imgs["wrist"])
            if frame_idx >= num_frames - 1:
                frame_idx = 0  # Loop back

        key = cv2.waitKeyEx(frame_delay if not paused else 50)
        if key < 0:
            continue

        if key == ord(' '):
            paused = not paused
            print(f"[INFO] {'Paused' if paused else 'Resumed'} at frame {frame_idx}")
            if paused:
                print(f"[INFO] Adjust {obj_name} with arrow keys. SPACE to resume.")
            continue

        if key == ord('q'):
            print("[INFO] Quit")
            break

        if paused:
            moved = False
            if key in KEY_LEFT:
                if adjust_sticker:
                    sticker_pos[0] -= STICKER_STEP_M
                else:
                    data.qpos[cube_qpos_addr] -= STICKER_STEP_M
                moved = True
            elif key in KEY_RIGHT:
                if adjust_sticker:
                    sticker_pos[0] += STICKER_STEP_M
                else:
                    data.qpos[cube_qpos_addr] += STICKER_STEP_M
                moved = True
            elif key in KEY_UP:
                if adjust_sticker:
                    sticker_pos[1] += STICKER_STEP_M
                else:
                    data.qpos[cube_qpos_addr + 1] += STICKER_STEP_M
                moved = True
            elif key in KEY_DOWN:
                if adjust_sticker:
                    sticker_pos[1] -= STICKER_STEP_M
                else:
                    data.qpos[cube_qpos_addr + 1] -= STICKER_STEP_M
                moved = True

            if moved:
                if adjust_sticker:
                    model.body_pos[sticker_body_id] = sticker_pos
                pos = sticker_pos[:2] if adjust_sticker else data.qpos[cube_qpos_addr : cube_qpos_addr + 2]
                blends, mujoco_imgs = render_and_blend(frame_idx)
                cv2.imshow(win_stat_alpha, blends["stationary"])
                cv2.imshow(win_stat_mj, mujoco_imgs["stationary"])
                cv2.imshow(win_wrist_alpha, blends["wrist"])
                cv2.imshow(win_wrist_mj, mujoco_imgs["wrist"])
                print(f"[INFO] {obj_name.capitalize()} pos: x={pos[0]:.4f} y={pos[1]:.4f}")

        if key in (ord('+'), ord('=')):
            alpha = min(1.0, alpha + 0.05)
            blends, mujoco_imgs = render_and_blend(frame_idx)
            cv2.imshow(win_stat_alpha, blends["stationary"])
            cv2.imshow(win_stat_mj, mujoco_imgs["stationary"])
            cv2.imshow(win_wrist_alpha, blends["wrist"])
            cv2.imshow(win_wrist_mj, mujoco_imgs["wrist"])
        elif key in (ord('-'), ord('_')):
            alpha = max(0.0, alpha - 0.05)
            blends, mujoco_imgs = render_and_blend(frame_idx)
            cv2.imshow(win_stat_alpha, blends["stationary"])
            cv2.imshow(win_stat_mj, mujoco_imgs["stationary"])
            cv2.imshow(win_wrist_alpha, blends["wrist"])
            cv2.imshow(win_wrist_mj, mujoco_imgs["wrist"])

    cv2.destroyAllWindows()
    final_pos = sticker_pos[:2] if adjust_sticker else data.qpos[cube_qpos_addr : cube_qpos_addr + 2]
    print(f"[INFO] Final {obj_name} position: x={final_pos[0]:.4f} y={final_pos[1]:.4f}")


if __name__ == "__main__":
    main()
