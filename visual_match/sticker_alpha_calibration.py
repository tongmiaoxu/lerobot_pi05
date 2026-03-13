#!/usr/bin/env python3
"""
Sticker Alpha Calibration Tool

Position calibration mode (--sticker and/or --mug):
  Two windows: stationary_cam and wrist_cam views with alpha blend of MuJoCo render + first frame.
  Arrows: move x,y (5mm). w/s: increase/decrease z. m/t/b: select mug/sticker/table. +/-: alpha. q: save & quit. Esc: discard & quit.
  Use --show-sources to add Real and MuJoCo source windows for comparison.

Replay mode (--cube only, or legacy):
  Replays episode with alpha blend. SPACE pauses for alignment.

Usage:
    python visual_match/sticker_alpha_calibration.py --dataset-path data --sticker --mug
    python visual_match/sticker_alpha_calibration.py --dataset-path data --sticker
    python visual_match/sticker_alpha_calibration.py --dataset-path data --cube
"""

import sys
import os
import argparse
import re
import signal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from camera_config import load_camera_config, set_mujoco_camera_from_config
from composite_rendering import get_robot_geom_ids, shift_for_principal_point
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
SCENE_XML_PATH = _PROJECT_ROOT / "xarm7" / "scene.xml"
XARM7_XML_PATH = _PROJECT_ROOT / "xarm7" / "xarm7.xml"

# Mug euler "0 0 1.5708" (90° around Z) as quat (w,x,y,z)
MUG_QUAT = "0.7071 0 0 0.7071"


def update_scene_xml_positions(mug_pos, sticker_pos, table_pos=None):
    """Update mug, sticker, and table positions in scene.xml and xarm7.xml (home keyframe)."""
    if mug_pos is not None:
        new_pos = f'{mug_pos[0]:.4f} {mug_pos[1]:.4f} {mug_pos[2]:.4f}'
        # scene.xml: body pos
        if SCENE_XML_PATH.exists():
            text = SCENE_XML_PATH.read_text()
            text = re.sub(
                r'(<body name="mug"[^>]*pos=")[^"]*(")',
                r'\g<1>' + new_pos + r'\2',
                text,
                count=1,
            )
            SCENE_XML_PATH.write_text(text)
            print(f"[INFO] Updated mug pos in {SCENE_XML_PATH}")
        # xarm7.xml: home keyframe qpos (last 7 values = mug freejoint: xyz + quat wxyz)
        # Match only the active key (has 0.7071 quat) to avoid modifying commented lines
        if XARM7_XML_PATH.exists():
            text = XARM7_XML_PATH.read_text()
            mug_qpos = f'{new_pos} {MUG_QUAT}'
            text = re.sub(
                r'(^    <key name="home" qpos=")([^"]*?)(\s+[\d.-]+\s+[\d.-]+\s+[\d.-]+\s+[\d.-]+\s+[\d.-]+\s+0\.7071)(")',
                r'\g<1>\g<2> ' + mug_qpos + r'\g<4>',
                text,
                count=1,
                flags=re.MULTILINE,
            )
            XARM7_XML_PATH.write_text(text)
            print(f"[INFO] Updated mug pos in home keyframe ({XARM7_XML_PATH})")

    if sticker_pos is not None:
        new_pos = f'{sticker_pos[0]:.4f} {sticker_pos[1]:.4f} {sticker_pos[2]:.4f}'
        if SCENE_XML_PATH.exists():
            text = SCENE_XML_PATH.read_text()
            text = re.sub(
                r'(<body name="sticker"[^>]*pos=")[^"]*(")',
                r'\g<1>' + new_pos + r'\2',
                text,
                count=1,
            )
            SCENE_XML_PATH.write_text(text)
            print(f"[INFO] Updated sticker pos in {SCENE_XML_PATH}")
    if table_pos is not None:
        new_pos = f'{table_pos[0]:.4f} {table_pos[1]:.4f} {table_pos[2]:.4f}'
        if SCENE_XML_PATH.exists():
            text = SCENE_XML_PATH.read_text()
            text = re.sub(
                r'(<geom name="table"[^>]*pos=")[^"]*(")' ,
                r'\g<1>' + new_pos + r'\2',
                text,
                count=1,
            )
            SCENE_XML_PATH.write_text(text)
            print(f"[INFO] Updated table pos in {SCENE_XML_PATH}")

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
    p.add_argument("--alpha", type=float, default=0.7,
                  help="Alpha for blending (0=fully real, 1=fully MuJoCo)")
    p.add_argument("--sticker", action="store_true",
                   help="Adjust sticker position with arrow keys")
    p.add_argument("--mug", action="store_true",
                   help="Adjust mug position with arrow keys")
    p.add_argument("--table", action="store_true",
                   help="Adjust table position with arrow keys")
    p.add_argument("--cube", action="store_true",
                   help="Adjust cube position (replay mode)")
    p.add_argument("--show-sources", action="store_true",
                   help="Show extra windows with pure Real and MuJoCo for comparison")
    p.add_argument("--calib-frame", type=int, default=0,
                   help="Which frame to use for position calibration (default: 9)")
    args = p.parse_args()

    # Position calibration: mug, sticker, and/or table. Replay mode: cube only.
    position_calibration = args.sticker or args.mug or args.table
    if not position_calibration and not args.cube:
        p.error("Use at least one of: --sticker, --mug, --table, --cube")

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

    # Initialize sim from dataset's first frame (instead of home keyframe) for aligned replay
    data.qpos[:7] = ctrl_seq[0, :7]
    data.qpos[7] = ctrl_seq[0, 7] / 255.0 * 0.85  # gripper ctrl -> qpos
    data.qvel[:8] = 0
    mujoco.mj_forward(model, data)
    print("[INFO] Initialized sim from dataset first frame (aligned with replay start)")

    # Fix fovy to match real camera intrinsics (same as compare_recorded_vs_mujoco).
    for cam_key, cam_cfg in CAMERA_CONFIG.items():
        K = cam_cfg["config"]["intrinsics"]
        fy = K[1, 1]
        correct_fovy = float(2.0 * np.degrees(np.arctan(RENDER_H / (2.0 * fy))))
        mj_cam = cam_cfg["mujoco_cam"]
        cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, mj_cam)
        if cid >= 0:
            old_fovy = model.cam_fovy[cid]
            model.cam_fovy[cid] = correct_fovy
            print(f"[INFO] Corrected fovy for '{mj_cam}': {old_fovy:.2f}° → {correct_fovy:.2f}°")

    # Apply camera calibration (same as compare_recorded_vs_mujoco).
    # Wrist cam: patches model (local pose) — only needs to be done once.
    # Stationary cam: patches data (world pose) — re-applied after every mj_forward.
    mujoco.mj_forward(model, data)
    for cam_key, cam_cfg in CAMERA_CONFIG.items():
        set_mujoco_camera_from_config(data, model, cam_cfg["mujoco_cam"], cam_cfg["config"])

    # Object for position adjustment (sticker, mug, or cube)
    adjust_sticker = args.sticker
    adjust_mug = args.mug
    adjust_cube = args.cube
    sticker_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "sticker")
    sticker_pos = model.body_pos[sticker_body_id].copy() if sticker_body_id >= 0 else None
    mug_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "mug")
    mug_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "mug_joint")
    mug_qpos_addr = model.jnt_qposadr[mug_joint_id] if mug_joint_id >= 0 else -1
    # Mug: use data.qpos if freejoint (position is in qpos), else model.body_pos
    if mug_body_id >= 0 and mug_qpos_addr >= 0:
        mug_pos = data.qpos[mug_qpos_addr : mug_qpos_addr + 3].copy()
    elif mug_body_id >= 0:
        mug_pos = model.body_pos[mug_body_id].copy()
    else:
        mug_pos = None
    table_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "table")
    table_pos = model.geom_pos[table_geom_id].copy() if table_geom_id >= 0 else None
    adjust_table = args.table
    cube_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint")
    cube_qpos_addr = model.jnt_qposadr[cube_joint_id] if cube_joint_id >= 0 else -1
    cube_pos = data.qpos[cube_qpos_addr : cube_qpos_addr + 3].copy() if cube_qpos_addr >= 0 else None

    if adjust_table and table_geom_id < 0:
        print("[ERROR] 'table' geom not found in scene.xml")
        sys.exit(1)
    if adjust_sticker and sticker_body_id < 0:
        print("[ERROR] 'sticker' body not found in scene.xml")
        sys.exit(1)
    if adjust_mug and mug_body_id < 0:
        print("[ERROR] 'mug' body not found in scene.xml")
        sys.exit(1)
    if adjust_cube and cube_joint_id < 0:
        print("[ERROR] 'cube_joint' not found in scene.xml")
        sys.exit(1)

    # Which object is selected for arrow-key adjustment (position calibration mode)
    current_obj = "mug" if adjust_mug else ("sticker" if adjust_sticker else ("table" if adjust_table else "cube"))

    renderer = mujoco.Renderer(model, height=RENDER_H, width=RENDER_W)
    seg_renderer = mujoco.Renderer(model, height=RENDER_H, width=RENDER_W)
    seg_renderer.enable_segmentation_rendering()
    robot_geom_ids = get_robot_geom_ids(model)

    alpha = args.alpha

    if position_calibration:
        if not cam_frames.get("stationary") or len(cam_frames["stationary"]) == 0:
            print("[WARN] No stationary cam frames; alpha blend will show MuJoCo only.")
        if not cam_frames.get("wrist") or len(cam_frames["wrist"]) == 0:
            print("[WARN] No wrist cam frames; wrist view will show MuJoCo only.")
        # Position calibration mode: stationary + wrist cam windows, first-frame alpha blend
        win_stat = f"Stationary - Alpha Blend (frame {args.calib_frame})"
        win_wrist = f"Wrist - Alpha Blend (frame {args.calib_frame})"
        cv2.namedWindow(win_stat, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win_stat, RENDER_W, RENDER_H)
        cv2.namedWindow(win_wrist, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win_wrist, RENDER_W, RENDER_H)
        cv2.moveWindow(win_stat, 50, 30)
        cv2.moveWindow(win_wrist, 50 + RENDER_W + 20, 30)
        if args.show_sources:
            win_real = "Stationary - Real (source)"
            win_mj = "Stationary - MuJoCo (source)"
            cv2.namedWindow(win_real, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(win_real, RENDER_W, RENDER_H)
            cv2.namedWindow(win_mj, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(win_mj, RENDER_W, RENDER_H)
            cv2.moveWindow(win_real, 50, 40 + RENDER_H)
            cv2.moveWindow(win_mj, 50 + RENDER_W + 20, 40 + RENDER_H)
        obj_list = []
        if adjust_mug:
            obj_list.append("mug")
        if adjust_sticker:
            obj_list.append("sticker")
        print(f"[INFO] Position calibration. Arrows: xy | w/s: z | m/t/b: mug/sticker/table | +/-: alpha | q: save & quit | Esc: discard")
    else:
        # Replay mode: four windows
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
        obj_name = "cube"
        print(f"[INFO] Replay: SPACE=pause for cube alignment. Arrow keys: move 5mm. +/-: alpha. q: save & quit. Esc: discard.")

    dt = 1.0 / args.fps

    def set_frame_pose(frame_idx: int):
        """Set ctrl and step simulation (same as run_prerecorded_traj_mujoco) so gripper works."""
        data.ctrl[:] = ctrl_seq[frame_idx]
        sim_target = data.time + dt
        while data.time < sim_target:
            mujoco.mj_step(model, data)

    def get_current_pos():
        """Get [x, y] or [x, y, z] of the currently selected object."""
        if current_obj == "sticker":
            return sticker_pos
        if current_obj == "mug":
            return mug_pos
        if current_obj == "table":
            return table_pos
        return data.qpos[cube_qpos_addr : cube_qpos_addr + 3]

    def get_pos_str():
        """Format position for display."""
        pos = get_current_pos()
        if len(pos) >= 3 and current_obj in ("sticker", "mug"):
            return f"[{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]"
        return f"[{pos[0]:.4f}, {pos[1]:.4f}]" if len(pos) >= 2 else str(pos)

    def render_and_blend(frame_idx: int, cams: list[str] | None = None, show_calib_help: bool = False):
        """Render and blend for given cameras using segmentation-masked alpha
        (same formula as compare_recorded_vs_mujoco)."""
        mujoco.mj_forward(model, data)
        for cam_key, cam_cfg in CAMERA_CONFIG.items():
            if cam_cfg["config"].get("type", "stationary") == "stationary":
                set_mujoco_camera_from_config(data, model, cam_cfg["mujoco_cam"], cam_cfg["config"])

        blends = {}
        mujoco_imgs = {}
        cam_list = cams if cams is not None else list(CAMERA_CONFIG.keys())
        for cam_key in cam_list:
            cam_cfg = CAMERA_CONFIG[cam_key]
            mujoco_cam = cam_cfg["mujoco_cam"]
            K_cam = cam_cfg["config"]["intrinsics"]

            # Render MuJoCo RGB
            renderer.update_scene(data, camera=mujoco_cam)
            mj_rgb = renderer.render()
            mj_bgr = cv2.cvtColor(mj_rgb, cv2.COLOR_RGB2BGR)

            # Render segmentation mask
            seg_renderer.update_scene(data, camera=mujoco_cam)
            seg_mask = seg_renderer.render()
            seg_labels = seg_mask[:, :, 0].astype(np.int32)
            seg_labels[seg_labels == -1] = 0

            # Shift both to compensate for off-center principal point
            mj_bgr = shift_for_principal_point(mj_bgr, K_cam)
            seg_labels = shift_for_principal_point(seg_labels, K_cam, seg=True)

            # Build foreground mask (robot + mug + sticker)
            mask = np.isin(seg_labels, list(robot_geom_ids))
            mask_uint8 = mask.astype(np.uint8) * 255

            mj_labeled = mj_bgr.copy()
            if not show_calib_help:
                cv2.putText(mj_labeled, f"Frame: {frame_idx}/{num_frames}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            mujoco_imgs[cam_key] = mj_labeled

            # Masked alpha blend (same as compare_recorded_vs_mujoco)
            frames_list = cam_frames.get(cam_key, [])
            real = frames_list[frame_idx] if frame_idx < len(frames_list) else None
            if real is None or real.size == 0:
                blended = mj_bgr.copy()
            else:
                if real.shape[:2] != mj_bgr.shape[:2]:
                    real = cv2.resize(real, (mj_bgr.shape[1], mj_bgr.shape[0]))
                alpha_mask = (mask_uint8 / 255.0).astype(np.float32)
                alpha_mask_3ch = np.stack([alpha_mask] * 3, axis=-1)
                fg = mj_bgr.astype(np.float32)
                bg = real.astype(np.float32)
                blended = (alpha * fg + (1 - alpha) * bg) * alpha_mask_3ch + \
                          bg * (1 - alpha_mask_3ch)
                blended = blended.astype(np.uint8)

            pos_str = get_pos_str()
            cv2.putText(blended, f"{current_obj.capitalize()}: {pos_str}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            if show_calib_help:
                cv2.putText(blended, "m/t/b: mug/sticker/table | Arrows: xy | w/s: z | +/-: alpha", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            else:
                cv2.putText(blended, f"Frame: {frame_idx}/{num_frames}", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            blends[cam_key] = blended
        return blends, mujoco_imgs

    if num_frames == 0:
        print("[ERROR] No frames in episode")
        sys.exit(1)

    # Save initial positions for reset
    mug_pos_init = mug_pos.copy() if mug_pos is not None else None
    sticker_pos_init = sticker_pos.copy() if sticker_pos is not None else None
    table_pos_init = table_pos.copy() if table_pos is not None else None
    cube_pos_init = cube_pos.copy() if cube_pos is not None else None

    def apply_pos_to_model():
        """Apply body/geom positions to model/data."""
        if sticker_body_id >= 0 and sticker_pos is not None:
            model.body_pos[sticker_body_id] = sticker_pos
        if mug_pos is not None:
            if mug_qpos_addr >= 0:
                # Mug has freejoint: position in data.qpos (preserve quat)
                data.qpos[mug_qpos_addr : mug_qpos_addr + 3] = mug_pos
            elif mug_body_id >= 0:
                model.body_pos[mug_body_id] = mug_pos
        if table_geom_id >= 0 and table_pos is not None:
            model.geom_pos[table_geom_id] = table_pos

    def do_move(dx: float, dy: float):
        """Move current object by dx, dy (meters)."""
        if current_obj == "sticker" and sticker_pos is not None:
            sticker_pos[0] += dx
            sticker_pos[1] += dy
        elif current_obj == "mug" and mug_pos is not None:
            mug_pos[0] += dx
            mug_pos[1] += dy
        elif current_obj == "table" and table_pos is not None:
            table_pos[0] += dx
            table_pos[1] += dy
        elif current_obj == "cube" and cube_qpos_addr >= 0:
            data.qpos[cube_qpos_addr] += dx
            data.qpos[cube_qpos_addr + 1] += dy

    def do_move_z(dz: float):
        """Move current object by dz in z (meters). Only for mug, sticker, table."""
        if current_obj == "sticker" and sticker_pos is not None:
            sticker_pos[2] += dz
        elif current_obj == "mug" and mug_pos is not None:
            mug_pos[2] += dz
        elif current_obj == "table" and table_pos is not None:
            table_pos[2] += dz

    frame_delay = max(1, int(1000 / args.fps))
    paused = False
    frame_idx = 0

    if position_calibration:
        # Position calibration: single frame, stationary + wrist cam, no playback
        cf = args.calib_frame
        if cf >= num_frames:
            print(f"[WARN] --calib-frame {cf} >= num_frames {num_frames}, clamping to {num_frames - 1}")
            cf = num_frames - 1
        # Step simulation to the calibration frame
        for _f in range(cf + 1):
            set_frame_pose(_f)
        apply_pos_to_model()
        blends, mujoco_imgs = render_and_blend(cf, cams=["stationary", "wrist"], show_calib_help=True)
        cv2.imshow(win_stat, blends["stationary"])
        cv2.imshow(win_wrist, blends["wrist"])
        if args.show_sources:
            real_stat = cam_frames.get("stationary", [])
            if cf < len(real_stat):
                cv2.imshow(win_real, real_stat[cf])
            cv2.imshow(win_mj, mujoco_imgs["stationary"])

        def on_sigint(*_):
            """On Ctrl+C: discard changes and exit."""
            cv2.destroyAllWindows()
            print("[INFO] Ctrl+C — changes discarded.")
            sys.exit(0)

        signal.signal(signal.SIGINT, on_sigint)

        while True:
            key = cv2.waitKeyEx(50)
            if key < 0:
                continue

            if key == 27:  # Esc
                print("[INFO] Esc — changes discarded.")
                cv2.destroyAllWindows()
                return

            if key == ord('r'):
                print("[INFO] Resetting all poses to original values.")
                mug_pos = mug_pos_init.copy() if mug_pos_init is not None else None
                sticker_pos = sticker_pos_init.copy() if sticker_pos_init is not None else None
                table_pos = table_pos_init.copy() if table_pos_init is not None else None
                cube_pos = cube_pos_init.copy() if cube_pos_init is not None else None
                apply_pos_to_model()
                blends, mujoco_imgs = render_and_blend(cf)
                cv2.imshow(win_stat_alpha, blends["stationary"])
                cv2.imshow(win_stat_mj, mujoco_imgs["stationary"])
                cv2.imshow(win_wrist_alpha, blends["wrist"])
                cv2.imshow(win_wrist_mj, mujoco_imgs["wrist"])
                continue

            if key == ord('q'):
                print("[INFO] Quit — saving positions.")
                break

            moved = False
            if key in KEY_LEFT:
                do_move(-STICKER_STEP_M, 0)
                moved = True
            elif key in KEY_RIGHT:
                do_move(STICKER_STEP_M, 0)
                moved = True
            elif key in KEY_UP:
                do_move(0, STICKER_STEP_M)
                moved = True
            elif key in KEY_DOWN:
                do_move(0, -STICKER_STEP_M)
                moved = True
            elif key == ord('w') and current_obj in ("mug", "sticker", "table"):
                do_move_z(STICKER_STEP_M)
                moved = True
            elif key == ord('s') and current_obj in ("mug", "sticker", "table"):
                do_move_z(-STICKER_STEP_M)
                moved = True

            if key == ord('m') and adjust_mug:
                current_obj = "mug"
                print("[INFO] Selected: mug")
            elif key == ord('t') and adjust_sticker:
                current_obj = "sticker"
                print("[INFO] Selected: sticker")
            elif key == ord('b') and adjust_table:
                current_obj = "table"
                print("[INFO] Selected: table")

            if moved:
                apply_pos_to_model()
                print(f"[INFO] {current_obj.capitalize()} pos: {get_pos_str()}")

            if key in (ord('+'), ord('=')):
                alpha = min(1.0, alpha + 0.05)
            elif key in (ord('-'), ord('_')):
                alpha = max(0.0, alpha - 0.05)

            if moved or key in (ord('+'), ord('='), ord('-'), ord('_'), ord('m'), ord('t')):
                apply_pos_to_model()
                blends, mujoco_imgs = render_and_blend(cf, cams=["stationary", "wrist"], show_calib_help=True)
                cv2.imshow(win_stat, blends["stationary"])
                cv2.imshow(win_wrist, blends["wrist"])
                if args.show_sources:
                    real_stat = cam_frames.get("stationary", [])
                    if cf < len(real_stat):
                        cv2.imshow(win_real, real_stat[cf])
                    cv2.imshow(win_mj, mujoco_imgs["stationary"])

        cv2.destroyAllWindows()
        if adjust_mug and mug_pos is not None:
            print(f"[INFO] Final mug position: x={mug_pos[0]:.4f} y={mug_pos[1]:.4f} z={mug_pos[2]:.4f}")
        if adjust_sticker and sticker_pos is not None:
            print(f"[INFO] Final sticker position: x={sticker_pos[0]:.4f} y={sticker_pos[1]:.4f} z={sticker_pos[2]:.4f}")
        if adjust_table and table_pos is not None:
            print(f"[INFO] Final table position: x={table_pos[0]:.4f} y={table_pos[1]:.4f} z={table_pos[2]:.4f}")
        if adjust_mug or adjust_sticker or adjust_table:
            update_scene_xml_positions(
                mug_pos if adjust_mug else None,
                sticker_pos if adjust_sticker else None,
                table_pos if adjust_table else None,
            )
    else:
        # Replay mode
        set_frame_pose(frame_idx)
        blends, mujoco_imgs = render_and_blend(frame_idx)
        cv2.imshow(win_stat_alpha, blends["stationary"])
        cv2.imshow(win_stat_mj, mujoco_imgs["stationary"])
        cv2.imshow(win_wrist_alpha, blends["wrist"])
        cv2.imshow(win_wrist_mj, mujoco_imgs["wrist"])

        while True:
            if not paused:
                frame_idx = min(frame_idx + 1, num_frames - 1)
                set_frame_pose(frame_idx)
                blends, mujoco_imgs = render_and_blend(frame_idx)
                cv2.imshow(win_stat_alpha, blends["stationary"])
                cv2.imshow(win_stat_mj, mujoco_imgs["stationary"])
                cv2.imshow(win_wrist_alpha, blends["wrist"])
                cv2.imshow(win_wrist_mj, mujoco_imgs["wrist"])
                if frame_idx >= num_frames - 1:
                    frame_idx = 0

            key = cv2.waitKeyEx(frame_delay if not paused else 50)
            if key < 0:
                continue

            if key == ord(' '):
                paused = not paused
                print(f"[INFO] {'Paused' if paused else 'Resumed'} at frame {frame_idx}")
                continue

            if key == 27:  # Esc
                print("[INFO] Esc — changes discarded.")
                cv2.destroyAllWindows()
                return

            if key == ord('q'):
                print("[INFO] Quit")
                break

            if paused:
                moved = False
                if key in KEY_LEFT:
                    do_move(-STICKER_STEP_M, 0)
                    moved = True
                elif key in KEY_RIGHT:
                    do_move(STICKER_STEP_M, 0)
                    moved = True
                elif key in KEY_UP:
                    do_move(0, STICKER_STEP_M)
                    moved = True
                elif key in KEY_DOWN:
                    do_move(0, -STICKER_STEP_M)
                    moved = True

                if moved:
                    apply_pos_to_model()
                    blends, mujoco_imgs = render_and_blend(frame_idx)
                    cv2.imshow(win_stat_alpha, blends["stationary"])
                    cv2.imshow(win_stat_mj, mujoco_imgs["stationary"])
                    cv2.imshow(win_wrist_alpha, blends["wrist"])
                    cv2.imshow(win_wrist_mj, mujoco_imgs["wrist"])
                    print(f"[INFO] cube pos: x={data.qpos[cube_qpos_addr]:.4f} y={data.qpos[cube_qpos_addr+1]:.4f}")

            if key in (ord('+'), ord('=')):
                alpha = min(1.0, alpha + 0.05)
                blends, mujoco_imgs = render_and_blend(frame_idx)
                cv2.imshow(win_stat_alpha, blends["stationary"])
                cv2.imshow(win_wrist_alpha, blends["wrist"])
            elif key in (ord('-'), ord('_')):
                alpha = max(0.0, alpha - 0.05)
                blends, mujoco_imgs = render_and_blend(frame_idx)
                cv2.imshow(win_stat_alpha, blends["stationary"])
                cv2.imshow(win_wrist_alpha, blends["wrist"])

        cv2.destroyAllWindows()
        final_pos = data.qpos[cube_qpos_addr : cube_qpos_addr + 2]
        print(f"[INFO] Final cube position: x={final_pos[0]:.4f} y={final_pos[1]:.4f}")


if __name__ == "__main__":
    main()
