#!/usr/bin/env python3
"""
Task-aware alpha calibration tool with a hardcoded default task.

Run:
  python visual_match/sticker_alpha_calibration.py

The script automatically chooses the scene XML, dataset root, and default
editable objects from `_DEFAULT_RECORD_TASK_ID`.
"""

import os
import re
import signal
import sys
from contextlib import nullcontext
from pathlib import Path

import cv2
import mujoco
import mujoco.viewer
import numpy as np
from mujoco import MjData, MjModel

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from camera_config import load_camera_config, set_mujoco_camera_from_config
from compare_recorded_vs_mujoco import load_episode
from composite_rendering import get_robot_geom_ids, shift_for_principal_point
from lerobot.datasets.video_utils import decode_video_frames
from lerobot.tasks import get_task_profile, resolve_task_scene_xml, resolve_task_xarm7_xml
from lerobot_mujoco_utils import lerobot_state_to_mujoco_ctrl


RENDER_W, RENDER_H = 640, 480
POSITION_STEP_M = 0.005
ROTATION_STEP_RAD = np.deg2rad(1.0)
TABLE_YAW_STEP_RAD = np.deg2rad(1)

KEY_LEFT = (65361, 81, 2)
KEY_UP = (65362, 82, 0)
KEY_RIGHT = (65363, 83, 3)
KEY_DOWN = (65364, 84, 1)

_stationary_cfg = load_camera_config("stationary_cam")
_wrist_cfg = load_camera_config("wrist_cam")
CAMERA_CONFIG = {
    "stationary": {"dataset_cam": "cam_high", "mujoco_cam": "stationary_cam", "config": _stationary_cfg},
    "wrist": {"dataset_cam": "cam_wrist", "mujoco_cam": "wrist_cam", "config": _wrist_cfg},
}

PROJECT_ROOT = Path(__file__).parent.parent
_DEFAULT_RECORD_TASK_ID = "hang_mug"
_DEFAULT_EPISODE = 1
_DEFAULT_FPS = 30.0
_DEFAULT_ALPHA = 0.7
_DEFAULT_CALIB_FRAME = 0
_DEFAULT_DATASET_ROOT = None
_SHOW_SOURCES = False
_SHOW_MUJOCO_VIEWER = True
_YAW_ROTATABLE_BODIES = {"rack"}


def quat_normalize(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    norm = np.linalg.norm(quat)
    if norm <= 0:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return quat / norm


def quat_from_euler_xyz(euler: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = [float(v) for v in euler]
    cr, sr = np.cos(roll / 2.0), np.sin(roll / 2.0)
    cp, sp = np.cos(pitch / 2.0), np.sin(pitch / 2.0)
    cy, sy = np.cos(yaw / 2.0), np.sin(yaw / 2.0)
    quat = np.array(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
        dtype=np.float64,
    )
    return quat_normalize(quat)


def euler_xyz_from_quat(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = [float(v) for v in quat_normalize(quat)]
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = np.copysign(np.pi / 2.0, sinp)
    else:
        pitch = np.arcsin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return np.array([roll, pitch, yaw], dtype=np.float64)


def fmt_xyz(vec: np.ndarray | None) -> str:
    if vec is None:
        return "n/a"
    return f"[{vec[0]:.4f}, {vec[1]:.4f}, {vec[2]:.4f}]"


def fmt_euler_deg(euler: np.ndarray | None) -> str:
    if euler is None:
        return "n/a"
    deg = np.degrees(euler)
    return f"[{deg[0]:.1f}, {deg[1]:.1f}, {deg[2]:.1f}] deg"


def load_episode_frames(dataset_path: str, episode_idx: int, dataset_root: str | None = None):
    episode_data = load_episode(dataset_path, episode_idx, dataset_root=dataset_root)
    dataset = episode_data["dataset"]
    ep_meta = dataset.meta.episodes[episode_idx]
    num_frames = episode_data["num_frames"]
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
                frames_list.append(cv2.cvtColor((frame * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
            cam_frames[cam_key] = frames_list
            print(f"[INFO] Loaded {len(frames_list)} frames for {cam_key}")
        except Exception as exc:
            print(f"[WARN] Failed to load {cam_key} camera: {exc}")
            cam_frames[cam_key] = []

    return cam_frames, episode_data, num_frames


def resolve_scene_path(task_id: str, scene_xml_path: str | None) -> Path:
    if scene_xml_path is not None:
        path = Path(scene_xml_path)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path
    return resolve_task_scene_xml(task_id, PROJECT_ROOT / "xarm7")


def replace_xml_attr(text: str, element: str, name: str, attr: str, value: str) -> str:
    pattern = rf'^([ \t]*<{element} name="{re.escape(name)}"[^>]*{attr}=")[^"]*(".*)$'
    updated, count = re.subn(pattern, rf"\g<1>{value}\g<2>", text, count=1, flags=re.MULTILINE)
    if count == 0:
        raise ValueError(f"Could not find {element} {name!r} attribute {attr!r}")
    return updated


def update_scene_files(
    scene_xml_path: Path,
    xarm7_xml_path: Path,
    mug_pos: np.ndarray | None,
    mug_euler: np.ndarray | None,
    body_positions: dict[str, np.ndarray],
    body_eulers: dict[str, np.ndarray],
    table_pos: np.ndarray | None,
    table_euler: np.ndarray | None,
) -> None:
    scene_text = scene_xml_path.read_text()
    if mug_pos is not None:
        scene_text = replace_xml_attr(
            scene_text, "body", "mug", "pos", f"{mug_pos[0]:.4f} {mug_pos[1]:.4f} {mug_pos[2]:.4f}"
        )
    if mug_euler is not None:
        scene_text = replace_xml_attr(
            scene_text, "body", "mug", "euler", f"{mug_euler[0]:.4f} {mug_euler[1]:.4f} {mug_euler[2]:.4f}"
        )
    for name, pos in body_positions.items():
        scene_text = replace_xml_attr(
            scene_text, "body", name, "pos", f"{pos[0]:.4f} {pos[1]:.4f} {pos[2]:.4f}"
        )
    for name, euler in body_eulers.items():
        scene_text = replace_xml_attr(
            scene_text, "body", name, "euler", f"{euler[0]:.4f} {euler[1]:.4f} {euler[2]:.4f}"
        )
    if table_pos is not None:
        scene_text = replace_xml_attr(
            scene_text, "body", "table", "pos", f"{table_pos[0]:.4f} {table_pos[1]:.4f} {table_pos[2]:.4f}"
        )
    if table_euler is not None:
        scene_text = replace_xml_attr(
            scene_text, "body", "table", "euler", f"{table_euler[0]:.4f} {table_euler[1]:.4f} {table_euler[2]:.4f}"
        )
    scene_xml_path.write_text(scene_text)
    print(f"[INFO] Updated scene XML: {scene_xml_path}")

    if mug_pos is not None and mug_euler is not None and xarm7_xml_path.exists():
        quat = quat_from_euler_xyz(mug_euler)
        xarm_text = xarm7_xml_path.read_text()
        match = re.search(r'^([ \t]*<key name="home" qpos=")([^"]+)(" ctrl=")', xarm_text, flags=re.MULTILINE)
        if not match:
            raise ValueError(f"Could not find home keyframe qpos in {xarm7_xml_path}")
        qpos_tokens = match.group(2).split()
        if len(qpos_tokens) < 7:
            raise ValueError(f"Unexpected home qpos in {xarm7_xml_path}: {match.group(2)}")
        qpos_tokens[-7:] = [
            f"{mug_pos[0]:.4f}",
            f"{mug_pos[1]:.4f}",
            f"{mug_pos[2]:.4f}",
            f"{quat[0]:.4f}",
            f"{quat[1]:.4f}",
            f"{quat[2]:.4f}",
            f"{quat[3]:.4f}",
        ]
        new_qpos = " ".join(qpos_tokens)
        xarm_text = xarm_text[: match.start(2)] + new_qpos + xarm_text[match.end(2) :]
        xarm7_xml_path.write_text(xarm_text)
        print(f"[INFO] Updated mug home pose in: {xarm7_xml_path}")


def main():
    task_profile = get_task_profile(_DEFAULT_RECORD_TASK_ID)
    adjustable_objects = list(dict.fromkeys(task_profile.calibration_adjustable_object_names))
    if not adjustable_objects:
        print(f"[ERROR] Task {_DEFAULT_RECORD_TASK_ID!r} has no calibration objects configured.")
        sys.exit(1)

    dataset_path = PROJECT_ROOT / task_profile.dataset_root
    if not dataset_path.is_dir():
        print(f"[ERROR] Dataset path not found: {dataset_path}")
        sys.exit(1)

    xarm_dir = PROJECT_ROOT / "xarm7"
    scene_xml_path = resolve_task_scene_xml(_DEFAULT_RECORD_TASK_ID, xarm_dir)
    xarm7_xml_path = resolve_task_xarm7_xml(_DEFAULT_RECORD_TASK_ID, xarm_dir)
    print(f"[INFO] Task: {_DEFAULT_RECORD_TASK_ID}")
    print(f"[INFO] Using scene XML: {scene_xml_path}")
    print(f"[INFO] Robot / home keyframe XML: {xarm7_xml_path}")
    print(f"[INFO] Adjustable objects: {', '.join(adjustable_objects)}")

    print(f"[INFO] Loading episode {_DEFAULT_EPISODE}...")
    cam_frames, episode_data, num_frames = load_episode_frames(
        str(dataset_path), _DEFAULT_EPISODE, dataset_root=_DEFAULT_DATASET_ROOT
    )
    observations = episode_data["observation.state"].numpy()

    original_cwd = os.getcwd()
    try:
        os.chdir(str(scene_xml_path.parent))
        model = MjModel.from_xml_path(scene_xml_path.name)
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

    data.qpos[:7] = ctrl_seq[0, :7]
    data.qpos[7] = ctrl_seq[0, 7] / 255.0 * 0.85
    data.qvel[:8] = 0
    mujoco.mj_forward(model, data)
    print("[INFO] Initialized sim from dataset first frame")

    for cam_cfg in CAMERA_CONFIG.values():
        K = cam_cfg["config"]["intrinsics"]
        fy = K[1, 1]
        correct_fovy = float(2.0 * np.degrees(np.arctan(RENDER_H / (2.0 * fy))))
        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_cfg["mujoco_cam"])
        if cam_id >= 0:
            model.cam_fovy[cam_id] = correct_fovy
    mujoco.mj_forward(model, data)
    for cam_cfg in CAMERA_CONFIG.values():
        set_mujoco_camera_from_config(data, model, cam_cfg["mujoco_cam"], cam_cfg["config"])

    renderer = mujoco.Renderer(model, height=RENDER_H, width=RENDER_W)
    seg_renderer = mujoco.Renderer(model, height=RENDER_H, width=RENDER_W)
    seg_renderer.enable_segmentation_rendering()
    robot_geom_ids = get_robot_geom_ids(model)

    mug_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "mug")
    mug_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "mug_joint")
    mug_qpos_addr = model.jnt_qposadr[mug_joint_id] if mug_joint_id >= 0 else -1
    mug_pos = data.qpos[mug_qpos_addr : mug_qpos_addr + 3].copy() if mug_qpos_addr >= 0 else None
    mug_quat = data.qpos[mug_qpos_addr + 3 : mug_qpos_addr + 7].copy() if mug_qpos_addr >= 0 else None
    mug_euler = euler_xyz_from_quat(mug_quat) if mug_quat is not None else None

    body_positions: dict[str, np.ndarray] = {}
    body_eulers: dict[str, np.ndarray] = {}
    body_ids: dict[str, int] = {}
    table_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "table")
    table_pos = None
    table_euler = None

    if "mug" in adjustable_objects and mug_qpos_addr < 0:
        print(f"[ERROR] 'mug_joint' not found in {scene_xml_path.name}")
        sys.exit(1)

    for name in adjustable_objects:
        if name == "mug":
            continue
        if name == "table":
            if table_body_id < 0:
                print(f"[ERROR] 'table' body not found in {scene_xml_path.name}")
                sys.exit(1)
            table_pos = model.body_pos[table_body_id].copy()
            table_euler = euler_xyz_from_quat(model.body_quat[table_body_id].copy())
            continue
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id < 0:
            print(f"[ERROR] '{name}' body not found in {scene_xml_path.name}")
            sys.exit(1)
        body_ids[name] = body_id
        body_positions[name] = model.body_pos[body_id].copy()
        if name in _YAW_ROTATABLE_BODIES:
            body_eulers[name] = euler_xyz_from_quat(model.body_quat[body_id].copy())

    alpha = _DEFAULT_ALPHA
    position_step = POSITION_STEP_M
    current_obj = adjustable_objects[0]
    viewer_handle = None

    mug_pos_init = mug_pos.copy() if mug_pos is not None else None
    mug_quat_init = mug_quat.copy() if mug_quat is not None else None
    mug_euler_init = mug_euler.copy() if mug_euler is not None else None
    body_positions_init = {k: v.copy() for k, v in body_positions.items()}
    body_eulers_init = {k: v.copy() for k, v in body_eulers.items()}
    table_pos_init = table_pos.copy() if table_pos is not None else None
    table_euler_init = table_euler.copy() if table_euler is not None else None
    def set_frame_pose(frame_idx: int):
        data.ctrl[:] = ctrl_seq[frame_idx]
        sim_target = data.time + 1.0 / _DEFAULT_FPS
        while data.time < sim_target:
            mujoco.mj_step(model, data)

    def sync_viewer():
        nonlocal viewer_handle
        if viewer_handle is None:
            return
        try:
            if hasattr(viewer_handle, "is_running") and not viewer_handle.is_running():
                viewer_handle = None
                return
            viewer_handle.sync()
        except Exception as exc:
            print(f"[WARN] MuJoCo viewer sync failed: {exc}")
            viewer_handle = None

    def apply_pose_changes():
        viewer_lock = viewer_handle.lock() if viewer_handle is not None else nullcontext()
        with viewer_lock:
            if "mug" in adjustable_objects and mug_qpos_addr >= 0 and mug_pos is not None and mug_quat is not None:
                data.qpos[mug_qpos_addr : mug_qpos_addr + 3] = mug_pos
                data.qpos[mug_qpos_addr + 3 : mug_qpos_addr + 7] = quat_normalize(mug_quat)
            for name, pos in body_positions.items():
                model.body_pos[body_ids[name]] = pos
                if name in body_eulers:
                    model.body_quat[body_ids[name]] = quat_normalize(quat_from_euler_xyz(body_eulers[name]))
            if table_pos is not None:
                model.body_pos[table_body_id] = table_pos
            if table_euler is not None:
                model.body_quat[table_body_id] = quat_normalize(quat_from_euler_xyz(table_euler))
        mujoco.mj_forward(model, data)
        for cam_cfg in CAMERA_CONFIG.values():
            if cam_cfg["config"].get("type", "stationary") == "stationary":
                set_mujoco_camera_from_config(data, model, cam_cfg["mujoco_cam"], cam_cfg["config"])
        sync_viewer()

    def reset_adjustments():
        nonlocal mug_pos, mug_quat, mug_euler, table_pos, table_euler
        mug_pos = mug_pos_init.copy() if mug_pos_init is not None else None
        mug_quat = mug_quat_init.copy() if mug_quat_init is not None else None
        mug_euler = mug_euler_init.copy() if mug_euler_init is not None else None
        body_positions.clear()
        for name, pos in body_positions_init.items():
            body_positions[name] = pos.copy()
        body_eulers.clear()
        for name, euler in body_eulers_init.items():
            body_eulers[name] = euler.copy()
        table_pos = table_pos_init.copy() if table_pos_init is not None else None
        table_euler = table_euler_init.copy() if table_euler_init is not None else None
        apply_pose_changes()

    def move_current(dx: float, dy: float, dz: float):
        if current_obj == "mug" and mug_pos is not None:
            mug_pos[0] += dx
            mug_pos[1] += dy
            mug_pos[2] += dz
        elif current_obj in body_positions:
            body_positions[current_obj][0] += dx
            body_positions[current_obj][1] += dy
            body_positions[current_obj][2] += dz
        elif current_obj == "table" and table_pos is not None:
            table_pos[0] += dx
            table_pos[1] += dy
            table_pos[2] += dz

    def rotate_current_yaw(delta: float):
        nonlocal mug_euler, mug_quat, table_euler
        if current_obj == "mug" and mug_euler is not None:
            mug_euler[2] += delta
            mug_quat = quat_from_euler_xyz(mug_euler)
            return True
        if current_obj in body_eulers:
            body_eulers[current_obj][2] += delta
            return True
        if current_obj == "table" and table_euler is not None:
            table_euler[2] += delta
            return True
        return False

    def current_status() -> str:
        if current_obj == "mug":
            return f"mug pos={fmt_xyz(mug_pos)} euler={fmt_euler_deg(mug_euler)}"
        if current_obj in body_positions:
            status = f"{current_obj} pos={fmt_xyz(body_positions[current_obj])}"
            if current_obj in body_eulers:
                status += f" euler={fmt_euler_deg(body_eulers[current_obj])}"
            return status
        if current_obj == "table":
            return f"table pos={fmt_xyz(table_pos)} euler={fmt_euler_deg(table_euler)}"
        return current_obj

    def render_and_blend(frame_idx: int, cams: list[str] | None = None, show_help: bool = False):
        apply_pose_changes()
        blends = {}
        mujoco_imgs = {}
        cam_list = cams if cams is not None else list(CAMERA_CONFIG.keys())
        for cam_key in cam_list:
            cam_cfg = CAMERA_CONFIG[cam_key]
            mujoco_cam = cam_cfg["mujoco_cam"]
            intrinsics = cam_cfg["config"]["intrinsics"]

            renderer.update_scene(data, camera=mujoco_cam)
            mj_rgb = renderer.render()
            mj_bgr = cv2.cvtColor(mj_rgb, cv2.COLOR_RGB2BGR)

            seg_renderer.update_scene(data, camera=mujoco_cam)
            seg_mask = seg_renderer.render()
            seg_labels = seg_mask[:, :, 0].astype(np.int32)
            seg_labels[seg_labels == -1] = 0

            mj_bgr = shift_for_principal_point(mj_bgr, intrinsics)
            seg_labels = shift_for_principal_point(seg_labels, intrinsics, seg=True)
            mask = np.isin(seg_labels, list(robot_geom_ids))
            alpha_mask = np.stack([mask.astype(np.float32)] * 3, axis=-1)

            real_frames = cam_frames.get(cam_key, [])
            real = real_frames[frame_idx] if frame_idx < len(real_frames) else None
            if real is None or real.size == 0:
                blended = mj_bgr.copy()
            else:
                if real.shape[:2] != mj_bgr.shape[:2]:
                    real = cv2.resize(real, (mj_bgr.shape[1], mj_bgr.shape[0]))
                blended = (alpha * mj_bgr.astype(np.float32) + (1.0 - alpha) * real.astype(np.float32))
                blended = blended * alpha_mask + real.astype(np.float32) * (1.0 - alpha_mask)
                blended = blended.astype(np.uint8)

            mj_labeled = mj_bgr.copy()
            cv2.putText(mj_labeled, f"Frame: {frame_idx}/{num_frames}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            mujoco_imgs[cam_key] = mj_labeled

            cv2.putText(blended, current_status(), (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
            if show_help:
                help_text = (
                    "Arrows: XY | w/s: Z | j/l: yaw (mug/rack/table) | "
                    "m/r/p/t/b: select | x: reset | +/-: alpha"
                )
                cv2.putText(blended, help_text, (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
            else:
                cv2.putText(blended, f"Frame: {frame_idx}/{num_frames}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            blends[cam_key] = blended
        return blends, mujoco_imgs

    if num_frames == 0:
        print("[ERROR] No frames in episode")
        sys.exit(1)

    cf = min(_DEFAULT_CALIB_FRAME, num_frames - 1)
    for frame in range(cf + 1):
        set_frame_pose(frame)
    apply_pose_changes()

    if _SHOW_MUJOCO_VIEWER:
        try:
            viewer_handle = mujoco.viewer.launch_passive(model, data, show_left_ui=True, show_right_ui=True)
            viewer_handle.cam.lookat[:] = np.array([0.45, 0.0, 0.12], dtype=np.float64)
            viewer_handle.cam.distance = 0.75
            viewer_handle.cam.azimuth = 135.0
            viewer_handle.cam.elevation = -20.0
            sync_viewer()
            print("[INFO] MuJoCo viewer opened for free-camera inspection.")
        except Exception as exc:
            print(f"[WARN] Failed to open MuJoCo viewer: {exc}")
            viewer_handle = None

    win_stat = f"Stationary - Alpha Blend (frame {cf})"
    win_wrist = f"Wrist - Alpha Blend (frame {cf})"
    cv2.namedWindow(win_stat, cv2.WINDOW_NORMAL)
    cv2.namedWindow(win_wrist, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_stat, RENDER_W, RENDER_H)
    cv2.resizeWindow(win_wrist, RENDER_W, RENDER_H)
    cv2.moveWindow(win_stat, 50, 30)
    cv2.moveWindow(win_wrist, 50 + RENDER_W + 20, 30)

    if _SHOW_SOURCES:
        win_real = "Stationary - Real (source)"
        win_mj = "Stationary - MuJoCo (source)"
        cv2.namedWindow(win_real, cv2.WINDOW_NORMAL)
        cv2.namedWindow(win_mj, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win_real, RENDER_W, RENDER_H)
        cv2.resizeWindow(win_mj, RENDER_W, RENDER_H)
        cv2.moveWindow(win_real, 50, 40 + RENDER_H)
        cv2.moveWindow(win_mj, 50 + RENDER_W + 20, 40 + RENDER_H)

    selection_key_by_object = {
        "mug": "m",
        "rack": "r",
        "saucer": "p",
        "sticker": "t",
        "table": "b",
    }
    select_help = " ".join(
        f"{selection_key_by_object[obj]}={obj}" for obj in adjustable_objects if obj in selection_key_by_object
    )
    print("[INFO] Position calibration mode")
    print("[INFO] Arrows: XY | w/s: Z | j/l: yaw (mug/rack/table)")
    print(f"[INFO] Select: {select_help} | x=reset | q=save | Esc=discard")

    def refresh():
        blends, mujoco_imgs = render_and_blend(cf, cams=["stationary", "wrist"], show_help=True)
        cv2.imshow(win_stat, blends["stationary"])
        cv2.imshow(win_wrist, blends["wrist"])
        if _SHOW_SOURCES:
            real_stat = cam_frames.get("stationary", [])
            if cf < len(real_stat):
                cv2.imshow(win_real, real_stat[cf])
            cv2.imshow(win_mj, mujoco_imgs["stationary"])

    def on_sigint(*_):
        if viewer_handle is not None:
            viewer_handle.close()
        cv2.destroyAllWindows()
        print("[INFO] Ctrl+C — changes discarded.")
        sys.exit(0)

    signal.signal(signal.SIGINT, on_sigint)
    refresh()

    while True:
        key = cv2.waitKeyEx(50)
        if key < 0:
            continue
        if key == 27:
            print("[INFO] Esc — changes discarded.")
            if viewer_handle is not None:
                viewer_handle.close()
            cv2.destroyAllWindows()
            return
        if key == ord("q"):
            print("[INFO] Saving updated poses.")
            break
        if key == ord("x"):
            reset_adjustments()
            print("[INFO] Reset all editable poses.")
            refresh()
            continue

        moved = False
        rotated = False
        if key in KEY_LEFT:
            move_current(-position_step, 0.0, 0.0)
            moved = True
        elif key in KEY_RIGHT:
            move_current(position_step, 0.0, 0.0)
            moved = True
        elif key in KEY_UP:
            move_current(0.0, position_step, 0.0)
            moved = True
        elif key in KEY_DOWN:
            move_current(0.0, -position_step, 0.0)
            moved = True
        elif key == ord("w"):
            move_current(0.0, 0.0, position_step)
            moved = True
        elif key == ord("s"):
            move_current(0.0, 0.0, -position_step)
            moved = True
        elif key == ord("j"):
            yaw_step = TABLE_YAW_STEP_RAD if current_obj == "table" else ROTATION_STEP_RAD
            rotated = rotate_current_yaw(-yaw_step)
        elif key == ord("l"):
            yaw_step = TABLE_YAW_STEP_RAD if current_obj == "table" else ROTATION_STEP_RAD
            rotated = rotate_current_yaw(yaw_step)

        for obj_name, key_char in selection_key_by_object.items():
            if key == ord(key_char) and obj_name in adjustable_objects:
                current_obj = obj_name
                print(f"[INFO] Selected: {obj_name}")
                break
        if key in (ord("+"), ord("=")):
            alpha = min(1.0, alpha + 0.05)
            print(f"[INFO] Alpha: {alpha:.2f}")
        elif key in (ord("-"), ord("_")):
            alpha = max(0.0, alpha - 0.05)
            print(f"[INFO] Alpha: {alpha:.2f}")

        if moved or rotated:
            apply_pose_changes()
            print(f"[INFO] {current_status()}")
        if moved or rotated or key in (
            ord("m"), ord("r"), ord("p"), ord("t"), ord("b"),
            ord("+"), ord("="), ord("-"), ord("_")
        ):
            refresh()

    if viewer_handle is not None:
        viewer_handle.close()
    cv2.destroyAllWindows()
    body_updates = {name: pos for name, pos in body_positions.items()}
    body_euler_updates = {name: euler for name, euler in body_eulers.items()}
    update_scene_files(
        scene_xml_path=scene_xml_path,
        xarm7_xml_path=xarm7_xml_path,
        mug_pos=mug_pos if "mug" in adjustable_objects else None,
        mug_euler=mug_euler if "mug" in adjustable_objects else None,
        body_positions=body_updates,
        body_eulers=body_euler_updates,
        table_pos=table_pos if "table" in adjustable_objects else None,
        table_euler=table_euler if "table" in adjustable_objects else None,
    )
    print(f"[INFO] Final mug pose: pos={fmt_xyz(mug_pos)} euler={fmt_euler_deg(mug_euler)}")
    for name, pos in body_updates.items():
        suffix = f" euler={fmt_euler_deg(body_euler_updates[name])}" if name in body_euler_updates else ""
        print(f"[INFO] Final {name} position: {fmt_xyz(pos)}{suffix}")
    if table_pos is not None:
        print(f"[INFO] Final table pose: pos={fmt_xyz(table_pos)} euler={fmt_euler_deg(table_euler)}")


if __name__ == "__main__":
    main()
