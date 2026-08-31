#!/usr/bin/env python3
"""
Task-aware alpha calibration tool.

Run:
  python visual_match/obj_calibration_mujoco.py

The script chooses the scene XML, dataset root, and default editable objects
from the selected task. It can either save global XML calibration edits or
write a per-episode auto-align cache override.
"""

import os
import re
import signal
import sys
import argparse
import subprocess
from contextlib import nullcontext
from dataclasses import dataclass
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
from lerobot.tasks import get_task_profile, get_task_profiles, resolve_task_scene_xml, resolve_task_xarm7_xml
from lerobot_mujoco_utils import lerobot_state_to_mujoco_ctrl
from object_pose_auto_align import (
    ObjectPoseAlignConfig,
    ObjectPoseAlignResult,
    cache_path_for_episode,
    load_result,
    save_result,
    selection_object_names,
)


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
_DEFAULT_RECORD_TASK_ID = "book_shelving"
_DEFAULT_EPISODE = 0
_DEFAULT_FPS = 30.0
_DEFAULT_ALPHA = 0.7
_DEFAULT_CALIB_FRAME = 0
_DEFAULT_DATASET_ROOT = None
_SHOW_SOURCES = False
_SHOW_MUJOCO_VIEWER = True


@dataclass
class _FreeJointEdit:
    joint_name: str
    qpos_addr: int
    pos: np.ndarray
    quat: np.ndarray
    euler: np.ndarray


def normalize_cli_tokens(argv: list[str]) -> list[str]:
    """Allow accidental ':--flag' tokens from copied command snippets."""
    return [token[1:] if token.startswith(":--") else token for token in argv]


def parse_episode_spec(spec: str | int) -> list[int]:
    text = str(spec).strip().strip(";")
    if not text:
        raise argparse.ArgumentTypeError("--episode cannot be empty")

    episodes: list[int] = []
    seen: set[int] = set()
    for part in text.split(","):
        part = part.strip().strip(";")
        if not part:
            continue
        if "-" in part:
            bounds = [item.strip() for item in part.split("-", maxsplit=1)]
            if len(bounds) != 2 or not bounds[0] or not bounds[1]:
                raise argparse.ArgumentTypeError(f"Invalid episode range: {part!r}")
            start, end = (int(bounds[0]), int(bounds[1]))
            if end < start:
                raise argparse.ArgumentTypeError(f"Episode range must be ascending: {part!r}")
            values = range(start, end + 1)
        else:
            values = (int(part),)
        for episode_idx in values:
            if episode_idx < 0:
                raise argparse.ArgumentTypeError("Episode indices must be non-negative")
            if episode_idx not in seen:
                episodes.append(episode_idx)
                seen.add(episode_idx)
    if not episodes:
        raise argparse.ArgumentTypeError("--episode did not contain any indices")
    return episodes


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


def replace_body_rotation_from_euler(text: str, body_name: str, euler: np.ndarray) -> str:
    """Patch a <body> orientation; MJCF may use euler= or quat= (w x y z)."""
    euler_str = f"{euler[0]:.4f} {euler[1]:.4f} {euler[2]:.4f}"
    quat = quat_from_euler_xyz(euler)
    quat_str = f"{quat[0]:.4f} {quat[1]:.4f} {quat[2]:.4f} {quat[3]:.4f}"
    for attr, val in (("euler", euler_str), ("quat", quat_str)):
        pattern = rf'^([ \t]*<body name="{re.escape(body_name)}"[^>]*{attr}=")[^"]*(".*)$'
        updated, count = re.subn(pattern, rf"\g<1>{val}\g<2>", text, count=1, flags=re.MULTILINE)
        if count:
            return updated
    pattern = rf'^([ \t]*<body name="{re.escape(body_name)}"(?=[ \t>/])[^>]*)(/?>.*)$'
    updated, count = re.subn(
        pattern,
        rf'\g<1> euler="{euler_str}"\g<2>',
        text,
        count=1,
        flags=re.MULTILINE,
    )
    if count:
        return updated
    raise ValueError(
        f"Could not find body {body_name!r} to set orientation"
    )


def update_scene_files(
    scene_xml_path: Path,
    xarm7_xml_path: Path,
    *,
    free_joint_bodies_pos_euler: dict[str, tuple[np.ndarray, np.ndarray]],
    xarm7_home_free_joint_body: str | None,
    body_positions: dict[str, np.ndarray],
    body_eulers: dict[str, np.ndarray],
    table_pos: np.ndarray | None,
    table_euler: np.ndarray | None,
) -> None:
    scene_text = scene_xml_path.read_text()
    for body_name, (pos, euler) in free_joint_bodies_pos_euler.items():
        scene_text = replace_xml_attr(
            scene_text, "body", body_name, "pos", f"{pos[0]:.4f} {pos[1]:.4f} {pos[2]:.4f}"
        )
        scene_text = replace_body_rotation_from_euler(scene_text, body_name, euler)
    for name, pos in body_positions.items():
        scene_text = replace_xml_attr(
            scene_text, "body", name, "pos", f"{pos[0]:.4f} {pos[1]:.4f} {pos[2]:.4f}"
        )
    for name, euler in body_eulers.items():
        scene_text = replace_body_rotation_from_euler(scene_text, name, euler)
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

    if (
        xarm7_home_free_joint_body is not None
        and xarm7_home_free_joint_body in free_joint_bodies_pos_euler
        and xarm7_xml_path.exists()
    ):
        pos, euler = free_joint_bodies_pos_euler[xarm7_home_free_joint_body]
        quat = quat_from_euler_xyz(euler)
        xarm_text = xarm7_xml_path.read_text()
        match = re.search(r'^([ \t]*<key name="home" qpos=")([^"]+)(" ctrl=")', xarm_text, flags=re.MULTILINE)
        if not match:
            raise ValueError(f"Could not find home keyframe qpos in {xarm7_xml_path}")
        qpos_tokens = match.group(2).split()
        if len(qpos_tokens) < 7:
            raise ValueError(f"Unexpected home qpos in {xarm7_xml_path}: {match.group(2)}")
        qpos_tokens[-7:] = [
            f"{pos[0]:.4f}",
            f"{pos[1]:.4f}",
            f"{pos[2]:.4f}",
            f"{quat[0]:.4f}",
            f"{quat[1]:.4f}",
            f"{quat[2]:.4f}",
            f"{quat[3]:.4f}",
        ]
        new_qpos = " ".join(qpos_tokens)
        xarm_text = xarm_text[: match.start(2)] + new_qpos + xarm_text[match.end(2) :]
        xarm7_xml_path.write_text(xarm_text)
        print(f"[INFO] Updated free-joint home pose ({xarm7_home_free_joint_body}) in: {xarm7_xml_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Manually align task objects against recorded frames."
    )
    parser.add_argument(
        "--task",
        dest="task_id",
        choices=sorted(get_task_profiles()),
        default=_DEFAULT_RECORD_TASK_ID,
        help="Task profile used for scene XML, dataset defaults, and editable objects.",
    )
    parser.add_argument(
        "--episode",
        type=str,
        default=str(_DEFAULT_EPISODE),
        help=(
            "Episode index or comma/range spec to display, e.g. '0-4,95-99'. "
            "Range mode saves each episode to the auto-align cache."
        ),
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Dataset path to load frames from. Defaults to the task's 480x640 dataset when available.",
    )
    parser.add_argument("--dataset-root", type=str, default=_DEFAULT_DATASET_ROOT)
    parser.add_argument(
        "--calib-frame",
        type=int,
        default=_DEFAULT_CALIB_FRAME,
        help="Frame index to use for manual alignment preview.",
    )
    parser.add_argument("--fps", type=float, default=_DEFAULT_FPS)
    parser.add_argument("--alpha", type=float, default=_DEFAULT_ALPHA)
    parser.add_argument(
        "--save-auto-align-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Save adjusted selection-object poses to the auto-align cache for this episode "
            "instead of updating scene XML files. Pass --no-save-auto-align-cache to write "
            "adjusted poses (e.g. robot_table) directly into the scene XML."
        ),
    )
    parser.add_argument(
        "--auto-align-cache-dir",
        type=str,
        default=None,
        help="Override auto-align cache directory. Defaults to <initial-states-dir>/auto_object_poses.",
    )
    parser.add_argument(
        "--initial-states-dir",
        type=str,
        default=None,
        help="Dataset root used for auto-align cache naming. Defaults to the task dataset root.",
    )
    parser.add_argument(
        "--object-name",
        type=str,
        default=None,
        help="Object names for auto-align cache, e.g. 'mug, saucer'. Defaults to task selection object name.",
    )
    parser.add_argument("--no-mujoco-viewer", action="store_true")
    parser.add_argument("--show-sources", action="store_true")
    args = parser.parse_args(normalize_cli_tokens(sys.argv[1:]))
    try:
        args.episodes = parse_episode_spec(args.episode)
    except (argparse.ArgumentTypeError, ValueError) as exc:
        parser.error(str(exc))
    args.episode = args.episodes[0]
    return args


def save_auto_align_cache(
    *,
    task_profile,
    episode_idx: int,
    initial_states_dir: str | Path,
    object_name: str,
    cache_dir: str | Path | None,
    free_joint_poses: dict[str, tuple[np.ndarray, np.ndarray]],
    body_positions: dict[str, np.ndarray],
    body_eulers: dict[str, np.ndarray],
    model: MjModel,
    body_ids: dict[str, int],
) -> Path:
    object_names = selection_object_names(object_name)
    poses: dict[str, dict[str, np.ndarray]] = {}
    free_joint_bodies = frozenset(task_profile.calibration_free_joint_pair_dict())
    body_aliases = task_profile.object_body_name_aliases

    for name in object_names:
        body_name = body_aliases.get(name, name)
        if body_name in free_joint_bodies:
            if body_name not in free_joint_poses:
                raise ValueError(
                    f"Cannot save {name!r} cache: free-joint pose is not available "
                    f"(task {task_profile.task_id!r})"
                )
            pos, quat = free_joint_poses[body_name]
            poses[name] = {
                "kind": np.array("freejoint"),
                "pos": np.asarray(pos, dtype=np.float64).copy(),
                "quat": quat_normalize(np.asarray(quat, dtype=np.float64)),
            }
            continue

        if body_name not in body_positions or body_name not in body_ids:
            raise ValueError(
                f"Cannot save {name!r} cache: object is not editable for task {task_profile.task_id!r}"
            )
        quat = (
            quat_from_euler_xyz(body_eulers[body_name])
            if body_name in body_eulers
            else model.body_quat[body_ids[body_name]].copy()
        )
        poses[name] = {
            "kind": np.array("body"),
            "pos": np.asarray(body_positions[body_name], dtype=np.float64).copy(),
            "quat": quat_normalize(np.asarray(quat, dtype=np.float64)),
        }

    config = ObjectPoseAlignConfig(
        initial_states_dir=initial_states_dir,
        object_name=object_name,
        cache_dir=cache_dir,
        body_name_aliases=body_aliases,
    )
    cache_path = cache_path_for_episode(config, episode_idx)
    result = ObjectPoseAlignResult(
        episode_idx=episode_idx,
        object_names=object_names,
        loss=float("nan"),
        iou_by_object={},
        poses=poses,
        cache_path=cache_path,
    )
    save_result(result, cache_path)
    return cache_path


def child_argv_for_episode(episode_idx: int) -> list[str]:
    argv = normalize_cli_tokens(sys.argv[1:])
    child_args: list[str] = []
    replaced_episode = False
    skip_next = False

    for token in argv:
        if skip_next:
            skip_next = False
            continue
        if token == "--episode":
            child_args.extend(["--episode", str(episode_idx)])
            replaced_episode = True
            skip_next = True
            continue
        if token.startswith("--episode="):
            child_args.append(f"--episode={episode_idx}")
            replaced_episode = True
            continue
        child_args.append(token)

    if not replaced_episode:
        child_args.extend(["--episode", str(episode_idx)])
    if "--save-auto-align-cache" not in child_args:
        child_args.append("--save-auto-align-cache")

    return [sys.executable, str(Path(__file__).resolve()), *child_args]


def run_episode_sequence(episodes: list[int]) -> int:
    print(f"[INFO] Episode sequence: {episodes}")
    print(f"[INFO] Each episode opens at calibration frame {_DEFAULT_CALIB_FRAME} unless --calib-frame is set.")
    print("[INFO] Press q in each episode to save its .npz and continue; press Esc to stop the sequence.")

    env = os.environ.copy()
    env["OBJ_CALIB_RANGE_CHILD"] = "1"
    for index, episode_idx in enumerate(episodes, start=1):
        print(f"[INFO] Starting episode {episode_idx} ({index}/{len(episodes)})")
        completed = subprocess.run(child_argv_for_episode(episode_idx), env=env)
        if completed.returncode == 2:
            print("[INFO] Episode sequence stopped.")
            return 0
        if completed.returncode != 0:
            print(f"[ERROR] Episode {episode_idx} exited with code {completed.returncode}.")
            return completed.returncode
    print("[INFO] Episode sequence complete.")
    return 0


def main():
    args = parse_args()
    if len(args.episodes) > 1 and os.environ.get("OBJ_CALIB_RANGE_CHILD") != "1":
        sys.exit(run_episode_sequence(args.episodes))

    if os.environ.get("OBJ_CALIB_RANGE_CHILD") == "1":
        args.save_auto_align_cache = True

    task_profile = get_task_profile(args.task_id)
    adjustable_objects = list(dict.fromkeys(task_profile.calibration_adjustable_object_names))
    if not adjustable_objects:
        print(f"[ERROR] Task {args.task_id!r} has no calibration objects configured.")
        sys.exit(1)

    dataset_path = Path(args.dataset_path) if args.dataset_path else PROJECT_ROOT / task_profile.dataset_root_480640
    if not dataset_path.is_absolute():
        dataset_path = PROJECT_ROOT / dataset_path
    if args.dataset_path is None and not dataset_path.is_dir():
        dataset_path = PROJECT_ROOT / task_profile.dataset_root
    if not dataset_path.is_dir():
        print(f"[ERROR] Dataset path not found: {dataset_path}")
        sys.exit(1)

    xarm_dir = PROJECT_ROOT / "xarm7"
    scene_xml_path = resolve_task_scene_xml(args.task_id, xarm_dir)
    xarm7_xml_path = resolve_task_xarm7_xml(args.task_id, xarm_dir)
    print(f"[INFO] Task: {args.task_id}")
    print(f"[INFO] Using scene XML: {scene_xml_path}")
    print(f"[INFO] Robot / home keyframe XML: {xarm7_xml_path}")
    print(f"[INFO] Dataset path: {dataset_path}")
    print(f"[INFO] Adjustable objects: {', '.join(adjustable_objects)}")
    cache_initial_states_dir = args.initial_states_dir or task_profile.dataset_root
    cache_object_name = args.object_name or task_profile.selection_object_name
    preview_cache_config = ObjectPoseAlignConfig(
        initial_states_dir=cache_initial_states_dir,
        object_name=cache_object_name,
        cache_dir=args.auto_align_cache_dir,
        body_name_aliases=task_profile.object_body_name_aliases,
    )
    if args.save_auto_align_cache:
        print(f"[INFO] Auto-align cache mode for objects: {cache_object_name}")
        print(f"[INFO] Will save cache to: {cache_path_for_episode(preview_cache_config, args.episode)}")

    print(f"[INFO] Loading episode {args.episode}...")
    cam_frames, episode_data, num_frames = load_episode_frames(
        str(dataset_path), args.episode, dataset_root=args.dataset_root
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
    robot_geom_ids = get_robot_geom_ids(model, extra_geom_names=[
        "robot_table_leg_1", "robot_table_leg_2", "robot_table_leg_3",
        "robot_table_leg_4", "robot_table_ledger",
    ])

    yaw_rotatable = frozenset(task_profile.calibration_body_yaw_rotatable_names)
    free_edits: dict[str, _FreeJointEdit] = {}
    for body_name, joint_name in task_profile.calibration_free_joint_pairs:
        if body_name not in adjustable_objects:
            continue
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if jid < 0:
            print(
                f"[ERROR] Joint {joint_name!r} for body {body_name!r} not found "
                f"in {scene_xml_path.name}"
            )
            sys.exit(1)
        addr = int(model.jnt_qposadr[jid])
        pos = data.qpos[addr : addr + 3].copy()
        quat = data.qpos[addr + 3 : addr + 7].copy()
        euler = euler_xyz_from_quat(quat)
        free_edits[body_name] = _FreeJointEdit(
            joint_name=joint_name, qpos_addr=addr, pos=pos, quat=quat, euler=euler
        )

    body_positions: dict[str, np.ndarray] = {}
    body_eulers: dict[str, np.ndarray] = {}
    body_ids: dict[str, int] = {}
    table_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "table")
    table_pos = None
    table_euler = None

    for name in adjustable_objects:
        if name == "table":
            if table_body_id < 0:
                print(f"[ERROR] 'table' body not found in {scene_xml_path.name}")
                sys.exit(1)
            table_pos = model.body_pos[table_body_id].copy()
            table_euler = euler_xyz_from_quat(model.body_quat[table_body_id].copy())
            continue
        if name in free_edits:
            continue
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id < 0:
            print(f"[ERROR] '{name}' body not found in {scene_xml_path.name}")
            sys.exit(1)
        body_ids[name] = body_id
        body_positions[name] = model.body_pos[body_id].copy()
        if name in yaw_rotatable:
            body_eulers[name] = euler_xyz_from_quat(model.body_quat[body_id].copy())

    cache_path = cache_path_for_episode(preview_cache_config, args.episode)
    if cache_path.exists():
        try:
            cached_result = load_result(cache_path)
            for name, pose in cached_result.poses.items():
                body_name = task_profile.object_body_name_aliases.get(name, name)
                pos = np.asarray(pose["pos"], dtype=np.float64).copy()
                quat = quat_normalize(np.asarray(pose["quat"], dtype=np.float64))
                if body_name in free_edits:
                    fe = free_edits[body_name]
                    fe.pos[:] = pos
                    fe.quat[:] = quat.copy()
                    fe.euler[:] = euler_xyz_from_quat(fe.quat)
                elif body_name == "table" and table_pos is not None:
                    table_pos = pos
                    table_euler = euler_xyz_from_quat(quat)
                elif body_name in body_positions:
                    body_positions[body_name] = pos
                    if body_name in body_eulers:
                        body_eulers[body_name] = euler_xyz_from_quat(quat)
                    elif body_name in body_ids:
                        model.body_quat[body_ids[body_name]] = quat
            print(f"[INFO] Loaded auto-align cache for initial pose: {cache_path}")
        except Exception as exc:
            print(f"[WARN] Failed to load auto-align cache {cache_path}: {exc}")
    else:
        print(f"[INFO] No auto-align cache found; starting from XML/home pose: {cache_path}")

    alpha = args.alpha
    position_step = POSITION_STEP_M
    current_obj = adjustable_objects[0]
    viewer_handle = None

    free_edits_init = {
        name: _FreeJointEdit(
            joint_name=fe.joint_name,
            qpos_addr=fe.qpos_addr,
            pos=fe.pos.copy(),
            quat=fe.quat.copy(),
            euler=fe.euler.copy(),
        )
        for name, fe in free_edits.items()
    }
    body_positions_init = {k: v.copy() for k, v in body_positions.items()}
    body_eulers_init = {k: v.copy() for k, v in body_eulers.items()}
    table_pos_init = table_pos.copy() if table_pos is not None else None
    table_euler_init = table_euler.copy() if table_euler is not None else None
    def set_frame_pose(frame_idx: int):
        data.ctrl[:] = ctrl_seq[frame_idx]
        sim_target = data.time + 1.0 / args.fps
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
            for fe in free_edits.values():
                data.qpos[fe.qpos_addr : fe.qpos_addr + 3] = fe.pos
                data.qpos[fe.qpos_addr + 3 : fe.qpos_addr + 7] = quat_normalize(fe.quat)
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

    def reset_sim_to_episode_start():
        viewer_lock = viewer_handle.lock() if viewer_handle is not None else nullcontext()
        with viewer_lock:
            try:
                home_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
                mujoco.mj_resetDataKeyframe(model, data, home_id)
            except Exception:
                mujoco.mj_resetData(model, data)
            data.qpos[:7] = ctrl_seq[0, :7]
            data.qpos[7] = ctrl_seq[0, 7] / 255.0 * 0.85
            data.qvel[:8] = 0
            data.ctrl[:] = ctrl_seq[0]
            mujoco.mj_forward(model, data)
        apply_pose_changes()

    def seek_to_frame(frame_idx: int):
        reset_sim_to_episode_start()
        for frame in range(frame_idx + 1):
            set_frame_pose(frame)
        apply_pose_changes()

    def reset_adjustments():
        nonlocal table_pos, table_euler
        for name, fe in free_edits.items():
            ini = free_edits_init[name]
            fe.pos[:] = ini.pos
            fe.quat[:] = ini.quat
            fe.euler[:] = ini.euler
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
        if current_obj in free_edits:
            fe = free_edits[current_obj]
            fe.pos[0] += dx
            fe.pos[1] += dy
            fe.pos[2] += dz
        elif current_obj in body_positions:
            body_positions[current_obj][0] += dx
            body_positions[current_obj][1] += dy
            body_positions[current_obj][2] += dz
        elif current_obj == "table" and table_pos is not None:
            table_pos[0] += dx
            table_pos[1] += dy
            table_pos[2] += dz

    def rotate_current_yaw(delta: float):
        nonlocal table_euler
        if current_obj in free_edits:
            fe = free_edits[current_obj]
            fe.euler[2] += delta
            fe.quat[:] = quat_from_euler_xyz(fe.euler)
            return True
        if current_obj in body_eulers:
            body_eulers[current_obj][2] += delta
            return True
        if current_obj == "table" and table_euler is not None:
            table_euler[2] += delta
            return True
        return False

    def current_status() -> str:
        if current_obj in free_edits:
            fe = free_edits[current_obj]
            return f"{current_obj} pos={fmt_xyz(fe.pos)} euler={fmt_euler_deg(fe.euler)}"
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
                    "Arrows: XY | w/s: Z | j/l: yaw (free-joint + yaw bodies + table) | "
                    "letter keys: select object | a/d: frame -/+1 | Space: replay | x: reset | +/-: alpha"
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

    cf = min(max(args.calib_frame, 0), num_frames - 1)
    seek_to_frame(cf)

    if _SHOW_MUJOCO_VIEWER and not args.no_mujoco_viewer:
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

    if _SHOW_SOURCES or args.show_sources:
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
        "book": "o",
        "book_shelf_target": "h",
        "left_shoe": "f",
        "right_shoe": "g",
        "robot_table": "u",
        "carton": "c",
    }
    select_help = " ".join(
        f"{selection_key_by_object[obj]}={obj}" for obj in adjustable_objects if obj in selection_key_by_object
    )
    print("[INFO] Position calibration mode")
    print("[INFO] Arrows: XY | w/s: Z | j/l: yaw (free-joint objects, yaw-tracked bodies, table)")
    q_help = "save and next episode" if os.environ.get("OBJ_CALIB_RANGE_CHILD") == "1" else "save"
    print(
        f"[INFO] Select: {select_help} | a/d=frame -/+1 | "
        f"Space=replay from frame {cf} | x=reset | q={q_help} | Esc=discard"
    )
    if args.save_auto_align_cache:
        print("[INFO] Save target: per-episode auto-align cache (.npz); XML files will not be modified.")

    def show_frame(frame_idx: int, show_help: bool):
        blends, mujoco_imgs = render_and_blend(frame_idx, cams=["stationary", "wrist"], show_help=show_help)
        cv2.imshow(win_stat, blends["stationary"])
        cv2.imshow(win_wrist, blends["wrist"])
        if _SHOW_SOURCES or args.show_sources:
            real_stat = cam_frames.get("stationary", [])
            if frame_idx < len(real_stat):
                cv2.imshow(win_real, real_stat[frame_idx])
            cv2.imshow(win_mj, mujoco_imgs["stationary"])

    def refresh():
        show_frame(cf, show_help=True)

    def step_calib_frame(delta: int):
        nonlocal cf
        next_cf = min(max(cf + delta, 0), num_frames - 1)
        if next_cf == cf:
            print(f"[INFO] Frame unchanged: {cf}/{num_frames - 1}")
            return
        cf = next_cf
        seek_to_frame(cf)
        print(f"[INFO] Calibration frame: {cf}/{num_frames - 1}")
        refresh()

    def replay_from_calib_frame():
        nonlocal cf
        print(f"[INFO] Replaying episode {args.episode} from frame {cf}/{num_frames - 1}")
        seek_to_frame(cf)
        frame_delay_ms = max(1, int(round(1000.0 / max(args.fps, 1e-6))))
        for replay_frame in range(cf, num_frames):
            if replay_frame > cf:
                set_frame_pose(replay_frame)
            show_frame(replay_frame, show_help=False)
            key = cv2.waitKeyEx(frame_delay_ms)
            if key == ord(" "):
                cf = replay_frame
                print(f"[INFO] Replay stopped at frame {cf}/{num_frames - 1}.")
                refresh()
                return None
            if key in (27, ord("q")):
                seek_to_frame(cf)
                refresh()
                return key
        print("[INFO] Replay finished.")
        seek_to_frame(cf)
        refresh()
        return None

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
            if os.environ.get("OBJ_CALIB_RANGE_CHILD") == "1":
                sys.exit(2)
            return
        if key == ord("q"):
            print("[INFO] Saving updated poses.")
            break
        if key == ord("x"):
            reset_adjustments()
            print("[INFO] Reset all editable poses.")
            refresh()
            continue
        if key == ord("a"):
            step_calib_frame(-1)
            continue
        if key == ord("d"):
            step_calib_frame(1)
            continue
        if key == ord(" "):
            replay_key = replay_from_calib_frame()
            if replay_key is None:
                continue
            if replay_key == 27:
                print("[INFO] Esc — changes discarded.")
                if viewer_handle is not None:
                    viewer_handle.close()
                cv2.destroyAllWindows()
                if os.environ.get("OBJ_CALIB_RANGE_CHILD") == "1":
                    sys.exit(2)
                return
            if replay_key == ord("q"):
                print("[INFO] Saving updated poses.")
                break

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
        select_key_codes = [
            ord(selection_key_by_object[obj])
            for obj in adjustable_objects
            if obj in selection_key_by_object
        ]
        if moved or rotated or key in (
            *select_key_codes,
            ord("+"), ord("="), ord("-"), ord("_")
        ):
            refresh()

    if viewer_handle is not None:
        viewer_handle.close()
    cv2.destroyAllWindows()
    body_updates = {name: pos for name, pos in body_positions.items()}
    body_euler_updates = {name: euler for name, euler in body_eulers.items()}
    free_joint_poses_save = {name: (fe.pos.copy(), fe.quat.copy()) for name, fe in free_edits.items()}
    free_joint_scene_pos_euler = {name: (fe.pos.copy(), fe.euler.copy()) for name, fe in free_edits.items()}
    xarm7_home_body = task_profile.calibration_xarm7_home_free_joint_body_resolved()
    if args.save_auto_align_cache:
        cache_path = save_auto_align_cache(
            task_profile=task_profile,
            episode_idx=args.episode,
            initial_states_dir=args.initial_states_dir or task_profile.dataset_root,
            object_name=args.object_name or task_profile.selection_object_name,
            cache_dir=args.auto_align_cache_dir,
            free_joint_poses=free_joint_poses_save,
            body_positions=body_updates,
            body_eulers=body_euler_updates,
            model=model,
            body_ids=body_ids,
        )
        print(f"[INFO] Saved auto-align cache: {cache_path}")
        print("[INFO] Re-run compare_recorded_vs_mujoco.py without --auto-align-force to use this pose.")
    else:
        update_scene_files(
            scene_xml_path=scene_xml_path,
            xarm7_xml_path=xarm7_xml_path,
            free_joint_bodies_pos_euler=free_joint_scene_pos_euler,
            xarm7_home_free_joint_body=xarm7_home_body,
            body_positions=body_updates,
            body_eulers=body_euler_updates,
            table_pos=table_pos if "table" in adjustable_objects else None,
            table_euler=table_euler if "table" in adjustable_objects else None,
        )
    for name, fe in free_edits.items():
        print(f"[INFO] Final {name} pose: pos={fmt_xyz(fe.pos)} euler={fmt_euler_deg(fe.euler)}")
    for name, pos in body_updates.items():
        suffix = f" euler={fmt_euler_deg(body_euler_updates[name])}" if name in body_euler_updates else ""
        print(f"[INFO] Final {name} position: {fmt_xyz(pos)}{suffix}")
    if table_pos is not None:
        print(f"[INFO] Final table pose: pos={fmt_xyz(table_pos)} euler={fmt_euler_deg(table_euler)}")


if __name__ == "__main__":
    main()
