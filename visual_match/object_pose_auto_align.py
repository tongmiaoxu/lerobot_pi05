"""Automatic initial object pose alignment from saved SAM masks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import mujoco
import numpy as np

from camera_config import set_mujoco_camera_from_config
from composite_rendering import shift_for_principal_point


@dataclass
class ObjectPoseAlignConfig:
    initial_states_dir: str | Path
    object_name: str
    cache_dir: str | Path | None = None
    camera_key: str = "stationary"
    optimize_z: bool = False
    xy_range_m: float = 0.12
    z_range_m: float = 0.03
    yaw_range_rad: float = np.deg2rad(120.0)
    min_iou: float = 0.08
    force: bool = False
    camera_adjust_delta: np.ndarray | None = None


@dataclass
class ObjectPoseAlignResult:
    episode_idx: int
    object_names: list[str]
    loss: float
    iou_by_object: dict[str, float]
    poses: dict[str, dict[str, np.ndarray]]
    cache_path: Path | None = None


def selection_object_names(object_name: str) -> list[str]:
    return [
        name.strip().replace("/", "_").replace(" ", "_")
        for name in object_name.split(",")
        if name.strip()
    ]


def default_cache_dir(initial_states_dir: str | Path) -> Path:
    return Path(initial_states_dir) / "auto_object_poses"


def cache_path_for_episode(config: ObjectPoseAlignConfig, episode_idx: int) -> Path:
    cache_root = Path(config.cache_dir) if config.cache_dir is not None else default_cache_dir(config.initial_states_dir)
    object_tag = "_".join(selection_object_names(config.object_name))
    return cache_root / object_tag / f"episode_{episode_idx:06d}.npz"


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
    pitch = np.copysign(np.pi / 2.0, sinp) if abs(sinp) >= 1.0 else np.arcsin(sinp)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return np.array([roll, pitch, yaw], dtype=np.float64)


def load_target_masks(
    initial_states_dir: str | Path,
    object_names: list[str],
    episode_idx: int,
) -> dict[str, np.ndarray]:
    masks = {}
    missing = []
    for object_name in object_names:
        path = Path(initial_states_dir) / object_name / "individual_masks" / f"ep_{episode_idx:03d}_mask.png"
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            missing.append(str(path))
            continue
        masks[object_name] = mask > 127
    if missing:
        raise FileNotFoundError("Missing initial-state mask(s):\n" + "\n".join(missing))
    return masks


def _target_precompute(target_masks: dict[str, np.ndarray]) -> dict[str, dict[str, np.ndarray | float]]:
    precomp = {}
    for name, mask in target_masks.items():
        dt = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 3)
        ys, xs = np.where(mask)
        centroid = np.array([xs.mean(), ys.mean()], dtype=np.float64) if len(xs) else np.array([np.nan, np.nan])
        precomp[name] = {"mask": mask, "dt": dt, "centroid": centroid, "area": float(mask.sum())}
    return precomp


def _geom_id_for_object(model: mujoco.MjModel, object_name: str) -> int:
    for geom_name in (f"{object_name}_visual", object_name):
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        if geom_id >= 0:
            return int(geom_id)
    raise ValueError(f"Could not find visual geom for object {object_name!r}")


def _body_id_for_object(model: mujoco.MjModel, object_name: str) -> int:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, object_name)
    if body_id < 0:
        raise ValueError(f"Could not find MuJoCo body {object_name!r}")
    return int(body_id)


def _mug_qpos_addr(model: mujoco.MjModel) -> int:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "mug_joint")
    return int(model.jnt_qposadr[joint_id]) if joint_id >= 0 else -1


def _capture_pose(model: mujoco.MjModel, data: mujoco.MjData, object_name: str) -> dict[str, np.ndarray]:
    if object_name == "mug":
        qpos_addr = _mug_qpos_addr(model)
        if qpos_addr >= 0:
            return {
                "kind": np.array("freejoint"),
                "pos": data.qpos[qpos_addr:qpos_addr + 3].copy(),
                "quat": quat_normalize(data.qpos[qpos_addr + 3:qpos_addr + 7].copy()),
            }
    body_id = _body_id_for_object(model, object_name)
    return {
        "kind": np.array("body"),
        "pos": model.body_pos[body_id].copy(),
        "quat": quat_normalize(model.body_quat[body_id].copy()),
    }


def capture_object_poses(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    object_names: list[str],
) -> dict[str, dict[str, np.ndarray]]:
    return {name: _capture_pose(model, data, name) for name in object_names}


def apply_object_poses(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    poses: dict[str, dict[str, np.ndarray]],
) -> None:
    for object_name, pose in poses.items():
        kind = str(pose.get("kind", "body"))
        pos = np.asarray(pose["pos"], dtype=np.float64)
        quat = quat_normalize(np.asarray(pose["quat"], dtype=np.float64))
        if kind == "freejoint":
            qpos_addr = _mug_qpos_addr(model)
            if qpos_addr < 0:
                continue
            data.qpos[qpos_addr:qpos_addr + 3] = pos
            data.qpos[qpos_addr + 3:qpos_addr + 7] = quat
        else:
            body_id = _body_id_for_object(model, object_name)
            model.body_pos[body_id] = pos
            model.body_quat[body_id] = quat
    mujoco.mj_forward(model, data)


def _pack_params(
    poses: dict[str, dict[str, np.ndarray]],
    object_names: list[str],
    optimize_z: bool,
) -> tuple[np.ndarray, list[tuple[str, str]]]:
    values = []
    specs = []
    for object_name in object_names:
        pos = np.asarray(poses[object_name]["pos"], dtype=np.float64)
        euler = euler_xyz_from_quat(poses[object_name]["quat"])
        for axis in range(2):
            values.append(pos[axis])
            specs.append((object_name, f"pos{axis}"))
        if optimize_z:
            values.append(pos[2])
            specs.append((object_name, "pos2"))
        values.append(euler[2])
        specs.append((object_name, "yaw"))
    return np.asarray(values, dtype=np.float64), specs


def _poses_from_params(
    base_poses: dict[str, dict[str, np.ndarray]],
    params: np.ndarray,
    specs: list[tuple[str, str]],
) -> dict[str, dict[str, np.ndarray]]:
    poses = {
        name: {
            "kind": pose["kind"],
            "pos": np.asarray(pose["pos"], dtype=np.float64).copy(),
            "quat": quat_normalize(np.asarray(pose["quat"], dtype=np.float64).copy()),
        }
        for name, pose in base_poses.items()
    }
    eulers = {name: euler_xyz_from_quat(pose["quat"]) for name, pose in poses.items()}
    for value, (object_name, field) in zip(params, specs):
        if field.startswith("pos"):
            poses[object_name]["pos"][int(field[-1])] = float(value)
        elif field == "yaw":
            eulers[object_name][2] = float(value)
    for object_name, euler in eulers.items():
        poses[object_name]["quat"] = quat_from_euler_xyz(euler)
    return poses


def _render_sim_masks(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    seg_renderer: mujoco.Renderer,
    camera_cfg: dict,
    intrinsics: np.ndarray | None,
    geom_ids: dict[str, int],
    camera_adjust_delta: np.ndarray | None,
) -> dict[str, np.ndarray]:
    camera_name = camera_cfg["mujoco_cam"]
    set_mujoco_camera_from_config(data, model, camera_name, camera_cfg["config"])
    if camera_adjust_delta is not None:
        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        if cam_id >= 0:
            camera_pose = np.eye(4)
            camera_pose[:3, :3] = data.cam_xmat[cam_id].reshape(3, 3)
            camera_pose[:3, 3] = data.cam_xpos[cam_id]
            adjusted = camera_pose @ camera_adjust_delta
            data.cam_xpos[cam_id] = adjusted[:3, 3]
            data.cam_xmat[cam_id] = adjusted[:3, :3].flatten()
    seg_renderer.update_scene(data, camera=camera_name)
    seg = seg_renderer.render()[:, :, 0].astype(np.int32)
    if intrinsics is not None:
        seg = shift_for_principal_point(seg, intrinsics, seg=True)
    return {name: seg == geom_id for name, geom_id in geom_ids.items()}


def _mask_metrics(sim_mask: np.ndarray, target_info: dict[str, np.ndarray | float]) -> tuple[float, float]:
    target_mask = target_info["mask"]
    target_area = float(target_info["area"])
    sim_area = float(sim_mask.sum())
    if target_area <= 0.0 or sim_area <= 0.0:
        return 0.0, 10.0
    inter = float(np.logical_and(sim_mask, target_mask).sum())
    union = float(np.logical_or(sim_mask, target_mask).sum())
    iou = inter / max(union, 1.0)
    target_dt = target_info["dt"]
    target_centroid = target_info["centroid"]
    sim_dt = cv2.distanceTransform((~sim_mask).astype(np.uint8), cv2.DIST_L2, 3)
    dist_to_target = float(np.mean(target_dt[sim_mask]))
    dist_to_sim = float(np.mean(sim_dt[target_mask]))
    ys, xs = np.where(sim_mask)
    sim_centroid = np.array([xs.mean(), ys.mean()], dtype=np.float64)
    centroid_px = float(np.linalg.norm(sim_centroid - target_centroid))
    loss = (1.0 - iou) * 2.0 + (dist_to_target + dist_to_sim) / 80.0 + centroid_px / 220.0
    return iou, loss


def _coordinate_search(
    evaluate,
    start: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    yaw_indices: set[int],
) -> tuple[np.ndarray, float]:
    params = np.minimum(np.maximum(start.copy(), lower), upper)
    best_loss, _ = evaluate(params)
    xy_steps = [0.04, 0.02, 0.01, 0.005, 0.0025]
    z_steps = [0.01, 0.005, 0.0025]
    yaw_steps = [np.deg2rad(v) for v in (30.0, 15.0, 7.5, 3.0, 1.5)]
    for level in range(len(xy_steps)):
        improved = True
        while improved:
            improved = False
            for idx in range(len(params)):
                if idx in yaw_indices:
                    step = yaw_steps[level]
                elif upper[idx] - lower[idx] <= 2.0 * 0.031:
                    step = z_steps[min(level, len(z_steps) - 1)]
                else:
                    step = xy_steps[level]
                for sign in (1.0, -1.0):
                    trial = params.copy()
                    trial[idx] += sign * step
                    trial = np.minimum(np.maximum(trial, lower), upper)
                    loss, _ = evaluate(trial)
                    if loss + 1e-6 < best_loss:
                        params = trial
                        best_loss = loss
                        improved = True
                        break
                if improved:
                    break
    return params, best_loss


def save_result(result: ObjectPoseAlignResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "episode_idx": np.array(result.episode_idx, dtype=np.int64),
        "object_names": np.array(result.object_names),
        "loss": np.array(result.loss, dtype=np.float64),
    }
    for object_name, iou in result.iou_by_object.items():
        arrays[f"{object_name}__iou"] = np.array(iou, dtype=np.float64)
    for object_name, pose in result.poses.items():
        arrays[f"{object_name}__kind"] = pose["kind"]
        arrays[f"{object_name}__pos"] = np.asarray(pose["pos"], dtype=np.float64)
        arrays[f"{object_name}__quat"] = quat_normalize(np.asarray(pose["quat"], dtype=np.float64))
    np.savez_compressed(path, **arrays)


def load_result(path: str | Path) -> ObjectPoseAlignResult:
    path = Path(path)
    data = np.load(path, allow_pickle=False)
    object_names = [str(v) for v in data["object_names"]]
    poses = {}
    iou_by_object = {}
    for object_name in object_names:
        poses[object_name] = {
            "kind": np.array(str(data[f"{object_name}__kind"])),
            "pos": data[f"{object_name}__pos"].astype(np.float64),
            "quat": quat_normalize(data[f"{object_name}__quat"].astype(np.float64)),
        }
        iou_key = f"{object_name}__iou"
        if iou_key in data:
            iou_by_object[object_name] = float(data[iou_key])
    return ObjectPoseAlignResult(
        episode_idx=int(data["episode_idx"]),
        object_names=object_names,
        loss=float(data["loss"]),
        iou_by_object=iou_by_object,
        poses=poses,
        cache_path=path,
    )


def auto_align_object_poses(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    seg_renderer: mujoco.Renderer,
    camera_config: dict,
    config: ObjectPoseAlignConfig,
    episode_idx: int,
    apply: bool = True,
) -> ObjectPoseAlignResult:
    object_names = selection_object_names(config.object_name)
    cache_path = cache_path_for_episode(config, episode_idx)
    if cache_path.exists() and not config.force:
        result = load_result(cache_path)
        if apply:
            apply_object_poses(model, data, result.poses)
        return result

    if config.camera_key not in camera_config:
        raise KeyError(f"Unknown camera_key {config.camera_key!r}; available={sorted(camera_config)}")

    target_masks = load_target_masks(config.initial_states_dir, object_names, episode_idx)
    target_info = _target_precompute(target_masks)
    geom_ids = {name: _geom_id_for_object(model, name) for name in object_names}
    base_poses = capture_object_poses(model, data, object_names)
    start, specs = _pack_params(base_poses, object_names, config.optimize_z)

    lower = start.copy()
    upper = start.copy()
    yaw_indices = set()
    for idx, (_, field) in enumerate(specs):
        if field in ("pos0", "pos1"):
            lower[idx] -= config.xy_range_m
            upper[idx] += config.xy_range_m
        elif field == "pos2":
            lower[idx] -= config.z_range_m
            upper[idx] += config.z_range_m
        elif field == "yaw":
            lower[idx] -= config.yaw_range_rad
            upper[idx] += config.yaw_range_rad
            yaw_indices.add(idx)

    cam_cfg = camera_config[config.camera_key]
    intrinsics = cam_cfg["config"].get("intrinsics")

    def evaluate(params: np.ndarray) -> tuple[float, dict[str, float]]:
        poses = _poses_from_params(base_poses, params, specs)
        apply_object_poses(model, data, poses)
        sim_masks = _render_sim_masks(
            model,
            data,
            seg_renderer,
            cam_cfg,
            intrinsics,
            geom_ids,
            config.camera_adjust_delta,
        )
        total_loss = 0.0
        iou_by_object = {}
        for object_name in object_names:
            iou, loss = _mask_metrics(sim_masks[object_name], target_info[object_name])
            iou_by_object[object_name] = iou
            total_loss += loss
        return total_loss / max(len(object_names), 1), iou_by_object

    best_params, best_loss = _coordinate_search(evaluate, start, lower, upper, yaw_indices)
    best_poses = _poses_from_params(base_poses, best_params, specs)
    apply_object_poses(model, data, best_poses)
    _, best_iou = evaluate(best_params)

    result = ObjectPoseAlignResult(
        episode_idx=episode_idx,
        object_names=object_names,
        loss=best_loss,
        iou_by_object=best_iou,
        poses=best_poses,
        cache_path=cache_path,
    )
    save_result(result, cache_path)
    if apply:
        apply_object_poses(model, data, result.poses)
    return result
