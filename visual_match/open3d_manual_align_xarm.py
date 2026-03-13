#!/usr/bin/env python3
"""
Open3D manual alignment tool for xArm calibration.

Loads:
- Real-world point cloud (default: /home/tina/Documents/residual_physics-main/point_cloud_world.pcd)
- MuJoCo xArm robot geometry at a target real-world state

Then opens an Open3D window so you can manually align the simulated robot cloud
to the real cloud with keyboard controls.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np
import open3d as o3d

from lerobot_mujoco_utils import lerobot_state_to_mujoco_ctrl
from load_model_xarm import load_model


DEFAULT_JOINTS_DEG = [112.3, -102.8, -91.0, 82.4, -22.0, 75.0, 160.3]
DEFAULT_GRIPPER_MM = 800.0


def _robot_geom_ids(model: mujoco.MjModel) -> list[int]:
    ids: list[int] = []
    keywords = ("link", "xarm", "gripper", "finger", "knuckle")
    for geom_id in range(model.ngeom):
        body_id = int(model.geom_bodyid[geom_id])
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
        if not any(k in body_name for k in keywords):
            continue
        geom_type = int(model.geom_type[geom_id])
        if geom_type in (mujoco.mjtGeom.mjGEOM_MESH, mujoco.mjtGeom.mjGEOM_BOX):
            ids.append(geom_id)
    return ids


def _mesh_vertices_world(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    geom_id: int,
) -> np.ndarray:
    geom_type = int(model.geom_type[geom_id])
    rot = data.geom_xmat[geom_id].reshape(3, 3)
    pos = data.geom_xpos[geom_id]

    if geom_type == mujoco.mjtGeom.mjGEOM_MESH:
        mesh_id = int(model.geom_dataid[geom_id])
        if mesh_id < 0:
            return np.empty((0, 3), dtype=np.float64)
        vadr = int(model.mesh_vertadr[mesh_id])
        vnum = int(model.mesh_vertnum[mesh_id])
        verts_local = model.mesh_vert[vadr : vadr + vnum]
        return (rot @ verts_local.T).T + pos

    if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
        sx, sy, sz = model.geom_size[geom_id]
        corners = np.array(
            [
                [-sx, -sy, -sz],
                [-sx, -sy, sz],
                [-sx, sy, -sz],
                [-sx, sy, sz],
                [sx, -sy, -sz],
                [sx, -sy, sz],
                [sx, sy, -sz],
                [sx, sy, sz],
            ],
            dtype=np.float64,
        )
        return (rot @ corners.T).T + pos

    return np.empty((0, 3), dtype=np.float64)


def build_robot_point_cloud(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    max_points: int,
) -> o3d.geometry.PointCloud:
    all_pts: list[np.ndarray] = []
    for geom_id in _robot_geom_ids(model):
        pts = _mesh_vertices_world(model, data, geom_id)
        if pts.size > 0:
            all_pts.append(pts)

    if not all_pts:
        raise RuntimeError("No robot geometry points extracted from MuJoCo model.")

    pts = np.concatenate(all_pts, axis=0)
    if pts.shape[0] > max_points:
        idx = np.random.choice(pts.shape[0], size=max_points, replace=False)
        pts = pts[idx]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    return pcd


def get_body_world_pos(model: mujoco.MjModel, data: mujoco.MjData, body_name: str) -> np.ndarray | None:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        return None
    return np.asarray(data.xpos[body_id], dtype=np.float64).copy()


def set_state_from_lerobot(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    joints_deg: list[float],
    gripper_mm: float,
    settle_steps: int,
) -> np.ndarray:
    if len(joints_deg) != 7:
        raise ValueError(f"Expected 7 joint values, got {len(joints_deg)}")

    gripper_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper")
    if gripper_act_id < 0:
        raise RuntimeError("Could not find gripper actuator named 'gripper'.")

    gripper_mj_range = (
        float(model.actuator_ctrlrange[gripper_act_id, 0]),
        float(model.actuator_ctrlrange[gripper_act_id, 1]),
    )

    state = np.asarray(list(joints_deg) + [gripper_mm], dtype=np.float64)
    ctrl = lerobot_state_to_mujoco_ctrl(state, gripper_mj_range)

    data.qpos[:7] = ctrl[:7]
    if model.nq > 7:
        data.qpos[7] = ctrl[7] / 255.0 * 0.85
    data.qvel[:] = 0
    data.ctrl[:8] = ctrl[:8]
    mujoco.mj_forward(model, data)

    for _ in range(max(settle_steps, 0)):
        data.ctrl[:8] = ctrl[:8]
        mujoco.mj_step(model, data)

    return ctrl


def transform_points(pts: np.ndarray, T: np.ndarray) -> np.ndarray:
    return (T[:3, :3] @ pts.T).T + T[:3, 3]


def make_transform(tx: float, ty: float, tz: float, rx: float, ry: float, rz: float) -> np.ndarray:
    rxr, ryr, rzr = np.deg2rad([rx, ry, rz])
    cx, sx = np.cos(rxr), np.sin(rxr)
    cy, sy = np.cos(ryr), np.sin(ryr)
    cz, sz = np.cos(rzr), np.sin(rzr)

    rot_x = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float64)
    rot_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
    rot_z = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float64)
    R = rot_z @ rot_y @ rot_x

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = [tx, ty, tz]
    return T


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Open3D manual alignment: world PCD + MuJoCo xArm")
    parser.add_argument(
        "--pcd",
        type=Path,
        default=Path("/home/tina/Documents/residual_physics-main/point_cloud_world.pcd"),
        help="Path to calibrated world PCD",
    )
    parser.add_argument(
        "--joints",
        type=float,
        nargs=7,
        default=DEFAULT_JOINTS_DEG,
        metavar=("J1", "J2", "J3", "J4", "J5", "J6", "J7"),
        help="xArm joint angles in degrees",
    )
    parser.add_argument("--gripper", type=float, default=DEFAULT_GRIPPER_MM, help="Gripper in mm (0..800)")
    parser.add_argument("--settle-steps", type=int, default=500, help="MuJoCo settle steps")
    parser.add_argument("--max-robot-points", type=int, default=80000, help="Max robot points for Open3D")
    parser.add_argument("--voxel", type=float, default=0.003, help="Voxel downsample size in meters")
    parser.add_argument("--trans-step", type=float, default=0.002, help="Translation key step in meters")
    parser.add_argument("--rot-step-deg", type=float, default=1.0, help="Rotation key step in degrees")
    parser.add_argument(
        "--robot-frame",
        type=str,
        default="world",
        choices=("world", "base"),
        help=(
            "Robot cloud frame before manual alignment. "
            "'world' uses MuJoCo world frame. "
            "'base' subtracts link_base pose so base origin is (0,0,0)."
        ),
    )
    parser.add_argument(
        "--initial-offset",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.0),
        metavar=("DX", "DY", "DZ"),
        help="Initial translation (meters) applied to sim robot before manual keys",
    )
    parser.add_argument(
        "--save-transform",
        type=Path,
        default=Path("visual_match/manual_align_transform.npy"),
        help="Output transform path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.pcd.exists():
        raise FileNotFoundError(f"PCD file not found: {args.pcd}")

    model, data, _ = load_model()
    ctrl = set_state_from_lerobot(model, data, list(args.joints), args.gripper, args.settle_steps)

    world_pcd = o3d.io.read_point_cloud(str(args.pcd))
    if world_pcd.is_empty():
        raise RuntimeError(f"Loaded empty PCD: {args.pcd}")

    robot_pcd = build_robot_point_cloud(model, data, max_points=args.max_robot_points)
    if args.voxel > 0:
        world_pcd = world_pcd.voxel_down_sample(args.voxel)
        robot_pcd = robot_pcd.voxel_down_sample(args.voxel)

    world_pcd.paint_uniform_color([0.75, 0.75, 0.75])
    robot_pcd.paint_uniform_color([0.95, 0.15, 0.1])

    link_base_pos = get_body_world_pos(model, data, "link_base")
    if link_base_pos is None:
        link_base_pos = np.zeros(3, dtype=np.float64)

    pts0 = np.asarray(robot_pcd.points).copy()
    if args.robot_frame == "base":
        pts0 = pts0 - link_base_pos

    initial_offset = np.asarray(args.initial_offset, dtype=np.float64)
    if np.linalg.norm(initial_offset) > 0:
        pts0 = pts0 + initial_offset
    robot_pcd.points = o3d.utility.Vector3dVector(pts0)

    base_robot_pts = np.asarray(robot_pcd.points).copy()
    T_user = np.eye(4, dtype=np.float64)
    trans_step = float(args.trans_step)
    rot_step_deg = float(args.rot_step_deg)

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="xArm Manual Alignment", width=1600, height=1000)
    vis.add_geometry(world_pcd)
    vis.add_geometry(robot_pcd)
    vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.12))

    def _refresh() -> None:
        pts = transform_points(base_robot_pts, T_user)
        robot_pcd.points = o3d.utility.Vector3dVector(pts)
        vis.update_geometry(robot_pcd)
        vis.poll_events()
        vis.update_renderer()

    def _apply(tx: float, ty: float, tz: float, rx: float, ry: float, rz: float):
        nonlocal T_user
        T_user = make_transform(tx, ty, tz, rx, ry, rz) @ T_user
        _refresh()

    def _print_state():
        print("\n[ALIGN] Current transform (sim_robot -> world_pcd):")
        np.set_printoptions(precision=6, suppress=True)
        print(T_user)
        print(f"[ALIGN] trans_step={trans_step:.6f} m, rot_step={rot_step_deg:.3f} deg")

    def _save():
        args.save_transform.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.save_transform, T_user)
        print(f"[ALIGN] Saved transform to: {args.save_transform}")

    def key_x_plus(_):
        _apply(trans_step, 0, 0, 0, 0, 0)
        return False

    def key_x_minus(_):
        _apply(-trans_step, 0, 0, 0, 0, 0)
        return False

    def key_y_plus(_):
        _apply(0, trans_step, 0, 0, 0, 0)
        return False

    def key_y_minus(_):
        _apply(0, -trans_step, 0, 0, 0, 0)
        return False

    def key_z_plus(_):
        _apply(0, 0, trans_step, 0, 0, 0)
        return False

    def key_z_minus(_):
        _apply(0, 0, -trans_step, 0, 0, 0)
        return False

    def key_rx_plus(_):
        _apply(0, 0, 0, rot_step_deg, 0, 0)
        return False

    def key_rx_minus(_):
        _apply(0, 0, 0, -rot_step_deg, 0, 0)
        return False

    def key_ry_plus(_):
        _apply(0, 0, 0, 0, rot_step_deg, 0)
        return False

    def key_ry_minus(_):
        _apply(0, 0, 0, 0, -rot_step_deg, 0)
        return False

    def key_rz_plus(_):
        _apply(0, 0, 0, 0, 0, rot_step_deg)
        return False

    def key_rz_minus(_):
        _apply(0, 0, 0, 0, 0, -rot_step_deg)
        return False

    def key_step_up(_):
        nonlocal trans_step
        trans_step *= 2.0
        print(f"[ALIGN] trans_step={trans_step:.6f} m")
        return False

    def key_step_down(_):
        nonlocal trans_step
        trans_step = max(trans_step / 2.0, 1e-5)
        print(f"[ALIGN] trans_step={trans_step:.6f} m")
        return False

    def key_rot_up(_):
        nonlocal rot_step_deg
        rot_step_deg *= 2.0
        print(f"[ALIGN] rot_step={rot_step_deg:.3f} deg")
        return False

    def key_rot_down(_):
        nonlocal rot_step_deg
        rot_step_deg = max(rot_step_deg / 2.0, 0.05)
        print(f"[ALIGN] rot_step={rot_step_deg:.3f} deg")
        return False

    vis.register_key_callback(ord("W"), key_x_plus)
    vis.register_key_callback(ord("X"), key_x_minus)
    vis.register_key_callback(ord("A"), key_y_minus)
    vis.register_key_callback(ord("D"), key_y_plus)
    vis.register_key_callback(ord("Q"), key_z_plus)
    vis.register_key_callback(ord("E"), key_z_minus)
    vis.register_key_callback(ord("I"), key_rx_plus)
    vis.register_key_callback(ord("K"), key_rx_minus)
    vis.register_key_callback(ord("J"), key_ry_minus)
    vis.register_key_callback(ord("L"), key_ry_plus)
    vis.register_key_callback(ord("U"), key_rz_minus)
    vis.register_key_callback(ord("O"), key_rz_plus)
    vis.register_key_callback(ord("R"), key_step_up)
    vis.register_key_callback(ord("F"), key_step_down)
    vis.register_key_callback(ord("T"), key_rot_up)
    vis.register_key_callback(ord("G"), key_rot_down)
    vis.register_key_callback(ord("P"), lambda _: (_save(), False)[1])
    vis.register_key_callback(ord("V"), lambda _: (_print_state(), False)[1])

    print("\n[INFO] Open3D manual alignment controls")
    print("  Translate: W/X (+/-X), A/D (-/+Y), Q/E (+/-Z)")
    print("  Rotate:    I/K (+/-Rx), J/L (-/+Ry), U/O (-/+Rz)")
    print("  Step size: R/F (translation up/down), T/G (rotation up/down)")
    print("  Output:    V print transform, P save transform")
    print(f"\n[INFO] Input joints (deg): {np.array(args.joints)}")
    print(f"[INFO] Input gripper (mm): {args.gripper}")
    print(f"[INFO] Applied MuJoCo ctrl: {np.array2string(ctrl, precision=4)}")
    print(f"[INFO] link_base world position: {np.array2string(link_base_pos, precision=4)}")
    print(f"[INFO] robot_frame mode: {args.robot_frame}")
    print(f"[INFO] initial_offset (m): {np.array2string(initial_offset, precision=4)}")
    print(f"[INFO] World PCD: {args.pcd}")
    print(f"[INFO] Save path: {args.save_transform}")

    _refresh()
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()
