"""
Generate a ground-truth point cloud of the xArm7 robot in world frame.

Loads the MuJoCo xArm7 model, sets joint angles to match the real robot, runs forward kinematics,
extracts all mesh vertices in world frame, and saves as a PCD file.

Usage:
  # Sample more densely (default 5000 points per mesh):
  python tools/get_xarm_pointcloud.py \
      --points-per-mesh 10000 \
      --output pointclouds/output.pcd

"""

import argparse
import os
from pathlib import Path

import mujoco
import numpy as np

try:
    import open3d as o3d
except ImportError:
    raise ImportError("open3d is required: pip install open3d")


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = str(SCRIPT_DIR.parent / "xarm7" / "scene.xml")


def sample_mesh_surface(vertices: np.ndarray, faces: np.ndarray, num_points: int) -> np.ndarray:
    """
    Uniformly sample points on a triangle mesh surface.

    Args:
        vertices: (V, 3) mesh vertices.
        faces: (F, 3) triangle face indices.
        num_points: number of points to sample.

    Returns:
        (num_points, 3) sampled points.
    """
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    # Triangle areas for weighted sampling
    cross = np.cross(v1 - v0, v2 - v0)
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    total_area = areas.sum()
    if total_area < 1e-12:
        # Degenerate mesh — just return vertex positions
        idx = np.random.choice(len(vertices), size=min(num_points, len(vertices)), replace=True)
        return vertices[idx]

    probs = areas / total_area

    # Sample triangles proportional to area
    sampled_faces = np.random.choice(len(faces), size=num_points, p=probs)

    # Random barycentric coordinates
    r1 = np.random.rand(num_points, 1)
    r2 = np.random.rand(num_points, 1)
    sqrt_r1 = np.sqrt(r1)

    # Barycentric → Cartesian
    a = 1 - sqrt_r1
    b = sqrt_r1 * (1 - r2)
    c = sqrt_r1 * r2

    p0 = v0[sampled_faces]
    p1 = v1[sampled_faces]
    p2 = v2[sampled_faces]

    points = a * p0 + b * p1 + c * p2
    return points


def extract_robot_pointcloud(
    model_path: str,
    points_per_mesh: int = 5000,
) -> np.ndarray:
    """
    Load MuJoCo model, set joint angles, run FK, and extract mesh vertices in world frame.

    Args:
        model_path: Path to the MuJoCo XML model file.
        gripper_qpos: Optional (6,) array of gripper joint positions.
                      If None, defaults to zeros (gripper open).
        points_per_mesh: Number of points to sample per mesh geometry.

    Returns:
        (N, 3) array of 3D points in world frame.
    """
    print(f"Loading MuJoCo model: {model_path}")
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if key_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, key_id)
        print(f"Using 'home' keyframe: qpos = {data.qpos[:7]}")
    else:
        raise ValueError("'home' keyframe not found in the model.")

    # Run forward kinematics
    mujoco.mj_forward(model, data)

    # Collect points from all mesh geoms
    all_points = []
    geom_count = 0

    for geom_id in range(model.ngeom):
        geom_type = model.geom_type[geom_id]

        # Only process mesh geoms (type 7 in MuJoCo)
        if geom_type != mujoco.mjtGeom.mjGEOM_MESH:
            continue

        mesh_id = model.geom_dataid[geom_id]
        if mesh_id < 0:
            continue

        # Get mesh vertices
        vert_adr = model.mesh_vertadr[mesh_id]
        vert_num = model.mesh_vertnum[mesh_id]
        vertices = model.mesh_vert[vert_adr: vert_adr + vert_num].copy()  # (V, 3)

        # Get mesh faces
        face_adr = model.mesh_faceadr[mesh_id]
        face_num = model.mesh_facenum[mesh_id]
        faces = model.mesh_face[face_adr: face_adr + face_num].copy()  # (F, 3)

        # Sample points on the mesh surface
        if face_num > 0:
            sampled = sample_mesh_surface(vertices, faces, points_per_mesh)
        else:
            idx = np.random.choice(len(vertices), size=min(points_per_mesh, len(vertices)), replace=True)
            sampled = vertices[idx]

        # Transform to world frame: P_world = R @ P_local + t
        R = data.geom_xmat[geom_id].reshape(3, 3)
        t = data.geom_xpos[geom_id]
        points_world = (R @ sampled.T).T + t

        body_id = model.geom_bodyid[geom_id]
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or f"body_{body_id}"
        print(f"  Geom {geom_id} ({body_name}): {len(sampled)} points")

        all_points.append(points_world)
        geom_count += 1

    print(f"Processed {geom_count} mesh geoms")

    if len(all_points) == 0:
        raise RuntimeError("No mesh geoms found in the model.")

    return np.vstack(all_points)


# def read_xarm_joints(ip: str) -> tuple[np.ndarray, float]:
#     """
#     Read current joint angles from the real xArm.

#     Returns:
#         joint_angles: (7,) array in radians.
#         gripper_pos_mm: gripper position in mm.
#     """
#     from xarm.wrapper import XArmAPI

#     arm = XArmAPI(ip, is_radian=True)
#     arm.motion_enable(True)

#     code, angles = arm.get_servo_angle(is_radian=True)
#     if code != 0:
#         arm.disconnect()
#         raise RuntimeError(f"Failed to read xArm joint angles (code={code})")

#     joints = np.array(angles[:7])

#     code, gripper_pos = arm.get_gripper_position()
#     gripper_mm = gripper_pos if code == 0 and gripper_pos is not None else 800.0

#     print(f"xArm joints (rad): {joints}")
#     print(f"xArm joints (deg): {np.degrees(joints)}")
#     print(f"Gripper position: {gripper_mm:.1f} mm")

#     arm.disconnect()
#     return joints, gripper_mm


def gripper_mm_to_qpos(gripper_mm: float, open_mm: float = 800.0, close_mm: float = 0.0) -> np.ndarray:
    """
    Convert gripper mm position to MuJoCo qpos for the 6 gripper joints.

    The xArm gripper in MuJoCo is driven by a tendon that controls left/right driver joints.
    The driver joint range is [0, 0.85] rad where 0 = open, 0.85 = closed.
    """
    pct = (open_mm - gripper_mm) / (open_mm - close_mm)  # 0 = open, 1 = closed
    pct = max(0.0, min(1.0, pct))
    driver_angle = pct * 0.85

    # qpos order: left_driver, left_finger, left_inner_knuckle,
    #             right_driver, right_finger, right_inner_knuckle
    # The equality constraints couple them, but for FK we set the driver
    # and approximate the followers.
    return np.array([
        driver_angle,   # left_driver_joint
        driver_angle,   # left_finger_joint (coupled)
        driver_angle,   # left_inner_knuckle_joint (spring)
        driver_angle,   # right_driver_joint
        driver_angle,   # right_finger_joint (coupled)
        driver_angle,   # right_inner_knuckle_joint (spring)
    ])


def main():
    parser = argparse.ArgumentParser(
        description="Generate ground-truth point cloud of xArm7 in world frame from MuJoCo model."
    )
    parser.add_argument(
        "--output", type=str, default="pointclouds/output1.pcd",
        help="Output PCD file path.",
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL_PATH,
        help="Path to MuJoCo XML model (default: xarm7/scene.xml).",
    )
    parser.add_argument(
        "--points-per-mesh", type=int, default=5000,
        help="Points to sample per mesh geometry (default: 5000).",
    )

    # Joint angle source (pick one)
    source = parser.add_mutually_exclusive_group()
    parser.add_argument(
        "--gripper-mm", type=float, default=None,
        help="Gripper position in mm (800=open, 0=closed). "
             "Auto-read from xArm if --xarm-ip is used.",
    )
    parser.add_argument(
        "--visualize", action="store_true", default=True,
        help="Open an Open3D viewer after saving.",
    )

    args = parser.parse_args()

    gripper_qpos = None

    if args.gripper_mm is not None:
        gripper_qpos = gripper_mm_to_qpos(args.gripper_mm)

    # Generate point cloud
    points = extract_robot_pointcloud(
        model_path=args.model,
        points_per_mesh=args.points_per_mesh,
    )

    print(f"\nTotal points: {points.shape[0]}")
    print(f"Bounds: x=[{points[:, 0].min():.4f}, {points[:, 0].max():.4f}] "
          f"y=[{points[:, 1].min():.4f}, {points[:, 1].max():.4f}] "
          f"z=[{points[:, 2].min():.4f}, {points[:, 2].max():.4f}]")

    # Save as PCD
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    o3d.io.write_point_cloud(args.output, pcd, write_ascii=True)
    print(f"Saved to: {args.output}")

    if args.visualize:
        print("Opening viewer...")
        origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
        pcd.paint_uniform_color([0.7, 0.7, 0.7])
        o3d.visualization.draw_geometries([pcd, origin])


if __name__ == "__main__":
    main()
