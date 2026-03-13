#!/usr/bin/env python3
"""
Visualize alignment between MuJoCo foreground (robot + table + mug + sticker) and
transformed Gaussian Splatting point cloud (xarm7.ply).

Use this to verify if the table from MuJoCo aligns with the real-world table
in the Gaussian Splatting reconstruction.

Usage:
  python tools/visualize_table_alignment.py

Controls (when focused on Open3D window):
  Up/Down: adjust selected object's z
  1/2/3: select table / sticker / mug
  Escape: quit

Requires:
  - pointclouds/xarm7.ply (Gaussian Splatting full scene)
  - pointclouds/icp_transform.npy (ICP transform from GS to MuJoCo frame)
  - xarm7/scene.xml (MuJoCo model)
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np

try:
    import open3d as o3d
except ImportError:
    raise ImportError("open3d is required: pip install open3d")

try:
    import mujoco
except ImportError:
    raise ImportError("mujoco is required: pip install mujoco")

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "xarm7" / "scene.xml"
SCENE_XML_PATH = PROJECT_ROOT / "xarm7" / "scene.xml"
XARM7_XML_PATH = PROJECT_ROOT / "xarm7" / "xarm7.xml"
DEFAULT_MJ_PCD = PROJECT_ROOT / "pointclouds" / "mujoco_foreground.pcd"
DEFAULT_GS_PLY = PROJECT_ROOT / "pointclouds" / "xarm7.ply"
DEFAULT_ICP_TRANSFORM = PROJECT_ROOT / "pointclouds" / "icp_transform.npy"

Z_STEP = 0.005  # 5 mm per key press

# GLFW key codes (may conflict with Open3D defaults; use 1/2/3 and Escape)
KEY_UP = 265
KEY_DOWN = 264
KEY_1 = 49  # table
KEY_2 = 50  # sticker
KEY_3 = 51  # mug
KEY_ESCAPE = 256  # quit

# Geom names to include in MuJoCo foreground (robot meshes + table, mug, sticker)
def _sample_mesh_surface(vertices: np.ndarray, faces: np.ndarray, num_points: int) -> np.ndarray:
    """Uniformly sample points on a triangle mesh surface."""
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    total_area = areas.sum()
    if total_area < 1e-12:
        idx = np.random.choice(len(vertices), size=min(num_points, len(vertices)), replace=True)
        return vertices[idx]
    probs = areas / total_area
    sampled_faces = np.random.choice(len(faces), size=num_points, p=probs)
    r1, r2 = np.random.rand(num_points, 1), np.random.rand(num_points, 1)
    sqrt_r1 = np.sqrt(r1)
    a, b, c = 1 - sqrt_r1, sqrt_r1 * (1 - r2), sqrt_r1 * r2
    p0, p1, p2 = v0[sampled_faces], v1[sampled_faces], v2[sampled_faces]
    return (a * p0 + b * p1 + c * p2).astype(np.float64)


FOREGROUND_GEOM_NAMES = (
    "table", "mug", "sticker",
    # Robot body names (mesh geoms come from these)
    "link_base", "link1", "link2", "link3", "link4", "link5", "link6", "link7",
    "end_tool", "base_link",
    "left_outer_knuckle", "left_finger", "left_inner_knuckle",
    "right_outer_knuckle", "right_finger", "right_inner_knuckle",
)


def sample_box_surface(half_extents: np.ndarray, num_points: int) -> np.ndarray:
    """
    Uniformly sample points on the surface of a box (centered at origin).
    half_extents: (3,) = [sx, sy, sz] in meters.
    """
    sx, sy, sz = half_extents
    n = max(num_points // 6, 100)
    all_pts = []

    # 6 faces: (axis, sign) -> fixed coord = sign * half_extent[axis]
    for axis in range(3):
        for sign in (1, -1):
            u = np.random.uniform(-1, 1, (n,))
            v = np.random.uniform(-1, 1, (n,))
            pts = np.zeros((n, 3))
            pts[:, axis] = sign * half_extents[axis]
            other = [i for i in range(3) if i != axis]
            pts[:, other[0]] = u * half_extents[other[0]]
            pts[:, other[1]] = v * half_extents[other[1]]
            all_pts.append(pts)

    return np.vstack(all_pts)


def extract_foreground_pointcloud(
    model_path: Path,
    points_per_mesh: int = 5000,
    points_per_box: int = 2000,
    include_table: bool = True,
    verbose: bool = True,
) -> np.ndarray:
    """
    Extract point cloud from MuJoCo foreground: robot (meshes) + table + mug + sticker.

    Returns:
        (N, 3) points in world frame.
    """
    if verbose:
        print(f"Loading MuJoCo model: {model_path}")
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)

    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if key_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)

    all_points = []
    foreground_names = set(FOREGROUND_GEOM_NAMES)

    for geom_id in range(model.ngeom):
        geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        body_id = model.geom_bodyid[geom_id]
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""

        # Check if this geom belongs to foreground (by geom name or body name)
        if geom_name and geom_name in foreground_names:
            pass
        elif body_name and body_name in foreground_names:
            pass
        else:
            continue

        geom_type = model.geom_type[geom_id]
        R = data.geom_xmat[geom_id].reshape(3, 3)
        t = data.geom_xpos[geom_id]

        if geom_type == mujoco.mjtGeom.mjGEOM_MESH:
            mesh_id = model.geom_dataid[geom_id]
            if mesh_id < 0:
                continue
            vert_adr = model.mesh_vertadr[mesh_id]
            vert_num = model.mesh_vertnum[mesh_id]
            vertices = model.mesh_vert[vert_adr : vert_adr + vert_num].copy()
            face_adr = model.mesh_faceadr[mesh_id]
            face_num = model.mesh_facenum[mesh_id]
            faces = model.mesh_face[face_adr : face_adr + face_num].copy()

            if face_num > 0:
                sampled = _sample_mesh_surface(vertices, faces, points_per_mesh)
            else:
                idx = np.random.choice(len(vertices), size=min(points_per_mesh, len(vertices)), replace=True)
                sampled = vertices[idx]

            points_world = (R @ sampled.T).T + t
            if verbose:
                print(f"  Geom '{geom_name}' (mesh): {len(points_world)} points")
            all_points.append(points_world)

        elif geom_type == mujoco.mjtGeom.mjGEOM_BOX and include_table:
            # MuJoCo box: geom_size = half-extents (sx, sy, sz)
            half_extents = model.geom_size[geom_id].copy()
            sampled = sample_box_surface(half_extents, points_per_box)
            points_world = (R @ sampled.T).T + t
            if verbose:
                print(f"  Geom '{geom_name}' (box): {len(points_world)} points")
            all_points.append(points_world)

    if not all_points:
        raise RuntimeError("No foreground geoms found. Check FOREGROUND_GEOM_NAMES.")
    if verbose:
        print(f"  Total: {sum(len(p) for p in all_points)} points")

    return np.vstack(all_points)


def filter_gs_noise(pcd: o3d.geometry.PointCloud, max_abs: float = 1.0) -> o3d.geometry.PointCloud:
    """Remove points with |x|, |y|, or |z| > max_abs."""
    pts = np.asarray(pcd.points)
    mask = (np.abs(pts[:, 0]) <= max_abs) & (np.abs(pts[:, 1]) <= max_abs) & (np.abs(pts[:, 2]) <= max_abs)
    pcd_filtered = o3d.geometry.PointCloud()
    pcd_filtered.points = o3d.utility.Vector3dVector(pts[mask])
    n_removed = len(pts) - np.sum(mask)
    if n_removed > 0:
        print(f"  Removed {n_removed} points with |x|,|y|,|z| > {max_abs}")
    return pcd_filtered


def read_positions_from_scene() -> dict[str, float]:
    """Read table_z, sticker_z, mug_z from scene.xml."""
    text = SCENE_XML_PATH.read_text()
    out = {}
    m = re.search(r'<geom name="table"[^>]*pos="([^"]+)"', text)
    if m:
        parts = m.group(1).split()
        if len(parts) >= 3:
            out["table_z"] = float(parts[2])
    m = re.search(r'<body name="sticker"[^>]*pos="([^"]+)"', text)
    if m:
        parts = m.group(1).split()
        if len(parts) >= 3:
            out["sticker_z"] = float(parts[2])
    m = re.search(r'<body name="mug"[^>]*pos="([^"]+)"', text)
    if m:
        parts = m.group(1).split()
        if len(parts) >= 3:
            out["mug_z"] = float(parts[2])
    return out


def update_scene_xml_z(table_z: float | None, sticker_z: float | None, mug_z: float | None):
    """Update pos z in scene.xml for table (geom), sticker (body), mug (body)."""
    text = SCENE_XML_PATH.read_text()
    # Match pos="x y z" - numbers can be negative or decimal
    num = r"-?\d+\.?\d*"
    if table_z is not None:
        text = re.sub(
            rf'(<geom name="table"[^>]*pos=")({num})\s+({num})\s+{num}(")',
            lambda m: m.group(1) + m.group(2) + " " + m.group(3) + f" {table_z:.4f}" + m.group(4),
            text,
            count=1,
        )
    if sticker_z is not None:
        text = re.sub(
            rf'(<body name="sticker"[^>]*pos=")({num})\s+({num})\s+{num}(")',
            lambda m: m.group(1) + m.group(2) + " " + m.group(3) + f" {sticker_z:.4f}" + m.group(4),
            text,
            count=1,
        )
    if mug_z is not None:
        text = re.sub(
            rf'(<body name="mug"[^>]*pos=")({num})\s+({num})\s+{num}(")',
            lambda m: m.group(1) + m.group(2) + " " + m.group(3) + f" {mug_z:.4f}" + m.group(4),
            text,
            count=1,
        )
    SCENE_XML_PATH.write_text(text)


def update_xarm7_keyframe_mug_z(mug_z: float):
    """Update mug z in xarm7.xml home keyframe qpos (last 7 values = xyz + quat)."""
    text = XARM7_XML_PATH.read_text()
    # Match the active key line: last 7 values are mug x,y,z, qw,qx,qy,qz
    def repl(m):
        prefix, rest = m.group(1), m.group(2)
        parts = rest.split()
        if len(parts) >= 7:
            parts[-5] = f"{mug_z:.4f}"  # z is 3rd of last 7
        return prefix + " ".join(parts) + m.group(3)
    text = re.sub(
        r'(^    <key name="home" qpos=")([\d.\s-]+)(")',
        repl,
        text,
        count=1,
        flags=re.MULTILINE,
    )
    XARM7_XML_PATH.write_text(text)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize MuJoCo foreground + transformed GS point cloud for table alignment check."
    )
    parser.add_argument(
        "--model", type=Path, default=DEFAULT_MODEL_PATH,
        help="MuJoCo scene XML path.",
    )
    parser.add_argument(
        "--gs-ply", type=Path, default=DEFAULT_GS_PLY,
        help="Gaussian Splatting PLY file (default: pointclouds/xarm7.ply).",
    )
    parser.add_argument(
        "--icp-transform", type=Path, default=DEFAULT_ICP_TRANSFORM,
        help="ICP transform 4x4 matrix (default: pointclouds/icp_transform.npy).",
    )
    parser.add_argument(
        "--mujoco-pcd", type=Path, default=None,
        help="Output path for MuJoCo foreground PCD (optional, for caching).",
    )
    parser.add_argument(
        "--points-per-mesh", type=int, default=3000,
        help="Points per mesh geom.",
    )
    parser.add_argument(
        "--points-per-box", type=int, default=2000,
        help="Points per box geom (table, sticker).",
    )
    parser.add_argument(
        "--no-denoise", action="store_true",
        help="Disable GS noise filtering (remove points with |x|,|y|,|z| > 1).",
    )
    parser.add_argument(
        "--max-abs", type=float, default=1.0,
        help="Remove points with |x|, |y|, or |z| > this value (default: 1).",
    )
    args = parser.parse_args()

    # Resolve paths relative to project root
    model_path = args.model if args.model.is_absolute() else PROJECT_ROOT / args.model
    gs_ply = args.gs_ply if args.gs_ply.is_absolute() else PROJECT_ROOT / args.gs_ply
    icp_path = args.icp_transform if args.icp_transform.is_absolute() else PROJECT_ROOT / args.icp_transform

    if not model_path.exists():
        print(f"[ERROR] Model not found: {model_path}")
        sys.exit(1)
    if not gs_ply.exists():
        print(f"[ERROR] Gaussian Splatting PLY not found: {gs_ply}")
        print("  Export from 3DGS: pointclouds/xarm7.ply")
        sys.exit(1)
    if not icp_path.exists():
        print(f"[ERROR] ICP transform not found: {icp_path}")
        print("  Run: python tools/icp_register.py")
        sys.exit(1)

    # 1. Extract MuJoCo foreground (robot + table + mug + sticker)
    print("\n=== Extracting MuJoCo foreground point cloud (robot + table + mug + sticker) ===")
    mj_points = extract_foreground_pointcloud(
        model_path,
        points_per_mesh=args.points_per_mesh,
        points_per_box=args.points_per_box,
    )
    mj_pcd = o3d.geometry.PointCloud()
    mj_pcd.points = o3d.utility.Vector3dVector(mj_points)
    mj_pcd.paint_uniform_color([0.0, 0.8, 0.0])  # Green = MuJoCo

    if args.mujoco_pcd:
        out_path = args.mujoco_pcd if args.mujoco_pcd.is_absolute() else PROJECT_ROOT / args.mujoco_pcd
        out_path.parent.mkdir(parents=True, exist_ok=True)
        o3d.io.write_point_cloud(str(out_path), mj_pcd, write_ascii=True)
        print(f"Saved MuJoCo foreground: {out_path}")

    # 2. Load Gaussian Splatting PLY and apply ICP transform
    print("\n=== Loading Gaussian Splatting point cloud ===")
    try:
        gs_pcd = o3d.io.read_point_cloud(str(gs_ply))
    except Exception:
        # Fallback: 3DGS PLY format (vertex with x,y,z)
        try:
            from plyfile import PlyData
            plydata = PlyData.read(str(gs_ply))
            v = plydata["vertex"]
            xyz = np.column_stack([v["x"], v["y"], v["z"]]).astype(np.float64)
            gs_pcd = o3d.geometry.PointCloud()
            gs_pcd.points = o3d.utility.Vector3dVector(xyz)
        except Exception as e:
            print(f"[ERROR] Cannot load PLY: {e}")
            sys.exit(1)
    print(f"  GS points: {len(gs_pcd.points)}")

    icp_T = np.load(icp_path)
    print(f"  Applying ICP transform from {icp_path}")
    gs_transformed = gs_pcd.transform(icp_T)

    if not args.no_denoise:
        n_before = len(gs_transformed.points)
        gs_transformed = filter_gs_noise(gs_transformed, max_abs=args.max_abs)
        print(f"  GS points after denoise: {len(gs_transformed.points)} (was {n_before})")

    gs_transformed.paint_uniform_color([0.0, 0.0, 0.9])  # Blue = GS (transformed)

    # 3. Visualize with z-adjustment (Up/Down arrows, t/s/m to select)
    print("\n=== Visualizing ===")
    print("  Green: MuJoCo foreground (robot + table + mug + sticker)")
    print("  Blue:  Gaussian Splatting (xarm7.ply) transformed to MuJoCo frame")
    print("  Up/Down: adjust z | 1/2/3: select table/sticker/mug | Escape: quit")

    pos = read_positions_from_scene()
    table_z = pos.get("table_z", 0.045)
    sticker_z = pos.get("sticker_z", 0.046)
    mug_z = pos.get("mug_z", 0.1)
    current_obj = "sticker"

    origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="Table alignment: Up/Down=z, 1/2/3=select, Esc=quit", width=1920, height=1080)
    vis.add_geometry(mj_pcd)
    vis.add_geometry(gs_transformed)
    vis.add_geometry(origin)

    ro = vis.get_render_option()
    ro.point_size = 1.5
    ro.background_color = np.array([0.3, 0.3, 0.3])

    def refresh_mj_pcd():
        nonlocal mj_pcd
        pts = extract_foreground_pointcloud(
            model_path,
            points_per_mesh=args.points_per_mesh,
            points_per_box=args.points_per_box,
            verbose=False,
        )
        mj_pcd.points = o3d.utility.Vector3dVector(pts)
        mj_pcd.paint_uniform_color([0.0, 0.8, 0.0])
        vis.update_geometry(mj_pcd)

    def on_up(vis):
        nonlocal table_z, sticker_z, mug_z
        if current_obj == "table":
            table_z += Z_STEP
            update_scene_xml_z(table_z=table_z, sticker_z=None, mug_z=None)
        elif current_obj == "sticker":
            sticker_z += Z_STEP
            update_scene_xml_z(table_z=None, sticker_z=sticker_z, mug_z=None)
        else:
            mug_z += Z_STEP
            update_scene_xml_z(table_z=None, sticker_z=None, mug_z=mug_z)
            update_xarm7_keyframe_mug_z(mug_z)
        refresh_mj_pcd()
        z = table_z if current_obj == "table" else (sticker_z if current_obj == "sticker" else mug_z)
        print(f"  {current_obj} z = {z:.4f}")

    def on_down(vis):
        nonlocal table_z, sticker_z, mug_z
        if current_obj == "table":
            table_z -= Z_STEP
            update_scene_xml_z(table_z=table_z, sticker_z=None, mug_z=None)
        elif current_obj == "sticker":
            sticker_z -= Z_STEP
            update_scene_xml_z(table_z=None, sticker_z=sticker_z, mug_z=None)
        else:
            mug_z -= Z_STEP
            update_scene_xml_z(table_z=None, sticker_z=None, mug_z=mug_z)
            update_xarm7_keyframe_mug_z(mug_z)
        refresh_mj_pcd()
        z = table_z if current_obj == "table" else (sticker_z if current_obj == "sticker" else mug_z)
        print(f"  {current_obj} z = {z:.4f}")

    def on_1(vis):
        nonlocal current_obj
        current_obj = "table"
        print(f"  Selected: table (z={table_z:.4f})")

    def on_2(vis):
        nonlocal current_obj
        current_obj = "sticker"
        print(f"  Selected: sticker (z={sticker_z:.4f})")

    def on_3(vis):
        nonlocal current_obj
        current_obj = "mug"
        print(f"  Selected: mug (z={mug_z:.4f})")

    def on_escape(vis):
        vis.destroy_window()
        return False  # signal stop (some Open3D versions use this)

    vis.register_key_callback(KEY_UP, on_up)
    vis.register_key_callback(KEY_DOWN, on_down)
    vis.register_key_callback(KEY_1, on_1)
    vis.register_key_callback(KEY_2, on_2)
    vis.register_key_callback(KEY_3, on_3)
    vis.register_key_callback(KEY_ESCAPE, on_escape)

    vis.run()
    vis.destroy_window()
    print(f"Done. Final: table_z={table_z:.4f} sticker_z={sticker_z:.4f} mug_z={mug_z:.4f}")


if __name__ == "__main__":
    main()
