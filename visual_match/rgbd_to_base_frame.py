"""
Simple script: capture RGBD from a RealSense camera, build point cloud in camera frame,
then transform to robot base frame using calibration loaded from configs/.
"""

import sys
from pathlib import Path
import numpy as np
import cv2
import open3d as o3d
import pyrealsense2 as rs

# Add project root and real_world for imports
root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root))
sys.path.insert(0, str(Path(__file__).parent))
from utils.pcd_utils import depth2fgpcd
from camera_config import load_camera_config

# === Load calibration from config ===
_cam_cfg = load_camera_config("stationary_cam")

R_cam2board = _cam_cfg["R_cam2board"]
t_cam2board = _cam_cfg["t_cam2board"]
R_world2base = _cam_cfg["R_world2base"]
t_world2base = _cam_cfg["t_world2base"]


def capture_rgbd(serial_number: str = None, resolution=(1280, 720)):
    """Capture one RGBD frame from the given camera."""
    if serial_number is None:
        serial_number = _cam_cfg["serial_number"]
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial_number)
    config.enable_stream(rs.stream.color, resolution[0], resolution[1], rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, resolution[0], resolution[1], rs.format.z16, 30)

    pipeline.start(config)
    align = rs.align(rs.stream.color)

    frames = pipeline.wait_for_frames()
    aligned = align.process(frames)
    color_frame = aligned.get_color_frame()
    depth_frame = aligned.get_depth_frame()

    color = np.asanyarray(color_frame.get_data())
    depth = np.asanyarray(depth_frame.get_data())

    profile = pipeline.get_active_profile()
    color_stream = profile.get_stream(rs.stream.color)
    intr = color_stream.as_video_stream_profile().get_intrinsics()
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    pipeline.stop()

    intr_matrix = np.eye(3)
    intr_matrix[0, 0] = intr.fx
    intr_matrix[1, 1] = intr.fy
    intr_matrix[0, 2] = intr.ppx
    intr_matrix[1, 2] = intr.ppy

    return color, depth, intr_matrix, depth_scale


def rgbd_to_pointcloud_base(color, depth, intr, depth_scale, depth_range=(0.2, 2.0)):
    """Build point cloud in camera frame, then transform to robot base frame."""
    depth_m = depth.astype(np.float32) * depth_scale

    points_cam = depth2fgpcd(depth_m, intr)
    points_cam = points_cam.reshape(depth.shape[0], depth.shape[1], 3)

    mask = (depth_m > depth_range[0]) & (depth_m < depth_range[1])
    points_cam = points_cam[mask]
    colors = color[mask][:, ::-1].astype(np.float64) / 255.0

    # Camera -> board (world)
    points_board = (R_cam2board @ points_cam.T).T + t_cam2board

    # Board (world) -> robot base
    points_base = (R_world2base @ points_board.T).T + t_world2base

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_base)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd


def main():
    serial = _cam_cfg["serial_number"]
    print(f"Capturing from camera {serial}...")
    color, depth, intr, depth_scale = capture_rgbd(serial)
    print("Got frame.")

    pcd = rgbd_to_pointcloud_base(color, depth, intr, depth_scale)

    # Visualize and save
    # o3d.io.write_point_cloud("pcd_robot_base.ply", pcd)
    # print("Saved pcd_robot_base.ply (point cloud in robot base frame)")

    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([pcd, frame])


if __name__ == "__main__":
    main()
