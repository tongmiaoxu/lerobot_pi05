"""
Simple script: capture RGBD from camera 246322303954, build point cloud in camera frame,
then transform to robot base frame using provided calibration.
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

# === Calibration (camera 246322303954) ===
# Camera to board (from estimatePoseCharucoBoard: board pose in camera frame)
rvec_cam2board = np.array([[-0.62826474], [0.31336757], [0.69660772]])
tvec_cam2board = np.array([[-0.18496315], [-0.00416288], [0.69116146]])

R_board2cam = cv2.Rodrigues(rvec_cam2board)[0]
t_board2cam = tvec_cam2board.ravel()

# Camera to board (world = board frame in calibration)
R_cam2board = R_board2cam.T
t_cam2board = -R_cam2board @ t_board2cam

# Board to robot base (inverse of base2world)
R_base2world = np.array([
    [0.0318678, 0.99946283, -0.00764881],
    [0.99945614, -0.03180082, 0.00872485],
    [0.00847693, -0.00792269, -0.99993268],
])
t_base2world = np.array([0.10073883, -0.64318448, -0.06923716])

R_world2base = R_base2world.T
t_world2base = -R_world2base @ t_base2world


def capture_rgbd(serial_number: str = "246322303954", resolution=(1280, 720)):
    """Capture one RGBD frame from the given camera."""
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
    print("Capturing from camera 246322303954...")
    color, depth, intr, depth_scale = capture_rgbd("246322303954")
    print("Got frame.")

    pcd = rgbd_to_pointcloud_base(color, depth, intr, depth_scale)

    # Visualize and save
    # o3d.io.write_point_cloud("pcd_robot_base.ply", pcd)
    # print("Saved pcd_robot_base.ply (point cloud in robot base frame)")

    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([pcd, frame])


if __name__ == "__main__":
    main()
