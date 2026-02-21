# visualizeAligned.py
import numpy as np
import open3d as o3d
import sys
import copy

# Filenames: segmented point clouds for ICP alignment (cleaner alignment, less noise)
TARGET_FILE = "pointclouds/output.pcd"   # fixed: MuJoCo segmented robot
SOURCE_FILE = "pointclouds/xarm7_robot.ply"   # moving: Gaussian Splatting segmented robot (transformed to align)

ORIGINAL_TARGET_FILE = "pointclouds/output.pcd"   # original MuJoCo full scene (from camera render)

source_segmented = o3d.io.read_point_cloud(SOURCE_FILE)
target_segmented = o3d.io.read_point_cloud(TARGET_FILE)
original_target = o3d.io.read_point_cloud(ORIGINAL_TARGET_FILE)

ICP_TRANSFORM = "pointclouds/icp_transform.npy"
icp_result = np.load(ICP_TRANSFORM)
print("ICP result", icp_result)
source_transformed = source_segmented.transform(icp_result.copy())

# Set gaussian point cloud to white color for visibility on black background
# num_points = len(source_transformed.points)
# white_colors = np.ones((num_points, 3))  # White color [1, 1, 1] for all points
# source_transformed.colors = o3d.utility.Vector3dVector(white_colors)

# Visualize the aligned point clouds

vis = o3d.visualization.Visualizer()
vis.create_window(window_name="ICP Alignment (segmented) applied to ORIGINAL point clouds", width=1920, height=1080)
vis.add_geometry(source_transformed)
# vis.add_geometry(target)
vis.add_geometry(original_target)

# Set rendering options
render_option = vis.get_render_option()
render_option.point_size = 1.0
render_option.background_color = np.array([0.5 ,0.5, 0.5])  # Black background


vis.run()
vis.destroy_window()