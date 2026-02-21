# icp_register.py
import numpy as np
import open3d as o3d
import sys
import copy

# Filenames: segmented point clouds for ICP alignment (cleaner alignment, less noise)
SEGMENTED_TARGET_FILE = "pointclouds/output.pcd"   # moving: MuJoCo segmented robot
SEGMENTED_SOURCE_FILE = "pointclouds/xarm7_robot.ply"   # fixed: Gaussian Splatting segmented robot

# Filenames: original/full point clouds for visualization (to verify alignment on full scene)
ORIGINAL_TARGET_FILE = "pointclouds/output.pcd"   # original MuJoCo full scene (from camera render)
ORIGINAL_SOURCE_FILE = "pointclouds/xarm7.ply"   # original Gaussian Splatting full scene (PLY file)

OUT_TRANSFORMED = "pointclouds/gs_to_mujoco.pcd"
OUT_TRANSFORM = "pointclouds/icp_transform.npy"   # saves 4x4 matrix

# Parameters (tweak if needed)
voxel_size = 0.01             # initial downsample voxel (meters) — change based on scale
max_correspondence_dist = voxel_size * 1.5
icp_iteration = 50

# Helper: print transform nicely
def print_transform(T):
    np.set_printoptions(precision=6, suppress=True)
    print("4x4 transformation:\n", T)

def preprocess(pcd, voxel):
    pcd_down = pcd.voxel_down_sample(voxel)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 2.0, max_nn=30))
    return pcd_down
import copy

def multi_scale_refine_icp(source, target, init_trans,
                           voxel_size, icp_iterations=50,
                           scales=[4.0]):
    """
    Multi-scale refinement:
      - scales: multipliers for voxel_size to create coarse->fine levels.
      - icp_iterations: max iterations for each final full-resolution ICP.
    Returns final icp_result (RegistrationResult).
    """
    # keep originals untouched
    src_orig = copy.deepcopy(source)
    tgt_orig = copy.deepcopy(target)

    current_trans = init_trans.copy()

    for i, s in enumerate(scales):
        vs = voxel_size * s
        # downsample + normals for this scale
        src_down = src_orig.voxel_down_sample(vs)
        tgt_down = tgt_orig.voxel_down_sample(vs)

        radius_normal = vs * 2.0
        src_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=50))
        tgt_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=50))

        # set correspondence threshold relative to scale (coarse -> large, fine -> small)
        max_corr = vs * 1.5

        # use point-to-plane where normals exist (for better surface fit)
        print(f"[multi-scale] level {i+1}/{len(scales)}: voxel={vs:.5f}, max_corr={max_corr:.5f}")

        # Run ICP on downsampled clouds to get improved transform
        icp_local = o3d.pipelines.registration.registration_icp(
            src_down, tgt_down, max_corr, current_trans,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=icp_iterations // len(scales))
        )

        print(f"  -> level {i+1} fitness: {icp_local.fitness:.6f}, rmse: {icp_local.inlier_rmse:.6f}")
        current_trans = icp_local.transformation

    # Final: ensure normals on full-res clouds (larger radius) for best point-to-plane
    full_radius_normal = voxel_size * 4.0
    src_orig.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=full_radius_normal, max_nn=80))
    tgt_orig.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=full_radius_normal, max_nn=80))

    final_max_corr = voxel_size * 0.8  # tighten for final refine (tune if needed)
    print(f"[final refine] full-res point-to-plane ICP with max_corr={final_max_corr:.5f}")
    final_icp = o3d.pipelines.registration.registration_icp(
        src_orig, tgt_orig, final_max_corr, current_trans,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=icp_iterations)
    )
    print(f"[final] fitness: {final_icp.fitness:.6f}, rmse: {final_icp.inlier_rmse:.6f}")
    return final_icp

# --- How to call (replace your existing single registration_icp(...) call with this) ---
# icp_result = multi_scale_refine_icp(source, target, init_trans, voxel_size, icp_iteration)

if __name__ == "__main__":
    # Step 1: Load SEGMENTED point clouds for ICP alignment (cleaner, less noise)
    print("=" * 60)
    print("Step 1: Loading SEGMENTED point clouds for ICP alignment...")
    print("=" * 60)
    source_segmented = o3d.io.read_point_cloud(SEGMENTED_SOURCE_FILE)
    target_segmented = o3d.io.read_point_cloud(SEGMENTED_TARGET_FILE)
    print(f"Segmented source (GS) points: {len(source_segmented.points)}")
    print(f"Segmented target (MuJoCo) points: {len(target_segmented.points)}")
    
    # Use segmented clouds for ICP
    source = source_segmented
    target = target_segmented

    # Auto-estimate voxel_size if user didn't set manually, based on bounding box
    bbox = target.get_axis_aligned_bounding_box()
    extents = bbox.get_extent()   # xyz size
    diag = np.linalg.norm(extents)
    if voxel_size is None:
        voxel_size = diag * 0.01
    print(f"Using voxel_size = {voxel_size}")

    source_down = preprocess(source, voxel_size)
    target_down = preprocess(target, voxel_size)

    # Optional: if you have a rough initial transform, set here:
    # init_trans = np.eye(4)
    # source.transform(init_trans)

    print("Running global ICP (point-to-point) as a coarse alignment using RANSAC (optional)...")
    # Compute FPFH for coarse global alignment (helps avoid local minima)
    source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        source_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*5, max_nn=100))
    target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        target_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*5, max_nn=100))

    result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        mutual_filter=True,
        max_correspondence_distance=voxel_size*2.0,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=4,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(voxel_size*2.0),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9)
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500))
    print("RANSAC result fitness, inlier_rmse:", result_ransac.fitness, result_ransac.inlier_rmse)
    init_trans = result_ransac.transformation
    print_transform(init_trans)

    print("Refining with ICP (point-to-plane)...")
    # icp_result = o3d.pipelines.registration.registration_icp(
    #     source, target, max_correspondence_dist,
    #     init_trans,
    #     o3d.pipelines.registration.TransformationEstimationPointToPlane(),
    #     o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=icp_iteration))
    # print("ICP fitness:", icp_result.fitness, "inlier_rmse:", icp_result.inlier_rmse)
    # # ICP fitness: 0.8007742833529079 inlier_rmse: 0.006431244331667684
    # print_transform(icp_result.transformation)
    icp_result = multi_scale_refine_icp(source, target, init_trans, voxel_size, icp_iteration)
    print_transform(icp_result.transformation)
    segmented_source_transformed = source.transform(icp_result.transformation.copy())
    # Save transformation matrix (computed from segmented clouds)
    np.save(OUT_TRANSFORM, icp_result.transformation)
    print(f"Saved transform matrix (from segmented clouds) -> {OUT_TRANSFORM}")
    
    # Step 2: Load ORIGINAL point clouds and apply transformation
    print("\n" + "=" * 60)
    print("Step 2: Loading ORIGINAL point clouds for visualization...")
    print("=" * 60)
    original_source = o3d.io.read_point_cloud(ORIGINAL_SOURCE_FILE)
    original_target = o3d.io.read_point_cloud(ORIGINAL_TARGET_FILE)
    print(f"Original source (GS) points: {len(original_source.points)}")
    print(f"Original target (MuJoCo) points: {len(original_target.points)}")
    
    # Apply the transformation from segmented ICP to original GS point cloud
    print("\nApplying transformation (from segmented ICP) to original GS point cloud...")
    original_source_transformed = original_source.transform(icp_result.transformation.copy())
    
    # Save transformed original point cloud
    o3d.io.write_point_cloud(OUT_TRANSFORMED, original_source_transformed, write_ascii=True)
    print(f"Saved transformed original GS -> {OUT_TRANSFORMED}")
    
    # ===== Visualize the aligned ORIGINAL point clouds =====
    print("\n" + "=" * 60)
    print("Visualizing aligned ORIGINAL point clouds...")
    print("=" * 60)
    print("  Blue: Transformed segmented GS (3DGS)")
    print("  Original: MuJoCo target (original colors)")
    
    # Color the point clouds for visualization
    segmented_source_transformed.paint_uniform_color([0.0, 0.0, 1.0])  # Blue for 3dgs
    target.paint_uniform_color([0.0, 1.0, 0.0])  # Green for mujoco
    # original_source_transformed.paint_uniform_color([1.0, 0.0, 0.0])  # red
    # original_target.paint_uniform_color([0.0, 0.0, 1.0])  # green
    # Create visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="ICP Alignment (segmented) applied to ORIGINAL point clouds", width=1920, height=1080)
    vis.add_geometry(segmented_source_transformed)
    # vis.add_geometry(target)
    # vis.add_geometry(original_source_transformed)
    vis.add_geometry(original_target)
    
    # Set rendering options
    render_option = vis.get_render_option()
    render_option.point_size = 1.0
    render_option.background_color = np.array([1.0, 1.0, 1.0])  # White background
    
    # Set initial zoom
    view_control = vis.get_view_control()
    view_control.set_zoom(0.8)
    
    print("Close the visualization window to finish...")
    vis.run()
    vis.destroy_window()
    
    print("\nDone! Transformation computed from segmented clouds, applied to original clouds.")
    
