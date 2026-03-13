# visualizeAligned.py
# Supports manual tuning of the Gaussian Splatting pointcloud alignment
# Controls:
#   Arrow Keys: Translate in X/Y plane
#   W/S: Translate in Z axis
#   +/-: Adjust step size
#   R: Reset to ICP result
#   P: Print current transform
#   Ctrl+C or close window: Save updated transform and exit

import numpy as np
import open3d as o3d
import sys
import copy
import signal

# Filenames: segmented point clouds for ICP alignment (cleaner alignment, less noise)
TARGET_FILE = "pointclouds/output.pcd"   # fixed: MuJoCo segmented robot
SOURCE_FILE = "pointclouds/xarm7_robot_black.ply"   # moving: Gaussian Splatting segmented robot (transformed to align)

ORIGINAL_TARGET_FILE = "pointclouds/output.pcd"   # original MuJoCo full scene (from camera render)

ICP_TRANSFORM = "pointclouds/icp_transform.npy"

# Global state for manual adjustment
class AlignmentState:
    def __init__(self, icp_transform):
        self.icp_transform = icp_transform.copy()
        self.manual_transform = np.eye(4)  # Additional manual adjustment
        self.translation_step = 0.0025  # 5mm steps
        self.modified = False
        
    def get_combined_transform(self):
        """Combine ICP result with manual adjustments"""
        return self.manual_transform @ self.icp_transform
    
    def translate(self, axis, direction):
        """Translate along axis (0=X, 1=Y, 2=Z)"""
        delta = np.eye(4)
        delta[axis, 3] = direction * self.translation_step
        self.manual_transform = delta @ self.manual_transform
        self.modified = True
        
    def reset(self):
        """Reset manual adjustments"""
        self.manual_transform = np.eye(4)
        self.modified = False
        
    def save_transform(self):
        """Save the combined transform to file"""
        combined = self.get_combined_transform()
        np.save(ICP_TRANSFORM, combined)
        print(f"\n{'='*50}")
        print("SAVED updated transform to:", ICP_TRANSFORM)
        print("Combined transform matrix:")
        print(combined)
        print(f"{'='*50}")


def remove_outliers(pcd, nb_neighbors=20, std_ratio=2.0):
    """Remove statistical outliers from a point cloud."""
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    n_removed = len(pcd.points) - len(ind)
    if n_removed > 0:
        print(f"  Removed {n_removed} outlier points ({n_removed/len(pcd.points)*100:.1f}%)")
    return cl


def main():
    # Load point clouds
    source_segmented = o3d.io.read_point_cloud(SOURCE_FILE)
    target_segmented = o3d.io.read_point_cloud(TARGET_FILE)
    original_target = o3d.io.read_point_cloud(ORIGINAL_TARGET_FILE)

    # Remove outliers from source (GS point cloud often has stray points)
    print(f"Source points before filtering: {len(source_segmented.points)}")
    source_segmented = remove_outliers(source_segmented, nb_neighbors=20, std_ratio=2.0)
    print(f"Source points after filtering: {len(source_segmented.points)}")

    icp_result = np.load(ICP_TRANSFORM)
    print("Loaded ICP transform:")
    print(icp_result)
    
    # Initialize alignment state
    state = AlignmentState(icp_result)
    
    # Create transformed source (will be updated during manual tuning)
    source_transformed = copy.deepcopy(source_segmented)
    source_transformed.transform(state.get_combined_transform())

    # Create visualizer with key callbacks
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="Manual Alignment Tuning (Arrow keys, W/S | +/- step | R reset | P print)", 
                      width=1920, height=1080)
    vis.add_geometry(source_transformed)
    vis.add_geometry(original_target)

    # Set rendering options
    render_option = vis.get_render_option()
    render_option.point_size = 1.0
    render_option.background_color = np.array([0.5, 0.5, 0.5])

    # Focus camera on the target point cloud (not the potentially larger source)
    target_bbox = original_target.get_axis_aligned_bounding_box()
    vis.reset_view_point(True)
    ctr = vis.get_view_control()
    ctr.set_lookat(target_bbox.get_center())
    ctr.set_zoom(0.5)

    def update_visualization():
        """Update the source pointcloud with current transform"""
        nonlocal source_transformed
        # Reset to original and apply new transform
        source_transformed.points = copy.deepcopy(source_segmented.points)
        source_transformed.colors = copy.deepcopy(source_segmented.colors)
        source_transformed.transform(state.get_combined_transform())
        vis.update_geometry(source_transformed)
        vis.poll_events()
        vis.update_renderer()

    def print_status():
        print(f"\nStep size: {state.translation_step*1000:.1f}mm")
        print(f"Manual adjustment: modified={state.modified}")

    # Key callbacks
    def translate_left(vis):
        state.translate(0, -1)
        update_visualization()
        print("Translate X-")
        return False

    def translate_right(vis):
        state.translate(0, 1)
        update_visualization()
        print("Translate X+")
        return False

    def translate_forward(vis):
        state.translate(1, 1)
        update_visualization()
        print("Translate Y+")
        return False

    def translate_backward(vis):
        state.translate(1, -1)
        update_visualization()
        print("Translate Y-")
        return False

    def translate_up(vis):
        state.translate(2, 1)
        update_visualization()
        print("Translate Z+")
        return False

    def translate_down(vis):
        state.translate(2, -1)
        update_visualization()
        print("Translate Z-")
        return False

    def increase_step(vis):
        state.translation_step *= 2
        print_status()
        return False

    def decrease_step(vis):
        state.translation_step /= 2
        print_status()
        return False

    def reset_transform(vis):
        state.reset()
        update_visualization()
        print("Reset to ICP result")
        return False

    def print_transform(vis):
        print("\nCurrent combined transform:")
        print(state.get_combined_transform())
        print("\nManual adjustment:")
        print(state.manual_transform)
        return False

    # Register key callbacks
    # Arrow keys: 262=Right, 263=Left, 264=Down, 265=Up
    vis.register_key_callback(262, translate_right)   # Right arrow
    vis.register_key_callback(263, translate_left)    # Left arrow
    vis.register_key_callback(265, translate_forward) # Up arrow
    vis.register_key_callback(264, translate_backward)# Down arrow
    
    # W/S for Z translation
    vis.register_key_callback(ord('W'), translate_up)
    vis.register_key_callback(ord('S'), translate_down)
    
    # +/- for step size (= is + without shift on US keyboard)
    vis.register_key_callback(ord('='), increase_step)
    vis.register_key_callback(ord('-'), decrease_step)
    vis.register_key_callback(ord('+'), increase_step)
    
    # R to reset, P to print
    vis.register_key_callback(ord('R'), reset_transform)
    vis.register_key_callback(ord('P'), print_transform)

    # Handle Ctrl+C
    def signal_handler(sig, frame):
        print("\nCtrl+C detected!")
        if state.modified:
            state.save_transform()
        else:
            print("No modifications made, transform not saved.")
        vis.destroy_window()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)

    print("\n" + "="*60)
    print("MANUAL ALIGNMENT TUNING")
    print("="*60)
    print("Controls:")
    print("  Arrow Keys    : Translate in X/Y plane")
    print("  W/S           : Translate Z up/down")
    print("  +/-           : Increase/decrease step size")
    print("  R             : Reset to ICP result")
    print("  P             : Print current transform")
    print("  Close window  : Save transform and exit (if modified)")
    print("="*60)
    print_status()
    print()

    # Run visualizer
    vis.run()
    
    # Save on normal exit (window close)
    if state.modified:
        state.save_transform()
    else:
        print("\nNo modifications made, transform not saved.")
    
    vis.destroy_window()


if __name__ == "__main__":
    main()