#!/usr/bin/env python3
"""
Proper 3D Gaussian Splatting PLY to NPZ converter.
Reads all Gaussian properties from the PLY file.
"""
import numpy as np
from plyfile import PlyData
import argparse
from pathlib import Path


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def SH2RGB(sh):
    """Convert spherical harmonics DC component to RGB color."""
    # SH coefficient for DC is C0 = 0.28209479177387814
    C0 = 0.28209479177387814
    return sh * C0 + 0.5


def load_3dgs_ply(ply_path):
    """
    Load a 3D Gaussian Splatting PLY file with all properties.
    
    Returns dict with:
        means3D: (N, 3) - 3D positions
        rgb_colors: (N, 3) - RGB colors [0,1]
        opacities: (N, 1) - opacities [0,1]
        scales: (N, 3) - scales (already exp'd)
        rotations: (N, 4) - quaternions [w,x,y,z]
        sh_rest: (N, rest_dim, 3) - remaining SH coefficients (optional)
    """
    print(f"Loading 3DGS PLY file: {ply_path}")
    plydata = PlyData.read(ply_path)
    vertex = plydata['vertex']
    
    num_points = len(vertex['x'])
    print(f"Found {num_points} Gaussians")
    
    # Extract positions
    means3D = np.stack([
        vertex['x'],
        vertex['y'],
        vertex['z']
    ], axis=1).astype(np.float32)
    print(f"  means3D: shape={means3D.shape}, range=[{means3D.min():.3f}, {means3D.max():.3f}]")
    
    # Extract DC spherical harmonics (base color)
    f_dc_0 = vertex['f_dc_0']
    f_dc_1 = vertex['f_dc_1']
    f_dc_2 = vertex['f_dc_2']
    sh_dc = np.stack([f_dc_0, f_dc_1, f_dc_2], axis=1).astype(np.float32)
    
    # Convert SH DC to RGB
    rgb_colors = SH2RGB(sh_dc)
    rgb_colors = np.clip(rgb_colors, 0.0, 1.0)
    print(f"  rgb_colors: shape={rgb_colors.shape}, range=[{rgb_colors.min():.3f}, {rgb_colors.max():.3f}]")
    
    # Extract opacity (stored as logit, need sigmoid)
    opacity_raw = vertex['opacity'].astype(np.float32)
    opacities = sigmoid(opacity_raw)
    print(f"  opacities: range=[{opacities.min():.3f}, {opacities.max():.3f}]")
    
    # Extract scales (stored as log, need exp)
    scale_0 = vertex['scale_0']
    scale_1 = vertex['scale_1']
    scale_2 = vertex['scale_2']
    log_scales = np.stack([scale_0, scale_1, scale_2], axis=1).astype(np.float32)
    scales = np.exp(log_scales) * 0.01
    print(f"  scales: range=[{scales.min():.6f}, {scales.max():.3f}]")
    
    # Extract rotations (quaternions)
    rot_0 = vertex['rot_0']  # w
    rot_1 = vertex['rot_1']  # x
    rot_2 = vertex['rot_2']  # y
    rot_3 = vertex['rot_3']  # z
    rotations = np.stack([rot_0, rot_1, rot_2, rot_3], axis=1).astype(np.float32)
    # Normalize quaternions
    rotations = rotations / np.linalg.norm(rotations, axis=1, keepdims=True)
    print(f"  rotations: shape={rotations.shape}")
    
    # Count SH rest components
    sh_rest_names = [name for name in vertex.data.dtype.names if name.startswith('f_rest_')]
    num_rest = len(sh_rest_names)
    print(f"  SH rest components: {num_rest}")
    
    return {
        'means3D': means3D,
        'rgb_colors': rgb_colors,
        'opacities': opacities.reshape(-1, 1),
        'scales': scales,
        'log_scales': log_scales,
        'rotations': rotations,
        'logit_opacities': opacity_raw.reshape(-1, 1),
    }


def ply_to_npz(ply_path, output_path, org_width=640, org_height=480, intrinsics=None):
    """
    Convert 3DGS PLY to NPZ format for the Gaussian rasterizer.
    """
    # Load Gaussian data
    data = load_3dgs_ply(ply_path)
    
    # Create default camera intrinsics if not provided
    if intrinsics is None:
        fov_deg = 60.0
        f = (org_height / 2) / np.tan(np.radians(fov_deg / 2))
        intrinsics = np.array([
            [f, 0, org_width / 2],
            [0, f, org_height / 2],
            [0, 0, 1]
        ], dtype=np.float64)
    
    # Create default world-to-camera (identity)
    w2c = np.eye(4, dtype=np.float32)
    
    # Camera sequence data
    cam_unnorm_rots = np.zeros((4, 1), dtype=np.float32)
    cam_unnorm_rots[0, 0] = 1.0
    cam_trans = np.zeros((3, 1), dtype=np.float32)
    
    # Build NPZ dict
    params = {
        # Gaussian parameters
        'means3D': data['means3D'],
        'rgb_colors': data['rgb_colors'],
        'unnorm_rotations': data['rotations'],
        'logit_opacities': data['logit_opacities'],
        'log_scales': data['log_scales'],
        
        # Camera parameters
        'org_width': org_width,
        'org_height': org_height,
        'w2c': w2c,
        'intrinsics': np.vstack([intrinsics, np.zeros((1, 3))]),
        
        # Camera sequence
        'cam_unnorm_rots': cam_unnorm_rots,
        'cam_trans': cam_trans,
    }
    
    print(f"\nSaving to: {output_path}")
    np.savez_compressed(output_path, **params)
    print(f"✓ Saved {len(data['means3D'])} Gaussians to {output_path}")
    
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert 3DGS PLY to NPZ")
    parser.add_argument("--ply_path", type=str, help="Path to 3DGS PLY file")
    parser.add_argument("--output", "-o", type=str, default=None,
                       help="Output .npz path")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    
    args = parser.parse_args()
    
    output_path = args.output or str(Path(args.ply_path).with_suffix('.npz'))
    
    try:
        ply_to_npz(args.ply_path, output_path, args.width, args.height)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

