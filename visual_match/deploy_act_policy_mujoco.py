#!/usr/bin/env python3
"""
Deploy ACT policy in MuJoCo ALOHA simulation.

This script loads an ACT policy checkpoint and runs it in MuJoCo simulation.
The policy uses the old data conversion (not PI05 normalization), matching
the training data format.

Usage:
    python visual_match/deploy_act_policy_mujoco.py \
        --policy-path outputs/train/act_pick_cuber/checkpoints/080000/pretrained_model \
        --prompt "Pick up the cube" \
        --fps 30
"""

import sys
import os
import re
import argparse
from pathlib import Path
import time
import json

# Add src to path for lerobot imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Auto-detect display
def _detect_display():
    if os.environ.get("DISPLAY"):
        return True
    if os.environ.get("WAYLAND_DISPLAY"):
        return True
    if sys.platform in ("darwin", "win32"):
        return True
    return False

_HAS_DISPLAY = _detect_display()
if not _HAS_DISPLAY:
    os.environ["MUJOCO_GL"] = "egl"
    print("[WARN] No display detected, using EGL rendering")

import numpy as np
import torch
import cv2
import mujoco
from mujoco import MjModel, MjData

if _HAS_DISPLAY:
    import mujoco.viewer

from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.control_utils import predict_action, prepare_observation_for_inference
from lerobot.utils.utils import get_safe_torch_device
from lerobot.utils.constants import OBS_STATE, OBS_IMAGES

# ============================================================================
# Import Gaussian Splatting helpers from compare_recorded_vs_mujoco.py
# ============================================================================
sys.path.insert(0, str(Path(__file__).parent))
from compare_recorded_vs_mujoco import (
    get_robot_geom_ids,
    load_scene_data,
    render_gaussian,
    get_mujoco_camera_pose,
    mj_pose_to_gaussian_w2c,
    get_camera_intrinsics_from_model,
    T_splat2mj,
)

# ============================================================================
# Color calibration functions
# ============================================================================

def _get_aug(x: np.ndarray, add_ones: bool = False) -> np.ndarray:
    """Augment input features for quadratic polynomial regression."""
    if add_ones:
        ones = np.ones((x.shape[0], 1), np.float64)
        return np.hstack([x ** 2, x, ones])
    return np.hstack([x ** 2, x])


def load_color_mapping(yaml_path: str):
    """
    Load color transform from color_mapping.yaml file.
    
    Returns:
        A: Transform matrix (3, 6) for [R², G², B², R, G, B] terms
        b: Bias vector (3,), constant term
    """
    with open(yaml_path, 'r') as f:
        content = f.read()
    
    # Parse color_A matrix (flattened 18 values)
    a_match = re.search(r'color_A:\s*\[(.*?)\]', content, re.DOTALL)
    if not a_match:
        raise ValueError(f"Could not find color_A in {yaml_path}")
    a_values = [float(x.strip()) for x in a_match.group(1).replace('\n', '').split(',')]
    A = np.array(a_values, dtype=np.float32).reshape(3, 6)  # 3 channels, 6 features
    
    # Parse color_b vector (3 values)
    b_match = re.search(r'color_b:\s*\[(.*?)\]', content, re.DOTALL)
    if not b_match:
        raise ValueError(f"Could not find color_b in {yaml_path}")
    b_values = [float(x.strip()) for x in b_match.group(1).replace('\n', '').split(',')]
    b = np.array(b_values, dtype=np.float32)
    
    return A, b


def apply_color_transform(img: np.ndarray, A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Apply quadratic color transform to image.
    
    Args:
        img: Input image (H, W, 3) in RGB format, uint8 [0-255]
        A: Transform matrix (3, 6) for [R², G², B², R, G, B] terms
        b: Bias vector (3,), constant term
    
    Returns:
        Transformed image (H, W, 3) in RGB format, uint8 [0-255]
    """
    # Flatten and normalize to [0, 1]
    flat = img.reshape(-1, 3).astype(np.float32) / 255.0
    
    # Apply quadratic transform: [R², G², B², R, G, B] @ A.T + b
    flat_aug = _get_aug(flat, add_ones=False)  # Shape: (N, 6)
    out = flat_aug @ A.T + b  # Shape: (N, 3)
    
    # Clip and convert back to uint8
    out = np.clip(out, 0.0, 1.0)
    out_rgb = (out.reshape(img.shape) * 255.0).astype(np.uint8)
    
    return out_rgb


def display_camera_images(observation: dict, policy_config=None, window_name_prefix: str = "Camera"):
    """
    Display camera images from observation dict in OpenCV windows.
    Only displays images that are actually used by the policy.
    
    Args:
        observation: Observation dict containing image keys
        policy_config: Policy configuration to determine which images are used
        window_name_prefix: Prefix for window names
    """
    # Find all image keys in observation
    all_image_keys = [k for k in observation.keys() if "image" in k.lower()]
    
    # Filter to only images actually used by the policy
    if policy_config is not None and hasattr(policy_config, 'image_features') and policy_config.image_features:
        # Only display images that the policy actually uses
        image_keys = [k for k in all_image_keys if k in policy_config.image_features]
    else:
        # No filtering, show all images
        image_keys = all_image_keys
    
    for img_key in image_keys:
        img = observation[img_key]
        
        # Convert from [0, 1] float32 to [0, 255] uint8 for display
        if img.dtype == np.float32 and img.max() <= 1.0:
            img_display = (img * 255).astype(np.uint8)
        else:
            img_display = img.astype(np.uint8)
        
        # Convert RGB to BGR for OpenCV
        img_bgr = cv2.cvtColor(img_display, cv2.COLOR_RGB2BGR)
        
        # Extract a nice window name from the key
        # e.g., "observation.images.cam_high" -> "cam_high"
        if "." in img_key:
            window_name = img_key.split(".")[-1]
        else:
            window_name = img_key
        
        window_full_name = f"{window_name_prefix}: {window_name}"
        
        # Display image
        cv2.imshow(window_full_name, img_bgr)
    
    # Wait 1ms to allow windows to update (non-blocking)
    cv2.waitKey(1)


def load_policy(policy_path: str) -> tuple[PreTrainedPolicy, dict]:
    """Load ACT policy from checkpoint path."""
    print(f"[INFO] Loading policy from: {policy_path}")
    
    # Resolve path
    policy_path_obj = Path(policy_path)
    if not policy_path_obj.is_absolute():
        # Try relative to current working directory first
        if not policy_path_obj.exists():
            # Try relative to project root
            project_root = Path(__file__).parent.parent
            policy_path_obj = project_root / policy_path
        if not policy_path_obj.exists():
            raise FileNotFoundError(f"Policy path not found: {policy_path}")
    
    policy_path = str(policy_path_obj.resolve())
    
    # Load policy config
    config_path = Path(policy_path) / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    # If config is missing 'type' field, we need to manually construct the config
    # and load the policy weights separately
    if "type" not in config_dict:
        print("[INFO] Config missing 'type' field, manually constructing ACTConfig")
        from lerobot.policies.act.configuration_act import ACTConfig
        from lerobot.configs.types import PolicyFeature, FeatureType, NormalizationMode
        
        # Convert old format (input_shapes, input_normalization_modes) to new format (input_features)
        config_dict_new = config_dict.copy()
        
        # Remove old format fields that aren't valid for ACTConfig
        old_fields = ["input_shapes", "input_normalization_modes", "output_shapes", "output_normalization_modes", "type"]
        for field in old_fields:
            config_dict_new.pop(field, None)
        
        # Convert input_shapes + input_normalization_modes to input_features
        if "input_shapes" in config_dict or "input_normalization_modes" in config_dict:
            input_shapes = config_dict.get("input_shapes", {})
            input_norm_modes = config_dict.get("input_normalization_modes", {})
            
            input_features = {}
            for key, shape in input_shapes.items():
                # Determine feature type based on key
                if "image" in key.lower() or "images" in key.lower():
                    feature_type = FeatureType.VISUAL
                elif "state" in key.lower():
                    feature_type = FeatureType.STATE
                elif "environment_state" in key.lower():
                    feature_type = FeatureType.ENV
                else:
                    feature_type = FeatureType.STATE  # Default
                
                input_features[key] = PolicyFeature(
                    type=feature_type,
                    shape=tuple(shape)
                )
            
            config_dict_new["input_features"] = input_features
        
        # Convert output_shapes + output_normalization_modes to output_features
        if "output_shapes" in config_dict or "output_normalization_modes" in config_dict:
            output_shapes = config_dict.get("output_shapes", {})
            output_norm_modes = config_dict.get("output_normalization_modes", {})
            
            output_features = {}
            for key, shape in output_shapes.items():
                output_features[key] = PolicyFeature(
                    type=FeatureType.ACTION,
                    shape=tuple(shape)
                )
            
            config_dict_new["output_features"] = output_features
        
        # Construct ACTConfig from converted dict
        config = ACTConfig(**config_dict_new)
        config.pretrained_path = Path(policy_path)
        
        # Get policy class
        policy_class = get_policy_class("act")
        
        # Create policy instance with config
        policy = policy_class(config)
        
        # Load model weights using the same method as from_pretrained
        from lerobot.utils.utils import get_safe_torch_device
        from lerobot.policies.pretrained import SAFETENSORS_SINGLE_FILE
        
        model_file = Path(policy_path) / SAFETENSORS_SINGLE_FILE
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_file}")
        
        device_obj = get_safe_torch_device(config.device)
        
        # Load to CPU first (safest approach, works with all safetensors versions)
        # Then move to target device afterwards
        policy = policy_class._load_as_safetensor(policy, str(model_file), "cpu", strict=False)
        policy = policy.to(device_obj)
        
        print(f"[INFO] Policy loaded: act (manual loading)")
    else:
        policy_type = config_dict.get("type", "act")
        if policy_type != "act":
            print(f"[WARN] Policy type is {policy_type}, expected 'act'")
        
        # Get policy class
        policy_class = get_policy_class(policy_type)
        
        # Load policy using from_pretrained (standard path)
        policy = policy_class.from_pretrained(policy_path)
        print(f"[INFO] Policy loaded: {policy_type}")
    
    policy.eval()
    print(f"[INFO] Policy config: device={policy.config.device}, use_amp={policy.config.use_amp}")
    
    return policy, config_dict


def build_observation_from_mujoco(model: MjModel, data: MjData, renderer: mujoco.Renderer, 
                                  calib_dir: Path, gripper_ctrl_range: tuple,
                                  camera_name: str = "teleoperator_pov",
                                  seg_renderer: mujoco.Renderer = None,
                                  robot_geom_ids: set = None,
                                  gaussian_data: dict = None) -> dict:
    """
    Build observation dictionary from MuJoCo state with optional Gaussian Splatting composite.
    
    Returns observation in format expected by policy:
    {
        "observation.state": np.ndarray,  # (14,) joint positions
        "observation.images.cam_high": np.ndarray,  # (H, W, 3) RGB image (composite if Gaussian available)
        "observation.images.cam_low": np.ndarray,
        "observation.images.cam_left_wrist": np.ndarray,  # (composite if Gaussian available)
        "observation.images.cam_right_wrist": np.ndarray,  # (composite if Gaussian available)
    }
    
    Args:
        gaussian_data: Dict with 'scene_data', 'scene_depth_data', 'intrinsics_cache', 'viz_cfg' for composite rendering
    """
    # Convert MuJoCo state (radians) to lerobot format (normalized degrees)
    state = convert_mujoco_state_to_lerobot(data, calib_dir, gripper_ctrl_range)
    
    # Build observation dict
    observation = {
        OBS_STATE: state,
    }
    
    # Camera mapping: observation key -> MuJoCo camera name
    camera_names = {
        "observation.images.cam_high": "teleoperator_pov",
        "observation.images.cam_low": "depth_cam",
        "observation.images.cam_left_wrist": "wrist_cam_left",
        "observation.images.cam_right_wrist": "wrist_cam_right",
    }
    
    # Check if Gaussian composite rendering is available
    use_composite = (gaussian_data is not None and 
                     gaussian_data.get('scene_data') is not None and
                     seg_renderer is not None and
                     robot_geom_ids is not None)
    
    # Wrist cameras should use composite rendering (if available)
    wrist_cameras = {"wrist_cam_left", "wrist_cam_right"}
    
    # Render each camera view
    for obs_key, cam_name_to_use in camera_names.items():
        try:
            # Decide if we should use composite rendering for this camera
            should_composite = use_composite and cam_name_to_use in wrist_cameras
            
            if should_composite:
                # Render composite (Gaussian background + MuJoCo foreground)
                rgb_image = render_composite_view(
                    model, data, renderer, seg_renderer, robot_geom_ids,
                    cam_name_to_use, gaussian_data
                )
            else:
                # Render MuJoCo only
                renderer.update_scene(data, camera=cam_name_to_use)
                rgb_image = renderer.render()
            
            # Convert to [0, 1] range
            rgb_image = rgb_image.astype(np.float32) / 255.0
            observation[obs_key] = rgb_image
            
        except Exception as e:
            # Fallback to default camera on error
            print(f"[WARN] Camera '{cam_name_to_use}' rendering failed: {e}, using '{camera_name}' instead")
            renderer.update_scene(data, camera=camera_name)
            rgb_image = renderer.render()
            rgb_image = rgb_image.astype(np.float32) / 255.0
            observation[obs_key] = rgb_image
    
    return observation


def render_composite_view(model: MjModel, data: MjData, 
                         renderer: mujoco.Renderer, seg_renderer: mujoco.Renderer,
                         robot_geom_ids: set, cam_name: str, gaussian_data: dict) -> np.ndarray:
    """
    Render composite view: Gaussian Splatting background + MuJoCo robot foreground.
    Optionally applies color calibration if available.
    
    Returns:
        RGB image as uint8 numpy array (H, W, 3)
    """
    # Render MuJoCo foreground
    renderer.update_scene(data, camera=cam_name)
    fg_rgb = renderer.render()
    
    # Get segmentation mask for robot
    seg_renderer.update_scene(data, camera=cam_name)
    seg_mask = seg_renderer.render()
    seg_labels = seg_mask[:, :, 0].astype(np.int32)
    seg_labels[seg_labels == -1] = 0
    robot_mask = np.isin(seg_labels, list(robot_geom_ids))
    mask_uint8 = (robot_mask.astype(np.uint8)) * 255
    
    # Render Gaussian background
    try:
        # Get camera pose and intrinsics
        camera_pose = get_mujoco_camera_pose(model, data, cam_name)
        w2c = mj_pose_to_gaussian_w2c(camera_pose, T_splat2mj)
        
        # Get or compute intrinsics for this camera
        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
        intrinsics_cache = gaussian_data['intrinsics_cache']
        if cam_name not in intrinsics_cache:
            viz_cfg = gaussian_data['viz_cfg']
            k = get_camera_intrinsics_from_model(model, cam_id, viz_cfg['viz_w'], viz_cfg['viz_h'])
            intrinsics_cache[cam_name] = k
        else:
            k = intrinsics_cache[cam_name]
        
        # Render Gaussian background
        bg_im = render_gaussian(w2c, k, gaussian_data['scene_data'], 
                               gaussian_data['scene_depth_data'], gaussian_data['viz_cfg'])
        bg_np = bg_im.permute(1, 2, 0).cpu().numpy()
        bg_np = (bg_np * 255).astype(np.uint8)
        
        # Composite: background + foreground where mask is True
        composite = bg_np.copy()
        composite[mask_uint8 > 0] = fg_rgb[mask_uint8 > 0]
        
        # Apply color calibration if available
        if 'color_calib' in gaussian_data and gaussian_data['color_calib'] is not None:
            color_A, color_b = gaussian_data['color_calib']
            composite = apply_color_transform(composite, color_A, color_b)
        
        return composite
        
    except Exception as e:
        # Fallback to MuJoCo only on Gaussian rendering error
        print(f"[WARN] Gaussian rendering failed for {cam_name}: {e}, using MuJoCo only")
        return fg_rgb


def convert_action_to_mujoco(action: torch.Tensor, mujoco_keyframe_ctrl: np.ndarray,
                             gripper_ctrl_range: tuple, calib_dir: Path, use_new_normalization: bool = False) -> np.ndarray:
    """
    Convert policy action (18 dims) to MuJoCo control (14 dims).
    
    Uses convert_actions_to_mujoco_absolute from run_prerecorded_traj_mujoco.py
    to match the conversion used during training with old data format.
    """
    # Import the conversion function from run_prerecorded_traj_mujoco
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from run_prerecorded_traj_mujoco import convert_actions_to_mujoco_pi05,convert_actions_to_mujoco_absolute
    
    # Convert to numpy and reshape for the conversion function
    action_np = action.cpu().numpy()
    if action_np.ndim > 1:
        action_np = action_np[0]  # Remove batch dimension if present
    
    # Reshape to (1, 18) for the conversion function (expects batch dimension)
    actions_raw = action_np.reshape(1, -1)
    
    # Use the same conversion function as run_prerecorded_traj_mujoco.py
    if use_new_normalization:
        ctrl_sequence = convert_actions_to_mujoco_pi05(
            actions_raw, 
            mujoco_keyframe_ctrl, 
            gripper_ctrl_range
        )
    else:
        ctrl_sequence = convert_actions_to_mujoco_absolute(
            actions_raw, 
            mujoco_keyframe_ctrl, 
            gripper_ctrl_range
        )
    # Return first frame (remove batch dimension)
    return ctrl_sequence[0]


def convert_mujoco_state_to_lerobot(data: MjData, calib_dir: Path, gripper_ctrl_range: tuple) -> np.ndarray:
    """
    Convert MuJoCo state (qpos in radians) to lerobot normalized format (degrees).
    
    This is the inverse of the action conversion - converts MuJoCo observations
    back to lerobot format that the policy expects.
    
    Returns 18-dim state: [left_arm(9), right_arm(9)]
    """
    # Load calibration files
    try:
        with open(calib_dir / "right_follower.json") as f:
            right_calib = json.load(f)
        with open(calib_dir / "left_follower.json") as f:
            left_calib = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Calibration files not found: {calib_dir}")
    
    def radians_to_raw_encoder(mujoco_rad: float, resolution: int = 4096) -> float:
        """Convert MuJoCo radians to raw encoder value."""
        return (mujoco_rad * 4096 / (2 * np.pi)) + 2048
    
    def raw_to_calibrated_degrees(raw: float, homing_offset: int, drive_mode: int, resolution: int = 4096) -> float:
        """Convert raw encoder value to lerobot calibrated degrees."""
        if drive_mode:
            raw *= -1
        raw += homing_offset
        degrees = raw * 180 / (resolution // 2)
        return degrees
    
    gripper_min, gripper_max = gripper_ctrl_range
    gripper_range = gripper_max - gripper_min
    
    # Calculate slope and intercept for gripper conversion (inverse of action conversion)
    LEROBOT_OPEN_PCT = 140.0
    LEROBOT_CLOSED_PCT = 0.0
    RIGHT_GRIPPER_SLOPE = (LEROBOT_CLOSED_PCT - LEROBOT_OPEN_PCT) / gripper_range
    RIGHT_GRIPPER_INTERCEPT = LEROBOT_OPEN_PCT - RIGHT_GRIPPER_SLOPE * gripper_min
    
    def gripper_interbotix_to_lerobot(mujoco_rad: float, calib: dict, arm_side: str = "right") -> float:
        """Convert gripper from Interbotix radians to lerobot percentage."""
        if "motor_names" in calib:
            gripper_idx = calib["motor_names"].index("gripper")
            drive_mode = calib["drive_mode"][gripper_idx]
        else:
            gripper_calib = calib.get("gripper", {})
            drive_mode = gripper_calib.get("drive_mode", 0)
        
        # Inverse of gripper_lerobot_to_interbotix
        if arm_side == "right":
            lerobot_pct = RIGHT_GRIPPER_SLOPE * mujoco_rad + RIGHT_GRIPPER_INTERCEPT
        else:
            lerobot_pct = RIGHT_GRIPPER_SLOPE * mujoco_rad + RIGHT_GRIPPER_INTERCEPT
        
        return np.clip(lerobot_pct, 0.0, 140.0)
    
    # MuJoCo qpos structure: [right_arm(8), left_arm(8)]
    # qpos[0:7] = right arm (waist, shoulder, elbow, forearm_roll, wrist_angle, wrist_rotate, left_finger)
    # qpos[7] = right/right_finger (excluded, coupled via equality constraint)
    # qpos[8:14] = left arm (waist, shoulder, elbow, forearm_roll, wrist_angle, wrist_rotate, left_finger)
    # qpos[15] = left/right_finger (excluded, coupled via equality constraint)
    # Convert to lerobot format: [left_arm(9), right_arm(9)] with shadow joints
    state = np.zeros((18,), dtype=np.float32)
    
    # Mapping: (mujoco_qpos_idx, lerobot_state_idx, calib, calib_joint_idx, arm_side)
    # Left arm: qpos[8:14] → state[0:7] (with shadow joints, excluding right_finger at qpos[15])
    left_mapping = [
        (8, 0, left_calib, 0, "left"),   # waist
        (9, 1, left_calib, 1, "left"),   # shoulder
        (9, 2, left_calib, 1, "left"),   # shoulder_shadow (duplicate)
        (10, 3, left_calib, 3, "left"),   # elbow
        (10, 4, left_calib, 3, "left"),   # elbow_shadow (duplicate)
        (11, 5, left_calib, 5, "left"),   # forearm_roll
        (12, 6, left_calib, 6, "left"),   # wrist_angle
        (13, 7, left_calib, 7, "left"),   # wrist_rotate
    ]
    
    # Right arm: qpos[0:6] → state[9:16] (with shadow joints, excluding right_finger at qpos[7])
    right_mapping = [
        (0, 9, right_calib, 0, "right"),   # waist
        (1, 10, right_calib, 1, "right"),  # shoulder
        (1, 11, right_calib, 1, "right"),  # shoulder_shadow (duplicate)
        (2, 12, right_calib, 3, "right"),  # elbow
        (2, 13, right_calib, 3, "right"),  # elbow_shadow (duplicate)
        (3, 14, right_calib, 5, "right"),  # forearm_roll
        (4, 15, right_calib, 6, "right"),  # wrist_angle
        (5, 16, right_calib, 7, "right"),  # wrist_rotate
    ]
    
    # Convert joints (skip grippers for now)
    for mujoco_idx, lerobot_idx, calib, calib_idx, arm_side in left_mapping + right_mapping:
        if len(data.qpos) > mujoco_idx:
            mujoco_rad = data.qpos[mujoco_idx]
            joint_name = calib["motor_names"][calib_idx]
            
            # Convert radians → raw → calibrated degrees
            raw = radians_to_raw_encoder(mujoco_rad)
            homing_offset = calib["homing_offset"][calib_idx]
            drive_mode = calib["drive_mode"][calib_idx]
            degrees = raw_to_calibrated_degrees(raw, homing_offset, drive_mode)
            state[lerobot_idx] = degrees
    
    # Convert grippers: qpos[14] = left/left_finger, qpos[6] = right/left_finger
 
    # Left gripper (left/left_finger at qpos[14])
    left_gripper_rad = data.qpos[14]
    state[8] = gripper_interbotix_to_lerobot(left_gripper_rad, left_calib, "left")
    
    # Right gripper (right/left_finger at qpos[6])
    right_gripper_rad = data.qpos[6]
    state[17] = gripper_interbotix_to_lerobot(right_gripper_rad, right_calib, "right")

    return state


def main():
    parser = argparse.ArgumentParser(
        description="Deploy ACT policy in MuJoCo ALOHA simulation"
    )
    parser.add_argument(
        "--policy-path",
        type=str,
        required=True,
        help="Path to policy checkpoint directory (e.g., outputs/train/act_pick_cuber/checkpoints/080000/pretrained_model)"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Pick up the cube",
        help="Task prompt/instruction for the policy"
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Control frequency (default: 30.0)"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=1000,
        help="Maximum number of steps to run (default: 1000)"
    )
    parser.add_argument(
        "--camera",
        type=str,
        default="teleoperator_pov",
        help="MuJoCo camera name for observation (default: teleoperator_pov)"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without GUI (for headless servers)"
    )
    parser.add_argument(
        "--disable-left-arm",
        action="store_true",
        help="Disable left arm control (set to zero or hold position)"
    )
    parser.add_argument("--new", action="store_true",
                   help="Use PI05 normalization method: degrees = (raw - mid) * 360 / max_res")
    parser.add_argument(
        "--scene-path",
        type=str,
        default="pointclouds/N Goodwin Ave_w_o_Arm.npz",
        help="Path to Gaussian Splatting scene file for composite rendering (optional)"
    )
    parser.add_argument(
        "--color-calib-path",
        type=str,
        default="calibration_pairs_wrist/calibrated/color_mapping.yaml",
        help="Path to color calibration YAML file (optional)"
    )
    
    args = parser.parse_args()
    
    # Load policy
    policy, config_dict = load_policy(args.policy_path)
    device = get_safe_torch_device(policy.config.device)
    policy = policy.to(device)
    
    # Print policy action parameters
    print(f"[INFO] Policy action parameters:")
    if hasattr(policy.config, 'horizon'):
        print(f"  - horizon: {policy.config.horizon} (number of future steps predicted)")
    if hasattr(policy.config, 'n_action_steps'):
        print(f"  - n_action_steps: {policy.config.n_action_steps} (number of steps executed per prediction)")
    if hasattr(policy.config, 'chunk_size'):
        print(f"  - chunk_size: {policy.config.chunk_size} (ACT chunk size)")
    
    # Create pre/post processors
    # Try to load from pretrained path, fallback to creating from config if files don't exist
    processor_path = Path(args.policy_path) / "policy_preprocessor.json"
    if processor_path.exists():
        # Processors exist, load them
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=policy.config,
            pretrained_path=args.policy_path,
        )
    else:
        # Processors don't exist, create from config
        print("[WARN] Processor files not found in checkpoint")
        print("[INFO] Creating processors from config (normalization may not match training)")
        print("[INFO] To fix this, run: python src/lerobot/processor/migrate_policy_normalization.py --pretrained-path " + args.policy_path)
        # Create processors from config without loading from checkpoint
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=policy.config,
            pretrained_path=None,  # Don't load from checkpoint, create from config
        )
    
    # Load MuJoCo model
    project_root = Path(__file__).parent.parent
    aloha_dir = project_root / "aloha"
    original_cwd = os.getcwd()
    try:
        os.chdir(str(aloha_dir))
        model = MjModel.from_xml_path("robolab_setup.xml")
    finally:
        os.chdir(original_cwd)
    
    data = MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco_keyframe_ctrl = data.ctrl.copy()
    
    # Get gripper control range
    right_gripper_actuator_id = 6
    gripper_ctrl_range = (
        model.actuator_ctrlrange[right_gripper_actuator_id, 0],
        model.actuator_ctrlrange[right_gripper_actuator_id, 1]
    )
    print(f"[INFO] Gripper control range: [{gripper_ctrl_range[0]}, {gripper_ctrl_range[1]}]")
    
    # Load calibration directory for action conversion
    calib_dir = project_root / "aloha" / ".cache" / "calibration" / "aloha_default"
    if not calib_dir.exists():
        print(f"[WARN] Calibration directory not found: {calib_dir}")
        print("[WARN] Will use default calibration values")
    
    # Create renderer
    RENDER_W, RENDER_H = 640, 480
    renderer = mujoco.Renderer(model, height=RENDER_H, width=RENDER_W)
    
    # Create segmentation renderer for composite rendering
    seg_renderer = mujoco.Renderer(model, height=RENDER_H, width=RENDER_W)
    seg_renderer.enable_segmentation_rendering()
    
    # Get robot geom IDs for masking
    robot_geom_ids = get_robot_geom_ids(model)
    print(f"[INFO] Found {len(robot_geom_ids)} robot geoms for masking")
    
    # Load Gaussian Splatting scene (optional)
    gaussian_data = None
    if os.path.exists(args.scene_path):
        try:
            # Check if diff_gaussian_rasterization is available
            from diff_gaussian_rasterization import GaussianRasterizer
            from diff_gaussian_rasterization import GaussianRasterizationSettings
            
            # Get initial camera pose and intrinsics for wrist cameras
            # We'll use wrist_cam_right as reference for loading scene
            composite_cam = "wrist_cam_right"
            composite_cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, composite_cam)
            if composite_cam_id == -1:
                print(f"[WARN] Camera '{composite_cam}' not found, trying wrist_cam_left")
                composite_cam = "wrist_cam_left"
                composite_cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, composite_cam)
            
            if composite_cam_id != -1:
                # Get intrinsics
                k = get_camera_intrinsics_from_model(model, composite_cam_id, RENDER_W, RENDER_H)
                
                # Get initial w2c
                mujoco.mj_forward(model, data)
                init_pose = get_mujoco_camera_pose(model, data, composite_cam)
                w2c_init = mj_pose_to_gaussian_w2c(init_pose, T_splat2mj)
                
                # Load scene
                scene_data, scene_depth_data = load_scene_data(args.scene_path, w2c_init, k)
                
                # Load color calibration if available
                color_calib = None
                if args.color_calib_path and os.path.exists(args.color_calib_path):
                    try:
                        color_A, color_b = load_color_mapping(args.color_calib_path)
                        color_calib = (color_A, color_b)
                        print(f"[INFO] Loaded color calibration from: {args.color_calib_path}")
                    except Exception as e:
                        print(f"[INFO] Failed to load color calibration: {e}, Continuing without color calibration")
                else:
                    print("[INFO] Continuing without color calibration")
                
                # Package Gaussian data
                viz_cfg = {
                    'viz_w': RENDER_W, 'viz_h': RENDER_H,
                    'viz_near': 0.1, 'viz_far': 10.0
                }
                gaussian_data = {
                    'scene_data': scene_data,
                    'scene_depth_data': scene_depth_data,
                    'intrinsics_cache': {},  # Will cache intrinsics per camera
                    'viz_cfg': viz_cfg,
                    'color_calib': color_calib  # Add color calibration
                }
                print(f"[INFO] Loaded Gaussian Splatting scene from: {args.scene_path}")
                print(f"[INFO] Composite rendering enabled for wrist cameras")
            else:
                print(f"[WARN] No wrist cameras found, Gaussian rendering disabled")
                
        except ImportError:
            print("[WARN] diff_gaussian_rasterization not installed, composite rendering disabled")
            print("[INFO] To enable, install: pip install git+https://github.com/graphdeco-inria/diff-gaussian-rasterization")
        except Exception as e:
            print(f"[WARN] Failed to load Gaussian Splatting: {e}")
            print("[INFO] Continuing with MuJoCo-only rendering")
    else:
        print(f"[INFO] Scene file not found: {args.scene_path}")
        print("[INFO] Using MuJoCo-only rendering (no Gaussian background)")
    
    # Reset policy
    policy.reset()
    
    # Create OpenCV windows for camera display (if not headless)
    # Only create windows for cameras actually used by the policy
    if not args.headless:
        if hasattr(policy.config, 'image_features') and policy.config.image_features:
            print(f"[INFO] Policy uses {len(policy.config.image_features)} camera(s): {list(policy.config.image_features)}")
            for img_key in policy.config.image_features:
                # Extract camera name from key (e.g., "observation.images.cam_high" -> "cam_high")
                cam_name = img_key.split(".")[-1] if "." in img_key else img_key
                cv2.namedWindow(f"Camera: {cam_name}", cv2.WINDOW_NORMAL)
            print(f"[INFO] Created {len(policy.config.image_features)} camera display window(s)")
        else:
            print("[WARN] Policy config has no image_features, creating windows for all cameras")
            cv2.namedWindow("Camera: cam_high", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Camera: cam_low", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Camera: cam_left_wrist", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Camera: cam_right_wrist", cv2.WINDOW_NORMAL)
            print("[INFO] Camera display windows created")
    
    # Control loop
    print(f"[INFO] Starting policy deployment (max {args.max_steps} steps)")
    
    step_dt = 1.0 / args.fps
    step = 0
    
    # Viewer (optional)
    viewer = None
    if not args.headless and _HAS_DISPLAY:
        try:
            viewer = mujoco.viewer.launch_passive(model, data)
        except Exception as e:
            viewer = None
    
    try:
        while step < args.max_steps:
            step_start = time.perf_counter()
            
            # Build observation from MuJoCo (convert to lerobot format)
            observation = build_observation_from_mujoco(
                model, data, renderer, calib_dir, gripper_ctrl_range, args.camera,
                seg_renderer=seg_renderer,
                robot_geom_ids=robot_geom_ids,
                gaussian_data=gaussian_data
            )
            
            # Display camera images (if not headless)
            if not args.headless:
                display_camera_images(observation, policy_config=policy.config, window_name_prefix="Camera")
            
            # Add prompt if policy supports it
            if hasattr(policy.config, 'language_features') and policy.config.language_features:
                observation["observation.language"] = args.prompt
            
            # Predict action
            with torch.inference_mode():
                action = predict_action(
                    observation,
                    policy,
                    device,
                    preprocessor,
                    postprocessor,
                    policy.config.use_amp,
                    task=args.prompt,
                    robot_type="aloha_follower",
                )
            
            # Debug: show action prediction details on first step
            if step == 0:
                action_shape = action.shape if isinstance(action, torch.Tensor) else np.array(action).shape
                print(f"[INFO] First action prediction:")
                print(f"  - Action shape: {action_shape}")
                if hasattr(policy.config, 'n_action_steps'):
                    print(f"  - Executing step 0 of {policy.config.n_action_steps} predicted actions")
                    print(f"  - Policy will repredict every {policy.config.n_action_steps} steps")
            
            # Convert action to MuJoCo control
            ctrl = convert_action_to_mujoco(action, mujoco_keyframe_ctrl, gripper_ctrl_range, calib_dir, use_new_normalization=args.new)
            
            # Debug output
            if step == 0 or step % 100 == 0:
                action_np = action.cpu().numpy() if isinstance(action, torch.Tensor) else action
                if action_np.ndim > 1:
                    action_np = action_np[0]
                # print(f"\n[DEBUG Step {step}]")
                # print(f"Policy action shape: {action_np.shape}")
                # print(f"  Left arm (0:9):  {action_np[:9]}")
                # print(f"  Right arm (9:18): {action_np[9:18]}")
                # print(f"MuJoCo ctrl shape: {ctrl.shape}")
                # print(f"  Right arm ctrl (0:7):  {ctrl[:7]}")
                # print(f"  Left arm ctrl (7:14):  {ctrl[7:14]} {'[DISABLED]' if args.disable_left_arm else ''}")
                # print(f"MuJoCo qpos (current state):")
                # print(f"  Left arm qpos (0:7):  {data.qpos[0:7]}")
                # print(f"  Right arm qpos (7:14): {data.qpos[7:14]}")
                # print(f"  Grippers qpos (14:16): {data.qpos[14:16] if len(data.qpos) >= 16 else 'N/A'}")
            
            # Apply control
            data.ctrl[:] = ctrl
            
            # Step simulation
            mujoco.mj_step(model, data)
            
            # Update viewer
            if viewer is not None:
                viewer.sync()
            
            # Sleep to maintain control frequency
            elapsed = time.perf_counter() - step_start
            sleep_time = max(0, step_dt - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
            
            step += 1
            
            # Print progress every 100 steps
            if step % 100 == 0:
                print(f"[INFO] Step {step}/{args.max_steps}")
    
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
    finally:
        if viewer is not None:
            viewer.close()
        # Close OpenCV windows
        if not args.headless:
            cv2.destroyAllWindows()
        print("[INFO] Deployment finished")


if __name__ == "__main__":
    main()
