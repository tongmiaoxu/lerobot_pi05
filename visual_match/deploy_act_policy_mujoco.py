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
                                  camera_name: str = "teleoperator_pov") -> dict:
    """
    Build observation dictionary from MuJoCo state.
    
    Returns observation in format expected by policy:
    {
        "observation.state": np.ndarray,  # (14,) joint positions
        "observation.images.cam_high": np.ndarray,  # (H, W, 3) RGB image
        "observation.images.cam_low": np.ndarray,
        "observation.images.cam_left_wrist": np.ndarray,
        "observation.images.cam_right_wrist": np.ndarray,
    }
    """
    # Convert MuJoCo state (radians) to lerobot format (normalized degrees)
    # This converts qpos from MuJoCo radians to lerobot normalized degrees
    state = convert_mujoco_state_to_lerobot(data, calib_dir, gripper_ctrl_range)
    
    # Build observation dict
    observation = {
        OBS_STATE: state,
    }
    
    # Render all required camera images
    # Policy expects: cam_high, cam_low, cam_left_wrist, cam_right_wrist
    # Try to use specific cameras if they exist, otherwise use the default camera for all
    camera_names = {
        "observation.images.cam_high": "teleoperator_pov",  # Default to teleoperator view
        "observation.images.cam_low": "depth_cam",
        "observation.images.cam_left_wrist": "wrist_cam_left",
        "observation.images.cam_right_wrist": "wrist_cam_right",
    }
 
    
    # Render each camera view
    for obs_key, cam_name_to_use in camera_names.items():
        try:
            renderer.update_scene(data, camera=cam_name_to_use)
            rgb_image = renderer.render()
            # Convert to [0, 1] range
            rgb_image = rgb_image.astype(np.float32) / 255.0
            observation[obs_key] = rgb_image
        except Exception as e:
            # If camera doesn't exist, use the default camera
            print(f"[WARN] Camera '{cam_name_to_use}' not found, using '{camera_name}' instead")
            renderer.update_scene(data, camera=camera_name)
            rgb_image = renderer.render()
            rgb_image = rgb_image.astype(np.float32) / 255.0
            observation[obs_key] = rgb_image
    
    return observation


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
    
    args = parser.parse_args()
    
    # Load policy
    policy, config_dict = load_policy(args.policy_path)
    device = get_safe_torch_device(policy.config.device)
    policy = policy.to(device)
    
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
    
    # Reset policy
    policy.reset()
    
    # Control loop
    print(f"[INFO] Starting policy deployment (max {args.max_steps} steps)")
    print(f"[INFO] Task prompt: '{args.prompt}'")
    print(f"[INFO] Control frequency: {args.fps} Hz")
    
    step_dt = 1.0 / args.fps
    step = 0
    
    # Viewer (optional)
    viewer = None
    if not args.headless and _HAS_DISPLAY:
        try:
            viewer = mujoco.viewer.launch_passive(model, data)
            print("[INFO] MuJoCo viewer opened (close window to stop)")
        except Exception as e:
            print(f"[WARN] Could not open viewer: {e}")
            viewer = None
    
    try:
        while step < args.max_steps:
            step_start = time.perf_counter()
            
            # Build observation from MuJoCo (convert to lerobot format)
            observation = build_observation_from_mujoco(
                model, data, renderer, calib_dir, gripper_ctrl_range, args.camera
            )
            
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
            
            # Convert action to MuJoCo control
            ctrl = convert_action_to_mujoco(action, mujoco_keyframe_ctrl, gripper_ctrl_range, calib_dir, use_new_normalization=args.new)
            
            # Debug output
            if step == 0 or step % 100 == 0:
                action_np = action.cpu().numpy() if isinstance(action, torch.Tensor) else action
                if action_np.ndim > 1:
                    action_np = action_np[0]
                print(f"\n[DEBUG Step {step}]")
                print(f"Policy action shape: {action_np.shape}")
                print(f"  Left arm (0:9):  {action_np[:9]}")
                print(f"  Right arm (9:18): {action_np[9:18]}")
                print(f"MuJoCo ctrl shape: {ctrl.shape}")
                print(f"  Right arm ctrl (0:7):  {ctrl[:7]}")
                print(f"  Left arm ctrl (7:14):  {ctrl[7:14]} {'[DISABLED]' if args.disable_left_arm else ''}")
                print(f"MuJoCo qpos (current state):")
                print(f"  Left arm qpos (0:7):  {data.qpos[0:7]}")
                print(f"  Right arm qpos (7:14): {data.qpos[7:14]}")
                print(f"  Grippers qpos (14:16): {data.qpos[14:16] if len(data.qpos) >= 16 else 'N/A'}")
            
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
        print("[INFO] Deployment finished")


if __name__ == "__main__":
    main()
