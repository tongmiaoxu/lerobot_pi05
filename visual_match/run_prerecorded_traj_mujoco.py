#!/usr/bin/env python3
"""
Replay LeRobot v3.0 dataset in MuJoCo ALOHA simulation using ABSOLUTE joint positions.

================================================================================
DATA FORMAT: LeRobot v3.0 (.parquet data files)
================================================================================
- Recorded values are ABSOLUTE joint positions in DEGREES (motor encoder values)
- v3.0: 18 dimensions per frame: [left_arm(8), left_gripper(1), right_arm(8), right_gripper(1)]
- Includes "shadow joints" (shoulder_shadow, elbow_shadow) - mechanically coupled
  joints that mirror the primary joint for additional torque
================================================================================
| Issue                      | Solution                                        |
|----------------------------|-------------------------------------------------|
| Different dimensions       | Skip shadow joint indices (2, 4, 11, 13)        |
| (18 vs 14)                 | → MuJoCo doesn't model parallel linkage         |
|                            |                                                 |
| Different arm order        | Map left/right to MuJoCo's [right, left] order  |
| recorded: [left, right]    | → MuJoCo ctrl: [right_arm(7), left_arm(7)]      |
|                            |                                                 |
| Motor encoder calibration  | Uses calibration files from aloha/.cache/       |
|                            | → calibrated_degrees → raw_encoder → radians   |
|                            |                                                 |
| Absolute positions         | Direct conversion using calibration offsets     |
|                            | → No delta-based approach, true motor positions|
|                            |                                                 |
| Gripper percentage         | Linear mapping from lerobot % to MuJoCo meters |
|                            | Range read from XML, lerobot range configurable |
================================================================================

Recorded data layout (18-dim):
  Left arm:  [0:waist, 1:shoulder, 2:shoulder_shadow, 3:elbow, 4:elbow_shadow,
              5:forearm_roll, 6:wrist_angle, 7:wrist_rotate, 8:gripper]
  Right arm: [9:waist, 10:shoulder, 11:shoulder_shadow, 12:elbow, 13:elbow_shadow,
              14:forearm_roll, 15:wrist_angle, 16:wrist_rotate, 17:gripper]
"""

# -------- Standard imports & sys.path tweaks ---------------------------------
import sys, os, dataclasses, argparse
from pathlib import Path
import time
from typing import Optional

# Add src to path for lerobot imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Auto-detect if we have a display before importing mujoco
def _detect_display():
    """Check if a display is available (X11 on Linux, or macOS/Windows)."""
    if os.environ.get("DISPLAY"):
        return True
    if os.environ.get("WAYLAND_DISPLAY"):
        return True
    if sys.platform in ("darwin", "win32"):
        return True
    if os.environ.get("SSH_CONNECTION") and not os.environ.get("DISPLAY"):
        return False
    return False

_HAS_DISPLAY = _detect_display()
print(f"[INFO] Display detected: {_HAS_DISPLAY}")

if not _HAS_DISPLAY:
    os.environ["MUJOCO_GL"] = "egl"

import numpy as np
import torch

import mujoco
from mujoco import MjModel, MjData

if _HAS_DISPLAY:
    import mujoco.viewer

import plotly.graph_objects as go
import json

# Lerobot dataset loader
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# Import unified conversion module
from mujoco_lerobot_conversion import (
    convert_actions_to_mujoco_pi05,
    convert_actions_to_mujoco_absolute,
    convert_mujoco_state_to_lerobot,
    MuJoCoLeRobotConverter,
    LEROBOT_OPEN_PCT,
    LEROBOT_CLOSED_PCT,
)

# -------- Dataclasses ---------------------------------------------------------
@dataclasses.dataclass
class Args:
    default_prompt: Optional[str] = "pick cube"
    dataset_path: str = "/home/tongmiao/Documents/pick_cuber"
    dataset_root: Optional[str] = None
    action_key: str = "action"
    episode: int = 0
    fps: float = 30.0
    use_new_normalization: bool = False

def parse_cli() -> Args:
    p = argparse.ArgumentParser(
        description="Replay LeRobot v3.0 dataset in MuJoCo (no policy needed)."
    )
    p.add_argument("--default-prompt", type=str, default="pick cube")
    p.add_argument("--dataset-path", type=str, default="/home/tongmiao/Documents/pick_cuber",
                   help="Path to dataset directory (local) or repo_id (Hub)")
    p.add_argument("--dataset-root", type=str, default=None,
                   help="Root directory for local datasets (default: ~/.cache/huggingface/lerobot)")
    p.add_argument("--action-key", type=str, default="action")
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--fps", type=float, default=30.0, help="Replay frame rate (default: 30.0)")
    p.add_argument("--new", action="store_true",
                   help="Use PI05 normalization method: degrees = (raw - mid) * 360 / max_res")

    ns = p.parse_args()

    return Args(
        default_prompt=ns.default_prompt,
        dataset_path=ns.dataset_path,
        dataset_root=ns.dataset_root,
        action_key=ns.action_key,
        episode=ns.episode,
        fps=ns.fps,
        use_new_normalization=ns.new,
    )

# -------- Dataset loader for v3.0 using LeRobotDataset ----------------------
def load_episode(dataset_path: str, episode_idx: int, dataset_root: str | None = None):
    """
    Load a single episode from a LeRobot v3.0 dataset using LeRobotDataset.
    
    Args:
        dataset_path: Path to local dataset directory or repo_id for Hub dataset
        episode_idx: Episode index to load
        dataset_root: Root directory for local datasets (optional, used when dataset_path is repo_id)
    """
    print(f"[INFO] Loading episode {episode_idx} from dataset: {dataset_path}")
    
    # Check if dataset_path is a local directory or repo_id
    dataset_path_obj = Path(dataset_path)
    if dataset_path_obj.exists() and dataset_path_obj.is_dir():
        # Local dataset - check if it has the v3.0 structure (meta/info.json)
        meta_info = dataset_path_obj / "meta" / "info.json"
        if meta_info.exists():
            # Dataset is at the path directly - use it as root
            # LeRobotDataset expects root/repo_id/, but for local datasets we can pass
            # the dataset directory as root and use the directory name as repo_id
            # However, LeRobotDatasetMetadata uses root directly, so we need to pass
            # the dataset directory as root
            repo_id = dataset_path_obj.name
            # For local datasets, LeRobotDataset will use root directly for metadata
            # So we pass the dataset directory as root
            dataset = LeRobotDataset(
                repo_id=repo_id,
                root=dataset_path_obj,  # Pass dataset directory as root
                episodes=[episode_idx],
                video_backend="pyav"  # Use pyav instead of torchcodec to avoid FFmpeg library issues
            )
        else:
            # Path might be a parent directory - try using directory name as repo_id
            repo_id = dataset_path_obj.name
            root = dataset_path_obj.parent
            dataset = LeRobotDataset(
                repo_id=repo_id,
                root=root,
                episodes=[episode_idx],
                video_backend="pyav"  # Use pyav instead of torchcodec to avoid FFmpeg library issues
            )
    else:
        # Assume it's a repo_id - load from Hub or local cache
        dataset = LeRobotDataset(
            repo_id=dataset_path,
            root=dataset_root,
            episodes=[episode_idx],
            video_backend="pyav"  # Use pyav instead of torchcodec to avoid FFmpeg library issues
        )
    
    # Get episode metadata
    if episode_idx >= dataset.num_episodes:
        raise ValueError(f"Episode {episode_idx} not found. Dataset has {dataset.num_episodes} episodes")
    
    # Get episode frame range
    ep_meta = dataset.meta.episodes[episode_idx]
    start_idx = ep_meta["dataset_from_index"]
    end_idx = ep_meta["dataset_to_index"]
    # end_idx is inclusive, so the valid range is [start_idx, end_idx]
    # But we need to make sure we don't go beyond the dataset size
    dataset_size = len(dataset)
    end_idx = min(end_idx, dataset_size - 1)
    num_frames = end_idx - start_idx + 1
    
    print(f"[INFO] Episode {episode_idx} has {num_frames} frames (indices {start_idx} to {end_idx})")
    
    # Load all frames from this episode
    actions = []
    observations = []
    
    for frame_idx in range(start_idx, end_idx + 1):
        sample = dataset[frame_idx]
        actions.append(sample["action"].numpy())
        observations.append(sample["observation.state"].numpy())
    
    # Convert to tensors
    actions_tensor = torch.from_numpy(np.array(actions))
    observations_tensor = torch.from_numpy(np.array(observations))
    
    return {
        'action': actions_tensor,
        'observation.state': observations_tensor,
        'episode_index': episode_idx,
        'num_frames': num_frames,
    }

# -------- Observation builder -------------------------------------------------
# XML path - resolve relative to project root
_project_root = Path(__file__).parent.parent
XML_PATH = str(_project_root / "aloha" / "robolab_setup.xml")
RENDER_W, RENDER_H = 640, 480

def build_observation(model, data, recorded_observation, renderer, prompt):
    """Build observation from MuJoCo state.
    
    MuJoCo qpos structure (with cube freejoint):
    - qpos[0:7] = right arm (6 joints + left_finger)
    - qpos[7] = right/right_finger (coupled, excluded)
    - qpos[8:15] = left arm (6 joints + left_finger)
    - qpos[15] = left/right_finger (coupled, excluded)
    - qpos[16:23] = cube freejoint (3 pos + 4 quat)
    
    Observation state (14-dim): [left_arm(7), right_arm(7)]
    """
    state = np.zeros((14,), np.float32)
    # Left arm: qpos[8:15] (7 elements)
    state[:7] = data.qpos[8:15]
    # Right arm: qpos[0:7] (7 elements)
    state[7:] = data.qpos[0:7]
    images = {}
    return {"state": state, "images": images, "prompt": prompt}

# Conversion functions are now imported from mujoco_lerobot_conversion module
# See: convert_actions_to_mujoco_pi05, convert_actions_to_mujoco_absolute

# -------- Main ----------------------------------------------------------------
def main():
    args = parse_cli()

    print(f"[INFO] Loading dataset from: {args.dataset_path}")
    print(f"[INFO] Replaying episode {args.episode}")
    
    episode_data = load_episode(args.dataset_path, args.episode, dataset_root=args.dataset_root)
    
    # Extract raw actions (18-dim, degrees)
    actions_raw = episode_data[args.action_key].numpy()  # (num_frames, 18)
    observations = episode_data["observation.state"].numpy()
    num_frames = len(actions_raw)
    
    print(f"[INFO] Action shape: {actions_raw.shape}")
    print(f"[INFO] First frame action: {actions_raw[0]}")
    
    # ============================================================
    # Load MuJoCo model first to get gripper control range
    # ============================================================
    # Change to aloha directory so relative includes resolve correctly
    aloha_dir = _project_root / "aloha"
    original_cwd = os.getcwd()
    try:
        os.chdir(str(aloha_dir))
        model = MjModel.from_xml_path("robolab_setup.xml")
    finally:
        os.chdir(original_cwd)
    data = MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco_keyframe_ctrl = data.ctrl.copy()
    
    # Read gripper control range from model (right gripper is actuator 6)
    right_gripper_actuator_id = 6
    gripper_ctrl_range = (
        model.actuator_ctrlrange[right_gripper_actuator_id, 0],
        model.actuator_ctrlrange[right_gripper_actuator_id, 1]
    )
    print(f"[INFO] Gripper control range from XML: [{gripper_ctrl_range[0]}, {gripper_ctrl_range[1]}]")
    
    # ============================================================
    # Choose conversion method based on --new flag
    # ============================================================
    if args.use_new_normalization:
        print("[INFO] Using PI05 normalization method (--new flag)")
        ctrl_sequence = convert_actions_to_mujoco_pi05(
            actions_raw, gripper_ctrl_range
        )
    else:
        print("[INFO] Using legacy absolute calibration method")
        ctrl_sequence = convert_actions_to_mujoco_absolute(
            actions_raw, gripper_ctrl_range
        )
    
    print(f"[INFO] Ctrl range: [{ctrl_sequence.min():.3f}, {ctrl_sequence.max():.3f}]")
    renderer = mujoco.Renderer(model, height=RENDER_H, width=RENDER_W)

    times = []
    grip_ch6 = []
    grip_ch13 = []
    gri_ch6_action = []
    gri_ch13_action = []

    def run_simulation(viewer=None):
        TIMESTEP = 1.0 / args.fps  # Match video frame rate
        print(f"[INFO] Using TIMESTEP={TIMESTEP:.4f}s (FPS={args.fps})")
        
        for frame_idx in range(num_frames):
            recorded_observation = observations[frame_idx]
            prompt = args.default_prompt or ""
            obs = build_observation(model, data, recorded_observation, renderer, prompt)
            
            # Apply ctrl directly (like working script)
            data.ctrl[:] = ctrl_sequence[frame_idx]
            
            # Step simulation forward by TIMESTEP
            sim_time_target = data.time + TIMESTEP
            while data.time < sim_time_target:
                mujoco.mj_step(model, data)
            
            # Sync viewer once per frame (not per physics step) for better performance
            if viewer is not None:
                viewer.sync()
            
            times.append(data.time)
            grip_ch6.append(recorded_observation[6])
            grip_ch13.append(recorded_observation[13])
            gri_ch6_action.append(obs['state'][6])
            gri_ch13_action.append(obs['state'][13])
            
            if frame_idx % 100 == 0:
                print(f"[INFO] Frame {frame_idx}/{num_frames}, ctrl[1]={data.ctrl[1]:.3f}")

    if _HAS_DISPLAY:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            run_simulation(viewer)
    else:
        print("[INFO] Running in headless mode (no viewer)")
        run_simulation(None)

    idx = list(range(len(grip_ch6)))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=idx, y=grip_ch6, mode='lines', name='grip_ch6'))
    fig.add_trace(go.Scatter(x=idx, y=grip_ch13, mode='lines', name='grip_ch13'))
    fig.add_trace(go.Scatter(x=idx, y=gri_ch6_action, mode='lines', name='gri_ch6_action'))
    fig.add_trace(go.Scatter(x=idx, y=gri_ch13_action, mode='lines', name='gri_ch13_action'))
    fig.update_layout(title='Gripper Actions Over Time', xaxis_title='Index', yaxis_title='Value')
    fig.write_html("gripper_actions_v16.html")
    print("Saved plot → gripper_actions_v16.html")
    print("[DONE] playback finished")

if __name__ == "__main__":
    main()
