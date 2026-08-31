#!/usr/bin/env python3
"""
Roll out a remote openpi pi05 policy (served via scripts/serve_policy.py in the
openpi repo) inside the xArm7 MuJoCo scene, over a websocket.

This intentionally does NOT load any JAX/openpi code in this process — it only
talks to the policy server via `openpi_client`, so the mujoco/torch/lerobot
env here stays completely separate from the openpi training env, per
https://github.com/Physical-Intelligence/openpi/blob/main/docs/remote_inference.md

Works for any task with a checkpoint under
/home/tina/openpi/ouputs/pi05_xarm_<task>/pi05_xarm_<task>_v1/<step> -- pass
--task to pick which one (choices: pick_shoe, place_mug, hang_mug,
book_shelving, pouring; see lerobot.tasks.get_task_profiles). --task selects
the MuJoCo scene XML, default --prompt, color-calibration path, and turbo
checkpoints automatically; only --policy.config / --policy.dir on the server
side need to match it manually.

Start the server first, in the openpi repo, with the config/dir for whichever
task you're running (example: pick_shoe). XLA_PYTHON_CLIENT_MEM_FRACTION caps
JAX's default (greedy, training-oriented) GPU preallocation -- without it, JAX
grabs ~18GB for a checkpoint that only needs ~6GB, leaving no room for --turbo's
two GPU-resident translation models on the client side:

    XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 uv run scripts/serve_policy.py --port 8010 policy:checkpoint \\
        --policy.config=pi05_xarm_pick_shoe \\
        --policy.dir=/home/tina/openpi/ouputs/pi05_xarm_pick_shoe/pi05_xarm_pick_shoe_v1/18000

Then, in this env (needs `openpi_client` installed, see packages/openpi-client):

    python visual_match/deploy_pi05_remote_mujoco.py --host localhost --port 8010 \\
        --task pick_shoe --checkpoint-name 18000 --num_eval_episodes 10

(--prompt defaults to the task profile's single_task string, e.g. "Pick up the
shoe" -- pass --prompt explicitly to override it.)

By default this composites a Gaussian-Splatting background (from
--scene-path) with the MuJoCo-rendered robot foreground, same pipeline as
deploy_act_policy_mujoco.py, so the policy sees photoreal-ish images instead
of raw sim renders. Pass --no-composite to fall back to plain MuJoCo
rendering. --color-calibrate and --turbo apply on top of that composite;
--turbo needs the task profile to define turbo_output_stationary/wrist
(book_shelving and pouring don't yet -- it's a no-op there until those are
trained).

Like deploy_act_policy_mujoco.py, this runs --num_eval_episodes episodes with
per-episode object-pose auto-alignment against real recorded initial states
(object_pose_auto_align, contour-selected via select_contours_auto), and saves
each episode under episode_NNN/ (states.npy, actions.npy, per-camera videos,
prediction-event snapshots) in the task profile's sim-eval directory --
data_sim/ for raw sim, data_sim_kaifeng/ for --color-calibrate, data_sim_turbo/
for --turbo (see TaskProfile.sim_eval_root_for_policy). Unlike the ACT script,
there is no interactive warmup UI: object alignment either succeeds or a
warning is printed and the episode proceeds anyway -- this script is meant to
run unattended.

Since the checkpoint is loaded server-side (a separate openpi process this
script never starts), it can't be swapped from here. To evaluate multiple
checkpoints back-to-back, start one server per checkpoint on its own port,
then pass --checkpoints "name:port,name:port" (e.g.
"18000:8010,24000:8011") -- each checkpoint is run as its own subprocess,
labeled by name for the output directory. --checkpoint-name labels a single
run the same way when --checkpoints isn't used; the client cannot verify that
a given port is actually serving the checkpoint you're naming it after (the
server never sends its checkpoint path over the wire) -- get_server_metadata()
is printed on connect as a best-effort sanity check only.

--all runs the four visual-input baselines back-to-back (raw sim,
--color-calibrate, --turbo, --turbo_mujoco), same as deploy_act_policy_mujoco.py.
--checkpoints and --all compose: each checkpoint subprocess run itself fans out
over the four variants. The first three composite a Gaussian-Splatting
background with the MuJoCo robot foreground (robot_table/legs/ledge rendered by
GS, excluded from the MuJoCo foreground mask); --turbo_mujoco instead skips GS
entirely and renders robot_table + its legs/ledge as MuJoCo foreground (mirrors
compare_recorded_vs_mujoco.py's --skip-gs).

It also opens two windows so you can watch it run: a 3D MuJoCo viewer, and a
2D window showing exactly the cam_high/cam_wrist frames the policy receives.
Pass --no-mujoco-view / --no-camera-view to disable either (or --headless for both),
or --save-video to also
write an mp4 of the whole run's camera feed (independent of the per-episode
saving above).
"""

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path


def _detect_display():
    if os.environ.get("DISPLAY"):
        return True
    if os.environ.get("WAYLAND_DISPLAY"):
        return True
    if sys.platform in ("darwin", "win32"):
        return True
    return False


_HAS_DISPLAY = _detect_display()
if "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "glx" if _HAS_DISPLAY else "egl"

import cv2  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

if _HAS_DISPLAY and os.environ.get("MUJOCO_GL") != "egl":
    try:
        import mujoco.viewer

        _HAS_MJ_VIEWER = True
    except ImportError:
        _HAS_MJ_VIEWER = False
else:
    _HAS_MJ_VIEWER = False

_VISUAL_MATCH_DIR = Path(__file__).parent
_PROJECT_ROOT = _VISUAL_MATCH_DIR.parent
sys.path.insert(0, str(_VISUAL_MATCH_DIR))

from camera_config import load_camera_config, set_mujoco_camera_from_config  # noqa: E402
from composite_rendering import (  # noqa: E402
    T_splat2mj,
    get_mujoco_camera_pose,
    get_robot_geom_ids,
    load_scene_data,
    mj_pose_to_gaussian_w2c,
)
from deploy_act_policy_mujoco import (  # noqa: E402
    CAMERA_CONFIG as ACT_CAMERA_CONFIG,
    _delete_episode_output_dir,
    _max_prediction_events_per_trajectory,
    _next_saved_episode_index,
    _reserve_episode_output_dir,
    build_combined_window_frame,
    build_last_episode_state_grid,
    load_color_mapping,
    load_initial_state_contours,
    render_composite_view,
    save_selection_grid,
    select_contours_auto,
)
from lerobot.tasks import get_task_profile, get_task_profiles, resolve_task_scene_xml  # noqa: E402
from lerobot.utils.constants import OBS_STATE  # noqa: E402
from lerobot_mujoco_utils import lerobot_state_to_mujoco_ctrl, mujoco_qpos_to_lerobot_state  # noqa: E402
from load_model_xarm import load_model  # noqa: E402
from object_pose_auto_align import ObjectPoseAlignConfig, auto_align_object_poses  # noqa: E402
from openpi_client import websocket_client_policy  # noqa: E402
from sim2real import SimToRealTranslator  # noqa: E402

CAMERA_CONFIG = {
    "stationary": {"obs_key": "cam_high", "mujoco_cam": "stationary_cam"},
    "wrist": {"obs_key": "cam_wrist", "mujoco_cam": "wrist_cam"},
}

# Fixed to match the resolution the camera intrinsics were calibrated at
# (see deploy_act_policy_mujoco.py) -- composite rendering / segmentation
# masking assumes this exact aspect ratio.
RENDER_W, RENDER_H = 640, 480

# How many actions to execute open-loop from each predicted chunk before
# querying the server again. The server predicts 50 steps at a time; you
# don't need a fresh policy call every single sim step.
ACTIONS_PER_CHUNK = 40

_NUM_EVAL_EPISODES_DEFAULT = 10

# Flags that select a visual-input baseline; --all drives these itself, one per
# subprocess run, so they are stripped from the forwarded argv.
_ALL_VARIANT_FLAGS = ("--all", "--color-calibrate", "--turbo", "--turbo_mujoco")
_ALL_VARIANTS = (
    ("raw sim", ()),
    ("color-calibrate", ("--color-calibrate",)),
    ("turbo", ("--turbo",)),
    ("turbo_mujoco", ("--turbo_mujoco",)),
)


def _run_all_variants() -> None:
    """Re-run this script once per --all baseline (raw sim / color-calibrate / turbo / turbo_mujoco)."""
    base_argv = [a for a in sys.argv[1:] if a not in _ALL_VARIANT_FLAGS]
    script_path = str(Path(__file__).resolve())

    for label, extra_flags in _ALL_VARIANTS:
        print(f"\n{'=' * 70}\n[ALL] Running variant: {label}\n{'=' * 70}\n", flush=True)
        cmd = [sys.executable, script_path, *base_argv, *extra_flags]
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(
                f"[ALL] Variant '{label}' exited with code {result.returncode}; stopping remaining variants.",
                flush=True,
            )
            sys.exit(result.returncode)

    print("\n[ALL] All four variants (raw sim, color-calibrate, turbo, turbo_mujoco) completed.\n", flush=True)


def _parse_checkpoints(spec: str) -> list[tuple[str, int]]:
    pairs = []
    for item in re.split(r"[;,]", spec):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"--checkpoints entry {item!r} must be 'name:port'")
        name, port_str = item.rsplit(":", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"--checkpoints entry {item!r} has an empty name")
        pairs.append((name, int(port_str.strip())))
    return pairs


def _run_all_checkpoints(checkpoints: list[tuple[str, int]]) -> None:
    """Re-run this script once per --checkpoints entry, one subprocess each.

    Mirrors deploy_act_policy_mujoco.py's --policy-paths loop: strips
    --checkpoints/--checkpoint-name/--port from the forwarded argv and sets
    them explicitly per checkpoint, so each run writes to that checkpoint's
    own output directory and talks to that checkpoint's own server port.
    """
    base_argv = []
    skip_next = False
    for a in sys.argv[1:]:
        if skip_next:
            skip_next = False
            continue
        if a in ("--checkpoints", "--checkpoint-name", "--port"):
            skip_next = True
            continue
        if a.startswith(("--checkpoints=", "--checkpoint-name=", "--port=")):
            continue
        base_argv.append(a)
    script_path = str(Path(__file__).resolve())

    for i, (name, port) in enumerate(checkpoints, start=1):
        print(
            f"\n{'=' * 70}\n[CHECKPOINTS] Running checkpoint {i}/{len(checkpoints)}: "
            f"{name} (port {port})\n{'=' * 70}\n",
            flush=True,
        )
        cmd = [sys.executable, script_path, *base_argv, "--checkpoint-name", name, "--port", str(port)]
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(
                f"[CHECKPOINTS] Checkpoint {name!r} exited with code "
                f"{result.returncode}; stopping remaining checkpoints.",
                flush=True,
            )
            sys.exit(result.returncode)

    print(f"\n[CHECKPOINTS] All {len(checkpoints)} checkpoints completed.\n", flush=True)


def _act_style_observation(images: dict, state: np.ndarray) -> dict:
    """Convert {obs_key: image} + state into deploy_act_policy_mujoco.py's flat
    'observation.images.<dataset_cam>' / OBS_STATE layout, for reuse with its
    display/save helper functions."""
    obs = {OBS_STATE: state}
    for cam_cfg in CAMERA_CONFIG.values():
        obs[f"observation.images.{cam_cfg['obs_key']}"] = images[cam_cfg["obs_key"]]
    return obs


def render_observation_images(
    model, data, renderer, seg_renderer, robot_geom_ids, gaussian_data, camera_intrinsics, raw_camera_cfg,
) -> dict:
    """Render one composite (or plain MuJoCo) RGB image per camera, pre-translation."""
    for cam_key, cam_cfg in CAMERA_CONFIG.items():
        if raw_camera_cfg[cam_key].get("type", "stationary") == "stationary":
            set_mujoco_camera_from_config(data, model, cam_cfg["mujoco_cam"], raw_camera_cfg[cam_key])

    images = {}
    for cam_key, cam_cfg in CAMERA_CONFIG.items():
        mujoco_cam = cam_cfg["mujoco_cam"]
        if gaussian_data is not None:
            rgb = render_composite_view(
                model, data, renderer, seg_renderer, robot_geom_ids,
                cam_key, mujoco_cam, gaussian_data, camera_intrinsics.get(cam_key),
            )
        else:
            renderer.update_scene(data, camera=mujoco_cam)
            rgb = renderer.render()
        images[cam_cfg["obs_key"]] = rgb
    return images


def state_from_mujoco(model, data) -> np.ndarray:
    ld_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "left_driver_joint")
    g_adr = int(model.jnt_qposadr[ld_id])
    g_rad = (float(model.jnt_range[ld_id, 0]), float(model.jnt_range[ld_id, 1]))
    return mujoco_qpos_to_lerobot_state(data.qpos, g_rad, gripper_qpos_adr=g_adr)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument(
        "--checkpoint-name",
        type=str,
        default=None,
        help="Label for the checkpoint currently served on --port, used only for the sim-eval "
        "output directory name (e.g. '18000'). Required unless --output-dir or "
        "--no-save-sim-eval is given.",
    )
    parser.add_argument(
        "--checkpoints",
        type=str,
        default=None,
        help="Evaluate multiple checkpoints back-to-back, one per subprocess run. Each entry is "
        "'name:port' for an already-running server on that port; separate entries with ',' or "
        "';', e.g. --checkpoints '18000:8010,24000:8011'. Takes priority over --checkpoint-name/"
        "--port.",
    )
    parser.add_argument(
        "--task",
        default="pick_shoe",
        choices=sorted(get_task_profiles()),
        help="Selects the MuJoCo scene, default --prompt, color-calibration path, and turbo "
        "checkpoints. Must match whichever checkpoint --policy.config/--policy.dir the server "
        "was started with.",
    )
    parser.add_argument(
        "--prompt", default=None, help="Defaults to the task profile's single_task string if omitted."
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=1000,
        help="Maximum number of control ticks (each = 1/fps of sim time) per episode, matching "
        "deploy_act_policy_mujoco.py's default. This is a safety ceiling, not a target: "
        "your recorded data_pick_shoe episodes average ~338 ticks (~11.3s) at fps=30.",
    )
    parser.add_argument("--fps", type=int, default=30, help="Control rate; must match training data fps (30 for all your tasks).")
    parser.add_argument(
        "--scene-path",
        type=str,
        default="pointclouds/xarm7_black.npz",
        help="Gaussian Splatting scene file for composite rendering (relative to repo root unless absolute).",
    )
    parser.add_argument(
        "--no-composite", action="store_true", help="Disable Gaussian composite; use plain MuJoCo rendering."
    )
    parser.add_argument(
        "--color-calibrate",
        action="store_true",
        help="Apply per-camera color calibration on top of the composite image (requires composite; "
        "uses task_profile.color_calibration_path for pick_shoe). Ignored if --turbo is also set, "
        "matching deploy_act_policy_mujoco.py (turbo replaces color-calibrate, doesn't stack with it).",
    )
    parser.add_argument(
        "--turbo",
        action="store_true",
        help="Apply per-camera pix2pix-turbo sim2real translation on top of the composite image. "
        "Loads a diffusion-based model per camera onto the GPU -- check free VRAM first.",
    )
    parser.add_argument(
        "--turbo_mujoco",
        action="store_true",
        help="Like --turbo, but skips the Gaussian-Splatting background entirely and renders "
        "robot_table + its legs/ledge as MuJoCo foreground instead (mirrors "
        "compare_recorded_vs_mujoco.py's --skip-gs). Defaults to the TaskProfile "
        "turbo_mujoco_output_* checkpoints (pix2pix-turbo trained on MuJoCo-rendered rather than "
        "real-captured sim images). Use either --turbo or --turbo_mujoco, not both.",
    )
    parser.add_argument("--turbo-prompt", type=str, default="a real-world robot camera image")
    parser.add_argument("--turbo-resolution", type=int, default=224, help="Square side before encode, multiple of 8.")
    parser.add_argument("--turbo-device", type=str, default=None, help="Torch device (default: CUDA if available).")
    parser.add_argument("--no-mujoco-view", action="store_true", help="Disable the 3D MuJoCo viewer window.")
    parser.add_argument(
        "--no-camera-view", action="store_true", help="Disable the 2D window showing cam_high/cam_wrist."
    )
    parser.add_argument(
        "--headless", action="store_true", help="Shorthand for --no-mujoco-view --no-camera-view."
    )
    parser.add_argument(
        "--save-video", type=str, default=None,
        help="Optional path to save one mp4 of the whole run's camera feed (independent of the "
        "per-episode saving below).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run the deployment four times back-to-back, once per visual-input baseline: raw sim "
        "(no color calibration / translation), --color-calibrate, --turbo, and --turbo_mujoco. Each "
        "variant re-invokes this script with the same arguments (minus --all/--color-calibrate/"
        "--turbo/--turbo_mujoco) plus its own variant flag, so each writes to its own task-profile "
        "output directory.",
    )
    parser.add_argument(
        "--num_eval_episodes",
        type=int,
        default=_NUM_EVAL_EPISODES_DEFAULT,
        help=f"Number of evaluation episodes to run (default: {_NUM_EVAL_EPISODES_DEFAULT}).",
    )
    parser.add_argument(
        "--select",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Deterministically pick --num_eval_episodes real initial states (first/last half by "
        "index, via select_contours_auto) to condition per-episode object alignment on. When off, "
        "every episode aligns against a single fixed --episode index.",
    )
    parser.add_argument(
        "--episode", type=int, default=1,
        help="Fixed source episode index used for alignment when --no-select.",
    )
    parser.add_argument(
        "--initial-states-dir",
        type=str,
        default=None,
        help="Path to the dataset root containing <object>/individual_masks. Defaults to the "
        "selected task's dataset root.",
    )
    parser.add_argument(
        "--object-name",
        type=str,
        default=None,
        help="Object name(s) matching the segmentation prompt, e.g. 'mug' or 'mug, saucer'. "
        "Defaults to the selected task's selection_object_name.",
    )
    parser.add_argument(
        "--auto-align-initial-objects",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Before each episode, optimize MuJoCo object poses against saved initial-state SAM "
        "masks (object_pose_auto_align). On failure, a warning is printed and the episode proceeds "
        "anyway -- this script has no interactive fallback.",
    )
    parser.add_argument(
        "--auto-align-cache-dir",
        type=str,
        default=None,
        help="Directory for cached auto-aligned object poses. Defaults to <initial-states-dir>/auto_object_poses.",
    )
    parser.add_argument(
        "--auto-align-force",
        action="store_true",
        help="Recompute automatic object alignment even if a cached pose exists.",
    )
    parser.add_argument(
        "--auto-align-optimize-z",
        action="store_true",
        help="Also optimize object Z. By default only XY and yaw are optimized.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save sim evaluation data. Default is derived from task, 'pi05', and "
        "--checkpoint-name.",
    )
    parser.add_argument(
        "--no-save-sim-eval",
        action="store_true",
        help="Run evaluation but do not write sim outputs (episode npy/mp4 or output directory).",
    )
    parser.add_argument(
        "--record",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save a tiled MP4 of the displayed camera windows for each episode.",
    )
    args = parser.parse_args()
    if args.headless:
        args.no_mujoco_view = True
        args.no_camera_view = True

    if args.checkpoints:
        checkpoints = _parse_checkpoints(args.checkpoints)
        if not checkpoints:
            raise ValueError("--checkpoints was given but contained no non-empty entries")
        _run_all_checkpoints(checkpoints)
        return

    if args.all:
        _run_all_variants()
        return

    task_profile = get_task_profile(args.task)
    max_prediction_events_per_trajectory = _max_prediction_events_per_trajectory(args.task)
    if args.prompt is None:
        args.prompt = task_profile.single_task
    if args.initial_states_dir is None:
        args.initial_states_dir = task_profile.dataset_root
    if args.object_name is None:
        args.object_name = task_profile.selection_object_name

    if args.turbo and args.turbo_mujoco:
        raise ValueError("Use either --turbo or --turbo_mujoco, not both.")
    turbo_mujoco_enabled = args.turbo_mujoco
    turbo_enabled = args.turbo or turbo_mujoco_enabled
    if args.color_calibrate and turbo_enabled:
        print("[INFO] --turbo overrides --color-calibrate (matches deploy_act_policy_mujoco.py); skipping color-calibrate.")

    if args.no_save_sim_eval:
        args.output_dir = None
    elif args.output_dir is None:
        if not args.checkpoint_name:
            raise ValueError(
                "--checkpoint-name is required to derive the sim-eval output directory "
                "(pass it explicitly, set --output-dir, or use --no-save-sim-eval)."
            )
        if turbo_mujoco_enabled:
            sim_variant = "turbo_mujoco"
        elif turbo_enabled:
            sim_variant = "turbo"
        elif args.color_calibrate:
            sim_variant = "kaifeng"
        else:
            sim_variant = "default"
        args.output_dir = task_profile.sim_eval_root_for_policy(
            "pi05", args.checkpoint_name, sim_variant=sim_variant
        )

    output_dir = Path(args.output_dir) if args.output_dir is not None else None
    existing_episode_count = 0
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Sim eval output directory: {output_dir.resolve()}", flush=True)
        existing_episode_count = _next_saved_episode_index(output_dir)
        if existing_episode_count >= args.num_eval_episodes:
            print(
                f"[INFO] {output_dir} already has {existing_episode_count} episode(s) "
                f"(>= --num_eval_episodes={args.num_eval_episodes}); skipping this run.",
                flush=True,
            )
            return
        if existing_episode_count > 0:
            print(
                f"[INFO] {output_dir} already has {existing_episode_count} episode(s); "
                f"resuming from episode {existing_episode_count} up to {args.num_eval_episodes}.",
                flush=True,
            )

    print(f"Connecting to policy server at {args.host}:{args.port} ...")
    client = websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)
    print(f"Server metadata ({args.checkpoint_name or 'unlabeled'} @ port {args.port}):", client.get_server_metadata())

    scene_xml_path = resolve_task_scene_xml(args.task, _PROJECT_ROOT / "xarm7")
    model, data, _ = load_model(model_path=scene_xml_path)

    raw_camera_cfg = {
        cam_key: load_camera_config(cam_cfg["mujoco_cam"]) for cam_key, cam_cfg in CAMERA_CONFIG.items()
    }

    # Match MuJoCo vertical FOV to the calibrated camera intrinsics so the
    # foreground render aligns with the Gaussian-Splatting background.
    for cam_key, cam_cfg in CAMERA_CONFIG.items():
        K = raw_camera_cfg[cam_key]["intrinsics"]
        fy = K[1, 1]
        fovy = float(2.0 * np.degrees(np.arctan(RENDER_H / (2.0 * fy))))
        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_cfg["mujoco_cam"])
        if cam_id >= 0:
            model.cam_fovy[cam_id] = fovy

    gripper_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper")
    gripper_mj_range = (
        model.actuator_ctrlrange[gripper_act_id, 0],
        model.actuator_ctrlrange[gripper_act_id, 1],
    )

    renderer = mujoco.Renderer(model, height=RENDER_H, width=RENDER_W)
    seg_renderer = mujoco.Renderer(model, height=RENDER_H, width=RENDER_W)
    seg_renderer.enable_segmentation_rendering()
    if turbo_mujoco_enabled:
        # --turbo_mujoco has no Gaussian-Splatting background, so robot_table
        # and its legs/ledge must render as MuJoCo foreground instead of being
        # left to GS (mirrors compare_recorded_vs_mujoco.py's --skip-gs).
        robot_geom_ids = get_robot_geom_ids(model, extra_geom_names=[
            "robot_table_leg_1", "robot_table_leg_2", "robot_table_leg_3",
            "robot_table_leg_4", "robot_table_ledger",
        ])
    else:
        # GS already reconstructs the static robot_table (top + legs +
        # ledger); keep it as background rather than compositing it as a
        # foreground mesh alongside movable objects like mug/saucer.
        # (robot_table_top is a box geom, so get_robot_geom_ids's default
        # box sweep would otherwise pull it in as foreground.)
        robot_geom_ids = get_robot_geom_ids(model)
        for _name in (
            "robot_table_top", "robot_table_leg_1", "robot_table_leg_2",
            "robot_table_leg_3", "robot_table_leg_4", "robot_table_ledger",
        ):
            robot_geom_ids.discard(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, _name))

    try:
        home_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
        mujoco.mj_resetDataKeyframe(model, data, home_id)
    except Exception:
        mujoco.mj_resetData(model, data)

    mujoco.mj_forward(model, data)
    for cam_key, cam_cfg in CAMERA_CONFIG.items():
        set_mujoco_camera_from_config(data, model, cam_cfg["mujoco_cam"], raw_camera_cfg[cam_key])

    camera_intrinsics = {cam_key: raw_camera_cfg[cam_key]["intrinsics"] for cam_key in CAMERA_CONFIG}

    # Adjustable-object default poses, restored by _reset_sim() before each
    # episode's auto-align so alignment always starts from a clean baseline
    # (mirrors deploy_act_policy_mujoco.py's _reset_sim).
    adjustable_object_names = tuple(dict.fromkeys(task_profile.deploy_adjustable_object_names))
    adjustable_body_ids: dict[str, int] = {}
    adjustable_body_default_pos: dict[str, np.ndarray] = {}
    adjustable_body_default_quat: dict[str, np.ndarray] = {}
    for obj_name in adjustable_object_names:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, obj_name)
        if body_id < 0:
            continue
        adjustable_body_ids[obj_name] = body_id
        adjustable_body_default_pos[obj_name] = model.body_pos[body_id].copy()
        adjustable_body_default_quat[obj_name] = model.body_quat[body_id].copy()

    def _reset_sim():
        try:
            home_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
            mujoco.mj_resetDataKeyframe(model, data, home_id)
        except Exception:
            mujoco.mj_resetData(model, data)
        for obj_name, body_id in adjustable_body_ids.items():
            model.body_pos[body_id] = adjustable_body_default_pos[obj_name]
            model.body_quat[body_id] = adjustable_body_default_quat[obj_name]
        mujoco.mj_forward(model, data)
        for cam_key, cam_cfg in CAMERA_CONFIG.items():
            if raw_camera_cfg[cam_key].get("type", "stationary") == "stationary":
                set_mujoco_camera_from_config(data, model, cam_cfg["mujoco_cam"], raw_camera_cfg[cam_key])

    gaussian_data = None
    if turbo_mujoco_enabled:
        print("[INFO] --turbo_mujoco: skipping Gaussian-Splatting background; using plain MuJoCo rendering.")
    elif not args.no_composite:
        scene_path = Path(args.scene_path)
        if not scene_path.is_absolute():
            scene_path = _PROJECT_ROOT / scene_path
        if scene_path.exists():
            try:
                init_pose = get_mujoco_camera_pose(model, data, "stationary_cam")
                w2c_init = mj_pose_to_gaussian_w2c(init_pose, T_splat2mj)
                scene_data, scene_depth_data, _ = load_scene_data(
                    str(scene_path), w2c_init, camera_intrinsics["stationary"]
                )
                gaussian_data = {
                    "scene_data": scene_data,
                    "scene_depth_data": scene_depth_data,
                    "viz_cfg": {"viz_w": RENDER_W, "viz_h": RENDER_H, "viz_near": 0.1, "viz_far": 10.0},
                    "color_calib_by_camera": {},
                    "camera_intrinsics": camera_intrinsics,
                }
                print(f"[INFO] Loaded Gaussian Splatting scene from: {scene_path}")
            except Exception as e:
                print(f"[WARN] Failed to load Gaussian scene ({e}); falling back to plain MuJoCo rendering.")
        else:
            print(f"[WARN] Scene file not found: {scene_path}; falling back to plain MuJoCo rendering.")

    if args.color_calibrate and not turbo_enabled:
        if gaussian_data is None:
            print("[WARN] --color-calibrate requires composite rendering; ignoring (no gaussian_data).")
        else:
            for cam_key in CAMERA_CONFIG:
                calib_path = task_profile.color_calibration_path(cam_key)
                try:
                    gaussian_data["color_calib_by_camera"][cam_key] = load_color_mapping(str(calib_path))
                    print(f"[INFO] Loaded {cam_key} color calibration from: {calib_path}")
                except Exception as e:
                    print(f"[WARN] Failed to load {cam_key} color calibration from {calib_path}: {e}")

    turbo_translators: dict[str, object] = {}
    if turbo_enabled:
        ckpt_paths = (
            task_profile.turbo_mujoco_default_checkpoint_paths(_PROJECT_ROOT)
            if turbo_mujoco_enabled
            else task_profile.turbo_default_checkpoint_paths(_PROJECT_ROOT)
        )
        if ckpt_paths is None:
            profile_field = (
                "turbo_mujoco_output_stationary / turbo_mujoco_output_wrist"
                if turbo_mujoco_enabled
                else "turbo_output_stationary / turbo_output_wrist"
            )
            flag = "--turbo_mujoco" if turbo_mujoco_enabled else "--turbo"
            raise ValueError(
                f"Task {args.task!r} has no default turbo checkpoints ({profile_field} unset on "
                f"its TaskProfile); {flag} cannot run for this task."
            )
        else:
            ckpt_stationary, ckpt_wrist = ckpt_paths
            print(f"[INFO] Loading turbo translator (stationary) from: {ckpt_stationary}")
            turbo_translators["stationary"] = SimToRealTranslator(
                checkpoint_path=ckpt_stationary,
                prompt=args.turbo_prompt,
                resolution=args.turbo_resolution,
                device=args.turbo_device,
            )
            print(f"[INFO] Loading turbo translator (wrist) from: {ckpt_wrist}")
            turbo_translators["wrist"] = SimToRealTranslator(
                checkpoint_path=ckpt_wrist,
                prompt=args.turbo_prompt,
                resolution=args.turbo_resolution,
                device=args.turbo_device,
            )

    # Contour-based initial-state selection (deterministic; matches
    # deploy_act_policy_mujoco.py's default --select behavior).
    selected_episode_indices: list[int] | None = None
    if args.select:
        list_of_contours, _contour_mask_size = load_initial_state_contours(
            initial_states_dir=args.initial_states_dir,
            object_name=args.object_name,
        )
        _selected_contours, selected_episode_indices = select_contours_auto(
            list_of_contours, args.num_eval_episodes
        )
        print(
            f"[INFO] Auto-selected {len(selected_episode_indices)} initial-state episodes "
            f"(first/last half): {selected_episode_indices}",
            flush=True,
        )
        if output_dir is not None:
            save_selection_grid(
                initial_states_dir=args.initial_states_dir,
                object_name=args.object_name,
                list_of_contours=list_of_contours,
                selected_indices=selected_episode_indices,
                output_path=output_dir / "selected_states_grid.png",
            )

    auto_align_config = None
    if args.auto_align_initial_objects:
        auto_align_config = ObjectPoseAlignConfig(
            initial_states_dir=args.initial_states_dir,
            object_name=args.object_name,
            cache_dir=args.auto_align_cache_dir,
            optimize_z=args.auto_align_optimize_z,
            force=args.auto_align_force,
            free_joint_pairs=task_profile.calibration_free_joint_pairs,
            body_name_aliases=task_profile.object_body_name_aliases,
        )

    use_viewer = not args.no_mujoco_view and _HAS_DISPLAY and _HAS_MJ_VIEWER
    viewer = None
    viewer_ctx = None
    if use_viewer:
        try:
            viewer_ctx = mujoco.viewer.launch_passive(model, data)
            viewer = viewer_ctx.__enter__()
            print("[INFO] MuJoCo 3D viewer launched.")
        except Exception as e:
            print(f"[WARN] Could not launch 3D viewer ({e}); continuing without it.")
            viewer = None
    elif not args.no_mujoco_view:
        print(f"[INFO] 3D viewer unavailable (_HAS_DISPLAY={_HAS_DISPLAY}, _HAS_MJ_VIEWER={_HAS_MJ_VIEWER}).")

    use_camera_view = not args.no_camera_view
    window_name = "pi05 remote: cam_high | cam_wrist"
    if use_camera_view:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    video_writer = None
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(args.save_video, fourcc, 30, (RENDER_W * 2, RENDER_H))
        print(f"[INFO] Saving whole-run camera feed to {args.save_video}")

    WINDOW_W, WINDOW_H = 400, 300
    step_dt = 1.0 / args.fps
    completed_episodes = existing_episode_count
    episode_idx = existing_episode_count

    try:
        while completed_episodes < args.num_eval_episodes:
            current_episode_output_dir = None
            if output_dir is not None:
                _, current_episode_output_dir = _reserve_episode_output_dir(output_dir)

            _reset_sim()

            source_episode_idx = (
                selected_episode_indices[completed_episodes]
                if selected_episode_indices is not None
                else args.episode
            )

            if auto_align_config is not None:
                try:
                    align_result = auto_align_object_poses(
                        model=model,
                        data=data,
                        seg_renderer=seg_renderer,
                        camera_config=ACT_CAMERA_CONFIG,
                        config=auto_align_config,
                        episode_idx=source_episode_idx,
                        apply=True,
                    )
                    iou_text = ", ".join(
                        f"{name} IoU={iou:.3f}" for name, iou in align_result.iou_by_object.items()
                    )
                    print(
                        f"[INFO] Auto-aligned objects for training episode {source_episode_idx}: "
                        f"loss={align_result.loss:.4f} {iou_text}",
                        flush=True,
                    )
                except Exception as exc:
                    print(
                        f"[WARN] Automatic object alignment failed for episode {episode_idx} "
                        f"(source {source_episode_idx}): {exc}; proceeding with default pose.",
                        flush=True,
                    )

            print(f"\n{'=' * 60}\n  RECORDING episode {episode_idx} ({completed_episodes + 1}/{args.num_eval_episodes})\n{'=' * 60}", flush=True)

            episode_actions: list[np.ndarray] = []
            episode_states: list[np.ndarray] = []
            episode_frames: dict[str, list[np.ndarray]] = {cam_key: [] for cam_key in CAMERA_CONFIG}
            prediction_event_panels: list[tuple[int, int, np.ndarray]] = []
            episode_window_writer: cv2.VideoWriter | None = None
            episode_window_tmp_path: Path | None = None

            step = 0
            prediction_events = 0
            prediction_limit_reached = False
            action_chunk = None
            chunk_limit = 0
            actions_used_in_chunk = 0
            last_turbo_display_obs = None

            while step < args.max_steps:
                needs_prediction = action_chunk is None or actions_used_in_chunk >= chunk_limit
                if needs_prediction and prediction_events >= max_prediction_events_per_trajectory:
                    prediction_limit_reached = True
                    break

                raw_images = render_observation_images(
                    model, data, renderer, seg_renderer, robot_geom_ids, gaussian_data,
                    camera_intrinsics, raw_camera_cfg,
                )
                raw_obs = _act_style_observation(raw_images, state_from_mujoco(model, data))

                for cam_key, cam_cfg in CAMERA_CONFIG.items():
                    episode_frames[cam_key].append(raw_images[cam_cfg["obs_key"]].copy())

                if needs_prediction:
                    final_images = dict(raw_images)
                    for cam_key, cam_cfg in CAMERA_CONFIG.items():
                        tr = turbo_translators.get(cam_key)
                        if tr is not None:
                            final_images[cam_cfg["obs_key"]] = tr.translate(
                                np.ascontiguousarray(raw_images[cam_cfg["obs_key"]])
                            )
                    if turbo_translators:
                        last_turbo_display_obs = _act_style_observation(final_images, raw_obs[OBS_STATE])

                    obs_for_server = {
                        "images": final_images,
                        "state": raw_obs[OBS_STATE],
                        "prompt": args.prompt,
                    }

                    t0 = time.time()
                    result = client.infer(obs_for_server)
                    action_chunk = np.asarray(result["actions"])  # (horizon, 8)
                    infer_ms = result.get("policy_timing", {}).get("infer_ms", (time.time() - t0) * 1000)
                    prediction_events += 1
                    chunk_limit = min(ACTIONS_PER_CHUNK, action_chunk.shape[0])
                    actions_used_in_chunk = 0
                    print(
                        f"[episode {episode_idx} step {step}] prediction {prediction_events} "
                        f"infer_ms={infer_ms:.1f} chunk shape={action_chunk.shape}",
                        flush=True,
                    )

                    if output_dir is not None:
                        panel = build_combined_window_frame(
                            [("Composite", raw_obs)], tile_width=WINDOW_W, tile_height=WINDOW_H,
                        )
                        prediction_event_panels.append((prediction_events, step, panel))

                action = action_chunk[actions_used_in_chunk]
                actions_used_in_chunk += 1

                if use_camera_view or video_writer is not None:
                    combined = np.concatenate([raw_images["cam_high"], raw_images["cam_wrist"]], axis=1)
                    combined_bgr = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
                    if use_camera_view:
                        cv2.imshow(window_name, combined_bgr)
                        cv2.waitKey(1)
                    if video_writer is not None:
                        video_writer.write(combined_bgr)

                if args.record and output_dir is not None:
                    display_rows = [("Composite", raw_obs)]
                    if turbo_translators:
                        display_rows.append(("Turbo", last_turbo_display_obs))
                    combined_rgb = build_combined_window_frame(display_rows, tile_width=WINDOW_W, tile_height=WINDOW_H)
                    combined_bgr2 = cv2.cvtColor(combined_rgb, cv2.COLOR_RGB2BGR)
                    if episode_window_writer is None:
                        episode_window_tmp_path = current_episode_output_dir / "combined_windows_tmp.mp4"
                        episode_window_tmp_path.unlink(missing_ok=True)
                        frame_h, frame_w = combined_bgr2.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        episode_window_writer = cv2.VideoWriter(
                            str(episode_window_tmp_path), fourcc, args.fps, (frame_w, frame_h)
                        )
                    episode_window_writer.write(combined_bgr2)

                episode_states.append(raw_obs[OBS_STATE].copy())
                episode_actions.append(np.asarray(action).copy())

                ctrl = lerobot_state_to_mujoco_ctrl(np.asarray(action), gripper_mj_range)
                data.ctrl[:] = ctrl
                sim_target = data.time + step_dt
                while data.time < sim_target:
                    mujoco.mj_step(model, data)
                if viewer is not None:
                    viewer.sync()

                step += 1

            if episode_window_writer is not None:
                episode_window_writer.release()

            reason = (
                f"prediction limit reached ({max_prediction_events_per_trajectory})"
                if prediction_limit_reached
                else "max steps reached"
            )
            if output_dir is not None:
                ep_dir = current_episode_output_dir
                np.save(str(ep_dir / "states.npy"), np.array(episode_states))
                np.save(str(ep_dir / "actions.npy"), np.array(episode_actions))
                if prediction_event_panels:
                    prediction_dir = ep_dir / "prediction_events"
                    prediction_dir.mkdir(parents=True, exist_ok=True)
                    for event_idx, step_idx, panel_rgb in prediction_event_panels:
                        panel_path = prediction_dir / f"composite_prediction_event_{event_idx:02d}_step_{step_idx:04d}.png"
                        cv2.imwrite(str(panel_path), cv2.cvtColor(panel_rgb, cv2.COLOR_RGB2BGR))
                for cam_key, frames in episode_frames.items():
                    if not frames:
                        continue
                    dataset_cam = CAMERA_CONFIG[cam_key]["obs_key"]
                    video_path = ep_dir / f"{dataset_cam}.mp4"
                    h, w = frames[0].shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(str(video_path), fourcc, args.fps, (w, h))
                    for frame in frames:
                        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                    writer.release()
                if args.record and episode_window_tmp_path is not None:
                    combined_path = ep_dir / "combined_windows.mp4"
                    combined_path.unlink(missing_ok=True)
                    episode_window_tmp_path.replace(combined_path)
                print(f"[INFO] Episode data saved ({reason}) -> {ep_dir.resolve()}", flush=True)
            else:
                print(f"[INFO] Episode completed ({reason}) with sim eval disk output disabled.", flush=True)
            completed_episodes += 1

            episode_idx += 1
    finally:
        if viewer_ctx is not None:
            try:
                viewer_ctx.__exit__(None, None, None)
            except Exception:
                pass
        if use_camera_view:
            cv2.destroyWindow(window_name)
        if video_writer is not None:
            video_writer.release()

    if output_dir is not None:
        try:
            grid_path = build_last_episode_state_grid(output_dir)
            if grid_path is not None:
                print(f"[INFO] Saved last-episode-state grid -> {grid_path.resolve()}", flush=True)
            else:
                print("[WARN] No episode videos found; skipped last-episode-state grid.", flush=True)
        except Exception as e:
            print(f"[WARN] Failed to build last-episode-state grid: {e}", flush=True)

    print("Done.")


if __name__ == "__main__":
    main()
