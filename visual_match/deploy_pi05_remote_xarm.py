#!/usr/bin/env python3
"""
Roll out a remote openpi pi05 policy (served via scripts/serve_policy.py in the
openpi repo) on the real xArm7, over a websocket. Mirrors
deploy_pi05_remote_mujoco.py, but drives `XarmFollower` instead of MuJoCo.

SAFETY
------
This moves a real robot arm. By default this script only *prints* what it
would send (--dry-run, the default) so you can sanity-check the observation
shapes and predicted actions before anything moves. Only pass --live once
you've verified that, and only with the arm powered, unobstructed, and an
e-stop within reach. Start with --max-steps small (e.g. 20-30) for the first
live run.

Works for any task with a checkpoint under
/home/tina/openpi/ouputs/pi05_xarm_<task>/pi05_xarm_<task>_v1/<step> -- pass
--task to pick which one (choices: pick_shoe, place_mug, hang_mug,
book_shelving, pouring; see lerobot.tasks.get_task_profiles). --task only
selects the default --prompt here (no mujoco scene involved); --policy.config
/ --policy.dir on the server side still need to match it manually.

Start the server first, in the openpi repo, with the config/dir for whichever
task you're running (example: pick_shoe). XLA_PYTHON_CLIENT_MEM_FRACTION caps
JAX's default (greedy, training-oriented) GPU preallocation to leave headroom
for other GPU-resident tools on the client side (e.g. deploy_pi05_remote_mujoco.py's
--turbo) sharing the same GPU:

    XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 uv run scripts/serve_policy.py --port 8010 policy:checkpoint \\
        --policy.config=pi05_xarm_pick_shoe \\
        --policy.dir=/home/tina/openpi/ouputs/pi05_xarm_pick_shoe/pi05_xarm_pick_shoe_v1/18000

Then, in the env with `openpi_client` + lerobot installed (e.g. gello_lerobot):

    # Dry run first (no robot connection, no motion):
    python visual_match/deploy_pi05_remote_xarm.py --task pick_shoe

    # Live, once you've checked the dry-run output:
    python visual_match/deploy_pi05_remote_xarm.py --task pick_shoe \\
        --live --max-steps 30
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lerobot.robots.xarm_follower import XarmFollower, XarmFollowerConfig
from lerobot.tasks import get_task_profile, get_task_profiles
from openpi_client import websocket_client_policy

# How many actions to execute from each predicted chunk before querying the
# server again. Keep this small on the real robot so you're never running far
# ahead on a stale prediction.
ACTIONS_PER_CHUNK = 10

# Seconds to wait between sent actions within a chunk. All of your recorded
# datasets (data_pick_shoe, data_place_mug, data_hang_mug, data_book_shelving,
# data_pouring) were captured at fps=30 (see meta/info.json in each), so we
# match that here. This is NOT the same as XarmFollowerConfig.control_frequency
# (that's the background smoothing rate of the servo thread, 50Hz).
STEP_DT = 1.0 / 30.0


def obs_to_state(obs: dict) -> np.ndarray:
    """XarmFollower.get_observation() -> [joint1..7 deg, gripper mm] float32 vector."""
    state = np.zeros(8, dtype=np.float32)
    for i in range(7):
        state[i] = obs[f"joint{i + 1}.pos"]
    state[7] = obs["gripper.pos"]
    return state


def action_row_to_dict(action_row: np.ndarray) -> dict:
    """[joint1..7 deg, gripper mm] -> XarmFollower.send_action() dict."""
    action = {f"joint{i + 1}.pos": float(action_row[i]) for i in range(7)}
    action["gripper.pos"] = float(action_row[7])
    return action


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument(
        "--task",
        default="pick_shoe",
        choices=sorted(get_task_profiles()),
        help="Selects the default --prompt. Must match whichever checkpoint --policy.config/"
        "--policy.dir the server was started with.",
    )
    parser.add_argument(
        "--prompt", default=None, help="Defaults to the task profile's single_task string if omitted."
    )
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--ip", default="192.168.1.228", help="xArm IP address")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Actually connect to the robot and send actions. Without this flag, "
        "the script only prints what it would do.",
    )
    args = parser.parse_args()

    if args.prompt is None:
        args.prompt = get_task_profile(args.task).single_task

    print(f"Connecting to policy server at {args.host}:{args.port} ...")
    client = websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)
    print("Server metadata:", client.get_server_metadata())

    robot = None
    if args.live:
        print(f"\n*** LIVE MODE: connecting to physical xArm at {args.ip} ***")
        print("*** Make sure the arm is unobstructed and you can reach the e-stop. ***")
        input("Press Enter to continue, or Ctrl+C to abort... ")
        robot = XarmFollower(XarmFollowerConfig(ip=args.ip))
        robot.connect()
    else:
        print("\n*** DRY RUN (default): no robot connection, nothing will move. ***")
        print("*** Pass --live to actually connect and send actions. ***\n")

    try:
        step = 0
        while step < args.max_steps:
            if robot is not None:
                obs = robot.get_observation()
                state = obs_to_state(obs)
                images = {"cam_high": obs["cam_high"], "cam_wrist": obs["cam_wrist"]}
            else:
                # Dry run: fabricate a plausible-shaped observation so we can
                # exercise the client/server round trip without hardware.
                state = np.zeros(8, dtype=np.float32)
                images = {
                    "cam_high": np.zeros((480, 640, 3), dtype=np.uint8),
                    "cam_wrist": np.zeros((480, 640, 3), dtype=np.uint8),
                }

            policy_obs = {"images": images, "state": state, "prompt": args.prompt}

            t0 = time.time()
            result = client.infer(policy_obs)
            action_chunk = np.asarray(result["actions"])  # (action_horizon, 8)
            infer_ms = result.get("policy_timing", {}).get("infer_ms", (time.time() - t0) * 1000)
            print(f"[step {step}] state={state.round(1)}  infer_ms={infer_ms:.1f}  chunk={action_chunk.shape}")

            for i in range(min(ACTIONS_PER_CHUNK, action_chunk.shape[0])):
                action_dict = action_row_to_dict(action_chunk[i])
                if robot is not None:
                    robot.send_action(action_dict)
                else:
                    print(f"    would send: {action_dict}")
                time.sleep(STEP_DT)
                step += 1
                if step >= args.max_steps:
                    break
    finally:
        if robot is not None:
            robot.disconnect()

    print("Done.")


if __name__ == "__main__":
    main()
