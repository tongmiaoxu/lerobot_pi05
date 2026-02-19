#!/usr/bin/env python
"""
Compute GELLO joint offsets for xArm teleoperation.

Same approach as gello_software's gello_get_offset.py:
  xarm_angle = joint_sign * (gello_raw_rad - offset)

Place GELLO in the xArm's start_joints pose (default: [0,0,0,90,0,90,0] deg)
with the gripper fully open, then run this script.

Usage:
  python scripts/gello_get_offset.py --port /dev/ttyUSB0

To use the exact offsets from gello_software (if your GELLO was calibrated there):
  python scripts/gello_get_offset.py --port /dev/ttyUSB0 --use-gello-software-offsets
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.dynamixel import DynamixelMotorsBus

PULSES_PER_REV = 4096
SAVE_DIR = Path(".cache/calibration/gello_leader")

# Travel from open to closed position in radians (measured via gello_measure_gripper.py)
GRIPPER_OPEN_TO_CLOSED_RAD = -0.8529

# Offsets from gello_software's PORT_CONFIG_MAP for /dev/ttyUSB0 (xArm GELLO).
# Use --use-gello-software-offsets to apply these directly.
GELLO_SOFTWARE_OFFSETS = [
    3 * np.pi / 2,
    3 * np.pi / 2,
    1 * np.pi / 2,
    1 * np.pi / 2,
    3 * np.pi / 2,
    2.3 * np.pi / 2,
    3 * np.pi / 2,
]
GELLO_SOFTWARE_SIGNS = [1, 1, 1, 1, 1, 1, 1]
GELLO_SOFTWARE_GRIPPER_OPEN_CLOSE_DEG = (114.145703125, 72.345703125)


def main():
    parser = argparse.ArgumentParser(description="Compute GELLO joint offsets for xArm")
    parser.add_argument("--port", default="/dev/ttyUSB0")
    parser.add_argument("--dof", type=int, default=7)
    parser.add_argument("--joint-signs", nargs="+", type=int, default=None)
    parser.add_argument("--start-joints", nargs="+", type=float, default=None,
                        help="xArm target pose in radians (default: [0,0,0,π/2,0,π/2,0])")
    parser.add_argument("--no-gripper", action="store_true")
    parser.add_argument("--use-gello-software-offsets", action="store_true",
                        help="Use the exact offsets from gello_software instead of computing")
    args = parser.parse_args()

    dof = args.dof
    joint_signs = args.joint_signs or [1] * dof
    start_joints = args.start_joints or [0, 0, 0, np.pi / 2, 0, np.pi / 2, 0][:dof]
    use_gripper = not args.no_gripper
    assert len(joint_signs) == dof and len(start_joints) == dof

    # Connect
    motors = {}
    for i in range(dof):
        motors[f"joint{i+1}"] = Motor(i + 1, "xl330-m288", MotorNormMode.DEGREES)
    if use_gripper:
        motors["gripper"] = Motor(dof + 1, "xl330-m077", MotorNormMode.DEGREES)

    bus = DynamixelMotorsBus(port=args.port, motors=motors, calibration=None)
    bus.connect()
    bus.disable_torque()

    # Flush initial reads
    for _ in range(10):
        bus.sync_read("Present_Position", normalize=False)

    raw = bus.sync_read("Present_Position", normalize=False)
    raw_rad = np.array([raw[f"joint{i+1}"] for i in range(dof)]) / PULSES_PER_REV * 2 * np.pi

    if args.use_gello_software_offsets:
        # Use the exact hardcoded offsets from gello_software
        best_offsets = GELLO_SOFTWARE_OFFSETS[:dof]
        joint_signs = GELLO_SOFTWARE_SIGNS[:dof]
        print("\nUsing gello_software hardcoded offsets directly.")
    else:
        # Compute exact offsets: offset = raw_rad - start_joint * sign
        # (since sign is ±1, dividing by sign is the same as multiplying)
        # Then find the nearest multiple of 2π to keep the offset in a reasonable range
        best_offsets = []
        for i in range(dof):
            exact = raw_rad[i] - start_joints[i] * joint_signs[i]
            best_offsets.append(float(exact))

    # Verify: corrected should match start_joints
    corrected = [joint_signs[i] * (raw_rad[i] - best_offsets[i]) for i in range(dof)]
    print()
    print("offsets (rad):", [f"{x:.4f}" for x in best_offsets])
    print("corrected    :", [f"{x:.4f}" for x in corrected])
    print("expected     :", [f"{x:.4f}" for x in start_joints])
    max_err = max(abs(corrected[i] - start_joints[i]) for i in range(dof))
    print(f"max error    : {max_err:.6f} rad ({np.degrees(max_err):.4f} deg)")

    calib = {"joint_offsets": [float(x) for x in best_offsets], "joint_signs": joint_signs}

    if use_gripper:
        if args.use_gello_software_offsets:
            open_deg, close_deg = GELLO_SOFTWARE_GRIPPER_OPEN_CLOSE_DEG
            open_rad = math.radians(open_deg)
            close_rad = math.radians(close_deg)
        else:
            gripper_open_rad = raw["gripper"] / PULSES_PER_REV * 2 * np.pi
            open_rad = float(gripper_open_rad)
            close_rad = float(gripper_open_rad + GRIPPER_OPEN_TO_CLOSED_RAD)
        calib["gripper_range_rad"] = [close_rad, open_rad]
        print(f"gripper range (rad): [{close_rad:.4f}, {open_rad:.4f}]")

    bus.disconnect()

    # Save
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    save_path = SAVE_DIR / "gello_offsets.json"
    with open(save_path, "w") as f:
        json.dump(calib, f, indent=4)
    print(f"\nSaved to {save_path}")


if __name__ == "__main__":
    main()
