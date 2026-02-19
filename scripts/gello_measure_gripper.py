#!/usr/bin/env python
"""
Measure GELLO gripper travel by reading the gripper motor encoder.

1. Fully open the gripper (move GELLO gripper or use UFactory Studio)
2. Press Enter to record the open position
3. Fully close the gripper
4. Press Enter to record the closed position
5. The script prints GRIPPER_OPEN_TO_CLOSED_RAD

Usage:
  python scripts/gello_measure_gripper.py --port /dev/ttyUSB0 --gripper-id 8
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.dynamixel import DynamixelMotorsBus

PULSES_PER_REV = 4096


def main():
    parser = argparse.ArgumentParser(description="Measure GELLO gripper travel")
    parser.add_argument("--port", default="/dev/ttyUSB0")
    parser.add_argument("--gripper-id", type=int, default=8)
    parser.add_argument("--motor-model", default="xl330-m077")
    args = parser.parse_args()

    motors = {"gripper": Motor(args.gripper_id, args.motor_model, MotorNormMode.DEGREES)}
    bus = DynamixelMotorsBus(port=args.port, motors=motors, calibration=None)
    bus.connect()
    bus.disable_torque()

    def read_rad() -> float:
        raw = bus.sync_read("Present_Position", normalize=False)
        return raw["gripper"] / PULSES_PER_REV * 2 * 3.141592653589793

    print(f"Connected to gripper motor (ID {args.gripper_id}) on {args.port}")
    print(f"Current reading: {read_rad():.4f} rad\n")

    input("Fully OPEN the gripper, then press Enter...")
    open_rad = read_rad()
    print(f"  Open position: {open_rad:.4f} rad")

    input("Fully CLOSE the gripper, then press Enter...")
    closed_rad = read_rad()
    print(f"  Closed position: {closed_rad:.4f} rad")

    travel = closed_rad - open_rad
    print(f"\nGRIPPER_OPEN_TO_CLOSED_RAD = {travel:.4f}")
    print(f"gripper_range_rad = [{closed_rad:.4f}, {open_rad:.4f}]")

    bus.disconnect()


if __name__ == "__main__":
    main()
