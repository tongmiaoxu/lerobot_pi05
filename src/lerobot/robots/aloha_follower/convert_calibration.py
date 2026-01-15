#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Convert ALOHA calibration files from the old LeRobot format (v0.1.0) to the new format (v0.4.x).

Old format (array-based):
{
    "homing_offset": [2048, 3072, ...],
    "drive_mode": [1, 1, ...],
    "start_pos": [2042, 2886, ...],
    "end_pos": [-974, -1921, ...],
    "calib_mode": ["DEGREE", "DEGREE", ...],
    "motor_names": ["waist", "shoulder", ...]
}

New format (dict-based):
{
    "waist": {"id": 1, "drive_mode": 1, "homing_offset": 2048, "range_min": -974, "range_max": 2042},
    ...
}

Usage:
    python -m lerobot.robots.aloha_follower.convert_calibration \
        --input /path/to/old/calibration/left_follower.json \
        --output /path/to/new/calibration/left.json
"""

import argparse
import json
from pathlib import Path


def convert_calibration(input_path: Path, output_path: Path, motor_ids: dict[str, int] | None = None) -> None:
    """
    Convert an old-format ALOHA calibration file to the new format.

    Args:
        input_path: Path to the old calibration file
        output_path: Path to save the new calibration file
        motor_ids: Optional dict mapping motor names to IDs. If not provided, uses default ALOHA IDs.
    """
    # Default motor IDs for ALOHA
    if motor_ids is None:
        motor_ids = {
            "waist": 1,
            "shoulder": 2,
            "shoulder_shadow": 3,
            "elbow": 4,
            "elbow_shadow": 5,
            "forearm_roll": 6,
            "wrist_angle": 7,
            "wrist_rotate": 8,
            "gripper": 9,
        }

    with open(input_path) as f:
        old_calib = json.load(f)

    motor_names = old_calib["motor_names"]
    homing_offsets = old_calib["homing_offset"]
    drive_modes = old_calib["drive_mode"]
    start_positions = old_calib["start_pos"]
    end_positions = old_calib["end_pos"]

    new_calib = {}
    for i, name in enumerate(motor_names):
        # Determine range_min and range_max from start_pos and end_pos
        range_min = min(start_positions[i], end_positions[i])
        range_max = max(start_positions[i], end_positions[i])

        new_calib[name] = {
            "id": motor_ids.get(name, i + 1),
            "drive_mode": drive_modes[i],
            "homing_offset": homing_offsets[i],
            "range_min": range_min,
            "range_max": range_max,
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(new_calib, f, indent=4)

    print(f"Converted calibration saved to: {output_path}")


def convert_all_aloha_calibrations(old_dir: Path, new_dir: Path, robot_id: str = "aloha") -> None:
    """
    Convert all ALOHA calibration files from old directory to new directory.

    The old directory structure is expected to be:
        old_dir/
            left_follower.json
            right_follower.json
            left_leader.json
            right_leader.json

    The new directory structure will be:
        new_dir/robots/aloha_follower/
            {robot_id}_left.json
            {robot_id}_right.json
        new_dir/teleoperators/aloha_leader/
            {robot_id}_left.json
            {robot_id}_right.json

    Args:
        old_dir: Path to directory containing old calibration files
        new_dir: Base path for new calibration files (e.g., ~/.cache/huggingface/lerobot/calibration)
        robot_id: ID to use for the robot calibration files
    """
    # Convert follower calibrations
    for side in ["left", "right"]:
        old_follower = old_dir / f"{side}_follower.json"
        if old_follower.exists():
            new_follower = new_dir / "robots" / "aloha_follower" / f"{robot_id}_{side}.json"
            print(f"Converting {old_follower} -> {new_follower}")
            convert_calibration(old_follower, new_follower)

    # Convert leader calibrations
    for side in ["left", "right"]:
        old_leader = old_dir / f"{side}_leader.json"
        if old_leader.exists():
            new_leader = new_dir / "teleoperators" / "aloha_leader" / f"{robot_id}_{side}.json"
            print(f"Converting {old_leader} -> {new_leader}")
            convert_calibration(old_leader, new_leader)


def main():
    parser = argparse.ArgumentParser(description="Convert ALOHA calibration files from old to new format")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Single file conversion
    single_parser = subparsers.add_parser("single", help="Convert a single calibration file")
    single_parser.add_argument("--input", "-i", type=Path, required=True, help="Input calibration file (old format)")
    single_parser.add_argument("--output", "-o", type=Path, required=True, help="Output calibration file (new format)")

    # Batch conversion
    batch_parser = subparsers.add_parser("batch", help="Convert all ALOHA calibration files")
    batch_parser.add_argument(
        "--old-dir",
        type=Path,
        required=True,
        help="Directory containing old calibration files (e.g., .cache/calibration/aloha_default)",
    )
    batch_parser.add_argument(
        "--new-dir",
        type=Path,
        default=Path.home() / ".cache" / "huggingface" / "lerobot" / "calibration",
        help="Base directory for new calibration files",
    )
    batch_parser.add_argument(
        "--robot-id",
        type=str,
        default="aloha",
        help="Robot ID to use in the new calibration file names",
    )

    args = parser.parse_args()

    if args.command == "single":
        convert_calibration(args.input, args.output)
    elif args.command == "batch":
        convert_all_aloha_calibrations(args.old_dir, args.new_dir, args.robot_id)


if __name__ == "__main__":
    main()

