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

from dataclasses import dataclass, field
from pathlib import Path

from lerobot.cameras import CameraConfig
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig

from ..config import RobotConfig


def _default_aloha_cameras() -> dict[str, CameraConfig]:
    """Default camera configuration for ALOHA robot with Intel RealSense cameras."""
    return {
        "cam_high": RealSenseCameraConfig(
            serial_number_or_name="243522072650",
            fps=30,
            width=640,
            height=480,
        ),
        "cam_low": RealSenseCameraConfig(
            serial_number_or_name="246322301893",
            fps=30,
            width=640,
            height=480,
        ),
        "cam_left_wrist": RealSenseCameraConfig(
            serial_number_or_name="241122072859",
            fps=30,
            width=640,
            height=480,
        ),
        "cam_right_wrist": RealSenseCameraConfig(
            serial_number_or_name="241122071122",
            fps=30,
            width=640,
            height=480,
        ),
    }


@RobotConfig.register_subclass("aloha_follower")
@dataclass
class AlohaFollowerConfig(RobotConfig):
    """
    Configuration for the ALOHA bimanual robot follower arms.

    ALOHA is a bimanual robot developed by Trossen Robotics:
    https://www.trossenrobotics.com/aloha-stationary
    https://aloha-2.github.io

    Each arm has 9 motors (including shadow motors for shoulder and elbow):
    - waist, shoulder, shoulder_shadow, elbow, elbow_shadow, forearm_roll, wrist_angle, wrist_rotate, gripper
    """

    # Default id for ALOHA robot
    id: str = "aloha"

    # Default calibration directory for ALOHA follower arms
    # Contains aloha_left.json and aloha_right.json converted from old ALOHA calibration files
    calibration_dir: Path | None = Path(".cache/calibration/aloha_follower")

    # Which arms to use (set to False to disable an arm for single-arm operation)
    # Left arm disabled by default for testing
    use_left_arm: bool = True
    use_right_arm: bool = True

    # Use raw motor positions instead of normalized positions
    # Set to True for direct teleoperation (leader/follower raw mapping)
    # Set to False for policy deployment (policies expect normalized positions)
    use_raw_positions: bool = False

    # Ports to connect to the left and right follower arms
    left_port: str = "/dev/ttyDXL_puppet_left"
    right_port: str = "/dev/ttyDXL_puppet_right"

    # Whether to disable torque on disconnect for each arm
    left_disable_torque_on_disconnect: bool = True
    right_disable_torque_on_disconnect: bool = True

    # /!\ FOR SAFETY, READ THIS /!\
    # `max_relative_target` limits the magnitude of the relative positional target vector for safety purposes.
    # Set this to a positive scalar to have the same value for all motors, or a list that is the same length as
    # the number of motors in your follower arms.
    # For Aloha, for every goal position request, motor rotations are capped at 5 degrees by default.
    # When you feel more confident with teleoperation or running the policy, you can extend
    # this safety limit and even removing it by setting it to `null`.
    left_max_relative_target: float | None = 5.0
    right_max_relative_target: float | None = 5.0

    # cameras (shared between both arms)
    # Default cameras for ALOHA: cam_high, cam_low, cam_left_wrist, cam_right_wrist
    # To disable cameras, set cameras to empty dict: --robot.cameras='{}'
    # Troubleshooting: If one of your IntelRealSense cameras freeze during
    # data recording due to bandwidth limit, you might need to plug the camera
    # on another USB hub or PCIe card.
    cameras: dict[str, CameraConfig] = field(default_factory=_default_aloha_cameras)

