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

from lerobot.cameras import CameraConfig

from ..config import RobotConfig


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
    # Troubleshooting: If one of your IntelRealSense cameras freeze during
    # data recording due to bandwidth limit, you might need to plug the camera
    # on another USB hub or PCIe card.
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

