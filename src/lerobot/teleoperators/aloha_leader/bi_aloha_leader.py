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
ALOHA bimanual leader teleoperator implementation.

This is used for teleoperation of the ALOHA bimanual robot.
"""

import logging
from functools import cached_property

from ..teleoperator import Teleoperator
from .aloha_leader_arm import AlohaLeader
from .config_aloha_leader import AlohaLeaderConfig, BiAlohaLeaderConfig

logger = logging.getLogger(__name__)


class BiAlohaLeader(Teleoperator):
    """
    [ALOHA Bimanual Leader Arms](https://www.trossenrobotics.com/aloha-stationary) for teleoperation.

    This teleoperator reads positions from both leader arms and provides them as actions
    for controlling the follower arms.
    """

    config_class = BiAlohaLeaderConfig
    name = "bi_aloha_leader"

    def __init__(self, config: BiAlohaLeaderConfig):
        super().__init__(config)
        self.config = config

        # Create left and right leader arms
        left_arm_config = AlohaLeaderConfig(
            id=f"{config.id}_left" if config.id else "left",
            calibration_dir=config.calibration_dir,
            port=config.left_port,
        )

        right_arm_config = AlohaLeaderConfig(
            id=f"{config.id}_right" if config.id else "right",
            calibration_dir=config.calibration_dir,
            port=config.right_port,
        )

        self.left_arm = AlohaLeader(left_arm_config)
        self.right_arm = AlohaLeader(right_arm_config)

    @cached_property
    def action_features(self) -> dict[str, type]:
        return {f"left_{motor}.pos": float for motor in self.left_arm.bus.motors} | {
            f"right_{motor}.pos": float for motor in self.right_arm.bus.motors
        }

    @cached_property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self.left_arm.is_connected and self.right_arm.is_connected

    def connect(self, calibrate: bool = True) -> None:
        """Connect both leader arms."""
        self.left_arm.connect(calibrate)
        self.right_arm.connect(calibrate)
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        return self.left_arm.is_calibrated and self.right_arm.is_calibrated

    def calibrate(self) -> None:
        """Calibrate both leader arms."""
        self.left_arm.calibrate()
        self.right_arm.calibrate()

    def configure(self) -> None:
        """Configure both leader arms."""
        self.left_arm.configure()
        self.right_arm.configure()

    def get_action(self) -> dict[str, float]:
        """
        Read positions from both leader arms.

        Returns:
            Dictionary with keys like "left_waist.pos", "right_gripper.pos", etc.
        """
        action_dict = {}

        # Get left arm action with "left_" prefix
        left_action = self.left_arm.get_action()
        action_dict.update({f"left_{key}": value for key, value in left_action.items()})

        # Get right arm action with "right_" prefix
        right_action = self.right_arm.get_action()
        action_dict.update({f"right_{key}": value for key, value in right_action.items()})

        return action_dict

    def send_feedback(self, feedback: dict[str, float]) -> None:
        """
        Send feedback to the leader arms (not implemented for ALOHA).
        """
        # Split feedback by arm prefix
        left_feedback = {
            key.removeprefix("left_"): value
            for key, value in feedback.items()
            if key.startswith("left_")
        }
        right_feedback = {
            key.removeprefix("right_"): value
            for key, value in feedback.items()
            if key.startswith("right_")
        }

        if left_feedback:
            self.left_arm.send_feedback(left_feedback)
        if right_feedback:
            self.right_arm.send_feedback(right_feedback)

    def disconnect(self) -> None:
        """Disconnect both leader arms."""
        self.left_arm.disconnect()
        self.right_arm.disconnect()
        logger.info(f"{self} disconnected.")

