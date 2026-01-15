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
Single ALOHA leader arm implementation for teleoperation.
"""

import logging
import time
from functools import cached_property
from typing import Any

from lerobot.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.dynamixel import (
    DynamixelMotorsBus,
    OperatingMode,
)

from ..teleoperator import Teleoperator
from .config_aloha_leader import AlohaLeaderConfig

logger = logging.getLogger(__name__)


class AlohaLeader(Teleoperator):
    """
    A single ALOHA leader arm for teleoperation.

    ALOHA leader arms use Dynamixel motors:
    - xm430-w350 for main joints
    - xl430-w250 for wrist_rotate
    - xc430-w150 for gripper

    Each arm has 9 motors (including shadow motors):
    - waist, shoulder, shoulder_shadow, elbow, elbow_shadow, forearm_roll, wrist_angle, wrist_rotate, gripper
    """

    config_class = AlohaLeaderConfig
    name = "aloha_leader"

    def __init__(self, config: AlohaLeaderConfig):
        super().__init__(config)
        self.config = config

        # ALOHA leader arm motors configuration
        # Uses xm430-w350 for main joints, xl430-w250 for wrist_rotate, xc430-w150 for gripper
        self.bus = DynamixelMotorsBus(
            port=config.port,
            motors={
                "waist": Motor(1, "xm430-w350", MotorNormMode.RANGE_M100_100),
                "shoulder": Motor(2, "xm430-w350", MotorNormMode.RANGE_M100_100),
                "shoulder_shadow": Motor(3, "xm430-w350", MotorNormMode.RANGE_M100_100),
                "elbow": Motor(4, "xm430-w350", MotorNormMode.RANGE_M100_100),
                "elbow_shadow": Motor(5, "xm430-w350", MotorNormMode.RANGE_M100_100),
                "forearm_roll": Motor(6, "xm430-w350", MotorNormMode.RANGE_M100_100),
                "wrist_angle": Motor(7, "xm430-w350", MotorNormMode.RANGE_M100_100),
                "wrist_rotate": Motor(8, "xl430-w250", MotorNormMode.RANGE_M100_100),
                "gripper": Motor(9, "xc430-w150", MotorNormMode.RANGE_0_100),
            },
            calibration=self.calibration if self.calibration else None,
        )

    @cached_property
    def action_features(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in self.bus.motors}

    @cached_property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        self.bus.connect()
        if not self.is_calibrated and calibrate:
            logger.info(
                f"Mismatch between calibration values in the motor and the calibration file "
                f"or no calibration file found for {self}"
            )
            self.calibrate()

        self.configure()
        logger.info(f"{self} connected on port {self.config.port}")

    @property
    def is_calibrated(self) -> bool:
        return self.bus.is_calibrated

    def calibrate(self) -> None:
        """
        Run calibration for this ALOHA leader arm.
        ALOHA comes with default calibration files, so manual calibration is typically not needed.
        """
        if self.calibration:
            user_input = input(
                f"Press ENTER to use provided calibration file associated with the id {self.id}, "
                f"or type 'c' and press ENTER to run calibration: "
            )
            if user_input.strip().lower() != "c":
                logger.info(f"Writing calibration file associated with the id {self.id} to the motors")
                self.bus.write_calibration(self.calibration)
                return

        logger.info(f"\nRunning calibration of {self}")
        self.bus.disable_torque()

        for motor in self.bus.motors:
            self.bus.write("Operating_Mode", motor, OperatingMode.EXTENDED_POSITION.value)

        input(f"Move {self} to the middle of its range of motion and press ENTER....")
        homing_offsets = self.bus.set_half_turn_homings()

        # For ALOHA, forearm_roll can do full rotations
        full_turn_motors = ["forearm_roll"]
        unknown_range_motors = [motor for motor in self.bus.motors if motor not in full_turn_motors]
        print(
            f"Move all joints except {full_turn_motors} sequentially through their "
            "entire ranges of motion.\nRecording positions. Press ENTER to stop..."
        )
        range_mins, range_maxes = self.bus.record_ranges_of_motion(unknown_range_motors)
        for motor in full_turn_motors:
            range_mins[motor] = 0
            range_maxes[motor] = 4095

        self.calibration = {}
        for motor, m in self.bus.motors.items():
            self.calibration[motor] = MotorCalibration(
                id=m.id,
                drive_mode=0,
                homing_offset=homing_offsets[motor],
                range_min=range_mins[motor],
                range_max=range_maxes[motor],
            )

        self.bus.write_calibration(self.calibration)
        self._save_calibration()
        logger.info(f"Calibration saved to {self.calibration_fpath}")

    def configure(self) -> None:
        """Configure the ALOHA leader arm motors."""
        self.bus.disable_torque()
        self.bus.configure_motors()

        # Set secondary/shadow ID for shoulder and elbow
        shoulder_id = self.bus.motors["shoulder"].id
        self.bus.write("Secondary_ID", "shoulder_shadow", shoulder_id)

        elbow_id = self.bus.motors["elbow"].id
        self.bus.write("Secondary_ID", "elbow_shadow", elbow_id)

    def get_action(self) -> dict[str, float]:
        """Read the current position of all motors as the action to send to follower."""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        start = time.perf_counter()
        action = self.bus.sync_read("Present_Position")
        action = {f"{motor}.pos": val for motor, val in action.items()}
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read action: {dt_ms:.1f}ms")
        return action

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        """
        Send feedback to the leader arm (e.g., for force feedback).
        Not implemented for ALOHA as it doesn't support force feedback.
        """
        pass

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self.bus.disconnect()
        logger.info(f"{self} disconnected.")

