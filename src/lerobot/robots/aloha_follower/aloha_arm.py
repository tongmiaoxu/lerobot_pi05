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
Single ALOHA arm implementation for the follower robot.
This is an internal helper class used by AlohaFollower.
"""

import logging
import time
from typing import Any

from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.dynamixel import (
    DynamixelMotorsBus,
    OperatingMode,
)

from ..utils import ensure_safe_goal_position

logger = logging.getLogger(__name__)


class AlohaArm:
    """
    A single ALOHA follower arm implementation.

    ALOHA arms use Dynamixel motors:
    - Follower arms: xm540-w270 for most joints, xm430-w350 for wrist_rotate and gripper

    Each arm has 9 motors (including shadow motors):
    - waist, shoulder, shoulder_shadow, elbow, elbow_shadow, forearm_roll, wrist_angle, wrist_rotate, gripper
    """

    def __init__(
        self,
        port: str,
        arm_id: str,
        calibration: dict[str, MotorCalibration] | None = None,
        disable_torque_on_disconnect: bool = True,
        max_relative_target: float | None = 5.0,
        use_raw_positions: bool = False,
    ):
        self.port = port
        self.arm_id = arm_id
        self.disable_torque_on_disconnect = disable_torque_on_disconnect
        self.max_relative_target = max_relative_target
        # Use raw positions for teleoperation, normalized for policy deployment
        self.use_raw_positions = use_raw_positions

        # ALOHA follower arm motors configuration
        # Uses xm540-w270 for most joints and xm430-w350 for wrist_rotate and gripper
        # Using DEGREES mode for joints (like original ALOHA) and RANGE_0_100 for gripper (percentage)
        self.bus = DynamixelMotorsBus(
            port=port,
            motors={
                "waist": Motor(1, "xm540-w270", MotorNormMode.DEGREES),
                "shoulder": Motor(2, "xm540-w270", MotorNormMode.DEGREES),
                "shoulder_shadow": Motor(3, "xm540-w270", MotorNormMode.DEGREES),
                "elbow": Motor(4, "xm540-w270", MotorNormMode.DEGREES),
                "elbow_shadow": Motor(5, "xm540-w270", MotorNormMode.DEGREES),
                "forearm_roll": Motor(6, "xm540-w270", MotorNormMode.DEGREES),
                "wrist_angle": Motor(7, "xm540-w270", MotorNormMode.DEGREES),
                "wrist_rotate": Motor(8, "xm430-w350", MotorNormMode.DEGREES),
                "gripper": Motor(9, "xm430-w350", MotorNormMode.RANGE_0_100),
            },
            calibration=calibration,
        )

    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected

    @property
    def is_calibrated(self) -> bool:
        return self.bus.is_calibrated

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"AlohaArm {self.arm_id} already connected")

        self.bus.connect()
        
        # Skip calibration if calibration data was loaded from file
        if self.bus.calibration is not None and len(self.bus.calibration) > 0:
            logger.info(f"Using existing calibration data for {self.arm_id}, skipping calibration check")
        elif not self.is_calibrated and calibrate:
            logger.info(
                f"Mismatch between calibration values in the motor and the calibration file "
                f"or no calibration file found for {self.arm_id}"
            )
            self.calibrate()

        self.configure()
        logger.info(f"AlohaArm {self.arm_id} connected on port {self.port}")

    def calibrate(self) -> None:
        """
        Run calibration for this ALOHA arm.
        ALOHA comes with default calibration files, so manual calibration is typically not needed.
        """
        logger.info(f"\nRunning calibration of AlohaArm {self.arm_id}")
        self.bus.disable_torque()

        for motor in self.bus.motors:
            self.bus.write("Operating_Mode", motor, OperatingMode.EXTENDED_POSITION.value)

        input(f"Move {self.arm_id} arm to the middle of its range of motion and press ENTER....")
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

        calibration = {}
        for motor, m in self.bus.motors.items():
            calibration[motor] = MotorCalibration(
                id=m.id,
                drive_mode=0,
                homing_offset=homing_offsets[motor],
                range_min=range_mins[motor],
                range_max=range_maxes[motor],
            )

        self.bus.write_calibration(calibration)
        logger.info(f"Calibration complete for {self.arm_id}")
        return calibration

    def configure(self) -> None:
        """Configure the ALOHA arm motors with appropriate settings."""
        with self.bus.torque_disabled():
            self.bus.configure_motors()

            # Set secondary/shadow ID for shoulder and elbow. These joints have two motors.
            # As a result, if only one of them is required to move to a certain position,
            # the other will follow. This is to avoid breaking the motors.
            shoulder_id = self.bus.motors["shoulder"].id
            self.bus.write("Secondary_ID", "shoulder_shadow", shoulder_id)

            elbow_id = self.bus.motors["elbow"].id
            self.bus.write("Secondary_ID", "elbow_shadow", elbow_id)

            # Set a velocity limit of 131 as advised by Trossen Robotics
            for motor in self.bus.motors:
                self.bus.write("Velocity_Limit", motor, 131)

            # Use 'extended position mode' for all motors except gripper
            all_motors_except_gripper = [name for name in self.bus.motors if name != "gripper"]
            for motor in all_motors_except_gripper:
                self.bus.write("Operating_Mode", motor, OperatingMode.EXTENDED_POSITION.value)

            # Use 'position control current based' for follower gripper
            # It can grasp an object without forcing too much
            self.bus.write("Operating_Mode", "gripper", OperatingMode.CURRENT_POSITION.value)

    def get_observation(self) -> dict[str, Any]:
        """Read the current position of all motors.
        
        When use_raw_positions=True: Returns raw motor positions for direct teleoperation.
        When use_raw_positions=False: Returns normalized positions for policy deployment.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"AlohaArm {self.arm_id} is not connected.")

        start = time.perf_counter()
        # Use raw or normalized based on mode
        obs_dict = self.bus.sync_read("Present_Position", normalize=not self.use_raw_positions)
        obs_dict = {f"{motor}.pos": val for motor, val in obs_dict.items()}
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"AlohaArm {self.arm_id} read state: {dt_ms:.1f}ms")

        return obs_dict

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        """
        Command arm to move to a target joint configuration.

        When use_raw_positions=True: Expects raw motor positions for direct teleoperation.
        When use_raw_positions=False: Expects normalized positions for policy deployment.

        Args:
            action: Dictionary of motor positions (e.g., {"waist.pos": 0.5, ...})

        Returns:
            The action actually sent to the motors, potentially clipped.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"AlohaArm {self.arm_id} is not connected.")

        goal_pos = {key.removesuffix(".pos"): val for key, val in action.items() if key.endswith(".pos")}

        # Cap goal position when too far away from present position for safety
        if self.max_relative_target is not None:
            present_pos = self.bus.sync_read("Present_Position", normalize=not self.use_raw_positions)
            goal_present_pos = {key: (g_pos, present_pos[key]) for key, g_pos in goal_pos.items()}
            goal_pos = ensure_safe_goal_position(goal_present_pos, self.max_relative_target)

        # For raw mode, convert to integers (Dynamixel requires int values)
        if self.use_raw_positions:
            goal_pos = {key: int(round(val)) for key, val in goal_pos.items()}

        # Send goal position to the arm
        self.bus.sync_write("Goal_Position", goal_pos, normalize=not self.use_raw_positions)
        return {f"{motor}.pos": val for motor, val in goal_pos.items()}

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"AlohaArm {self.arm_id} is not connected.")

        self.bus.disconnect(self.disable_torque_on_disconnect)
        logger.info(f"AlohaArm {self.arm_id} disconnected.")

