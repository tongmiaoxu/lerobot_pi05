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
ALOHA bimanual follower robot implementation.

ALOHA is a bimanual robot developed by Trossen Robotics:
https://www.trossenrobotics.com/aloha-stationary
https://aloha-2.github.io
"""

import logging
import time
from functools import cached_property
from typing import Any

from lerobot.cameras.utils import make_cameras_from_configs

from ..robot import Robot
from .aloha_arm import AlohaArm
from .config_aloha_follower import AlohaFollowerConfig

logger = logging.getLogger(__name__)


class AlohaFollower(Robot):
    """
    [ALOHA Bimanual Robot](https://www.trossenrobotics.com/aloha-stationary) developed by Trossen Robotics.

    This is a bimanual robot with two follower arms, each having 9 Dynamixel motors.
    The robot can be used for imitation learning with the corresponding leader arms for teleoperation.
    """

    config_class = AlohaFollowerConfig
    name = "aloha_follower"

    def __init__(self, config: AlohaFollowerConfig):
        super().__init__(config)
        self.config = config

        # Create left and right follower arms based on config
        self.left_arm = None
        self.right_arm = None

        if config.use_left_arm:
            self.left_arm = AlohaArm(
                port=config.left_port,
                arm_id=f"{config.id}_left" if config.id else "left",
                calibration=self._get_arm_calibration("left"),
                disable_torque_on_disconnect=config.left_disable_torque_on_disconnect,
                max_relative_target=config.left_max_relative_target,
                use_raw_positions=config.use_raw_positions,
            )

        if config.use_right_arm:
            self.right_arm = AlohaArm(
                port=config.right_port,
                arm_id=f"{config.id}_right" if config.id else "right",
                calibration=self._get_arm_calibration("right"),
                disable_torque_on_disconnect=config.right_disable_torque_on_disconnect,
                max_relative_target=config.right_max_relative_target,
                use_raw_positions=config.use_raw_positions,
            )

        self.cameras = make_cameras_from_configs(config.cameras)

    def _get_arm_calibration(self, arm_name: str) -> dict | None:
        """Extract calibration for a specific arm from the combined calibration or individual files."""
        import draccus
        from lerobot.motors import MotorCalibration
        
        # First, try to load from individual arm calibration file (e.g., aloha_left.json, aloha_right.json)
        arm_calibration_fpath = self.calibration_dir / f"{self.id}_{arm_name}.json"
        if arm_calibration_fpath.is_file():
            logger.info(f"Loading calibration from {arm_calibration_fpath}")
            with open(arm_calibration_fpath) as f, draccus.config_type("json"):
                return draccus.load(dict[str, MotorCalibration], f)
        
        # Fall back to extracting from combined calibration
        if not self.calibration:
            return None

        arm_calibration = {}
        prefix = f"{arm_name}_"
        for key, value in self.calibration.items():
            if key.startswith(prefix):
                motor_name = key[len(prefix):]
                arm_calibration[motor_name] = value

        return arm_calibration if arm_calibration else None

    @property
    def _motors_ft(self) -> dict[str, type]:
        """Get motor features for enabled arms with left/right prefixes."""
        features = {}
        if self.left_arm is not None:
            features.update({f"left_{motor}.pos": float for motor in self.left_arm.bus.motors})
        if self.right_arm is not None:
            features.update({f"right_{motor}.pos": float for motor in self.right_arm.bus.motors})
        return features

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        """Get camera features."""
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3)
            for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        arms_connected = True
        if self.left_arm is not None:
            arms_connected = arms_connected and self.left_arm.is_connected
        if self.right_arm is not None:
            arms_connected = arms_connected and self.right_arm.is_connected
        return arms_connected and all(cam.is_connected for cam in self.cameras.values())

    def connect(self, calibrate: bool = True) -> None:
        """Connect enabled follower arms and all cameras."""
        if self.left_arm is not None:
            self.left_arm.connect(calibrate)
        if self.right_arm is not None:
            self.right_arm.connect(calibrate)

        for cam in self.cameras.values():
            cam.connect()

        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        calibrated = True
        if self.left_arm is not None:
            calibrated = calibrated and self.left_arm.is_calibrated
        if self.right_arm is not None:
            calibrated = calibrated and self.right_arm.is_calibrated
        return calibrated

    def calibrate(self) -> None:
        """Calibrate enabled arms."""
        left_calib = self.left_arm.calibrate() if self.left_arm is not None else None
        right_calib = self.right_arm.calibrate() if self.right_arm is not None else None

        # Combine calibrations with prefixes
        self.calibration = {}
        if left_calib:
            for key, value in left_calib.items():
                self.calibration[f"left_{key}"] = value
        if right_calib:
            for key, value in right_calib.items():
                self.calibration[f"right_{key}"] = value

        if self.calibration:
            self._save_calibration()

    def configure(self) -> None:
        """Configure enabled arms."""
        if self.left_arm is not None:
            self.left_arm.configure()
        if self.right_arm is not None:
            self.right_arm.configure()

    def get_observation(self) -> dict[str, Any]:
        """Get observations from enabled arms and all cameras."""
        obs_dict = {}

        # Get left arm observations with "left_" prefix
        if self.left_arm is not None:
            left_obs = self.left_arm.get_observation()
            obs_dict.update({f"left_{key}": value for key, value in left_obs.items()})

        # Get right arm observations with "right_" prefix
        if self.right_arm is not None:
            right_obs = self.right_arm.get_observation()
            obs_dict.update({f"right_{key}": value for key, value in right_obs.items()})

        # Capture images from cameras
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.async_read()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        return obs_dict

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """
        Send action commands to enabled arms.

        Args:
            action: Dictionary with keys like "left_waist.pos", "right_gripper.pos", etc.

        Returns:
            The actions actually sent to the motors, potentially clipped.
        """
        result = {}

        # Split and send actions for left arm
        if self.left_arm is not None:
            left_action = {
                key.removeprefix("left_"): value
                for key, value in action.items()
                if key.startswith("left_")
            }
            # Only send if there are actions for this arm
            if left_action:
                sent_left = self.left_arm.send_action(left_action)
                result.update({f"left_{k}": v for k, v in sent_left.items()})

        # Split and send actions for right arm
        if self.right_arm is not None:
            right_action = {
                key.removeprefix("right_"): value
                for key, value in action.items()
                if key.startswith("right_")
            }
            # Only send if there are actions for this arm
            if right_action:
                sent_right = self.right_arm.send_action(right_action)
                result.update({f"right_{k}": v for k, v in sent_right.items()})

        return result

    def disconnect(self) -> None:
        """Disconnect enabled arms and all cameras."""
        if self.left_arm is not None:
            self.left_arm.disconnect()
        if self.right_arm is not None:
            self.right_arm.disconnect()

        for cam in self.cameras.values():
            cam.disconnect()

        logger.info(f"{self} disconnected.")

