"""
GELLO teleoperator implementation for xArm teleoperation.

GELLO is a 3D-printed teleoperation device using Dynamixel XL330 motors
that mirrors the xArm's kinematic structure. Joint angles from GELLO
are read as raw encoder values and converted to xArm joint commands.

Calibration follows the gello_software approach:
  corrected_angle = joint_sign * (raw_radians - offset)
  offset is computed by scripts/gello_get_offset.py

Includes exponential smoothing (alpha=0.99) to reduce Dynamixel noise,
matching gello_software/gello/robots/dynamixel.py.

Reference: https://github.com/wuphilipp/gello_software
"""

import json
import logging
import math
import time
from functools import cached_property
from typing import Any

import numpy as np

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.dynamixel import DynamixelMotorsBus
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..teleoperator import Teleoperator
from .config_gello_leader import GelloLeaderConfig

logger = logging.getLogger(__name__)

PULSES_PER_REV = 4096


def _pulses_to_rad(pulses: int) -> float:
    return pulses / PULSES_PER_REV * 2 * math.pi


class GelloLeader(Teleoperator):
    """
    [GELLO](https://github.com/wuphilipp/gello_software) teleoperator for xArm.

    Reads raw encoder positions from Dynamixel XL330 motors, converts to
    radians, applies sign and offset corrections, and outputs xArm joint
    angles in degrees. Exponential smoothing (alpha=0.99) reduces noise.
    """

    config_class = GelloLeaderConfig
    name = "gello_leader"

    def __init__(self, config: GelloLeaderConfig):
        super().__init__(config)
        self.config = config

        if len(config.joint_signs) != config.dof:
            raise ValueError(
                f"joint_signs length ({len(config.joint_signs)}) must equal dof ({config.dof})"
            )
        if len(config.joint_offsets) != config.dof:
            raise ValueError(
                f"joint_offsets length ({len(config.joint_offsets)}) must equal dof ({config.dof})"
            )

        motors = {}
        for i in range(config.dof):
            motors[f"joint{i+1}"] = Motor(
                config.start_motor_id + i,
                config.motor_model,
                MotorNormMode.DEGREES,
            )
        if config.use_gripper:
            motors["gripper"] = Motor(
                config.start_motor_id + config.dof,
                config.gripper_motor_model,
                MotorNormMode.DEGREES,
            )

        self.bus = DynamixelMotorsBus(
            port=config.port,
            motors=motors,
            calibration=None,
        )

        self._offsets = list(config.joint_offsets)
        self._signs = list(config.joint_signs)
        self._gripper_open_close_rad: tuple[float, float] | None = None

        # Exponential smoothing state (matches gello_software alpha=0.99)
        self._alpha = 0.99
        self._last_pos: np.ndarray | None = None

        # Load calibration file (written by gello_get_offset.py) if it exists
        offset_file = self.calibration_dir / "gello_offsets.json"
        if offset_file.is_file():
            logger.info(f"Loading GELLO calibration from {offset_file}")
            with open(offset_file) as f:
                data = json.load(f)
            if "joint_offsets" in data:
                self._offsets = data["joint_offsets"]
            if "joint_signs" in data:
                self._signs = data["joint_signs"]
            if "gripper_range_rad" in data:
                gr = data["gripper_range_rad"]
                self._gripper_open_close_rad = (gr[1], gr[0])  # (open, closed)
            elif "gripper_open_deg" in data and "gripper_close_deg" in data:
                self._gripper_open_close_rad = (
                    math.radians(data["gripper_open_deg"]),
                    math.radians(data["gripper_close_deg"]),
                )

    @cached_property
    def action_features(self) -> dict[str, type]:
        names = [f"joint{i+1}" for i in range(self.config.dof)]
        if self.config.use_gripper:
            names.append("gripper")
        return {f"{n}.pos": float for n in names}

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
        self.configure()
        logger.info(f"{self} connected on port {self.config.port}")

        if all(o == 0.0 for o in self._offsets):
            logger.warning(
                "GELLO joint offsets are all zero. If you haven't calibrated yet, run:\n"
                "  python scripts/gello_get_offset.py "
                f"--port {self.config.port} --dof {self.config.dof}"
            )

    @property
    def is_calibrated(self) -> bool:
        return not all(o == 0.0 for o in self._offsets)

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        """Disable torque so GELLO can be freely moved by hand."""
        self.bus.disable_torque()

    def get_action(self) -> dict[str, float]:
        """
        Read GELLO joint positions and convert to xArm-compatible actions.

        Applies the gello_software formula: corrected = sign * (raw_rad - offset)
        then exponential smoothing, then gripper normalization.
        Outputs joint angles in degrees, gripper in mm.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        start = time.perf_counter()
        raw = self.bus.sync_read("Present_Position", normalize=False)

        dof = self.config.dof
        n_total = dof + (1 if self.config.use_gripper else 0)
        pos = np.zeros(n_total)

        for i in range(dof):
            raw_rad = _pulses_to_rad(raw[f"joint{i+1}"])
            pos[i] = self._signs[i] * (raw_rad - self._offsets[i])

        if self.config.use_gripper:
            pos[dof] = _pulses_to_rad(raw["gripper"])

        # Exponential smoothing (matches gello_software dynamixel.py)
        if self._last_pos is None:
            self._last_pos = pos.copy()
        else:
            pos = self._last_pos * (1 - self._alpha) + pos * self._alpha
            self._last_pos = pos.copy()

        # Build action dict — joints in degrees
        action = {}
        for i in range(dof):
            action[f"joint{i+1}.pos"] = math.degrees(pos[i])

        if self.config.use_gripper:
            gripper_rad = pos[dof]
            if self._gripper_open_close_rad is not None:
                open_rad, close_rad = self._gripper_open_close_rad
                denom = close_rad - open_rad
                if abs(denom) > 1e-6:
                    # g_pos: 0 = open, 1 = closed (matches gello_software)
                    g_pos = (gripper_rad - open_rad) / denom
                    g_pos = max(0.0, min(1.0, g_pos))
                else:
                    g_pos = 0.0
            else:
                g_pos = 0.0

            # Map to mm: 0 (open) → gripper_open_mm, 1 (closed) → gripper_close_mm
            gripper_mm = (
                self.config.gripper_open_mm
                + g_pos * (self.config.gripper_close_mm - self.config.gripper_open_mm)
            )
            action["gripper.pos"] = gripper_mm

        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read action: {dt_ms:.1f}ms")
        return action

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        pass

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self.bus.disconnect()
        logger.info(f"{self} disconnected.")
