"""
UFactory xArm follower robot implementation.

Controls an xArm robot via the xArm Python SDK over Ethernet.
Architecture matches gello_software's XArmRobot: a background thread
runs at a fixed frequency (default 50 Hz), reading the current joint
state and sending incremental servo commands toward the latest target
with norm-based velocity limiting.

Reference: gello_software/gello/robots/xarm_robot.py
"""

import logging
import threading
import time
from functools import cached_property
from typing import Any

import numpy as np

from lerobot.cameras.utils import make_cameras_from_configs

from ..robot import Robot
from .config_xarm_follower import XarmFollowerConfig

logger = logging.getLogger(__name__)


class _Rate:
    """Fixed-rate timer matching gello_software's Rate helper."""

    def __init__(self, duration: float):
        self.duration = duration
        self.last = time.time()

    def sleep(self) -> None:
        now = time.time()
        remaining = self.duration - (now - self.last)
        if remaining > 0.0001:
            time.sleep(remaining)
        self.last = time.time()


class XarmFollower(Robot):
    """
    [UFactory xArm](https://www.ufactory.cc/) robot controlled via the xArm Python SDK.

    Supports xArm5/6/7 with optional gripper. Connects over Ethernet.
    A background thread sends servo commands at a fixed rate, smoothly
    interpolating toward the latest target position.
    """

    config_class = XarmFollowerConfig
    name = "xarm_follower"

    def __init__(self, config: XarmFollowerConfig):
        super().__init__(config)
        self.config = config
        self.arm = None
        self._gripper_enabled = config.use_gripper
        self.cameras = make_cameras_from_configs(config.cameras)

        self._running = False
        self._command_thread: threading.Thread | None = None
        self._target_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._target_joints: np.ndarray | None = None
        self._target_gripper: float | None = None
        self._current_joints: np.ndarray | None = None
        self._current_gripper: float = 0.0

    @property
    def _joint_names(self) -> list[str]:
        names = [f"joint{i+1}" for i in range(self.config.dof)]
        if self._gripper_enabled:
            names.append("gripper")
        return names

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {f"{name}.pos": float for name in self._joint_names}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
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
        return self.arm is not None and all(cam.is_connected for cam in self.cameras.values())

    def _clear_error_states(self) -> None:
        """
        Clear errors and initialize xArm for servo mode.

        Matches gello_software's _clear_error_states exactly:
        clean errors → enable motion → mode 1 → collision off → state 0 → gripper init,
        with 1-second delays between each step for the controller to process.
        """
        if self.arm is None:
            return
        self.arm.clean_error()
        self.arm.clean_warn()
        self.arm.motion_enable(True)
        time.sleep(1)
        self.arm.set_mode(1)
        time.sleep(1)
        self.arm.set_collision_sensitivity(0)
        time.sleep(1)
        self.arm.set_state(state=0)
        time.sleep(1)
        if self._gripper_enabled:
            self.arm.set_gripper_enable(True)
            time.sleep(1)
            self.arm.set_gripper_mode(0)
            time.sleep(1)
            self.arm.set_gripper_speed(3000)
            time.sleep(1)

    def _get_gripper_pos_normalized(self) -> float:
        """Read gripper position and normalize to [0, 1] (0=open, 1=closed)."""
        if self.arm is None:
            return 0.0
        code, gripper_pos = self.arm.get_gripper_position()
        if code != 0 or gripper_pos is None:
            return 0.0
        return (gripper_pos - self.config.gripper_open_mm) / (
            self.config.gripper_close_mm - self.config.gripper_open_mm
        )

    def _control_loop(self) -> None:
        """
        Background thread: reads current state, computes delta toward target,
        limits delta norm to max_delta, and sends servo command.

        Matches gello_software's XArmRobot._robot_thread.
        """
        rate = _Rate(duration=1.0 / self.config.control_frequency)
        dof = self.config.dof

        while self._running:
            # Read current joint state
            code, servo_angle = self.arm.get_servo_angle(is_radian=True)
            if code != 0:
                logger.warning(f"get_servo_angle failed (code={code}), clearing errors")
                self._clear_error_states()
                continue

            current = np.array(servo_angle[:dof])
            with self._state_lock:
                self._current_joints = current
                self._current_gripper = self._get_gripper_pos_normalized()

            with self._target_lock:
                target = self._target_joints.copy()
                gripper_cmd = self._target_gripper

            # Compute delta and limit by L2 norm
            delta = target - current
            norm = np.linalg.norm(delta)
            if norm > self.config.max_delta:
                delta = delta / norm * self.config.max_delta

            # Send servo command
            command = current + delta
            ret = self.arm.set_servo_angle_j(command.tolist(), wait=False, is_radian=True)
            if ret in (1, 9):
                self._clear_error_states()

            # Gripper
            if gripper_cmd is not None and self._gripper_enabled:
                gripper_mm = (
                    self.config.gripper_open_mm
                    + gripper_cmd * (self.config.gripper_close_mm - self.config.gripper_open_mm)
                )
                self.arm.set_gripper_position(gripper_mm, wait=False)

            # Re-read state after command (matches gello_software)
            code2, servo_angle2 = self.arm.get_servo_angle(is_radian=True)
            if code2 == 0:
                with self._state_lock:
                    self._current_joints = np.array(servo_angle2[:dof])

            rate.sleep()

    def connect(self, calibrate: bool = True) -> None:
        from xarm.wrapper import XArmAPI

        self.arm = XArmAPI(self.config.ip, is_radian=True)
        self._clear_error_states()

        # Initialize target to current position (robot stays put until teleop starts)
        code, servo_angle = self.arm.get_servo_angle(is_radian=True)
        if code != 0:
            raise RuntimeError(f"Failed to read xArm joint angles (code={code})")
        self._current_joints = np.array(servo_angle[: self.config.dof])
        self._target_joints = self._current_joints.copy()
        self._target_gripper = None

        # Start background control thread
        self._running = True
        self._command_thread = threading.Thread(target=self._control_loop, daemon=True)
        self._command_thread.start()

        for cam in self.cameras.values():
            cam.connect()

        logger.info(f"{self} connected to xArm at {self.config.ip}")

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def get_observation(self) -> dict[str, Any]:
        obs_dict = {}

        with self._state_lock:
            joints = self._current_joints
            gripper = self._current_gripper

        if joints is not None:
            for i in range(self.config.dof):
                obs_dict[f"joint{i+1}.pos"] = np.degrees(joints[i])
        else:
            for i in range(self.config.dof):
                obs_dict[f"joint{i+1}.pos"] = 0.0

        if self._gripper_enabled:
            # Convert normalized (0=open, 1=closed) back to mm
            gripper_mm = (
                self.config.gripper_open_mm
                + gripper * (self.config.gripper_close_mm - self.config.gripper_open_mm)
            )
            obs_dict["gripper.pos"] = gripper_mm

        for cam_key, cam in self.cameras.items():
            obs_dict[cam_key] = cam.async_read()

        return obs_dict

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """
        Set the target joint positions. The background control thread will
        smoothly move the xArm toward this target.
        """
        goal_deg = []
        for i in range(self.config.dof):
            goal_deg.append(action.get(f"joint{i+1}.pos", 0.0))

        goal_rad = np.deg2rad(goal_deg)

        with self._target_lock:
            self._target_joints = goal_rad

            if self._gripper_enabled and "gripper.pos" in action:
                # Convert mm to normalized [0,1] (0=open, 1=closed)
                gripper_mm = action["gripper.pos"]
                if abs(self.config.gripper_open_mm - self.config.gripper_close_mm) > 1e-6:
                    pct = (gripper_mm - self.config.gripper_open_mm) / (
                        self.config.gripper_close_mm - self.config.gripper_open_mm
                    )
                    self._target_gripper = max(0.0, min(1.0, pct))
                else:
                    self._target_gripper = 0.0

        return action

    def disconnect(self) -> None:
        self._running = False
        if self._command_thread is not None:
            self._command_thread.join(timeout=3.0)
            self._command_thread = None

        if self.arm is not None:
            self.arm.set_mode(0)
            self.arm.set_state(state=0)
            self.arm.disconnect()
            self.arm = None

        for cam in self.cameras.values():
            cam.disconnect()

        logger.info(f"{self} disconnected.")
