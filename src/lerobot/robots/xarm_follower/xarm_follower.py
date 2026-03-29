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
        self._prev_command: np.ndarray | None = None
        # Offset between GELLO's initial pose and xArm's current pose.
        # Computed on the first send_action call so the robot stays put
        # until the operator actually moves the GELLO.
        self._action_offset: np.ndarray | None = None
        # Target pose for lerobot-record RIGHT ARROW resets (rad + normalized gripper).
        self._deployment_reset_joints_rad: np.ndarray | None = None
        self._deployment_reset_gripper_norm: float | None = None

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
        #Use 4 for velocity control, 0 for position control.
        mode = {
            "velocity": 4,
            "position": 0,
        }.get(self.config.control_mode, 1)
        self.arm.set_mode(mode)
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

    @property
    def _cur_gripper_pos_mm(self) -> float | None:
        """Current gripper position in mm, or None if unavailable."""
        if self.arm is None:
            return None
        code, gripper_pos = self.arm.get_gripper_position()
        if code != 0 or gripper_pos is None:
            return None
        return gripper_pos

    def _check_code(self, code: int, op_name: str) -> bool:
        """Check xArm API return code. Returns True if successful."""
        if code == 0:
            return True
        logger.warning(f"{op_name} failed (code={code})")
        return False

    def velocity_control(
        self,
        next_state: np.ndarray,
        current_state: np.ndarray,
        ignore_error: bool = True,
    ) -> None:
        """Execute velocity control by sending streaming velocity targets to robot."""
        # NOTE: velocity control don't use ema
        # next_joints = ema_factor * next_joints + (1 - ema_factor) * current_joints

        # NOTE: delta for velocity control
        next_joints = next_state[0 : self.config.dof] - current_state[0 : self.config.dof]

        # denormalize gripper position
        if self._gripper_enabled and len(next_state) > self.config.dof:
            gripper_pos = next_state[-1]
            denormalized_gripper_pos = (
                gripper_pos * (self.config.gripper_close_mm - self.config.gripper_open_mm)
                + self.config.gripper_open_mm
            )

        if not self._running or self.arm is None:
            raise ValueError("Robot is not alive!")
        if (
            self._gripper_enabled
            and len(next_state) > self.config.dof
            and self._cur_gripper_pos_mm is not None
        ):
            isclose = np.isclose(self._cur_gripper_pos_mm, denormalized_gripper_pos)
            if not isclose:
                self.arm.set_gripper_position(denormalized_gripper_pos, wait=False)

        v = next_joints * self.config.velocity_control_scale
        v = v.tolist()
        code = self.arm.vc_set_joint_velocity(v, is_radian=True, is_sync=False, duration=0)

        if not self._check_code(code, "vc_set_joint_velocity"):
            raise ValueError("velocity control error")
        if ignore_error:
            self.arm.clean_error()
            self.arm.clean_warn()

    def position_control(
        self,
        next_arm_goal: np.ndarray,
        prev_arm_goal: np.ndarray,
        next_gripper: float | None = None,
        ema_factor: float | None = None,
        ignore_error: bool = True,
    ) -> None:
        """Execute position control by sending streaming position/servo targets with EMA smoothing."""
        if ema_factor is None:
            ema_factor = self.config.ema_factor
        next_arm_goal = ema_factor * next_arm_goal + (1 - ema_factor) * prev_arm_goal

        # denormalize gripper position
        if self._gripper_enabled:
            assert next_gripper is not None, "next_gripper must be provided when gripper_enable is True"
            denormalized_gripper_pos = (
                next_gripper * (self.config.gripper_close_mm - self.config.gripper_open_mm)
                + self.config.gripper_open_mm
            )

        if not self._running or self.arm is None:
            raise ValueError("Robot is not alive!")

        if (
            self._gripper_enabled
            and len(next_arm_goal) == self.config.dof
            and self._cur_gripper_pos_mm is not None
        ):
            if np.abs(self._cur_gripper_pos_mm - denormalized_gripper_pos) > 5.0:
                self.arm.set_gripper_position(denormalized_gripper_pos, wait=False, speed=2500)

        next_arm_goal = next_arm_goal.tolist()
        code = self.arm.set_servo_angle_j(angles=next_arm_goal, is_radian=True, wait=False)

        if not self._check_code(code, "set_servo_angle_j"):
            raise ValueError("position control error")
        if ignore_error:
            self.arm.clean_error()
            self.arm.clean_warn()

    def _control_loop(self) -> None:
        """
        Background thread: reads current state and sends commands.
        Uses position control (EMA smoothing) or velocity control based on config.
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

            # # --- Safety: clamp target so joint delta doesn't exceed max_delta ---
            delta = target - current
            delta_norm = np.linalg.norm(delta)
            if delta_norm > self.config.max_delta:
                logger.warning(
                    f"Joint delta {delta_norm:.4f} rad exceeds max_delta "
                    f"{self.config.max_delta:.4f}, clamping."
                )
                target = current + delta * (self.config.max_delta / delta_norm)

            try:
                if self.config.control_mode == "velocity":
                    # Velocity control: send (target - current) as velocity
                    grip_target = (
                        gripper_cmd if gripper_cmd is not None else self._current_gripper
                    )
                    next_state = np.concatenate([target, [grip_target]])
                    current_state = np.concatenate([current, [self._current_gripper]])
                    self.velocity_control(next_state, current_state, ignore_error=True)
                else:
                    # Position control (default): EMA smoothing + set_servo_angle_j
                    prev = self._prev_command if self._prev_command is not None else current
                    self.position_control(
                        target,
                        prev,
                        next_gripper=gripper_cmd if gripper_cmd is not None else self._current_gripper,
                        ignore_error=True,
                    )
                    self._prev_command = (
                        self.config.ema_factor * target
                        + (1 - self.config.ema_factor) * prev
                    )
            except ValueError:
                self._clear_error_states()
                self._prev_command = current.copy()

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
        self._prev_command = self._current_joints.copy()
        self._target_gripper = None

        if self.config.deployment_reset_joint_deg is not None:
            deg = np.array(self.config.deployment_reset_joint_deg, dtype=float)
            if deg.size != self.config.dof:
                raise ValueError(
                    f"deployment_reset_joint_deg must have length {self.config.dof}, got {deg.size}"
                )
            self._deployment_reset_joints_rad = np.deg2rad(deg)
        else:
            self._deployment_reset_joints_rad = self._current_joints.copy()

        if self._gripper_enabled:
            if self.config.deployment_reset_gripper_mm is not None:
                mm = self.config.deployment_reset_gripper_mm
                span = self.config.gripper_close_mm - self.config.gripper_open_mm
                if abs(span) > 1e-6:
                    self._deployment_reset_gripper_norm = float(
                        max(0.0, min(1.0, (mm - self.config.gripper_open_mm) / span))
                    )
                else:
                    self._deployment_reset_gripper_norm = 0.0
            else:
                self._deployment_reset_gripper_norm = self._get_gripper_pos_normalized()
        else:
            self._deployment_reset_gripper_norm = None

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

        On the first call an offset is computed between the incoming GELLO
        position and the xArm's current position.  This offset is subtracted
        from every subsequent command so the robot stays in place until the
        operator actually moves the GELLO.
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

    def go_to_deployment_initial_pose(self, timeout_s: float = 120.0, position_tol_rad: float = 0.12) -> None:
        """
        Move toward the deployment "initial" pose (see config ``deployment_reset_*``
        or the pose captured at ``connect()``). Blocks until joints are within
        ``position_tol_rad`` (L2 norm) or ``timeout_s`` elapses.
        """
        if not self.is_connected or self._deployment_reset_joints_rad is None:
            return

        target = self._deployment_reset_joints_rad
        with self._target_lock:
            self._target_joints = target.copy()
            if self._gripper_enabled and self._deployment_reset_gripper_norm is not None:
                self._target_gripper = self._deployment_reset_gripper_norm

        deadline = time.time() + timeout_s
        while time.time() < deadline:
            with self._state_lock:
                cur = self._current_joints
            if cur is not None and np.linalg.norm(cur - target) <= position_tol_rad:
                with self._target_lock:
                    self._prev_command = cur.copy()
                return
            time.sleep(0.05)

        logger.warning(
            "go_to_deployment_initial_pose: timed out before reaching target "
            f"(timeout_s={timeout_s}, tol={position_tol_rad})"
        )

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
