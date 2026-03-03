"""
xArm simulation follower via built-in MuJoCo ZMQ server.

Provides the same interface as xarm_follower so teleoperation with GELLO
behaves identically in sim and on the real robot (joints in degrees,
gripper in mm: 0=closed, 800=open).
"""

import logging
import math
import pickle
from functools import cached_property
from typing import Any

import numpy as np

from lerobot.cameras.utils import make_cameras_from_configs

from ..robot import Robot
from .config_xarm_sim_follower import XarmSimFollowerConfig

logger = logging.getLogger(__name__)


class _ZMQSimClient:
    """Minimal ZMQ client matching gello_software's ZMQClientRobot protocol."""

    def __init__(self, host: str, port: int):
        import zmq

        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REQ)
        self._socket.setsockopt(zmq.RCVTIMEO, 2000)
        self._socket.connect(f"tcp://{host}:{port}")

    def num_dofs(self) -> int:
        request = {"method": "num_dofs"}
        self._socket.send(pickle.dumps(request))
        return pickle.loads(self._socket.recv())

    def get_joint_state(self) -> np.ndarray:
        request = {"method": "get_joint_state"}
        self._socket.send(pickle.dumps(request))
        result = pickle.loads(self._socket.recv())
        if isinstance(result, dict) and "error" in result:
            raise RuntimeError(result["error"])
        return np.asarray(result)

    def command_joint_state(self, joint_state: np.ndarray) -> None:
        request = {
            "method": "command_joint_state",
            "args": {"joint_state": joint_state},
        }
        self._socket.send(pickle.dumps(request))
        pickle.loads(self._socket.recv())

    def close(self) -> None:
        self._socket.close()
        self._context.term()


class XarmSimFollower(Robot):
    """
    xArm simulation follower via gello_software ZMQ server.

    Connects to launch_nodes.py (sim_xarm, sim_panda, sim_ur) and provides
    the same interface as XarmFollower for lerobot-teleoperate.
    """

    config_class = XarmSimFollowerConfig
    name = "xarm_sim_follower"

    def __init__(self, config: XarmSimFollowerConfig):
        super().__init__(config)
        self.config = config
        self._client: _ZMQSimClient | None = None
        self.cameras = make_cameras_from_configs(config.cameras)

    @property
    def _joint_names(self) -> list[str]:
        names = [f"joint{i+1}" for i in range(self.config.dof)]
        if self.config.use_gripper:
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
        return self._client is not None and all(
            cam.is_connected for cam in self.cameras.values()
        )

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def connect(self, calibrate: bool = True) -> None:
        self._client = _ZMQSimClient(self.config.host, self.config.port)
        num_dofs = self._client.num_dofs()
        if num_dofs != self.config.dof + (1 if self.config.use_gripper else 0):
            logger.warning(
                f"Sim reports {num_dofs} DoFs, config has dof={self.config.dof} "
                f"use_gripper={self.config.use_gripper}. Adjust config if needed."
            )
        for cam in self.cameras.values():
            cam.connect()
        logger.info(
            f"{self} connected to sim at {self.config.host}:{self.config.port}"
        )

    def get_observation(self) -> dict[str, Any]:
        if self._client is None:
            raise RuntimeError("Robot not connected")
        joints_rad = self._client.get_joint_state()
        obs_dict = {}
        n_joints = min(len(joints_rad), self.config.dof)
        for i in range(n_joints):
            obs_dict[f"joint{i+1}.pos"] = math.degrees(joints_rad[i])
        for i in range(n_joints, self.config.dof):
            obs_dict[f"joint{i+1}.pos"] = 0.0
        # Gripper: match real xArm (0=closed, 800=open). MuJoCo driver joint 0-0.85 rad.
        if self.config.use_gripper and len(joints_rad) > self.config.dof:
            g_rad = joints_rad[self.config.dof]
            GRIPPER_MAX_RAD = 0.85
            grip_frac = max(0.0, min(1.0, g_rad / GRIPPER_MAX_RAD))
            g_mm = (1.0 - grip_frac) * self.config.gripper_open_mm
            obs_dict["gripper.pos"] = g_mm
        for cam_key, cam in self.cameras.items():
            obs_dict[cam_key] = cam.async_read()
        return obs_dict

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if self._client is None:
            raise RuntimeError("Robot not connected")
        joints_rad = np.zeros(self.config.dof + (1 if self.config.use_gripper else 0))
        for i in range(self.config.dof):
            joints_rad[i] = math.radians(action.get(f"joint{i+1}.pos", 0.0))
        # Gripper: match real xArm (0=closed, 800=open). Sim expects 0-1 -> scales to 0-255.
        if self.config.use_gripper and "gripper.pos" in action:
            g_mm = action["gripper.pos"]
            denom = self.config.gripper_open_mm - self.config.gripper_close_mm
            if abs(denom) > 1e-6:
                grip_frac = (g_mm - self.config.gripper_close_mm) / denom
                joints_rad[self.config.dof] = max(0.0, min(1.0, grip_frac))
        self._client.command_joint_state(joints_rad)
        return action

    def disconnect(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None
        for cam in self.cameras.values():
            cam.disconnect()
        logger.info(f"{self} disconnected.")
