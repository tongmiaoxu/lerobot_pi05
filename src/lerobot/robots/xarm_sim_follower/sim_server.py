"""
Self-contained MuJoCo sim server for xarm_sim_follower teleoperation.

Uses the same ZMQ protocol as gello_software's run_env.py so xarm_sim_follower
can connect. No dependency on gello_software.

Usage:
    python -m lerobot.robots.xarm_sim_follower.sim_server --port 6001

Or from lerobot-teleoperate with --robot.launch_sim=true.
"""

import os
import pickle
import threading
import time
from pathlib import Path
from typing import Any, Dict

import mujoco
import numpy as np

try:
    import mujoco.viewer
except ImportError:
    mujoco.viewer = None  # type: ignore

try:
    import zmq
except ImportError:
    zmq = None


def _get_xarm7_path() -> Path:
    """Resolve xarm7 model path relative to this package.
    """
    pkg_dir = Path(__file__).resolve().parent
    project_root = pkg_dir.parents[3]
    return project_root / "xarm7" / "scene.xml"


class _ZMQRobotServer:
    """ZMQ server implementing the robot protocol (num_dofs, get_joint_state, command_joint_state, get_observations)."""

    def __init__(self, robot: "MujocoSimServer", host: str = "127.0.0.1", port: int = 6001):
        self._robot = robot
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REP)
        self._socket.bind(f"tcp://{host}:{port}")
        self._stop = threading.Event()
        self._socket.setsockopt(zmq.RCVTIMEO, 1000)

    def serve(self) -> None:
        while not self._stop.is_set():
            try:
                msg = self._socket.recv()
                req = pickle.loads(msg)
                method = req.get("method")
                args = req.get("args", {})
                if method == "num_dofs":
                    result = self._robot.num_dofs()
                elif method == "get_joint_state":
                    result = self._robot.get_joint_state()
                elif method == "command_joint_state":
                    self._robot.command_joint_state(**args)
                    result = None
                elif method == "get_observations":
                    result = self._robot.get_observations()
                else:
                    result = {"error": f"Invalid method: {method}"}
                self._socket.send(pickle.dumps(result))
            except zmq.Again:
                pass

    def stop(self) -> None:
        self._stop.set()
        self._socket.close()
        self._context.term()


class MujocoSimServer:
    """
    MuJoCo xArm7 sim with ZMQ server. Compatible with xarm_sim_follower client.
    """

    def __init__(
        self,
        xml_path: str | Path | None = None,
        host: str = "127.0.0.1",
        port: int = 6001,
    ):
        if zmq is None:
            raise ImportError("pyzmq is required for sim server. Install with: pip install pyzmq")
        xml_path = xml_path or _get_xarm7_path()
        xml_path = Path(xml_path)
        if not xml_path.exists():
            raise FileNotFoundError(f"Model not found: {xml_path}")

        xarm_dir = xml_path.parent
        orig_cwd = os.getcwd()
        try:
            os.chdir(str(xarm_dir))
            self._model = mujoco.MjModel.from_xml_path(xml_path.name)
        finally:
            os.chdir(orig_cwd)

        self._data = mujoco.MjData(self._model)
        self._num_joints = self._model.nu
        self._joint_state = np.zeros(self._num_joints)
        self._joint_cmd = np.array([0.0, 0.0, 0.0, 1.57, 0.0, 1.57, 0.0, 0.0])[: self._num_joints].copy()
        self._joint_state = self._joint_cmd.copy()

        self._zmq = _ZMQRobotServer(self, host=host, port=port)
        self._zmq_thread = threading.Thread(target=self._zmq.serve, daemon=True)

    def num_dofs(self) -> int:
        return self._num_joints

    def get_joint_state(self) -> np.ndarray:
        return self._joint_state.copy()

    def command_joint_state(self, joint_state: np.ndarray) -> None:
        if len(joint_state) != self._num_joints:
            raise ValueError(f"Expected {self._num_joints} joints, got {len(joint_state)}")
        cmd = np.asarray(joint_state, dtype=np.float64)
        if self._num_joints == 8:
            cmd = cmd.copy()
            cmd[-1] = cmd[-1] * 255
        self._joint_cmd = cmd

    def get_observations(self) -> Dict[str, np.ndarray]:
        n = self._num_joints
        return {
            "joint_positions": self._data.qpos[:n].copy(),
            "joint_velocities": self._data.qvel[:n].copy(),
            "ee_pos_quat": np.zeros(7),
            "gripper_position": self._data.qpos[n - 1].copy() if n > 0 else 0.0,
        }

    def serve(self) -> None:
        self._zmq_thread.start()
        if mujoco.viewer is None:
            raise ImportError("mujoco.viewer required for sim. Ensure mujoco supports viewer.")
        with mujoco.viewer.launch_passive(self._model, self._data) as viewer:
            while viewer.is_running():
                t0 = time.time()
                self._data.ctrl[:] = self._joint_cmd
                mujoco.mj_step(self._model, self._data)
                self._joint_state = self._data.qpos[: self._num_joints].copy()
                with viewer.lock():
                    viewer.sync()
                dt = self._model.opt.timestep - (time.time() - t0)
                if dt > 0:
                    time.sleep(dt)
        self._zmq.stop()


def main():
    import argparse

    p = argparse.ArgumentParser(description="Launch xArm7 MuJoCo sim server for teleoperation")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=6001)
    p.add_argument("--xml", default=None, help="Path to scene.xml (default: project xarm7/scene.xml)")
    args = p.parse_args()
    server = MujocoSimServer(host=args.host, port=args.port, xml_path=args.xml)
    print(f"xArm7 sim server on {args.host}:{args.port}")
    server.serve()


if __name__ == "__main__":
    main()
