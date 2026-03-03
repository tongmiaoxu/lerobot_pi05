from dataclasses import dataclass, field
from pathlib import Path

from lerobot.cameras import CameraConfig

from ..config import RobotConfig


@RobotConfig.register_subclass("xarm_sim_follower")
@dataclass
class XarmSimFollowerConfig(RobotConfig):
    """
    Configuration for xArm simulation via built-in MuJoCo ZMQ server.

    Connects to a robot server started by:
      python -m lerobot.robots.xarm_sim_follower.sim_server --port 6001

    Use with lerobot-teleoperate for teleop in simulation:
      lerobot-teleoperate --robot.type=xarm_sim_follower --teleop.type=gello_leader ...

    Optionally set --robot.launch_sim=true to spawn the sim in a subprocess.
    """

    id: str = "xarm_sim"

    # Sim does not use calibration; set None to avoid creating empty dir
    calibration_dir: Path | None = None

    # ZMQ server address (must match launch_nodes.py)
    host: str = "127.0.0.1"
    port: int = 6001

    # Degrees of freedom (xarm7 = 7 arm joints)
    dof: int = 7

    # Built-in sim uses xarm7.xml with gripper (8 actuators)
    use_gripper: bool = True

    # Gripper range in mm (used when use_gripper=True)
    gripper_open_mm: float = 800.0
    gripper_close_mm: float = 0.0

    # Cameras (sim typically has none; use empty dict)
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    # If True, spawn built-in MuJoCo sim server before connecting
    launch_sim: bool = False
