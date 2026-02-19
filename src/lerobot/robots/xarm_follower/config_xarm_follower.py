from dataclasses import dataclass, field
from pathlib import Path

from lerobot.cameras import CameraConfig

from ..config import RobotConfig


@RobotConfig.register_subclass("xarm_follower")
@dataclass
class XarmFollowerConfig(RobotConfig):
    """
    Configuration for a UFactory xArm robot controlled via the xArm Python SDK.

    The xArm connects over Ethernet (IP address). Supports xArm5, xArm6, and xArm7.
    Optionally includes a gripper (xArm gripper or Robotiq).

    Control architecture matches gello_software: a background thread sends
    servo commands at `control_frequency` Hz, smoothly interpolating toward
    the latest target with norm-based `max_delta` limiting.

    Requires: pip install xarm-python-sdk
    """

    id: str = "xarm"

    calibration_dir: Path | None = Path(".cache/calibration/xarm_follower")

    # xArm IP address (set to your robot's IP)
    ip: str = "192.168.1.228"

    # Degrees of freedom: 5, 6, or 7 depending on your xArm model
    dof: int = 7

    use_gripper: bool = True

    # Background control thread frequency (Hz).
    # The thread continuously moves the xArm toward the latest target.
    control_frequency: float = 50.0

    # Maximum joint movement per control step (radians, L2 norm across all joints).
    # At 50 Hz with max_delta=0.05, the maximum total joint velocity is ~2.5 rad/s.
    # Matches gello_software's DEFAULT_MAX_DELTA.
    max_delta: float = 0.05

    # xArm gripper range in mm
    gripper_open_mm: float = 800.0
    gripper_close_mm: float = 0.0

    # Cameras (e.g., wrist camera, overhead camera)
    # To disable cameras, set cameras to empty dict: --robot.cameras='{}'
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
