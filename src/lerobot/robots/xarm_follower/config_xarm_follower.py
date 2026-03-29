import json
from dataclasses import dataclass, field
from pathlib import Path

from lerobot.cameras import CameraConfig
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig

from ..config import RobotConfig


def _find_visual_match_configs_dir() -> Path | None:
    """Find visual_match/configs directory (project root has visual_match/ and src/)."""
    p = Path(__file__).resolve().parent
    for _ in range(10):
        configs_dir = p / "visual_match" / "configs"
        if configs_dir.exists():
            return configs_dir
        if p.parent == p:
            break
        p = p.parent
    return None


def _default_xarm_cameras() -> dict[str, CameraConfig]:
    """Load cameras from visual_match/configs/*.json (stationary_cam -> cam_high, wrist_cam -> cam_wrist)."""
    configs_dir = _find_visual_match_configs_dir()
    if configs_dir is None:
        return {}
    cameras: dict[str, CameraConfig] = {}
    mapping = [
        ("stationary_cam", "cam_high"),
        ("wrist_cam", "cam_wrist"),
    ]
    for config_name, lerobot_key in mapping:
        config_path = configs_dir / f"{config_name}.json"
        if not config_path.exists():
            continue
        with open(config_path) as f:
            cfg = json.load(f)
        cameras[lerobot_key] = RealSenseCameraConfig(
            serial_number_or_name=cfg["serial_number"],
            fps=cfg["fps"],
            width=cfg["width"],
            height=cfg["height"],
        )
    return cameras


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
    max_delta: float = 0.07

    # Control mode: "position" (default) uses set_servo_angle_j with EMA smoothing,
    # "velocity" uses vc_set_joint_velocity for streaming velocity targets.
    control_mode: str = "velocity"

    # Scale factor for velocity control (applied to joint delta before vc_set_joint_velocity).
    velocity_control_scale: float = 7.0

    # EMA factor for position control smoothing (0-1). Higher = more smoothing.
    ema_factor: float = 0.5

    # xArm gripper range in mm
    gripper_open_mm: float = 800.0
    gripper_close_mm: float = 0.0

    # Cameras (e.g., wrist camera, overhead camera).
    # Default: loaded from visual_match/configs/stationary_cam.json and wrist_cam.json
    # (cam_high, cam_wrist). To disable: --robot.cameras='{}'
    cameras: dict[str, CameraConfig] = field(default_factory=_default_xarm_cameras)

    # When using lerobot-record with the arrow-key state machine, each RIGHT ARROW
    # (start episode / save episode) can move the arm to this pose first.
    # If ``deployment_reset_joint_deg`` is None, the joint angles read at ``connect()``
    # are used (place the arm in your desired start pose before launching record).
    deployment_reset_joint_deg: list[float] | None = None
    # Optional gripper target in mm when resetting (open/close range as in gripper_*_mm).
    # If None, the gripper position at ``connect()`` is used.
    deployment_reset_gripper_mm: float | None = None
