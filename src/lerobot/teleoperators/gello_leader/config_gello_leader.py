import math
from dataclasses import dataclass, field
from pathlib import Path

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("gello_leader")
@dataclass
class GelloLeaderConfig(TeleoperatorConfig):
    """
    Configuration for a GELLO teleoperator for xArm.

    GELLO is a 3D-printed teleoperation device using Dynamixel XL330 motors.
    It mirrors the xArm's kinematic structure so joint angles map directly.

    Calibration uses the same approach as gello_software:
      1. Run `python scripts/gello_get_offset.py` with GELLO in a known pose
      2. The script computes joint_offsets and gripper_range_rad
      3. Values are saved to .cache/calibration/gello_leader/gello_offsets.json

    For best results, use --use-gello-software-offsets if your GELLO
    was previously calibrated with gello_software.

    Reference: https://github.com/wuphilipp/gello_software
    """

    id: str = "gello"

    calibration_dir: Path | None = Path(".cache/calibration/gello_leader")

    # USB serial port for the GELLO Dynamixel chain
    port: str = "/dev/ttyUSB0"

    # Number of robot joints (must match the xArm: 5, 6, or 7)
    dof: int = 7

    # Whether the GELLO has a gripper lever/trigger (extra motor after the joints)
    use_gripper: bool = True

    # Dynamixel motor model for joints (typically xl330-m288)
    motor_model: str = "xl330-m288"

    # Gripper motor model (often xl330-m077, different from the joint motors)
    gripper_motor_model: str = "xl330-m077"

    # Starting motor ID (IDs are sequential: start_id, start_id+1, ...)
    start_motor_id: int = 1

    # Per-joint sign correction: +1 or -1 to handle reversed motor mounting.
    # Length must equal dof.
    joint_signs: list[int] = field(default_factory=lambda: [1, 1, 1, 1, 1, 1, 1])

    # Per-joint offset in RADIANS.
    # Computed by scripts/gello_get_offset.py.
    # Corrected angle = sign * (raw_radians - offset).
    joint_offsets: list[float] = field(
        default_factory=lambda: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    )

    # The known xArm joint angles (in radians) that GELLO is placed in
    # during calibration. Used by gello_get_offset.py to compute offsets.
    start_joints: list[float] = field(
        default_factory=lambda: [0.0, 0.0, 0.0, math.pi / 2, 0.0, math.pi / 2, 0.0]
    )

    # xArm gripper output range in mm.
    # Matches gello_software: GRIPPER_OPEN=800, GRIPPER_CLOSE=0
    gripper_open_mm: float = 800.0
    gripper_close_mm: float = 0.0
