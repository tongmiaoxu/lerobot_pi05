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

from dataclasses import dataclass
from pathlib import Path

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("aloha_leader")
@dataclass
class AlohaLeaderConfig(TeleoperatorConfig):
    """
    Configuration for a single ALOHA leader arm.

    ALOHA leader arms use Dynamixel motors:
    - xm430-w350 for most joints
    - xl430-w250 for wrist_rotate
    - xc430-w150 for gripper
    """

    # Default calibration directory for ALOHA leader arms
    calibration_dir: Path | None = Path(".cache/calibration/aloha_leader")

    # Port to connect to the leader arm
    port: str = "/dev/ttyDXL_master_left"

    # Use raw motor positions instead of normalized positions
    # Must match follower's use_raw_positions setting!
    # Set to False for proper teleoperation (normalized positions work across different motor models)
    use_raw_positions: bool = False


@TeleoperatorConfig.register_subclass("bi_aloha_leader")
@dataclass
class BiAlohaLeaderConfig(TeleoperatorConfig):
    """
    Configuration for the ALOHA bimanual leader arms (for teleoperation).

    ALOHA is a bimanual robot developed by Trossen Robotics:
    https://www.trossenrobotics.com/aloha-stationary

    The leader arms are used for teleoperation to control the follower arms.
    Each arm has 9 motors (including shadow motors for shoulder and elbow).
    """

    # Default id for ALOHA teleoperator
    id: str = "aloha"

    # Default calibration directory for ALOHA leader arms
    # Contains aloha_left.json and aloha_right.json converted from old ALOHA calibration files
    calibration_dir: Path | None = Path(".cache/calibration/aloha_leader")

    # Which arms to use (set to False to disable an arm for single-arm operation)
    # Left arm disabled by default for testing
    use_left_arm: bool = True
    use_right_arm: bool = True

    # Use raw motor positions instead of normalized positions
    # Must match follower's use_raw_positions setting!
    # Set to False for proper teleoperation (normalized positions work across different motor models)
    use_raw_positions: bool = False

    # Ports to connect to the left and right leader arms
    left_port: str = "/dev/ttyDXL_master_left"
    right_port: str = "/dev/ttyDXL_master_right"

