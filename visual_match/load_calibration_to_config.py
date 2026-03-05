#!/usr/bin/env python3
"""
Load calibration data from residual_physics calibration folder and update
visual_match config JSON files (stationary_cam.json, wrist_cam.json).

The calibration folder contains:
- base.pkl: R_base2world, t_base2world (shared by all cameras)
- calibration_handeye_result.pkl: R_gripper2cam, t_gripper2cam (wrist camera only)
- intrinsics.npy: camera intrinsics, order = [fixed_cam_1, fixed_cam_2, ..., wrist_cam]
- rvecs.pkl: rvec_cam2board per fixed camera (keyed by serial number)
- tvecs.pkl: tvec_cam2board per fixed camera (keyed by serial number)

Usage:
    python load_calibration_to_config.py <stationary_cam_serial> <wrist_cam_serial> [--calib-dir PATH] [--config-dir PATH]
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np


def _to_list(arr):
    """Convert numpy array to nested list for JSON serialization."""
    return np.asarray(arr).tolist()


def load_calibration(calib_dir: Path):
    """Load all calibration data from the calibration directory."""
    calib_dir = Path(calib_dir)
    with open(calib_dir / "base.pkl", "rb") as f:
        base = pickle.load(f)
    with open(calib_dir / "calibration_handeye_result.pkl", "rb") as f:
        handeye = pickle.load(f)
    intrinsics = np.load(calib_dir / "intrinsics.npy", allow_pickle=True)
    with open(calib_dir / "rvecs.pkl", "rb") as f:
        rvecs = pickle.load(f)
    with open(calib_dir / "tvecs.pkl", "rb") as f:
        tvecs = pickle.load(f)
    return base, handeye, intrinsics, rvecs, tvecs


def get_intrinsics_index(serial: str, rvecs: dict, wrist_serial: str) -> int:
    """
    Get the intrinsics array index for a camera serial.
    Order: fixed cameras (from rvecs keys) first, then wrist camera last.
    """
    fixed_serials = list(rvecs.keys())
    if serial == wrist_serial:
        return len(fixed_serials)
    if serial in fixed_serials:
        return fixed_serials.index(serial)
    raise ValueError(
        f"Serial {serial} not found. Fixed cameras: {fixed_serials}, wrist: {wrist_serial}"
    )


def update_stationary_cam_config(
    config_path: Path,
    calib_dir: Path,
    stationary_serial: str,
    wrist_serial: str,
) -> None:
    """Update stationary_cam.json with calibration data for the given stationary camera."""
    base, _, intrinsics, rvecs, tvecs = load_calibration(calib_dir)

    if stationary_serial not in rvecs:
        raise ValueError(
            f"Stationary camera serial {stationary_serial} must be a fixed camera. "
            f"Available fixed cameras: {list(rvecs.keys())}"
        )

    idx = get_intrinsics_index(stationary_serial, rvecs, wrist_serial)
    intr = intrinsics[idx]

    rvec = rvecs[stationary_serial]
    tvec = tvecs[stationary_serial]

    config = {
        "name": "stationary_cam",
        "type": "stationary",
        "serial_number": stationary_serial,
        "fps": 30,
        "width": 640,
        "height": 480,
        "intrinsics_640x480": _to_list(intr),
        "rvec_cam2board": _to_list(rvec),
        "tvec_cam2board": _to_list(tvec),
        "R_base2world": _to_list(base["R_base2world"]),
        "t_base2world": _to_list(base["t_base2world"]),
    }

    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)
    print(f"Updated {config_path}")


def update_wrist_cam_config(
    config_path: Path,
    calib_dir: Path,
    stationary_serial: str,
    wrist_serial: str,
) -> None:
    """Update wrist_cam.json with calibration data for the given wrist camera."""
    base, handeye, intrinsics, rvecs, tvecs = load_calibration(calib_dir)

    idx = get_intrinsics_index(wrist_serial, rvecs, wrist_serial)
    intr = intrinsics[idx]

    config = {
        "name": "wrist_cam",
        "type": "wrist",
        "serial_number": wrist_serial,
        "fps": 30,
        "width": 640,
        "height": 480,
        "intrinsics_640x480": _to_list(intr),
        "R_gripper2cam": _to_list(handeye["R_gripper2cam"]),
        "t_gripper2cam": _to_list(handeye["t_gripper2cam"]),
        "R_base2world": _to_list(base["R_base2world"]),
        "t_base2world": _to_list(base["t_base2world"]),
    }

    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)
    print(f"Updated {config_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Load calibration from residual_physics folder into visual_match config JSONs"
    )
    parser.add_argument(
        "stationary_cam_serial",
        type=str,
        help="Serial number of the stationary camera (e.g. 311322300308)",
    )
    parser.add_argument(
        "wrist_cam_serial",
        type=str,
        help="Serial number of the wrist camera (e.g. 213622251153)",
    )
    parser.add_argument(
        "--calib-dir",
        type=Path,
        default=Path("/home/tina/Documents/residual_physics-main/experiments/log/latest_calibration"),
        help="Path to calibration folder (base.pkl, intrinsics.npy, etc.)",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "configs",
        help="Path to config directory containing stationary_cam.json and wrist_cam.json",
    )
    args = parser.parse_args()

    config_dir = Path(args.config_dir)
    stationary_path = config_dir / "stationary_cam.json"
    wrist_path = config_dir / "wrist_cam.json"

    update_stationary_cam_config(
        stationary_path,
        args.calib_dir,
        args.stationary_cam_serial,
        args.wrist_cam_serial,
    )
    update_wrist_cam_config(
        wrist_path,
        args.calib_dir,
        args.stationary_cam_serial,
        args.wrist_cam_serial,
    )
    print("Done.")


if __name__ == "__main__":
    main()
