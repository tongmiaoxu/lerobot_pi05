#!/usr/bin/env python3
"""
Connect to both RealSense cameras, read live factory intrinsics, load
extrinsic calibration from the residual_physics calibration folder, and
write complete visual_match config JSONs (stationary_cam.json, wrist_cam.json).

Intrinsics come directly from the cameras (pyrealsense2) — NOT from
intrinsics.npy in the calibration folder.

Usage:
    python tools/dump_camera_intrinsics.py [--calib-dir PATH] [--config-dir PATH]
"""

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pyrealsense2 as rs

# ── defaults ──────────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_CONFIG_DIR = _PROJECT_ROOT / "visual_match" / "configs"
DEFAULT_CALIB_DIR = Path(
    "/home/tina/Documents/residual_physics-main/experiments/log/latest_calibration"
)

CAMERAS = {
    "stationary_cam": {"key": "cam_high", "type": "stationary"},
    "wrist_cam":      {"key": "cam_wrist", "type": "wrist"},
}


# ── helpers ───────────────────────────────────────────────────────────────────
def _to_list(arr):
    """Convert numpy array to nested list for JSON serialization."""
    return np.asarray(arr).tolist()


def get_intrinsics(serial: str, width: int = 640, height: int = 480, fps: int = 30):
    """Start a pipeline at the requested resolution and return the color intrinsics."""
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)

    profile = pipeline.start(config)
    time.sleep(0.3)  # let it warm up briefly

    color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr = color_stream.get_intrinsics()

    pipeline.stop()
    return intr


def intrinsics_to_K(intr) -> list[list[float]]:
    """Convert pyrealsense2 intrinsics to a 3x3 matrix (nested list)."""
    return [
        [intr.fx, 0.0,     intr.ppx],
        [0.0,     intr.fy, intr.ppy],
        [0.0,     0.0,     1.0],
    ]


def load_calibration(calib_dir: Path):
    """Load extrinsic calibration data (base, handeye, rvecs, tvecs).

    Does NOT load intrinsics.npy — intrinsics come from the live cameras.
    """
    calib_dir = Path(calib_dir)

    with open(calib_dir / "base.pkl", "rb") as f:
        base = pickle.load(f)
    with open(calib_dir / "calibration_handeye_result.pkl", "rb") as f:
        handeye = pickle.load(f)
    with open(calib_dir / "rvecs.pkl", "rb") as f:
        rvecs = pickle.load(f)
    with open(calib_dir / "tvecs.pkl", "rb") as f:
        tvecs = pickle.load(f)

    return base, handeye, rvecs, tvecs


# ── per-camera config builders ────────────────────────────────────────────────
def build_stationary_config(
    serial: str,
    K: list[list[float]],
    distortion_model: str,
    distortion_coeffs: list[float],
    base: dict,
    rvecs: dict,
    tvecs: dict,
    width: int = 640,
    height: int = 480,
    fps: int = 30,
) -> dict:
    if serial not in rvecs:
        raise ValueError(
            f"Stationary camera serial {serial} not found in rvecs. "
            f"Available: {list(rvecs.keys())}"
        )
    return {
        "name": "stationary_cam",
        "type": "stationary",
        "serial_number": serial,
        "fps": fps,
        "width": width,
        "height": height,
        "intrinsics_640x480": K,
        "distortion_model": distortion_model,
        "distortion_coeffs": distortion_coeffs,
        "rvec_cam2board": _to_list(rvecs[serial]),
        "tvec_cam2board": _to_list(tvecs[serial]),
        "R_base2world": _to_list(base["R_base2world"]),
        "t_base2world": _to_list(base["t_base2world"]),
    }


def build_wrist_config(
    serial: str,
    K: list[list[float]],
    distortion_model: str,
    distortion_coeffs: list[float],
    base: dict,
    handeye: dict,
    width: int = 640,
    height: int = 480,
    fps: int = 30,
) -> dict:
    return {
        "name": "wrist_cam",
        "type": "wrist",
        "serial_number": serial,
        "fps": fps,
        "width": width,
        "height": height,
        "intrinsics_640x480": K,
        "distortion_model": distortion_model,
        "distortion_coeffs": distortion_coeffs,
        "R_gripper2cam": _to_list(handeye["R_gripper2cam"]),
        "t_gripper2cam": _to_list(handeye["t_gripper2cam"]),
        "R_base2world": _to_list(base["R_base2world"]),
        "t_base2world": _to_list(base["t_base2world"]),
    }


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description=(
            "Read live RealSense intrinsics, merge with extrinsic calibration, "
            "and write visual_match config JSONs."
        ),
    )
    parser.add_argument(
        "--calib-dir",
        type=Path,
        default=DEFAULT_CALIB_DIR,
        help="Path to calibration folder (base.pkl, rvecs.pkl, etc.)",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=DEFAULT_CONFIG_DIR,
        help="Output directory for config JSONs",
    )
    parser.add_argument(
        "--stationary-serial",
        type=str,
        default="311322300308",
        help="Serial number of the stationary camera",
    )
    parser.add_argument(
        "--wrist-serial",
        type=str,
        default="213622251153",
        help="Serial number of the wrist camera",
    )
    args = parser.parse_args()

    # ── load extrinsics from calibration folder ──
    print(f"[INFO] Loading extrinsic calibration from {args.calib_dir}")
    base, handeye, rvecs, tvecs = load_calibration(args.calib_dir)
    print(f"  Fixed cameras in rvecs: {list(rvecs.keys())}")

    # ── serials ──
    serials = {
        "stationary_cam": args.stationary_serial,
        "wrist_cam": args.wrist_serial,
    }

    args.config_dir.mkdir(parents=True, exist_ok=True)

    for cam_name, meta in CAMERAS.items():
        serial = serials[cam_name]
        w, h, fps = 640, 480, 30

        print(f"\n{'='*60}")
        print(f"Camera : {cam_name}  (lerobot key: {meta['key']})")
        print(f"Serial : {serial}")
        print(f"Stream : {w}x{h} @ {fps}fps")
        print(f"{'='*60}")

        # ── read live intrinsics from camera ──
        try:
            intr = get_intrinsics(serial, w, h, fps)
        except Exception as e:
            print(f"  [ERROR] Could not connect to {serial}: {e}")
            continue

        K = intrinsics_to_K(intr)
        dist_model = str(intr.model)
        dist_coeffs = list(intr.coeffs)

        print(f"  fx={intr.fx:.4f}  fy={intr.fy:.4f}  "
              f"cx={intr.ppx:.4f}  cy={intr.ppy:.4f}")
        print(f"  distortion: {dist_model}  coeffs={dist_coeffs}")
        print(f"  Intrinsic matrix:")
        for row in K:
            print(f"    [{row[0]:12.6f}, {row[1]:12.6f}, {row[2]:12.6f}]")

        # ── build config dict (live intrinsics + calibration extrinsics) ──
        if meta["type"] == "stationary":
            cfg = build_stationary_config(
                serial, K, dist_model, dist_coeffs, base, rvecs, tvecs, w, h, fps,
            )
        else:
            cfg = build_wrist_config(
                serial, K, dist_model, dist_coeffs, base, handeye, w, h, fps,
            )

        out_path = args.config_dir / f"{cam_name}.json"
        with open(out_path, "w") as f:
            json.dump(cfg, f, indent=4)
        print(f"\n  Saved to {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
