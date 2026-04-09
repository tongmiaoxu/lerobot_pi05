#!/usr/bin/env python
"""Resize local LeRobot dataset videos, update info.json, and recompute image stats."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from recompute_dataset_image_stats import recompute_image_stats


def _probe_video(video_path: Path) -> tuple[str, int, int]:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,width,height",
            "-of",
            "csv=p=0",
            str(video_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    codec, width, height = result.stdout.strip().split(",")
    return codec, int(width), int(height)


def _resize_videos(root: Path, height: int, width: int, codec: str) -> tuple[int, int]:
    processed = 0
    changed = 0
    for video_path in sorted((root / "videos").rglob("*.mp4")):
        processed += 1
        current_codec, current_width, current_height = _probe_video(video_path)
        if current_codec == codec and current_width == width and current_height == height:
            continue

        tmp_path = video_path.with_name(video_path.name + ".tmp224.mp4")
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(video_path),
                "-vf",
                f"scale={width}:{height}",
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "23",
                "-pix_fmt",
                "yuv420p",
                "-an",
                str(tmp_path),
            ],
            check=True,
        )
        tmp_path.replace(video_path)
        changed += 1
        print(f"Resized {video_path}")
    return processed, changed


def _update_info_json(root: Path, height: int, width: int, codec: str) -> None:
    info_path = root / "meta" / "info.json"
    with info_path.open() as f:
        info = json.load(f)

    for feature in info.get("features", {}).values():
        if feature.get("dtype") != "video":
            continue
        shape = feature.get("shape")
        if isinstance(shape, list) and len(shape) == 3:
            feature["shape"] = [height, width, shape[2]]
        feature_info = feature.setdefault("info", {})
        feature_info["video.height"] = height
        feature_info["video.width"] = width
        feature_info["video.codec"] = codec
        feature_info["video.pix_fmt"] = "yuv420p"

    with info_path.open("w") as f:
        json.dump(info, f, indent=4)
        f.write("\n")
    print("Wrote", info_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", help="Path to the local LeRobot dataset root")
    parser.add_argument("--repo-id", default="local/xarm", help="Repo id label for loading the local dataset")
    parser.add_argument("--height", type=int, default=224, help="Target video height")
    parser.add_argument("--width", type=int, default=224, help="Target video width")
    parser.add_argument(
        "--codec",
        default="h264",
        choices=["h264"],
        help="Codec to record in metadata after resize",
    )
    args = parser.parse_args()

    root = Path(args.dataset_root).expanduser().resolve()
    if not (root / "meta" / "info.json").is_file():
        raise FileNotFoundError(f"Dataset metadata not found at {root / 'meta' / 'info.json'}")

    processed, changed = _resize_videos(root=root, height=args.height, width=args.width, codec=args.codec)
    _update_info_json(root=root, height=args.height, width=args.width, codec=args.codec)
    recompute_image_stats(root=root, repo_id=args.repo_id)
    print(f"Checked {processed} videos, resized {changed}.")


if __name__ == "__main__":
    main()
