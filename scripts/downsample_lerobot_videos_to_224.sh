#!/usr/bin/env bash
# Downscale all dataset MP4s under ROOT/videos to 224x224 (faster training I/O).
# Usage: bash scripts/downsample_lerobot_videos_to_224.sh [ROOT]
# After running, update data/meta/info.json video shapes/codecs to match (see repo data/meta/info.json).
set -euo pipefail
ROOT="${1:-data}"
while IFS= read -r -d '' f; do
  tmp="${f}.tmp224.mp4"
  ffmpeg -y -i "$f" -vf scale=224:224 -c:v libx264 -preset fast -crf 23 -pix_fmt yuv420p -an "$tmp"
  mv "$tmp" "$f"
  echo "OK: $f"
done < <(find "$ROOT/videos" -name '*.mp4' -print0)
