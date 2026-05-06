"""
Extract the first frame of the stationary camera (cam_high) from every episode,
segment target objects using Grounding-DINO + SAM2, and save initial-state
visualisations directly under the dataset root.

Uses ffmpeg (subprocess) to decode dataset videos.
SAM2 checkpoint loaded from <repo-root>/weights/sam2/sam2.1_hiera_large.pt

Outputs (saved under DATA_DIR):
  - _initial_state_frames/ep_XXX.png          – cached first frame per episode
  - <object>/individual_masks/ep_XXX_mask.png – binary object mask per episode
  - <object>/contours_overlay.png             – all contours on median background
  - <object>/pixel_masks_overlay.png          – original-pixel transparent overlay
  - <object>/object_centroids.png             – centroid scatter + spread ellipse
  - <object>/all_episodes_grid.png            – grid montage of every first frame
  - pixel_masks_overlay.png                   – combined overlay across all objects
"""

import json
import math
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from segmentation_utils import segment_object_mask


ROOT = Path(__file__).resolve().parent.parent          # lerobot_pi05
TEXT_PROMPTS = ["shoe"]   # Comma-separated entries are split into separate objects
OUTPUT_OBJECT_NAMES = {"shoe": "right_shoe"}
DATA_DIR = ROOT / "data_pick_shoe_copy"
COMPONENT_SELECTION_MODE = "leftmost"  # "leftmost", "interactive_each", or "track_neighbors"


# ── helpers ──────────────────────────────────────────────────────────────────
def _make_colormap(n):
    """Return (n, 3) uint8 array of distinct RGB colours using HSV spacing."""
    colours = np.zeros((n, 3), dtype=np.uint8)
    for i in range(n):
        h = int(180 * i / n)   # OpenCV hue is 0-179
        hsv = np.uint8([[[h, 255, 220]]])
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0, 0]
        colours[i] = rgb
    return colours


def _expand_text_prompts(prompt_list):
    """Split comma-separated prompt entries into unique individual object prompts."""
    expanded = []
    for prompt in prompt_list:
        for part in prompt.split(","):
            token = part.strip()
            if token and token not in expanded:
                expanded.append(token)
    return expanded


def _sanitize_object_name(name: str) -> str:
    return name.strip().replace("/", "_").replace(" ", "_")


def _component_centroids(mask):
    n_components, labels = cv2.connectedComponents(mask.astype(np.uint8))
    centroids = {}
    for lbl in range(1, n_components):
        ys, xs = np.where(labels == lbl)
        if len(xs) > 0 and len(ys) > 0:
            centroids[lbl] = np.array([xs.mean(), ys.mean()])
    return labels, centroids


def _select_component_interactively(labels, centroids, window_name):
    n_components = int(labels.max()) + 1
    display = np.zeros((*labels.shape, 3), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for lbl, centroid in centroids.items():
        color = tuple(int(x) for x in np.random.randint(0, 255, 3))
        display[labels == lbl] = color
        cx, cy = int(centroid[0]), int(centroid[1])
        cv2.putText(display, str(lbl), (cx, cy), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow(window_name, display)
    print(f"Multiple components detected ({n_components - 1}). Press key 1-{n_components - 1} to select, or q to skip.")
    key = cv2.waitKey(0)
    cv2.destroyWindow(window_name)
    if key in [ord(str(i)) for i in range(1, n_components)]:
        chosen = int(chr(key))
        return labels == chosen

    print("No valid selection, returning full mask.")
    return labels > 0


def _filter_mask_by_neighbor_centroids(mask, neighbor_centroids):
    """If a mask has multiple disconnected components, keep only the one
    whose centroid is closest to the mean of *neighbor_centroids*.
    If there's only one component, return it unchanged.
    """
    labels, component_centroids = _component_centroids(mask)
    if len(component_centroids) <= 1:
        return mask

    if COMPONENT_SELECTION_MODE == "leftmost":
        # Camera image coordinates increase to the right; the desired right shoe
        # appears on the left side of the image.
        best_label = min(component_centroids, key=lambda lbl: component_centroids[lbl][0])
        return labels == best_label

    if COMPONENT_SELECTION_MODE == "interactive_each":
        return _select_component_interactively(
            labels,
            component_centroids,
            "Select component: press 1, 2, ...",
        )

    # Compute reference point from nearby episodes
    valid_neighbors = [c for c in neighbor_centroids if c is not None]
    if not valid_neighbors:
        return _select_component_interactively(
            labels,
            component_centroids,
            "Select component: press 1, 2, ...",
        )
    ref = np.array(valid_neighbors).mean(axis=0)  # (cx, cy)

    best_label, best_dist = None, float("inf")
    for lbl, centroid in component_centroids.items():
        dist = np.linalg.norm(centroid - ref)
        if dist < best_dist:
            best_dist = dist
            best_label = lbl

    return labels == best_label


# ── config ───────────────────────────────────────────────────────────────────

META_DIR = DATA_DIR / "meta"
VIDEO_DIR = DATA_DIR / "videos" / "observation.images.cam_high"
OUT_DIR = DATA_DIR
FRAMES_DIR = OUT_DIR / "_initial_state_frames"
FRAMES_DIR.mkdir(parents=True, exist_ok=True)


# ── load dataset info ────────────────────────────────────────────────────────
with open(META_DIR / "info.json") as f:
    info = json.load(f)

total_episodes = info["total_episodes"]
fps = info["fps"]
vid_h = info["features"]["observation.images.cam_high"]["shape"][0]
vid_w = info["features"]["observation.images.cam_high"]["shape"][1]

print(f"Dataset: {total_episodes} episodes, {fps} fps, {vid_w}x{vid_h}")


# ── build episode → (video_file, frame_offset) mapping from data parquets ────
data_parquets = sorted((DATA_DIR / "data").rglob("*.parquet"))
file_episode_map = {}  # {vid_path_str: [(ep_idx, offset_in_file), ...]}

for pq_path in data_parquets:
    file_idx = int(pq_path.stem.split("-")[1])
    chunk_idx = int(pq_path.parent.name.split("-")[1])
    df = pd.read_parquet(pq_path)
    grouped = df.groupby("episode_index").size()
    cumsum = 0
    for ep_idx, length in grouped.items():
        vid_path = VIDEO_DIR / f"chunk-{chunk_idx:03d}" / f"file-{file_idx:03d}.mp4"
        file_episode_map.setdefault(str(vid_path), []).append((ep_idx, cumsum))
        cumsum += length

episode_info = []
for vid_path, entries in file_episode_map.items():
    for ep_idx, offset in entries:
        episode_info.append((ep_idx, vid_path, offset))
episode_info.sort(key=lambda x: x[0])

print(f"Mapped {len(episode_info)} episodes across {len(file_episode_map)} video files")


# ── extract a single frame using ffmpeg ──────────────────────────────────────
def extract_frame_ffmpeg(video_path, frame_offset, width, height):
    """Use ffmpeg to extract a single frame at a given offset (frame number)."""
    timestamp = frame_offset / fps
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{timestamp:.6f}",
        "-i", str(video_path),
        "-frames:v", "1",
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-v", "error",
        "pipe:1",
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0 or len(result.stdout) == 0:
        return None
    frame = np.frombuffer(result.stdout, dtype=np.uint8)
    expected = height * width * 3
    if len(frame) < expected:
        return None
    return frame[:expected].reshape(height, width, 3)


# ── STEP 1: extract first frames ────────────────────────────────────────────
print("\n── Step 1: Extracting first frames ──")
frames = []          # list of (H, W, 3) uint8 RGB arrays
ep_indices = []      # parallel list of episode indices

for ep_idx, vid_path, offset in episode_info:
    # Re-use cached frame if it already exists
    cached = FRAMES_DIR / f"ep_{ep_idx:03d}.png"
    if cached.exists():
        frame_rgb = np.array(Image.open(cached).convert("RGB"))
    else:
        frame_rgb = extract_frame_ffmpeg(vid_path, offset, vid_w, vid_h)
        if frame_rgb is None:
            print(f"  [WARN] Failed ep={ep_idx}  offset={offset}")
            continue
        Image.fromarray(frame_rgb).save(cached)

    frames.append(frame_rgb)
    ep_indices.append(ep_idx)
    # print(f"  Episode {ep_idx:3d}  ✓")

print(f"Extracted {len(frames)} / {total_episodes} first frames")
if len(frames) == 0:
    raise RuntimeError("No frames extracted – check video paths / ffmpeg")


# ── STEP 2 & 3: segment + visualise for EACH object ─────────────────────────
# We run the segmentation + visualisation pipeline per object, saving outputs
# into per-object subdirectories under the dataset root.

# Compute median background once (shared across objects)
stack = np.stack(frames).astype(np.float64)
median_bg = np.median(stack, axis=0).astype(np.uint8)

n = len(frames)
cmap = _make_colormap(n)
object_prompts = _expand_text_prompts(TEXT_PROMPTS)
object_masks_by_name = {}

for text_prompt in object_prompts:
    print(f"\n{'='*60}")
    print(f"  Processing object: '{text_prompt}'")
    print(f"{'='*60}")

    object_name = OUTPUT_OBJECT_NAMES.get(text_prompt, _sanitize_object_name(text_prompt))
    obj_dir = OUT_DIR / object_name
    obj_masks_dir = obj_dir / "individual_masks"
    obj_masks_dir.mkdir(parents=True, exist_ok=True)

    # ── Segment the object in every first frame ─────────────────────────────
    print(f"\n── Segmenting '{text_prompt}' in each frame ──")
    masks = []           # list of (H, W) bool arrays (or None)
    centroids = []       # list of (cx, cy) for valid masks

    for i, (frame_rgb, ep_idx) in enumerate(zip(frames, ep_indices)):
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        mask = segment_object_mask(frame_bgr, text_prompt=text_prompt)

        if mask is not None and mask.any():
            lo = max(0, i - 3)
            neighbor_cents = centroids[lo:i]
            mask = _filter_mask_by_neighbor_centroids(mask, neighbor_cents)

            masks.append(mask)
            mask_img = (mask.astype(np.uint8) * 255)
            Image.fromarray(mask_img).save(obj_masks_dir / f"ep_{ep_idx:03d}_mask.png")
            ys, xs = np.where(mask)
            cx, cy = xs.mean(), ys.mean()
            centroids.append((cx, cy))
        else:
            masks.append(None)
            centroids.append(None)
            print(f"  Episode {ep_idx:3d}  [no mask found]")

    valid = sum(1 for m in masks if m is not None)
    print(f"Segmented {valid} / {len(frames)} episodes for '{text_prompt}'")

    if valid == 0:
        print(f"  [SKIP] No masks found for '{text_prompt}' – skipping visualisations")
        continue

    object_masks_by_name[object_name] = masks

    # --- Contours overlay ----------------------------------------------------
    contours_img_bgr = cv2.cvtColor(median_bg.copy(), cv2.COLOR_RGB2BGR)
    for i, m in enumerate(masks):
        if m is None:
            continue
        contours, _ = cv2.findContours(
            m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        colour_bgr = (int(cmap[i][2]), int(cmap[i][1]), int(cmap[i][0]))
        cv2.drawContours(contours_img_bgr, contours, -1, colour_bgr, 2)
    contours_rgb = cv2.cvtColor(contours_img_bgr, cv2.COLOR_BGR2RGB)
    Image.fromarray(contours_rgb).save(obj_dir / "contours_overlay.png")
    print(f"  Saved {object_name}/contours_overlay.png")

    # --- Transparent original-pixel masks overlay ----------------------------
    pixel_overlay = median_bg.astype(np.float64).copy()
    alpha_per_ep = min(0.5, 20 / max(valid, 1))
    for m, frame_rgb in zip(masks, frames):
        if m is None:
            continue
        pixel_overlay[m] = (
            (1 - alpha_per_ep) * pixel_overlay[m]
            + alpha_per_ep * frame_rgb[m].astype(np.float64)
        )
    pixel_overlay = np.clip(pixel_overlay, 0, 255).astype(np.uint8)
    Image.fromarray(pixel_overlay).save(obj_dir / "pixel_masks_overlay.png")
    print(f"  Saved {object_name}/pixel_masks_overlay.png")

    # --- Centroid scatter + spread ellipse ----------------------------------
    centroid_img_bgr = cv2.cvtColor(median_bg.copy(), cv2.COLOR_RGB2BGR)
    valid_centroids = [c for c in centroids if c is not None]
    for i, c in enumerate(centroids):
        if c is None:
            continue
        cx, cy = int(c[0]), int(c[1])
        colour_bgr = (int(cmap[i][2]), int(cmap[i][1]), int(cmap[i][0]))
        cv2.circle(centroid_img_bgr, (cx, cy), 5, colour_bgr, -1)
        cv2.circle(centroid_img_bgr, (cx, cy), 6, (255, 255, 255), 1)
    if len(valid_centroids) >= 2:
        pts = np.array(valid_centroids)
        mean_c = pts.mean(axis=0)
        cov = np.cov(pts.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        angle = np.degrees(np.arctan2(eigvecs[1, 1], eigvecs[0, 1]))
        axes = tuple((2 * np.sqrt(np.maximum(eigvals, 0))).astype(int))
        cv2.ellipse(
            centroid_img_bgr,
            (int(mean_c[0]), int(mean_c[1])),
            axes, angle, 0, 360,
            (0, 255, 255), 2,
        )
    centroid_rgb = cv2.cvtColor(centroid_img_bgr, cv2.COLOR_BGR2RGB)
    Image.fromarray(centroid_rgb).save(obj_dir / "object_centroids.png")
    print(f"  Saved {object_name}/object_centroids.png")

    # --- Grid montage --------------------------------------------------------
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    h, w = frames[0].shape[:2]
    thumb_w, thumb_h = w // 2, h // 2

    grid = np.zeros((rows * thumb_h, cols * thumb_w, 3), dtype=np.uint8)
    for idx, fr in enumerate(frames):
        r, c = divmod(idx, cols)
        thumb = cv2.resize(fr, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
        if masks[idx] is not None:
            m_small = cv2.resize(
                masks[idx].astype(np.uint8), (thumb_w, thumb_h),
                interpolation=cv2.INTER_NEAREST
            )
            cnts, _ = cv2.findContours(m_small, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(thumb, cnts, -1, (0, 255, 0), 1)
        grid[r * thumb_h:(r + 1) * thumb_h, c * thumb_w:(c + 1) * thumb_w] = thumb

    Image.fromarray(grid).save(obj_dir / "all_episodes_grid.png")
    print(f"  Saved {object_name}/all_episodes_grid.png")

# --- Combined overlay across all requested objects ---------------------------
combined_overlay = median_bg.astype(np.float64).copy()
num_object_masks = sum(
    1 for masks in object_masks_by_name.values() for mask in masks if mask is not None
)
alpha_per_mask = min(0.35, 20 / max(num_object_masks, 1))
for masks in object_masks_by_name.values():
    for mask, frame_rgb in zip(masks, frames):
        if mask is None:
            continue
        combined_overlay[mask] = (
            (1 - alpha_per_mask) * combined_overlay[mask]
            + alpha_per_mask * frame_rgb[mask].astype(np.float64)
        )
combined_overlay = np.clip(combined_overlay, 0, 255).astype(np.uint8)
Image.fromarray(combined_overlay).save(OUT_DIR / "pixel_masks_overlay.png")
print(f"Saved combined overlay → {OUT_DIR / 'pixel_masks_overlay.png'}")

# --- Combined grid montage across all requested objects ----------------------
cols = math.ceil(math.sqrt(n))
rows = math.ceil(n / cols)
h, w = frames[0].shape[:2]
thumb_w, thumb_h = w // 2, h // 2

combined_grid = np.zeros((rows * thumb_h, cols * thumb_w, 3), dtype=np.uint8)
for idx, fr in enumerate(frames):
    r, c = divmod(idx, cols)
    thumb = cv2.resize(fr, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
    for masks in object_masks_by_name.values():
        mask = masks[idx]
        if mask is None:
            continue
        m_small = cv2.resize(
            mask.astype(np.uint8), (thumb_w, thumb_h), interpolation=cv2.INTER_NEAREST
        )
        cnts, _ = cv2.findContours(m_small, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(thumb, cnts, -1, (0, 255, 0), 1)
    combined_grid[r * thumb_h:(r + 1) * thumb_h, c * thumb_w:(c + 1) * thumb_w] = thumb

Image.fromarray(combined_grid).save(OUT_DIR / "all_episodes_grid.png")
print(f"Saved combined grid → {OUT_DIR / 'all_episodes_grid.png'}")

print(f"\nAll outputs in: {OUT_DIR}")
