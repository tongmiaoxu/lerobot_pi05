"""
Extract the first frame of the stationary camera (cam_high) from every episode
in data_1, segment target objects using Grounding-DINO + SAM2, then produce
clear initial-state distribution / overlay visualisations that show where each
object appears across episodes without blurring.

Uses ffmpeg (subprocess) to decode AV1-encoded video files.
SAM2 checkpoint loaded from  <repo-root>/weights/sam2/sam2.1_hiera_large.pt

Outputs (saved to visual_match/initial_states/):
  - individual_frames/ep_XXX.png              – raw first frame per episode
  - <object>/individual_masks/ep_XXX_mask.png – binary object mask per episode
  - <object>/individual_masked/ep_XXX_masked.png – object cutout (transparent bg)
  - <object>/occupancy_heatmap.png   – per-pixel count heatmap (jet colourmap)
  - <object>/contours_overlay.png    – all contours on median background
  - <object>/pixel_masks_overlay.png – original-pixel transparent overlay
  - <object>/object_centroids.png    – centroid scatter + spread ellipse
  - <object>/all_episodes_grid.png   – grid montage of every first frame
"""

import json
import math
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor


# ── paths (defined early so helpers can reference ROOT) ──────────────────────
ROOT = Path(__file__).resolve().parent.parent          # lerobot_pi05


# ── segmentation helpers (merged from sam2_segmentation.py) ──────────────────
_SEGMENT_MODELS = None


def _get_segment_models(device):
    """Lazy-load Grounding-DINO + SAM2 models (cached across calls)."""
    global _SEGMENT_MODELS
    if _SEGMENT_MODELS is None or _SEGMENT_MODELS["device"] != device:
        checkpoint = str(ROOT / "weights" / "sam2" / "sam2.1_hiera_large.pt")
        model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        model_id = "IDEA-Research/grounding-dino-tiny"
        processor = AutoProcessor.from_pretrained(model_id)
        grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            model_id
        ).to(device)
        image_predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))
        _SEGMENT_MODELS = {
            "device": device,
            "processor": processor,
            "grounding_model": grounding_model,
            "image_predictor": image_predictor,
        }
    return (
        _SEGMENT_MODELS["processor"],
        _SEGMENT_MODELS["grounding_model"],
        _SEGMENT_MODELS["image_predictor"],
    )


def _segment_object_mask(image_bgr, text_prompt="plush toy"):
    """Segment an object in a BGR image using Grounding-DINO + SAM2."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor, grounding_model, image_predictor = _get_segment_models(device)

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_rgb)

    inputs = processor(images=image_pil, text=[text_prompt], return_tensors="pt").to(
        device
    )
    with torch.no_grad():
        outputs = grounding_model(**inputs)
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=0.325,
        text_threshold=0.3,
        target_sizes=[image_pil.size[::-1]],
    )

    input_boxes = results[0]["boxes"].cpu().numpy()
    if len(input_boxes) == 0:
        return None

    image_predictor.set_image(np.array(image_pil.convert("RGB")))
    masks_list = []
    for bx in input_boxes:
        m, _, _ = image_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=bx[None, :],
            multimask_output=False,
        )
        if m.ndim == 4:
            m = m.squeeze(1)
        m = m.squeeze(0)
        masks_list.append(m.astype(bool))

    if not masks_list:
        return None

    mask = np.logical_or.reduce(np.stack(masks_list, axis=0), axis=0)
    return mask


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

def _filter_mask_by_neighbor_centroids(mask, neighbor_centroids):
    """If a mask has multiple disconnected components, keep only the one
    whose centroid is closest to the mean of *neighbor_centroids*.
    If there's only one component, return it unchanged.
    """
    n_components, labels = cv2.connectedComponents(mask.astype(np.uint8))
    if n_components <= 2:          # background (0) + one component
        return mask

    # Compute reference point from nearby episodes
    valid_neighbors = [c for c in neighbor_centroids if c is not None]
    if not valid_neighbors:
        return mask                # no neighbours yet – keep full mask
    ref = np.array(valid_neighbors).mean(axis=0)  # (cx, cy)

    best_label, best_dist = None, float("inf")
    for lbl in range(1, n_components):
        ys, xs = np.where(labels == lbl)
        centroid = np.array([xs.mean(), ys.mean()])
        dist = np.linalg.norm(centroid - ref)
        if dist < best_dist:
            best_dist = dist
            best_label = lbl

    return (labels == best_label)


# ── config ───────────────────────────────────────────────────────────────────
TEXT_PROMPTS = ["mug"]   # Grounding-DINO text prompts for objects

# ── dataset paths ────────────────────────────────────────────────────────────
DATA_DIR = ROOT / "data"
META_DIR = DATA_DIR / "meta"
VIDEO_DIR = DATA_DIR / "videos" / "observation.images.cam_high"
OUT_DIR = Path(__file__).resolve().parent / "initial_states"
FRAMES_DIR = OUT_DIR / "individual_frames"
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
    print(f"  Episode {ep_idx:3d}  ✓")

print(f"Extracted {len(frames)} / {total_episodes} first frames")
if len(frames) == 0:
    raise RuntimeError("No frames extracted – check video paths / ffmpeg")


# ── STEP 2 & 3: segment + visualise for EACH object ─────────────────────────
# We run the full segmentation + visualisation pipeline per object prompt,
# saving outputs into per-object subdirectories.

# Compute median background once (shared across objects)
stack = np.stack(frames).astype(np.float64)
median_bg = np.median(stack, axis=0).astype(np.uint8)
Image.fromarray(median_bg).save(OUT_DIR / "median_background.png")

n = len(frames)
cmap = _make_colormap(n)

for TEXT_PROMPT in TEXT_PROMPTS:
    print(f"\n{'='*60}")
    print(f"  Processing object: '{TEXT_PROMPT}'")
    print(f"{'='*60}")

    # Per-object output dirs
    obj_dir = OUT_DIR / TEXT_PROMPT
    obj_masks_dir = obj_dir / "individual_masks"
    obj_masked_dir = obj_dir / "individual_masked"
    for d in [obj_masks_dir, obj_masked_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # ── Segment the object in every first frame ─────────────────────────────
    print(f"\n── Segmenting '{TEXT_PROMPT}' in each frame ──")
    masks = []           # list of (H, W) bool arrays (or None)
    centroids = []       # list of (cx, cy) for valid masks

    for i, (frame_rgb, ep_idx) in enumerate(zip(frames, ep_indices)):
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        mask = _segment_object_mask(frame_bgr, text_prompt=TEXT_PROMPT)

        if mask is not None and mask.any():
            # Filter spurious components using nearby episode centroids
            lo = max(0, i - 3)
            neighbor_cents = centroids[lo:i]
            mask = _filter_mask_by_neighbor_centroids(mask, neighbor_cents)

            masks.append(mask)
            # save binary mask
            mask_img = (mask.astype(np.uint8) * 255)
            Image.fromarray(mask_img).save(obj_masks_dir / f"ep_{ep_idx:03d}_mask.png")
            # save masked cutout (RGBA)
            rgba = np.zeros((*frame_rgb.shape[:2], 4), dtype=np.uint8)
            rgba[..., :3] = frame_rgb
            rgba[..., 3] = mask.astype(np.uint8) * 255
            Image.fromarray(rgba).save(obj_masked_dir / f"ep_{ep_idx:03d}_masked.png")
            # centroid
            ys, xs = np.where(mask)
            cx, cy = xs.mean(), ys.mean()
            centroids.append((cx, cy))
            print(f"  Episode {ep_idx:3d}  mask pixels={mask.sum():7d}  centroid=({cx:.0f},{cy:.0f})")
        else:
            masks.append(None)
            centroids.append(None)
            print(f"  Episode {ep_idx:3d}  [no mask found]")

    valid = sum(1 for m in masks if m is not None)
    print(f"Segmented {valid} / {len(frames)} episodes for '{TEXT_PROMPT}'")

    if valid == 0:
        print(f"  [SKIP] No masks found for '{TEXT_PROMPT}' – skipping visualisations")
        continue

    # ── Visualisations ──────────────────────────────────────────────────────

    # --- Occupancy heatmap ---------------------------------------------------
    occupancy = np.zeros((vid_h, vid_w), dtype=np.float64)
    for m in masks:
        if m is not None:
            occupancy += m.astype(np.float64)

    occ_norm = (occupancy / max(occupancy.max(), 1) * 255).astype(np.uint8)
    heatmap_bgr = cv2.applyColorMap(occ_norm, cv2.COLORMAP_JET)
    heat_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    blend = median_bg.copy()
    occ_mask = occupancy > 0
    alpha_heat = 0.65
    blend[occ_mask] = (
        alpha_heat * heat_rgb[occ_mask].astype(np.float64)
        + (1 - alpha_heat) * median_bg[occ_mask].astype(np.float64)
    ).astype(np.uint8)
    Image.fromarray(blend).save(obj_dir / "occupancy_heatmap.png")
    print(f"  Saved {TEXT_PROMPT}/occupancy_heatmap.png")

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
    print(f"  Saved {TEXT_PROMPT}/contours_overlay.png")

    # --- Transparent original-pixel masks overlay ----------------------------
    pixel_overlay = median_bg.astype(np.float64).copy()
    alpha_per_ep = min(0.5, 20 / max(valid, 1))
    for i, (m, frame_rgb) in enumerate(zip(masks, frames)):
        if m is None:
            continue
        pixel_overlay[m] = (
            (1 - alpha_per_ep) * pixel_overlay[m]
            + alpha_per_ep * frame_rgb[m].astype(np.float64)
        )
    pixel_overlay = np.clip(pixel_overlay, 0, 255).astype(np.uint8)
    Image.fromarray(pixel_overlay).save(obj_dir / "pixel_masks_overlay.png")
    print(f"  Saved {TEXT_PROMPT}/pixel_masks_overlay.png")

    # --- Centroid scatter + spread ellipse ------------------------------------
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
    print(f"  Saved {TEXT_PROMPT}/object_centroids.png")

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
    print(f"  Saved {TEXT_PROMPT}/all_episodes_grid.png")

print(f"\nAll outputs in: {OUT_DIR}")
