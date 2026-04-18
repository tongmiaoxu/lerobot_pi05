# VLA Attention Visualization

Tool to visualize **what part of the scene a VLA policy is looking at** when it
produces an action, during a live MuJoCo rollout.

Currently supports:
- **pi05** (PaliGemma + Gemma action expert) — first-class support
- **Groot** (Eagle2 backbone + DiT action head) — works for the common single-crop
  case; may need manual vision-token layout hints (see below).

Two visualizations are produced per control step:
1. **Per-layer raw attention maps** (pick any layer, plus `mean_layers` and
   `last_layer` shortcuts).
2. **Attention rollout** — layer-wise aggregated attention (Abnar & Zuidema 2020
   style) that summarizes how strongly each action query attends to each image
   patch across the whole stack.

Attention is sliced to `action_queries → image_patch_tokens` and reshaped to the
patch grid, then upsampled and overlaid on the image the policy actually saw
(i.e. after resizing).

## Files

- `visual_match/attention_viz.py` — self-contained capture + rollout + overlay
  utilities. Import this from any script.
- `visual_match/deploy_vla_attention_mujoco.py` — minimal standalone MuJoCo
  deployment that runs a VLA + saves / displays attention dashboards.

## Quick start — pi05 on `place_mug`

```bash
python visual_match/deploy_vla_attention_mujoco.py \
    --policy-path outputs/pi05_place_mug/checkpoints/100000/pretrained_model \
    --task place_mug \
    --mode rollout \
    --keep-denoise last \
    --num-episodes 1 --max-steps 300
```

Outputs:
- Live dashboard window (top row = raw images the policy sees, bottom row = same
  images with the attention heatmap burned on top, one panel per camera).
- `outputs/attention_viz/pi05_place_mug/episode_000_rollout.mp4`

Useful flags:
- `--mode` : `rollout | mean_layers | last_layer | layer=<i>` — which attention
  map to render. `layer=0` is the earliest expert layer (usually more local /
  patchy), `layer=17` is the last (usually more global / task-semantic).
- `--keep-denoise` : `last | mean | stack` — how to combine attention across
  flow-matching denoise steps. `last` is fast and usually the most informative.
- `--query-agg` : `mean | max | last` — how to aggregate over the 50 action
  tokens in the chunk. `last` highlights what drives the final action in the
  chunk; `mean` is a safer default.
- `--head-fuse` : `mean | max | min` — how to fuse the 8 attention heads.
- `--colormap`, `--alpha`, `--blur-sigma` : cosmetic overlay controls.
- `--no-display` : headless mode (just write the MP4).

## Quick start — Groot

Groot's Eagle backbone does dynamic image tiling, so the exact layout of
"visual tokens" inside the VL sequence depends on your image size / number of
tiles. You usually need to tell the script manually:

```bash
python visual_match/deploy_vla_attention_mujoco.py \
    --policy-path outputs/groot_place_mug/checkpoints/last/pretrained_model \
    --task place_mug --mode rollout \


If you omit the layout flags entirely, the script will try to infer a square
grid from the captured attention width — this only works when your prefix
contains nothing but visual tokens.

## Using the library directly

```python
from visual_match.attention_viz import PI05AttentionCapture, render_pi05_overlays

capture = PI05AttentionCapture(policy, keep_denoise="last")
with capture:
    action = policy.select_action(batch)

overlays = render_pi05_overlays(
    capture,
    images_by_cam={
        "observation.images.cam_high":  img_high_u8,    # (H,W,3) uint8
        "observation.images.cam_wrist": img_wrist_u8,
    },
    mode="rollout",
    alpha=0.5, colormap="turbo", blur_sigma=2.0,
)
# overlays["observation.images.cam_high"] is an RGB uint8 overlay image.
```

Raw per-layer tensors:

```python
capture.attentions  # list[torch.Tensor], one per expert layer.
                    # Each [H, Q=chunk, K=prefix+suffix] after keep_denoise reduction,
                    # or [T, H, Q, K] when keep_denoise="stack".

capture.layout      # PrefixLayout: image_spans, lang_start/end, prefix_len
```

## Interpretation tips

- **Very flat heatmaps** often mean the policy is attending almost uniformly —
  either it has memorized the trajectory and barely uses vision, or the layer
  you picked is an early/dispersed layer. Try `--mode last_layer` first.
- **Heatmap lit up on the robot arm / gripper only** is common and can indicate
  the policy is closed-loop on proprioception-in-the-image (good for precision,
  but brittle to appearance shift).
- **Heatmap lit up on the target object** is what you want at grasp / place
  moments.
- Compare `layer=0`, `layer=mid`, `layer=last` and `rollout` on the same frame
  — rollout should look like a weighted combination but often concentrates on
  the most task-relevant regions.
- Consider capturing multiple failure + success episodes and diff-ing the
  heatmaps (simple per-pixel subtract of rollouts) to localize the regions the
  policy *stops* attending to during failures.

## Known caveats

- Pi05's `denoise_step` runs with eager attention (required for capture to
  work) — this is already the case in the codebase, so you should not need to
  do anything special. If you add a SDPA/flash code path later, make sure
  eager remains enabled during visualization runs.
- Groot's DiT uses diffusers' `Attention`; we replace its processor with a
  custom one that computes attention in float32. Small numerical differences
  vs SDPA are expected but do not affect the final action meaningfully.
- Attention tensors are moved to CPU after each layer's hook to keep VRAM
  steady across denoise steps, at the cost of one GPU→CPU copy per layer.
  For 18 Gemma-expert layers × 10 denoise steps × [1,8,50,~760] float32 this
  is ~200 MB CPU RAM per inference step and resets every step.
