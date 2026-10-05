---
name: eval-turbo-checkpoint
description: Evaluate a pix2pix-turbo sim2real checkpoint (a per-task/camera "DINO-Align" checkpoint or a joint multi-task/camera checkpoint like "DINO-Align-Joint") against held-out data_val_set_<task> data. Reuses an existing reviewed GS-render eval set when one exists (for pixel/metric parity), runs inference, copies real_A/real_B mask overrides (as stable saved masks, not just points), pauses for manual fake_B mask review, freezes the reviewed masks and renders a fake_B grid under data_mask/<checkpoint>/ for a final visual check, then computes LPIPS + Grounding-DINO/SAM2 J&F metrics. Use whenever asked to eval a turbo/pix2pix-turbo checkpoint on one or more tasks/cameras, run eval_turbo/eval_pix2pix, compute eval_pix2pix_metrics for such a checkpoint, or generate/update its mask-review grid images.
---

# Evaluating a pix2pix-turbo sim2real checkpoint

This is a two-phase, task/camera-by-task/camera workflow for scoring a `sim2real/eval_turbo.py`
checkpoint against the held-out `data_val_set_<task>` sets, in a way that stays comparable to
every other checkpoint already evaluated the same way (same input images, same manually-reviewed
mask overrides where possible).

**Phase A (this session runs automatically):** prepare/reuse the eval dataset, run inference,
copy `real_A`/`real_B` mask overrides from an existing reviewed baseline.
**Phase B (blocks on the user):** the user manually reviews `fake_B` masks with
`review_pix2pix_masks.py` — an interactive OpenCV tool only a human can drive. Stop and wait
after Phase A; do not compute metrics until the user says review is done. Once they confirm,
freeze anything left un-clicked-but-correct and render a `fake_B` mask grid (Phase A step 5)
before moving on.
**Phase C (this session runs automatically once told review is done):** compute
`eval_pix2pix_metrics.py` for each task/camera and report a summary table.

Never skip straight to Phase C without the user confirming review is complete for that
checkpoint's result sets, and never silently fall back to computing metrics on an unreviewed
`mask_overrides.json` — ask if it's unclear whether review happened.

## Inputs you need from the user (ask if not given)

- The checkpoint path to evaluate, e.g. `outputs/turbo_sim2real_all_tasks_dino_mujoco/checkpoints/model_30001.pkl`.
  Its parent's parent is the `--output-dir` for every eval_turbo.py call below, e.g.
  `outputs/turbo_sim2real_all_tasks_dino_mujoco`.
- Which task/camera combinations to evaluate (e.g. "mug wrist and stationary", "the remaining 6
  tasks", "book/wrist only"). Cameras are always `stationary` or `wrist`.

## Task metadata table

Each task has a pix2pix `--name` prefix, a data_val_set folder, and a Grounding-DINO
`--text-prompt` (object list) used for both mask review and metrics — these must match exactly
what earlier baselines used for that task, or the manual review / metrics won't line up with
prior results.

| task    | name prefix     | data_val_set dir     | --text-prompt   |
|---------|------------------|-----------------------|-----------------|
| mug     | `place_mug`      | `data_val_set_mug`     | `mug, saucer`   |
| shoe    | `pick_shoe`      | `data_val_set_shoe`    | `shoe`          |
| book    | `book_shelving`  | `data_val_set_book`    | `book`          |
| pouring | `pouring`        | `data_val_set_pouring` | `juice, mug`    |

If asked to eval a task not in this table, don't guess the prompt or name prefix — look for an
existing `outputs/turbo_sim2real_*_dino*<task>*` baseline and read its
`results/*/test_latest/metrics.json` `summary.text_prompt` field (or ask the user).

## Phase A steps (per task/camera)

### 1. Find (or build) the eval dataset — GS renders only, never turbo-mujoco

Look for an existing reference baseline for this exact task+camera:
`outputs/turbo_sim2real_{camera}_dino{"" if task=="mug" else "_" + task}` (mug's baseline has no
task suffix since it was the first task done; every other task's baseline is
`turbo_sim2real_{camera}_dino_{task}`, e.g. `turbo_sim2real_stationary_dino_shoe`).

- **If that baseline's `eval_dataset/test_A` + `test_B` exist:** reuse them directly as
  `--sim-dir`/`--real-dir` for `eval_turbo.py` below — do **not** point at
  `data_val_set_<task>/<camera>/gs_renders` directly. The live `data_val_set_<task>` folder can
  grow over time (e.g. a `0000_256.png` calibration pair got added to `data_val_set_mug` after
  the mug baseline was built, which would silently inflate a fresh pair count from 25 to 26 and
  break parity/index-alignment with the already-reviewed overrides). Reusing the frozen
  `test_A`/`test_B` guarantees identical images and identical pair count/indexing every time.
- **If no such baseline exists yet** (brand new task/camera never evaluated before): build fresh
  from `data_val_set_<task>/<camera>/gs_renders` + `data_val_set_<task>/<camera>/real_captures`.
  There's nothing to copy overrides from in this case — skip step 3, tell the user this is a
  from-scratch review.
- Never use `data_val_set_<task>/turbo-mujoco/<camera>/mujoco_renders` for this workflow. That
  was a one-off mistake in an earlier run — always gs_render-based input, confirmed with the user.

### 2. Run inference

```bash
python sim2real/eval_turbo.py \
  --sim-dir <resolved sim-dir from step 1> \
  --real-dir <resolved real-dir from step 1> \
  --dataset-dir outputs/<this_checkpoint's_output_dir>/eval_dataset_<task>_<camera> \
  --checkpoint <checkpoint path> \
  --output-dir outputs/<this_checkpoint's_output_dir> \
  --name <name prefix>_<camera>_turbo<suffix matching this checkpoint's own naming, e.g. _mujoco> \
  --overwrite
```

Sanity-check the printed `n=<N>` matches the reference baseline's pair count (usually 25) before
moving on — if it doesn't and you reused an existing test_A/test_B, something is wrong (stop and
investigate rather than continuing).

### 3. Copy real_A/real_B mask overrides (skip if no reference baseline existed in step 1)

```bash
python scripts/seed_turbo_overrides.py \
  --src outputs/turbo_sim2real_{camera}_dino{_task}/results/<name prefix>_<camera>_turbo/test_latest/mask_overrides.json \
  --dst outputs/<this_checkpoint's_output_dir>/results/<name from step 2>/test_latest/mask_overrides.json
```

This only copies `real_A`/`real_B` entries (never `fake_B` — each checkpoint's `fake_B` is its
own generated image and always needs its own fresh review). Safe to re-run; it overwrites
matching keys rather than duplicating.

`mask_overrides.json` entries are one of three kinds (see `scripts/pix2pix_mask_common.py`):
a stable saved `mask` (a PNG blob, used verbatim, never re-derived — the current/preferred kind),
a legacy `point`/`candidate_index` (re-resolved by rerunning SAM2/Grounding-DINO every time,
which can occasionally reproduce a *different* mask than what a reviewer actually confirmed, due
to ordinary GPU float nondeterminism nudging Grounding-DINO's box scores/order), or a confirmed
`missing`. Since `real_A`/`real_B` are visually identical across every checkpoint evaluated on
the same task/camera (same source images, same `--metrics-resolution`/`--resolution`), a
`mask`-kind override copied by `seed_turbo_overrides.py` is exactly as valid on the destination
checkpoint as it was on the source — no re-review needed there, ever. This holds across an
`eval_turbo.py` run and an `eval_pix2pix.py` run too (e.g. copying from a turbo baseline into a
plain-pix2pix checkpoint's result set), even though those two scripts' saved real_A/real_B PNGs
aren't literally byte-identical — `eval_pix2pix.py`'s dataloader round-trips them through the
network's normalized tensor range, introducing up to ±1/255 rounding noise on a small fraction
of pixels (see `eval_turbo.py`'s docstring) — that's sub-perceptual and doesn't change which mask
is correct, so it doesn't block reuse. If the source set you're copying from still has
`legacy` entries (reviewed before mask-caching, or the reviewer left a correct-looking default
un-clicked — see `review_pix2pix_masks.py`'s docstring on why not clicking is often intentional),
freeze it to the stable format first (safe to re-run, no-op on anything already a `mask`):

```bash
python scripts/freeze_pix2pix_masks.py \
  --images-dir outputs/<src output_dir>/results/<src name>/test_latest/images \
  --text-prompt "<task's prompt>"
```

If you're unsure whether a set is fully converted, check before copying/reporting metrics —
count `mask`/`legacy`/`missing`/`none` per idx/obj/role in its `mask_overrides.json` and flag any
non-`mask`, non-confirmed-`missing` entries to the user rather than assuming the review is final.

### 4. Hand off to the user

For a joint/multi-task checkpoint (every task/camera evaluated this pass shares one
`--output-dir`), give a single command covering all of them — `--checkpoint-dir` auto-discovers
every `results/<name>/test_latest/` under it, maps each to its task's fixed `--text-prompt` via
`pix2pix_mask_common.TASK_PROMPTS`, and reviews them all in one continuous session (`n`/`p` walk
a flattened list across result sets, `[`/`]` jump straight to the previous/next one):

```bash
python scripts/review_pix2pix_masks.py --checkpoint-dir outputs/<output_dir>
```

Otherwise (a single task/camera checkpoint, or only some of a joint checkpoint's result sets
were evaluated this pass), give the single-set command per task/camera instead:

```bash
python scripts/review_pix2pix_masks.py \
  --images-dir outputs/<output_dir>/results/<name>/test_latest/images \
  --text-prompt "<task's prompt>"
```

Then stop. Do not proceed to Phase C until the user confirms review is done for these result sets.

### 5. After the user confirms manual correction is done: freeze + grid, per task/camera

Once the user says `fake_B` review is complete for a result set, first freeze anything they
left un-clicked because it already looked correct (see step 3's note — not clicking is often
intentional, not incomplete review):

```bash
python scripts/freeze_pix2pix_masks.py \
  --images-dir outputs/<output_dir>/results/<name>/test_latest/images \
  --text-prompt "<task's prompt>"
```

Then render a `fake_B` grid for a final visual sanity check, under
`data_mask/<output_dir's basename>/` (one subfolder per checkpoint — never flatten multiple
checkpoints' `fake_B` grids into the same folder, since the whole point is telling which
checkpoint generated which `fake_B`):

```bash
python scripts/make_mask_grid.py \
  --images-dir outputs/<output_dir>/results/<name>/test_latest/images \
  --text-prompt "<task's prompt>" --role fake_B \
  --out data_mask/<output_dir's basename>/<name>_fake_B.png
```

`real_A`/`real_B` grids do **not** need one copy per checkpoint — since their masks are
pixel-identical across every checkpoint sharing a task/camera (see step 3), render them once
into a shared `data_mask/real_A_real_B/<task>_<camera>_<role>.png` from any one already-reviewed
checkpoint, instead of duplicating them under every checkpoint's own subfolder.

## Phase C: metrics (only after user confirms review is done)

```bash
python scripts/eval_pix2pix_metrics.py \
  --images-dir outputs/<output_dir>/results/<name>/test_latest/images \
  --text-prompt "<task's prompt>"
```

Pull these three numbers out of the written `metrics.json`'s `summary` for the report table:
- `lpips.mean` → **LPIPS**
- `jf_af_avg_objects.mean` → **J&F (real_A vs fake_B)** — this is geometry alignment of the
  translation itself (does the object stay where the sim input put it), not `jf_mean_avg_objects`
  (which compares against real_B instead — a different, also-valid number, but not what past
  tables in this project have reported; ask if unsure which the user wants).
- `missing_masks` (`real_A`/`fake_B`/`real_B` counts) → **missing (A/fake/real)**

Report as one table, one row per task/camera, e.g.:

```
┌───────────────┬────────┬────────────────────────┬───────────────────────┐
│   Task/Cam    │ LPIPS  │ J&F (real_A vs fake_B) │ missing (A/fake/real) │
├───────────────┼────────┼────────────────────────┼───────────────────────┤
│ Mug/Stat      │ 0.1045 │ 0.9264                 │ 0/0/0                 │
└───────────────┴────────┴────────────────────────┴───────────────────────┘
```

## Notes

- `eval_turbo.py`'s translator is deterministic (`deterministic=True`, no sampling noise) — the
  same checkpoint + same input image always reproduces the same `fake_B` byte-for-byte. This
  matters if you ever need to explain why re-running inference invalidated a manual review: the
  images only change if the *input* images changed, not from run-to-run noise.
- `outputs/` is gitignored — there's no version history to fall back on if a `fake_B` review gets
  invalidated by regenerating images from different inputs. If you're about to overwrite an
  `images/` dir that has a populated `mask_overrides.json` next to it, flag that to the user
  *before* running, not after.
- If the user wants a new checkpoint's results reported/plotted alongside existing baselines
  (e.g. in a paper figure), that naming decision is the user's call — confirm the exact label
  with them before sending it to any other session/collaborator.
