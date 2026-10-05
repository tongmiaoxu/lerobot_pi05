#!/usr/bin/env python3
"""Grid viewer for eyeballing Grounding-DINO+SAM2 detection over an entire pix2pix eval set,
and fixing the bad ones without re-running the whole metrics pass.

Shows all <idx>_<role>.png images (default role: fake_B, the one most likely to trip up the
detector) as a thumbnail grid, each bordered by its current detection status for --text-prompt's
first object (green = one clean detection, yellow = multiple boxes found — candidate #1, drawn
in green and labeled "1", is what eval_pix2pix_metrics.py uses by default, so only click a
yellow cell if #1 is NOT actually on the right object, red = none found, orange = point override
saved, purple = confirmed no object present).
Click a thumbnail to open it full-size and fix it: number keys pick a candidate box (saved as
that candidate's index — detection is deterministic, so a future run re-selects the exact same
mask verbatim instead of re-segmenting anything), a left click runs a fresh SAM2 point-prompt,
'x' confirms the object is genuinely absent from this image (important
for red/missing cells — eval_pix2pix_metrics.py now scores an unconfirmed missing mask as a
failed detection when the other side of the fake_B/real_B pair has the object, so mark true
negatives explicitly to keep them out of that penalty), 's' skips without saving. Every fix is
written to mask_overrides.json (next to metrics.json) immediately, not just on quit, so nothing
is lost if the window hangs and you have to kill the process. Picked up automatically the next
time eval_pix2pix_metrics.py runs — this tool never computes/writes metrics itself.

Multi-object prompts (e.g. "mug, saucer") are reviewed one object at a time: press 'n'/'p' to
switch which object's detection the grid is showing, or 'r' to switch which image role
(real_A/fake_B/real_B) is shown.

Reviewing a whole joint checkpoint (all 8 task/camera result sets) needs no restarts: pass
--checkpoint-dir instead of --images-dir/--text-prompt and every results/<name>/test_latest/
under it is auto-discovered, matched to its task's fixed --text-prompt via
pix2pix_mask_common.TASK_PROMPTS (by name prefix — place_mug/pick_shoe/book_shelving/pouring),
and queued up in one continuous session. 'n'/'p' walk a single flattened (task/camera, object)
list — reaching the last object of one result set's prompt just continues into the next result
set's first object, loading its own images/overrides automatically. '['/']' jump straight to the
previous/next result set (skip past one you already finished, without stepping through every
object). The window title always shows which result set (e.g. "3/8: place_mug_wrist_..."). A
checkpoint trained for a single task/camera (e.g. turbo_sim2real_stationary_dino_book) still
works with --checkpoint-dir — it just discovers the one result set under it.

--manual mode (for when the text prompt is unreliable for this object): skips Grounding-DINO
entirely and has two sub-modes you toggle between with 'd':

  PLACING (the starting sub-mode): every cell not yet overridden starts blank/"unclicked" (gray
  border). Left-click directly on a thumbnail to drop a point on the object — this only queues
  the point (yellow dot, blue border) and does NOT run SAM yet, so you can click through the
  whole grid quickly. Right-click a cell to mark it "confirmed no object here" (purple) without
  placing a point. Press 'd' to run SAM2 on every queued point, save the results, and switch into
  REVIEW.

  REVIEW: the grid shows the current final state of every cell (a saved point override is
  re-segmented and shown, a "missing" override shows purple, anything never touched stays
  "unclicked"). real_A/fake_B/real_B are pixel-aligned renders of the same frame, so a point
  saved for one role is automatically borrowed and persisted for the other roles too the first
  time you view them here — you only need to click each image once, in whichever role you
  reviewed it in, not once per role. ("missing" is NOT shared this way, since whether an object
  actually renders is exactly what can differ between roles — e.g. fake_B failing to draw
  something real_A/real_B clearly have.) Clicking a cell here opens the same full-size fixer
  popup the tool always used —
  click a point / 'x' confirm no object / 's' skip — for correcting a bad result, exactly like
  before this mode existed (there are never numbered candidates to pick here, since manual mode
  never runs Grounding-DINO). Press 'd' again to switch back to PLACING and queue points for any
  images you haven't touched yet.

'n'/'p'/'r' still switch object/role from either sub-mode (any queued-but-undetected points are
saved first so nothing is silently dropped), and 'q' quits (also saving first).

Usage:
  python scripts/review_pix2pix_masks.py \\
    --images-dir outputs/pix2pix_stationary_mug/results/place_mug_stationary_pix2pix/test_latest/images \\
    --text-prompt "mug, saucer"

  python scripts/review_pix2pix_masks.py \\
    --images-dir outputs/pix2pix_wrist_mug/results/place_mug_wrist_pix2pix/test_latest/images \\
    --text-prompt "mug, plate" --manual

  # One session for all 8 task/camera result sets under a joint checkpoint's output dir:
  python scripts/review_pix2pix_masks.py --checkpoint-dir outputs/pix2pix_dino_all_tasks --manual
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "scripts"))
from pix2pix_mask_common import (  # noqa: E402
    TASK_PROMPTS,
    default_overrides_path,
    draw_candidates,
    find_triplet_indices,
    get_override_candidate_index,
    get_override_mask,
    get_override_point,
    is_override_missing,
    load_overrides,
    parse_object_list,
    prompt_for_mask,
    save_overrides,
    set_override_mask,
    set_override_missing,
)
from segmentation_utils import segment_candidate_masks, segment_point_mask  # noqa: E402

THUMB_W, THUMB_H = 160, 120
PAD = 8
ROLES = ("real_A", "fake_B", "real_B")

STATUS_COLOR = {
    "ok": (0, 200, 0),
    "ambiguous": (0, 200, 255),
    "missing": (0, 0, 255),
    "override": (255, 180, 0),
    "override_missing": (128, 0, 128),
    "unclicked": (120, 120, 120),
    "queued": (255, 255, 0),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--images-dir", type=Path, default=None,
        help="Single result set: folder of <idx>_real_A/fake_B/real_B.png triplets. Requires --text-prompt. "
        "Mutually exclusive with --checkpoint-dir.",
    )
    parser.add_argument(
        "--text-prompt", type=str, default=None,
        help="Object(s), comma-separated, e.g. 'mug, saucer'. Required with --images-dir; unused/invalid "
        "with --checkpoint-dir (each result set's prompt is looked up from TASK_PROMPTS instead). In "
        "--manual mode these are only labels (no Grounding-DINO text detection is run).",
    )
    parser.add_argument(
        "--checkpoint-dir", type=Path, default=None,
        help="A checkpoint's output dir (e.g. outputs/pix2pix_dino_all_tasks) — auto-discovers every "
        "results/<name>/test_latest/images under it and reviews them all in one continuous session, "
        "each with its task's fixed --text-prompt from TASK_PROMPTS. Mutually exclusive with "
        "--images-dir/--text-prompt/--overrides-json.",
    )
    parser.add_argument(
        "--manual", action="store_true",
        default=True,
        help="Skip Grounding-DINO auto-detection; click a point per image directly on the grid "
        "instead, then press 'd' to run SAM2 on all queued points. Use when --text-prompt is "
        "unreliable for this object.",
    )
    parser.add_argument("--role", choices=ROLES, default="fake_B", help="Which image to show/review first.")
    parser.add_argument("--box-threshold", type=float, default=0.325)
    parser.add_argument("--text-threshold", type=float, default=0.3)
    parser.add_argument("--columns", type=int, default=5)
    parser.add_argument(
        "--overrides-json", type=Path, default=None,
        help="Default: <images-dir>/../mask_overrides.json (shared with eval_pix2pix_metrics.py). "
        "Only valid with --images-dir; each --checkpoint-dir result set always uses its own default path.",
    )
    args = parser.parse_args()

    if args.checkpoint_dir is not None:
        if args.images_dir is not None or args.text_prompt is not None or args.overrides_json is not None:
            parser.error("--checkpoint-dir is mutually exclusive with --images-dir/--text-prompt/--overrides-json")
    elif args.images_dir is None or args.text_prompt is None:
        parser.error("either --checkpoint-dir, or both --images-dir and --text-prompt, are required")

    return args


def discover_combos(checkpoint_dir: Path) -> list[dict]:
    """Find every results/<name>/test_latest/images under checkpoint_dir, matching each <name>
    to its task's fixed --text-prompt via TASK_PROMPTS (longest-prefix-first, so no task prefix
    can accidentally shadow another). Sorted by TASK_PROMPTS' own order then name, so repeated
    invocations (e.g. resuming a review) always walk the result sets in the same order."""
    results_dir = checkpoint_dir / "results"
    if not results_dir.is_dir():
        raise FileNotFoundError(f"No results/ dir under {checkpoint_dir}")

    prefixes_by_length = sorted(TASK_PROMPTS, key=len, reverse=True)
    task_order = {prefix: i for i, prefix in enumerate(TASK_PROMPTS)}

    combos = []
    for images_dir in sorted(results_dir.glob("*/test_latest/images")):
        if not images_dir.is_dir():
            continue
        name = images_dir.parent.parent.name
        prefix = next((p for p in prefixes_by_length if name.startswith(p)), None)
        if prefix is None:
            print(f"[WARN] Skipping {name!r}: no task prefix in TASK_PROMPTS matches it, don't know its --text-prompt")
            continue
        combos.append(
            {
                "name": name,
                "images_dir": images_dir,
                "overrides_path": default_overrides_path(images_dir),
                "objects": parse_object_list(TASK_PROMPTS[prefix]),
                "_sort_key": (task_order[prefix], name),
            }
        )
    if not combos:
        raise FileNotFoundError(f"No task/camera result sets with a recognized task prefix found under {results_dir}")
    combos.sort(key=lambda c: c.pop("_sort_key"))
    return combos


def status_for(masks: list[np.ndarray], override_kind: str | None) -> str:
    if override_kind == "unclicked":
        return "unclicked"
    if override_kind == "missing":
        return "override_missing"
    if override_kind in ("point", "candidate", "mask"):
        return "override"
    if len(masks) == 0:
        return "missing"
    if len(masks) > 1:
        return "ambiguous"
    return "ok"


def make_thumb(image_bgr: np.ndarray, masks: list[np.ndarray], idx: str, status: str) -> np.ndarray:
    vis = draw_candidates(image_bgr, masks) if masks else image_bgr.copy()
    thumb = cv2.resize(vis, (THUMB_W, THUMB_H))
    color = STATUS_COLOR[status]
    thumb = cv2.copyMakeBorder(thumb, PAD, PAD, PAD, PAD, cv2.BORDER_CONSTANT, value=color)
    cv2.putText(thumb, idx, (PAD + 4, PAD + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return thumb


def build_grid(thumbs: list[np.ndarray], columns: int) -> np.ndarray:
    rows = (len(thumbs) + columns - 1) // columns
    cell_h, cell_w = thumbs[0].shape[:2]
    grid = np.zeros((rows * cell_h, columns * cell_w, 3), dtype=np.uint8)
    for i, thumb in enumerate(thumbs):
        r, c = divmod(i, columns)
        grid[r * cell_h : (r + 1) * cell_h, c * cell_w : (c + 1) * cell_w] = thumb
    return grid, cell_h, cell_w


def draw_cell_border(grid: np.ndarray, row: int, col: int, cell_h: int, cell_w: int, color: tuple) -> None:
    """Repaint just one cell's PAD-thick border in place, without recomputing/rebuilding the grid."""
    top, left = row * cell_h, col * cell_w
    grid[top : top + PAD, left : left + cell_w] = color
    grid[top + cell_h - PAD : top + cell_h, left : left + cell_w] = color
    grid[top : top + cell_h, left : left + PAD] = color
    grid[top : top + cell_h, left + cell_w - PAD : left + cell_w] = color


def draw_cell_marker(grid: np.ndarray, row: int, col: int, cell_h: int, cell_w: int, local_xy: tuple) -> None:
    """Draw a small dot at a queued point's location within its thumbnail, in place."""
    top, left = row * cell_h, col * cell_w
    lx, ly = local_xy
    cv2.circle(grid, (left + PAD + lx, top + PAD + ly), 4, (0, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(grid, (left + PAD + lx, top + PAD + ly), 5, (0, 0, 0), 1, cv2.LINE_AA)


def locate_cell(x: int, y: int, columns: int, cell_h: int, cell_w: int, n_indices: int):
    """Map a raw grid-pixel click to (cell_idx, row, col, local_xy_within_thumbnail), or None if
    the click missed every cell (a trailing partial row) or landed on the padding border."""
    col, row = x // cell_w, y // cell_h
    cell_idx = row * columns + col
    if not (0 <= cell_idx < n_indices):
        return None
    local_x = min(max(x - col * cell_w - PAD, 0), THUMB_W - 1)
    local_y = min(max(y - row * cell_h - PAD, 0), THUMB_H - 1)
    return cell_idx, row, col, (local_x, local_y)


def local_to_image_point(local_xy: tuple, image_bgr: np.ndarray) -> tuple[int, int]:
    orig_h, orig_w = image_bgr.shape[:2]
    lx, ly = local_xy
    return int(lx * orig_w / THUMB_W), int(ly * orig_h / THUMB_H)


def repaint_cell(view: dict, idx: str, row: int, col: int) -> None:
    """Redraw one cell from its cached (already-segmented-or-not) image/masks, with no SAM call.
    Used before drawing a fresh queued-point marker so re-clicking a cell doesn't stack old dots."""
    image_bgr, masks = view["cache"][idx]
    status = "override" if masks else "unclicked"
    thumb = make_thumb(image_bgr, masks, idx, status)
    cell_h, cell_w = view["cell_h"], view["cell_w"]
    top, left = row * cell_h, col * cell_w
    view["grid"][top : top + cell_h, left : left + cell_w] = thumb


def main() -> int:
    args = parse_args()

    if args.checkpoint_dir is not None:
        combos = discover_combos(args.checkpoint_dir)
    else:
        combos = [
            {
                "name": args.images_dir.name,
                "images_dir": args.images_dir,
                "overrides_path": args.overrides_json or default_overrides_path(args.images_dir),
                "objects": parse_object_list(args.text_prompt),
            }
        ]

    ctx: dict = {}

    def load_combo(combo_idx: int) -> None:
        """Point ctx at combos[combo_idx]'s own images/overrides/indices/objects — called on
        startup and whenever navigation crosses into a different task/camera result set."""
        combo = combos[combo_idx]
        ctx["images_dir"] = combo["images_dir"]
        ctx["overrides_path"] = combo["overrides_path"]
        ctx["overrides"] = load_overrides(combo["overrides_path"])
        ctx["indices"] = find_triplet_indices(combo["images_dir"])
        ctx["objects"] = combo["objects"]

    state = {"combo_idx": 0, "role": args.role, "obj_idx": 0, "mode": "placing"}  # mode only meaningful when args.manual
    load_combo(state["combo_idx"])
    pending: dict[str, tuple[int, int]] = {}  # idx -> queued point in original-image coords (manual mode only)
    key_hint = "d:toggle placing/review" if args.manual else "click:fix"
    combo_hint = ", [/]:result set" if len(combos) > 1 else ""
    window = f"Detection review (n/p: object{combo_hint}, r: role, {key_hint}, q: quit)"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    def current_object() -> str:
        return ctx["objects"][state["obj_idx"]]

    def advance_object(delta: int) -> None:
        """Move to the next/previous object, crossing into the next/previous combo (task/camera
        result set) at either end of its object list — flattens every combo's objects into one
        continuous walk so a whole joint checkpoint needs no restarts between result sets."""
        obj_idx = state["obj_idx"] + delta
        if 0 <= obj_idx < len(ctx["objects"]):
            state["obj_idx"] = obj_idx
            return
        combo_idx = (state["combo_idx"] + delta) % len(combos)
        state["combo_idx"] = combo_idx
        load_combo(combo_idx)
        state["obj_idx"] = 0 if delta > 0 else len(ctx["objects"]) - 1

    def jump_combo(delta: int) -> None:
        """Jump straight to the previous/next result set, skipping the rest of the current
        one's objects — for skipping past an already-finished result set quickly."""
        combo_idx = (state["combo_idx"] + delta) % len(combos)
        state["combo_idx"] = combo_idx
        load_combo(combo_idx)
        state["obj_idx"] = 0

    def detect(idx: str, role: str, obj_name: str):
        overrides, overrides_path = ctx["overrides"], ctx["overrides_path"]
        image_bgr = cv2.imread(str(ctx["images_dir"] / f"{idx}_{role}.png"))
        override_mask = get_override_mask(overrides, idx, obj_name, role)
        if override_mask is not None:
            return image_bgr, [override_mask], "mask"
        if is_override_missing(overrides, idx, obj_name, role):
            return image_bgr, [], "missing"
        override_point = get_override_point(overrides, idx, obj_name, role)
        if override_point is None and args.manual:
            # real_A/fake_B/real_B are pixel-aligned renders of the same frame, so a point placed
            # while reviewing one role is a valid point for the others too. Borrow it from
            # whichever other role already has one, re-segment it for *this* role's own image,
            # and persist the resulting mask (plus the point, so it can still be borrowed again)
            # under this role as well — eval_pix2pix_metrics.py reads overrides per-role directly
            # and has no such fallback, so without this, roles you never happened to click
            # through here would stay ungrounded and get scored as missing detections downstream.
            for other_role in ROLES:
                if other_role == role:
                    continue
                borrowed_point = get_override_point(overrides, idx, obj_name, other_role)
                if borrowed_point is not None:
                    mask = segment_point_mask(image_bgr, borrowed_point)
                    set_override_mask(overrides, idx, obj_name, role, mask, point=borrowed_point)
                    save_overrides(overrides_path, overrides)
                    return image_bgr, [mask], "mask"
        if override_point is not None:
            # Legacy override predating mask-caching: still re-derived from the saved point
            # every time, rather than a stable saved mask (see get_override_mask).
            return image_bgr, [segment_point_mask(image_bgr, override_point)], "point"
        if args.manual:
            return image_bgr, [], "unclicked"
        masks = segment_candidate_masks(
            image_bgr, text_prompt=obj_name, box_threshold=args.box_threshold, text_threshold=args.text_threshold
        )
        candidate_index = get_override_candidate_index(overrides, idx, obj_name, role)
        if candidate_index is not None and 0 <= candidate_index < len(masks):
            # Legacy override predating mask-caching: still re-derived by re-running detection
            # and indexing in, rather than a stable saved mask (see get_override_mask).
            return image_bgr, [masks[candidate_index]], "candidate"
        return image_bgr, masks, None

    def rebuild():
        role, obj_name = state["role"], current_object()
        print(f"[INFO] Scanning {len(ctx['indices'])} images for object={obj_name!r} role={role!r} "
              f"(result set {state['combo_idx'] + 1}/{len(combos)}: {combos[state['combo_idx']]['name']}) ...")
        thumbs = []
        cache = {}
        for idx in ctx["indices"]:
            image_bgr, masks, override_kind = detect(idx, role, obj_name)
            cache[idx] = (image_bgr, masks)
            thumbs.append(make_thumb(image_bgr, masks, idx, status_for(masks, override_kind)))
        grid, cell_h, cell_w = build_grid(thumbs, args.columns)
        if args.manual:
            mode_hint = (
                "PLACING: left-click=queue point, right-click=missing, d=detect+review"
                if state["mode"] == "placing"
                else "REVIEW: click a cell to fix (point/x/s) | d=back to placing"
            )
        else:
            mode_hint = "yellow=#1 used by default, click only if #1 wrong"
        combo_prefix = f"[{state['combo_idx'] + 1}/{len(combos)}: {combos[state['combo_idx']]['name']}] " if len(combos) > 1 else ""
        title = f"{combo_prefix}object={obj_name} role={role} | n/p:object r:role q:quit | {mode_hint}"
        cv2.putText(grid, title, (10, grid.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        return grid, cell_h, cell_w, cache

    view = {}
    view["grid"], view["cell_h"], view["cell_w"], view["cache"] = rebuild()
    clicked = {"xy": None, "button": None}

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked["xy"], clicked["button"] = (x, y), "left"
        elif event == cv2.EVENT_RBUTTONDOWN:
            clicked["xy"], clicked["button"] = (x, y), "right"

    cv2.setMouseCallback(window, on_mouse)
    cv2.imshow(window, view["grid"])

    def apply_pending():
        """Segment every queued manual point and save the resulting mask (no rebuild/redraw).
        Must be called before any navigation (advance_object/jump_combo/state["role"] change)
        actually switches ctx away from the combo/object/role these points were queued under."""
        if not pending:
            return
        overrides, overrides_path = ctx["overrides"], ctx["overrides_path"]
        obj_name, role = current_object(), state["role"]
        for idx, point in pending.items():
            image_bgr, _ = view["cache"][idx]
            mask = segment_point_mask(image_bgr, point)
            set_override_mask(overrides, idx, obj_name, role, mask, point=point)
        pending.clear()
        save_overrides(overrides_path, overrides)

    def freeze_current_view():
        """Persist every currently-resolved cell in the active object/role view as a stable
        saved mask, including ones the reviewer deliberately left un-clicked because the shown
        default already looked correct (a clean single detection, an accepted #1 on an ambiguous
        cell, or an existing legacy point/candidate override) — not just ones just fixed by a
        click. Confirmed-missing and truly unresolved cells (no detection, no override) are left
        exactly as-is; there's no mask to freeze there. Must be called before any navigation
        actually switches ctx away from the combo/object/role this view is showing."""
        overrides, overrides_path = ctx["overrides"], ctx["overrides_path"]
        obj_name, role = current_object(), state["role"]
        changed = False
        for idx in ctx["indices"]:
            if get_override_mask(overrides, idx, obj_name, role) is not None:
                continue
            if is_override_missing(overrides, idx, obj_name, role):
                continue
            _, masks = view["cache"][idx]
            if not masks:
                continue
            point = get_override_point(overrides, idx, obj_name, role)
            set_override_mask(overrides, idx, obj_name, role, masks[0], point=point)
            changed = True
        if changed:
            save_overrides(overrides_path, overrides)

    dirty = False
    while True:
        key = cv2.waitKeyEx(30)
        if clicked["xy"] is not None:
            x, y = clicked["xy"]
            button = clicked["button"]
            clicked["xy"], clicked["button"] = None, None
            located = locate_cell(x, y, args.columns, view["cell_h"], view["cell_w"], len(ctx["indices"]))
            if located is not None:
                cell_idx, row, col, local_xy = located
                idx = ctx["indices"][cell_idx]
                overrides, overrides_path = ctx["overrides"], ctx["overrides_path"]
                obj_name, role = current_object(), state["role"]

                if args.manual and state["mode"] == "placing":
                    if button == "right":
                        pending.pop(idx, None)
                        set_override_missing(overrides, idx, obj_name, role)
                        dirty = True
                        save_overrides(overrides_path, overrides)
                        print(f"[INFO] {idx}/{obj_name}/{role}: confirmed no object present, saved override")
                        view["grid"], view["cell_h"], view["cell_w"], view["cache"] = rebuild()
                        cv2.imshow(window, view["grid"])
                    else:
                        image_bgr, _ = view["cache"][idx]
                        pending[idx] = local_to_image_point(local_xy, image_bgr)
                        dirty = True
                        repaint_cell(view, idx, row, col)
                        draw_cell_border(view["grid"], row, col, view["cell_h"], view["cell_w"], STATUS_COLOR["queued"])
                        draw_cell_marker(view["grid"], row, col, view["cell_h"], view["cell_w"], local_xy)
                        cv2.imshow(window, view["grid"])
                        print(f"[INFO] {idx}/{obj_name}/{role}: queued point {pending[idx]} (press 'd' to detect)")
                else:
                    # Non-manual, or manual mode's REVIEW sub-mode: the original click-to-fix popup.
                    image_bgr, masks = view["cache"][idx]
                    mask, tag, point, candidate_index = prompt_for_mask(image_bgr, masks, f"{idx} {obj_name} {role}")
                    if point is not None:
                        set_override_mask(overrides, idx, obj_name, role, mask, point=point)
                        dirty = True
                        save_overrides(overrides_path, overrides)
                        print(f"[INFO] {idx}/{obj_name}/{role}: saved mask from point override at {point}")
                    elif tag == "missing":
                        set_override_missing(overrides, idx, obj_name, role)
                        dirty = True
                        save_overrides(overrides_path, overrides)
                        print(f"[INFO] {idx}/{obj_name}/{role}: confirmed no object present, saved override")
                    elif tag == "pick" and not args.manual:
                        # Meaningless in manual mode: the "candidate" shown there is just a saved
                        # point override re-segmented, not an independent Grounding-DINO detection,
                        # so picking it would silently downgrade a point override to a broken
                        # candidate_index one. Only real (text-prompt) runs have real candidates.
                        set_override_mask(overrides, idx, obj_name, role, mask)
                        dirty = True
                        save_overrides(overrides_path, overrides)
                        print(f"[INFO] {idx}/{obj_name}/{role}: saved mask from candidate #{candidate_index + 1} (reused verbatim, no re-segmentation)")
                    view["grid"], view["cell_h"], view["cell_w"], view["cache"] = rebuild()
                    cv2.imshow(window, view["grid"])
            continue
        if key == ord("q"):
            apply_pending()
            freeze_current_view()
            break
        if key == ord("d") and args.manual:
            if state["mode"] == "placing":
                apply_pending()
                freeze_current_view()
                state["mode"] = "review"
            else:
                state["mode"] = "placing"
            view["grid"], view["cell_h"], view["cell_w"], view["cache"] = rebuild()
            cv2.imshow(window, view["grid"])
        elif key == ord("n"):
            apply_pending()
            freeze_current_view()
            advance_object(+1)
            view["grid"], view["cell_h"], view["cell_w"], view["cache"] = rebuild()
            cv2.imshow(window, view["grid"])
        elif key == ord("p"):
            apply_pending()
            freeze_current_view()
            advance_object(-1)
            view["grid"], view["cell_h"], view["cell_w"], view["cache"] = rebuild()
            cv2.imshow(window, view["grid"])
        elif key == ord("r"):
            apply_pending()
            freeze_current_view()
            state["role"] = ROLES[(ROLES.index(state["role"]) + 1) % len(ROLES)]
            view["grid"], view["cell_h"], view["cell_w"], view["cache"] = rebuild()
            cv2.imshow(window, view["grid"])
        elif key == ord("]") and len(combos) > 1:
            apply_pending()
            freeze_current_view()
            jump_combo(+1)
            view["grid"], view["cell_h"], view["cell_w"], view["cache"] = rebuild()
            cv2.imshow(window, view["grid"])
        elif key == ord("[") and len(combos) > 1:
            apply_pending()
            freeze_current_view()
            jump_combo(-1)
            view["grid"], view["cell_h"], view["cell_w"], view["cache"] = rebuild()
            cv2.imshow(window, view["grid"])

    cv2.destroyAllWindows()
    if dirty:
        print(f"[INFO] Saved overrides across {len(combos)} result set(s).")
    else:
        print("[INFO] No changes made.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
