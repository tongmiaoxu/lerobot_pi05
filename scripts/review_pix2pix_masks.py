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
    default_overrides_path,
    draw_candidates,
    find_triplet_indices,
    get_override_candidate_index,
    get_override_point,
    is_override_missing,
    load_overrides,
    parse_object_list,
    prompt_for_mask,
    save_overrides,
    set_override_candidate_index,
    set_override_missing,
    set_override_point,
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
    parser.add_argument("--images-dir", type=Path, required=True, help="Folder of <idx>_real_A/fake_B/real_B.png triplets.")
    parser.add_argument(
        "--text-prompt", type=str, required=True,
        help="Object(s), comma-separated, e.g. 'mug, saucer'. In --manual mode these are only "
        "labels (no Grounding-DINO text detection is run).",
    )
    parser.add_argument(
        "--manual", action="store_true",
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
        help="Default: <images-dir>/../mask_overrides.json (shared with eval_pix2pix_metrics.py).",
    )
    return parser.parse_args()


def status_for(masks: list[np.ndarray], override_kind: str | None) -> str:
    if override_kind == "unclicked":
        return "unclicked"
    if override_kind == "missing":
        return "override_missing"
    if override_kind in ("point", "candidate"):
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
    objects = parse_object_list(args.text_prompt)
    overrides_path = args.overrides_json or default_overrides_path(args.images_dir)
    overrides = load_overrides(overrides_path)
    indices = find_triplet_indices(args.images_dir)

    state = {"role": args.role, "obj_idx": 0, "mode": "placing"}  # mode only meaningful when args.manual
    pending: dict[str, tuple[int, int]] = {}  # idx -> queued point in original-image coords (manual mode only)
    key_hint = "d:toggle placing/review" if args.manual else "click:fix"
    window = f"Detection review (n/p: object, r: role, {key_hint}, q: quit)"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    def current_object() -> str:
        return objects[state["obj_idx"]]

    def detect(idx: str, role: str, obj_name: str):
        image_bgr = cv2.imread(str(args.images_dir / f"{idx}_{role}.png"))
        if is_override_missing(overrides, idx, obj_name, role):
            return image_bgr, [], "missing"
        override_point = get_override_point(overrides, idx, obj_name, role)
        if override_point is None and args.manual:
            # real_A/fake_B/real_B are pixel-aligned renders of the same frame, so a point placed
            # while reviewing one role is a valid point for the others too. Borrow it from
            # whichever other role already has one, and persist it under this role as well —
            # eval_pix2pix_metrics.py reads overrides per-role directly and has no such fallback,
            # so without this, roles you never happened to click through here would stay
            # ungrounded and get scored as missing detections downstream.
            for other_role in ROLES:
                if other_role == role:
                    continue
                borrowed = get_override_point(overrides, idx, obj_name, other_role)
                if borrowed is not None:
                    set_override_point(overrides, idx, obj_name, role, borrowed)
                    save_overrides(overrides_path, overrides)
                    override_point = borrowed
                    break
        if override_point is not None:
            return image_bgr, [segment_point_mask(image_bgr, override_point)], "point"
        if args.manual:
            return image_bgr, [], "unclicked"
        masks = segment_candidate_masks(
            image_bgr, text_prompt=obj_name, box_threshold=args.box_threshold, text_threshold=args.text_threshold
        )
        candidate_index = get_override_candidate_index(overrides, idx, obj_name, role)
        if candidate_index is not None and 0 <= candidate_index < len(masks):
            return image_bgr, [masks[candidate_index]], "candidate"
        return image_bgr, masks, None

    def rebuild():
        role, obj_name = state["role"], current_object()
        print(f"[INFO] Scanning {len(indices)} images for object={obj_name!r} role={role!r} ...")
        thumbs = []
        cache = {}
        for idx in indices:
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
        title = f"object={obj_name} role={role} | n/p:object r:role q:quit | {mode_hint}"
        cv2.putText(grid, title, (10, grid.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
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
        """Save every queued manual point as a point override (no rebuild/redraw)."""
        if not pending:
            return
        obj_name, role = current_object(), state["role"]
        for idx, point in pending.items():
            set_override_point(overrides, idx, obj_name, role, point)
        pending.clear()
        save_overrides(overrides_path, overrides)

    dirty = False
    while True:
        key = cv2.waitKeyEx(30)
        if clicked["xy"] is not None:
            x, y = clicked["xy"]
            button = clicked["button"]
            clicked["xy"], clicked["button"] = None, None
            located = locate_cell(x, y, args.columns, view["cell_h"], view["cell_w"], len(indices))
            if located is not None:
                cell_idx, row, col, local_xy = located
                idx = indices[cell_idx]
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
                        set_override_point(overrides, idx, obj_name, role, point)
                        dirty = True
                        save_overrides(overrides_path, overrides)
                        print(f"[INFO] {idx}/{obj_name}/{role}: saved point override at {point}")
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
                        set_override_candidate_index(overrides, idx, obj_name, role, candidate_index)
                        dirty = True
                        save_overrides(overrides_path, overrides)
                        print(f"[INFO] {idx}/{obj_name}/{role}: saved candidate #{candidate_index + 1} as override (reused verbatim, no re-segmentation)")
                    view["grid"], view["cell_h"], view["cell_w"], view["cache"] = rebuild()
                    cv2.imshow(window, view["grid"])
            continue
        if key == ord("q"):
            apply_pending()
            break
        if key == ord("d") and args.manual:
            if state["mode"] == "placing":
                apply_pending()
                state["mode"] = "review"
            else:
                state["mode"] = "placing"
            view["grid"], view["cell_h"], view["cell_w"], view["cache"] = rebuild()
            cv2.imshow(window, view["grid"])
        elif key == ord("n"):
            apply_pending()
            state["obj_idx"] = (state["obj_idx"] + 1) % len(objects)
            view["grid"], view["cell_h"], view["cell_w"], view["cache"] = rebuild()
            cv2.imshow(window, view["grid"])
        elif key == ord("p"):
            apply_pending()
            state["obj_idx"] = (state["obj_idx"] - 1) % len(objects)
            view["grid"], view["cell_h"], view["cell_w"], view["cache"] = rebuild()
            cv2.imshow(window, view["grid"])
        elif key == ord("r"):
            apply_pending()
            state["role"] = ROLES[(ROLES.index(state["role"]) + 1) % len(ROLES)]
            view["grid"], view["cell_h"], view["cell_w"], view["cache"] = rebuild()
            cv2.imshow(window, view["grid"])

    cv2.destroyAllWindows()
    if dirty:
        save_overrides(overrides_path, overrides)
        print(f"[INFO] Saved overrides to {overrides_path}")
    else:
        print("[INFO] No changes made.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
