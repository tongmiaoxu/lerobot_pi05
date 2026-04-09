#!/usr/bin/env python
"""Recompute per-channel image/video stats from decoded frames."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from lerobot.datasets.compute_stats import aggregate_stats, estimate_num_samples, get_feature_stats
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import load_stats, write_stats


def _sample_indices(data_len: int) -> list[int]:
    num_samples = estimate_num_samples(data_len)
    return np.round(np.linspace(0, data_len - 1, num_samples)).astype(int).tolist()


def _episode_image_stats(
    ds: LeRobotDataset, from_idx: int, length: int, video_keys: list[str]
) -> dict[str, dict[str, np.ndarray]]:
    if length < 2:
        raise ValueError("episode too short for stats")
    idxs = _sample_indices(length)
    out: dict[str, dict[str, np.ndarray]] = {}
    for key in video_keys:
        imgs = []
        for j in idxs:
            g = from_idx + int(j)
            imgs.append(ds[g][key].numpy())
        arr = np.stack(imgs, axis=0).astype(np.float64)
        raw = get_feature_stats(arr, axis=(0, 2, 3), keepdims=True)
        out[key] = {k: v if k == "count" else np.squeeze(v, axis=0) for k, v in raw.items()}
    return out


def recompute_image_stats(root: Path, repo_id: str = "local/xarm") -> None:
    ds = LeRobotDataset(repo_id, root=root, download_videos=False)
    ep_df = pd.read_parquet(root / "meta/episodes/chunk-000/file-000.parquet")
    video_keys = list(ds.meta.video_keys)

    ep_stats_list: list[dict[str, dict[str, np.ndarray]]] = []
    rows_updated: list[tuple[int, dict[str, dict[str, np.ndarray]]]] = []

    for _, row in tqdm(ep_df.iterrows(), total=len(ep_df), desc="episodes"):
        ep_idx = int(row["episode_index"])
        from_idx = int(row["dataset_from_index"])
        to_idx = int(row["dataset_to_index"])
        length = to_idx - from_idx
        st = _episode_image_stats(ds, from_idx, length, video_keys)
        ep_stats_list.append(st)
        rows_updated.append((ep_idx, st))

    existing = load_stats(root)
    if existing is None:
        raise FileNotFoundError(root / "meta/stats.json")
    agg_images = aggregate_stats(ep_stats_list)
    for k, v in agg_images.items():
        existing[k] = v
    write_stats(existing, root)

    for ep_idx, st in rows_updated:
        row_ix = ep_df.index[ep_df["episode_index"] == ep_idx]
        assert len(row_ix) == 1
        i = row_ix[0]
        for feat_name, feat_stats in st.items():
            for stat_name, arr in feat_stats.items():
                col = f"stats/{feat_name}/{stat_name}"
                # Match LeRobot `compute_episode_stats` + `flatten_dict` serialization: per-channel
                # stats are shape (3, 1, 1) so JSON/Parquet use nested lists [[[r]],[[g]],[[b]]].
                # Training reads normalization from `meta/stats.json` only; episode stats are for
                # metadata/tools (e.g. dataset merge) and must match library expectations.
                a = np.asarray(arr)
                if stat_name == "count":
                    ep_df.at[i, col] = a.reshape(-1).astype(np.int64).tolist()
                else:
                    ep_df.at[i, col] = a.astype(np.float64).reshape(3, 1, 1).tolist()

    ep_path = root / "meta/episodes/chunk-000/file-000.parquet"
    ep_df.to_parquet(ep_path, index=False)
    print("Wrote", root / "meta/stats.json")
    print("Wrote", ep_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset_root",
        nargs="?",
        default=str(Path(__file__).resolve().parents[1] / "data"),
        help="Path to the local LeRobot dataset root (default: ./data)",
    )
    parser.add_argument(
        "--repo-id",
        default="local/xarm",
        help="Repo id label used when opening the local dataset (default: local/xarm)",
    )
    args = parser.parse_args()

    root = Path(args.dataset_root).expanduser().resolve()
    recompute_image_stats(root=root, repo_id=args.repo_id)


if __name__ == "__main__":
    main()
