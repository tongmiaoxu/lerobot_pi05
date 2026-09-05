#!/usr/bin/env python3
"""Aggregate scripts/eval_pix2pix_metrics.py's metrics.json across multiple runs (task x
camera) into one visual-alignment number and one geometry-alignment number, for the baseline
(no GAN) and model pairs:

  Baseline (sim vs real, no translation):
    visual   = lpips_base       = LPIPS(real_A, real_B)
    geometry = jf_base_avg_objects = J&F(real_A, real_B)

  Model:
    visual   = lpips            = LPIPS(fake_B, real_B)          -- output vs real target
    geometry = jf_af_avg_objects   = J&F(real_A, fake_B)         -- does the translation preserve
                                                                     the sim layout/shape

The "1 metric" per axis is the unweighted mean of each run's per-example mean (each run/task
gets equal weight regardless of its example count).

Usage:
  python scripts/aggregate_pix2pix_metrics.py outputs/*/results/*/test_latest/metrics.json
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def run_label(metrics_path: Path) -> str:
    # .../outputs/<run_dir>/results/<result_dir>/test_latest/metrics.json
    parts = metrics_path.parts
    return parts[parts.index("outputs") + 1] if "outputs" in parts else str(metrics_path.parent)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("metrics_json", nargs="+", type=Path, help="metrics.json files, one per run.")
    args = parser.parse_args()

    rows = []
    for path in args.metrics_json:
        summary = json.loads(path.read_text())["summary"]
        for key in ("lpips_base", "jf_af_avg_objects"):
            if key not in summary or summary[key]["mean"] is None:
                raise KeyError(
                    f"{path} is missing '{key}' -- rerun scripts/eval_pix2pix_metrics.py "
                    "(this field was added after that file was last generated)."
                )
        rows.append({
            "run": run_label(path),
            "lpips_base": summary["lpips_base"]["mean"],
            "lpips_model": summary["lpips"]["mean"],
            "jf_base": summary["jf_base_avg_objects"]["mean"],
            "jf_model": summary["jf_af_avg_objects"]["mean"],
        })

    rows.sort(key=lambda r: r["run"])

    name_w = max(len("run"), max(len(r["run"]) for r in rows))
    header = f"{'run':<{name_w}}  {'visual_base':>12}  {'visual_model':>13}  {'geom_base':>10}  {'geom_model':>11}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['run']:<{name_w}}  {r['lpips_base']:>12.4f}  {r['lpips_model']:>13.4f}  "
            f"{r['jf_base']:>10.4f}  {r['jf_model']:>11.4f}"
        )
    print("-" * len(header))

    overall = {
        "visual_alignment_baseline_lpips(real_A,real_B)": statistics.fmean(r["lpips_base"] for r in rows),
        "visual_alignment_model_lpips(fake_B,real_B)": statistics.fmean(r["lpips_model"] for r in rows),
        "geometry_alignment_baseline_jf(real_A,real_B)": statistics.fmean(r["jf_base"] for r in rows),
        "geometry_alignment_model_jf(real_A,fake_B)": statistics.fmean(r["jf_model"] for r in rows),
    }
    print(f"\nOverall ({len(rows)} runs, mean of per-run means; lower is better for LPIPS, higher is better for J&F):")
    for k, v in overall.items():
        print(f"  {k}: {v:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
