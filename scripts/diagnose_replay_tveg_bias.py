#!/usr/bin/env python3
"""Relate daily coupled-replay vegetation-temperature error to LAI and shortwave."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from netCDF4 import Dataset
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case", action="append", nargs=3, metavar=("LABEL", "EMULATOR_NC", "COMPARISON_CSV"), required=True
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def correlation(x: np.ndarray, y: np.ndarray) -> float:
    valid = np.isfinite(x) & np.isfinite(y)
    return float(np.corrcoef(x[valid], y[valid])[0, 1]) if np.count_nonzero(valid) > 1 else np.nan


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases = []
    summary_rows = []
    for label, emulator_path, comparison_path in args.case:
        with Dataset(emulator_path) as ds:
            lai = np.asarray(ds.variables["lai_mean"][:, 0], dtype=np.float64)
            fsds = np.asarray(ds.variables["fsds_mean"][:, 0], dtype=np.float64)
        comparison = list(csv.DictReader(Path(comparison_path).open()))
        tv_error = np.asarray(
            [float(row["TV_emulator"]) - float(row["TV_native"]) for row in comparison],
            dtype=np.float64,
        )
        tsoi_error = np.asarray(
            [float(row["TSOI_emulator"]) - float(row["TSOI_native"]) for row in comparison],
            dtype=np.float64,
        )
        if not (tv_error.size == lai.size == fsds.size):
            raise ValueError(f"Daily record mismatch for {label}")
        cases.append((label, lai, fsds, tsoi_error, tv_error))
        summary_rows.append({
            "site": label,
            "n": tv_error.size,
            "tv_bias_k": float(np.mean(tv_error)),
            "corr_tv_error_lai": correlation(tv_error, lai),
            "corr_tv_error_fsds": correlation(tv_error, fsds),
            "corr_tv_error_tsoi_error": correlation(tv_error, tsoi_error),
        })

    with (args.output_dir / "tveg_replay_correlations.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    colors = ("#1f77b4", "#d62728", "#2ca02c", "#9467bd")
    for (label, lai, fsds, tsoi_error, tv_error), color in zip(cases, colors):
        axes[0].scatter(lai, tv_error, s=8, alpha=0.22, color=color, label=label, rasterized=True)
        axes[1].scatter(fsds, tv_error, s=8, alpha=0.22, color=color, label=label, rasterized=True)
        axes[2].scatter(tsoi_error, tv_error, s=8, alpha=0.22, color=color, label=label, rasterized=True)
    for ax in axes:
        ax.axhline(0, color="black", linewidth=0.8)
        ax.grid(alpha=0.25)
    axes[0].set_xlabel("Daily mean LAI")
    axes[1].set_xlabel(r"Daily mean incoming shortwave (W m$^{-2}$)")
    axes[2].set_xlabel("Daily top-soil temperature error (K)")
    axes[0].set_ylabel("Daily vegetation-temperature error (K)")
    axes[0].set_title("Error versus LAI")
    axes[1].set_title("Error versus shortwave")
    axes[2].set_title("Error versus soil-temperature error")
    axes[0].legend(frameon=False)
    fig.suptitle("Coupled replay vegetation-temperature bias")
    fig.tight_layout()
    fig.savefig(args.output_dir / "tveg_replay_error_relationships.png", dpi=180)
    plt.close(fig)
    print(f"Wrote replay diagnostics for {len(cases)} sites to {args.output_dir}")


if __name__ == "__main__":
    main()
