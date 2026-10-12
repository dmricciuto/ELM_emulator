#!/usr/bin/env python3
"""Plot standardized bias and RMSE from a native/emulator daily comparison."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


VARIABLES = (
    ("FPSN", "Photosynthesis", "µmol CO₂ m⁻² s⁻¹"),
    ("EFLX_LH_TOT", "Latent heat", "W m⁻²"),
    ("FSH", "Sensible heat", "W m⁻²"),
    ("QVEGT", "Transpiration", "mm s⁻¹"),
    ("QSOIL", "Ground evaporation", "mm s⁻¹"),
    ("TV", "Vegetation temperature", "K"),
    ("TSOI", "Top-layer soil temperature", "K"),
    ("H2OSOI", "Top-layer soil moisture", "m³ m⁻³"),
    ("H2OSNO", "Snow water equivalent", "kg m⁻²"),
    ("H2OCAN", "Canopy water", "mm"),
    ("ZWT", "Water-table depth", "m"),
    ("QDRAI", "Subsurface drainage", "mm s⁻¹"),
    ("QCHARGE", "Aquifer recharge", "mm s⁻¹"),
    ("WA", "Aquifer water", "mm"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--daily-csv", type=Path, required=True)
    parser.add_argument("--stats-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default="ELM-emulator versus native ELM")
    return parser.parse_args()


def format_value(value: float) -> str:
    magnitude = abs(value)
    if magnitude != 0.0 and (magnitude < 1.0e-3 or magnitude >= 1.0e4):
        return f"{value:.2e}"
    return f"{value:.3g}"


def main() -> None:
    args = parse_args()
    with args.stats_csv.open(newline="", encoding="utf-8") as handle:
        stats = {row["variable"]: row for row in csv.DictReader(handle)}
    with args.daily_csv.open(newline="", encoding="utf-8") as handle:
        daily = list(csv.DictReader(handle))

    labels: list[str] = []
    normalized_bias: list[float] = []
    normalized_rmse: list[float] = []
    annotations: list[str] = []
    for name, label, units in VARIABLES:
        native = np.asarray([float(row[f"{name}_native"]) for row in daily])
        native_std = float(np.nanstd(native))
        bias = float(stats[name]["bias"])
        rmse = float(stats[name]["rmse"])
        scale = native_std if native_std > 1.0e-20 else 1.0
        labels.append(label)
        normalized_bias.append(bias / scale)
        normalized_rmse.append(rmse / scale)
        annotations.append(
            f"bias {format_value(bias)}, RMSE {format_value(rmse)} {units}"
        )

    y = np.arange(len(labels))
    fig, (ax_bias, ax_rmse) = plt.subplots(
        1,
        2,
        figsize=(16, 9),
        sharey=True,
        gridspec_kw={"width_ratios": (1.0, 1.25)},
    )
    bias_values = np.asarray(normalized_bias)
    bias_colors = np.where(bias_values >= 0.0, "#d95f02", "#1b9e77")
    ax_bias.barh(y, bias_values, color=bias_colors, alpha=0.88)
    ax_bias.axvline(0.0, color="black", linewidth=0.9)
    ax_bias.set_xlabel("Bias / native standard deviation")
    ax_bias.set_yticks(y, labels)
    ax_bias.grid(axis="x", alpha=0.25)
    ax_bias.invert_yaxis()

    rmse_values = np.asarray(normalized_rmse)
    ax_rmse.barh(y, rmse_values, color="#7570b3", alpha=0.88)
    ax_rmse.set_xlabel("RMSE / native standard deviation")
    ax_rmse.grid(axis="x", alpha=0.25)
    for idx, (value, annotation) in enumerate(zip(rmse_values, annotations)):
        ax_rmse.text(
            value + max(0.01, 0.015 * np.nanmax(rmse_values)),
            idx,
            annotation,
            va="center",
            fontsize=8.5,
        )
    ax_rmse.set_xlim(0.0, max(1.0, float(np.nanmax(rmse_values)) * 1.9))

    fig.suptitle(args.title, fontsize=15)
    fig.text(
        0.5,
        0.015,
        "Positive bias means emulator > native. Raw daily bias and RMSE are annotated at right.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0.0, 0.035, 1.0, 0.96))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200)
    plt.close(fig)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
