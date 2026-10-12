#!/usr/bin/env python3
"""Diagnose vegetation-temperature bias by PFT, LAI, and shortwave forcing."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from elm_emulator.coupled_surface_hydrology_soiltemp import (
    load_surface_model_bundle,
    predict_surface_targets,
)
from elm_emulator.train_patch_models import iter_patch_datasets, load_training_data


LAI_EDGES = np.asarray([0.0, 0.5, 1.0, 2.0, 3.0, 4.0, np.inf])
LAI_LABELS = ("0–0.5", "0.5–1", "1–2", "2–3", "3–4", "≥4")
SW_EDGES = np.asarray([0.0, 1.0, 100.0, 300.0, 600.0, np.inf])
SW_LABELS = ("night", "1–100", "100–300", "300–600", "≥600")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("training_file", type=Path)
    parser.add_argument("--surfdata-file", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--holdout-gridcells", type=int, nargs="+", default=(1, 100, 200))
    parser.add_argument("--batch-size", type=int, default=65536)
    return parser.parse_args()


def metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    valid = np.isfinite(truth) & np.isfinite(prediction)
    truth = truth[valid]
    prediction = prediction[valid]
    if truth.size == 0:
        return {"n": 0, "truth_mean": np.nan, "prediction_mean": np.nan,
                "bias": np.nan, "mae": np.nan, "rmse": np.nan, "correlation": np.nan}
    error = prediction - truth
    return {
        "n": int(truth.size),
        "truth_mean": float(np.mean(truth)),
        "prediction_mean": float(np.mean(prediction)),
        "bias": float(np.mean(error)),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "correlation": (
            float(np.corrcoef(truth, prediction)[0, 1])
            if truth.size > 1 and np.std(truth) > 0 and np.std(prediction) > 0
            else np.nan
        ),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    patch_index, cell_index, features, targets, time_info, names = load_training_data(
        args.training_file,
        args.surfdata_file,
    )
    gridcell_index = np.asarray(time_info["gridcell_index"], dtype=np.int64)
    keep = np.isin(gridcell_index, np.asarray(args.holdout_gridcells, dtype=np.int64))
    patch_index = patch_index[keep]
    cell_index = cell_index[keep] if cell_index is not None else None
    features = features[keep]
    targets = targets[keep]
    time_info = {
        key: value[keep] if isinstance(value, np.ndarray) and value.shape[:1] == keep.shape else value
        for key, value in time_info.items()
    }

    input_names = list(names["input_names"])
    target_names = list(names["target_names"])
    lai_idx = input_names.index("elai_patch")
    sw_idx = input_names.index("forc_swrad")
    tveg_idx = target_names.index("t_veg")
    device = torch.device("cpu")
    bundle_cache = {}
    records = []

    datasets = iter_patch_datasets(
        patch_index,
        features,
        targets,
        time_info,
        selected_patch_indices=None,
        spval_threshold=1.0e30,
        cell_index=cell_index,
        input_names=input_names,
        target_names=target_names,
        state_target_lag_steps=1,
        state_target_names=["t_veg", "t_ref2m"],
    )
    for dataset in datasets:
        pft = int(dataset.patch_index)
        model_patch = pft if pft in (0, 14) else -1
        if model_patch not in bundle_cache:
            model_file = args.model_dir / f"patch_{model_patch}" / "model.pt"
            bundle_cache[model_patch] = load_surface_model_bundle(
                model_file, device=device, dtype=torch.float32
            )
        bundle = bundle_cache[model_patch]
        columns = np.asarray([input_names.index(name) for name in bundle.input_names], dtype=np.intp)
        prediction_parts = []
        for start in range(0, dataset.features.shape[0], args.batch_size):
            stop = min(start + args.batch_size, dataset.features.shape[0])
            prediction = predict_surface_targets(
                bundle,
                np.asarray(dataset.features[start:stop, columns], dtype=np.float64),
                device=device,
                dtype=torch.float32,
            )
            prediction_parts.append(np.asarray(prediction["t_veg"], dtype=np.float64))
        prediction = np.concatenate(prediction_parts)
        truth = np.asarray(dataset.targets[:, tveg_idx], dtype=np.float64)
        lai = np.asarray(dataset.features[:, lai_idx], dtype=np.float64)
        shortwave = np.asarray(dataset.features[:, sw_idx], dtype=np.float64)
        records.append((pft, truth, prediction, lai, shortwave))

    rows: list[dict[str, object]] = []
    for pft, truth, prediction, lai, shortwave in records:
        rows.append({"pft": pft, "dimension": "all", "bin": "all", **metrics(truth, prediction)})
        for idx, label in enumerate(LAI_LABELS):
            selected = (lai >= LAI_EDGES[idx]) & (lai < LAI_EDGES[idx + 1])
            rows.append({"pft": pft, "dimension": "lai", "bin": label,
                         **metrics(truth[selected], prediction[selected])})
        for idx, label in enumerate(SW_LABELS):
            selected = (shortwave >= SW_EDGES[idx]) & (shortwave < SW_EDGES[idx + 1])
            rows.append({"pft": pft, "dimension": "shortwave", "bin": label,
                         **metrics(truth[selected], prediction[selected])})

    csv_path = args.output_dir / "tveg_holdout_bias_bins.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    pfts = [record[0] for record in records]
    colors = plt.cm.tab20(np.linspace(0, 1, max(len(pfts), 1)))
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    overall = [next(row for row in rows if row["pft"] == pft and row["dimension"] == "all") for pft in pfts]
    axes[0].bar([str(pft) for pft in pfts], [row["bias"] for row in overall], color=colors)
    axes[0].set_xlabel("PFT")
    axes[0].set_ylabel("TV bias (K)")
    axes[0].set_title("Spatial holdout mean bias")
    for pft, color in zip(pfts, colors):
        lai_rows = [row for row in rows if row["pft"] == pft and row["dimension"] == "lai"]
        axes[1].plot(LAI_LABELS, [row["bias"] for row in lai_rows], "o-", color=color, label=f"PFT {pft}")
        sw_rows = [row for row in rows if row["pft"] == pft and row["dimension"] == "shortwave"]
        axes[2].plot(SW_LABELS, [row["bias"] for row in sw_rows], "o-", color=color, label=f"PFT {pft}")
    axes[1].set_xlabel("Exposed LAI")
    axes[1].set_ylabel("TV bias (K)")
    axes[1].set_title("Bias by LAI")
    axes[2].set_xlabel(r"Incoming shortwave (W m$^{-2}$)")
    axes[2].set_ylabel("TV bias (K)")
    axes[2].set_title("Bias by shortwave")
    for ax in axes:
        ax.axhline(0, color="black", linewidth=0.8)
        ax.grid(alpha=0.25)
        ax.tick_params(axis="x", rotation=30)
    axes[2].legend(frameon=False, fontsize=8, ncol=2)
    gridcell_label = ", ".join(str(value) for value in args.holdout_gridcells)
    fig.suptitle(f"Vegetation-temperature error on selected grid cells {gridcell_label}")
    fig.tight_layout()
    fig.savefig(args.output_dir / "tveg_holdout_bias_by_pft_lai_shortwave.png", dpi=180)
    plt.close(fig)
    print(f"Evaluated {sum(row[1].size for row in records):,} held-out rows; wrote {csv_path}")


if __name__ == "__main__":
    main()
