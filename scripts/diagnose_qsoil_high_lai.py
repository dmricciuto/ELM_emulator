#!/usr/bin/env python3
"""Evaluate universal-model soil evaporation across summer LAI bins."""

from __future__ import annotations

import argparse
import csv
import gc
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
from elm_emulator.train_patch_models import load_training_data, split_train_validation


LAI_EDGES = np.asarray([0.0, 1.0, 2.0, 3.0, 4.0, np.inf])
LAI_LABELS = ("0–1", "1–2", "2–3", "3–4", "≥4")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("training_file", type=Path)
    parser.add_argument("--surfdata-file", type=Path, required=True)
    parser.add_argument("--model-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pfts", type=int, nargs="+", default=(1, 7))
    parser.add_argument("--months", type=int, nargs="+", default=(6, 7, 8))
    parser.add_argument("--batch-size", type=int, default=65536)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument(
        "--holdout-gridcells",
        type=int,
        nargs="*",
        default=(1, 100, 200),
    )
    return parser.parse_args()


def summary(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    truth = y_true[valid]
    pred = y_pred[valid]
    if truth.size == 0:
        return {
            "n": 0,
            "true_mean": np.nan,
            "pred_mean": np.nan,
            "bias": np.nan,
            "relative_bias_pct": np.nan,
            "mae": np.nan,
            "rmse": np.nan,
            "correlation": np.nan,
            "true_dew_fraction": np.nan,
            "pred_dew_fraction": np.nan,
        }
    error = pred - truth
    true_mean = float(np.mean(truth))
    return {
        "n": int(truth.size),
        "true_mean": true_mean,
        "pred_mean": float(np.mean(pred)),
        "bias": float(np.mean(error)),
        "relative_bias_pct": float(100.0 * np.mean(error) / true_mean) if true_mean != 0.0 else np.nan,
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "correlation": (
            float(np.corrcoef(truth, pred)[0, 1])
            if truth.size > 1 and np.std(truth) > 0.0 and np.std(pred) > 0.0
            else np.nan
        ),
        "true_dew_fraction": float(np.mean(truth < -1.0e-12)),
        "pred_dew_fraction": float(np.mean(pred < -1.0e-12)),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    patch_index, _, features, targets, time_info, names = load_training_data(
        args.training_file,
        args.surfdata_file,
    )
    input_names = list(names["input_names"])
    target_names = list(names["target_names"])
    lai_idx = input_names.index("elai_patch")
    qsoil_idx = target_names.index("qflx_ev_soil")
    selected = (
        np.isin(patch_index, np.asarray(args.pfts))
        & np.isin(time_info["month"], np.asarray(args.months))
        & np.isfinite(features[:, lai_idx])
        & np.isfinite(targets[:, qsoil_idx])
        & (np.abs(targets[:, qsoil_idx]) < 1.0e30)
    )
    selected_indices = np.flatnonzero(selected)
    gridcell_index = (
        np.asarray(time_info["gridcell_index"], dtype=np.int64)
        if "gridcell_index" in time_info
        else np.full(patch_index.shape, -1, dtype=np.int64)
    )
    holdout_mask = np.isin(gridcell_index, np.asarray(args.holdout_gridcells))
    split_pool_indices = np.flatnonzero(~holdout_mask)
    _, validation_pool_indices = split_train_validation(
        split_pool_indices.size,
        args.val_fraction,
        args.random_seed,
    )
    validation_mask = np.zeros(patch_index.shape, dtype=bool)
    validation_mask[split_pool_indices[validation_pool_indices]] = True
    selected_validation = validation_mask[selected_indices]
    selected_holdout = holdout_mask[selected_indices]

    selected_features = np.asarray(features[selected], dtype=np.float64)
    selected_truth = np.asarray(targets[selected, qsoil_idx], dtype=np.float64)
    selected_pft = np.asarray(patch_index[selected], dtype=np.int64)
    selected_lai = np.asarray(features[selected, lai_idx], dtype=np.float64)
    del patch_index, features, targets, time_info
    gc.collect()

    device = torch.device("cpu")
    bundle = load_surface_model_bundle(args.model_file, device, torch.float32)
    missing = [name for name in bundle.input_names if name not in input_names]
    if missing:
        raise KeyError(f"Training features do not contain model inputs: {missing}")
    model_columns = np.asarray([input_names.index(name) for name in bundle.input_names], dtype=np.intp)
    selected_features = selected_features[:, model_columns]

    prediction_parts: list[np.ndarray] = []
    for start in range(0, selected_features.shape[0], args.batch_size):
        stop = min(start + args.batch_size, selected_features.shape[0])
        prediction = predict_surface_targets(
            bundle,
            selected_features[start:stop],
            device=device,
            dtype=torch.float32,
        )
        prediction_parts.append(np.asarray(prediction["qflx_ev_soil"], dtype=np.float64))
    selected_prediction = np.concatenate(prediction_parts)

    rows: list[dict[str, object]] = []
    subsets = (
        ("all_gridcells", np.ones(selected_truth.shape, dtype=bool)),
        ("random_validation", selected_validation),
        (
            "spatial_holdout_" + "_".join(str(value) for value in args.holdout_gridcells),
            selected_holdout,
        ),
    )
    for subset_name, subset_mask in subsets:
        for pft in args.pfts:
            for bin_idx, label in enumerate(LAI_LABELS):
                mask = (
                    subset_mask
                    & (selected_pft == pft)
                    & (selected_lai >= LAI_EDGES[bin_idx])
                    & (selected_lai < LAI_EDGES[bin_idx + 1])
                )
                rows.append(
                    {
                        "subset": subset_name,
                        "pft": int(pft),
                        "lai_bin": label,
                        "lai_min": float(LAI_EDGES[bin_idx]),
                        "lai_max": float(LAI_EDGES[bin_idx + 1]),
                        **summary(selected_truth[mask], selected_prediction[mask]),
                    }
                )

    csv_path = args.output_dir / "qsoil_jja_lai_bin_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    x = np.arange(len(LAI_LABELS))
    for pft, color in zip(args.pfts, ("#1f77b4", "#d62728")):
        pft_rows = [
            row for row in rows
            if row["subset"] == "random_validation" and row["pft"] == pft
        ]
        axes[0, 0].plot(x, [row["true_mean"] for row in pft_rows], "o--", color=color, label=f"PFT {pft} truth")
        axes[0, 0].plot(x, [row["pred_mean"] for row in pft_rows], "o-", color=color, label=f"PFT {pft} MLP")
        axes[0, 1].plot(x, [row["relative_bias_pct"] for row in pft_rows], "o-", color=color, label=f"PFT {pft}")
        axes[1, 0].plot(x, [row["rmse"] for row in pft_rows], "o-", color=color, label=f"PFT {pft}")
        axes[1, 1].plot(x, [row["correlation"] for row in pft_rows], "o-", color=color, label=f"PFT {pft}")

    axes[0, 0].set_title("Mean soil evaporation")
    axes[0, 0].set_ylabel(r"kg m$^{-2}$ s$^{-1}$")
    axes[0, 1].set_title("Mean bias relative to truth")
    axes[0, 1].set_ylabel("%")
    axes[0, 1].axhline(0.0, color="black", linewidth=0.8)
    axes[1, 0].set_title("RMSE")
    axes[1, 0].set_ylabel(r"kg m$^{-2}$ s$^{-1}$")
    axes[1, 1].set_title("Correlation")
    axes[1, 1].set_ylim(-0.05, 1.05)
    for ax in axes.flat:
        ax.set_xticks(x, LAI_LABELS)
        ax.set_xlabel("Exposed LAI")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Universal MLP qflx_ev_soil performance: JJA random validation rows")
    fig.tight_layout()
    fig.savefig(args.output_dir / "qsoil_jja_lai_bin_performance.png", dpi=180)
    plt.close(fig)

    high_lai = (selected_lai >= 3.0) & selected_validation
    rng = np.random.default_rng(42)
    high_indices = np.flatnonzero(high_lai)
    if high_indices.size > 30000:
        high_indices = rng.choice(high_indices, size=30000, replace=False)
    fig, ax = plt.subplots(figsize=(7, 6))
    for pft, color in zip(args.pfts, ("#1f77b4", "#d62728")):
        idx = high_indices[selected_pft[high_indices] == pft]
        ax.scatter(
            selected_truth[idx],
            selected_prediction[idx],
            s=6,
            alpha=0.18,
            color=color,
            label=f"PFT {pft}",
            rasterized=True,
        )
    values = np.concatenate((selected_truth[high_indices], selected_prediction[high_indices]))
    low, high = np.nanpercentile(values, [0.2, 99.8])
    ax.plot([low, high], [low, high], "k--", linewidth=1)
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_xlabel(r"ELM target (kg m$^{-2}$ s$^{-1}$)")
    ax.set_ylabel(r"MLP prediction (kg m$^{-2}$ s$^{-1}$)")
    ax.set_title("JJA random validation: soil evaporation at exposed LAI ≥ 3")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(args.output_dir / "qsoil_jja_high_lai_scatter.png", dpi=180)
    plt.close(fig)

    print(f"Evaluated {selected_truth.size:,} JJA rows; wrote {csv_path}")


if __name__ == "__main__":
    main()
