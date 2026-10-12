#!/usr/bin/env python3
"""Compare exact US-MOz replay MLP inputs with matched training rows."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import netCDF4
import numpy as np
import torch
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from elm_emulator.btran_utils import compute_btran_and_root_fraction
from elm_emulator.train_patch_models import _load_surfdata_watsat_profiles


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("training_file", type=Path)
    parser.add_argument("--replay-features", type=Path, required=True)
    parser.add_argument("--model-file", type=Path, required=True)
    parser.add_argument(
        "--training-surfdata",
        type=Path,
        default=Path("data/surface_data/surfdata_sg256.nc"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pfts", type=int, nargs="+", default=(1, 7))
    parser.add_argument("--months", type=int, nargs="+", default=(6, 7, 8))
    parser.add_argument("--lai-min", type=float, default=3.0)
    parser.add_argument("--daylight-swrad-min", type=float, default=20.0)
    parser.add_argument("--chunk-size", type=int, default=250_000)
    parser.add_argument("--max-training-rows-per-pft", type=int, default=250_000)
    parser.add_argument("--random-seed", type=int, default=42)
    return parser.parse_args()


def safe_name(name: str) -> str:
    return "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in name)


def forcing_vpd(forc_t: np.ndarray, forc_q: np.ndarray, forc_pbot: np.ndarray) -> np.ndarray:
    """Match the replay/training VPD derivation."""
    tc = np.asarray(forc_t, dtype=np.float64) - 273.15
    exponent = np.clip((17.2694 * tc) / (tc + 237.3), -80.0, 80.0)
    saturation = 610.78 * np.exp(exponent)
    q = np.maximum(np.asarray(forc_q, dtype=np.float64), 0.0)
    pressure = np.maximum(np.asarray(forc_pbot, dtype=np.float64), 1.0)
    actual = q * pressure / np.maximum(0.622 + (1.0 - 0.622) * q, 1.0e-12)
    return np.maximum(saturation - actual, 0.0)


def reservoir_update(
    reservoir: np.ndarray | None,
    rows: np.ndarray,
    *,
    maximum: int,
    seen: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, int]:
    """Update a fixed-size uniform reservoir with a batch of rows."""
    if rows.size == 0:
        return reservoir, seen
    if reservoir is None:
        reservoir = np.empty((maximum, rows.shape[1]), dtype=np.float64)
    for row in rows:
        seen += 1
        if seen <= maximum:
            reservoir[seen - 1] = row
        else:
            replacement = int(rng.integers(0, seen))
            if replacement < maximum:
                reservoir[replacement] = row
    return reservoir, seen


def mahalanobis_summary(training: np.ndarray, site: np.ndarray) -> dict[str, float | int]:
    """Return a regularized multivariate distance comparison."""
    finite = np.all(np.isfinite(training), axis=0) & np.all(np.isfinite(site), axis=0)
    training = training[:, finite]
    site = site[:, finite]
    center = np.mean(training, axis=0)
    scale = np.std(training, axis=0)
    varying = scale > np.maximum(1.0e-12, 1.0e-8 * np.abs(center))
    training = (training[:, varying] - center[varying]) / scale[varying]
    site = (site[:, varying] - center[varying]) / scale[varying]
    covariance = np.cov(training, rowvar=False)
    covariance = np.atleast_2d(covariance)
    shrinkage = 0.05
    covariance = (1.0 - shrinkage) * covariance + shrinkage * np.eye(covariance.shape[0])
    precision = np.linalg.pinv(covariance, hermitian=True)
    train_distance = np.sqrt(np.einsum("ij,jk,ik->i", training, precision, training))
    site_distance = np.sqrt(np.einsum("ij,jk,ik->i", site, precision, site))
    reference = np.sort(train_distance)
    site_percentile = np.searchsorted(reference, site_distance, side="right") / reference.size
    return {
        "varying_feature_count": int(varying.sum()),
        "training_distance_median": float(np.median(train_distance)),
        "training_distance_p95": float(np.quantile(train_distance, 0.95)),
        "training_distance_p99": float(np.quantile(train_distance, 0.99)),
        "usmoz_distance_median": float(np.median(site_distance)),
        "usmoz_distance_p95": float(np.quantile(site_distance, 0.95)),
        "usmoz_training_percentile_median": float(np.median(site_percentile)),
        "usmoz_fraction_above_training_p95": float(
            np.mean(site_distance > np.quantile(train_distance, 0.95))
        ),
        "usmoz_fraction_above_training_p99": float(
            np.mean(site_distance > np.quantile(train_distance, 0.99))
        ),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.random_seed)

    checkpoint = torch.load(args.model_file, map_location="cpu", weights_only=False)
    input_names = list(checkpoint["input_names"])
    model_mean = np.asarray(checkpoint["x_mean"], dtype=np.float64)
    model_std = np.asarray(checkpoint["x_std"], dtype=np.float64)

    replay = xr.open_dataset(args.replay_features, decode_times=False)
    replay_month = np.asarray(replay["month"].values, dtype=np.int64).reshape(-1)
    site_by_pft: dict[int, np.ndarray] = {}
    for pft in args.pfts:
        columns = []
        for name in input_names:
            variable = f"pft_{pft}_feature_{safe_name(name)}"
            if variable not in replay:
                raise KeyError(f"Replay feature file lacks {variable}")
            columns.append(
                np.asarray(replay[variable].values, dtype=np.float64).reshape(
                    replay_month.size,
                    -1,
                )[:, 0]
            )
        values = np.column_stack(columns)
        lai = values[:, input_names.index("elai_patch")]
        swrad = values[:, input_names.index("forc_swrad")]
        selected = (
            np.isin(replay_month, np.asarray(args.months))
            & (lai >= args.lai_min)
            & (swrad > args.daylight_swrad_min)
            & np.all(np.isfinite(values), axis=1)
        )
        site_by_pft[pft] = values[selected]
    replay.close()

    reservoirs: dict[int, np.ndarray | None] = {pft: None for pft in args.pfts}
    seen = {pft: 0 for pft in args.pfts}
    watsat_by_cell = _load_surfdata_watsat_profiles(args.training_surfdata, 10)
    with netCDF4.Dataset(args.training_file) as training:
        training_names = training.getncattr("feature_names").split(",")
        raw_input_names = [
            name for name in input_names if name not in {"forc_vpd", "btran"}
        ]
        btran_names = [
            *(f"h2osoi_liqvol_lev{level}" for level in range(1, 11)),
            *(f"t_soisno_lev{level}" for level in range(1, 11)),
            *(f"rootfr_lev{level}" for level in range(1, 11)),
            *(f"eff_porosity_lev{level}" for level in range(1, 11)),
            *(f"smp_l_lev{level}" for level in range(1, 11)),
            "frac_h2osfc",
            "smpso",
            "smpsc",
        ]
        requested_read_names = list(dict.fromkeys([*raw_input_names, *btran_names]))
        training_columns = np.asarray(
            sorted(training_names.index(name) for name in requested_read_names),
            dtype=np.int64,
        )
        read_names = [training_names[index] for index in training_columns]
        raw_position = {name: index for index, name in enumerate(read_names)}
        lai_column = training_names.index("elai_patch")
        swrad_column = training_names.index("forc_swrad")
        nsample = training.dimensions["sample"].size
        for start in range(0, nsample, args.chunk_size):
            stop = min(start + args.chunk_size, nsample)
            patch = np.asarray(training["patch_index"][start:stop])
            month = np.asarray(training["month"][start:stop])
            lai = np.asarray(training["features"][start:stop, lai_column])
            swrad = np.asarray(training["features"][start:stop, swrad_column])
            common = (
                np.isin(month, np.asarray(args.months))
                & (lai >= args.lai_min)
                & (swrad > args.daylight_swrad_min)
            )
            if not np.any(common):
                continue
            raw_features = np.asarray(
                training["features"][start:stop, training_columns],
                dtype=np.float64,
            )
            gridcell = np.asarray(training["gridcell_index"][start:stop], dtype=np.int64)
            if np.min(gridcell) >= 1 and np.max(gridcell) <= watsat_by_cell.shape[0]:
                mapped_gridcell = gridcell - 1
            elif np.min(gridcell) >= 0 and np.max(gridcell) < watsat_by_cell.shape[0]:
                mapped_gridcell = gridcell
            else:
                raise ValueError(
                    "Training gridcell_index cannot be mapped directly to training surfdata"
                )
            liqvol = np.column_stack(
                [
                    raw_features[:, raw_position[f"h2osoi_liqvol_lev{level}"]]
                    for level in range(1, 11)
                ]
            )
            tempk = np.column_stack(
                [
                    raw_features[:, raw_position[f"t_soisno_lev{level}"]]
                    for level in range(1, 11)
                ]
            )
            rootfr = np.column_stack(
                [
                    raw_features[:, raw_position[f"rootfr_lev{level}"]]
                    for level in range(1, 11)
                ]
            )
            eff_porosity = np.column_stack(
                [
                    raw_features[:, raw_position[f"eff_porosity_lev{level}"]]
                    for level in range(1, 11)
                ]
            )
            smp_l = np.column_stack(
                [
                    raw_features[:, raw_position[f"smp_l_lev{level}"]]
                    for level in range(1, 11)
                ]
            )
            derived_btran, _ = compute_btran_and_root_fraction(
                liqvol=liqvol,
                tempk=tempk,
                frac_h2osfc=raw_features[:, raw_position["frac_h2osfc"]],
                rootfr=rootfr,
                smpso=raw_features[:, raw_position["smpso"]],
                smpsc=raw_features[:, raw_position["smpsc"]],
                tc_stress_c=-2.0,
                eff_porosity=eff_porosity,
                smp_l=smp_l,
                watsat=watsat_by_cell[mapped_gridcell],
            )
            feature_parts = []
            for name in input_names:
                if name == "forc_vpd":
                    feature_parts.append(
                        forcing_vpd(
                            raw_features[:, raw_position["forc_t"]],
                            raw_features[:, raw_position["forc_q"]],
                            raw_features[:, raw_position["forc_pbot"]],
                        )
                    )
                elif name == "btran":
                    feature_parts.append(derived_btran)
                else:
                    feature_parts.append(raw_features[:, raw_position[name]])
            features = np.column_stack(feature_parts)
            for pft in args.pfts:
                rows = features[common & (patch == pft)]
                rows = rows[np.all(np.isfinite(rows), axis=1)]
                reservoirs[pft], seen[pft] = reservoir_update(
                    reservoirs[pft],
                    rows,
                    maximum=args.max_training_rows_per_pft,
                    seen=seen[pft],
                    rng=rng,
                )

    rows_out: list[dict[str, float | int | str]] = []
    multivariate: dict[str, dict[str, float | int]] = {}
    for pft in args.pfts:
        if reservoirs[pft] is None or seen[pft] == 0:
            raise RuntimeError(f"No matched training rows found for PFT {pft}")
        n_kept = min(seen[pft], args.max_training_rows_per_pft)
        training_values = reservoirs[pft][:n_kept]
        site_values = site_by_pft[pft]
        multivariate[str(pft)] = {
            "training_rows_matched": int(seen[pft]),
            "training_rows_sampled": int(n_kept),
            "usmoz_rows": int(site_values.shape[0]),
            **mahalanobis_summary(training_values, site_values),
        }
        for index, name in enumerate(input_names):
            train = training_values[:, index]
            site = site_values[:, index]
            tq = np.quantile(train, [0.005, 0.05, 0.5, 0.95, 0.995])
            sq = np.quantile(site, [0.05, 0.5, 0.95])
            std = model_std[index]
            range_tolerance = max(
                1.0e-10,
                1.0e-6
                * max(
                    abs(float(tq[0])),
                    abs(float(tq[-1])),
                    abs(float(sq[1])),
                    abs(float(std)),
                ),
            )
            rows_out.append(
                {
                    "pft": pft,
                    "feature": name,
                    "training_n_matched": seen[pft],
                    "training_n_sampled": n_kept,
                    "usmoz_n": site.size,
                    "training_p005": tq[0],
                    "training_p05": tq[1],
                    "training_median": tq[2],
                    "training_p95": tq[3],
                    "training_p995": tq[4],
                    "usmoz_p05": sq[0],
                    "usmoz_median": sq[1],
                    "usmoz_p95": sq[2],
                    "usmoz_median_model_z": (
                        (sq[1] - model_mean[index]) / std if std > 0.0 else np.nan
                    ),
                    "median_shift_model_std": (
                        (sq[1] - tq[2]) / std if std > 0.0 else np.nan
                    ),
                    "usmoz_fraction_below_training_p005": float(
                        np.mean(site < tq[0] - range_tolerance)
                    ),
                    "usmoz_fraction_above_training_p995": float(
                        np.mean(site > tq[4] + range_tolerance)
                    ),
                }
            )

    csv_path = args.output_dir / "usmoz_mlp_input_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows_out[0]))
        writer.writeheader()
        writer.writerows(rows_out)

    summary_path = args.output_dir / "usmoz_mlp_input_multivariate_summary.json"
    summary_path.write_text(json.dumps(multivariate, indent=2) + "\n", encoding="utf-8")

    for pft in args.pfts:
        pft_rows = [row for row in rows_out if row["pft"] == pft]
        pft_rows.sort(
            key=lambda row: (
                abs(float(row["median_shift_model_std"]))
                + 3.0
                * (
                    float(row["usmoz_fraction_below_training_p005"])
                    + float(row["usmoz_fraction_above_training_p995"])
                )
            ),
            reverse=True,
        )
        top = pft_rows[:15][::-1]
        labels = [str(row["feature"]) for row in top]
        shifts = [float(row["median_shift_model_std"]) for row in top]
        outside = [
            100.0
            * (
                float(row["usmoz_fraction_below_training_p005"])
                + float(row["usmoz_fraction_above_training_p995"])
            )
            for row in top
        ]
        fig, axes = plt.subplots(1, 2, figsize=(13, 7), sharey=True)
        axes[0].barh(labels, shifts, color="#4477aa")
        axes[0].axvline(0.0, color="black", linewidth=0.8)
        axes[0].set_xlabel("US-MOz median shift (model training SD)")
        axes[1].barh(labels, outside, color="#cc6677")
        axes[1].set_xlabel("US-MOz outside training 0.5–99.5% range (%)")
        fig.suptitle(
            f"PFT {pft}: daylight JJA MLP inputs at LAI ≥ {args.lai_min:g}"
        )
        fig.tight_layout()
        fig.savefig(args.output_dir / f"usmoz_mlp_input_shift_pft_{pft}.png", dpi=180)
        plt.close(fig)

    print(f"Wrote {csv_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
