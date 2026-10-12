#!/usr/bin/env python3
"""Compare daily US-MOz ELM-emulator output with native ELM history files."""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset, num2date


@dataclass(frozen=True)
class VariableSpec:
    native_name: str
    emulator_name: str
    label: str
    units: str
    native_level: int | None = None


VARIABLES = (
    VariableSpec("FPSN", "fpsn_mean", "Photosynthesis", r"$\mu$mol CO$_2$ m$^{-2}$ s$^{-1}$"),
    VariableSpec("EFLX_LH_TOT", "eflx_lh_tot_mean", "Latent heat", r"W m$^{-2}$"),
    VariableSpec("QVEGT", "qvegt_mean", "Transpiration", r"mm s$^{-1}$"),
    VariableSpec("QSOIL", "qsoil_mean", "Ground evaporation", r"mm s$^{-1}$"),
    VariableSpec("FSH", "eflx_sh_tot_mean", "Sensible heat", r"W m$^{-2}$"),
    VariableSpec("TV", "t_veg_mean", "Vegetation temperature", "K"),
    VariableSpec("TSOI", "t_soil_mean", "Top-layer soil temperature", "K", 0),
    VariableSpec("H2OSOI", "h2osoi_liqvol_lev1_mean", "Top-layer soil moisture", r"m$^3$ m$^{-3}$", 0),
    VariableSpec("H2OSNO", "h2osno_mean", "Snow water equivalent", r"kg m$^{-2}$"),
    VariableSpec("H2OCAN", "h2ocan_mean", "Canopy water", "mm"),
    VariableSpec("ZWT", "zwt_mean", "Water-table depth", "m"),
    VariableSpec("QDRAI", "qflx_drain_mean", "Subsurface drainage", r"mm s$^{-1}$"),
    VariableSpec("QCHARGE", "qflx_deep_drain_mean", "Aquifer recharge", r"mm s$^{-1}$"),
    VariableSpec("WA", "wa_mean", "Unconfined aquifer water", "mm"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emulator", type=Path, required=True)
    parser.add_argument("--native-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefix", default="usmoz_emulator_vs_native_2004_2014")
    parser.add_argument(
        "--site-label",
        default="US-MOz",
        help="Site or experiment label used in plot titles (default: US-MOz).",
    )
    parser.add_argument("--start-year", type=int, default=None)
    parser.add_argument("--end-year", type=int, default=None)
    parser.add_argument(
        "--native-record-offset",
        type=int,
        default=1,
        help=(
            "Native record index relative to the emulator record (default: 1). "
            "ELM daily history is interval-end stamped, so emulator[i] is "
            "compared with native[i+1]."
        ),
    )
    parser.add_argument(
        "--timeseries-frequency",
        choices=("daily", "monthly"),
        default="monthly",
    )
    return parser.parse_args()


def clean(values: np.ndarray, fill_value: float | None = None) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if fill_value is not None:
        result = np.where(result == float(fill_value), np.nan, result)
    return np.where(np.isfinite(result) & (np.abs(result) < 1.0e30), result, np.nan)


def load_native(native_dir: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    paths = sorted(native_dir.glob("*.elm.h0.*.nc"))
    if not paths:
        raise FileNotFoundError(f"No native ELM h0 files found in {native_dir}")

    all_dates: list[date] = []
    chunks: dict[str, list[np.ndarray]] = {spec.native_name: [] for spec in VARIABLES}
    for path in paths:
        with Dataset(path) as ds:
            time_var = ds.variables["time"]
            decoded = num2date(
                time_var[:],
                units=time_var.units,
                calendar=getattr(time_var, "calendar", "standard"),
            )
            all_dates.extend(date(int(item.year), int(item.month), int(item.day)) for item in decoded)
            for spec in VARIABLES:
                var = ds.variables[spec.native_name]
                if spec.native_level is None:
                    values = var[:, 0]
                else:
                    values = var[:, spec.native_level, 0]
                chunks[spec.native_name].append(
                    clean(values, getattr(var, "_FillValue", None))
                )
    return np.asarray(all_dates, dtype=object), {
        name: np.concatenate(parts) for name, parts in chunks.items()
    }


def load_emulator(path: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    with Dataset(path) as ds:
        dates = np.asarray(
            [
                date(int(year), int(month), int(day))
                for year, month, day in zip(
                    ds.variables["year"][:],
                    ds.variables["month"][:],
                    ds.variables["day"][:],
                )
            ],
            dtype=object,
        )
        values = {
            spec.emulator_name: clean(
                ds.variables[spec.emulator_name][:, 0],
                getattr(ds.variables[spec.emulator_name], "_FillValue", None),
            )
            for spec in VARIABLES
        }
    return dates, values


def align(
    native_dates: np.ndarray,
    native: dict[str, np.ndarray],
    emulator_dates: np.ndarray,
    emulator: dict[str, np.ndarray],
    native_record_offset: int,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    native_lookup = {value: idx for idx, value in enumerate(native_dates)}
    emulator_lookup = {value: idx for idx, value in enumerate(emulator_dates)}
    common = np.asarray(sorted(set(native_lookup) & set(emulator_lookup)), dtype=object)
    native_idx = np.asarray(
        [native_lookup[value] + native_record_offset for value in common],
        dtype=np.intp,
    )
    emulator_idx = np.asarray([emulator_lookup[value] for value in common], dtype=np.intp)
    valid = (native_idx >= 0) & (native_idx < native_dates.size)
    common = common[valid]
    native_idx = native_idx[valid]
    emulator_idx = emulator_idx[valid]
    return (
        common,
        {name: values[native_idx] for name, values in native.items()},
        {name: values[emulator_idx] for name, values in emulator.items()},
    )


def metrics(native: np.ndarray, emulator: np.ndarray) -> dict[str, float | int]:
    valid = np.isfinite(native) & np.isfinite(emulator)
    if not np.any(valid):
        return {"n": 0, "bias": np.nan, "mae": np.nan, "rmse": np.nan, "correlation": np.nan}
    obs = native[valid]
    pred = emulator[valid]
    error = pred - obs
    correlation = (
        float(np.corrcoef(obs, pred)[0, 1])
        if obs.size > 1 and np.std(obs) > 0.0 and np.std(pred) > 0.0
        else np.nan
    )
    return {
        "n": int(obs.size),
        "bias": float(np.mean(error)),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "correlation": correlation,
    }


def monthly_means(dates: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    keys = [(item.year, item.month) for item in dates]
    unique = sorted(set(keys))
    output_dates: list[date] = []
    output_values: list[float] = []
    for year, month in unique:
        selected = np.asarray([key == (year, month) for key in keys], dtype=bool)
        output_dates.append(date(year, month, 15))
        finite = values[selected][np.isfinite(values[selected])]
        output_values.append(float(np.mean(finite)) if finite.size else np.nan)
    return np.asarray(output_dates, dtype=object), np.asarray(output_values)


def write_outputs(
    output_dir: Path,
    prefix: str,
    site_label: str,
    dates: np.ndarray,
    native: dict[str, np.ndarray],
    emulator: dict[str, np.ndarray],
    timeseries_frequency: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    ncols = 3
    nrows = math.ceil(len(VARIABLES) / ncols)

    with (output_dir / f"{prefix}_daily.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["date"]
        for spec in VARIABLES:
            fieldnames.extend([f"{spec.native_name}_native", f"{spec.native_name}_emulator"])
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for idx, item in enumerate(dates):
            row: dict[str, object] = {"date": item.isoformat()}
            for spec in VARIABLES:
                row[f"{spec.native_name}_native"] = native[spec.native_name][idx]
                row[f"{spec.native_name}_emulator"] = emulator[spec.emulator_name][idx]
            writer.writerow(row)

    with (output_dir / f"{prefix}_stats.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["variable", "label", "units", "n", "bias", "mae", "rmse", "correlation"],
        )
        writer.writeheader()
        for spec in VARIABLES:
            writer.writerow(
                {
                    "variable": spec.native_name,
                    "label": spec.label,
                    "units": spec.units,
                    **metrics(native[spec.native_name], emulator[spec.emulator_name]),
                }
            )

    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4 * nrows), sharex=True)
    for ax, spec in zip(axes.flat, VARIABLES):
        if timeseries_frequency == "daily":
            plot_dates = dates
            native_plot = native[spec.native_name]
            emulator_plot = emulator[spec.emulator_name]
            linewidth = 0.8
        else:
            plot_dates, native_plot = monthly_means(dates, native[spec.native_name])
            _, emulator_plot = monthly_means(dates, emulator[spec.emulator_name])
            linewidth = 1.5
        ax.plot(plot_dates, native_plot, color="#1f77b4", linewidth=linewidth, label="Native ELM")
        ax.plot(plot_dates, emulator_plot, color="#d62728", linewidth=linewidth, label="ELM-emulator")
        ax.set_title(spec.label)
        ax.set_ylabel(spec.units)
        ax.grid(alpha=0.25)
    for ax in axes.flat[len(VARIABLES):]:
        ax.set_visible(False)
    axes.flat[0].legend(frameon=False)
    fig.suptitle(
        f"{site_label}: {timeseries_frequency} ELM-emulator versus native ELM",
        fontsize=15,
    )
    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}_{timeseries_frequency}_timeseries.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 4.3 * nrows))
    for ax, spec in zip(axes.flat, VARIABLES):
        obs = native[spec.native_name]
        pred = emulator[spec.emulator_name]
        valid = np.isfinite(obs) & np.isfinite(pred)
        ax.scatter(obs[valid], pred[valid], s=5, alpha=0.18, color="#6a3d9a", rasterized=True)
        if np.any(valid):
            low = float(min(np.min(obs[valid]), np.min(pred[valid])))
            high = float(max(np.max(obs[valid]), np.max(pred[valid])))
            ax.plot([low, high], [low, high], "k--", linewidth=1)
        summary = metrics(obs, pred)
        ax.set_title(f"{spec.label}\nRMSE={summary['rmse']:.3g}, r={summary['correlation']:.3f}")
        ax.set_xlabel(f"Native ({spec.units})")
        ax.set_ylabel(f"Emulator ({spec.units})")
        ax.grid(alpha=0.2)
    for ax in axes.flat[len(VARIABLES):]:
        ax.set_visible(False)
    fig.suptitle(f"{site_label}: daily ELM-emulator versus native ELM", fontsize=15)
    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}_daily_scatter.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.7 * nrows))
    month = np.asarray([item.month for item in dates])
    for ax, spec in zip(axes.flat, VARIABLES):
        obs = native[spec.native_name]
        pred = emulator[spec.emulator_name]
        obs_monthly = [
            np.nanmean(obs[month == value]) if np.any(np.isfinite(obs[month == value])) else np.nan
            for value in range(1, 13)
        ]
        pred_monthly = [
            np.nanmean(pred[month == value]) if np.any(np.isfinite(pred[month == value])) else np.nan
            for value in range(1, 13)
        ]
        ax.plot(range(1, 13), obs_monthly, "o-", color="#1f77b4", label="Native ELM")
        ax.plot(range(1, 13), pred_monthly, "o-", color="#d62728", label="ELM-emulator")
        ax.set_title(spec.label)
        ax.set_ylabel(spec.units)
        ax.set_xticks(range(1, 13))
        ax.grid(alpha=0.25)
    for ax in axes.flat[len(VARIABLES):]:
        ax.set_visible(False)
    axes.flat[0].legend(frameon=False)
    fig.suptitle(
        f"{site_label}: {dates[0].year}–{dates[-1].year} monthly climatology",
        fontsize=15,
    )
    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}_monthly_climatology.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    native_dates, native = load_native(args.native_dir)
    emulator_dates, emulator = load_emulator(args.emulator)
    dates, native, emulator = align(
        native_dates,
        native,
        emulator_dates,
        emulator,
        args.native_record_offset,
    )
    selected = np.ones(dates.size, dtype=bool)
    if args.start_year is not None:
        selected &= np.asarray([item.year >= args.start_year for item in dates])
    if args.end_year is not None:
        selected &= np.asarray([item.year <= args.end_year for item in dates])
    dates = dates[selected]
    native = {name: values[selected] for name, values in native.items()}
    emulator = {name: values[selected] for name, values in emulator.items()}
    if dates.size == 0:
        raise RuntimeError("Native and emulator outputs have no dates in common")
    write_outputs(
        args.output_dir,
        args.prefix,
        args.site_label,
        dates,
        native,
        emulator,
        args.timeseries_frequency,
    )
    print(
        f"Wrote comparison for {dates.size} daily records to {args.output_dir} "
        f"(native record offset={args.native_record_offset})"
    )


if __name__ == "__main__":
    main()
