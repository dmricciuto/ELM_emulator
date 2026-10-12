#!/usr/bin/env python3
"""Plot Central PA native, no-aging, and SNICAR-aging spring snow behavior."""

from __future__ import annotations

import csv
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset, num2date


ROOT = Path(__file__).resolve().parents[1]
OLD_CSV = ROOT / "output/central_pa_trendy_2004_2005_joint_surface_thermal/comparison/central_pa_joint_surface_thermal_vs_native_daily.csv"
NEW_DAILY = ROOT / "output/central_pa_trendy_2004_2005_snicar_aging/central_pa_trendy_emulator_2004_2005_daily.nc"
NEW_HOURLY = ROOT / "output/central_pa_trendy_2004_2005_snicar_aging/central_pa_trendy_emulator_2004_2005_hourly_diagnostics.nc"
NATIVE = ROOT.parent / "elm-olmt/setup_runs/20260804_CentralPA_trendy_global_ICBELMBC_native_daily/run/20260804_CentralPA_trendy_global_ICBELMBC_native_daily.elm.h0.2005-01-01-00000.nc"
OUT_DIR = ROOT / "output/central_pa_trendy_2004_2005_snicar_aging/comparison"


def _dates_from_ymd(year, month, day):
    return np.asarray(
        [date(int(y), int(m), int(d)) for y, m, d in zip(year, month, day)],
        dtype=object,
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with Dataset(NATIVE) as dataset:
        time = dataset["time"]
        native_dates = np.asarray(
            [date(int(d.year), int(d.month), int(d.day)) for d in num2date(time[:], time.units, getattr(time, "calendar", "noleap"))],
            dtype=object,
        )
        native_swe = np.asarray(dataset["H2OSNO"][:, 0], dtype=np.float64)
        native_tsoil = np.asarray(dataset["TSOI"][:, 0, 0], dtype=np.float64)
        native_fsh = np.asarray(dataset["FSH"][:, 0], dtype=np.float64)

    with Dataset(NEW_DAILY) as dataset:
        new_dates = _dates_from_ymd(dataset["year"][:], dataset["month"][:], dataset["day"][:])
        new_swe = np.asarray(dataset["h2osno_mean"][:, 0], dtype=np.float64)
        new_tsoil = np.asarray(dataset["t_soisno_lev1_mean"][:, 0], dtype=np.float64)
        new_fsh = np.asarray(dataset["eflx_sh_tot_mean"][:, 0], dtype=np.float64)

    old_rows = list(csv.DictReader(OLD_CSV.open()))
    old_dates = np.asarray([date.fromisoformat(row["date"]) for row in old_rows], dtype=object)
    old_swe = np.asarray([float(row["H2OSNO_emulator"]) for row in old_rows])
    old_tsoil = np.asarray([float(row["TSOI_emulator"]) for row in old_rows])
    old_fsh = np.asarray([float(row["FSH_emulator"]) for row in old_rows])

    with Dataset(NEW_HOURLY) as dataset:
        hourly_dates = _dates_from_ymd(dataset["year"][:], dataset["month"][:], dataset["day"][:])
        daily_dates = np.asarray(sorted(set(hourly_dates)), dtype=object)

        def daily_mean(name):
            values = np.asarray(dataset[name][:, 0], dtype=np.float64)
            output = []
            for daily_date in daily_dates:
                selected = values[hourly_dates == daily_date]
                finite = selected[np.isfinite(selected)]
                output.append(np.mean(finite) if finite.size else np.nan)
            return np.asarray(output)

        grain_radius = daily_mean("snow_grain_radius_top")
        albedo_vis = daily_mean("snow_albedo_vis")
        albedo_nir = daily_mean("snow_albedo_nir")
        frac_sno = daily_mean("frac_sno")
        frac_sno_input = daily_mean("frac_sno_input")
        sabg_snow = daily_mean("sabg_snow_input")
        snow_ice = daily_mean("snow_ice_total")
        snow_liq = daily_mean("snow_liq_total")
        surface_water = daily_mean("h2osfc")

    start, end = date(2005, 3, 15), date(2005, 4, 30)
    fig, axes = plt.subplots(6, 1, figsize=(12, 17), sharex=True)

    for ax, native_y, old_y, new_y, ylabel in (
        (axes[0], native_swe, old_swe, new_swe, "SWE (kg m$^{-2}$)"),
        (axes[1], native_tsoil, old_tsoil, new_tsoil, "Top soil T (K)"),
        (axes[2], native_fsh, old_fsh, new_fsh, "Sensible heat (W m$^{-2}$)"),
    ):
        ax.plot(native_dates, native_y, color="black", lw=2.2, label="Native ELM")
        ax.plot(old_dates, old_y, color="0.55", lw=1.8, label="Emulator, no aging")
        ax.plot(new_dates, new_y, color="#d62728", lw=2.0, label="Emulator, SNICAR aging")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)

    axes[0].legend(ncol=3, fontsize=9)
    axes[3].plot(daily_dates, snow_ice, label="Snow ice")
    axes[3].plot(daily_dates, snow_liq, label="Snow liquid")
    axes[3].plot(daily_dates, surface_water, label="Standing water")
    axes[3].set_ylabel("Water storage (kg m$^{-2}$)")
    axes[3].grid(alpha=0.25)
    axes[3].legend(ncol=3, fontsize=9)

    axes[4].plot(daily_dates, grain_radius, color="#8c564b", label="Top grain radius")
    axes[4].set_ylabel("Grain radius (µm)")
    axes[4].grid(alpha=0.25)
    albedo_axis = axes[4].twinx()
    albedo_axis.plot(daily_dates, albedo_vis, color="#1f77b4", ls="--", label="VIS albedo")
    albedo_axis.plot(daily_dates, albedo_nir, color="#17becf", ls="--", label="NIR albedo")
    albedo_axis.set_ylabel("Snow albedo")
    lines = axes[4].lines + albedo_axis.lines
    axes[4].legend(lines, [line.get_label() for line in lines], ncol=3, fontsize=9)

    axes[5].plot(daily_dates, frac_sno_input, color="#9467bd", ls="--", label="FSNO feature input")
    axes[5].plot(daily_dates, frac_sno, color="#2ca02c", label="FSNO post-step effective")
    axes[5].set_ylabel("Snow fraction")
    axes[5].set_ylim(-0.03, 1.03)
    axes[5].grid(alpha=0.25)
    radiation_axis = axes[5].twinx()
    radiation_axis.plot(daily_dates, sabg_snow, color="#ff7f0e", alpha=0.8, label="SABG snow input")
    radiation_axis.set_ylabel("Absorbed SW (W m$^{-2}$)")
    lines = axes[5].lines + radiation_axis.lines
    axes[5].legend(lines, [line.get_label() for line in lines], ncol=3, fontsize=9)

    for ax in axes:
        ax.set_xlim(start, end)
    axes[-1].set_xlabel("Date")
    fig.suptitle("Central PA spring melt: effect of activating SNICAR grain aging")
    fig.tight_layout()
    figure_path = OUT_DIR / "central_pa_spring_2005_snicar_aging_comparison.png"
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)

    def meltout(dates, swe):
        mask = (dates >= date(2005, 3, 20)) & (swe < 1.0)
        return next(iter(dates[mask]), None)

    new_by_date = {d: i for i, d in enumerate(new_dates)}
    old_by_date = {d: i for i, d in enumerate(old_dates)}
    native_by_date = {d: i for i, d in enumerate(native_dates)}

    def comparison_stats(values_native, values_old, values_new, window_start, window_end):
        dates = sorted(
            d for d in new_by_date
            if window_start <= d <= window_end and d in old_by_date and d in native_by_date
        )
        native_values = np.asarray([values_native[native_by_date[d]] for d in dates])
        old_values = np.asarray([values_old[old_by_date[d]] for d in dates])
        new_values = np.asarray([values_new[new_by_date[d]] for d in dates])
        return (
            float(np.sqrt(np.mean((old_values - native_values) ** 2))),
            float(np.sqrt(np.mean((new_values - native_values) ** 2))),
        )

    melt_window = (date(2005, 3, 20), date(2005, 4, 10))
    swe_rmse = comparison_stats(native_swe, old_swe, new_swe, *melt_window)
    tsoil_rmse = comparison_stats(native_tsoil, old_tsoil, new_tsoil, *melt_window)
    fsh_rmse = comparison_stats(native_fsh, old_fsh, new_fsh, *melt_window)
    april_mask = (daily_dates >= date(2005, 4, 1)) & (daily_dates <= date(2005, 4, 30))

    stats_path = OUT_DIR / "central_pa_spring_2005_snicar_aging_stats.txt"
    stats_path.write_text(
        "\n".join(
            (
                f"native_meltout_lt_1={meltout(native_dates, native_swe)}",
                f"old_emulator_meltout_lt_1={meltout(old_dates, old_swe)}",
                f"snicar_emulator_meltout_lt_1={meltout(new_dates, new_swe)}",
                f"max_daily_top_grain_radius_um={np.nanmax(grain_radius):.6f}",
                f"min_daily_nir_albedo={np.nanmin(albedo_nir):.6f}",
                f"max_daily_sabg_snow_input_wm2={np.nanmax(sabg_snow):.6f}",
                f"melt_window_swe_rmse_old={swe_rmse[0]:.6f}",
                f"melt_window_swe_rmse_snicar={swe_rmse[1]:.6f}",
                f"melt_window_tsoil_rmse_old={tsoil_rmse[0]:.6f}",
                f"melt_window_tsoil_rmse_snicar={tsoil_rmse[1]:.6f}",
                f"melt_window_fsh_rmse_old={fsh_rmse[0]:.6f}",
                f"melt_window_fsh_rmse_snicar={fsh_rmse[1]:.6f}",
                f"april_mean_snow_ice={np.nanmean(snow_ice[april_mask]):.6f}",
                f"april_mean_snow_liquid={np.nanmean(snow_liq[april_mask]):.6f}",
                f"april_mean_standing_water={np.nanmean(surface_water[april_mask]):.6f}",
            )
        )
        + "\n"
    )
    print(figure_path)
    print(stats_path)


if __name__ == "__main__":
    main()
