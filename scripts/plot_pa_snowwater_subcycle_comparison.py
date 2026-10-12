#!/usr/bin/env python3
"""Compare the Central PA SnowWater replay before and after subcycling."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset, num2date


ROOT = Path(__file__).resolve().parents[1]
OLD_DIR = ROOT / "output/central_pa_trendy_2004_2005_snicar_aging"
NEW_DIR = ROOT / "output/central_pa_trendy_2004_2005_snowwater_subcycle"
NATIVE = (
    ROOT.parent
    / "elm-olmt/setup_runs/20260804_CentralPA_trendy_global_ICBELMBC_native_daily/run"
    / "20260804_CentralPA_trendy_global_ICBELMBC_native_daily.elm.h0.2005-01-01-00000.nc"
)
OUT_DIR = NEW_DIR / "comparison"


def ymd_dates(dataset: Dataset) -> np.ndarray:
    return np.asarray(
        [date(int(y), int(m), int(d)) for y, m, d in zip(dataset["year"][:], dataset["month"][:], dataset["day"][:])],
        dtype=object,
    )


def load_daily(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with Dataset(path) as dataset:
        return (
            ymd_dates(dataset),
            np.asarray(dataset["h2osno_mean"][:, 0], dtype=np.float64),
            np.asarray(dataset["t_soisno_lev1_mean"][:, 0], dtype=np.float64),
        )


def load_hourly(path: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    with Dataset(path) as dataset:
        dates = ymd_dates(dataset)
        physics = np.asarray(dataset["physics_step"][:, 0], dtype=np.float64) > 0.5
        names = ("snow_liq_total", "h2osfc", "qflx_snow_to_soil", "qflx_rain_grnd")
        raw = {name: np.asarray(dataset[name][:, 0], dtype=np.float64) for name in names}

    unique = np.asarray(sorted(set(dates)), dtype=object)
    daily: dict[str, list[float]] = {name: [] for name in names}
    for current_date in unique:
        selected = dates == current_date
        snow_liquid = raw["snow_liq_total"][selected]
        snow_liquid = snow_liquid[np.isfinite(snow_liquid)]
        daily["snow_liq_total"].append(float(np.mean(snow_liquid)) if snow_liquid.size else np.nan)
        daily["h2osfc"].append(float(raw["h2osfc"][selected][-1]))
        # SnowWater returns mass per coupled call; rain is a rate and needs dt=10800 s.
        coupled = selected & physics
        daily["qflx_snow_to_soil"].append(float(np.nansum(raw["qflx_snow_to_soil"][coupled])))
        daily["qflx_rain_grnd"].append(float(np.nansum(raw["qflx_rain_grnd"][coupled]) * 10800.0))
    return unique, {name: np.asarray(values) for name, values in daily.items()}


def values_on(dates: np.ndarray, values: np.ndarray, wanted: list[date]) -> np.ndarray:
    lookup = {d: values[i] for i, d in enumerate(dates)}
    return np.asarray([lookup[d] for d in wanted], dtype=np.float64)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    old_dates, old_swe, old_tsoil = load_daily(OLD_DIR / "central_pa_trendy_emulator_2004_2005_daily.nc")
    new_dates, new_swe, new_tsoil = load_daily(NEW_DIR / "central_pa_trendy_emulator_2004_2005_daily.nc")
    old_hour_dates, old_hour = load_hourly(OLD_DIR / "central_pa_trendy_emulator_2004_2005_hourly_diagnostics.nc")
    new_hour_dates, new_hour = load_hourly(NEW_DIR / "central_pa_trendy_emulator_2004_2005_hourly_diagnostics.nc")

    with Dataset(NATIVE) as dataset:
        time = dataset["time"]
        native_dates = np.asarray(
            [date(x.year, x.month, x.day) for x in num2date(time[:], time.units, getattr(time, "calendar", "noleap"))],
            dtype=object,
        )
        native_swe = np.asarray(dataset["H2OSNO"][:, 0], dtype=np.float64)
        native_tsoil = np.asarray(dataset["TSOI"][:, 0, 0], dtype=np.float64)

    start, end = date(2005, 3, 15), date(2005, 3, 31)
    fig, axes = plt.subplots(4, 1, figsize=(12, 13), sharex=True)
    axes[0].plot(native_dates, native_swe, color="black", lw=2.2, label="Native ELM")
    axes[0].plot(old_dates, old_swe, color="0.55", lw=1.8, label="Old one-sweep replay")
    axes[0].plot(new_dates, new_swe, color="#d62728", lw=2.0, label="Subcycled replay")
    axes[0].set_ylabel("SWE (kg m$^{-2}$)")
    axes[0].legend(ncol=3, fontsize=9)

    axes[1].plot(old_hour_dates, old_hour["snow_liq_total"], color="0.55", label="Old snow liquid")
    axes[1].plot(new_hour_dates, new_hour["snow_liq_total"], color="#d62728", label="Subcycled snow liquid")
    axes[1].set_ylabel("Snow liquid (kg m$^{-2}$)")
    axes[1].legend(fontsize=9)

    axes[2].plot(old_hour_dates, old_hour["h2osfc"], color="0.55", label="Old standing water")
    axes[2].plot(new_hour_dates, new_hour["h2osfc"], color="#1f77b4", label="Subcycled standing water")
    axes[2].set_ylabel("H2OSFC (kg m$^{-2}$)")
    axes[2].legend(fontsize=9)

    axes[3].plot(old_hour_dates, old_hour["qflx_snow_to_soil"], color="0.55", marker="o", ms=3, label="Old snow drainage")
    axes[3].plot(new_hour_dates, new_hour["qflx_snow_to_soil"], color="#2ca02c", marker="o", ms=3, label="Subcycled snow drainage")
    axes[3].plot(new_hour_dates, new_hour["qflx_rain_grnd"], color="#1f77b4", ls="--", label="Rain to snow")
    axes[3].set_ylabel("Daily water mass (kg m$^{-2}$)")
    axes[3].legend(ncol=3, fontsize=9)
    axes[3].set_xlabel("Date")

    for axis in axes:
        axis.set_xlim(start, end)
        axis.grid(alpha=0.25)
    fig.suptitle("Central PA March 2005: SnowWater subcycling replay test")
    fig.tight_layout()
    figure_path = OUT_DIR / "central_pa_march_2005_snowwater_subcycle_comparison.png"
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)

    window = [date(2005, 3, day) for day in range(20, 32)]
    native_window = values_on(native_dates, native_swe, window)
    old_window = values_on(old_dates, old_swe, window)
    new_window = values_on(new_dates, new_swe, window)
    event = [date(2005, 3, day) for day in (28, 29)]
    stats = {
        "mar20_31_swe_rmse_old": np.sqrt(np.mean((old_window - native_window) ** 2)),
        "mar20_31_swe_rmse_subcycled": np.sqrt(np.mean((new_window - native_window) ** 2)),
        "mar28_29_rain_to_snow": np.sum(values_on(new_hour_dates, new_hour["qflx_rain_grnd"], event)),
        "mar28_29_snow_drain_old": np.sum(values_on(old_hour_dates, old_hour["qflx_snow_to_soil"], event)),
        "mar28_29_snow_drain_subcycled": np.sum(values_on(new_hour_dates, new_hour["qflx_snow_to_soil"], event)),
        "mar20_31_snow_drain_old": np.sum(values_on(old_hour_dates, old_hour["qflx_snow_to_soil"], window)),
        "mar20_31_snow_drain_subcycled": np.sum(values_on(new_hour_dates, new_hour["qflx_snow_to_soil"], window)),
        "mar31_swe_native": native_window[-1],
        "mar31_swe_old": old_window[-1],
        "mar31_swe_subcycled": new_window[-1],
        "mar31_h2osfc_old": values_on(old_hour_dates, old_hour["h2osfc"], [window[-1]])[0],
        "mar31_h2osfc_subcycled": values_on(new_hour_dates, new_hour["h2osfc"], [window[-1]])[0],
    }
    stats_path = OUT_DIR / "central_pa_march_2005_snowwater_subcycle_stats.txt"
    stats_path.write_text("".join(f"{name}={value:.6f}\n" for name, value in stats.items()))
    print(figure_path)
    print(stats_path)
    for name, value in stats.items():
        print(f"{name}={value:.6f}")


if __name__ == "__main__":
    main()
