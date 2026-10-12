#!/usr/bin/env python3
"""Compare native and replay snow-layer phase change for Central PA."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset, num2date


ROOT = Path(__file__).resolve().parents[1]
NATIVE = (
    ROOT.parent
    / "elm-olmt/setup_runs/20260807_CentralPA_trendy_global_ICBELMBC_native_snowlayers_hourly/run"
    / "20260807_CentralPA_trendy_global_ICBELMBC_native_snowlayers_hourly.elm.h0.2005-01-01-00000.nc"
)
REPLAY = ROOT / "output/central_pa_trendy_2004_2005_layer_profiles/central_pa_trendy_emulator_2004_2005_hourly_diagnostics.nc"
OUT_DIR = ROOT / "output/central_pa_trendy_2004_2005_layer_profiles/comparison"


def _daily_native() -> tuple[np.ndarray, dict[str, np.ndarray]]:
    with Dataset(NATIVE) as dataset:
        time = dataset["time"]
        stamps = np.asarray(num2date(time[:], time.units, getattr(time, "calendar", "noleap")))
        dates = np.asarray([date(x.year, x.month, x.day) for x in stamps], dtype=object)
        melt = np.ma.filled(dataset["QSNOMELT_LYR"][:, :, 0], 0.0)
        melt = np.where(melt > 1.0e20, 0.0, melt)
        exists = np.asarray(dataset["SNO_EXISTENCE"][:, :, 0]) > 0.5
        ice = np.ma.filled(dataset["SNO_ICE"][:, :, 0], 0.0)
        liquid = np.ma.filled(dataset["SNO_LIQH2O"][:, :, 0], 0.0)
        ice = np.where(ice > 1.0e20, 0.0, ice)
        liquid = np.where(liquid > 1.0e20, 0.0, liquid)
        swe = np.asarray(dataset["H2OSNO"][:, 0], dtype=np.float64)
        h2osfc = np.asarray(dataset["H2OSFC"][:, 0], dtype=np.float64)

    unique = np.asarray(sorted(set(dates)), dtype=object)
    output = {name: [] for name in ("swe", "h2osfc", "snow_liquid", "melt", "bottom_melt")}
    for current in unique:
        selected = np.flatnonzero(dates == current)
        last = selected[-1]
        bottom_melt = 0.0
        for index in selected:
            active = np.flatnonzero(exists[index])
            if active.size:
                bottom_melt += melt[index, active[-1]] * 3600.0
        output["swe"].append(swe[last])
        output["h2osfc"].append(h2osfc[last])
        output["snow_liquid"].append(np.sum(liquid[last]))
        output["melt"].append(np.sum(melt[selected]) * 3600.0)
        output["bottom_melt"].append(bottom_melt)
    return unique, {name: np.asarray(values) for name, values in output.items()}


def _daily_replay() -> tuple[np.ndarray, dict[str, np.ndarray]]:
    with Dataset(REPLAY) as dataset:
        dates = np.asarray(
            [date(int(y), int(m), int(d)) for y, m, d in zip(dataset["year"][:], dataset["month"][:], dataset["day"][:])],
            dtype=object,
        )
        physics = np.asarray(dataset["physics_step"][:, 0]) > 0.5
        snl = np.asarray(dataset["snow_snl"][:, 0], dtype=np.float64)
        melt = np.column_stack(
            [np.asarray(dataset[f"qflx_snow_melt_lev{level}"][:, 0], dtype=np.float64) for level in range(1, 6)]
        )
        swe = np.asarray(dataset["h2osno"][:, 0], dtype=np.float64)
        h2osfc = np.asarray(dataset["h2osfc"][:, 0], dtype=np.float64)
        liquid = np.column_stack(
            [np.asarray(dataset[f"snow_liq_lev{level}"][:, 0], dtype=np.float64) for level in range(1, 6)]
        )

    unique = np.asarray(sorted(set(dates)), dtype=object)
    output = {name: [] for name in ("swe", "h2osfc", "snow_liquid", "melt", "bottom_melt")}
    for current in unique:
        selected = np.flatnonzero(dates == current)
        coupled = selected[physics[selected]]
        last = selected[-1]
        bottom_melt = 0.0
        for index in coupled:
            layer_count = (
                int(np.clip(-snl[index], 0, 5))
                if np.isfinite(snl[index])
                else 0
            )
            if layer_count:
                bottom_melt += melt[index, layer_count - 1] * 10800.0
        output["swe"].append(swe[last])
        output["h2osfc"].append(h2osfc[last])
        output["snow_liquid"].append(np.nansum(liquid[last]))
        output["melt"].append(np.nansum(melt[coupled]) * 10800.0)
        output["bottom_melt"].append(bottom_melt)
    return unique, {name: np.asarray(values) for name, values in output.items()}


def _sum_window(dates: np.ndarray, values: np.ndarray, start: date, end: date) -> float:
    return float(np.sum(values[(dates >= start) & (dates <= end)]))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    native_dates, native = _daily_native()
    replay_dates, replay = _daily_replay()
    start, end = date(2005, 3, 15), date(2005, 3, 31)

    fig, axes = plt.subplots(4, 1, figsize=(12, 13), sharex=True)
    for axis, name, ylabel in (
        (axes[0], "swe", "SWE (kg m$^{-2}$)"),
        (axes[1], "h2osfc", "H2OSFC (kg m$^{-2}$)"),
        (axes[2], "snow_liquid", "Snow liquid (kg m$^{-2}$)"),
    ):
        axis.plot(native_dates, native[name], color="black", lw=2.1, label="Native ELM")
        axis.plot(replay_dates, replay[name], color="#d62728", lw=2.0, label="Replay")
        axis.set_ylabel(ylabel)
        axis.legend(fontsize=9)

    axes[3].plot(native_dates, native["melt"], color="black", lw=2.1, label="Native total melt")
    axes[3].plot(native_dates, native["bottom_melt"], color="black", ls="--", label="Native bottom-layer melt")
    axes[3].plot(replay_dates, replay["melt"], color="#d62728", lw=2.0, label="Replay total melt")
    axes[3].plot(replay_dates, replay["bottom_melt"], color="#d62728", ls="--", label="Replay bottom-layer melt")
    axes[3].set_ylabel("Daily melt (kg m$^{-2}$)")
    axes[3].set_xlabel("Date")
    axes[3].legend(ncol=2, fontsize=9)
    for axis in axes:
        axis.set_xlim(start, end)
        axis.grid(alpha=0.25)
    fig.suptitle("Central PA snow-layer phase change: native ELM versus replay")
    fig.tight_layout()
    figure_path = OUT_DIR / "central_pa_snow_layer_phase_comparison.png"
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)

    stats = {}
    for label, window_start in (("mar20_31", date(2005, 3, 20)), ("mar28_31", date(2005, 3, 28))):
        for source_name, source_dates, source in (
            ("native", native_dates, native),
            ("replay", replay_dates, replay),
        ):
            total = _sum_window(source_dates, source["melt"], window_start, end)
            bottom = _sum_window(source_dates, source["bottom_melt"], window_start, end)
            stats[f"{label}_{source_name}_total_melt"] = total
            stats[f"{label}_{source_name}_bottom_melt"] = bottom
            stats[f"{label}_{source_name}_bottom_melt_fraction"] = bottom / total if total > 0.0 else np.nan
    stats_path = OUT_DIR / "central_pa_snow_layer_phase_stats.txt"
    stats_path.write_text("".join(f"{name}={value:.6f}\n" for name, value in stats.items()))
    print(figure_path)
    print(stats_path)
    for name, value in stats.items():
        print(f"{name}={value:.6f}")


if __name__ == "__main__":
    main()
