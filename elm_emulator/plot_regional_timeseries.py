"""Plot regional replay diagnostic time series from NetCDF output."""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import numpy as np
from netCDF4 import Dataset


SECONDS_PER_DAY = 86400.0


def _masked_to_float(data) -> np.ndarray:
    return np.ma.asarray(data, dtype=np.float64).filled(np.nan)


def _dates_from_dataset(ds: Dataset) -> list[dt.datetime]:
    if "date" in ds.variables:
        values = np.asarray(ds.variables["date"][:], dtype=np.int64).reshape(-1)
        dates: list[dt.datetime] = []
        for value in values:
            text = f"{int(value):08d}" if int(value) >= 10_000_00 else f"{int(value):06d}"
            if len(text) == 8:
                dates.append(dt.datetime(int(text[:4]), int(text[4:6]), int(text[6:8])))
            else:
                dates.append(dt.datetime(int(text[:4]), int(text[4:6]), 15))
        return dates
    years = np.asarray(ds.variables["year"][:], dtype=np.int32)
    months = np.asarray(ds.variables["month"][:], dtype=np.int32)
    days = np.asarray(ds.variables.get("day", np.ones_like(years))[:], dtype=np.int32)
    return [
        dt.datetime(int(y), int(m), max(1, int(d)))
        for y, m, d in zip(years, months, days)
    ]


def _nearest_cell(ds: Dataset, lat: float, lon: float) -> tuple[int, float]:
    lat_arr = np.asarray(ds.variables["lat"][:], dtype=np.float64)
    lon_arr = np.asarray(ds.variables["lon"][:], dtype=np.float64)
    lon_query = float(lon)
    if np.nanmax(lon_arr) > 180.0 and lon_query < 0.0:
        lon_query += 360.0
    elif np.nanmax(lon_arr) <= 180.0 and lon_query > 180.0:
        lon_query -= 360.0
    dlat = np.deg2rad(lat_arr - float(lat))
    dlon = np.deg2rad(((lon_arr - lon_query + 180.0) % 360.0) - 180.0)
    lat_mid = np.deg2rad(0.5 * (lat_arr + float(lat)))
    dist_rad = np.sqrt(dlat * dlat + (np.cos(lat_mid) * dlon) ** 2)
    idx = int(np.nanargmin(dist_rad))
    return idx, float(dist_rad[idx] * 6371.0)


def _selector_label(ds: Dataset, selector: int | None, lat: float | None, lon: float | None) -> str:
    if selector is None:
        return "domain mean"
    lat_val = float(np.asarray(ds.variables["lat"][:], dtype=np.float64)[selector])
    lon_val = float(np.asarray(ds.variables["lon"][:], dtype=np.float64)[selector])
    if lat is not None and lon is not None:
        _, distance_km = _nearest_cell(ds, lat, lon)
        return f"cell {selector} at {lat_val:.3f}, {lon_val:.3f} ({distance_km:.1f} km from requested)"
    return f"cell {selector} at {lat_val:.3f}, {lon_val:.3f}"


def _read_series(
    ds: Dataset,
    names: tuple[str, ...],
    *,
    selector: int | None,
    scale: float = 1.0,
    offset: float = 0.0,
) -> tuple[np.ndarray | None, str | None]:
    for name in names:
        if name not in ds.variables:
            continue
        var = ds.variables[name]
        values = _masked_to_float(var[:])
        dims = tuple(getattr(var, "dimensions", ()))
        if values.ndim == 1:
            series = values
        elif "time" in dims and "cell" in dims:
            time_axis = dims.index("time")
            cell_axis = dims.index("cell")
            values = np.moveaxis(values, (time_axis, cell_axis), (0, 1))
            if selector is None:
                series = np.nanmean(values, axis=1)
            else:
                series = values[:, int(selector)]
        elif values.ndim == 2:
            if selector is None:
                series = np.nanmean(values, axis=1)
            else:
                series = values[:, int(selector)]
        else:
            raise ValueError(f"{name} has unsupported shape {values.shape}")
        return series * float(scale) + float(offset), name
    return None, None


def _plot_optional_lines(ax, dates, series_specs, *, ylabel: str, title: str) -> bool:
    plotted = False
    for label, values in series_specs:
        if values is None:
            continue
        ax.plot(dates, values, marker="o", linewidth=1.7, markersize=3.5, label=label)
        plotted = True
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontsize=10)
    ax.grid(True, alpha=0.25)
    if plotted and len(series_specs) > 1:
        ax.legend(loc="best", fontsize=8)
    if not plotted:
        ax.text(0.5, 0.5, "not available", ha="center", va="center", transform=ax.transAxes)
    return plotted


def plot_regional_timeseries(
    input_path: Path,
    output_path: Path,
    *,
    selector: int | None = None,
    lat: float | None = None,
    lon: float | None = None,
    title: str | None = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with Dataset(input_path) as ds:
        dates = _dates_from_dataset(ds)
        label = _selector_label(ds, selector, lat, lon)

        lai, _ = _read_series(ds, ("lai_mean",), selector=selector)
        et, _ = _read_series(ds, ("qflx_evap_tot_mean",), selector=selector, scale=SECONDS_PER_DAY)
        sh, _ = _read_series(ds, ("eflx_sh_tot_mean",), selector=selector)
        runoff, _ = _read_series(ds, ("qflx_runoff_tot_mean",), selector=selector, scale=SECONDS_PER_DAY)
        btran, _ = _read_series(ds, ("btran_mean",), selector=selector)

        soil_t_1, name_t1 = _read_series(
            ds,
            ("t_soisno_lev1_mean", "t_soil_mean"),
            selector=selector,
            offset=-273.15,
        )
        soil_t_3, name_t3 = _read_series(ds, ("t_soisno_lev3_mean",), selector=selector, offset=-273.15)
        soil_t_5, name_t5 = _read_series(ds, ("t_soisno_lev5_mean",), selector=selector, offset=-273.15)

        soil_m_1, _ = _read_series(ds, ("h2osoi_liqvol_lev1_mean", "liq_mean"), selector=selector)
        soil_m_3, _ = _read_series(ds, ("h2osoi_liqvol_lev3_mean",), selector=selector)
        soil_m_5, _ = _read_series(ds, ("h2osoi_liqvol_lev5_mean",), selector=selector)

        missing = []
        if name_t3 is None:
            missing.append("t_soisno_lev3_mean")
        if name_t5 is None:
            missing.append("t_soisno_lev5_mean")
        if soil_m_5 is None:
            missing.append("h2osoi_liqvol_lev5_mean")
        if missing:
            print("Missing variables in input; skipped: " + ", ".join(missing))

    fig, axes = plt.subplots(7, 1, figsize=(11, 13.5), sharex=True)
    _plot_optional_lines(axes[0], dates, (("LAI", lai),), ylabel="m2 m-2", title="LAI")
    _plot_optional_lines(axes[1], dates, (("ET", et),), ylabel="mm d-1", title="ET")
    _plot_optional_lines(axes[2], dates, (("SH", sh),), ylabel="W m-2", title="Sensible heat")
    _plot_optional_lines(axes[3], dates, (("Runoff", runoff),), ylabel="mm d-1", title="Runoff")
    _plot_optional_lines(
        axes[4],
        dates,
        (("Layer 1", soil_t_1), ("Layer 3", soil_t_3), ("Layer 5", soil_t_5)),
        ylabel="deg C",
        title="Soil temperature",
    )
    _plot_optional_lines(
        axes[5],
        dates,
        (("Layer 1", soil_m_1), ("Layer 3", soil_m_3), ("Layer 5", soil_m_5)),
        ylabel="m3 m-3",
        title="Soil moisture",
    )
    _plot_optional_lines(axes[6], dates, (("BTRAN", btran),), ylabel="1", title="BTRAN")
    axes[-1].set_xlabel("Time")
    fig.autofmt_xdate()
    fig.suptitle(title or f"{input_path.name} - {label}", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {output_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot regional replay monthly/daily diagnostic time series from NetCDF output."
    )
    parser.add_argument("input", type=Path, help="Regional diagnostic NetCDF file.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG path. Default: <input_stem>_flux_state_timeseries.png",
    )
    parser.add_argument("--cell-index", type=int, default=None, help="0-based cell index to plot.")
    parser.add_argument("--lat", type=float, default=None, help="Latitude for nearest-cell selection.")
    parser.add_argument("--lon", type=float, default=None, help="Longitude for nearest-cell selection.")
    parser.add_argument("--title", default=None, help="Optional figure title.")
    args = parser.parse_args()
    if (args.lat is None) != (args.lon is None):
        parser.error("--lat and --lon must be supplied together")
    if args.cell_index is not None and args.lat is not None:
        parser.error("--cell-index cannot be combined with --lat/--lon")
    return args


def main() -> None:
    args = _parse_args()
    selector = args.cell_index
    with Dataset(args.input) as ds:
        if args.lat is not None and args.lon is not None:
            selector, distance_km = _nearest_cell(ds, args.lat, args.lon)
            print(f"nearest cell: {selector} ({distance_km:.2f} km)")
        elif selector is not None:
            ncell = len(ds.dimensions["cell"])
            if selector < 0 or selector >= ncell:
                raise SystemExit(f"--cell-index must be in [0, {ncell - 1}], got {selector}")
    output = args.output or args.input.with_name(f"{args.input.stem}_flux_state_timeseries.png")
    plot_regional_timeseries(
        args.input,
        output,
        selector=selector,
        lat=args.lat,
        lon=args.lon,
        title=args.title,
    )


if __name__ == "__main__":
    main()
