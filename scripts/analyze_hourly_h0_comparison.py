#!/usr/bin/env python
"""Analyze hourly emulator/native ELM h0 comparisons for the 2012 site cases."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset


LV = 2.501e6
SECSPDAY = 86400.0

DEFAULT_CASES = {
    "central_tx": {
        "label": "Central Texas",
        "emulator": Path(
            "/code/ELM_emulator/output/central_tx_h0align_diag2_2012/run/"
            "central_tx_CentralTexas_ICBELMBC_SP_emulator_h0align_diag2_2012_1yr.elm.h0.2012-01-01-00000.nc"
        ),
        "native": Path(
            "/output/e3sm_run/central_tx_CentralTexas_ICBELMBC_SP_E3SM_ORNL_h0align_diag2_2012_1yr/run"
        ),
    },
    "east_tn": {
        "label": "East Tennessee",
        "emulator": Path(
            "/output/e3sm_run/east_tn_EastTennessee_ICBELMBC_SP_emulator_hourly_2012_1yr/run/"
            "east_tn_EastTennessee_ICBELMBC_SP_emulator_hourly_2012_1yr.elm.h0.2012-01-01-00000.nc"
        ),
        "native": Path(
            "/output/e3sm_run/east_tn_EastTennessee_ICBELMBC_SP_E3SM_ORNL_subset_dailyfiles_2012_1yr/run"
        ),
    },
}

SCALAR_VARS = (
    "TBOT",
    "QBOT",
    "PBOT",
    "PSRF",
    "FLDS",
    "WIND",
    "RH2M",
    "FSDS",
    "TLAI",
    "FPSN",
    "EFLX_LH_TOT",
    "QVEGT",
    "QVEGE",
    "QSOIL",
    "FCTR",
    "FCEV",
    "FGEV",
    "FSH",
    "QRUNOFF",
    "QOVER",
    "BTRAN",
    "ZWT",
    "TG",
    "FGR",
    "FGR12",
    "HS_SOIL",
    "HS_TOPSNOW",
    "HS_H2OSFC",
    "DHSDT",
    "FSA",
    "FSR",
    "SABG",
    "SABV",
    "FSDSVD",
    "FSDSVI",
    "FSDSND",
    "FSDSNI",
    "FSRVD",
    "FSRVI",
    "FSRND",
    "FSRNI",
)

NUMRAD_VARS = ("ALBD", "ALBI", "ALBGRD", "ALBGRI")
NUMRAD_BANDS = ("VIS", "NIR")
LAYER_INDICES = (0, 2, 4)
LAYER_LABELS = {0: "L1", 2: "L3", 4: "L5"}

PLOT_KEYS = (
    ("TLAI", "LAI", "1"),
    ("FPSN", "FPSN", "umol m-2 s-1"),
    ("ET_MM_D", "ET", "mm d-1"),
    ("FSH", "SH", "W m-2"),
    ("TG", "TG", "K"),
    ("FGR", "FGR", "W m-2"),
    ("HS_SOIL", "HS_SOIL", "W m-2"),
    ("DHSDT", "DHSDT", "W m-2 K-1"),
    ("TSOI_L1", "TSOI L1", "K"),
    ("TSOI_L3", "TSOI L3", "K"),
    ("TSOI_L5", "TSOI L5", "K"),
    ("H2OSOI_L1", "H2OSOI L1", "m3 m-3"),
    ("BTRAN", "BTRAN", "1"),
    ("SAB_FRAC_GROUND", "SABG/FSA", "1"),
)


def _filled(var) -> np.ndarray:
    return np.ma.filled(var[:], np.nan).astype(np.float64, copy=False)


def _time_first(arr: np.ndarray, dims: tuple[str, ...]) -> tuple[np.ndarray, tuple[str, ...]]:
    if "time" not in dims:
        raise ValueError(f"variable has no time dimension: {dims}")
    time_axis = dims.index("time")
    if time_axis == 0:
        return arr, dims
    return np.moveaxis(arr, time_axis, 0), ("time",) + tuple(
        dim for idx, dim in enumerate(dims) if idx != time_axis
    )


def _mean_non_time(arr: np.ndarray, dims: tuple[str, ...]) -> np.ndarray:
    arr, dims = _time_first(arr, dims)
    while arr.ndim > 1:
        arr = np.nanmean(arr, axis=-1)
    return arr


def _read_scalar(ds: Dataset, name: str) -> np.ndarray | None:
    if name not in ds.variables:
        return None
    var = ds.variables[name]
    return _mean_non_time(_filled(var), tuple(var.dimensions))


def _read_layer(ds: Dataset, name: str, layer: int) -> np.ndarray | None:
    if name not in ds.variables:
        return None
    var = ds.variables[name]
    dims = tuple(var.dimensions)
    if "levgrnd" not in dims:
        return None
    arr = _filled(var)
    arr = np.take(arr, int(layer), axis=dims.index("levgrnd"))
    dims = tuple(dim for dim in dims if dim != "levgrnd")
    return _mean_non_time(arr, dims)


def _read_numrad(ds: Dataset, name: str, band: int) -> np.ndarray | None:
    if name not in ds.variables:
        return None
    var = ds.variables[name]
    dims = tuple(var.dimensions)
    if "numrad" not in dims:
        return None
    arr = _filled(var)
    arr = np.take(arr, int(band), axis=dims.index("numrad"))
    dims = tuple(dim for dim in dims if dim != "numrad")
    return _mean_non_time(arr, dims)


def _safe_ratio(num: np.ndarray | None, den: np.ndarray | None, min_den: float = 1.0) -> np.ndarray | None:
    if num is None or den is None:
        return None
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(np.abs(den) >= min_den, num / den, np.nan)


def _sum_keys(data: dict[str, np.ndarray | None], keys: tuple[str, ...]) -> np.ndarray | None:
    arrays = [data.get(key) for key in keys]
    if any(arr is None for arr in arrays):
        return None
    return np.sum(np.stack([arr for arr in arrays if arr is not None], axis=0), axis=0)


def _read_soil_profile(ds: Dataset, name: str) -> np.ndarray | None:
    if name not in ds.variables:
        return None
    var = ds.variables[name]
    dims = tuple(var.dimensions)
    if "levgrnd" not in dims:
        return None
    arr, dims = _time_first(_filled(var), dims)
    lev_axis = dims.index("levgrnd")
    if lev_axis != 1:
        arr = np.moveaxis(arr, lev_axis, 1)
        dims = ("time", "levgrnd") + tuple(
            dim for idx, dim in enumerate(dims) if idx not in (0, lev_axis)
        )
    while arr.ndim > 2:
        arr = np.nanmean(arr, axis=-1)
    return arr


def read_h0_file(path: Path) -> dict[str, np.ndarray | None]:
    out: dict[str, np.ndarray | None] = {}
    with Dataset(path) as ds:
        out["_time"] = np.asarray(ds.variables["time"][:], dtype=np.float64)
        out["_mcdate"] = np.asarray(ds.variables["mcdate"][:], dtype=np.int64)
        out["_mcsec"] = np.asarray(ds.variables["mcsec"][:], dtype=np.int64)
        for name in SCALAR_VARS:
            out[name] = _read_scalar(ds, name)
        for name in ("TSOI", "H2OSOI"):
            out[f"_{name}_PROFILE"] = _read_soil_profile(ds, name)
            for layer in LAYER_INDICES:
                out[f"{name}_{LAYER_LABELS[layer]}"] = _read_layer(ds, name, layer)
        for name in NUMRAD_VARS:
            for band, label in enumerate(NUMRAD_BANDS):
                out[f"{name}_{label}"] = _read_numrad(ds, name, band)

    out["FSDS_SUM"] = _sum_keys(out, ("FSDSVD", "FSDSVI", "FSDSND", "FSDSNI"))
    out["FSDS_TOTAL"] = out.get("FSDS") if out.get("FSDS") is not None else out.get("FSDS_SUM")
    out["FSR_SUM"] = _sum_keys(out, ("FSRVD", "FSRVI", "FSRND", "FSRNI"))
    out["ET_MM_D"] = None if out.get("EFLX_LH_TOT") is None else out["EFLX_LH_TOT"] * SECSPDAY / LV
    out["APP_ALB"] = _safe_ratio(out.get("FSR"), out.get("FSDS_TOTAL"))
    out["SAB_FRAC_GROUND"] = _safe_ratio(out.get("SABG"), out.get("FSA"))
    out["SAB_CLOSURE"] = (
        None
        if out.get("SABG") is None or out.get("SABV") is None or out.get("FSA") is None
        else out["SABG"] + out["SABV"] - out["FSA"]
    )
    out["SW_CLOSURE"] = (
        None
        if out.get("FSDS_TOTAL") is None or out.get("FSR") is None or out.get("FSA") is None
        else out["FSDS_TOTAL"] - out["FSR"] - out["FSA"]
    )
    return out


def _history_files(path: Path) -> list[Path]:
    path = Path(path)
    if path.is_dir():
        files = sorted(path.glob("*.elm.h0.*.nc"))
        if not files:
            raise FileNotFoundError(f"No h0 files found in {path}")
        return files
    return [path]


def _concat_h0(parts: list[dict[str, np.ndarray | None]]) -> dict[str, np.ndarray | None]:
    if len(parts) == 1:
        return parts[0]
    keys = sorted(set().union(*(part.keys() for part in parts)))
    out: dict[str, np.ndarray | None] = {}
    for key in keys:
        arrays = [part.get(key) for part in parts]
        if any(arr is None for arr in arrays):
            out[key] = None
            continue
        out[key] = np.concatenate([arr for arr in arrays if arr is not None], axis=0)
    return out


def read_h0(path: Path) -> dict[str, np.ndarray | None]:
    return _concat_h0([read_h0_file(file_path) for file_path in _history_files(path)])


def _slice_time(value: np.ndarray | None, start: int, stop: int) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.asarray(value)
    if arr.ndim == 0 or arr.shape[0] < stop:
        return value
    return arr[start:stop]


def _take_time_indices(value: np.ndarray | None, indices: np.ndarray) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.asarray(value)
    if arr.ndim == 0 or arr.shape[0] <= int(np.max(indices, initial=-1)):
        return value
    return arr[indices]


def _record_labels(data: dict[str, np.ndarray | None]) -> list[tuple[int, int]] | None:
    mcdate = data.get("_mcdate")
    mcsec = data.get("_mcsec")
    if mcdate is None or mcsec is None:
        return None
    dates = np.asarray(mcdate)
    secs = np.asarray(mcsec)
    if dates.shape[0] != secs.shape[0]:
        return None
    return [(int(date), int(sec)) for date, sec in zip(dates, secs)]


def _align_h0_by_record_label(
    emu: dict[str, np.ndarray | None],
    elm: dict[str, np.ndarray | None],
) -> tuple[dict[str, np.ndarray | None], dict[str, np.ndarray | None]]:
    emu_labels = _record_labels(emu)
    elm_labels = _record_labels(elm)
    if not emu_labels or not elm_labels:
        return emu, elm
    if len(emu_labels) == len(elm_labels) and emu_labels == elm_labels:
        return emu, elm

    elm_lookup = {label: idx for idx, label in enumerate(elm_labels)}
    emu_indices: list[int] = []
    elm_indices: list[int] = []
    for emu_idx, label in enumerate(emu_labels):
        elm_idx = elm_lookup.get(label)
        if elm_idx is None:
            continue
        emu_indices.append(emu_idx)
        elm_indices.append(elm_idx)
    if len(emu_indices) < 2:
        return emu, elm
    emu_take = np.asarray(emu_indices, dtype=np.int64)
    elm_take = np.asarray(elm_indices, dtype=np.int64)
    return (
        {key: _take_time_indices(value, emu_take) for key, value in emu.items()},
        {key: _take_time_indices(value, elm_take) for key, value in elm.items()},
    )


def align_h0(
    emu: dict[str, np.ndarray | None],
    elm: dict[str, np.ndarray | None],
    *,
    elm_shift_records: int = 0,
    align_by_record_label: bool = True,
) -> tuple[dict[str, np.ndarray | None], dict[str, np.ndarray | None]]:
    """Align time-first h0 dictionaries.

    Positive ``elm_shift_records`` compares emulator record ``t`` against ELM
    record ``t + shift``. This is useful for native h0 streams whose timestamp
    names the end of the averaging interval.
    """
    if elm_shift_records == 0:
        if align_by_record_label:
            return _align_h0_by_record_label(emu, elm)
        return emu, elm
    emu_n = int(emu["_time"].shape[0])  # type: ignore[index]
    elm_n = int(elm["_time"].shape[0])  # type: ignore[index]
    emu_start = max(0, -int(elm_shift_records))
    elm_start = max(0, int(elm_shift_records))
    n = min(emu_n - emu_start, elm_n - elm_start)
    if n <= 0:
        raise ValueError(f"Cannot apply elm_shift_records={elm_shift_records}: no overlapping records")
    emu_stop = emu_start + n
    elm_stop = elm_start + n
    return (
        {key: _slice_time(value, emu_start, emu_stop) for key, value in emu.items()},
        {key: _slice_time(value, elm_start, elm_stop) for key, value in elm.items()},
    )


def _common_length(emu: dict[str, np.ndarray | None], elm: dict[str, np.ndarray | None]) -> int:
    return min(int(emu["_time"].shape[0]), int(elm["_time"].shape[0]))  # type: ignore[index]


def _metric_row(case: str, key: str, emu: np.ndarray | None, elm: np.ndarray | None) -> dict[str, object]:
    row: dict[str, object] = {
        "case": case,
        "metric": key,
        "n": 0,
        "elm_mean": np.nan,
        "emulator_mean": np.nan,
        "bias_emulator_minus_elm": np.nan,
        "mae": np.nan,
        "rmse": np.nan,
        "p95_abs": np.nan,
        "max_abs": np.nan,
        "max_abs_hour": np.nan,
        "corr": np.nan,
    }
    if emu is None or elm is None:
        return row
    n = min(int(emu.shape[0]), int(elm.shape[0]))
    lhs = np.asarray(emu[:n], dtype=np.float64)
    rhs = np.asarray(elm[:n], dtype=np.float64)
    valid = np.isfinite(lhs) & np.isfinite(rhs)
    if not np.any(valid):
        return row
    diff = lhs[valid] - rhs[valid]
    valid_idx = np.flatnonzero(valid)
    max_pos = int(np.nanargmax(np.abs(diff)))
    row["n"] = int(valid.sum())
    row["elm_mean"] = float(np.nanmean(rhs[valid]))
    row["emulator_mean"] = float(np.nanmean(lhs[valid]))
    row["bias_emulator_minus_elm"] = float(np.nanmean(diff))
    row["mae"] = float(np.nanmean(np.abs(diff)))
    row["rmse"] = float(np.sqrt(np.nanmean(diff * diff)))
    row["p95_abs"] = float(np.nanpercentile(np.abs(diff), 95))
    row["max_abs"] = float(np.nanmax(np.abs(diff)))
    row["max_abs_hour"] = int(valid_idx[max_pos])
    if valid.sum() >= 2 and np.nanstd(lhs[valid]) > 0.0 and np.nanstd(rhs[valid]) > 0.0:
        row["corr"] = float(np.corrcoef(lhs[valid], rhs[valid])[0, 1])
    return row


def _daily_mean(arr: np.ndarray | None) -> np.ndarray | None:
    if arr is None:
        return None
    nday = int(arr.shape[0]) // 24
    if nday <= 0:
        return None
    return np.nanmean(arr[: nday * 24].reshape(nday, 24), axis=1)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _hour_label(hour: int) -> str:
    day = int(hour) // 24 + 1
    hod = int(hour) % 24
    return f"DOY {day:03d} hour {hod:02d}"


def plot_daily_overview(case: str, label: str, outdir: Path, emu: dict[str, np.ndarray | None], elm: dict[str, np.ndarray | None]) -> Path:
    days = np.arange(1, int(_common_length(emu, elm)) // 24 + 1)
    ncol = 3
    nrow = int(math.ceil(len(PLOT_KEYS) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(14, 2.5 * nrow), sharex=True)
    flat_axes = axes.ravel()
    for ax, (key, title, units) in zip(flat_axes, PLOT_KEYS):
        e = _daily_mean(emu.get(key))
        n = _daily_mean(elm.get(key))
        if e is not None:
            ax.plot(days[: e.shape[0]], e, color="#c43c39", lw=1.5, label="emulator")
        if n is not None:
            ax.plot(days[: n.shape[0]], n, color="#2f5f9f", lw=1.2, ls="--", label="ELM")
        ax.set_title(title)
        ax.set_ylabel(units)
        ax.grid(True, alpha=0.25)
    for ax in flat_axes[len(PLOT_KEYS):]:
        ax.set_visible(False)
    axes[0, 0].legend(loc="best", fontsize=8)
    for ax in axes[-1, :]:
        ax.set_xlabel("Day of 2012")
    fig.suptitle(f"{label}: daily means from hourly h0")
    fig.tight_layout()
    path = outdir / f"{case}_daily_overview.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_spike_window(case: str, label: str, outdir: Path, emu: dict[str, np.ndarray | None], elm: dict[str, np.ndarray | None]) -> tuple[Path, int]:
    tsoi = emu.get("TSOI_L1")
    if tsoi is None or not np.any(np.isfinite(tsoi)):
        raise ValueError(f"{case} has no finite emulator TSOI_L1 values")
    peak = int(np.nanargmax(tsoi))
    start = max(0, peak - 7 * 24)
    stop = min(int(tsoi.shape[0]), peak + 7 * 24 + 1)
    x = np.arange(start, stop) / 24.0 + 1.0

    fig, axes = plt.subplots(4, 1, figsize=(13, 10), sharex=True)
    for key, color in (("TSOI_L1", "#c43c39"), ("TSOI_L3", "#dd8f3a")):
        if emu.get(key) is not None:
            axes[0].plot(x, emu[key][start:stop], color=color, lw=1.4, label=f"emulator {key}")
        if elm.get(key) is not None:
            axes[0].plot(x, elm[key][start:stop], color=color, lw=1.0, ls="--", label=f"ELM {key}")
    axes[0].set_ylabel("K")
    axes[0].set_title("Soil temperature")

    for source, data, style in (("emulator", emu, "-"), ("ELM", elm, "--")):
        if data.get("TG") is not None:
            axes[1].plot(x, data["TG"][start:stop], lw=1.3, ls=style, label=f"{source} TG")
    axes[1].set_ylabel("K")
    axes[1].set_title("Ground temperature")

    for source, data, style in (("emulator", emu, "-"), ("ELM", elm, "--")):
        if data.get("FGR") is not None:
            axes[2].plot(x, data["FGR"][start:stop], lw=1.2, ls=style, label=f"{source} FGR")
    axes[2].axhline(0.0, color="0.2", lw=0.7)
    axes[2].set_ylabel("W m-2")
    axes[2].set_title("Ground heat flux")

    if emu.get("HS_SOIL") is not None:
        axes[2].plot(x, emu["HS_SOIL"][start:stop], color="#c43c39", lw=1.0, alpha=0.8, label="emulator HS_SOIL")
    if elm.get("HS_SOIL") is not None:
        axes[2].plot(x, elm["HS_SOIL"][start:stop], color="#c43c39", lw=0.9, ls=":", alpha=0.9, label="ELM HS_SOIL")

    for key, color in (("FSA", "#333333"), ("SABG", "#2d8a57"), ("SABV", "#8050a0")):
        if emu.get(key) is not None:
            axes[3].plot(x, emu[key][start:stop], color=color, lw=1.2, label=f"emulator {key}")
        if elm.get(key) is not None:
            axes[3].plot(x, elm[key][start:stop], color=color, lw=0.9, ls="--", label=f"ELM {key}")
    axes[3].set_ylabel("W m-2")
    axes[3].set_title("Absorbed shortwave partitioning")
    axes[3].set_xlabel("Day of 2012")

    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=8, ncol=2)
    fig.suptitle(f"{label}: hourly context around emulator TSOI L1 peak ({_hour_label(peak)})")
    fig.tight_layout()
    path = outdir / f"{case}_spike_window.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path, peak


def plot_soil_heatmap(case: str, label: str, outdir: Path, emu: dict[str, np.ndarray | None], elm: dict[str, np.ndarray | None]) -> Path | None:
    e = emu.get("_TSOI_PROFILE")
    n = elm.get("_TSOI_PROFILE")
    if e is None or n is None:
        return None
    nh = min(e.shape[0], n.shape[0])
    nl = min(e.shape[1], n.shape[1])
    nday = nh // 24
    diff = e[: nday * 24, :nl] - n[: nday * 24, :nl]
    daily = np.nanmean(diff.reshape(nday, 24, nl), axis=1).T
    vmax = float(np.nanpercentile(np.abs(daily), 98))
    vmax = max(vmax, 1.0)
    fig, ax = plt.subplots(figsize=(13, 5))
    im = ax.imshow(daily, aspect="auto", origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_title(f"{label}: daily mean TSOI difference, emulator - ELM")
    ax.set_xlabel("Day of 2012")
    ax.set_ylabel("Soil layer index")
    ax.set_yticks(np.arange(nl))
    ax.set_yticklabels([str(i + 1) for i in range(nl)])
    fig.colorbar(im, ax=ax, label="K")
    fig.tight_layout()
    path = outdir / f"{case}_tsoi_daily_difference_heatmap.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def top_event_rows(case: str, emu: dict[str, np.ndarray | None], elm: dict[str, np.ndarray | None], nrows: int = 25) -> list[dict[str, object]]:
    e = emu.get("TSOI_L1")
    n = elm.get("TSOI_L1")
    if e is None or n is None:
        return []
    nh = min(e.shape[0], n.shape[0])
    diff = e[:nh] - n[:nh]
    order = np.argsort(np.where(np.isfinite(diff), np.abs(diff), -np.inf))[::-1][:nrows]
    rows: list[dict[str, object]] = []
    for hour in order:
        row: dict[str, object] = {
            "case": case,
            "hour": int(hour),
            "day": int(hour) // 24 + 1,
            "hour_of_day": int(hour) % 24,
            "TSOI_L1_emulator": float(e[hour]),
            "TSOI_L1_ELM": float(n[hour]),
            "TSOI_L1_diff": float(diff[hour]),
        }
        for key in (
            "TBOT",
            "QBOT",
            "PBOT",
            "FLDS",
            "WIND",
            "RH2M",
            "FSDS",
            "TG",
            "FGR",
            "FGR12",
            "HS_SOIL",
            "HS_TOPSNOW",
            "HS_H2OSFC",
            "DHSDT",
            "FSA",
            "SABG",
            "SABV",
            "ET_MM_D",
            "EFLX_LH_TOT",
            "FSH",
            "BTRAN",
            "H2OSOI_L1",
        ):
            if emu.get(key) is not None:
                row[f"{key}_emulator"] = float(emu[key][hour])
            if elm.get(key) is not None:
                row[f"{key}_ELM"] = float(elm[key][hour])
        rows.append(row)
    return rows


def analyze_case(
    case: str,
    label: str,
    emulator_path: Path,
    native_path: Path,
    outdir: Path,
    *,
    elm_shift_records: int = 0,
    align_by_record_label: bool = True,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[Path], int]:
    if not emulator_path.exists():
        raise FileNotFoundError(emulator_path)
    if not native_path.exists():
        raise FileNotFoundError(native_path)
    emu, elm = align_h0(
        read_h0(emulator_path),
        read_h0(native_path),
        elm_shift_records=elm_shift_records,
        align_by_record_label=align_by_record_label,
    )

    keys = [
        "TLAI",
        "TBOT",
        "QBOT",
        "PBOT",
        "FLDS",
        "WIND",
        "RH2M",
        "FSDS",
        "FPSN",
        "ET_MM_D",
        "EFLX_LH_TOT",
        "QVEGT",
        "QVEGE",
        "QSOIL",
        "FCTR",
        "FCEV",
        "FGEV",
        "FSH",
        "QRUNOFF",
        "QOVER",
        "BTRAN",
        "ZWT",
        "TG",
        "FGR",
        "FGR12",
        "HS_SOIL",
        "HS_TOPSNOW",
        "HS_H2OSFC",
        "DHSDT",
        "FSA",
        "FSR",
        "SABG",
        "SABV",
        "FSDS_SUM",
        "FSDS_TOTAL",
        "FSR_SUM",
        "APP_ALB",
        "SAB_FRAC_GROUND",
        "SAB_CLOSURE",
        "SW_CLOSURE",
    ]
    for name in ("TSOI", "H2OSOI"):
        for layer in LAYER_INDICES:
            keys.append(f"{name}_{LAYER_LABELS[layer]}")
    for name in NUMRAD_VARS:
        for band in NUMRAD_BANDS:
            keys.append(f"{name}_{band}")

    metrics = [_metric_row(case, key, emu.get(key), elm.get(key)) for key in keys]
    daily_metrics = [
        _metric_row(case, key, _daily_mean(emu.get(key)), _daily_mean(elm.get(key)))
        for key in keys
    ]
    plot_paths = [
        plot_daily_overview(case, label, outdir, emu, elm),
    ]
    spike_path, peak_hour = plot_spike_window(case, label, outdir, emu, elm)
    plot_paths.append(spike_path)
    heatmap = plot_soil_heatmap(case, label, outdir, emu, elm)
    if heatmap is not None:
        plot_paths.append(heatmap)
    return metrics, daily_metrics, plot_paths, peak_hour


def write_summary(
    path: Path,
    metrics: list[dict[str, object]],
    peak_hours: dict[str, int],
    *,
    elm_shift_records: int = 0,
    align_by_record_label: bool = True,
) -> None:
    important = {
        "TLAI",
        "TBOT",
        "QBOT",
        "PBOT",
        "FLDS",
        "WIND",
        "RH2M",
        "FSDS",
        "FPSN",
        "ET_MM_D",
        "EFLX_LH_TOT",
        "QVEGT",
        "QVEGE",
        "QSOIL",
        "FCTR",
        "FCEV",
        "FGEV",
        "FSH",
        "TG",
        "FGR",
        "TSOI_L1",
        "TSOI_L3",
        "H2OSOI_L1",
        "BTRAN",
        "FSA",
        "SABG",
        "SABV",
        "SAB_FRAC_GROUND",
        "APP_ALB",
    }
    lines = ["Hourly h0 emulator-vs-native ELM comparison", ""]
    if elm_shift_records:
        lines.append(f"ELM records shifted by +{elm_shift_records} relative to emulator before comparison.")
    elif align_by_record_label:
        lines.append("Records aligned by matching mcdate/mcsec labels before comparison.")
    if elm_shift_records or align_by_record_label:
        lines.append("")
    for case, peak in peak_hours.items():
        lines.append(f"{case}: emulator TSOI L1 peak at {_hour_label(peak)}")
    lines.append("")
    lines.append("Selected hourly metrics; bias, MAE, RMSE are emulator minus ELM.")
    lines.append("case, metric, elm_mean, emu_mean, bias, mae, rmse, corr")
    for row in metrics:
        if row["metric"] not in important:
            continue
        lines.append(
            f"{row['case']}, {row['metric']}, "
            f"{row['elm_mean']:.6g}, {row['emulator_mean']:.6g}, "
            f"{row['bias_emulator_minus_elm']:.6g}, {row['mae']:.6g}, "
            f"{row['rmse']:.6g}, {row['corr']:.4f}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/code/ELM_emulator/output/hourly_h0_2012_analysis"),
    )
    parser.add_argument(
        "--case",
        choices=tuple(DEFAULT_CASES),
        action="append",
        help="Case key to analyze. May be provided more than once. Defaults to all cases.",
    )
    parser.add_argument(
        "--elm-shift-records",
        type=int,
        default=0,
        help="Compare emulator record t against ELM record t + shift. Mainly for legacy start-labelled emulator h0.",
    )
    parser.add_argument(
        "--no-align-by-record-label",
        action="store_true",
        help="Disable default mcdate/mcsec alignment and compare records by raw index.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_metrics: list[dict[str, object]] = []
    all_daily_metrics: list[dict[str, object]] = []
    all_events: list[dict[str, object]] = []
    peak_hours: dict[str, int] = {}
    plot_paths: list[Path] = []

    selected_cases = args.case or list(DEFAULT_CASES)
    for case in selected_cases:
        spec = DEFAULT_CASES[case]
        metrics, daily_metrics, paths, peak = analyze_case(
            case,
            str(spec["label"]),
            Path(spec["emulator"]),
            Path(spec["native"]),
            args.output_dir,
            elm_shift_records=args.elm_shift_records,
            align_by_record_label=not args.no_align_by_record_label,
        )
        all_metrics.extend(metrics)
        all_daily_metrics.extend(daily_metrics)
        peak_hours[case] = peak
        plot_paths.extend(paths)
        emu, elm = align_h0(
            read_h0(Path(spec["emulator"])),
            read_h0(Path(spec["native"])),
            elm_shift_records=args.elm_shift_records,
            align_by_record_label=not args.no_align_by_record_label,
        )
        all_events.extend(top_event_rows(case, emu, elm))

    _write_csv(args.output_dir / "hourly_metrics.csv", all_metrics)
    _write_csv(args.output_dir / "daily_mean_metrics.csv", all_daily_metrics)
    if all_events:
        _write_csv(args.output_dir / "top_tsoi_l1_difference_hours.csv", all_events)
    write_summary(
        args.output_dir / "summary.txt",
        all_metrics,
        peak_hours,
        elm_shift_records=args.elm_shift_records,
        align_by_record_label=not args.no_align_by_record_label,
    )

    print(f"Wrote {args.output_dir / 'hourly_metrics.csv'}")
    print(f"Wrote {args.output_dir / 'daily_mean_metrics.csv'}")
    print(f"Wrote {args.output_dir / 'top_tsoi_l1_difference_hours.csv'}")
    print(f"Wrote {args.output_dir / 'summary.txt'}")
    for path in plot_paths:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
