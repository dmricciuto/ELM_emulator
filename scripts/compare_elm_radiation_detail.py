#!/usr/bin/env python
"""Compare emulator and native ELM shortwave radiation diagnostics.

The h0 albedo variables are model diagnostic states, while reflected-flux
ratios are apparent albedos over a finite averaging period.  This script
reports both so mismatches in native history conventions are not confused with
true shortwave budget differences.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from netCDF4 import Dataset, num2date


FLUX_VARS = (
    "FSDS",
    "FSA",
    "FSR",
    "SABG",
    "SABV",
    "FGR",
    "TG",
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


def _infer_case_name(run_dir: Path) -> str:
    run_dir = Path(run_dir)
    return run_dir.parent.name if run_dir.name == "run" else run_dir.name


def _history_files(run_dir: Path, case_name: str | None, years: list[int]) -> list[Path]:
    run_dir = Path(run_dir)
    case = case_name or _infer_case_name(run_dir)
    paths = [
        run_dir / f"{case}.elm.h0.{year:04d}-01-01-00000.nc"
        for year in years
    ]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing h0 file(s): " + ", ".join(missing))
    return paths


def _filled(values) -> np.ndarray:
    return np.ma.filled(values, np.nan).astype(np.float64, copy=False)


def _reduce_to_time(arr: np.ndarray, dims: tuple[str, ...]) -> np.ndarray:
    if "time" not in dims:
        raise ValueError(f"variable has no time dimension: {dims}")
    time_axis = dims.index("time")
    if time_axis != 0:
        arr = np.moveaxis(arr, time_axis, 0)
        dims = ("time",) + tuple(dim for idx, dim in enumerate(dims) if idx != time_axis)
    while arr.ndim > 1:
        arr = np.nanmean(arr, axis=-1)
    return arr


def _read_scalar(files: list[Path], name: str) -> np.ndarray | None:
    rows = []
    for path in files:
        with Dataset(path) as ds:
            if name not in ds.variables:
                return None
            var = ds.variables[name]
            rows.append(_reduce_to_time(_filled(var[:]), tuple(var.dimensions)))
    return np.concatenate(rows, axis=0)


def _read_numrad(files: list[Path], name: str, band: int) -> np.ndarray | None:
    rows = []
    for path in files:
        with Dataset(path) as ds:
            if name not in ds.variables:
                return None
            var = ds.variables[name]
            dims = tuple(var.dimensions)
            if "numrad" not in dims:
                raise ValueError(f"{name} has no numrad dimension in {path}")
            arr = _filled(var[:])
            arr = np.take(arr, indices=int(band), axis=dims.index("numrad"))
            dims = tuple(dim for dim in dims if dim != "numrad")
            rows.append(_reduce_to_time(arr, dims))
    return np.concatenate(rows, axis=0)


def _read_time(files: list[Path]) -> np.ndarray:
    labels: list[str] = []
    for path in files:
        with Dataset(path) as ds:
            if "time" not in ds.variables:
                raise ValueError(f"{path} has no time coordinate")
            time_var = ds.variables["time"]
            dates = num2date(
                time_var[:],
                time_var.units,
                calendar=getattr(time_var, "calendar", "standard"),
                only_use_cftime_datetimes=False,
                only_use_python_datetimes=False,
            )
            labels.extend(f"{int(date.year):04d}-{int(date.month):02d}-{int(date.day):02d}" for date in dates)
    return np.asarray(labels)


def _safe_ratio(num: np.ndarray | None, den: np.ndarray | None, min_den: float) -> np.ndarray | None:
    if num is None or den is None:
        return None
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(np.abs(den) >= min_den, num / den, np.nan)


def _metric_row(case: str, metric: str, emu: np.ndarray | None, elm: np.ndarray | None) -> dict[str, object]:
    row: dict[str, object] = {
        "case": case,
        "metric": metric,
        "n": 0,
        "elm_mean": np.nan,
        "emulator_mean": np.nan,
        "bias_emulator_minus_elm": np.nan,
        "mae": np.nan,
        "rmse": np.nan,
        "corr": np.nan,
    }
    if emu is None or elm is None:
        return row
    n = min(emu.shape[0], elm.shape[0])
    lhs = emu[:n]
    rhs = elm[:n]
    valid = np.isfinite(lhs) & np.isfinite(rhs)
    if not np.any(valid):
        return row
    diff = lhs[valid] - rhs[valid]
    row["n"] = int(valid.sum())
    row["elm_mean"] = float(np.nanmean(rhs[valid]))
    row["emulator_mean"] = float(np.nanmean(lhs[valid]))
    row["bias_emulator_minus_elm"] = float(np.nanmean(diff))
    row["mae"] = float(np.nanmean(np.abs(diff)))
    row["rmse"] = float(np.sqrt(np.nanmean(diff * diff)))
    if valid.sum() >= 2 and np.nanstd(lhs[valid]) > 0.0 and np.nanstd(rhs[valid]) > 0.0:
        row["corr"] = float(np.corrcoef(lhs[valid], rhs[valid])[0, 1])
    return row


def _residual_row(case: str, metric: str, value: np.ndarray | None) -> dict[str, object]:
    row: dict[str, object] = {
        "case": case,
        "metric": metric,
        "n": 0,
        "mean": np.nan,
        "mae": np.nan,
        "rmse": np.nan,
        "max_abs": np.nan,
    }
    if value is None:
        return row
    valid = np.isfinite(value)
    if not np.any(valid):
        return row
    v = value[valid]
    row["n"] = int(valid.sum())
    row["mean"] = float(np.nanmean(v))
    row["mae"] = float(np.nanmean(np.abs(v)))
    row["rmse"] = float(np.sqrt(np.nanmean(v * v)))
    row["max_abs"] = float(np.nanmax(np.abs(v)))
    return row


def _load_case(run_dir: Path, case_name: str | None, years: list[int]) -> tuple[dict[str, np.ndarray | None], np.ndarray]:
    files = _history_files(run_dir, case_name, years)
    data: dict[str, np.ndarray | None] = {name: _read_scalar(files, name) for name in FLUX_VARS}
    for name in NUMRAD_VARS:
        data[f"{name}_VIS"] = _read_numrad(files, name, 0)
        data[f"{name}_NIR"] = _read_numrad(files, name, 1)
    return data, _read_time(files)


def _take_time(data: dict[str, np.ndarray | None], indices: list[int]) -> dict[str, np.ndarray | None]:
    out: dict[str, np.ndarray | None] = {}
    for key, value in data.items():
        out[key] = None if value is None else value[np.asarray(indices, dtype=np.int64)]
    return out


def _align_by_time(
    emu: dict[str, np.ndarray | None],
    elm: dict[str, np.ndarray | None],
    emu_time: np.ndarray,
    elm_time: np.ndarray,
) -> tuple[dict[str, np.ndarray | None], dict[str, np.ndarray | None], list[str]]:
    emu_lookup = {str(label): idx for idx, label in enumerate(emu_time)}
    elm_lookup = {str(label): idx for idx, label in enumerate(elm_time)}
    common = [str(label) for label in elm_time if str(label) in emu_lookup]
    if not common:
        raise ValueError("No overlapping h0 time coordinates between emulator and native files")
    return (
        _take_time(emu, [emu_lookup[label] for label in common]),
        _take_time(elm, [elm_lookup[label] for label in common]),
        common,
    )


def _add_apparent_albedos(data: dict[str, np.ndarray | None], min_sw: float) -> None:
    data["APP_ALB_TOTAL"] = _safe_ratio(data.get("FSR"), data.get("FSDS"), min_sw)
    data["APP_ALBD_VIS"] = _safe_ratio(data.get("FSRVD"), data.get("FSDSVD"), min_sw)
    data["APP_ALBI_VIS"] = _safe_ratio(data.get("FSRVI"), data.get("FSDSVI"), min_sw)
    data["APP_ALBD_NIR"] = _safe_ratio(data.get("FSRND"), data.get("FSDSND"), min_sw)
    data["APP_ALBI_NIR"] = _safe_ratio(data.get("FSRNI"), data.get("FSDSNI"), min_sw)


def _sum_existing(data: dict[str, np.ndarray | None], names: tuple[str, ...]) -> np.ndarray | None:
    arrays = [data.get(name) for name in names]
    if any(arr is None for arr in arrays):
        return None
    return np.sum(np.stack([arr for arr in arrays if arr is not None], axis=0), axis=0)


def compare_case(
    *,
    case: str,
    emulator_run_dir: Path,
    native_run_dir: Path,
    emulator_case_name: str | None,
    native_case_name: str | None,
    years: list[int],
    min_sw: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    emu_raw, emu_time = _load_case(emulator_run_dir, emulator_case_name, years)
    elm_raw, elm_time = _load_case(native_run_dir, native_case_name, years)
    emu, elm, common_time = _align_by_time(emu_raw, elm_raw, emu_time, elm_time)
    print(
        f"Aligned {case}: {len(common_time)} common records "
        f"({common_time[0]} to {common_time[-1]})"
    )
    _add_apparent_albedos(emu, min_sw)
    _add_apparent_albedos(elm, min_sw)

    metrics: list[dict[str, object]] = []
    for name in FLUX_VARS:
        metrics.append(_metric_row(case, name, emu.get(name), elm.get(name)))
    for name in (
        "ALBD_VIS",
        "ALBD_NIR",
        "ALBI_VIS",
        "ALBI_NIR",
        "ALBGRD_VIS",
        "ALBGRD_NIR",
        "ALBGRI_VIS",
        "ALBGRI_NIR",
        "APP_ALB_TOTAL",
        "APP_ALBD_VIS",
        "APP_ALBD_NIR",
        "APP_ALBI_VIS",
        "APP_ALBI_NIR",
    ):
        metrics.append(_metric_row(case, name, emu.get(name), elm.get(name)))

    residuals: list[dict[str, object]] = []
    for label, data in (("emulator", emu), ("native", elm)):
        fsds_parts = _sum_existing(data, ("FSDSVD", "FSDSVI", "FSDSND", "FSDSNI"))
        fsr_parts = _sum_existing(data, ("FSRVD", "FSRVI", "FSRND", "FSRNI"))
        residuals.append(_residual_row(case, f"{label}: sum(FSDS bands)-FSDS", None if fsds_parts is None else fsds_parts - data.get("FSDS")))
        residuals.append(_residual_row(case, f"{label}: sum(FSR bands)-FSR", None if fsr_parts is None else fsr_parts - data.get("FSR")))
        residuals.append(_residual_row(case, f"{label}: FSDS-FSR-FSA", None if fsr_parts is None else data.get("FSDS") - data.get("FSR") - data.get("FSA")))
        if data.get("SABG") is not None and data.get("SABV") is not None:
            sab = data["SABG"] + data["SABV"]  # type: ignore[operator]
            residuals.append(_residual_row(case, f"{label}: SABG+SABV-FSA", sab - data.get("FSA")))  # type: ignore[operator]
            residuals.append(_residual_row(case, f"{label}: SABG/FSA", _safe_ratio(data.get("SABG"), data.get("FSA"), min_sw)))
        else:
            residuals.append(_residual_row(case, f"{label}: SABG+SABV-FSA", None))
            residuals.append(_residual_row(case, f"{label}: SABG/FSA", None))
    for name in ("ALBD_VIS", "ALBD_NIR", "ALBI_VIS", "ALBI_NIR"):
        apparent = "APP_" + name
        residuals.append(_residual_row(case, f"native: {name}-apparent", None if elm.get(name) is None or elm.get(apparent) is None else elm[name] - elm[apparent]))  # type: ignore[operator]

    return metrics, residuals


def _print_metrics(rows: list[dict[str, object]]) -> None:
    print("\nField comparison: emulator minus native")
    print("case, metric, n, elm_mean, emu_mean, bias, mae, rmse, corr")
    for row in rows:
        print(
            f"{row['case']}, {row['metric']}, {row['n']}, "
            f"{row['elm_mean']:.6g}, {row['emulator_mean']:.6g}, "
            f"{row['bias_emulator_minus_elm']:.6g}, {row['mae']:.6g}, "
            f"{row['rmse']:.6g}, {row['corr']:.4f}"
        )


def _print_residuals(rows: list[dict[str, object]]) -> None:
    print("\nBudget/internal residuals")
    print("case, metric, n, mean, mae, rmse, max_abs")
    for row in rows:
        print(
            f"{row['case']}, {row['metric']}, {row['n']}, "
            f"{row['mean']:.6g}, {row['mae']:.6g}, "
            f"{row['rmse']:.6g}, {row['max_abs']:.6g}"
        )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", default="case", help="Label written to the report.")
    p.add_argument("--emulator-run-dir", type=Path, required=True)
    p.add_argument("--native-run-dir", type=Path, required=True)
    p.add_argument("--emulator-case-name", default=None)
    p.add_argument("--native-case-name", default=None)
    p.add_argument("--years", type=int, nargs="+", default=[2012])
    p.add_argument("--min-sw", type=float, default=1.0, help="Minimum denominator for apparent albedo ratios.")
    p.add_argument("--metrics-csv", type=Path, default=None)
    p.add_argument("--residuals-csv", type=Path, default=None)
    args = p.parse_args()

    metrics, residuals = compare_case(
        case=args.case,
        emulator_run_dir=args.emulator_run_dir,
        native_run_dir=args.native_run_dir,
        emulator_case_name=args.emulator_case_name,
        native_case_name=args.native_case_name,
        years=list(args.years),
        min_sw=float(args.min_sw),
    )
    _print_metrics(metrics)
    _print_residuals(residuals)

    if args.metrics_csv is not None:
        args.metrics_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.metrics_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(metrics[0]))
            writer.writeheader()
            writer.writerows(metrics)
    if args.residuals_csv is not None:
        args.residuals_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.residuals_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(residuals[0]))
            writer.writeheader()
            writer.writerows(residuals)


if __name__ == "__main__":
    main()
