"""Regional meteorological forcing loaders and interpolation helpers."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

try:
    import torch
except ImportError as exc:  # pragma: no cover
    raise SystemExit("PyTorch is required. Install torch first.") from exc

from .coupled_surface_hydrology_soiltemp import saturation_vapor_pressure_pa
from .regional_time import _NOLEAP_MONTH_LENGTHS

_MET_FILE_TYPES: dict[str, list[str]] = {
    "TPQWL": ["TBOT", "QBOT", "PSRF", "FLDS", "WIND"],
    "Prec":  ["PRECTmms"],
    "Solr":  ["FSDS"],
}
# Canonical 7-variable order kept throughout the regional code.
_MET_VAR_ORDER = ["TBOT", "QBOT", "PSRF", "FLDS", "WIND", "PRECTmms", "FSDS"]
_MET_VAR_IDX   = {v: i for i, v in enumerate(_MET_VAR_ORDER)}

_TRENDY_START_YEAR = 1901
_TRENDY_END_YEAR = 2024


def _detect_regional_met_format(met_dir: Path, requested: str = "auto") -> str:
    """Resolve regional forcing layout."""
    requested = str(requested).lower()
    if requested != "auto":
        return requested
    if _site_met_file_exists(Path(met_dir)):
        return "site"
    if list(Path(met_dir).glob("clmforc.Daymet_ERA5_TESSFA.4km.TPQWL.*.nc")):
        return "tessfa"
    if list(Path(met_dir).glob("*_TBOT_*_z*.nc")) or list(Path(met_dir).glob("*_TBOT_*.nc")):
        return "trendy"
    raise FileNotFoundError(
        f"Could not detect regional met format under {met_dir}. "
        "Expected site all_hourly.nc, TESSFA monthly files, or TRENDY "
        "single-variable time-series files."
    )


def _site_met_file(met_dir: Path) -> Path:
    """Return the CLM1PT/site forcing file path for a site forcing directory."""
    met_dir = Path(met_dir)
    if met_dir.is_file():
        return met_dir
    path = met_dir / "all_hourly.nc"
    if path.exists():
        return path
    matches = sorted(met_dir.glob("*all_hourly*.nc"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"No site forcing all_hourly.nc found under {met_dir}")


def _site_met_file_exists(met_dir: Path) -> bool:
    try:
        _site_met_file(met_dir)
    except FileNotFoundError:
        return False
    return True


def _trendy_met_file(met_dir: Path, var_name: str) -> Path:
    matches = sorted(Path(met_dir).glob(f"*_{var_name}_*_z*.nc"))
    if not matches:
        matches = sorted(Path(met_dir).glob(f"*_{var_name}_*.nc"))
    if not matches:
        raise FileNotFoundError(f"No TRENDY forcing file for {var_name} under {met_dir}")
    if len(matches) > 1:
        print(f"  [TRENDY met] using {matches[0].name} for {var_name} ({len(matches)} matches)")
    return matches[0]


def load_met_month_cpu(
    met_dir: Path,
    year: int,
    month: int,
    grid_row: np.ndarray,   # (ncell,) gridYID → lat index
    grid_col: np.ndarray,   # (ncell,) gridXID → lon index
    *,
    cache_dir: Path | None = None,
    met_index: np.ndarray | None = None,
    met_format: str = "auto",
    calendar: str = "gregorian",
    pad_steps: int = 0,
) -> np.ndarray:
    """Load one month of met forcing for the active cells.

    Returns ``float32`` array of shape ``(ntimes, ncell, 7)`` with variables
    in the order defined by ``_MET_VAR_ORDER``.
    """
    met_format = _detect_regional_met_format(met_dir, met_format)
    if met_format == "site":
        return load_site_met_month_cpu(
            met_dir,
            year,
            month,
            np.arange(np.asarray(grid_row).size, dtype=np.int64) if met_index is None else met_index,
            cache_dir=cache_dir,
            calendar=calendar,
        )
    if met_format == "trendy":
        return load_trendy_met_month_cpu(
            met_dir,
            year,
            month,
            np.arange(np.asarray(grid_row).size, dtype=np.int64) if met_index is None else met_index,
            cache_dir=cache_dir,
            calendar=calendar,
            pad_steps=pad_steps,
        )
    if met_format != "tessfa":
        raise ValueError(f"Unsupported regional met format: {met_format}")

    ym = f"{year:04d}-{month:02d}"
    grid_row = np.asarray(grid_row, dtype=np.int64)
    grid_col = np.asarray(grid_col, dtype=np.int64)
    ncell = int(grid_row.size)
    single_forcing_point = (
        ncell > 0
        and np.all(grid_row == grid_row[0])
        and np.all(grid_col == grid_col[0])
    )
    cache_path: Path | None = None
    if single_forcing_point and cache_dir is not None:
        cache_path = (
            Path(cache_dir)
            / f"met_cell_r{int(grid_row[0]):04d}_c{int(grid_col[0]):04d}_{ym}_v1.npy"
        )
        if cache_path.exists():
            cached = np.load(cache_path).astype(np.float32, copy=False)
            if ncell == 1:
                return cached
            return np.broadcast_to(
                cached,
                (cached.shape[0], ncell, cached.shape[2]),
            ).copy()

    parts: list[np.ndarray] = []
    for ftype, varnames in _MET_FILE_TYPES.items():
        fpath = met_dir / f"clmforc.Daymet_ERA5_TESSFA.4km.{ftype}.{ym}.nc"
        with Dataset(fpath) as ds:
            for vname in varnames:
                var = ds.variables[vname]
                if single_forcing_point:
                    # Single-point extraction: avoid reading the full regional
                    # met grid just to clone one point across a synthetic grid.
                    point = np.asarray(
                        var[:, int(grid_row[0]), int(grid_col[0])],
                        dtype=np.float32,
                    )
                    parts.append(point[:, None])                              # (nt, 1)
                else:
                    data = np.asarray(var[:], dtype=np.float32)               # (nt, nlat, nlon)
                    parts.append(data[:, grid_row, grid_col])                 # (nt, ncell)
    result = np.stack(parts, axis=-1).astype(np.float32, copy=False)  # (nt, ncell, 7)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = cache_path.with_name(cache_path.name + ".tmp.npy")
        np.save(tmp_path, result)
        tmp_path.replace(cache_path)
    if single_forcing_point and ncell > 1:
        result = np.broadcast_to(result, (result.shape[0], ncell, result.shape[2])).copy()
    return result


def _site_rh_to_specific_humidity(
    tbot: np.ndarray,
    rh: np.ndarray,
    psrf: np.ndarray,
) -> np.ndarray:
    """Convert site relative humidity to specific humidity [kg kg-1]."""
    rh_frac = np.asarray(rh, dtype=np.float64)
    if np.nanmax(rh_frac) > 1.5:
        rh_frac = rh_frac * 0.01
    rh_frac = np.clip(rh_frac, 0.0, 1.0)
    es = saturation_vapor_pressure_pa(np.asarray(tbot, dtype=np.float64))
    e = np.minimum(rh_frac * es, np.asarray(psrf, dtype=np.float64) * 0.99)
    epsilon = 0.622
    q = epsilon * e / np.maximum(np.asarray(psrf, dtype=np.float64) - (1.0 - epsilon) * e, 1.0)
    return np.asarray(np.clip(q, 0.0, 1.0), dtype=np.float32)


def load_site_met_month_cpu(
    met_dir: Path,
    year: int,
    month: int,
    met_index: np.ndarray,
    *,
    cache_dir: Path | None = None,
    calendar: str = "noleap",
) -> np.ndarray:
    """Load one no-leap month from a CLM1PT/site ``all_hourly.nc`` file.

    Site files used by OLMT/CPL bypass store one hourly stream with dimensions
    ``(gridcell, DTIME)`` and variables such as ``TBOT``, ``RH``, ``WIND``,
    ``FSDS``, ``FLDS``, ``PSRF``, and ``PRECTmms``.  The replay expects the
    canonical regional order with specific humidity in the ``QBOT`` slot, so
    relative humidity is converted when a native ``QBOT`` variable is absent.
    """
    if calendar != "noleap":
        raise ValueError("Site all_hourly forcing is treated as no-leap; pass --regional-calendar noleap")
    year_i = int(year)
    month_i = int(month)
    met_index = np.asarray(met_index, dtype=np.int64)
    ncell = int(met_index.size)
    fpath = _site_met_file(met_dir)

    with Dataset(fpath) as ds:
        start_year = int(np.asarray(ds.variables.get("start_year", [year_i])[:]).reshape(-1)[0])
        end_year = int(np.asarray(ds.variables.get("end_year", [year_i])[:]).reshape(-1)[0])
        if year_i < start_year or year_i > end_year:
            raise ValueError(f"Site forcing {fpath} supports years {start_year}-{end_year}, got {year_i}")

        nmet = int(ds.dimensions["gridcell"].size)
        if ncell > 0 and (met_index.min() < 0 or met_index.max() >= nmet):
            raise IndexError(
                f"Site met index range {int(met_index.min())}..{int(met_index.max())} "
                f"is outside forcing gridcell dimension 0..{nmet - 1}"
            )

        dtime_sample = np.asarray(
            ds.variables["DTIME"][: min(1000, int(ds.dimensions["DTIME"].size))],
            dtype=np.float64,
        )
        dt_days = float(np.nanmedian(np.diff(dtime_sample)))
        site_steps_per_day = int(round(1.0 / dt_days))
        if site_steps_per_day <= 0 or not np.isclose(dt_days * site_steps_per_day, 1.0, rtol=0.0, atol=1.0e-4):
            raise ValueError(f"Cannot infer regular site forcing timestep from {fpath}: median DTIME step={dt_days}")
        if site_steps_per_day % 24 != 0:
            raise ValueError(
                f"Site forcing {fpath} has {site_steps_per_day} records/day; "
                "only integer multiples of hourly forcing are currently supported."
            )
        hourly_agg = site_steps_per_day // 24

        cache_path: Path | None = None
        if cache_dir is not None:
            first = int(met_index[0]) if ncell else -1
            last = int(met_index[-1]) if ncell else -1
            cache_path = (
                Path(cache_dir)
                / f"site_met_n{ncell}_i{first}_to_{last}_{year_i:04d}-{month_i:02d}_s{site_steps_per_day}_v2.npy"
            )
            if cache_path.exists():
                return np.load(cache_path).astype(np.float32, copy=False)

        days_before_year = (year_i - start_year) * 365
        days_before_month = sum(_NOLEAP_MONTH_LENGTHS[:month_i - 1])
        t0 = (days_before_year + days_before_month) * site_steps_per_day
        nt_site = _NOLEAP_MONTH_LENGTHS[month_i - 1] * site_steps_per_day
        t_slice = slice(t0, t0 + nt_site)
        if t0 + nt_site > int(ds.dimensions["DTIME"].size):
            raise ValueError(
                f"Site forcing {fpath} does not contain {year_i}-{month_i:02d}; "
                f"needed DTIME slice {t0}:{t0 + nt_site}"
            )

        def read_var(name: str) -> np.ndarray:
            data = np.asarray(ds.variables[name][met_index, t_slice], dtype=np.float32)
            if data.ndim == 1:
                data = data[None, :]
            return data.T  # (nt, ncell)

        tbot = read_var("TBOT")
        psrf = read_var("PSRF")
        if "QBOT" in ds.variables:
            qbot = read_var("QBOT")
        elif "RH" in ds.variables:
            qbot = _site_rh_to_specific_humidity(tbot, read_var("RH"), psrf)
        else:
            raise KeyError(f"Site forcing {fpath} needs either QBOT or RH")
        parts = [
            tbot,
            qbot,
            psrf,
            read_var("FLDS"),
            read_var("WIND"),
            read_var("PRECTmms"),
            read_var("FSDS"),
        ]

    result = np.stack(parts, axis=-1).astype(np.float32, copy=False)
    if hourly_agg > 1:
        nt_site, ncell_result, nvar = result.shape
        result = result.reshape(nt_site // hourly_agg, hourly_agg, ncell_result, nvar).mean(axis=1)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = cache_path.with_name(cache_path.name + ".tmp.npy")
        np.save(tmp_path, result)
        tmp_path.replace(cache_path)
    return result


def load_trendy_met_month_cpu(
    met_dir: Path,
    year: int,
    month: int,
    met_index: np.ndarray,
    *,
    cache_dir: Path | None = None,
    calendar: str = "noleap",
    pad_steps: int = 0,
) -> np.ndarray:
    """Load one no-leap month from TRENDY 0.5-degree compressed land forcing.

    TRENDY forcing files are one variable per file with shape ``(n, DTIME)``.
    ``DTIME`` is 6-hourly from 1901-01-01 through 2024 on a no-leap calendar.
    """
    if calendar != "noleap":
        raise ValueError("TRENDY forcing uses a no-leap calendar; pass --regional-calendar noleap")
    year_i = int(year)
    month_i = int(month)
    if year_i < _TRENDY_START_YEAR or year_i > _TRENDY_END_YEAR:
        raise ValueError(
            f"TRENDY forcing supports years {_TRENDY_START_YEAR}-{_TRENDY_END_YEAR}, got {year_i}"
        )
    met_index = np.asarray(met_index, dtype=np.int64)
    ncell = int(met_index.size)
    pad_steps_i = max(int(pad_steps), 0)
    cache_path: Path | None = None
    if cache_dir is not None:
        first = int(met_index[0]) if ncell else -1
        last = int(met_index[-1]) if ncell else -1
        cache_path = (
            Path(cache_dir)
            / f"trendy_met_n{ncell}_i{first}_to_{last}_{year_i:04d}-{month_i:02d}_pad{pad_steps_i}_v2.npy"
        )
        if cache_path.exists():
            return np.load(cache_path).astype(np.float32, copy=False)

    steps_per_day = 4
    days_before_year = (year_i - _TRENDY_START_YEAR) * 365
    days_before_month = sum(_NOLEAP_MONTH_LENGTHS[:month_i - 1])
    t0 = (days_before_year + days_before_month) * steps_per_day
    nt = _NOLEAP_MONTH_LENGTHS[month_i - 1] * steps_per_day
    t_start = max(t0 - pad_steps_i, 0)
    t_end = t0 + nt + pad_steps_i

    parts: list[np.ndarray] = []
    full_index = ncell > 0 and np.array_equal(met_index, np.arange(ncell, dtype=np.int64))
    for vname in _MET_VAR_ORDER:
        fpath = _trendy_met_file(met_dir, vname)
        with Dataset(fpath) as ds:
            var = ds.variables[vname]
            t_end_clipped = min(t_end, int(var.shape[1]))
            t_slice = slice(t_start, t_end_clipped)
            if ncell == 1:
                data = np.asarray(var[int(met_index[0]), t_slice], dtype=np.float32)[None, :]
            elif full_index and ncell == var.shape[0]:
                data = np.asarray(var[:, t_slice], dtype=np.float32)
            else:
                # NetCDF4 fancy indexing over both axes can be very slow.  The
                # monthly time slice is modest, so read all land points once.
                data_all = np.asarray(var[:, t_slice], dtype=np.float32)
                data = data_all[met_index, :]
            if pad_steps_i:
                missing_left = max(pad_steps_i - t0, 0)
                missing_right = max(t_end - int(var.shape[1]), 0)
                if missing_left:
                    data = np.concatenate(
                        [np.repeat(data[:, :1], missing_left, axis=1), data],
                        axis=1,
                    )
                if missing_right:
                    data = np.concatenate(
                        [data, np.repeat(data[:, -1:], missing_right, axis=1)],
                        axis=1,
                    )
            parts.append(data.T)  # (nt, ncell)

    result = np.stack(parts, axis=-1).astype(np.float32, copy=False)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = cache_path.with_name(cache_path.name + ".tmp.npy")
        np.save(tmp_path, result)
        tmp_path.replace(cache_path)
    return result


def interp_met_to_1h_device(
    data: "torch.Tensor",  # (nstep, ncell, nvars) – already on device
    input_step_hours: int,
    *,
    elm_stream_methods: bool = False,
    padding_steps: int = 0,
) -> "torch.Tensor":
    """Linearly interpolate regularly spaced met data to hourly on device.

    Assumes each input value represents the start of its interval.
    Returns shape ``(nstep * input_step_hours, ncell, nvars)``.
    """
    step = int(input_step_hours)
    if step <= 0:
        raise ValueError("input_step_hours must be positive")
    pad = max(int(padding_steps), 0)
    if elm_stream_methods:
        return interp_met_to_1h_elm_stream_device(data, step, padding_steps=pad)
    if step == 1:
        return data[pad : data.shape[0] - pad] if pad else data
    # Duplicate last step so we can always look one step ahead.
    x1 = torch.cat([data[1:], data[-1:]], dim=0)
    diff = (x1 - data).unsqueeze(1)
    x0   = data.unsqueeze(1)
    wts  = (
        torch.arange(step, device=data.device, dtype=data.dtype)
        / torch.as_tensor(float(step), device=data.device, dtype=data.dtype)
    ).reshape(1, step, 1, 1)
    out = (x0 + wts * diff).reshape(data.shape[0] * step, data.shape[1], data.shape[2])
    return out


def interp_met_to_1h_elm_stream_device(
    data: "torch.Tensor",
    input_step_hours: int,
    *,
    padding_steps: int = 0,
) -> "torch.Tensor":
    """Interpolate regional met with ELM/DATM stream methods.

    For ELM cpl-bypass streams, solar is handled later with coszen
    redistribution, precipitation uses nearest-neighbor interval values, and
    state variables use the native cpl-bypass linear interpolation convention.
    With 6-hour TRENDY forcing, ELM's linear variables are evaluated at the
    model timestamp ending the current step, using the stream record just before
    the timestamp and the following record. A padded record on each side lets
    month boundaries match native ELM.
    """
    step = int(input_step_hours)
    if step <= 0:
        raise ValueError("input_step_hours must be positive")
    pad = max(int(padding_steps), 0)
    if step == 1:
        return data[pad : data.shape[0] - pad] if pad else data
    if pad > 0 and data.shape[0] <= 2 * pad:
        raise ValueError(
            f"Cannot drop {pad} padded met step(s) from data with {data.shape[0]} steps"
        )
    core = data[pad : data.shape[0] - pad] if pad else data
    ncore = int(core.shape[0])
    nout = ncore * step
    if nout == 0:
        return core.repeat_interleave(step, dim=0)

    out = core.repeat_interleave(step, dim=0).clone()

    linear_names = ("TBOT", "QBOT", "PSRF", "FLDS", "WIND")
    linear_idx = [_MET_VAR_IDX[name] for name in linear_names]
    if pad > 0:
        source = data[:, :, linear_idx]
        if 24 % step != 0:
            raise ValueError(
                "ELM stream interpolation currently requires input_step_hours "
                f"to evenly divide 24, got {step}"
            )
        label_hour = torch.arange(1, nout + 1, device=data.device, dtype=torch.long)
        tod_hour = torch.remainder(label_hour, 24)
        day_index = torch.div(label_hour, 24, rounding_mode="floor")
        source_per_day = 24 // step
        # Mirrors E3SM_ORNL components/elm/src/cpl/lnd_import_export.F90:
        # tindex advances after the stream center is used at tod=npf/2.
        source_offset = torch.div(
            tod_hour + max(step // 2 - 1, 0),
            step,
            rounding_mode="floor",
        )
        lower = (day_index * source_per_day + source_offset).clamp(0, source.shape[0] - 2)
        phase = torch.remainder(
            tod_hour.to(dtype=data.dtype)
            + torch.as_tensor(24.0 - 0.5 * float(step), device=data.device, dtype=data.dtype),
            torch.as_tensor(float(step), device=data.device, dtype=data.dtype),
        )
        wt1 = (1.0 - phase / float(step)).reshape(nout, 1, 1)
        linear = source[lower] * wt1 + source[lower + 1] * (1.0 - wt1)
    else:
        # Fallback for unpadded non-hourly forcing. This preserves the old
        # data availability requirements while still using per-variable methods.
        x1 = torch.cat([core[1:, :, linear_idx], core[-1:, :, linear_idx]], dim=0)
        diff = (x1 - core[:, :, linear_idx]).unsqueeze(1)
        x0 = core[:, :, linear_idx].unsqueeze(1)
        wts = (
            torch.arange(step, device=data.device, dtype=data.dtype)
            / torch.as_tensor(float(step), device=data.device, dtype=data.dtype)
        ).reshape(1, step, 1, 1)
        linear = (x0 + wts * diff).reshape(nout, data.shape[1], len(linear_idx))
    out[:, :, linear_idx] = linear

    # DATM stream metadata uses nearest-neighbor precipitation. Repeating each
    # interval value preserves native daily totals for cpl-bypass PRECTmms.
    out[:, :, _MET_VAR_IDX["PRECTmms"]] = core[
        :, :, _MET_VAR_IDX["PRECTmms"]
    ].repeat_interleave(step, dim=0)
    return out


def redistribute_shortwave_to_1h_device(
    data: "torch.Tensor",  # (nstep, ncell, nvars) – already on device
    coszen_hourly: "torch.Tensor",  # (n1h, ncell), signed like native ELM
    input_step_hours: int,
    *,
    n1h: int | None = None,
) -> "torch.Tensor":
    """Redistribute interval-mean FSDS to hourly values using native ELM coszen weights."""
    step = int(input_step_hours)
    if step <= 1:
        hourly = torch.clamp(data[:, :, _MET_VAR_IDX["FSDS"]], min=0.0)
        return hourly if n1h is None else hourly[: int(n1h)]
    nout = int(coszen_hourly.shape[0] if n1h is None else n1h)
    ninterval = int(math.ceil(nout / float(step)))
    fsds = torch.clamp(data[:ninterval, :, _MET_VAR_IDX["FSDS"]], min=0.0)
    if fsds.shape[0] < ninterval:
        raise ValueError(
            f"Need {ninterval} shortwave intervals for {nout} hourly steps, "
            f"but met input has {fsds.shape[0]}"
        )
    cos = coszen_hourly[: ninterval * step]
    if cos.shape[0] < ninterval * step:
        pad = cos[-1:].expand(ninterval * step - cos.shape[0], -1)
        cos = torch.cat([cos, pad], dim=0)
    cos = cos.reshape(ninterval, step, cos.shape[1])
    cos_for_avg = torch.clamp(cos, min=0.001)
    avgcos = torch.clamp(cos_for_avg.mean(dim=1), min=1.0e-12)
    weights = torch.where(
        cos > 0.001,
        torch.clamp(cos_for_avg / avgcos[:, None, :], max=10.0),
        torch.zeros_like(cos),
    )
    return (fsds[:, None, :] * weights).reshape(ninterval * step, fsds.shape[1])[:nout]


def interp_3h_to_1h_device(
    data_3h: "torch.Tensor",
) -> "torch.Tensor":
    """Backward-compatible wrapper for existing callers."""
    return interp_met_to_1h_device(data_3h, 3)
