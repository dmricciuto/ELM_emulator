"""Initial state and restart helpers for replay drivers."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

try:
    import torch
except ImportError as exc:  # pragma: no cover
    raise SystemExit("PyTorch is required. Install torch first.") from exc

from .btran_utils import inverse_frac_h2osfc
from .coupled_surface_hydrology_soiltemp import (
    CoupledColumnState,
    CoupledStaticParameters,
    SnowLayerState,
    _snow_layers_from_storage_numpy,
    coupled_state_to_cpu,
    coupled_state_to_device,
    force_scalar_snow_state_torch,
    h2osno_from_depth_and_frac,
)
from .elm_hydrology import elm_soil_grid

_FILL_VALUE_ABS_LIMIT = 1.0e20


def _first_valid_feature(
    features: np.ndarray,
    fmap: dict[str, int],
    name: str,
    fallback: float,
    *,
    lower: float | None = None,
    upper: float | None = None,
) -> float:
    """Return the first finite, non-fill feature value in the replay window."""
    idx = fmap.get(name)
    if idx is None:
        return float(fallback)
    values = np.asarray(features[:, idx], dtype=np.float64)
    valid = np.isfinite(values) & (np.abs(values) < _FILL_VALUE_ABS_LIMIT)
    if lower is not None:
        valid &= values >= lower
    if upper is not None:
        valid &= values <= upper
    if np.any(valid):
        return float(values[np.argmax(valid)])
    return float(fallback)


def tile_state(state: CoupledColumnState, ncell: int, nens: int) -> CoupledColumnState:
    """Broadcast a ``(1, 1, ...)`` state to ``(ncell, nens, ...)``."""

    def tile(arr):
        if arr is None:
            return None
        a = np.asarray(arr, dtype=np.float64)
        return np.broadcast_to(a, (ncell, nens) + a.shape[2:]).copy()

    snow_layers = None
    if state.snow_layers is not None:
        sl = state.snow_layers
        sl_snw_rds = getattr(sl, "snw_rds", None)
        snow_layers = SnowLayerState(
            snl=tile(sl.snl),
            dz=tile(sl.dz),
            h2osoi_ice=tile(sl.h2osoi_ice),
            h2osoi_liq=tile(sl.h2osoi_liq),
            t_snow=tile(sl.t_snow),
            snw_rds=tile(sl_snw_rds) if sl_snw_rds is not None else None,
        )

    return CoupledColumnState(
        h2osoi_liq=tile(state.h2osoi_liq),
        h2osoi_ice=tile(state.h2osoi_ice),
        h2ocan=tile(state.h2ocan),
        h2osfc=tile(state.h2osfc),
        frac_h2osfc=tile(state.frac_h2osfc),
        frac_sno=tile(state.frac_sno),
        zwt=tile(state.zwt),
        t_soisno=tile(state.t_soisno),
        wa=tile(state.wa),
        t_h2osfc=tile(state.t_h2osfc),
        t10=tile(state.t10),
        t10_history=tile(state.t10_history),
        h2osno=tile(state.h2osno),
        snow_depth=tile(state.snow_depth),
        snow_t=tile(state.snow_t),
        snow_layers=snow_layers,
        int_snow=tile(state.int_snow),
    )


def build_initial_state_1x1(
    feature_names: list[str],
    features: np.ndarray,
    static: CoupledStaticParameters,
    dz: np.ndarray,
) -> CoupledColumnState:
    """Build a ``(1, 1, nlev)`` initial state from the first training row."""
    fmap = {name: i for i, name in enumerate(feature_names)}
    nlev = dz.size
    n_obs = min(8, nlev)

    watsat = np.asarray(static.hydrology_numpy.watsat, dtype=np.float64)
    if watsat.size < nlev:
        watsat = np.concatenate([watsat, np.full(nlev - watsat.size, watsat[-1], dtype=np.float64)])
    watsat = watsat[:nlev]

    liqvol = np.empty(nlev, dtype=np.float64)
    eff_por = np.empty(nlev, dtype=np.float64)
    temp = np.full(nlev, 283.15, dtype=np.float64)
    for lev in range(n_obs):
        eff_raw = _first_valid_feature(
            features,
            fmap,
            f"eff_porosity_lev{lev + 1}",
            float(watsat[lev]),
            lower=0.0,
            upper=float(watsat[lev]) + 1.0e-8,
        )
        eff_por[lev] = np.clip(eff_raw, 0.0, watsat[lev])
        liq_raw = _first_valid_feature(
            features,
            fmap,
            f"h2osoi_liqvol_lev{lev + 1}",
            0.5 * float(eff_por[lev]),
            lower=0.0,
            upper=float(watsat[lev]) + 1.0e-8,
        )
        liqvol[lev] = np.clip(liq_raw, 0.0, eff_por[lev])
        temp[lev] = _first_valid_feature(features, fmap, f"t_soisno_lev{lev + 1}", 283.15, lower=150.0, upper=350.0)
    if nlev > n_obs:
        liqvol[n_obs:] = liqvol[n_obs - 1]
        eff_por[n_obs:] = eff_por[n_obs - 1]

    frozen = temp < (273.15 - 0.5)
    vol_ice_eff = np.maximum(np.minimum(watsat - eff_por, watsat - liqvol), 0.0)
    vol_ice = np.where(frozen, vol_ice_eff, 0.0)

    t10_value = _first_valid_feature(features, fmap, "t10", 283.15, lower=150.0, upper=350.0)
    t10 = np.array([[t10_value]], dtype=np.float64)
    snow_depth0 = np.array([[_first_valid_feature(features, fmap, "snow_depth", 0.0, lower=0.0)]], dtype=np.float64)
    frac_sno0 = np.array([[_first_valid_feature(features, fmap, "frac_sno", 0.0, lower=0.0, upper=1.0)]], dtype=np.float64)
    h2osno0 = h2osno_from_depth_and_frac(snow_depth0, frac_sno0)
    frac_h2osfc0 = np.array([[_first_valid_feature(features, fmap, "frac_h2osfc", 0.0, lower=0.0, upper=1.0)]], dtype=np.float64)
    h2osfc_value = _first_valid_feature(features, fmap, "h2osfc", np.nan, lower=0.0)
    if not np.isfinite(h2osfc_value):
        h2osfc_value = float(inverse_frac_h2osfc(frac_h2osfc0)[0, 0])

    return CoupledColumnState(
        h2osoi_liq=(liqvol[None, None, :] * dz[None, None, :] * 1000.0),
        h2osoi_ice=(vol_ice[None, None, :] * dz[None, None, :] * 917.0),
        h2ocan=np.array([[_first_valid_feature(features, fmap, "h2ocan", 0.0, lower=0.0)]], dtype=np.float64),
        h2osfc=np.array([[h2osfc_value]], dtype=np.float64),
        frac_h2osfc=frac_h2osfc0,
        frac_sno=frac_sno0,
        zwt=np.array([[_first_valid_feature(features, fmap, "zwt_col", 8.0, lower=0.0)]], dtype=np.float64),
        t_soisno=temp[None, None, :],
        t_h2osfc=np.array([[_first_valid_feature(features, fmap, "t_h2osfc", float(temp[0]), lower=150.0, upper=350.0)]], dtype=np.float64),
        t10=t10,
        t10_history=None,
        h2osno=h2osno0,
        snow_depth=snow_depth0,
        snow_layers=_snow_layers_from_storage_numpy(
            h2osno0,
            snow_depth0,
            snow_t=np.full((1, 1, 5), 273.15, dtype=np.float64),
        ),
    )


def regional_initial_state(
    ncell: int,
    nlev: int,
    *,
    nlevbed: int = 10,
    nens: int = 1,
    device: torch.device,
    dtype: torch.dtype,
) -> CoupledColumnState:
    """Native-ELM-style cold-start initial state for regional replay."""
    dz, _, _ = elm_soil_grid(nlevsoi=nlev)
    _, _, zi_hydro = elm_soil_grid(nlevsoi=10)
    active_soil_layers = max(0, min(int(nlevbed), nlev))
    native_elm_wa = 4000.0  # mm, natural non-lake branch in SoilHydrologyType.F90
    native_elm_zwt = (25.0 + float(zi_hydro[-1])) - native_elm_wa / 0.2 / 1000.0
    native_elm_soil_temp = 274.0  # K, natural non-lake branch in ColumnDataType.F90
    native_elm_h2osfc_temp = 274.0
    native_elm_t10 = 273.15 + 20.0
    native_elm_h2osoi_vol = 0.15  # use_hydrstress=.false. natural soil/crop cold start
    h2osoi_ice = np.zeros((ncell, nens, nlev), dtype=np.float64)
    h2osoi_liq = np.zeros((ncell, nens, nlev), dtype=np.float64)
    if active_soil_layers > 0:
        if native_elm_soil_temp <= 273.15:
            h2osoi_ice[..., :active_soil_layers] = (
                native_elm_h2osoi_vol * dz[None, None, :active_soil_layers] * 917.0
            )
        else:
            h2osoi_liq[..., :active_soil_layers] = (
                native_elm_h2osoi_vol * dz[None, None, :active_soil_layers] * 1000.0
            )
    t_soisno = np.full((ncell, nens, nlev), native_elm_soil_temp, dtype=np.float64)
    snow_depth = np.zeros((ncell, nens), dtype=np.float64)
    frac_sno = np.zeros((ncell, nens), dtype=np.float64)
    h2osno = h2osno_from_depth_and_frac(snow_depth, frac_sno)
    state = CoupledColumnState(
        h2osoi_liq=h2osoi_liq,
        h2osoi_ice=h2osoi_ice,
        h2ocan=np.zeros((ncell, nens), dtype=np.float64),
        h2osfc=np.zeros((ncell, nens), dtype=np.float64),
        frac_h2osfc=np.zeros((ncell, nens), dtype=np.float64),
        frac_sno=frac_sno,
        zwt=np.full((ncell, nens), native_elm_zwt, dtype=np.float64),
        t_soisno=t_soisno,
        wa=np.full((ncell, nens), native_elm_wa, dtype=np.float64),
        t_h2osfc=np.full((ncell, nens), native_elm_h2osfc_temp, dtype=np.float64),
        t10=np.full((ncell, nens), native_elm_t10, dtype=np.float64),
        t10_history=None,
        h2osno=h2osno,
        snow_depth=snow_depth,
        snow_layers=_snow_layers_from_storage_numpy(
            h2osno,
            snow_depth,
            snow_t=np.full((ncell, nens, 5), 273.15, dtype=np.float64),
        ),
    )
    return coupled_state_to_device(state, device, dtype)


def prepare_regional_initial_state(
    *,
    surf: dict,
    pft_ids: list[int],
    pft_weights: np.ndarray,
    ncell: int,
    nlev: int,
    nlevbed: int,
    nens: int,
    device: torch.device,
    dtype: torch.dtype,
    restart_in: Path | None,
    start_year: int,
    start_month: int,
    snow_model_enabled: bool,
    snow_scalar_only: bool,
) -> CoupledColumnState:
    """Build or load the initial regional coupled physical state."""
    if restart_in is None:
        state = regional_initial_state(
            ncell,
            nlev,
            nlevbed=nlevbed,
            nens=nens,
            device=device,
            dtype=dtype,
        )
    else:
        state = load_regional_restart(
            restart_in,
            surf=surf,
            pft_ids=pft_ids,
            pft_weights=pft_weights,
            nens=nens,
            expected_start_year=start_year,
            expected_start_month=start_month,
            snow_model_enabled=snow_model_enabled,
            device=device,
            dtype=dtype,
        )
    if snow_model_enabled and snow_scalar_only:
        state = force_scalar_snow_state_torch(state, device=device, dtype=dtype)
    return state


_RESTART_STATE_FIELDS = (
    "h2osoi_liq",
    "h2osoi_ice",
    "h2ocan",
    "h2osfc",
    "frac_h2osfc",
    "frac_sno",
    "zwt",
    "t_soisno",
    "wa",
    "t_h2osfc",
    "t10",
    "t10_history",
    "h2osno",
    "snow_depth",
    "snow_t",
    "int_snow",
)


def _restart_array(value):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _restart_nc_type(arr: np.ndarray) -> str:
    arr = np.asarray(arr)
    if arr.dtype.kind == "f":
        return "f4" if arr.dtype.itemsize <= 4 else "f8"
    if arr.dtype.kind in {"i", "u"}:
        return "i4" if arr.dtype.itemsize <= 4 else "i8"
    raise TypeError(f"Unsupported restart dtype {arr.dtype}")


def _restart_dims_for_array(ds: Dataset, name: str, arr: np.ndarray, *, ncell: int, nens: int) -> tuple[str, ...]:
    dims: list[str] = []
    for axis, size_raw in enumerate(arr.shape):
        size = int(size_raw)
        if axis == 0 and size == ncell:
            dim = "cell"
        elif axis == 1 and size == nens:
            dim = "ensemble"
        else:
            dim = f"{name}_dim{axis}"
        if dim not in ds.dimensions:
            ds.createDimension(dim, size)
        dims.append(dim)
    return tuple(dims)


def _write_restart_var(ds: Dataset, name: str, value, *, ncell: int, nens: int) -> bool:
    arr = _restart_array(value)
    if arr is None:
        return False
    arr = np.asarray(arr)
    dims = _restart_dims_for_array(ds, name, arr, ncell=ncell, nens=nens)
    kwargs = {}
    if arr.dtype.kind == "f" and arr.size > 0:
        kwargs.update(zlib=True, complevel=1, shuffle=True)
    var = ds.createVariable(name, _restart_nc_type(arr), dims, **kwargs)
    var[:] = arr
    return True


def save_regional_restart(
    path: Path,
    *,
    state: CoupledColumnState,
    surf: dict,
    pft_ids: list[int],
    pft_weights: np.ndarray,
    start_year: int,
    start_month: int,
    completed_year: int,
    completed_month: int,
    completed_hours_last_month: int,
    next_year: int,
    next_month: int,
    calendar: str,
    snow_model_enabled: bool,
    dtype: torch.dtype,
    regional_max_hours: int | None,
) -> None:
    """Write a restart file for the regional coupled physical state."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state_cpu = coupled_state_to_cpu(state)
    ncell = int(np.asarray(surf["lat_deg"]).size)
    nens = int(np.asarray(state_cpu.h2osfc).shape[1])
    with Dataset(path, "w") as ds:
        ds.createDimension("cell", ncell)
        ds.createDimension("ensemble", nens)
        ds.createDimension("pft", len(pft_ids))
        ds.title = "Regional surface emulator restart"
        ds.history = f"Created by benchmark_replay.py at {time.strftime('%Y-%m-%d %H:%M:%S %Z')}"
        ds.restart_format_version = 1
        ds.start_year = int(start_year)
        ds.start_month = int(start_month)
        ds.completed_year = int(completed_year)
        ds.completed_month = int(completed_month)
        ds.completed_hours_last_month = int(completed_hours_last_month)
        ds.next_year = int(next_year)
        ds.next_month = int(next_month)
        ds.calendar = str(calendar)
        ds.snow_model_enabled = int(bool(snow_model_enabled))
        ds.torch_dtype = str(dtype).replace("torch.", "")
        ds.regional_max_hours = -1 if regional_max_hours is None else int(regional_max_hours)

        for name, values, units in (
            ("gridcell_index", np.asarray(surf.get("active_idx", np.arange(ncell)), dtype=np.int64), "0-based global gridcell index"),
            ("grid_row", np.asarray(surf["grid_row"], dtype=np.int32), "0-based grid row"),
            ("grid_col", np.asarray(surf["grid_col"], dtype=np.int32), "0-based grid column"),
            ("lat", np.asarray(surf["lat_deg"], dtype=np.float64), "degrees_north"),
            ("lon", np.asarray(surf["lon_deg"], dtype=np.float64), "degrees_east"),
        ):
            var = ds.createVariable(name, _restart_nc_type(values), ("cell",))
            var.long_name = units
            var[:] = values
        pft_var = ds.createVariable("pft_id", "i4", ("pft",))
        pft_var[:] = np.asarray(pft_ids, dtype=np.int32)
        weights = ds.createVariable("pft_weights", "f8", ("pft", "cell"), zlib=True, complevel=1, shuffle=True)
        weights[:] = np.asarray(pft_weights, dtype=np.float64)

        written: list[str] = []
        for field_name in _RESTART_STATE_FIELDS:
            if _write_restart_var(ds, f"state_{field_name}", getattr(state_cpu, field_name), ncell=ncell, nens=nens):
                written.append(f"state_{field_name}")
        snow_layers = state_cpu.snow_layers
        if snow_layers is not None:
            for layer_name in ("snl", "dz", "h2osoi_ice", "h2osoi_liq", "t_snow", "snw_rds"):
                if _write_restart_var(ds, f"snow_layers_{layer_name}", getattr(snow_layers, layer_name, None), ncell=ncell, nens=nens):
                    written.append(f"snow_layers_{layer_name}")
        ds.written_state_variables = ",".join(written)
    print(f"  wrote regional restart {path}")


def load_regional_restart(
    path: Path,
    *,
    surf: dict,
    pft_ids: list[int],
    pft_weights: np.ndarray,
    nens: int,
    expected_start_year: int,
    expected_start_month: int,
    snow_model_enabled: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> CoupledColumnState:
    """Read and validate a regional restart file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Regional restart file does not exist: {path}")
    ncell = int(np.asarray(surf["lat_deg"]).size)

    def required(ds: Dataset, name: str) -> np.ndarray:
        if name not in ds.variables:
            raise KeyError(f"Restart {path} is missing required variable {name}")
        return np.asarray(ds.variables[name][:])

    with Dataset(path) as ds:
        if int(len(ds.dimensions["cell"])) != ncell:
            raise ValueError(f"Restart cell count {len(ds.dimensions['cell'])} does not match current domain {ncell}")
        if int(len(ds.dimensions["ensemble"])) != int(nens):
            raise ValueError(f"Restart ensemble count {len(ds.dimensions['ensemble'])} does not match --regional-ensemble-size {nens}")
        restart_grid = required(ds, "gridcell_index").astype(np.int64)
        current_grid = np.asarray(surf.get("active_idx", np.arange(ncell)), dtype=np.int64)
        if restart_grid.shape != current_grid.shape or not np.array_equal(restart_grid, current_grid):
            raise ValueError("Restart gridcell_index does not match the current regional domain/subset")
        restart_pfts = required(ds, "pft_id").astype(np.int32)
        current_pfts = np.asarray(pft_ids, dtype=np.int32)
        if restart_pfts.shape != current_pfts.shape or not np.array_equal(restart_pfts, current_pfts):
            raise ValueError(
                f"Restart PFT IDs {restart_pfts.tolist()} do not match current PFT IDs {current_pfts.tolist()}"
            )
        restart_weights = required(ds, "pft_weights")
        if restart_weights.shape != np.asarray(pft_weights).shape or not np.allclose(restart_weights, pft_weights, rtol=0.0, atol=1.0e-12):
            raise ValueError("Restart PFT weights do not match the current surfdata/PFT selection")
        restart_snow_enabled = getattr(ds, "snow_model_enabled", None)
        if restart_snow_enabled is not None and bool(int(restart_snow_enabled)) != bool(snow_model_enabled):
            raise ValueError(
                f"Restart snow_model_enabled={int(restart_snow_enabled)} does not match "
                f"current snow setting {int(bool(snow_model_enabled))}"
            )

        def state_var(name: str, *, required_var: bool = False):
            var_name = f"state_{name}"
            if var_name not in ds.variables:
                if required_var:
                    raise KeyError(f"Restart {path} is missing required state variable {var_name}")
                return None
            return np.asarray(ds.variables[var_name][:])

        snow_layers = None
        if "snow_layers_snl" in ds.variables:
            snow_layers = SnowLayerState(
                snl=np.asarray(ds.variables["snow_layers_snl"][:], dtype=np.int32),
                dz=np.asarray(ds.variables["snow_layers_dz"][:]),
                h2osoi_ice=np.asarray(ds.variables["snow_layers_h2osoi_ice"][:]),
                h2osoi_liq=np.asarray(ds.variables["snow_layers_h2osoi_liq"][:]),
                t_snow=np.asarray(ds.variables["snow_layers_t_snow"][:]),
                snw_rds=(
                    np.asarray(ds.variables["snow_layers_snw_rds"][:])
                    if "snow_layers_snw_rds" in ds.variables
                    else None
                ),
            )
        state = CoupledColumnState(
            h2osoi_liq=state_var("h2osoi_liq", required_var=True),
            h2osoi_ice=state_var("h2osoi_ice", required_var=True),
            h2ocan=state_var("h2ocan", required_var=True),
            h2osfc=state_var("h2osfc", required_var=True),
            frac_h2osfc=state_var("frac_h2osfc", required_var=True),
            frac_sno=state_var("frac_sno", required_var=True),
            zwt=state_var("zwt", required_var=True),
            t_soisno=state_var("t_soisno", required_var=True),
            wa=state_var("wa"),
            t_h2osfc=state_var("t_h2osfc"),
            t10=state_var("t10"),
            t10_history=state_var("t10_history"),
            h2osno=state_var("h2osno"),
            snow_depth=state_var("snow_depth"),
            snow_t=state_var("snow_t"),
            snow_layers=snow_layers,
            int_snow=state_var("int_snow"),
        )
        next_year = getattr(ds, "next_year", None)
        next_month = getattr(ds, "next_month", None)
        if next_year is not None and next_month is not None:
            if int(next_year) != int(expected_start_year) or int(next_month) != int(expected_start_month):
                raise ValueError(
                    f"Restart {path} expects resume at {int(next_year)}-{int(next_month):02d}, "
                    f"but this run starts at {int(expected_start_year)}-{int(expected_start_month):02d}"
                )
    msg = f"  loaded regional restart {path}"
    if next_year is not None and next_month is not None:
        msg += f" (next suggested start {int(next_year)}-{int(next_month):02d})"
    print(msg)
    return coupled_state_to_device(state, device, dtype)
