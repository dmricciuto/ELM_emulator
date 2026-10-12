#!/usr/bin/env python3
"""Benchmark the full coupled replay across ncell and ensemble-size sweeps.

Unlike ``coupled_surface_hydrology_soiltemp.py``, which drives the benchmark
with random synthetic state, this script loads the actual training-file forcing
(atmospheric, vegetation, geometry) and replays a contiguous window of
``--nsteps`` timesteps for a ``(ncell, ensemble_size)`` batch in each timed
pass.  This exercises the realistic per-timestep feature-build, MLP inference,
hydrology, and soil-temperature paths with real forcing sequences.

Throughput is reported as ``(ncell * ensemble_size * nsteps) / pass_time``,
matching the units used by the single-step benchmark in
``coupled_surface_hydrology_soiltemp.py``.

Usage example::

    python benchmark_replay.py \\
        --training-file canopyflux_training.nc \\
        --surfdata-file surfdata.nc \\
        --model-dir canopyflux_models \\
        --ncell 64 --ensemble-size 16 --nsteps 168 \\
        --device auto --dtype float32 \\
        --hydrology-backend torch --soil-temperature-backend torch
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from netCDF4 import Dataset, num2date

try:
    import torch
except ImportError as exc:  # pragma: no cover
    raise SystemExit("PyTorch is required. Install torch first.") from exc

from .coupled_surface_hydrology_soiltemp import (
    AtmosphericForcingInputs,
    CoupledColumnState,
    CoupledStaticParameters,
    MAX_SNOW_LAYERS,
    SnowLayerState,
    SurfaceGeometryInputs,
    SurfaceModelBundle,
    VegetationSurfaceInputs,
    apply_canopy_interception,
    apply_canopy_evaporation,
    build_static_parameters_from_surfdata,
    distribute_root_uptake,
    effective_root_fraction,
    build_surface_features_from_inputs,
    build_surface_features_from_inputs_torch,
    configure_tf32_matmul,
    coupled_state_to_cpu,
    coupled_state_to_device,
    coupled_step,
    diagnostic_soil_surface_humidity_torch,
    force_scalar_snow_state_numpy,
    force_scalar_snow_state_torch,
    load_snicar_aging_tables,
    load_surface_model_bundle,
    predict_surface_targets,
    select_device,
    snicar_aging_tables_loaded,
    _snow_albedo_from_grain_radius,
    _active_snow_depth_torch,
    _active_snow_fraction_torch,
    saturation_vapor_pressure_pa_torch,
    _total_snow_depth_torch,
    _total_snow_fraction_torch,
    _zero_snow_state,
    vapor_pressure_deficit_from_forcing_torch,
)
from .thermal_energy_closure import (
    derive_surface_thermal_boundaries_torch,
    derive_soil_thermal_boundary_torch,
    has_energy_closed_soil_targets,
    has_energy_closed_surface_targets,
)
from .elm_hydrology import elm_soil_grid
from .elm_history import ElmHistoryWriter, infer_elm_history_case_name
from .elm_hydrology_torch import SoilHydrologyTorchWorkspace
from .elm_soil_temperature_torch import SoilTemperatureTorchWorkspace
from .paths import DEFAULT_CLM_PARAMS_SP, DEFAULT_MODEL_DIR, DEFAULT_TESSFA_DOMAIN2_SURFDATA
from .replay_initialization import (
    build_initial_state_1x1,
    prepare_regional_initial_state,
    regional_initial_state,
    save_regional_restart,
    tile_state,
)
from .regional_met import (
    _MET_VAR_IDX,
    _detect_regional_met_format,
    interp_met_to_1h_device,
    load_met_month_cpu,
    redistribute_shortwave_to_1h_device,
)
from .regional_precompute import _load_or_precompute_month_cpu, _precompute_month_cpu
from .regional_surfdata import (
    _add_center_square_precip_treatment,
    _clone_single_regional_cell_to_synthetic_grid,
    _month_intersects_date_window,
    _nearest_regional_cell_index,
    _parse_optional_yyyymmdd,
    _subset_regional_surf_cells,
    _synthetic_trait_value_for_surf,
    load_regional_surfdata,
)
from .regional_time import _day_of_year_start, _days_in_month
from .regional_output import (
    _REGIONAL_DAILY_DIAG_KEYS,
    _REGIONAL_DAILY_OUTPUT_KEYS,
    _REGIONAL_TRAIT_STD_TARGET_MAP,
    _WATER_DIAG_NLEV,
    _write_regional_daily_output,
    _write_regional_single_cell_timeseries,
)
from .regional_plotting import _filter_regional_plot_diag_maps, _plot_diag_maps

STEFAN_BOLTZMANN = 5.670374419e-8
SNICAR_AGING_FILENAME = "snicar_drdt_bst_fit_60_c070416.nc"

STATIC_PARAMETER_FEATURE_NAMES = (
    "pft_type",
    "flnr",
    "mbbopt",
    "smpso",
    "smpsc",
    "soil_color",
    "leafcn",
    "slatop",
    "z0mr",
    "rholvis",
    "rholnir",
    "taulvis",
    "taulnir",
    "rhosvis",
    "rhosnir",
    "tausvis",
    "tausnir",
    "xl",
)
ENSEMBLE_PARAMETER_FEATURE_NAMES = ("flnr", "mbbopt")
REGIONAL_TRAIT_FEATURE_NAMES = (
    "flnr",
    "mbbopt",
    "smpso",
    "smpsc",
    "leafcn",
    "slatop",
    "z0mr",
    "rholvis",
    "rholnir",
    "taulvis",
    "taulnir",
    "rhosvis",
    "rhosnir",
    "tausvis",
    "tausnir",
    "xl",
)
NONNEGATIVE_REGIONAL_TRAIT_FEATURE_NAMES = frozenset({
    "flnr",
    "mbbopt",
    "leafcn",
    "slatop",
    "z0mr",
    "rholvis",
    "rholnir",
    "taulvis",
    "taulnir",
    "rhosvis",
    "rhosnir",
    "tausvis",
    "tausnir",
    "xl",
})
LATENT_HEAT_VAPORIZATION = 2.501e6
LATENT_HEAT_SUBLIMATION = LATENT_HEAT_VAPORIZATION + 3.337e5
GPP_UMOL_CO2_TO_GC = 12.011e-6


# ---------------------------------------------------------------------------
# Training-data helpers (simplified subset of replay_coupled_training_year)
# ---------------------------------------------------------------------------

def _decode_names(attr: str) -> list[str]:
    return [item.strip() for item in attr.split(",") if item.strip()]


def load_training_slice(
    training_file: Path,
    patch_index: int,
    nsteps: int,
    start_year: int | None = None,
) -> tuple[list[str], list[str], list[str], np.ndarray, np.ndarray, np.ndarray]:
    """Load *nsteps* rows for *patch_index* from *training_file*.

    When *start_year* is supplied, rows before that year are skipped before
    taking the contiguous window.

    Returns ``(feature_names, target_names, param_names, features, targets,
    parameters)`` where ``features`` has shape ``(nsteps, nfeat)``.
    """
    with Dataset(training_file) as ds:
        feature_names = _decode_names(getattr(ds, "feature_names"))
        target_names = _decode_names(getattr(ds, "target_names"))
        param_names = _decode_names(getattr(ds, "param_names"))
        patch = np.asarray(ds.variables["patch_index"][:], dtype=np.int32)
        year = np.asarray(ds.variables["year"][:], dtype=np.int32)
        nstep_var = np.asarray(ds.variables["nstep"][:], dtype=np.int32)
        features_all = np.asarray(ds.variables["features"][:], dtype=np.float64)
        targets_all = np.asarray(ds.variables["targets"][:], dtype=np.float64)
        params_all = np.asarray(ds.variables["parameters"][:], dtype=np.float64)

    mask = patch == patch_index
    if start_year is not None:
        mask &= year >= int(start_year)
    if not np.any(mask):
        if start_year is None:
            raise ValueError(f"No samples for patch {patch_index} in {training_file}")
        raise ValueError(
            f"No samples for patch {patch_index} with year >= {start_year} in {training_file}"
        )
    idx = np.where(mask)[0]
    order = np.lexsort((nstep_var[idx], year[idx]))
    idx = idx[order]
    if nsteps > idx.size:
        window = f"year >= {start_year}" if start_year is not None else "all years"
        raise ValueError(
            f"Requested --nsteps {nsteps} but only {idx.size} timesteps available "
            f"for patch {patch_index} ({window})"
        )
    idx = idx[:nsteps]
    return (
        feature_names,
        target_names,
        param_names,
        features_all[idx],
        targets_all[idx],
        params_all[idx],
    )


def _build_static(
    surfdata_file: Path,
    rootfr: np.ndarray,
    device: torch.device,
    dtype: torch.dtype,
    nlev_total: int = 15,
) -> tuple[CoupledStaticParameters, np.ndarray]:
    """Build CoupledStaticParameters with split hydro/thermal levels."""
    rootfr_hydro = np.maximum(rootfr[:10], 0.0)
    rootfr_hydro /= np.maximum(rootfr_hydro.sum(), 1.0e-12)
    rootfr_thermal = np.zeros(nlev_total, dtype=np.float64)
    rootfr_thermal[: rootfr_hydro.size] = rootfr_hydro

    static_full, _, dz, _, _ = build_static_parameters_from_surfdata(
        surfdata_file, device=device, dtype=dtype, nlevsoi=nlev_total, rootfr=rootfr_thermal
    )
    static_hydro, _, _, _, _ = build_static_parameters_from_surfdata(
        surfdata_file, device=device, dtype=dtype, nlevsoi=10, rootfr=rootfr_hydro
    )
    static_full.rootfr = static_hydro.rootfr
    static_full.hydrology_numpy = static_hydro.hydrology_numpy
    static_full.hydrology_torch = static_hydro.hydrology_torch
    return static_full, dz


# ---------------------------------------------------------------------------
# Batched forcing builder
# ---------------------------------------------------------------------------

def _row_inputs_batched(
    fmap: dict[str, int],
    row: np.ndarray,
    state: CoupledColumnState,
    ncell: int,
    nens: int,
) -> tuple[AtmosphericForcingInputs, VegetationSurfaceInputs, SurfaceGeometryInputs]:
    """Build (ncell, nens) forcing inputs from one training row."""
    batch = (ncell, nens)

    def s(name: str) -> np.ndarray:
        return np.full(batch, row[fmap[name]], dtype=np.float64)

    atmosphere = AtmosphericForcingInputs(
        forc_t=s("forc_t"),
        forc_q=s("forc_q"),
        forc_pbot=s("forc_pbot"),
        forc_lwrad=s("forc_lwrad"),
        forc_rain=s("forc_rain"),
        forc_snow=s("forc_snow"),
        forc_swrad=s("forc_swrad"),
        forc_solad_vis=s("forc_solad_vis"),
        forc_solad_nir=s("forc_solad_nir"),
        forc_solai_vis=s("forc_solai_vis"),
        forc_solai_nir=s("forc_solai_nir"),
        wind_speed=s("wind_speed"),
        forc_rho=s("forc_rho"),
        forc_pco2=s("forc_pco2"),
        dayl=s("dayl"),
        max_dayl=s("max_dayl"),
        coszen_col=s("coszen_col"),
    )
    def state_np(field) -> np.ndarray:
        """Move a state field to CPU and convert to float64 numpy."""
        if isinstance(field, torch.Tensor):
            field = field.cpu()
        return np.asarray(field, dtype=np.float64)

    vegetation = VegetationSurfaceInputs(
        frac_veg_nosno_patch=s("frac_veg_nosno_patch"),
        elai_patch=s("elai_patch"),
        esai_patch=s("esai_patch"),
        snow_depth=(
            state_np(state.snow_depth)
            if state.snow_depth is not None
            else s("snow_depth")
        ),
        t10=(
            state_np(state.t10)
            if state.t10 is not None
            else s("t10")
        ),
        htop_patch=s("htop_patch") if "htop_patch" in fmap else None,
    )
    geometry = SurfaceGeometryInputs(
        forc_hgt_u_patch=s("forc_hgt_u_patch"),
        z0mg_col=s("z0mg_col"),
        ugust=s("ugust"),
    )
    return atmosphere, vegetation, geometry


def _row_inputs_batched_torch(
    fmap: dict[str, int],
    row: torch.Tensor,
    state: CoupledColumnState,
    ncell: int,
    nens: int,
) -> tuple[AtmosphericForcingInputs, VegetationSurfaceInputs, SurfaceGeometryInputs]:
    """Build (ncell, nens) Torch forcing inputs from one device-resident row."""
    batch = (ncell, nens)

    def s(name: str) -> torch.Tensor:
        return row[fmap[name]].expand(batch)

    atmosphere = AtmosphericForcingInputs(
        forc_t=s("forc_t"),
        forc_q=s("forc_q"),
        forc_pbot=s("forc_pbot"),
        forc_lwrad=s("forc_lwrad"),
        forc_rain=s("forc_rain"),
        forc_snow=s("forc_snow"),
        forc_swrad=s("forc_swrad"),
        forc_solad_vis=s("forc_solad_vis"),
        forc_solad_nir=s("forc_solad_nir"),
        forc_solai_vis=s("forc_solai_vis"),
        forc_solai_nir=s("forc_solai_nir"),
        wind_speed=s("wind_speed"),
        forc_rho=s("forc_rho"),
        forc_pco2=s("forc_pco2"),
        dayl=s("dayl"),
        max_dayl=s("max_dayl"),
        coszen_col=s("coszen_col"),
    )
    vegetation = VegetationSurfaceInputs(
        frac_veg_nosno_patch=s("frac_veg_nosno_patch"),
        elai_patch=s("elai_patch"),
        esai_patch=s("esai_patch"),
        snow_depth=state.snow_depth if state.snow_depth is not None else s("snow_depth"),
        t10=state.t10 if state.t10 is not None else s("t10"),
        htop_patch=s("htop_patch") if "htop_patch" in fmap else None,
    )
    geometry = SurfaceGeometryInputs(
        forc_hgt_u_patch=s("forc_hgt_u_patch"),
        z0mg_col=s("z0mg_col"),
        ugust=s("ugust"),
    )
    return atmosphere, vegetation, geometry


def _inject_row_overrides(
    features: np.ndarray,
    fmap: dict[str, int],
    row: np.ndarray,
    bundle: SurfaceModelBundle,
    ens_params: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    """Inject static PFT-parameter feature columns into the feature tensor.

    When *ens_params* is provided, per-ensemble member values (shape
    ``(1, nens)``) are broadcast over the cell axis.  Otherwise the
    uniform training-row value is used.
    """
    inames = bundle.input_names
    if not any(col in fmap and col in inames for col in STATIC_PARAMETER_FEATURE_NAMES):
        return features
    features = np.array(features, copy=True)
    for col in STATIC_PARAMETER_FEATURE_NAMES:
        if col not in fmap or col not in inames:
            continue
        idx = inames.index(col)
        if ens_params is not None and col in ens_params:
            features[..., idx] = ens_params[col]  # (1, nens) broadcasts to (ncell, nens)
        else:
            features[..., idx] = row[fmap[col]]
    return features


def _inject_row_overrides_torch(
    features: torch.Tensor,
    fmap: dict[str, int],
    row: torch.Tensor,
    bundle: SurfaceModelBundle,
    ens_params: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Torch-native variant of ``_inject_row_overrides``."""
    inames = bundle.input_names
    if not any(col in fmap and col in inames for col in STATIC_PARAMETER_FEATURE_NAMES):
        return features
    for col in STATIC_PARAMETER_FEATURE_NAMES:
        if col not in fmap or col not in inames:
            continue
        idx = inames.index(col)
        if ens_params is not None and col in ens_params:
            features[..., idx] = ens_params[col].to(device=features.device, dtype=features.dtype)
        else:
            features[..., idx] = row[fmap[col]]
    return features


def sample_ensemble_params(
    feature_names: list[str],
    features_rows: np.ndarray,
    bundle: SurfaceModelBundle,
    nens: int,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Sample fixed unique per-ensemble values of flnr and mbbopt.

    Values are drawn uniformly over [mean − 2σ, mean + 2σ] of the
    training distribution.  Returns ``{feature_name: (1, nens) array}``.
    """
    fmap_local = {name: i for i, name in enumerate(feature_names)}
    rng = np.random.default_rng(seed)
    params: dict[str, np.ndarray] = {}
    for col in ENSEMBLE_PARAMETER_FEATURE_NAMES:
        if col not in fmap_local or col not in bundle.input_names:
            continue
        vals = features_rows[:, fmap_local[col]]
        mean = float(np.mean(vals))
        std = max(float(np.std(vals)), 1.0e-6)
        params[col] = rng.uniform(mean - 2.0 * std, mean + 2.0 * std, size=(1, nens))
    return params


# ---------------------------------------------------------------------------
# Surfdata patch helpers
# ---------------------------------------------------------------------------

def _select_nonzero_surfdata_patches(
    surfdata_file: Path | str,
    *,
    gridcell: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return 1-based PFT patch indices and their normalized area weights.

    Reads ``PCT_NAT_PFT`` (shape ``[npft, ncell]``) from *surfdata_file* and
    returns the indices (0-based into the natpft axis, which equals the 1-based
    PFT number for CLM/ELM native PFTs) and the normalized fractional weights
    for all patches that have non-zero area in the selected *gridcell*.
    """
    with Dataset(surfdata_file) as ds:
        if "PCT_NAT_PFT" not in ds.variables:
            raise ValueError(f"surfdata file {surfdata_file} does not contain PCT_NAT_PFT")
        pct = np.asarray(ds.variables["PCT_NAT_PFT"][:, gridcell], dtype=np.float64)
    idx = np.where(pct > 0.0)[0]
    weights = pct[idx]
    weights = weights / np.maximum(np.sum(weights), 1.0e-12)
    return idx.astype(np.int32), weights


def _blend_diag_outputs(outputs_list: list[dict], weights: np.ndarray) -> dict:
    """Weighted average of per-patch diagnostic output dicts.

    Each dict contains arrays of shape ``(nsteps, ncell, nens)``.  Keys that
    are ``None`` in all patches are returned as ``None``; keys missing from some
    patches but present in others are averaged over the patches that have them
    (with their weights renormalized).
    """
    w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    all_keys: set[str] = set()
    for o in outputs_list:
        all_keys.update(o.keys())
    result: dict = {}
    for key in all_keys:
        valid = [(i, o[key]) for i, o in enumerate(outputs_list) if o.get(key) is not None]
        if not valid:
            result[key] = None
            continue
        valid_w = np.array([w[i] for i, _ in valid], dtype=np.float64)
        valid_w /= valid_w.sum()
        blended = np.zeros_like(np.asarray(valid[0][1], dtype=np.float64))
        for wi, (_, arr) in zip(valid_w, valid):
            blended += wi * np.asarray(arr, dtype=np.float64)
        result[key] = blended
    return result


# ---------------------------------------------------------------------------
# Replay pass
# ---------------------------------------------------------------------------

def _to_np(x) -> np.ndarray:
    """Convert a torch.Tensor or array-like to a float64 NumPy array."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().astype(np.float64)
    return np.asarray(x, dtype=np.float64)


def run_replay_pass(
    features_rows: np.ndarray,
    fmap: dict[str, int],
    bundle: SurfaceModelBundle,
    static: CoupledStaticParameters,
    initial_state: CoupledColumnState,
    *,
    ncell: int,
    nens: int,
    dt: float,
    hydrology_backend: str,
    soil_temperature_backend: str,
    device: torch.device,
    dtype: torch.dtype,
    snow_model_enabled: bool,
    snow_scalar_only: bool = False,
    ens_params: dict[str, np.ndarray] | None = None,
    record_outputs: bool = False,
    profile_sections: dict[str, float] | None = None,
) -> tuple[CoupledColumnState, dict[str, float], dict[str, np.ndarray] | None]:
    """Run one full replay pass over *features_rows* timesteps.

    Returns ``(final_state, sanity_scalars, outputs)`` where *outputs* is
    ``None`` unless *record_outputs* is ``True``, in which case it contains
    ``fpsn_patch`` and ``qflx_tran_veg`` arrays of shape
    ``(nsteps, ncell, nens)``.
    """
    use_torch = hydrology_backend in ("torch", "full") or soil_temperature_backend == "torch"
    state: CoupledColumnState = tile_state(initial_state, ncell, nens)
    if use_torch:
        state = coupled_state_to_device(state, device, dtype)
        if not snow_model_enabled:
            state = _zero_snow_state(state)
        elif snow_scalar_only:
            state = force_scalar_snow_state_torch(state, device=device, dtype=dtype)
        features_rows_device = torch.as_tensor(features_rows, device=device, dtype=dtype)
        ens_params_device = (
            {name: torch.as_tensor(value, device=device, dtype=dtype) for name, value in ens_params.items()}
            if ens_params
            else None
        )
        x_mean_t = (
            bundle.x_mean_torch
            if bundle.x_mean_torch is not None and bundle.x_mean_torch.device == device and bundle.x_mean_torch.dtype == dtype
            else torch.as_tensor(bundle.x_mean, device=device, dtype=dtype)
        )
        feature_template = torch.empty((ncell, nens, x_mean_t.numel()), device=device, dtype=dtype)
        feature_template.copy_(x_mean_t)
        feature_buffer = torch.empty_like(feature_template)
    else:
        if snow_model_enabled and snow_scalar_only:
            state = force_scalar_snow_state_numpy(state)
        features_rows_device = None
        ens_params_device = None
        feature_template = None
        feature_buffer = None

    hydrology_workspace = (
        SoilHydrologyTorchWorkspace()
        if device.type != "cpu" and hydrology_backend in ("torch", "full")
        else None
    )
    soil_temperature_workspace = (
        SoilTemperatureTorchWorkspace()
        if device.type != "cpu" and soil_temperature_backend == "torch"
        else None
    )

    nsteps = features_rows.shape[0]
    _buf_shape = (nsteps, ncell, nens)
    if record_outputs:
        if use_torch:
            fpsn_out   = torch.zeros(_buf_shape, device=device, dtype=dtype)
            tran_out   = torch.zeros(_buf_shape, device=device, dtype=dtype)
            zwt_out    = torch.zeros(_buf_shape, device=device, dtype=dtype)
            h2osfc_out = torch.zeros(_buf_shape, device=device, dtype=dtype)
            lai_out    = torch.zeros(_buf_shape, device=device, dtype=dtype)
            btran_out  = (
                torch.zeros(_buf_shape, device=device, dtype=dtype)
                if "btran" in bundle.input_names else None
            )
        else:
            fpsn_out   = np.zeros(_buf_shape, dtype=np.float64)
            tran_out   = np.zeros(_buf_shape, dtype=np.float64)
            zwt_out    = np.zeros(_buf_shape, dtype=np.float64)
            h2osfc_out = np.zeros(_buf_shape, dtype=np.float64)
            lai_out    = np.zeros(_buf_shape, dtype=np.float64)
            btran_out  = (
                np.zeros(_buf_shape, dtype=np.float64)
                if "btran" in bundle.input_names else None
            )
    else:
        fpsn_out = tran_out = zwt_out = h2osfc_out = lai_out = btran_out = None

    def _sync_time(fn):
        """Call fn(), synchronize device, return (result, elapsed_s)."""
        if device.type not in ("cuda", "mps"):
            t0 = time.perf_counter()
            out = fn()
            return out, time.perf_counter() - t0
        from torch import cuda as _cuda
        if device.type == "cuda":
            _cuda.synchronize(device)
        t0 = time.perf_counter()
        out = fn()
        if device.type == "cuda":
            _cuda.synchronize(device)
        return out, time.perf_counter() - t0

    last_result = None
    _btran_feat_idx: int | None = (
        bundle.input_names.index("btran")
        if btran_out is not None and "btran" in bundle.input_names
        else None
    )
    row_iter = features_rows_device if use_torch and features_rows_device is not None else features_rows
    for step_i, row in enumerate(row_iter):
        if use_torch:
            assert isinstance(row, torch.Tensor)
            assert feature_template is not None
            assert feature_buffer is not None

            def _assemble():
                atm, veg, geo = _row_inputs_batched_torch(fmap, row, state, ncell, nens)
                feature_buffer.copy_(feature_template)
                feat = build_surface_features_from_inputs_torch(
                    bundle, atm, veg, geo, state, static,
                    device=device, dtype=dtype, features=feature_buffer,
                    snow_model_enabled=snow_model_enabled,
                )
                return _inject_row_overrides_torch(feat, fmap, row, bundle, ens_params=ens_params_device)

            if profile_sections is not None:
                features, _dt = _sync_time(_assemble)
                profile_sections["feature_assembly"] = profile_sections.get("feature_assembly", 0.0) + _dt
            else:
                features = _assemble()
            if btran_out is not None and _btran_feat_idx is not None:
                btran_out[step_i] = features[..., _btran_feat_idx].reshape(ncell, nens)
        else:
            atmosphere, vegetation, geometry = _row_inputs_batched(fmap, row, state, ncell, nens)
            features = build_surface_features_from_inputs(
                bundle,
                atmosphere,
                vegetation,
                geometry,
                state,
                static,
                snow_model_enabled=snow_model_enabled,
            )
            features = _inject_row_overrides(features, fmap, row, bundle, ens_params=ens_params)
            if btran_out is not None and _btran_feat_idx is not None:
                btran_out[step_i] = np.asarray(features[..., _btran_feat_idx], dtype=np.float64).reshape(ncell, nens)

        last_result = coupled_step(
            bundle,
            features,
            state,
            static,
            dt,
            hydrology_backend=hydrology_backend,
            soil_temperature_backend=soil_temperature_backend,
            device=device,
            dtype=dtype,
            hydrology_workspace=hydrology_workspace,
            soil_temperature_workspace=soil_temperature_workspace,
            snow_model_enabled=snow_model_enabled,
            snow_scalar_only=snow_scalar_only,
            copy_surface_features=not use_torch,
            snow_state_already_disabled=use_torch and not snow_model_enabled,
            profile_sections=profile_sections,
        )
        state = last_result.state

        if record_outputs and fpsn_out is not None and tran_out is not None:
            ft = last_result.surface_targets
            if use_torch:
                fpsn_out[step_i] = ft["fpsn_patch"]
                if all(k in ft for k in ("laisun_patch", "laisha_patch")):
                    lsun = ft["laisun_patch"].clamp(min=0.0)
                    lsha = ft["laisha_patch"].clamp(min=0.0)
                    lai_out[step_i] = (lsun + lsha).reshape(ncell, nens)
                tran_out[step_i] = ft["qflx_tran_veg"]
                if zwt_out is not None and last_result.state.zwt is not None:
                    zwt_out[step_i] = last_result.state.zwt.reshape(ncell, nens)
                if h2osfc_out is not None and last_result.state.h2osfc is not None:
                    h2osfc_out[step_i] = last_result.state.h2osfc.reshape(ncell, nens)
            else:
                fpsn_val = _to_np(ft["fpsn_patch"])
                if all(k in ft for k in ("laisun_patch", "laisha_patch")):
                    lai_val = (
                        np.maximum(_to_np(ft["laisun_patch"]), 0.0)
                        + np.maximum(_to_np(ft["laisha_patch"]), 0.0)
                    )
                else:
                    lai_val  = np.zeros_like(fpsn_val)
                fpsn_out[step_i] = fpsn_val
                tran_out[step_i] = _to_np(ft["qflx_tran_veg"])
                if zwt_out is not None:
                    zwt_arr = last_result.state.zwt
                    zwt_out[step_i] = _to_np(zwt_arr).reshape(ncell, nens) if zwt_arr is not None else 0.0
                if h2osfc_out is not None:
                    sfc_arr = last_result.state.h2osfc
                    h2osfc_out[step_i] = _to_np(sfc_arr).reshape(ncell, nens) if sfc_arr is not None else 0.0
                if lai_out is not None:
                    lai_out[step_i] = lai_val.reshape(ncell, nens)

    if use_torch:
        state = coupled_state_to_cpu(state)
        if record_outputs and fpsn_out is not None:
            fpsn_out   = fpsn_out.cpu().numpy().astype(np.float64)
            tran_out   = tran_out.cpu().numpy().astype(np.float64)
            zwt_out    = zwt_out.cpu().numpy().astype(np.float64)
            h2osfc_out = h2osfc_out.cpu().numpy().astype(np.float64)
            lai_out    = lai_out.cpu().numpy().astype(np.float64)
            if btran_out is not None:
                btran_out = btran_out.cpu().numpy().astype(np.float64)

    sanity: dict[str, float] = {}
    if last_result is not None:
        h2osfc = _to_np(last_result.state.h2osfc)
        t_top = _to_np(last_result.state.t_soisno[..., 0])
        sanity["mean_h2osfc"] = float(h2osfc.mean())
        sanity["mean_t_top"] = float(t_top.mean())

    outputs: dict[str, np.ndarray] | None = None
    if record_outputs and fpsn_out is not None and tran_out is not None:
        outputs = {
            "fpsn_patch": fpsn_out,
            "qflx_tran_veg": tran_out,
            "zwt": zwt_out if zwt_out is not None else np.zeros_like(fpsn_out),
            "h2osfc": h2osfc_out if h2osfc_out is not None else np.zeros_like(fpsn_out),
            "lai": lai_out if lai_out is not None else np.zeros_like(fpsn_out),
        }
        if btran_out is not None:
            outputs["btran"] = btran_out
    return state, sanity, outputs


# ---------------------------------------------------------------------------
# Blended-patch replay pass
# ---------------------------------------------------------------------------

def run_blended_replay_pass(
    patch_setups: list[dict],
    patch_weights: np.ndarray,
    *,
    patch_statics: list[CoupledStaticParameters],
    static: CoupledStaticParameters,
    initial_state: CoupledColumnState,
    ncell: int,
    nens: int,
    dt: float,
    hydrology_backend: str,
    soil_temperature_backend: str,
    device: torch.device,
    dtype: torch.dtype,
    snow_model_enabled: bool,
    snow_scalar_only: bool = False,
    record_outputs: bool = False,
    profile_sections: dict[str, float] | None = None,
) -> tuple[CoupledColumnState, dict[str, float], dict[str, np.ndarray] | None]:
    """Run one blended-patch replay pass.

    For each timestep:
    1. Builds per-patch feature vectors from the shared column state.
    2. Runs canopy interception + MLP inference + canopy evaporation per patch.
    3. Blends all surface targets by area weight (on-device for torch).
    4. Advances a single shared column state through snow, hydrology, and soil
       temperature using the blended targets (via ``coupled_step`` with
       ``precomputed_surface_targets``).

    Each element of *patch_setups* must contain:
        ``"features_rows"``: np.ndarray  shape ``(nsteps, nfeat)``
        ``"fmap"``:          ``dict[str, int]``
        ``"bundle"``:        ``SurfaceModelBundle``
        ``"ens_params"``:    ``dict[str, np.ndarray] | None``
    """
    use_torch = hydrology_backend in ("torch", "full") or soil_temperature_backend == "torch"
    w_arr = np.asarray(patch_weights, dtype=np.float64)
    w_arr = w_arr / w_arr.sum()
    w_torch = torch.as_tensor(w_arr, device=device, dtype=dtype) if use_torch else None

    state: CoupledColumnState = tile_state(initial_state, ncell, nens)
    if use_torch:
        state = coupled_state_to_device(state, device, dtype)
        if not snow_model_enabled:
            state = _zero_snow_state(state)
        elif snow_scalar_only:
            state = force_scalar_snow_state_torch(state, device=device, dtype=dtype)
    elif snow_model_enabled and snow_scalar_only:
        state = force_scalar_snow_state_numpy(state)

    # Pre-build per-patch device-resident tensors (allocated once, reused each step).
    pp_list: list[dict] = []
    for p in patch_setups:
        p_bundle = p["bundle"]
        rows_dev = torch.as_tensor(p["features_rows"], device=device, dtype=dtype) if use_torch else None
        ens_dev = (
            {nm: torch.as_tensor(v, device=device, dtype=dtype) for nm, v in p["ens_params"].items()}
            if use_torch and p.get("ens_params") else None
        )
        if use_torch:
            x_mean_t = (
                p_bundle.x_mean_torch
                if (p_bundle.x_mean_torch is not None
                    and p_bundle.x_mean_torch.device == device
                    and p_bundle.x_mean_torch.dtype == dtype)
                else torch.as_tensor(p_bundle.x_mean, device=device, dtype=dtype)
            )
            feat_tmpl = torch.empty((ncell, nens, x_mean_t.numel()), device=device, dtype=dtype)
            feat_tmpl.copy_(x_mean_t)
            feat_buf = torch.empty_like(feat_tmpl)
        else:
            feat_tmpl = feat_buf = None
        pp_list.append({
            "bundle": p_bundle,
            "fmap": p["fmap"],
            "rows_dev": rows_dev,
            "rows_cpu": p["features_rows"],
            "ens_dev": ens_dev,
            "ens_cpu": p.get("ens_params"),
            "feat_tmpl": feat_tmpl,
            "feat_buf": feat_buf,
        })

    hydrology_workspace = (
        SoilHydrologyTorchWorkspace()
        if device.type != "cpu" and hydrology_backend in ("torch", "full") else None
    )
    soil_temperature_workspace = (
        SoilTemperatureTorchWorkspace()
        if device.type != "cpu" and soil_temperature_backend == "torch" else None
    )

    nsteps = patch_setups[0]["features_rows"].shape[0]
    _buf_shape = (nsteps, ncell, nens)
    ref_pp = pp_list[0]
    ref_bundle = ref_pp["bundle"]

    if record_outputs:
        if use_torch:
            fpsn_out   = torch.zeros(_buf_shape, device=device, dtype=dtype)
            tran_out   = torch.zeros(_buf_shape, device=device, dtype=dtype)
            zwt_out    = torch.zeros(_buf_shape, device=device, dtype=dtype)
            h2osfc_out = torch.zeros(_buf_shape, device=device, dtype=dtype)
            lai_out    = torch.zeros(_buf_shape, device=device, dtype=dtype)
            btran_out  = (
                torch.zeros(_buf_shape, device=device, dtype=dtype)
                if "btran" in ref_bundle.input_names else None
            )
        else:
            fpsn_out   = np.zeros(_buf_shape, dtype=np.float64)
            tran_out   = np.zeros(_buf_shape, dtype=np.float64)
            zwt_out    = np.zeros(_buf_shape, dtype=np.float64)
            h2osfc_out = np.zeros(_buf_shape, dtype=np.float64)
            lai_out    = np.zeros(_buf_shape, dtype=np.float64)
            btran_out  = (
                np.zeros(_buf_shape, dtype=np.float64)
                if "btran" in ref_bundle.input_names else None
            )
    else:
        fpsn_out = tran_out = zwt_out = h2osfc_out = lai_out = btran_out = None

    last_result = None
    for step_i in range(nsteps):
        # --- Per-patch: feature assembly + canopy interception + MLP + canopy evaporation ---
        all_targets: list[dict] = []
        canopy_h2ocans: list = []
        blended_btran_step = None
        ref_feat = None  # reference features saved for the physics step

        for p_idx, pp in enumerate(pp_list):
            p_bundle = pp["bundle"]
            p_fmap = pp["fmap"]
            p_static = patch_statics[p_idx]  # per-patch static for correct smpso/smpsc/rootfr
            if use_torch:
                row = pp["rows_dev"][step_i]
                atm, veg, geo = _row_inputs_batched_torch(p_fmap, row, state, ncell, nens)
                pp["feat_buf"].copy_(pp["feat_tmpl"])
                feat = build_surface_features_from_inputs_torch(
                    p_bundle, atm, veg, geo, state, p_static,
                    device=device, dtype=dtype, features=pp["feat_buf"],
                    snow_model_enabled=snow_model_enabled,
                )
                feat = _inject_row_overrides_torch(feat, p_fmap, row, p_bundle, ens_params=pp["ens_dev"])
            else:
                row = pp["rows_cpu"][step_i]
                atm, veg, geo = _row_inputs_batched(p_fmap, row, state, ncell, nens)
                feat = build_surface_features_from_inputs(
                    p_bundle, atm, veg, geo, state, p_static, snow_model_enabled=snow_model_enabled
                )
                feat = _inject_row_overrides(feat, p_fmap, row, p_bundle, ens_params=pp["ens_cpu"])

            # Save reference patch features (atmospheric forcing is identical across patches;
            # used by update_snow_state and t10 update inside coupled_step).
            if p_idx == 0:
                ref_feat = feat

            # Accumulate area-weighted btran blend.
            if btran_out is not None and "btran" in p_bundle.input_names:
                p_btran_idx = p_bundle.input_names.index("btran")
                if use_torch:
                    contrib = w_torch[p_idx] * feat[..., p_btran_idx].reshape(ncell, nens)
                else:
                    contrib = w_arr[p_idx] * np.asarray(feat[..., p_btran_idx], dtype=np.float64).reshape(ncell, nens)
                blended_btran_step = contrib if blended_btran_step is None else blended_btran_step + contrib

            # Canopy interception.
            canopy_precip, intercept_state = apply_canopy_interception(p_bundle, feat, state, dt)

            # Update h2ocan in the feature tensor (mirrors coupled_step behaviour).
            p_h2ocan_idx = next((i for i, n in enumerate(p_bundle.input_names) if n == "h2ocan"), None)
            if p_h2ocan_idx is not None:
                if use_torch:
                    feat[..., p_h2ocan_idx] = (
                        intercept_state.h2ocan if isinstance(intercept_state.h2ocan, torch.Tensor)
                        else torch.as_tensor(intercept_state.h2ocan, device=device, dtype=dtype)
                    )
                else:
                    feat[..., p_h2ocan_idx] = np.asarray(intercept_state.h2ocan, dtype=np.float64)

            # MLP inference.
            targets = predict_surface_targets(
                p_bundle,
                feat,
                device=device,
                dtype=dtype,
            )

            # hs_top_snow fallback.
            if "hs_top_snow" not in targets:
                if isinstance(next(iter(targets.values())), torch.Tensor):
                    targets["hs_top_snow"] = targets["hs_soil"]
                else:
                    targets["hs_top_snow"] = np.asarray(targets["hs_soil"], dtype=np.float64)

            # Merge canopy precipitation fluxes.
            targets.update(canopy_precip)

            # Canopy evaporation.
            targets, canopy_state_p = apply_canopy_evaporation(
                targets, intercept_state, dt, device=device, dtype=dtype
            )

            all_targets.append(targets)
            canopy_h2ocans.append(
                canopy_state_p.h2ocan if use_torch
                else np.asarray(canopy_state_p.h2ocan, dtype=np.float64)
            )

        # --- Area-weighted blend of surface targets (on-device for torch) ---
        all_keys: set[str] = set()
        for t in all_targets:
            all_keys.update(t.keys())
        blended: dict = {}
        for key in all_keys:
            acc = None
            for p_idx, t in enumerate(all_targets):
                v = t.get(key)
                if v is None:
                    continue
                if use_torch:
                    v_t = v if isinstance(v, torch.Tensor) else torch.as_tensor(v, device=device, dtype=dtype)
                    term = w_torch[p_idx] * v_t
                else:
                    term = w_arr[p_idx] * np.asarray(v, dtype=np.float64)
                acc = term if acc is None else acc + term
            if acc is not None:
                blended[key] = acc

        # --- Transpiration-weighted root-fraction blend (matches replay_coupled_training_year) ---
        # Per-patch: rootfrac_p * qtran_p.  Blend: sum(w_p * rootfrac_p * qtran_p).
        # The normalize-then-re-multiply in native_blended_root_profile cancels to this.
        # Torch path stays fully on-device; numpy path uses the numpy backend.
        _rootsoi_acc: torch.Tensor | np.ndarray | None = None
        for _p_idx, (_t_p, _ps) in enumerate(zip(all_targets, patch_statics)):
            _p_qtran = _t_p["qflx_tran_veg"]
            if use_torch:
                _qroot_p = distribute_root_uptake(_p_qtran, state, _ps, "torch", device=device, dtype=dtype)
                _term: torch.Tensor | np.ndarray = w_torch[_p_idx] * _qroot_p
            else:
                _qroot_p = distribute_root_uptake(
                    np.asarray(_p_qtran, dtype=np.float64), state, _ps, "numpy"
                )
                _term = w_arr[_p_idx] * np.asarray(_qroot_p, dtype=np.float64)
            _rootsoi_acc = _term if _rootsoi_acc is None else _rootsoi_acc + _term
        blended["qflx_rootsoi"] = _rootsoi_acc

        # Blend per-patch h2ocan for the canopy state passed to update_snow_state.
        if use_torch:
            blended_h2ocan: torch.Tensor = sum(  # type: ignore[assignment]
                w_torch[i] * (h if isinstance(h, torch.Tensor) else torch.as_tensor(h, device=device, dtype=dtype))
                for i, h in enumerate(canopy_h2ocans)
            )
        else:
            blended_h2ocan_np: np.ndarray = sum(w_arr[i] * h for i, h in enumerate(canopy_h2ocans))  # type: ignore[assignment]

        # The blended canopy state carries blended h2ocan; all soil/snow fields
        # come from the shared column state.
        blended_canopy_state = CoupledColumnState(
            h2osoi_liq=state.h2osoi_liq,
            h2osoi_ice=state.h2osoi_ice,
            h2ocan=blended_h2ocan if use_torch else blended_h2ocan_np,
            h2osfc=state.h2osfc,
            frac_h2osfc=state.frac_h2osfc,
            frac_sno=state.frac_sno,
            zwt=state.zwt,
            t_soisno=state.t_soisno,
            wa=state.wa,
            t_h2osfc=state.t_h2osfc,
            t10=state.t10,
            t10_history=state.t10_history,
            h2osno=state.h2osno,
            snow_depth=state.snow_depth,
            snow_t=state.snow_t,
            snow_layers=state.snow_layers,
        )

        # --- Single physics step (snow + hydrology + soil temperature) ---
        last_result = coupled_step(
            ref_bundle,
            ref_feat,
            state,
            static,
            dt,
            hydrology_backend=hydrology_backend,
            soil_temperature_backend=soil_temperature_backend,
            device=device,
            dtype=dtype,
            hydrology_workspace=hydrology_workspace,
            soil_temperature_workspace=soil_temperature_workspace,
            snow_model_enabled=snow_model_enabled,
            snow_scalar_only=snow_scalar_only,
            copy_surface_features=not use_torch,
            snow_state_already_disabled=use_torch and not snow_model_enabled,
            profile_sections=profile_sections,
            precomputed_surface_targets=blended,
            precomputed_canopy_state=blended_canopy_state,
        )
        state = last_result.state

        # --- Record outputs ---
        if record_outputs and fpsn_out is not None and tran_out is not None:
            ft = last_result.surface_targets
            if use_torch:
                fpsn_out[step_i] = ft["fpsn_patch"]
                if all(k in ft for k in ("laisun_patch", "laisha_patch")):
                    lsun = ft["laisun_patch"].clamp(min=0.0)
                    lsha = ft["laisha_patch"].clamp(min=0.0)
                    lai_out[step_i] = (lsun + lsha).reshape(ncell, nens)
                tran_out[step_i] = ft["qflx_tran_veg"]
                if zwt_out is not None and last_result.state.zwt is not None:
                    zwt_out[step_i] = last_result.state.zwt.reshape(ncell, nens)
                if h2osfc_out is not None and last_result.state.h2osfc is not None:
                    h2osfc_out[step_i] = last_result.state.h2osfc.reshape(ncell, nens)
                if btran_out is not None and blended_btran_step is not None:
                    btran_out[step_i] = blended_btran_step
            else:
                fpsn_val = _to_np(ft.get("fpsn_patch", np.zeros((ncell, nens))))
                if all(k in ft for k in ("laisun_patch", "laisha_patch")):
                    lai_val = (
                        np.maximum(_to_np(ft["laisun_patch"]), 0.0)
                        + np.maximum(_to_np(ft["laisha_patch"]), 0.0)
                    )
                else:
                    lai_val = np.zeros_like(fpsn_val)
                fpsn_out[step_i] = fpsn_val
                tran_out[step_i] = _to_np(ft["qflx_tran_veg"])
                if zwt_out is not None:
                    zwt_arr = last_result.state.zwt
                    zwt_out[step_i] = _to_np(zwt_arr).reshape(ncell, nens) if zwt_arr is not None else 0.0
                if h2osfc_out is not None:
                    sfc_arr = last_result.state.h2osfc
                    h2osfc_out[step_i] = _to_np(sfc_arr).reshape(ncell, nens) if sfc_arr is not None else 0.0
                if lai_out is not None:
                    lai_out[step_i] = lai_val.reshape(ncell, nens)
                if btran_out is not None and blended_btran_step is not None:
                    btran_out[step_i] = blended_btran_step

    if use_torch:
        state = coupled_state_to_cpu(state)
        if record_outputs and fpsn_out is not None:
            fpsn_out   = fpsn_out.cpu().numpy().astype(np.float64)
            tran_out   = tran_out.cpu().numpy().astype(np.float64)
            zwt_out    = zwt_out.cpu().numpy().astype(np.float64)
            h2osfc_out = h2osfc_out.cpu().numpy().astype(np.float64)
            lai_out    = lai_out.cpu().numpy().astype(np.float64)
            if btran_out is not None:
                btran_out = btran_out.cpu().numpy().astype(np.float64)

    sanity: dict[str, float] = {}
    if last_result is not None:
        h2osfc_s = _to_np(last_result.state.h2osfc)
        t_top_s  = _to_np(last_result.state.t_soisno[..., 0])
        sanity["mean_h2osfc"] = float(h2osfc_s.mean())
        sanity["mean_t_top"]  = float(t_top_s.mean())

    outputs: dict[str, np.ndarray] | None = None
    if record_outputs and fpsn_out is not None and tran_out is not None:
        outputs = {
            "fpsn_patch": fpsn_out,
            "qflx_tran_veg": tran_out,
            "zwt": zwt_out if zwt_out is not None else np.zeros_like(fpsn_out),
            "h2osfc": h2osfc_out if h2osfc_out is not None else np.zeros_like(fpsn_out),
            "lai": lai_out if lai_out is not None else np.zeros_like(fpsn_out),
        }
        if btran_out is not None:
            outputs["btran"] = btran_out
    return state, sanity, outputs


# ---------------------------------------------------------------------------
# Diagnostic plot
# ---------------------------------------------------------------------------

def plot_diagnostics(
    output_path: Path,
    fpsn: np.ndarray,
    qflx_tran_veg: np.ndarray,
    ens_params: dict[str, np.ndarray],
    dt: float,
    daily_average: bool = False,
    zwt: np.ndarray | None = None,
    h2osfc: np.ndarray | None = None,
    lai: np.ndarray | None = None,
    btran: np.ndarray | None = None,
    t_soil: np.ndarray | None = None,
    frac_h2osfc: np.ndarray | None = None,
    snow_depth: np.ndarray | None = None,
    snow_depth_eff: np.ndarray | None = None,
    forc_t: np.ndarray | None = None,
    forc_pbot: np.ndarray | None = None,
) -> None:
    """Save a time-series plot of fpsn, transpiration, LAI, water-table depth,
    and surface water.

    Individual ensemble-member lines are drawn semi-transparently and coloured
    by flnr value (or steelblue when flnr is absent).  The ensemble mean and
    ±1σ band are overlaid.

    Parameters
    ----------
    fpsn, qflx_tran_veg:
        Arrays of shape ``(nsteps, ncell, nens)``.
    ens_params:
        Per-ensemble parameter overrides, e.g. ``{"flnr": (1, nens), ...}``.
    zwt, h2osfc, lai:
        Optional extra state arrays of shape ``(nsteps, ncell, nens)``.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    nsteps, ncell, nens = fpsn.shape

    # Keep hourly values by default so this plot is directly comparable to
    # replay_coupled_training_year.py. Daily means include nighttime FPSN
    # zeros and can make daytime photosynthesis look artificially small.
    steps_per_day = max(1, round(86400.0 / dt))
    if daily_average and nsteps > 1000:
        n_full = (nsteps // steps_per_day) * steps_per_day
        def _daily_avg(arr):  # arr: (nsteps, ...)
            return arr[:n_full].reshape(n_full // steps_per_day, steps_per_day, *arr.shape[1:]).mean(axis=1)
        fpsn          = _daily_avg(fpsn)
        qflx_tran_veg = _daily_avg(qflx_tran_veg)
        if zwt    is not None: zwt    = _daily_avg(zwt)
        if h2osfc is not None: h2osfc = _daily_avg(h2osfc)
        if lai    is not None: lai    = _daily_avg(lai)
        if btran  is not None: btran  = _daily_avg(btran)
        if t_soil is not None: t_soil = _daily_avg(t_soil)
        if frac_h2osfc is not None: frac_h2osfc = _daily_avg(frac_h2osfc)
        if snow_depth is not None: snow_depth = _daily_avg(snow_depth)
        if snow_depth_eff is not None: snow_depth_eff = _daily_avg(snow_depth_eff)
        if forc_t is not None: forc_t = _daily_avg(forc_t)
        if forc_pbot is not None: forc_pbot = _daily_avg(forc_pbot)
        nsteps  = fpsn.shape[0]
        time_h  = np.arange(nsteps) * steps_per_day * dt / 3600.0
        xlabel  = "Time (hours, 24 h averages)"
    else:
        time_h = np.arange(nsteps) * dt / 3600.0
        xlabel = "Time (hours)"

    # Colour lines by flnr if available, else uniform steelblue.
    flnr_vals = ens_params.get("flnr")
    if flnr_vals is not None:
        fv = flnr_vals.reshape(-1)  # (nens,)
        lo, hi = fv.min(), fv.max()
        norm = (fv - lo) / max(hi - lo, 1.0e-12)
        colors = [cm.plasma(float(n)) for n in norm]
        cbar_label = "flnr"
        cbar_ticks = [lo, hi]
    else:
        colors = ["steelblue"] * nens
        cbar_label = None
        cbar_ticks = None

    show_lines = ncell * nens <= 1000

    # Build panel list: always fpsn + transpiration; add lai/zwt/h2osfc when provided.
    panels: list[tuple[np.ndarray, np.ndarray, str]] = [
        (fpsn[:, 0, :],          fpsn.reshape(nsteps, -1),          "fpsn_patch  (µmol CO₂ m⁻² s⁻¹)"),
        (qflx_tran_veg[:, 0, :] * 86400.0,
         qflx_tran_veg.reshape(nsteps, -1) * 86400.0,
         "qflx_tran_veg  (mm d⁻¹)"),
    ]
    if forc_t is not None:
        panels.append((forc_t[:, 0, :], forc_t.reshape(nsteps, -1), "2-m air temperature  (K)"))
    if t_soil is not None:
        panels.append((t_soil[:, 0, :], t_soil.reshape(nsteps, -1), "Top-layer soil temperature  (K)"))
    if forc_pbot is not None:
        panels.append((forc_pbot[:, 0, :] / 1000.0, forc_pbot.reshape(nsteps, -1) / 1000.0, "Surface pressure  (kPa)"))
    if lai is not None:
        panels.append((lai[:, 0, :], lai.reshape(nsteps, -1), "LAI  (m² m⁻²)"))
    if zwt is not None:
        panels.append((zwt[:, 0, :], zwt.reshape(nsteps, -1), "Water table depth  (m)"))
    if h2osfc is not None:
        panels.append((h2osfc[:, 0, :], h2osfc.reshape(nsteps, -1), "Surface water  (mm)"))
    if frac_h2osfc is not None:
        panels.append((frac_h2osfc[:, 0, :], frac_h2osfc.reshape(nsteps, -1), "Surface water fraction"))
    if snow_depth is not None:
        panels.append((snow_depth[:, 0, :], snow_depth.reshape(nsteps, -1), "Snow depth  (m)"))
    if snow_depth_eff is not None:
        panels.append((snow_depth_eff[:, 0, :], snow_depth_eff.reshape(nsteps, -1), "Snow fraction x depth  (m)"))
    if btran is not None:
        panels.append((btran[:, 0, :], btran.reshape(nsteps, -1), "btran  (plant water stress, 0–1)"))

    nrows = len(panels)
    fig, axes = plt.subplots(nrows, 1, figsize=(11, 2.5 * nrows + 0.5), sharex=True)
    if nrows == 1:
        axes = [axes]  # ensure iterable

    for ax, (cell0_data, flat_data, ylabel) in zip(axes, panels):
        if show_lines:
            for j in range(nens):
                ax.plot(time_h, cell0_data[:, j], lw=0.8, alpha=0.35, color=colors[j])
        mean = flat_data.mean(axis=1)
        std  = flat_data.std(axis=1)
        ax.plot(time_h, mean, lw=2.0, color="black", label="ensemble mean")
        ax.fill_between(time_h, mean - std, mean + std,
                        alpha=0.20, color="black", label="±1σ")
        ax.set_ylabel(ylabel, fontsize=8)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7, loc="upper right")
        # Invert y-axis for water table (deeper = lower on plot)
        if "Water table" in ylabel:
            ax.invert_yaxis()

    axes[-1].set_xlabel(xlabel)
    fig.suptitle(
        f"Coupled replay  |  {ncell} cells × {nens} ensemble members  |  {nsteps} steps",
        fontsize=10,
    )

    # Colourbar for flnr when available.
    if flnr_vals is not None and cbar_ticks is not None:
        sm = cm.ScalarMappable(cmap="plasma",
                               norm=plt.Normalize(vmin=cbar_ticks[0], vmax=cbar_ticks[1]))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axes, fraction=0.015, pad=0.01)
        cbar.set_label(cbar_label, fontsize=8)

    fig.tight_layout(rect=[0, 0, 0.88 if flnr_vals is not None else 1.0, 1.0])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved diagnostic plot → {output_path}")


# ---------------------------------------------------------------------------
# Regional benchmark helpers
# ---------------------------------------------------------------------------

_DEFAULT_CO2_FILE_RELATIVE = Path("atm/datm7/CO2/fco2_datm_rcp4.5_1765-2500_c130312.nc")
_CO2_PPM_FALLBACK = 338.0
_CO2_FILE_CACHE: dict[Path, tuple[np.ndarray, np.ndarray]] = {}


def _infer_inputdata_root(path: Path) -> Path | None:
    """Return the nearest ``inputdata`` ancestor for OLMT/E3SM-style paths."""
    path = Path(path).expanduser()
    for candidate in (path, *path.parents):
        if candidate.name == "inputdata":
            return candidate
    return None


def _default_co2_file_for_met_dir(met_dir: Path) -> Path | None:
    inputdata_root = _infer_inputdata_root(Path(met_dir))
    if inputdata_root is None:
        return None
    return inputdata_root / _DEFAULT_CO2_FILE_RELATIVE


def _decimal_year(year: int, month: int, day: int, *, calendar: str = "noleap") -> float:
    mdays = [_days_in_month(year, m, calendar) for m in range(1, 13)]
    doy = int(day) + sum(mdays[: max(0, int(month) - 1)])
    return float(year) + (float(doy) - 0.5) / float(sum(mdays))


def _co2_var_name(ds: Dataset) -> str:
    for name in ("CO2", "co2", "CO2_PPM", "co2_ppm"):
        if name in ds.variables:
            return name
    raise ValueError("CO2 file does not contain a CO2 variable")


def _load_co2_file_series(path: Path) -> tuple[np.ndarray, np.ndarray]:
    path = Path(path).expanduser()
    cache_key = path.resolve() if path.exists() else path
    cached = _CO2_FILE_CACHE.get(cache_key)
    if cached is not None:
        return cached
    with Dataset(path) as ds:
        co2_name = _co2_var_name(ds)
        co2 = np.asarray(ds.variables[co2_name][:], dtype=np.float64)
        if co2.ndim > 1:
            co2 = np.nanmean(co2.reshape((co2.shape[0], -1)), axis=1)
        if "date" in ds.variables:
            dates = np.asarray(ds.variables["date"][:], dtype=np.int64)
            years = dates // 10000
            months = (dates // 100) % 100
            days = dates % 100
            x = np.asarray(
                [
                    _decimal_year(int(year), int(month), int(day), calendar="noleap")
                    for year, month, day in zip(years, months, days)
                ],
                dtype=np.float64,
            )
        elif "time" in ds.variables:
            time_var = ds.variables["time"]
            calendar = str(getattr(time_var, "calendar", "noleap")).lower()
            dates = num2date(
                time_var[:],
                time_var.units,
                calendar=calendar,
                only_use_cftime_datetimes=False,
                only_use_python_datetimes=False,
            )
            x = np.asarray(
                [
                    _decimal_year(
                        int(date.year),
                        int(date.month),
                        int(date.day),
                        calendar="noleap" if calendar == "noleap" else "gregorian",
                    )
                    for date in dates
                ],
                dtype=np.float64,
            )
        else:
            raise ValueError("CO2 file must contain either a date or time coordinate")
    if x.size != co2.size:
        raise ValueError(f"CO2 coordinate length {x.size} does not match CO2 length {co2.size}")
    order = np.argsort(x)
    result = (x[order], co2[order])
    _CO2_FILE_CACHE[cache_key] = result
    return result


def _co2_ppm_from_file(path: Path, year: int, month: int, *, calendar: str = "noleap") -> float:
    x, co2 = _load_co2_file_series(path)
    mdays = _days_in_month(year, month, calendar)
    target = _decimal_year(year, month, max(1, int(round(mdays * 0.5))), calendar=calendar)
    return float(np.interp(target, x, co2, left=co2[0], right=co2[-1]))


def _resolve_regional_co2_file(co2_file: Path | None, met_dir: Path) -> Path | None:
    if co2_file is not None:
        return Path(co2_file).expanduser()
    default_path = _default_co2_file_for_met_dir(met_dir)
    if default_path is not None and default_path.exists():
        return default_path
    return None


def _resolve_snicar_aging_file(
    explicit_path: Path | None,
    clm_params: Path | None,
) -> Path | None:
    """Locate the ELM SNICAR grain-aging lookup table."""
    if explicit_path is not None:
        return explicit_path.expanduser().resolve()
    project_root = Path(__file__).resolve().parents[2]
    candidates = []
    if clm_params is not None:
        candidates.append(Path(clm_params).expanduser().resolve().parent / SNICAR_AGING_FILENAME)
    candidates.extend(
        (
            project_root / "e3sm_inputdata/lnd/clm2/snicardata" / SNICAR_AGING_FILENAME,
            project_root / "E3SM-emulator/components/elm/tools/canopyflux_emulator" / SNICAR_AGING_FILENAME,
            project_root / "surface_emulator" / SNICAR_AGING_FILENAME,
        )
    )
    return next((path for path in candidates if path.is_file()), None)

def _find_elm_history_file(path: Path, year: int) -> Path:
    """Find the ELM h0 history file for ``year`` under a case run directory."""
    path = Path(path)
    year_s = f"{int(year):04d}"
    if path.is_file():
        if year_s in path.name:
            return path
        sibling_matches = sorted(path.parent.glob(f"*.elm.h0.{year_s}-01-01-00000.nc"))
        if sibling_matches:
            return sibling_matches[0]
        raise FileNotFoundError(f"Could not find sibling ELM h0 history file for {year_s} near {path}")
    matches = sorted(path.glob(f"*.elm.h0.{year_s}-01-01-00000.nc"))
    if not matches:
        raise FileNotFoundError(f"Could not find ELM h0 history file for {year_s} under {path}")
    return matches[0]


def _read_elm_tsoi_history(path: Path, ncell: int) -> np.ndarray:
    """Read ELM TSOI as (time, ncell, levgrnd), with single-cell broadcast allowed."""
    with Dataset(path) as ds:
        if "TSOI" not in ds.variables:
            raise KeyError(f"{path} does not contain TSOI")
        var = ds.variables["TSOI"]
        data = np.asarray(var[:], dtype=np.float64)
        dims = tuple(getattr(var, "dimensions", ()))
    if data.ndim != 3:
        raise ValueError(f"Expected TSOI to be 3-D in {path}, got shape {data.shape}")
    dim_l = tuple(str(d).lower() for d in dims)
    if "time" in dim_l and "levgrnd" in dim_l:
        time_axis = dim_l.index("time")
        lev_axis = dim_l.index("levgrnd")
        grid_axis = next((i for i, d in enumerate(dim_l) if d in {"lndgrid", "gridcell", "cell"}), None)
        if grid_axis is None:
            grid_axis = next(i for i in range(3) if i not in (time_axis, lev_axis))
        data = np.moveaxis(data, (time_axis, grid_axis, lev_axis), (0, 1, 2))
    else:
        # Native ELM h0 convention is (time, levgrnd, lndgrid).
        data = np.moveaxis(data, (0, 2, 1), (0, 1, 2))
    if data.shape[1] == ncell:
        return data
    if data.shape[1] == 1:
        return np.broadcast_to(data, (data.shape[0], ncell, data.shape[2])).copy()
    raise ValueError(
        f"ELM TSOI lndgrid dimension {data.shape[1]} does not match replay ncell={ncell}; "
        "only matching-grid or single-cell ELM history is supported."
    )


def _read_elm_soil_water_history(
    path: Path,
    var_name: str,
    ncell: int,
) -> np.ndarray:
    """Read ELM SOILLIQ/SOILICE as (time, ncell, levgrnd)."""
    with Dataset(path) as ds:
        if var_name not in ds.variables:
            raise KeyError(f"{path} does not contain {var_name}")
        var = ds.variables[var_name]
        data = np.ma.asarray(var[:], dtype=np.float64).filled(np.nan)
        dims = tuple(str(d).lower() for d in getattr(var, "dimensions", ()))
    if data.ndim != 3:
        raise ValueError(f"Expected {var_name} to be 3-D in {path}, got shape {data.shape}")
    if "time" in dims and "levgrnd" in dims:
        time_axis = dims.index("time")
        lev_axis = dims.index("levgrnd")
        grid_axis = next(
            (i for i, d in enumerate(dims) if d in {"lndgrid", "gridcell", "cell"}),
            None,
        )
        if grid_axis is None:
            grid_axis = next(i for i in range(3) if i not in (time_axis, lev_axis))
        data = np.moveaxis(data, (time_axis, grid_axis, lev_axis), (0, 1, 2))
    else:
        data = np.moveaxis(data, (0, 2, 1), (0, 1, 2))
    if data.shape[1] == ncell:
        return data
    if data.shape[1] == 1:
        return np.broadcast_to(data, (data.shape[0], ncell, data.shape[2])).copy()
    raise ValueError(
        f"ELM {var_name} lndgrid dimension {data.shape[1]} does not match replay "
        f"ncell={ncell}; only matching-grid or single-cell ELM history is supported."
    )


def _read_elm_scalar_history(path: Path, var_name: str, ncell: int) -> np.ndarray:
    """Read a scalar ELM h0 history variable as ``(time, cell)``."""
    var_name = str(var_name)
    with Dataset(path, "r") as ds:
        if var_name not in ds.variables:
            raise KeyError(f"{path} does not contain {var_name}")
        var = ds.variables[var_name]
        data = np.ma.asarray(var[:], dtype=np.float64).filled(np.nan)
        dims = tuple(str(d).lower() for d in getattr(var, "dimensions", ()))
    if data.ndim == 1:
        data = data[:, None]
    elif data.ndim == 2:
        if "time" in dims:
            time_axis = dims.index("time")
            grid_axis = next((i for i, d in enumerate(dims) if d in {"lndgrid", "gridcell", "cell"}), None)
            if grid_axis is None:
                grid_axis = 1 - time_axis
            data = np.moveaxis(data, (time_axis, grid_axis), (0, 1))
        else:
            data = np.moveaxis(data, (0, 1), (0, 1))
    else:
        raise ValueError(f"Expected {var_name} to be 1-D or 2-D in {path}, got shape {data.shape}")
    if data.shape[1] == ncell:
        return data
    if data.shape[1] == 1:
        return np.broadcast_to(data, (data.shape[0], ncell)).copy()
    raise ValueError(
        f"ELM {var_name} lndgrid dimension {data.shape[1]} does not match replay ncell={ncell}; "
        "only matching-grid or single-cell ELM history is supported."
    )


def _elm_history_override_window(
    *,
    sample_count: int,
    year: int,
    month: int,
    n1h: int,
    calendar: str,
) -> tuple[int, int, int]:
    """Return records/day, lag-aligned start record, and records needed."""
    days_in_year = sum(_days_in_month(year, value, calendar) for value in range(1, 13))
    records_per_day_f = float(sample_count) / float(days_in_year)
    records_per_day = int(round(records_per_day_f))
    if records_per_day not in (1, 24) or not np.isclose(
        records_per_day_f,
        records_per_day,
        rtol=0.0,
        atol=1.0e-6,
    ):
        raise ValueError(
            "ELM override history must contain either daily or hourly records; "
            f"found {sample_count} records for {days_in_year} days"
        )
    day_offset = _day_of_year_start(year, month, calendar) - 1
    # ELM history is interval-end stamped. Record 0 covers the interval ending
    # at the run start, so replay hour/day 0 aligns with history record 1.
    start_record = day_offset * records_per_day + 1
    records_needed = int(math.ceil(int(n1h) * records_per_day / 24.0))
    return records_per_day, start_record, records_needed


def _elm_history_records_to_hourly(
    records: np.ndarray,
    *,
    records_per_day: int,
    n1h: int,
) -> np.ndarray:
    if records_per_day == 24:
        return records[:int(n1h)]
    return np.repeat(records, 24, axis=0)[:int(n1h)]


def _load_elm_history_soil_temperature_override(
    path: Path,
    *,
    year: int,
    month: int,
    ncell: int,
    nlev: int,
    n1h: int,
    calendar: str,
) -> np.ndarray:
    """Load lag-aligned daily or hourly ELM TSOI for replay steps."""
    current = _read_elm_tsoi_history(_find_elm_history_file(path, year), ncell)
    records_per_day, start_record, records_needed = _elm_history_override_window(
        sample_count=current.shape[0],
        year=year,
        month=month,
        n1h=n1h,
        calendar=calendar,
    )
    pieces = [current]
    if start_record + records_needed > current.shape[0]:
        pieces.append(_read_elm_tsoi_history(_find_elm_history_file(path, year + 1), ncell))
    records = np.concatenate(pieces, axis=0)[
        start_record:start_record + records_needed, :, :
    ]
    if records.shape[0] < records_needed:
        raise ValueError(
            f"ELM TSOI override has only {records.shape[0]} records for "
            f"{year}-{month:02d}; need {records_needed}"
        )
    if records.shape[2] < 1:
        raise ValueError("ELM TSOI override contains no soil levels")
    if records.shape[2] < nlev:
        pad = np.broadcast_to(
            records[:, :, -1:],
            (records.shape[0], records.shape[1], nlev - records.shape[2]),
        )
        records = np.concatenate([records, pad], axis=2)
    elif records.shape[2] > nlev:
        records = records[:, :, :nlev]
    hourly = _elm_history_records_to_hourly(
        records,
        records_per_day=records_per_day,
        n1h=n1h,
    )
    return np.asarray(hourly, dtype=np.float32)


def _load_elm_history_soil_moisture_override(
    path: Path,
    *,
    year: int,
    month: int,
    ncell: int,
    nlev: int,
    n1h: int,
    calendar: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Load lag-aligned daily or hourly ELM SOILLIQ/SOILICE."""

    def _load_year(year_i: int) -> tuple[np.ndarray, np.ndarray]:
        history_file = _find_elm_history_file(path, year_i)
        return (
            _read_elm_soil_water_history(history_file, "SOILLIQ", ncell),
            _read_elm_soil_water_history(history_file, "SOILICE", ncell),
        )

    liquid_current, ice_current = _load_year(year)
    records_per_day, start_record, records_needed = _elm_history_override_window(
        sample_count=liquid_current.shape[0],
        year=year,
        month=month,
        n1h=n1h,
        calendar=calendar,
    )
    liquid_pieces = [liquid_current]
    ice_pieces = [ice_current]
    if start_record + records_needed > liquid_current.shape[0]:
        liquid_next, ice_next = _load_year(year + 1)
        liquid_pieces.append(liquid_next)
        ice_pieces.append(ice_next)
    liquid_records = np.concatenate(liquid_pieces, axis=0)[
        start_record:start_record + records_needed, :, :
    ]
    ice_records = np.concatenate(ice_pieces, axis=0)[
        start_record:start_record + records_needed, :, :
    ]
    if liquid_records.shape[0] < records_needed or ice_records.shape[0] < records_needed:
        raise ValueError(
            f"ELM soil-moisture override has only {liquid_records.shape[0]} "
            f"records for {year}-{month:02d}; need {records_needed}"
        )
    if liquid_records.shape[2] < nlev or ice_records.shape[2] < nlev:
        raise ValueError(
            "ELM SOILLIQ/SOILICE override does not contain the requested "
            f"{nlev} soil levels"
        )
    liquid_hourly = _elm_history_records_to_hourly(
        liquid_records[:, :, :nlev],
        records_per_day=records_per_day,
        n1h=n1h,
    )
    ice_hourly = _elm_history_records_to_hourly(
        ice_records[:, :, :nlev],
        records_per_day=records_per_day,
        n1h=n1h,
    )
    return (
        np.asarray(liquid_hourly, dtype=np.float32),
        np.asarray(ice_hourly, dtype=np.float32),
    )


def _load_elm_history_scalar_hourly_override(
    path: Path,
    *,
    var_name: str,
    aliases: tuple[str, ...] | list[str] | None = None,
    year: int,
    month: int,
    ncell: int,
    n1h: int,
    calendar: str,
) -> np.ndarray:
    """Load a lag-aligned daily or hourly ELM scalar history field."""
    names = tuple(dict.fromkeys((str(var_name), *(str(name) for name in (aliases or ())))))

    def _read_any(path_i: Path) -> np.ndarray:
        last_error: Exception | None = None
        for name in names:
            try:
                return _read_elm_scalar_history(path_i, name, ncell)
            except KeyError as exc:
                last_error = exc
        raise KeyError(
            f"{path_i} does not contain any of: {', '.join(names)}"
        ) from last_error

    current = _read_any(_find_elm_history_file(path, year))
    records_per_day, start_record, records_needed = _elm_history_override_window(
        sample_count=current.shape[0],
        year=year,
        month=month,
        n1h=n1h,
        calendar=calendar,
    )
    pieces = [current]
    if start_record + records_needed > current.shape[0]:
        pieces.append(_read_any(_find_elm_history_file(path, year + 1)))
    records = np.concatenate(pieces, axis=0)[
        start_record:start_record + records_needed, :
    ]
    if records.shape[0] < records_needed:
        raise ValueError(
            f"ELM {var_name} override has only {records.shape[0]} records for "
            f"{year}-{month:02d}; need {records_needed}"
        )
    if not np.all(np.isfinite(records)):
        raise ValueError(f"ELM {var_name} override contains missing/non-finite values for {year}-{month:02d}")
    hourly = _elm_history_records_to_hourly(
        records,
        records_per_day=records_per_day,
        n1h=n1h,
    )
    return np.asarray(hourly, dtype=np.float32)


def zeng2001_rootfr(roota: float, rootb: float, zi_interfaces: np.ndarray) -> np.ndarray:
    """Root fraction per soil layer using the Zeng (2001) formula (ELM default).

    ``zi_interfaces`` must be ``[0.0, zi[0], zi[1], ..., zi[nlevsoi-1]]``
    (i.e. the surface interface prepended to the lower-interface depths returned
    by :func:`elm_soil_grid`).  Returns a normalised ``(nlevsoi,)`` array.
    """
    nlevsoi = len(zi_interfaces) - 1
    if roota == 0.0 and rootb == 0.0:
        # Bare soil / not_vegetated: transpiration is zero regardless, use uniform.
        return np.ones(nlevsoi, dtype=np.float64) / nlevsoi
    rootfr = np.empty(nlevsoi, dtype=np.float64)
    for lev in range(nlevsoi - 1):
        zi_up = zi_interfaces[lev]
        zi_lo = zi_interfaces[lev + 1]
        rootfr[lev] = 0.5 * (
            np.exp(-roota * zi_up) + np.exp(-rootb * zi_up)
            - np.exp(-roota * zi_lo) - np.exp(-rootb * zi_lo)
        )
    # Last layer has an open bottom; CDF evaluated at upper interface only.
    rootfr[-1] = 0.5 * (
        np.exp(-roota * zi_interfaces[-2]) + np.exp(-rootb * zi_interfaces[-2])
    )
    total = rootfr.sum()
    if total > 1.0e-12:
        rootfr /= total
    return rootfr


def root_parameter_index_for_pft(pft_id: int, nparam: int) -> int:
    """Parameter-file index used for roota/rootb in native ELM replay."""
    root_idx = int(pft_id)
    if root_idx < 0 or root_idx >= nparam:
        raise IndexError(
            f"root parameter index {root_idx} for PFT {pft_id} is outside "
            f"the parameter-file PFT dimension [0, {nparam - 1}]"
        )
    return root_idx


def build_regional_static_vectorized(
    surf: dict,
    *,
    pft_id: int,
    rootfr_pft: np.ndarray,   # (nlev_hydro,) – Zeng 2001 root fractions for this PFT
    smpso: float | None,
    smpsc: float | None,
    flnr: float | np.ndarray,
    mbbopt: float | np.ndarray,
    leafcn: float,
    slatop: float,
    z0mr: float | None,
    rholvis: float,
    rholnir: float,
    taulvis: float,
    taulnir: float,
    rhosvis: float,
    rhosnir: float,
    tausvis: float,
    tausnir: float,
    xl: float,
    rsub_top_globalmax: float = 10.0,
    device: "torch.device",
    dtype: "torch.dtype",
    nlev_hydro: int = 10,
    nlev_thermal: int = 15,
) -> "CoupledStaticParameters":
    """Build a vectorised ``CoupledStaticParameters`` for the full domain.

    Soil hydraulic parameters (watsat, bsw, sucsat, hksat) are stored as
    ``(ncell, 1, nlev)`` tensors so they broadcast correctly against
    ``(ncell, nens, nlev)`` state tensors in the torch physics kernels.

    Per-cell scalar fields (wtfact, fover, h2osfc_thresh, micro_sigma, n_melt)
    are stored as ``(ncell, 1)`` tensors / arrays.
    """
    from .coupled_surface_hydrology_soiltemp import (
        SoilTextureProfile,
        hydraulic_params_from_texture,
        thermal_params_from_texture,
        thermal_params_from_texture_torch,
        micro_sigma_from_slope,
        h2osfc_threshold_from_micro_sigma,
        _extend_rootfr_to_nlev,
    )

    ncell = surf["sand"].shape[0]
    dz_h, z_h, zi_h = elm_soil_grid(nlevsoi=nlev_hydro)
    dz_t, z_t, zi_t = elm_soil_grid(nlevsoi=nlev_thermal)

    # --- Per-cell soil hydraulic parameters (ncell, nlev_hydro) ---
    sand_h    = surf["sand"].astype(np.float64)      # (ncell, nlev_hydro)
    clay_h    = surf["clay"].astype(np.float64)
    org_frac_h = surf["organic_frac"].astype(np.float64)

    texture_h = SoilTextureProfile(
        sand_pct=sand_h,
        clay_pct=clay_h,
        organic_frac=org_frac_h,
    )
    rootfr_h = _extend_rootfr_to_nlev(rootfr_pft, nlev_hydro)
    hydro_np = hydraulic_params_from_texture(
        dz=dz_h, z=z_h, zi=zi_h,
        texture=texture_h,
        rootfr=rootfr_h,
        topo_slope=surf["slope"].astype(np.float64),
    )
    # Per-cell scalar hydro fields.
    fmax   = surf["fmax"].astype(np.float64)           # (ncell,)
    slope  = surf["slope"].astype(np.float64)
    micro_sigma_arr = np.array(
        [micro_sigma_from_slope(float(s)) for s in slope], dtype=np.float64
    )  # (ncell,)
    h2osfc_thresh_arr = np.array(
        [h2osfc_threshold_from_micro_sigma(float(ms)) for ms in micro_sigma_arr], dtype=np.float64
    )  # (ncell,)

    # Override per-cell hydraulic fields so both numpy and torch physics
    # broadcast correctly against (ncell, nens, nlev) state arrays.
    hydro_np.watsat = hydro_np.watsat[:, None, :]
    hydro_np.sucsat = hydro_np.sucsat[:, None, :]
    hydro_np.bsw    = hydro_np.bsw[:, None, :]
    hydro_np.hksat  = hydro_np.hksat[:, None, :]

    # Override per-cell scalar fields in hydro_np with (ncell, 1) arrays.
    # These broadcast to (ncell, nens) in both numpy and torch kernels.
    hydro_np.wtfact         = fmax.reshape(ncell, 1)
    hydro_np.h2osfc_thresh  = h2osfc_thresh_arr.reshape(ncell, 1)
    hydro_np.topo_slope     = slope.reshape(ncell, 1)
    hydro_np.fover          = np.full((ncell, 1), 0.5, dtype=np.float64)
    hydro_np.rsub_top_globalmax = np.full((ncell, 1), rsub_top_globalmax, dtype=np.float64)
    # fdrain and pc remain at their scalar defaults.

    # --- Torch hydro params with (ncell, 1, nlev) 3-D tensors ---
    from .elm_hydrology_torch import SoilWaterCoreParameters
    hydro_torch = SoilWaterCoreParameters(
        dz       = torch.as_tensor(dz_h, device=device, dtype=dtype),
        z        = torch.as_tensor(z_h,  device=device, dtype=dtype),
        zi       = torch.as_tensor(zi_h, device=device, dtype=dtype),
        watsat   = torch.as_tensor(hydro_np.watsat, device=device, dtype=dtype),   # (ncell,1,nlev)
        sucsat   = torch.as_tensor(hydro_np.sucsat, device=device, dtype=dtype),
        bsw      = torch.as_tensor(hydro_np.bsw,    device=device, dtype=dtype),
        hksat    = torch.as_tensor(hydro_np.hksat,  device=device, dtype=dtype),
        smpmin   = torch.as_tensor(np.array([[-1.0e8]]), device=device, dtype=dtype),
        wtfact   = torch.as_tensor(fmax.reshape(ncell, 1), device=device, dtype=dtype),
        fover    = torch.as_tensor(hydro_np.fover, device=device, dtype=dtype),
        fdrain   = torch.as_tensor(np.full((ncell, 1), 2.5), device=device, dtype=dtype),
        h2osfc_thresh = torch.as_tensor(h2osfc_thresh_arr.reshape(ncell, 1), device=device, dtype=dtype),
        pc       = torch.as_tensor(np.full((ncell, 1), 0.4), device=device, dtype=dtype),
        topo_slope = torch.as_tensor(slope.reshape(ncell, 1), device=device, dtype=dtype),
        rsub_top_globalmax=torch.as_tensor(
            hydro_np.rsub_top_globalmax, device=device, dtype=dtype
        ),
    )

    # --- Per-cell soil thermal parameters (ncell, nlev_thermal) ---
    sand_t    = np.repeat(
        np.pad(sand_h, ((0, 0), (0, max(0, nlev_thermal - nlev_hydro))), mode="edge"),
        1, axis=0,
    )[:, :nlev_thermal]
    clay_t    = np.repeat(
        np.pad(clay_h, ((0, 0), (0, max(0, nlev_thermal - nlev_hydro))), mode="edge"),
        1, axis=0,
    )[:, :nlev_thermal]
    org_frac_t = np.repeat(
        np.pad(org_frac_h, ((0, 0), (0, max(0, nlev_thermal - nlev_hydro))), mode="edge"),
        1, axis=0,
    )[:, :nlev_thermal]
    texture_t = SoilTextureProfile(
        sand_pct=sand_t, clay_pct=clay_t, organic_frac=org_frac_t,
    )
    soil_torch = thermal_params_from_texture_torch(
        texture=texture_t, dz=dz_t, z=z_t, zi=zi_t, nlevbed=nlev_hydro,
        device=device, dtype=dtype,
    )
    # When texture_t is (ncell, nlev), thermal_params_from_texture_torch returns
    # parameter tensors of shape (ncell, nlev).  _broadcast_layers in the soil
    # temperature kernel expects either (nlev,) or (..., 1, nlev) so it can
    # broadcast to (ncell, nens, nlev).  Unsqueeze to (ncell, 1, nlev) here.
    from dataclasses import replace as _dc_replace
    from .elm_soil_temperature_torch import SoilThermalParametersTorch as _STP
    def _unsqueeze_if_2d(t: "torch.Tensor") -> "torch.Tensor":
        return t.unsqueeze(1) if t.ndim == 2 else t
    soil_torch = _STP(
        dz    = soil_torch.dz,
        z     = soil_torch.z,
        zi    = soil_torch.zi,
        watsat= _unsqueeze_if_2d(soil_torch.watsat),
        bsw   = _unsqueeze_if_2d(soil_torch.bsw),
        sucsat= _unsqueeze_if_2d(soil_torch.sucsat),
        tkmg  = _unsqueeze_if_2d(soil_torch.tkmg),
        tkdry = _unsqueeze_if_2d(soil_torch.tkdry),
        csol  = _unsqueeze_if_2d(soil_torch.csol),
        nlevbed = soil_torch.nlevbed,
    )
    # thermal_params_from_texture (numpy) only accepts 1D texture.  Build a
    # domain-mean placeholder — it is never called in the torch-backend regional path.
    mean_texture_t = SoilTextureProfile(
        sand_pct    = sand_t.mean(axis=0),
        clay_pct    = clay_t.mean(axis=0),
        organic_frac= org_frac_t.mean(axis=0),
    )
    soil_np = thermal_params_from_texture(
        dz=dz_t, z=z_t, zi=zi_t, texture=mean_texture_t, nlevbed=nlev_hydro
    )

    # --- Rootfr for full thermal nlev ---
    rootfr_full = _extend_rootfr_to_nlev(rootfr_pft, nlev_thermal)
    rootfr_t   = torch.as_tensor(np.maximum(rootfr_full, 0.0), device=device, dtype=dtype)
    rootfr_norm_t = rootfr_t / torch.clamp(rootfr_t.sum(), min=1.0e-12)

    # --- Per-cell microtopography fields ---
    micro_sigma_field = micro_sigma_arr.reshape(ncell, 1)
    std_elev = surf["std_elev"].astype(np.float64)
    n_melt_field = (200.0 / np.maximum(10.0, std_elev)).reshape(ncell, 1)

    from .coupled_surface_hydrology_soiltemp import CoupledStaticParameters
    static = CoupledStaticParameters(
        rootfr             = rootfr_full,
        hydrology_numpy    = hydro_np,
        hydrology_torch    = hydro_torch,
        soil_temperature_numpy  = soil_np,
        soil_temperature_torch  = soil_torch,
        micro_sigma        = micro_sigma_field,
        pc                 = 0.4,
        topo_slope         = None,
        rootfr_torch       = rootfr_t,
        rootfr_norm_torch  = rootfr_norm_t,
        pft_type           = float(pft_id),
        smpso              = smpso,
        smpsc              = smpsc,
        soil_color         = torch.as_tensor(surf["soil_color"].reshape(ncell, 1), device=device, dtype=dtype),
        leafcn             = leafcn,
        slatop             = slatop,
        flnr               = flnr,
        mbbopt             = mbbopt,
        z0mr               = z0mr,
        rholvis            = rholvis,
        rholnir            = rholnir,
        taulvis            = taulvis,
        taulnir            = taulnir,
        rhosvis            = rhosvis,
        rhosnir            = rhosnir,
        tausvis            = tausvis,
        tausnir            = tausnir,
        xl                 = xl,
    )
    static.n_melt = n_melt_field
    return static


def _regional_ood_tensors(
    bundle,
    *,
    device: "torch.device",
    dtype: "torch.dtype",
) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    """Feature indices and standardization tensors for MLP input OOD maps."""
    x_mean = np.asarray(bundle.x_mean, dtype=np.float64)
    x_std = np.asarray(bundle.x_std, dtype=np.float64)
    tol = 10.0 * np.finfo(np.float32).eps * np.maximum(np.abs(x_mean), 1.0)
    idx = np.nonzero(x_std > tol)[0].astype(np.int64)
    if idx.size == 0:
        empty_i = torch.empty(0, device=device, dtype=torch.long)
        empty_f = torch.empty(0, device=device, dtype=dtype)
        return empty_i, empty_f, empty_f
    return (
        torch.as_tensor(idx, device=device, dtype=torch.long),
        torch.as_tensor(x_mean[idx], device=device, dtype=dtype),
        torch.as_tensor(1.0 / x_std[idx], device=device, dtype=dtype),
    )


def _mlp_input_ood_stats_torch(
    features: "torch.Tensor",
    idx: "torch.Tensor",
    mean: "torch.Tensor",
    inv_std: "torch.Tensor",
    *,
    threshold: float,
    chunk_size: int = 32,
) -> tuple["torch.Tensor", "torch.Tensor"]:
    """Return per-sample max |z| and fraction of standardized inputs beyond threshold."""
    batch_shape = features.shape[:-1]
    if idx.numel() == 0:
        zeros = torch.zeros(batch_shape, device=features.device, dtype=features.dtype)
        return zeros, zeros
    max_abs_z = torch.zeros(batch_shape, device=features.device, dtype=features.dtype)
    exceed = torch.zeros(batch_shape, device=features.device, dtype=features.dtype)
    n_used = 0
    for start in range(0, int(idx.numel()), chunk_size):
        stop = min(start + chunk_size, int(idx.numel()))
        vals = torch.index_select(features, -1, idx[start:stop])
        z_abs = torch.abs((vals - mean[start:stop]) * inv_std[start:stop])
        max_abs_z = torch.maximum(max_abs_z, z_abs.amax(dim=-1))
        exceed = exceed + (z_abs > threshold).sum(dim=-1).to(dtype=features.dtype)
        n_used += stop - start
    return max_abs_z, exceed / max(n_used, 1)


_DEFAULT_REGIONAL_OOD_FEATURE_MAPS = (
    "forc_pbot",
    "forc_vpd",
    "zwt_col",
    "btran",
    "frac_h2osfc",
    "t_h2osfc",
    "t_grnd",
    "h2ocan",
    "soil_color",
)
_ALL_REGIONAL_OOD_FEATURE_MAPS_SENTINEL = "__all__"


def _parse_ood_feature_map_names(value: str | None) -> tuple[str, ...]:
    """Parse a comma-separated feature list for regional OOD maps."""
    if value is None:
        return (_ALL_REGIONAL_OOD_FEATURE_MAPS_SENTINEL,)
    names = tuple(
        dict.fromkeys(
            item.strip()
            for item in str(value).split(",")
            if item.strip()
        )
    )
    if len(names) == 1 and names[0].lower() in {"none", "off", "false", "0"}:
        return ()
    if any(name.lower() == "all" for name in names):
        return (_ALL_REGIONAL_OOD_FEATURE_MAPS_SENTINEL,)
    return names


_NUMRAD_SHORTWAVE_DIAGNOSTICS = frozenset(
    {"albgrd_mean", "albgri_mean", "albd_mean", "albi_mean"}
)


def _reduce_regional_shortwave_diagnostic(
    key: str,
    value: "torch.Tensor",
    *,
    pft_weights: "torch.Tensor",
    universal_pft_weights: "torch.Tensor",
    universal_pft_count: int,
    device: "torch.device",
    dtype: "torch.dtype",
) -> "torch.Tensor":
    """Area-blend PFT-dependent radiation fields and ensemble-average them.

    ``surface_shortwave_diagnostics_torch`` intentionally returns a mixture of
    PFT-dependent fields (for example reflected radiation) and PFT-invariant
    forcing/ground fields.  Their respective layouts are
    ``(npft,ncell,nens[,2])`` and ``(ncell,nens[,2])``.  Treating every field as
    PFT-major fails whenever ``ncell != npft`` and is especially visible in
    large regional runs.
    """
    value_t = (
        value.to(device=device, dtype=dtype)
        if isinstance(value, torch.Tensor)
        else torch.as_tensor(value, device=device, dtype=dtype)
    )
    is_numrad = key in _NUMRAD_SHORTWAVE_DIAGNOSTICS
    pft_rank = 4 if is_numrad else 3
    cell_rank = pft_rank - 1
    if value_t.ndim == pft_rank:
        if value_t.shape[0] == int(universal_pft_count):
            weights = universal_pft_weights
        elif value_t.shape[0] == pft_weights.shape[0]:
            weights = pft_weights
        else:
            raise RuntimeError(
                f"{key} has PFT-major shape {tuple(value_t.shape)}, but its first "
                f"dimension matches neither npft={pft_weights.shape[0]} nor "
                f"nuniversal={int(universal_pft_count)}"
            )
        while weights.ndim < value_t.ndim:
            weights = weights.unsqueeze(-1)
        reduced = torch.sum(weights * value_t, dim=0)
    elif value_t.ndim == cell_rank:
        reduced = value_t
    else:
        expected = (
            "(npft,ncell,nens,2) or (ncell,nens,2)"
            if is_numrad else "(npft,ncell,nens) or (ncell,nens)"
        )
        raise RuntimeError(
            f"{key} shortwave diagnostic has shape {tuple(value_t.shape)}; "
            f"expected {expected}"
        )

    if is_numrad:
        if reduced.ndim == 3 and reduced.shape[-1] == 2:
            return reduced.mean(1)
        if reduced.ndim == 2 and reduced.shape[-1] == 2:
            return reduced
    else:
        if reduced.ndim == 2:
            return reduced.mean(1)
        if reduced.ndim == 1:
            return reduced
    raise RuntimeError(
        f"{key} reduced shortwave diagnostic has unsupported shape "
        f"{tuple(reduced.shape)}"
    )


@torch.inference_mode()
def run_regional_pass(
    *,
    surf: dict,
    pft_ids: list[int],             # active PFT indices (0-based into PCT_NAT_PFT)
    pft_weights: np.ndarray,        # (npft, ncell) normalised area fractions
    pft_bundles: list,              # list[SurfaceModelBundle]
    pft_statics: list,              # list[CoupledStaticParameters]  (per-PFT)
    shared_static,                  # CoupledStaticParameters with per-cell soil params
    met_3h: "torch.Tensor",         # (nstep, ncell, 7) already on device
    year: int,
    month: int,
    met_input_step_hours: int = 3,
    met_padding_steps: int = 0,
    elm_stream_met_methods: bool = False,
    calendar: str = "gregorian",
    dt: float = 3600.0,
    device: "torch.device",
    dtype: "torch.dtype",
    mlp_dtype: "torch.dtype | None" = None,
    hydrology_backend: str = "torch",
    soil_temperature_backend: str = "torch",
    snow_model_enabled: bool = True,
    snow_scalar_only: bool = False,
    blend_partial_snow_soil_temperature: bool = False,
    co2_ppm: float = _CO2_PPM_FALLBACK,
    initial_state=None,             # if None, cold start
    profile_sections: "dict[str, float] | None" = None,
    precomputed: "dict | None" = None,  # output of _precompute_month_cpu; avoids recompute
    nens: int = 1,                  # ensemble size; flnr/mbbopt randomised when > 1
    ensemble_seed: int = 42,
    ensemble_flnr_factors: tuple[float, ...] | list[float] | None = None,
    ensemble_mbbopt_factors: tuple[float, ...] | list[float] | None = None,
    step_callback=None,             # optional callable(step_i, last_result, blended, tbot, dayl)
    carbon_model=None,              # optional OnlineDailyCarbonModel; advances once per day and feeds back LAI
    ood_diagnostics_stride: int = 0,
    ood_threshold: float = 4.0,
    ood_feature_map_names: tuple[str, ...] | list[str] | None = _DEFAULT_REGIONAL_OOD_FEATURE_MAPS,
    progress_stride: int = 24,
    record_diag_maps: bool = True,
    record_daily_diag_maps: bool = False,
    record_hourly_diag_maps: bool = False,
    record_day_night_flux_maps: bool = False,
    day_night_fsds_threshold: float = 20.0,
    safety_clamps: bool = True,
    temperature_min_k: float = 180.0,
    temperature_max_k: float = 330.0,
    photosynthesis_max: float = 100.0,
    thermal_target_z_clamp: float = 3.0,
    soil_energy_sabg_source: str = "learned",
    physics_step_hours: float = 1.0,
    universal_mlp_batch: bool = False,
    sparse_universal_mlp_batch: bool = True,
    sparse_individual_feature_assembly: bool = False,
    neutral_patch0_flnr: float | None = None,
    regional_lai_scale: float = 1.0,
    regional_sai_scale: float = 1.0,
    regional_lai_add: float = 0.0,
    regional_lai_add_months: tuple[int, ...] | list[int] | None = None,
    regional_lai_add_include_bare: bool = False,
    regional_trait_ensemble_size: int = 1,
    regional_trait_chunk_size: int = 20,
    regional_trait_sigma: float = 2.0,
    regional_trait_relative_sd: float = 0.20,
    regional_trait_feature_names: tuple[str, ...] | list[str] | None = None,
    soil_temperature_override: np.ndarray | None = None,
    ground_temperature_override: np.ndarray | None = None,
    soil_liquid_override: np.ndarray | None = None,
    soil_ice_override: np.ndarray | None = None,
    ground_heat_flux_override: np.ndarray | None = None,
    ground_heat_flux_override_targets: tuple[str, ...] | list[str] | None = None,
    zwt_override: np.ndarray | None = None,
    precip_multiplier: np.ndarray | None = None,
    max_hours: int | None = None,
) -> tuple:
    """Run one month of the full-domain regional replay.

    Per-PFT canopy interception, MLP, and canopy evaporation are applied over
    all cells simultaneously; outputs are blended by the cell-specific area
    weights; a single shared column physics step is then run.

    Returns ``(final_state, sanity, timing_info)`` where ``timing_info`` is a
    dict with keys ``"n3h"``, ``"n1h"``, ``"elapsed_s"``.
    """
    if soil_energy_sabg_source not in {"learned", "diagnostic"}:
        raise ValueError(
            "soil_energy_sabg_source must be 'learned' or 'diagnostic', got "
            f"{soil_energy_sabg_source!r}"
        )
    from .coupled_surface_hydrology_soiltemp import (
        apply_canopy_interception, apply_canopy_evaporation,
        predict_surface_targets, AtmosphericForcingInputs,
        VegetationSurfaceInputs, SurfaceGeometryInputs,
        build_surface_features_from_inputs_torch,
        precompute_surface_feature_state_torch,
        coupled_step, coupled_state_to_device,
        distribute_root_uptake,
        sabg_snow_from_state_torch,
        _absorbed_ground_shortwave_torch,
        _snow_albedo_from_grain_radius_torch,
        surface_shortwave_diagnostics_torch,
        coupled_t_grnd_from_state_torch,
        coupled_soilbeta_from_state_torch,
        coupled_btran_from_state_torch,
        TFRZ,
        SNOW_MIN_ACTIVE_DEPTH,
        SNW_RDS_MIN,
        DENH2O,
        DENICE,
        SECSPDAY,
        _zero_snow_state,
        _index_copy_feature_columns_torch,
        _batch_update_feature_columns_torch,
        _soil_profile_feature_indices,
    )
    from .elm_hydrology_torch import SoilHydrologyTorchWorkspace
    from .elm_soil_temperature_torch import SoilTemperatureTorchWorkspace

    mlp_dtype = dtype if mlp_dtype is None else mlp_dtype
    ncell, npft = pft_weights.shape[1], len(pft_ids)
    _carbon_day_start = (
        len(getattr(carbon_model, "daily_domain_gpp", ()))
        if carbon_model is not None
        else 0
    )
    _daily_carbon_pool_rows: dict[str, list[np.ndarray]] = {}
    lai_scale = float(regional_lai_scale)
    sai_scale = float(regional_sai_scale)
    lai_add = float(regional_lai_add)
    lai_add_months = tuple(int(m) for m in (regional_lai_add_months or ()))
    if not np.isfinite(lai_scale) or lai_scale < 0.0:
        raise ValueError("--regional-lai-scale must be a finite non-negative value")
    if not np.isfinite(sai_scale) or sai_scale < 0.0:
        raise ValueError("--regional-sai-scale must be a finite non-negative value")
    if not np.isfinite(lai_add):
        raise ValueError("--regional-lai-add must be finite")
    invalid_lai_add_months = [m for m in lai_add_months if m < 1 or m > 12]
    if invalid_lai_add_months:
        raise ValueError(
            "--regional-lai-add-months entries must be in [1, 12], got "
            + ", ".join(str(m) for m in invalid_lai_add_months)
        )
    lai_add_applies = (
        lai_add != 0.0
        and (not lai_add_months or int(month) in set(lai_add_months))
    )
    if lai_scale != 1.0 or sai_scale != 1.0:
        print(f"    scaling prescribed LAI/SAI: LAI x{lai_scale:g}, SAI x{sai_scale:g}", flush=True)
    if lai_add_applies:
        target = "all active PFTs" if regional_lai_add_include_bare else "active non-bare PFTs"
        month_label = f"{int(month):02d}"
        print(f"    adding {lai_add:g} to prescribed LAI for {target} in month {month_label}", flush=True)
    trait_ensemble_size = int(regional_trait_ensemble_size)
    trait_chunk_size = int(regional_trait_chunk_size)
    trait_sigma = float(regional_trait_sigma)
    trait_relative_sd = float(regional_trait_relative_sd)
    if trait_ensemble_size <= 0:
        raise ValueError("--regional-trait-ensemble-size must be positive")
    if trait_chunk_size <= 0:
        raise ValueError("--regional-trait-chunk-size must be positive")
    if not np.isfinite(trait_sigma) or trait_sigma < 0.0:
        raise ValueError("--regional-trait-sigma must be a finite non-negative value")
    if not np.isfinite(trait_relative_sd) or trait_relative_sd < 0.0:
        raise ValueError("--regional-trait-relative-sd must be a finite non-negative value")
    universal_batch_indices: list[int] = []
    universal_batch_index_set: set[int] = set()
    universal_batch_slot_for_pft: dict[int, int] = {}
    universal_ref_bundle = None
    universal_feature_assembly_enabled = False
    universal_fast_feature_assembly = False
    if universal_mlp_batch:
        if not pft_bundles:
            raise ValueError("Universal MLP batching requires at least one model bundle.")
        universal_batch_indices = [
            idx
            for idx, bundle in enumerate(pft_bundles)
            if int(getattr(bundle, "patch_index", pft_ids[idx])) == -1
        ]
        if not universal_batch_indices:
            universal_mlp_batch = False
        else:
            universal_ref_bundle = pft_bundles[universal_batch_indices[0]]
            universal_batch_index_set = set(universal_batch_indices)
            universal_batch_slot_for_pft = {
                p_idx: u_idx for u_idx, p_idx in enumerate(universal_batch_indices)
            }
            universal_feature_assembly_enabled = True
        for idx in universal_batch_indices[1:]:
            bundle = pft_bundles[idx]
            if (
                bundle.input_names != universal_ref_bundle.input_names
                or bundle.target_names != universal_ref_bundle.target_names
            ):
                raise ValueError(
                    "Universal MLP batching requires all active PFTs to use the same model "
                    f"schema; PFT slot {idx} differs."
                )
        if universal_mlp_batch:
            universal_fast_feature_assembly = len(universal_batch_indices) == npft
            if universal_fast_feature_assembly:
                print(
                    f"    universal MLP batching enabled: one forward over {npft} PFTs",
                    flush=True,
                )
            else:
                individual_ids = [
                    str(int(pft_ids[idx]))
                    for idx in range(npft)
                    if idx not in universal_batch_index_set
                ]
                print(
                    f"    hybrid universal replay: patch-specific model(s) for PFT(s) "
                    f"{', '.join(individual_ids)}; universal batch covers "
                    f"{len(universal_batch_indices)} PFT(s)",
                    flush=True,
                )
    if trait_ensemble_size > 1 and not universal_mlp_batch:
        raise ValueError("--regional-trait-ensemble-size > 1 requires --universal")
    # Interpolate met to hourly on device.
    met_1h = interp_met_to_1h_device(   # (n1h, ncell, 7)
        met_3h,
        met_input_step_hours,
        elm_stream_methods=elm_stream_met_methods,
        padding_steps=met_padding_steps,
    )
    if max_hours is not None:
        max_hours_i = int(max_hours)
        if max_hours_i <= 0:
            raise ValueError("max_hours must be positive")
        if max_hours_i < met_1h.shape[0]:
            met_1h = met_1h[:max_hours_i]
    if precip_multiplier is not None:
        precip_multiplier_np = np.asarray(precip_multiplier, dtype=np.float32).reshape(-1)
        if precip_multiplier_np.shape != (ncell,):
            raise ValueError(
                f"precip_multiplier has shape {precip_multiplier_np.shape}, expected {(ncell,)}"
            )
        if not np.all(np.isfinite(precip_multiplier_np)) or np.any(precip_multiplier_np < 0.0):
            raise ValueError("precip_multiplier must be finite and non-negative")
        if not np.allclose(precip_multiplier_np, 1.0):
            precip_multiplier_t = torch.as_tensor(
                precip_multiplier_np,
                device=device,
                dtype=dtype,
            ).reshape(1, ncell)
            met_1h[:, :, _MET_VAR_IDX["PRECTmms"]] = (
                torch.clamp(met_1h[:, :, _MET_VAR_IDX["PRECTmms"]], min=0.0)
                * precip_multiplier_t
            )
            print(
                "    precipitation multiplier: "
                f"min={float(np.min(precip_multiplier_np)):.3f} "
                f"mean={float(np.mean(precip_multiplier_np)):.6f} "
                f"max={float(np.max(precip_multiplier_np)):.3f}",
                flush=True,
            )
    n1h    = met_1h.shape[0]
    soil_temperature_override_t = None
    if soil_temperature_override is not None:
        override_np = np.asarray(soil_temperature_override, dtype=np.float32)
        if override_np.ndim == 2:
            override_np = override_np[:, None, :]
        if override_np.ndim != 3:
            raise ValueError(
                "soil_temperature_override must have shape (hour, cell, lev) "
                f"or (hour, lev); got {override_np.shape}"
            )
        if override_np.shape[0] < n1h:
            raise ValueError(
                f"soil_temperature_override has {override_np.shape[0]} hours but replay needs {n1h}"
            )
        if override_np.shape[1] not in (1, ncell):
            raise ValueError(
                f"soil_temperature_override cell dimension {override_np.shape[1]} "
                f"does not match ncell={ncell}"
            )
        if override_np.shape[1] == 1 and ncell > 1:
            override_np = np.broadcast_to(
                override_np,
                (override_np.shape[0], ncell, override_np.shape[2]),
            ).copy()
        soil_temperature_override_t = torch.as_tensor(
            override_np[:n1h],
            device=device,
            dtype=dtype,
        )
    ground_temperature_override_t = None
    if ground_temperature_override is not None:
        override_np = np.asarray(ground_temperature_override, dtype=np.float32)
        if override_np.ndim == 1:
            override_np = override_np[:, None]
        if override_np.ndim != 2:
            raise ValueError(
                "ground_temperature_override must have shape (hour, cell) or "
                f"(hour,), got {override_np.shape}"
            )
        if override_np.shape[0] < n1h:
            raise ValueError(
                f"ground_temperature_override has {override_np.shape[0]} hours "
                f"but replay needs {n1h}"
            )
        if override_np.shape[1] not in (1, ncell):
            raise ValueError(
                "ground_temperature_override cell dimension "
                f"{override_np.shape[1]} does not match ncell={ncell}"
            )
        if override_np.shape[1] == 1 and ncell > 1:
            override_np = np.broadcast_to(
                override_np,
                (override_np.shape[0], ncell),
            ).copy()
        if not np.all(np.isfinite(override_np[:n1h])):
            raise ValueError("ground_temperature_override must contain finite temperatures")
        ground_temperature_override_t = torch.as_tensor(
            override_np[:n1h],
            device=device,
            dtype=dtype,
        )
    soil_liquid_override_t = None
    soil_ice_override_t = None
    if (soil_liquid_override is None) != (soil_ice_override is None):
        raise ValueError("soil_liquid_override and soil_ice_override must be provided together")
    if soil_liquid_override is not None and soil_ice_override is not None:
        water_overrides = []
        for name, override in (
            ("soil_liquid_override", soil_liquid_override),
            ("soil_ice_override", soil_ice_override),
        ):
            override_np = np.asarray(override, dtype=np.float32)
            if override_np.ndim == 2:
                override_np = override_np[:, None, :]
            if override_np.ndim != 3:
                raise ValueError(
                    f"{name} must have shape (hour, cell, lev) or (hour, lev); "
                    f"got {override_np.shape}"
                )
            if override_np.shape[0] < n1h:
                raise ValueError(
                    f"{name} has {override_np.shape[0]} hours but replay needs {n1h}"
                )
            if override_np.shape[1] not in (1, ncell):
                raise ValueError(
                    f"{name} cell dimension {override_np.shape[1]} does not match ncell={ncell}"
                )
            if override_np.shape[1] == 1 and ncell > 1:
                override_np = np.broadcast_to(
                    override_np,
                    (override_np.shape[0], ncell, override_np.shape[2]),
                ).copy()
            if not np.all(np.isfinite(override_np[:n1h])) or np.any(override_np[:n1h] < 0.0):
                raise ValueError(f"{name} must contain finite non-negative water masses")
            water_overrides.append(
                torch.as_tensor(override_np[:n1h], device=device, dtype=dtype)
            )
        soil_liquid_override_t, soil_ice_override_t = water_overrides
    ground_heat_flux_override_t = None
    ground_heat_flux_override_targets = tuple(
        str(name) for name in (ground_heat_flux_override_targets or ("hs_soil", "hs_top_snow"))
    )
    invalid_ground_heat_targets = [
        name for name in ground_heat_flux_override_targets
        if name not in {"hs_soil", "hs_top_snow", "hs_h2osfc"}
    ]
    if invalid_ground_heat_targets:
        raise ValueError(
            "ground_heat_flux_override_targets entries must be hs_soil, hs_top_snow, or hs_h2osfc; got "
            + ", ".join(invalid_ground_heat_targets)
        )
    if ground_heat_flux_override is not None:
        override_np = np.asarray(ground_heat_flux_override, dtype=np.float32)
        if override_np.ndim == 1:
            override_np = override_np[:, None]
        if override_np.ndim != 2:
            raise ValueError(
                "ground_heat_flux_override must have shape (hour, cell) or (hour,), "
                f"got {override_np.shape}"
            )
        if override_np.shape[0] < n1h:
            raise ValueError(
                f"ground_heat_flux_override has {override_np.shape[0]} hours but replay needs {n1h}"
            )
        if override_np.shape[1] not in (1, ncell):
            raise ValueError(
                f"ground_heat_flux_override cell dimension {override_np.shape[1]} does not match ncell={ncell}"
            )
        if override_np.shape[1] == 1 and ncell > 1:
            override_np = np.broadcast_to(override_np, (override_np.shape[0], ncell)).copy()
        ground_heat_flux_override_t = torch.as_tensor(
            override_np[:n1h],
            device=device,
            dtype=dtype,
        )
    zwt_override_t = None
    if zwt_override is not None:
        override_np = np.asarray(zwt_override, dtype=np.float32)
        if override_np.ndim == 1:
            override_np = override_np[:, None]
        if override_np.ndim != 2:
            raise ValueError(
                "zwt_override must have shape (hour, cell) or (hour,), "
                f"got {override_np.shape}"
            )
        if override_np.shape[0] < n1h:
            raise ValueError(
                f"zwt_override has {override_np.shape[0]} hours but replay needs {n1h}"
            )
        if override_np.shape[1] not in (1, ncell):
            raise ValueError(
                f"zwt_override cell dimension {override_np.shape[1]} does not match ncell={ncell}"
            )
        if not np.all(np.isfinite(override_np[:n1h])):
            raise ValueError("zwt_override contains missing/non-finite values")
        if override_np.shape[1] == 1 and ncell > 1:
            override_np = np.broadcast_to(override_np, (override_np.shape[0], ncell)).copy()
        zwt_override_t = torch.as_tensor(
            override_np[:n1h],
            device=device,
            dtype=dtype,
        )
    if snow_model_enabled:
        with torch.no_grad():
            _snow_possible = (
                (met_1h[:, :, _MET_VAR_IDX["PRECTmms"]] > 1.0e-12)
                & (met_1h[:, :, _MET_VAR_IDX["TBOT"]] < (TFRZ + 2.0))
            )
            _snowfall_possible_steps = torch.any(_snow_possible, dim=1).detach().cpu().numpy()
    else:
        _snowfall_possible_steps = np.zeros(n1h, dtype=bool)

    # Precompute per-cell lat/lon.
    lat_rad  = np.asarray(surf["lat_deg"],  dtype=np.float64) * (np.pi / 180.0)
    lon_deg  = np.asarray(surf["lon_deg"],  dtype=np.float64)

    # Compute doy for the first hour of this month (needed for LAI interp).
    day_of_year_start = _day_of_year_start(year, month, calendar)

    # ------------------------------------------------------------------
    # Coszen / dayl / LAI: use prefetched numpy arrays when available,
    # otherwise compute them here (single-month / cold-start fallback).
    # ------------------------------------------------------------------
    if precomputed is not None:
        _pc = precomputed
        print("    using prefetched coszen/dayl/LAI")
    else:
        print("    precomputing coszen/dayl for all steps ...", end=" ", flush=True)
        _pc = _precompute_month_cpu(surf, year, month, calendar=calendar, n1h=n1h)
        print("done")

    coszen_dev   = torch.as_tensor(_pc["coszen_all"],  device=device, dtype=dtype)
    dayl_dev     = torch.as_tensor(_pc["dayl_all"],    device=device, dtype=dtype)
    prev_dayl_dev = torch.as_tensor(_pc["prev_dayl_all"], device=device, dtype=dtype)
    max_dayl_dev = torch.as_tensor(_pc["max_dayl_np"], device=device, dtype=dtype).unsqueeze(1)
    del _pc["coszen_all"], _pc["dayl_all"], _pc["prev_dayl_all"], _pc["max_dayl_np"]
    if coszen_dev.shape[0] < n1h:
        pc_calendar = precomputed.get("calendar", calendar) if precomputed is not None else calendar
        raise ValueError(
            f"Solar/LAI precompute for {year}-{month:02d} has {coszen_dev.shape[0]} hourly steps, "
            f"but interpolated met has {n1h}. Check --regional-calendar or met file month length "
            f"(precompute calendar={pc_calendar!r})."
        )
    if met_input_step_hours > 1:
        met_interval = (
            met_3h[met_padding_steps : met_3h.shape[0] - met_padding_steps]
            if met_padding_steps
            else met_3h
        )
        met_1h[:, :, _MET_VAR_IDX["FSDS"]] = redistribute_shortwave_to_1h_device(
            met_interval,
            coszen_dev,
            met_input_step_hours,
            n1h=n1h,
        )

    # ------------------------------------------------------------------
    # Precompute LAI/SAI/canopy height for every unique DOY in this month on the GPU.
    # ------------------------------------------------------------------
    print("    uploading LAI/SAI/canopy height to device ...", end=" ", flush=True)
    lai_by_doy: dict[int, "torch.Tensor"] = {
        d: torch.as_tensor(arr, device=device, dtype=dtype)
        for d, arr in _pc["lai_by_doy"].items()
    }
    sai_by_doy: dict[int, "torch.Tensor"] = {
        d: torch.as_tensor(arr, device=device, dtype=dtype)
        for d, arr in _pc["sai_by_doy"].items()
    }
    htop_by_doy: dict[int, "torch.Tensor"] = {
        d: torch.as_tensor(arr, device=device, dtype=dtype)
        for d, arr in _pc["htop_by_doy"].items()
    }
    hbot_by_doy: dict[int, "torch.Tensor"] = {
        d: torch.as_tensor(arr, device=device, dtype=dtype)
        for d, arr in _pc["hbot_by_doy"].items()
    }
    del _pc
    print("done")

    # ------------------------------------------------------------------
    # Precompute h2ocan feature index per PFT (avoid linear scan per step).
    # ------------------------------------------------------------------
    h2ocan_feat_idx = [
        next((i for i, n in enumerate(b.input_names) if n == "h2ocan"), None)
        for b in pft_bundles
    ]

    # Area-weight tensor (npft, ncell, 1) on device for blending.
    w_dev = torch.as_tensor(
        pft_weights.astype(np.float64),  # (npft, ncell)
        device=device, dtype=dtype,
    ).unsqueeze(-1)  # (npft, ncell, 1)

    universal_sparse_mlp_enabled = False
    universal_sparse_pft_idx_t: "torch.Tensor | None" = None
    universal_sparse_weight_pft_idx_t: "torch.Tensor | None" = None
    universal_sparse_cell_idx_t: "torch.Tensor | None" = None
    sparse_cells_by_pft: dict[int, "torch.Tensor"] = {}
    if sparse_universal_mlp_batch or sparse_individual_feature_assembly:
        pft_weight_np_all = np.asarray(pft_weights, dtype=np.float64)
        for p_idx in range(npft):
            cells_np = np.nonzero(pft_weight_np_all[p_idx, :] > 1.0e-12)[0]
            sparse_cells_by_pft[p_idx] = torch.as_tensor(cells_np, device=device, dtype=torch.long)
    if (
        sparse_universal_mlp_batch
        and universal_mlp_batch
        and universal_batch_indices
    ):
        universal_batch_np = np.asarray(universal_batch_indices, dtype=np.int64)
        universal_weight_np = np.asarray(pft_weights[universal_batch_np, :], dtype=np.float64)
        sparse_local_pft, sparse_cell = np.nonzero(universal_weight_np > 1.0e-12)
        sparse_pair_count = int(sparse_cell.size)
        dense_pair_count = int(universal_weight_np.size)
        if 0 < sparse_pair_count < dense_pair_count:
            sparse_weight_pft = universal_batch_np[sparse_local_pft]
            universal_sparse_pft_idx_t = torch.as_tensor(sparse_local_pft, device=device, dtype=torch.long)
            universal_sparse_weight_pft_idx_t = torch.as_tensor(sparse_weight_pft, device=device, dtype=torch.long)
            universal_sparse_cell_idx_t = torch.as_tensor(sparse_cell, device=device, dtype=torch.long)
            universal_sparse_mlp_enabled = True
            print(
                "    universal sparse PFT-cell batching enabled: "
                f"{sparse_pair_count:,}/{dense_pair_count:,} nonzero PFT-cell pairs "
                f"({dense_pair_count / max(sparse_pair_count, 1):.2f}x smaller canopy/forward work)",
                flush=True,
            )
        elif sparse_pair_count == 0:
            print("    universal sparse MLP batching skipped: no nonzero universal PFT-cell pairs", flush=True)

    # Initial state.
    nlev = 15
    nlev_hydro_static = int(shared_static.hydrology_torch.dz.shape[-1])
    if initial_state is None:
        state = regional_initial_state(
            ncell,
            nlev,
            nlevbed=nlev_hydro_static,
            nens=nens,
            device=device,
            dtype=dtype,
        )
    else:
        state = (
            coupled_state_to_device(initial_state, device, dtype)
            if device.type != "cpu"
            else initial_state
        )

    def _active_snow_cell_count(state_check) -> int:
        if not snow_model_enabled:
            return 0
        h2osno = getattr(state_check, "h2osno", None)
        snow_depth = getattr(state_check, "snow_depth", None)
        snow_layers = getattr(state_check, "snow_layers", None)
        if snow_layers is not None and getattr(snow_layers, "snl", None) is not None:
            snl = snow_layers.snl
            if isinstance(snl, torch.Tensor):
                active = snl < 0
                if h2osno is not None:
                    h2osno_t = h2osno.to(device=device, dtype=dtype) if isinstance(h2osno, torch.Tensor) else torch.as_tensor(h2osno, device=device, dtype=dtype)
                    active = active & (h2osno_t > 1.0e-8)
                if snow_depth is not None:
                    depth_t = snow_depth.to(device=device, dtype=dtype) if isinstance(snow_depth, torch.Tensor) else torch.as_tensor(snow_depth, device=device, dtype=dtype)
                    active = active & (depth_t >= 0.01)
                if active.ndim > 1:
                    active = torch.any(active.reshape(active.shape[0], -1), dim=1)
                return int(active.sum().detach().cpu().item())
            active_np = np.asarray(snl, dtype=np.int32) < 0
            if h2osno is not None:
                active_np = active_np & (np.asarray(h2osno, dtype=np.float64) > 1.0e-8)
            if snow_depth is not None:
                active_np = active_np & (np.asarray(snow_depth, dtype=np.float64) >= 0.01)
            if active_np.ndim > 1:
                active_np = np.any(active_np.reshape(active_np.shape[0], -1), axis=1)
            return int(np.count_nonzero(active_np))
        if h2osno is None:
            return 0
        if isinstance(h2osno, torch.Tensor):
            h2osno_t = h2osno.to(device=device, dtype=dtype)
            depth_t = (
                snow_depth.to(device=device, dtype=dtype)
                if isinstance(snow_depth, torch.Tensor)
                else torch.as_tensor(0.0 if snow_depth is None else snow_depth, device=device, dtype=dtype)
            )
            active = (h2osno_t > 1.0e-8) & (depth_t >= 0.01)
            if active.ndim > 1:
                active = torch.any(active.reshape(active.shape[0], -1), dim=1)
            return int(active.sum().detach().cpu().item())
        h2osno_np = np.asarray(h2osno, dtype=np.float64)
        depth_np = (
            np.asarray(snow_depth, dtype=np.float64)
            if snow_depth is not None
            else np.zeros_like(h2osno_np, dtype=np.float64)
        )
        active_np = (h2osno_np > 1.0e-8) & (depth_np >= 0.01)
        if active_np.ndim > 1:
            active_np = np.any(active_np.reshape(active_np.shape[0], -1), axis=1)
        return int(np.count_nonzero(active_np))

    def _state_has_active_snow_bool(state_check) -> bool:
        return _active_snow_cell_count(state_check) > 0

    def _snow_present_cell_count(state_check) -> int:
        if not snow_model_enabled:
            return 0
        h2osno = getattr(state_check, "h2osno", None)
        if h2osno is None:
            return 0
        snow_depth = getattr(state_check, "snow_depth", None)
        if isinstance(h2osno, torch.Tensor):
            h2osno_t = h2osno.to(device=device, dtype=dtype)
            present = h2osno_t > 1.0e-8
            if snow_depth is not None:
                depth_t = (
                    snow_depth.to(device=device, dtype=dtype)
                    if isinstance(snow_depth, torch.Tensor)
                    else torch.as_tensor(snow_depth, device=device, dtype=dtype)
                )
                present = present & (depth_t > 0.0)
            if present.ndim > 1:
                present = torch.any(present.reshape(present.shape[0], -1), dim=1)
            return int(present.sum().detach().cpu().item())
        h2osno_np = np.asarray(h2osno, dtype=np.float64)
        present_np = h2osno_np > 1.0e-8
        if snow_depth is not None:
            present_np = present_np & (np.asarray(snow_depth, dtype=np.float64) > 0.0)
        if present_np.ndim > 1:
            present_np = np.any(present_np.reshape(present_np.shape[0], -1), axis=1)
        return int(np.count_nonzero(present_np))

    hydrology_workspace      = SoilHydrologyTorchWorkspace() if device.type != "cpu" else None
    soil_temperature_workspace = SoilTemperatureTorchWorkspace() if device.type != "cpu" else None

    # Per-PFT working feature buffers. They start at x_mean and are updated
    # in place each timestep; unavailable regional features are left at x_mean.
    feat_bufs = []
    feat_idx_maps = []
    universal_feature_batch = None
    universal_sparse_feature_batch = None
    individual_sparse_feature_bufs: dict[int, "torch.Tensor"] = {}
    from .coupled_surface_hydrology_soiltemp import _bundle_input_index_map
    if universal_mlp_batch:
        p_bundle = universal_ref_bundle
        assert p_bundle is not None
        x_mean_t = (
            p_bundle.x_mean_torch
            if p_bundle.x_mean_torch is not None
               and p_bundle.x_mean_torch.device == device
               and p_bundle.x_mean_torch.dtype == dtype
            else torch.as_tensor(p_bundle.x_mean, device=device, dtype=dtype)
        )
        nuniversal = len(universal_batch_indices)
        universal_feature_batch = torch.empty((nuniversal, ncell, nens, x_mean_t.numel()), device=device, dtype=dtype)
        universal_feature_batch.copy_(x_mean_t)
        shared_idx_map = _bundle_input_index_map(tuple(p_bundle.input_names))
        for p_idx, p_bundle_i in enumerate(pft_bundles):
            if p_idx in universal_batch_index_set:
                feat_bufs.append(universal_feature_batch[universal_batch_slot_for_pft[p_idx]])
                feat_idx_maps.append(shared_idx_map)
                continue
            x_mean_i = (
                p_bundle_i.x_mean_torch
                if p_bundle_i.x_mean_torch is not None
                   and p_bundle_i.x_mean_torch.device == device
                   and p_bundle_i.x_mean_torch.dtype == dtype
                else torch.as_tensor(p_bundle_i.x_mean, device=device, dtype=dtype)
            )
            tmpl = torch.empty((ncell, nens, x_mean_i.numel()), device=device, dtype=dtype)
            tmpl.copy_(x_mean_i)
            feat_bufs.append(tmpl)
            feat_idx_maps.append(_bundle_input_index_map(tuple(p_bundle_i.input_names)))
    else:
        for p_bundle in pft_bundles:
            x_mean_t = (
                p_bundle.x_mean_torch
                if p_bundle.x_mean_torch is not None
                   and p_bundle.x_mean_torch.device == device
                   and p_bundle.x_mean_torch.dtype == dtype
                else torch.as_tensor(p_bundle.x_mean, device=device, dtype=dtype)
            )
            tmpl = torch.empty((ncell, nens, x_mean_t.numel()), device=device, dtype=dtype)
            tmpl.copy_(x_mean_t)  # broadcast over (ncell, nens): same x_mean for all members
            feat_bufs.append(tmpl)
            feat_idx_maps.append(_bundle_input_index_map(tuple(p_bundle.input_names)))

    # Pre-index static PFT-parameter feature positions.
    pft_static_feature_idx = {
        name: [
            (p_bundle.input_names.index(name) if name in p_bundle.input_names else None)
            for p_bundle in pft_bundles
        ]
        for name in STATIC_PARAMETER_FEATURE_NAMES
    }
    pft_flnr_idx = pft_static_feature_idx["flnr"]
    pft_mbbopt_idx = pft_static_feature_idx["mbbopt"]
    neutral_universal_static_pft_idx = {
        p_idx
        for p_idx, (pft_id, p_bundle) in enumerate(zip(pft_ids, pft_bundles))
        if int(pft_id) == 0 and int(getattr(p_bundle, "patch_index", pft_id)) == -1
    }
    if neutral_universal_static_pft_idx:
        neutral_ids = ", ".join(str(int(pft_ids[p_idx])) for p_idx in sorted(neutral_universal_static_pft_idx))
        flnr_note = (
            f"; flnr={neutral_patch0_flnr:.6g}"
            if neutral_patch0_flnr is not None and np.isfinite(neutral_patch0_flnr)
            else ""
        )
        print(
            f"    universal fallback: keeping static plant-parameter features at training means "
            f"for PFT(s) {neutral_ids}, except low-end flnr{flnr_note}",
            flush=True,
        )

    flnr_factor_t = None
    mbbopt_factor_t = None
    if ensemble_flnr_factors is not None or ensemble_mbbopt_factors is not None:
        flnr_factor_np = np.asarray(
            [1.0] if ensemble_flnr_factors is None else ensemble_flnr_factors,
            dtype=np.float64,
        ).reshape(-1)
        mbbopt_factor_np = np.asarray(
            [1.0] if ensemble_mbbopt_factors is None else ensemble_mbbopt_factors,
            dtype=np.float64,
        ).reshape(-1)
        if flnr_factor_np.size != nens or mbbopt_factor_np.size != nens:
            raise ValueError(
                "explicit regional ensemble trait factors must have one value per ensemble member "
                f"(got flnr={flnr_factor_np.size}, mbbopt={mbbopt_factor_np.size}, nens={nens})"
            )
        flnr_factor_t = torch.as_tensor(flnr_factor_np.reshape(1, nens), device=device, dtype=dtype)
        mbbopt_factor_t = torch.as_tensor(mbbopt_factor_np.reshape(1, nens), device=device, dtype=dtype)

    # For nens > 1, sample per-ensemble flnr / mbbopt from ±2σ of training distribution,
    # unless explicit multiplicative factors are supplied. Shape (1, nens) so values
    # broadcast over the cell axis in the feature tensor.
    _rng_ens = np.random.default_rng(ensemble_seed)
    ens_flnr_dev: list = []   # per-PFT: (1, nens) tensor or None
    ens_mbbopt_dev: list = []
    for p_idx, p_bundle in enumerate(pft_bundles):
        def _ens_param(feat_idx, static_val, factor_t=None):
            if feat_idx is None:
                return None
            static_arr = np.asarray(static_val)
            if factor_t is not None:
                base_t = torch.as_tensor(static_arr, device=device, dtype=dtype)
                if base_t.ndim == 0:
                    base_t = base_t.reshape(1, 1)
                elif base_t.ndim == 1:
                    base_t = base_t.reshape(-1, 1)
                elif base_t.ndim == 2 and base_t.shape[1] == 1:
                    pass
                else:
                    base_t = base_t.reshape(base_t.shape[0], -1)
                    if base_t.shape[1] != 1:
                        raise ValueError(
                            "explicit ensemble trait factors require scalar or per-cell baseline traits; "
                            f"got shape {tuple(base_t.shape)} for feature index {feat_idx}"
                        )
                return base_t * factor_t
            if static_arr.ndim > 0 and static_arr.size > 1:
                val_t = torch.as_tensor(static_arr, device=device, dtype=dtype)
                if val_t.ndim == 1:
                    val_t = val_t.reshape(-1, 1)
                if val_t.ndim == 2 and val_t.shape[1] == 1 and nens > 1:
                    val_t = val_t.expand(-1, nens)
                return val_t
            if nens == 1:
                # Scalar broadcast — same as before.
                return torch.tensor(
                    [[float(static_arr.reshape(-1)[0])]],
                    device=device,
                    dtype=dtype,
                )
            x_mean_np = p_bundle.x_mean if not isinstance(p_bundle.x_mean, torch.Tensor) \
                        else p_bundle.x_mean.cpu().numpy()
            x_std_np  = p_bundle.x_std  if not isinstance(p_bundle.x_std,  torch.Tensor) \
                        else p_bundle.x_std.cpu().numpy()
            mean_v = float(x_mean_np[feat_idx])
            std_v  = max(float(x_std_np[feat_idx]),  1e-9)
            vals   = _rng_ens.uniform(mean_v - 2.0 * std_v, mean_v + 2.0 * std_v,
                                      size=(1, nens)).astype(np.float64)
            return torch.as_tensor(vals, device=device, dtype=dtype)
        ens_flnr_dev.append(_ens_param(
            pft_flnr_idx[p_idx],
            getattr(pft_statics[p_idx], 'flnr', 0.0),
            flnr_factor_t,
        ))
        ens_mbbopt_dev.append(_ens_param(
            pft_mbbopt_idx[p_idx],
            getattr(pft_statics[p_idx], 'mbbopt', 0.0),
            mbbopt_factor_t,
        ))

    # Import CoupledColumnState once outside the hot loop.
    from .coupled_surface_hydrology_soiltemp import CoupledColumnState as _CCS

    def _apply_soil_temperature_override(state_in: "_CCS", step_i: int) -> "_CCS":
        if soil_temperature_override_t is None:
            return state_in
        source = soil_temperature_override_t[min(int(step_i), soil_temperature_override_t.shape[0] - 1)]
        current = (
            state_in.t_soisno.to(device=device, dtype=dtype)
            if isinstance(state_in.t_soisno, torch.Tensor)
            else torch.as_tensor(state_in.t_soisno, device=device, dtype=dtype)
        )
        nlev_override = min(int(source.shape[-1]), int(current.shape[-1]))
        t_soisno = current.clone()
        t_soisno[..., :nlev_override] = source[:, None, :nlev_override].expand(
            -1,
            t_soisno.shape[1],
            -1,
        )
        return _CCS(
            h2osoi_liq=state_in.h2osoi_liq,
            h2osoi_ice=state_in.h2osoi_ice,
            h2ocan=state_in.h2ocan,
            h2osfc=state_in.h2osfc,
            frac_h2osfc=state_in.frac_h2osfc,
            frac_sno=state_in.frac_sno,
            zwt=state_in.zwt,
            t_soisno=t_soisno,
            wa=state_in.wa,
            t_h2osfc=state_in.t_h2osfc,
            t10=state_in.t10,
            t10_history=state_in.t10_history,
            h2osno=state_in.h2osno,
            snow_depth=state_in.snow_depth,
            snow_t=state_in.snow_t,
            snow_layers=state_in.snow_layers,
            int_snow=state_in.int_snow,
        )

    def _apply_soil_moisture_override(state_in: "_CCS", step_i: int) -> "_CCS":
        if soil_liquid_override_t is None or soil_ice_override_t is None:
            return state_in
        source_liquid = soil_liquid_override_t[
            min(int(step_i), soil_liquid_override_t.shape[0] - 1)
        ]
        source_ice = soil_ice_override_t[
            min(int(step_i), soil_ice_override_t.shape[0] - 1)
        ]
        current_liquid = (
            state_in.h2osoi_liq.to(device=device, dtype=dtype)
            if isinstance(state_in.h2osoi_liq, torch.Tensor)
            else torch.as_tensor(state_in.h2osoi_liq, device=device, dtype=dtype)
        )
        current_ice = (
            state_in.h2osoi_ice.to(device=device, dtype=dtype)
            if isinstance(state_in.h2osoi_ice, torch.Tensor)
            else torch.as_tensor(state_in.h2osoi_ice, device=device, dtype=dtype)
        )
        nlev_override = min(
            int(source_liquid.shape[-1]),
            int(source_ice.shape[-1]),
            int(current_liquid.shape[-1]),
            int(current_ice.shape[-1]),
        )
        h2osoi_liq = current_liquid.clone()
        h2osoi_ice = current_ice.clone()
        h2osoi_liq[..., :nlev_override] = source_liquid[
            :, None, :nlev_override
        ].expand(-1, h2osoi_liq.shape[1], -1)
        h2osoi_ice[..., :nlev_override] = source_ice[
            :, None, :nlev_override
        ].expand(-1, h2osoi_ice.shape[1], -1)
        return _CCS(
            h2osoi_liq=h2osoi_liq,
            h2osoi_ice=h2osoi_ice,
            h2ocan=state_in.h2ocan,
            h2osfc=state_in.h2osfc,
            frac_h2osfc=state_in.frac_h2osfc,
            frac_sno=state_in.frac_sno,
            zwt=state_in.zwt,
            t_soisno=state_in.t_soisno,
            wa=state_in.wa,
            t_h2osfc=state_in.t_h2osfc,
            t10=state_in.t10,
            t10_history=state_in.t10_history,
            h2osno=state_in.h2osno,
            snow_depth=state_in.snow_depth,
            snow_t=state_in.snow_t,
            snow_layers=state_in.snow_layers,
            int_snow=state_in.int_snow,
        )

    state = _apply_soil_temperature_override(state, 0)
    state = _apply_soil_moisture_override(state, 0)

    def _apply_zwt_override(state_in: "_CCS", step_i: int) -> "_CCS":
        if zwt_override_t is None:
            return state_in
        source = zwt_override_t[min(int(step_i), zwt_override_t.shape[0] - 1)]
        current = (
            state_in.zwt.to(device=device, dtype=dtype)
            if isinstance(state_in.zwt, torch.Tensor)
            else torch.as_tensor(state_in.zwt, device=device, dtype=dtype)
        )
        zwt = source.reshape(ncell, 1).expand_as(current)
        return _CCS(
            h2osoi_liq=state_in.h2osoi_liq,
            h2osoi_ice=state_in.h2osoi_ice,
            h2ocan=state_in.h2ocan,
            h2osfc=state_in.h2osfc,
            frac_h2osfc=state_in.frac_h2osfc,
            frac_sno=state_in.frac_sno,
            zwt=zwt,
            t_soisno=state_in.t_soisno,
            wa=state_in.wa,
            t_h2osfc=state_in.t_h2osfc,
            t10=state_in.t10,
            t10_history=state_in.t10_history,
            h2osno=state_in.h2osno,
            snow_depth=state_in.snow_depth,
            snow_t=state_in.snow_t,
            snow_layers=state_in.snow_layers,
            int_snow=state_in.int_snow,
        )

    state = _apply_zwt_override(state, 0)

    # Pre-allocate zero tensors for struct fields we do not have real data for.
    # The regional builder skips writing these ablated feature names so their
    # feature columns stay at the x_mean values seeded in feat_bufs.
    _zeros_ncell1 = torch.zeros(ncell, 1, device=device, dtype=dtype)
    _zeros_ncell1_geo = torch.zeros(ncell, 1, device=device, dtype=dtype)

    # Ablated feature names we do not have real regional data for. Radiation
    # components and air density are reconstructed below so they stay
    # internally consistent with FSDS / pressure / humidity.
    _ABLATED_NAMES = frozenset({
        "forc_hgt_u_patch", "z0mg_col", "ugust",
    })
    if universal_feature_assembly_enabled and snow_model_enabled:
        print(
            "    universal MLP batching will use fast feature assembly with live active snow features",
            flush=True,
        )

    # Seed static per-PFT feature columns once. The hourly feature builder
    # updates only state/forcing columns, leaving these static values intact.
    _feature_batch_shape = (ncell, nens)

    def _static_feature_value(value) -> "torch.Tensor":
        if value is None:
            return torch.zeros((), device=device, dtype=dtype)
        value_t = value if isinstance(value, torch.Tensor) else torch.as_tensor(value, device=device, dtype=dtype)
        value_t = value_t.to(device=device, dtype=dtype)
        if value_t.ndim == 1 and value_t.shape[0] == ncell:
            return value_t.reshape(ncell, 1)
        return value_t

    for p_idx, (p_bundle, p_static) in enumerate(zip(pft_bundles, pft_statics)):
        feat = feat_bufs[p_idx]
        input_names_t = tuple(p_bundle.input_names)
        nlev_static = int(p_static.hydrology_torch.dz.shape[-1])
        fidx, lidx = _soil_profile_feature_indices(input_names_t, "rootfr", nlev_static)
        if fidx:
            rootfr = (
                p_static.rootfr_norm_torch
                if p_static.rootfr_norm_torch is not None
                   and p_static.rootfr_norm_torch.device == device
                   and p_static.rootfr_norm_torch.dtype == dtype
                else torch.as_tensor(np.maximum(np.asarray(p_static.rootfr, dtype=np.float64), 0.0), device=device, dtype=dtype)
            )
            if p_static.rootfr_norm_torch is None or p_static.rootfr_norm_torch.device != device or p_static.rootfr_norm_torch.dtype != dtype:
                rootfr = rootfr / torch.clamp(rootfr.sum(), min=1.0e-12)
            root_values = torch.broadcast_to(rootfr[lidx], _feature_batch_shape + (len(lidx),))
            _index_copy_feature_columns_torch(feat, fidx, root_values)
        for col in STATIC_PARAMETER_FEATURE_NAMES:
            idx = pft_static_feature_idx[col][p_idx]
            if idx is None:
                continue
            if p_idx in neutral_universal_static_pft_idx and col not in ("pft_type", "soil_color"):
                if col == "flnr" and neutral_patch0_flnr is not None and np.isfinite(neutral_patch0_flnr):
                    feat[..., idx] = float(neutral_patch0_flnr)
                continue
            if col == "flnr" and ens_flnr_dev[p_idx] is not None:
                feat[..., idx] = ens_flnr_dev[p_idx]
            elif col == "mbbopt" and ens_mbbopt_dev[p_idx] is not None:
                feat[..., idx] = ens_mbbopt_dev[p_idx]
            else:
                feat[..., idx] = torch.broadcast_to(
                    _static_feature_value(getattr(p_static, col, 0.0)),
                    _feature_batch_shape,
                )

    # Keep the model-facing universal features in active-pair order.  The
    # legacy dense allocation remains as a compatibility/reference row for
    # canopy and coupled-physics code, but hourly assembly no longer writes a
    # full PFT-by-cell grid only to gather it again before inference.
    if (
        universal_feature_batch is not None
        and universal_sparse_mlp_enabled
        and universal_sparse_pft_idx_t is not None
        and universal_sparse_cell_idx_t is not None
    ):
        universal_sparse_feature_batch = universal_feature_batch[
            universal_sparse_pft_idx_t,
            universal_sparse_cell_idx_t,
            ...,
        ].clone()

    # Snow-disabled regional benchmarks do not need a full-cell individual
    # feature row in coupled_step (surface targets are already precomputed).
    # Keep patch-specific models packed over their nonzero-weight cells so
    # feature derivation and MLP inference avoid inactive grid cells.  Snow
    # replays retain the dense path because update_snow_state consumes the
    # complete reference feature row.
    if sparse_individual_feature_assembly and not snow_model_enabled:
        for p_idx, feat in enumerate(feat_bufs):
            if p_idx in universal_batch_index_set:
                continue
            cells = sparse_cells_by_pft.get(p_idx)
            if cells is not None and 0 < cells.numel() < ncell:
                individual_sparse_feature_bufs[p_idx] = feat.index_select(0, cells).clone()
        if individual_sparse_feature_bufs:
            packed_summary = ", ".join(
                f"PFT {int(pft_ids[p_idx])}: {int(sparse_cells_by_pft[p_idx].numel()):,}/{ncell:,}"
                for p_idx in sorted(individual_sparse_feature_bufs)
            )
            print(
                f"    individual sparse feature assembly enabled ({packed_summary} active cells)",
                flush=True,
            )

    # Fast universal feature assembly support.  This avoids calling the generic
    # feature builder once per PFT when every PFT uses the same input schema.
    universal_bundle = universal_ref_bundle if universal_mlp_batch else None
    universal_idx_map = (
        feat_idx_maps[universal_batch_indices[0]]
        if universal_mlp_batch and universal_batch_indices
        else None
    )
    universal_input_names_t = tuple(universal_bundle.input_names) if universal_bundle is not None else ()
    universal_trait_enabled = False
    universal_trait_feature_names_active: tuple[str, ...] = ()
    universal_trait_feature_indices_t: "torch.Tensor | None" = None
    universal_trait_z: "torch.Tensor | None" = None
    universal_trait_low: "torch.Tensor | None" = None
    universal_trait_high: "torch.Tensor | None" = None
    if trait_ensemble_size > 1 and universal_mlp_batch:
        assert universal_idx_map is not None
        assert universal_bundle is not None
        requested_trait_names = tuple(regional_trait_feature_names or REGIONAL_TRAIT_FEATURE_NAMES)
        if len(requested_trait_names) == 1 and str(requested_trait_names[0]).lower() == "all":
            requested_trait_names = REGIONAL_TRAIT_FEATURE_NAMES
        idx_rows: list[int] = []
        active_names: list[str] = []
        missing_names: list[str] = []
        constant_names: list[str] = []
        x_std_np = np.asarray(universal_bundle.x_std, dtype=np.float64)
        for name in requested_trait_names:
            name_s = str(name)
            idx = universal_idx_map.get(name_s)
            if idx is None:
                missing_names.append(name_s)
                continue
            std_v = float(x_std_np[idx])
            if not np.isfinite(std_v) or std_v <= 1.0e-12:
                constant_names.append(name_s)
                continue
            active_names.append(name_s)
            idx_rows.append(int(idx))
        if missing_names:
            print(
                "    trait ensemble: skipping missing universal feature(s): "
                + ", ".join(missing_names),
                flush=True,
            )
        if constant_names:
            print(
                "    trait ensemble: skipping constant universal feature(s): "
                + ", ".join(constant_names),
                flush=True,
            )
        if not idx_rows:
            print(
                "    trait ensemble requested but no varying trait features are available; using baseline universal MLP",
                flush=True,
            )
        else:
            rng_trait = np.random.default_rng(int(ensemble_seed) + 7919)
            cols_np = np.asarray(idx_rows, dtype=np.int64)
            mean_np = np.asarray(universal_bundle.x_mean, dtype=np.float64)[cols_np]
            std_np = x_std_np[cols_np]
            z_np = rng_trait.standard_normal(
                size=(len(universal_batch_indices), trait_ensemble_size, len(cols_np)),
            ).astype(np.float64)
            # Keep one exact baseline trait member in the quadrature.
            z_np[:, 0, :] = 0.0
            low_np = mean_np - trait_sigma * std_np
            high_np = mean_np + trait_sigma * std_np
            for j, name in enumerate(active_names):
                if name in NONNEGATIVE_REGIONAL_TRAIT_FEATURE_NAMES:
                    low_np[j] = max(low_np[j], 0.0)
            universal_trait_feature_names_active = tuple(active_names)
            universal_trait_feature_indices_t = torch.as_tensor(idx_rows, device=device, dtype=torch.long)
            universal_trait_z = torch.as_tensor(z_np, device=device, dtype=dtype)
            universal_trait_low = torch.as_tensor(low_np, device=device, dtype=dtype)
            universal_trait_high = torch.as_tensor(high_np, device=device, dtype=dtype)
            universal_trait_enabled = True
            print(
                f"    same-state trait ensemble enabled for universal MLP: "
                f"ntrait={trait_ensemble_size}, chunk={trait_chunk_size}, "
                f"relative_sd={trait_relative_sd:g}, range=mean±{trait_sigma:g}σ, "
                f"features={', '.join(universal_trait_feature_names_active)}",
                flush=True,
            )
    universal_profile_indices = (
        {
            prefix: _soil_profile_feature_indices(universal_input_names_t, prefix, nlev_hydro_static)
            for prefix in ("smp_l", "eff_porosity", "h2osoi_liqvol", "t_soisno")
        }
        if universal_mlp_batch
        else {}
    )
    pft_profile_indices = [
        {
            prefix: _soil_profile_feature_indices(
                tuple(p_bundle.input_names),
                prefix,
                int(p_static.hydrology_torch.dz.shape[-1]),
            )
            for prefix in ("smp_l", "eff_porosity", "h2osoi_liqvol", "t_soisno")
        }
        for p_bundle, p_static in zip(pft_bundles, pft_statics)
    ]
    universal_btran_idx = (
        universal_idx_map.get("btran") if universal_idx_map is not None else None
    )
    universal_sabg_idx = (
        universal_idx_map.get("sabg_snow") if universal_idx_map is not None else None
    )
    universal_soilbeta_idx = (
        universal_idx_map.get("soilbeta_col") if universal_idx_map is not None else None
    )
    soilbeta_dz0_top = None
    soilbeta_watfc_top = None
    if universal_soilbeta_idx is not None:
        hyd_top = shared_static.hydrology_torch

        def _top_layer_for_soilbeta(param: "torch.Tensor") -> "torch.Tensor":
            param_t = param.to(device=device, dtype=dtype) if isinstance(param, torch.Tensor) else torch.as_tensor(param, device=device, dtype=dtype)
            if param_t.ndim == 0:
                return param_t
            top = param_t[..., 0]
            if param_t.ndim == 1:
                return top
            while top.ndim < 2:
                top = top.unsqueeze(-1)
            return top

        soilbeta_dz0_top = _top_layer_for_soilbeta(hyd_top.dz)
        soilbeta_watsat_top = _top_layer_for_soilbeta(hyd_top.watsat)
        soilbeta_hksat_top = torch.clamp(_top_layer_for_soilbeta(hyd_top.hksat), min=1.0e-20)
        soilbeta_bsw_top = _top_layer_for_soilbeta(hyd_top.bsw)
        soilbeta_watfc_top = soilbeta_watsat_top * torch.pow(
            torch.as_tensor(0.1, device=device, dtype=dtype) / (soilbeta_hksat_top * SECSPDAY),
            1.0 / (2.0 * soilbeta_bsw_top + 3.0),
        )
    universal_dynamic_feature_names = (
        "forc_t",
        "forc_q",
        "forc_pbot",
        "forc_vpd",
        "qg_soil_diag",
        "forc_lwrad",
        "forc_rain",
        "forc_snow",
        "forc_swrad",
        "forc_solad_vis",
        "forc_solad_nir",
        "forc_solai_vis",
        "forc_solai_nir",
        "wind_speed",
        "forc_rho",
        "forc_pco2",
        "frac_veg_nosno_patch",
        "elai_patch",
        "esai_patch",
        "htop_patch",
        "frac_sno",
        "snow_depth",
        "zwt_col",
        "h2ocan",
        "t10",
        "dayl",
        "max_dayl",
        "forc_hgt_u_patch",
        "z0mg_col",
        "frac_h2osfc",
        "coszen_col",
        "t_grnd",
        "t_h2osfc",
        "ugust",
        "h2osfc",
        "h2osno",
        "snow_layer_count",
        "t_snow_top",
    )
    universal_dynamic_feature_active = tuple(
        (int(idx), pos)
        for pos, name in enumerate(universal_dynamic_feature_names)
        for idx in ((universal_idx_map.get(name) if universal_idx_map is not None and name not in _ABLATED_NAMES else None),)
        if idx is not None
    )
    universal_rootfr_btran = None
    universal_smpso = None
    universal_smpsc = None
    universal_btran_valid = None
    universal_watsat = None
    universal_tc_stress_c = None
    if universal_mlp_batch:
        rootfr_rows = []
        smpso_rows = []
        smpsc_rows = []
        btran_valid_rows = []
        tc_stress_rows = []
        for p_idx in universal_batch_indices:
            p_static = pft_statics[p_idx]
            rootfr_t = (
                p_static.rootfr_norm_torch
                if p_static.rootfr_norm_torch is not None
                   and p_static.rootfr_norm_torch.device == device
                   and p_static.rootfr_norm_torch.dtype == dtype
                else torch.as_tensor(np.maximum(np.asarray(p_static.rootfr, dtype=np.float64), 0.0), device=device, dtype=dtype)
            )
            if p_static.rootfr_norm_torch is None or p_static.rootfr_norm_torch.device != device or p_static.rootfr_norm_torch.dtype != dtype:
                rootfr_t = rootfr_t / torch.clamp(rootfr_t.sum(), min=1.0e-12)
            rootfr_rows.append(rootfr_t[:nlev_hydro_static])
            is_valid = p_static.smpso is not None and p_static.smpsc is not None
            smpso_rows.append(float(p_static.smpso) if p_static.smpso is not None else 0.0)
            smpsc_rows.append(float(p_static.smpsc) if p_static.smpsc is not None else 0.0)
            btran_valid_rows.append(is_valid)
            tc_stress_rows.append(float(getattr(p_static, "tc_stress_c", -2.0)))
        nuniversal = len(universal_batch_indices)
        universal_rootfr_btran = torch.stack(rootfr_rows, dim=0).reshape(nuniversal, 1, 1, nlev_hydro_static)
        universal_smpso = torch.as_tensor(smpso_rows, device=device, dtype=dtype).reshape(nuniversal, 1, 1, 1)
        universal_smpsc = torch.as_tensor(smpsc_rows, device=device, dtype=dtype).reshape(nuniversal, 1, 1, 1)
        universal_btran_valid = torch.as_tensor(btran_valid_rows, device=device, dtype=torch.bool).reshape(nuniversal, 1, 1)
        universal_tc_stress_c = torch.as_tensor(tc_stress_rows, device=device, dtype=dtype).reshape(nuniversal, 1, 1, 1)
        universal_watsat = shared_static.hydrology_torch.watsat
        if not isinstance(universal_watsat, torch.Tensor):
            universal_watsat = torch.as_tensor(universal_watsat, device=device, dtype=dtype)
        universal_watsat = universal_watsat.to(device=device, dtype=dtype)
        if universal_watsat.ndim == 1:
            universal_watsat = universal_watsat.reshape(1, 1, -1)
        elif universal_watsat.ndim == 2:
            universal_watsat = universal_watsat.unsqueeze(1)

    water_diag_rootfr = None
    water_diag_smpso = None
    water_diag_smpsc = None
    water_diag_valid = None
    water_diag_tc_stress_c = None
    water_diag_watsat = None
    if record_daily_diag_maps:
        rootfr_rows = []
        smpso_rows = []
        smpsc_rows = []
        valid_rows = []
        tc_stress_rows = []
        for p_static in pft_statics:
            rootfr_t = (
                p_static.rootfr_norm_torch
                if p_static.rootfr_norm_torch is not None
                   and p_static.rootfr_norm_torch.device == device
                   and p_static.rootfr_norm_torch.dtype == dtype
                else torch.as_tensor(np.maximum(np.asarray(p_static.rootfr, dtype=np.float64), 0.0), device=device, dtype=dtype)
            )
            if p_static.rootfr_norm_torch is None or p_static.rootfr_norm_torch.device != device or p_static.rootfr_norm_torch.dtype != dtype:
                rootfr_t = rootfr_t / torch.clamp(rootfr_t.sum(), min=1.0e-12)
            rootfr_rows.append(rootfr_t[:nlev_hydro_static])
            is_valid = p_static.smpso is not None and p_static.smpsc is not None
            smpso_rows.append(float(p_static.smpso) if p_static.smpso is not None else 0.0)
            smpsc_rows.append(float(p_static.smpsc) if p_static.smpsc is not None else 0.0)
            valid_rows.append(is_valid)
            tc_stress_rows.append(float(getattr(p_static, "tc_stress_c", -2.0)))
        water_diag_rootfr = torch.stack(rootfr_rows, dim=0).reshape(npft, 1, 1, nlev_hydro_static)
        water_diag_smpso = torch.as_tensor(smpso_rows, device=device, dtype=dtype).reshape(npft, 1, 1, 1)
        water_diag_smpsc = torch.as_tensor(smpsc_rows, device=device, dtype=dtype).reshape(npft, 1, 1, 1)
        water_diag_valid = torch.as_tensor(valid_rows, device=device, dtype=torch.bool).reshape(npft, 1, 1, 1)
        water_diag_tc_stress_c = torch.as_tensor(tc_stress_rows, device=device, dtype=dtype).reshape(npft, 1, 1, 1)
        water_diag_watsat = shared_static.hydrology_torch.watsat
        if not isinstance(water_diag_watsat, torch.Tensor):
            water_diag_watsat = torch.as_tensor(water_diag_watsat, device=device, dtype=dtype)
        water_diag_watsat = water_diag_watsat.to(device=device, dtype=dtype)
        if water_diag_watsat.ndim == 1:
            water_diag_watsat = water_diag_watsat.reshape(1, 1, -1)
        elif water_diag_watsat.ndim == 2:
            water_diag_watsat = water_diag_watsat.unsqueeze(1)

    def _area_weighted_rresis_layers_from_common(common: dict) -> "torch.Tensor | None":
        if (
            water_diag_rootfr is None
            or water_diag_smpso is None
            or water_diag_smpsc is None
            or water_diag_valid is None
            or water_diag_tc_stress_c is None
            or water_diag_watsat is None
        ):
            return None
        liqvol = common["liqvol"][..., :nlev_hydro_static]
        eff_porosity = common["eff_porosity"][..., :nlev_hydro_static]
        smp_l = common["smp_l"][..., :nlev_hydro_static]
        tempk = common["t_soisno_t"][..., :nlev_hydro_static]
        h2osfc_t = common["h2osfc_t"]
        assert isinstance(liqvol, torch.Tensor)
        assert isinstance(eff_porosity, torch.Tensor)
        assert isinstance(smp_l, torch.Tensor)
        assert isinstance(tempk, torch.Tensor)
        assert isinstance(h2osfc_t, torch.Tensor)

        liqvol4 = liqvol.unsqueeze(0)
        temp4 = tempk.unsqueeze(0)
        smp_node = torch.maximum(water_diag_smpsc, smp_l.unsqueeze(0))
        denom = torch.clamp(water_diag_smpso - water_diag_smpsc, min=1.0e-12)
        ep_over_ws = eff_porosity.unsqueeze(0) / torch.clamp(water_diag_watsat.unsqueeze(0), min=1.0e-12)
        rresis = torch.minimum(
            ep_over_ws * (smp_node - water_diag_smpsc) / denom,
            torch.ones_like(smp_node),
        )
        floodf = torch.where(
            h2osfc_t <= 1000.0,
            torch.clamp(1.0 - 0.001 * torch.clamp(h2osfc_t, min=0.0), min=0.0),
            torch.zeros_like(h2osfc_t),
        )
        active = (liqvol4 > 0.0) & (temp4 > (TFRZ + water_diag_tc_stress_c))
        rresis = torch.where(active, rresis * floodf.unsqueeze(0).unsqueeze(-1), torch.zeros_like(rresis))
        weights = w_dev.unsqueeze(-1)
        valid_weights = torch.where(water_diag_valid, weights, torch.zeros_like(weights))
        weight_sum = torch.sum(valid_weights, dim=0)
        weighted = torch.sum(torch.where(water_diag_valid, weights * rresis, torch.zeros_like(rresis)), dim=0)
        return torch.where(
            weight_sum > 1.0e-12,
            weighted / torch.clamp(weight_sum, min=1.0e-12),
            torch.full_like(weighted, float("nan")),
        )

    def _broadcast_universal_feature_value(value: "torch.Tensor | float | int") -> "torch.Tensor":
        nuniversal = len(universal_batch_indices)
        value_t = value if isinstance(value, torch.Tensor) else torch.as_tensor(value, device=device, dtype=dtype)
        value_t = value_t.to(device=device, dtype=dtype)
        if value_t.ndim == 0:
            return value_t.reshape(1, 1, 1).expand(nuniversal, ncell, nens)
        if value_t.shape == (ncell,):
            return value_t.reshape(1, ncell, 1).expand(nuniversal, ncell, nens)
        if value_t.shape == (ncell, 1):
            return value_t.unsqueeze(0).expand(nuniversal, ncell, nens)
        if value_t.shape == (ncell, nens):
            return value_t.unsqueeze(0).expand(nuniversal, ncell, nens)
        if value_t.shape == (nuniversal, ncell):
            return value_t.unsqueeze(-1).expand(nuniversal, ncell, nens)
        if value_t.shape == (nuniversal, ncell, 1):
            return value_t.expand(nuniversal, ncell, nens)
        if value_t.shape == (nuniversal, ncell, nens):
            return value_t
        return torch.broadcast_to(value_t, (nuniversal, ncell, nens))

    def _use_sparse_universal_feature_assembly() -> bool:
        return bool(
            universal_sparse_mlp_enabled
            and universal_sparse_pft_idx_t is not None
            and universal_sparse_cell_idx_t is not None
            and 0 < universal_sparse_pft_idx_t.numel() < len(universal_batch_indices) * ncell
        )

    def _gather_sparse_universal_value(value: "torch.Tensor | float | int") -> "torch.Tensor":
        """Gather a scalar, cell, PFT, or PFT-cell value onto active pairs."""
        assert universal_sparse_pft_idx_t is not None
        assert universal_sparse_cell_idx_t is not None
        nuniversal = len(universal_batch_indices)
        npair = int(universal_sparse_pft_idx_t.numel())
        value_t = value if isinstance(value, torch.Tensor) else torch.as_tensor(value, device=device, dtype=dtype)
        value_t = value_t.to(device=device, dtype=dtype)
        if value_t.ndim == 0:
            return value_t.reshape(1, 1).expand(npair, nens)
        if value_t.ndim >= 2 and value_t.shape[0] == nuniversal and value_t.shape[1] == ncell:
            gathered = value_t[universal_sparse_pft_idx_t, universal_sparse_cell_idx_t, ...]
        elif value_t.shape[0] == ncell:
            gathered = value_t.index_select(0, universal_sparse_cell_idx_t)
        elif value_t.shape[0] == nuniversal:
            gathered = value_t.index_select(0, universal_sparse_pft_idx_t)
        elif value_t.shape[0] == 1:
            gathered = value_t.expand((npair,) + tuple(value_t.shape[1:]))
        else:
            gathered = torch.broadcast_to(value_t, (npair, nens))
        gathered = gathered.reshape(npair, -1)
        if gathered.shape[1] == 1:
            return gathered.expand(npair, nens)
        if gathered.shape[1] != nens:
            raise ValueError(
                f"cannot gather universal feature shape {tuple(value_t.shape)} "
                f"onto ({npair}, {nens}) sparse PFT-cell pairs"
            )
        return gathered

    def _sync_packed_universal_columns_to_dense(names: tuple[str, ...]) -> None:
        """Sync the small physical-feature subset still used by dense consumers."""
        if (
            universal_sparse_feature_batch is None
            or universal_feature_batch is None
            or universal_idx_map is None
            or universal_sparse_pft_idx_t is None
            or universal_sparse_cell_idx_t is None
        ):
            return
        for name in names:
            idx = universal_idx_map.get(name)
            if idx is None:
                continue
            universal_feature_batch[
                universal_sparse_pft_idx_t,
                universal_sparse_cell_idx_t,
                :,
                idx,
            ] = universal_sparse_feature_batch[..., idx]

    def _universal_model_feature_batch() -> "torch.Tensor | None":
        return (
            universal_sparse_feature_batch
            if _use_sparse_universal_feature_assembly()
            else universal_feature_batch
        )

    _universal_state_cached_dynamic_positions = frozenset({
        20, 21, 22, 24, 29, 31, 32, 34, 35, 36, 37,
    })
    _universal_daily_vegetation_positions = frozenset({16, 17, 18, 19})

    def _update_universal_dynamic_feature_columns(
        values: tuple["torch.Tensor | float | int", ...],
        *,
        update_state_features: bool,
        update_vegetation_features: bool,
    ) -> None:
        if universal_feature_batch is None or universal_idx_map is None:
            return
        if not universal_dynamic_feature_active:
            return
        sparse = _use_sparse_universal_feature_assembly()
        for idx, pos in universal_dynamic_feature_active:
            if pos in _universal_state_cached_dynamic_positions and not update_state_features:
                continue
            if pos in _universal_daily_vegetation_positions and not update_vegetation_features:
                continue
            value = values[pos]
            if sparse:
                assert universal_sparse_pft_idx_t is not None
                assert universal_sparse_cell_idx_t is not None
                assert universal_sparse_feature_batch is not None
                # Canopy interception uses row 0 as the source of column-wide
                # meteorology.  Keep that row current, then update only the
                # nonzero pairs consumed by the universal MLP.
                universal_feature_batch[0, ..., idx] = _broadcast_universal_feature_value(value)[0]
                universal_sparse_feature_batch[..., idx] = _gather_sparse_universal_value(value)
            else:
                universal_feature_batch[..., idx] = _broadcast_universal_feature_value(value)

    def _active_snow_feature_tensors(state_check: CoupledColumnState) -> tuple[torch.Tensor, torch.Tensor]:
        if not snow_model_enabled or getattr(state_check, "frac_sno", None) is None:
            return _zeros_ncell1, _zeros_ncell1
        return (
            _active_snow_fraction_torch(state_check, device=device, dtype=dtype),
            _active_snow_depth_torch(state_check, device=device, dtype=dtype),
        )

    def _surface_snow_feature_tensors(state_check: CoupledColumnState) -> tuple[torch.Tensor, torch.Tensor]:
        if not snow_model_enabled or getattr(state_check, "frac_sno", None) is None:
            return _zeros_ncell1, _zeros_ncell1
        return (
            _total_snow_fraction_torch(state_check, device=device, dtype=dtype),
            _total_snow_depth_torch(state_check, device=device, dtype=dtype),
        )

    def _snow_energy_feature_tensors(
        state_check: CoupledColumnState,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        def _state_tensor(value) -> "torch.Tensor":
            if isinstance(value, torch.Tensor):
                return value.to(device=device, dtype=dtype)
            return torch.as_tensor(value, device=device, dtype=dtype)

        soil_top = _state_tensor(state_check.t_soisno[..., 0])
        if not snow_model_enabled:
            zeros = torch.zeros_like(soil_top)
            return zeros, zeros, soil_top
        h2osno = (
            _state_tensor(state_check.h2osno)
            if state_check.h2osno is not None
            else torch.zeros_like(soil_top)
        )
        if state_check.snow_layers is not None:
            snl = torch.as_tensor(state_check.snow_layers.snl, device=device, dtype=torch.int32)
            layer_count = torch.clamp(-snl, min=0).to(dtype=dtype)
            snow_top = _state_tensor(state_check.snow_layers.t_snow[..., 0])
        elif state_check.snow_t is not None:
            snow_top = _state_tensor(state_check.snow_t[..., 0])
            layer_count = (h2osno > 0.0).to(dtype=dtype)
        else:
            snow_top = soil_top
            layer_count = torch.zeros_like(soil_top)
        snow_top = torch.where(layer_count > 0.0, snow_top, soil_top)
        return h2osno, layer_count, snow_top

    def _soilbeta_from_common_features(
        common: dict,
        *,
        frac_sno: "torch.Tensor",
    ) -> "torch.Tensor":
        if soilbeta_dz0_top is None or soilbeta_watfc_top is None:
            return coupled_soilbeta_from_state_torch(
                state,
                shared_static,
                device=device,
                dtype=dtype,
                h2osoi_liq=common["h2osoi_liq_t"],
                h2osoi_ice=common["h2osoi_ice_t"],
                frac_sno=frac_sno,
                frac_h2osfc=common["frac_h2osfc_t"],
            )
        liq_top = common["h2osoi_liq_t"][..., 0]
        ice_top = common["h2osoi_ice_t"][..., 0]
        frac_h2osfc = common["frac_h2osfc_t"]
        assert isinstance(liq_top, torch.Tensor)
        assert isinstance(ice_top, torch.Tensor)
        assert isinstance(frac_h2osfc, torch.Tensor)
        wx = (liq_top / DENH2O + ice_top / DENICE) / torch.clamp(soilbeta_dz0_top, min=1.0e-12)
        fac_fc = torch.clamp(wx / torch.clamp(soilbeta_watfc_top, min=1.0e-12), min=0.01, max=1.0)
        beta_dry = (
            (1.0 - frac_sno - frac_h2osfc)
            * 0.25
            * torch.pow(1.0 - torch.cos(math.pi * fac_fc), 2.0)
            + frac_sno
            + frac_h2osfc
        )
        return torch.where(wx < soilbeta_watfc_top, beta_dry, torch.ones_like(wx))

    def _copy_universal_profile(prefix: str, values: "torch.Tensor") -> None:
        if universal_feature_batch is None:
            return
        fidx, lidx = universal_profile_indices.get(prefix, ([], []))
        if not fidx:
            return
        if _use_sparse_universal_feature_assembly():
            assert universal_sparse_pft_idx_t is not None
            assert universal_sparse_cell_idx_t is not None
            assert universal_sparse_feature_batch is not None
            pair_values = values.index_select(0, universal_sparse_cell_idx_t)
            for feature_idx, layer_idx in zip(fidx, lidx):
                universal_sparse_feature_batch[..., feature_idx] = pair_values[..., layer_idx]
                # Row 0 is the physical reference passed to coupled_step.
                universal_feature_batch[0, ..., feature_idx] = values[..., layer_idx]
            return
        selected = values[..., lidx].unsqueeze(0).expand(len(universal_batch_indices), *values.shape[:-1], len(lidx))
        _index_copy_feature_columns_torch(universal_feature_batch, fidx, selected)

    def _copy_pft_profile(
        p_idx: int,
        features: "torch.Tensor",
        prefix: str,
        values: "torch.Tensor",
        cell_idx: "torch.Tensor | None" = None,
    ) -> None:
        fidx, lidx = pft_profile_indices[p_idx].get(prefix, ([], []))
        if fidx:
            if cell_idx is not None:
                values = values.index_select(0, cell_idx)
            _index_copy_feature_columns_torch(features, fidx, values[..., lidx])

    def _assemble_individual_features_fast(
        p_idx: int,
        p_bundle,
        p_static,
        features: "torch.Tensor",
        common: dict,
        *,
        tbot: "torch.Tensor",
        qbot: "torch.Tensor",
        psrf: "torch.Tensor",
        forc_vpd: "torch.Tensor",
        flds: "torch.Tensor",
        forc_rain: "torch.Tensor",
        forc_snow: "torch.Tensor",
        fsds: "torch.Tensor",
        forc_solad_vis: "torch.Tensor",
        forc_solad_nir: "torch.Tensor",
        forc_solai_vis: "torch.Tensor",
        forc_solai_nir: "torch.Tensor",
        wind: "torch.Tensor",
        forc_rho: "torch.Tensor",
        forc_pco2: "torch.Tensor",
        dayl: "torch.Tensor",
        max_dayl: "torch.Tensor",
        coszen: "torch.Tensor",
        lai_pft: "torch.Tensor",
        sai_pft: "torch.Tensor",
        frac_veg: "torch.Tensor",
        htop_pft: "torch.Tensor",
        atmosphere: AtmosphericForcingInputs,
        update_state_features: bool = True,
        update_vegetation_features: bool = True,
        cell_idx: "torch.Tensor | None" = None,
    ) -> "torch.Tensor":
        idx_map = feat_idx_maps[p_idx]
        assert isinstance(common["h2ocan_t"], torch.Tensor)
        assert isinstance(common["h2osfc_t"], torch.Tensor)
        assert isinstance(common["frac_h2osfc_t"], torch.Tensor)
        assert isinstance(common["t_soisno_t"], torch.Tensor)
        assert isinstance(common["t_h2osfc_t"], torch.Tensor)
        assert isinstance(common["t_grnd"], torch.Tensor)
        assert isinstance(common["t10"], torch.Tensor)
        assert isinstance(common["liqvol"], torch.Tensor)
        assert isinstance(common["eff_porosity"], torch.Tensor)
        assert isinstance(common["smp_l"], torch.Tensor)

        def _select_cells(value):
            if cell_idx is None or value is None:
                return value
            value_t = value if isinstance(value, torch.Tensor) else torch.as_tensor(value, device=device)
            if value_t.ndim >= 1 and value_t.shape[0] == ncell:
                return value_t.index_select(0, cell_idx)
            return value_t

        if snow_model_enabled:
            feature_forc_rain = forc_rain
            feature_forc_snow = forc_snow
            feature_frac_sno, feature_snow_depth = _surface_snow_feature_tensors(state)
            sabg_state = state
        else:
            feature_forc_rain = forc_rain + forc_snow
            feature_forc_snow = _zeros_ncell1
            feature_frac_sno = _zeros_ncell1
            feature_snow_depth = _zeros_ncell1
            sabg_state = _zero_snow_state(state)
        feature_h2osno, feature_snow_layer_count, feature_t_snow_top = (
            _snow_energy_feature_tensors(state)
        )

        updates: dict[str, torch.Tensor | float] = {
            "forc_t": tbot,
            "forc_q": qbot,
            "forc_pbot": psrf,
            "forc_vpd": forc_vpd,
            "qg_soil_diag": common["qg_soil_diag"],
            "forc_lwrad": flds,
            "forc_rain": feature_forc_rain,
            "forc_snow": feature_forc_snow,
            "forc_swrad": fsds,
            "forc_solad_vis": forc_solad_vis,
            "forc_solad_nir": forc_solad_nir,
            "forc_solai_vis": forc_solai_vis,
            "forc_solai_nir": forc_solai_nir,
            "wind_speed": wind,
            "forc_rho": forc_rho,
            "forc_pco2": forc_pco2,
            "frac_veg_nosno_patch": frac_veg,
            "elai_patch": lai_pft,
            "esai_patch": sai_pft,
            "htop_patch": htop_pft,
            "frac_sno": feature_frac_sno,
            "snow_depth": feature_snow_depth,
            "zwt_col": state.zwt,
            "h2ocan": common["h2ocan_t"],
            "t10": common["t10"],
            "dayl": dayl,
            "max_dayl": max_dayl,
            "frac_h2osfc": common["frac_h2osfc_t"],
            "coszen_col": coszen,
            "t_grnd": common["t_grnd"],
            "t_h2osfc": common["t_h2osfc_t"],
            "h2osfc": common["h2osfc_t"],
            "h2osno": feature_h2osno,
            "snow_layer_count": feature_snow_layer_count,
            "t_snow_top": feature_t_snow_top,
        }
        if cell_idx is not None:
            updates = {name: _select_cells(value) for name, value in updates.items()}
        updates = {name: value for name, value in updates.items() if name not in _ABLATED_NAMES}
        if not update_state_features:
            cached_state_names = {
                "frac_sno", "snow_depth", "zwt_col", "t10", "frac_h2osfc",
                "t_grnd", "t_h2osfc", "h2osfc", "h2osno",
                "snow_layer_count", "t_snow_top",
            }
            updates = {name: value for name, value in updates.items() if name not in cached_state_names}
        if not update_vegetation_features:
            cached_vegetation_names = {
                "frac_veg_nosno_patch", "elai_patch", "esai_patch", "htop_patch",
            }
            updates = {name: value for name, value in updates.items() if name not in cached_vegetation_names}

        if update_state_features and "btran" in idx_map:
            if p_static.smpso is None or p_static.smpsc is None:
                updates["btran"] = _select_cells(torch.ones_like(common["h2osfc_t"]))
            else:
                btran_value = coupled_btran_from_state_torch(
                    state,
                    p_static,
                    device=device,
                    dtype=dtype,
                    h2osfc_storage=common["h2osfc_t"],
                    liqvol=common["liqvol"],
                    eff_porosity=common["eff_porosity"],
                    smp_l=common["smp_l"],
                )
                updates["btran"] = _select_cells(btran_value)
        if update_state_features and "soilbeta_col" in idx_map:
            soilbeta_value = coupled_soilbeta_from_state_torch(
                state,
                p_static,
                device=device,
                dtype=dtype,
                h2osoi_liq=common["h2osoi_liq_t"],
                h2osoi_ice=common["h2osoi_ice_t"],
                frac_sno=feature_frac_sno,
                frac_h2osfc=common["frac_h2osfc_t"],
            )
            updates["soilbeta_col"] = _select_cells(soilbeta_value)
        if "sabg_snow" in idx_map:
            sabg_fsds = _select_cells(fsds)
            sabg_lai = _select_cells(lai_pft)
            sabg_sai = _select_cells(sai_pft)
            sabg_atmosphere = atmosphere
            sabg_state_selected = sabg_state
            if cell_idx is not None:
                sabg_atmosphere = AtmosphericForcingInputs(**{
                    name: _select_cells(getattr(atmosphere, name))
                    for name in atmosphere.__dataclass_fields__
                })
                sabg_state_selected = _select_state_cells_for_sparse_pairs(sabg_state, cell_idx)
            updates["sabg_snow"] = (
                _absorbed_ground_shortwave_torch(
                    sabg_fsds,
                    device=device,
                    dtype=dtype,
                    elai=sabg_lai,
                    esai=sabg_sai,
                    atmosphere=sabg_atmosphere,
                    soil_color=_select_cells(p_static.soil_color),
                    h2osoi_vol_top=_select_cells(common["h2osoi_vol"])[..., 0],
                    rholvis=_select_cells(p_static.rholvis),
                    rholnir=_select_cells(p_static.rholnir),
                    taulvis=_select_cells(p_static.taulvis),
                    taulnir=_select_cells(p_static.taulnir),
                    rhosvis=_select_cells(p_static.rhosvis),
                    rhosnir=_select_cells(p_static.rhosnir),
                    tausvis=_select_cells(p_static.tausvis),
                    tausnir=_select_cells(p_static.tausnir),
                    xl=_select_cells(p_static.xl),
                    require_native_like=True,
                    context="regional sabg_snow feature",
                )
                if not snow_model_enabled
                else sabg_snow_from_state_torch(
                    sabg_state_selected,
                    sabg_fsds,
                    _select_cells(feature_frac_sno),
                    device=device,
                    dtype=dtype,
                    elai=sabg_lai,
                    esai=sabg_sai,
                    atmosphere=sabg_atmosphere,
                    soil_color=_select_cells(p_static.soil_color),
                    h2osoi_vol_top=_select_cells(common["h2osoi_vol"])[..., 0],
                    rholvis=_select_cells(p_static.rholvis),
                    rholnir=_select_cells(p_static.rholnir),
                    taulvis=_select_cells(p_static.taulvis),
                    taulnir=_select_cells(p_static.taulnir),
                    rhosvis=_select_cells(p_static.rhosvis),
                    rhosnir=_select_cells(p_static.rhosnir),
                    tausvis=_select_cells(p_static.tausvis),
                    tausnir=_select_cells(p_static.tausnir),
                    xl=_select_cells(p_static.xl),
                    require_native_like=True,
                    context="regional sabg_snow feature",
                )
            )

        _batch_update_feature_columns_torch(features, idx_map, updates)
        if update_state_features:
            _copy_pft_profile(p_idx, features, "smp_l", common["smp_l"], cell_idx)
            _copy_pft_profile(p_idx, features, "eff_porosity", common["eff_porosity"], cell_idx)
            _copy_pft_profile(p_idx, features, "h2osoi_liqvol", common["liqvol"], cell_idx)
            _copy_pft_profile(p_idx, features, "t_soisno", common["t_soisno_t"], cell_idx)
        return features

    def _universal_btran_and_root_fraction_from_common(common: dict) -> tuple["torch.Tensor | None", "torch.Tensor | None"]:
        if (
            universal_rootfr_btran is None
            or universal_smpso is None
            or universal_smpsc is None
            or universal_btran_valid is None
            or universal_watsat is None
            or universal_tc_stress_c is None
        ):
            return None, None
        liqvol = common["liqvol"]
        eff_porosity = common["eff_porosity"]
        smp_l = common["smp_l"]
        t_soisno_t = common["t_soisno_t"][..., :nlev_hydro_static]
        h2osfc_t = common["h2osfc_t"]
        assert isinstance(liqvol, torch.Tensor)
        assert isinstance(eff_porosity, torch.Tensor)
        assert isinstance(smp_l, torch.Tensor)
        assert isinstance(t_soisno_t, torch.Tensor)
        assert isinstance(h2osfc_t, torch.Tensor)

        sparse = _use_sparse_universal_feature_assembly()
        if sparse:
            assert universal_sparse_pft_idx_t is not None
            assert universal_sparse_cell_idx_t is not None
            pidx = universal_sparse_pft_idx_t
            cidx = universal_sparse_cell_idx_t
            liqvol4 = liqvol.index_select(0, cidx)
            temp4 = t_soisno_t.index_select(0, cidx)
            smp_l4 = smp_l.index_select(0, cidx)
            eff_porosity4 = eff_porosity.index_select(0, cidx)
            smpso4 = universal_smpso[pidx, 0, ...]
            smpsc4 = universal_smpsc[pidx, 0, ...]
            tc_stress4 = universal_tc_stress_c[pidx, 0, ...]
            rootfr4 = universal_rootfr_btran[pidx, 0, ...]
            valid4 = universal_btran_valid[pidx, 0, ...]
            if universal_watsat.shape[0] == ncell:
                watsat4 = universal_watsat.index_select(0, cidx)
            else:
                watsat4 = universal_watsat.expand(ncell, *universal_watsat.shape[1:]).index_select(0, cidx)
        else:
            liqvol4 = liqvol.unsqueeze(0)
            temp4 = t_soisno_t.unsqueeze(0)
            smp_l4 = smp_l.unsqueeze(0)
            eff_porosity4 = eff_porosity.unsqueeze(0)
            smpso4 = universal_smpso
            smpsc4 = universal_smpsc
            tc_stress4 = universal_tc_stress_c
            rootfr4 = universal_rootfr_btran
            valid4 = universal_btran_valid
            watsat4 = universal_watsat.unsqueeze(0)
        active = (liqvol4 > 0.0) & (temp4 > (TFRZ + tc_stress4))
        smp_node = torch.maximum(smpsc4, smp_l4)
        denom = torch.clamp(smpso4 - smpsc4, min=1.0e-12)
        ep_over_ws = eff_porosity4 / torch.clamp(watsat4, min=1.0e-12)
        rresis = torch.minimum(
            ep_over_ws * (smp_node - smpsc4) / denom,
            torch.ones_like(smp_node),
        )
        floodf = torch.where(
            h2osfc_t <= 1000.0,
            torch.clamp(1.0 - 0.001 * torch.clamp(h2osfc_t, min=0.0), min=0.0),
            torch.zeros_like(h2osfc_t),
        )
        flood_scale = floodf.index_select(0, cidx).unsqueeze(-1) if sparse else floodf.unsqueeze(0).unsqueeze(-1)
        rresis = torch.where(active, rresis * flood_scale, torch.zeros_like(rresis))
        rootfr_unf = torch.where(
            temp4 > (TFRZ + tc_stress4),
            rootfr4,
            torch.zeros_like(rootfr4),
        )
        root_sum = rootfr_unf.sum(dim=-1, keepdim=True)
        rootfr_unf = torch.where(root_sum > 1.0e-12, rootfr_unf / root_sum, torch.zeros_like(rootfr_unf))
        root_contrib = rootfr_unf * rresis
        btran = torch.sum(root_contrib, dim=-1)
        rootr = torch.where(
            btran.unsqueeze(-1) > 1.0e-12,
            root_contrib / torch.clamp(btran.unsqueeze(-1), min=1.0e-12),
            torch.zeros_like(root_contrib),
        )
        rootr = torch.where(
            valid4.unsqueeze(-1),
            rootr,
            rootfr4.expand_as(rootr),
        )
        btran = torch.where(valid4, btran, torch.ones_like(btran))
        if sparse:
            if rootr.shape[:2] != (pidx.numel(), nens):
                raise RuntimeError(
                    "unexpected sparse btran shape: "
                    f"rootr={tuple(rootr.shape)} liq={tuple(liqvol4.shape)} "
                    f"smpso={tuple(smpso4.shape)} smpsc={tuple(smpsc4.shape)} "
                    f"tc={tuple(tc_stress4.shape)} rootfr={tuple(rootfr4.shape)} "
                    f"valid={tuple(valid4.shape)} watsat={tuple(watsat4.shape)}"
                )
        return btran, rootr

    ood_diagnostics_stride = int(ood_diagnostics_stride)
    pft_ood_tensors = (
        [
            _regional_ood_tensors(p_bundle, device=device, dtype=dtype)
            for p_bundle in pft_bundles
        ]
        if ood_diagnostics_stride > 0
        else []
    )
    ood_feature_map_names = tuple(
        dict.fromkeys(str(name).strip() for name in (ood_feature_map_names or ()) if str(name).strip())
    )
    if _ALL_REGIONAL_OOD_FEATURE_MAPS_SENTINEL in ood_feature_map_names:
        all_names: list[str] = []
        for p_bundle in pft_bundles:
            all_names.extend(p_bundle.input_names)
        ood_feature_map_names = tuple(dict.fromkeys(all_names))
    pft_ood_feature_stats: list[dict[str, tuple[int, float, float]]] = []
    for p_bundle in pft_bundles:
        x_mean = np.asarray(p_bundle.x_mean, dtype=np.float64)
        x_std = np.asarray(p_bundle.x_std, dtype=np.float64)
        tol = 10.0 * np.finfo(np.float32).eps * np.maximum(np.abs(x_mean), 1.0)
        stats: dict[str, tuple[int, float, float]] = {}
        for name in (ood_feature_map_names if ood_diagnostics_stride > 0 else ()):
            if name not in p_bundle.input_names:
                continue
            feat_idx = p_bundle.input_names.index(name)
            if x_std[feat_idx] <= tol[feat_idx]:
                continue
            stats[name] = (feat_idx, float(x_mean[feat_idx]), float(1.0 / x_std[feat_idx]))
        pft_ood_feature_stats.append(stats)

    # Area-weight tensor for LAI blending: (npft, ncell) on device.
    w_lai_dev = w_dev[:, :, 0]  # (npft, ncell)

    # -----------------------------------------------------------------------
    # Main hourly loop
    # -----------------------------------------------------------------------
    physics_step_seconds_requested = max(float(dt), float(physics_step_hours) * 3600.0)
    requested_cadence_steps = physics_step_seconds_requested / float(dt)
    physics_cadence_steps = max(1, int(round(requested_cadence_steps)))
    if not np.isclose(requested_cadence_steps, physics_cadence_steps, rtol=0.0, atol=1.0e-6):
        raise ValueError(
            "--regional-physics-step-hours must be an integer multiple of the hourly replay dt"
        )
    physics_step_seconds = physics_cadence_steps * float(dt)
    if physics_cadence_steps > 1:
        print(
            f"    soil T/hydrology coupling every {physics_cadence_steps} hourly steps "
            f"({physics_step_seconds / 3600.0:g} h)",
            flush=True,
        )
    t0 = time.perf_counter()
    last_result = None
    _physics_accum: "dict | None" = None
    _physics_count = 0
    _physics_qtran_universal_accum: "torch.Tensor | None" = None
    _physics_qtran_individual_accum: dict[int, "torch.Tensor"] = {}
    _physics_universal_rootr: "torch.Tensor | None" = None
    _last_qflx_rootsoi: "torch.Tensor | None" = None
    _physics_ref_feat = None
    _physics_canopy_state = None
    _physics_ncalls = 0
    _physics_snow_enabled_calls = 0
    _physics_active_snow_calls = 0
    _physics_pending_snow_calls = 0
    _active_snow_cells = _active_snow_cell_count(state)
    _snow_present_cells = _snow_present_cell_count(state)
    _max_active_snow_cells = _active_snow_cells
    _state_has_active_snow = _active_snow_cells > 0
    _state_has_snow = _snow_present_cells > 0
    _snow_pending_in_accum = False
    _universal_fast_steps = 0
    _state_feature_refreshes = 0
    _vegetation_feature_refreshes = 0
    # Accumulators for diagnostic maps (time-mean, kept on device).
    _acc_forc_t: "torch.Tensor | None" = None
    _acc_forc_q: "torch.Tensor | None" = None
    _acc_forc_pbot: "torch.Tensor | None" = None
    _acc_forc_lwrad: "torch.Tensor | None" = None
    _acc_forc_wind: "torch.Tensor | None" = None
    _acc_rh2m: "torch.Tensor | None" = None
    _acc_forc_rain: "torch.Tensor | None" = None
    _acc_forc_snow: "torch.Tensor | None" = None
    _acc_fsds: "torch.Tensor | None" = None
    _acc_fpsn:   "torch.Tensor | None" = None
    _acc_lai:    "torch.Tensor | None" = None
    _acc_t_soil: "torch.Tensor | None" = None
    _acc_t_soil_layers: "torch.Tensor | None" = None
    _acc_elm_h0_tsoi: "torch.Tensor | None" = None
    _acc_elm_h0_h2osoi: "torch.Tensor | None" = None
    _acc_t_veg:  "torch.Tensor | None" = None
    _acc_liq:    "torch.Tensor | None" = None
    _acc_liq_layers: "torch.Tensor | None" = None
    _acc_h2osfc: "torch.Tensor | None" = None
    _acc_h2osno: "torch.Tensor | None" = None
    _acc_frac_sno: "torch.Tensor | None" = None
    _acc_active_frac_sno: "torch.Tensor | None" = None
    _acc_zwt: "torch.Tensor | None" = None
    _acc_btran: "torch.Tensor | None" = None
    _acc_tran: "torch.Tensor | None" = None
    _acc_canopy_evap: "torch.Tensor | None" = None
    _acc_ev_soil: "torch.Tensor | None" = None
    _acc_evap_soi: "torch.Tensor | None" = None
    _acc_evap_tot: "torch.Tensor | None" = None
    _acc_sh: "torch.Tensor | None" = None
    _acc_tg: "torch.Tensor | None" = None
    _acc_fire: "torch.Tensor | None" = None
    _acc_fira: "torch.Tensor | None" = None
    _acc_fgr: "torch.Tensor | None" = None
    _acc_hs_soil: "torch.Tensor | None" = None
    _acc_hs_top_snow: "torch.Tensor | None" = None
    _acc_hs_h2osfc: "torch.Tensor | None" = None
    _acc_dhsdT: "torch.Tensor | None" = None
    _acc_runoff_surf: "torch.Tensor | None" = None
    _acc_runoff_h2osfc_surf: "torch.Tensor | None" = None
    _acc_runoff_drain: "torch.Tensor | None" = None
    _acc_runoff_deep_drain: "torch.Tensor | None" = None
    _acc_runoff_tot: "torch.Tensor | None" = None
    _acc_lh: "torch.Tensor | None" = None
    _acc_snow_depth: "torch.Tensor | None" = None
    _acc_active_snow_depth: "torch.Tensor | None" = None
    _acc_snow_depth_eff: "torch.Tensor | None" = None
    _acc_shortwave_diag: "dict[str, torch.Tensor]" = {}
    _acc_energy_diag: "dict[str, torch.Tensor]" = {}
    _closure_residual_cumulative = torch.zeros((ncell,), device=device, dtype=dtype)
    _acc_ood_max_z: "torch.Tensor | None" = None
    _acc_ood_frac:  "torch.Tensor | None" = None
    _acc_ood_feature_z: "dict[str, torch.Tensor]" = {}
    _acc_ood_n = 0
    _acc_n = 0
    _daily_diag_rows: list[np.ndarray] = []
    _daily_diag_count = 0
    _daily_diag_sum: "torch.Tensor | None" = None
    _daily_day_night_rows: "dict[str, list[np.ndarray]]" = {}
    _daily_day_count_sum: "torch.Tensor | None" = None
    _daily_night_count_sum: "torch.Tensor | None" = None
    _daily_day_flux_sum: "dict[str, torch.Tensor]" = {}
    _daily_night_flux_sum: "dict[str, torch.Tensor]" = {}
    _daily_fpsn_ens_rows: list[np.ndarray] = []
    _daily_fpsn_ens_sum: "torch.Tensor | None" = None
    _period_fpsn_ens_sum: "torch.Tensor | None" = None
    _period_fpsn_ens_count = 0
    _daily_runoff_ens_rows: list[np.ndarray] = []
    _daily_runoff_ens_sum: "torch.Tensor | None" = None
    _period_runoff_ens_sum: "torch.Tensor | None" = None
    _period_runoff_ens_count = 0
    _acc_trait_std: "dict[str, torch.Tensor]" = {}
    _daily_trait_std_rows: "dict[str, list[np.ndarray]]" = {}
    _daily_trait_std_sum: "dict[str, torch.Tensor]" = {}
    _period_trait_std_sum: "dict[str, torch.Tensor]" = {}
    _period_trait_std_count = 0
    _daily_shortwave_rows: "dict[str, list[np.ndarray]]" = {}
    _daily_shortwave_sum: "dict[str, torch.Tensor]" = {}
    _daily_energy_rows: "dict[str, list[np.ndarray]]" = {}
    _daily_energy_sum: "dict[str, torch.Tensor]" = {}
    _daily_elm_h0_tsoi_rows: list[np.ndarray] = []
    _daily_elm_h0_h2osoi_rows: list[np.ndarray] = []
    _daily_elm_h0_tsoi_sum: "torch.Tensor | None" = None
    _daily_elm_h0_h2osoi_sum: "torch.Tensor | None" = None
    _hourly_h0_rows: "dict[str, list[np.ndarray]]" = {}
    # Hoist layer thicknesses out of the per-step loop (avoids GPU→CPU sync each step).
    _dz0_diag = float(shared_static.hydrology_torch.dz[0])  # [m]
    _dz_liq_diag = torch.as_tensor(
        shared_static.hydrology_torch.dz[:5],
        device=device,
        dtype=dtype,
    )  # [m], layers 1-5
    _dz_hydro_diag = torch.as_tensor(
        shared_static.hydrology_torch.dz[:nlev_hydro_static],
        device=device,
        dtype=dtype,
    )  # [m], hydrology layers
    _outlier_topk = 8
    _outlier_thresholds = {
        "fpsn": 100.0,
        "pft_fpsn": 100.0,
        "t_soil": 330.0,
        "t_soil_raw": 330.0,
        "lai": 20.0,
        "liq_vol": 2.0,
        "h2osfc": 100.0,
        "h2osno": 10.0,
        "snow_depth": 1.0,
        "snow_depth_eff": 1.0,
        "frac_sno": 0.95,
    }
    _outlier_units = {
        "fpsn": "umol/m2/s",
        "pft_fpsn": "umol/m2/s",
        "t_soil": "K",
        "t_soil_raw": "K",
        "lai": "m2/m2",
        "liq_vol": "m3/m3",
        "h2osfc": "mm",
        "h2osno": "mm",
        "snow_depth": "m",
        "snow_depth_eff": "m",
        "frac_sno": "fraction",
    }
    _outlier_max = {
        name: torch.full((ncell,), -torch.inf, device=device, dtype=dtype)
        for name in _outlier_thresholds
    }
    _outlier_step = {
        name: torch.full((ncell,), -1, device=device, dtype=torch.int32)
        for name in _outlier_thresholds
    }
    _outlier_pft_id = torch.full((ncell,), -1, device=device, dtype=torch.int32)
    _pft_id_t = torch.as_tensor(pft_ids, device=device, dtype=torch.int32)
    _lai_add_pft_mask = (
        torch.ones((npft, 1), device=device, dtype=dtype)
        if regional_lai_add_include_bare
        else (_pft_id_t != 0).to(dtype=dtype).reshape(npft, 1)
    )
    _universal_batch_idx_t = (
        torch.as_tensor(universal_batch_indices, device=device, dtype=torch.long)
        if universal_batch_indices
        else None
    )
    w_universal_dev = (
        w_dev.index_select(0, _universal_batch_idx_t)
        if _universal_batch_idx_t is not None
        else w_dev[:0]
    )
    _blend_target_names = (
        "qflx_tran_veg",
        "qflx_evap_canopy",
        "qflx_ev_soil",
        "qflx_ev_snow",
        "qflx_ev_h2osfc",
        "qflx_prec_grnd",
        "qflx_snow_grnd",
        "qflx_rain_grnd",
        "hs_soil",
        "hs_top_snow",
        "hs_h2osfc",
        "dhsdT",
        "eflx_sh_soil",
        "sabg_soil",
        "sabg_soil_learned",
        "sabg_soil_diagnostic",
        "sabg_soil_learned_minus_diagnostic",
        "dlrad",
        "cgrnds",
        "cgrndl",
        "surface_energy_closure_residual",
        "t_veg",
        "t_ref2m",
        "fpsn_patch",
        "eflx_sh_tot",
    )
    _clip_count_dtype = torch.float32 if device.type == "mps" else torch.float64
    _clip_counts = {
        "state_temperature": torch.zeros((), device=device, dtype=_clip_count_dtype),
        "photosynthesis": torch.zeros((), device=device, dtype=_clip_count_dtype),
        "thermal_target": torch.zeros((), device=device, dtype=_clip_count_dtype),
    }

    def _profile_start() -> float:
        if profile_sections is None:
            return 0.0
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elif device.type == "mps" and hasattr(torch, "mps"):
            torch.mps.synchronize()
        return time.perf_counter()

    def _profile_stop(name: str, started_at: float) -> None:
        if profile_sections is None:
            return
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elif device.type == "mps" and hasattr(torch, "mps"):
            torch.mps.synchronize()
        profile_sections[name] = profile_sections.get(name, 0.0) + (time.perf_counter() - started_at)

    def _profile_call(name: str, fn):
        started_at = _profile_start()
        out = fn()
        _profile_stop(name, started_at)
        return out

    def _target_dict_to_state_dtype(targets: dict) -> dict:
        if mlp_dtype == dtype:
            return targets
        out = {}
        for key, value in targets.items():
            if isinstance(value, torch.Tensor):
                out[key] = value.to(device=device, dtype=dtype)
            else:
                out[key] = torch.as_tensor(value, device=device, dtype=dtype)
        return out

    def _predict_surface_targets_mlp(bundle, features) -> dict:
        targets = predict_surface_targets(
            bundle,
            features,
            device=device,
            dtype=mlp_dtype,
        )
        return _target_dict_to_state_dtype(targets)

    def _apply_energy_closed_soil_boundary(
        targets: dict,
        frac_veg_nosno: "torch.Tensor",
        sabg_soil_diagnostic: "torch.Tensor | None" = None,
    ) -> dict:
        """Replace learned thermal boundaries with closed energy residuals.

        The original primitive contract reconstructs soil and the common
        derivative. The extended contract additionally reconstructs top-snow
        and standing-water heat fluxes. Legacy checkpoints are unchanged.
        """

        if not has_energy_closed_soil_targets(targets):
            return targets
        if sabg_soil_diagnostic is not None:
            sabg_diag = sabg_soil_diagnostic.to(device=device, dtype=dtype)
            targets["sabg_soil_learned"] = targets["sabg_soil"]
            targets["sabg_soil_diagnostic"] = sabg_diag
            targets["sabg_soil_learned_minus_diagnostic"] = (
                targets["sabg_soil"] - sabg_diag
            )
        if soil_energy_sabg_source == "diagnostic":
            if sabg_soil_diagnostic is None:
                raise RuntimeError(
                    "Diagnostic soil-energy SABG was requested, but the model input "
                    "does not contain the reconstructed sabg_snow feature"
                )
            targets["sabg_soil"] = targets["sabg_soil_diagnostic"]
        qsoil = targets["qflx_ev_soil"]

        def _match(value: "torch.Tensor") -> "torch.Tensor":
            value_t = value.to(device=device, dtype=dtype)
            while value_t.ndim < qsoil.ndim:
                value_t = value_t.unsqueeze(0)
            return value_t

        t_soil_top = _match(state.t_soisno[..., 0])
        t_grnd_now = _match(coupled_t_grnd_from_state_torch(state, device, dtype))
        _, _, t_snow_top_now = _snow_energy_feature_tensors(state)
        t_snow_top_now = _match(t_snow_top_now)
        t_h2osfc_now = _match(
            state.t_h2osfc if state.t_h2osfc is not None else state.t_soisno[..., 0]
        )
        flds_now = _match(flds.expand(-1, nens) if flds.shape[1] == 1 else flds)
        frac_veg_now = frac_veg_nosno.to(device=device, dtype=dtype)
        while frac_veg_now.ndim < qsoil.ndim:
            frac_veg_now = frac_veg_now.unsqueeze(-1)
        top_liq = state.h2osoi_liq[..., 0]
        top_ice = state.h2osoi_ice[..., 0]
        htvp_now = torch.where(
            (top_liq <= 0.0) & (top_ice > 0.0),
            torch.full_like(top_liq, LATENT_HEAT_SUBLIMATION),
            torch.full_like(top_liq, LATENT_HEAT_VAPORIZATION),
        )
        htvp_now = _match(htvp_now)
        full_surface_closure = has_energy_closed_surface_targets(targets)
        if full_surface_closure:
            hs_soil, hs_top_snow, hs_h2osfc, dhsdT, components = (
                derive_surface_thermal_boundaries_torch(
                    targets,
                    forc_lwrad=flds_now,
                    frac_veg_nosno=frac_veg_now,
                    t_soil_top=t_soil_top,
                    t_snow_top=t_snow_top_now,
                    t_h2osfc=t_h2osfc_now,
                    t_grnd=t_grnd_now,
                    htvp=htvp_now,
                    emg=0.96,
                )
            )
            targets["hs_top_snow"] = hs_top_snow
            targets["hs_h2osfc"] = hs_h2osfc
        else:
            hs_soil, dhsdT, components = derive_soil_thermal_boundary_torch(
                targets,
                forc_lwrad=flds_now,
                frac_veg_nosno=frac_veg_now,
                t_soil_top=t_soil_top,
                t_grnd=t_grnd_now,
                htvp=htvp_now,
                emg=0.96,
            )
        targets["hs_soil"] = hs_soil
        targets["dhsdT"] = dhsdT
        targets["surface_energy_closure_residual"] = torch.zeros_like(hs_soil)
        return targets

    def _cell_mean_tensor(value) -> "torch.Tensor":
        value_t = value if isinstance(value, torch.Tensor) else torch.as_tensor(value, device=device, dtype=dtype)
        value_t = value_t.to(device=device, dtype=dtype)
        return value_t.mean(1) if value_t.ndim > 1 else value_t

    def _update_outlier(name: str, values: "torch.Tensor") -> None:
        vals = torch.nan_to_num(values.to(device=device, dtype=dtype), nan=-torch.inf, posinf=torch.inf, neginf=-torch.inf)
        mask = vals > _outlier_max[name]
        _outlier_max[name] = torch.where(mask, vals, _outlier_max[name])
        _outlier_step[name] = torch.where(
            mask,
            torch.full_like(_outlier_step[name], step_i),
            _outlier_step[name],
        )

    def _safety_clamp_tensor(
        value,
        *,
        min_value: float,
        max_value: float,
        count_name: str,
    ):
        value_t = value if isinstance(value, torch.Tensor) else torch.as_tensor(value, device=device, dtype=dtype)
        value_t = value_t.to(device=device, dtype=dtype)
        clipped = torch.nan_to_num(
            value_t,
            nan=0.5 * (min_value + max_value),
            posinf=max_value,
            neginf=min_value,
        )
        clipped = torch.clamp(clipped, min=min_value, max=max_value)
        changed = (
            (~torch.isfinite(value_t))
            | (value_t < min_value)
            | (value_t > max_value)
        )
        _clip_counts[count_name] = _clip_counts[count_name] + changed.sum().to(dtype=_clip_count_dtype)
        return clipped

    def _clone_accum_value(value):
        if isinstance(value, torch.Tensor):
            return value.clone()
        if isinstance(value, list):
            return [
                None if item is None else (item.clone() if isinstance(item, torch.Tensor) else torch.as_tensor(item, device=device, dtype=dtype).clone())
                for item in value
            ]
        return value

    def _add_accum_value(acc, value):
        if isinstance(acc, torch.Tensor):
            value_t = value if isinstance(value, torch.Tensor) else torch.as_tensor(value, device=device, dtype=dtype)
            return acc.add(value_t.to(device=device, dtype=dtype))
        if isinstance(acc, list):
            value_l = value if isinstance(value, list) else [value] * len(acc)
            for idx, item in enumerate(value_l):
                if idx >= len(acc) or item is None:
                    continue
                item_t = item if isinstance(item, torch.Tensor) else torch.as_tensor(item, device=device, dtype=dtype)
                if acc[idx] is None:
                    acc[idx] = item_t.to(device=device, dtype=dtype).clone()
                else:
                    acc[idx].add_(item_t.to(device=device, dtype=dtype))
            return acc
        return acc

    def _accumulate_physics_targets(accum: "dict | None", targets: dict) -> dict:
        if accum is None:
            return {key: _clone_accum_value(value) for key, value in targets.items()}
        for key, value in targets.items():
            if key not in accum:
                accum[key] = _clone_accum_value(value)
            else:
                accum[key] = _add_accum_value(accum[key], value)
        return accum

    def _average_accum_value(value, count: int):
        if isinstance(value, torch.Tensor):
            return value / float(count)
        if isinstance(value, list):
            return [None if item is None else item / float(count) for item in value]
        return value

    def _average_physics_targets(accum: dict, count: int) -> dict:
        return {key: _average_accum_value(value, count) for key, value in accum.items()}

    def _clip_photosynthesis_targets(targets: dict) -> dict:
        if not safety_clamps:
            return targets
        for name in ("psnsun_patch", "psnsha_patch"):
            if name in targets:
                targets[name] = _safety_clamp_tensor(
                    targets[name],
                    min_value=0.0,
                    max_value=photosynthesis_max,
                    count_name="photosynthesis",
                )
        if "fpsn_patch" in targets:
            targets["fpsn_patch"] = _safety_clamp_tensor(
                targets["fpsn_patch"],
                min_value=0.0,
                max_value=photosynthesis_max,
                count_name="photosynthesis",
            )
        return targets

    _thermal_target_names = ("hs_soil", "hs_top_snow", "hs_h2osfc")
    _thermal_target_z_clamp = float(thermal_target_z_clamp)
    _thermal_target_ref_bundle = universal_ref_bundle or (pft_bundles[0] if pft_bundles else None)
    _thermal_target_limits: dict[str, tuple[float, float]] = {}
    if _thermal_target_z_clamp > 0.0 and _thermal_target_ref_bundle is not None:
        _target_idx = {name: idx for idx, name in enumerate(_thermal_target_ref_bundle.target_names)}
        for _target_name in _thermal_target_names:
            _stats_name = "hs_soil" if _target_name == "hs_top_snow" else _target_name
            _idx = _target_idx.get(_stats_name)
            if _idx is None:
                continue
            _mean = float(np.asarray(_thermal_target_ref_bundle.y_mean, dtype=np.float64)[_idx])
            _std = float(np.asarray(_thermal_target_ref_bundle.y_std, dtype=np.float64)[_idx])
            if np.isfinite(_mean) and np.isfinite(_std) and _std > 0.0:
                _span = _thermal_target_z_clamp * _std
                _thermal_target_limits[_target_name] = (_mean - _span, _mean + _span)

    def _clip_thermal_boundary_targets(targets: dict) -> dict:
        if not safety_clamps or not _thermal_target_limits:
            return targets
        for name, (lower, upper) in _thermal_target_limits.items():
            value = targets.get(name)
            if value is None:
                continue
            key = f"thermal_target_{name}"
            if key not in _clip_counts:
                _clip_counts[key] = torch.zeros((), device=device, dtype=_clip_count_dtype)
            before = _clip_counts["thermal_target"]
            targets[name] = _safety_clamp_tensor(
                value,
                min_value=lower,
                max_value=upper,
                count_name="thermal_target",
            )
            _clip_counts[key] = _clip_counts[key] + (_clip_counts["thermal_target"] - before)
        return targets

    def _bare_ground_target_fallback(
        targets: dict,
        p_idx: int,
        *,
        stacked: bool,
        stack_row: int | None = None,
    ) -> dict:
        if int(pft_ids[p_idx]) != 0:
            return targets
        using_patch0_model = int(getattr(pft_bundles[p_idx], "patch_index", pft_ids[p_idx])) == 0
        t_top = state.t_soisno[..., 0]
        air = tbot.expand_as(t_top)
        hs = torch.clamp(20.0 * (air - t_top), min=-150.0, max=150.0)
        dhsdT = torch.full_like(hs, -20.0)
        zero = torch.zeros_like(hs)

        def set_target(name: str, value: torch.Tensor) -> None:
            if name not in targets:
                return
            if stacked:
                target = targets[name]
                target[p_idx if stack_row is None else stack_row] = value.to(device=target.device, dtype=target.dtype)
            else:
                target = targets[name]
                targets[name] = value.to(device=target.device, dtype=target.dtype)

        for name in (
            "fpsn_patch",
            "psnsun_patch",
            "psnsha_patch",
            "laisun_patch",
            "laisha_patch",
            "qflx_tran_veg",
            "qflx_evap_canopy",
        ):
            set_target(name, zero)
        if using_patch0_model:
            return targets
        for name in ("hs_soil", "hs_top_snow", "hs_h2osfc"):
            set_target(name, hs)
        set_target("dhsdT", dhsdT)
        return targets

    def _state_with_h2ocan(state_in: "_CCS", h2ocan_value: "torch.Tensor") -> "_CCS":
        return _CCS(
            h2osoi_liq=state_in.h2osoi_liq,
            h2osoi_ice=state_in.h2osoi_ice,
            h2ocan=h2ocan_value,
            h2osfc=state_in.h2osfc,
            frac_h2osfc=state_in.frac_h2osfc,
            frac_sno=state_in.frac_sno,
            zwt=state_in.zwt,
            t_soisno=state_in.t_soisno,
            wa=state_in.wa,
            t_h2osfc=state_in.t_h2osfc,
            t10=state_in.t10,
            t10_history=state_in.t10_history,
            h2osno=state_in.h2osno,
            snow_depth=state_in.snow_depth,
            snow_t=state_in.snow_t,
            snow_layers=state_in.snow_layers,
            int_snow=state_in.int_snow,
        )

    def _select_state_cells_for_sparse_pairs(state_in: "_CCS", cell_idx: "torch.Tensor") -> "_CCS":
        def sel(value):
            if value is None:
                return None
            value_t = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
            if value_t.is_floating_point():
                value_t = value_t.to(device=device, dtype=dtype)
            elif value_t.device != device:
                value_t = value_t.to(device=device)
            if value_t.ndim >= 1 and value_t.shape[0] == ncell:
                return value_t.index_select(0, cell_idx)
            return value_t

        snow_layers = None
        if state_in.snow_layers is not None:
            sl = state_in.snow_layers
            sl_snw_rds = getattr(sl, "snw_rds", None)
            snow_layers = SnowLayerState(
                snl=sel(sl.snl),
                dz=sel(sl.dz),
                h2osoi_ice=sel(sl.h2osoi_ice),
                h2osoi_liq=sel(sl.h2osoi_liq),
                t_snow=sel(sl.t_snow),
                snw_rds=sel(sl_snw_rds) if sl_snw_rds is not None else None,
            )

        return _CCS(
            h2osoi_liq=sel(state_in.h2osoi_liq),
            h2osoi_ice=sel(state_in.h2osoi_ice),
            h2ocan=sel(state_in.h2ocan),
            h2osfc=sel(state_in.h2osfc),
            frac_h2osfc=sel(state_in.frac_h2osfc),
            frac_sno=sel(state_in.frac_sno),
            zwt=sel(state_in.zwt),
            t_soisno=sel(state_in.t_soisno),
            wa=sel(state_in.wa),
            t_h2osfc=sel(state_in.t_h2osfc),
            t10=sel(state_in.t10),
            t10_history=sel(state_in.t10_history),
            h2osno=sel(state_in.h2osno),
            snow_depth=sel(state_in.snow_depth),
            snow_t=sel(state_in.snow_t),
            snow_layers=snow_layers,
            int_snow=sel(state_in.int_snow),
        )

    def _default_canopy_precip_targets(bundle, features: "torch.Tensor") -> dict[str, "torch.Tensor"]:
        idx_map = _bundle_input_index_map(tuple(bundle.input_names))
        precip_rain = torch.clamp(features[..., idx_map["forc_rain"]], min=0.0)
        precip_snow = torch.clamp(features[..., idx_map["forc_snow"]], min=0.0)
        zero = torch.zeros_like(precip_rain)
        return {
            "qflx_prec_grnd": precip_rain + precip_snow,
            "qflx_snow_grnd": precip_snow.clone(),
            "qflx_rain_grnd": precip_rain.clone(),
            "qflx_leafdrip": zero,
        }

    def _precip_active_cells_for_interception(
        bundle,
        features: "torch.Tensor",
        candidate_cells: "torch.Tensor | None" = None,
    ) -> "torch.Tensor":
        idx_map = _bundle_input_index_map(tuple(bundle.input_names))
        if candidate_cells is not None:
            if candidate_cells.numel() == 0:
                return candidate_cells
            feat = features.index_select(0, candidate_cells)
        else:
            feat = features
        precip_total = (
            torch.clamp(feat[..., idx_map["forc_rain"]], min=0.0)
            + torch.clamp(feat[..., idx_map["forc_snow"]], min=0.0)
        )
        if precip_total.ndim <= 1:
            active = precip_total > 0.0
        else:
            active = torch.any(precip_total > 0.0, dim=tuple(range(1, precip_total.ndim)))
        active_pos = torch.nonzero(active, as_tuple=False).flatten()
        if candidate_cells is not None:
            return candidate_cells.index_select(0, active_pos)
        return active_pos.to(device=features.device, dtype=torch.long)

    def _h2ocan_like_features(state_in: "_CCS", features: "torch.Tensor") -> "torch.Tensor":
        h2ocan_t = (
            state_in.h2ocan
            if isinstance(state_in.h2ocan, torch.Tensor)
            else torch.as_tensor(state_in.h2ocan, device=device, dtype=dtype)
        ).to(device=device, dtype=dtype)
        target_shape = tuple(features.shape[:-1])
        while h2ocan_t.ndim < len(target_shape):
            h2ocan_t = h2ocan_t.unsqueeze(0)
        return torch.broadcast_to(h2ocan_t, target_shape).clone()

    def _state_tensor_like_features(value, features: "torch.Tensor", *, clone: bool = False) -> "torch.Tensor":
        value_t = (
            value
            if isinstance(value, torch.Tensor)
            else torch.as_tensor(value, device=device, dtype=dtype)
        ).to(device=device)
        if value_t.is_floating_point():
            value_t = value_t.to(dtype=dtype)
        target_shape = tuple(features.shape[:-1])
        while value_t.ndim < len(target_shape):
            value_t = value_t.unsqueeze(0)
        value_t = torch.broadcast_to(value_t, target_shape)
        return value_t.clone() if clone else value_t

    def _apply_canopy_interception_fast(
        bundle,
        features: "torch.Tensor",
        state_in: "_CCS",
        h2ocan_old: "torch.Tensor | None" = None,
        frac_sno_value: "torch.Tensor | None" = None,
    ):
        idx_map = _bundle_input_index_map(tuple(bundle.input_names))
        precip_rain = torch.clamp(features[..., idx_map["forc_rain"]], min=0.0)
        precip_snow = torch.clamp(features[..., idx_map["forc_snow"]], min=0.0)
        precip_total = precip_rain + precip_snow
        zero = torch.zeros_like(precip_total)

        elai_idx = idx_map.get("elai_patch")
        esai_idx = idx_map.get("esai_patch")
        elai = features[..., elai_idx] if elai_idx is not None else zero
        esai = features[..., esai_idx] if esai_idx is not None else zero
        vegt = torch.clamp(elai + esai, min=0.0)

        frac_veg_idx = idx_map.get("frac_veg_nosno_patch")
        if frac_veg_idx is not None:
            frac_veg_nosno = torch.clamp(features[..., frac_veg_idx], min=0.0, max=1.0)
        else:
            frac_sno_t = (
                _state_tensor_like_features(frac_sno_value, features)
                if frac_sno_value is not None
                else _state_tensor_like_features(state_in.frac_sno, features)
            )
            snow_free = 1.0 - torch.clamp(frac_sno_t, min=0.0, max=1.0)
            frac_veg_nosno = torch.where(vegt > 0.0, snow_free, zero)

        h2ocan_old_t = (
            _state_tensor_like_features(h2ocan_old, features)
            if h2ocan_old is not None
            else _state_tensor_like_features(state_in.h2ocan, features)
        )
        active_canopy = (frac_veg_nosno > 0.0) & (precip_total > 0.0)
        inv_precip = torch.where(
            precip_total > 0.0,
            torch.reciprocal(torch.clamp(precip_total, min=1.0e-30)),
            zero,
        )
        fracsnow = precip_snow * inv_precip
        fracrain = precip_rain * inv_precip
        fpi = 0.25 * (1.0 - torch.exp(-0.5 * vegt))
        through_snow = precip_snow * (1.0 - fpi)
        through_rain = precip_rain * (1.0 - fpi)
        intercepted = precip_total * fpi
        h2ocan_after_interception = torch.clamp(
            h2ocan_old_t + dt * torch.where(active_canopy, intercepted, zero),
            min=0.0,
        )
        capacity = 0.1 * vegt
        candrip = torch.where(
            active_canopy,
            torch.clamp((h2ocan_after_interception - capacity) / dt, min=0.0),
            zero,
        )
        h2ocan_after_drip = torch.where(
            active_canopy & (candrip > 0.0),
            capacity,
            h2ocan_after_interception,
        )
        qflx_snow_grnd = torch.where(active_canopy, through_snow + candrip * fracsnow, precip_snow)
        qflx_rain_grnd = torch.where(active_canopy, through_rain + candrip * fracrain, precip_rain)
        qflx_leafdrip = torch.where(active_canopy, candrip * fracrain, zero)
        targets = {
            "qflx_prec_grnd": torch.clamp(qflx_snow_grnd + qflx_rain_grnd, min=0.0),
            "qflx_snow_grnd": qflx_snow_grnd,
            "qflx_rain_grnd": qflx_rain_grnd,
            "qflx_leafdrip": qflx_leafdrip,
        }
        return targets, _state_with_h2ocan(state_in, h2ocan_after_drip)

    def _apply_canopy_interception_replay(bundle, features, state_in: "_CCS"):
        if isinstance(features, torch.Tensor):
            return _apply_canopy_interception_fast(bundle, features, state_in)
        return apply_canopy_interception(bundle, features, state_in, dt)

    def _apply_canopy_interception_sparse_pft(
        bundle,
        features: "torch.Tensor",
        state_in: "_CCS",
        cell_idx: "torch.Tensor | None",
    ):
        active_cell_idx = _precip_active_cells_for_interception(bundle, features, cell_idx)
        if active_cell_idx.numel() == 0:
            targets = _default_canopy_precip_targets(bundle, features)
            return targets, _state_with_h2ocan(state_in, _h2ocan_like_features(state_in, features))
        if active_cell_idx.numel() >= ncell:
            return _apply_canopy_interception_fast(bundle, features, state_in)
        sparse_features = features.index_select(0, active_cell_idx)
        sparse_h2ocan_old = _state_tensor_like_features(state_in.h2ocan, features).index_select(0, active_cell_idx)
        sparse_frac_sno = None
        if "frac_veg_nosno_patch" not in tuple(bundle.input_names):
            sparse_frac_sno = _state_tensor_like_features(state_in.frac_sno, features).index_select(0, active_cell_idx)
        sparse_targets, sparse_state_out = _apply_canopy_interception_fast(
            bundle,
            sparse_features,
            state_in,
            h2ocan_old=sparse_h2ocan_old,
            frac_sno_value=sparse_frac_sno,
        )
        targets = _default_canopy_precip_targets(bundle, features)
        for name, sparse_value in sparse_targets.items():
            sparse_value_t = (
                sparse_value
                if isinstance(sparse_value, torch.Tensor)
                else torch.as_tensor(sparse_value, device=device, dtype=dtype)
            ).to(device=device, dtype=dtype)
            targets[name].index_copy_(0, active_cell_idx, sparse_value_t)
        h2ocan_full = _h2ocan_like_features(state_in, features)
        sparse_h2ocan = (
            sparse_state_out.h2ocan
            if isinstance(sparse_state_out.h2ocan, torch.Tensor)
            else torch.as_tensor(sparse_state_out.h2ocan, device=device, dtype=dtype)
        ).to(device=device, dtype=dtype)
        h2ocan_full.index_copy_(0, active_cell_idx, sparse_h2ocan)
        return targets, _state_with_h2ocan(state_in, h2ocan_full)

    def _apply_canopy_interception_packed_pft(
        bundle,
        features: "torch.Tensor",
        state_in: "_CCS",
        cell_idx: "torch.Tensor",
    ):
        """Run individual-PFT interception in packed active-cell order."""
        active_pos = _precip_active_cells_for_interception(bundle, features)
        sparse_targets = _default_canopy_precip_targets(bundle, features)
        h2ocan_global = (
            state_in.h2ocan
            if isinstance(state_in.h2ocan, torch.Tensor)
            else torch.as_tensor(state_in.h2ocan, device=device, dtype=dtype)
        ).to(device=device, dtype=dtype)
        sparse_h2ocan = h2ocan_global.index_select(0, cell_idx)
        while sparse_h2ocan.ndim < features.ndim - 1:
            sparse_h2ocan = sparse_h2ocan.unsqueeze(-1)
        sparse_h2ocan = torch.broadcast_to(sparse_h2ocan, features.shape[:-1]).clone()
        if active_pos.numel() > 0:
            active_global_cells = cell_idx.index_select(0, active_pos)
            active_features = features.index_select(0, active_pos)
            active_state = _select_state_cells_for_sparse_pairs(state_in, active_global_cells)
            active_targets, active_state_out = _apply_canopy_interception_fast(
                bundle,
                active_features,
                active_state,
            )
            for name, active_value in active_targets.items():
                sparse_targets[name].index_copy_(0, active_pos, active_value)
            active_h2ocan = (
                active_state_out.h2ocan
                if isinstance(active_state_out.h2ocan, torch.Tensor)
                else torch.as_tensor(active_state_out.h2ocan, device=device, dtype=dtype)
            ).to(device=device, dtype=dtype)
            sparse_h2ocan.index_copy_(0, active_pos, active_h2ocan)
        full_targets: dict[str, torch.Tensor] = {}
        for name, sparse_value in sparse_targets.items():
            sparse_value_t = (
                sparse_value
                if isinstance(sparse_value, torch.Tensor)
                else torch.as_tensor(sparse_value, device=device, dtype=dtype)
            ).to(device=device, dtype=dtype)
            full_value = torch.zeros(
                (ncell,) + tuple(sparse_value_t.shape[1:]),
                device=device,
                dtype=dtype,
            )
            full_value.index_copy_(0, cell_idx, sparse_value_t)
            full_targets[name] = full_value
        h2ocan_full = torch.zeros(
            (ncell,) + tuple(sparse_h2ocan.shape[1:]),
            device=device,
            dtype=dtype,
        )
        h2ocan_full.index_copy_(0, cell_idx, sparse_h2ocan)
        return full_targets, _state_with_h2ocan(state_in, h2ocan_full), sparse_h2ocan

    def _apply_canopy_interception_sparse_stack(bundle, features: "torch.Tensor", state_in: "_CCS"):
        if (
            not universal_sparse_mlp_enabled
            or universal_sparse_pft_idx_t is None
            or universal_sparse_weight_pft_idx_t is None
            or universal_sparse_cell_idx_t is None
            or universal_sparse_pft_idx_t.numel() == 0
        ):
            return _apply_canopy_interception_fast(bundle, features, state_in)
        if universal_sparse_pft_idx_t.numel() >= len(universal_batch_indices) * ncell:
            return _apply_canopy_interception_fast(bundle, features, state_in)
        idx_map = _bundle_input_index_map(tuple(bundle.input_names))
        sparse_features = features[
            universal_sparse_pft_idx_t,
            universal_sparse_cell_idx_t,
            ...,
        ]
        sparse_precip_total = (
            torch.clamp(sparse_features[..., idx_map["forc_rain"]], min=0.0)
            + torch.clamp(sparse_features[..., idx_map["forc_snow"]], min=0.0)
        )
        if sparse_precip_total.ndim <= 1:
            active_pair = sparse_precip_total > 0.0
        else:
            active_pair = torch.any(sparse_precip_total > 0.0, dim=tuple(range(1, sparse_precip_total.ndim)))
        active_pair_pos = torch.nonzero(active_pair, as_tuple=False).flatten()
        if active_pair_pos.numel() == 0:
            targets = _default_canopy_precip_targets(bundle, features)
            return targets, _state_with_h2ocan(state_in, _h2ocan_like_features(state_in, features))
        active_pft_idx = universal_sparse_pft_idx_t.index_select(0, active_pair_pos)
        active_cell_idx = universal_sparse_cell_idx_t.index_select(0, active_pair_pos)
        active_sparse_features = sparse_features.index_select(0, active_pair_pos)
        h2ocan_dense = _state_tensor_like_features(state_in.h2ocan, features)
        sparse_h2ocan_old = h2ocan_dense[
            active_pft_idx,
            active_cell_idx,
            ...,
        ]
        sparse_frac_sno = None
        if "frac_veg_nosno_patch" not in tuple(bundle.input_names):
            frac_sno_dense = _state_tensor_like_features(state_in.frac_sno, features)
            sparse_frac_sno = frac_sno_dense[
                active_pft_idx,
                active_cell_idx,
                ...,
            ]
        sparse_targets, sparse_state_out = _apply_canopy_interception_fast(
            bundle,
            active_sparse_features,
            state_in,
            h2ocan_old=sparse_h2ocan_old,
            frac_sno_value=sparse_frac_sno,
        )
        targets = _default_canopy_precip_targets(bundle, features)
        for name, sparse_value in sparse_targets.items():
            sparse_value_t = (
                sparse_value
                if isinstance(sparse_value, torch.Tensor)
                else torch.as_tensor(sparse_value, device=device, dtype=dtype)
            ).to(device=device, dtype=dtype)
            targets[name][
                active_pft_idx,
                active_cell_idx,
                ...,
            ] = sparse_value_t
        h2ocan_full = _h2ocan_like_features(state_in, features)
        sparse_h2ocan = (
            sparse_state_out.h2ocan
            if isinstance(sparse_state_out.h2ocan, torch.Tensor)
            else torch.as_tensor(sparse_state_out.h2ocan, device=device, dtype=dtype)
        ).to(device=device, dtype=dtype)
        h2ocan_full[
            active_pft_idx,
            active_cell_idx,
            ...,
        ] = sparse_h2ocan
        return targets, _state_with_h2ocan(state_in, h2ocan_full)

    def _apply_canopy_interception_sparse_stack_blended(bundle, features: "torch.Tensor", state_in: "_CCS"):
        """Sparse universal interception for production runs.

        Returns area-weighted precipitation-to-ground fields directly, avoiding
        dense per-PFT precipitation tensors that are only needed by single-cell
        diagnostics.
        """
        def blend_stack(value: "torch.Tensor") -> "torch.Tensor":
            w = w_universal_dev
            while w.ndim < value.ndim:
                w = w.unsqueeze(-1)
            return torch.sum(w * value, dim=0)

        if (
            not universal_sparse_mlp_enabled
            or universal_sparse_pft_idx_t is None
            or universal_sparse_weight_pft_idx_t is None
            or universal_sparse_cell_idx_t is None
            or universal_sparse_pft_idx_t.numel() == 0
        ):
            dense_targets, dense_state = _apply_canopy_interception_fast(bundle, features, state_in)
            return (
                {name: blend_stack(value) for name, value in dense_targets.items() if name != "qflx_leafdrip"},
                dense_state,
            )
        if universal_sparse_pft_idx_t.numel() >= len(universal_batch_indices) * ncell:
            dense_targets, dense_state = _apply_canopy_interception_fast(bundle, features, state_in)
            return (
                {name: blend_stack(value) for name, value in dense_targets.items() if name != "qflx_leafdrip"},
                dense_state,
            )

        idx_map = _bundle_input_index_map(tuple(bundle.input_names))
        rain_idx = idx_map["forc_rain"]
        snow_idx = idx_map["forc_snow"]
        elai_idx = idx_map.get("elai_patch")
        esai_idx = idx_map.get("esai_patch")
        frac_veg_idx = idx_map.get("frac_veg_nosno_patch")

        if not snow_model_enabled:
            t_canopy_detail = _profile_start()
            pair_pft_idx = universal_sparse_pft_idx_t
            pair_cell_idx = universal_sparse_cell_idx_t
            pair_rain = torch.clamp(
                features[
                    pair_pft_idx,
                    pair_cell_idx,
                    ...,
                    rain_idx,
                ],
                min=0.0,
            )
            zero = torch.zeros_like(pair_rain)
            pair_elai = (
                features[pair_pft_idx, pair_cell_idx, ..., elai_idx]
                if elai_idx is not None
                else zero
            )
            pair_esai = (
                features[pair_pft_idx, pair_cell_idx, ..., esai_idx]
                if esai_idx is not None
                else zero
            )
            pair_vegt = torch.clamp(pair_elai + pair_esai, min=0.0)
            if frac_veg_idx is not None:
                pair_frac_veg_nosno = torch.clamp(
                    features[pair_pft_idx, pair_cell_idx, ..., frac_veg_idx],
                    min=0.0,
                    max=1.0,
                )
            else:
                pair_frac_veg_nosno = torch.where(pair_vegt > 0.0, torch.ones_like(pair_vegt), zero)
            h2ocan_dense = _state_tensor_like_features(state_in.h2ocan, features)
            pair_h2ocan_old = h2ocan_dense[
                pair_pft_idx,
                pair_cell_idx,
                ...,
            ]
            _profile_stop("canopy:rain_setup", t_canopy_detail)

            t_canopy_detail = _profile_start()
            active_canopy = (pair_frac_veg_nosno > 0.0) & (pair_rain > 0.0)
            fpi = 0.25 * (1.0 - torch.exp(-0.5 * pair_vegt))
            through_rain = pair_rain * (1.0 - fpi)
            pair_h2ocan_after_interception = torch.clamp(
                pair_h2ocan_old + dt * torch.where(active_canopy, pair_rain * fpi, zero),
                min=0.0,
            )
            capacity = 0.1 * pair_vegt
            candrip = torch.where(
                active_canopy,
                torch.clamp((pair_h2ocan_after_interception - capacity) / dt, min=0.0),
                zero,
            )
            pair_h2ocan_after_drip = torch.where(
                active_canopy & (candrip > 0.0),
                capacity,
                pair_h2ocan_after_interception,
            )
            pair_rain_grnd = torch.where(active_canopy, through_rain + candrip, pair_rain)
            _profile_stop("canopy:rain_math", t_canopy_detail)

            t_canopy_detail = _profile_start()
            blended_rain = torch.clamp(features[0, ..., rain_idx], min=0.0).clone()
            w_pair = w_dev[
                universal_sparse_weight_pft_idx_t,
                pair_cell_idx,
                ...,
            ]
            blended_rain.index_add_(
                0,
                pair_cell_idx,
                w_pair * (pair_rain_grnd - pair_rain),
            )
            blended_snow = torch.zeros_like(blended_rain)
            blended_targets = {
                "qflx_rain_grnd": blended_rain,
                "qflx_snow_grnd": blended_snow,
                "qflx_prec_grnd": torch.clamp(blended_rain, min=0.0),
            }
            _profile_stop("canopy:rain_blend", t_canopy_detail)

            t_canopy_detail = _profile_start()
            h2ocan_full = h2ocan_dense.clone()
            h2ocan_full[
                pair_pft_idx,
                pair_cell_idx,
                ...,
            ] = pair_h2ocan_after_drip
            state_out = _state_with_h2ocan(state_in, h2ocan_full)
            _profile_stop("canopy:rain_state", t_canopy_detail)
            return blended_targets, state_out

        # Gather only the few columns needed for interception. Pulling the full
        # MLP feature vector here was expensive in the global sparse path.
        sparse_precip_rain = torch.clamp(
            features[
                universal_sparse_pft_idx_t,
                universal_sparse_cell_idx_t,
                ...,
                rain_idx,
            ],
            min=0.0,
        )
        sparse_precip_snow = torch.clamp(
            features[
                universal_sparse_pft_idx_t,
                universal_sparse_cell_idx_t,
                ...,
                snow_idx,
            ],
            min=0.0,
        )
        sparse_precip_total = sparse_precip_rain + sparse_precip_snow
        if sparse_precip_total.ndim <= 1:
            active_pair = sparse_precip_total > 0.0
        else:
            active_pair = torch.any(sparse_precip_total > 0.0, dim=tuple(range(1, sparse_precip_total.ndim)))
        active_pair_pos = torch.nonzero(active_pair, as_tuple=False).flatten()

        # Because rain/snow forcing is column-level, the un-intercepted
        # area-weighted default is just the first PFT's forcing.
        rain0 = torch.clamp(features[0, ..., rain_idx], min=0.0)
        snow0 = torch.clamp(features[0, ..., snow_idx], min=0.0)
        blended_rain = rain0.clone()
        blended_snow = snow0.clone()

        if active_pair_pos.numel() == 0:
            blended_targets = {
                "qflx_rain_grnd": blended_rain,
                "qflx_snow_grnd": blended_snow,
            }
            blended_targets["qflx_prec_grnd"] = torch.clamp(blended_rain + blended_snow, min=0.0)
            return blended_targets, _state_with_h2ocan(state_in, _h2ocan_like_features(state_in, features))

        active_pft_idx = universal_sparse_pft_idx_t.index_select(0, active_pair_pos)
        active_weight_pft_idx = universal_sparse_weight_pft_idx_t.index_select(0, active_pair_pos)
        active_cell_idx = universal_sparse_cell_idx_t.index_select(0, active_pair_pos)
        active_base_rain = sparse_precip_rain.index_select(0, active_pair_pos)
        active_base_snow = sparse_precip_snow.index_select(0, active_pair_pos)
        active_precip_total = active_base_rain + active_base_snow
        zero = torch.zeros_like(active_precip_total)
        active_elai = (
            features[active_pft_idx, active_cell_idx, ..., elai_idx]
            if elai_idx is not None
            else zero
        )
        active_esai = (
            features[active_pft_idx, active_cell_idx, ..., esai_idx]
            if esai_idx is not None
            else zero
        )
        active_vegt = torch.clamp(active_elai + active_esai, min=0.0)
        if frac_veg_idx is not None:
            active_frac_veg_nosno = torch.clamp(
                features[active_pft_idx, active_cell_idx, ..., frac_veg_idx],
                min=0.0,
                max=1.0,
            )
        else:
            frac_sno_dense = _state_tensor_like_features(state_in.frac_sno, features)
            active_frac_sno = frac_sno_dense[
                active_pft_idx,
                active_cell_idx,
                ...,
            ]
            snow_free = 1.0 - torch.clamp(active_frac_sno, min=0.0, max=1.0)
            active_frac_veg_nosno = torch.where(active_vegt > 0.0, snow_free, zero)
        h2ocan_dense = _state_tensor_like_features(state_in.h2ocan, features)
        sparse_h2ocan_old = h2ocan_dense[
            active_pft_idx,
            active_cell_idx,
            ...,
        ]

        active_canopy = (active_frac_veg_nosno > 0.0) & (active_precip_total > 0.0)
        inv_precip = torch.where(
            active_precip_total > 0.0,
            torch.reciprocal(torch.clamp(active_precip_total, min=1.0e-30)),
            zero,
        )
        fracsnow = active_base_snow * inv_precip
        fracrain = active_base_rain * inv_precip
        fpi = 0.25 * (1.0 - torch.exp(-0.5 * active_vegt))
        through_snow = active_base_snow * (1.0 - fpi)
        through_rain = active_base_rain * (1.0 - fpi)
        intercepted = active_precip_total * fpi
        h2ocan_after_interception = torch.clamp(
            sparse_h2ocan_old + dt * torch.where(active_canopy, intercepted, zero),
            min=0.0,
        )
        capacity = 0.1 * active_vegt
        candrip = torch.where(
            active_canopy,
            torch.clamp((h2ocan_after_interception - capacity) / dt, min=0.0),
            zero,
        )
        sparse_h2ocan = torch.where(
            active_canopy & (candrip > 0.0),
            capacity,
            h2ocan_after_interception,
        )
        sparse_rain_grnd = torch.where(active_canopy, through_rain + candrip * fracrain, active_base_rain)
        sparse_snow_grnd = torch.where(active_canopy, through_snow + candrip * fracsnow, active_base_snow)
        w_pair = w_dev[
            active_weight_pft_idx,
            active_cell_idx,
            ...,
        ]
        blended_rain.index_add_(
            0,
            active_cell_idx,
            w_pair * (sparse_rain_grnd - active_base_rain),
        )
        blended_snow.index_add_(
            0,
            active_cell_idx,
            w_pair * (sparse_snow_grnd - active_base_snow),
        )

        h2ocan_full = h2ocan_dense.clone()
        h2ocan_full[
            active_pft_idx,
            active_cell_idx,
            ...,
        ] = sparse_h2ocan
        blended_targets = {
            "qflx_rain_grnd": blended_rain,
            "qflx_snow_grnd": blended_snow,
        }
        blended_targets["qflx_prec_grnd"] = torch.clamp(blended_rain + blended_snow, min=0.0)
        return blended_targets, _state_with_h2ocan(state_in, h2ocan_full)

    def _predict_universal_trait_mean(
        bundle,
        pair_features: "torch.Tensor",
        pair_local_pft_idx: "torch.Tensor",
    ) -> tuple[dict, dict | None]:
        if (
            not universal_trait_enabled
            or universal_trait_feature_indices_t is None
            or universal_trait_z is None
            or universal_trait_low is None
            or universal_trait_high is None
        ):
            targets = _profile_call(
                "mlp_forward",
                lambda: _predict_surface_targets_mlp(
                    bundle,
                    pair_features,
                ),
            )
            return targets, None
        n_pairs = int(pair_features.shape[0])
        accum: dict[str, "torch.Tensor"] = {}
        accum_sq: dict[str, "torch.Tensor"] = {}
        trait_cols = universal_trait_feature_indices_t
        low = universal_trait_low.reshape(1, 1, 1, -1)
        high = universal_trait_high.reshape(1, 1, 1, -1)
        base_trait_values = pair_features.index_select(-1, trait_cols)
        trait_sd = torch.abs(base_trait_values) * trait_relative_sd
        for start in range(0, trait_ensemble_size, trait_chunk_size):
            stop = min(trait_ensemble_size, start + trait_chunk_size)
            chunk = stop - start
            t_trait0 = _profile_start()
            work = pair_features.unsqueeze(1).expand(n_pairs, chunk, *pair_features.shape[1:]).clone()
            z_trait = universal_trait_z[pair_local_pft_idx, start:stop, :]
            varied = base_trait_values.unsqueeze(1) + z_trait.unsqueeze(2) * trait_sd.unsqueeze(1)
            varied = torch.maximum(torch.minimum(varied, high), low)
            if start == 0:
                varied[:, 0, ...] = base_trait_values
            work[..., trait_cols] = varied
            _profile_stop("trait_feature_expansion", t_trait0)

            chunk_targets = _profile_call(
                "mlp_forward",
                lambda work=work: _predict_surface_targets_mlp(
                    bundle,
                    work,
                ),
            )
            chunk_targets = _clip_photosynthesis_targets(chunk_targets)
            for target_name, target_value in chunk_targets.items():
                value_t = (
                    target_value
                    if isinstance(target_value, torch.Tensor)
                    else torch.as_tensor(target_value, device=device, dtype=dtype)
                ).to(device=device, dtype=dtype)
                value_sum = value_t.sum(dim=1)
                value_sumsq = (value_t * value_t).sum(dim=1)
                accum[target_name] = (
                    value_sum
                    if target_name not in accum
                    else accum[target_name] + value_sum
                )
                accum_sq[target_name] = (
                    value_sumsq
                    if target_name not in accum_sq
                    else accum_sq[target_name] + value_sumsq
                )
        inv_ntrait = 1.0 / float(trait_ensemble_size)
        mean_targets = {name: value * inv_ntrait for name, value in accum.items()}
        var_targets = {
            name: torch.clamp(accum_sq[name] * inv_ntrait - mean_value * mean_value, min=0.0)
            for name, mean_value in mean_targets.items()
            if name in accum_sq
        }
        return mean_targets, var_targets

    def _predict_universal_targets_stack(bundle, features: "torch.Tensor") -> tuple[dict, dict | None]:
        if (
            universal_sparse_mlp_enabled
            and universal_sparse_pft_idx_t is not None
            and universal_sparse_cell_idx_t is not None
            and 0 < universal_sparse_pft_idx_t.numel() < len(universal_batch_indices) * ncell
        ):
            sparse_features = (
                features
                if features is universal_sparse_feature_batch
                else features[
                    universal_sparse_pft_idx_t,
                    universal_sparse_cell_idx_t,
                    ...,
                ]
            )
            sparse_targets, sparse_trait_vars = _predict_universal_trait_mean(
                bundle,
                sparse_features,
                universal_sparse_pft_idx_t,
            )
            targets_stack = {}
            for target_name, sparse_value in sparse_targets.items():
                sparse_value_t = (
                    sparse_value
                    if isinstance(sparse_value, torch.Tensor)
                    else torch.as_tensor(sparse_value, device=device, dtype=dtype)
                ).to(device=device, dtype=dtype)
                full_shape = (len(universal_batch_indices), ncell) + tuple(sparse_value_t.shape[1:])
                full_value = torch.zeros(full_shape, device=device, dtype=dtype)
                full_value[
                    universal_sparse_pft_idx_t,
                    universal_sparse_cell_idx_t,
                    ...,
                ] = sparse_value_t
                targets_stack[target_name] = full_value
            trait_var_stack = None
            if sparse_trait_vars is not None:
                trait_var_stack = {}
                for target_name, sparse_value in sparse_trait_vars.items():
                    sparse_value_t = (
                        sparse_value
                        if isinstance(sparse_value, torch.Tensor)
                        else torch.as_tensor(sparse_value, device=device, dtype=dtype)
                    ).to(device=device, dtype=dtype)
                    full_shape = (len(universal_batch_indices), ncell) + tuple(sparse_value_t.shape[1:])
                    full_value = torch.zeros(full_shape, device=device, dtype=dtype)
                    full_value[
                        universal_sparse_pft_idx_t,
                        universal_sparse_cell_idx_t,
                        ...,
                    ] = sparse_value_t
                    trait_var_stack[target_name] = full_value
            return targets_stack, trait_var_stack
        if universal_trait_enabled:
            nuniversal = len(universal_batch_indices)
            pair_features = features.reshape(nuniversal * ncell, *features.shape[2:])
            pair_local_pft_idx = (
                torch.arange(nuniversal, device=device, dtype=torch.long)
                .reshape(nuniversal, 1)
                .expand(nuniversal, ncell)
                .reshape(-1)
            )
            pair_targets, pair_trait_vars = _predict_universal_trait_mean(bundle, pair_features, pair_local_pft_idx)
            targets_stack = {
                target_name: target_value.reshape(nuniversal, ncell, *target_value.shape[1:])
                for target_name, target_value in pair_targets.items()
            }
            trait_var_stack = (
                {
                    target_name: target_value.reshape(nuniversal, ncell, *target_value.shape[1:])
                    for target_name, target_value in pair_trait_vars.items()
                }
                if pair_trait_vars is not None
                else None
            )
            return targets_stack, trait_var_stack
        targets = _profile_call(
            "mlp_forward",
            lambda: _predict_surface_targets_mlp(
                bundle,
                features,
            ),
        )
        return targets, None

    def _clip_state_temperatures(state_in: "_CCS") -> "_CCS":
        if not safety_clamps:
            return state_in

        def _clip_temp(value, field_name: str):
            if value is None:
                return None
            key = f"state_temperature_{field_name}"
            if key not in _clip_counts:
                _clip_counts[key] = torch.zeros((), device=device, dtype=_clip_count_dtype)
            before = _clip_counts["state_temperature"]
            clipped = _safety_clamp_tensor(
                value,
                min_value=temperature_min_k,
                max_value=temperature_max_k,
                count_name="state_temperature",
            )
            _clip_counts[key] = _clip_counts[key] + (_clip_counts["state_temperature"] - before)
            return clipped

        def _clip_active_snow_temp(value, snl_value, field_name: str):
            key = f"state_temperature_{field_name}"
            if key not in _clip_counts:
                _clip_counts[key] = torch.zeros((), device=device, dtype=_clip_count_dtype)
            before = _clip_counts["state_temperature"]
            clipped = _clip_active_snow_temp_impl(value, snl_value)
            _clip_counts[key] = _clip_counts[key] + (_clip_counts["state_temperature"] - before)
            return clipped

        def _clip_active_snow_temp_impl(value, snl_value):
            if value is None:
                return None
            if snl_value is None:
                return _clip_temp(value, "snow_no_layer")
            value_t = value if isinstance(value, torch.Tensor) else torch.as_tensor(value, device=device, dtype=dtype)
            value_t = value_t.to(device=device, dtype=dtype)
            snl_t = torch.as_tensor(snl_value, device=device, dtype=torch.int32)
            if value_t.ndim != snl_t.ndim + 1:
                return _clip_temp(value, "snow_shape_fallback")
            layer_idx = torch.arange(value_t.shape[-1], device=device, dtype=torch.int32)
            active = layer_idx < torch.clamp(-snl_t, min=0).unsqueeze(-1)
            clipped = torch.nan_to_num(
                value_t,
                nan=0.5 * (temperature_min_k + temperature_max_k),
                posinf=temperature_max_k,
                neginf=temperature_min_k,
            )
            clipped = torch.clamp(clipped, min=temperature_min_k, max=temperature_max_k)
            changed = active & (
                (~torch.isfinite(value_t))
                | (value_t < temperature_min_k)
                | (value_t > temperature_max_k)
            )
            _clip_counts["state_temperature"] = (
                _clip_counts["state_temperature"] + changed.sum().to(dtype=_clip_count_dtype)
            )
            return torch.where(active, clipped, torch.full_like(clipped, 273.15))

        def _unused_old_clip_temp(value):
            if value is None:
                return None
            return _safety_clamp_tensor(
                value,
                min_value=temperature_min_k,
                max_value=temperature_max_k,
                count_name="state_temperature",
            )

        snow_layers = state_in.snow_layers
        if snow_layers is not None:
            snow_layers = SnowLayerState(
                snl=snow_layers.snl,
                dz=snow_layers.dz,
                h2osoi_ice=snow_layers.h2osoi_ice,
                h2osoi_liq=snow_layers.h2osoi_liq,
                t_snow=_clip_active_snow_temp(snow_layers.t_snow, snow_layers.snl, "snow_layers"),
                snw_rds=getattr(snow_layers, "snw_rds", None),
            )

        return _CCS(
            h2osoi_liq=state_in.h2osoi_liq,
            h2osoi_ice=state_in.h2osoi_ice,
            h2ocan=state_in.h2ocan,
            h2osfc=state_in.h2osfc,
            frac_h2osfc=state_in.frac_h2osfc,
            frac_sno=state_in.frac_sno,
            zwt=state_in.zwt,
            t_soisno=_clip_temp(state_in.t_soisno, "t_soisno"),
            wa=state_in.wa,
            t_h2osfc=_clip_temp(state_in.t_h2osfc, "t_h2osfc"),
            t10=_clip_temp(state_in.t10, "t10"),
            t10_history=state_in.t10_history,
            h2osno=state_in.h2osno,
            snow_depth=state_in.snow_depth,
            snow_t=_clip_active_snow_temp(
                state_in.snow_t,
                None if state_in.snow_layers is None else state_in.snow_layers.snl,
                "snow_t",
            ),
            snow_layers=snow_layers,
            int_snow=state_in.int_snow,
        )

    state = _clip_state_temperatures(state)

    def _solar_components_from_fsds(fsds_t: "torch.Tensor", coszen_t: "torch.Tensor", doy_i: int) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        """Split total FSDS using native ELM cpl-bypass solar partitioning.

        ELM first applies the stream/coszen weighting to total shortwave, then
        splits the weighted flux 50/50 into VIS/NIR and uses separate cubic
        fits for the direct fraction in each band.
        """
        sw = torch.clamp(fsds_t, min=0.0)
        half_sw = 0.50 * sw
        ratio_nir = torch.clamp(
            0.29548
            + 0.00504 * half_sw
            - 1.4957e-05 * half_sw * half_sw
            + 1.4881e-08 * half_sw * half_sw * half_sw,
            min=0.01,
            max=0.99,
        )
        ratio_vis = torch.clamp(
            0.17639
            + 0.00380 * half_sw
            - 9.0039e-06 * half_sw * half_sw
            + 8.1351e-09 * half_sw * half_sw * half_sw,
            min=0.01,
            max=0.99,
        )
        forc_solad_vis = ratio_vis * half_sw
        forc_solad_nir = ratio_nir * half_sw
        forc_solai_vis = (1.0 - ratio_vis) * half_sw
        forc_solai_nir = (1.0 - ratio_nir) * half_sw
        return forc_solad_vis, forc_solad_nir, forc_solai_vis, forc_solai_nir

    def _prescribed_frac_veg_nosno(lai: torch.Tensor, sai: torch.Tensor) -> torch.Tensor:
        # ELM treats exposed stem area as canopy, so no-carbon vegetation is
        # active when either prescribed LAI or SAI is present.
        exposed_area = torch.clamp(lai, min=0.0) + torch.clamp(sai, min=0.0)
        return (exposed_area > 0.0).to(dtype=dtype)

    universal_step_rootr: "torch.Tensor | None" = None

    def _assemble_universal_features(
        common: dict,
        *,
        tbot: "torch.Tensor",
        qbot: "torch.Tensor",
        psrf: "torch.Tensor",
        forc_vpd: "torch.Tensor",
        flds: "torch.Tensor",
        forc_rain: "torch.Tensor",
        forc_snow: "torch.Tensor",
        fsds: "torch.Tensor",
        forc_solad_vis: "torch.Tensor",
        forc_solad_nir: "torch.Tensor",
        forc_solai_vis: "torch.Tensor",
        forc_solai_nir: "torch.Tensor",
        wind: "torch.Tensor",
        forc_rho: "torch.Tensor",
        forc_pco2: "torch.Tensor",
        dayl: "torch.Tensor",
        max_dayl: "torch.Tensor",
        coszen: "torch.Tensor",
        lai_active: "torch.Tensor",
        sai_active: "torch.Tensor",
        frac_veg_active: "torch.Tensor",
        htop_active: "torch.Tensor",
        update_state_features: bool = True,
        update_vegetation_features: bool = True,
    ) -> "torch.Tensor":
        nonlocal universal_step_rootr
        assert universal_feature_batch is not None
        assert isinstance(common["h2osoi_liq_t"], torch.Tensor)
        assert isinstance(common["h2osoi_ice_t"], torch.Tensor)
        assert isinstance(common["h2ocan_t"], torch.Tensor)
        assert isinstance(common["h2osfc_t"], torch.Tensor)
        assert isinstance(common["frac_h2osfc_t"], torch.Tensor)
        assert isinstance(common["t_soisno_t"], torch.Tensor)
        assert isinstance(common["t_h2osfc_t"], torch.Tensor)
        assert isinstance(common["t_grnd"], torch.Tensor)
        assert isinstance(common["t10"], torch.Tensor)
        assert isinstance(common["liqvol"], torch.Tensor)
        assert isinstance(common["eff_porosity"], torch.Tensor)
        assert isinstance(common["smp_l"], torch.Tensor)

        if snow_model_enabled:
            feature_forc_rain = forc_rain
            feature_forc_snow = forc_snow
            feature_frac_sno, feature_snow_depth = _surface_snow_feature_tensors(state)
        else:
            feature_forc_rain = forc_rain + forc_snow
            feature_forc_snow = _zeros_ncell1
            feature_frac_sno = _zeros_ncell1
            feature_snow_depth = _zeros_ncell1
        feature_h2osno, feature_snow_layer_count, feature_t_snow_top = (
            _snow_energy_feature_tensors(state)
        )

        qg_soil_diag = common["qg_soil_diag"]
        _profile_call("feature_dynamic_columns", lambda: _update_universal_dynamic_feature_columns((
            tbot,
            qbot,
            psrf,
            forc_vpd,
            qg_soil_diag,
            flds,
            feature_forc_rain,
            feature_forc_snow,
            fsds,
            forc_solad_vis,
            forc_solad_nir,
            forc_solai_vis,
            forc_solai_nir,
            wind,
            forc_rho,
            forc_pco2,
            frac_veg_active,
            lai_active,
            sai_active,
            htop_active,
            feature_frac_sno,
            feature_snow_depth,
            state.zwt,
            common["h2ocan_t"],
            common["t10"],
            dayl,
            max_dayl,
            _zeros_ncell1_geo,
            _zeros_ncell1_geo,
            common["frac_h2osfc_t"],
            coszen,
            common["t_grnd"],
            common["t_h2osfc_t"],
            _zeros_ncell1_geo,
            common["h2osfc_t"],
            feature_h2osno,
            feature_snow_layer_count,
            feature_t_snow_top,
        ), update_state_features=update_state_features, update_vegetation_features=update_vegetation_features))
        if _use_sparse_universal_feature_assembly():
            _sync_packed_universal_columns_to_dense((
                "forc_rain", "forc_snow", "elai_patch", "esai_patch",
                "frac_veg_nosno_patch", "h2ocan",
            ))
        def _copy_dynamic_profiles() -> None:
            _copy_universal_profile("smp_l", common["smp_l"])
            _copy_universal_profile("eff_porosity", common["eff_porosity"])
            _copy_universal_profile("h2osoi_liqvol", common["liqvol"])
            _copy_universal_profile("t_soisno", common["t_soisno_t"][..., :nlev_hydro_static])
        if update_state_features:
            _profile_call("feature_profile_columns", _copy_dynamic_profiles)

        btran = None
        if update_state_features:
            btran, universal_step_rootr = _profile_call(
                "feature_derive_btran",
                lambda: _universal_btran_and_root_fraction_from_common(common),
            )
        if btran is not None and universal_btran_idx is not None:
            if _use_sparse_universal_feature_assembly():
                assert universal_sparse_pft_idx_t is not None
                assert universal_sparse_cell_idx_t is not None
                assert universal_sparse_feature_batch is not None
                universal_sparse_feature_batch[..., universal_btran_idx] = btran
                universal_feature_batch[
                    universal_sparse_pft_idx_t,
                    universal_sparse_cell_idx_t,
                    :,
                    universal_btran_idx,
                ] = btran
            else:
                universal_feature_batch[..., universal_btran_idx] = btran
        if update_state_features and universal_soilbeta_idx is not None:
            soilbeta_col = _profile_call(
                "feature_derive_soilbeta",
                lambda: _soilbeta_from_common_features(common, frac_sno=feature_frac_sno),
            )
            if _use_sparse_universal_feature_assembly():
                assert universal_sparse_pft_idx_t is not None
                assert universal_sparse_cell_idx_t is not None
                assert universal_sparse_feature_batch is not None
                universal_sparse_feature_batch[..., universal_soilbeta_idx] = soilbeta_col.index_select(0, universal_sparse_cell_idx_t)
            else:
                universal_feature_batch[..., universal_soilbeta_idx] = soilbeta_col.unsqueeze(0).expand(len(universal_batch_indices), ncell, nens)
        if universal_sabg_idx is not None:
            def _universal_static_col(name: str):
                idx = universal_idx_map.get(name) if universal_idx_map is not None else None
                if idx is not None:
                    return universal_feature_batch[..., idx]
                values = [
                    getattr(pft_statics[p_idx], name, None)
                    for p_idx in universal_batch_indices
                ]
                if any(value is None for value in values):
                    return None
                return torch.as_tensor(values, device=device, dtype=dtype).reshape(len(universal_batch_indices), 1, 1)

            sabg_atmosphere = AtmosphericForcingInputs(
                forc_t=tbot,
                forc_q=qbot,
                forc_pbot=psrf,
                forc_lwrad=flds,
                forc_rain=feature_forc_rain,
                forc_snow=feature_forc_snow,
                forc_swrad=fsds,
                forc_solad_vis=forc_solad_vis,
                forc_solad_nir=forc_solad_nir,
                forc_solai_vis=forc_solai_vis,
                forc_solai_nir=forc_solai_nir,
                wind_speed=wind,
                forc_rho=forc_rho,
                forc_pco2=forc_pco2,
                dayl=dayl,
                max_dayl=max_dayl,
                coszen_col=coszen,
            )
            def _derive_sabg_snow_feature():
                if _use_sparse_universal_feature_assembly():
                    assert universal_sparse_pft_idx_t is not None
                    assert universal_sparse_cell_idx_t is not None
                    pidx = universal_sparse_pft_idx_t
                    cidx = universal_sparse_cell_idx_t

                    def _pair_cell(value):
                        value_t = value if isinstance(value, torch.Tensor) else torch.as_tensor(value, device=device, dtype=dtype)
                        value_t = value_t.to(device=device, dtype=dtype)
                        return value_t.index_select(0, cidx)

                    def _pair_static(name: str):
                        value = _universal_static_col(name)
                        return None if value is None else _gather_sparse_universal_value(value)

                    pair_atmosphere = AtmosphericForcingInputs(
                        forc_t=_pair_cell(tbot),
                        forc_q=_pair_cell(qbot),
                        forc_pbot=_pair_cell(psrf),
                        forc_lwrad=_pair_cell(flds),
                        forc_rain=_pair_cell(feature_forc_rain),
                        forc_snow=_pair_cell(feature_forc_snow),
                        forc_swrad=_pair_cell(fsds),
                        forc_solad_vis=_pair_cell(forc_solad_vis),
                        forc_solad_nir=_pair_cell(forc_solad_nir),
                        forc_solai_vis=_pair_cell(forc_solai_vis),
                        forc_solai_nir=_pair_cell(forc_solai_nir),
                        wind_speed=_pair_cell(wind),
                        forc_rho=_pair_cell(forc_rho),
                        forc_pco2=_pair_cell(forc_pco2),
                        dayl=_pair_cell(dayl),
                        max_dayl=_pair_cell(max_dayl),
                        coszen_col=_pair_cell(coszen),
                    )
                    pair_kwargs = dict(
                        device=device,
                        dtype=dtype,
                        elai=lai_active[pidx, cidx].unsqueeze(-1).expand(-1, nens),
                        esai=sai_active[pidx, cidx].unsqueeze(-1).expand(-1, nens),
                        atmosphere=pair_atmosphere,
                        soil_color=_gather_sparse_universal_value(shared_static.soil_color),
                        h2osoi_vol_top=_pair_cell(common["h2osoi_vol"][..., 0]),
                        rholvis=_pair_static("rholvis"),
                        rholnir=_pair_static("rholnir"),
                        taulvis=_pair_static("taulvis"),
                        taulnir=_pair_static("taulnir"),
                        rhosvis=_pair_static("rhosvis"),
                        rhosnir=_pair_static("rhosnir"),
                        tausvis=_pair_static("tausvis"),
                        tausnir=_pair_static("tausnir"),
                        xl=_pair_static("xl"),
                        require_native_like=True,
                        context="regional sparse universal sabg_snow feature",
                    )
                    pair_fsds = _pair_cell(fsds)
                    if not snow_model_enabled:
                        return _absorbed_ground_shortwave_torch(pair_fsds, **pair_kwargs)

                    def _subset(value):
                        if value is None:
                            return None
                        value_t = value if isinstance(value, torch.Tensor) else torch.as_tensor(value, device=device)
                        return value_t.to(device=device).index_select(0, cidx)

                    pair_layers = None
                    if state.snow_layers is not None:
                        pair_layers = SnowLayerState(
                            snl=_subset(state.snow_layers.snl),
                            dz=_subset(state.snow_layers.dz),
                            h2osoi_ice=_subset(state.snow_layers.h2osoi_ice),
                            h2osoi_liq=_subset(state.snow_layers.h2osoi_liq),
                            t_snow=_subset(state.snow_layers.t_snow),
                            snw_rds=_subset(state.snow_layers.snw_rds),
                        )
                    pair_state = CoupledColumnState(
                        h2osoi_liq=_subset(state.h2osoi_liq),
                        h2osoi_ice=_subset(state.h2osoi_ice),
                        h2ocan=_subset(state.h2ocan),
                        h2osfc=_subset(state.h2osfc),
                        frac_h2osfc=_subset(state.frac_h2osfc),
                        frac_sno=_subset(state.frac_sno),
                        zwt=_subset(state.zwt),
                        t_soisno=_subset(state.t_soisno),
                        wa=_subset(state.wa),
                        t_h2osfc=_subset(state.t_h2osfc),
                        t10=_subset(state.t10),
                        t10_history=_subset(state.t10_history),
                        h2osno=_subset(state.h2osno),
                        snow_depth=_subset(state.snow_depth),
                        snow_t=_subset(state.snow_t),
                        snow_layers=pair_layers,
                        int_snow=_subset(state.int_snow),
                    )
                    return sabg_snow_from_state_torch(
                        pair_state,
                        pair_fsds,
                        _pair_cell(feature_frac_sno),
                        **pair_kwargs,
                    )
                kwargs = dict(
                    device=device,
                    dtype=dtype,
                    elai=lai_active.unsqueeze(-1),
                    esai=sai_active.unsqueeze(-1),
                    atmosphere=sabg_atmosphere,
                    soil_color=shared_static.soil_color,
                    h2osoi_vol_top=common["h2osoi_vol"][..., 0],
                    rholvis=_universal_static_col("rholvis"),
                    rholnir=_universal_static_col("rholnir"),
                    taulvis=_universal_static_col("taulvis"),
                    taulnir=_universal_static_col("taulnir"),
                    rhosvis=_universal_static_col("rhosvis"),
                    rhosnir=_universal_static_col("rhosnir"),
                    tausvis=_universal_static_col("tausvis"),
                    tausnir=_universal_static_col("tausnir"),
                    xl=_universal_static_col("xl"),
                    require_native_like=True,
                    context="regional universal sabg_snow feature",
                )
                if not snow_model_enabled:
                    return _absorbed_ground_shortwave_torch(fsds, **kwargs)
                return sabg_snow_from_state_torch(
                    state,
                    fsds,
                    feature_frac_sno,
                    **kwargs,
                )
            sabg_snow = _profile_call(
                "feature_derive_sabg_snow",
                _derive_sabg_snow_feature,
            )
            if _use_sparse_universal_feature_assembly():
                assert universal_sparse_pft_idx_t is not None
                assert universal_sparse_cell_idx_t is not None
                assert universal_sparse_feature_batch is not None
                universal_sparse_feature_batch[..., universal_sabg_idx] = sabg_snow
                # SABG is also consumed by dense energy closure/blending.
                universal_feature_batch[
                    universal_sparse_pft_idx_t,
                    universal_sparse_cell_idx_t,
                    :,
                    universal_sabg_idx,
                ] = sabg_snow
            else:
                universal_feature_batch[..., universal_sabg_idx] = _broadcast_universal_feature_value(sabg_snow)
        return universal_feature_batch

    _cached_common_feature_state: "dict | None" = None
    _cached_vegetation_doy: int | None = None
    _cached_vegetation_values: "tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None" = None
    _cached_universal_vegetation_values: "tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None" = None
    _state_feature_overrides_active = any(
        value is not None
        for value in (
            soil_temperature_override_t,
            soil_liquid_override_t,
            soil_ice_override_t,
            ground_temperature_override_t,
            zwt_override_t,
        )
    )

    for step_i in range(n1h):
        state = _apply_soil_temperature_override(state, step_i)
        state = _apply_soil_moisture_override(state, step_i)
        state = _apply_zwt_override(state, step_i)
        row = met_1h[step_i]  # (ncell, 7) – device tensor, float dtype

        # Unpack met vars (ncell,) → expand to (ncell, 1) for broadcasting.
        def _col(name: str) -> "torch.Tensor":
            return row[:, _MET_VAR_IDX[name]].unsqueeze(1)  # (ncell, 1)

        tbot   = _col("TBOT")
        qbot   = _col("QBOT")
        psrf   = _col("PSRF")
        flds   = _col("FLDS")
        wind   = _col("WIND")
        prec   = _col("PRECTmms")  # mm s-1
        fsds   = torch.clamp(_col("FSDS"), min=0.0)

        # Rain/snow partitioning (ELM lnd_import_export formula).
        rain_frac = torch.clamp((tbot - 273.15) * 0.5, min=0.0, max=1.0)
        forc_rain = rain_frac * prec
        forc_snow = (1.0 - rain_frac) * prec
        step_has_snowfall = bool(_snowfall_possible_steps[step_i])
        _use_universal_fast_this_step = bool(universal_fast_feature_assembly)
        _assemble_universal_features_this_step = bool(universal_feature_assembly_enabled)
        if step_has_snowfall:
            _snow_pending_in_accum = True
        if _assemble_universal_features_this_step:
            _universal_fast_steps += 1

        # CO2 partial pressure (Pa).
        forc_pco2 = (co2_ppm * 1.0e-6) * psrf  # (ncell, 1)
        forc_vpd = vapor_pressure_deficit_from_forcing_torch(tbot, qbot, psrf)

        # Coszen/dayl/vegetation-structure lookup from precomputed tables.
        doy = day_of_year_start + step_i // 24
        coszen = coszen_dev[step_i].unsqueeze(1)  # (ncell, 1), signed like native ELM
        daylight = coszen > 0.0
        fsds = torch.where(daylight, torch.clamp(fsds, min=0.0), torch.zeros_like(fsds))
        dayl   = dayl_dev[step_i].unsqueeze(1)
        prev_dayl = prev_dayl_dev[step_i].unsqueeze(1)
        max_dayl = max_dayl_dev  # (ncell, 1) – constant across steps
        forc_solad_vis, forc_solad_nir, forc_solai_vis, forc_solai_nir = _solar_components_from_fsds(fsds, coszen, doy)
        tv = torch.clamp(tbot * (1.0 + 0.61 * torch.clamp(qbot, min=0.0)), min=150.0)
        forc_rho = torch.clamp(psrf / (287.042 * tv), min=0.5, max=1.7)
        no_carbon_prescribed_veg = carbon_model is None
        update_vegetation_features = bool(
            carbon_model is not None
            or _cached_vegetation_values is None
            or _cached_vegetation_doy != doy
        )
        if update_vegetation_features:
            _vegetation_feature_refreshes += 1
        if not update_vegetation_features:
            assert _cached_vegetation_values is not None
            (
                lai_active_today,
                sai_active_today,
                htop_active_today,
                frac_veg_active_today,
            ) = _cached_vegetation_values
        else:
            lai_today = lai_by_doy[doy]  # (17, ncell) GPU tensor
            sai_today = sai_by_doy[doy]
            htop_today = htop_by_doy[doy]
            hbot_today = hbot_by_doy[doy]
            prescribed_lai_active = torch.stack([lai_today[pid] for pid in pft_ids], dim=0)  # (npft, ncell)
            prescribed_sai_active = torch.stack([sai_today[pid] for pid in pft_ids], dim=0)
            htop_active = torch.stack([htop_today[pid] for pid in pft_ids], dim=0)
            hbot_active = torch.stack([hbot_today[pid] for pid in pft_ids], dim=0)

        if update_vegetation_features and carbon_model is not None:
            veg_structure = carbon_model.update_vegetation_structure_by_pft(
                fallback_lai_by_pft=prescribed_lai_active,
                fallback_sai_by_pft=prescribed_sai_active,
                htop_by_pft=htop_active,
                hbot_by_pft=hbot_active,
                snow_depth=state.snow_depth,
                dt=dt,
            )
            lai_active_today = veg_structure["elai"]
            sai_active_today = veg_structure["esai"]
            htop_active_today = veg_structure["htop"]
            frac_veg_active_today = veg_structure["frac_veg_nosno"]
        elif update_vegetation_features:
            lai_active_today = prescribed_lai_active
            sai_active_today = prescribed_sai_active
            htop_active_today = htop_active
            frac_veg_active_today = _prescribed_frac_veg_nosno(lai_active_today, sai_active_today)
        if update_vegetation_features and lai_scale != 1.0:
            lai_active_today = torch.clamp(lai_active_today * lai_scale, min=0.0)
        if update_vegetation_features and lai_add_applies:
            lai_active_today = torch.clamp(lai_active_today + lai_add * _lai_add_pft_mask, min=0.0)
        if update_vegetation_features and sai_scale != 1.0:
            sai_active_today = torch.clamp(sai_active_today * sai_scale, min=0.0)
        if update_vegetation_features and no_carbon_prescribed_veg and (lai_add_applies or lai_scale != 1.0 or sai_scale != 1.0):
            frac_veg_active_today = _prescribed_frac_veg_nosno(lai_active_today, sai_active_today)
        elif update_vegetation_features and (not no_carbon_prescribed_veg) and (lai_add_applies or lai_scale != 1.0):
            frac_veg_active_today = (lai_active_today > 0.01).to(dtype=dtype)
        if update_vegetation_features and carbon_model is None:
            _cached_vegetation_doy = doy
            _cached_vegetation_values = (
                lai_active_today,
                sai_active_today,
                htop_active_today,
                frac_veg_active_today,
            )

        update_state_features = bool(
            _cached_common_feature_state is None or _state_feature_overrides_active
        )
        if update_state_features:
            _state_feature_refreshes += 1
        if update_state_features:
            _cached_common_feature_state = _profile_call(
                "feature_common_state",
                lambda: precompute_surface_feature_state_torch(
                    state,
                    shared_static,
                    device=device,
                    dtype=dtype,
                ),
            )
        assert _cached_common_feature_state is not None
        common_feature_state = _cached_common_feature_state
        # Canopy storage advances hourly even while soil/hydrology state is
        # held between coupled physics calls.
        common_feature_state["h2ocan_t"] = (
            state.h2ocan.to(device=device, dtype=dtype)
            if isinstance(state.h2ocan, torch.Tensor)
            else torch.as_tensor(state.h2ocan, device=device, dtype=dtype)
        )
        # This diagnostic is column-only and was previously recomputed once
        # for the universal model and again for every individual model.
        common_feature_state["qg_soil_diag"] = _profile_call(
            "feature_derive_surface_humidity",
            lambda: diagnostic_soil_surface_humidity_torch(
                common_feature_state["t_soisno_t"][..., 0],
                common_feature_state["smp_l"][..., 0],
                psrf,
                qbot,
            ),
        )
        if ground_temperature_override_t is not None:
            common_feature_state["t_grnd"] = ground_temperature_override_t[
                min(step_i, ground_temperature_override_t.shape[0] - 1)
            ].unsqueeze(-1).expand(-1, nens)

        # Build AtmosphericForcingInputs with what we have. The four solar
        # bands are reconstructed from total FSDS so their sum is always
        # consistent with forc_swrad instead of leaving daylight means in
        # nighttime/low-light rows.
        atmosphere = AtmosphericForcingInputs(
            forc_t       = tbot,
            forc_q       = qbot,
            forc_pbot    = psrf,
            forc_lwrad   = flds,
            forc_rain    = forc_rain,
            forc_snow    = forc_snow,
            forc_swrad   = fsds,
            forc_solad_vis = forc_solad_vis,
            forc_solad_nir = forc_solad_nir,
            forc_solai_vis = forc_solai_vis,
            forc_solai_nir = forc_solai_nir,
            wind_speed   = wind,
            forc_rho     = forc_rho,
            forc_pco2    = forc_pco2,
            dayl         = dayl,
            max_dayl     = max_dayl,
            coszen_col   = coszen,
        )
        geometry = SurfaceGeometryInputs(
            forc_hgt_u_patch = _zeros_ncell1_geo,  # ablation
            z0mg_col         = _zeros_ncell1_geo,  # ablation
            ugust            = _zeros_ncell1_geo,  # ablation
        )
        if _assemble_universal_features_this_step:
            assert _universal_batch_idx_t is not None
            if update_vegetation_features or _cached_universal_vegetation_values is None:
                _cached_universal_vegetation_values = (
                    lai_active_today.index_select(0, _universal_batch_idx_t),
                    sai_active_today.index_select(0, _universal_batch_idx_t),
                    frac_veg_active_today.index_select(0, _universal_batch_idx_t),
                    htop_active_today.index_select(0, _universal_batch_idx_t),
                )
            (
                lai_universal_today,
                sai_universal_today,
                frac_veg_universal_today,
                htop_universal_today,
            ) = _cached_universal_vegetation_values
            _profile_call(
                "feature_assembly_universal",
                lambda: _assemble_universal_features(
                    common_feature_state,
                    tbot=tbot,
                    qbot=qbot,
                    psrf=psrf,
                    forc_vpd=forc_vpd,
                    flds=flds,
                    forc_rain=forc_rain,
                    forc_snow=forc_snow,
                    fsds=fsds,
                    forc_solad_vis=forc_solad_vis,
                    forc_solad_nir=forc_solad_nir,
                    forc_solai_vis=forc_solai_vis,
                    forc_solai_nir=forc_solai_nir,
                    wind=wind,
                    forc_rho=forc_rho,
                    forc_pco2=forc_pco2,
                    dayl=dayl,
                    max_dayl=max_dayl,
                    coszen=coszen,
                    lai_active=lai_universal_today,
                    sai_active=sai_universal_today,
                    frac_veg_active=frac_veg_universal_today,
                    htop_active=htop_universal_today,
                    update_state_features=update_state_features,
                    update_vegetation_features=update_vegetation_features,
                ),
            )

        # --- Per-PFT canopy + MLP forward pass ---
        all_targets: list[dict | None] = [None] * npft
        canopy_h2ocans: list = [None] * npft
        universal_canopy_precips: list[dict | None] = [None] * npft
        universal_intercept_states: list = [None] * npft
        universal_targets_stack: "dict | None" = None
        universal_trait_var_targets_stack: "dict | None" = None
        universal_canopy_h2ocan_stack: "torch.Tensor | None" = None
        universal_canopy_precip_blended: "dict | None" = None
        ref_feat = None
        compute_ood_this_step = ood_diagnostics_stride > 0 and (step_i % ood_diagnostics_stride == 0)
        if compute_ood_this_step and universal_sparse_feature_batch is not None:
            _sync_packed_universal_columns_to_dense(tuple(universal_input_names_t))
        ood_max_col: "torch.Tensor | None" = None
        ood_frac_col: "torch.Tensor | None" = None
        ood_feature_col: "dict[str, torch.Tensor]" = {}
        btran_input_col: "torch.Tensor | None" = None
        sabg_snow_input_col: "torch.Tensor | None" = None
        if _use_universal_fast_this_step:
            assert universal_feature_batch is not None
            universal_bundle = universal_ref_bundle
            assert universal_bundle is not None
            ref_feat = universal_feature_batch[0]

            if universal_btran_idx is not None:
                btran_input_col = torch.sum(
                    w_universal_dev * universal_feature_batch[..., universal_btran_idx],
                    dim=0,
                )
            if universal_sabg_idx is not None:
                sabg_snow_input_col = torch.sum(
                    w_universal_dev * universal_feature_batch[..., universal_sabg_idx],
                    dim=0,
                )
            if compute_ood_this_step:
                for u_idx, p_idx in enumerate(universal_batch_indices):
                    feat = universal_feature_batch[u_idx]
                    zmax_sample, zfrac_sample = _profile_call(
                        "ood_diagnostics",
                        lambda feat=feat, p_idx=p_idx: _mlp_input_ood_stats_torch(
                            feat,
                            *pft_ood_tensors[p_idx],
                            threshold=ood_threshold,
                        ),
                    )
                    zmax_cell = zmax_sample.mean(1)
                    zfrac_cell = zfrac_sample.mean(1)
                    ood_max_col = zmax_cell if ood_max_col is None else torch.maximum(ood_max_col, zmax_cell)
                    weighted_zfrac = w_lai_dev[p_idx] * zfrac_cell
                    ood_frac_col = weighted_zfrac if ood_frac_col is None else ood_frac_col + weighted_zfrac
                    for name, (feat_idx, mean_value, inv_std_value) in pft_ood_feature_stats[p_idx].items():
                        z_cell = torch.abs((feat[..., feat_idx] - mean_value) * inv_std_value).mean(1)
                        weighted_z = w_lai_dev[p_idx] * z_cell
                        if name in ood_feature_col:
                            ood_feature_col[name] = ood_feature_col[name] + weighted_z
                        else:
                            ood_feature_col[name] = weighted_z

            if step_callback is None:
                universal_canopy_precip_blended, intercept_state_stack = _profile_call(
                    "canopy_interception_universal",
                    lambda: _apply_canopy_interception_sparse_stack_blended(
                        universal_bundle,
                        universal_feature_batch,
                        state,
                    ),
                )
                canopy_precip_stack = {}
            else:
                canopy_precip_stack, intercept_state_stack = _profile_call(
                    "canopy_interception_universal",
                    lambda: _apply_canopy_interception_sparse_stack(universal_bundle, universal_feature_batch, state),
                )
            h2ocan_idx = h2ocan_feat_idx[0]
            if h2ocan_idx is not None:
                ic_h2o = intercept_state_stack.h2ocan
                universal_feature_batch[..., h2ocan_idx] = (
                    ic_h2o if isinstance(ic_h2o, torch.Tensor)
                    else torch.as_tensor(ic_h2o, device=device, dtype=dtype)
                )
                if universal_sparse_feature_batch is not None:
                    assert universal_sparse_pft_idx_t is not None
                    assert universal_sparse_cell_idx_t is not None
                    universal_sparse_feature_batch[..., h2ocan_idx] = universal_feature_batch[
                        universal_sparse_pft_idx_t,
                        universal_sparse_cell_idx_t,
                        ...,
                        h2ocan_idx,
                    ]
            _universal_model_features = _universal_model_feature_batch()
            assert _universal_model_features is not None
            targets_stack, trait_var_stack = _predict_universal_targets_stack(universal_bundle, _universal_model_features)
            targets_stack = _clip_photosynthesis_targets(targets_stack)
            targets_stack = _apply_energy_closed_soil_boundary(
                targets_stack,
                frac_veg_active_today.index_select(0, _universal_batch_idx_t),
                (
                    universal_feature_batch[..., universal_sabg_idx]
                    if universal_sabg_idx is not None
                    else None
                ),
            )
            if "hs_top_snow" not in targets_stack:
                targets_stack["hs_top_snow"] = targets_stack["hs_soil"]
                if trait_var_stack is not None and "hs_soil" in trait_var_stack:
                    trait_var_stack["hs_top_snow"] = trait_var_stack["hs_soil"]
            for u_idx, p_idx in enumerate(universal_batch_indices):
                targets_stack = _bare_ground_target_fallback(
                    targets_stack,
                    p_idx,
                    stacked=True,
                    stack_row=u_idx,
                )
            targets_stack.update(canopy_precip_stack)
            targets_stack, canopy_state_stack = _profile_call(
                "canopy_evaporation",
                lambda: apply_canopy_evaporation(
                    targets_stack, intercept_state_stack, dt, device=device, dtype=dtype
                ),
            )
            universal_targets_stack = targets_stack
            universal_trait_var_targets_stack = trait_var_stack
            universal_canopy_h2ocan_stack = (
                canopy_state_stack.h2ocan if isinstance(canopy_state_stack.h2ocan, torch.Tensor)
                else torch.as_tensor(canopy_state_stack.h2ocan, device=device, dtype=dtype)
            )

            for u_idx, p_idx in enumerate(universal_batch_indices):
                targets = {
                    name: value[u_idx]
                    for name, value in targets_stack.items()
                }
                all_targets[p_idx] = targets
                canopy_h2ocans[p_idx] = (
                    canopy_state_stack.h2ocan[u_idx] if isinstance(canopy_state_stack.h2ocan, torch.Tensor)
                    else torch.as_tensor(canopy_state_stack.h2ocan[u_idx], device=device, dtype=dtype)
                )
        else:
            hybrid_universal_canopy_precip_stack = None
            hybrid_universal_intercept_state_stack = None
            hybrid_universal_h2ocan_stack = None
            if (
                universal_mlp_batch
                and universal_feature_batch is not None
                and universal_batch_indices
                and step_callback is None
            ):
                universal_bundle = universal_ref_bundle
                assert universal_bundle is not None
                hybrid_universal_canopy_precip_stack, hybrid_universal_intercept_state_stack = _profile_call(
                    "canopy_interception_universal",
                    lambda: _apply_canopy_interception_sparse_stack(
                        universal_bundle,
                        universal_feature_batch,
                        state,
                    ),
                )
                hybrid_universal_h2ocan_stack = (
                    hybrid_universal_intercept_state_stack.h2ocan
                    if isinstance(hybrid_universal_intercept_state_stack.h2ocan, torch.Tensor)
                    else torch.as_tensor(
                        hybrid_universal_intercept_state_stack.h2ocan,
                        device=device,
                        dtype=dtype,
                    )
                )
                h2ocan_idx = h2ocan_feat_idx[universal_batch_indices[0]]
                if h2ocan_idx is not None:
                    universal_feature_batch[..., h2ocan_idx] = hybrid_universal_h2ocan_stack
                    if universal_sparse_feature_batch is not None:
                        assert universal_sparse_pft_idx_t is not None
                        assert universal_sparse_cell_idx_t is not None
                        universal_sparse_feature_batch[..., h2ocan_idx] = universal_feature_batch[
                            universal_sparse_pft_idx_t,
                            universal_sparse_cell_idx_t,
                            ...,
                            h2ocan_idx,
                        ]

            for p_idx, (pft_id, p_bundle, p_static) in enumerate(
                zip(pft_ids, pft_bundles, pft_statics)
            ):
                individual_cells = sparse_cells_by_pft.get(p_idx)
                individual_packed = p_idx in individual_sparse_feature_bufs
                # Per-PFT exposed LAI/SAI for all cells (already GPU tensors).
                lai_pft = lai_active_today[p_idx].unsqueeze(1)  # (ncell, 1)
                sai_pft = sai_active_today[p_idx].unsqueeze(1)
                frac_veg = frac_veg_active_today[p_idx].unsqueeze(1)

                vegetation = VegetationSurfaceInputs(
                    frac_veg_nosno_patch = frac_veg,
                    elai_patch           = lai_pft,
                    esai_patch           = sai_pft,
                    snow_depth = state.snow_depth,
                    t10        = state.t10,
                    htop_patch = htop_active_today[p_idx].unsqueeze(1),
                )

                if p_idx in universal_batch_index_set and universal_feature_batch is not None:
                    feat = universal_feature_batch[universal_batch_slot_for_pft[p_idx]]
                else:
                    def _assemble_features():
                        return _assemble_individual_features_fast(
                            p_idx,
                            p_bundle,
                            p_static,
                            (
                                individual_sparse_feature_bufs[p_idx]
                                if individual_packed
                                else feat_bufs[p_idx]
                            ),
                            common_feature_state,
                            tbot=tbot,
                            qbot=qbot,
                            psrf=psrf,
                            forc_vpd=forc_vpd,
                            flds=flds,
                            forc_rain=forc_rain,
                            forc_snow=forc_snow,
                            fsds=fsds,
                            forc_solad_vis=forc_solad_vis,
                            forc_solad_nir=forc_solad_nir,
                            forc_solai_vis=forc_solai_vis,
                            forc_solai_nir=forc_solai_nir,
                            wind=wind,
                            forc_rho=forc_rho,
                            forc_pco2=forc_pco2,
                            dayl=dayl,
                            max_dayl=max_dayl,
                            coszen=coszen,
                            lai_pft=lai_pft,
                            sai_pft=sai_pft,
                            frac_veg=frac_veg,
                            htop_pft=htop_active_today[p_idx].unsqueeze(1),
                            atmosphere=atmosphere,
                            update_state_features=update_state_features,
                            update_vegetation_features=update_vegetation_features,
                            cell_idx=individual_cells if individual_packed else None,
                        )

                    feat = _profile_call("feature_assembly_individual", _assemble_features)

                # Canopy interception.
                if (
                    p_idx in universal_batch_index_set
                    and hybrid_universal_canopy_precip_stack is not None
                    and hybrid_universal_h2ocan_stack is not None
                ):
                    canopy_precip = {
                        name: value[universal_batch_slot_for_pft[p_idx]]
                        for name, value in hybrid_universal_canopy_precip_stack.items()
                    }
                    intercept_state = _state_with_h2ocan(
                        state,
                        hybrid_universal_h2ocan_stack[universal_batch_slot_for_pft[p_idx]],
                    )
                else:
                    if individual_packed:
                        assert individual_cells is not None
                        canopy_precip, intercept_state, packed_h2ocan = _profile_call(
                            "canopy_interception_individual",
                            lambda: _apply_canopy_interception_packed_pft(
                                p_bundle,
                                feat,
                                state,
                                individual_cells,
                            ),
                        )
                    elif individual_cells is not None:
                        sparse_cells = individual_cells
                        candidate_cells = sparse_cells if sparse_cells.numel() < ncell else None
                        canopy_precip, intercept_state = _profile_call(
                            "canopy_interception_individual",
                            lambda candidate_cells=candidate_cells: _apply_canopy_interception_sparse_pft(
                                p_bundle,
                                feat,
                                state,
                                candidate_cells,
                            ),
                        )
                    else:
                        canopy_precip, intercept_state = _profile_call(
                            "canopy_interception_individual",
                            lambda: _apply_canopy_interception_replay(p_bundle, feat, state),
                        )
                h2ocan_idx = h2ocan_feat_idx[p_idx]
                if h2ocan_idx is not None:
                    if individual_packed:
                        feat[..., h2ocan_idx] = packed_h2ocan
                    else:
                        ic_h2o = intercept_state.h2ocan
                        feat[..., h2ocan_idx] = (
                            ic_h2o if isinstance(ic_h2o, torch.Tensor)
                            else torch.as_tensor(ic_h2o, device=device, dtype=dtype)
                        )

                _btran_idx = feat_idx_maps[p_idx].get("btran")
                if _btran_idx is not None:
                    if individual_packed:
                        assert individual_cells is not None
                        weighted_btran = torch.zeros((ncell, nens), device=device, dtype=dtype)
                        weighted_btran.index_copy_(
                            0,
                            individual_cells,
                            w_dev[p_idx].index_select(0, individual_cells) * feat[..., _btran_idx],
                        )
                    else:
                        weighted_btran = w_dev[p_idx] * feat[..., _btran_idx]
                    btran_input_col = weighted_btran if btran_input_col is None else btran_input_col + weighted_btran
                _sabg_snow_idx = feat_idx_maps[p_idx].get("sabg_snow")
                if _sabg_snow_idx is not None:
                    if individual_packed:
                        assert individual_cells is not None
                        weighted_sabg_snow = torch.zeros((ncell, nens), device=device, dtype=dtype)
                        weighted_sabg_snow.index_copy_(
                            0,
                            individual_cells,
                            w_dev[p_idx].index_select(0, individual_cells) * feat[..., _sabg_snow_idx],
                        )
                    else:
                        weighted_sabg_snow = w_dev[p_idx] * feat[..., _sabg_snow_idx]
                    sabg_snow_input_col = (
                        weighted_sabg_snow
                        if sabg_snow_input_col is None
                        else sabg_snow_input_col + weighted_sabg_snow
                    )

                if p_idx == 0:
                    ref_feat = feat_bufs[p_idx] if individual_packed else feat
                    if individual_packed and "forc_t" in feat_idx_maps[p_idx]:
                        # coupled_step uses this full-cell fallback to update
                        # the running 10-day temperature when t_ref2m is absent.
                        ref_feat[..., feat_idx_maps[p_idx]["forc_t"]] = tbot
                if compute_ood_this_step:
                    zmax_sample, zfrac_sample = _profile_call(
                        "ood_diagnostics",
                        lambda: _mlp_input_ood_stats_torch(
                            feat,
                            *pft_ood_tensors[p_idx],
                            threshold=ood_threshold,
                        ),
                    )
                    zmax_cell = zmax_sample.mean(1)
                    zfrac_cell = zfrac_sample.mean(1)
                    if individual_packed:
                        assert individual_cells is not None
                        zmax_full = torch.zeros(ncell, device=device, dtype=dtype)
                        zfrac_full = torch.zeros(ncell, device=device, dtype=dtype)
                        zmax_full.index_copy_(0, individual_cells, zmax_cell)
                        zfrac_full.index_copy_(0, individual_cells, zfrac_cell)
                        zmax_cell = zmax_full
                        zfrac_cell = zfrac_full
                    ood_max_col = zmax_cell if ood_max_col is None else torch.maximum(ood_max_col, zmax_cell)
                    weighted_zfrac = w_lai_dev[p_idx] * zfrac_cell
                    ood_frac_col = weighted_zfrac if ood_frac_col is None else ood_frac_col + weighted_zfrac
                    for name, (feat_idx, mean_value, inv_std_value) in pft_ood_feature_stats[p_idx].items():
                        z_cell = torch.abs((feat[..., feat_idx] - mean_value) * inv_std_value).mean(1)
                        if individual_packed:
                            assert individual_cells is not None
                            z_cell_full = torch.zeros(ncell, device=device, dtype=dtype)
                            z_cell_full.index_copy_(0, individual_cells, z_cell)
                            z_cell = z_cell_full
                        weighted_z = w_lai_dev[p_idx] * z_cell
                        if name in ood_feature_col:
                            ood_feature_col[name] = ood_feature_col[name] + weighted_z
                        else:
                            ood_feature_col[name] = weighted_z

                if p_idx in universal_batch_index_set:
                    universal_canopy_precips[p_idx] = canopy_precip
                    universal_intercept_states[p_idx] = intercept_state
                    continue

                # MLP inference.
                targets = _profile_call(
                    "mlp_forward",
                    lambda: _predict_surface_targets_mlp(
                        p_bundle,
                        feat,
                    ),
                )
                targets = _clip_photosynthesis_targets(targets)
                if individual_packed:
                    assert individual_cells is not None
                    dense_targets = {}
                    for target_name, target_value in targets.items():
                        target_value_t = (
                            target_value
                            if isinstance(target_value, torch.Tensor)
                            else torch.as_tensor(target_value, device=device, dtype=dtype)
                        ).to(device=device, dtype=dtype)
                        full_value = torch.zeros(
                            (ncell,) + tuple(target_value_t.shape[1:]),
                            device=device,
                            dtype=dtype,
                        )
                        full_value.index_copy_(0, individual_cells, target_value_t)
                        dense_targets[target_name] = full_value
                    targets = dense_targets
                    sabg_for_closure = None
                    if "sabg_snow" in feat_idx_maps[p_idx]:
                        sabg_for_closure = torch.zeros((ncell, nens), device=device, dtype=dtype)
                        sabg_for_closure.index_copy_(
                            0,
                            individual_cells,
                            feat[..., feat_idx_maps[p_idx]["sabg_snow"]],
                        )
                else:
                    sabg_for_closure = (
                        feat[..., feat_idx_maps[p_idx]["sabg_snow"]]
                        if "sabg_snow" in feat_idx_maps[p_idx]
                        else None
                    )
                targets = _apply_energy_closed_soil_boundary(
                    targets,
                    frac_veg,
                    sabg_for_closure,
                )
                if "hs_top_snow" not in targets:
                    targets["hs_top_snow"] = targets["hs_soil"]
                targets = _bare_ground_target_fallback(targets, p_idx, stacked=False)
                targets.update(canopy_precip)

                # Canopy evaporation.
                targets, canopy_state_p = _profile_call(
                    "canopy_evaporation",
                    lambda: apply_canopy_evaporation(
                        targets, intercept_state, dt, device=device, dtype=dtype
                    ),
                )
                all_targets[p_idx] = targets
                canopy_h2ocans[p_idx] = (
                    canopy_state_p.h2ocan if isinstance(canopy_state_p.h2ocan, torch.Tensor)
                    else torch.as_tensor(canopy_state_p.h2ocan, device=device, dtype=dtype)
                )

            if universal_mlp_batch:
                assert universal_feature_batch is not None
                universal_bundle = universal_ref_bundle
                assert universal_bundle is not None
                _universal_model_features = _universal_model_feature_batch()
                assert _universal_model_features is not None
                targets_stack, trait_var_stack = _predict_universal_targets_stack(universal_bundle, _universal_model_features)
                targets_stack = _clip_photosynthesis_targets(targets_stack)
                targets_stack = _apply_energy_closed_soil_boundary(
                    targets_stack,
                    frac_veg_active_today.index_select(0, _universal_batch_idx_t),
                    (
                        universal_feature_batch[..., universal_sabg_idx]
                        if universal_sabg_idx is not None
                        else None
                    ),
                )
                if "hs_top_snow" not in targets_stack:
                    targets_stack["hs_top_snow"] = targets_stack["hs_soil"]
                    if trait_var_stack is not None and "hs_soil" in trait_var_stack:
                        trait_var_stack["hs_top_snow"] = trait_var_stack["hs_soil"]
                for p_idx in universal_batch_indices:
                    targets_stack = _bare_ground_target_fallback(
                        targets_stack,
                        p_idx,
                        stacked=True,
                        stack_row=universal_batch_slot_for_pft[p_idx],
                    )

                for p_idx in universal_batch_indices:
                    u_idx = universal_batch_slot_for_pft[p_idx]
                    targets = {
                        name: value[u_idx]
                        for name, value in targets_stack.items()
                    }
                    canopy_precip = universal_canopy_precips[p_idx]
                    intercept_state = universal_intercept_states[p_idx]
                    if canopy_precip is None or intercept_state is None:
                        raise RuntimeError(f"Missing universal canopy state for PFT slot {p_idx}")
                    targets.update(canopy_precip)
                    targets, canopy_state_p = _profile_call(
                        "canopy_evaporation",
                        lambda targets=targets, state_p=intercept_state: apply_canopy_evaporation(
                            targets, state_p, dt, device=device, dtype=dtype
                        ),
                    )
                    all_targets[p_idx] = targets
                    canopy_h2ocans[p_idx] = (
                        canopy_state_p.h2ocan if isinstance(canopy_state_p.h2ocan, torch.Tensor)
                        else torch.as_tensor(canopy_state_p.h2ocan, device=device, dtype=dtype)
                    )
                universal_trait_var_targets_stack = trait_var_stack

        # --- Area-weighted blend of surface targets ---
        _t_blend0 = _profile_start()
        if compute_ood_this_step and ood_max_col is not None and ood_frac_col is not None:
            if _acc_ood_max_z is None:
                _acc_ood_max_z = ood_max_col.clone()
                _acc_ood_frac = ood_frac_col.clone()
            else:
                _acc_ood_max_z.add_(ood_max_col)
                _acc_ood_frac.add_(ood_frac_col)
            for name, value in ood_feature_col.items():
                if name in _acc_ood_feature_z:
                    _acc_ood_feature_z[name].add_(value)
                else:
                    _acc_ood_feature_z[name] = value.clone()
            _acc_ood_n += 1

        def _blend_pft_stack(value: "torch.Tensor") -> "torch.Tensor":
            value_t = value if isinstance(value, torch.Tensor) else torch.as_tensor(value, device=device, dtype=dtype)
            w = w_universal_dev if value_t.shape[0] == len(universal_batch_indices) else w_dev
            while w.ndim < value_t.ndim:
                w = w.unsqueeze(-1)
            return torch.sum(w * value_t.to(device=device, dtype=dtype), dim=0)

        def _blend_universal_batch_stack(value: "torch.Tensor") -> "torch.Tensor":
            value_t = value if isinstance(value, torch.Tensor) else torch.as_tensor(value, device=device, dtype=dtype)
            value_t = value_t.to(device=device, dtype=dtype)
            if not universal_batch_indices:
                raise RuntimeError("Universal batch blend requested without universal PFT indices")
            w = w_universal_dev
            while w.ndim < value_t.ndim:
                w = w.unsqueeze(-1)
            return torch.sum(w * value_t, dim=0)

        def _blend_independent_trait_std_stack(value: "torch.Tensor") -> "torch.Tensor":
            value_t = value if isinstance(value, torch.Tensor) else torch.as_tensor(value, device=device, dtype=dtype)
            value_t = torch.clamp(value_t.to(device=device, dtype=dtype), min=0.0)
            if not universal_batch_indices:
                raise RuntimeError("Universal trait blend requested without universal PFT indices")
            w = w_universal_dev
            while w.ndim < value_t.ndim:
                w = w.unsqueeze(-1)
            return torch.sqrt(torch.clamp(torch.sum((w * w) * value_t, dim=0), min=0.0))

        def _add_weighted_pft_value(acc, p_idx: int, value):
            value_t = value if isinstance(value, torch.Tensor) else torch.as_tensor(value, device=device, dtype=dtype)
            value_t = value_t.to(device=device, dtype=dtype)
            w = w_dev[p_idx]
            while w.ndim < value_t.ndim:
                w = w.unsqueeze(-1)
            term = w * value_t
            return term if acc is None else acc + term

        if _use_universal_fast_this_step and universal_targets_stack is not None:
            blended: dict = {
                key: _blend_pft_stack(value)
                for key, value in universal_targets_stack.items()
                if key in _blend_target_names
            }
            if universal_canopy_precip_blended is not None:
                blended.update(universal_canopy_precip_blended)
            blended["fpsn_per_pft"] = [
                universal_targets_stack["fpsn_patch"][universal_batch_slot_for_pft[p_idx]]
                if p_idx in universal_batch_slot_for_pft else None
                for p_idx in range(npft)
            ] if "fpsn_patch" in universal_targets_stack else [None] * npft
            all_targets_filled = [t for t in all_targets if t is not None]
            canopy_h2ocans_filled = [h for h in canopy_h2ocans if h is not None]
        elif universal_targets_stack is not None and _universal_batch_idx_t is not None:
            if any(t is None for t in all_targets):
                missing = [str(int(pft_ids[idx])) for idx, t in enumerate(all_targets) if t is None]
                raise RuntimeError(f"Missing surface targets for PFT(s): {', '.join(missing)}")
            if any(h is None for h in canopy_h2ocans):
                missing = [str(int(pft_ids[idx])) for idx, h in enumerate(canopy_h2ocans) if h is None]
                raise RuntimeError(f"Missing canopy states for PFT(s): {', '.join(missing)}")
            all_targets_filled = [t for t in all_targets if t is not None]
            canopy_h2ocans_filled = [h for h in canopy_h2ocans if h is not None]
            blended = {}
            for key in _blend_target_names:
                acc = (
                    _blend_universal_batch_stack(universal_targets_stack[key])
                    if key in universal_targets_stack
                    else None
                )
                for p_idx, t in enumerate(all_targets_filled):
                    if p_idx in universal_batch_index_set:
                        continue
                    v = t.get(key)
                    if v is not None:
                        acc = _add_weighted_pft_value(acc, p_idx, v)
                if acc is not None:
                    blended[key] = acc
            blended["fpsn_per_pft"] = [t.get("fpsn_patch") for t in all_targets_filled]
        else:
            if any(t is None for t in all_targets):
                missing = [str(int(pft_ids[idx])) for idx, t in enumerate(all_targets) if t is None]
                raise RuntimeError(f"Missing surface targets for PFT(s): {', '.join(missing)}")
            if any(h is None for h in canopy_h2ocans):
                missing = [str(int(pft_ids[idx])) for idx, h in enumerate(canopy_h2ocans) if h is None]
                raise RuntimeError(f"Missing canopy states for PFT(s): {', '.join(missing)}")
            all_targets_filled = [t for t in all_targets if t is not None]
            canopy_h2ocans_filled = [h for h in canopy_h2ocans if h is not None]
            blended = {}
            for key in _blend_target_names:
                acc = None
                for p_idx, t in enumerate(all_targets_filled):
                    v = t.get(key)
                    if v is None:
                        continue
                    acc = _add_weighted_pft_value(acc, p_idx, v)
                if acc is not None:
                    blended[key] = acc

            # Per-PFT (unweighted) fpsn list for carbon step_callback.
            # Index matches pft_ids order; each entry is (ncell, nens) or None.
            blended["fpsn_per_pft"] = [t.get("fpsn_patch") for t in all_targets_filled]
        lai_area_weighted = (w_lai_dev * lai_active_today).sum(0).unsqueeze(1).expand(-1, nens)
        sai_area_weighted = (w_lai_dev * sai_active_today).sum(0).unsqueeze(1).expand(-1, nens)
        blended["diag_lai_area_weighted"] = lai_area_weighted
        blended["diag_sai_area_weighted"] = sai_area_weighted
        blended["diag_forc_pbot"] = psrf.expand(-1, nens)
        blended["diag_frac_h2osfc_input"] = state.frac_h2osfc
        blended["diag_frac_sno_input"], _ = _surface_snow_feature_tensors(state)
        blended["diag_h2osoi_liq_input"] = state.h2osoi_liq
        blended["diag_h2osoi_ice_input"] = state.h2osoi_ice
        if btran_input_col is not None:
            blended["diag_btran_input"] = btran_input_col
        if sabg_snow_input_col is not None:
            blended["diag_sabg_snow_input"] = sabg_snow_input_col
        blended = _clip_thermal_boundary_targets(blended)
        if ground_heat_flux_override_t is not None:
            _gh_override = ground_heat_flux_override_t[min(int(step_i), ground_heat_flux_override_t.shape[0] - 1)]
            _gh_override = _gh_override.unsqueeze(1).expand(-1, nens)
            for _gh_name in ground_heat_flux_override_targets:
                if _gh_name in blended:
                    blended[_gh_name] = _gh_override
            blended["diag_ground_heat_flux_override"] = _gh_override

        _universal_qtran_accumulated_this_step = False
        if (
            universal_targets_stack is not None
            and universal_step_rootr is not None
            and "qflx_tran_veg" in universal_targets_stack
            and _universal_batch_idx_t is not None
            and _universal_batch_idx_t.numel() > 0
        ):
            q_tran_u = universal_targets_stack["qflx_tran_veg"]
            _physics_qtran_universal_accum = (
                q_tran_u.clone()
                if _physics_qtran_universal_accum is None
                else _physics_qtran_universal_accum.add(q_tran_u)
            )
            _physics_universal_rootr = universal_step_rootr
            _universal_qtran_accumulated_this_step = True
        for p_idx, t_p in enumerate(all_targets_filled):
            universal_already_accumulated = _universal_qtran_accumulated_this_step and p_idx in universal_batch_index_set
            if universal_already_accumulated or int(pft_ids[p_idx]) == 0:
                continue
            q_tran = t_p.get("qflx_tran_veg")
            if q_tran is None:
                continue
            q_tran_t = q_tran if isinstance(q_tran, torch.Tensor) else torch.as_tensor(q_tran, device=device, dtype=dtype)
            q_tran_t = q_tran_t.to(device=device, dtype=dtype)
            if p_idx in _physics_qtran_individual_accum:
                _physics_qtran_individual_accum[p_idx].add_(q_tran_t)
            else:
                _physics_qtran_individual_accum[p_idx] = q_tran_t.clone()

        # Blended canopy state.
        if _use_universal_fast_this_step and universal_canopy_h2ocan_stack is not None:
            blended_h2ocan = torch.sum(w_universal_dev * universal_canopy_h2ocan_stack, dim=0)
        else:
            blended_h2ocan: "torch.Tensor" = sum(
                w_dev[i] * h2ocan for i, h2ocan in enumerate(canopy_h2ocans_filled)
            )
        blended_canopy_state = _CCS(
            h2osoi_liq=state.h2osoi_liq, h2osoi_ice=state.h2osoi_ice,
            h2ocan=blended_h2ocan, h2osfc=state.h2osfc,
            frac_h2osfc=state.frac_h2osfc, frac_sno=state.frac_sno,
            zwt=state.zwt, t_soisno=state.t_soisno, wa=state.wa,
            t_h2osfc=state.t_h2osfc, t10=state.t10, t10_history=state.t10_history,
            h2osno=state.h2osno, snow_depth=state.snow_depth, snow_t=state.snow_t,
            snow_layers=state.snow_layers, int_snow=state.int_snow,
        )
        _profile_stop("target_blend", _t_blend0)

        _physics_accum = _accumulate_physics_targets(_physics_accum, blended)
        _physics_count += 1
        _physics_ref_feat = ref_feat
        _physics_canopy_state = blended_canopy_state
        run_physics_this_step = (
            _physics_count >= physics_cadence_steps
            or (step_i + 1) == n1h
        )

        if run_physics_this_step:
            physics_targets = _average_physics_targets(_physics_accum, _physics_count)
            physics_dt = dt * _physics_count
            rootsoi_acc: "torch.Tensor | None" = None
            if (
                _physics_qtran_universal_accum is not None
                and _physics_universal_rootr is not None
                and _universal_batch_idx_t is not None
            ):
                q_tran_u = _physics_qtran_universal_accum / float(_physics_count)
                rootr_u = _physics_universal_rootr
                sparse_rootr = bool(
                    _use_sparse_universal_feature_assembly()
                    and rootr_u.ndim == 3
                    and universal_sparse_pft_idx_t is not None
                    and universal_sparse_cell_idx_t is not None
                )
                if universal_rootfr_btran is not None:
                    rootr_fallback = (
                        universal_rootfr_btran.index_select(0, universal_sparse_pft_idx_t).squeeze(1).expand_as(rootr_u)
                        if sparse_rootr
                        else universal_rootfr_btran.expand_as(rootr_u)
                    )
                    rootr_sum = torch.sum(rootr_u, dim=-1, keepdim=True)
                    q_tran_for_root = (
                        q_tran_u[universal_sparse_pft_idx_t, universal_sparse_cell_idx_t, ...]
                        if sparse_rootr
                        else q_tran_u
                    )
                    rootr_u = torch.where(
                        (rootr_sum > 1.0e-12) | (q_tran_for_root.unsqueeze(-1) <= 1.0e-15),
                        rootr_u,
                        rootr_fallback,
                    )
                if sparse_rootr:
                    assert universal_sparse_pft_idx_t is not None
                    assert universal_sparse_cell_idx_t is not None
                    q_tran_pair = q_tran_u[
                        universal_sparse_pft_idx_t,
                        universal_sparse_cell_idx_t,
                        ...,
                    ]
                    weight_pair = w_universal_dev[
                        universal_sparse_pft_idx_t,
                        universal_sparse_cell_idx_t,
                        ...,
                    ]
                    q_root_pair = weight_pair.unsqueeze(-1) * q_tran_pair.unsqueeze(-1) * rootr_u
                    rootsoi_acc = torch.zeros(
                        (ncell, nens, rootr_u.shape[-1]), device=device, dtype=dtype
                    )
                    rootsoi_acc.index_add_(0, universal_sparse_cell_idx_t, q_root_pair)
                else:
                    w_u = w_universal_dev
                    q_root_stack = q_tran_u.unsqueeze(-1) * rootr_u
                    rootsoi_acc = torch.sum(w_u.unsqueeze(-1) * q_root_stack, dim=0)
            for p_idx, qtran_acc in _physics_qtran_individual_accum.items():
                q_tran = qtran_acc / float(_physics_count)
                q_root = distribute_root_uptake(
                    q_tran,
                    state,
                    pft_statics[p_idx],
                    "torch",
                    device=device,
                    dtype=dtype,
                )
                q_root_sum = torch.sum(q_root, dim=-1, keepdim=True)
                rootfr_fallback = (
                    pft_statics[p_idx].rootfr_norm_torch
                    if pft_statics[p_idx].rootfr_norm_torch is not None
                       and pft_statics[p_idx].rootfr_norm_torch.device == device
                       and pft_statics[p_idx].rootfr_norm_torch.dtype == dtype
                    else torch.as_tensor(
                        np.maximum(np.asarray(pft_statics[p_idx].rootfr, dtype=np.float64), 0.0),
                        device=device,
                        dtype=dtype,
                    )
                )
                if (
                    pft_statics[p_idx].rootfr_norm_torch is None
                    or pft_statics[p_idx].rootfr_norm_torch.device != device
                    or pft_statics[p_idx].rootfr_norm_torch.dtype != dtype
                ):
                    rootfr_fallback = rootfr_fallback / torch.clamp(rootfr_fallback.sum(), min=1.0e-12)
                rootfr_fallback = rootfr_fallback[:nlev_hydro_static].reshape((1,) * q_tran.ndim + (-1,))
                q_root_fallback = q_tran.unsqueeze(-1) * rootfr_fallback
                q_root = torch.where(
                    (q_root_sum > 1.0e-15) | (q_tran.unsqueeze(-1) <= 1.0e-15),
                    q_root,
                    q_root_fallback,
                )
                term = w_dev[p_idx].unsqueeze(-1) * q_root
                rootsoi_acc = term if rootsoi_acc is None else rootsoi_acc + term
            if rootsoi_acc is None:
                rootsoi_acc = torch.zeros(
                    (ncell, nens, nlev_hydro_static),
                    device=device,
                    dtype=dtype,
                )
            physics_targets["qflx_rootsoi"] = rootsoi_acc
            _last_qflx_rootsoi = rootsoi_acc
            snow_enabled_for_physics = bool(
                snow_model_enabled and (_state_has_snow or _snow_pending_in_accum)
            )
            _physics_ncalls += 1
            if snow_enabled_for_physics:
                _physics_snow_enabled_calls += 1
            if _state_has_active_snow:
                _physics_active_snow_calls += 1
                _max_active_snow_cells = max(_max_active_snow_cells, _active_snow_cells)
            if _snow_pending_in_accum:
                _physics_pending_snow_calls += 1
            physics_detail_profile: dict[str, float] | None = (
                {} if profile_sections is not None else None
            )
            _t_phys0 = _profile_start()
            last_result = coupled_step(
                pft_bundles[0],
                _physics_ref_feat,
                state,
                shared_static,
                physics_dt,
                hydrology_backend=hydrology_backend,
                soil_temperature_backend=soil_temperature_backend,
                device=device,
                dtype=dtype,
                hydrology_workspace=hydrology_workspace,
                soil_temperature_workspace=soil_temperature_workspace,
                snow_model_enabled=snow_enabled_for_physics,
                snow_scalar_only=snow_scalar_only,
                blend_partial_snow_soil_temperature=blend_partial_snow_soil_temperature,
                copy_surface_features=False,
                snow_state_already_disabled=not snow_enabled_for_physics,
                profile_sections=physics_detail_profile,
                precomputed_surface_targets=physics_targets,
                precomputed_canopy_state=_physics_canopy_state,
            )
            _profile_stop("physics_step", _t_phys0)
            if profile_sections is not None and physics_detail_profile:
                for detail_name, detail_s in physics_detail_profile.items():
                    key = f"physics:{detail_name}"
                    profile_sections[key] = profile_sections.get(key, 0.0) + detail_s
            _update_outlier("t_soil_raw", _cell_mean_tensor(last_result.state.t_soisno[..., 0]))
            state = _clip_state_temperatures(last_result.state)
            state = _apply_soil_temperature_override(state, step_i)
            state = _apply_soil_moisture_override(state, step_i)
            state = _apply_zwt_override(state, step_i)
            last_result.state = state
            _cached_common_feature_state = None
            _active_snow_cells = _active_snow_cell_count(state)
            _snow_present_cells = _snow_present_cell_count(state)
            _max_active_snow_cells = max(_max_active_snow_cells, _active_snow_cells)
            _state_has_active_snow = _active_snow_cells > 0
            _state_has_snow = _snow_present_cells > 0
            _snow_pending_in_accum = False
            if carbon_model is not None:
                _t_carbon0 = _profile_start()
                _carbon_advanced = carbon_model.accumulate_step(
                    fpsn_per_pft=physics_targets.get("fpsn_per_pft"),
                    state=state,
                    hydrology_result=last_result.hydrology_result,
                    tbot=tbot,
                    dayl=dayl,
                    prev_dayl=prev_dayl,
                    doy=doy,
                    dt=physics_dt,
                )
                if _carbon_advanced:
                    for _key, _value in carbon_model.pool_diagnostics().items():
                        _daily_carbon_pool_rows.setdefault(_key, []).append(
                            _value.detach().cpu().numpy().astype(np.float32, copy=False)
                        )
                _profile_stop("carbon_step", _t_carbon0)
            _physics_accum = None
            _physics_count = 0
            _physics_qtran_universal_accum = None
            _physics_qtran_individual_accum.clear()
            _physics_universal_rootr = None
            _physics_ref_feat = None
            _physics_canopy_state = None
        else:
            # Canopy water is still advanced hourly; soil T/hydrology stay at
            # the last coupled value until the next aggregated physics step.
            state = _clip_state_temperatures(blended_canopy_state)
            state = _apply_soil_temperature_override(state, step_i)
            state = _apply_soil_moisture_override(state, step_i)
            state = _apply_zwt_override(state, step_i)
            if last_result is not None:
                last_result.state = state

        if step_callback is not None:
            callback_result = last_result if last_result is not None else SimpleNamespace(state=state)
            step_callback(
                step_i,
                callback_result,
                blended,
                all_targets,
                universal_targets_stack,
                tbot,
                fsds,
                dayl,
                ood_max_col,
                ood_frac_col,
                ood_feature_col,
                tuple(feat_bufs),
            )

        with torch.no_grad():
            fpsn_cell = blended.get("fpsn_patch")
            if fpsn_cell is not None:
                _update_outlier("fpsn", _cell_mean_tensor(fpsn_cell))
            pft_fpsn_values = []
            for pft_targets in all_targets:
                if pft_targets is None:
                    pft_fpsn_values.append(torch.full((ncell,), -torch.inf, device=device, dtype=dtype))
                    continue
                pft_fpsn = pft_targets.get("fpsn_patch")
                if pft_fpsn is None:
                    pft_fpsn_values.append(torch.full((ncell,), -torch.inf, device=device, dtype=dtype))
                else:
                    pft_fpsn_values.append(_cell_mean_tensor(pft_fpsn))
            if pft_fpsn_values:
                pft_stack = torch.stack(
                    [
                        torch.nan_to_num(v, nan=-torch.inf, posinf=torch.inf, neginf=-torch.inf)
                        for v in pft_fpsn_values
                    ],
                    dim=0,
                )
                pft_best, pft_best_idx = torch.max(pft_stack, dim=0)
                pft_mask = pft_best > _outlier_max["pft_fpsn"]
                _outlier_max["pft_fpsn"] = torch.where(pft_mask, pft_best, _outlier_max["pft_fpsn"])
                _outlier_step["pft_fpsn"] = torch.where(
                    pft_mask,
                    torch.full_like(_outlier_step["pft_fpsn"], step_i),
                    _outlier_step["pft_fpsn"],
                )
                _outlier_pft_id = torch.where(
                    pft_mask,
                    _pft_id_t[pft_best_idx],
                    _outlier_pft_id,
                )
            _update_outlier("lai", lai_area_weighted.mean(1))
            _update_outlier("t_soil", _cell_mean_tensor(state.t_soisno[..., 0]))
            _update_outlier("liq_vol", _cell_mean_tensor(state.h2osoi_liq[..., 0]) / (_dz0_diag * 1000.0))
            _update_outlier("h2osfc", _cell_mean_tensor(state.h2osfc))
            if getattr(state, "h2osno", None) is not None:
                _update_outlier("h2osno", _cell_mean_tensor(state.h2osno))
            if getattr(state, "snow_depth", None) is not None:
                _update_outlier("snow_depth", _cell_mean_tensor(state.snow_depth))
                if getattr(state, "frac_sno", None) is not None:
                    _frac_sno_out = (
                        state.frac_sno.to(device=device, dtype=dtype)
                        if isinstance(state.frac_sno, torch.Tensor)
                        else torch.as_tensor(state.frac_sno, device=device, dtype=dtype)
                    )
                    _snow_depth_out = (
                        state.snow_depth.to(device=device, dtype=dtype)
                        if isinstance(state.snow_depth, torch.Tensor)
                        else torch.as_tensor(state.snow_depth, device=device, dtype=dtype)
                    )
                    _update_outlier(
                        "snow_depth_eff",
                        _cell_mean_tensor(_frac_sno_out * _snow_depth_out),
                    )
            if getattr(state, "frac_sno", None) is not None:
                _update_outlier("frac_sno", _cell_mean_tensor(state.frac_sno))

        if record_diag_maps or record_daily_diag_maps or record_hourly_diag_maps:
            # Accumulate diagnostic maps (squeezed to (ncell,) on device).
            _t = tbot.expand(-1, nens).mean(1) if nens > 1 else tbot.squeeze(1)  # (ncell,)
            _q = qbot.expand(-1, nens).mean(1) if nens > 1 else qbot.squeeze(1)
            _pbot = psrf.expand(-1, nens).mean(1) if nens > 1 else psrf.squeeze(1)
            _lwrad = flds.expand(-1, nens).mean(1) if nens > 1 else flds.squeeze(1)
            _wind = wind.expand(-1, nens).mean(1) if nens > 1 else wind.squeeze(1)
            _esat = torch.clamp(saturation_vapor_pressure_pa_torch(tbot), min=1.0e-6)
            _rh2m = torch.clamp(100.0 * (_esat - forc_vpd) / _esat, min=0.0, max=100.0)
            _rh2m_diag = _rh2m.expand(-1, nens).mean(1) if nens > 1 else _rh2m.squeeze(1)
            _f_raw = blended.get("fpsn_patch")
            # Area-weighted LAI across active PFTs for this step.
            _lai_col = lai_area_weighted.mean(1)  # (ncell,)
            # Soil temperature: layer 0 — ensemble-mean over members -> (ncell,).
            def _diag_tensor(value) -> "torch.Tensor":
                return (
                    value.to(device=device, dtype=dtype)
                    if isinstance(value, torch.Tensor)
                    else torch.as_tensor(value, device=device, dtype=dtype)
                )

            _fpsn_ens_full = (
                _diag_tensor(_f_raw)
                if _f_raw is not None
                else torch.zeros_like(_diag_tensor(state.h2osfc))
            )
            _f_diag = _fpsn_ens_full.mean(1) if _f_raw is not None else torch.zeros_like(_t)
            _gpp_gc_diag = _f_diag * (SECSPDAY * GPP_UMOL_CO2_TO_GC)
            _t_soil = _diag_tensor(state.t_soisno[..., 0]).mean(1)   # (ncell,)
            _t_soil_layers = _diag_tensor(state.t_soisno[..., [0, 2, 4]]).mean(1)
            _t_soil_all_layers = _diag_tensor(state.t_soisno[..., :nlev]).mean(1)
            _t_veg = blended.get("t_veg")
            _t_veg_diag = (
                _diag_tensor(_t_veg).mean(1)
                if _t_veg is not None
                else torch.full_like(_t, torch.nan)
            )
            _rain_diag = forc_rain.mean(1)
            _snow_diag = forc_snow.mean(1)
            _fsds_diag = fsds.mean(1)
            _shortwave_diag: dict[str, "torch.Tensor"] = {}
            def _pft_static_stack(name: str) -> "torch.Tensor | None":
                rows = []
                for pft_static in pft_statics:
                    raw = getattr(pft_static, name, None)
                    if raw is None:
                        return None
                    value_t = _diag_tensor(raw).reshape(-1)
                    if value_t.numel() == 0:
                        return None
                    rows.append(value_t[0].reshape(1, 1).expand(ncell, 1))
                return torch.stack(rows, dim=0)

            _rad_static = {
                _name: _pft_static_stack(_name)
                for _name in (
                    "rholvis",
                    "rholnir",
                    "taulvis",
                    "taulnir",
                    "rhosvis",
                    "rhosnir",
                    "tausvis",
                    "tausnir",
                    "xl",
                )
            }
            if all(_value is not None for _value in _rad_static.values()):
                _frac_sno_for_rad = blended.get("diag_frac_sno_input", state.frac_sno)
                _snow_albedo_for_rad = None
                if (
                    _frac_sno_for_rad is not None
                    and state.snow_layers is not None
                    and state.snow_layers.snw_rds is not None
                ):
                    _frac_sno_t = torch.clamp(_diag_tensor(_frac_sno_for_rad), min=0.0, max=1.0)
                    _has_snow = _frac_sno_t > 1.0e-8
                    if state.h2osno is not None:
                        _has_snow = _has_snow & (_diag_tensor(state.h2osno) > 1.0e-8)
                    if state.snow_depth is not None:
                        _has_snow = _has_snow & (_diag_tensor(state.snow_depth) >= SNOW_MIN_ACTIVE_DEPTH)
                    _snl_raw = state.snow_layers.snl
                    _snl_t = (
                        _snl_raw.to(device=device)
                        if isinstance(_snl_raw, torch.Tensor)
                        else torch.as_tensor(_snl_raw, device=device)
                    )
                    _has_snow = _has_snow & (_snl_t < 0)
                    _snw_rds = _diag_tensor(state.snow_layers.snw_rds)
                    _snw_rds_top = torch.where(
                        _has_snow,
                        _snw_rds[..., 0],
                        torch.full_like(_snw_rds[..., 0], SNW_RDS_MIN),
                    )
                    _snow_albedo_for_rad = _snow_albedo_from_grain_radius_torch(
                        _snw_rds_top,
                        device=device,
                        dtype=dtype,
                    )
                _rad_by_pft = surface_shortwave_diagnostics_torch(
                    fsds,
                    device=device,
                    dtype=dtype,
                    elai=lai_active_today.unsqueeze(-1),
                    esai=sai_active_today.unsqueeze(-1),
                    atmosphere=atmosphere,
                    soil_color=shared_static.soil_color,
                    h2osoi_vol_top=common_feature_state["h2osoi_vol"][..., 0],
                    frac_sno=_frac_sno_for_rad,
                    snow_albedo=_snow_albedo_for_rad,
                    rholvis=_rad_static["rholvis"],
                    rholnir=_rad_static["rholnir"],
                    taulvis=_rad_static["taulvis"],
                    taulnir=_rad_static["taulnir"],
                    rhosvis=_rad_static["rhosvis"],
                    rhosnir=_rad_static["rhosnir"],
                    tausvis=_rad_static["tausvis"],
                    tausnir=_rad_static["tausnir"],
                    xl=_rad_static["xl"],
                )
                def _reduce_shortwave_diag(_key: str, _value: "torch.Tensor") -> "torch.Tensor":
                    return _reduce_regional_shortwave_diagnostic(
                        _key,
                        _value,
                        pft_weights=w_dev,
                        universal_pft_weights=w_universal_dev,
                        universal_pft_count=len(universal_batch_indices),
                        device=device,
                        dtype=dtype,
                    )
                _shortwave_diag = {
                    _key: _reduce_shortwave_diag(_key, _value)
                    for _key, _value in _rad_by_pft.items()
                }
            # Volumetric liquid water content: liq [kg/m²] / (dz [m] * DENH2O [kg/m³]).
            _soil_liq_hydro_full = _diag_tensor(state.h2osoi_liq[..., :nlev_hydro_static])
            _soil_ice_hydro_full = _diag_tensor(state.h2osoi_ice[..., :nlev_hydro_static])
            _liq_vol_layers = _soil_liq_hydro_full[..., :5].mean(1) / (_dz_liq_diag * 1000.0)
            _h2osoi_vol_all_layers = (
                (_soil_liq_hydro_full + _soil_ice_hydro_full).mean(1)
                / (_dz_hydro_diag * 1000.0)
            )
            _liq_vol = _liq_vol_layers[:, 0]  # (ncell,) m³/m³
            _h2osfc = _diag_tensor(state.h2osfc).mean(1)  # (ncell,) kg/m² == mm water equivalent
            _zwt = (
                _diag_tensor(state.zwt).mean(1)
                if getattr(state, "zwt", None) is not None
                else torch.zeros_like(_t)
            )
            _btran = blended.get("diag_btran_input")
            _btran_diag = (
                _diag_tensor(_btran).mean(1)
                if _btran is not None
                else torch.full_like(_t, torch.nan)
            )
            _tran = blended.get("qflx_tran_veg")
            _tran_full = (
                _diag_tensor(_tran)
                if _tran is not None
                else torch.zeros_like(_diag_tensor(state.h2osfc))
            )
            _tran_diag = (
                _tran_full.mean(1)
                if _tran is not None
                else torch.zeros_like(_t)
            )
            _canopy_evap = blended.get("qflx_evap_canopy")
            _canopy_evap_full = (
                _diag_tensor(_canopy_evap)
                if _canopy_evap is not None
                else torch.zeros_like(_tran_full)
            )
            _canopy_evap_diag = (
                _canopy_evap_full.mean(1)
                if _canopy_evap is not None
                else torch.zeros_like(_t)
            )
            _ev_soil = blended.get("qflx_ev_soil")
            _ev_soil_diag = (
                _diag_tensor(_ev_soil).mean(1)
                if _ev_soil is not None
                else torch.zeros_like(_t)
            )
            _ev_snow = blended.get("qflx_ev_snow")
            _ev_snow_diag = (
                _diag_tensor(_ev_snow).mean(1)
                if _ev_snow is not None
                else torch.zeros_like(_t)
            )
            _ev_h2osfc = blended.get("qflx_ev_h2osfc")
            _ev_h2osfc_diag = (
                _diag_tensor(_ev_h2osfc).mean(1)
                if _ev_h2osfc is not None
                else torch.zeros_like(_t)
            )
            _frac_sno_flux = torch.clamp(
                _diag_tensor(blended.get("diag_frac_sno_input", state.frac_sno)),
                min=0.0,
                max=1.0,
            )
            _frac_h2osfc_flux = torch.clamp(
                _diag_tensor(blended.get("diag_frac_h2osfc_input", state.frac_h2osfc)),
                min=0.0,
                max=1.0,
            )
            _soil_frac_flux = torch.clamp(
                1.0 - _frac_sno_flux - _frac_h2osfc_flux,
                min=0.0,
            )
            _ev_snow_full = (
                _diag_tensor(_ev_snow)
                if _ev_snow is not None
                else torch.zeros_like(_frac_sno_flux)
            )
            _ev_soil_full = (
                _diag_tensor(_ev_soil)
                if _ev_soil is not None
                else torch.zeros_like(_soil_frac_flux)
            )
            _ev_h2osfc_full = (
                _diag_tensor(_ev_h2osfc)
                if _ev_h2osfc is not None
                else torch.zeros_like(_frac_h2osfc_flux)
            )
            # Hydrology may supply-limit the potential bare-soil flux before
            # removing water from layer 1.  Report that realized, already
            # area-weighted flux so QSOIL and latent heat match the water
            # actually exchanged with the soil column.
            _hydro_for_evap = (
                getattr(last_result, "hydrology_result", None)
                if last_result is not None
                else None
            )
            _realized_soil_evap = (
                getattr(_hydro_for_evap, "qflx_soil_evap", None)
                if _hydro_for_evap is not None
                else None
            )
            _soil_evap_full = (
                _diag_tensor(_realized_soil_evap)
                if _realized_soil_evap is not None
                else _soil_frac_flux * _ev_soil_full
            )
            _evap_soi_full = (
                _frac_sno_flux * _ev_snow_full
                + _soil_evap_full
                + _frac_h2osfc_flux * _ev_h2osfc_full
            )
            _evap_soi_diag = _evap_soi_full.mean(1)
            _evap_tot_diag = _tran_diag + _canopy_evap_diag + _evap_soi_diag
            _liq_input = _diag_tensor(
                blended.get("diag_h2osoi_liq_input", state.h2osoi_liq)
            )[..., 0]
            _ice_input = _diag_tensor(
                blended.get("diag_h2osoi_ice_input", state.h2osoi_ice)
            )[..., 0]
            _htvp = torch.where(
                (_liq_input <= 0.0) & (_ice_input > 0.0),
                torch.full_like(_liq_input, LATENT_HEAT_SUBLIMATION),
                torch.full_like(_liq_input, LATENT_HEAT_VAPORIZATION),
            )
            _lh_full = (
                LATENT_HEAT_VAPORIZATION * (_tran_full + _canopy_evap_full)
                + _htvp * _evap_soi_full
            )
            _lh_diag = _lh_full.mean(1)
            _sh = blended.get("eflx_sh_tot")
            _sh_diag = (
                _diag_tensor(_sh).mean(1)
                if _sh is not None
                else torch.zeros_like(_t)
            )
            _soil_temp_result = (
                getattr(last_result, "soil_temperature_result", None)
                if last_result is not None
                else None
            )
            _eflx_fgr12_raw = (
                None
                if _soil_temp_result is None
                else getattr(_soil_temp_result, "eflx_fgr12", None)
            )
            _eflx_fgr12_diag = (
                _diag_tensor(_eflx_fgr12_raw).mean(1)
                if _eflx_fgr12_raw is not None
                else torch.zeros_like(_t)
            )
            def _target_diag(name: str) -> "torch.Tensor":
                _raw = blended.get(name)
                if _raw is None:
                    return torch.full_like(_t, torch.nan)
                _value = _diag_tensor(_raw)
                return _value.mean(1) if _value.ndim > 1 else _value

            _hs_soil_diag = _target_diag("hs_soil")
            _hs_top_snow_diag = _target_diag("hs_top_snow")
            _hs_h2osfc_diag = _target_diag("hs_h2osfc")
            _dhsdT_diag = _target_diag("dhsdT")
            _energy_diag = {}
            if "surface_energy_closure_residual" in blended:
                _energy_diag = {
                    "eflx_sh_soil_mean": _target_diag("eflx_sh_soil"),
                    "sabg_soil_mean": _target_diag("sabg_soil"),
                    "sabg_soil_learned_mean": _target_diag("sabg_soil_learned"),
                    "sabg_soil_diagnostic_mean": _target_diag("sabg_soil_diagnostic"),
                    "sabg_soil_learned_minus_diagnostic_mean": _target_diag(
                        "sabg_soil_learned_minus_diagnostic"
                    ),
                    "dlrad_mean": _target_diag("dlrad"),
                    "cgrnds_mean": _target_diag("cgrnds"),
                    "cgrndl_mean": _target_diag("cgrndl"),
                    "surface_energy_closure_residual_mean": _target_diag(
                        "surface_energy_closure_residual"
                    ),
                }
                _closure_residual_cumulative.add_(
                    _energy_diag["surface_energy_closure_residual_mean"] * dt
                )
            _tg_full = _diag_tensor(coupled_t_grnd_from_state_torch(state, device, dtype))
            _tg_diag = _tg_full.mean(1)

            def _pft_target_stack(name: str) -> "torch.Tensor | None":
                rows = []
                for _pft_idx in range(npft):
                    _value = None
                    if (
                        universal_targets_stack is not None
                        and _pft_idx in universal_batch_slot_for_pft
                        and name in universal_targets_stack
                    ):
                        _value = universal_targets_stack[name][universal_batch_slot_for_pft[_pft_idx]]
                    if _value is None and _pft_idx < len(all_targets) and all_targets[_pft_idx] is not None:
                        _value = all_targets[_pft_idx].get(name)
                    if _value is None:
                        return None
                    rows.append(_diag_tensor(_value))
                return torch.stack(rows, dim=0) if rows else None

            _fire_diag = torch.full_like(_t, torch.nan)
            _fira_diag = torch.full_like(_t, torch.nan)
            _t_veg_stack = _pft_target_stack("t_veg")
            if _t_veg_stack is not None:
                _flds_full = flds.expand(-1, nens) if flds.shape[1] == 1 else _diag_tensor(flds)
                _frac_sno_lw = torch.clamp(_diag_tensor(blended.get("diag_frac_sno_input", state.frac_sno)), min=0.0, max=1.0)
                _emg = (1.0 - _frac_sno_lw) * 0.96 + _frac_sno_lw * 0.97
                _lai_lw = torch.clamp(lai_active_today, min=0.0)
                _sai_lw = torch.clamp(sai_active_today, min=0.0)
                _emv = (1.0 - torch.exp(-(_lai_lw + _sai_lw))).unsqueeze(-1)
                _frac_veg_lw = torch.clamp(frac_veg_active_today, min=0.0, max=1.0).unsqueeze(-1)
                _tg_emit4 = torch.clamp(_tg_full, min=150.0, max=400.0).pow(4).unsqueeze(0)
                _tv_emit4 = torch.clamp(_t_veg_stack, min=150.0, max=400.0).pow(4)
                _flds_b = _flds_full.unsqueeze(0)
                _emg_b = _emg.unsqueeze(0)
                _ulrad = (
                    (1.0 - _emg_b) * (1.0 - _emv).pow(2) * _flds_b
                    + _emv * (1.0 + (1.0 - _emg_b) * (1.0 - _emv)) * STEFAN_BOLTZMANN * _tv_emit4
                    + _emg_b * (1.0 - _emv) * STEFAN_BOLTZMANN * _tg_emit4
                )
                _open_ground_lw = (
                    (1.0 - _frac_veg_lw) * (1.0 - _emg_b) * _flds_b
                    + (1.0 - _frac_veg_lw) * _emg_b * STEFAN_BOLTZMANN * _tg_emit4
                )
                _fire_pft = torch.where(
                    _emv > 1.0e-8,
                    _ulrad + _open_ground_lw,
                    _open_ground_lw,
                )
                _fire_full = _blend_pft_stack(_fire_pft)
                _fire_diag = _fire_full.mean(1) if _fire_full.ndim > 1 else _fire_full
                _fira_full = _fire_full - _flds_full
                _fira_diag = _fira_full.mean(1) if _fira_full.ndim > 1 else _fira_full
            _trait_std_diag: dict[str, "torch.Tensor"] = {}
            if universal_trait_var_targets_stack is not None:
                for _target_name, _out_key in _REGIONAL_TRAIT_STD_TARGET_MAP.items():
                    _var_stack = universal_trait_var_targets_stack.get(_target_name)
                    if _var_stack is None:
                        continue
                    _trait_std_full = _blend_independent_trait_std_stack(_var_stack)
                    _trait_std_diag[_out_key] = (
                        _trait_std_full.mean(1)
                        if _trait_std_full.ndim > 1
                        else _trait_std_full
                    )

            def _hydro_flux_full(name: str) -> "torch.Tensor":
                hydro = getattr(last_result, "hydrology_result", None) if last_result is not None else None
                value = getattr(hydro, name, None) if hydro is not None else None
                if value is None:
                    return torch.zeros_like(_diag_tensor(state.h2osfc))
                return _diag_tensor(value)

            _runoff_surf_full = _hydro_flux_full("qflx_surf")
            _runoff_h2osfc_surf_full = _hydro_flux_full("qflx_h2osfc_surf")
            _runoff_drain_full = _hydro_flux_full("qflx_drain")
            _runoff_deep_drain_full = _hydro_flux_full("qflx_deep_drain")
            _runoff_tot_full = (
                _runoff_surf_full
                + _runoff_h2osfc_surf_full
                + _runoff_drain_full
                + _runoff_deep_drain_full
            )
            _runoff_surf_diag = _runoff_surf_full.mean(1)
            _runoff_h2osfc_surf_diag = _runoff_h2osfc_surf_full.mean(1)
            _runoff_drain_diag = _runoff_drain_full.mean(1)
            _runoff_deep_drain_diag = _runoff_deep_drain_full.mean(1)
            _runoff_tot_diag = _runoff_tot_full.mean(1)
            _snow_depth_diag = (
                _diag_tensor(state.snow_depth).mean(1)
                if getattr(state, "snow_depth", None) is not None
                else torch.zeros_like(_t)
            )
            _h2osno_diag = (
                _diag_tensor(state.h2osno).mean(1)
                if getattr(state, "h2osno", None) is not None
                else torch.zeros_like(_t)
            )
            _frac_sno_diag_full = (
                _diag_tensor(state.frac_sno)
                if getattr(state, "frac_sno", None) is not None
                else torch.zeros_like(_diag_tensor(state.snow_depth))
                if getattr(state, "snow_depth", None) is not None
                else torch.zeros_like(_diag_tensor(state.h2osfc))
            )
            _snow_depth_full = (
                _diag_tensor(state.snow_depth)
                if getattr(state, "snow_depth", None) is not None
                else torch.zeros_like(_frac_sno_diag_full)
            )
            _frac_sno_diag = _frac_sno_diag_full.mean(1)
            _active_frac_sno_full, _active_snow_depth_full = _active_snow_feature_tensors(state)
            _active_frac_sno_diag = _diag_tensor(_active_frac_sno_full).mean(1)
            _active_snow_depth_diag = _diag_tensor(_active_snow_depth_full).mean(1)
            _snow_depth_eff_diag = (_frac_sno_diag_full * _snow_depth_full).mean(1)
            if record_hourly_diag_maps:
                def _append_hourly_h0(_key: str, _value: "torch.Tensor") -> None:
                    _hourly_h0_rows.setdefault(_key, []).append(
                        _value.detach().cpu().numpy().astype(np.float32, copy=False)
                    )

                for _key, _value in (
                    ("forc_t_mean", _t),
                    ("forc_q_mean", _q),
                    ("forc_pbot_mean", _pbot),
                    ("forc_lwrad_mean", _lwrad),
                    ("forc_wind_mean", _wind),
                    ("rh2m_mean", _rh2m_diag),
                    ("forc_rain_mean", _rain_diag),
                    ("forc_snow_mean", _snow_diag),
                    ("fsds_mean", _fsds_diag),
                    ("fpsn_mean", _f_diag),
                    ("gpp_gc_m2_d_mean", _gpp_gc_diag),
                    ("lai_mean", _lai_col),
                    ("tg_mean", _tg_diag),
                    ("eflx_lwrad_out_mean", _fire_diag),
                    ("eflx_lwrad_net_mean", _fira_diag),
                    ("eflx_fgr12_mean", _eflx_fgr12_diag),
                    ("hs_soil_mean", _hs_soil_diag),
                    ("hs_top_snow_mean", _hs_top_snow_diag),
                    ("hs_h2osfc_mean", _hs_h2osfc_diag),
                    ("dhsdT_mean", _dhsdT_diag),
                    ("elm_h0_TSOI", _t_soil_all_layers),
                    ("elm_h0_H2OSOI", _h2osoi_vol_all_layers),
                    ("t_veg_mean", _t_veg_diag),
                    ("liq_mean", _liq_vol),
                    ("h2osfc_mean", _h2osfc),
                    ("h2osno_mean", _h2osno_diag),
                    ("frac_sno_mean", _frac_sno_diag),
                    ("active_frac_sno_mean", _active_frac_sno_diag),
                    ("zwt_mean", _zwt),
                    ("btran_mean", _btran_diag),
                    ("qvegt_mean", _tran_diag),
                    ("qflx_tran_veg_mean", _tran_diag),
                    ("qvege_mean", _canopy_evap_diag),
                    ("qflx_ev_soil_mean", _ev_soil_diag),
                    ("qsoil_mean", _evap_soi_diag),
                    ("qflx_evap_soi_mean", _evap_soi_diag),
                    ("qflx_evap_tot_mean", _evap_tot_diag),
                    ("eflx_lh_tot_mean", _lh_diag),
                    ("eflx_sh_tot_mean", _sh_diag),
                    ("qflx_surf_mean", _runoff_surf_diag),
                    ("qflx_h2osfc_surf_mean", _runoff_h2osfc_surf_diag),
                    ("qflx_drain_mean", _runoff_drain_diag),
                    ("qflx_deep_drain_mean", _runoff_deep_drain_diag),
                    ("qflx_runoff_tot_mean", _runoff_tot_diag),
                    ("snow_depth_mean", _snow_depth_diag),
                    ("active_snow_depth_mean", _active_snow_depth_diag),
                    ("snow_depth_eff_mean", _snow_depth_eff_diag),
                ):
                    _append_hourly_h0(_key, _value)
                for _key, _value in _energy_diag.items():
                    _append_hourly_h0(_key, _value)
                for _key, _value in _shortwave_diag.items():
                    _append_hourly_h0(_key, _value)
            if record_daily_diag_maps:
                if _f_raw is not None:
                    if _daily_fpsn_ens_sum is None:
                        _daily_fpsn_ens_sum = _fpsn_ens_full.clone()
                    else:
                        _daily_fpsn_ens_sum.add_(_fpsn_ens_full)
                    if _period_fpsn_ens_sum is None:
                        _period_fpsn_ens_sum = _fpsn_ens_full.clone()
                    else:
                        _period_fpsn_ens_sum.add_(_fpsn_ens_full)
                    _period_fpsn_ens_count += 1
                if _daily_runoff_ens_sum is None:
                    _daily_runoff_ens_sum = _runoff_tot_full.clone()
                else:
                    _daily_runoff_ens_sum.add_(_runoff_tot_full)
                if _period_runoff_ens_sum is None:
                    _period_runoff_ens_sum = _runoff_tot_full.clone()
                else:
                    _period_runoff_ens_sum.add_(_runoff_tot_full)
                _period_runoff_ens_count += 1
                if _trait_std_diag:
                    for _key, _value in _trait_std_diag.items():
                        if _key in _daily_trait_std_sum:
                            _daily_trait_std_sum[_key].add_(_value)
                        else:
                            _daily_trait_std_sum[_key] = _value.clone()
                        if _key in _period_trait_std_sum:
                            _period_trait_std_sum[_key].add_(_value)
                        else:
                            _period_trait_std_sum[_key] = _value.clone()
                    _period_trait_std_count += 1
                for _key, _value in _shortwave_diag.items():
                    if _key in _daily_shortwave_sum:
                        _daily_shortwave_sum[_key].add_(_value)
                    else:
                        _daily_shortwave_sum[_key] = _value.clone()
                for _key, _value in _energy_diag.items():
                    if _key in _daily_energy_sum:
                        _daily_energy_sum[_key].add_(_value)
                    else:
                        _daily_energy_sum[_key] = _value.clone()
                _water_diag_nlev = min(_WATER_DIAG_NLEV, nlev_hydro_static)
                _soil_liq_full = _diag_tensor(state.h2osoi_liq[..., :nlev_hydro_static])
                _soil_ice_full = _diag_tensor(state.h2osoi_ice[..., :nlev_hydro_static])
                _soil_liq_total_diag = _soil_liq_full.sum(dim=-1).mean(1)
                _soil_ice_total_diag = _soil_ice_full.sum(dim=-1).mean(1)
                _h2ocan_diag = _diag_tensor(state.h2ocan).mean(1)
                _wa_diag = (
                    _diag_tensor(state.wa).mean(1)
                    if getattr(state, "wa", None) is not None
                    else torch.zeros_like(_t)
                )
                _tws_diag = (
                    _soil_liq_total_diag
                    + _soil_ice_total_diag
                    + _h2ocan_diag
                    + _h2osfc
                    + _h2osno_diag
                    + _wa_diag
                )
                _smp_l_layers = _diag_tensor(common_feature_state["smp_l"][..., :_water_diag_nlev]).mean(1)
                _rresis_full = _area_weighted_rresis_layers_from_common(common_feature_state)
                _rresis_layers = (
                    _diag_tensor(_rresis_full[..., :_water_diag_nlev]).mean(1)
                    if _rresis_full is not None
                    else torch.full((ncell, _water_diag_nlev), float("nan"), device=device, dtype=dtype)
                )
                _qroot_full = (
                    _diag_tensor(_last_qflx_rootsoi[..., :_water_diag_nlev])
                    if _last_qflx_rootsoi is not None
                    else torch.zeros((ncell, nens, _water_diag_nlev), device=device, dtype=dtype)
                )
                _qroot_layers = _qroot_full.mean(1)
                _rootr_layers = torch.where(
                    _tran_diag.unsqueeze(-1) > 1.0e-15,
                    _qroot_layers / torch.clamp(_tran_diag.unsqueeze(-1), min=1.0e-15),
                    torch.zeros_like(_qroot_layers),
                )
            if record_diag_maps:
                if _acc_forc_t is None:
                    _acc_forc_t = _t.clone()
                    _acc_forc_q = _q.clone()
                    _acc_forc_pbot = _pbot.clone()
                    _acc_forc_lwrad = _lwrad.clone()
                    _acc_forc_wind = _wind.clone()
                    _acc_rh2m = _rh2m_diag.clone()
                    _acc_forc_rain = _rain_diag.clone()
                    _acc_forc_snow = _snow_diag.clone()
                    _acc_fsds = _fsds_diag.clone()
                    _acc_fpsn   = _f_diag.clone()
                    _acc_lai    = _lai_col.clone()
                    _acc_t_soil = _t_soil.clone()
                    _acc_t_soil_layers = _t_soil_layers.clone()
                    _acc_elm_h0_tsoi = _t_soil_all_layers.clone()
                    _acc_elm_h0_h2osoi = _h2osoi_vol_all_layers.clone()
                    _acc_t_veg  = _t_veg_diag.clone()
                    _acc_liq    = _liq_vol.clone()
                    _acc_liq_layers = _liq_vol_layers.clone()
                    _acc_h2osfc = _h2osfc.clone()
                    _acc_h2osno = _h2osno_diag.clone()
                    _acc_frac_sno = _frac_sno_diag.clone()
                    _acc_active_frac_sno = _active_frac_sno_diag.clone()
                    _acc_zwt = _zwt.clone()
                    _acc_btran = _btran_diag.clone()
                    _acc_tran = _tran_diag.clone()
                    _acc_canopy_evap = _canopy_evap_diag.clone()
                    _acc_ev_soil = _ev_soil_diag.clone()
                    _acc_evap_soi = _evap_soi_diag.clone()
                    _acc_evap_tot = _evap_tot_diag.clone()
                    _acc_sh = _sh_diag.clone()
                    _acc_tg = _tg_diag.clone()
                    _acc_fire = _fire_diag.clone()
                    _acc_fira = _fira_diag.clone()
                    _acc_fgr = _eflx_fgr12_diag.clone()
                    _acc_hs_soil = _hs_soil_diag.clone()
                    _acc_hs_top_snow = _hs_top_snow_diag.clone()
                    _acc_hs_h2osfc = _hs_h2osfc_diag.clone()
                    _acc_dhsdT = _dhsdT_diag.clone()
                    _acc_runoff_surf = _runoff_surf_diag.clone()
                    _acc_runoff_h2osfc_surf = _runoff_h2osfc_surf_diag.clone()
                    _acc_runoff_drain = _runoff_drain_diag.clone()
                    _acc_runoff_deep_drain = _runoff_deep_drain_diag.clone()
                    _acc_runoff_tot = _runoff_tot_diag.clone()
                    _acc_lh = _lh_diag.clone()
                    _acc_snow_depth = _snow_depth_diag.clone()
                    _acc_active_snow_depth = _active_snow_depth_diag.clone()
                    _acc_snow_depth_eff = _snow_depth_eff_diag.clone()
                else:
                    _acc_forc_t.add_(_t)
                    _acc_forc_q.add_(_q)
                    _acc_forc_pbot.add_(_pbot)
                    _acc_forc_lwrad.add_(_lwrad)
                    _acc_forc_wind.add_(_wind)
                    _acc_rh2m.add_(_rh2m_diag)
                    _acc_forc_rain.add_(_rain_diag)
                    _acc_forc_snow.add_(_snow_diag)
                    _acc_fsds.add_(_fsds_diag)
                    _acc_fpsn.add_(_f_diag)
                    _acc_lai.add_(_lai_col)
                    _acc_t_soil.add_(_t_soil)
                    _acc_t_soil_layers.add_(_t_soil_layers)
                    _acc_elm_h0_tsoi.add_(_t_soil_all_layers)
                    _acc_elm_h0_h2osoi.add_(_h2osoi_vol_all_layers)
                    _acc_t_veg.add_(_t_veg_diag)
                    _acc_liq.add_(_liq_vol)
                    _acc_liq_layers.add_(_liq_vol_layers)
                    _acc_h2osfc.add_(_h2osfc)
                    _acc_h2osno.add_(_h2osno_diag)
                    _acc_frac_sno.add_(_frac_sno_diag)
                    _acc_active_frac_sno.add_(_active_frac_sno_diag)
                    _acc_zwt.add_(_zwt)
                    _acc_btran.add_(_btran_diag)
                    _acc_tran.add_(_tran_diag)
                    _acc_canopy_evap.add_(_canopy_evap_diag)
                    _acc_ev_soil.add_(_ev_soil_diag)
                    _acc_evap_soi.add_(_evap_soi_diag)
                    _acc_evap_tot.add_(_evap_tot_diag)
                    _acc_sh.add_(_sh_diag)
                    _acc_tg.add_(_tg_diag)
                    _acc_fire.add_(_fire_diag)
                    _acc_fira.add_(_fira_diag)
                    _acc_fgr.add_(_eflx_fgr12_diag)
                    _acc_hs_soil.add_(_hs_soil_diag)
                    _acc_hs_top_snow.add_(_hs_top_snow_diag)
                    _acc_hs_h2osfc.add_(_hs_h2osfc_diag)
                    _acc_dhsdT.add_(_dhsdT_diag)
                    _acc_runoff_surf.add_(_runoff_surf_diag)
                    _acc_runoff_h2osfc_surf.add_(_runoff_h2osfc_surf_diag)
                    _acc_runoff_drain.add_(_runoff_drain_diag)
                    _acc_runoff_deep_drain.add_(_runoff_deep_drain_diag)
                    _acc_runoff_tot.add_(_runoff_tot_diag)
                    _acc_lh.add_(_lh_diag)
                    _acc_snow_depth.add_(_snow_depth_diag)
                    _acc_active_snow_depth.add_(_active_snow_depth_diag)
                    _acc_snow_depth_eff.add_(_snow_depth_eff_diag)
                for _key, _value in _shortwave_diag.items():
                    if _key in _acc_shortwave_diag:
                        _acc_shortwave_diag[_key].add_(_value)
                    else:
                        _acc_shortwave_diag[_key] = _value.clone()
                for _key, _value in _energy_diag.items():
                    if _key in _acc_energy_diag:
                        _acc_energy_diag[_key].add_(_value)
                    else:
                        _acc_energy_diag[_key] = _value.clone()
                for _key, _value in _trait_std_diag.items():
                    if _key in _acc_trait_std:
                        _acc_trait_std[_key].add_(_value)
                    else:
                        _acc_trait_std[_key] = _value.clone()
                _acc_n += 1
            if record_daily_diag_maps:
                _water_storage_stack = (
                    _soil_liq_total_diag,
                    _soil_ice_total_diag,
                    _h2ocan_diag,
                    _wa_diag,
                    _tws_diag,
                )
                _water_layer_stack = tuple(
                    layer_tensor[:, lev]
                    for layer_tensor in (_smp_l_layers, _rresis_layers, _rootr_layers, _qroot_layers)
                    for lev in range(_WATER_DIAG_NLEV)
                )
                _daily_stack = torch.stack(
                    (
                        _t,
                        _q,
                        _pbot,
                        _lwrad,
                        _wind,
                        _rh2m_diag,
                        _rain_diag,
                        _snow_diag,
                        _fsds_diag,
                        _tg_diag,
                        _fire_diag,
                        _fira_diag,
                        _eflx_fgr12_diag,
                        _hs_soil_diag,
                        _hs_top_snow_diag,
                        _hs_h2osfc_diag,
                        _dhsdT_diag,
                        _f_diag,
                        _gpp_gc_diag,
                        _lai_col,
                        _t_soil,
                        _t_soil_layers[:, 0],
                        _t_soil_layers[:, 1],
                        _t_soil_layers[:, 2],
                        _t_veg_diag,
                        _liq_vol,
                        _liq_vol_layers[:, 0],
                        _liq_vol_layers[:, 1],
                        _liq_vol_layers[:, 2],
                        _liq_vol_layers[:, 3],
                        _liq_vol_layers[:, 4],
                        _h2osfc,
                        _h2osno_diag,
                        _frac_sno_diag,
                        _active_frac_sno_diag,
                        _zwt,
                        _btran_diag,
                        _tran_diag,
                        _canopy_evap_diag,
                        _evap_soi_diag,
                        _tran_diag,
                        _ev_soil_diag,
                        _evap_soi_diag,
                        _sh_diag,
                        _runoff_surf_diag,
                        _runoff_h2osfc_surf_diag,
                        _runoff_drain_diag,
                        _runoff_deep_drain_diag,
                        _runoff_tot_diag,
                        _evap_tot_diag,
                        _lh_diag,
                        _snow_depth_diag,
                        _active_snow_depth_diag,
                        _snow_depth_eff_diag,
                        *_water_storage_stack,
                        *_water_layer_stack,
                    ),
                    dim=0,
                )
                if _daily_diag_sum is None:
                    _daily_diag_sum = _daily_stack.clone()
                    _daily_elm_h0_tsoi_sum = _t_soil_all_layers.clone()
                    _daily_elm_h0_h2osoi_sum = _h2osoi_vol_all_layers.clone()
                else:
                    _daily_diag_sum.add_(_daily_stack)
                    _daily_elm_h0_tsoi_sum.add_(_t_soil_all_layers)
                    _daily_elm_h0_h2osoi_sum.add_(_h2osoi_vol_all_layers)
                _daily_diag_count += 1
                if record_day_night_flux_maps:
                    _day_weight = (_fsds_diag > float(day_night_fsds_threshold)).to(dtype=dtype)
                    _night_weight = 1.0 - _day_weight
                    if _daily_day_count_sum is None:
                        _daily_day_count_sum = _day_weight.clone()
                        _daily_night_count_sum = _night_weight.clone()
                    else:
                        _daily_day_count_sum.add_(_day_weight)
                        _daily_night_count_sum.add_(_night_weight)
                    for _key, _value in (
                        ("gpp_gc_m2_d", _gpp_gc_diag),
                        ("qflx_evap_tot", _evap_tot_diag),
                        ("eflx_sh_tot", _sh_diag),
                    ):
                        _day_value = _value * _day_weight
                        _night_value = _value * _night_weight
                        if _key in _daily_day_flux_sum:
                            _daily_day_flux_sum[_key].add_(_day_value)
                            _daily_night_flux_sum[_key].add_(_night_value)
                        else:
                            _daily_day_flux_sum[_key] = _day_value.clone()
                            _daily_night_flux_sum[_key] = _night_value.clone()
                if (step_i % 24 == 23) or ((step_i + 1) == n1h):
                    _daily_mean = _daily_diag_sum / max(_daily_diag_count, 1)
                    _daily_diag_rows.append(_daily_mean.detach().cpu().numpy().astype(np.float32, copy=False))
                    if _daily_elm_h0_tsoi_sum is not None:
                        _daily_elm_h0_tsoi_rows.append(
                            (_daily_elm_h0_tsoi_sum / max(_daily_diag_count, 1))
                            .detach()
                            .cpu()
                            .numpy()
                            .astype(np.float32, copy=False)
                        )
                    if _daily_elm_h0_h2osoi_sum is not None:
                        _daily_elm_h0_h2osoi_rows.append(
                            (_daily_elm_h0_h2osoi_sum / max(_daily_diag_count, 1))
                            .detach()
                            .cpu()
                            .numpy()
                            .astype(np.float32, copy=False)
                        )
                    if record_day_night_flux_maps and _daily_day_count_sum is not None:
                        _day_count = _daily_day_count_sum
                        _night_count = (
                            _daily_night_count_sum
                            if _daily_night_count_sum is not None
                            else torch.zeros_like(_day_count)
                        )
                        _daily_day_night_rows.setdefault("daytime_sample_count", []).append(
                            _day_count.detach().cpu().numpy().astype(np.float32, copy=False)
                        )
                        _daily_day_night_rows.setdefault("nighttime_sample_count", []).append(
                            _night_count.detach().cpu().numpy().astype(np.float32, copy=False)
                        )
                        for _key in ("gpp_gc_m2_d", "qflx_evap_tot", "eflx_sh_tot"):
                            _day_sum = _daily_day_flux_sum.get(_key)
                            _night_sum = _daily_night_flux_sum.get(_key)
                            if _day_sum is not None:
                                _day_mean = torch.where(
                                    _day_count > 0.0,
                                    _day_sum / torch.clamp(_day_count, min=1.0),
                                    torch.full_like(_day_sum, float("nan")),
                                )
                                _daily_day_night_rows.setdefault(f"{_key}_day_mean", []).append(
                                    _day_mean.detach().cpu().numpy().astype(np.float32, copy=False)
                                )
                            if _night_sum is not None:
                                _night_mean = torch.where(
                                    _night_count > 0.0,
                                    _night_sum / torch.clamp(_night_count, min=1.0),
                                    torch.full_like(_night_sum, float("nan")),
                                )
                                _daily_day_night_rows.setdefault(f"{_key}_night_mean", []).append(
                                    _night_mean.detach().cpu().numpy().astype(np.float32, copy=False)
                                )
                    if _daily_fpsn_ens_sum is not None:
                        _daily_fpsn_ens_mean = _daily_fpsn_ens_sum / max(_daily_diag_count, 1)
                        _daily_fpsn_ens_rows.append(
                            _daily_fpsn_ens_mean.std(dim=1, unbiased=False)
                            .detach()
                            .cpu()
                            .numpy()
                            .astype(np.float32, copy=False)
                        )
                    if _daily_runoff_ens_sum is not None:
                        _daily_runoff_ens_mean = _daily_runoff_ens_sum / max(_daily_diag_count, 1)
                        _daily_runoff_ens_rows.append(
                            _daily_runoff_ens_mean.std(dim=1, unbiased=False)
                            .detach()
                            .cpu()
                            .numpy()
                            .astype(np.float32, copy=False)
                        )
                    for _key, _value_sum in _daily_trait_std_sum.items():
                        _daily_trait_std_rows.setdefault(_key, []).append(
                            (_value_sum / max(_daily_diag_count, 1))
                            .detach()
                            .cpu()
                            .numpy()
                            .astype(np.float32, copy=False)
                        )
                    for _key, _value_sum in _daily_shortwave_sum.items():
                        _daily_shortwave_rows.setdefault(_key, []).append(
                            (_value_sum / max(_daily_diag_count, 1))
                            .detach()
                            .cpu()
                            .numpy()
                            .astype(np.float32, copy=False)
                        )
                    for _key, _value_sum in _daily_energy_sum.items():
                        _daily_energy_rows.setdefault(_key, []).append(
                            (_value_sum / max(_daily_diag_count, 1))
                            .detach()
                            .cpu()
                            .numpy()
                            .astype(np.float32, copy=False)
                        )
                    _daily_diag_sum = None
                    _daily_day_count_sum = None
                    _daily_night_count_sum = None
                    _daily_day_flux_sum.clear()
                    _daily_night_flux_sum.clear()
                    _daily_fpsn_ens_sum = None
                    _daily_runoff_ens_sum = None
                    _daily_trait_std_sum.clear()
                    _daily_shortwave_sum.clear()
                    _daily_energy_sum.clear()
                    _daily_elm_h0_tsoi_sum = None
                    _daily_elm_h0_h2osoi_sum = None
                    _daily_diag_count = 0

        if progress_stride > 0 and ((step_i + 1) % progress_stride == 0 or (step_i + 1) == n1h):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elif device.type == "mps" and hasattr(torch, "mps"):
                torch.mps.synchronize()
            elapsed_so_far = time.perf_counter() - t0
            print(
                f"    {year}-{month:02d}: hour {step_i + 1}/{n1h} "
                f"elapsed={elapsed_so_far:.1f}s",
                flush=True,
            )

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()
    elapsed = time.perf_counter() - t0
    sanity: dict[str, float] = {}
    diag_maps: dict[str, np.ndarray] = {}
    sanity["mean_h2osfc"] = float(_to_np(state.h2osfc).mean())
    sanity["mean_t_top"] = float(_to_np(state.t_soisno[..., 0]).mean())
    _frac_sno = state.frac_sno
    if _frac_sno is not None:
        sanity["mean_frac_sno"] = float(_to_np(_frac_sno).mean())
    _h2osno = state.h2osno
    if _h2osno is not None:
        sanity["mean_h2osno"] = float(_to_np(_h2osno).mean())
        sanity["max_h2osno"] = float(_to_np(_h2osno).max())
    _snow_depth = state.snow_depth
    if _snow_depth is not None:
        sanity["max_snow_depth"] = float(_to_np(_snow_depth).max())
    if _acc_ood_n > 0 and _acc_ood_max_z is not None and _acc_ood_frac is not None:
        sanity["mean_mlp_max_abs_z"] = float((_acc_ood_max_z / _acc_ood_n).mean().detach().cpu())
        sanity["mean_mlp_frac_ood"] = float((_acc_ood_frac / _acc_ood_n).mean().detach().cpu())
    for _name, _values in _outlier_max.items():
        _finite_values = torch.where(torch.isfinite(_values), _values, torch.full_like(_values, -torch.inf))
        sanity[f"max_{_name}"] = float(torch.max(_finite_values).detach().cpu())
    for _name, _value in _clip_counts.items():
        sanity[f"clip_{_name}"] = float(_value.detach().cpu())
    if _acc_n > 0 and _acc_forc_t is not None:
        diag_maps["forc_t_mean"] = (_acc_forc_t / _acc_n).cpu().numpy()  # (ncell,)
        diag_maps["forc_q_mean"] = (_acc_forc_q / _acc_n).cpu().numpy()  # (ncell,)
        diag_maps["forc_pbot_mean"] = (_acc_forc_pbot / _acc_n).cpu().numpy()  # (ncell,)
        diag_maps["forc_lwrad_mean"] = (_acc_forc_lwrad / _acc_n).cpu().numpy()  # (ncell,)
        diag_maps["forc_wind_mean"] = (_acc_forc_wind / _acc_n).cpu().numpy()  # (ncell,)
        diag_maps["rh2m_mean"] = (_acc_rh2m / _acc_n).cpu().numpy()  # (ncell,)
        diag_maps["forc_rain_mean"] = (_acc_forc_rain / _acc_n).cpu().numpy()  # (ncell,)
        diag_maps["forc_snow_mean"] = (_acc_forc_snow / _acc_n).cpu().numpy()  # (ncell,)
        diag_maps["fsds_mean"] = (_acc_fsds / _acc_n).cpu().numpy()  # (ncell,)
        diag_maps["fpsn_mean"]   = (_acc_fpsn   / _acc_n).cpu().numpy()  # (ncell,)
        diag_maps["gpp_gc_m2_d_mean"] = (
            (_acc_fpsn / _acc_n) * (86400.0 * GPP_UMOL_CO2_TO_GC)
        ).cpu().numpy()  # (ncell,)
        diag_maps["lai_mean"]    = (_acc_lai    / _acc_n).cpu().numpy()  # (ncell,)
        diag_maps["t_soil_mean"] = (_acc_t_soil / _acc_n).cpu().numpy()  # (ncell,)
        if _acc_t_soil_layers is not None:
            _t_soil_layers_mean = (_acc_t_soil_layers / _acc_n).cpu().numpy()
            diag_maps["t_soisno_lev1_mean"] = _t_soil_layers_mean[:, 0]
            diag_maps["t_soisno_lev3_mean"] = _t_soil_layers_mean[:, 1]
            diag_maps["t_soisno_lev5_mean"] = _t_soil_layers_mean[:, 2]
        if _acc_elm_h0_tsoi is not None:
            diag_maps["elm_h0_TSOI"] = (_acc_elm_h0_tsoi / _acc_n).cpu().numpy()
        if _acc_elm_h0_h2osoi is not None:
            diag_maps["elm_h0_H2OSOI"] = (_acc_elm_h0_h2osoi / _acc_n).cpu().numpy()
        if _acc_tg is not None:
            diag_maps["tg_mean"] = (_acc_tg / _acc_n).cpu().numpy()
        if _acc_fire is not None:
            diag_maps["eflx_lwrad_out_mean"] = (_acc_fire / _acc_n).cpu().numpy()
        if _acc_fira is not None:
            diag_maps["eflx_lwrad_net_mean"] = (_acc_fira / _acc_n).cpu().numpy()
        if _acc_fgr is not None:
            diag_maps["eflx_fgr12_mean"] = (_acc_fgr / _acc_n).cpu().numpy()
        if _acc_hs_soil is not None:
            diag_maps["hs_soil_mean"] = (_acc_hs_soil / _acc_n).cpu().numpy()
        if _acc_hs_top_snow is not None:
            diag_maps["hs_top_snow_mean"] = (_acc_hs_top_snow / _acc_n).cpu().numpy()
        if _acc_hs_h2osfc is not None:
            diag_maps["hs_h2osfc_mean"] = (_acc_hs_h2osfc / _acc_n).cpu().numpy()
        if _acc_dhsdT is not None:
            diag_maps["dhsdT_mean"] = (_acc_dhsdT / _acc_n).cpu().numpy()
        diag_maps["t_veg_mean"] = (_acc_t_veg / _acc_n).cpu().numpy()  # (ncell,)
        diag_maps["liq_mean"]    = (_acc_liq    / _acc_n).cpu().numpy()  # (ncell,)
        if _acc_liq_layers is not None:
            _liq_layers_mean = (_acc_liq_layers / _acc_n).cpu().numpy()
            diag_maps["h2osoi_liqvol_lev1_mean"] = _liq_layers_mean[:, 0]
            diag_maps["h2osoi_liqvol_lev2_mean"] = _liq_layers_mean[:, 1]
            diag_maps["h2osoi_liqvol_lev3_mean"] = _liq_layers_mean[:, 2]
            diag_maps["h2osoi_liqvol_lev4_mean"] = _liq_layers_mean[:, 3]
            diag_maps["h2osoi_liqvol_lev5_mean"] = _liq_layers_mean[:, 4]
        if _acc_h2osfc is not None:
            diag_maps["h2osfc_mean"] = (_acc_h2osfc / _acc_n).cpu().numpy()  # (ncell,)
        if _acc_h2osno is not None:
            diag_maps["h2osno_mean"] = (_acc_h2osno / _acc_n).cpu().numpy()  # (ncell,)
        if _acc_frac_sno is not None:
            diag_maps["frac_sno_mean"] = (_acc_frac_sno / _acc_n).cpu().numpy()  # (ncell,)
        if _acc_active_frac_sno is not None:
            diag_maps["active_frac_sno_mean"] = (_acc_active_frac_sno / _acc_n).cpu().numpy()  # (ncell,)
        if _acc_zwt is not None:
            diag_maps["zwt_mean"] = (_acc_zwt / _acc_n).cpu().numpy()  # (ncell,)
        if _acc_btran is not None:
            diag_maps["btran_mean"] = (_acc_btran / _acc_n).cpu().numpy()  # (ncell,)
        if _acc_tran is not None:
            diag_maps["qvegt_mean"] = (_acc_tran / _acc_n).cpu().numpy()  # (ncell,)
            diag_maps["qflx_tran_veg_mean"] = (_acc_tran / _acc_n).cpu().numpy()  # (ncell,)
        if _acc_canopy_evap is not None:
            diag_maps["qvege_mean"] = (_acc_canopy_evap / _acc_n).cpu().numpy()  # (ncell,)
        if _acc_ev_soil is not None:
            diag_maps["qflx_ev_soil_mean"] = (_acc_ev_soil / _acc_n).cpu().numpy()  # (ncell,)
        if _acc_evap_soi is not None:
            diag_maps["qsoil_mean"] = (_acc_evap_soi / _acc_n).cpu().numpy()  # (ncell,)
            diag_maps["qflx_evap_soi_mean"] = (_acc_evap_soi / _acc_n).cpu().numpy()  # (ncell,)
        if _acc_evap_tot is not None:
            diag_maps["qflx_evap_tot_mean"] = (_acc_evap_tot / _acc_n).cpu().numpy()  # (ncell,)
        if _acc_sh is not None:
            diag_maps["eflx_sh_tot_mean"] = (_acc_sh / _acc_n).cpu().numpy()  # (ncell,)
        if _acc_runoff_surf is not None:
            diag_maps["qflx_surf_mean"] = (_acc_runoff_surf / _acc_n).cpu().numpy()  # (ncell,)
        if _acc_runoff_h2osfc_surf is not None:
            diag_maps["qflx_h2osfc_surf_mean"] = (_acc_runoff_h2osfc_surf / _acc_n).cpu().numpy()  # (ncell,)
        if _acc_runoff_drain is not None:
            diag_maps["qflx_drain_mean"] = (_acc_runoff_drain / _acc_n).cpu().numpy()  # (ncell,)
        if _acc_runoff_deep_drain is not None:
            diag_maps["qflx_deep_drain_mean"] = (_acc_runoff_deep_drain / _acc_n).cpu().numpy()  # (ncell,)
        if _acc_runoff_tot is not None:
            diag_maps["qflx_runoff_tot_mean"] = (_acc_runoff_tot / _acc_n).cpu().numpy()  # (ncell,)
        if _acc_lh is not None:
            diag_maps["eflx_lh_tot_mean"] = (_acc_lh / _acc_n).cpu().numpy()  # (ncell,)
        if _acc_snow_depth is not None:
            diag_maps["snow_depth_mean"] = (_acc_snow_depth / _acc_n).cpu().numpy()  # (ncell,)
        if _acc_active_snow_depth is not None:
            diag_maps["active_snow_depth_mean"] = (_acc_active_snow_depth / _acc_n).cpu().numpy()  # (ncell,)
        if _acc_snow_depth_eff is not None:
            diag_maps["snow_depth_eff_mean"] = (_acc_snow_depth_eff / _acc_n).cpu().numpy()  # (ncell,)
        for _key, _acc_value in _acc_shortwave_diag.items():
            diag_maps[_key] = (_acc_value / _acc_n).cpu().numpy()
        for _key, _acc_value in _acc_energy_diag.items():
            diag_maps[_key] = (_acc_value / _acc_n).cpu().numpy()
        if _acc_energy_diag:
            diag_maps["surface_energy_closure_residual_cumulative_j_m2"] = (
                _closure_residual_cumulative.cpu().numpy()
            )
        for _key, _acc_value in _acc_trait_std.items():
            diag_maps[_key] = (_acc_value / _acc_n).cpu().numpy()  # (ncell,)
        if _acc_ood_n > 0 and _acc_ood_max_z is not None and _acc_ood_frac is not None:
            diag_maps["mlp_ood_max_abs_z"] = (_acc_ood_max_z / _acc_ood_n).cpu().numpy()
            diag_maps["mlp_ood_frac"] = (_acc_ood_frac / _acc_ood_n).cpu().numpy()
            for name, acc in _acc_ood_feature_z.items():
                diag_maps[f"mlp_ood_abs_z_{name}"] = (acc / _acc_ood_n).cpu().numpy()
        for _key in (
            "synthetic_flnr_factor",
            "synthetic_mbbopt_factor",
            "synthetic_pct_sand_delta",
            "synthetic_pct_sand_lev1",
            "synthetic_pct_sand_mean",
        ):
            if _key in surf:
                diag_maps[_key] = np.asarray(surf[_key], dtype=np.float32)
    if _hourly_h0_rows:
        diag_maps["_hourly"] = {
            key: np.stack(rows, axis=0).astype(np.float32, copy=False)
            for key, rows in _hourly_h0_rows.items()
            if rows
        }
    if _daily_diag_rows:
        daily_stack = np.stack(_daily_diag_rows, axis=1)  # (nvar, nday, ncell)
        if daily_stack.shape[0] != len(_REGIONAL_DAILY_DIAG_KEYS):
            raise RuntimeError(
                "Regional daily diagnostic stack/key mismatch: "
                f"stack has {daily_stack.shape[0]} variables but "
                f"{len(_REGIONAL_DAILY_DIAG_KEYS)} keys are configured"
            )
        diag_maps["_daily"] = {
            key: daily_stack[idx]
            for idx, key in enumerate(_REGIONAL_DAILY_DIAG_KEYS)
        }
        if _daily_elm_h0_tsoi_rows:
            elm_h0_tsoi_daily = np.stack(_daily_elm_h0_tsoi_rows, axis=0).astype(
                np.float32,
                copy=False,
            )
            if elm_h0_tsoi_daily.shape[0] == daily_stack.shape[1]:
                diag_maps["_daily"]["elm_h0_TSOI"] = elm_h0_tsoi_daily
        if _daily_elm_h0_h2osoi_rows:
            elm_h0_h2osoi_daily = np.stack(_daily_elm_h0_h2osoi_rows, axis=0).astype(
                np.float32,
                copy=False,
            )
            if elm_h0_h2osoi_daily.shape[0] == daily_stack.shape[1]:
                diag_maps["_daily"]["elm_h0_H2OSOI"] = elm_h0_h2osoi_daily
        if _daily_fpsn_ens_rows:
            fpsn_ens_std_daily = np.stack(_daily_fpsn_ens_rows, axis=0)
            if fpsn_ens_std_daily.shape[0] == daily_stack.shape[1]:
                diag_maps["_daily"]["fpsn_ens_std"] = fpsn_ens_std_daily
        if _daily_runoff_ens_rows:
            runoff_ens_std_daily = np.stack(_daily_runoff_ens_rows, axis=0)
            if runoff_ens_std_daily.shape[0] == daily_stack.shape[1]:
                diag_maps["_daily"]["qflx_runoff_tot_ens_std"] = runoff_ens_std_daily
        for key, rows in _daily_trait_std_rows.items():
            trait_std_daily = np.stack(rows, axis=0)
            if trait_std_daily.shape[0] == daily_stack.shape[1]:
                diag_maps["_daily"][key] = trait_std_daily
        for key, rows in _daily_shortwave_rows.items():
            shortwave_daily = np.stack(rows, axis=0).astype(np.float32, copy=False)
            if shortwave_daily.shape[0] == daily_stack.shape[1]:
                diag_maps["_daily"][key] = shortwave_daily
        for key, rows in _daily_energy_rows.items():
            energy_daily = np.stack(rows, axis=0).astype(np.float32, copy=False)
            if energy_daily.shape[0] == daily_stack.shape[1]:
                diag_maps["_daily"][key] = energy_daily
        for key, rows in _daily_day_night_rows.items():
            split_values = np.stack(rows, axis=0).astype(np.float32, copy=False)
            if split_values.shape[0] == daily_stack.shape[1]:
                diag_maps["_daily"][key] = split_values
    if _period_fpsn_ens_sum is not None and _period_fpsn_ens_count > 0:
        fpsn_ens_mean_period = _period_fpsn_ens_sum / _period_fpsn_ens_count
        diag_maps.setdefault("_monthly", {})
        diag_maps["_monthly"].update({
            "fpsn_ens_std": (
                fpsn_ens_mean_period.std(dim=1, unbiased=False)
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32, copy=False)[None, :]
            )
        })
    if _period_runoff_ens_sum is not None and _period_runoff_ens_count > 0:
        runoff_ens_mean_period = _period_runoff_ens_sum / _period_runoff_ens_count
        diag_maps.setdefault("_monthly", {})
        diag_maps["_monthly"].update({
            "qflx_runoff_tot_ens_std": (
                runoff_ens_mean_period.std(dim=1, unbiased=False)
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32, copy=False)[None, :]
            )
        })
    if _period_trait_std_count > 0 and _period_trait_std_sum:
        diag_maps.setdefault("_monthly", {})
        diag_maps["_monthly"].update({
            key: (
                (acc_value / _period_trait_std_count)
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32, copy=False)[None, :]
            )
            for key, acc_value in _period_trait_std_sum.items()
        })

    if carbon_model is not None:
        carbon_series = {
            "carbon_gpp_daily": np.asarray(
                carbon_model.daily_domain_gpp[_carbon_day_start:], dtype=np.float32
            ),
            "carbon_npp_daily": np.asarray(
                carbon_model.daily_domain_npp[_carbon_day_start:], dtype=np.float32
            ),
            "carbon_hr_daily": np.asarray(
                carbon_model.daily_domain_hr[_carbon_day_start:], dtype=np.float32
            ),
        }
        carbon_series = {
            key: value
            for key, value in carbon_series.items()
            if value.size > 0 and np.all(np.isfinite(value))
        }
        if carbon_series:
            daily_diag = diag_maps.setdefault("_daily", {})
            monthly_diag = diag_maps.setdefault("_monthly", {})
            existing_daily_nday = (
                int(next(iter(daily_diag.values())).shape[0])
                if daily_diag
                else None
            )
            for key, values in carbon_series.items():
                if existing_daily_nday is None or values.size == existing_daily_nday:
                    daily_diag[key] = np.broadcast_to(values[:, None], (values.size, ncell)).copy()
                monthly_diag[f"{key}_mean"] = np.full(
                    (1, ncell),
                    float(values.mean()),
                    dtype=np.float32,
                )
        if _daily_carbon_pool_rows:
            daily_diag = diag_maps.setdefault("_daily", {})
            monthly_diag = diag_maps.setdefault("_monthly", {})
            existing_daily_nday = (
                int(next(iter(daily_diag.values())).shape[0])
                if daily_diag
                else None
            )
            for key, rows in _daily_carbon_pool_rows.items():
                values = np.stack(rows, axis=0).astype(np.float32, copy=False)
                if existing_daily_nday is None or values.shape[0] == existing_daily_nday:
                    daily_diag[key] = values
                monthly_diag[key] = np.nanmean(values, axis=0, keepdims=True).astype(
                    np.float32,
                    copy=False,
                )

    def _outlier_location(cell: int) -> str:
        active = surf.get("active_idx")
        gridcell = int(np.asarray(active)[cell]) + 1 if active is not None else cell + 1
        return (
            f"cell={cell} gridcell={gridcell} "
            f"lat={float(np.asarray(surf['lat_deg'])[cell]):.5f} "
            f"lon={float(np.asarray(surf['lon_deg'])[cell]):.5f} "
            f"row={int(np.asarray(surf['grid_row'])[cell])} "
            f"col={int(np.asarray(surf['grid_col'])[cell])}"
        )

    def _print_outlier_table(name: str) -> None:
        values_t = torch.where(
            torch.isfinite(_outlier_max[name]),
            _outlier_max[name],
            torch.full_like(_outlier_max[name], -torch.inf),
        )
        k = min(_outlier_topk, int(values_t.numel()))
        if k <= 0:
            return
        vals, idx = torch.topk(values_t, k=k)
        vals_np = vals.detach().cpu().numpy()
        if vals_np.size == 0 or not np.isfinite(vals_np[0]) or vals_np[0] <= _outlier_thresholds[name]:
            return
        idx_np = idx.detach().cpu().numpy().astype(np.int64)
        steps_np = _outlier_step[name][idx].detach().cpu().numpy().astype(np.int64)
        pft_np = _outlier_pft_id[idx].detach().cpu().numpy().astype(np.int64) if name == "pft_fpsn" else None
        print(f"    outlier {name} top cells (threshold { _outlier_thresholds[name]:.3g} {_outlier_units[name]}):")
        for rank, (value, cell, step) in enumerate(zip(vals_np, idx_np, steps_np), start=1):
            if not np.isfinite(value) or value <= _outlier_thresholds[name]:
                continue
            day = int(step // 24) + 1 if step >= 0 else -1
            hour = int(step % 24) if step >= 0 else -1
            pft_text = f" pft={int(pft_np[rank - 1])}" if pft_np is not None else ""
            print(
                f"      {rank:2d}. {value:.6g} {_outlier_units[name]} "
                f"day={day} hour={hour}{pft_text} {_outlier_location(int(cell))}"
            )

    for _outlier_name in ("fpsn", "pft_fpsn", "t_soil_raw", "t_soil", "lai", "liq_vol", "h2osfc", "h2osno", "snow_depth", "snow_depth_eff", "frac_sno"):
        _print_outlier_table(_outlier_name)

    return state, sanity, {
        "n_met_input": met_3h.shape[0],
        "n1h": n1h,
        "physics_cadence_steps": physics_cadence_steps,
        "physics_calls": _physics_ncalls,
        "snowfall_possible_steps": int(np.count_nonzero(_snowfall_possible_steps)),
        "snow_physics_calls": _physics_snow_enabled_calls,
        "active_snow_physics_calls": _physics_active_snow_calls,
        "pending_snow_physics_calls": _physics_pending_snow_calls,
        "max_active_snow_cells": int(_max_active_snow_cells),
        "final_active_snow_cells": int(_active_snow_cells),
        "physics_step_s": physics_step_seconds,
        "universal_fast_steps": _universal_fast_steps,
        "state_feature_refreshes": _state_feature_refreshes,
        "vegetation_feature_refreshes": _vegetation_feature_refreshes,
        "elapsed_s": elapsed,
    }, diag_maps


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    argv = sys.argv[1:]
    ensemble_size_given = any(
        arg == "--ensemble-size" or arg.startswith("--ensemble-size=")
        for arg in argv
    )
    regional_ensemble_size_given = any(
        arg == "--regional-ensemble-size" or arg.startswith("--regional-ensemble-size=")
        for arg in argv
    )
    regional_min_lon_given = any(
        arg == "--regional-min-lon" or arg.startswith("--regional-min-lon=")
        for arg in argv
    )
    p = argparse.ArgumentParser(
        description="Benchmark the full coupled replay with real training-file forcing.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--training-file", type=Path, default=Path("./canopyflux_training.nc"))
    p.add_argument("--surfdata-file", type=Path, default=Path("./surfdata.nc"))
    p.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    p.add_argument("--patch-index", type=int, default=1)
    p.add_argument(
        "--start-year",
        type=int,
        default=None,
        help="First training-file year to include in the replay window.",
    )
    p.add_argument("--ncell", type=int, default=64, help="Number of spatial cells in the batch.")
    p.add_argument("--ensemble-size", type=int, default=16, help="Number of ensemble members per cell.")
    p.add_argument("--nsteps", type=int, default=168, help="Timesteps per timed pass (hours of forcing to replay).")
    p.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    p.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    p.add_argument(
        "--mlp-dtype",
        choices=("same", "float32", "float64", "float16", "bfloat16"),
        default="same",
        help=(
            "Regional replay MLP forward dtype. Use bfloat16 on CUDA to run the "
            "canopy MLP on BF16 Tensor Cores while keeping coupled physics/state "
            "in --dtype. 'same' uses --dtype."
        ),
    )
    p.add_argument("--hydrology-backend", choices=("numpy", "numpy_full", "torch", "full"), default="torch")
    p.add_argument("--soil-temperature-backend", choices=("numpy", "torch"), default="torch")
    p.add_argument("--warmup-iters", type=int, default=1)
    p.add_argument(
        "--warmup-nsteps",
        type=int,
        default=None,
        help="Timesteps per warmup pass (default: same as --nsteps). "
             "A small value (e.g. 1) is sufficient to trigger torch.compile "
             "tracing without replaying the full window.",
    )
    p.add_argument("--timed-iters", type=int, default=3)
    p.add_argument("--disable-snow-model", action="store_true")
    p.add_argument(
        "--snow-grain-aging",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable SNICAR snow grain aging when the snow model is active.",
    )
    p.add_argument(
        "--snicar-aging-file",
        type=Path,
        default=None,
        help=(
            "SNICAR grain-aging lookup table. When omitted, search the CLM input-data "
            "and E3SM-emulator locations adjacent to this project."
        ),
    )
    p.add_argument(
        "--snow-scalar-only",
        action="store_true",
        help=(
            "Diagnostic snow mode: keep H2OSNO/FSNO/snow_depth but never "
            "promote snow to resolved active snow layers."
        ),
    )
    p.add_argument(
        "--blend-partial-snow-soil-temperature",
        action="store_true",
        help=(
            "Diagnostic: blend snow-covered and bare-soil thermal solutions "
            "by frac_sno for partially snow-covered columns."
        ),
    )
    p.add_argument("--compile", action=argparse.BooleanOptionalAction, default=None,
                   help="torch.compile the surface MLP (default: on when using torch backends + CUDA)")
    p.add_argument("--compile-physics", action=argparse.BooleanOptionalAction, default=None,
                   help="torch.compile hydrology and soil-temperature step functions (default: on when using torch backends + CUDA)")
    p.add_argument("--profile-sections", action="store_true",
                   help="Print per-section timing breakdown after the timed pass")
    p.add_argument(
        "--tf32-matmul",
        choices=("highest", "high", "medium"),
        default="highest",
        help="CUDA float32 matmul precision policy.",
    )
    p.add_argument(
        "--torch-num-threads",
        type=int,
        default=None,
        help=(
            "Set torch CPU intra-op and inter-op thread counts. For tiny CPU "
            "regional runs, 1 avoids large thread-pool overhead in the MLP."
        ),
    )
    p.add_argument("--dt", type=float, default=3600.0, help="Timestep length in seconds.")
    p.add_argument(
        "--plot-output",
        type=Path,
        default=Path("./plots/benchmark_replay_diagnostics.png"),
        help="Output path for the fpsn / transpiration diagnostic plot.",
    )
    p.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip the diagnostic pass and plot.",
    )
    p.add_argument(
        "--plot-daily-average",
        action="store_true",
        help="Daily-average diagnostic time series before plotting.",
    )
    p.add_argument("--ensemble-seed", type=int, default=42,
                   help="RNG seed for sampling per-ensemble flnr / mbbopt values.")
    p.add_argument(
        "--blend-patches",
        action="store_true",
        help=(
            "Run every nonzero-fraction PFT patch from surfdata sequentially and "
            "blend their outputs by fractional area weight.  --patch-index is ignored; "
            "a separate model and training slice is loaded per patch."
        ),
    )

    # ---- Regional benchmark ------------------------------------------------
    p.add_argument(
        "--regional",
        action="store_true",
        help=(
            "Full-domain regional benchmark using the TESSFA surface data and "
            "Daymet/ERA5 met forcing.  Overrides most other options."
        ),
    )
    p.add_argument(
        "--universal",
        action="store_true",
        help=(
            "Regional mode: use <model-dir>/patch_-1/model.pt for active vegetated PFTs "
            "except PFT 14, and use the individual patch_0 and patch_14 models "
            "for PFTs 0 and 14. Universal-model PFTs use one batched MLP forward "
            "when their schemas match."
        ),
    )
    p.add_argument(
        "--universal-all-pfts",
        action="store_true",
        help=(
            "Regional universal mode: use patch_-1/model.pt for PFTs 0 and 14 "
            "as well as the other active PFTs. This is intended for universal "
            "checkpoints trained with rows from all PFT types; otherwise PFTs "
            "0 and 14 continue to require individual checkpoints."
        ),
    )
    p.add_argument(
        "--regional-sparse-pft-batch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Regional universal mode: evaluate the universal MLP only for nonzero "
            "PFT-cell area weights, then scatter results back for blending."
        ),
    )
    p.add_argument(
        "--regional-sparse-individual-features",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Regional mode: assemble and evaluate patch-specific models only over "
            "their nonzero-area cells when the snow model is disabled."
        ),
    )
    p.add_argument(
        "--regional-pft-min-percent",
        type=float,
        default=0.0,
        help=(
            "Regional mode: drop PFT-cell fractions below this percentage-point "
            "threshold and renormalize the retained fractions within each cell. "
            "Zero disables pruning."
        ),
    )
    p.add_argument(
        "--regional-exclude-pft",
        nargs="+",
        type=int,
        default=(),
        metavar="PFT",
        help=(
            "Regional mode: exclude one or more PFT indices, renormalize the "
            "remaining fractions, and discard cells containing only excluded PFTs."
        ),
    )
    p.add_argument(
        "--regional-surfdata",
        type=Path,
        default=DEFAULT_TESSFA_DOMAIN2_SURFDATA,
        help="Path to the TESSFA regional surfdata NetCDF4 file.",
    )
    p.add_argument(
        "--met-dir",
        type=Path,
        default=Path("met_forcing"),
        help=(
            "Directory containing regional/site met forcing. Supports CLM1PT "
            "site all_hourly.nc, TESSFA monthly files, or TRENDY single-variable "
            "1901-2024 time-series files."
        ),
    )
    p.add_argument(
        "--regional-met-format",
        choices=("auto", "site", "tessfa", "trendy"),
        default="auto",
        help=(
            "Regional forcing layout. 'site' uses CLM1PT all_hourly.nc; "
            "'tessfa' uses monthly gridded 3-hour files; "
            "'trendy' uses compressed 0.5-degree 6-hour 1901-2024 time-series files."
        ),
    )
    p.add_argument(
        "--regional-calendar",
        choices=("gregorian", "noleap"),
        default="gregorian",
        help="Calendar for regional month lengths and day-of-year calculations.",
    )
    p.add_argument(
        "--regional-met-cycle-years",
        type=int,
        default=0,
        help=(
            "Regional mode: cycle meteorological forcing over this many years while "
            "the simulation year keeps advancing. Use 0 to disable cycling."
        ),
    )
    p.add_argument(
        "--regional-met-cycle-start-year",
        type=int,
        default=None,
        help=(
            "First forcing year used by --regional-met-cycle-years. "
            "Default: the simulation --year."
        ),
    )
    p.add_argument(
        "--lat",
        type=float,
        default=None,
        help="Regional mode: run only the nearest active grid cell to this latitude. Requires --lon.",
    )
    p.add_argument(
        "--lon",
        type=float,
        default=None,
        help="Regional mode: run only the nearest active grid cell to this longitude. Requires --lat.",
    )
    p.add_argument(
        "--pft",
        type=int,
        default=None,
        help=(
            "Regional mode: force every active cell to use this single PFT index "
            "with 100%% area weight, e.g. --pft 1 or --pft 7."
        ),
    )
    p.add_argument(
        "--clm-params",
        type=Path,
        default=DEFAULT_CLM_PARAMS_SP,
        help="CLM/ELM parameter file (provides roota/rootb and PFT trait parameters).",
    )
    p.add_argument(
        "--year",
        type=int,
        default=1980,
        help="Year to run (regional mode).",
    )
    p.add_argument(
        "--month",
        type=int,
        default=1,
        help="Starting month to run (1–12, regional mode).",
    )
    p.add_argument(
        "--nmonths",
        type=int,
        default=1,
        help="Number of months to run (regional mode).",
    )
    p.add_argument(
        "--regional-ensemble-size",
        type=int,
        default=1,
        help="Ensemble size for regional run.  When > 1, flnr and mbbopt are "
             "randomised uniformly over ±2σ of the training distribution per member.",
    )
    p.add_argument(
        "--regional-ensemble-flnr-factors",
        nargs="+",
        type=float,
        default=None,
        help=(
            "Explicit multiplicative flnr factors for a deterministic regional ensemble. "
            "When used with --regional-ensemble-mbbopt-factors, the Cartesian product is "
            "used and --regional-ensemble-size is set automatically."
        ),
    )
    p.add_argument(
        "--regional-ensemble-mbbopt-factors",
        nargs="+",
        type=float,
        default=None,
        help=(
            "Explicit multiplicative mbbopt factors for a deterministic regional ensemble. "
            "When used with --regional-ensemble-flnr-factors, the Cartesian product is "
            "used and --regional-ensemble-size is set automatically."
        ),
    )
    p.add_argument(
        "--regional-trait-ensemble-size",
        type=int,
        default=1,
        help=(
            "Same-state trait ensemble size for universal regional MLP inference. "
            "When > 1, each universal PFT-cell state is evaluated for multiple "
            "trait vectors and the trait mean flux feeds the coupled replay. "
            "Requires --universal."
        ),
    )
    p.add_argument(
        "--regional-trait-chunk-size",
        type=int,
        default=20,
        help="Number of same-state trait members evaluated per universal MLP chunk.",
    )
    p.add_argument(
        "--regional-trait-sigma",
        type=float,
        default=2.0,
        help=(
            "Approximate training-distribution clamp half-width in training standard deviations "
            "for same-state trait sampling."
        ),
    )
    p.add_argument(
        "--regional-trait-relative-sd",
        type=float,
        default=0.20,
        help=(
            "Same-state trait sampling standard deviation as a fraction of the current "
            "PFT trait value. The first trait member is always the unperturbed baseline."
        ),
    )
    p.add_argument(
        "--regional-trait-feature",
        nargs="+",
        default=list(REGIONAL_TRAIT_FEATURE_NAMES),
        help=(
            "Universal input feature names varied along the same-state trait axis. "
            "Use 'all' for the built-in PFT trait/parameter set."
        ),
    )
    p.add_argument(
        "--regional-clone-grid-rows",
        type=int,
        default=None,
        help=(
            "Regional mode: after --lat/--lon single-cell selection, clone that "
            "cell to this many synthetic grid rows. Use with --regional-clone-grid-cols."
        ),
    )
    p.add_argument(
        "--regional-clone-grid-cols",
        type=int,
        default=None,
        help=(
            "Regional mode: after --lat/--lon single-cell selection, clone that "
            "cell to this many synthetic grid columns. Use with --regional-clone-grid-rows."
        ),
    )
    p.add_argument(
        "--regional-clone-corr-length-cells",
        type=float,
        default=20.0,
        help="Synthetic clone-grid Gaussian autocorrelation length in grid cells.",
    )
    p.add_argument(
        "--regional-clone-flnr-relative-sd",
        type=float,
        default=0.20,
        help="Synthetic clone-grid flnr perturbation standard deviation as a fraction of baseline.",
    )
    p.add_argument(
        "--regional-clone-mbbopt-relative-sd",
        type=float,
        default=0.20,
        help="Synthetic clone-grid mbbopt perturbation standard deviation as a fraction of baseline.",
    )
    p.add_argument(
        "--regional-clone-sand-absolute-sd",
        type=float,
        default=8.0,
        help="Synthetic clone-grid PCT_SAND perturbation standard deviation in percentage points.",
    )
    p.add_argument(
        "--regional-clone-trait-min-factor",
        type=float,
        default=0.5,
        help="Synthetic clone-grid lower multiplicative bound for flnr/mbbopt.",
    )
    p.add_argument(
        "--regional-clone-trait-max-factor",
        type=float,
        default=1.5,
        help="Synthetic clone-grid upper multiplicative bound for flnr/mbbopt.",
    )
    p.add_argument(
        "--regional-clone-sand-min-pct",
        type=float,
        default=1.0,
        help="Synthetic clone-grid lower bound for perturbed PCT_SAND.",
    )
    p.add_argument(
        "--regional-clone-sand-max-pct",
        type=float,
        default=99.0,
        help="Synthetic clone-grid upper bound for perturbed PCT_SAND.",
    )
    p.add_argument(
        "--regional-clone-grid-spacing-km",
        type=float,
        default=4.0,
        help=(
            "Synthetic clone-grid spacing used only for map axes. The physics "
            "lat/lon remains the selected source cell so solar geometry is identical."
        ),
    )
    p.add_argument(
        "--regional-clone-seed",
        type=int,
        default=42,
        help="Random seed for synthetic clone-grid spatial fields.",
    )
    p.add_argument(
        "--regional-ncell",
        type=int,
        default=None,
        help=(
            "Regional mode: run only the first N active cells after surfdata/met "
            "screening.  Useful for profiling a smaller deterministic subdomain."
        ),
    )
    p.add_argument(
        "--regional-cell-shards",
        type=int,
        default=1,
        help=(
            "Regional mode: split the active cells into this many contiguous "
            "shards after surfdata/met screening and any --regional-ncell limit."
        ),
    )
    p.add_argument(
        "--regional-cell-shard-index",
        type=int,
        default=0,
        help=(
            "Regional mode: zero-based shard index to run when "
            "--regional-cell-shards is greater than 1."
        ),
    )
    p.add_argument(
        "--regional-min-lon",
        type=float,
        default=-95.0,
        help=(
            "Regional mode: discard active cells west of this longitude after "
            "met/surfdata screening. Use -180 to keep the full TESSFA domain."
        ),
    )
    p.add_argument(
        "--regional-max-lon",
        type=float,
        default=None,
        help=(
            "Regional mode: discard active cells east of this longitude after "
            "met/surfdata screening. Longitudes may be -180..180 or 0..360. "
            "If min lon is greater than max lon after normalization, the box "
            "is treated as crossing the dateline."
        ),
    )
    p.add_argument(
        "--regional-min-lat",
        type=float,
        default=None,
        help="Regional mode: discard active cells south of this latitude.",
    )
    p.add_argument(
        "--regional-max-lat",
        type=float,
        default=None,
        help="Regional mode: discard active cells north of this latitude.",
    )
    p.add_argument(
        "--regional-max-hours",
        type=int,
        default=None,
        help=(
            "Regional mode: limit each month pass to the first N hourly replay "
            "steps. Useful for quick profiling smoke tests."
        ),
    )
    p.add_argument(
        "--regional-prefetch",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Regional mode: prefetch the next month's met forcing and CPU "
            "precomputations in a background thread.  Default is enabled except "
            "for single-cell CPU runs, where it can dominate timing by competing "
            "with the foreground loop."
        ),
    )
    p.add_argument(
        "--regional-met-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Regional mode: cache extracted monthly met forcing slices as .npy "
            "files so repeated experiments avoid decompressing large forcing "
            "NetCDF chunks. Full-domain caches can be large."
        ),
    )
    p.add_argument(
        "--regional-met-cache-dir",
        type=Path,
        default=None,
        help=(
            "Directory for --regional-met-cache. Default: <met-dir>/.cell_cache."
        ),
    )
    p.add_argument(
        "--regional-precompute-cache",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Regional mode: cache monthly CPU coszen/daylength/LAI precompute "
            "bundles as .npz files. Useful when repeating the same domain/month."
        ),
    )
    p.add_argument(
        "--regional-precompute-cache-dir",
        type=Path,
        default=None,
        help=(
            "Directory for --regional-precompute-cache. Default: cache/regional_precompute."
        ),
    )
    p.add_argument(
        "--regional-progress-stride",
        type=int,
        default=24,
        help=(
            "Regional mode: print progress every N simulated hours. "
            "Use 0 to disable."
        ),
    )
    p.add_argument(
        "--regional-cell-timeseries",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Regional single-cell mode: record and write the diagnostic time-series "
            "plot. Use --no-regional-cell-timeseries for timing-only runs."
        ),
    )
    p.add_argument(
        "--regional-timeseries-output",
        type=Path,
        default=None,
        help=(
            "Regional single-cell mode: write hourly diagnostics to this NetCDF "
            "file. Intended for debugging state jumps."
        ),
    )
    p.add_argument(
        "--regional-feature-output",
        type=Path,
        default=None,
        help=(
            "Regional single-cell mode: write the exact hourly per-PFT MLP input "
            "features used during replay to this NetCDF file."
        ),
    )
    p.add_argument(
        "--regional-physics-step-hours",
        type=float,
        default=1.0,
        help=(
            "Regional mode: coupling interval for soil temperature and hydrology in hours. "
            "Canopy/MLP still run hourly; surface targets are averaged over this interval."
        ),
    )
    p.add_argument(
        "--regional-lai-scale",
        type=float,
        default=1.0,
        help="Regional mode: multiply LAI fed to the canopy/MLP replay by this factor.",
    )
    p.add_argument(
        "--regional-lai-add",
        type=float,
        default=0.0,
        help=(
            "Regional mode: add this many LAI units to prescribed exposed LAI. "
            "Use --regional-lai-add-months to restrict the perturbation to selected months."
        ),
    )
    p.add_argument(
        "--regional-lai-add-months",
        nargs="+",
        type=int,
        default=None,
        help=(
            "Regional mode: month numbers where --regional-lai-add is applied. "
            "Omit to apply the additive perturbation in every month."
        ),
    )
    p.add_argument(
        "--regional-lai-add-include-bare",
        action="store_true",
        help=(
            "Regional mode: also apply --regional-lai-add to PFT 0. By default, "
            "PFT 0 bare/unsupported area is left unperturbed."
        ),
    )
    p.add_argument(
        "--regional-sai-scale",
        type=float,
        default=1.0,
        help="Regional mode: multiply SAI fed to the canopy/MLP replay by this factor.",
    )
    p.add_argument(
        "--regional-precip-treatment-area-ha",
        type=float,
        default=0.0,
        help=(
            "Regional clone-grid mode: apply a centered square precipitation "
            "treatment with this area in hectares. Edge cells use fractional "
            "overlap so the effective treated area is exact."
        ),
    )
    p.add_argument(
        "--regional-precip-treatment-reduction",
        type=float,
        default=0.0,
        help=(
            "Regional clone-grid mode: fractional precipitation reduction inside "
            "the treatment area; 0.30 means 30 percent less precipitation."
        ),
    )
    p.add_argument(
        "--regional-precip-treatment-cell-size-m",
        type=float,
        default=20.0,
        help="Regional clone-grid precipitation treatment cell size in meters.",
    )
    p.add_argument(
        "--regional-precip-treatment-center-x-m",
        type=float,
        default=0.0,
        help="Regional clone-grid precipitation treatment center east coordinate in meters.",
    )
    p.add_argument(
        "--regional-precip-treatment-center-y-m",
        type=float,
        default=0.0,
        help="Regional clone-grid precipitation treatment center north coordinate in meters.",
    )
    p.add_argument(
        "--regional-precip-treatment-start-date",
        type=str,
        default=None,
        help=(
            "Regional clone-grid precipitation treatment active-window start date "
            "as YYYYMMDD. Omit to activate from the beginning of the run."
        ),
    )
    p.add_argument(
        "--regional-precip-treatment-end-date",
        type=str,
        default=None,
        help=(
            "Regional clone-grid precipitation treatment active-window end date "
            "as YYYYMMDD, inclusive. Omit to activate through the end of the run."
        ),
    )
    p.add_argument(
        "--regional-soil-temp-override-elm-history",
        type=Path,
        default=None,
        help=(
            "Regional mode: clamp the replay t_soisno state to lag-aligned ELM "
            "daily TSOI from this h0 file or case run directory. Daily values are "
            "held constant through each replay day."
        ),
    )
    p.add_argument(
        "--regional-initial-soil-temp-elm-history",
        type=Path,
        default=None,
        help=(
            "Regional mode: initialize replay t_soisno once from ELM TSOI "
            "record 0 in this h0 file or case run directory. Temperatures are "
            "prognostic after initialization."
        ),
    )
    p.add_argument(
        "--regional-soil-temp-override-nlev",
        type=int,
        choices=range(1, 16),
        default=15,
        metavar="N",
        help=(
            "Number of leading TSOI levels to clamp when "
            "--regional-soil-temp-override-elm-history is used. Use 1 to force "
            "only level 1 and leave deeper soil temperatures prognostic."
        ),
    )
    p.add_argument(
        "--regional-ground-temp-override-elm-history",
        type=Path,
        default=None,
        help=(
            "Diagnostic regional mode: replace the MLP t_grnd input with "
            "lag-aligned daily ELM TG from this h0 file or case run directory. "
            "Daily values are held constant through each replay day."
        ),
    )
    p.add_argument(
        "--regional-soil-moisture-override-elm-history",
        type=Path,
        default=None,
        help=(
            "Diagnostic regional mode: clamp replay h2osoi_liq and h2osoi_ice "
            "to lag-aligned daily ELM SOILLIQ and SOILICE from this h0 file or "
            "case run directory. Daily values are held constant through each "
            "replay day."
        ),
    )
    p.add_argument(
        "--regional-soil-moisture-override-nlev",
        type=int,
        choices=range(1, 16),
        default=15,
        metavar="N",
        help=(
            "Number of leading SOILLIQ/SOILICE levels to clamp when "
            "--regional-soil-moisture-override-elm-history is used."
        ),
    )
    p.add_argument(
        "--regional-ground-heat-override-elm-history",
        type=Path,
        default=None,
        help=(
            "Diagnostic regional mode: replace learned ground heat boundary "
            "target(s) with lag-aligned daily values from this ELM h0 file or "
            "case run directory."
        ),
    )
    p.add_argument(
        "--regional-ground-heat-override-var",
        type=str,
        default="FGR",
        help="ELM h0 scalar variable to use for --regional-ground-heat-override-elm-history.",
    )
    p.add_argument(
        "--regional-ground-heat-override-targets",
        nargs="+",
        choices=("hs_soil", "hs_top_snow", "hs_h2osfc"),
        default=("hs_soil", "hs_top_snow"),
        help=(
            "Learned thermal target names to replace with the ELM history "
            "ground-heat override. Default: hs_soil hs_top_snow."
        ),
    )
    p.add_argument(
        "--regional-zwt-override-elm-history",
        type=Path,
        default=None,
        help=(
            "Diagnostic regional mode: clamp the replay water-table depth state "
            "to lag-aligned daily ZWT from this ELM h0 file or case run directory. "
            "Daily values are held constant through each replay day."
        ),
    )
    p.add_argument(
        "--regional-zwt-override-var",
        type=str,
        default="ZWT",
        help=(
            "ELM h0 scalar variable to use for --regional-zwt-override-elm-history. "
            "ZWT0 is also accepted as an alias."
        ),
    )
    p.add_argument(
        "--regional-safety-clamps",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Regional mode: clamp replay-only state temperatures and "
            "photosynthesis predictions to prevent a single runaway cell from "
            "poisoning later timesteps."
        ),
    )
    p.add_argument(
        "--regional-temperature-min-k",
        type=float,
        default=180.0,
        help="Regional safety clamp lower bound for dynamic temperature states [K].",
    )
    p.add_argument(
        "--regional-temperature-max-k",
        type=float,
        default=330.0,
        help="Regional safety clamp upper bound for dynamic temperature states [K].",
    )
    p.add_argument(
        "--regional-photosynthesis-max",
        type=float,
        default=100.0,
        help="Regional safety clamp upper bound for psn/fpsn targets [umol m-2 s-1].",
    )
    p.add_argument(
        "--regional-thermal-target-z-clamp",
        type=float,
        default=3.0,
        help=(
            "Regional safety clamp for learned thermal boundary targets "
            "(hs_soil, hs_top_snow, hs_h2osfc), expressed as training "
            "standard deviations around the model target mean. Use <=0 to disable."
        ),
    )
    p.add_argument(
        "--soil-energy-sabg-source",
        choices=("learned", "diagnostic"),
        default="learned",
        help=(
            "For primitive energy-closure checkpoints, use either the MLP-predicted "
            "sabg_soil target or Python's native-like two-stream absorbed-ground "
            "shortwave diagnostic. This supports an otherwise identical A/B replay."
        ),
    )
    p.add_argument(
        "--regional-ood-stride",
        type=int,
        default=0,
        help=(
            "Regional mode: sample MLP input out-of-distribution diagnostics every "
            "N hours and plot max |z| / fraction beyond threshold.  Use 0 to disable."
        ),
    )
    p.add_argument(
        "--regional-ood-threshold",
        type=float,
        default=4.0,
        help="Regional mode: standardized-input threshold for MLP OOD fraction maps.",
    )
    p.add_argument(
        "--regional-ood-feature-maps",
        default="none",
        help=(
            "Regional mode: comma-separated MLP input names for additional mean |z| "
            "diagnostic maps.  Use 'all' for every MLP input or 'none' to disable "
            "these feature-specific maps."
        ),
    )
    p.add_argument(
        "--regional-daily-output",
        type=Path,
        default=None,
        help=(
            "Regional mode: write one NetCDF file with diagnostic means for the "
            "same base variables used in the regional map plots. The file is "
            "overwritten at the start of the run and appended across months."
        ),
    )
    p.add_argument(
        "--regional-output-frequency",
        choices=("daily", "5day", "monthly"),
        default="daily",
        help=(
            "Frequency for --regional-daily-output records. '5day' writes "
            "5-day period means with a final partial period as needed; "
            "'monthly' writes one time record per simulated month."
        ),
    )
    p.add_argument(
        "--elm-history-output-dir",
        "--elm-history-run-dir",
        dest="elm_history_output_dir",
        type=Path,
        default=None,
        help=(
            "Regional mode: write ELM-style .elm.h0.<year>-01-01-00000.nc "
            "history files into this run directory. This is intended for "
            "elm-olmt integration and comparison against native ELM h0 files."
        ),
    )
    p.add_argument(
        "--elm-history-case",
        default=None,
        help=(
            "Case name used in ELM-style h0 filenames. Default: infer from "
            "--elm-history-output-dir, using the parent directory when the "
            "output directory is named 'run'."
        ),
    )
    p.add_argument(
        "--elm-history-frequency",
        choices=("hourly", "daily", "5day", "monthly"),
        default="monthly",
        help=(
            "Averaging frequency for ELM-style h0 output. Monthly is the most "
            "compact; hourly writes one record per replay hour; daily and 5day "
            "use the daily diagnostic path."
        ),
    )
    p.add_argument(
        "--regional-day-night-flux-output",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Regional mode: when writing daily or 5-day output, also write "
            "daytime and nighttime GPP, ET, and sensible heat means split by FSDS."
        ),
    )
    p.add_argument(
        "--regional-day-night-fsds-threshold",
        type=float,
        default=20.0,
        help="FSDS threshold in W m-2 used to split day and night regional flux output.",
    )
    p.add_argument(
        "--regional-plot-dir",
        type=Path,
        default=Path("plots"),
        help="Directory for regional map PNG output.",
    )
    p.add_argument(
        "--regional-plot-variable",
        nargs="+",
        default=None,
        help=(
            "Regional map variables to plot. Omit to plot every available map; "
            "for this experiment useful values include gpp_gc_m2_d_mean, "
            "eflx_lh_tot_mean, eflx_sh_tot_mean, and qflx_runoff_tot_mean."
        ),
    )
    p.add_argument(
        "--regional-restart-in",
        type=Path,
        default=None,
        help=(
            "Regional mode: initialize the coupled physical replay state from this "
            "restart NetCDF file instead of the cold-start state."
        ),
    )
    p.add_argument(
        "--regional-restart-out",
        type=Path,
        default=None,
        help="Regional mode: write the final coupled physical replay state to this restart NetCDF file.",
    )
    p.add_argument(
        "--co2-ppm",
        type=float,
        default=None,
        help=(
            "Scalar atmospheric CO2 concentration in ppm, used only when no "
            "--co2-file/default CO2 stream is available."
        ),
    )
    p.add_argument(
        "--co2-file",
        type=Path,
        default=None,
        help=(
            "NetCDF CO2 stream file with a CO2 variable in ppmv. Default: infer "
            "the OLMT/E3SM cpl-bypass CO2 file from the inputdata root containing --met-dir."
        ),
    )
    p.add_argument(
        "--online-carbon",
        action="store_true",
        help="Regional mode: advance the ELM carbon pools once per day and feed carbon-derived exposed LAI/SAI back to the hourly surface model.",
    )
    p.add_argument(
        "--carbon-som-init",
        type=float,
        default=5000.0,
        help="Regional online carbon: initial total SOM [gC m-2].",
    )
    p.add_argument(
        "--carbon-leafc-init",
        type=float,
        default=10.0,
        help="Regional online carbon: initial evergreen active leaf carbon [gC m-2].",
    )
    p.add_argument(
        "--carbon-decid-leafc-storage-init",
        type=float,
        default=50.0,
        help="Regional online carbon: initial deciduous leaf carbon storage [gC m-2].",
    )
    p.add_argument(
        "--carbon-accel",
        type=float,
        default=1.0,
        help="Regional online carbon: acceleration factor passed to the daily carbon step.",
    )
    p.add_argument(
        "--carbon-accel-mode",
        choices=("constant", "elm-native", "native"),
        default="constant",
        help=(
            "Regional online carbon: 'constant' preserves --carbon-accel behavior; "
            "'elm-native' uses native ELM DecompCascadeCN pool-specific accelerated "
            "decomposition factors. The year-40 scalar-average correction is not "
            "applied in this emulator mode."
        ),
    )
    args = p.parse_args()
    if args.regional and ensemble_size_given and not regional_ensemble_size_given:
        args.regional_ensemble_size = (
            int(args.ensemble_size)
        )
        print(
            f"  [args] using --ensemble-size={args.ensemble_size} as "
            f"--regional-ensemble-size={args.regional_ensemble_size}"
        )
    if args.regional and (
        args.regional_ensemble_flnr_factors is not None
        or args.regional_ensemble_mbbopt_factors is not None
    ):
        flnr_values = (
            [1.0]
            if args.regional_ensemble_flnr_factors is None
            else [float(v) for v in args.regional_ensemble_flnr_factors]
        )
        mbbopt_values = (
            [1.0]
            if args.regional_ensemble_mbbopt_factors is None
            else [float(v) for v in args.regional_ensemble_mbbopt_factors]
        )
        bad_values = [
            v for v in (*flnr_values, *mbbopt_values)
            if not np.isfinite(v) or v <= 0.0
        ]
        if bad_values:
            raise SystemExit("regional ensemble trait factors must be finite positive values")
        flnr_grid = [f for f in flnr_values for _ in mbbopt_values]
        mbbopt_grid = [m for _ in flnr_values for m in mbbopt_values]
        grid_size = len(flnr_grid)
        if regional_ensemble_size_given and int(args.regional_ensemble_size) != grid_size:
            print(
                "  [args] overriding --regional-ensemble-size with explicit "
                f"trait-factor grid size {grid_size}"
            )
        args.regional_ensemble_size = grid_size
        args.regional_ensemble_flnr_factors = flnr_grid
        args.regional_ensemble_mbbopt_factors = mbbopt_grid
        print(
            "  [args] regional ensemble trait-factor grid: "
            f"{len(flnr_values)} flnr x {len(mbbopt_values)} mbbopt = {grid_size} members"
        )
    if args.universal and not args.regional:
        raise SystemExit("--universal is currently supported only with --regional")
    if args.elm_history_output_dir is not None and not args.regional:
        raise SystemExit("--elm-history-output-dir is currently supported only with --regional")
    if args.elm_history_output_dir is not None and args.elm_history_case is None:
        args.elm_history_case = infer_elm_history_case_name(args.elm_history_output_dir)
    if args.regional_met_cycle_years < 0:
        raise SystemExit("--regional-met-cycle-years must be non-negative")
    if args.regional_met_cycle_years > 0 and args.regional_met_cycle_start_year is None:
        args.regional_met_cycle_start_year = int(args.year)
    if args.regional_cell_shards <= 0:
        raise SystemExit("--regional-cell-shards must be positive")
    if args.regional_cell_shard_index < 0:
        raise SystemExit("--regional-cell-shard-index must be non-negative")
    if args.regional_cell_shard_index >= args.regional_cell_shards:
        raise SystemExit("--regional-cell-shard-index must be less than --regional-cell-shards")
    if (
        args.regional_min_lat is not None
        and args.regional_max_lat is not None
        and args.regional_min_lat > args.regional_max_lat
    ):
        raise SystemExit("--regional-min-lat must be <= --regional-max-lat")
    if args.regional_precip_treatment_area_ha < 0.0:
        raise SystemExit("--regional-precip-treatment-area-ha must be non-negative")
    if args.regional_precip_treatment_reduction < 0.0 or args.regional_precip_treatment_reduction > 1.0:
        raise SystemExit("--regional-precip-treatment-reduction must be in [0, 1]")
    if args.regional_precip_treatment_cell_size_m <= 0.0:
        raise SystemExit("--regional-precip-treatment-cell-size-m must be positive")
    try:
        args.regional_precip_treatment_start_date = _parse_optional_yyyymmdd(
            args.regional_precip_treatment_start_date,
            name="--regional-precip-treatment-start-date",
        )
        args.regional_precip_treatment_end_date = _parse_optional_yyyymmdd(
            args.regional_precip_treatment_end_date,
            name="--regional-precip-treatment-end-date",
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if (
        args.regional_precip_treatment_start_date is not None
        and args.regional_precip_treatment_end_date is not None
        and args.regional_precip_treatment_end_date < args.regional_precip_treatment_start_date
    ):
        raise SystemExit("--regional-precip-treatment-end-date must be >= start date")
    if (args.regional_clone_grid_rows is None) != (args.regional_clone_grid_cols is None):
        raise SystemExit(
            "--regional-clone-grid-rows and --regional-clone-grid-cols must be supplied together"
        )
    if args.regional_clone_grid_rows is not None:
        if args.regional_clone_grid_rows <= 0 or args.regional_clone_grid_cols <= 0:
            raise SystemExit("--regional-clone-grid-rows/cols must be positive")
        if args.lat is None or args.lon is None:
            raise SystemExit("--regional clone-grid mode requires --lat and --lon")
        if args.regional_cell_shards > 1:
            raise SystemExit("Synthetic clone-grid mode does not support --regional-cell-shards > 1")
    args.regional_min_lon_given = regional_min_lon_given
    return args




# ---------------------------------------------------------------------------
# Regional benchmark driver
# ---------------------------------------------------------------------------

def _run_regional_benchmark(
    args,
    device: "torch.device",
    dtype: "torch.dtype",
    mlp_dtype: "torch.dtype",
) -> None:
    """End-to-end regional benchmark driver (called from main when --regional)."""
    snow_model_enabled = not args.disable_snow_model
    regional_met_format = _detect_regional_met_format(args.met_dir, args.regional_met_format)
    if regional_met_format == "site" and args.regional_calendar != "noleap":
        print("  [SITE met] switching regional calendar to noleap")
        args.regional_calendar = "noleap"
    if regional_met_format == "trendy" and args.regional_calendar != "noleap":
        print("  [TRENDY met] switching regional calendar to noleap")
        args.regional_calendar = "noleap"
    if (
        regional_met_format == "trendy"
        and not getattr(args, "regional_min_lon_given", False)
        and float(args.regional_min_lon) == -95.0
    ):
        print("  [TRENDY met] using full longitude range (--regional-min-lon -180)")
        args.regional_min_lon = -180.0
    if regional_met_format == "trendy":
        met_input_step_hours = 6
    elif regional_met_format == "site":
        met_input_step_hours = 1
    else:
        met_input_step_hours = 3
    met_padding_steps = 1 if regional_met_format == "trendy" and met_input_step_hours > 1 else 0
    elm_stream_met_methods = regional_met_format == "trendy"

    # ------------------------------------------------------------------
    # 1. Load regional surfdata (active cells only).
    # ------------------------------------------------------------------
    print(f"Loading regional surfdata from {args.regional_surfdata} …")
    surf = load_regional_surfdata(
        args.regional_surfdata,
        met_dir=args.met_dir,
        min_lon_deg=args.regional_min_lon,
        max_lon_deg=args.regional_max_lon,
        min_lat_deg=args.regional_min_lat,
        max_lat_deg=args.regional_max_lat,
        met_format=regional_met_format,
    )
    ncell = surf["lat_deg"].shape[0]
    print(f"  Active cells: {ncell:,}")
    single_cell_mode = args.lat is not None or args.lon is not None
    selected_cell_info: dict[str, float | int] | None = None
    if (args.lat is None) != (args.lon is None):
        raise SystemExit("--lat and --lon must be supplied together")
    if single_cell_mode:
        cell_idx, distance_km = _nearest_regional_cell_index(surf, float(args.lat), float(args.lon))
        selected_cell_info = {
            "original_index": int(cell_idx),
            "active_idx": int(np.asarray(surf["active_idx"])[cell_idx]),
            "lat": float(np.asarray(surf["lat_deg"])[cell_idx]),
            "lon": float(np.asarray(surf["lon_deg"])[cell_idx]),
            "grid_row": int(np.asarray(surf["grid_row"])[cell_idx]),
            "grid_col": int(np.asarray(surf["grid_col"])[cell_idx]),
            "distance_km": float(distance_km),
        }
        print(
            "  Single-cell regional run: requested "
            f"lat={float(args.lat):.5f} lon={float(args.lon):.5f}; "
            f"using active cell {selected_cell_info['active_idx']} at "
            f"lat={selected_cell_info['lat']:.5f} lon={selected_cell_info['lon']:.5f} "
            f"({selected_cell_info['distance_km']:.2f} km away)"
        )
        surf = _subset_regional_surf_cells(surf, [cell_idx])
        ncell = 1
        if args.regional_clone_grid_rows is not None:
            surf = _clone_single_regional_cell_to_synthetic_grid(
                surf,
                nrow=int(args.regional_clone_grid_rows),
                ncol=int(args.regional_clone_grid_cols),
                corr_length_cells=float(args.regional_clone_corr_length_cells),
                flnr_relative_sd=float(args.regional_clone_flnr_relative_sd),
                mbbopt_relative_sd=float(args.regional_clone_mbbopt_relative_sd),
                sand_absolute_sd=float(args.regional_clone_sand_absolute_sd),
                trait_min_factor=float(args.regional_clone_trait_min_factor),
                trait_max_factor=float(args.regional_clone_trait_max_factor),
                sand_min_pct=float(args.regional_clone_sand_min_pct),
                sand_max_pct=float(args.regional_clone_sand_max_pct),
                grid_spacing_km=float(args.regional_clone_grid_spacing_km),
                seed=int(args.regional_clone_seed),
            )
            ncell = int(np.asarray(surf["lat_deg"]).size)
            single_cell_mode = False
            print(
                "  Synthetic clone grid: "
                f"{int(args.regional_clone_grid_rows)}x{int(args.regional_clone_grid_cols)} "
                f"({ncell:,} cells), forcing row={selected_cell_info['grid_row']} "
                f"col={selected_cell_info['grid_col']}, corr_length="
                f"{float(args.regional_clone_corr_length_cells):g} cells"
            )
            for _name in (
                "synthetic_flnr_factor",
                "synthetic_mbbopt_factor",
                "synthetic_pct_sand_delta",
                "synthetic_pct_sand_lev1",
            ):
                _arr = np.asarray(surf[_name], dtype=np.float64)
                print(
                    f"    {_name}: mean={float(np.nanmean(_arr)):.4g} "
                    f"sd={float(np.nanstd(_arr)):.4g} "
                    f"min={float(np.nanmin(_arr)):.4g} "
                    f"max={float(np.nanmax(_arr)):.4g}"
                )
    elif args.regional_ncell is not None:
        regional_ncell = int(args.regional_ncell)
        if regional_ncell <= 0:
            raise SystemExit("--regional-ncell must be positive")
        if regional_ncell < ncell:
            surf = _subset_regional_surf_cells(surf, np.arange(regional_ncell, dtype=np.int64))
            ncell = regional_ncell
            print(f"  Regional subdomain: first {ncell:,} active cells after screening")
        else:
            print(
                f"  Regional subdomain request {regional_ncell:,} >= active cells; using all {ncell:,}"
            )
    if not single_cell_mode and int(args.regional_cell_shards) > 1:
        shard_count = int(args.regional_cell_shards)
        shard_index = int(args.regional_cell_shard_index)
        start = (ncell * shard_index) // shard_count
        stop = (ncell * (shard_index + 1)) // shard_count
        if stop <= start:
            raise SystemExit(
                f"--regional-cell-shard-index {shard_index} has no cells "
                f"for ncell={ncell} and --regional-cell-shards={shard_count}"
            )
        surf = _subset_regional_surf_cells(
            surf,
            np.arange(start, stop, dtype=np.int64),
        )
        ncell = stop - start
        print(
            f"  Regional cell shard: {shard_index + 1}/{shard_count} "
            f"active-cell slice [{start:,}, {stop:,}) -> {ncell:,} cells"
        )

    if float(args.regional_precip_treatment_area_ha) > 0.0:
        surf = _add_center_square_precip_treatment(
            surf,
            area_ha=float(args.regional_precip_treatment_area_ha),
            reduction=float(args.regional_precip_treatment_reduction),
            cell_size_m=float(args.regional_precip_treatment_cell_size_m),
            center_x_m=float(args.regional_precip_treatment_center_x_m),
            center_y_m=float(args.regional_precip_treatment_center_y_m),
        )
        surf["precip_treatment_start_date"] = (
            0
            if args.regional_precip_treatment_start_date is None
            else int(args.regional_precip_treatment_start_date)
        )
        surf["precip_treatment_end_date"] = (
            0
            if args.regional_precip_treatment_end_date is None
            else int(args.regional_precip_treatment_end_date)
        )
        _window_label = "entire run"
        if args.regional_precip_treatment_start_date is not None or args.regional_precip_treatment_end_date is not None:
            _window_label = (
                f"{args.regional_precip_treatment_start_date or 'run-start'}.."
                f"{args.regional_precip_treatment_end_date or 'run-end'}"
            )
        print(
            "  Precipitation treatment ROI: "
            f"{float(surf['precip_treatment_reduction']) * 100.0:.1f}% reduction, "
            f"requested={float(surf['precip_treatment_area_ha_requested']):.3f} ha, "
            f"effective={float(surf['precip_treatment_area_ha_effective']):.3f} ha, "
            f"side={float(surf['precip_treatment_side_m']):.1f} m, "
            f"cells={int(surf['precip_treatment_cell_count_any_overlap'])} "
            f"({float(surf['precip_treatment_cell_equivalent_count']):.1f} cell-equivalent), "
            f"active={_window_label}"
        )

    # ------------------------------------------------------------------
    # 2. Determine which PFTs have non-zero area across any active cell.
    # ------------------------------------------------------------------
    pct = np.asarray(surf["pct_nat_pft"], dtype=np.float64)   # (17, ncell)
    excluded_pfts = sorted(set(int(pft) for pft in args.regional_exclude_pft))
    if excluded_pfts:
        invalid = [pft for pft in excluded_pfts if pft < 0 or pft >= pct.shape[0]]
        if invalid:
            raise SystemExit(
                f"--regional-exclude-pft values must be in [0, {pct.shape[0] - 1}], got {invalid}"
            )
        if args.pft is not None and int(args.pft) in excluded_pfts:
            raise SystemExit("--pft cannot also be listed in --regional-exclude-pft")
        original_total = max(float(pct.sum()), 1.0e-12)
        excluded_area_pct = 100.0 * float(pct[excluded_pfts, :].sum()) / original_total
        pct = pct.copy()
        pct[excluded_pfts, :] = 0.0
        keep_cells = pct.sum(axis=0) > 1.0e-12
        removed_cells = int(np.count_nonzero(~keep_cells))
        if not np.any(keep_cells):
            raise SystemExit(
                f"Excluding PFTs {excluded_pfts} removes every selected regional cell"
            )
        if removed_cells:
            keep_idx = np.where(keep_cells)[0]
            surf = _subset_regional_surf_cells(surf, keep_idx)
            pct = pct[:, keep_idx]
            ncell = int(keep_idx.size)
        surf["pct_nat_pft"] = pct.astype(np.float32)
        print(
            f"  Excluding PFTs {excluded_pfts}: removed {excluded_area_pct:.2f}% of "
            f"original PFT area and {removed_cells:,} cells with no remaining PFT; "
            f"renormalizing {ncell:,} retained cells",
            flush=True,
        )
    pft_min_percent = float(args.regional_pft_min_percent)
    if pft_min_percent < 0.0:
        raise SystemExit("--regional-pft-min-percent must be nonnegative")
    if pft_min_percent > 0.0 and args.pft is None:
        pct_original = pct
        pct = pct_original.copy()
        drop = (pct > 0.0) & (pct < pft_min_percent)
        pct[drop] = 0.0
        empty = pct.sum(axis=0) <= 1.0e-12
        if np.any(empty):
            # Never erase a cell completely, even for an aggressive threshold.
            empty_idx = np.where(empty)[0]
            dominant = np.argmax(pct_original[:, empty_idx], axis=0)
            pct[dominant, empty_idx] = pct_original[dominant, empty_idx]
            drop[dominant, empty_idx] = False
        removed_pairs = int(np.count_nonzero(drop))
        original_pairs = int(np.count_nonzero(pct_original > 0.0))
        removed_area_pct = 100.0 * float(pct_original[drop].sum()) / max(
            float(pct_original.sum()), 1.0e-12
        )
        affected_cells = int(np.count_nonzero(np.any(drop, axis=0)))
        surf["pct_nat_pft"] = pct.astype(np.float32)
        print(
            f"  PFT fraction pruning < {pft_min_percent:g}%: removed "
            f"{removed_pairs:,}/{original_pairs:,} PFT-cell pairs "
            f"({100.0 * removed_pairs / max(original_pairs, 1):.2f}%), "
            f"redistributing {removed_area_pct:.3f}% of domain area across retained PFTs "
            f"in {affected_cells:,} cells",
            flush=True,
        )
    natural_pft_present = sorted(int(p) for p in np.where(np.any(pct > 0.0, axis=1))[0])
    print(f"  PFTs present: {natural_pft_present}")
    if args.pft is not None:
        forced_pft = int(args.pft)
        if forced_pft < 0 or forced_pft >= pct.shape[0]:
            raise SystemExit(f"--pft must be in [0, {pct.shape[0] - 1}], got {forced_pft}")
        natural_frac = np.asarray(pct[forced_pft, :], dtype=np.float64)
        print(
            f"  Forcing PFT {forced_pft} to 100% area weight "
            f"(natural fraction range {natural_frac.min():.3g}-{natural_frac.max():.3g}%)"
        )
        if not np.any(natural_frac > 0.0):
            print(f"  WARNING: PFT {forced_pft} has zero natural area in the selected active cells")
        pft_present = [forced_pft]
        pft_weights = np.ones((1, ncell), dtype=np.float64)
    else:
        pft_present = natural_pft_present

        # Per-cell normalised area weights, shape (npft, ncell).
        pct_active = pct[pft_present, :]  # (npft, ncell)
        row_sum = np.maximum(pct_active.sum(axis=0, keepdims=True), 1.0e-12)
        pft_weights = pct_active / row_sum   # (npft, ncell)

    # ------------------------------------------------------------------
    # 3. Load CLM parameters: rootfr, smpso, smpsc, and PFT traits.
    # ------------------------------------------------------------------
    print(f"Loading CLM params from {args.clm_params} …")
    with Dataset(args.clm_params) as ds:
        roota_all  = np.asarray(ds.variables["roota_par"][:], dtype=np.float64)
        rootb_all  = np.asarray(ds.variables["rootb_par"][:], dtype=np.float64)
        smpso_all  = np.asarray(ds.variables["smpso"][:],    dtype=np.float64)
        smpsc_all  = np.asarray(ds.variables["smpsc"][:],    dtype=np.float64)
        flnr_all   = np.asarray(ds.variables["flnr"][:],     dtype=np.float64)
        mbbopt_all = np.asarray(ds.variables["mbbopt"][:],   dtype=np.float64)
        leafcn_all = np.asarray(ds.variables["leafcn"][:],   dtype=np.float64)
        slatop_all = np.asarray(ds.variables["slatop"][:],   dtype=np.float64)
        z0mr_all   = np.asarray(ds.variables["z0mr"][:],     dtype=np.float64)
        rholvis_all = np.asarray(ds.variables["rholvis"][:], dtype=np.float64)
        rholnir_all = np.asarray(ds.variables["rholnir"][:], dtype=np.float64)
        taulvis_all = np.asarray(ds.variables["taulvis"][:], dtype=np.float64)
        taulnir_all = np.asarray(ds.variables["taulnir"][:], dtype=np.float64)
        rhosvis_all = np.asarray(ds.variables["rhosvis"][:], dtype=np.float64)
        rhosnir_all = np.asarray(ds.variables["rhosnir"][:], dtype=np.float64)
        tausvis_all = np.asarray(ds.variables["tausvis"][:], dtype=np.float64)
        tausnir_all = np.asarray(ds.variables["tausnir"][:], dtype=np.float64)
        xl_all      = np.asarray(ds.variables["xl"][:],      dtype=np.float64)
        rsub_top_globalmax = (
            float(np.asarray(ds.variables["rsub_top_globalmax"][:]).reshape(-1)[0])
            if "rsub_top_globalmax" in ds.variables else 10.0
        )
    valid_flnr_mask = np.isfinite(flnr_all) & (flnr_all > 0.0)
    if valid_flnr_mask.size > 0:
        valid_flnr_mask[0] = False
    if valid_flnr_mask.size > 14:
        valid_flnr_mask[14] = False
    neutral_patch0_flnr = float(np.min(flnr_all[valid_flnr_mask])) if np.any(valid_flnr_mask) else None

    # Soil interface depths for Zeng rootfr computation.
    dz_h, _, zi_h = elm_soil_grid(nlevsoi=10)
    zi_interfaces = np.concatenate([[0.0], zi_h])  # length 11

    # ------------------------------------------------------------------
    # 4. Load model bundles and build per-PFT static parameters.
    # ------------------------------------------------------------------
    pft_bundles: list = []
    pft_statics: list = []

    universal_bundle = None  # loaded on demand from patch_-1
    fallback_bundle = None  # loaded on demand from patch_1
    if args.universal:
        universal_file = args.model_dir / "patch_-1" / "model.pt"
        if not universal_file.exists():
            raise FileNotFoundError(
                f"--universal requested, but {universal_file} does not exist. "
                "Train with train_patch_models.py --universal-model first."
            )
        print(f"  Universal replay: loading shared model from {universal_file} …")
        universal_bundle = load_surface_model_bundle(
            universal_file, device=device, dtype=mlp_dtype, use_compile=args.compile
        )

    for pft_id in pft_present:
        model_file = args.model_dir / f"patch_{pft_id}" / "model.pt"
        if args.universal:
            if int(pft_id) in (0, 14) and not args.universal_all_pfts:
                if not model_file.exists():
                    raise FileNotFoundError(
                        f"--universal now uses the individual PFT {int(pft_id)} model, "
                        f"but {model_file} does not exist. Train patch {int(pft_id)} first."
                    )
                print(f"  Universal replay: loading individual model for PFT {int(pft_id)} from {model_file} …")
                b = load_surface_model_bundle(
                    model_file, device=device, dtype=mlp_dtype, use_compile=args.compile
                )
            else:
                b = universal_bundle
        elif model_file.exists():
            print(f"  Loading model for PFT {pft_id} from {model_file} …")
            b = load_surface_model_bundle(model_file, device=device, dtype=mlp_dtype, use_compile=args.compile)
        else:
            universal_file = args.model_dir / "patch_-1" / "model.pt"
            if universal_file.exists():
                if universal_bundle is None:
                    print(f"  no model for PFT {pft_id} — loading universal patch_-1 model")
                    universal_bundle = load_surface_model_bundle(
                        universal_file, device=device, dtype=mlp_dtype, use_compile=args.compile
                    )
                else:
                    print(f"  no model for PFT {pft_id} — reusing universal patch_-1 model")
                b = universal_bundle
            elif pft_id in (0, 14):
                raise FileNotFoundError(
                    f"no individual model for PFT {pft_id}, and required universal model "
                    f"{universal_file} does not exist"
                )
            else:
                fallback_file = args.model_dir / "patch_1" / "model.pt"
                if fallback_bundle is None:
                    print(f"  WARNING: no model for PFT {pft_id} — loading patch_1 as fallback")
                    fallback_bundle = load_surface_model_bundle(
                        fallback_file, device=device, dtype=mlp_dtype, use_compile=args.compile
                    )
                else:
                    print(f"  WARNING: no model for PFT {pft_id} — reusing patch_1 fallback")
                b = fallback_bundle
        pft_bundles.append(b)

        root_param_idx = root_parameter_index_for_pft(int(pft_id), len(roota_all))
        rootfr = zeng2001_rootfr(
            float(roota_all[root_param_idx]), float(rootb_all[root_param_idx]), zi_interfaces
        )
        smpso = float(smpso_all[pft_id]) if smpso_all[pft_id] != 0.0 else None
        smpsc = float(smpsc_all[pft_id]) if smpsc_all[pft_id] != 0.0 else None
        p_static = build_regional_static_vectorized(
            surf,
            pft_id=pft_id,
            rootfr_pft=rootfr,
            smpso=smpso,
            smpsc=smpsc,
            flnr=_synthetic_trait_value_for_surf(
                surf,
                "synthetic_flnr_factor",
                float(flnr_all[pft_id]),
            ),
            mbbopt=_synthetic_trait_value_for_surf(
                surf,
                "synthetic_mbbopt_factor",
                float(mbbopt_all[pft_id]),
            ),
            leafcn=float(leafcn_all[pft_id]),
            slatop=float(slatop_all[pft_id]),
            z0mr=float(z0mr_all[pft_id]),
            rholvis=float(rholvis_all[pft_id]),
            rholnir=float(rholnir_all[pft_id]),
            taulvis=float(taulvis_all[pft_id]),
            taulnir=float(taulnir_all[pft_id]),
            rhosvis=float(rhosvis_all[pft_id]),
            rhosnir=float(rhosnir_all[pft_id]),
            tausvis=float(tausvis_all[pft_id]),
            tausnir=float(tausnir_all[pft_id]),
            xl=float(xl_all[pft_id]),
            rsub_top_globalmax=rsub_top_globalmax,
            device=device,
            dtype=dtype,
        )
        pft_statics.append(p_static)

    # Shared static uses the PFT-0 (dominant) soil params for the column physics.
    # (Soil texture already embedded in each pft_static; shared_static = first PFT's.)
    shared_static = pft_statics[0]

    # ------------------------------------------------------------------
    # 5. Optional physics compile.
    # ------------------------------------------------------------------
    if args.compile_physics and hasattr(torch, "compile"):
        from . import elm_hydrology_torch as _hyd_mod
        from . import elm_soil_temperature_torch as _temp_mod
        from . import coupled_surface_hydrology_soiltemp as _coupled_mod
        if device.type == "cuda":
            _hyd_mod.batched_tridiagonal_torch = torch.compile(
                _hyd_mod.batched_tridiagonal_torch, fullgraph=True, dynamic=True
            )
        else:
            print("  [torch.compile] leaving hydrology Thomas solver eager on CPU")
        if snow_model_enabled and not args.snow_scalar_only:
            print(
                "  [torch.compile] leaving soil-temperature Thomas solver eager "
                "because active snow changes the matrix width"
            )
        elif device.type == "cuda":
            _temp_mod._batched_tridiagonal_torch = torch.compile(
                _temp_mod._batched_tridiagonal_torch, fullgraph=True, dynamic=True
            )
        else:
            print("  [torch.compile] leaving soil-temperature Thomas solver eager on CPU")
        _compiled_hydrology_torch = torch.compile(_hyd_mod.hydrology_torch, fullgraph=False)
        _compiled_soil_temperature_step_torch = torch.compile(
            _temp_mod.soil_temperature_step_torch, fullgraph=False
        )
        # coupled_surface_hydrology_soiltemp imports these functions directly,
        # so replacing only the defining module's attribute leaves its aliases
        # eager.  Install the compiled callables at both lookup sites.
        _hyd_mod.hydrology_torch = _compiled_hydrology_torch
        _coupled_mod.hydrology_torch = _compiled_hydrology_torch
        _temp_mod.soil_temperature_step_torch = _compiled_soil_temperature_step_torch
        _coupled_mod.soil_temperature_step_torch = _compiled_soil_temperature_step_torch

    if args.online_carbon and (args.regional_restart_in is not None or args.regional_restart_out is not None):
        raise SystemExit(
            "Regional restart currently covers the coupled physical state only; "
            "--online-carbon restart support has not been implemented yet."
        )

    # ------------------------------------------------------------------
    # 6. Initial coupled physical state.
    # ------------------------------------------------------------------
    nlev_hydro_static = int(shared_static.hydrology_torch.dz.shape[-1])
    state = prepare_regional_initial_state(
        surf=surf,
        pft_ids=pft_present,
        pft_weights=pft_weights,
        ncell=ncell,
        nlev=15,
        nlevbed=nlev_hydro_static,
        nens=args.regional_ensemble_size,
        device=device,
        dtype=dtype,
        restart_in=args.regional_restart_in,
        start_year=args.year,
        start_month=args.month,
        snow_model_enabled=snow_model_enabled,
        snow_scalar_only=args.snow_scalar_only,
    )
    if args.regional_initial_soil_temp_elm_history is not None:
        initial_tsoi = _read_elm_tsoi_history(
            _find_elm_history_file(args.regional_initial_soil_temp_elm_history, args.year),
            ncell,
        )[0]
        state_nlev = int(state.t_soisno.shape[-1])
        if initial_tsoi.shape[-1] < state_nlev:
            initial_tsoi = np.concatenate(
                [
                    initial_tsoi,
                    np.broadcast_to(
                        initial_tsoi[:, -1:],
                        (initial_tsoi.shape[0], state_nlev - initial_tsoi.shape[-1]),
                    ),
                ],
                axis=-1,
            )
        initial_tsoi_t = torch.as_tensor(
            initial_tsoi[:, None, :state_nlev],
            device=device,
            dtype=dtype,
        ).expand(ncell, args.regional_ensemble_size, state_nlev).clone()
        state.t_soisno = initial_tsoi_t
        print(
            "  initialized regional soil temperatures once from ELM TSOI record 0: "
            f"{args.regional_initial_soil_temp_elm_history}"
        )
    carbon_model = None
    if args.online_carbon:
        from .coupled_surface_hydrology_soiltemp import OnlineDailyCarbonModel
        carbon_model = OnlineDailyCarbonModel.from_clm_params(
            clm_params=args.clm_params,
            pft_ids=pft_present,
            pft_weights=pft_weights,
            ncell=ncell,
            nens=args.regional_ensemble_size,
            device=device,
            dtype=dtype,
            som_init=args.carbon_som_init,
            evergreen_leafc_init=args.carbon_leafc_init,
            decid_leafc_storage_init=args.carbon_decid_leafc_storage_init,
            accel_fac=args.carbon_accel,
            accel_mode=args.carbon_accel_mode,
        )
        print(
            "  Online daily carbon enabled: carbon-derived exposed LAI/SAI will feed "
            f"back to surface features; acceleration={carbon_model.accel_summary}"
        )

    # ------------------------------------------------------------------
    # 7. Loop over months.
    # ------------------------------------------------------------------
    print(f"\nRunning {args.nmonths} month(s) starting {args.year}-{args.month:02d}")
    print(
        f"  device={device.type}  dtype={dtype}  mlp_dtype={mlp_dtype}  ncell={ncell:,}"
        f"  npft={len(pft_present)}  nens={args.regional_ensemble_size}"
        f"  ntrait={args.regional_trait_ensemble_size}  "
        f"snow={'scalar-only' if snow_model_enabled and args.snow_scalar_only else snow_model_enabled}  "
        f"soil_energy_sabg={args.soil_energy_sabg_source}"
    )
    print()

    total_elapsed = 0.0
    total_nhours  = 0
    last_n1h = 0
    _all_diag: dict[str, np.ndarray] = {}
    _diag_months = 0
    _prof: dict[str, float] = {}  # profile sections accumulator, cleared each month
    year, month = args.year, args.month
    regional_ts_records: dict[str, list[np.ndarray]] | None = None
    regional_feature_records: dict[str, list[np.ndarray]] | None = None
    regional_step_callback = None
    default_plot_output = Path("plots/benchmark_replay_diagnostics.png")
    regional_cell_plot_output: Path | None = None
    record_single_cell_timeseries = (
        single_cell_mode
        and (
            args.regional_cell_timeseries
            or args.regional_timeseries_output is not None
            or args.regional_feature_output is not None
        )
    )
    if record_single_cell_timeseries:
        if args.regional_cell_timeseries and args.plot_output == default_plot_output and selected_cell_info is not None:
            regional_cell_plot_output = args.regional_plot_dir / (
                f"regional_cell_timeseries_{args.year}_{args.month:02d}_"
                f"lat{selected_cell_info['lat']:.4f}_lon{selected_cell_info['lon']:.4f}.png"
            )
        elif args.regional_cell_timeseries:
            regional_cell_plot_output = args.plot_output
        if args.regional_cell_timeseries:
            if args.no_plot:
                print("  Single-cell mode: writing the regional time-series plot despite --no-plot")
            print(f"  Single-cell time-series plot: {regional_cell_plot_output}")
        if args.regional_timeseries_output is not None:
            print(f"  Single-cell hourly diagnostics: {args.regional_timeseries_output}")
        if args.regional_feature_output is not None:
            print(f"  Single-cell hourly MLP features: {args.regional_feature_output}")
        regional_ts_records = {
            "hour_index": [],
            "year": [],
            "month": [],
            "day": [],
            "hour_of_day": [],
            "physics_step": [],
            "ensemble_flnr_factor": [],
            "ensemble_mbbopt_factor": [],
            "fpsn_patch": [],
            "qflx_tran_veg": [],
            "lai": [],
            "sai": [],
            "zwt": [],
            "h2osfc": [],
            "t_h2osfc": [],
            "btran": [],
            "t_soil": [],
            "t_soil_lev2": [],
            "liq_vol": [],
            "ice_vol": [],
            "frac_h2osfc": [],
            "frac_sno": [],
            "frac_sno_input": [],
            "h2osno": [],
            "int_snow": [],
            "snow_depth": [],
            "snow_depth_eff": [],
            "snow_snl": [],
            "snow_ice_total": [],
            "snow_liq_total": [],
            "snow_t_min": [],
            "snow_grain_radius_top": [],
            "snow_albedo_vis": [],
            "snow_albedo_nir": [],
            "sabg_snow_input": [],
            "forc_t": [],
            "forc_pbot": [],
            "forc_swrad": [],
            "eflx_sh_tot": [],
            "qflx_evap_canopy": [],
            "qflx_ev_soil_mlp_blended": [],
            "t_veg": [],
            "t_ref2m": [],
            "hs_soil": [],
            "hs_top_snow": [],
            "hs_h2osfc": [],
            "dhsdT": [],
            "eflx_fgr12": [],
            "diag_ground_heat_flux_override": [],
            "qflx_prec_grnd": [],
            "qflx_rain_grnd": [],
            "qflx_snow_grnd": [],
            "qflx_ev_soil": [],
            "qflx_ev_h2osfc": [],
            "qflx_ev_snow": [],
            "qflx_snow_melt": [],
            "qflx_snow_freeze": [],
            "qflx_soil_melt": [],
            "qflx_soil_freeze": [],
            "qflx_snow_to_soil": [],
            "qflx_snwcp_ice": [],
            "qflx_snwcp_liq": [],
            "phase_xm_h2osfc": [],
            "phase_temp1_h2osfc": [],
            "fn_h2osfc_old": [],
            "fn_h2osfc_new": [],
            "tk_h2osfc": [],
            "partial_snow_blend_energy_residual_uncorrected": [],
            "partial_snow_blend_energy_residual_corrected": [],
            "partial_snow_blend_water_residual": [],
        }
        for _snow_lev in range(MAX_SNOW_LAYERS):
            regional_ts_records[f"snow_dz_lev{_snow_lev + 1}"] = []
            regional_ts_records[f"snow_ice_lev{_snow_lev + 1}"] = []
            regional_ts_records[f"snow_liq_lev{_snow_lev + 1}"] = []
            regional_ts_records[f"snow_t_lev{_snow_lev + 1}"] = []
            regional_ts_records[f"qflx_snow_melt_lev{_snow_lev + 1}"] = []
            regional_ts_records[f"qflx_snow_freeze_lev{_snow_lev + 1}"] = []
        for _pft_id in pft_present:
            _pft_label = f"pft_{int(_pft_id)}"
            regional_ts_records[f"{_pft_label}_area_weight"] = []
            regional_ts_records[f"{_pft_label}_eflx_sh_tot"] = []
            regional_ts_records[f"{_pft_label}_hs_soil"] = []
            regional_ts_records[f"{_pft_label}_hs_top_snow"] = []
            regional_ts_records[f"{_pft_label}_hs_h2osfc"] = []
            regional_ts_records[f"{_pft_label}_dhsdT"] = []
            regional_ts_records[f"{_pft_label}_qflx_tran_veg"] = []
            regional_ts_records[f"{_pft_label}_qflx_evap_canopy"] = []
            regional_ts_records[f"{_pft_label}_qflx_ev_soil"] = []
            regional_ts_records[f"{_pft_label}_qflx_ev_soil_weighted"] = []
            regional_ts_records[f"{_pft_label}_fpsn_patch"] = []
        if args.regional_feature_output is not None:
            regional_feature_records = {
                "hour_index": [],
                "year": [],
                "month": [],
                "day": [],
                "hour_of_day": [],
            }
        dz_profile_ts = np.asarray(_to_np(shared_static.hydrology_torch.dz), dtype=np.float64)
        soil_profile_nlev_ts = min(10, int(dz_profile_ts.size))
        dz0_ts = float(dz_profile_ts[0])
        physics_cadence_ts = max(1, int(round(float(args.regional_physics_step_hours))))

        def _cell_ens_array(value) -> np.ndarray:
            arr = _to_np(value)
            arr = np.asarray(arr, dtype=np.float64)
            if arr.ndim == 0:
                arr = arr.reshape(1, 1)
            elif arr.ndim == 1:
                arr = arr.reshape(1, -1)
            else:
                arr = arr.reshape(arr.shape[0], -1)
            if arr.shape[0] != 1:
                raise ValueError(f"Expected a single regional cell, got shape {arr.shape}")
            if arr.shape[1] == 1 and args.regional_ensemble_size > 1:
                arr = np.broadcast_to(arr, (1, args.regional_ensemble_size)).copy()
            return arr

        def _record_ts_value(name: str, value, default: float = np.nan) -> None:
            assert regional_ts_records is not None
            if name not in regional_ts_records:
                n_previous = max(0, len(regional_ts_records.get("hour_index", [])) - 1)
                fill = np.full((1, args.regional_ensemble_size), np.nan, dtype=np.float64)
                regional_ts_records[name] = [fill.copy() for _ in range(n_previous)]
            if value is None:
                value = np.full((1, args.regional_ensemble_size), default, dtype=np.float64)
            regional_ts_records[name].append(_cell_ens_array(value))

        def _safe_feature_name(feature_name: str) -> str:
            return "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in str(feature_name))

        def _record_feature_value(name: str, value, default: float = np.nan) -> None:
            assert regional_feature_records is not None
            if name not in regional_feature_records:
                n_previous = max(0, len(regional_feature_records.get("hour_index", [])) - 1)
                fill = np.full((1, args.regional_ensemble_size), np.nan, dtype=np.float64)
                regional_feature_records[name] = [fill.copy() for _ in range(n_previous)]
            if value is None:
                value = np.full((1, args.regional_ensemble_size), default, dtype=np.float64)
            regional_feature_records[name].append(_cell_ens_array(value))

        def _record_feature_snapshot(step_i: int, feature_buffers) -> None:
            if regional_feature_records is None or feature_buffers is None:
                return
            _record_feature_value("hour_index", len(regional_feature_records["hour_index"]))
            _record_feature_value("year", year)
            _record_feature_value("month", month)
            _record_feature_value("day", step_i // 24 + 1)
            _record_feature_value("hour_of_day", step_i % 24)
            for _pft_idx, (_pft_id, _p_bundle) in enumerate(zip(pft_present, pft_bundles)):
                if _pft_idx >= len(feature_buffers):
                    continue
                _feat = feature_buffers[_pft_idx]
                _pft_label = f"pft_{int(_pft_id)}"
                _record_feature_value(f"{_pft_label}_area_weight", pft_weights[_pft_idx])
                for _feature_idx, _feature_name in enumerate(_p_bundle.input_names):
                    _key = f"{_pft_label}_feature_{_safe_feature_name(_feature_name)}"
                    _record_feature_value(_key, _feat[..., _feature_idx])

        def _ood_var_name(feature_name: str) -> str:
            safe = "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in str(feature_name))
            return f"mlp_ood_abs_z_{safe}"

        def _snow_layer_sum(state_step, attr: str):
            snow_layers = state_step.snow_layers
            if snow_layers is None:
                return None
            value = getattr(snow_layers, attr, None)
            if value is None:
                return None
            arr = np.asarray(_to_np(value), dtype=np.float64)
            return np.sum(arr, axis=-1) if arr.ndim >= 2 else arr

        def _snow_layer_min_temp(state_step):
            snow_layers = state_step.snow_layers
            if snow_layers is None:
                return None
            temp = np.asarray(_to_np(snow_layers.t_snow), dtype=np.float64)
            mass = (
                np.asarray(_to_np(snow_layers.h2osoi_ice), dtype=np.float64)
                + np.asarray(_to_np(snow_layers.h2osoi_liq), dtype=np.float64)
            )
            if temp.ndim < 2:
                return None
            masked = np.where(mass > 1.0e-8, temp, np.inf)
            out = np.min(masked, axis=-1)
            return np.where(np.isfinite(out), out, np.nan)

        def _snow_layer_top_grain_radius(state_step):
            snow_layers = state_step.snow_layers
            if snow_layers is None or snow_layers.snw_rds is None:
                return None
            rds = np.asarray(_to_np(snow_layers.snw_rds), dtype=np.float64)
            snl = np.asarray(_to_np(snow_layers.snl), dtype=np.int32)
            if rds.ndim < 2:
                return None
            return np.where(snl < 0, rds[..., 0], np.nan)

        def _target_from(result, name: str):
            return (getattr(result, "surface_targets", None) or {}).get(name)

        def _soil_result_from(result, name: str):
            soil_result = getattr(result, "soil_temperature_result", None)
            return None if soil_result is None else getattr(soil_result, name, None)

        def _snow_water_from(result, name: str):
            snow_water = getattr(result, "snow_water_result", None)
            return None if snow_water is None else getattr(snow_water, name, None)

        def _record_regional_cell(
            step_i,
            last_result,
            blended,
            all_targets_step,
            universal_targets_step,
            tbot,
            fsds,
            dayl,
            ood_max_col=None,
            ood_frac_col=None,
            ood_feature_col=None,
            feature_buffers=None,
        ) -> None:
            assert regional_ts_records is not None
            state_step = last_result.state

            def _pft_target_value(p_idx: int, name: str):
                target = all_targets_step[p_idx] if p_idx < len(all_targets_step) else None
                if target is not None and target.get(name) is not None:
                    return target.get(name)
                if universal_targets_step is not None and name in universal_targets_step:
                    return universal_targets_step[name][p_idx]
                return None

            def _pft_weighted_target_value(p_idx: int, name: str):
                value = _pft_target_value(p_idx, name)
                if value is None:
                    return None
                weight = float(np.asarray(pft_weights[p_idx], dtype=np.float64).reshape(-1)[0])
                return value * weight

            hour_index = len(regional_ts_records["hour_index"])
            _record_ts_value("hour_index", hour_index)
            _record_ts_value("year", year)
            _record_ts_value("month", month)
            _record_ts_value("day", step_i // 24 + 1)
            _record_ts_value("hour_of_day", step_i % 24)
            if args.regional_ensemble_flnr_factors is not None:
                _record_ts_value(
                    "ensemble_flnr_factor",
                    np.asarray(args.regional_ensemble_flnr_factors, dtype=np.float64).reshape(1, -1),
                )
            if args.regional_ensemble_mbbopt_factors is not None:
                _record_ts_value(
                    "ensemble_mbbopt_factor",
                    np.asarray(args.regional_ensemble_mbbopt_factors, dtype=np.float64).reshape(1, -1),
                )
            _record_ts_value(
                "physics_step",
                1
                if (
                    (step_i + 1) % physics_cadence_ts == 0
                    or (step_i + 1)
                    == (max(len(met_dev) - 2 * int(met_padding_steps), 0) * int(met_input_step_hours))
                )
                else 0,
            )
            _record_ts_value("fpsn_patch", blended["fpsn_patch"])
            _record_ts_value("qflx_tran_veg", blended["qflx_tran_veg"])
            _record_ts_value("lai", blended["diag_lai_area_weighted"])
            _record_ts_value("sai", blended["diag_sai_area_weighted"])
            _record_ts_value("zwt", state_step.zwt)
            _record_ts_value("h2osfc", state_step.h2osfc)
            _record_ts_value("t_h2osfc", state_step.t_h2osfc)
            _record_ts_value("btran", blended["diag_btran_input"])
            _record_ts_value("t_soil", state_step.t_soisno[..., 0])
            for _soil_lev in range(soil_profile_nlev_ts):
                _record_ts_value(f"t_soil_lev{_soil_lev + 1}", state_step.t_soisno[..., _soil_lev])
            _record_ts_value("liq_vol", state_step.h2osoi_liq[..., 0] / (dz0_ts * 1000.0))
            for _liq_lev in range(soil_profile_nlev_ts):
                _record_ts_value(
                    f"h2osoi_liqvol_lev{_liq_lev + 1}",
                    state_step.h2osoi_liq[..., _liq_lev] / (dz_profile_ts[_liq_lev] * 1000.0),
                )
                _record_ts_value(f"h2osoi_liq_lev{_liq_lev + 1}", state_step.h2osoi_liq[..., _liq_lev])
            _record_ts_value("ice_vol", state_step.h2osoi_ice[..., 0] / (dz0_ts * 1000.0))
            for _ice_lev in range(soil_profile_nlev_ts):
                _record_ts_value(
                    f"h2osoi_icevol_lev{_ice_lev + 1}",
                    state_step.h2osoi_ice[..., _ice_lev] / (dz_profile_ts[_ice_lev] * 1000.0),
                )
                _record_ts_value(f"h2osoi_ice_lev{_ice_lev + 1}", state_step.h2osoi_ice[..., _ice_lev])
            # Record both surface fractions from the same post-physics state.
            # On a physics hour the feature input above is the pre-step water
            # fraction, while frac_sno below is post-step; mixing those two can
            # falsely report overlap even after native-style reconciliation.
            _record_ts_value("frac_h2osfc", state_step.frac_h2osfc)
            _record_ts_value("frac_sno", state_step.frac_sno)
            _record_ts_value("frac_sno_input", blended.get("diag_frac_sno_input"))
            _record_ts_value("h2osno", state_step.h2osno)
            _record_ts_value("int_snow", getattr(state_step, "int_snow", None))
            _record_ts_value("snow_depth", state_step.snow_depth)
            _record_ts_value(
                "snow_depth_eff",
                None if state_step.frac_sno is None or state_step.snow_depth is None else state_step.frac_sno * state_step.snow_depth,
            )
            _record_ts_value("snow_snl", getattr(state_step.snow_layers, "snl", None) if state_step.snow_layers is not None else None)
            for _snow_lev in range(MAX_SNOW_LAYERS):
                for _field, _prefix in (
                    ("dz", "snow_dz"),
                    ("h2osoi_ice", "snow_ice"),
                    ("h2osoi_liq", "snow_liq"),
                    ("t_snow", "snow_t"),
                ):
                    _layer_values = (
                        None
                        if state_step.snow_layers is None
                        else getattr(state_step.snow_layers, _field, None)
                    )
                    _record_ts_value(
                        f"{_prefix}_lev{_snow_lev + 1}",
                        None if _layer_values is None else _layer_values[..., _snow_lev],
                    )
            _record_ts_value("snow_ice_total", _snow_layer_sum(state_step, "h2osoi_ice"))
            _record_ts_value("snow_liq_total", _snow_layer_sum(state_step, "h2osoi_liq"))
            _record_ts_value("snow_t_min", _snow_layer_min_temp(state_step))
            _top_grain_radius = _snow_layer_top_grain_radius(state_step)
            _record_ts_value("snow_grain_radius_top", _top_grain_radius)
            if _top_grain_radius is None:
                _record_ts_value("snow_albedo_vis", None)
                _record_ts_value("snow_albedo_nir", None)
            else:
                _snow_albedo_vis, _snow_albedo_nir = _snow_albedo_from_grain_radius(_top_grain_radius)
                _record_ts_value("snow_albedo_vis", _snow_albedo_vis)
                _record_ts_value("snow_albedo_nir", _snow_albedo_nir)
            _record_ts_value("sabg_snow_input", blended.get("diag_sabg_snow_input"))
            _soil_temperature_result = getattr(last_result, "soil_temperature_result", None)
            _record_ts_value(
                "partial_snow_blend_energy_residual_uncorrected",
                getattr(_soil_temperature_result, "partial_snow_blend_energy_residual_uncorrected", None),
            )
            _record_ts_value(
                "partial_snow_blend_energy_residual_corrected",
                getattr(_soil_temperature_result, "partial_snow_blend_energy_residual_corrected", None),
            )
            _record_ts_value(
                "partial_snow_blend_water_residual",
                getattr(_soil_temperature_result, "partial_snow_blend_water_residual", None),
            )
            _record_ts_value("forc_t", tbot)
            _record_ts_value("forc_pbot", blended["diag_forc_pbot"])
            _record_ts_value("forc_swrad", fsds)
            _ood_written = set()
            _record_ts_value("mlp_ood_max_abs_z", ood_max_col)
            _record_ts_value("mlp_ood_frac", ood_frac_col)
            _ood_written.update(("mlp_ood_max_abs_z", "mlp_ood_frac"))
            for _ood_feature, _ood_value in (ood_feature_col or {}).items():
                _ood_key = _ood_var_name(_ood_feature)
                _record_ts_value(_ood_key, _ood_value)
                _ood_written.add(_ood_key)
            for _ood_key in tuple(
                key for key in regional_ts_records
                if key.startswith("mlp_ood") and key not in _ood_written
            ):
                _record_ts_value(_ood_key, None)
            _record_ts_value("eflx_sh_tot", blended.get("eflx_sh_tot"))
            _record_ts_value("qflx_evap_canopy", blended.get("qflx_evap_canopy"))
            _record_ts_value("qflx_ev_soil_mlp_blended", blended.get("qflx_ev_soil"))
            _record_ts_value("t_veg", blended.get("t_veg"))
            _record_ts_value("t_ref2m", blended.get("t_ref2m"))
            _record_ts_value("diag_ground_heat_flux_override", blended.get("diag_ground_heat_flux_override"))
            for _pft_idx, _pft_id in enumerate(pft_present):
                _pft_label = f"pft_{int(_pft_id)}"
                _record_ts_value(f"{_pft_label}_area_weight", pft_weights[_pft_idx])
                _record_ts_value(f"{_pft_label}_eflx_sh_tot", _pft_target_value(_pft_idx, "eflx_sh_tot"))
                _record_ts_value(f"{_pft_label}_hs_soil", _pft_target_value(_pft_idx, "hs_soil"))
                _record_ts_value(f"{_pft_label}_hs_top_snow", _pft_target_value(_pft_idx, "hs_top_snow"))
                _record_ts_value(f"{_pft_label}_hs_h2osfc", _pft_target_value(_pft_idx, "hs_h2osfc"))
                _record_ts_value(f"{_pft_label}_dhsdT", _pft_target_value(_pft_idx, "dhsdT"))
                _record_ts_value(f"{_pft_label}_qflx_tran_veg", _pft_target_value(_pft_idx, "qflx_tran_veg"))
                _record_ts_value(f"{_pft_label}_qflx_evap_canopy", _pft_target_value(_pft_idx, "qflx_evap_canopy"))
                _record_ts_value(f"{_pft_label}_qflx_ev_soil", _pft_target_value(_pft_idx, "qflx_ev_soil"))
                _record_ts_value(
                    f"{_pft_label}_qflx_ev_soil_weighted",
                    _pft_weighted_target_value(_pft_idx, "qflx_ev_soil"),
                )
                _record_ts_value(f"{_pft_label}_fpsn_patch", _pft_target_value(_pft_idx, "fpsn_patch"))
            for _name in (
                "hs_soil",
                "hs_top_snow",
                "hs_h2osfc",
                "dhsdT",
                "qflx_prec_grnd",
                "qflx_rain_grnd",
                "qflx_snow_grnd",
                "qflx_ev_soil",
                "qflx_ev_h2osfc",
                "qflx_ev_snow",
            ):
                _record_ts_value(_name, _target_from(last_result, _name))
            for _name in (
                "qflx_snow_melt",
                "qflx_snow_freeze",
                "qflx_soil_melt",
                "qflx_soil_freeze",
                "eflx_fgr12",
                "phase_hm_h2osfc",
                "phase_xm_h2osfc",
                "phase_temp1_h2osfc",
                "fn_h2osfc_old",
                "fn_h2osfc_new",
                "tk_h2osfc",
            ):
                _record_ts_value(_name, _soil_result_from(last_result, _name))
            _imelt = _soil_result_from(last_result, "imelt")
            _soil_melt_profile = _soil_result_from(last_result, "qflx_soil_melt_profile")
            _soil_freeze_profile = _soil_result_from(last_result, "qflx_soil_freeze_profile")
            _snow_melt_profile = _soil_result_from(last_result, "qflx_snow_melt_profile")
            _snow_freeze_profile = _soil_result_from(last_result, "qflx_snow_freeze_profile")
            for _snow_lev in range(MAX_SNOW_LAYERS):
                _record_ts_value(
                    f"qflx_snow_melt_lev{_snow_lev + 1}",
                    None if _snow_melt_profile is None else _snow_melt_profile[..., _snow_lev],
                )
                _record_ts_value(
                    f"qflx_snow_freeze_lev{_snow_lev + 1}",
                    None if _snow_freeze_profile is None else _snow_freeze_profile[..., _snow_lev],
                )
            for _soil_lev in range(soil_profile_nlev_ts):
                _record_ts_value(
                    f"imelt_lev{_soil_lev + 1}",
                    None if _imelt is None else _imelt[..., _soil_lev],
                )
                _record_ts_value(
                    f"qflx_soil_melt_lev{_soil_lev + 1}",
                    None if _soil_melt_profile is None else _soil_melt_profile[..., _soil_lev],
                )
                _record_ts_value(
                    f"qflx_soil_freeze_lev{_soil_lev + 1}",
                    None if _soil_freeze_profile is None else _soil_freeze_profile[..., _soil_lev],
                )
            for _name in ("qflx_snow_to_soil", "qflx_snwcp_ice", "qflx_snwcp_liq"):
                _record_ts_value(_name, _snow_water_from(last_result, _name))
            _record_feature_snapshot(step_i, feature_buffers)

        regional_step_callback = _record_regional_cell
    elif single_cell_mode:
        print("  Single-cell time-series plot disabled")

    def _next_ym(y: int, m: int) -> "tuple[int, int]":
        m += 1
        if m > 12:
            m, y = 1, y + 1
        return y, m

    # Build the full list of (year, month) to process.
    _ym_list: list[tuple[int, int]] = []
    _y, _m = year, month
    for _ in range(args.nmonths):
        _ym_list.append((_y, _m))
        _y, _m = _next_ym(_y, _m)

    co2_file = _resolve_regional_co2_file(args.co2_file, args.met_dir)
    co2_scalar = None
    if co2_file is not None:
        if not co2_file.exists():
            raise SystemExit(f"CO2 file not found: {co2_file}")
        _load_co2_file_series(co2_file)
        print(f"  CO2 forcing: {co2_file}")
    elif args.co2_ppm is not None:
        co2_scalar = float(args.co2_ppm)
        print(f"  CO2 forcing: scalar {co2_scalar:.6g} ppm")
    else:
        co2_scalar = float(_CO2_PPM_FALLBACK)
        print(
            "  CO2 forcing: no CO2 stream found from --met-dir; "
            f"using fallback scalar {co2_scalar:.6g} ppm"
        )

    def _co2_ppm_for_ym(y: int, m: int, *, calendar: str) -> float:
        if co2_file is not None:
            return _co2_ppm_from_file(co2_file, int(y), int(m), calendar=calendar)
        assert co2_scalar is not None
        return float(co2_scalar)

    met_cycle_years = int(args.regional_met_cycle_years)
    met_cycle_start_year = (
        int(args.regional_met_cycle_start_year)
        if args.regional_met_cycle_start_year is not None
        else int(year)
    )

    def _met_year_for_sim_year(y: int) -> int:
        if met_cycle_years <= 0:
            return int(y)
        return met_cycle_start_year + ((int(y) - met_cycle_start_year) % met_cycle_years)

    def _ym_load_label(y: int, m: int) -> str:
        met_y = _met_year_for_sim_year(y)
        label = f"{int(y):04d}-{int(m):02d}"
        if met_y != int(y):
            label += f" (met {met_y:04d}-{int(m):02d})"
        return label

    if met_cycle_years > 0:
        print(
            f"  Regional met cycling enabled: years {met_cycle_start_year}-"
            f"{met_cycle_start_year + met_cycle_years - 1} repeat"
        )

    met_cache_dir = None
    if args.regional_met_cache:
        met_cache_dir = (
            args.regional_met_cache_dir
            if args.regional_met_cache_dir is not None
            else args.met_dir / ".cell_cache"
        )
        print(f"  Regional met cache: {met_cache_dir}")
    precompute_cache_dir = None
    if args.regional_precompute_cache:
        precompute_cache_dir = (
            args.regional_precompute_cache_dir
            if args.regional_precompute_cache_dir is not None
            else Path("cache/regional_precompute")
        )
        print(f"  Regional precompute cache: {precompute_cache_dir}")

    def _load_and_precompute(y: int, m: int) -> "tuple[np.ndarray, dict]":
        """Disk read + CPU precomputation — runs concurrently with GPU pass."""
        met_y = _met_year_for_sim_year(y)
        met = load_met_month_cpu(
            args.met_dir,
            met_y,
            m,
            surf.get("met_grid_row", surf["grid_row"]),
            surf.get("met_grid_col", surf["grid_col"]),
            cache_dir=met_cache_dir,
            met_index=surf.get("met_index"),
            met_format=regional_met_format,
            calendar=args.regional_calendar,
            pad_steps=met_padding_steps,
        )
        core_met_steps = int(met.shape[0]) - 2 * int(met_padding_steps)
        if core_met_steps <= 0:
            raise ValueError(
                f"{regional_met_format} met load for {y}-{m:02d} returned {met.shape[0]} "
                f"steps, too few for padding={met_padding_steps}"
            )
        n1h_expected = core_met_steps * int(met_input_step_hours)
        calendar_for_precompute = args.regional_calendar
        expected_days = _days_in_month(met_y, m, calendar_for_precompute)
        actual_days = n1h_expected / 24.0
        if not np.isclose(actual_days, expected_days):
            alternate_calendar = "gregorian" if calendar_for_precompute == "noleap" else "noleap"
            alternate_days = _days_in_month(met_y, m, alternate_calendar)
            if np.isclose(actual_days, alternate_days):
                print(
                    f"  [{regional_met_format.upper()} met] {y}-{m:02d} has "
                    f"{actual_days:.0f} days; using {alternate_calendar} calendar "
                    "for solar/LAI precompute for this month",
                    flush=True,
                )
                calendar_for_precompute = alternate_calendar
            else:
                print(
                    f"  [{regional_met_format.upper()} met] {y}-{m:02d} has "
                    f"{actual_days:.2f} days, not {expected_days} for "
                    f"{calendar_for_precompute}; precomputing exactly {n1h_expected} hours",
                    flush=True,
                )
        pc = _load_or_precompute_month_cpu(
            surf,
            y,
            m,
            calendar=calendar_for_precompute,
            n1h=n1h_expected,
            cache_dir=precompute_cache_dir,
        )
        pc["calendar"] = calendar_for_precompute
        pc["met_year"] = int(met_y)
        return met, pc

    regional_prefetch = args.regional_prefetch
    if regional_prefetch is None:
        regional_prefetch = True
    if not regional_prefetch:
        print("  Regional prefetch disabled")

    from concurrent.futures import ThreadPoolExecutor
    _executor = (
        ThreadPoolExecutor(max_workers=1, thread_name_prefix="met_prefetch")
        if regional_prefetch and len(_ym_list) > 1
        else None
    )

    # Eagerly load + precompute the first month (blocking), then submit month 2.
    print(f"  {_ym_load_label(*_ym_list[0])}:  loading met + precomputing coszen/LAI … ", end="", flush=True)
    t_load = time.perf_counter()
    met_cpu, precomputed_cpu = _load_and_precompute(*_ym_list[0])
    t_load_done = time.perf_counter()
    print(f"done in {t_load_done - t_load:.2f}s  (shape {tuple(met_cpu.shape)})")

    _prefetch_future = (
        _executor.submit(_load_and_precompute, *_ym_list[1])
        if _executor is not None
        else None
    )

    _regional_output_written = False
    _five_day_pending_diag: dict[str, np.ndarray] = {}
    _five_day_pending_dates: list[tuple[int, int, int]] = []
    elm_history_writer = (
        ElmHistoryWriter(
            args.elm_history_output_dir,
            case_name=str(args.elm_history_case),
            calendar=args.regional_calendar,
            surface_dataset=args.regional_surfdata,
            clm_params=args.clm_params,
        )
        if args.elm_history_output_dir is not None
        else None
    )
    _elm_h0_pending_diag: dict[str, np.ndarray] = {}
    _elm_h0_pending_dates: list[tuple[int, int, int]] = []
    if elm_history_writer is not None:
        print(
            "  ELM-style h0 output: "
            f"{args.elm_history_output_dir}  case={args.elm_history_case}  "
            f"frequency={args.elm_history_frequency}"
        )

    def _period_index_for_daily(year: int, month: int, nday_local: int) -> list[tuple[int, int, int, int]]:
        return [
            (int(year), int(month), int(day), 1)
            for day in range(1, int(nday_local) + 1)
        ]

    def _period_index_for_hourly(year: int, month: int, nhour_local: int) -> list[tuple[int, int, int, int, int]]:
        return [
            (
                int(year),
                int(month),
                int(hour_i // 24 + 1),
                int((hour_i % 24) * 3600),
                3600,
            )
            for hour_i in range(int(nhour_local))
        ]

    def _write_elm_h0_records(
        diag: dict[str, np.ndarray],
        *,
        period_index: list[tuple],
    ) -> None:
        if elm_history_writer is None or not diag or not period_index:
            return
        t_write = time.perf_counter()
        paths = elm_history_writer.write(
            surf=surf,
            diag=diag,
            period_index=period_index,
        )
        if paths:
            path_text = ", ".join(str(path) for path in paths)
            print(f"    ELM h0 write: {time.perf_counter() - t_write:.2f}s  {path_text}")

    def _flush_elm_h0_five_day_pending(*, final: bool = False) -> None:
        nonlocal _elm_h0_pending_diag, _elm_h0_pending_dates
        if elm_history_writer is None or not _elm_h0_pending_dates:
            return
        total_days = len(_elm_h0_pending_dates)
        n_groups = total_days // 5
        if final and total_days % 5:
            n_groups += 1
        if n_groups <= 0:
            return

        period_index: list[tuple[int, int, int, int]] = []
        rows: dict[str, list[np.ndarray]] = {key: [] for key in _elm_h0_pending_diag}
        used_days = 0
        for group_idx in range(n_groups):
            start_day = group_idx * 5
            end_day = min(start_day + 5, total_days)
            start_year, start_month, start_month_day = _elm_h0_pending_dates[start_day]
            period_index.append((
                int(start_year),
                int(start_month),
                int(start_month_day),
                int(end_day - start_day),
            ))
            for key, arr in _elm_h0_pending_diag.items():
                rows[key].append(np.nanmean(arr[start_day:end_day], axis=0))
            used_days = end_day

        output_diag = {
            key: np.stack(value, axis=0).astype(np.float32, copy=False)
            for key, value in rows.items()
            if value
        }
        _write_elm_h0_records(output_diag, period_index=period_index)

        if used_days >= total_days:
            _elm_h0_pending_dates = []
            _elm_h0_pending_diag = {
                key: arr[:0]
                for key, arr in _elm_h0_pending_diag.items()
            }
        else:
            _elm_h0_pending_dates = _elm_h0_pending_dates[used_days:]
            _elm_h0_pending_diag = {
                key: arr[used_days:]
                for key, arr in _elm_h0_pending_diag.items()
            }

    def _accumulate_elm_h0_five_day(
        daily_diag: dict[str, np.ndarray],
        *,
        year: int,
        month: int,
    ) -> None:
        nonlocal _elm_h0_pending_diag, _elm_h0_pending_dates
        if elm_history_writer is None or not daily_diag:
            return
        incoming = {
            key: np.asarray(value, dtype=np.float32)
            for key, value in daily_diag.items()
            if not key.startswith("mlp_ood_abs_z_")
        }
        if not incoming:
            return
        nday_local = int(next(iter(incoming.values())).shape[0])
        if nday_local <= 0:
            return
        pending_len = len(_elm_h0_pending_dates)
        month_dates = [
            (int(year), int(month), int(day))
            for day in range(1, nday_local + 1)
        ]
        for key, arr in incoming.items():
            if arr.shape[0] != nday_local:
                raise ValueError(
                    f"{key} h0 daily diagnostics have {arr.shape[0]} records, "
                    f"expected {nday_local}"
                )
            if key not in _elm_h0_pending_diag:
                _elm_h0_pending_diag[key] = np.full(
                    (pending_len, *arr.shape[1:]),
                    np.nan,
                    dtype=np.float32,
                )
        for key in tuple(_elm_h0_pending_diag):
            arr = incoming.get(key)
            if arr is None:
                arr = np.full(
                    (nday_local, *_elm_h0_pending_diag[key].shape[1:]),
                    np.nan,
                    dtype=np.float32,
                )
            _elm_h0_pending_diag[key] = np.concatenate(
                (_elm_h0_pending_diag[key], arr),
                axis=0,
            )
        _elm_h0_pending_dates.extend(month_dates)
        _flush_elm_h0_five_day_pending(final=False)

    def _flush_five_day_pending(*, final: bool = False) -> None:
        nonlocal _regional_output_written, _five_day_pending_diag, _five_day_pending_dates
        if args.regional_daily_output is None or not _five_day_pending_dates:
            return
        total_days = len(_five_day_pending_dates)
        n_groups = total_days // 5
        if final and total_days % 5:
            n_groups += 1
        if n_groups <= 0:
            return

        period_index: list[tuple[int, int, int, int]] = []
        output_rows: dict[str, list[np.ndarray]] = {
            key: [] for key in _five_day_pending_diag
        }
        used_days = 0
        for group_idx in range(n_groups):
            start_day = group_idx * 5
            end_day = min(start_day + 5, total_days)
            start_year, start_month, start_month_day = _five_day_pending_dates[start_day]
            period_index.append((
                int(start_year),
                int(start_month),
                int(start_month_day),
                int(end_day - start_day),
            ))
            for key, arr in _five_day_pending_diag.items():
                if key.endswith("_day_mean") and "daytime_sample_count" in _five_day_pending_diag:
                    _counts = _five_day_pending_diag["daytime_sample_count"][start_day:end_day]
                    _values = arr[start_day:end_day]
                    _denom = np.nansum(_counts, axis=0)
                    output_rows[key].append(
                        np.divide(
                            np.nansum(_values * _counts, axis=0),
                            _denom,
                            out=np.full_like(_denom, np.nan, dtype=np.float32),
                            where=_denom > 0.0,
                        )
                    )
                elif key.endswith("_night_mean") and "nighttime_sample_count" in _five_day_pending_diag:
                    _counts = _five_day_pending_diag["nighttime_sample_count"][start_day:end_day]
                    _values = arr[start_day:end_day]
                    _denom = np.nansum(_counts, axis=0)
                    output_rows[key].append(
                        np.divide(
                            np.nansum(_values * _counts, axis=0),
                            _denom,
                            out=np.full_like(_denom, np.nan, dtype=np.float32),
                            where=_denom > 0.0,
                        )
                    )
                elif key in {"daytime_sample_count", "nighttime_sample_count"}:
                    output_rows[key].append(np.nansum(arr[start_day:end_day], axis=0))
                else:
                    output_rows[key].append(np.nanmean(arr[start_day:end_day], axis=0))
            used_days = end_day

        output_diag = {
            key: np.stack(rows, axis=0).astype(np.float32, copy=False)
            for key, rows in output_rows.items()
            if rows
        }
        if output_diag:
            _t_write = time.perf_counter()
            _write_regional_daily_output(
                args.regional_daily_output,
                surf=surf,
                year=period_index[0][0],
                month=period_index[0][1],
                daily_diag=output_diag,
                append=_regional_output_written,
                calendar=args.regional_calendar,
                frequency="5day",
                period_index=period_index,
                day_night_fsds_threshold=(
                    args.regional_day_night_fsds_threshold
                    if args.regional_day_night_flux_output
                    else None
                ),
            )
            _regional_output_written = True
            print(f"    regional output write: {time.perf_counter() - _t_write:.2f}s")

        if used_days >= total_days:
            _five_day_pending_dates = []
            _five_day_pending_diag = {
                key: arr[:0]
                for key, arr in _five_day_pending_diag.items()
            }
        else:
            _five_day_pending_dates = _five_day_pending_dates[used_days:]
            _five_day_pending_diag = {
                key: arr[used_days:]
                for key, arr in _five_day_pending_diag.items()
            }

    def _accumulate_five_day_daily(
        daily_diag: dict[str, np.ndarray],
        *,
        year: int,
        month: int,
    ) -> None:
        nonlocal _five_day_pending_diag, _five_day_pending_dates
        if not daily_diag:
            return
        output_keys = tuple(key for key in _REGIONAL_DAILY_OUTPUT_KEYS if key in daily_diag)
        if not output_keys:
            return
        incoming = {
            key: np.asarray(daily_diag[key], dtype=np.float32)
            for key in output_keys
        }
        nday_local = int(next(iter(incoming.values())).shape[0])
        if nday_local <= 0:
            return
        pending_len = len(_five_day_pending_dates)
        month_dates = [
            (int(year), int(month), int(day))
            for day in range(1, nday_local + 1)
        ]
        for key, arr in incoming.items():
            if arr.shape != (nday_local, ncell):
                raise ValueError(
                    f"{key} daily diagnostics have shape {arr.shape}, "
                    f"expected {(nday_local, ncell)}"
                )
            if key not in _five_day_pending_diag:
                _five_day_pending_diag[key] = np.full(
                    (pending_len, ncell),
                    np.nan,
                    dtype=np.float32,
                )
        for key in tuple(_five_day_pending_diag):
            arr = incoming.get(key)
            if arr is None:
                arr = np.full((nday_local, ncell), np.nan, dtype=np.float32)
            _five_day_pending_diag[key] = np.concatenate(
                (_five_day_pending_diag[key], arr),
                axis=0,
            )
        _five_day_pending_dates.extend(month_dates)
        _flush_five_day_pending(final=False)

    for m_idx, (year, month) in enumerate(_ym_list):
        run_calendar = str(precomputed_cpu.get("calendar", args.regional_calendar))
        precip_multiplier_this_month = None
        if (
            "precip_treatment_multiplier" in surf
            and float(args.regional_precip_treatment_area_ha) > 0.0
            and float(args.regional_precip_treatment_reduction) > 0.0
            and _month_intersects_date_window(
                year,
                month,
                calendar=run_calendar,
                start_date=args.regional_precip_treatment_start_date,
                end_date=args.regional_precip_treatment_end_date,
            )
        ):
            precip_multiplier_this_month = surf.get("precip_treatment_multiplier")
        co2_ppm_this_month = _co2_ppm_for_ym(year, month, calendar=run_calendar)
        if m_idx == 0 or month == 1:
            print(f"  {year}-{month:02d}:  CO2 {co2_ppm_this_month:.3f} ppm")
        # Transfer current month to device.
        t_xfer = time.perf_counter()
        met_dev = torch.as_tensor(met_cpu, device=device, dtype=dtype)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t_ready = time.perf_counter()
        print(f"  {year}-{month:02d}:  GPU transfer {t_ready - t_xfer:.2f}s")
        soil_temperature_override = None
        if args.regional_soil_temp_override_elm_history is not None:
            n1h_override = (
                int(met_cpu.shape[0]) - 2 * int(met_padding_steps)
            ) * int(met_input_step_hours)
            if args.regional_max_hours is not None:
                n1h_override = min(n1h_override, int(args.regional_max_hours))
            t_override_start = time.perf_counter()
            soil_temperature_override = _load_elm_history_soil_temperature_override(
                args.regional_soil_temp_override_elm_history,
                year=year,
                month=month,
                ncell=ncell,
                nlev=int(args.regional_soil_temp_override_nlev),
                n1h=n1h_override,
                calendar=run_calendar,
            )
            print(
                f"  {year}-{month:02d}:  loaded ELM TSOI override "
                f"{tuple(soil_temperature_override.shape)} "
                f"(first {int(args.regional_soil_temp_override_nlev)} level(s)) "
                f"in {time.perf_counter() - t_override_start:.2f}s"
            )
        ground_temperature_override = None
        if args.regional_ground_temp_override_elm_history is not None:
            n1h_override = (
                int(met_cpu.shape[0]) - 2 * int(met_padding_steps)
            ) * int(met_input_step_hours)
            if args.regional_max_hours is not None:
                n1h_override = min(n1h_override, int(args.regional_max_hours))
            t_override_start = time.perf_counter()
            ground_temperature_override = _load_elm_history_scalar_hourly_override(
                args.regional_ground_temp_override_elm_history,
                var_name="TG",
                year=year,
                month=month,
                ncell=ncell,
                n1h=n1h_override,
                calendar=run_calendar,
            )
            print(
                f"  {year}-{month:02d}:  loaded ELM TG feature override "
                f"{tuple(ground_temperature_override.shape)} "
                f"in {time.perf_counter() - t_override_start:.2f}s"
            )
        soil_liquid_override = None
        soil_ice_override = None
        if args.regional_soil_moisture_override_elm_history is not None:
            n1h_override = (
                int(met_cpu.shape[0]) - 2 * int(met_padding_steps)
            ) * int(met_input_step_hours)
            if args.regional_max_hours is not None:
                n1h_override = min(n1h_override, int(args.regional_max_hours))
            t_override_start = time.perf_counter()
            soil_liquid_override, soil_ice_override = (
                _load_elm_history_soil_moisture_override(
                    args.regional_soil_moisture_override_elm_history,
                    year=year,
                    month=month,
                    ncell=ncell,
                    nlev=int(args.regional_soil_moisture_override_nlev),
                    n1h=n1h_override,
                    calendar=run_calendar,
                )
            )
            print(
                f"  {year}-{month:02d}:  loaded ELM SOILLIQ/SOILICE override "
                f"{tuple(soil_liquid_override.shape)} "
                f"(first {int(args.regional_soil_moisture_override_nlev)} level(s)) "
                f"in {time.perf_counter() - t_override_start:.2f}s"
            )
        ground_heat_flux_override = None
        if args.regional_ground_heat_override_elm_history is not None:
            n1h_override = (
                int(met_cpu.shape[0]) - 2 * int(met_padding_steps)
            ) * int(met_input_step_hours)
            if args.regional_max_hours is not None:
                n1h_override = min(n1h_override, int(args.regional_max_hours))
            t_override_start = time.perf_counter()
            ground_heat_flux_override = _load_elm_history_scalar_hourly_override(
                args.regional_ground_heat_override_elm_history,
                var_name=args.regional_ground_heat_override_var,
                year=year,
                month=month,
                ncell=ncell,
                n1h=n1h_override,
                calendar=run_calendar,
            )
            print(
                f"  {year}-{month:02d}:  loaded ELM {args.regional_ground_heat_override_var} "
                f"ground-heat override {tuple(ground_heat_flux_override.shape)} "
                f"for {', '.join(args.regional_ground_heat_override_targets)} "
                f"in {time.perf_counter() - t_override_start:.2f}s"
            )
        zwt_override = None
        if args.regional_zwt_override_elm_history is not None:
            n1h_override = (
                int(met_cpu.shape[0]) - 2 * int(met_padding_steps)
            ) * int(met_input_step_hours)
            if args.regional_max_hours is not None:
                n1h_override = min(n1h_override, int(args.regional_max_hours))
            t_override_start = time.perf_counter()
            zwt_override = _load_elm_history_scalar_hourly_override(
                args.regional_zwt_override_elm_history,
                var_name=args.regional_zwt_override_var,
                aliases=("ZWT0",),
                year=year,
                month=month,
                ncell=ncell,
                n1h=n1h_override,
                calendar=run_calendar,
            )
            print(
                f"  {year}-{month:02d}:  loaded ELM {args.regional_zwt_override_var} "
                f"ZWT override {tuple(zwt_override.shape)} "
                f"in {time.perf_counter() - t_override_start:.2f}s"
            )

        state, sanity, timing, diag_maps = run_regional_pass(
            surf=surf,
            pft_ids=pft_present,
            pft_weights=pft_weights,
            pft_bundles=pft_bundles,
            pft_statics=pft_statics,
            shared_static=shared_static,
            met_3h=met_dev,
            year=year,
            month=month,
            met_input_step_hours=met_input_step_hours,
            met_padding_steps=met_padding_steps,
            elm_stream_met_methods=elm_stream_met_methods,
            calendar=run_calendar,
            dt=3600.0,
            device=device,
            dtype=dtype,
            mlp_dtype=mlp_dtype,
            hydrology_backend=args.hydrology_backend,
            soil_temperature_backend=args.soil_temperature_backend,
            snow_model_enabled=snow_model_enabled,
            snow_scalar_only=args.snow_scalar_only,
            blend_partial_snow_soil_temperature=args.blend_partial_snow_soil_temperature,
            co2_ppm=co2_ppm_this_month,
            initial_state=state,
            precomputed=precomputed_cpu,
            profile_sections=_prof,
            nens=args.regional_ensemble_size,
            ensemble_seed=args.ensemble_seed,
            ensemble_flnr_factors=args.regional_ensemble_flnr_factors,
            ensemble_mbbopt_factors=args.regional_ensemble_mbbopt_factors,
            carbon_model=carbon_model,
            ood_diagnostics_stride=args.regional_ood_stride,
            ood_threshold=args.regional_ood_threshold,
            ood_feature_map_names=_parse_ood_feature_map_names(args.regional_ood_feature_maps),
            step_callback=regional_step_callback,
            progress_stride=args.regional_progress_stride,
            record_diag_maps=(
                (not args.no_plot and not single_cell_mode)
                or (
                    args.regional_daily_output is not None
                    and args.regional_output_frequency == "monthly"
                )
                or (
                    elm_history_writer is not None
                    and args.elm_history_frequency == "monthly"
                )
            ),
            record_daily_diag_maps=(
                (
                    args.regional_daily_output is not None
                    and args.regional_output_frequency in {"daily", "5day"}
                )
                or (
                    elm_history_writer is not None
                    and args.elm_history_frequency in {"daily", "5day"}
                )
            ),
            record_hourly_diag_maps=(
                elm_history_writer is not None
                and args.elm_history_frequency == "hourly"
            ),
            record_day_night_flux_maps=(
                args.regional_day_night_flux_output
                and args.regional_daily_output is not None
                and args.regional_output_frequency in {"daily", "5day"}
            ),
            day_night_fsds_threshold=args.regional_day_night_fsds_threshold,
            safety_clamps=args.regional_safety_clamps,
            temperature_min_k=args.regional_temperature_min_k,
            temperature_max_k=args.regional_temperature_max_k,
            photosynthesis_max=args.regional_photosynthesis_max,
            thermal_target_z_clamp=args.regional_thermal_target_z_clamp,
            soil_energy_sabg_source=args.soil_energy_sabg_source,
            physics_step_hours=args.regional_physics_step_hours,
            universal_mlp_batch=args.universal,
            sparse_universal_mlp_batch=args.regional_sparse_pft_batch,
            sparse_individual_feature_assembly=args.regional_sparse_individual_features,
            neutral_patch0_flnr=neutral_patch0_flnr,
            regional_lai_scale=args.regional_lai_scale,
            regional_sai_scale=args.regional_sai_scale,
            regional_lai_add=args.regional_lai_add,
            regional_lai_add_months=args.regional_lai_add_months,
            regional_lai_add_include_bare=args.regional_lai_add_include_bare,
            regional_trait_ensemble_size=args.regional_trait_ensemble_size,
            regional_trait_chunk_size=args.regional_trait_chunk_size,
            regional_trait_sigma=args.regional_trait_sigma,
            regional_trait_relative_sd=args.regional_trait_relative_sd,
            regional_trait_feature_names=tuple(args.regional_trait_feature),
            soil_temperature_override=soil_temperature_override,
            ground_temperature_override=ground_temperature_override,
            soil_liquid_override=soil_liquid_override,
            soil_ice_override=soil_ice_override,
            ground_heat_flux_override=ground_heat_flux_override,
            ground_heat_flux_override_targets=tuple(args.regional_ground_heat_override_targets),
            zwt_override=zwt_override,
            precip_multiplier=precip_multiplier_this_month,
            max_hours=args.regional_max_hours,
        )

        n1h = timing["n1h"]
        last_n1h = int(n1h)
        elapsed = timing["elapsed_s"]
        total_elapsed += elapsed
        total_nhours  += n1h
        cell_steps = ncell * n1h
        print(
            f"  {year}-{month:02d}:  {n1h} hourly steps  "
            f"{elapsed:.2f}s  ({cell_steps / elapsed:.0f} cell-steps/s  "
            f"{1e6 * elapsed / cell_steps:.2f} µs/cell-step)"
        )
        if int(timing.get("physics_cadence_steps", 1)) > 1:
            print(
                f"    physics cadence: every {int(timing['physics_cadence_steps'])} h "
                f"({int(timing.get('physics_calls', 0))} soil T/hydrology calls)"
            )
        if snow_model_enabled:
            print(
                "    snow gates: "
                f"possible_snowfall_steps={int(timing.get('snowfall_possible_steps', 0))}/{n1h}  "
                f"snow_physics_calls={int(timing.get('snow_physics_calls', 0))}/"
                f"{int(timing.get('physics_calls', 0))}  "
                f"active_snow_calls={int(timing.get('active_snow_physics_calls', 0))}  "
                f"pending_snow_calls={int(timing.get('pending_snow_physics_calls', 0))}  "
                f"active_cells_max={int(timing.get('max_active_snow_cells', 0))}  "
                f"active_cells_final={int(timing.get('final_active_snow_cells', 0))}"
            )
        if args.universal:
            print(
                f"    universal fast feature steps: "
                f"{int(timing.get('universal_fast_steps', 0))}/{n1h}"
            )
        print(
            "    feature cache refreshes: "
            f"state={int(timing.get('state_feature_refreshes', n1h))}/{n1h}  "
            f"vegetation={int(timing.get('vegetation_feature_refreshes', n1h))}/{n1h}"
        )
        if sanity:
            print("    sanity: " + "  ".join(f"{k}={v:.4e}" for k, v in sanity.items()))
        if _prof:
            top_prof = {
                key: value
                for key, value in _prof.items()
                if not key.startswith("physics:")
                and not key.startswith("canopy:")
            }
            total_prof = sum(top_prof.values()) or 1.0
            parts = "  ".join(
                f"{k}={v:.2f}s({100*v/total_prof:.0f}%)"
                for k, v in sorted(top_prof.items(), key=lambda x: -x[1])
            )
            print(f"    profile: {parts}")
            canopy_prof = {
                key.split(":", 1)[1]: value
                for key, value in _prof.items()
                if key.startswith("canopy:")
            }
            if canopy_prof:
                canopy_total = _prof.get("canopy_interception", sum(canopy_prof.values())) or 1.0
                canopy_parts = "  ".join(
                    f"{k}={v:.2f}s({100*v/canopy_total:.0f}%)"
                    for k, v in sorted(canopy_prof.items(), key=lambda x: -x[1])
                )
                print(f"    canopy detail: {canopy_parts}")
            physics_prof = {
                key.split(":", 1)[1]: value
                for key, value in _prof.items()
                if key.startswith("physics:")
            }
            if physics_prof:
                physics_total = _prof.get("physics_step", sum(physics_prof.values())) or 1.0
                hydrology_detail = {
                    key: value
                    for key, value in physics_prof.items()
                    if key.startswith("hydrology_full_")
                }
                physics_top = {
                    key: value
                    for key, value in physics_prof.items()
                    if not key.startswith("hydrology_full_")
                }
                physics_parts = "  ".join(
                    f"{k}={v:.2f}s({100*v/physics_total:.0f}%)"
                    for k, v in sorted(physics_top.items(), key=lambda x: -x[1])
                )
                print(f"    physics profile: {physics_parts}")
                if hydrology_detail:
                    hydrology_total = physics_prof.get(
                        "hydrology_torch_full",
                        sum(hydrology_detail.values()),
                    ) or 1.0
                    hydrology_parts = "  ".join(
                        f"{k}={v:.2f}s({100*v/hydrology_total:.0f}%)"
                        for k, v in sorted(hydrology_detail.items(), key=lambda x: -x[1])
                    )
                    print(f"    full hydrology profile: {hydrology_parts}")
            _prof.clear()

        # Free cached GPU memory between months to avoid fragmentation.
        if device.type == "cuda":
            torch.cuda.empty_cache()

        daily_diag_maps = diag_maps.pop("_daily", None)
        hourly_diag_maps = diag_maps.pop("_hourly", None)
        monthly_diag_maps = diag_maps.pop("_monthly", None)
        output_diag_maps = None
        if args.regional_daily_output is not None:
            if args.regional_output_frequency in {"daily", "5day"}:
                output_diag_maps = daily_diag_maps
            else:
                output_diag_maps = {}
                for _key in _REGIONAL_DAILY_OUTPUT_KEYS:
                    if _key not in diag_maps:
                        continue
                    _arr = np.asarray(diag_maps[_key], dtype=np.float32)
                    if _arr.shape == (ncell,):
                        output_diag_maps[_key] = _arr[None, :]
                if monthly_diag_maps:
                    output_diag_maps.update(monthly_diag_maps)
        if args.regional_daily_output is not None and output_diag_maps:
            if args.regional_output_frequency == "5day":
                _accumulate_five_day_daily(
                    output_diag_maps,
                    year=year,
                    month=month,
                )
            else:
                _t_write = time.perf_counter()
                _write_regional_daily_output(
                    args.regional_daily_output,
                    surf=surf,
                    year=year,
                    month=month,
                    daily_diag=output_diag_maps,
                    append=_regional_output_written,
                    calendar=run_calendar,
                    frequency=args.regional_output_frequency,
                    day_night_fsds_threshold=(
                        args.regional_day_night_fsds_threshold
                        if args.regional_day_night_flux_output
                        else None
                    ),
                )
                _regional_output_written = True
                print(f"    regional output write: {time.perf_counter() - _t_write:.2f}s")

        if elm_history_writer is not None:
            if args.elm_history_frequency == "hourly":
                if hourly_diag_maps:
                    _write_elm_h0_records(
                        hourly_diag_maps,
                        period_index=_period_index_for_hourly(
                            year,
                            month,
                            int(next(iter(hourly_diag_maps.values())).shape[0]),
                        ),
                    )
            elif args.elm_history_frequency == "daily":
                if daily_diag_maps:
                    _write_elm_h0_records(
                        daily_diag_maps,
                        period_index=_period_index_for_daily(
                            year,
                            month,
                            int(next(iter(daily_diag_maps.values())).shape[0]),
                        ),
                    )
            elif args.elm_history_frequency == "5day":
                if daily_diag_maps:
                    _accumulate_elm_h0_five_day(
                        daily_diag_maps,
                        year=year,
                        month=month,
                    )
            else:
                h0_monthly_diag: dict[str, np.ndarray] = {}
                for _key, _value in diag_maps.items():
                    _arr = np.asarray(_value, dtype=np.float32)
                    if _arr.ndim == 1 and _arr.shape == (ncell,):
                        h0_monthly_diag[_key] = _arr[None, :]
                    elif _key in {"elm_h0_TSOI", "elm_h0_H2OSOI"} and _arr.ndim == 2 and _arr.shape[0] == ncell:
                        h0_monthly_diag[_key] = _arr[None, :, :]
                    elif _key in {"albgrd_mean", "albgri_mean", "albd_mean", "albi_mean"} and _arr.shape == (ncell, 2):
                        h0_monthly_diag[_key] = _arr[None, :, :]
                if monthly_diag_maps:
                    for _key, _value in monthly_diag_maps.items():
                        _arr = np.asarray(_value, dtype=np.float32)
                        if _arr.shape[0] == 1:
                            h0_monthly_diag[_key] = _arr
                if h0_monthly_diag:
                    _write_elm_h0_records(
                        h0_monthly_diag,
                        period_index=[(
                            int(year),
                            int(month),
                            1,
                            int(_days_in_month(year, month, run_calendar)),
                        )],
                    )

        if not args.no_plot and not single_cell_mode:
            plot_diag_maps = _filter_regional_plot_diag_maps(
                diag_maps,
                args.regional_plot_variable,
                monthly_diag_maps=monthly_diag_maps,
            )
            # Accumulate monthly diagnostic maps.
            for key, arr in plot_diag_maps.items():
                if key in _all_diag:
                    _all_diag[key] = _all_diag[key] + arr
                else:
                    _all_diag[key] = arr.copy()
            if plot_diag_maps:
                _diag_months += 1

            # Per-month plots.
            if plot_diag_maps:
                print(f"  {year}-{month:02d}:  saving diagnostic maps ...", flush=True)
                _plot_diag_maps(
                    diag=plot_diag_maps,
                    n_months=1,
                    grid_row=surf["grid_row"],
                    grid_col=surf["grid_col"],
                    nlat=int(surf.get("map_nlat", 361)), nlon=int(surf.get("map_nlon", 625)),
                    lat_min=float(surf.get("map_lat_min", 25.0)), lat_max=float(surf.get("map_lat_max", 40.0)),
                    lon_min=float(surf.get("map_lon_min", -100.0)), lon_max=float(surf.get("map_lon_max", -74.0)),
                    out_dir=args.regional_plot_dir,
                    label=f"{year}_{month:02d}",
                    map_lat_centers=surf.get("map_lat_centers"),
                    map_lon_centers=surf.get("map_lon_centers"),
                )

        # Retrieve or synchronously load data for next month.
        next_idx = m_idx + 1
        if _prefetch_future is not None:
            next_ym = _ym_list[next_idx]
            print(f"  {_ym_load_label(*next_ym)}:  waiting for prefetch … ", end="", flush=True)
            t_wait = time.perf_counter()
            met_cpu, precomputed_cpu = _prefetch_future.result()
            print(f"done ({time.perf_counter() - t_wait:.2f}s wait)")
            # Submit the one after that, if it exists.
            _prefetch_future = (
                _executor.submit(_load_and_precompute, *_ym_list[next_idx + 1])
                if _executor is not None and next_idx + 1 < len(_ym_list)
                else None
            )
        elif next_idx < len(_ym_list):
            next_ym = _ym_list[next_idx]
            print(f"  {_ym_load_label(*next_ym)}:  loading met + precomputing coszen/LAI … ", end="", flush=True)
            t_load = time.perf_counter()
            met_cpu, precomputed_cpu = _load_and_precompute(*next_ym)
            print(f"done in {time.perf_counter() - t_load:.2f}s  (shape {tuple(met_cpu.shape)})")

    if args.regional_daily_output is not None and args.regional_output_frequency == "5day":
        _flush_five_day_pending(final=True)
    if elm_history_writer is not None and args.elm_history_frequency == "5day":
        _flush_elm_h0_five_day_pending(final=True)

    if _executor is not None:
        _executor.shutdown(wait=False)

    if args.regional_restart_out is not None:
        completed_year, completed_month = _ym_list[-1]
        next_year, next_month = _next_ym(completed_year, completed_month)
        save_regional_restart(
            args.regional_restart_out,
            state=state,
            surf=surf,
            pft_ids=pft_present,
            pft_weights=pft_weights,
            start_year=args.year,
            start_month=args.month,
            completed_year=completed_year,
            completed_month=completed_month,
            completed_hours_last_month=last_n1h,
            next_year=next_year,
            next_month=next_month,
            calendar=args.regional_calendar,
            snow_model_enabled=snow_model_enabled,
            dtype=dtype,
            regional_max_hours=args.regional_max_hours,
        )

    # ------------------------------------------------------------------
    # 8. Summary.
    # ------------------------------------------------------------------
    if args.nmonths > 1:
        print()
        total_cs = ncell * total_nhours
        print(
            f"TOTAL  {total_nhours}h  {total_elapsed:.2f}s  "
            f"{total_cs / total_elapsed:.0f} cell-steps/s  "
            f"{1e6 * total_elapsed / total_cs:.2f} µs/cell-step"
        )
    if carbon_model is not None:
        print(
            f"Online carbon: {carbon_model.day_count} daily steps  "
            f"last GPP={carbon_model.daily_domain_gpp[-1] if carbon_model.daily_domain_gpp else float('nan'):.2f}  "
            f"last NPP={carbon_model.daily_domain_npp[-1] if carbon_model.daily_domain_npp else float('nan'):.2f}  "
            f"last HR={carbon_model.daily_domain_hr[-1] if carbon_model.daily_domain_hr else float('nan'):.2f}"
        )
    if regional_ts_records is not None and regional_ts_records["fpsn_patch"]:
        if args.regional_timeseries_output is not None:
            _write_regional_single_cell_timeseries(
                args.regional_timeseries_output,
                records=regional_ts_records,
                selected_cell_info=selected_cell_info,
            )
        if regional_feature_records is not None and regional_feature_records["hour_index"]:
            assert args.regional_feature_output is not None
            _write_regional_single_cell_timeseries(
                args.regional_feature_output,
                records=regional_feature_records,
                selected_cell_info=selected_cell_info,
            )
        if args.regional_cell_timeseries:
            ts = {
                key: np.stack(values, axis=0)
                for key, values in regional_ts_records.items()
                if values
            }
            plot_output = regional_cell_plot_output if regional_cell_plot_output is not None else args.plot_output
            plot_diagnostics(
                plot_output,
                ts["fpsn_patch"],
                ts["qflx_tran_veg"],
                {},
                dt=3600.0,
                daily_average=args.plot_daily_average,
                zwt=ts["zwt"],
                h2osfc=ts["h2osfc"],
                lai=ts["lai"],
                btran=ts["btran"],
                t_soil=ts["t_soil"],
                frac_h2osfc=ts["frac_h2osfc"],
                snow_depth=ts["snow_depth"],
                snow_depth_eff=ts["snow_depth_eff"],
                forc_t=ts["forc_t"],
                forc_pbot=ts["forc_pbot"],
            )

    # ------------------------------------------------------------------
    # 9. Diagnostic maps.
    # ------------------------------------------------------------------
    if (not args.no_plot) and (not single_cell_mode) and _diag_months > 0 and _all_diag:
        print("  saving aggregate diagnostic maps ...", flush=True)
        _plot_diag_maps(
            diag=_all_diag,
            n_months=_diag_months,
            grid_row=surf["grid_row"],
            grid_col=surf["grid_col"],
            nlat=int(surf.get("map_nlat", 361)), nlon=int(surf.get("map_nlon", 625)),
            lat_min=float(surf.get("map_lat_min", 25.0)), lat_max=float(surf.get("map_lat_max", 40.0)),
            lon_min=float(surf.get("map_lon_min", -100.0)), lon_max=float(surf.get("map_lon_max", -74.0)),
            out_dir=args.regional_plot_dir,
            label=f"{args.year}_{args.month:02d}_mean_{_diag_months}mo",
            map_lat_centers=surf.get("map_lat_centers"),
            map_lon_centers=surf.get("map_lon_centers"),
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    device = select_device(args.device)

    if not args.disable_snow_model and args.snow_grain_aging:
        snicar_path = _resolve_snicar_aging_file(
            args.snicar_aging_file,
            getattr(args, "clm_params", None),
        )
        if snicar_path is None or not snicar_path.is_file():
            requested = "auto-discovery" if args.snicar_aging_file is None else str(args.snicar_aging_file)
            raise SystemExit(
                "Snow grain aging is enabled, but the SNICAR lookup table was not found "
                f"({requested}). Supply --snicar-aging-file or use --no-snow-grain-aging."
            )
        load_snicar_aging_tables(snicar_path)
        if not snicar_aging_tables_loaded():
            raise SystemExit(f"Failed to load SNICAR grain-aging table: {snicar_path}")
        print(f"  SNICAR snow grain aging: {snicar_path}")

    def _dtype_from_name(name: str) -> "torch.dtype":
        if name == "float32":
            return torch.float32
        if name == "float64":
            return torch.float64
        if name == "float16":
            return torch.float16
        if name == "bfloat16":
            return torch.bfloat16
        raise SystemExit(f"Unsupported dtype {name!r}")

    dtype = _dtype_from_name(args.dtype)
    mlp_dtype = dtype if args.mlp_dtype == "same" else _dtype_from_name(args.mlp_dtype)
    if device.type == "mps" and (dtype == torch.float64 or mlp_dtype == torch.float64):
        raise SystemExit("Apple MPS does not support float64 tensors; use --dtype float32 and --mlp-dtype same/float32.")
    if args.mlp_dtype != "same" and not args.regional:
        print("  [mlp-dtype] --mlp-dtype currently applies to --regional replay only")
    if args.mlp_dtype != "same":
        print(f"  [mlp-dtype] canopy MLP dtype={mlp_dtype}; replay state/physics dtype={dtype}")
        if device.type != "cuda" and mlp_dtype in (torch.float16, torch.bfloat16):
            print("  [mlp-dtype] warning: CPU/MPS half/bfloat16 is not representative of H100 Tensor Cores")
    _auto_single_cell_cpu = (
        device.type == "cpu"
        and args.regional
        and (
            (args.lat is not None and args.lon is not None)
            or (args.regional_ncell is not None and args.regional_ncell <= 16)
        )
    )
    _torch_threads = args.torch_num_threads
    if _torch_threads is None and _auto_single_cell_cpu:
        _torch_threads = 1
    if _torch_threads is not None:
        _torch_threads = int(_torch_threads)
        if _torch_threads <= 0:
            raise SystemExit("--torch-num-threads must be positive")
        torch.set_num_threads(_torch_threads)
        try:
            torch.set_num_interop_threads(_torch_threads)
        except RuntimeError:
            pass
        print(f"  [torch] CPU threads set to {_torch_threads}")
    configure_tf32_matmul(args.tf32_matmul, device)
    snow_model_enabled = not args.disable_snow_model
    if args.pft is not None and not args.regional:
        raise SystemExit("--pft is only supported with --regional")

    # Auto-enable compile flags when running torch backends on CUDA.
    _torch_backend = (
        args.hydrology_backend in ("torch", "full")
        or args.soil_temperature_backend == "torch"
    )
    if args.compile is None:
        args.compile = _torch_backend and device.type == "cuda"
    if args.compile_physics is None:
        args.compile_physics = _torch_backend and device.type == "cuda"

    # --- Load model bundle (single-patch path only) ---
    if not args.blend_patches and not args.regional:
        model_file = args.model_dir / f"patch_{args.patch_index}" / "model.pt"
        bundle = load_surface_model_bundle(model_file, device=device, dtype=dtype, use_compile=args.compile)
    else:
        bundle = None  # loaded per-patch inside the blend / regional path below

    if args.compile_physics and not args.regional and hasattr(torch, "compile"):
        from . import elm_hydrology_torch as _hyd_mod
        from . import elm_soil_temperature_torch as _temp_mod
        # Compile the Thomas solvers with fullgraph=True: they are now
        # workspace-free and contain only Python index branches, so the
        # compiler can unroll the fixed-length loops and fuse all layer
        # operations into a single CUDA kernel instead of n separate ones.
        if device.type == "cuda":
            _hyd_mod.batched_tridiagonal_torch = torch.compile(  # type: ignore[attr-defined]
                _hyd_mod.batched_tridiagonal_torch, fullgraph=True, dynamic=True
            )
        else:
            print("  [torch.compile] leaving hydrology Thomas solver eager on CPU")
        if snow_model_enabled and not args.snow_scalar_only:
            print(
                "  [torch.compile] leaving soil-temperature Thomas solver eager "
                "because active snow changes the matrix width"
            )
        elif device.type == "cuda":
            _temp_mod._batched_tridiagonal_torch = torch.compile(  # type: ignore[attr-defined]
                _temp_mod._batched_tridiagonal_torch, fullgraph=True, dynamic=True
            )
        else:
            print("  [torch.compile] leaving soil-temperature Thomas solver eager on CPU")
        # Also compile the full physics step functions (fullgraph=False to
        # tolerate the tensor-conditional branches they still contain).
        _hyd_mod.hydrology_no_drainage_torch = torch.compile(  # type: ignore[attr-defined]
            _hyd_mod.hydrology_no_drainage_torch, fullgraph=False
        )
        _hyd_mod.drainage_torch = torch.compile(  # type: ignore[attr-defined]
            _hyd_mod.drainage_torch, fullgraph=False
        )
        _hyd_mod.hydrology_torch = torch.compile(  # type: ignore[attr-defined]
            _hyd_mod.hydrology_torch, fullgraph=False
        )
        _temp_mod.soil_temperature_step_torch = torch.compile(  # type: ignore[attr-defined]
            _temp_mod.soil_temperature_step_torch, fullgraph=False
        )
        if args.warmup_iters < 3:
            print(
                f"[warning] --compile-physics is active; at least 3 warmup passes are "
                f"recommended for torch.compile (got --warmup-iters {args.warmup_iters})"
            )

    # -----------------------------------------------------------------------
    # Blended-patch benchmark path
    # -----------------------------------------------------------------------
    if args.blend_patches:
        patch_ids, patch_weights = _select_nonzero_surfdata_patches(args.surfdata_file)
        if len(patch_ids) < 2:
            raise ValueError(
                "--blend-patches requires at least 2 nonzero PFT patches in surfdata "
                f"(found {len(patch_ids)} in {args.surfdata_file})"
            )
        year_msg = f" starting at year {args.start_year}" if args.start_year is not None else ""
        print(
            f"Blend mode: {len(patch_ids)} patches: "
            + ", ".join(f"patch_{int(pid)} ({w:.1%})" for pid, w in zip(patch_ids, patch_weights))
        )

        # --- Load per-patch data, build shared blended static + initial state ---
        patch_setups: list[dict] = []
        per_patch_rootfr: list[np.ndarray] = []
        per_patch_initial: list[CoupledColumnState] = []
        per_patch_static: list = []

        for pid, pw in zip(patch_ids, patch_weights):
            pid = int(pid)
            model_file_p = args.model_dir / f"patch_{pid}" / "model.pt"
            bundle_p = load_surface_model_bundle(
                model_file_p, device=device, dtype=dtype, use_compile=args.compile
            )
            print(f"  Loading {args.nsteps} timesteps for patch_{pid}{year_msg} ...")
            fn_p, _, prn_p, feats_p, _, params_p = load_training_slice(
                args.training_file, pid, args.nsteps, start_year=args.start_year
            )
            fmap_p = {name: i for i, name in enumerate(fn_p)}
            pmap_p = {name: i for i, name in enumerate(prn_p)}
            rootfr_p = np.array(
                [feats_p[0, fmap_p[f"rootfr_lev{i + 1}"]] for i in range(10)], dtype=np.float64
            )
            static_p, dz_p = _build_static(args.surfdata_file, rootfr_p, device, dtype, nlev_total=15)
            if "pft_type" in pmap_p:
                static_p.pft_type = float(params_p[0, pmap_p["pft_type"]])
            if "smpso" in pmap_p:
                static_p.smpso = float(params_p[0, pmap_p["smpso"]])
            if "smpsc" in pmap_p:
                static_p.smpsc = float(params_p[0, pmap_p["smpsc"]])
            initial_state_p = build_initial_state_1x1(fn_p, feats_p, static_p, dz_p)
            ens_params_p = sample_ensemble_params(
                fn_p, feats_p, bundle_p, args.ensemble_size, seed=args.ensemble_seed
            )
            per_patch_rootfr.append(rootfr_p)
            per_patch_initial.append(initial_state_p)
            per_patch_static.append((static_p, dz_p))
            patch_setups.append({
                "patch_id": pid,
                "weight": float(pw),
                "bundle": bundle_p,
                "fmap": fmap_p,
                "features_rows": feats_p,
                "ens_params": ens_params_p,
            })

        # Blend root fractions by area weight for the shared soil column.
        blended_rootfr = sum(
            patch_weights[i] * per_patch_rootfr[i] for i in range(len(patch_weights))
        )
        shared_static, shared_dz = _build_static(
            args.surfdata_file, blended_rootfr, device, dtype, nlev_total=15
        )
        # Use smpso/smpsc from first patch (column-level params, same for all patches).
        ref_static_p = per_patch_static[0][0]
        if ref_static_p.smpso is not None:
            shared_static.smpso = ref_static_p.smpso
        if ref_static_p.smpsc is not None:
            shared_static.smpsc = ref_static_p.smpsc

        # Blend initial states by area weight.
        def _blend_np_field(arrays, weights):
            return sum(w * np.asarray(a, dtype=np.float64) for w, a in zip(weights, arrays))

        def _safe(s, attr):
            v = getattr(s, attr, None)
            return v if v is not None else None

        shared_initial_state = CoupledColumnState(
            h2osoi_liq=_blend_np_field([s.h2osoi_liq for s in per_patch_initial], patch_weights),
            h2osoi_ice=_blend_np_field([s.h2osoi_ice for s in per_patch_initial], patch_weights),
            h2ocan=_blend_np_field([s.h2ocan for s in per_patch_initial], patch_weights),
            h2osfc=_blend_np_field([s.h2osfc for s in per_patch_initial], patch_weights),
            frac_h2osfc=_blend_np_field([s.frac_h2osfc for s in per_patch_initial], patch_weights),
            frac_sno=_blend_np_field([s.frac_sno for s in per_patch_initial], patch_weights),
            zwt=_blend_np_field([s.zwt for s in per_patch_initial], patch_weights),
            t_soisno=_blend_np_field([s.t_soisno for s in per_patch_initial], patch_weights),
            wa=_blend_np_field(
                [s.wa if s.wa is not None else np.zeros((1, 1)) for s in per_patch_initial],
                patch_weights,
            ),
            t_h2osfc=_blend_np_field(
                [s.t_h2osfc if s.t_h2osfc is not None else np.full((1, 1), 273.15) for s in per_patch_initial],
                patch_weights,
            ),
            t10=_blend_np_field(
                [s.t10 if s.t10 is not None else np.full((1, 1), 283.15) for s in per_patch_initial],
                patch_weights,
            ),
            t10_history=None,
            h2osno=_blend_np_field(
                [s.h2osno if s.h2osno is not None else np.zeros((1, 1)) for s in per_patch_initial],
                patch_weights,
            ),
            snow_depth=_blend_np_field(
                [s.snow_depth if s.snow_depth is not None else np.zeros((1, 1)) for s in per_patch_initial],
                patch_weights,
            ),
            snow_t=None,
            snow_layers=per_patch_initial[0].snow_layers,  # use first patch; blended h2osno ≈ 0 at init
        )

        n_patches = len(patch_setups)
        nsample = args.ncell * args.ensemble_size * args.nsteps

        _blended_pass_kw = dict(
            patch_weights=patch_weights,
            patch_statics=[s for s, _ in per_patch_static],
            static=shared_static,
            initial_state=shared_initial_state,
            ncell=args.ncell,
            nens=args.ensemble_size,
            dt=args.dt,
            hydrology_backend=args.hydrology_backend,
            soil_temperature_backend=args.soil_temperature_backend,
            device=device,
            dtype=dtype,
            snow_model_enabled=snow_model_enabled,
            snow_scalar_only=args.snow_scalar_only,
        )

        warmup_nsteps = args.warmup_nsteps if args.warmup_nsteps is not None else args.nsteps
        print(
            f"device={device.type} dtype={dtype} patches={n_patches} "
            f"hydrology={args.hydrology_backend} soil_temperature={args.soil_temperature_backend} "
            f"ncell={args.ncell} ensemble_size={args.ensemble_size} nsteps={args.nsteps}"
        )
        print(f"Warming up ({args.warmup_iters} pass(es), {warmup_nsteps} step(s)) ...")
        warmup_setups = [{**p, "features_rows": p["features_rows"][:warmup_nsteps]} for p in patch_setups]
        for _ in range(args.warmup_iters):
            run_blended_replay_pass(warmup_setups, **_blended_pass_kw)

        print(f"Timing ({args.timed_iters} pass(es)) ...")
        timings: list[float] = []
        sanity: dict[str, float] = {}
        prof: dict[str, float] | None = {} if args.profile_sections else None
        diag_outputs: dict | None = None
        for i in range(args.timed_iters):
            record = (not args.no_plot) and (i == args.timed_iters - 1)
            t0 = time.perf_counter()
            _, sanity, diag_outputs = run_blended_replay_pass(
                patch_setups, **_blended_pass_kw,
                profile_sections=prof, record_outputs=record,
            )
            timings.append(time.perf_counter() - t0)

        mean_s = float(np.mean(timings))
        median_s = float(np.median(timings))

        print()
        print(
            f"{'patches':>7} {'cells':>6} {'nens':>6} {'nsteps':>7} {'blnd_samp':>12}  "
            f"{'mean_s':>12} {'median_s':>12} {'blnd_samp/s':>14} {'us/blnd':>12}"
        )
        print("-" * 102)
        print(
            f"{n_patches:7d} {args.ncell:6d} {args.ensemble_size:6d} {args.nsteps:7d} {nsample:12d}  "
            f"{mean_s:12.6e} {median_s:12.6e} {nsample / mean_s:14.3f} {1.0e6 * mean_s / nsample:12.3f}"
        )
        if sanity:
            print("sanity: " + "  ".join(f"{k}={v:.6e}" for k, v in sanity.items()))
        if prof:
            total_prof = sum(prof.values())
            nsteps_total = args.nsteps * args.timed_iters
            print()
            print(f"Section timing ({nsteps_total} column-steps across {n_patches} patches blended, wall={mean_s * args.timed_iters:.3f}s):")
            print(f"  {'section':<30}  {'total_s':>10}  {'ms/step':>10}  {'%':>6}")
            for sec, secs in sorted(prof.items(), key=lambda kv: -kv[1]):
                pct = 100.0 * secs / max(total_prof, 1e-12)
                print(f"  {sec:<30}  {secs:10.4f}  {1000*secs/nsteps_total:10.3f}  {pct:6.1f}%")
            print(f"  {'(unaccounted)':<30}  {mean_s*args.timed_iters - total_prof:10.4f}")

        if not args.no_plot and diag_outputs is not None:
            ens_params_ref = patch_setups[0]["ens_params"]
            plot_diagnostics(
                args.plot_output,
                diag_outputs["fpsn_patch"],
                diag_outputs["qflx_tran_veg"],
                ens_params_ref,
                args.dt,
                daily_average=args.plot_daily_average,
                zwt=diag_outputs.get("zwt"),
                h2osfc=diag_outputs.get("h2osfc"),
                lai=diag_outputs.get("lai"),
                btran=diag_outputs.get("btran"),
            )
        return

    # -----------------------------------------------------------------------
    # Regional full-domain benchmark path
    # -----------------------------------------------------------------------
    if args.regional:
        _run_regional_benchmark(args, device, dtype, mlp_dtype)
        return

    # -----------------------------------------------------------------------
    # Single-patch benchmark path (original)
    # -----------------------------------------------------------------------

    # --- Load training forcing slice ---
    year_msg = f" starting at year {args.start_year}" if args.start_year is not None else ""
    print(f"Loading {args.nsteps} timesteps from {args.training_file}{year_msg} ...")
    feature_names, target_names, param_names, features_rows, targets_rows, params_rows = (
        load_training_slice(args.training_file, args.patch_index, args.nsteps, start_year=args.start_year)
    )
    fmap = {name: i for i, name in enumerate(feature_names)}
    pmap = {name: i for i, name in enumerate(param_names)}

    # --- Build static parameters from surfdata ---
    rootfr = np.array(
        [features_rows[0, fmap[f"rootfr_lev{i + 1}"]] for i in range(10)],
        dtype=np.float64,
    )
    static, dz = _build_static(args.surfdata_file, rootfr, device, dtype, nlev_total=15)
    if "pft_type" in pmap:
        static.pft_type = float(params_rows[0, pmap["pft_type"]])
    if "smpso" in pmap:
        static.smpso = float(params_rows[0, pmap["smpso"]])
    if "smpsc" in pmap:
        static.smpsc = float(params_rows[0, pmap["smpsc"]])

    # --- Build initial state (1, 1, nlev) then tile later in each pass ---
    initial_state_1x1 = build_initial_state_1x1(feature_names, features_rows, static, dz)

    # --- Sample unique per-ensemble flnr / mbbopt values ---
    ens_params = sample_ensemble_params(
        feature_names, features_rows, bundle, args.ensemble_size, seed=args.ensemble_seed
    )
    if ens_params:
        print("Ensemble parameter ranges:")
        for col, vals in ens_params.items():
            v = vals.reshape(-1)
            print(f"  {col}: [{v.min():.4g}, {v.max():.4g}]  (mean {v.mean():.4g})")

    nsample = args.ncell * args.ensemble_size * args.nsteps

    _pass_kw = dict(
        fmap=fmap,
        bundle=bundle,
        static=static,
        initial_state=initial_state_1x1,
        ncell=args.ncell,
        nens=args.ensemble_size,
        dt=args.dt,
        hydrology_backend=args.hydrology_backend,
        soil_temperature_backend=args.soil_temperature_backend,
        device=device,
        dtype=dtype,
        snow_model_enabled=snow_model_enabled,
        snow_scalar_only=args.snow_scalar_only,
        ens_params=ens_params,
    )

    # --- Warmup ---
    warmup_nsteps = args.warmup_nsteps if args.warmup_nsteps is not None else args.nsteps
    warmup_features = features_rows[:warmup_nsteps]
    print(
        f"device={device.type} dtype={dtype} patch={bundle.patch_index} "
        f"hydrology={args.hydrology_backend} soil_temperature={args.soil_temperature_backend} "
        f"ncell={args.ncell} ensemble_size={args.ensemble_size} nsteps={args.nsteps}"
    )
    print(f"Warming up ({args.warmup_iters} pass(es), {warmup_nsteps} step(s) each) ...")
    for _ in range(args.warmup_iters):
        run_replay_pass(warmup_features, **_pass_kw)

    # --- Timed passes ---
    print(f"Timing ({args.timed_iters} pass(es)) ...")
    timings: list[float] = []
    sanity: dict[str, float] = {}
    prof: dict[str, float] | None = {} if args.profile_sections else None
    diag_outputs: dict | None = None
    for i in range(args.timed_iters):
        record = (not args.no_plot) and (i == args.timed_iters - 1)
        t0 = time.perf_counter()
        _, sanity, diag_outputs = run_replay_pass(
            features_rows, **_pass_kw, profile_sections=prof, record_outputs=record
        )
        timings.append(time.perf_counter() - t0)

    mean_s = float(np.mean(timings))
    median_s = float(np.median(timings))

    print()
    print(
        f"{'cells':>6} {'nens':>6} {'nsteps':>7} {'samples':>12}  "
        f"{'mean_s':>12} {'median_s':>12} {'samples/s':>14} {'us/sample':>12}"
    )
    print("-" * 96)
    print(
        f"{args.ncell:6d} {args.ensemble_size:6d} {args.nsteps:7d} {nsample:12d}  "
        f"{mean_s:12.6e} {median_s:12.6e} {nsample / mean_s:14.3f} {1.0e6 * mean_s / nsample:12.3f}"
    )
    if sanity:
        print(
            "sanity: "
            + "  ".join(f"{k}={v:.6e}" for k, v in sanity.items())
        )
    if prof:
        total = sum(prof.values())
        nsteps = args.nsteps * args.timed_iters
        print()
        print(f"Section timing ({nsteps} total steps, wall={mean_s * args.timed_iters:.3f}s):")
        print(f"  {'section':<30}  {'total_s':>10}  {'ms/step':>10}  {'%':>6}")
        for sec, secs in sorted(prof.items(), key=lambda kv: -kv[1]):
            pct = 100.0 * secs / max(total, 1e-12)
            print(f"  {sec:<30}  {secs:10.4f}  {1000*secs/nsteps:10.3f}  {pct:6.1f}%")
        print(f"  {'(unaccounted)':<30}  {mean_s*args.timed_iters - total:10.4f}")

    # --- Diagnostic plot from last timed pass ---
    if not args.no_plot:
        if diag_outputs is not None:
            plot_diagnostics(
                args.plot_output,
                diag_outputs["fpsn_patch"],
                diag_outputs["qflx_tran_veg"],
                ens_params,
                args.dt,
                daily_average=args.plot_daily_average,
                zwt=diag_outputs.get("zwt"),
                h2osfc=diag_outputs.get("h2osfc"),
                lai=diag_outputs.get("lai"),
                btran=diag_outputs.get("btran"),
            )


if __name__ == "__main__":
    main()
