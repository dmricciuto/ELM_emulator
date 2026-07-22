#!/usr/bin/env python3
"""Couple the surface surrogate, no-drainage hydrology, and soil temperature.

This module keeps the interfaces explicit:

- surface surrogate: PyTorch bundle, driven by feature vectors in bundle order
- hydrology: NumPy or Torch no-drainage step
- soil temperature: NumPy or Torch non-snow, optional-standing-water step

The intended use is offline replay or large ensemble evaluation where the
surface surrogate provides:

- ``qflx_tran_veg``
- ``qflx_ev_soil``
- ``qflx_ev_h2osfc``
- ``hs_soil``
- ``hs_h2osfc``
- ``dhsdT``

and the soil column state is updated by the physical hydrology and thermal
solvers.
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass, field, replace
from functools import lru_cache
from math import erf
from pathlib import Path
from typing import Mapping

import numpy as np
from netCDF4 import Dataset

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover - runtime dependency
    raise SystemExit("PyTorch is required for the coupled driver.") from exc

from .elm_hydrology import (
    HydrologyForcing,
    HydrologyParameters,
    HydrologyResult,
    HydrologyState,
    SoilTextureProfile,
    elm_soil_grid,
    hydraulic_params_from_texture,
    hydrology_no_drainage_step,
    hydrology_step,
)
from .elm_hydrology_torch import (
    SoilHydrologyNoDrainageState,
    SoilHydrologyTorchResult,
    SoilHydrologyTorchWorkspace,
    SoilWaterCoreParameters,
    hydrology_torch,
    hydrology_no_drainage_torch,
)
from .elm_soil_temperature import (
    SoilTemperatureForcing,
    SoilTemperatureResult,
    SoilTemperatureState,
    SoilThermalParameters,
    soil_temperature_step,
    thermal_params_from_texture,
)
from .elm_soil_temperature_torch import (
    SoilTemperatureForcingTorch,
    SoilTemperatureResultTorch,
    SoilTemperatureStateTorch,
    SoilThermalParametersTorch,
    SoilTemperatureTorchWorkspace,
    soil_temperature_step_torch,
    thermal_params_from_texture_torch,
)
from .btran_utils import compute_btran_and_root_fraction, inverse_frac_h2osfc
from .elm_carbon_torch import (
    LitterFluxTorch,
    PlantStateTorch,
    SoilStateTorch,
    build_carbon_params_torch,
    carbon_plant_step_torch,
    carbon_soil_step_torch,
    elm_native_decomp_spinup_factors,
    init_plant_state_torch,
    init_soil_state_torch,
    rootfr_profile_torch,
)
from .paths import DEFAULT_MODEL_DIR


class PatchMLP(nn.Module):
    """Simple fully connected multi-output regressor."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_layers: tuple[int, ...],
        *,
        pft_embedding_dim: int = 0,
        pft_input_index: int | None = None,
        pft_embedding_categories: tuple[int, ...] | list[int] | None = None,
        pft_x_mean: float = 0.0,
        pft_x_std: float = 1.0,
    ) -> None:
        super().__init__()
        categories = tuple(sorted({int(value) for value in (pft_embedding_categories or ())}))
        use_pft_embedding = pft_embedding_dim > 0 and pft_input_index is not None and len(categories) > 0
        self.pft_input_index = int(pft_input_index) if use_pft_embedding else None
        self.pft_embedding_dim = int(pft_embedding_dim) if use_pft_embedding else 0
        if use_pft_embedding:
            self.pft_unknown_embedding_index = len(categories)
            self.pft_embedding = nn.Embedding(len(categories) + 1, self.pft_embedding_dim)
            max_category = max(max(categories), 0)
            lookup = torch.full(
                (max_category + 1,),
                self.pft_unknown_embedding_index,
                dtype=torch.long,
            )
            for embedding_index, category in enumerate(categories):
                if category >= 0:
                    lookup[category] = embedding_index
            self.register_buffer("pft_embedding_lookup", lookup, persistent=True)
            self.register_buffer("pft_x_mean", torch.tensor(float(pft_x_mean)), persistent=True)
            self.register_buffer("pft_x_std", torch.tensor(float(pft_x_std) if pft_x_std != 0.0 else 1.0), persistent=True)
        else:
            self.pft_unknown_embedding_index = None
            self.pft_embedding = None

        layers: list[nn.Module] = []
        prev_dim = input_dim + self.pft_embedding_dim
        for width in hidden_layers:
            layers.append(nn.Linear(prev_dim, width))
            layers.append(nn.GELU())
            prev_dim = width
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def _pft_embedding_indices(self, x: torch.Tensor) -> torch.Tensor:
        assert self.pft_input_index is not None
        assert self.pft_embedding is not None
        assert self.pft_unknown_embedding_index is not None
        raw_pft = torch.round(x[..., self.pft_input_index] * self.pft_x_std + self.pft_x_mean).to(torch.long)
        unknown = torch.full_like(raw_pft, self.pft_unknown_embedding_index)
        valid = (raw_pft >= 0) & (raw_pft < self.pft_embedding_lookup.numel())
        safe_raw = torch.where(valid, raw_pft, torch.zeros_like(raw_pft))
        return torch.where(valid, self.pft_embedding_lookup[safe_raw], unknown)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pft_embedding is not None:
            pft_embedding = self.pft_embedding(self._pft_embedding_indices(x))
            x_continuous = x.clone()
            assert self.pft_input_index is not None
            x_continuous[..., self.pft_input_index] = 0.0
            x = torch.cat((x_continuous, pft_embedding), dim=-1)
        return self.network(x)


@dataclass
class SurfaceModelBundle:
    patch_index: int
    input_names: list[str]
    target_names: list[str]
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: np.ndarray
    y_std: np.ndarray
    model: nn.Module
    # Per-target input masks: {target_name: [feature_name, ...]}
    # Applied at inference — these features are zeroed before predicting the named target.
    target_feature_masks: dict[str, list[str]] | None = None
    log1p_target_names: list[str] | None = None
    x_mean_torch: torch.Tensor | None = None
    x_std_safe_torch: torch.Tensor | None = None
    x_float32_constant_mask_torch: torch.Tensor | None = None
    y_mean_torch: torch.Tensor | None = None
    y_std_torch: torch.Tensor | None = None


@dataclass
class CoupledColumnState:
    """Dynamic state shared by the hydrology and soil-temperature pieces."""

    h2osoi_liq: np.ndarray
    h2osoi_ice: np.ndarray
    h2ocan: np.ndarray
    h2osfc: np.ndarray
    frac_h2osfc: np.ndarray
    frac_sno: np.ndarray
    zwt: np.ndarray
    t_soisno: np.ndarray
    wa: np.ndarray | None = None
    t_h2osfc: np.ndarray | None = None
    t10: np.ndarray | None = None
    t10_history: np.ndarray | None = None
    h2osno: np.ndarray | None = None
    snow_depth: np.ndarray | None = None
    snow_t: np.ndarray | None = None
    snow_layers: "SnowLayerState" | None = None
    int_snow: np.ndarray | torch.Tensor | None = None
    # Per-timestep btran/rootr memoization: {id(static): (btran, rootr)}
    # init=False so it is never part of __init__ or equality checks.
    _btran_cache: dict | None = field(default=None, init=False, repr=False, compare=False)


@dataclass
class SnowLayerState:
    """Persistent layered snow storage used to seed the thermal solve."""

    snl: np.ndarray | torch.Tensor
    dz: np.ndarray | torch.Tensor
    h2osoi_ice: np.ndarray | torch.Tensor
    h2osoi_liq: np.ndarray | torch.Tensor
    t_snow: np.ndarray | torch.Tensor
    snw_rds: np.ndarray | torch.Tensor | None = None  # effective grain radius [µm]


@dataclass
class SnowWaterResult:
    """Outcome of the simplified SnowWater + SnowCapping update."""

    state: CoupledColumnState
    qflx_snow_to_soil: np.ndarray | torch.Tensor
    qflx_snwcp_ice: np.ndarray | torch.Tensor
    qflx_snwcp_liq: np.ndarray | torch.Tensor


def native_root_parameter_index_for_pft(pft_id: int, nparam: int) -> int:
    """Parameter-file index used for native-replay roota/rootb compatibility."""
    root_idx = int(pft_id)
    if root_idx < 0 or root_idx >= nparam:
        raise IndexError(
            f"root parameter index {root_idx} for PFT {pft_id} is outside "
            f"the parameter-file PFT dimension [0, {nparam - 1}]"
        )
    return root_idx


def _zero_snow_state(state: CoupledColumnState) -> CoupledColumnState:
    """Return a copy of *state* with snow storage removed."""
    def zero_like(value: np.ndarray | torch.Tensor | None) -> np.ndarray | torch.Tensor | None:
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            return torch.zeros_like(value)
        return np.zeros_like(np.asarray(value, dtype=np.float64))

    return CoupledColumnState(
        h2osoi_liq=state.h2osoi_liq,
        h2osoi_ice=state.h2osoi_ice,
        h2ocan=state.h2ocan,
        h2osno=zero_like(state.h2osno) if state.h2osno is not None else zero_like(state.h2osfc),
        snow_depth=zero_like(state.snow_depth) if state.snow_depth is not None else zero_like(state.h2osfc),
        h2osfc=state.h2osfc,
        frac_h2osfc=state.frac_h2osfc,
        frac_sno=zero_like(state.frac_sno) if state.frac_sno is not None else zero_like(state.h2osfc),
        zwt=state.zwt,
        t_soisno=state.t_soisno,
        wa=state.wa,
        t_h2osfc=state.t_h2osfc,
        t10=state.t10,
        t10_history=state.t10_history,
        snow_t=None,
        snow_layers=None,
        int_snow=zero_like(state.int_snow) if state.int_snow is not None else zero_like(state.h2osfc),
    )


def _repair_snow_fraction_depth_numpy(state: CoupledColumnState) -> CoupledColumnState:
    """Keep snow mass, fraction, depth, and layers in a physically consistent range."""
    if state.h2osno is None:
        return state
    h2osno_raw = _to_np(state.h2osno)
    h2osno = np.where(np.isfinite(h2osno_raw), np.maximum(h2osno_raw, 0.0), 0.0)
    frac = (
        _to_np(state.frac_sno)
        if state.frac_sno is not None
        else np.zeros_like(h2osno, dtype=np.float64)
    )
    depth = (
        _to_np(state.snow_depth)
        if state.snow_depth is not None
        else snow_depth_from_storage(h2osno)
    )
    frac_finite = np.isfinite(frac)
    depth_finite = np.isfinite(depth)
    mass_active = h2osno > 1.0e-8
    frac_clipped = np.clip(np.where(frac_finite, frac, 0.0), 0.0, 1.0)
    depth_clean = np.where(depth_finite, np.maximum(depth, 0.0), 0.0)
    effective_depth = frac_clipped * depth_clean
    max_effective_depth = np.maximum(
        h2osno / SNOW_GEOMETRY_MIN_DENSITY * SNOW_GEOMETRY_DEPTH_MARGIN,
        snow_depth_from_storage(h2osno) + 1.0e-6,
    )
    bad = (
        ~np.isfinite(h2osno_raw)
        | (h2osno_raw < 0.0)
        | (~frac_finite)
        | (~depth_finite)
        | (frac < -1.0e-8)
        | (frac > 1.0 + 1.0e-8)
        | (depth < -1.0e-8)
        | (mass_active & (frac <= 1.0e-8))
        | (mass_active & (depth_clean > SNOW_GEOMETRY_MAX_DEPTH))
        | (mass_active & (effective_depth > max_effective_depth))
        | (~mass_active & ((depth_clean > 1.0e-10) | (frac_clipped > 1.0e-10)))
    )
    snow_shape = tuple(h2osno.shape)
    snow_layers = state.snow_layers if _snow_layers_shape_ok(state.snow_layers, snow_shape) else None
    if state.snow_layers is not None and snow_layers is None:
        bad = np.ones_like(bad, dtype=bool)
    if snow_layers is not None:
        layer_depth = _snow_depth_from_layers_numpy(snow_layers)
        layer_mass = _snow_mass_from_layers_numpy(snow_layers)
        layer_depth_clean = np.where(np.isfinite(layer_depth), np.maximum(layer_depth, 0.0), 0.0)
        layer_effective_depth = frac_clipped * layer_depth_clean
        layer_bad = (
            ~np.isfinite(layer_depth)
            | ~np.isfinite(layer_mass)
            | (layer_depth < -1.0e-8)
            | (layer_mass < -1.0e-8)
            | (mass_active & (layer_depth_clean > SNOW_GEOMETRY_MAX_DEPTH))
            | (mass_active & (layer_effective_depth > max_effective_depth))
            | (~mass_active & ((layer_depth_clean > 1.0e-10) | (np.maximum(layer_mass, 0.0) > 1.0e-10)))
        )
        bad = bad | layer_bad
    if not np.any(bad):
        return state
    fixed_depth = snow_depth_from_storage(h2osno)
    fixed_frac = frac_sno_from_snow_storage(h2osno, fixed_depth)
    depth = np.where(bad, fixed_depth, depth_clean)
    frac = np.where(bad, fixed_frac, frac_clipped)
    snow_t_seed = state.snow_t
    if snow_t_seed is not None and _shape_tuple(snow_t_seed) != snow_shape + (MAX_SNOW_LAYERS,):
        snow_t_seed = None
    if snow_t_seed is None and snow_layers is not None:
        snow_t_seed = snow_layers.t_snow
    fixed_layers = _snow_layers_from_storage_numpy(
        h2osno,
        fixed_depth,
        snow_t=_to_np(snow_t_seed) if snow_t_seed is not None else None,
        frac_sno=fixed_frac,
    )
    if snow_layers is not None:
        base_layers = _snow_layers_to_cpu(snow_layers)
        assert base_layers is not None
        layer_mask = bad[..., None]
        layers = SnowLayerState(
            snl=np.where(bad, fixed_layers.snl, base_layers.snl),
            dz=np.where(layer_mask, fixed_layers.dz, base_layers.dz),
            h2osoi_ice=np.where(layer_mask, fixed_layers.h2osoi_ice, base_layers.h2osoi_ice),
            h2osoi_liq=np.where(layer_mask, fixed_layers.h2osoi_liq, base_layers.h2osoi_liq),
            t_snow=np.where(layer_mask, fixed_layers.t_snow, base_layers.t_snow),
            snw_rds=(
                np.where(layer_mask, fixed_layers.snw_rds, base_layers.snw_rds)
                if base_layers.snw_rds is not None and fixed_layers.snw_rds is not None
                else fixed_layers.snw_rds
            ),
        )
    else:
        layers = fixed_layers
    int_snow = _to_np(state.int_snow) if state.int_snow is not None else None
    if int_snow is not None:
        int_snow = np.where(
            h2osno > 1.0e-8,
            np.maximum(np.where(np.isfinite(int_snow), int_snow, 0.0), h2osno),
            0.0,
        )
    return CoupledColumnState(
        h2osoi_liq=_to_np(state.h2osoi_liq),
        h2osoi_ice=_to_np(state.h2osoi_ice),
        h2ocan=_to_np(state.h2ocan),
        h2osfc=_to_np(state.h2osfc),
        frac_h2osfc=_to_np(state.frac_h2osfc),
        frac_sno=np.clip(frac, 0.0, 1.0),
        zwt=_to_np(state.zwt),
        t_soisno=_to_np(state.t_soisno),
        wa=_to_np(state.wa) if state.wa is not None else None,
        t_h2osfc=_to_np(state.t_h2osfc) if state.t_h2osfc is not None else None,
        t10=_to_np(state.t10) if state.t10 is not None else None,
        t10_history=_to_np(state.t10_history) if state.t10_history is not None else None,
        h2osno=h2osno,
        snow_depth=depth,
        snow_t=_to_np(state.snow_t) if state.snow_t is not None else None,
        snow_layers=layers,
        int_snow=int_snow,
    )


def _repair_snow_fraction_depth_torch(
    state: CoupledColumnState,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> CoupledColumnState:
    """Torch analogue of _repair_snow_fraction_depth_numpy."""
    if state.h2osno is None:
        return state
    h2osno_raw = _to_torch(state.h2osno, device, dtype)
    h2osno = torch.where(torch.isfinite(h2osno_raw), torch.clamp(h2osno_raw, min=0.0), torch.zeros_like(h2osno_raw))
    frac = (
        _to_torch(state.frac_sno, device, dtype)
        if state.frac_sno is not None
        else torch.zeros_like(h2osno)
    )
    depth = (
        _to_torch(state.snow_depth, device, dtype)
        if state.snow_depth is not None
        else snow_depth_from_storage_torch(h2osno, device=device, dtype=dtype)
    )
    frac_finite = torch.isfinite(frac)
    depth_finite = torch.isfinite(depth)
    mass_active = h2osno > 1.0e-8
    frac_clipped = torch.clamp(torch.where(frac_finite, frac, torch.zeros_like(frac)), min=0.0, max=1.0)
    depth_clean = torch.where(depth_finite, torch.clamp(depth, min=0.0), torch.zeros_like(depth))
    effective_depth = frac_clipped * depth_clean
    max_effective_depth = torch.maximum(
        h2osno / SNOW_GEOMETRY_MIN_DENSITY * SNOW_GEOMETRY_DEPTH_MARGIN,
        snow_depth_from_storage_torch(h2osno, device=device, dtype=dtype) + 1.0e-6,
    )
    bad = (
        ~torch.isfinite(h2osno_raw)
        | (h2osno_raw < 0.0)
        | (~frac_finite)
        | (~depth_finite)
        | (frac < -1.0e-8)
        | (frac > 1.0 + 1.0e-8)
        | (depth < -1.0e-8)
        | (mass_active & (frac <= 1.0e-8))
        | (mass_active & (depth_clean > SNOW_GEOMETRY_MAX_DEPTH))
        | (mass_active & (effective_depth > max_effective_depth))
        | (~mass_active & ((depth_clean > 1.0e-10) | (frac_clipped > 1.0e-10)))
    )
    snow_shape = tuple(h2osno.shape)
    snow_layers = state.snow_layers if _snow_layers_shape_ok(state.snow_layers, snow_shape) else None
    if state.snow_layers is not None and snow_layers is None:
        bad = torch.ones_like(bad, dtype=torch.bool)
    if snow_layers is not None:
        layer_depth = _snow_depth_from_layers_torch(snow_layers, device=device, dtype=dtype)
        layer_mass = _snow_mass_from_layers_torch(snow_layers, device=device, dtype=dtype)
        layer_depth_clean = torch.where(torch.isfinite(layer_depth), torch.clamp(layer_depth, min=0.0), torch.zeros_like(layer_depth))
        layer_effective_depth = frac_clipped * layer_depth_clean
        layer_bad = (
            ~torch.isfinite(layer_depth)
            | ~torch.isfinite(layer_mass)
            | (layer_depth < -1.0e-8)
            | (layer_mass < -1.0e-8)
            | (mass_active & (layer_depth_clean > SNOW_GEOMETRY_MAX_DEPTH))
            | (mass_active & (layer_effective_depth > max_effective_depth))
            | (~mass_active & ((layer_depth_clean > 1.0e-10) | (torch.clamp(layer_mass, min=0.0) > 1.0e-10)))
        )
        bad = bad | layer_bad
    if not bool(torch.any(bad).detach().cpu().item()):
        return state
    fixed_depth = snow_depth_from_storage_torch(h2osno, device=device, dtype=dtype)
    fixed_frac = frac_sno_from_snow_storage_torch(h2osno, fixed_depth, device=device, dtype=dtype)
    depth = torch.where(bad, fixed_depth, depth_clean)
    frac = torch.where(bad, fixed_frac, frac_clipped)
    snow_t_seed = state.snow_t
    if snow_t_seed is not None and _shape_tuple(snow_t_seed) != snow_shape + (MAX_SNOW_LAYERS,):
        snow_t_seed = None
    if snow_t_seed is None and snow_layers is not None:
        snow_t_seed = snow_layers.t_snow
    fixed_layers = _snow_layers_from_storage_torch(
        h2osno,
        fixed_depth,
        snow_t=_to_torch(snow_t_seed, device, dtype) if snow_t_seed is not None else None,
        frac_sno=fixed_frac,
        device=device,
        dtype=dtype,
    )
    if snow_layers is not None:
        base_layers = _snow_layers_to_device(snow_layers, device, dtype)
        assert base_layers is not None
        layer_mask = bad[..., None]
        layers = SnowLayerState(
            snl=torch.where(bad, fixed_layers.snl, base_layers.snl),
            dz=torch.where(layer_mask, fixed_layers.dz, base_layers.dz),
            h2osoi_ice=torch.where(layer_mask, fixed_layers.h2osoi_ice, base_layers.h2osoi_ice),
            h2osoi_liq=torch.where(layer_mask, fixed_layers.h2osoi_liq, base_layers.h2osoi_liq),
            t_snow=torch.where(layer_mask, fixed_layers.t_snow, base_layers.t_snow),
            snw_rds=(
                torch.where(layer_mask, fixed_layers.snw_rds, base_layers.snw_rds)
                if base_layers.snw_rds is not None and fixed_layers.snw_rds is not None
                else fixed_layers.snw_rds
            ),
        )
    else:
        layers = fixed_layers
    int_snow = _to_torch(state.int_snow, device, dtype) if state.int_snow is not None else None
    if int_snow is not None:
        int_snow = torch.where(
            h2osno > 1.0e-8,
            torch.maximum(torch.where(torch.isfinite(int_snow), int_snow, torch.zeros_like(int_snow)), h2osno),
            torch.zeros_like(h2osno),
        )
    return CoupledColumnState(
        h2osoi_liq=_to_torch(state.h2osoi_liq, device, dtype),
        h2osoi_ice=_to_torch(state.h2osoi_ice, device, dtype),
        h2ocan=_to_torch(state.h2ocan, device, dtype),
        h2osfc=_to_torch(state.h2osfc, device, dtype),
        frac_h2osfc=_to_torch(state.frac_h2osfc, device, dtype),
        frac_sno=torch.clamp(frac, min=0.0, max=1.0),
        zwt=_to_torch(state.zwt, device, dtype),
        t_soisno=_to_torch(state.t_soisno, device, dtype),
        wa=_to_torch(state.wa, device, dtype) if state.wa is not None else None,
        t_h2osfc=_to_torch(state.t_h2osfc, device, dtype) if state.t_h2osfc is not None else None,
        t10=_to_torch(state.t10, device, dtype) if state.t10 is not None else None,
        t10_history=_to_torch(state.t10_history, device, dtype) if state.t10_history is not None else None,
        h2osno=h2osno,
        snow_depth=depth,
        snow_t=_to_torch(state.snow_t, device, dtype) if state.snow_t is not None else None,
        snow_layers=layers,
        int_snow=int_snow,
    )


def _drain_inactive_snow_storage_numpy(
    state: CoupledColumnState,
) -> tuple[CoupledColumnState, np.ndarray]:
    """Move invalid zero-depth snow mass to the soil-drainage flux."""
    if state.h2osno is None:
        return state, np.zeros_like(_to_np(state.h2osfc), dtype=np.float64)

    h2osno_raw = _to_np(state.h2osno)
    h2osno = np.where(np.isfinite(h2osno_raw), np.maximum(h2osno_raw, 0.0), 0.0)
    depth_raw = (
        _to_np(state.snow_depth)
        if state.snow_depth is not None
        else snow_depth_from_storage(h2osno)
    )
    depth = np.where(np.isfinite(depth_raw), np.maximum(depth_raw, 0.0), 0.0)
    frac_raw = (
        _to_np(state.frac_sno)
        if state.frac_sno is not None
        else np.zeros_like(h2osno, dtype=np.float64)
    )
    frac = np.clip(np.where(np.isfinite(frac_raw), frac_raw, 0.0), 0.0, 1.0)
    mass_present = h2osno > 1.0e-8
    clear = (~np.isfinite(h2osno_raw)) | (h2osno_raw < 0.0) | (mass_present & (depth <= 1.0e-10))
    if not np.any(clear):
        return state, np.zeros_like(h2osno, dtype=np.float64)

    drained = np.where(clear & mass_present, h2osno, 0.0)
    h2osno_new = np.where(clear, 0.0, h2osno)
    depth_new = np.where(h2osno_new > 1.0e-8, depth, 0.0)
    frac_new = np.where(h2osno_new > 1.0e-8, frac, 0.0)
    int_snow = _to_np(state.int_snow) if state.int_snow is not None else None
    int_snow_new = (
        np.where(h2osno_new > 1.0e-8, np.maximum(np.where(np.isfinite(int_snow), int_snow, 0.0), h2osno_new), 0.0)
        if int_snow is not None
        else None
    )

    snow_layers = state.snow_layers
    if _snow_layers_shape_ok(snow_layers, tuple(h2osno.shape)):
        assert snow_layers is not None
        layer_mask = clear[..., None]
        base_rds = getattr(snow_layers, "snw_rds", None)
        snow_layers = SnowLayerState(
            snl=np.where(clear, 0, np.asarray(snow_layers.snl, dtype=np.int32)),
            dz=np.where(layer_mask, 0.0, _to_np(snow_layers.dz)),
            h2osoi_ice=np.where(layer_mask, 0.0, _to_np(snow_layers.h2osoi_ice)),
            h2osoi_liq=np.where(layer_mask, 0.0, _to_np(snow_layers.h2osoi_liq)),
            t_snow=np.where(layer_mask, TFRZ, _to_np(snow_layers.t_snow)),
            snw_rds=(
                np.where(layer_mask, 0.0, _to_np(base_rds))
                if base_rds is not None
                else None
            ),
        )

    snow_t_new = _to_np(state.snow_t) if state.snow_t is not None else None
    if snow_t_new is not None:
        if tuple(snow_t_new.shape) == tuple(h2osno.shape) + (MAX_SNOW_LAYERS,):
            snow_t_new = np.where(clear[..., None], TFRZ, snow_t_new)
        elif tuple(snow_t_new.shape) == tuple(h2osno.shape):
            snow_t_new = np.where(clear, TFRZ, snow_t_new)

    return (
        CoupledColumnState(
            h2osoi_liq=_to_np(state.h2osoi_liq),
            h2osoi_ice=_to_np(state.h2osoi_ice),
            h2ocan=_to_np(state.h2ocan),
            h2osfc=_to_np(state.h2osfc),
            frac_h2osfc=_to_np(state.frac_h2osfc),
            frac_sno=frac_new,
            zwt=_to_np(state.zwt),
            t_soisno=_to_np(state.t_soisno),
            wa=_to_np(state.wa) if state.wa is not None else None,
            t_h2osfc=_to_np(state.t_h2osfc) if state.t_h2osfc is not None else None,
            t10=_to_np(state.t10) if state.t10 is not None else None,
            t10_history=_to_np(state.t10_history) if state.t10_history is not None else None,
            h2osno=h2osno_new,
            snow_depth=depth_new,
            snow_t=snow_t_new,
            snow_layers=snow_layers,
            int_snow=int_snow_new,
        ),
        drained,
    )


def _drain_inactive_snow_storage_torch(
    state: CoupledColumnState,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[CoupledColumnState, torch.Tensor]:
    """Torch analogue of _drain_inactive_snow_storage_numpy."""
    if state.h2osno is None:
        ref = _to_torch(state.h2osfc, device, dtype)
        return state, torch.zeros_like(ref)

    h2osno_raw = _to_torch(state.h2osno, device, dtype)
    h2osno = torch.where(torch.isfinite(h2osno_raw), torch.clamp(h2osno_raw, min=0.0), torch.zeros_like(h2osno_raw))
    depth_raw = (
        _to_torch(state.snow_depth, device, dtype)
        if state.snow_depth is not None
        else snow_depth_from_storage_torch(h2osno, device=device, dtype=dtype)
    )
    depth = torch.where(torch.isfinite(depth_raw), torch.clamp(depth_raw, min=0.0), torch.zeros_like(depth_raw))
    frac_raw = (
        _to_torch(state.frac_sno, device, dtype)
        if state.frac_sno is not None
        else torch.zeros_like(h2osno)
    )
    frac = torch.clamp(torch.where(torch.isfinite(frac_raw), frac_raw, torch.zeros_like(frac_raw)), min=0.0, max=1.0)
    mass_present = h2osno > 1.0e-8
    clear = (~torch.isfinite(h2osno_raw)) | (h2osno_raw < 0.0) | (mass_present & (depth <= 1.0e-10))
    if not bool(torch.any(clear).detach().cpu().item()):
        return state, torch.zeros_like(h2osno)

    drained = torch.where(clear & mass_present, h2osno, torch.zeros_like(h2osno))
    h2osno_new = torch.where(clear, torch.zeros_like(h2osno), h2osno)
    depth_new = torch.where(h2osno_new > 1.0e-8, depth, torch.zeros_like(depth))
    frac_new = torch.where(h2osno_new > 1.0e-8, frac, torch.zeros_like(frac))
    int_snow = _to_torch(state.int_snow, device, dtype) if state.int_snow is not None else None
    int_snow_new = (
        torch.where(
            h2osno_new > 1.0e-8,
            torch.maximum(torch.where(torch.isfinite(int_snow), int_snow, torch.zeros_like(int_snow)), h2osno_new),
            torch.zeros_like(h2osno_new),
        )
        if int_snow is not None
        else None
    )

    snow_layers = state.snow_layers
    if _snow_layers_shape_ok(snow_layers, tuple(h2osno.shape)):
        assert snow_layers is not None
        layer_mask = clear[..., None]
        base_rds = getattr(snow_layers, "snw_rds", None)
        snow_layers = SnowLayerState(
            snl=torch.where(
                clear,
                torch.zeros_like(torch.as_tensor(snow_layers.snl, device=device, dtype=torch.int32)),
                torch.as_tensor(snow_layers.snl, device=device, dtype=torch.int32),
            ),
            dz=torch.where(layer_mask, torch.zeros_like(_to_torch(snow_layers.dz, device, dtype)), _to_torch(snow_layers.dz, device, dtype)),
            h2osoi_ice=torch.where(layer_mask, torch.zeros_like(_to_torch(snow_layers.h2osoi_ice, device, dtype)), _to_torch(snow_layers.h2osoi_ice, device, dtype)),
            h2osoi_liq=torch.where(layer_mask, torch.zeros_like(_to_torch(snow_layers.h2osoi_liq, device, dtype)), _to_torch(snow_layers.h2osoi_liq, device, dtype)),
            t_snow=torch.where(layer_mask, torch.full_like(_to_torch(snow_layers.t_snow, device, dtype), TFRZ), _to_torch(snow_layers.t_snow, device, dtype)),
            snw_rds=(
                torch.where(layer_mask, torch.zeros_like(_to_torch(base_rds, device, dtype)), _to_torch(base_rds, device, dtype))
                if base_rds is not None
                else None
            ),
        )

    snow_t_new = _to_torch(state.snow_t, device, dtype) if state.snow_t is not None else None
    if snow_t_new is not None:
        if tuple(snow_t_new.shape) == tuple(h2osno.shape) + (MAX_SNOW_LAYERS,):
            snow_t_new = torch.where(clear[..., None], torch.full_like(snow_t_new, TFRZ), snow_t_new)
        elif tuple(snow_t_new.shape) == tuple(h2osno.shape):
            snow_t_new = torch.where(clear, torch.full_like(snow_t_new, TFRZ), snow_t_new)

    return (
        CoupledColumnState(
            h2osoi_liq=_to_torch(state.h2osoi_liq, device, dtype),
            h2osoi_ice=_to_torch(state.h2osoi_ice, device, dtype),
            h2ocan=_to_torch(state.h2ocan, device, dtype),
            h2osfc=_to_torch(state.h2osfc, device, dtype),
            frac_h2osfc=_to_torch(state.frac_h2osfc, device, dtype),
            frac_sno=frac_new,
            zwt=_to_torch(state.zwt, device, dtype),
            t_soisno=_to_torch(state.t_soisno, device, dtype),
            wa=_to_torch(state.wa, device, dtype) if state.wa is not None else None,
            t_h2osfc=_to_torch(state.t_h2osfc, device, dtype) if state.t_h2osfc is not None else None,
            t10=_to_torch(state.t10, device, dtype) if state.t10 is not None else None,
            t10_history=_to_torch(state.t10_history, device, dtype) if state.t10_history is not None else None,
            h2osno=h2osno_new,
            snow_depth=depth_new,
            snow_t=snow_t_new,
            snow_layers=snow_layers,
            int_snow=int_snow_new,
        ),
        drained,
    )


@dataclass
class AtmosphericForcingInputs:
    forc_t: np.ndarray
    forc_q: np.ndarray
    forc_pbot: np.ndarray
    forc_lwrad: np.ndarray
    forc_rain: np.ndarray
    forc_snow: np.ndarray
    forc_swrad: np.ndarray
    forc_solad_vis: np.ndarray
    forc_solad_nir: np.ndarray
    forc_solai_vis: np.ndarray
    forc_solai_nir: np.ndarray
    wind_speed: np.ndarray
    forc_rho: np.ndarray
    forc_pco2: np.ndarray
    dayl: np.ndarray
    max_dayl: np.ndarray
    coszen_col: np.ndarray


@dataclass
class VegetationSurfaceInputs:
    frac_veg_nosno_patch: np.ndarray
    elai_patch: np.ndarray
    esai_patch: np.ndarray
    snow_depth: np.ndarray
    t10: np.ndarray
    htop_patch: np.ndarray | None = None


@dataclass
class SurfaceGeometryInputs:
    forc_hgt_u_patch: np.ndarray
    z0mg_col: np.ndarray
    ugust: np.ndarray


@dataclass
class CoupledStaticParameters:
    """Static column parameters and both backend-specific parameter objects."""

    rootfr: np.ndarray
    hydrology_numpy: HydrologyParameters
    hydrology_torch: SoilWaterCoreParameters
    soil_temperature_numpy: SoilThermalParameters
    soil_temperature_torch: SoilThermalParametersTorch
    micro_sigma: float
    pc: float = 0.4
    topo_slope: float | None = None
    pft_type: float | None = None
    smpso: float | None = None
    smpsc: float | None = None
    soil_color: float | np.ndarray | torch.Tensor | None = None
    leafcn: float | None = None
    slatop: float | None = None
    flnr: float | None = None
    mbbopt: float | None = None
    z0mr: float | None = None
    rholvis: float | None = None
    rholnir: float | None = None
    taulvis: float | None = None
    taulnir: float | None = None
    rhosvis: float | None = None
    rhosnir: float | None = None
    tausvis: float | None = None
    tausnir: float | None = None
    xl: float | None = None
    tc_stress_c: float = -2.0
    n_melt: float = 10.0
    rootfr_torch: torch.Tensor | None = None
    rootfr_norm_torch: torch.Tensor | None = None


@dataclass
class CoupledStepResult:
    """Outputs from one coupled surface -> hydrology -> soil-temperature step."""

    surface_targets: dict[str, np.ndarray]
    hydrology_backend: str
    soil_temperature_backend: str
    hydrology_result: HydrologyResult | SoilHydrologyTorchResult
    soil_temperature_result: SoilTemperatureResult | SoilTemperatureResultTorch
    qflx_snow_to_soil: np.ndarray | torch.Tensor | None
    state: CoupledColumnState
    canopy_state: CoupledColumnState | None = None
    snow_state: CoupledColumnState | None = None
    thermal_state: CoupledColumnState | None = None
    snow_water_result: SnowWaterResult | None = None


@dataclass
class SurfdataSiteInfo:
    gridcell: int
    lat: float
    lon_deg_east: float
    area_km2: float
    slope: float
    std_elev_m: float
    micro_sigma_m: float
    pc: float
    fmax: float
    zwt0_m: float | None
    soil_color: float | None
    sand_pct: np.ndarray
    clay_pct: np.ndarray
    organic_density: np.ndarray
    organic_frac: np.ndarray
    dominant_natpft_index: int | None = None
    dominant_natpft_pct: float | None = None
    monthly_lai: np.ndarray | None = None
    monthly_sai: np.ndarray | None = None
    monthly_height_top: np.ndarray | None = None
    monthly_height_bot: np.ndarray | None = None


REQUIRED_SURFACE_TARGETS = (
    "qflx_tran_veg",
    "qflx_ev_soil",
    "qflx_ev_h2osfc",
    "hs_soil",
    "hs_h2osfc",
    "dhsdT",
)

OPTIONAL_SURFACE_TARGETS = ("hs_top_snow",)

NONNEGATIVE_SURFACE_TARGETS = {
    "fpsn_patch",
    "laisun_patch",
    "laisha_patch",
    "psnsun_patch",
    "psnsha_patch",
    "qflx_tran_veg",
    "qflx_prec_grnd",
}

SIGNED_EVAPORATION_SURFACE_TARGETS = {
    "qflx_evap_canopy",
    "qflx_ev_soil",
    "qflx_ev_snow",
    "qflx_ev_h2osfc",
}

LOG1P_SURFACE_TARGETS = {
    "gh_canopy",
    "psnsun_patch",
}
DEFAULT_LOG1P_SURFACE_TARGETS = tuple(sorted(LOG1P_SURFACE_TARGETS))

PHOTOSYNTHESIS_SURFACE_TARGETS = ("psnsun_patch", "psnsha_patch", "fpsn_patch")
PATCH0_ZERO_SURFACE_TARGETS = {
    "fpsn_patch",
    "psnsun_patch",
    "psnsha_patch",
    "laisun_patch",
    "laisha_patch",
    "qflx_tran_veg",
    "qflx_evap_canopy",
}
FPSN_LAI_CUTOFF = 1.0e-2
ZERO_RADIATION_THRESHOLD = 1.0e-12
ZERO_BTRAN_THRESHOLD = 1.0e-2
H2OSFC_ACTIVE_FRACTION_THRESHOLD = 1.0e-8

# The regional replay can drive the top-layer matric potential substantially
# drier than the surface-energy training support during hot, high-radiation
# periods. Keep that extrapolation out of the learned energy boundary targets
# while leaving water-stress and hydrology targets on the original state.
SURFACE_ENERGY_SMP_L_FEATURE = "smp_l_lev1"
SURFACE_ENERGY_SMP_L_LOWER_BOUND = -2.3e7
SURFACE_ENERGY_TARGETS: frozenset[str] = frozenset({
    "eflx_sh_tot",
    "hs_soil",
    "hs_top_snow",
    "hs_h2osfc",
    "dhsdT",
})

# ELM's soil/snow temperature solver expects the surface heat-flux
# linearization to be stabilizing. The native training targets are strictly
# negative, so keep small extrapolation errors from making the implicit solve
# anti-diffusive.
DHS_DT_MIN = -350.0
DHS_DT_MAX = -2.5
THERMAL_LINEARIZATION_MAX_DELTA_K = 20.0

# Atmospheric forcing features that are externally prescribed and should never
# be clipped by the manifold constraint — their values are always physically
# valid regardless of how extreme they appear relative to training statistics.
EXTERNAL_FORCING_FEATURES: frozenset[str] = frozenset({
    "forc_t",
    "forc_q",
    "forc_pbot",
    "forc_vpd",
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
    "dayl",
    "max_dayl",
    "coszen_col",
})

SMPL_FEATURES: frozenset[str] = frozenset({f"smp_l_lev{i}" for i in range(1, 9)})

DENICE = 917.0
DENH2O = 1000.0
CPICE = 2117.27
CPLIQ = 4188.0
TFRZ = 273.15
SECSPDAY = 86400.0
EPSILON_WATER_VAPOR = 0.622
GPP_UMOL_TO_GC = 12.011e-6
DT_CARBON_DAILY = 86400.0
HFUS = 3.337e5
MAX_SNOW_LAYERS = 5
MIN_SNOW_LAYER_MASS = 1.0e-6
H2OSNO_MAX = 1000.0
MIN_SNOW_TO_KEEP = 1.0e-9
SNOW_IRREDUCIBLE_SAT = 0.033


def saturation_vapor_pressure_pa(tk: np.ndarray | float) -> np.ndarray:
    """Saturation vapor pressure over water [Pa] from air temperature [K]."""
    tc = np.asarray(tk, dtype=np.float64) - TFRZ
    exponent = np.clip((17.2694 * tc) / (tc + 237.3), -80.0, 80.0)
    return 610.78 * np.exp(exponent)


def vapor_pressure_deficit_from_forcing(
    forc_t: np.ndarray | float,
    forc_q: np.ndarray | float,
    forc_pbot: np.ndarray | float,
) -> np.ndarray:
    """Return atmospheric vapor pressure deficit [Pa] from T, q, and pressure."""
    t = np.asarray(forc_t, dtype=np.float64)
    q = np.maximum(np.asarray(forc_q, dtype=np.float64), 0.0)
    p = np.maximum(np.asarray(forc_pbot, dtype=np.float64), 1.0)
    denom = np.maximum(EPSILON_WATER_VAPOR + (1.0 - EPSILON_WATER_VAPOR) * q, 1.0e-12)
    e_actual = q * p / denom
    return np.maximum(saturation_vapor_pressure_pa(t) - e_actual, 0.0)


def saturation_vapor_pressure_pa_torch(tk: torch.Tensor) -> torch.Tensor:
    """Torch saturation vapor pressure over water [Pa] from air temperature [K]."""
    tc = torch.as_tensor(tk) - TFRZ
    exponent = torch.clamp((17.2694 * tc) / (tc + 237.3), min=-80.0, max=80.0)
    return torch.as_tensor(610.78, device=tc.device, dtype=tc.dtype) * torch.exp(exponent)


def vapor_pressure_deficit_from_forcing_torch(
    forc_t: torch.Tensor,
    forc_q: torch.Tensor,
    forc_pbot: torch.Tensor,
) -> torch.Tensor:
    """Torch atmospheric vapor pressure deficit [Pa] from T, q, and pressure."""
    t = torch.as_tensor(forc_t)
    q = torch.clamp(torch.as_tensor(forc_q, device=t.device, dtype=t.dtype), min=0.0)
    p = torch.clamp(torch.as_tensor(forc_pbot, device=t.device, dtype=t.dtype), min=1.0)
    eps = torch.as_tensor(EPSILON_WATER_VAPOR, device=t.device, dtype=t.dtype)
    denom = torch.clamp(eps + (1.0 - eps) * q, min=1.0e-12)
    e_actual = q * p / denom
    return torch.clamp(saturation_vapor_pressure_pa_torch(t) - e_actual, min=0.0)
SNOW_WATER_TO_DEPTH = 100.0  # mm H2O to m snow depth for fresh snow at ~100 kg/m^3
SNOW_GEOMETRY_MAX_DEPTH = 10.0
SNOW_GEOMETRY_MIN_DENSITY = 50.0
SNOW_GEOMETRY_DEPTH_MARGIN = 1.2


def _carbon_lai_from_leafc(
    leafc: torch.Tensor,
    slatop: torch.Tensor,
    dsladlai: torch.Tensor,
) -> torch.Tensor:
    """Native ELM VegStructUpdate leaf-carbon to one-sided LAI conversion."""
    if float(dsladlai.item()) > 0.0:
        lai = slatop * torch.expm1(leafc * dsladlai) / dsladlai
    else:
        lai = slatop * leafc
    return lai.clamp(min=0.0)


def _carbon_leafc_from_lai(
    lai: torch.Tensor,
    slatop: torch.Tensor,
    dsladlai: torch.Tensor,
) -> torch.Tensor:
    """Inverse of ELM VegStructUpdate LAI formula, used for cold-start seeding."""
    lai = lai.clamp(min=0.0)
    if float(slatop.item()) <= 0.0:
        return torch.zeros_like(lai)
    if float(dsladlai.item()) > 0.0:
        return torch.log1p(lai * dsladlai / slatop) / dsladlai
    return lai / slatop


@dataclass
class OnlineDailyCarbonModel:
    """Carbon pools advanced daily from hourly coupled-model fluxes.

    The hourly surface model can call :meth:`update_vegetation_structure_by_pft`
    before feature construction, while :meth:`accumulate_step` gathers hourly
    photosynthesis and soil state and advances carbon once a full day is
    available.
    """

    pft_ids: list[int]
    pft_weights: torch.Tensor       # (npft, ncell, 1)
    veg_pft_orig_idx: list[int]
    carbon_params: list
    rootfrs: list[torch.Tensor]
    plant_states: list[PlantStateTorch]
    soil_state: SoilStateTorch
    decomp_params: object
    dz: torch.Tensor
    slatop: torch.Tensor            # (nveg,)
    dsladlai: torch.Tensor          # (nveg,)
    crop_flags: torch.Tensor        # (nveg,)
    woody_flags: torch.Tensor       # (nveg,)
    dwood: torch.Tensor             # (nveg,) gC m-3
    stocking: torch.Tensor          # (nveg,) stems m-2
    taper: torch.Tensor             # (nveg,) height:radius
    displar: torch.Tensor           # (nveg,)
    z0mr: torch.Tensor              # (nveg,)
    device: torch.device
    dtype: torch.dtype
    evergreen_leafc_init: float = 10.0
    decid_leafc_storage_init: float = 50.0
    accel_fac: float = 1.0
    accel_mode: str = "constant"

    _acc_gpp: torch.Tensor = field(init=False, repr=False)
    _acc_tsoil: torch.Tensor = field(init=False, repr=False)
    _acc_smpl: torch.Tensor = field(init=False, repr=False)
    _acc_tref: torch.Tensor = field(init=False, repr=False)
    _dayl_last: torch.Tensor = field(init=False, repr=False)
    _prev_dayl_last: torch.Tensor = field(init=False, repr=False)
    _ann_tref_sum: torch.Tensor = field(init=False, repr=False)
    _annavg_tref: torch.Tensor = field(init=False, repr=False)
    _tlai_prev: torch.Tensor = field(init=False, repr=False)
    _tsai: torch.Tensor = field(init=False, repr=False)
    _steps_in_day: int = field(default=0, init=False)
    _day_count: int = field(default=0, init=False)
    _ann_tref_days: int = field(default=0, init=False)
    _last_doy: int | None = field(default=None, init=False, repr=False)
    daily_domain_gpp: list[float] = field(default_factory=list)
    daily_domain_npp: list[float] = field(default_factory=list)
    daily_domain_hr: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        accel_mode = str(self.accel_mode).strip().lower().replace("_", "-")
        if accel_mode == "native":
            accel_mode = "elm-native"
        if accel_mode not in {"constant", "elm-native"}:
            raise ValueError(f"Unsupported carbon acceleration mode: {self.accel_mode!r}")
        self.accel_mode = accel_mode
        npft, ncell, _ = self.pft_weights.shape
        self._acc_gpp = torch.zeros(npft, ncell, device=self.device, dtype=self.dtype)
        self._acc_tsoil = torch.zeros(ncell, 10, device=self.device, dtype=self.dtype)
        self._acc_smpl = torch.zeros(ncell, 10, device=self.device, dtype=self.dtype)
        self._acc_tref = torch.zeros(ncell, device=self.device, dtype=self.dtype)
        self._dayl_last = torch.zeros(ncell, device=self.device, dtype=self.dtype)
        self._prev_dayl_last = torch.zeros(ncell, device=self.device, dtype=self.dtype)
        self._ann_tref_sum = torch.zeros(ncell, device=self.device, dtype=self.dtype)
        self._annavg_tref = torch.full((ncell,), 280.0, device=self.device, dtype=self.dtype)
        self._initialize_leaf_pools()
        self._tlai_prev = self._carbon_tlai_by_veg()
        self._tsai = torch.zeros_like(self._tlai_prev)

    def _initialize_leaf_pools(self) -> None:
        """Cold-start canopy carbon without using prescribed LAI."""
        for params, state in zip(self.carbon_params, self.plant_states):
            state.leafc.zero_()
            state.frootc.zero_()
            state.livestemc.zero_()
            state.deadstemc.zero_()
            state.livecrootc.zero_()
            state.deadcrootc.zero_()
            state.leafc_xfer.zero_()
            state.frootc_xfer.zero_()
            state.livestemc_xfer.zero_()
            state.deadstemc_xfer.zero_()
            state.livecrootc_xfer.zero_()
            state.deadcrootc_xfer.zero_()
            state.leafc_stor.zero_()
            state.frootc_stor.zero_()
            state.livestemc_stor.zero_()
            state.deadstemc_stor.zero_()
            state.livecrootc_stor.zero_()
            state.deadcrootc_stor.zero_()

            if params.is_evergreen:
                leafc = torch.full_like(state.leafc, float(self.evergreen_leafc_init))
                stem_leaf = params.stem_leaf
                croot_stem = params.croot_stem
                flivewd = params.flivewd
                state.leafc.copy_(leafc)
                state.frootc.copy_(leafc * params.froot_leaf)
                state.livestemc.copy_(leafc * stem_leaf * flivewd)
                state.deadstemc.copy_(leafc * stem_leaf * (1.0 - flivewd))
                state.livecrootc.copy_(leafc * stem_leaf * croot_stem * flivewd)
                state.deadcrootc.copy_(leafc * stem_leaf * croot_stem * (1.0 - flivewd))
            elif params.is_any_decid:
                state.leafc_stor.fill_(float(self.decid_leafc_storage_init))

    @classmethod
    def from_clm_params(
        cls,
        *,
        clm_params: Path | str,
        pft_ids: list[int],
        pft_weights: np.ndarray,
        ncell: int,
        nens: int,
        device: torch.device,
        dtype: torch.dtype,
        som_init: float = 5000.0,
        evergreen_leafc_init: float = 10.0,
        decid_leafc_storage_init: float = 50.0,
        accel_fac: float = 1.0,
        accel_mode: str = "constant",
    ) -> "OnlineDailyCarbonModel":
        dz_np, z_node_np, _ = elm_soil_grid(nlevsoi=10)
        with Dataset(clm_params) as ds:
            roota_all = np.asarray(ds.variables["roota_par"][:], dtype=np.float64)
            rootb_all = np.asarray(ds.variables["rootb_par"][:], dtype=np.float64)
            slatop_all = np.asarray(ds.variables["slatop"][:], dtype=np.float64)
            dsladlai_all = np.asarray(ds.variables["dsladlai"][:], dtype=np.float64)
            crop_all = (
                np.asarray(ds.variables["crop"][:], dtype=np.float64)
                if "crop" in ds.variables else np.zeros_like(slatop_all)
            )
            woody_all = (
                np.asarray(ds.variables["woody"][:], dtype=np.float64)
                if "woody" in ds.variables else np.zeros_like(slatop_all)
            )
            if "dwood" in ds.variables:
                dwood_var = np.asarray(ds.variables["dwood"][...], dtype=np.float64)
                dwood_all = (
                    np.full_like(slatop_all, float(dwood_var))
                    if dwood_var.ndim == 0 else dwood_var
                )
            else:
                dwood_all = np.full_like(slatop_all, 2.5e5)
            if "stocking" in ds.variables:
                stocking_all = np.asarray(ds.variables["stocking"][:], dtype=np.float64) / 10000.0
            else:
                stocking_all = np.full_like(slatop_all, 0.1)
            if "taper" in ds.variables:
                taper_all = np.asarray(ds.variables["taper"][:], dtype=np.float64)
            else:
                taper_all = np.full_like(slatop_all, 200.0)
                taper_all = np.where(woody_all == 2.0, 10.0, taper_all)
            displar_all = (
                np.asarray(ds.variables["displar"][:], dtype=np.float64)
                if "displar" in ds.variables else np.full_like(slatop_all, 0.67)
            )
            z0mr_all = (
                np.asarray(ds.variables["z0mr"][:], dtype=np.float64)
                if "z0mr" in ds.variables else np.full_like(slatop_all, 0.055)
            )

        carbon_params = []
        rootfrs = []
        veg_pft_orig_idx: list[int] = []
        slatop_veg: list[float] = []
        dsladlai_veg: list[float] = []
        crop_veg: list[float] = []
        woody_veg: list[float] = []
        dwood_veg: list[float] = []
        stocking_veg: list[float] = []
        taper_veg: list[float] = []
        displar_veg: list[float] = []
        z0mr_veg: list[float] = []
        for raw_idx, pft_id in enumerate(pft_ids):
            if pft_id == 0:
                continue
            params = build_carbon_params_torch(
                clm_params, pft_id, dz=dz_np, device=device, dtype=dtype
            )
            carbon_params.append(params)
            root_idx = native_root_parameter_index_for_pft(int(pft_id), len(roota_all))
            rootfrs.append(
                rootfr_profile_torch(
                    float(roota_all[root_idx]), float(rootb_all[root_idx]), z_node_np,
                    device=device, dtype=dtype,
                )
            )
            veg_pft_orig_idx.append(raw_idx)
            slatop_veg.append(float(slatop_all[pft_id]))
            dsladlai_veg.append(float(dsladlai_all[pft_id]))
            crop_veg.append(float(crop_all[pft_id]))
            woody_veg.append(float(woody_all[pft_id]))
            dwood_veg.append(float(dwood_all[pft_id]))
            stocking_veg.append(float(stocking_all[pft_id]))
            taper_veg.append(float(taper_all[pft_id]))
            displar_veg.append(float(displar_all[pft_id]))
            z0mr_veg.append(float(z0mr_all[pft_id]))

        if not carbon_params:
            raise ValueError("OnlineDailyCarbonModel requires at least one vegetation PFT")

        return cls(
            pft_ids=list(pft_ids),
            pft_weights=torch.as_tensor(pft_weights, device=device, dtype=dtype).unsqueeze(-1),
            veg_pft_orig_idx=veg_pft_orig_idx,
            carbon_params=carbon_params,
            rootfrs=rootfrs,
            plant_states=[
                init_plant_state_torch(ncell, nens, params, leafc_init=0.0)
                for params in carbon_params
            ],
            soil_state=init_soil_state_torch(
                ncell, nens, 10, som_init=som_init, device=device, dtype=dtype
            ),
            decomp_params=carbon_params[0],
            dz=torch.as_tensor(dz_np, device=device, dtype=dtype),
            slatop=torch.as_tensor(slatop_veg, device=device, dtype=dtype),
            dsladlai=torch.as_tensor(dsladlai_veg, device=device, dtype=dtype),
            crop_flags=torch.as_tensor(crop_veg, device=device, dtype=dtype),
            woody_flags=torch.as_tensor(woody_veg, device=device, dtype=dtype),
            dwood=torch.as_tensor(dwood_veg, device=device, dtype=dtype),
            stocking=torch.as_tensor(stocking_veg, device=device, dtype=dtype),
            taper=torch.as_tensor(taper_veg, device=device, dtype=dtype),
            displar=torch.as_tensor(displar_veg, device=device, dtype=dtype),
            z0mr=torch.as_tensor(z0mr_veg, device=device, dtype=dtype),
            device=device,
            dtype=dtype,
            evergreen_leafc_init=evergreen_leafc_init,
            decid_leafc_storage_init=decid_leafc_storage_init,
            accel_fac=accel_fac,
            accel_mode=accel_mode,
        )

    @property
    def day_count(self) -> int:
        return self._day_count

    @property
    def accel_summary(self) -> str:
        if self.accel_mode == "elm-native":
            factors = elm_native_decomp_spinup_factors(self.decomp_params)
            parts = " ".join(f"{key}={value:.3g}" for key, value in factors.items())
            return f"elm-native ({parts})"
        return f"constant factor {self.accel_fac:g}"

    def pool_diagnostics(self) -> dict[str, torch.Tensor]:
        """Return ensemble-mean carbon pool diagnostics by cell in gC m-2."""
        ncell = self.soil_state.soil1c.shape[0]
        nens = self.soil_state.soil1c.shape[1]
        vegc_full = torch.zeros(ncell, nens, device=self.device, dtype=self.dtype)
        deadstemc_full = torch.zeros_like(vegc_full)
        for p_idx, state in enumerate(self.plant_states):
            orig_idx = self.veg_pft_orig_idx[p_idx]
            w_p = self.pft_weights[orig_idx]
            vegc_full.addcmul_(state.total(), w_p, value=1.0)
            deadstemc_full.addcmul_(state.deadstemc, w_p, value=1.0)

        soil = self.soil_state
        somc_full = (
            soil.soil1c + soil.soil2c + soil.soil3c + soil.soil4c
        ).sum(dim=-1)
        dead_organic_full = soil.total().sum(dim=-1)
        ecosysc_full = vegc_full + dead_organic_full

        return {
            "carbon_totvegc": vegc_full.mean(dim=1),
            "carbon_totsomc": somc_full.mean(dim=1),
            "carbon_deadstemc": deadstemc_full.mean(dim=1),
            "carbon_totecosysc": ecosysc_full.mean(dim=1),
        }

    def _carbon_tlai_by_veg(self) -> torch.Tensor:
        """Return carbon-derived total LAI for vegetation PFTs, shaped ``(nveg, ncell)``."""
        nveg = len(self.plant_states)
        ncell = self.plant_states[0].leafc.shape[0]
        tlai = torch.zeros(nveg, ncell, device=self.device, dtype=self.dtype)
        for p_idx, state in enumerate(self.plant_states):
            tlai[p_idx] = _carbon_lai_from_leafc(
                state.leafc,
                self.slatop[p_idx],
                self.dsladlai[p_idx],
            ).mean(1)
        return tlai

    def current_lai_by_pft(self, fallback_lai_by_pft: torch.Tensor | None = None) -> torch.Tensor:
        """Return per-PFT LAI table shaped ``(npft, ncell)`` for surface features."""
        if fallback_lai_by_pft is None:
            npft, ncell, _ = self.pft_weights.shape
            lai = torch.zeros(npft, ncell, device=self.device, dtype=self.dtype)
        else:
            lai = fallback_lai_by_pft.to(device=self.device, dtype=self.dtype).clone()
        tlai_veg = self._carbon_tlai_by_veg()
        for p_idx, orig_idx in enumerate(self.veg_pft_orig_idx):
            lai[orig_idx] = tlai_veg[p_idx]
        return lai

    def update_vegetation_structure_by_pft(
        self,
        *,
        fallback_lai_by_pft: torch.Tensor | None = None,
        fallback_sai_by_pft: torch.Tensor | None = None,
        htop_by_pft: torch.Tensor | None = None,
        hbot_by_pft: torch.Tensor | None = None,
        snow_depth: torch.Tensor | None = None,
        forc_hgt_u_patch: torch.Tensor | None = None,
        dt: float = 3600.0,
    ) -> dict[str, torch.Tensor]:
        """Update carbon-derived SAI and return exposed vegetation structure.

        Mirrors the local E3SM ``VegStructUpdateMod.F90`` contract for the pieces
        needed by the surface emulator: total LAI from leaf carbon, persistent
        total SAI from the Zeng-style update, woody height from dead stem carbon,
        and exposed LAI/SAI after snow burial. Prescribed heights are retained
        for non-woody PFTs when available.
        """
        npft, ncell, _ = self.pft_weights.shape
        if fallback_lai_by_pft is None:
            tlai = torch.zeros(npft, ncell, device=self.device, dtype=self.dtype)
        else:
            tlai = fallback_lai_by_pft.to(device=self.device, dtype=self.dtype).clone()
        if fallback_sai_by_pft is None:
            tsai = torch.zeros_like(tlai)
        else:
            tsai = fallback_sai_by_pft.to(device=self.device, dtype=self.dtype).clone()

        tlai_veg = self._carbon_tlai_by_veg()
        dtsmonth = 2592000.0
        tsai_alpha = torch.where(
            self.crop_flags > 0.5,
            torch.full_like(self.crop_flags, 1.0 - float(dt) / dtsmonth),
            torch.full_like(self.crop_flags, 1.0 - 0.5 * float(dt) / dtsmonth),
        ).reshape(-1, 1)
        tsai_min = torch.where(
            self.crop_flags > 0.5,
            torch.full_like(self.crop_flags, 0.05),
            torch.full_like(self.crop_flags, 0.5),
        ).reshape(-1, 1)
        self._tsai.copy_(
            torch.maximum(
                tsai_alpha * self._tsai + torch.clamp(self._tlai_prev - tlai_veg, min=0.0),
                tsai_min,
            )
        )
        self._tlai_prev.copy_(tlai_veg)

        for p_idx, orig_idx in enumerate(self.veg_pft_orig_idx):
            tlai[orig_idx] = tlai_veg[p_idx]
            tsai[orig_idx] = self._tsai[p_idx]

        if htop_by_pft is None:
            htop = torch.zeros_like(tlai)
        else:
            htop = htop_by_pft.to(device=self.device, dtype=self.dtype).clone()
        woody_full = torch.zeros(npft, 1, device=self.device, dtype=self.dtype)
        woody_htop = torch.zeros_like(tlai)
        woody_hbot = torch.zeros_like(tlai)
        for p_idx, orig_idx in enumerate(self.veg_pft_orig_idx):
            woody_full[orig_idx, 0] = self.woody_flags[p_idx]
            deadstemc = torch.clamp(self.plant_states[p_idx].deadstemc.mean(1), min=0.0)
            denom = torch.clamp(
                torch.as_tensor(torch.pi, device=self.device, dtype=self.dtype)
                * self.stocking[p_idx]
                * self.dwood[p_idx],
                min=1.0e-12,
            )
            htop_p = torch.pow(
                torch.clamp(
                    3.0 * deadstemc * self.taper[p_idx] * self.taper[p_idx] / denom,
                    min=0.0,
                ),
                1.0 / 3.0,
            )
            htop_p = torch.clamp(htop_p, min=0.01)
            if forc_hgt_u_patch is None:
                forc_hgt = torch.full_like(htop_p, 30.0)
            else:
                forc_hgt_t = (
                    forc_hgt_u_patch
                    if isinstance(forc_hgt_u_patch, torch.Tensor)
                    else torch.as_tensor(forc_hgt_u_patch, device=self.device, dtype=self.dtype)
                )
                forc_hgt_t = forc_hgt_t.to(device=self.device, dtype=self.dtype)
                forc_hgt = forc_hgt_t.mean(1) if forc_hgt_t.ndim == 2 else forc_hgt_t.reshape(-1)
            htop_cap = forc_hgt / torch.clamp(self.displar[p_idx] + self.z0mr[p_idx], min=1.0e-6) - 3.0
            htop_p = torch.minimum(htop_p, htop_cap)
            htop_p = torch.clamp(htop_p, min=0.01)
            woody_htop[orig_idx] = htop_p
            woody_hbot[orig_idx] = torch.clamp(htop_p - 1.0, min=0.0, max=3.0)
        nonwoody_htop = torch.maximum(torch.full_like(tlai, 0.25), tlai * 0.25)
        htop_bad = (~torch.isfinite(htop)) | (htop <= 0.0)
        htop = torch.where(
            htop_bad,
            torch.where(woody_full >= 1.0, torch.full_like(htop, 0.01), nonwoody_htop),
            htop,
        )
        htop = torch.where(woody_full >= 1.0, woody_htop, htop)
        hbot_default = torch.where(
            woody_full >= 1.0,
            woody_hbot,
            torch.clamp(htop - 0.20, min=0.0, max=0.05),
        )
        if hbot_by_pft is None:
            hbot = hbot_default
        else:
            hbot = hbot_by_pft.to(device=self.device, dtype=self.dtype).clone()
            hbot_bad = (~torch.isfinite(hbot)) | (hbot < 0.0)
            hbot = torch.where(hbot_bad, hbot_default, hbot)
        hbot = torch.where(woody_full >= 1.0, woody_hbot, hbot)

        if snow_depth is None:
            snow = torch.zeros(ncell, device=self.device, dtype=self.dtype)
        else:
            snow_t = (
                snow_depth if isinstance(snow_depth, torch.Tensor)
                else torch.as_tensor(snow_depth, device=self.device, dtype=self.dtype)
            )
            snow_t = snow_t.to(device=self.device, dtype=self.dtype)
            snow = snow_t.mean(1) if snow_t.ndim == 2 else snow_t.reshape(ncell)
        snow = snow.reshape(1, ncell)

        ol_woody = torch.minimum(
            torch.clamp(snow - hbot, min=0.0),
            torch.clamp(htop - hbot, min=0.0),
        )
        fb_woody = ol_woody / torch.clamp(htop - hbot, min=1.0e-6)
        fb_short = torch.clamp(snow, min=0.0, max=0.2) / 0.2
        fb = torch.where(woody_full >= 1.0, fb_woody, fb_short)
        fb = torch.clamp(fb, min=0.0, max=1.0)

        elai = torch.clamp(tlai * (1.0 - fb), min=0.0)
        esai = torch.clamp(tsai * (1.0 - fb), min=0.0)
        frac_veg_nosno = ((elai + esai) > 0.0).to(dtype=self.dtype)
        return {
            "tlai": tlai,
            "tsai": tsai,
            "elai": elai,
            "esai": esai,
            "htop": htop,
            "hbot": hbot,
            "frac_veg_nosno": frac_veg_nosno,
        }

    def accumulate_step(
        self,
        *,
        fpsn_per_pft: list[torch.Tensor | None] | None,
        state: CoupledColumnState,
        hydrology_result: HydrologyResult | SoilHydrologyTorchResult,
        tbot: torch.Tensor,
        dayl: torch.Tensor,
        prev_dayl: torch.Tensor | None,
        doy: int | None,
        dt: float,
    ) -> bool:
        """Accumulate one coupled timestep; return True when carbon advanced."""
        steps_per_day = max(1, int(round(DT_CARBON_DAILY / float(dt))))
        npft, ncell, _ = self.pft_weights.shape
        nens = self.plant_states[0].leafc.shape[1]

        if doy is not None:
            doy_i = int(doy)
            if self._last_doy is not None and doy_i < self._last_doy:
                self._roll_annual_temperature()
            self._last_doy = doy_i

        if fpsn_per_pft is None:
            fpsn_per_pft = [None] * npft
        for p_idx in range(npft):
            fpsn = fpsn_per_pft[p_idx] if p_idx < len(fpsn_per_pft) else None
            if fpsn is None:
                continue
            fpsn_t = fpsn if isinstance(fpsn, torch.Tensor) else torch.as_tensor(fpsn, device=self.device, dtype=self.dtype)
            fpsn_t = fpsn_t.to(device=self.device, dtype=self.dtype).clamp(min=0.0)
            self._acc_gpp[p_idx].add_(fpsn_t.mean(1) if fpsn_t.ndim == 2 else fpsn_t.reshape(ncell))

        t_soil = state.t_soisno if isinstance(state.t_soisno, torch.Tensor) else torch.as_tensor(state.t_soisno, device=self.device, dtype=self.dtype)
        self._acc_tsoil.add_(t_soil[..., :10].mean(1))

        smp_l = getattr(hydrology_result, "smp_l", None)
        if smp_l is None:
            smp_l_t = torch.zeros(ncell, nens, 10, device=self.device, dtype=self.dtype)
        else:
            smp_l_t = smp_l if isinstance(smp_l, torch.Tensor) else torch.as_tensor(smp_l, device=self.device, dtype=self.dtype)
        self._acc_smpl.add_(smp_l_t[..., :10].mean(1))

        tbot_t = tbot if isinstance(tbot, torch.Tensor) else torch.as_tensor(tbot, device=self.device, dtype=self.dtype)
        dayl_t = dayl if isinstance(dayl, torch.Tensor) else torch.as_tensor(dayl, device=self.device, dtype=self.dtype)
        if prev_dayl is None:
            prev_dayl_t = None
        else:
            prev_dayl_t = prev_dayl if isinstance(prev_dayl, torch.Tensor) else torch.as_tensor(prev_dayl, device=self.device, dtype=self.dtype)
        self._acc_tref.add_(tbot_t.reshape(ncell, -1).mean(1))
        self._dayl_last.copy_(dayl_t.reshape(ncell, -1).mean(1))
        if prev_dayl_t is None:
            self._prev_dayl_last.copy_(self._dayl_last)
        else:
            self._prev_dayl_last.copy_(prev_dayl_t.reshape(ncell, -1).mean(1))

        self._steps_in_day += 1
        if self._steps_in_day < steps_per_day:
            return False

        self._advance_day(steps_per_day)
        self._reset_daily_accumulators()
        return True

    def _roll_annual_temperature(self) -> None:
        if self._ann_tref_days > 0:
            self._annavg_tref.copy_(self._ann_tref_sum / float(self._ann_tref_days))
        self._ann_tref_sum.zero_()
        self._ann_tref_days = 0

    def _advance_day(self, steps_per_day: int) -> None:
        inv_steps = 1.0 / float(steps_per_day)
        gpp_d = self._acc_gpp * (inv_steps * GPP_UMOL_TO_GC)
        t_soil_d = (self._acc_tsoil * inv_steps).unsqueeze(1)
        smp_l_d = (self._acc_smpl * inv_steps).unsqueeze(1)
        tref_d = (self._acc_tref * inv_steps).unsqueeze(1)
        t_pheno_d = t_soil_d[..., 2]
        dayl_d = self._dayl_last.unsqueeze(1)
        prev_dayl_d = self._prev_dayl_last.unsqueeze(1)
        crit_onset_gdd_d = torch.exp(
            torch.as_tensor(4.8, device=self.device, dtype=self.dtype)
            + torch.as_tensor(0.13, device=self.device, dtype=self.dtype)
            * (self._annavg_tref.unsqueeze(1) - TFRZ)
        )

        ncell = self._acc_tref.shape[0]
        step_gpp = torch.zeros(ncell, 1, device=self.device, dtype=self.dtype)
        step_npp = torch.zeros(ncell, 1, device=self.device, dtype=self.dtype)
        plant_accel_fac = self.accel_fac if self.accel_mode == "constant" else 1.0
        litr1_acc = torch.zeros(ncell, 1, 10, device=self.device, dtype=self.dtype)
        litr2_acc = torch.zeros_like(litr1_acc)
        litr3_acc = torch.zeros_like(litr1_acc)
        cwd_acc = torch.zeros_like(litr1_acc)

        for p_idx, (params_p, rootfr_p) in enumerate(zip(self.carbon_params, self.rootfrs)):
            orig_idx = self.veg_pft_orig_idx[p_idx]
            w_p = self.pft_weights[orig_idx]
            gpp_p = gpp_d[orig_idx].unsqueeze(1)
            self.plant_states[p_idx], litter_p, diag_p = carbon_plant_step_torch(
                self.plant_states[p_idx],
                gpp_p, tref_d, t_pheno_d, dayl_d,
                rootfr_p, params_p, DT_CARBON_DAILY,
                smp_l=smp_l_d,
                crit_onset_gdd=crit_onset_gdd_d,
                accel_fac=plant_accel_fac,
                prev_dayl=prev_dayl_d,
            )
            litr1_acc.addcmul_(litter_p.litr1_flux, w_p.unsqueeze(-1), value=1.0)
            litr2_acc.addcmul_(litter_p.litr2_flux, w_p.unsqueeze(-1), value=1.0)
            litr3_acc.addcmul_(litter_p.litr3_flux, w_p.unsqueeze(-1), value=1.0)
            cwd_acc.addcmul_(litter_p.cwd_flux, w_p.unsqueeze(-1), value=1.0)
            step_gpp.addcmul_(diag_p["gpp"], w_p, value=1.0)
            step_npp.addcmul_(diag_p["npp"], w_p, value=1.0)

        litter = LitterFluxTorch(litr1_acc, litr2_acc, litr3_acc, cwd_acc)
        self.soil_state, hr_d = carbon_soil_step_torch(
            self.soil_state, litter, t_soil_d, smp_l_d, self.dz, self.decomp_params,
            DT_CARBON_DAILY, accel_fac=self.accel_fac, accel_mode=self.accel_mode,
        )

        self.daily_domain_gpp.append(float((step_gpp.squeeze(1) * DT_CARBON_DAILY).mean().detach().cpu()))
        self.daily_domain_npp.append(float((step_npp.squeeze(1) * DT_CARBON_DAILY).mean().detach().cpu()))
        self.daily_domain_hr.append(float((hr_d.sum(-1).squeeze(1) * DT_CARBON_DAILY).mean().detach().cpu()))
        self._ann_tref_sum.add_(tref_d.squeeze(1))
        self._ann_tref_days += 1
        self._day_count += 1

    def _reset_daily_accumulators(self) -> None:
        self._acc_gpp.zero_()
        self._acc_tsoil.zero_()
        self._acc_smpl.zero_()
        self._acc_tref.zero_()
        self._steps_in_day = 0
SNOW_ZLND = 0.01  # native ELM roughness length for snow cover fraction relation
SNOW_COMPACTION_C2 = 23.0e-3
SNOW_COMPACTION_C3 = 2.777e-6
SNOW_COMPACTION_C4 = 0.04
SNOW_COMPACTION_C5 = 2.0
SNOW_COMPACTION_DM = 100.0
SNOW_COMPACTION_ETA0 = 9.0e5
SNOW_COMBINE_DZMIN = np.array([0.010, 0.015, 0.025, 0.055, 0.115], dtype=np.float64)
SNOW_COMBINE_MIN_DENSITY = 50.0
SNOW_MIN_ACTIVE_DEPTH = 0.010
SNOW_UPDATE_MAX_DT = 1800.0
SNOW_DIVIDE_TOP_SINGLE = 0.03
SNOW_DIVIDE_TOP_MULTI = 0.02
SNOW_DENSITY_SPLIT_LIMIT = 800.0
SNW_RDS_MIN = 54.526    # µm  fresh snow effective grain radius [Flanner & Zender 2006]
SNW_RDS_MAX = 1500.0    # µm  maximum effective grain radius
SNW_RDS_REFRZ = 1000.0  # µm  refrozen snow effective grain radius
C2_LIQ_BRUN89 = 4.22e-13  # m3 s-1  wet grain-growth coefficient [Brun 1989]


def select_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        raise RuntimeError("Requested CUDA device, but torch.cuda.is_available() is False.")
    if requested == "mps":
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        raise RuntimeError("Requested MPS device, but torch.backends.mps.is_available() is False.")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def configure_tf32_matmul(precision: str, device: torch.device) -> None:
    if device.type != "cuda":
        return
    if precision not in {"highest", "high", "medium"}:
        raise ValueError(f"Unsupported matmul precision: {precision}")
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision(precision)
    allow_reduced_precision = precision != "highest"
    torch.backends.cuda.matmul.allow_tf32 = allow_reduced_precision
    torch.backends.cudnn.allow_tf32 = allow_reduced_precision


def _extend_profile_to_nlev(profile: np.ndarray, nlev: int, *, fill_value: float | None = None) -> np.ndarray:
    """Extend or truncate a 1D profile to `nlev` layers."""
    arr = np.asarray(profile, dtype=np.float64).reshape(-1)
    if arr.size >= nlev:
        return arr[:nlev]
    fill = arr[-1] if fill_value is None else float(fill_value)
    return np.concatenate([arr, np.full(nlev - arr.size, fill, dtype=np.float64)])


def _extend_rootfr_to_nlev(rootfr: np.ndarray, nlev: int) -> np.ndarray:
    """Preserve the supplied root fractions and pad deeper layers with zeros."""
    arr = np.maximum(np.asarray(rootfr, dtype=np.float64).reshape(-1), 0.0)
    if arr.size >= nlev:
        out = arr[:nlev]
    else:
        out = np.zeros(nlev, dtype=np.float64)
        out[: arr.size] = arr
    total = out.sum()
    if total > 0.0:
        out /= total
    return out


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()


def load_surface_model_bundle(
    model_file: Path,
    device: torch.device,
    dtype: torch.dtype,
    use_compile: bool = False,
) -> SurfaceModelBundle:
    bundle = torch.load(model_file, map_location="cpu", weights_only=False)
    input_names = [str(name) for name in bundle["input_names"]]
    target_names = [str(name) for name in bundle["target_names"]]
    hidden_layers = tuple(int(width) for width in bundle["hidden_layers"])
    x_mean_np = np.asarray(bundle["x_mean"], dtype=np.float64)
    x_std_np = np.asarray(bundle["x_std"], dtype=np.float64)
    pft_embedding_dim = int(bundle.get("pft_embedding_dim", 0) or 0)
    pft_embedding_input_name = bundle.get("pft_embedding_input_name") or "pft_type"
    pft_input_index = (
        input_names.index(str(pft_embedding_input_name))
        if pft_embedding_dim > 0 and str(pft_embedding_input_name) in input_names
        else None
    )
    pft_embedding_categories = [
        int(value)
        for value in bundle.get("pft_embedding_categories", [])
    ]
    if pft_embedding_dim > 0 and pft_input_index is None:
        raise ValueError(
            f"{model_file} requests a PFT embedding, but {pft_embedding_input_name!r} "
            "is not present in input_names."
        )
    if pft_embedding_dim > 0 and not pft_embedding_categories:
        raise ValueError(f"{model_file} requests a PFT embedding but has no pft_embedding_categories.")
    model = PatchMLP(
        input_dim=len(input_names),
        output_dim=len(target_names),
        hidden_layers=hidden_layers,
        pft_embedding_dim=pft_embedding_dim,
        pft_input_index=pft_input_index,
        pft_embedding_categories=pft_embedding_categories,
        pft_x_mean=float(bundle.get("pft_embedding_x_mean", x_mean_np[pft_input_index] if pft_input_index is not None else 0.0)),
        pft_x_std=float(bundle.get("pft_embedding_x_std", x_std_np[pft_input_index] if pft_input_index is not None else 1.0)),
    )
    tensor_state_dict: dict[str, torch.Tensor] = {}
    for name, value in bundle["state_dict"].items():
        if isinstance(value, torch.Tensor):
            tensor_state_dict[name] = value.detach().cpu()
        else:
            tensor_state_dict[name] = torch.as_tensor(np.asarray(value))
    model.load_state_dict(tensor_state_dict)
    model.eval()
    model.to(device=device, dtype=dtype)
    if use_compile and hasattr(torch, "compile"):
        model = torch.compile(model)  # type: ignore[assignment]

    # Load and resolve target-feature masks (stored by name, resolved to indices here).
    raw_masks: dict[str, list[str]] = bundle.get("target_feature_masks") or {}
    log1p_target_names = [str(name) for name in bundle.get("log1p_target_names", DEFAULT_LOG1P_SURFACE_TARGETS)]

    # Float32 cannot reliably resolve deviations much smaller than eps * mean.
    # Treat those columns as constant in scaled space to avoid catastrophic
    # cancellation for features such as max_dayl.
    float32_constant_tol = np.maximum(
        1.0e-6,
        10.0 * np.finfo(np.float32).eps * np.maximum(np.abs(x_mean_np), 1.0),
    )
    x_float32_constant_mask = x_std_np <= float32_constant_tol

    return SurfaceModelBundle(
        patch_index=int(bundle["patch_index"]),
        input_names=input_names,
        target_names=target_names,
        x_mean=x_mean_np,
        x_std=x_std_np,
        y_mean=np.asarray(bundle["y_mean"], dtype=np.float64),
        y_std=np.asarray(bundle["y_std"], dtype=np.float64),
        model=model,
        target_feature_masks=raw_masks if raw_masks else None,
        log1p_target_names=log1p_target_names,
        x_mean_torch=torch.as_tensor(x_mean_np, device=device, dtype=dtype),
        x_std_safe_torch=torch.as_tensor(np.where(x_std_np > 0.0, x_std_np, 1.0), device=device, dtype=dtype),
        x_float32_constant_mask_torch=torch.as_tensor(x_float32_constant_mask, device=device, dtype=torch.bool),
        y_mean_torch=torch.as_tensor(np.asarray(bundle["y_mean"], dtype=np.float64), device=device, dtype=dtype),
        y_std_torch=torch.as_tensor(np.asarray(bundle["y_std"], dtype=np.float64), device=device, dtype=dtype),
    )


def required_inputs(bundle: SurfaceModelBundle) -> dict[str, list[str]]:
    """Return the full coupled input contract for one column batch."""
    return {
        "surface_feature_order": list(bundle.input_names),
        "atmospheric_forcing": [
            "forc_t",
            "forc_q",
            "forc_pbot",
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
            "dayl",
            "max_dayl",
            "coszen_col",
        ],
        "derived_forcing_features": [
            name for name in bundle.input_names if name == "forc_vpd"
        ],
        "vegetation_surface_state": [
            "frac_veg_nosno_patch",
            "elai_patch",
            "esai_patch",
            "snow_depth",
            "t10",
            "btran",
        ],
        "dynamic_column_state": [
            "h2osoi_liq",
            "h2osoi_ice",
            "h2ocan",
            "h2osfc",
            "frac_h2osfc",
            "frac_sno",
            "zwt",
            "t_soisno",
            "t_h2osfc",
        ],
        "optional_surface_targets": [
            "h2ocan",
        ],
        "surface_geometry": [
            "forc_hgt_u_patch",
            "z0mg_col",
            "ugust",
        ],
        "static_column_parameters": [
            "rootfr",
            "dz",
            "z",
            "zi",
            "texture.sand_pct",
            "texture.clay_pct",
            "texture.organic_frac",
            "smpso",
            "smpsc",
            "leafcn",
            "slatop",
            "flnr",
            "mbbopt",
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
        ],
        "surface_targets_consumed_downstream": list(REQUIRED_SURFACE_TARGETS),
    }


def describe_inputs(bundle: SurfaceModelBundle) -> str:
    lines = ["Surface feature order:"]
    lines.extend(f"  - {name}" for name in bundle.input_names)
    inputs = required_inputs(bundle)
    lines.append("Atmospheric forcing inputs:")
    lines.extend(f"  - {name}" for name in inputs["atmospheric_forcing"])
    if inputs["derived_forcing_features"]:
        lines.append("Derived forcing features:")
        lines.extend(f"  - {name}" for name in inputs["derived_forcing_features"])
    lines.append("Vegetation and surface-state inputs:")
    lines.extend(f"  - {name}" for name in inputs["vegetation_surface_state"])
    lines.append("Dynamic column state:")
    lines.extend(f"  - {name}" for name in inputs["dynamic_column_state"])
    lines.append("Surface geometry inputs:")
    lines.extend(f"  - {name}" for name in inputs["surface_geometry"])
    lines.append("Static column parameters:")
    lines.extend(f"  - {name}" for name in inputs["static_column_parameters"])
    lines.append("Surface-emulator targets consumed by hydrology/soil temperature:")
    lines.extend(f"  - {name}" for name in REQUIRED_SURFACE_TARGETS)
    return "\n".join(lines)


def assemble_surface_features(
    bundle: SurfaceModelBundle,
    feature_values: Mapping[str, np.ndarray | float],
    batch_shape: tuple[int, ...],
) -> np.ndarray:
    """Assemble surface features in exactly the bundle's expected order."""
    out = np.empty(batch_shape + (len(bundle.input_names),), dtype=np.float64)
    missing = [name for name in bundle.input_names if name not in feature_values]
    if missing:
        raise KeyError(f"Missing surface features: {', '.join(missing[:8])}")
    for idx, name in enumerate(bundle.input_names):
        out[..., idx] = np.broadcast_to(np.asarray(feature_values[name], dtype=np.float64), batch_shape)
    return out


def update_feature_columns(
    bundle: SurfaceModelBundle,
    features: np.ndarray,
    feature_values: Mapping[str, np.ndarray | float],
) -> None:
    """Overwrite selected feature columns in place."""
    idx_map = _bundle_input_index_map(tuple(bundle.input_names))
    for name, value in feature_values.items():
        idx = idx_map.get(name)
        if idx is None:
            continue
        # Move device tensors to CPU before assigning into the numpy features array.
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        features[..., idx] = value


def _target_map(y: np.ndarray, target_names: list[str]) -> dict[str, np.ndarray]:
    return {name: y[..., idx] for idx, name in enumerate(target_names)}


def _target_map_torch(y: torch.Tensor, target_names: list[str]) -> dict[str, torch.Tensor]:
    return {name: y[..., idx] for idx, name in enumerate(target_names)}


def _apply_patch0_zero_surface_targets(
    targets: dict[str, np.ndarray] | dict[str, torch.Tensor],
    bundle: SurfaceModelBundle,
    batch_shape: tuple[int, ...],
    *,
    like: np.ndarray | torch.Tensor,
) -> dict[str, np.ndarray] | dict[str, torch.Tensor]:
    if bundle.patch_index != 0:
        return targets
    for name in PATCH0_ZERO_SURFACE_TARGETS:
        if isinstance(like, torch.Tensor):
            targets[name] = torch.zeros(batch_shape, device=like.device, dtype=like.dtype)
        else:
            targets[name] = np.zeros(batch_shape, dtype=np.asarray(like).dtype)
    return targets


def _apply_surface_energy_smp_l_clip_for_targets(
    bundle: SurfaceModelBundle,
    x2_scaled: torch.Tensor,
    y_scaled: torch.Tensor,
    input_idx_map: Mapping[str, int],
    *,
    already_zero_feature_names: set[str] | frozenset[str] | None = None,
) -> torch.Tensor:
    """Recompute energy targets with top-layer matric potential clipped.

    The clip is target-specific: the rest of the predicted target vector still
    sees the raw hydrology state. This avoids corrupting transpiration and other
    water-stress-sensitive outputs while preventing a rare, strongly negative
    ``smp_l_lev1`` extrapolation from dominating surface energy targets.
    """
    if already_zero_feature_names and SURFACE_ENERGY_SMP_L_FEATURE in already_zero_feature_names:
        return y_scaled
    feature_idx = input_idx_map.get(SURFACE_ENERGY_SMP_L_FEATURE)
    if feature_idx is None:
        return y_scaled
    x_std = float(bundle.x_std[feature_idx]) if feature_idx < bundle.x_std.size else 1.0
    if not np.isfinite(x_std) or x_std <= 0.0:
        return y_scaled

    # In scaled space, clipping physical smp_l_lev1 to be no more negative than
    # the lower bound is a lower-bound clamp because x_std is positive.
    clipped_scaled_value = (SURFACE_ENERGY_SMP_L_LOWER_BOUND - float(bundle.x_mean[feature_idx])) / x_std
    current = x2_scaled[:, feature_idx]
    clipped = torch.clamp(
        current,
        min=torch.as_tensor(clipped_scaled_value, device=x2_scaled.device, dtype=x2_scaled.dtype),
    )
    if not torch.any(clipped != current):
        return y_scaled

    target_feature_masks = bundle.target_feature_masks or {}
    target_indices = [
        idx
        for idx, target_name in enumerate(bundle.target_names)
        if target_name in SURFACE_ENERGY_TARGETS and target_name not in target_feature_masks
    ]
    if not target_indices:
        return y_scaled

    x2_clipped = x2_scaled.clone()
    x2_clipped[:, feature_idx] = clipped
    y_clipped = bundle.model(x2_clipped)
    out = y_scaled.clone()
    out[:, target_indices] = y_clipped[:, target_indices]
    return out


@lru_cache(maxsize=8)
def _nonneg_target_indices(target_names: tuple[str, ...]) -> np.ndarray:
    return np.array([i for i, n in enumerate(target_names) if n in NONNEGATIVE_SURFACE_TARGETS], dtype=np.intp)


def postprocess_surface_predictions(
    y_scaled: np.ndarray,
    bundle: SurfaceModelBundle,
    surface_features: np.ndarray | None = None,
) -> np.ndarray:
    """Convert scaled network outputs back to physical units."""
    y = y_scaled * bundle.y_std + bundle.y_mean
    y_out = np.array(y, copy=True)
    log1p_target_names = set(bundle.log1p_target_names or DEFAULT_LOG1P_SURFACE_TARGETS)
    for idx, name in enumerate(bundle.target_names):
        if name in log1p_target_names:
            y_out[..., idx] = np.expm1(y_out[..., idx])
    nn_idx = _nonneg_target_indices(tuple(bundle.target_names))
    if nn_idx.size > 0:
        y_out[..., nn_idx] = np.maximum(y_out[..., nn_idx], 0.0)
    target_idx_map = _bundle_input_index_map(tuple(bundle.target_names))
    if "dhsdT" in target_idx_map:
        idx = target_idx_map["dhsdT"]
        y_out[..., idx] = np.clip(y_out[..., idx], DHS_DT_MIN, DHS_DT_MAX)
        thermal_flux = None
        frac_h2osfc = None
        if surface_features is not None and "frac_h2osfc" in bundle.input_names:
            frac_idx = bundle.input_names.index("frac_h2osfc")
            frac_h2osfc = np.asarray(surface_features[..., frac_idx], dtype=np.float64)
        for flux_name in ("hs_soil", "hs_h2osfc", "hs_top_snow"):
            flux_idx = target_idx_map.get(flux_name)
            if flux_idx is None:
                continue
            flux_abs = np.abs(y_out[..., flux_idx])
            if flux_name == "hs_h2osfc" and frac_h2osfc is not None:
                flux_abs = np.where(frac_h2osfc > H2OSFC_ACTIVE_FRACTION_THRESHOLD, flux_abs, 0.0)
            thermal_flux = flux_abs if thermal_flux is None else np.maximum(thermal_flux, flux_abs)
        if thermal_flux is not None:
            required = -thermal_flux / THERMAL_LINEARIZATION_MAX_DELTA_K
            y_out[..., idx] = np.clip(np.minimum(y_out[..., idx], required), DHS_DT_MIN, DHS_DT_MAX)
    if (
        surface_features is not None
        and "elai_patch" in bundle.input_names
        and "laisun_patch" in target_idx_map
        and "laisha_patch" in target_idx_map
    ):
        lai_idx = bundle.input_names.index("elai_patch")
        elai = np.maximum(np.asarray(surface_features[..., lai_idx], dtype=np.float64), 0.0)
        laisun_idx = target_idx_map["laisun_patch"]
        laisha_idx = target_idx_map["laisha_patch"]
        lai_sum = y_out[..., laisun_idx] + y_out[..., laisha_idx]
        scale = np.minimum(1.0, elai / np.maximum(lai_sum, 1.0e-12))
        y_out[..., laisun_idx] = y_out[..., laisun_idx] * scale
        y_out[..., laisha_idx] = y_out[..., laisha_idx] * scale

    if surface_features is not None:
        has_zero_rule = False
        zero_mask = np.zeros(y_out.shape[:-1], dtype=bool)
        if "forc_swrad" in bundle.input_names:
            sw_idx = bundle.input_names.index("forc_swrad")
            zero_mask |= np.asarray(surface_features[..., sw_idx], dtype=np.float64) <= ZERO_RADIATION_THRESHOLD
            has_zero_rule = True
        if "coszen_col" in bundle.input_names:
            cosz_idx = bundle.input_names.index("coszen_col")
            zero_mask |= np.asarray(surface_features[..., cosz_idx], dtype=np.float64) <= 0.0
            has_zero_rule = True
        lai_idx = bundle.input_names.index("elai_patch") if "elai_patch" in bundle.input_names else None
        if lai_idx is not None:
            zero_mask |= np.asarray(surface_features[..., lai_idx], dtype=np.float64) <= FPSN_LAI_CUTOFF
            has_zero_rule = True
        btran_idx = bundle.input_names.index("btran") if "btran" in bundle.input_names else None
        if btran_idx is not None:
            zero_mask |= np.asarray(surface_features[..., btran_idx], dtype=np.float64) < ZERO_BTRAN_THRESHOLD
            has_zero_rule = True
        if has_zero_rule and np.any(zero_mask):
            for target_name in PHOTOSYNTHESIS_SURFACE_TARGETS:
                if target_name in target_idx_map:
                    target_idx = target_idx_map[target_name]
                    y_out[..., target_idx] = np.where(zero_mask, 0.0, y_out[..., target_idx])
    return y_out


def postprocess_surface_predictions_torch(
    y_scaled: torch.Tensor,
    bundle: SurfaceModelBundle,
    surface_features: torch.Tensor | None = None,
) -> torch.Tensor:
    """Convert scaled network outputs back to physical units on-device."""
    y_mean = (
        bundle.y_mean_torch
        if bundle.y_mean_torch is not None and bundle.y_mean_torch.device == y_scaled.device and bundle.y_mean_torch.dtype == y_scaled.dtype
        else torch.as_tensor(bundle.y_mean, device=y_scaled.device, dtype=y_scaled.dtype)
    )
    y_std = (
        bundle.y_std_torch
        if bundle.y_std_torch is not None and bundle.y_std_torch.device == y_scaled.device and bundle.y_std_torch.dtype == y_scaled.dtype
        else torch.as_tensor(bundle.y_std, device=y_scaled.device, dtype=y_scaled.dtype)
    )
    y = y_scaled * y_std + y_mean
    y_out = y.clone()
    idx_map = _bundle_input_index_map(tuple(bundle.input_names))
    target_idx_map = _bundle_input_index_map(tuple(bundle.target_names))
    log1p_target_names = set(bundle.log1p_target_names or DEFAULT_LOG1P_SURFACE_TARGETS)
    for idx, name in enumerate(bundle.target_names):
        if name in log1p_target_names:
            y_out[..., idx] = torch.expm1(y_out[..., idx])
        if name in NONNEGATIVE_SURFACE_TARGETS:
            y_out[..., idx] = torch.clamp(y_out[..., idx], min=0.0)
    if "dhsdT" in target_idx_map:
        idx = target_idx_map["dhsdT"]
        y_out[..., idx] = torch.clamp(y_out[..., idx], min=DHS_DT_MIN, max=DHS_DT_MAX)
        thermal_flux = None
        frac_h2osfc = None
        if surface_features is not None:
            frac_idx = idx_map.get("frac_h2osfc")
            if frac_idx is not None:
                frac_h2osfc = surface_features[..., frac_idx]
        for flux_name in ("hs_soil", "hs_h2osfc", "hs_top_snow"):
            flux_idx = target_idx_map.get(flux_name)
            if flux_idx is None:
                continue
            flux_abs = torch.abs(y_out[..., flux_idx])
            if flux_name == "hs_h2osfc" and frac_h2osfc is not None:
                flux_abs = torch.where(
                    frac_h2osfc > H2OSFC_ACTIVE_FRACTION_THRESHOLD,
                    flux_abs,
                    torch.zeros_like(flux_abs),
                )
            thermal_flux = flux_abs if thermal_flux is None else torch.maximum(thermal_flux, flux_abs)
        if thermal_flux is not None:
            required = -thermal_flux / THERMAL_LINEARIZATION_MAX_DELTA_K
            y_out[..., idx] = torch.clamp(torch.minimum(y_out[..., idx], required), min=DHS_DT_MIN, max=DHS_DT_MAX)
    if (
        surface_features is not None
        and "elai_patch" in idx_map
        and "laisun_patch" in target_idx_map
        and "laisha_patch" in target_idx_map
    ):
        lai_idx = idx_map["elai_patch"]
        elai = torch.clamp(surface_features[..., lai_idx], min=0.0)
        laisun_idx = target_idx_map["laisun_patch"]
        laisha_idx = target_idx_map["laisha_patch"]
        lai_sum = y_out[..., laisun_idx] + y_out[..., laisha_idx]
        scale = torch.minimum(torch.ones_like(lai_sum), elai / torch.clamp(lai_sum, min=1.0e-12))
        y_out[..., laisun_idx] = y_out[..., laisun_idx] * scale
        y_out[..., laisha_idx] = y_out[..., laisha_idx] * scale

    if surface_features is not None:
        has_zero_rule = False
        zero_mask = torch.zeros(y_out.shape[:-1], device=y_out.device, dtype=torch.bool)
        if "forc_swrad" in idx_map:
            sw_idx = idx_map["forc_swrad"]
            zero_mask = zero_mask | (surface_features[..., sw_idx] <= ZERO_RADIATION_THRESHOLD)
            has_zero_rule = True
        cosz_idx = idx_map.get("coszen_col")
        if cosz_idx is not None:
            zero_mask = zero_mask | (surface_features[..., cosz_idx] <= 0.0)
            has_zero_rule = True
        lai_idx = idx_map.get("elai_patch")
        if lai_idx is not None:
            zero_mask = zero_mask | (surface_features[..., lai_idx] <= FPSN_LAI_CUTOFF)
            has_zero_rule = True
        btran_idx = idx_map.get("btran")
        if btran_idx is not None:
            zero_mask = zero_mask | (surface_features[..., btran_idx] < ZERO_BTRAN_THRESHOLD)
            has_zero_rule = True
        if has_zero_rule and torch.any(zero_mask):
            for target_name in PHOTOSYNTHESIS_SURFACE_TARGETS:
                if target_name in target_idx_map:
                    target_idx = target_idx_map[target_name]
                    y_out[..., target_idx] = torch.where(
                        zero_mask,
                        torch.zeros_like(y_out[..., target_idx]),
                        y_out[..., target_idx],
                    )
    return y_out


def _ensure_surface_targets(bundle: SurfaceModelBundle, targets: Mapping[str, np.ndarray]) -> None:
    missing = [name for name in REQUIRED_SURFACE_TARGETS if name not in targets]
    if missing:
        raise KeyError(
            "Surface model is missing required coupled targets: " + ", ".join(missing)
        )


@lru_cache(maxsize=8)
def _bundle_input_index_map(input_names: tuple[str, ...]) -> dict[str, int]:
    return {name: idx for idx, name in enumerate(input_names)}


@lru_cache(maxsize=16)
def _soil_profile_feature_indices(
    input_names: tuple[str, ...], prefix: str, nlev: int
) -> tuple[list[int], list[int]]:
    """Return (feat_indices, lev_indices) for bulk slice assignment of a soil profile."""
    idx_map = _bundle_input_index_map(input_names)
    feat_idx: list[int] = []
    lev_idx: list[int] = []
    for lev in range(nlev):
        i = idx_map.get(f"{prefix}_lev{lev + 1}")
        if i is not None:
            feat_idx.append(i)
            lev_idx.append(lev)
    return feat_idx, lev_idx


_TORCH_INDEX_CACHE: dict[tuple[tuple[int, ...], str], torch.Tensor] = {}


def _torch_index_tensor(indices: list[int] | tuple[int, ...], device: torch.device) -> torch.Tensor:
    """Return a cached LongTensor for feature-column index_copy operations."""
    key = (tuple(int(i) for i in indices), str(device))
    out = _TORCH_INDEX_CACHE.get(key)
    if out is None or out.device != device:
        out = torch.tensor(key[0], device=device, dtype=torch.long)
        _TORCH_INDEX_CACHE[key] = out
    return out


def _index_copy_feature_columns_torch(
    features: torch.Tensor,
    indices: list[int] | tuple[int, ...],
    values: torch.Tensor,
) -> None:
    """Overwrite feature columns with a single grouped index_copy_ call."""
    if not indices:
        return
    if features.device.type == "cpu":
        values = values.to(device=features.device, dtype=features.dtype)
        for j, idx in enumerate(indices):
            features[..., idx] = values[..., j]
        return
    idx = _torch_index_tensor(indices, features.device)
    flat_features = features.reshape(-1, features.shape[-1])
    flat_values = values.reshape(-1, len(indices)).to(device=features.device, dtype=features.dtype)
    flat_features.index_copy_(1, idx, flat_values)


def _batch_update_feature_columns_torch(
    features: torch.Tensor,
    idx_map: Mapping[str, int],
    feature_values: Mapping[str, np.ndarray | float | torch.Tensor],
) -> None:
    """Overwrite many scalar feature columns with one grouped write."""
    batch_shape = features.shape[:-1]
    if features.device.type == "cpu":
        for name, value in feature_values.items():
            idx = idx_map.get(name)
            if idx is None:
                continue
            features[..., idx] = torch.broadcast_to(_to_torch(value, features.device, features.dtype), batch_shape)
        return
    indices: list[int] = []
    cols: list[torch.Tensor] = []
    for name, value in feature_values.items():
        idx = idx_map.get(name)
        if idx is None:
            continue
        indices.append(idx)
        cols.append(torch.broadcast_to(_to_torch(value, features.device, features.dtype), batch_shape))
    if cols:
        _index_copy_feature_columns_torch(features, indices, torch.stack(cols, dim=-1))


def _to_numpy(x: np.ndarray | torch.Tensor | None) -> np.ndarray | None:
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _to_np(x, dtype=np.float64) -> np.ndarray:
    """Convert to numpy (float64 by default), moving a device tensor to CPU first."""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu()
    return np.asarray(x, dtype=dtype)


def _surface_target_to_numpy(value):
    """Convert tensor-valued surface targets to NumPy, preserving diagnostic lists."""
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return np.asarray(value.detach().cpu().numpy(), dtype=np.float64)
    if isinstance(value, list):
        return [_surface_target_to_numpy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_surface_target_to_numpy(item) for item in value)
    return np.asarray(value, dtype=np.float64)


def _to_torch(x: np.ndarray | torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.as_tensor(x, device=device, dtype=dtype)


def precompute_surface_feature_state_torch(
    state: CoupledColumnState,
    static: CoupledStaticParameters,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor | tuple[int, ...] | None]:
    """Precompute shared state-derived feature tensors for one coupled timestep."""
    h2osoi_liq_t = _to_torch(state.h2osoi_liq, device, dtype)
    h2osoi_ice_t = _to_torch(state.h2osoi_ice, device, dtype)
    h2ocan_t = _to_torch(state.h2ocan, device, dtype)
    h2osfc_t = _to_torch(state.h2osfc, device, dtype)
    frac_h2osfc_t = _to_torch(state.frac_h2osfc, device, dtype)
    frac_sno_t = _total_snow_fraction_torch(state, device=device, dtype=dtype)
    t_soisno_t = _to_torch(state.t_soisno, device, dtype)
    t_h2osfc_t = _to_torch(
        state.t_h2osfc if state.t_h2osfc is not None else state.t_soisno[..., 0],
        device,
        dtype,
    )
    snow_depth_t = _total_snow_depth_torch(state, device=device, dtype=dtype)
    t_grnd = coupled_t_grnd_from_state_torch(state, device, dtype)
    t10 = coupled_t10_from_state_torch(state, device, dtype)

    hyd = static.hydrology_torch
    dz = hyd.dz.reshape((1, 1, -1))
    watsat = _prep_param_3d(hyd.watsat)
    sucsat = _prep_param_3d(hyd.sucsat)
    bsw = _prep_param_3d(hyd.bsw)
    nlev_hydro = int(hyd.dz.shape[-1])
    liqvol = h2osoi_liq_t[..., :nlev_hydro] / (dz * DENH2O)
    vol_ice = torch.minimum(watsat, h2osoi_ice_t[..., :nlev_hydro] / (dz * DENICE))
    h2osoi_vol = liqvol + vol_ice
    eff_porosity = torch.clamp(watsat - vol_ice, min=0.01)
    s_node = torch.clamp(liqvol / torch.clamp(watsat, min=1.0e-12), min=0.01, max=1.0)
    smp_l = torch.clamp(-sucsat * torch.pow(s_node, -bsw), min=-1.0e8)

    return {
        "batch_shape": state.h2osfc.shape,
        "h2osoi_liq_t": h2osoi_liq_t,
        "h2osoi_ice_t": h2osoi_ice_t,
        "h2ocan_t": h2ocan_t,
        "h2osfc_t": h2osfc_t,
        "frac_h2osfc_t": frac_h2osfc_t,
        "frac_sno_t": frac_sno_t,
        "t_soisno_t": t_soisno_t,
        "t_h2osfc_t": t_h2osfc_t,
        "snow_depth_t": snow_depth_t,
        "t_grnd": t_grnd,
        "t10": t10,
        "liqvol": liqvol,
        "h2osoi_vol": h2osoi_vol,
        "eff_porosity": eff_porosity,
        "smp_l": smp_l,
    }


def _prep_param_3d(x: torch.Tensor) -> torch.Tensor:
    """Ensure a soil parameter tensor is broadcastable against (ncell, nens, nlev) state.

    Accepts:
      * 1-D (nlev,)         → (1, 1, nlev)   – scalar-per-layer (original case)
      * 2-D (ncell, nlev)   → (ncell, 1, nlev) – per-cell per-layer
      * 3-D (ncell, 1, nlev) – already prepared, returned unchanged
    """
    if x.ndim == 1:
        return x.reshape(1, 1, -1)
    if x.ndim == 2:
        return x.unsqueeze(1)       # (ncell, nlev) → (ncell, 1, nlev)
    return x


def _top_layer_param_numpy(param: np.ndarray, batch_ndim: int) -> np.ndarray:
    arr = np.asarray(param, dtype=np.float64)
    if arr.ndim == 0:
        return arr
    top = arr[..., 0]
    if arr.ndim == 1:
        return top
    while top.ndim < batch_ndim:
        top = top[..., None]
    return top


def _top_layer_param_torch(param: torch.Tensor, batch_ndim: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    arr = _to_torch(param, device, dtype)
    if arr.ndim == 0:
        return arr
    top = arr[..., 0]
    if arr.ndim == 1:
        return top
    while top.ndim < batch_ndim:
        top = top.unsqueeze(-1)
    return top


def _watfc_from_hydraulic_params_numpy(static: CoupledStaticParameters, batch_ndim: int) -> np.ndarray:
    hyd = static.hydrology_numpy
    watsat = _top_layer_param_numpy(hyd.watsat, batch_ndim)
    hksat = np.maximum(_top_layer_param_numpy(hyd.hksat, batch_ndim), 1.0e-20)
    bsw = _top_layer_param_numpy(hyd.bsw, batch_ndim)
    return watsat * np.power(0.1 / (hksat * SECSPDAY), 1.0 / (2.0 * bsw + 3.0))


def _watfc_from_hydraulic_params_torch(
    static: CoupledStaticParameters,
    batch_ndim: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    hyd = static.hydrology_torch
    watsat = _top_layer_param_torch(hyd.watsat, batch_ndim, device, dtype)
    hksat = torch.clamp(_top_layer_param_torch(hyd.hksat, batch_ndim, device, dtype), min=1.0e-20)
    bsw = _top_layer_param_torch(hyd.bsw, batch_ndim, device, dtype)
    return watsat * torch.pow(
        torch.as_tensor(0.1, device=device, dtype=dtype) / (hksat * SECSPDAY),
        1.0 / (2.0 * bsw + 3.0),
    )


def coupled_soilbeta_from_state(
    state: CoupledColumnState,
    static: CoupledStaticParameters,
    *,
    frac_sno: np.ndarray | None = None,
    frac_h2osfc: np.ndarray | None = None,
) -> np.ndarray:
    """ELM SurfaceResistanceMod Lee-Pielke top-layer ground-evaporation beta."""
    h2osoi_liq = _to_np(state.h2osoi_liq)[..., 0]
    h2osoi_ice = _to_np(state.h2osoi_ice)[..., 0]
    batch_ndim = h2osoi_liq.ndim
    dz0 = _top_layer_param_numpy(static.hydrology_numpy.dz, batch_ndim)
    watfc = _watfc_from_hydraulic_params_numpy(static, batch_ndim)
    frac_sno_eff = _active_snow_fraction_numpy(state) if frac_sno is None else np.asarray(frac_sno, dtype=np.float64)
    frac_h2osfc_eff = _to_np(state.frac_h2osfc) if frac_h2osfc is None else np.asarray(frac_h2osfc, dtype=np.float64)
    wx = (h2osoi_liq / DENH2O + h2osoi_ice / DENICE) / np.maximum(dz0, 1.0e-12)
    fac_fc = np.clip(wx / np.maximum(watfc, 1.0e-12), 0.01, 1.0)
    beta_dry = (
        (1.0 - frac_sno_eff - frac_h2osfc_eff)
        * 0.25
        * (1.0 - np.cos(math.pi * fac_fc)) ** 2.0
        + frac_sno_eff
        + frac_h2osfc_eff
    )
    return np.where(wx < watfc, beta_dry, 1.0)


def coupled_soilbeta_from_state_torch(
    state: CoupledColumnState,
    static: CoupledStaticParameters,
    *,
    device: torch.device,
    dtype: torch.dtype,
    h2osoi_liq: torch.Tensor | None = None,
    h2osoi_ice: torch.Tensor | None = None,
    frac_sno: torch.Tensor | None = None,
    frac_h2osfc: torch.Tensor | None = None,
) -> torch.Tensor:
    """Torch-native ELM SurfaceResistanceMod Lee-Pielke soilbeta."""
    h2osoi_liq_t = _to_torch(state.h2osoi_liq, device, dtype) if h2osoi_liq is None else h2osoi_liq.to(device=device, dtype=dtype)
    h2osoi_ice_t = _to_torch(state.h2osoi_ice, device, dtype) if h2osoi_ice is None else h2osoi_ice.to(device=device, dtype=dtype)
    liq_top = h2osoi_liq_t[..., 0]
    ice_top = h2osoi_ice_t[..., 0]
    batch_ndim = liq_top.ndim
    dz0 = _top_layer_param_torch(static.hydrology_torch.dz, batch_ndim, device, dtype)
    watfc = _watfc_from_hydraulic_params_torch(static, batch_ndim, device=device, dtype=dtype)
    frac_sno_eff = (
        _active_snow_fraction_torch(state, device=device, dtype=dtype)
        if frac_sno is None
        else frac_sno.to(device=device, dtype=dtype)
    )
    frac_h2osfc_eff = (
        _to_torch(state.frac_h2osfc, device, dtype)
        if frac_h2osfc is None
        else frac_h2osfc.to(device=device, dtype=dtype)
    )
    wx = (liq_top / DENH2O + ice_top / DENICE) / torch.clamp(dz0, min=1.0e-12)
    fac_fc = torch.clamp(wx / torch.clamp(watfc, min=1.0e-12), min=0.01, max=1.0)
    beta_dry = (
        (1.0 - frac_sno_eff - frac_h2osfc_eff)
        * 0.25
        * torch.pow(1.0 - torch.cos(math.pi * fac_fc), 2.0)
        + frac_sno_eff
        + frac_h2osfc_eff
    )
    return torch.where(wx < watfc, beta_dry, torch.ones_like(wx))


def frac_h2osfc_from_storage(
    h2osfc: np.ndarray,
    micro_sigma: float | np.ndarray,
) -> np.ndarray:
    """ELM non-polygonal ``FracH2OSfc`` closure from storage and microtopography."""
    h2osfc = np.asarray(h2osfc, dtype=np.float64)
    sigma_m = np.broadcast_to(np.asarray(micro_sigma, dtype=np.float64), h2osfc.shape)
    frac = np.zeros_like(h2osfc)
    active = h2osfc > 1.0e-8
    if not np.any(active):
        return frac

    sigma = 1.0e3 * np.maximum(sigma_m[active], 1.0e-6)  # m -> mm
    d = np.zeros_like(sigma)
    target = h2osfc[active]
    sqrt_two = np.sqrt(2.0)
    sqrt_two_pi = np.sqrt(2.0 * np.pi)
    # Vectorized erf via Abramowitz & Stegun 7.1.26 rational approximation (max |err| < 1.5e-7)
    def _erf(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        sign = np.sign(x)
        t = 1.0 / (1.0 + 0.3275911 * np.abs(x))
        poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))))
        return sign * (1.0 - poly * np.exp(-x * x))
    for _ in range(10):
        erf_term = _erf(d / (sigma * sqrt_two))
        exp_term = np.exp(-d**2 / (2.0 * sigma**2))
        fd = 0.5 * d * (1.0 + erf_term) + sigma / sqrt_two_pi * exp_term - target
        dfdd = np.maximum(0.5 * (1.0 + erf_term), 1.0e-12)
        d = d - fd / dfdd
    frac[active] = 0.5 * (1.0 + _erf(d / (sigma * sqrt_two)))
    return np.clip(frac, 0.0, 0.99)


def frac_h2osfc_from_storage_torch(
    h2osfc: np.ndarray | torch.Tensor,
    micro_sigma: float | np.ndarray | torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Torch-native ``FracH2OSfc`` closure from storage and microtopography."""
    h2osfc_t = _to_torch(h2osfc, device, dtype)
    micro_sigma_t = _to_torch(micro_sigma, device, dtype)
    sigma_m = torch.broadcast_to(micro_sigma_t, h2osfc_t.shape)
    frac = torch.zeros_like(h2osfc_t)
    active = h2osfc_t > 1.0e-8
    if not torch.any(active):
        return frac

    sigma = 1.0e3 * torch.clamp(sigma_m[active], min=1.0e-6)
    target = h2osfc_t[active]
    d = torch.zeros_like(sigma)
    sqrt_two = math.sqrt(2.0)
    sqrt_two_pi = math.sqrt(2.0 * math.pi)
    for _ in range(10):
        erf_term = torch.erf(d / (sigma * sqrt_two))
        exp_term = torch.exp(-d**2 / (2.0 * sigma**2))
        fd = 0.5 * d * (1.0 + erf_term) + sigma / sqrt_two_pi * exp_term - target
        dfdd = torch.clamp(0.5 * (1.0 + erf_term), min=1.0e-12)
        d = d - fd / dfdd
    frac[active] = 0.5 * (1.0 + torch.erf(d / (sigma * sqrt_two)))
    return torch.clamp(frac, 0.0, 0.99)


def inverse_frac_h2osfc_torch(frac: np.ndarray | torch.Tensor, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    frac_t = _to_torch(frac, device, dtype)
    frac_t = torch.clamp(frac_t, 0.0, 0.999999)
    return -5.0 * torch.log(torch.clamp(1.0 - frac_t, min=1.0e-12))


def canopy_water_capacity(elai: np.ndarray, esai: np.ndarray) -> np.ndarray:
    """Approximate ELM-style canopy water capacity in mm."""
    return 0.1 * np.maximum(np.asarray(elai, dtype=np.float64) + np.asarray(esai, dtype=np.float64), 0.0)


def canopy_water_capacity_torch(elai: np.ndarray | torch.Tensor, esai: np.ndarray | torch.Tensor, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    elai_t = _to_torch(elai, device, dtype)
    esai_t = _to_torch(esai, device, dtype)
    return 0.1 * torch.clamp(elai_t + esai_t, min=0.0)


def micro_sigma_from_slope(slope: float) -> float:
    """ELM `initVerticalMod` mapping from topographic slope to `micro_sigma`."""
    slopebeta = 3.0
    slopemax = 0.4
    slope0 = slopemax ** (-1.0 / slopebeta)
    topo_slope = max(float(slope), 0.2)
    return float((topo_slope + slope0) ** (-slopebeta))


def h2osfc_threshold_from_micro_sigma(micro_sigma: float, pc: float = 0.4) -> float:
    """ELM `SoilHydrologyType` fill-and-spill threshold in mm."""
    sigma = max(float(micro_sigma), 1.0e-6)
    d = 0.0
    sqrt_two = np.sqrt(2.0)
    sqrt_two_pi = np.sqrt(2.0 * np.pi)
    for _ in range(4):
        fd = 0.5 * (1.0 + erf(d / (sigma * sqrt_two))) - pc
        dfdd = np.exp(-d**2 / (2.0 * sigma**2)) / (sigma * sqrt_two_pi)
        d = d - fd / max(dfdd, 1.0e-12)
    thresh_m = 0.5 * d * (1.0 + erf(d / (sigma * sqrt_two))) + sigma / sqrt_two_pi * np.exp(
        -d**2 / (2.0 * sigma**2)
    )
    return float(1.0e3 * thresh_m)


def organic_fraction_from_surfdata(
    organic_density: np.ndarray,
    *,
    organic_max: float = 130.0,
    more_vertlayers: bool = False,
) -> np.ndarray:
    """Convert surfdata ORGANIC (kg m-3) to the ELM organic fraction used in soil physics."""
    organic_density = np.asarray(organic_density, dtype=np.float64)
    if more_vertlayers:
        om_frac = organic_density / organic_max
    else:
        om_frac = (organic_density / organic_max) ** 2
    return np.clip(om_frac, 0.0, 1.0)


def load_site_info_from_surfdata(
    surfdata_file: Path | str,
    *,
    gridcell: int = 0,
    organic_max: float = 130.0,
    more_vertlayers: bool = False,
) -> SurfdataSiteInfo:
    """Extract location-specific static information from an ELM surfdata file."""
    with Dataset(surfdata_file) as ds:
        sand = np.asarray(ds.variables["PCT_SAND"][:, gridcell], dtype=np.float64)
        clay = np.asarray(ds.variables["PCT_CLAY"][:, gridcell], dtype=np.float64)
        organic_density = np.asarray(ds.variables["ORGANIC"][:, gridcell], dtype=np.float64)
        slope = float(np.asarray(ds.variables["SLOPE"][gridcell], dtype=np.float64))
        std_elev_m = float(np.asarray(ds.variables["STD_ELEV"][gridcell], dtype=np.float64))
        fmax = float(np.asarray(ds.variables["FMAX"][gridcell], dtype=np.float64))
        pc = 0.4
        if "pc" in ds.variables:
            pc = float(np.asarray(ds.variables["pc"][gridcell], dtype=np.float64))
        lat = float(np.asarray(ds.variables["LATIXY"][gridcell], dtype=np.float64))
        lon = float(np.asarray(ds.variables["LONGXY"][gridcell], dtype=np.float64))
        area_km2 = float(np.asarray(ds.variables["AREA"][gridcell], dtype=np.float64))
        zwt0_m = None
        if "ZWT0" in ds.variables:
            zwt0_m = float(np.asarray(ds.variables["ZWT0"][gridcell], dtype=np.float64))
        soil_color = None
        if "SOIL_COLOR" in ds.variables:
            soil_color = float(np.asarray(ds.variables["SOIL_COLOR"][gridcell], dtype=np.float64))

        dominant_natpft_index = None
        dominant_natpft_pct = None
        monthly_lai = None
        monthly_sai = None
        monthly_height_top = None
        monthly_height_bot = None
        if "PCT_NAT_PFT" in ds.variables:
            pct_nat_pft = np.asarray(ds.variables["PCT_NAT_PFT"][:, gridcell], dtype=np.float64)
            dominant_natpft_index = int(np.argmax(pct_nat_pft))
            dominant_natpft_pct = float(pct_nat_pft[dominant_natpft_index])
            if "MONTHLY_LAI" in ds.variables:
                monthly_lai = np.asarray(ds.variables["MONTHLY_LAI"][:, dominant_natpft_index, gridcell], dtype=np.float64)
            if "MONTHLY_SAI" in ds.variables:
                monthly_sai = np.asarray(ds.variables["MONTHLY_SAI"][:, dominant_natpft_index, gridcell], dtype=np.float64)
            if "MONTHLY_HEIGHT_TOP" in ds.variables:
                monthly_height_top = np.asarray(ds.variables["MONTHLY_HEIGHT_TOP"][:, dominant_natpft_index, gridcell], dtype=np.float64)
            if "MONTHLY_HEIGHT_BOT" in ds.variables:
                monthly_height_bot = np.asarray(ds.variables["MONTHLY_HEIGHT_BOT"][:, dominant_natpft_index, gridcell], dtype=np.float64)

    micro_sigma = micro_sigma_from_slope(slope)
    organic_frac = organic_fraction_from_surfdata(
        organic_density,
        organic_max=organic_max,
        more_vertlayers=more_vertlayers,
    )
    return SurfdataSiteInfo(
        gridcell=gridcell,
        lat=lat,
        lon_deg_east=lon,
        area_km2=area_km2,
        slope=slope,
        std_elev_m=std_elev_m,
        micro_sigma_m=micro_sigma,
        pc=pc,
        fmax=fmax,
        zwt0_m=zwt0_m,
        soil_color=soil_color,
        sand_pct=sand,
        clay_pct=clay,
        organic_density=organic_density,
        organic_frac=organic_frac,
        dominant_natpft_index=dominant_natpft_index,
        dominant_natpft_pct=dominant_natpft_pct,
        monthly_lai=monthly_lai,
        monthly_sai=monthly_sai,
        monthly_height_top=monthly_height_top,
        monthly_height_bot=monthly_height_bot,
    )


def apply_canopy_interception(
    bundle: SurfaceModelBundle,
    surface_features: np.ndarray | torch.Tensor,
    state: CoupledColumnState,
    dt: float,
) -> tuple[dict[str, np.ndarray] | dict[str, torch.Tensor], CoupledColumnState]:
    """Port ELM CanopyHydrology interception/drip before surface flux prediction."""
    if isinstance(surface_features, torch.Tensor):
        features_t = surface_features
        device = features_t.device
        dtype = features_t.dtype
        idx_map = _bundle_input_index_map(tuple(bundle.input_names))
        h2ocan_old = _to_torch(state.h2ocan, device, dtype)
        h2osfc_t = _to_torch(state.h2osfc, device, dtype)
        frac_h2osfc_t = _to_torch(state.frac_h2osfc, device, dtype)
        frac_sno_t = _to_torch(state.frac_sno, device, dtype)
        zwt_t = _to_torch(state.zwt, device, dtype)
        t_soisno_t = _to_torch(state.t_soisno, device, dtype)
        t_h2osfc_t = None if state.t_h2osfc is None else _to_torch(state.t_h2osfc, device, dtype)
        t10_t = None if state.t10 is None else _to_torch(state.t10, device, dtype)
        t10_history_t = None if state.t10_history is None else _to_torch(state.t10_history, device, dtype)
        snow_t_t = None if state.snow_t is None else _to_torch(state.snow_t, device, dtype)

        def feature(name: str) -> torch.Tensor:
            idx = idx_map[name]
            return features_t[..., idx]

        precip_rain = torch.clamp(feature("forc_rain"), min=0.0)
        precip_snow = torch.clamp(feature("forc_snow"), min=0.0)
        zero = torch.zeros_like(precip_rain)
        elai_idx = idx_map.get("elai_patch")
        esai_idx = idx_map.get("esai_patch")
        elai = features_t[..., elai_idx] if elai_idx is not None else zero
        esai = features_t[..., esai_idx] if esai_idx is not None else zero
        vegt = torch.clamp(elai + esai, min=0.0)
        frac_veg_idx = idx_map.get("frac_veg_nosno_patch")
        if frac_veg_idx is not None:
            frac_veg_nosno = torch.clamp(features_t[..., frac_veg_idx], min=0.0, max=1.0)
        else:
            snow_free = 1.0 - torch.clamp(frac_sno_t, min=0.0, max=1.0)
            frac_veg_nosno = torch.where(vegt > 0.0, snow_free, zero)
        capacity = canopy_water_capacity_torch(elai, esai, device=device, dtype=dtype)

        precip_total = precip_rain + precip_snow
        active_canopy = (frac_veg_nosno > 0.0) & (precip_total > 0.0)
        fracsnow = torch.where(precip_total > 0.0, precip_snow / torch.clamp(precip_total, min=1.0e-30), torch.zeros_like(precip_total))
        fracrain = torch.where(precip_total > 0.0, precip_rain / torch.clamp(precip_total, min=1.0e-30), torch.zeros_like(precip_total))
        fpi = 0.25 * (1.0 - torch.exp(-0.5 * vegt))
        through_snow = precip_snow * (1.0 - fpi)
        through_rain = precip_rain * (1.0 - fpi)
        intercepted = precip_total * fpi
        h2ocan_after_interception = torch.clamp(h2ocan_old + dt * torch.where(active_canopy, intercepted, torch.zeros_like(intercepted)), min=0.0)
        candrip = torch.where(active_canopy, torch.clamp((h2ocan_after_interception - capacity) / dt, min=0.0), torch.zeros_like(precip_total))
        h2ocan_after_drip = torch.where(active_canopy & (candrip > 0.0), capacity, h2ocan_after_interception)

        qflx_snow_grnd = torch.where(active_canopy, through_snow + candrip * fracsnow, precip_snow)
        qflx_rain_grnd = torch.where(active_canopy, through_rain + candrip * fracrain, precip_rain)
        qflx_leafdrip = torch.where(active_canopy, candrip * fracrain, torch.zeros_like(candrip))
        qprec_grnd = torch.clamp(qflx_snow_grnd + qflx_rain_grnd, min=0.0)

        adjusted_targets: dict[str, torch.Tensor] = {}
        adjusted_targets["qflx_prec_grnd"] = qprec_grnd
        adjusted_targets["qflx_snow_grnd"] = qflx_snow_grnd
        adjusted_targets["qflx_rain_grnd"] = qflx_rain_grnd
        adjusted_targets["qflx_leafdrip"] = qflx_leafdrip

        adjusted_state = CoupledColumnState(
            h2osoi_liq=_to_torch(state.h2osoi_liq, device, dtype),
            h2osoi_ice=_to_torch(state.h2osoi_ice, device, dtype),
            h2ocan=h2ocan_after_drip,
            h2osfc=h2osfc_t,
            frac_h2osfc=frac_h2osfc_t,
            frac_sno=frac_sno_t,
            zwt=zwt_t,
            t_soisno=t_soisno_t,
            t_h2osfc=t_h2osfc_t,
            t10=t10_t,
            t10_history=t10_history_t,
            h2osno=_to_torch(snow_storage_from_state(state), device, dtype),
            snow_depth=_to_torch(state.snow_depth, device, dtype) if state.snow_depth is not None else snow_depth_from_storage_torch(_to_torch(snow_storage_from_state(state), device, dtype), device=device, dtype=dtype),
            snow_t=snow_t_t,
            snow_layers=_snow_layers_to_device(state.snow_layers, device, dtype),
            int_snow=None if state.int_snow is None else _to_torch(state.int_snow, device, dtype),
        )
        return adjusted_targets, adjusted_state

    features_np = _to_numpy(surface_features)
    assert features_np is not None
    idx_map = _bundle_input_index_map(tuple(bundle.input_names))

    def feature(name: str) -> np.ndarray:
        idx = idx_map[name]
        return np.asarray(features_np[..., idx], dtype=np.float64)

    precip_rain = feature("forc_rain")
    precip_snow = feature("forc_snow")
    zero = np.zeros_like(precip_rain)
    elai_idx = idx_map.get("elai_patch")
    esai_idx = idx_map.get("esai_patch")
    elai = np.asarray(features_np[..., elai_idx], dtype=np.float64) if elai_idx is not None else zero
    esai = np.asarray(features_np[..., esai_idx], dtype=np.float64) if esai_idx is not None else zero
    vegt = np.maximum(elai + esai, 0.0)
    frac_veg_idx = idx_map.get("frac_veg_nosno_patch")
    if frac_veg_idx is not None:
        frac_veg_nosno = np.clip(np.asarray(features_np[..., frac_veg_idx], dtype=np.float64), 0.0, 1.0)
    else:
        snow_free = 1.0 - np.clip(_to_np(state.frac_sno), 0.0, 1.0)
        frac_veg_nosno = np.where(vegt > 0.0, snow_free, zero)
    capacity = canopy_water_capacity(elai, esai)

    h2ocan_old = _to_np(state.h2ocan)

    precip_total = precip_rain + precip_snow
    active_canopy = (frac_veg_nosno > 0.0) & (precip_total > 0.0)
    fracsnow = np.where(precip_total > 0.0, precip_snow / np.maximum(precip_total, 1.0e-30), 0.0)
    fracrain = np.where(precip_total > 0.0, precip_rain / np.maximum(precip_total, 1.0e-30), 0.0)
    fpi = 0.25 * (1.0 - np.exp(-0.5 * vegt))
    through_snow = precip_snow * (1.0 - fpi)
    through_rain = precip_rain * (1.0 - fpi)
    intercepted = precip_total * fpi
    h2ocan_after_interception = np.maximum(h2ocan_old + dt * np.where(active_canopy, intercepted, 0.0), 0.0)
    candrip = np.where(active_canopy, np.maximum((h2ocan_after_interception - capacity) / dt, 0.0), 0.0)
    h2ocan_after_drip = np.where(active_canopy & (candrip > 0.0), capacity, h2ocan_after_interception)

    qflx_snow_grnd = np.where(active_canopy, through_snow + candrip * fracsnow, precip_snow)
    qflx_rain_grnd = np.where(active_canopy, through_rain + candrip * fracrain, precip_rain)
    qflx_leafdrip = np.where(active_canopy, candrip * fracrain, 0.0)
    qprec_grnd = np.maximum(qflx_snow_grnd + qflx_rain_grnd, 0.0)

    adjusted_targets: dict[str, np.ndarray] = {}
    adjusted_targets["qflx_prec_grnd"] = qprec_grnd
    adjusted_targets["qflx_snow_grnd"] = qflx_snow_grnd
    adjusted_targets["qflx_rain_grnd"] = qflx_rain_grnd
    adjusted_targets["qflx_leafdrip"] = qflx_leafdrip

    adjusted_state = CoupledColumnState(
        h2osoi_liq=_to_np(state.h2osoi_liq),
        h2osoi_ice=_to_np(state.h2osoi_ice),
        h2ocan=h2ocan_after_drip,
        h2osfc=_to_np(state.h2osfc),
        frac_h2osfc=_to_np(state.frac_h2osfc),
        frac_sno=_active_snow_fraction_numpy(state),
        zwt=_to_np(state.zwt),
        t_soisno=_to_np(state.t_soisno),
        t_h2osfc=_to_np(state.t_h2osfc) if state.t_h2osfc is not None else None,
        t10=_to_np(state.t10) if state.t10 is not None else None,
        t10_history=_to_np(state.t10_history) if state.t10_history is not None else None,
        h2osno=snow_storage_from_state(state),
        snow_depth=_to_np(state.snow_depth) if state.snow_depth is not None else snow_depth_from_storage(snow_storage_from_state(state)),
        snow_t=_to_np(state.snow_t) if state.snow_t is not None else None,
        snow_layers=state.snow_layers,
        int_snow=_to_np(state.int_snow) if state.int_snow is not None else None,
    )
    return adjusted_targets, adjusted_state


def apply_canopy_evaporation(
    surface_targets: dict[str, np.ndarray] | dict[str, torch.Tensor],
    state: CoupledColumnState,
    dt: float,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> tuple[dict[str, np.ndarray] | dict[str, torch.Tensor], CoupledColumnState]:
    """Drain post-interception canopy storage using predicted canopy evaporation."""
    adjusted_targets = dict(surface_targets)
    first = next(iter(adjusted_targets.values()))
    if isinstance(first, torch.Tensor):
        if device is None:
            device = first.device
        if dtype is None:
            dtype = first.dtype
        h2ocan_old = _to_torch(state.h2ocan, device, dtype)
        evap_pred = _to_torch(adjusted_targets.get("qflx_evap_canopy", 0.0), device, dtype)
        evap_pos = torch.minimum(torch.clamp(evap_pred, min=0.0), torch.clamp(h2ocan_old, min=0.0) / dt)
        evap = torch.where(evap_pred > 0.0, evap_pos, evap_pred)
        h2ocan_new = torch.clamp(h2ocan_old - dt * evap, min=0.0)
        adjusted_targets["qflx_evap_canopy"] = evap
        adjusted_state = CoupledColumnState(
            h2osoi_liq=_to_torch(state.h2osoi_liq, device, dtype),
            h2osoi_ice=_to_torch(state.h2osoi_ice, device, dtype),
            h2ocan=h2ocan_new,
            h2osfc=_to_torch(state.h2osfc, device, dtype),
            frac_h2osfc=_to_torch(state.frac_h2osfc, device, dtype),
            frac_sno=_to_torch(state.frac_sno, device, dtype),
            zwt=_to_torch(state.zwt, device, dtype),
            t_soisno=_to_torch(state.t_soisno, device, dtype),
            t_h2osfc=None if state.t_h2osfc is None else _to_torch(state.t_h2osfc, device, dtype),
            t10=None if state.t10 is None else _to_torch(state.t10, device, dtype),
            t10_history=None if state.t10_history is None else _to_torch(state.t10_history, device, dtype),
            h2osno=_to_torch(snow_storage_from_state(state), device, dtype),
            snow_depth=_to_torch(state.snow_depth, device, dtype) if state.snow_depth is not None else snow_depth_from_storage_torch(_to_torch(snow_storage_from_state(state), device, dtype), device=device, dtype=dtype),
            snow_t=None if state.snow_t is None else _to_torch(state.snow_t, device, dtype),
            snow_layers=_snow_layers_to_device(state.snow_layers, device, dtype),
            int_snow=None if state.int_snow is None else _to_torch(state.int_snow, device, dtype),
        )
        return adjusted_targets, adjusted_state

    h2ocan_old = _to_np(state.h2ocan)
    evap_pred = np.asarray(adjusted_targets.get("qflx_evap_canopy", 0.0), dtype=np.float64)
    evap_pos = np.minimum(np.maximum(evap_pred, 0.0), np.maximum(h2ocan_old, 0.0) / dt)
    evap = np.where(evap_pred > 0.0, evap_pos, evap_pred)
    h2ocan_new = np.maximum(h2ocan_old - dt * evap, 0.0)
    adjusted_targets["qflx_evap_canopy"] = evap
    adjusted_state = CoupledColumnState(
        h2osoi_liq=_to_np(state.h2osoi_liq),
        h2osoi_ice=_to_np(state.h2osoi_ice),
        h2ocan=h2ocan_new,
        h2osfc=_to_np(state.h2osfc),
        frac_h2osfc=_to_np(state.frac_h2osfc),
        frac_sno=_active_snow_fraction_numpy(state),
        zwt=_to_np(state.zwt),
        t_soisno=_to_np(state.t_soisno),
        t_h2osfc=_to_np(state.t_h2osfc) if state.t_h2osfc is not None else None,
        t10=_to_np(state.t10) if state.t10 is not None else None,
        t10_history=_to_np(state.t10_history) if state.t10_history is not None else None,
        h2osno=snow_storage_from_state(state),
        snow_depth=_to_np(state.snow_depth) if state.snow_depth is not None else snow_depth_from_storage(snow_storage_from_state(state)),
        snow_t=_to_np(state.snow_t) if state.snow_t is not None else None,
        snow_layers=state.snow_layers,
        int_snow=_to_np(state.int_snow) if state.int_snow is not None else None,
    )
    return adjusted_targets, adjusted_state


def coupled_soil_state_diagnostics(
    state: CoupledColumnState,
    static: CoupledStaticParameters,
    nprofile: int = 8,
) -> dict[str, np.ndarray]:
    """Derive coupled soil diagnostics for the surface emulator from current state."""
    dz = static.hydrology_numpy.dz[None, None, :]
    watsat = static.hydrology_numpy.watsat[None, None, :]
    sucsat = static.hydrology_numpy.sucsat[None, None, :]
    bsw = static.hydrology_numpy.bsw[None, None, :]
    nlev_hydro = static.hydrology_numpy.dz.shape[-1]
    h2osoi_liq = _to_np(state.h2osoi_liq)[..., :nlev_hydro]
    h2osoi_ice = _to_np(state.h2osoi_ice)[..., :nlev_hydro]
    h2osoi_liqvol = np.array(h2osoi_liq / (dz * DENH2O))  # writable copy
    vol_ice = np.minimum(watsat, h2osoi_ice / (dz * DENICE))
    h2osoi_vol = h2osoi_liqvol + vol_ice
    eff_porosity = np.maximum(watsat - vol_ice, 0.01)
    icefrac = np.minimum(1.0, vol_ice / np.maximum(watsat, 1.0e-12))
    # Use effective porosity (pore space not filled by ice) so that partial
    # freezing doesn't collapse liquid saturation toward zero and cause
    # spurious -1e8 spikes in smp_l that are never seen in ELM training data.
    s_node = np.clip(h2osoi_liqvol / np.maximum(eff_porosity, 1.0e-12), 0.01, 1.0)
    smp_l = -sucsat * s_node ** (-bsw)
    smp_l = np.maximum(smp_l, -1.0e8)
    return {
        "h2osoi_liqvol": h2osoi_liqvol[..., :nprofile],
        "h2osoi_vol": h2osoi_vol[..., :nprofile],
        "eff_porosity": eff_porosity[..., :nprofile],
        "icefrac": icefrac[..., :nprofile],
        "smp_l": smp_l[..., :nprofile],
    }


def coupled_t10_from_state(state: CoupledColumnState) -> np.ndarray:
    """Return the prognostic 10-day running mean 2 m temperature."""
    if state.t10 is not None:
        return _to_np(state.t10)
    if state.t10_history is not None:
        return _to_np(state.t10_history).mean(axis=-1)
    raise ValueError("Cannot derive t10 without t10 or t10_history; seed it from t_ref2m.")


def coupled_t10_from_state_torch(state: CoupledColumnState, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Torch-native variant of ``coupled_t10_from_state`` for benchmark/device-resident paths."""
    if state.t10 is not None:
        return _to_torch(state.t10, device, dtype)
    if state.t10_history is not None:
        return _to_torch(_to_np(state.t10_history).mean(axis=-1), device, dtype)
    raise ValueError("Cannot derive t10 without t10 or t10_history; seed it from t_ref2m.")


def _snow_top_temp_from_state(state: CoupledColumnState) -> np.ndarray:
    if state.snow_layers is not None:
        return _to_np(state.snow_layers.t_snow)[..., 0]
    if state.snow_t is not None:
        return _to_np(state.snow_t)[..., 0]
    return _to_np(state.t_soisno)[..., 0]


def coupled_t_grnd_from_state(state: CoupledColumnState) -> np.ndarray:
    """Compute `t_grnd` following the ELM non-lake hydrology formulation."""
    t_soil_top = _to_np(state.t_soisno)[..., 0]
    if state.t_h2osfc is None:
        if (state.snow_layers is not None or state.snow_t is not None) and np.any(_to_np(state.frac_sno) > 0.0):
            return _to_np(state.frac_sno) * _snow_top_temp_from_state(state) + (
                1.0 - _to_np(state.frac_sno)
            ) * t_soil_top
        return t_soil_top
    frac_sno_eff = _to_np(state.frac_sno)
    frac_h2osfc = _to_np(state.frac_h2osfc)
    t_h2osfc = _to_np(state.t_h2osfc)
    if np.any(frac_sno_eff > 0.0):
        snow_top = _snow_top_temp_from_state(state)
        return (
            frac_sno_eff * snow_top
            + (1.0 - frac_sno_eff - frac_h2osfc) * t_soil_top
            + frac_h2osfc * t_h2osfc
        )
    return (1.0 - frac_h2osfc) * t_soil_top + frac_h2osfc * t_h2osfc


def coupled_t_grnd_from_state_torch(state: CoupledColumnState, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Torch-native ``t_grnd`` used by the benchmark path."""
    t_soil_top = _to_torch(state.t_soisno[..., 0], device, dtype)
    frac_h2osfc = _to_torch(state.frac_h2osfc, device, dtype)
    frac_sno_eff = _to_torch(state.frac_sno, device, dtype)
    if state.t_h2osfc is None:
        if (state.snow_layers is not None or state.snow_t is not None) and torch.any(frac_sno_eff > 0.0):
            snow_top = _to_torch(_snow_top_temp_from_state(state), device, dtype)
            return frac_sno_eff * snow_top + (1.0 - frac_sno_eff) * t_soil_top
        return t_soil_top
    t_h2osfc = _to_torch(state.t_h2osfc, device, dtype)
    if torch.any(frac_sno_eff > 0.0):
        snow_top = _to_torch(_snow_top_temp_from_state(state), device, dtype)
        return frac_sno_eff * snow_top + (1.0 - frac_sno_eff - frac_h2osfc) * t_soil_top + frac_h2osfc * t_h2osfc
    return (1.0 - frac_h2osfc) * t_soil_top + frac_h2osfc * t_h2osfc

def snow_storage_from_state(state: CoupledColumnState) -> np.ndarray:
    if state.h2osno is not None:
        if isinstance(state.h2osno, torch.Tensor):
            return state.h2osno
        return _to_np(state.h2osno)
    if state.snow_depth is not None:
        if isinstance(state.snow_depth, torch.Tensor):
            return state.snow_depth * SNOW_WATER_TO_DEPTH
        return _to_np(state.snow_depth) * SNOW_WATER_TO_DEPTH
    return np.zeros_like(_to_np(state.h2osfc))


def _snow_layer_count(depth: float) -> int:
    if depth < 0.01:
        return 0
    if depth <= 0.03:
        return 1
    if depth <= 0.07:
        return 2
    if depth <= 0.18:
        return 3
    if depth <= 0.41:
        return 4
    return 5


def _elm_snow_layer_depths_numpy(snow_depth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ELM's active snow-layer count and top-to-bottom layer depths."""
    depth = np.asarray(snow_depth, dtype=np.float64)
    nsnow = np.zeros(depth.shape, dtype=np.int32)
    dz = np.zeros(depth.shape + (MAX_SNOW_LAYERS,), dtype=np.float64)

    def assign(mask: np.ndarray, count: int, values: tuple[np.ndarray | float, ...]) -> None:
        if not np.any(mask):
            return
        nsnow[mask] = count
        for k, value in enumerate(values):
            val = value[mask] if isinstance(value, np.ndarray) else value
            dz[mask, k] = val

    assign((depth >= 0.01) & (depth <= 0.03), 1, (depth,))
    assign((depth > 0.03) & (depth <= 0.04), 2, (0.5 * depth, 0.5 * depth))
    assign((depth > 0.04) & (depth <= 0.07), 2, (0.02, depth - 0.02))
    assign((depth > 0.07) & (depth <= 0.12), 3, (0.02, 0.5 * (depth - 0.02), 0.5 * (depth - 0.02)))
    assign((depth > 0.12) & (depth <= 0.18), 3, (0.02, 0.05, depth - 0.07))
    assign((depth > 0.18) & (depth <= 0.29), 4, (0.02, 0.05, 0.5 * (depth - 0.07), 0.5 * (depth - 0.07)))
    assign((depth > 0.29) & (depth <= 0.41), 4, (0.02, 0.05, 0.11, depth - 0.18))
    assign((depth > 0.41) & (depth <= 0.64), 5, (0.02, 0.05, 0.11, 0.5 * (depth - 0.18), 0.5 * (depth - 0.18)))
    assign(depth > 0.64, 5, (0.02, 0.05, 0.11, 0.23, depth - 0.41))
    return nsnow, np.maximum(dz, 0.0)


def _elm_snow_layer_depths_torch(snow_depth: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Torch analogue of ELM's active snow-layer geometry."""
    depth = torch.as_tensor(snow_depth)
    nsnow = torch.zeros(depth.shape, device=depth.device, dtype=torch.int32)
    dz = torch.zeros(depth.shape + (MAX_SNOW_LAYERS,), device=depth.device, dtype=depth.dtype)

    def assign(mask: torch.Tensor, count: int, values: tuple[torch.Tensor | float, ...]) -> None:
        nonlocal nsnow, dz
        if not torch.any(mask):
            return
        nsnow = torch.where(mask, torch.full_like(nsnow, count), nsnow)
        for k, value in enumerate(values):
            val = torch.as_tensor(value, device=depth.device, dtype=depth.dtype)
            dz[..., k] = torch.where(mask, val, dz[..., k])

    assign((depth >= 0.01) & (depth <= 0.03), 1, (depth,))
    assign((depth > 0.03) & (depth <= 0.04), 2, (0.5 * depth, 0.5 * depth))
    assign((depth > 0.04) & (depth <= 0.07), 2, (0.02, depth - 0.02))
    assign((depth > 0.07) & (depth <= 0.12), 3, (0.02, 0.5 * (depth - 0.02), 0.5 * (depth - 0.02)))
    assign((depth > 0.12) & (depth <= 0.18), 3, (0.02, 0.05, depth - 0.07))
    assign((depth > 0.18) & (depth <= 0.29), 4, (0.02, 0.05, 0.5 * (depth - 0.07), 0.5 * (depth - 0.07)))
    assign((depth > 0.29) & (depth <= 0.41), 4, (0.02, 0.05, 0.11, depth - 0.18))
    assign((depth > 0.41) & (depth <= 0.64), 5, (0.02, 0.05, 0.11, 0.5 * (depth - 0.18), 0.5 * (depth - 0.18)))
    assign(depth > 0.64, 5, (0.02, 0.05, 0.11, 0.23, depth - 0.41))
    return nsnow, torch.clamp(dz, min=0.0)


def _snow_layers_from_storage_numpy(
    h2osno: np.ndarray,
    snow_depth: np.ndarray,
    snow_t: np.ndarray | None = None,
    frac_sno: np.ndarray | None = None,
) -> SnowLayerState:
    h2osno = np.asarray(h2osno, dtype=np.float64)
    snow_depth = np.asarray(snow_depth, dtype=np.float64)
    shape = h2osno.shape
    if frac_sno is None:
        frac = np.ones(shape, dtype=np.float64)
    else:
        frac_arr = np.asarray(frac_sno, dtype=np.float64)
        if frac_arr.shape != shape and frac_arr.ndim + 1 == len(shape) and frac_arr.shape == shape[:frac_arr.ndim]:
            frac_arr = frac_arr[..., None]
        frac = np.clip(np.broadcast_to(frac_arr, shape), 0.0, 1.0)
    layer_depth_selector = np.where(frac * np.maximum(snow_depth, 0.0) >= SNOW_MIN_ACTIVE_DEPTH, snow_depth, 0.0)
    snl = np.zeros(shape, dtype=np.int32)
    dz = np.zeros(shape + (MAX_SNOW_LAYERS,), dtype=np.float64)
    ice = np.zeros_like(dz)
    liq = np.zeros_like(dz)
    temp = np.zeros_like(dz)
    if snow_t is not None:
        temp_seed = np.broadcast_to(np.asarray(snow_t, dtype=np.float64), shape + (MAX_SNOW_LAYERS,))
    else:
        temp_seed = np.broadcast_to(np.full(shape, TFRZ, dtype=np.float64)[..., None], shape + (MAX_SNOW_LAYERS,))
    nsnow, dz_template = _elm_snow_layer_depths_numpy(layer_depth_selector)
    for k in range(1, MAX_SNOW_LAYERS + 1):
        mask = nsnow == k
        if not np.any(mask):
            continue
        snl[mask] = -k
        dz[mask, :k] = np.maximum(dz_template[mask, :k], 1.0e-6)
        mass_frac = dz_template[mask, :k] / np.maximum(snow_depth[mask, None], 1.0e-12)
        ice[mask, :k] = np.maximum(h2osno[mask, None] * mass_frac, 0.0)
        temp[mask, :k] = temp_seed[mask, :k]
    inactive_scalar = (nsnow == 0) & (h2osno > 1.0e-8)
    if np.any(inactive_scalar):
        dz[inactive_scalar, 0] = np.maximum(snow_depth[inactive_scalar], 0.0)
        ice[inactive_scalar, 0] = h2osno[inactive_scalar]
        temp[inactive_scalar, 0] = temp_seed[inactive_scalar, 0]
    # Initialise grain radius to fresh-snow value everywhere.
    rds = np.zeros(shape + (MAX_SNOW_LAYERS,), dtype=np.float64)
    for k in range(1, MAX_SNOW_LAYERS + 1):
        mask = nsnow >= k
        rds[mask, k - 1] = SNW_RDS_MIN
    return SnowLayerState(snl=snl, dz=dz, h2osoi_ice=ice, h2osoi_liq=liq, t_snow=temp, snw_rds=rds)


def _snow_layers_from_storage_torch(
    h2osno: np.ndarray | torch.Tensor,
    snow_depth: np.ndarray | torch.Tensor,
    snow_t: np.ndarray | torch.Tensor | None = None,
    frac_sno: np.ndarray | torch.Tensor | None = None,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> SnowLayerState:
    h2osno_t = _to_torch(h2osno, device, dtype)
    snow_depth_t = _to_torch(snow_depth, device, dtype)
    shape = h2osno_t.shape
    if frac_sno is None:
        frac_t = torch.ones(shape, device=device, dtype=dtype)
    else:
        frac_in = _to_torch(frac_sno, device, dtype)
        if tuple(frac_in.shape) != shape and frac_in.ndim + 1 == len(shape) and tuple(frac_in.shape) == shape[:frac_in.ndim]:
            frac_in = frac_in.unsqueeze(-1)
        frac_t = torch.clamp(torch.broadcast_to(frac_in, shape), min=0.0, max=1.0)
    layer_depth_selector = torch.where(
        frac_t * torch.clamp(snow_depth_t, min=0.0) >= SNOW_MIN_ACTIVE_DEPTH,
        snow_depth_t,
        torch.zeros_like(snow_depth_t),
    )
    snl = torch.zeros(shape, device=device, dtype=torch.int32)
    dz = torch.zeros(shape + (MAX_SNOW_LAYERS,), device=device, dtype=dtype)
    ice = torch.zeros_like(dz)
    liq = torch.zeros_like(dz)
    temp = torch.zeros_like(dz)
    if snow_t is not None:
        temp_seed = torch.broadcast_to(_to_torch(snow_t, device, dtype), shape + (MAX_SNOW_LAYERS,))
    else:
        temp_seed = torch.broadcast_to(torch.full(shape, TFRZ, device=device, dtype=dtype)[..., None], shape + (MAX_SNOW_LAYERS,))
    nsnow, dz_template = _elm_snow_layer_depths_torch(layer_depth_selector)
    for k in range(1, MAX_SNOW_LAYERS + 1):
        mask = nsnow == k
        if not torch.any(mask):
            continue
        snl = torch.where(mask, torch.full_like(snl, -k), snl)
        dz[mask, :k] = torch.clamp(dz_template[mask, :k], min=1.0e-6)
        mass_frac = dz_template[mask, :k] / torch.clamp(snow_depth_t[mask, None], min=1.0e-12)
        ice[mask, :k] = torch.clamp(h2osno_t[mask, None] * mass_frac, min=0.0)
        temp[mask, :k] = temp_seed[mask, :k]
    inactive_scalar = (nsnow == 0) & (h2osno_t > 1.0e-8)
    if torch.any(inactive_scalar):
        dz[inactive_scalar, 0] = torch.clamp(snow_depth_t[inactive_scalar], min=0.0)
        ice[inactive_scalar, 0] = h2osno_t[inactive_scalar]
        temp[inactive_scalar, 0] = temp_seed[inactive_scalar, 0]
    rds_t = torch.full(shape + (MAX_SNOW_LAYERS,), SNW_RDS_MIN, device=device, dtype=dtype)
    for k in range(MAX_SNOW_LAYERS):
        rds_t[..., k] = torch.where(nsnow > k, torch.full(shape, SNW_RDS_MIN, device=device, dtype=dtype), torch.zeros(shape, device=device, dtype=dtype))
    return SnowLayerState(snl=snl, dz=dz, h2osoi_ice=ice, h2osoi_liq=liq, t_snow=temp, snw_rds=rds_t)


def _scalar_snow_layers_from_storage_numpy(
    h2osno: np.ndarray,
    snow_depth: np.ndarray,
    snow_t: np.ndarray | None = None,
) -> SnowLayerState:
    """Represent all snow as scalar storage, never as resolved active layers."""
    h2osno = np.asarray(h2osno, dtype=np.float64)
    snow_depth = np.asarray(snow_depth, dtype=np.float64)
    shape = h2osno.shape
    present = h2osno > 1.0e-8
    dz = np.zeros(shape + (MAX_SNOW_LAYERS,), dtype=np.float64)
    ice = np.zeros_like(dz)
    liq = np.zeros_like(dz)
    temp = np.full_like(dz, TFRZ)
    if snow_t is not None:
        temp_seed = np.broadcast_to(np.asarray(snow_t, dtype=np.float64), shape + (MAX_SNOW_LAYERS,))
    else:
        temp_seed = temp
    dz[..., 0] = np.where(present, np.maximum(snow_depth, 0.0), 0.0)
    ice[..., 0] = np.where(present, np.maximum(h2osno, 0.0), 0.0)
    temp[..., 0] = np.where(present, temp_seed[..., 0], TFRZ)
    return SnowLayerState(
        snl=np.zeros(shape, dtype=np.int32),
        dz=dz,
        h2osoi_ice=ice,
        h2osoi_liq=liq,
        t_snow=temp,
        snw_rds=np.zeros(shape + (MAX_SNOW_LAYERS,), dtype=np.float64),
    )


def _scalar_snow_layers_from_storage_torch(
    h2osno: np.ndarray | torch.Tensor,
    snow_depth: np.ndarray | torch.Tensor,
    snow_t: np.ndarray | torch.Tensor | None = None,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> SnowLayerState:
    """Torch analogue of scalar-only snow storage."""
    h2osno_t = _to_torch(h2osno, device, dtype)
    snow_depth_t = _to_torch(snow_depth, device, dtype)
    shape = tuple(h2osno_t.shape)
    present = h2osno_t > 1.0e-8
    dz = torch.zeros(shape + (MAX_SNOW_LAYERS,), device=device, dtype=dtype)
    ice = torch.zeros_like(dz)
    liq = torch.zeros_like(dz)
    temp = torch.full_like(dz, TFRZ)
    if snow_t is not None:
        temp_seed = torch.broadcast_to(_to_torch(snow_t, device, dtype), shape + (MAX_SNOW_LAYERS,))
    else:
        temp_seed = temp
    dz[..., 0] = torch.where(present, torch.clamp(snow_depth_t, min=0.0), torch.zeros_like(snow_depth_t))
    ice[..., 0] = torch.where(present, torch.clamp(h2osno_t, min=0.0), torch.zeros_like(h2osno_t))
    temp[..., 0] = torch.where(present, temp_seed[..., 0], torch.full_like(h2osno_t, TFRZ))
    return SnowLayerState(
        snl=torch.zeros(shape, device=device, dtype=torch.int32),
        dz=dz,
        h2osoi_ice=ice,
        h2osoi_liq=liq,
        t_snow=temp,
        snw_rds=torch.zeros(shape + (MAX_SNOW_LAYERS,), device=device, dtype=dtype),
    )


def force_scalar_snow_state_numpy(state: CoupledColumnState) -> CoupledColumnState:
    """Collapse snow storage to scalar H2OSNO/SNOWDP with no active snow layers."""
    if state.h2osno is None:
        return state
    h2osno_raw = np.asarray(state.h2osno, dtype=np.float64)
    h2osno = np.where(np.isfinite(h2osno_raw), np.maximum(h2osno_raw, 0.0), 0.0)
    snow_depth_raw = (
        np.asarray(state.snow_depth, dtype=np.float64)
        if state.snow_depth is not None
        else snow_depth_from_storage(h2osno)
    )
    snow_depth = np.where(h2osno > 1.0e-8, np.maximum(np.where(np.isfinite(snow_depth_raw), snow_depth_raw, 0.0), 0.0), 0.0)
    frac_raw = (
        np.asarray(state.frac_sno, dtype=np.float64)
        if state.frac_sno is not None
        else frac_sno_from_snow_storage(h2osno, snow_depth)
    )
    frac_sno = np.where(h2osno > 1.0e-8, np.clip(np.where(np.isfinite(frac_raw), frac_raw, 0.0), 0.0, 1.0), 0.0)
    snow_t = np.asarray(state.snow_t, dtype=np.float64) if state.snow_t is not None else None
    snow_layers = _scalar_snow_layers_from_storage_numpy(h2osno, snow_depth, snow_t=snow_t)
    int_snow = None
    if state.int_snow is not None:
        int_raw = np.asarray(state.int_snow, dtype=np.float64)
        int_snow = np.where(h2osno > 1.0e-8, np.maximum(np.where(np.isfinite(int_raw), int_raw, 0.0), h2osno), 0.0)
    return CoupledColumnState(
        h2osoi_liq=_to_np(state.h2osoi_liq),
        h2osoi_ice=_to_np(state.h2osoi_ice),
        h2ocan=_to_np(state.h2ocan),
        h2osfc=_to_np(state.h2osfc),
        frac_h2osfc=_to_np(state.frac_h2osfc),
        frac_sno=frac_sno,
        zwt=_to_np(state.zwt),
        t_soisno=_to_np(state.t_soisno),
        wa=_to_np(state.wa) if state.wa is not None else None,
        t_h2osfc=_to_np(state.t_h2osfc) if state.t_h2osfc is not None else None,
        t10=_to_np(state.t10) if state.t10 is not None else None,
        t10_history=_to_np(state.t10_history) if state.t10_history is not None else None,
        h2osno=h2osno,
        snow_depth=snow_depth,
        snow_t=snow_layers.t_snow,
        snow_layers=snow_layers,
        int_snow=int_snow,
    )


def force_scalar_snow_state_torch(
    state: CoupledColumnState,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> CoupledColumnState:
    """Torch analogue of force_scalar_snow_state_numpy."""
    if state.h2osno is None:
        return state
    h2osno_raw = _to_torch(state.h2osno, device, dtype)
    h2osno = torch.where(torch.isfinite(h2osno_raw), torch.clamp(h2osno_raw, min=0.0), torch.zeros_like(h2osno_raw))
    snow_depth_raw = (
        _to_torch(state.snow_depth, device, dtype)
        if state.snow_depth is not None
        else snow_depth_from_storage_torch(h2osno, device=device, dtype=dtype)
    )
    snow_depth = torch.where(
        h2osno > 1.0e-8,
        torch.clamp(torch.where(torch.isfinite(snow_depth_raw), snow_depth_raw, torch.zeros_like(snow_depth_raw)), min=0.0),
        torch.zeros_like(h2osno),
    )
    if state.frac_sno is not None:
        frac_raw = _to_torch(state.frac_sno, device, dtype)
    else:
        frac_raw = frac_sno_from_snow_storage_torch(h2osno, snow_depth, device=device, dtype=dtype)
    frac_sno = torch.where(
        h2osno > 1.0e-8,
        torch.clamp(torch.where(torch.isfinite(frac_raw), frac_raw, torch.zeros_like(frac_raw)), min=0.0, max=1.0),
        torch.zeros_like(h2osno),
    )
    snow_t = _to_torch(state.snow_t, device, dtype) if state.snow_t is not None else None
    snow_layers = _scalar_snow_layers_from_storage_torch(h2osno, snow_depth, snow_t=snow_t, device=device, dtype=dtype)
    int_snow = None
    if state.int_snow is not None:
        int_raw = _to_torch(state.int_snow, device, dtype)
        int_snow = torch.where(
            h2osno > 1.0e-8,
            torch.maximum(torch.where(torch.isfinite(int_raw), int_raw, torch.zeros_like(int_raw)), h2osno),
            torch.zeros_like(h2osno),
        )
    return CoupledColumnState(
        h2osoi_liq=_to_torch(state.h2osoi_liq, device, dtype),
        h2osoi_ice=_to_torch(state.h2osoi_ice, device, dtype),
        h2ocan=_to_torch(state.h2ocan, device, dtype),
        h2osfc=_to_torch(state.h2osfc, device, dtype),
        frac_h2osfc=_to_torch(state.frac_h2osfc, device, dtype),
        frac_sno=frac_sno,
        zwt=_to_torch(state.zwt, device, dtype),
        t_soisno=_to_torch(state.t_soisno, device, dtype),
        wa=_to_torch(state.wa, device, dtype) if state.wa is not None else None,
        t_h2osfc=_to_torch(state.t_h2osfc, device, dtype) if state.t_h2osfc is not None else None,
        t10=_to_torch(state.t10, device, dtype) if state.t10 is not None else None,
        t10_history=_to_torch(state.t10_history, device, dtype) if state.t10_history is not None else None,
        h2osno=h2osno,
        snow_depth=snow_depth,
        snow_t=snow_layers.t_snow,
        snow_layers=snow_layers,
        int_snow=int_snow,
    )


def _snow_layers_to_device(snow_layers: SnowLayerState | None, device: torch.device, dtype: torch.dtype) -> SnowLayerState | None:
    if snow_layers is None:
        return None
    snw_rds = getattr(snow_layers, "snw_rds", None)
    return SnowLayerState(
        snl=torch.as_tensor(snow_layers.snl, device=device, dtype=torch.int32),
        dz=torch.as_tensor(snow_layers.dz, device=device, dtype=dtype),
        h2osoi_ice=torch.as_tensor(snow_layers.h2osoi_ice, device=device, dtype=dtype),
        h2osoi_liq=torch.as_tensor(snow_layers.h2osoi_liq, device=device, dtype=dtype),
        t_snow=torch.as_tensor(snow_layers.t_snow, device=device, dtype=dtype),
        snw_rds=torch.as_tensor(snw_rds, device=device, dtype=dtype) if snw_rds is not None else None,
    )


def _snow_layers_to_cpu(snow_layers: SnowLayerState | None) -> SnowLayerState | None:
    if snow_layers is None:
        return None
    def arr(x, *, int32: bool = False):
        if isinstance(x, torch.Tensor):
            return np.asarray(x.detach().cpu().numpy(), dtype=np.int32 if int32 else np.float64)
        return np.asarray(x, dtype=np.int32 if int32 else np.float64)
    snw_rds = getattr(snow_layers, "snw_rds", None)
    return SnowLayerState(
        snl=arr(snow_layers.snl, int32=True),
        dz=arr(snow_layers.dz),
        h2osoi_ice=arr(snow_layers.h2osoi_ice),
        h2osoi_liq=arr(snow_layers.h2osoi_liq),
        t_snow=arr(snow_layers.t_snow),
        snw_rds=arr(snw_rds) if snw_rds is not None else None,
    )


def _snow_layers_slice(snow_layers: SnowLayerState | None, start: int, stop: int) -> SnowLayerState | None:
    if snow_layers is None:
        return None
    return SnowLayerState(
        snl=snow_layers.snl[start:stop],
        dz=snow_layers.dz[start:stop],
        h2osoi_ice=snow_layers.h2osoi_ice[start:stop],
        h2osoi_liq=snow_layers.h2osoi_liq[start:stop],
        t_snow=snow_layers.t_snow[start:stop],
        snw_rds=snow_layers.snw_rds[start:stop] if snow_layers.snw_rds is not None else None,
    )


def _snow_layers_cat(states: list[SnowLayerState | None]) -> SnowLayerState | None:
    values = [s for s in states if s is not None]
    if not values:
        return None
    first = values[0]
    if isinstance(first.snl, torch.Tensor):
        rds_vals = [s.snw_rds for s in values]
        rds_cat = torch.cat(rds_vals, dim=0) if all(r is not None for r in rds_vals) else None
        return SnowLayerState(
            snl=torch.cat([s.snl for s in values], dim=0),
            dz=torch.cat([s.dz for s in values], dim=0),
            h2osoi_ice=torch.cat([s.h2osoi_ice for s in values], dim=0),
            h2osoi_liq=torch.cat([s.h2osoi_liq for s in values], dim=0),
            t_snow=torch.cat([s.t_snow for s in values], dim=0),
            snw_rds=rds_cat,
        )
    rds_np_vals = [s.snw_rds for s in values]
    rds_np = np.concatenate([np.asarray(r, dtype=np.float64) for r in rds_np_vals], axis=0) if all(r is not None for r in rds_np_vals) else None
    return SnowLayerState(
        snl=np.concatenate([np.asarray(s.snl, dtype=np.int32) for s in values], axis=0),
        dz=np.concatenate([np.asarray(s.dz, dtype=np.float64) for s in values], axis=0),
        h2osoi_ice=np.concatenate([np.asarray(s.h2osoi_ice, dtype=np.float64) for s in values], axis=0),
        h2osoi_liq=np.concatenate([np.asarray(s.h2osoi_liq, dtype=np.float64) for s in values], axis=0),
        t_snow=np.concatenate([np.asarray(s.t_snow, dtype=np.float64) for s in values], axis=0),
        snw_rds=rds_np,
    )


def _cell_index_select(value, indices: torch.Tensor, *, ncell: int, allow_1d: bool = False):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.ndim >= 1 and value.shape[0] == ncell and (allow_1d or value.ndim >= 2):
            if indices.numel() > 0:
                idx_min = int(indices.detach().min().cpu().item())
                idx_max = int(indices.detach().max().cpu().item())
                if idx_min < 0 or idx_max >= value.shape[0]:
                    raise IndexError(
                        f"cell index range [{idx_min}, {idx_max}] is invalid for tensor "
                        f"with leading dimension {value.shape[0]}"
                    )
            return value.index_select(0, indices.to(device=value.device))
        return value
    if isinstance(value, Mapping):
        return {
            key: _cell_index_select(item, indices, ncell=ncell, allow_1d=allow_1d)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _cell_index_select(item, indices, ncell=ncell, allow_1d=allow_1d)
            for item in value
        ]
    if isinstance(value, tuple):
        return tuple(
            _cell_index_select(item, indices, ncell=ncell, allow_1d=allow_1d)
            for item in value
        )
    try:
        arr = np.asarray(value)
    except (TypeError, ValueError):
        return value
    if arr.ndim >= 1 and arr.shape[0] == ncell and (allow_1d or arr.ndim >= 2):
        idx_np = indices.detach().cpu().numpy()
        if idx_np.size:
            idx_min = int(idx_np.min())
            idx_max = int(idx_np.max())
            if idx_min < 0 or idx_max >= arr.shape[0]:
                raise IndexError(
                    f"cell index range [{idx_min}, {idx_max}] is invalid for array "
                    f"with leading dimension {arr.shape[0]}"
                )
        return arr[idx_np]
    return value


def _cell_index_mapping(
    values: Mapping[str, np.ndarray | torch.Tensor],
    indices: torch.Tensor,
    *,
    ncell: int,
) -> dict[str, np.ndarray | torch.Tensor]:
    return {
        name: _cell_index_select(value, indices, ncell=ncell)
        for name, value in values.items()
    }


def _cell_index_dataclass(obj, indices: torch.Tensor, *, ncell: int):
    if obj is None:
        return None
    return type(obj)(
        **{
            name: _cell_index_select(getattr(obj, name), indices, ncell=ncell)
            for name in obj.__dataclass_fields__
        }
    )


def _cell_index_static_parameters(
    static: CoupledStaticParameters | None,
    indices: torch.Tensor,
    *,
    ncell: int,
) -> CoupledStaticParameters | None:
    if static is None:
        return None
    subset = CoupledStaticParameters(
        rootfr=static.rootfr,
        hydrology_numpy=_cell_index_dataclass(static.hydrology_numpy, indices, ncell=ncell),
        hydrology_torch=_cell_index_dataclass(static.hydrology_torch, indices, ncell=ncell),
        soil_temperature_numpy=_cell_index_dataclass(static.soil_temperature_numpy, indices, ncell=ncell),
        soil_temperature_torch=_cell_index_dataclass(static.soil_temperature_torch, indices, ncell=ncell),
        micro_sigma=_cell_index_select(static.micro_sigma, indices, ncell=ncell, allow_1d=True),
        pc=_cell_index_select(static.pc, indices, ncell=ncell, allow_1d=True),
        topo_slope=_cell_index_select(static.topo_slope, indices, ncell=ncell, allow_1d=True),
        pft_type=static.pft_type,
        smpso=static.smpso,
        smpsc=static.smpsc,
        soil_color=_cell_index_select(static.soil_color, indices, ncell=ncell, allow_1d=True),
        leafcn=static.leafcn,
        slatop=static.slatop,
        flnr=static.flnr,
        mbbopt=static.mbbopt,
        z0mr=static.z0mr,
        rholvis=static.rholvis,
        rholnir=static.rholnir,
        taulvis=static.taulvis,
        taulnir=static.taulnir,
        rhosvis=static.rhosvis,
        rhosnir=static.rhosnir,
        tausvis=static.tausvis,
        tausnir=static.tausnir,
        xl=static.xl,
        tc_stress_c=static.tc_stress_c,
        n_melt=_cell_index_select(static.n_melt, indices, ncell=ncell, allow_1d=True),
        rootfr_torch=static.rootfr_torch,
        rootfr_norm_torch=static.rootfr_norm_torch,
    )
    if hasattr(static, "n_melt"):
        subset.n_melt = _cell_index_select(getattr(static, "n_melt"), indices, ncell=ncell, allow_1d=True)
    return subset


def _shape_tuple(value) -> tuple[int, ...] | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return tuple(value.shape)
    try:
        return tuple(np.asarray(value).shape)
    except (TypeError, ValueError):
        return None


def _snow_layers_shape_ok(snow_layers: SnowLayerState | None, batch_shape: tuple[int, ...]) -> bool:
    if snow_layers is None:
        return False
    layer_shape = batch_shape + (MAX_SNOW_LAYERS,)
    if _shape_tuple(getattr(snow_layers, "snl", None)) != batch_shape:
        return False
    for name in ("dz", "h2osoi_ice", "h2osoi_liq", "t_snow"):
        if _shape_tuple(getattr(snow_layers, name, None)) != layer_shape:
            return False
    snw_rds = getattr(snow_layers, "snw_rds", None)
    return snw_rds is None or _shape_tuple(snw_rds) == layer_shape


def _cell_index_snow_layers(
    snow_layers: SnowLayerState | None,
    indices: torch.Tensor,
    *,
    batch_shape: tuple[int, ...],
) -> SnowLayerState | None:
    if not _snow_layers_shape_ok(snow_layers, batch_shape):
        return None

    def sel(value):
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            return value.index_select(0, indices.to(device=value.device))
        return np.asarray(value)[indices.detach().cpu().numpy()]

    return SnowLayerState(
        snl=sel(snow_layers.snl),
        dz=sel(snow_layers.dz),
        h2osoi_ice=sel(snow_layers.h2osoi_ice),
        h2osoi_liq=sel(snow_layers.h2osoi_liq),
        t_snow=sel(snow_layers.t_snow),
        snw_rds=sel(getattr(snow_layers, "snw_rds", None)),
    )


def _cell_index_coupled_state(state: CoupledColumnState, indices: torch.Tensor) -> CoupledColumnState:
    ncell = int(state.h2osfc.shape[0])
    batch_shape = tuple(state.h2osfc.shape)

    def sel(value):
        return _cell_index_select(value, indices, ncell=ncell, allow_1d=True)

    def sel_batch_optional(value):
        return sel(value) if _shape_tuple(value) == batch_shape else None

    def sel_layer_optional(value):
        return sel(value) if _shape_tuple(value) == batch_shape + (MAX_SNOW_LAYERS,) else None

    return CoupledColumnState(
        h2osoi_liq=sel(state.h2osoi_liq),
        h2osoi_ice=sel(state.h2osoi_ice),
        h2ocan=sel(state.h2ocan),
        h2osfc=sel(state.h2osfc),
        frac_h2osfc=sel(state.frac_h2osfc),
        frac_sno=sel(state.frac_sno),
        zwt=sel(state.zwt),
        t_soisno=sel(state.t_soisno),
        wa=sel(state.wa),
        t_h2osfc=sel(state.t_h2osfc),
        t10=sel(state.t10),
        t10_history=sel(state.t10_history),
        h2osno=sel(state.h2osno),
        snow_depth=sel(state.snow_depth),
        snow_t=sel_layer_optional(state.snow_t),
        snow_layers=_cell_index_snow_layers(state.snow_layers, indices, batch_shape=batch_shape),
        int_snow=sel_batch_optional(state.int_snow),
    )


def _cell_scatter_value(base, indices: torch.Tensor, update):
    if update is None:
        return base
    if base is None:
        return update
    if isinstance(base, torch.Tensor):
        out = base.clone()
        upd = update.to(device=base.device, dtype=base.dtype) if isinstance(update, torch.Tensor) else torch.as_tensor(update, device=base.device, dtype=base.dtype)
        target_shape = (int(indices.numel()),) + tuple(base.shape[1:])
        if tuple(upd.shape) != target_shape and upd.numel() == int(np.prod(target_shape)):
            upd = upd.reshape(target_shape)
        out.index_copy_(0, indices.to(device=base.device), upd)
        return out
    out = np.array(base, copy=True)
    upd = np.asarray(update, dtype=out.dtype)
    target_shape = (int(indices.numel()),) + tuple(out.shape[1:])
    if upd.shape != target_shape and upd.size == int(np.prod(target_shape)):
        upd = upd.reshape(target_shape)
    out[indices.detach().cpu().numpy()] = upd
    return out


def _cell_scatter_snow_layers(
    base: SnowLayerState | None,
    indices: torch.Tensor,
    update: SnowLayerState | None,
) -> SnowLayerState | None:
    if update is None:
        return base
    if base is None:
        return update
    return SnowLayerState(
        snl=_cell_scatter_value(base.snl, indices, update.snl),
        dz=_cell_scatter_value(base.dz, indices, update.dz),
        h2osoi_ice=_cell_scatter_value(base.h2osoi_ice, indices, update.h2osoi_ice),
        h2osoi_liq=_cell_scatter_value(base.h2osoi_liq, indices, update.h2osoi_liq),
        t_snow=_cell_scatter_value(base.t_snow, indices, update.t_snow),
        snw_rds=_cell_scatter_value(getattr(base, "snw_rds", None), indices, getattr(update, "snw_rds", None)),
    )


def _empty_snow_layers_like(value_like, update: SnowLayerState) -> SnowLayerState:
    if isinstance(value_like, torch.Tensor):
        shape = tuple(value_like.shape)
        device = value_like.device
        dtype = value_like.dtype
        snw_rds_update = getattr(update, "snw_rds", None)
        return SnowLayerState(
            snl=torch.zeros(shape, device=device, dtype=torch.int32),
            dz=torch.zeros(shape + (MAX_SNOW_LAYERS,), device=device, dtype=dtype),
            h2osoi_ice=torch.zeros(shape + (MAX_SNOW_LAYERS,), device=device, dtype=dtype),
            h2osoi_liq=torch.zeros(shape + (MAX_SNOW_LAYERS,), device=device, dtype=dtype),
            t_snow=torch.full(shape + (MAX_SNOW_LAYERS,), TFRZ, device=device, dtype=dtype),
            snw_rds=(
                torch.zeros(shape + (MAX_SNOW_LAYERS,), device=device, dtype=dtype)
                if snw_rds_update is not None
                else None
            ),
        )
    arr = np.asarray(value_like)
    shape = tuple(arr.shape)
    snw_rds_update = getattr(update, "snw_rds", None)
    return SnowLayerState(
        snl=np.zeros(shape, dtype=np.int32),
        dz=np.zeros(shape + (MAX_SNOW_LAYERS,), dtype=np.float64),
        h2osoi_ice=np.zeros(shape + (MAX_SNOW_LAYERS,), dtype=np.float64),
        h2osoi_liq=np.zeros(shape + (MAX_SNOW_LAYERS,), dtype=np.float64),
        t_snow=np.full(shape + (MAX_SNOW_LAYERS,), TFRZ, dtype=np.float64),
        snw_rds=np.zeros(shape + (MAX_SNOW_LAYERS,), dtype=np.float64) if snw_rds_update is not None else None,
    )


def _empty_snow_t_like(value_like):
    if isinstance(value_like, torch.Tensor):
        return torch.full(
            tuple(value_like.shape) + (MAX_SNOW_LAYERS,),
            TFRZ,
            device=value_like.device,
            dtype=value_like.dtype,
        )
    arr = np.asarray(value_like)
    return np.full(tuple(arr.shape) + (MAX_SNOW_LAYERS,), TFRZ, dtype=np.float64)


def _cell_scatter_coupled_state(
    base: CoupledColumnState,
    indices: torch.Tensor,
    update: CoupledColumnState,
) -> CoupledColumnState:
    base_snow_layers = base.snow_layers
    layer_shape_source = base.h2osno if base.h2osno is not None else base.h2osfc
    expected_layer_shape = tuple(layer_shape_source.shape)
    if (
        update.snow_layers is not None
        and (
            base_snow_layers is None
            or not _snow_layers_shape_ok(base_snow_layers, expected_layer_shape)
        )
    ):
        base_snow_layers = _empty_snow_layers_like(layer_shape_source, update.snow_layers)
    base_snow_t = base.snow_t
    if (
        update.snow_t is not None
        and (
            base_snow_t is None
            or tuple(base_snow_t.shape) != expected_layer_shape + (MAX_SNOW_LAYERS,)
        )
    ):
        base_snow_t = _empty_snow_t_like(layer_shape_source)
    base_int_snow = base.int_snow
    if (
        update.int_snow is not None
        and (
            base_int_snow is None
            or tuple(base_int_snow.shape) != expected_layer_shape
        )
    ):
        if isinstance(layer_shape_source, torch.Tensor):
            base_int_snow = torch.zeros_like(layer_shape_source)
        else:
            base_int_snow = np.zeros_like(np.asarray(layer_shape_source, dtype=np.float64))
    return CoupledColumnState(
        h2osoi_liq=_cell_scatter_value(base.h2osoi_liq, indices, update.h2osoi_liq),
        h2osoi_ice=_cell_scatter_value(base.h2osoi_ice, indices, update.h2osoi_ice),
        h2ocan=_cell_scatter_value(base.h2ocan, indices, update.h2ocan),
        h2osfc=_cell_scatter_value(base.h2osfc, indices, update.h2osfc),
        frac_h2osfc=_cell_scatter_value(base.frac_h2osfc, indices, update.frac_h2osfc),
        frac_sno=_cell_scatter_value(base.frac_sno, indices, update.frac_sno),
        zwt=_cell_scatter_value(base.zwt, indices, update.zwt),
        t_soisno=_cell_scatter_value(base.t_soisno, indices, update.t_soisno),
        wa=_cell_scatter_value(base.wa, indices, update.wa),
        t_h2osfc=_cell_scatter_value(base.t_h2osfc, indices, update.t_h2osfc),
        t10=_cell_scatter_value(base.t10, indices, update.t10),
        t10_history=_cell_scatter_value(base.t10_history, indices, update.t10_history),
        h2osno=_cell_scatter_value(base.h2osno, indices, update.h2osno),
        snow_depth=_cell_scatter_value(base.snow_depth, indices, update.snow_depth),
        snow_t=_cell_scatter_value(base_snow_t, indices, update.snow_t),
        snow_layers=_cell_scatter_snow_layers(base_snow_layers, indices, update.snow_layers),
        int_snow=_cell_scatter_value(base_int_snow, indices, update.int_snow),
    )


def _cell_indices_from_mask(mask: torch.Tensor) -> torch.Tensor:
    if mask.ndim == 0:
        return torch.zeros((1,), device=mask.device, dtype=torch.long) if bool(mask.detach().cpu().item()) else torch.empty((0,), device=mask.device, dtype=torch.long)
    if mask.ndim == 1:
        cell_mask = mask
    else:
        cell_mask = torch.any(mask.reshape(mask.shape[0], -1), dim=1)
    return torch.nonzero(cell_mask, as_tuple=False).reshape(-1).to(dtype=torch.long)


def _active_snow_cell_indices_torch(
    state: CoupledColumnState,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    batch_shape = tuple(_to_torch(state.h2osfc, device, dtype).shape)
    if (
        state.snow_layers is not None
        and _snow_layers_shape_ok(state.snow_layers, batch_shape)
    ):
        active = torch.as_tensor(state.snow_layers.snl, device=device, dtype=torch.int32) < 0
        if state.h2osno is not None:
            active = active & (_to_torch(state.h2osno, device, dtype) > MIN_SNOW_LAYER_MASS)
        if state.snow_depth is not None:
            active = active & (_to_torch(state.snow_depth, device, dtype) >= SNOW_MIN_ACTIVE_DEPTH)
    else:
        ref = _to_torch(state.h2osfc, device, dtype)
        h2osno = _to_torch(state.h2osno, device, dtype) if state.h2osno is not None else torch.zeros_like(ref)
        snow_depth = _to_torch(state.snow_depth, device, dtype) if state.snow_depth is not None else torch.zeros_like(h2osno)
        active = (h2osno > MIN_SNOW_LAYER_MASS) & (snow_depth >= SNOW_MIN_ACTIVE_DEPTH)
    return _cell_indices_from_mask(active)


def _repair_snow_layer_state_shape_torch(
    state: CoupledColumnState,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> CoupledColumnState:
    ref = _to_torch(state.h2osfc, device, dtype)
    batch_shape = tuple(ref.shape)
    layers = state.snow_layers
    snow_t = _to_torch(state.snow_t, device, dtype) if state.snow_t is not None else None
    snow_t_ok = snow_t is not None and tuple(snow_t.shape) == batch_shape + (MAX_SNOW_LAYERS,)
    layers_ok = _snow_layers_shape_ok(layers, batch_shape)
    int_snow_raw = _to_torch(state.int_snow, device, dtype) if state.int_snow is not None else None
    int_snow = None
    int_snow_ok = False
    if int_snow_raw is not None:
        try:
            int_snow = _snow_batch_like_torch(int_snow_raw, batch_shape, device=device, dtype=dtype)
            int_snow_ok = True
        except RuntimeError:
            int_snow = int_snow_raw
    if layers_ok and snow_t_ok and (state.int_snow is None or int_snow_ok):
        return state
    h2osno = _to_torch(state.h2osno, device, dtype) if state.h2osno is not None else torch.zeros_like(ref)
    snow_depth = _to_torch(state.snow_depth, device, dtype) if state.snow_depth is not None else torch.zeros_like(h2osno)
    repaired_layers = (
        _snow_layers_to_device(layers, device, dtype)
        if layers_ok
        else _snow_layers_from_storage_torch(
            h2osno,
            snow_depth,
            snow_t=snow_t if snow_t_ok else None,
            frac_sno=state.frac_sno,
            device=device,
            dtype=dtype,
        )
    )
    repaired_snow_t = (
        snow_t
        if snow_t_ok
        else (
            repaired_layers.t_snow
            if repaired_layers is not None
            else torch.full(batch_shape + (MAX_SNOW_LAYERS,), TFRZ, device=device, dtype=dtype)
        )
    )
    repaired_int_snow = (
        int_snow
        if int_snow_ok
        else torch.where(h2osno > 1.0e-8, h2osno, torch.zeros_like(h2osno))
    )
    return CoupledColumnState(
        h2osoi_liq=state.h2osoi_liq,
        h2osoi_ice=state.h2osoi_ice,
        h2ocan=state.h2ocan,
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
        snow_t=repaired_snow_t,
        snow_layers=repaired_layers,
        int_snow=repaired_int_snow,
    )


def _state_with_inactive_snow_layers_torch(
    state: CoupledColumnState,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> CoupledColumnState:
    ref = _to_torch(state.h2osfc, device, dtype)
    shape = tuple(ref.shape)
    snow_t = (
        _to_torch(state.snow_t, device, dtype)
        if state.snow_t is not None
        else torch.full(shape + (MAX_SNOW_LAYERS,), TFRZ, device=device, dtype=dtype)
    )
    if tuple(snow_t.shape) != shape + (MAX_SNOW_LAYERS,):
        snow_t = torch.broadcast_to(snow_t, shape + (MAX_SNOW_LAYERS,))
    zero_layers = SnowLayerState(
        snl=torch.zeros(shape, device=device, dtype=torch.int32),
        dz=torch.zeros(shape + (MAX_SNOW_LAYERS,), device=device, dtype=dtype),
        h2osoi_ice=torch.zeros(shape + (MAX_SNOW_LAYERS,), device=device, dtype=dtype),
        h2osoi_liq=torch.zeros(shape + (MAX_SNOW_LAYERS,), device=device, dtype=dtype),
        t_snow=snow_t,
        snw_rds=torch.zeros(shape + (MAX_SNOW_LAYERS,), device=device, dtype=dtype),
    )
    return CoupledColumnState(
        h2osoi_liq=state.h2osoi_liq,
        h2osoi_ice=state.h2osoi_ice,
        h2ocan=state.h2ocan,
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
        snow_t=snow_t,
        snow_layers=zero_layers,
        int_snow=state.int_snow,
    )


def _snow_depth_from_layers_numpy(snow_layers: SnowLayerState) -> np.ndarray:
    snl = np.asarray(snow_layers.snl, dtype=np.int32)
    dz = np.asarray(snow_layers.dz, dtype=np.float64)
    ice = np.asarray(snow_layers.h2osoi_ice, dtype=np.float64)
    liq = np.asarray(snow_layers.h2osoi_liq, dtype=np.float64)
    nsnow = np.maximum(-snl, 0)
    depth = np.zeros(snl.shape, dtype=np.float64)
    for k in range(MAX_SNOW_LAYERS):
        depth = np.where(nsnow > k, depth + dz[..., k], depth)
    scalar_snow = (nsnow == 0) & (ice[..., 0] + liq[..., 0] > 1.0e-8)
    depth = np.where(scalar_snow, np.maximum(dz[..., 0], 0.0), depth)
    return np.maximum(depth, 0.0)


def _snow_depth_from_layers_torch(snow_layers: SnowLayerState, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    snl = torch.as_tensor(snow_layers.snl, device=device, dtype=torch.int32)
    dz = torch.as_tensor(snow_layers.dz, device=device, dtype=dtype)
    ice = torch.as_tensor(snow_layers.h2osoi_ice, device=device, dtype=dtype)
    liq = torch.as_tensor(snow_layers.h2osoi_liq, device=device, dtype=dtype)
    nsnow = torch.clamp(-snl, min=0)
    depth = torch.zeros(snl.shape, device=device, dtype=dtype)
    for k in range(MAX_SNOW_LAYERS):
        depth = torch.where(nsnow > k, depth + dz[..., k], depth)
    scalar_snow = (nsnow == 0) & (ice[..., 0] + liq[..., 0] > 1.0e-8)
    depth = torch.where(scalar_snow, torch.clamp(dz[..., 0], min=0.0), depth)
    return torch.clamp(depth, min=0.0)


def _snow_mass_from_layers_numpy(snow_layers: SnowLayerState) -> np.ndarray:
    snl = np.asarray(snow_layers.snl, dtype=np.int32)
    ice = np.asarray(snow_layers.h2osoi_ice, dtype=np.float64)
    liq = np.asarray(snow_layers.h2osoi_liq, dtype=np.float64)
    nsnow = np.maximum(-snl, 0)
    mass = np.zeros(snl.shape, dtype=np.float64)
    for k in range(MAX_SNOW_LAYERS):
        mass = np.where(nsnow > k, mass + np.maximum(ice[..., k] + liq[..., k], 0.0), mass)
    mass = np.where((nsnow == 0) & (ice[..., 0] > 0.0), np.maximum(ice[..., 0], 0.0), mass)
    return mass


def _snow_mass_from_layers_torch(snow_layers: SnowLayerState, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    snl = torch.as_tensor(snow_layers.snl, device=device, dtype=torch.int32)
    ice = torch.as_tensor(snow_layers.h2osoi_ice, device=device, dtype=dtype)
    liq = torch.as_tensor(snow_layers.h2osoi_liq, device=device, dtype=dtype)
    nsnow = torch.clamp(-snl, min=0)
    mass = torch.zeros(snl.shape, device=device, dtype=dtype)
    for k in range(MAX_SNOW_LAYERS):
        mass = torch.where(nsnow > k, mass + torch.clamp(ice[..., k] + liq[..., k], min=0.0), mass)
    mass = torch.where((nsnow == 0) & (ice[..., 0] > 0.0), torch.clamp(ice[..., 0], min=0.0), mass)
    return mass


def _active_snow_fraction_numpy(state: CoupledColumnState) -> np.ndarray:
    frac = np.clip(_to_np(state.frac_sno), 0.0, 1.0)
    h2osno = _to_np(state.h2osno) if state.h2osno is not None else None
    depth = _to_np(state.snow_depth) if state.snow_depth is not None else (snow_depth_from_storage(h2osno) if h2osno is not None else None)
    layers = state.snow_layers
    if layers is not None and _snow_layers_shape_ok(layers, tuple(frac.shape)):
        active = np.asarray(layers.snl, dtype=np.int32) < 0
        if h2osno is not None:
            active = active & (h2osno > MIN_SNOW_LAYER_MASS)
        if depth is not None:
            active = active & (depth > 0.0)
    elif h2osno is not None:
        active = h2osno > MIN_SNOW_LAYER_MASS
        if depth is not None:
            active = active & (frac * np.maximum(depth, 0.0) >= SNOW_MIN_ACTIVE_DEPTH)
    else:
        active = frac > 0.0
    return np.where(active, frac, 0.0)


def _total_snow_fraction_numpy(state: CoupledColumnState) -> np.ndarray:
    frac = np.clip(_to_np(state.frac_sno), 0.0, 1.0)
    h2osno = _to_np(state.h2osno) if state.h2osno is not None else None
    depth = _to_np(state.snow_depth) if state.snow_depth is not None else (snow_depth_from_storage(h2osno) if h2osno is not None else None)
    if h2osno is not None:
        snow_present = h2osno > 1.0e-8
        if depth is not None:
            snow_present = snow_present & (depth > 0.0)
    else:
        snow_present = frac > 0.0
    return np.where(snow_present, frac, 0.0)


def _active_snow_depth_numpy(state: CoupledColumnState) -> np.ndarray:
    h2osno = (
        _to_np(state.h2osno)
        if state.h2osno is not None
        else np.zeros_like(_to_np(state.frac_sno), dtype=np.float64)
    )
    depth = _to_np(state.snow_depth) if state.snow_depth is not None else snow_depth_from_storage(h2osno)
    frac = np.clip(_to_np(state.frac_sno), 0.0, 1.0)
    layers = state.snow_layers
    if layers is not None and _snow_layers_shape_ok(layers, tuple(h2osno.shape)):
        active = np.asarray(layers.snl, dtype=np.int32) < 0
        active = active & (h2osno > MIN_SNOW_LAYER_MASS) & (depth > 0.0)
    else:
        active = (h2osno > MIN_SNOW_LAYER_MASS) & (frac * np.maximum(depth, 0.0) >= SNOW_MIN_ACTIVE_DEPTH)
    return np.where(active, np.maximum(depth, 0.0), 0.0)


def _total_snow_depth_numpy(state: CoupledColumnState) -> np.ndarray:
    h2osno = (
        _to_np(state.h2osno)
        if state.h2osno is not None
        else np.zeros_like(_to_np(state.frac_sno), dtype=np.float64)
    )
    depth = _to_np(state.snow_depth) if state.snow_depth is not None else snow_depth_from_storage(h2osno)
    snow_present = (h2osno > 1.0e-8) & (depth > 0.0)
    return np.where(snow_present, np.maximum(depth, 0.0), 0.0)


def _active_snow_fraction_torch(
    state: CoupledColumnState,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    frac = torch.clamp(_to_torch(state.frac_sno, device, dtype), min=0.0, max=1.0)
    h2osno = _to_torch(state.h2osno, device, dtype) if state.h2osno is not None else None
    depth = _to_torch(state.snow_depth, device, dtype) if state.snow_depth is not None else (snow_depth_from_storage_torch(h2osno, device=device, dtype=dtype) if h2osno is not None else None)
    layers = state.snow_layers
    if layers is not None and _snow_layers_shape_ok(layers, tuple(frac.shape)):
        active = torch.as_tensor(layers.snl, device=device, dtype=torch.int32) < 0
        if h2osno is not None:
            active = active & (h2osno > MIN_SNOW_LAYER_MASS)
        if depth is not None:
            active = active & (depth > 0.0)
    elif h2osno is not None:
        active = h2osno > MIN_SNOW_LAYER_MASS
        if depth is not None:
            active = active & (frac * torch.clamp(depth, min=0.0) >= SNOW_MIN_ACTIVE_DEPTH)
    else:
        active = frac > 0.0
    return torch.where(active, frac, torch.zeros_like(frac))


def _total_snow_fraction_torch(
    state: CoupledColumnState,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    frac = torch.clamp(_to_torch(state.frac_sno, device, dtype), min=0.0, max=1.0)
    h2osno = _to_torch(state.h2osno, device, dtype) if state.h2osno is not None else None
    depth = _to_torch(state.snow_depth, device, dtype) if state.snow_depth is not None else (snow_depth_from_storage_torch(h2osno, device=device, dtype=dtype) if h2osno is not None else None)
    if h2osno is not None:
        snow_present = h2osno > 1.0e-8
        if depth is not None:
            snow_present = snow_present & (depth > 0.0)
    else:
        snow_present = frac > 0.0
    return torch.where(snow_present, frac, torch.zeros_like(frac))


def _active_snow_depth_torch(
    state: CoupledColumnState,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    h2osno = (
        _to_torch(state.h2osno, device, dtype)
        if state.h2osno is not None
        else torch.zeros_like(_to_torch(state.frac_sno, device, dtype))
    )
    depth = _to_torch(state.snow_depth, device, dtype) if state.snow_depth is not None else snow_depth_from_storage_torch(h2osno, device=device, dtype=dtype)
    frac = torch.clamp(_to_torch(state.frac_sno, device, dtype), min=0.0, max=1.0)
    layers = state.snow_layers
    if layers is not None and _snow_layers_shape_ok(layers, tuple(h2osno.shape)):
        active = torch.as_tensor(layers.snl, device=device, dtype=torch.int32) < 0
        active = active & (h2osno > MIN_SNOW_LAYER_MASS) & (depth > 0.0)
    else:
        active = (h2osno > MIN_SNOW_LAYER_MASS) & (frac * torch.clamp(depth, min=0.0) >= SNOW_MIN_ACTIVE_DEPTH)
    return torch.where(active, torch.clamp(depth, min=0.0), torch.zeros_like(depth))


def _total_snow_depth_torch(
    state: CoupledColumnState,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    h2osno = (
        _to_torch(state.h2osno, device, dtype)
        if state.h2osno is not None
        else torch.zeros_like(_to_torch(state.frac_sno, device, dtype))
    )
    depth = _to_torch(state.snow_depth, device, dtype) if state.snow_depth is not None else snow_depth_from_storage_torch(h2osno, device=device, dtype=dtype)
    snow_present = (h2osno > 1.0e-8) & (depth > 0.0)
    return torch.where(snow_present, torch.clamp(depth, min=0.0), torch.zeros_like(depth))


def _adjust_snow_depth_by_fraction_numpy(snow_depth: np.ndarray, frac_sno: np.ndarray) -> np.ndarray:
    snow_depth = np.asarray(snow_depth, dtype=np.float64)
    frac_sno = np.clip(np.asarray(frac_sno, dtype=np.float64), 1.0e-12, 1.0)
    return np.where(snow_depth > 0.0, snow_depth / frac_sno, 0.0)


def _adjust_snow_depth_by_fraction_torch(snow_depth: torch.Tensor, frac_sno: torch.Tensor) -> torch.Tensor:
    frac = torch.clamp(frac_sno, min=1.0e-12, max=1.0)
    return torch.where(snow_depth > 0.0, snow_depth / frac, torch.zeros_like(snow_depth))


def _snow_batch_like_numpy(value: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape == shape:
        return arr
    if arr.ndim + 1 == len(shape) and arr.shape == shape[:arr.ndim]:
        return arr[..., None]
    if arr.size == int(np.prod(shape)):
        return arr.reshape(shape)
    return np.broadcast_to(arr, shape)


def _snow_batch_like_torch(
    value: torch.Tensor,
    shape: tuple[int, ...],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    arr = torch.as_tensor(value, device=device, dtype=dtype)
    if tuple(arr.shape) == tuple(shape):
        return arr
    if arr.ndim + 1 == len(shape) and tuple(arr.shape) == tuple(shape[:arr.ndim]):
        return arr.unsqueeze(-1)
    if arr.numel() == int(np.prod(shape)):
        return arr.reshape(shape)
    return torch.broadcast_to(arr, shape)


def _compact_snow_layers_numpy(
    snow_layers: SnowLayerState,
    frac_sno_eff: np.ndarray,
    dt: float,
) -> SnowLayerState:
    snl = np.asarray(snow_layers.snl, dtype=np.int32)
    dz = np.array(snow_layers.dz, dtype=np.float64, copy=True)
    ice = np.array(snow_layers.h2osoi_ice, dtype=np.float64, copy=True)
    liq = np.array(snow_layers.h2osoi_liq, dtype=np.float64, copy=True)
    t_snow = np.array(snow_layers.t_snow, dtype=np.float64, copy=True)
    frac_sno_eff = np.clip(_snow_batch_like_numpy(frac_sno_eff, snl.shape), 1.0e-12, 1.0)
    active = np.maximum(-snl, 0)
    burden = np.zeros_like(frac_sno_eff)
    for k in range(MAX_SNOW_LAYERS):
        layer_mask = active > k
        if not np.any(layer_mask):
            continue
        dzk = dz[..., k]
        icek = np.maximum(ice[..., k], 0.0)
        liqk = np.maximum(liq[..., k], 0.0)
        wx = icek + liqk
        thick = np.maximum(dzk * frac_sno_eff, 1.0e-12)
        void = 1.0 - (icek / DENICE + liqk / DENH2O) / thick
        active_mask = layer_mask & (void > 0.001) & (icek > 0.1)
        if np.any(active_mask):
            bi = np.where(active_mask, icek / thick, 0.0)
            td = np.maximum(TFRZ - t_snow[..., k], 0.0)
            dexpf = np.exp(-SNOW_COMPACTION_C4 * td)
            ddz1 = -SNOW_COMPACTION_C3 * dexpf
            ddz1 = np.where(bi > SNOW_COMPACTION_DM, ddz1 * np.exp(-46.0e-3 * (bi - SNOW_COMPACTION_DM)), ddz1)
            ddz1 = np.where(liqk > 0.01 * thick, ddz1 * SNOW_COMPACTION_C5, ddz1)
            ddz2 = -(burden + wx / 2.0) * np.exp(-0.08 * td - SNOW_COMPACTION_C2 * bi) / SNOW_COMPACTION_ETA0
            pdzdtc = ddz1 + ddz2
            new_dz = np.maximum(dzk * (1.0 + pdzdtc * dt), (icek / DENICE + liqk / DENH2O) / frac_sno_eff)
            dz[..., k] = np.where(active_mask, new_dz, dzk)
        burden = burden + wx
    return SnowLayerState(snl=snl, dz=dz, h2osoi_ice=ice, h2osoi_liq=liq, t_snow=t_snow,
                          snw_rds=np.array(snow_layers.snw_rds, dtype=np.float64, copy=True) if snow_layers.snw_rds is not None else None)


def _compact_snow_layers_torch(
    snow_layers: SnowLayerState,
    frac_sno_eff: torch.Tensor,
    dt: float,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> SnowLayerState:
    snl = torch.as_tensor(snow_layers.snl, device=device, dtype=torch.int32)
    dz = torch.as_tensor(snow_layers.dz, device=device, dtype=dtype).clone()
    ice = torch.as_tensor(snow_layers.h2osoi_ice, device=device, dtype=dtype).clone()
    liq = torch.as_tensor(snow_layers.h2osoi_liq, device=device, dtype=dtype).clone()
    t_snow = torch.as_tensor(snow_layers.t_snow, device=device, dtype=dtype).clone()
    frac_sno_eff = torch.clamp(
        _snow_batch_like_torch(frac_sno_eff, tuple(snl.shape), device=device, dtype=dtype),
        min=1.0e-12,
        max=1.0,
    )
    active = torch.clamp(-snl, min=0)
    burden = torch.zeros_like(frac_sno_eff)
    for k in range(MAX_SNOW_LAYERS):
        layer_mask = active > k
        if not torch.any(layer_mask):
            continue
        dzk = dz[..., k]
        icek = torch.clamp(ice[..., k], min=0.0)
        liqk = torch.clamp(liq[..., k], min=0.0)
        wx = icek + liqk
        thick = torch.clamp(dzk * frac_sno_eff, min=1.0e-12)
        void = 1.0 - (icek / DENICE + liqk / DENH2O) / thick
        active_mask = layer_mask & (void > 0.001) & (icek > 0.1)
        if torch.any(active_mask):
            bi = torch.where(active_mask, icek / thick, torch.zeros_like(icek))
            td = torch.clamp(TFRZ - t_snow[..., k], min=0.0)
            dexpf = torch.exp(-SNOW_COMPACTION_C4 * td)
            ddz1 = -SNOW_COMPACTION_C3 * dexpf
            ddz1 = torch.where(bi > SNOW_COMPACTION_DM, ddz1 * torch.exp(-46.0e-3 * (bi - SNOW_COMPACTION_DM)), ddz1)
            ddz1 = torch.where(liqk > 0.01 * thick, ddz1 * SNOW_COMPACTION_C5, ddz1)
            ddz2 = -(burden + wx / 2.0) * torch.exp(-0.08 * td - SNOW_COMPACTION_C2 * bi) / SNOW_COMPACTION_ETA0
            pdzdtc = ddz1 + ddz2
            new_dz = torch.maximum(dzk * (1.0 + pdzdtc * dt), (icek / DENICE + liqk / DENH2O) / frac_sno_eff)
            dz[..., k] = torch.where(active_mask, new_dz, dzk)
        burden = burden + wx
    rds_t = torch.as_tensor(snow_layers.snw_rds, device=device, dtype=dtype).clone() if snow_layers.snw_rds is not None else None
    return SnowLayerState(snl=snl, dz=dz, h2osoi_ice=ice, h2osoi_liq=liq, t_snow=t_snow, snw_rds=rds_t)


def _snow_combo_one_numpy(
    dz_a: float,
    ice_a: float,
    liq_a: float,
    t_a: float,
    dz_b: float,
    ice_b: float,
    liq_b: float,
    t_b: float,
) -> tuple[float, float, float, float]:
    dzc = dz_a + dz_b
    wicec = ice_a + ice_b
    wliqc = liq_a + liq_b
    h1 = (CPICE * ice_a + CPLIQ * liq_a) * (t_a - TFRZ) + HFUS * liq_a
    h2 = (CPICE * ice_b + CPLIQ * liq_b) * (t_b - TFRZ) + HFUS * liq_b
    hc = h1 + h2
    denom = CPICE * wicec + CPLIQ * wliqc
    tc = TFRZ if denom <= 0.0 else TFRZ + (hc - HFUS * wliqc) / denom
    return dzc, wicec, wliqc, tc


def _snow_combo_one_torch(
    dz_a: torch.Tensor,
    ice_a: torch.Tensor,
    liq_a: torch.Tensor,
    t_a: torch.Tensor,
    dz_b: torch.Tensor,
    ice_b: torch.Tensor,
    liq_b: torch.Tensor,
    t_b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    dzc = dz_a + dz_b
    wicec = ice_a + ice_b
    wliqc = liq_a + liq_b
    h1 = (torch.as_tensor(CPICE, device=dz_a.device, dtype=dz_a.dtype) * ice_a + torch.as_tensor(CPLIQ, device=dz_a.device, dtype=dz_a.dtype) * liq_a) * (t_a - TFRZ) + torch.as_tensor(HFUS, device=dz_a.device, dtype=dz_a.dtype) * liq_a
    h2 = (torch.as_tensor(CPICE, device=dz_a.device, dtype=dz_a.dtype) * ice_b + torch.as_tensor(CPLIQ, device=dz_a.device, dtype=dz_a.dtype) * liq_b) * (t_b - TFRZ) + torch.as_tensor(HFUS, device=dz_a.device, dtype=dz_a.dtype) * liq_b
    hc = h1 + h2
    denom = torch.as_tensor(CPICE, device=dz_a.device, dtype=dz_a.dtype) * wicec + torch.as_tensor(CPLIQ, device=dz_a.device, dtype=dz_a.dtype) * wliqc
    tc = torch.where(denom > 0.0, TFRZ + (hc - torch.as_tensor(HFUS, device=dz_a.device, dtype=dz_a.dtype) * wliqc) / denom, torch.full_like(dzc, TFRZ))
    return dzc, wicec, wliqc, tc


def _rebalance_snow_layers_numpy(snow_layers: SnowLayerState, frac_sno_eff: np.ndarray) -> SnowLayerState:
    snl = np.asarray(snow_layers.snl, dtype=np.int32).reshape(-1)
    dz = np.array(snow_layers.dz, dtype=np.float64, copy=True).reshape(-1, MAX_SNOW_LAYERS)
    ice = np.array(snow_layers.h2osoi_ice, dtype=np.float64, copy=True).reshape(-1, MAX_SNOW_LAYERS)
    liq = np.array(snow_layers.h2osoi_liq, dtype=np.float64, copy=True).reshape(-1, MAX_SNOW_LAYERS)
    t_snow = np.array(snow_layers.t_snow, dtype=np.float64, copy=True).reshape(-1, MAX_SNOW_LAYERS)
    frac = np.clip(np.asarray(frac_sno_eff, dtype=np.float64).reshape(-1), 1.0e-12, 1.0)
    # Grain radius — optional; if absent use fresh-snow placeholder.
    has_rds = snow_layers.snw_rds is not None
    if has_rds:
        rds = np.array(snow_layers.snw_rds, dtype=np.float64, copy=True).reshape(-1, MAX_SNOW_LAYERS)
    else:
        rds = np.full((snl.size, MAX_SNOW_LAYERS), SNW_RDS_MIN, dtype=np.float64)
    ncol = snl.size

    def _rds_combo(ice_a, liq_a, rds_a, ice_b, liq_b, rds_b):
        """Mass-weighted grain radius for two layers being combined."""
        mass_a = max(ice_a + liq_a, 1.0e-12)
        mass_b = max(ice_b + liq_b, 1.0e-12)
        return np.clip((rds_a * mass_a + rds_b * mass_b) / (mass_a + mass_b), SNW_RDS_MIN, SNW_RDS_MAX)

    for c in range(ncol):
        n = int(max(-snl[c], 0))
        if n <= 0:
            scalar_mass = max(ice[c, 0], 0.0) + max(liq[c, 0], 0.0)
            if scalar_mass > 1.0e-8:
                dz[c, 1:] = 0.0
                ice[c, 1:] = 0.0
                liq[c, 1:] = 0.0
                t_snow[c, 1:] = TFRZ
                if has_rds:
                    rds[c, 1:] = 0.0
            else:
                dz[c, :] = 0.0
                ice[c, :] = 0.0
                liq[c, :] = 0.0
                t_snow[c, :] = TFRZ
                if has_rds:
                    rds[c, :] = 0.0
            continue
        if np.sum(np.maximum(ice[c, :n], 0.0) + np.maximum(liq[c, :n], 0.0)) <= 1.0e-8:
            snl[c] = 0
            dz[c, :] = 0.0
            ice[c, :] = 0.0
            liq[c, :] = 0.0
            t_snow[c, :] = TFRZ
            rds[c, :] = 0.0
            continue
        # Divide the top layer if it grows too thick.
        top_depth = frac[c] * dz[c, 0]
        split_limit = SNOW_DIVIDE_TOP_SINGLE if n == 1 else SNOW_DIVIDE_TOP_MULTI
        if top_depth > split_limit:
            target_dz = split_limit / frac[c]
            if n == 1:
                if target_dz < dz[c, 0] and n < MAX_SNOW_LAYERS:
                    orig_dz = dz[c, 0]
                    orig_ice = ice[c, 0]
                    orig_liq = liq[c, 0]
                    orig_t = t_snow[c, 0]
                    orig_rds = rds[c, 0]
                    frac_top = 0.5
                    top = [target_dz, orig_ice * frac_top, orig_liq * frac_top, orig_t]
                    bot = [orig_dz - target_dz, orig_ice * (1.0 - frac_top), orig_liq * (1.0 - frac_top), orig_t]
                    dz[c, 0], ice[c, 0], liq[c, 0], t_snow[c, 0] = top
                    dz[c, 1], ice[c, 1], liq[c, 1], t_snow[c, 1] = bot
                    rds[c, 0] = orig_rds
                    rds[c, 1] = orig_rds
                    n = 2
            else:
                if target_dz < dz[c, 0]:
                    if n < MAX_SNOW_LAYERS:
                        orig_dz = dz[c, 0]
                        orig_ice = ice[c, 0]
                        orig_liq = liq[c, 0]
                        orig_t = t_snow[c, 0]
                        orig_rds = rds[c, 0]
                        remainder = orig_dz - target_dz
                        frac_top = target_dz / orig_dz
                        # shift existing layers down
                        dz[c, 2 : n + 1] = dz[c, 1:n]
                        ice[c, 2 : n + 1] = ice[c, 1:n]
                        liq[c, 2 : n + 1] = liq[c, 1:n]
                        t_snow[c, 2 : n + 1] = t_snow[c, 1:n]
                        rds[c, 2 : n + 1] = rds[c, 1:n]
                        dz[c, 0] = target_dz
                        ice[c, 0] = orig_ice * frac_top
                        liq[c, 0] = orig_liq * frac_top
                        t_snow[c, 0] = orig_t
                        rds[c, 0] = orig_rds
                        dz[c, 1] = remainder
                        ice[c, 1] = orig_ice * (1.0 - frac_top)
                        liq[c, 1] = orig_liq * (1.0 - frac_top)
                        t_snow[c, 1] = orig_t
                        rds[c, 1] = orig_rds
                        n = min(n + 1, MAX_SNOW_LAYERS)
                    else:
                        # No room to insert a layer: split the top layer and fold the remainder into layer 1.
                        orig_dz = dz[c, 0]
                        orig_ice = ice[c, 0]
                        orig_liq = liq[c, 0]
                        orig_t = t_snow[c, 0]
                        orig_rds = rds[c, 0]
                        frac_top = target_dz / orig_dz
                        top_ice = orig_ice * frac_top
                        top_liq = orig_liq * frac_top
                        rem = (orig_dz - target_dz, orig_ice * (1.0 - frac_top), orig_liq * (1.0 - frac_top), orig_t)
                        dz[c, 0] = target_dz
                        ice[c, 0] = top_ice
                        liq[c, 0] = top_liq
                        rds[c, 0] = orig_rds
                        dz[c, 1], ice[c, 1], liq[c, 1], t_snow[c, 1] = _snow_combo_one_numpy(
                            dz[c, 1], ice[c, 1], liq[c, 1], t_snow[c, 1], *rem
                        )
                        rds[c, 1] = _rds_combo(ice[c, 1], liq[c, 1], rds[c, 1],
                                                rem[1], rem[2], orig_rds)
        # Combine any thin / low-density layers.
        changed = True
        while changed and n > 1:
            changed = False
            for k in range(n):
                dzsno = frac[c] * dz[c, k]
                mass = ice[c, k] + liq[c, k]
                dens = mass / max(dzsno, 1.0e-12)
                thresh = SNOW_COMBINE_DZMIN[min(k, SNOW_COMBINE_DZMIN.size - 1)]
                if dzsno < thresh or dens < SNOW_COMBINE_MIN_DENSITY:
                    if k == 0:
                        victim = 0
                        target = 1
                    elif k == n - 1:
                        victim = k
                        target = k - 1
                    else:
                        if (dz[c, k - 1] + dz[c, k]) <= (dz[c, k + 1] + dz[c, k]):
                            victim = k
                            target = k - 1
                        else:
                            victim = k
                            target = k + 1
                    old_dz = dz[c, :n].copy()
                    old_ice = ice[c, :n].copy()
                    old_liq = liq[c, :n].copy()
                    old_t = t_snow[c, :n].copy()
                    old_rds = rds[c, :n].copy()
                    combo = _snow_combo_one_numpy(
                        old_dz[target],
                        old_ice[target],
                        old_liq[target],
                        old_t[target],
                        old_dz[victim],
                        old_ice[victim],
                        old_liq[victim],
                        old_t[victim],
                    )
                    combo_rds = _rds_combo(
                        old_ice[target], old_liq[target], old_rds[target],
                        old_ice[victim], old_liq[victim], old_rds[victim],
                    )
                    insert = min(victim, target)
                    new_layers = []
                    for kk in range(n):
                        if kk == insert:
                            new_layers.append((*combo, combo_rds))
                        if kk == victim or kk == target:
                            continue
                        new_layers.append((old_dz[kk], old_ice[kk], old_liq[kk], old_t[kk], old_rds[kk]))
                    for kk, (dz_k, ice_k, liq_k, t_k, rds_k) in enumerate(new_layers):
                        dz[c, kk] = dz_k
                        ice[c, kk] = ice_k
                        liq[c, kk] = liq_k
                        t_snow[c, kk] = t_k
                        rds[c, kk] = rds_k
                    n -= 1
                    changed = True
                    break
        total_depth = np.sum(np.maximum(dz[c, :n], 0.0))
        total_mass = np.sum(np.maximum(ice[c, :n], 0.0) + np.maximum(liq[c, :n], 0.0))
        eff_depth = frac[c] * total_depth
        bulk_density = total_mass / max(eff_depth, 1.0e-12)
        if n > 0 and (eff_depth < SNOW_MIN_ACTIVE_DEPTH or bulk_density < SNOW_COMBINE_MIN_DENSITY):
            # Native CombineSnowLayers collapses sub-1 cm or very low-density
            # packs to no active snow layer.  Keeping them active lets a trace
            # snow event falsely freeze large standing-surface-water stores.
            inactive_ice = np.sum(np.maximum(ice[c, :n], 0.0))
            n = 0
            dz[c, :] = 0.0
            ice[c, :] = 0.0
            liq[c, :] = 0.0
            dz[c, 0] = total_depth
            ice[c, 0] = inactive_ice
            t_snow[c, :] = TFRZ
            rds[c, :] = 0.0
        snl[c] = -n if n > 0 else 0
        if n < MAX_SNOW_LAYERS:
            start = 1 if n == 0 and ice[c, 0] > 0.0 else n
            dz[c, start:] = 0.0
            ice[c, start:] = 0.0
            liq[c, start:] = 0.0
            t_snow[c, start:] = TFRZ
            rds[c, start:] = 0.0
    return SnowLayerState(
        snl=snl.reshape(snow_layers.snl.shape),
        dz=dz.reshape(snow_layers.dz.shape),
        h2osoi_ice=ice.reshape(snow_layers.h2osoi_ice.shape),
        h2osoi_liq=liq.reshape(snow_layers.h2osoi_liq.shape),
        t_snow=t_snow.reshape(snow_layers.t_snow.shape),
        snw_rds=rds.reshape(snow_layers.snw_rds.shape) if has_rds else None,
    )


def _rebalance_snow_layers_torch_slow(snow_layers: SnowLayerState, frac_sno_eff: torch.Tensor) -> SnowLayerState:
    snl = torch.as_tensor(snow_layers.snl).reshape(-1).to(dtype=torch.int32)
    dz = torch.as_tensor(snow_layers.dz).clone().reshape(-1, MAX_SNOW_LAYERS)
    ice = torch.as_tensor(snow_layers.h2osoi_ice).clone().reshape(-1, MAX_SNOW_LAYERS)
    liq = torch.as_tensor(snow_layers.h2osoi_liq).clone().reshape(-1, MAX_SNOW_LAYERS)
    t_snow = torch.as_tensor(snow_layers.t_snow).clone().reshape(-1, MAX_SNOW_LAYERS)
    frac = torch.clamp(frac_sno_eff.reshape(-1), min=1.0e-12, max=1.0)
    ncol = snl.numel()
    cpice = torch.as_tensor(CPICE, device=dz.device, dtype=dz.dtype)
    cpliq = torch.as_tensor(CPLIQ, device=dz.device, dtype=dz.dtype)
    hfus = torch.as_tensor(HFUS, device=dz.device, dtype=dz.dtype)
    for c in range(ncol):
        n = int(max(int(-snl[c].item()), 0))
        if n <= 0:
            scalar_mass = torch.clamp(ice[c, 0], min=0.0) + torch.clamp(liq[c, 0], min=0.0)
            if bool((scalar_mass > 1.0e-8).item()):
                dz[c, 1:] = 0.0
                ice[c, 1:] = 0.0
                liq[c, 1:] = 0.0
                t_snow[c, 1:] = TFRZ
            else:
                dz[c, :] = 0.0
                ice[c, :] = 0.0
                liq[c, :] = 0.0
                t_snow[c, :] = TFRZ
            continue
        if n > 0 and float(torch.sum(torch.clamp(ice[c, :n], min=0.0) + torch.clamp(liq[c, :n], min=0.0)).item()) <= 1.0e-8:
            snl[c] = 0
            dz[c, :] = 0.0
            ice[c, :] = 0.0
            liq[c, :] = 0.0
            t_snow[c, :] = TFRZ
            continue
        top_depth = float((frac[c] * dz[c, 0]).item())
        split_limit = SNOW_DIVIDE_TOP_SINGLE if n == 1 else SNOW_DIVIDE_TOP_MULTI
        if top_depth > split_limit:
            target_dz = split_limit / max(float(frac[c].item()), 1.0e-12)
            if n == 1 and target_dz < float(dz[c, 0].item()) and n < MAX_SNOW_LAYERS:
                orig_dz = dz[c, 0].clone()
                orig_ice = ice[c, 0].clone()
                orig_liq = liq[c, 0].clone()
                orig_t = t_snow[c, 0].clone()
                frac_top = 0.5
                dz[c, 0] = target_dz
                ice[c, 0] = orig_ice * frac_top
                liq[c, 0] = orig_liq * frac_top
                t_snow[c, 0] = orig_t
                dz[c, 1] = orig_dz - target_dz
                ice[c, 1] = orig_ice * (1.0 - frac_top)
                liq[c, 1] = orig_liq * (1.0 - frac_top)
                t_snow[c, 1] = orig_t
                n = 2
            elif n > 1 and target_dz < float(dz[c, 0].item()):
                if n < MAX_SNOW_LAYERS:
                    orig_dz = dz[c, 0].clone()
                    orig_ice = ice[c, 0].clone()
                    orig_liq = liq[c, 0].clone()
                    orig_t = t_snow[c, 0].clone()
                    remainder = orig_dz - target_dz
                    frac_top = target_dz / max(float(orig_dz.item()), 1.0e-12)
                    dz[c, 2 : n + 1] = dz[c, 1:n].clone()
                    ice[c, 2 : n + 1] = ice[c, 1:n].clone()
                    liq[c, 2 : n + 1] = liq[c, 1:n].clone()
                    t_snow[c, 2 : n + 1] = t_snow[c, 1:n].clone()
                    dz[c, 0] = target_dz
                    ice[c, 0] = orig_ice * frac_top
                    liq[c, 0] = orig_liq * frac_top
                    t_snow[c, 0] = orig_t
                    dz[c, 1] = remainder
                    ice[c, 1] = orig_ice * (1.0 - frac_top)
                    liq[c, 1] = orig_liq * (1.0 - frac_top)
                    t_snow[c, 1] = orig_t
                    n = min(n + 1, MAX_SNOW_LAYERS)
                else:
                    orig_dz = dz[c, 0].clone()
                    orig_ice = ice[c, 0].clone()
                    orig_liq = liq[c, 0].clone()
                    orig_t = t_snow[c, 0].clone()
                    frac_top = target_dz / max(float(orig_dz.item()), 1.0e-12)
                    rem_dz = orig_dz - target_dz
                    rem_ice = orig_ice * (1.0 - frac_top)
                    rem_liq = orig_liq * (1.0 - frac_top)
                    rem_t = orig_t
                    dz[c, 0] = target_dz
                    ice[c, 0] = orig_ice * frac_top
                    liq[c, 0] = orig_liq * frac_top
                    t_snow[c, 0] = orig_t
                    dz[c, 1], ice[c, 1], liq[c, 1], t_snow[c, 1] = _snow_combo_one_torch(
                        dz[c, 1], ice[c, 1], liq[c, 1], t_snow[c, 1], torch.as_tensor(rem_dz, device=dz.device, dtype=dz.dtype), torch.as_tensor(rem_ice, device=dz.device, dtype=dz.dtype), torch.as_tensor(rem_liq, device=dz.device, dtype=dz.dtype), torch.as_tensor(rem_t, device=dz.device, dtype=dz.dtype)
                    )
        changed = True
        while changed and n > 1:
            changed = False
            for k in range(n):
                dzsno = float((frac[c] * dz[c, k]).item())
                mass = float((ice[c, k] + liq[c, k]).item())
                dens = mass / max(dzsno, 1.0e-12)
                thresh = float(SNOW_COMBINE_DZMIN[min(k, SNOW_COMBINE_DZMIN.size - 1)])
                if dzsno < thresh or dens < SNOW_COMBINE_MIN_DENSITY:
                    if k == 0:
                        victim = 0
                        target = 1
                    elif k == n - 1:
                        victim = k
                        target = k - 1
                    else:
                        if float((dz[c, k - 1] + dz[c, k]).item()) <= float((dz[c, k + 1] + dz[c, k]).item()):
                            victim = k
                            target = k - 1
                        else:
                            victim = k
                            target = k + 1
                    old_dz = dz[c, :n].clone()
                    old_ice = ice[c, :n].clone()
                    old_liq = liq[c, :n].clone()
                    old_t = t_snow[c, :n].clone()
                    combo = _snow_combo_one_torch(
                        old_dz[target], old_ice[target], old_liq[target], old_t[target],
                        old_dz[victim], old_ice[victim], old_liq[victim], old_t[victim],
                    )
                    insert = min(victim, target)
                    new_layers = []
                    for kk in range(n):
                        if kk == insert:
                            new_layers.append(combo)
                        if kk == victim or kk == target:
                            continue
                        new_layers.append((old_dz[kk], old_ice[kk], old_liq[kk], old_t[kk]))
                    for kk, (dz_k, ice_k, liq_k, t_k) in enumerate(new_layers):
                        dz[c, kk] = dz_k
                        ice[c, kk] = ice_k
                        liq[c, kk] = liq_k
                        t_snow[c, kk] = t_k
                    n -= 1
                    changed = True
                    break
        total_depth = float(torch.sum(torch.clamp(dz[c, :n], min=0.0)).item())
        total_mass = float(torch.sum(torch.clamp(ice[c, :n], min=0.0) + torch.clamp(liq[c, :n], min=0.0)).item())
        eff_depth = float(frac[c].item()) * total_depth
        bulk_density = total_mass / max(eff_depth, 1.0e-12)
        if n > 0 and (eff_depth < SNOW_MIN_ACTIVE_DEPTH or bulk_density < SNOW_COMBINE_MIN_DENSITY):
            inactive_ice = torch.sum(torch.clamp(ice[c, :n], min=0.0))
            n = 0
            dz[c, :] = 0.0
            ice[c, :] = 0.0
            liq[c, :] = 0.0
            dz[c, 0] = total_depth
            ice[c, 0] = inactive_ice
            t_snow[c, :] = TFRZ
        snl[c] = -n if n > 0 else 0
        if n < MAX_SNOW_LAYERS:
            start = 1 if n == 0 and float(ice[c, 0].item()) > 0.0 else n
            dz[c, start:] = 0.0
            ice[c, start:] = 0.0
            liq[c, start:] = 0.0
            t_snow[c, start:] = TFRZ
    return SnowLayerState(
        snl=snl.reshape(snow_layers.snl.shape),
        dz=dz.reshape(snow_layers.dz.shape),
        h2osoi_ice=ice.reshape(snow_layers.h2osoi_ice.shape),
        h2osoi_liq=liq.reshape(snow_layers.h2osoi_liq.shape),
        t_snow=t_snow.reshape(snow_layers.t_snow.shape),
    )


def _rebalance_snow_layers_torch(snow_layers: SnowLayerState, frac_sno_eff: torch.Tensor) -> SnowLayerState:
    """Vectorized torch snow-layer cleanup with subset fallback for structural edits."""
    device = frac_sno_eff.device
    dtype = frac_sno_eff.dtype if frac_sno_eff.is_floating_point() else torch.float32
    if device.type == "mps" and dtype == torch.float64:
        dtype = torch.float32
    snl = torch.as_tensor(snow_layers.snl, device=device, dtype=torch.int32)
    dz = torch.as_tensor(snow_layers.dz, device=device, dtype=dtype).clone()
    ice = torch.as_tensor(snow_layers.h2osoi_ice, device=device, dtype=dtype).clone()
    liq = torch.as_tensor(snow_layers.h2osoi_liq, device=device, dtype=dtype).clone()
    t_snow = torch.as_tensor(snow_layers.t_snow, device=device, dtype=dtype).clone()
    rds_in = getattr(snow_layers, "snw_rds", None)
    rds = torch.as_tensor(rds_in, device=device, dtype=dtype).clone() if rds_in is not None else None
    batch_shape = tuple(snl.shape)
    frac = torch.clamp(
        _snow_batch_like_torch(frac_sno_eff, batch_shape, device=device, dtype=dtype),
        min=1.0e-12,
        max=1.0,
    )

    nsnow = torch.clamp(-snl, min=0)
    layer_axis_shape = (1,) * len(batch_shape) + (MAX_SNOW_LAYERS,)
    layer_index = torch.arange(MAX_SNOW_LAYERS, device=device).reshape(layer_axis_shape)
    active_layers = layer_index < nsnow[..., None]
    layer_mass = torch.clamp(ice, min=0.0) + torch.clamp(liq, min=0.0)
    active_mass = torch.sum(torch.where(active_layers, layer_mass, torch.zeros_like(layer_mass)), dim=-1)
    active_depth = torch.sum(torch.where(active_layers, torch.clamp(dz, min=0.0), torch.zeros_like(dz)), dim=-1)

    active = nsnow > 0
    empty_active = active & (active_mass <= 1.0e-8)
    top_depth = frac * dz[..., 0]
    split_limit = torch.where(
        nsnow == 1,
        torch.full_like(frac, SNOW_DIVIDE_TOP_SINGLE),
        torch.full_like(frac, SNOW_DIVIDE_TOP_MULTI),
    )
    target_dz = split_limit / frac
    split_needed = active & (top_depth > split_limit) & (target_dz < dz[..., 0])

    thresh = torch.as_tensor(SNOW_COMBINE_DZMIN, device=device, dtype=dtype).reshape(layer_axis_shape)
    dzsno = frac[..., None] * dz
    dens = layer_mass / torch.clamp(dzsno, min=1.0e-12)
    combine_needed = (nsnow > 1) & torch.any(
        active_layers & ((dzsno < thresh) | (dens < SNOW_COMBINE_MIN_DENSITY)),
        dim=-1,
    )

    eff_depth = frac * active_depth
    bulk_density = active_mass / torch.clamp(eff_depth, min=1.0e-12)
    collapse_needed = active & ((eff_depth < SNOW_MIN_ACTIVE_DEPTH) | (bulk_density < SNOW_COMBINE_MIN_DENSITY))
    slow_mask = empty_active | split_needed | combine_needed | collapse_needed

    # Fast cleanup for columns whose layer topology is already valid.
    scalar_mass = torch.clamp(ice[..., 0], min=0.0) + torch.clamp(liq[..., 0], min=0.0)
    inactive_scalar = (nsnow <= 0) & (scalar_mass > 1.0e-8)
    inactive_empty = (nsnow <= 0) & ~inactive_scalar
    tail_index = (nsnow > 0)[..., None] & (layer_index >= torch.clamp(nsnow, min=0)[..., None])
    inactive_tail = inactive_scalar[..., None] & (layer_index >= 1)
    inactive_all = inactive_empty[..., None]
    cleanup_mask = (~slow_mask)[..., None] & (tail_index | inactive_tail | inactive_all)
    dz = torch.where(cleanup_mask, torch.zeros_like(dz), dz)
    ice = torch.where(cleanup_mask, torch.zeros_like(ice), ice)
    liq = torch.where(cleanup_mask, torch.zeros_like(liq), liq)
    t_snow = torch.where(cleanup_mask, torch.full_like(t_snow, TFRZ), t_snow)
    if rds is not None:
        rds = torch.where(cleanup_mask, torch.zeros_like(rds), rds)

    snl = torch.where((~slow_mask) & (nsnow > 0), -nsnow, snl)
    snl = torch.where((~slow_mask) & (nsnow <= 0), torch.zeros_like(snl), snl)
    fast_layers = SnowLayerState(
        snl=snl,
        dz=dz,
        h2osoi_ice=ice,
        h2osoi_liq=liq,
        t_snow=t_snow,
        snw_rds=rds,
    )

    if not torch.any(slow_mask):
        return fast_layers

    slow_idx = _cell_indices_from_mask(slow_mask)
    if slow_idx.numel() == 0:
        return fast_layers
    ncell = int(batch_shape[0]) if batch_shape else 1
    sub_layers = _cell_index_snow_layers(snow_layers, slow_idx, batch_shape=batch_shape)
    if sub_layers is None:
        return _rebalance_snow_layers_torch_slow(snow_layers, frac_sno_eff)
    sub_frac = _cell_index_select(frac, slow_idx, ncell=ncell, allow_1d=True)
    sub_rebalanced_np = _rebalance_snow_layers_numpy(
        _snow_layers_to_cpu(sub_layers),
        np.asarray(sub_frac.detach().cpu().numpy(), dtype=np.float64),
    )
    sub_rebalanced = _snow_layers_to_device(sub_rebalanced_np, device, dtype)
    return _cell_scatter_snow_layers(fast_layers, slow_idx, sub_rebalanced)


def snow_depth_from_storage(h2osno: np.ndarray) -> np.ndarray:
    return np.maximum(np.asarray(h2osno, dtype=np.float64), 0.0) / SNOW_WATER_TO_DEPTH


def snow_depth_from_storage_torch(
    h2osno: np.ndarray | torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    h2osno_t = _to_torch(h2osno, device, dtype)
    return torch.clamp(h2osno_t, min=0.0) / SNOW_WATER_TO_DEPTH


def _new_snow_bulk_density_numpy(forc_t: np.ndarray, wind_speed: np.ndarray) -> np.ndarray:
    """ELM-style bulk density for newly fallen snow."""
    forc_t = np.asarray(forc_t, dtype=np.float64)
    wind_speed = np.asarray(wind_speed, dtype=np.float64)
    dens = np.empty_like(forc_t, dtype=np.float64)
    warm = forc_t > TFRZ + 2.0
    mid = (forc_t > TFRZ - 15.0) & ~warm
    cold = ~warm & ~mid
    dens[warm] = 50.0 + 1.7 * (17.0 ** 1.5)
    dens[mid] = 50.0 + 1.7 * np.maximum(forc_t[mid] - TFRZ + 15.0, 0.0) ** 1.5
    t_cold = np.maximum(forc_t[cold] - TFRZ, -57.55)
    dens[cold] = -((50.0 / 15.0) + 0.0333 * 15.0) * t_cold - 0.0333 * t_cold**2
    wind = np.asarray(wind_speed, dtype=np.float64)
    wind_mask = wind > 0.1
    dens[wind_mask] = dens[wind_mask] + 266.861 * ((1.0 + np.tanh(wind[wind_mask] / 5.0)) / 2.0) ** 8.8
    return np.clip(dens, 50.0, 400.0)


def _new_snow_bulk_density_torch(forc_t: torch.Tensor, wind_speed: torch.Tensor) -> torch.Tensor:
    """Torch port of the ELM fresh-snow bulk-density rule."""
    forc_t = torch.as_tensor(forc_t)
    wind_speed = torch.as_tensor(wind_speed)
    dens = torch.empty_like(forc_t)
    warm = forc_t > (TFRZ + 2.0)
    mid = (forc_t > (TFRZ - 15.0)) & ~warm
    cold = ~warm & ~mid
    dens[warm] = 50.0 + 1.7 * (17.0 ** 1.5)
    dens[mid] = 50.0 + 1.7 * torch.clamp(forc_t[mid] - TFRZ + 15.0, min=0.0) ** 1.5
    t_cold = torch.clamp(forc_t[cold] - TFRZ, min=-57.55)
    dens[cold] = -((50.0 / 15.0) + 0.0333 * 15.0) * t_cold - 0.0333 * t_cold**2
    wind_mask = wind_speed > 0.1
    dens[wind_mask] = dens[wind_mask] + 266.861 * ((1.0 + torch.tanh(wind_speed[wind_mask] / 5.0)) / 2.0) ** 8.8
    return torch.clamp(dens, 50.0, 400.0)


def _snow_frac_after_new_snow_numpy(
    frac_sno_prev: np.ndarray,
    snow_old: np.ndarray,
    snow_new: np.ndarray,
    newsnow: np.ndarray,
) -> np.ndarray:
    """ELM accumulation update for subgrid snow-covered fraction."""
    frac_prev = np.clip(np.asarray(frac_sno_prev, dtype=np.float64), 0.0, 1.0)
    snow_old = np.asarray(snow_old, dtype=np.float64)
    snow_new = np.asarray(snow_new, dtype=np.float64)
    newsnow = np.maximum(np.asarray(newsnow, dtype=np.float64), 0.0)
    accum_factor = 0.1
    new_event = newsnow > 0.0
    existing = snow_old > 0.0
    frac_new = np.where(snow_new > 0.0, frac_prev, 0.0)
    frac_new = np.where(
        new_event & existing,
        1.0 - (1.0 - np.tanh(accum_factor * newsnow)) * (1.0 - frac_prev),
        frac_new,
    )
    frac_new = np.where(new_event & ~existing, np.tanh(accum_factor * newsnow), frac_new)
    return np.clip(np.where(snow_new > 1.0e-12, frac_new, 0.0), 0.0, 1.0)


def _snow_frac_after_new_snow_torch(
    frac_sno_prev: torch.Tensor,
    snow_old: torch.Tensor,
    snow_new: torch.Tensor,
    newsnow: torch.Tensor,
) -> torch.Tensor:
    """Torch analogue of ELM's accumulation update for snow-covered fraction."""
    frac_prev = torch.clamp(frac_sno_prev, min=0.0, max=1.0)
    newsnow = torch.clamp(newsnow, min=0.0)
    accum_factor = 0.1
    new_event = newsnow > 0.0
    existing = snow_old > 0.0
    frac_new = torch.where(snow_new > 0.0, frac_prev, torch.zeros_like(frac_prev))
    frac_new = torch.where(
        new_event & existing,
        1.0 - (1.0 - torch.tanh(accum_factor * newsnow)) * (1.0 - frac_prev),
        frac_new,
    )
    frac_new = torch.where(
        new_event & ~existing,
        torch.tanh(accum_factor * newsnow),
        frac_new,
    )
    return torch.clamp(torch.where(snow_new > 1.0e-12, frac_new, torch.zeros_like(frac_new)), min=0.0, max=1.0)


def _snow_n_melt_numpy(
    static: CoupledStaticParameters | None,
    shape: tuple[int, ...],
) -> np.ndarray:
    """Broadcast ELM's snow-covered-area melt shape parameter to *shape*."""
    n_melt = 10.0 if static is None else getattr(static, "n_melt", 10.0)
    arr = np.asarray(_to_numpy(n_melt), dtype=np.float64)
    if arr.ndim == 0:
        return np.full(shape, max(float(arr), 1.0e-6), dtype=np.float64)
    return np.maximum(_snow_batch_like_numpy(arr, shape), 1.0e-6)


def _snow_n_melt_torch(
    static: CoupledStaticParameters | None,
    shape: tuple[int, ...],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Broadcast ELM's snow-covered-area melt shape parameter to *shape*."""
    n_melt = 10.0 if static is None else getattr(static, "n_melt", 10.0)
    arr = _to_torch(n_melt, device, dtype)
    if arr.ndim == 0:
        return torch.full(shape, max(float(arr.detach().cpu().item()), 1.0e-6), device=device, dtype=dtype)
    return torch.clamp(_snow_batch_like_torch(arr, shape, device=device, dtype=dtype), min=1.0e-6)


def _snow_area_factor_from_frac_numpy(frac_sno: np.ndarray, n_melt: np.ndarray) -> np.ndarray:
    """Area-weighted SWE fraction implied by ELM's depletion curve."""
    frac = np.clip(np.asarray(frac_sno, dtype=np.float64), 1.0e-6, 1.0)
    exponent = 1.0 / np.maximum(n_melt, 1.0e-6)
    return 0.5 * (np.cos(np.pi * (1.0 - frac) ** exponent) + 1.0)


def _snow_area_factor_from_frac_torch(frac_sno: torch.Tensor, n_melt: torch.Tensor) -> torch.Tensor:
    """Area-weighted SWE fraction implied by ELM's depletion curve."""
    frac = torch.clamp(frac_sno, min=1.0e-6, max=1.0)
    exponent = 1.0 / torch.clamp(n_melt, min=1.0e-6)
    return 0.5 * (torch.cos(math.pi * (1.0 - frac) ** exponent) + 1.0)


def _snow_frac_from_swe_ratio_numpy(swe_ratio: np.ndarray, n_melt: np.ndarray) -> np.ndarray:
    """ELM snow-covered fraction during melt as a function of SWE / int_snow."""
    ratio = np.clip(np.asarray(swe_ratio, dtype=np.float64), 0.0, 1.0)
    arg = np.clip(2.0 * ratio - 1.0, -1.0, 1.0)
    frac = 1.0 - (np.arccos(arg) / np.pi) ** np.maximum(n_melt, 1.0e-6)
    return np.clip(frac, 0.0, 1.0)


def _snow_frac_from_swe_ratio_torch(swe_ratio: torch.Tensor, n_melt: torch.Tensor) -> torch.Tensor:
    """Torch analogue of ELM snow-covered fraction during melt."""
    ratio = torch.clamp(swe_ratio, min=0.0, max=1.0)
    arg = torch.clamp(2.0 * ratio - 1.0, min=-1.0, max=1.0)
    frac = 1.0 - (torch.arccos(arg) / math.pi) ** torch.clamp(n_melt, min=1.0e-6)
    return torch.clamp(frac, min=0.0, max=1.0)


def _apply_post_melt_snow_fraction_update(
    state: CoupledColumnState,
    snow_melt: np.ndarray | torch.Tensor | None,
    dt: float,
    static: CoupledStaticParameters | None,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> CoupledColumnState:
    """Apply ELM's melt-depletion FSNO update after the thermal melt step."""
    if snow_melt is None or state.h2osno is None:
        return state
    if isinstance(snow_melt, torch.Tensor) or isinstance(state.h2osno, torch.Tensor):
        h2osno = _to_torch(state.h2osno, device, dtype)
        shape = tuple(h2osno.shape)
        melt_mass = torch.clamp(
            _snow_batch_like_torch(_to_torch(snow_melt, device, dtype), shape, device=device, dtype=dtype),
            min=0.0,
        ) * float(dt)
        has_melt = (melt_mass > 0.0) & (h2osno > 1.0e-8)
        if not torch.any(has_melt):
            return state
        frac_prev = (
            _snow_batch_like_torch(_to_torch(state.frac_sno, device, dtype), shape, device=device, dtype=dtype)
            if state.frac_sno is not None
            else torch.zeros_like(h2osno)
        )
        int_snow = (
            _snow_batch_like_torch(_to_torch(state.int_snow, device, dtype), shape, device=device, dtype=dtype)
            if state.int_snow is not None
            else h2osno
        )
        int_snow = torch.where(h2osno > 1.0e-8, torch.maximum(int_snow, h2osno), torch.zeros_like(int_snow))
        n_melt = _snow_n_melt_torch(static, shape, device=device, dtype=dtype)
        frac_melt = _snow_frac_from_swe_ratio_torch(h2osno / torch.clamp(int_snow, min=1.0e-12), n_melt)
        frac_sno = torch.where(has_melt, frac_melt, frac_prev)
        frac_sno = torch.where(h2osno > 1.0e-8, torch.clamp(frac_sno, min=0.0, max=1.0), torch.zeros_like(frac_sno))
        return CoupledColumnState(
            h2osoi_liq=state.h2osoi_liq,
            h2osoi_ice=state.h2osoi_ice,
            h2ocan=state.h2ocan,
            h2osfc=state.h2osfc,
            frac_h2osfc=state.frac_h2osfc,
            frac_sno=frac_sno,
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
            int_snow=int_snow,
        )

    h2osno = np.asarray(state.h2osno, dtype=np.float64)
    shape = h2osno.shape
    melt_mass = np.maximum(_snow_batch_like_numpy(np.asarray(snow_melt, dtype=np.float64), shape), 0.0) * float(dt)
    has_melt = (melt_mass > 0.0) & (h2osno > 1.0e-8)
    if not np.any(has_melt):
        return state
    frac_prev = (
        _snow_batch_like_numpy(_to_np(state.frac_sno), shape)
        if state.frac_sno is not None
        else np.zeros_like(h2osno, dtype=np.float64)
    )
    int_snow = (
        _snow_batch_like_numpy(_to_np(state.int_snow), shape)
        if state.int_snow is not None
        else h2osno
    )
    int_snow = np.where(h2osno > 1.0e-8, np.maximum(int_snow, h2osno), 0.0)
    n_melt = _snow_n_melt_numpy(static, shape)
    frac_melt = _snow_frac_from_swe_ratio_numpy(h2osno / np.maximum(int_snow, 1.0e-12), n_melt)
    frac_sno = np.where(has_melt, frac_melt, frac_prev)
    frac_sno = np.where(h2osno > 1.0e-8, np.clip(frac_sno, 0.0, 1.0), 0.0)
    return CoupledColumnState(
        h2osoi_liq=state.h2osoi_liq,
        h2osoi_ice=state.h2osoi_ice,
        h2ocan=state.h2ocan,
        h2osfc=state.h2osfc,
        frac_h2osfc=state.frac_h2osfc,
        frac_sno=frac_sno,
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
        int_snow=int_snow,
    )


def _initial_int_snow_numpy(
    h2osno: np.ndarray,
    frac_sno: np.ndarray,
    n_melt: np.ndarray,
) -> np.ndarray:
    """Recover a conservative accumulation memory when no restart value exists."""
    area_factor = np.maximum(_snow_area_factor_from_frac_numpy(frac_sno, n_melt), 1.0e-12)
    return np.where(h2osno > 1.0e-8, h2osno / area_factor, 0.0)


def _initial_int_snow_torch(
    h2osno: torch.Tensor,
    frac_sno: torch.Tensor,
    n_melt: torch.Tensor,
) -> torch.Tensor:
    """Recover a conservative accumulation memory when no restart value exists."""
    area_factor = torch.clamp(_snow_area_factor_from_frac_torch(frac_sno, n_melt), min=1.0e-12)
    return torch.where(h2osno > 1.0e-8, h2osno / area_factor, torch.zeros_like(h2osno))


def frac_sno_from_snow_storage(h2osno: np.ndarray, snow_depth: np.ndarray) -> np.ndarray:
    h2osno = np.asarray(h2osno, dtype=np.float64)
    snow_depth = np.asarray(snow_depth, dtype=np.float64)
    frac = np.zeros_like(h2osno, dtype=np.float64)
    mask = (h2osno > 1.0e-8) & (snow_depth > 0.0)
    if np.any(mask):
        with np.errstate(divide="ignore", invalid="ignore"):
            snowbd = np.zeros_like(h2osno, dtype=np.float64)
            snowbd[mask] = np.minimum(400.0, h2osno[mask] / np.maximum(snow_depth[mask], 1.0e-12))
            fmelt = np.maximum((snowbd / 100.0) ** 1.0, 1.0e-6)
            frac[mask] = np.tanh(snow_depth[mask] / (2.5 * SNOW_ZLND * fmelt[mask]))
    return np.clip(frac, 0.0, 1.0)


def frac_sno_from_snow_storage_torch(
    h2osno: np.ndarray | torch.Tensor,
    snow_depth: np.ndarray | torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    h2osno_t = _to_torch(h2osno, device, dtype)
    snow_depth_t = _to_torch(snow_depth, device, dtype)
    frac = torch.zeros_like(h2osno_t)
    mask = (h2osno_t > 1.0e-8) & (snow_depth_t > 0.0)
    if torch.any(mask):
        snowbd = torch.zeros_like(h2osno_t)
        snowbd[mask] = torch.minimum(
            torch.full_like(h2osno_t[mask], 400.0),
            h2osno_t[mask] / torch.clamp(snow_depth_t[mask], min=1.0e-12),
        )
        fmelt = torch.clamp((snowbd / 100.0) ** 1.0, min=1.0e-6)
        frac[mask] = torch.tanh(snow_depth_t[mask] / (2.5 * SNOW_ZLND * fmelt[mask]))
    return torch.clamp(frac, 0.0, 1.0)


def update_snow_state(
    state: CoupledColumnState,
    bundle: SurfaceModelBundle,
    surface_features: np.ndarray | torch.Tensor,
    surface_targets: Mapping[str, np.ndarray],
    dt: float,
    static: CoupledStaticParameters | None = None,
    *,
    scalar_snow_only: bool = False,
) -> CoupledColumnState:
    if dt > SNOW_UPDATE_MAX_DT:
        nsub = max(1, int(math.ceil(float(dt) / SNOW_UPDATE_MAX_DT)))
        sub_dt = float(dt) / float(nsub)
        sub_state = state
        for _ in range(nsub):
            sub_state = update_snow_state(
                sub_state,
                bundle,
                surface_features,
                surface_targets,
                sub_dt,
                static=static,
                scalar_snow_only=scalar_snow_only,
            )
        return sub_state

    if isinstance(surface_features, torch.Tensor) or any(
        isinstance(v, torch.Tensor) for v in (state.h2osno, state.snow_depth, state.snow_t) if v is not None
    ):
        device = surface_features.device if isinstance(surface_features, torch.Tensor) else _to_torch(state.h2osno, torch.device("cpu"), torch.float32).device
        dtype = surface_features.dtype if isinstance(surface_features, torch.Tensor) else torch.float32
        features_t = _to_torch(surface_features, device, dtype)
        idx_map = _bundle_input_index_map(tuple(bundle.input_names))
        forc_snow_t = torch.clamp(features_t[..., idx_map["forc_snow"]], min=0.0)
        qflx_snow_t = torch.clamp(_to_torch(surface_targets.get("qflx_snow_grnd", forc_snow_t), device, dtype), min=0.0)
        snow_old_t = _to_torch(snow_storage_from_state(state), device, dtype)
        snow_shape_t = tuple(snow_old_t.shape)
        forc_snow_t = torch.clamp(
            _snow_batch_like_torch(forc_snow_t, snow_shape_t, device=device, dtype=dtype),
            min=0.0,
        )
        qflx_snow_t = torch.clamp(
            _snow_batch_like_torch(qflx_snow_t, snow_shape_t, device=device, dtype=dtype),
            min=0.0,
        )
        active_idx = _cell_indices_from_mask((snow_old_t > 1.0e-8) | (qflx_snow_t > 0.0))
        if active_idx.numel() == 0:
            return state
        ncell = int(snow_shape_t[0]) if snow_shape_t else 1
        if active_idx.numel() < ncell:
            sub_state = _cell_index_coupled_state(state, active_idx)
            sub_features_t = _cell_index_select(features_t, active_idx, ncell=ncell, allow_1d=True)
            sub_targets_t = {
                name: _cell_index_select(value, active_idx, ncell=ncell)
                for name, value in surface_targets.items()
                if name in ("qflx_snow_grnd", "qflx_ev_snow")
            }
            sub_static = _cell_index_static_parameters(static, active_idx, ncell=ncell)
            snow_state_t = update_snow_state(
                sub_state,
                bundle,
                sub_features_t,
                sub_targets_t,
                dt,
                static=sub_static,
                scalar_snow_only=scalar_snow_only,
            )
            base_state_t = coupled_state_to_device(state, device, dtype)
            return _cell_scatter_coupled_state(base_state_t, active_idx, snow_state_t)

        forc_snow = qflx_snow_t
        forc_t = _snow_batch_like_torch(features_t[..., idx_map["forc_t"]], snow_shape_t, device=device, dtype=dtype)
        wind_speed = _snow_batch_like_torch(features_t[..., idx_map["wind_speed"]], snow_shape_t, device=device, dtype=dtype)
        qflx_ev_snow = _snow_batch_like_torch(
            _to_torch(surface_targets.get("qflx_ev_snow", 0.0), device, dtype),
            snow_shape_t,
            device=device,
            dtype=dtype,
        )
        snow_old = snow_old_t
        newsnow = torch.clamp(dt * forc_snow, min=0.0)
        frac_sno_prev = (
            _snow_batch_like_torch(_to_torch(state.frac_sno, device, dtype), snow_shape_t, device=device, dtype=dtype)
            if state.frac_sno is not None
            else torch.zeros_like(snow_old)
        )
        frac_sno_prev = torch.clamp(frac_sno_prev, min=0.0, max=1.0)
        qflx_ev_snow_store = qflx_ev_snow * frac_sno_prev
        if dt > 0.0:
            qflx_ev_snow_store = torch.where(
                qflx_ev_snow_store > 0.0,
                torch.minimum(
                    qflx_ev_snow_store,
                    torch.clamp((snow_old + newsnow) / float(dt), min=0.0),
                ),
                qflx_ev_snow_store,
            )
        deposition = torch.clamp(-float(dt) * qflx_ev_snow_store, min=0.0)
        snow_new = torch.clamp(snow_old + newsnow - float(dt) * qflx_ev_snow_store, min=0.0)
        n_melt = _snow_n_melt_torch(static, snow_shape_t, device=device, dtype=dtype)
        int_snow_prev = (
            _snow_batch_like_torch(_to_torch(state.int_snow, device, dtype), snow_shape_t, device=device, dtype=dtype)
            if state.int_snow is not None
            else _initial_int_snow_torch(snow_old, frac_sno_prev, n_melt)
        )
        int_snow_prev = torch.where(
            snow_old > 1.0e-8,
            torch.maximum(int_snow_prev, snow_old),
            torch.zeros_like(int_snow_prev),
        )
        swe_for_depletion = torch.where(newsnow > 0.0, snow_old, snow_new)
        depleting = (swe_for_depletion > 1.0e-8) & (int_snow_prev > swe_for_depletion + 1.0e-8)
        frac_sno_base = torch.where(
            depleting,
            _snow_frac_from_swe_ratio_torch(
                swe_for_depletion / torch.clamp(int_snow_prev, min=1.0e-12),
                n_melt,
            ),
            frac_sno_prev,
        )
        snow_after_new = torch.clamp(snow_old + newsnow, min=0.0)
        frac_sno_new = _snow_frac_after_new_snow_torch(frac_sno_base, snow_old, snow_after_new, newsnow)
        frac_sno_new = torch.where(snow_new > 1.0e-8, frac_sno_new, torch.zeros_like(frac_sno_new))
        frac_sno_new = _snow_batch_like_torch(frac_sno_new, snow_shape_t, device=device, dtype=dtype)
        area_factor = torch.clamp(_snow_area_factor_from_frac_torch(frac_sno_new, n_melt), min=1.0e-12)
        int_snow_reset = torch.minimum(snow_after_new / area_factor, torch.full_like(snow_after_new, 1.0e8))
        int_snow_new = torch.where(
            (newsnow > 0.0) & (snow_after_new > 1.0e-8),
            int_snow_reset + newsnow,
            int_snow_prev,
        )
        int_snow_new = torch.where(
            snow_new > 1.0e-8,
            torch.maximum(int_snow_new, snow_new),
            torch.zeros_like(int_snow_new),
        )
        snow_depth_prev = (
            _snow_batch_like_torch(_to_torch(state.snow_depth, device, dtype), snow_shape_t, device=device, dtype=dtype)
            if state.snow_depth is not None
            else snow_depth_from_storage_torch(snow_old, device=device, dtype=dtype)
        )
        fresh_density = _new_snow_bulk_density_torch(forc_t, wind_speed)
        snow_mass_gain = newsnow + deposition
        snow_depth_gain = torch.minimum(
            snow_mass_gain / torch.clamp(fresh_density * torch.clamp(frac_sno_new, min=1.0e-12), min=1.0e-12),
            snow_mass_gain / SNOW_GEOMETRY_MIN_DENSITY,
        )
        snow_depth_new = torch.where(
            snow_old > 0.0,
            torch.clamp(snow_depth_prev + snow_depth_gain, min=0.0),
            torch.where(
                (newsnow > 0.0) & (snow_new > 0.0),
                snow_depth_gain,
                torch.zeros_like(snow_new),
            ),
        )

        prev_layers = (
            _snow_layers_to_device(state.snow_layers, device, dtype)
            if _snow_layers_shape_ok(state.snow_layers, snow_shape_t)
            else None
        )
        prev_snl = torch.as_tensor(prev_layers.snl, device=device, dtype=torch.int32) if prev_layers is not None else None
        if prev_layers is not None and torch.any(prev_snl < 0):
            ice = torch.as_tensor(prev_layers.h2osoi_ice, device=device, dtype=dtype).clone()
            liq = torch.as_tensor(prev_layers.h2osoi_liq, device=device, dtype=dtype).clone()
            dz = torch.as_tensor(prev_layers.dz, device=device, dtype=dtype).clone()
            t_snow = torch.as_tensor(prev_layers.t_snow, device=device, dtype=dtype).clone()
            snl = prev_snl.clone()
            rds_prev = getattr(prev_layers, "snw_rds", None)
            rds = torch.as_tensor(rds_prev, device=device, dtype=dtype).clone() if rds_prev is not None else None
            active = snl < 0
            gain = torch.where(active, newsnow + deposition, torch.zeros_like(newsnow))
            evap = torch.where(active, torch.clamp(float(dt) * qflx_ev_snow_store, min=0.0), torch.zeros_like(newsnow))
            top_mass0 = torch.clamp(ice[..., 0] + liq[..., 0], min=0.0)
            frac_prev_eff = torch.clamp(frac_sno_prev, min=1.0e-12, max=1.0)
            top_density0 = torch.where(
                top_mass0 > 0.0,
                torch.maximum(
                    top_mass0 / torch.clamp(dz[..., 0] * frac_prev_eff, min=1.0e-12),
                    torch.full_like(top_mass0, 50.0),
                ),
                fresh_density,
            )
            ice[..., 0] = ice[..., 0] + gain
            gain_depth = torch.minimum(
                gain / torch.clamp(fresh_density * torch.clamp(frac_sno_new, min=1.0e-12), min=1.0e-12),
                gain / SNOW_GEOMETRY_MIN_DENSITY,
            )
            dz[..., 0] = torch.clamp(
                dz[..., 0]
                + gain_depth
                - evap / torch.clamp(top_density0 * frac_prev_eff, min=1.0e-12),
                min=1.0e-6,
            )
            ice_e = torch.minimum(evap, torch.clamp(ice[..., 0], min=0.0))
            ice[..., 0] = torch.clamp(ice[..., 0] - ice_e, min=0.0)
            liq_e = torch.minimum(evap - ice_e, torch.clamp(liq[..., 0], min=0.0))
            liq[..., 0] = torch.clamp(liq[..., 0] - liq_e, min=0.0)
            new_snow = ~active & (newsnow > 0.0) & (snow_new > 0.0)
            if torch.any(new_snow):
                fresh_layers = _snow_layers_from_storage_torch(
                    torch.where(new_snow, snow_new, torch.zeros_like(snow_new)),
                    torch.where(new_snow, snow_depth_new, torch.zeros_like(snow_depth_new)),
                    snow_t=_to_torch(state.snow_t, device, dtype) if state.snow_t is not None else None,
                    frac_sno=torch.where(new_snow, frac_sno_new, torch.zeros_like(frac_sno_new)),
                    device=device,
                    dtype=dtype,
                )
                m = new_snow[..., None]
                snl = torch.where(new_snow, fresh_layers.snl, snl)
                dz = torch.where(m, fresh_layers.dz, dz)
                ice = torch.where(m, fresh_layers.h2osoi_ice, ice)
                liq = torch.where(m, fresh_layers.h2osoi_liq, liq)
                t_snow = torch.where(m, fresh_layers.t_snow, t_snow)
                if rds is not None and fresh_layers.snw_rds is not None:
                    rds = torch.where(m, fresh_layers.snw_rds, rds)
            gone = snow_new <= 0.0
            if torch.any(gone):
                m = gone[..., None]
                snl = torch.where(gone, torch.zeros_like(snl), snl)
                dz = torch.where(m, torch.zeros_like(dz), dz)
                ice = torch.where(m, torch.zeros_like(ice), ice)
                liq = torch.where(m, torch.zeros_like(liq), liq)
            snow_layers_new = SnowLayerState(
                snl=snl,
                dz=dz,
                h2osoi_ice=ice,
                h2osoi_liq=liq,
                t_snow=t_snow,
                snw_rds=rds,
            )
        else:
            promote_frac = torch.where(newsnow > 0.0, frac_sno_new, torch.zeros_like(frac_sno_new))
            snow_layers_new = _snow_layers_from_storage_torch(
                snow_new,
                snow_depth_new,
                snow_t=_to_torch(state.snow_t, device, dtype) if state.snow_t is not None else None,
                frac_sno=promote_frac,
                device=device,
                dtype=dtype,
            )
        if scalar_snow_only:
            snow_layers_new = _scalar_snow_layers_from_storage_torch(
                snow_new,
                snow_depth_new,
                snow_t=_to_torch(state.snow_t, device, dtype) if state.snow_t is not None else None,
                device=device,
                dtype=dtype,
            )
        else:
            snow_layers_new = _compact_snow_layers_torch(
                snow_layers_new,
                frac_sno_new,
                dt,
                device=device,
                dtype=dtype,
            )
            snow_layers_new = _rebalance_snow_layers_torch(snow_layers_new, frac_sno_new)
            snow_new = _snow_mass_from_layers_torch(snow_layers_new, device=device, dtype=dtype)
            snow_depth_new = _snow_depth_from_layers_torch(snow_layers_new, device=device, dtype=dtype)
        frac_sno_new = _snow_batch_like_torch(frac_sno_new, tuple(snow_new.shape), device=device, dtype=dtype)
        frac_sno_new = torch.where(
            snow_new > 1.0e-8,
            torch.clamp(frac_sno_new, min=0.0, max=1.0),
            torch.zeros_like(frac_sno_new),
        )
        int_snow_new = torch.where(
            snow_new > 1.0e-8,
            torch.maximum(int_snow_new, snow_new),
            torch.zeros_like(int_snow_new),
        )
        out_state = CoupledColumnState(
            h2osoi_liq=_to_torch(state.h2osoi_liq, device, dtype),
            h2osoi_ice=_to_torch(state.h2osoi_ice, device, dtype),
            h2ocan=_to_torch(state.h2ocan, device, dtype),
            h2osno=snow_new,
            snow_depth=snow_depth_new,
            h2osfc=_to_torch(state.h2osfc, device, dtype),
            frac_h2osfc=_to_torch(state.frac_h2osfc, device, dtype),
            frac_sno=frac_sno_new,
            zwt=_to_torch(state.zwt, device, dtype),
            t_soisno=_to_torch(state.t_soisno, device, dtype),
            wa=_to_torch(state.wa, device, dtype) if state.wa is not None else None,
            t_h2osfc=_to_torch(state.t_h2osfc, device, dtype) if state.t_h2osfc is not None else None,
            t10=_to_torch(state.t10, device, dtype) if state.t10 is not None else None,
            t10_history=_to_torch(state.t10_history, device, dtype) if state.t10_history is not None else None,
            snow_t=_to_torch(state.snow_t, device, dtype) if state.snow_t is not None else None,
            snow_layers=snow_layers_new,
            int_snow=int_snow_new,
        )
        return force_scalar_snow_state_torch(out_state, device=device, dtype=dtype) if scalar_snow_only else out_state
    features_np = _to_numpy(surface_features)
    assert features_np is not None
    fidx = {name: i for i, name in enumerate(bundle.input_names)}
    idx_map = _bundle_input_index_map(tuple(bundle.input_names))
    forc_snow = np.maximum(
        np.asarray(surface_targets.get("qflx_snow_grnd", features_np[..., fidx["forc_snow"]]), dtype=np.float64),
        0.0,
    )
    qflx_ev_snow = np.asarray(surface_targets.get("qflx_ev_snow", 0.0), dtype=np.float64)
    snow_old = np.asarray(snow_storage_from_state(state), dtype=np.float64)
    snow_shape = snow_old.shape
    forc_snow = np.maximum(_snow_batch_like_numpy(forc_snow, snow_shape), 0.0)
    qflx_ev_snow = _snow_batch_like_numpy(qflx_ev_snow, snow_shape)
    forc_t = _snow_batch_like_numpy(features_np[..., idx_map["forc_t"]], snow_shape)
    wind_speed = _snow_batch_like_numpy(features_np[..., idx_map["wind_speed"]], snow_shape)
    newsnow = np.maximum(dt * forc_snow, 0.0)
    frac_sno_prev = (
        _snow_batch_like_numpy(_to_np(state.frac_sno), snow_shape)
        if state.frac_sno is not None
        else np.zeros_like(snow_old, dtype=np.float64)
    )
    frac_sno_prev = np.clip(frac_sno_prev, 0.0, 1.0)
    qflx_ev_snow_store = qflx_ev_snow * frac_sno_prev
    if dt > 0.0:
        qflx_ev_snow_store = np.where(
            qflx_ev_snow_store > 0.0,
            np.minimum(qflx_ev_snow_store, np.maximum((snow_old + newsnow) / float(dt), 0.0)),
            qflx_ev_snow_store,
        )
    deposition = np.maximum(-float(dt) * qflx_ev_snow_store, 0.0)
    snow_new = np.maximum(snow_old + newsnow - float(dt) * qflx_ev_snow_store, 0.0)
    n_melt = _snow_n_melt_numpy(static, snow_old.shape)
    int_snow_prev = (
        _snow_batch_like_numpy(_to_np(state.int_snow), snow_shape)
        if state.int_snow is not None
        else _initial_int_snow_numpy(snow_old, frac_sno_prev, n_melt)
    )
    int_snow_prev = np.where(snow_old > 1.0e-8, np.maximum(int_snow_prev, snow_old), 0.0)
    swe_for_depletion = np.where(newsnow > 0.0, snow_old, snow_new)
    depleting = (swe_for_depletion > 1.0e-8) & (int_snow_prev > swe_for_depletion + 1.0e-8)
    frac_sno_base = np.where(
        depleting,
        _snow_frac_from_swe_ratio_numpy(
            swe_for_depletion / np.maximum(int_snow_prev, 1.0e-12),
            n_melt,
        ),
        frac_sno_prev,
    )
    snow_after_new = np.maximum(snow_old + newsnow, 0.0)
    frac_sno_new = _snow_frac_after_new_snow_numpy(frac_sno_base, snow_old, snow_after_new, newsnow)
    frac_sno_new = np.where(snow_new > 1.0e-8, frac_sno_new, 0.0)
    frac_sno_new = _snow_batch_like_numpy(frac_sno_new, snow_new.shape)
    area_factor = np.maximum(_snow_area_factor_from_frac_numpy(frac_sno_new, n_melt), 1.0e-12)
    int_snow_reset = np.minimum(snow_after_new / area_factor, 1.0e8)
    int_snow_new = np.where((newsnow > 0.0) & (snow_after_new > 1.0e-8), int_snow_reset + newsnow, int_snow_prev)
    int_snow_new = np.where(snow_new > 1.0e-8, np.maximum(int_snow_new, snow_new), 0.0)
    fresh_density0 = _new_snow_bulk_density_numpy(
        forc_t,
        wind_speed,
    )
    snow_depth_prev0 = (
        _snow_batch_like_numpy(_to_np(state.snow_depth), snow_shape)
        if state.snow_depth is not None
        else snow_depth_from_storage(snow_old)
    )
    snow_mass_gain = newsnow + deposition
    snow_depth_gain = np.minimum(
        snow_mass_gain / np.maximum(fresh_density0 * np.maximum(frac_sno_new, 1.0e-12), 1.0e-12),
        snow_mass_gain / SNOW_GEOMETRY_MIN_DENSITY,
    )
    snow_depth_new = np.where(
        snow_old > 0.0,
        np.maximum(snow_depth_prev0 + snow_depth_gain, 0.0),
        np.where(
            (newsnow > 0.0) & (snow_new > 0.0),
            snow_depth_gain,
            0.0,
        ),
    )
    # Debug: report initial snow_depth_from_storage and frac_sno before layering
    try:
        import os
        if os.environ.get("CF_DEBUG_SNOW") == "1":
            # only print when there is new snow or a 0->>0 transition to keep logs concise
            try:
                cond = np.any(snow_new > 0.0) or np.any((np.asarray(snow_old) == 0.0) & (np.asarray(snow_new) > 0.0))
            except Exception:
                cond = False
            if cond:
                print("[CF_DEBUG_SNOW] initial_snow_depth_from_storage=", np.asarray(snow_depth_new).reshape(-1)[:5])
                print("[CF_DEBUG_SNOW] initial_frac_sno=", np.asarray(frac_sno_new).reshape(-1)[:5])
    except Exception:
        pass
    # Use existing snow layers as base so that liquid water produced by the
    # thermal phase-change solver persists across timesteps rather than being
    # reset to all-ice every step.  Fall back to a full re-initialisation only
    # when no valid layers are available yet.
    _prev = state.snow_layers
    _prev_snl = np.asarray(_prev.snl, dtype=np.int32) if _prev is not None else None
    if _prev is not None and np.any(_prev_snl < 0):
        _ice = np.array(_prev.h2osoi_ice, dtype=np.float64, copy=True)
        _liq = np.array(_prev.h2osoi_liq, dtype=np.float64, copy=True)
        _dz  = np.array(_prev.dz,          dtype=np.float64, copy=True)
        _t   = np.array(_prev.t_snow,      dtype=np.float64, copy=True)
        _snl = _prev_snl.copy()
        _rds = (np.array(_prev.snw_rds, dtype=np.float64, copy=True)
                if _prev.snw_rds is not None else None)
        _active = _snl < 0  # per-element: existing active snow
        _fresh_density = _new_snow_bulk_density_numpy(
            forc_t,
            wind_speed,
        )
        # Add new snowfall as fresh ice to the top active layer.  Also grow
        # the top-layer thickness so accumulated mass increases snow depth
        # (previously we only increased ice mass which inflated density
        # without changing depth).
        _gain = np.where(_active, newsnow + deposition, 0.0)
        _evap = np.where(_active, np.maximum(float(dt) * qflx_ev_snow_store, 0.0), 0.0)
        _top_mass0 = np.maximum(_ice[..., 0] + _liq[..., 0], 0.0)
        _frac_prev = np.clip(frac_sno_prev, 1.0e-12, 1.0)
        _top_density0 = np.where(
            _top_mass0 > 0.0,
            np.maximum(_top_mass0 / np.maximum(_dz[..., 0] * _frac_prev, 1.0e-12), 50.0),
            _fresh_density,
        )
        _ice[..., 0] = _ice[..., 0] + _gain
        # Convert added mass (kg/m²) to fresh-snow depth using an ELM-style
        # bulk density for new snow; remove evaporated mass using the current
        # pack density so shallow events do not linger.
        _gain_depth = np.minimum(
            _gain / np.maximum(_fresh_density * np.maximum(frac_sno_new, 1.0e-12), 1.0e-12),
            _gain / SNOW_GEOMETRY_MIN_DENSITY,
        )
        _dz[..., 0] = np.maximum(
            _dz[..., 0]
            + _gain_depth
            - (_evap / np.maximum(_top_density0 * _frac_prev, 1.0e-12)),
            1.0e-6,
        )
        # Remove evaporated mass from the top layer (ice first, then liquid).
        _ice_e = np.minimum(_evap, np.maximum(_ice[..., 0], 0.0))
        _ice[..., 0] = np.maximum(_ice[..., 0] - _ice_e, 0.0)
        _liq_e = np.minimum(_evap - _ice_e, np.maximum(_liq[..., 0], 0.0))
        _liq[..., 0] = np.maximum(_liq[..., 0] - _liq_e, 0.0)
        # Elements with no prior snow that now have snow: create fresh layers.
        _new_snow = ~_active & (newsnow > 0.0) & (snow_new > 0.0)
        if np.any(_new_snow):
            _fresh = _snow_layers_from_storage_numpy(
                np.where(_new_snow, snow_new, 0.0),
                np.where(_new_snow, snow_depth_new, 0.0),
                snow_t=(_to_np(state.snow_t)
                        if state.snow_t is not None else None),
                frac_sno=np.where(_new_snow, frac_sno_new, 0.0),
            )
            _m = _new_snow[..., None]
            _snl = np.where(_new_snow, _fresh.snl,        _snl)
            _dz  = np.where(_m,        _fresh.dz,          _dz)
            _ice = np.where(_m,        _fresh.h2osoi_ice,  _ice)
            _liq = np.where(_m,        _fresh.h2osoi_liq,  _liq)
            _t   = np.where(_m,        _fresh.t_snow,      _t)
            if _rds is not None and _fresh.snw_rds is not None:
                _rds = np.where(_m, _fresh.snw_rds, _rds)
        # Elements where all snow has vanished: zero out all layers.
        _gone = np.asarray(snow_new <= 0.0)
        if np.any(_gone):
            _snl = np.where(_gone, 0, _snl)
            _dz[_gone] = 0.0
            _ice[_gone] = 0.0
            _liq[_gone] = 0.0
        snow_layers_new = SnowLayerState(
            snl=_snl, dz=_dz, h2osoi_ice=_ice, h2osoi_liq=_liq,
            t_snow=_t, snw_rds=_rds,
        )
    else:
        _promote_frac = np.where(newsnow > 0.0, frac_sno_new, 0.0)
        snow_layers_new = _snow_layers_from_storage_numpy(
            snow_new,
            snow_depth_new,
            snow_t=(_to_np(state.snow_t)
                    if state.snow_t is not None else None),
            frac_sno=_promote_frac,
        )
    if scalar_snow_only:
        snow_layers_new = _scalar_snow_layers_from_storage_numpy(
            snow_new,
            snow_depth_new,
            snow_t=(_to_np(state.snow_t) if state.snow_t is not None else None),
        )
    else:
        snow_layers_new = _compact_snow_layers_numpy(snow_layers_new, frac_sno_new, dt)
        snow_layers_new = _rebalance_snow_layers_numpy(snow_layers_new, frac_sno_new)
        snow_new = _snow_mass_from_layers_numpy(snow_layers_new)
        snow_depth_new = _snow_depth_from_layers_numpy(snow_layers_new)
    frac_sno_new = _snow_batch_like_numpy(frac_sno_new, snow_new.shape)
    frac_sno_new = np.where(snow_new > 1.0e-8, np.clip(frac_sno_new, 0.0, 1.0), 0.0)
    int_snow_new = np.where(snow_new > 1.0e-8, np.maximum(int_snow_new, snow_new), 0.0)
    # Grain aging — only runs when SNICAR tables have been loaded.
    if (not scalar_snow_only) and _SNICAR_AGING_TABLES is not None and snow_layers_new.snw_rds is not None:
        layers = snow_layers_new
        ice_l = np.asarray(layers.h2osoi_ice, dtype=np.float64)
        liq_l = np.asarray(layers.h2osoi_liq, dtype=np.float64)
        dz_l  = np.maximum(np.asarray(layers.dz, dtype=np.float64), 1.0e-6)
        t_l   = np.asarray(layers.t_snow, dtype=np.float64)
        snl_l = np.asarray(layers.snl, dtype=np.int32)
        rho_l = np.maximum((ice_l + liq_l) / (dz_l * np.clip(frac_sno_new[..., None], 1.0e-12, 1.0)), 50.0)
        # Temperature gradient between layer centres.
        nlyr = t_l.shape[-1]
        dTdz_l = np.zeros_like(t_l)
        for k in range(nlyr):
            if k + 1 < nlyr:
                mid = (dz_l[..., k] + dz_l[..., k + 1]) / 2.0
                dTdz_l[..., k] = np.abs((t_l[..., k] - t_l[..., k + 1]) / np.maximum(mid * np.clip(frac_sno_new[..., None], 1.0e-12, 1.0), 1.0e-6))
        mass_l = ice_l + liq_l
        frc_liq_l = np.where(mass_l > 0, np.clip(liq_l / np.maximum(mass_l, 1.0e-12), 0.0, 0.1), 0.0)
        # New snowfall fraction (top layer only).
        frc_newsnow_l = np.zeros_like(t_l)
        top_mass = np.maximum(ice_l[..., 0] + liq_l[..., 0], 1.0e-12)
        frc_newsnow_l[..., 0] = np.where(snl_l < 0, np.clip(forc_snow * dt / top_mass, 0.0, 1.0), 0.0)
        frc_refrz_l = np.zeros_like(t_l)  # refreezing not tracked in emulator
        new_rds = snow_grain_aging_step(
            np.asarray(layers.snw_rds, dtype=np.float64),
            t_l, dTdz_l, rho_l, frc_liq_l, frc_newsnow_l, frc_refrz_l,
            snl_l, dt,
        )
        snow_layers_new = SnowLayerState(
            snl=layers.snl, dz=layers.dz,
            h2osoi_ice=layers.h2osoi_ice, h2osoi_liq=layers.h2osoi_liq,
            t_snow=layers.t_snow, snw_rds=new_rds,
        )
    out_state = CoupledColumnState(
        h2osoi_liq=_to_np(state.h2osoi_liq),
        h2osoi_ice=_to_np(state.h2osoi_ice),
        h2ocan=_to_np(state.h2ocan),
        h2osno=snow_new,
        snow_depth=snow_depth_new,
        h2osfc=_to_np(state.h2osfc),
        frac_h2osfc=_to_np(state.frac_h2osfc),
        frac_sno=frac_sno_new,
        zwt=_to_np(state.zwt),
        t_soisno=_to_np(state.t_soisno),
        wa=_to_np(state.wa) if state.wa is not None else None,
        t_h2osfc=_to_np(state.t_h2osfc) if state.t_h2osfc is not None else None,
        t10=_to_np(state.t10) if state.t10 is not None else None,
        t10_history=_to_np(state.t10_history) if state.t10_history is not None else None,
        snow_t=_to_np(state.snow_t) if state.snow_t is not None else None,
        snow_layers=snow_layers_new,
        int_snow=int_snow_new,
    )
    # Optional debug prints when environment variable CF_DEBUG_SNOW=1
    try:
        import os
        if os.environ.get("CF_DEBUG_SNOW") == "1":
            try:
                cond = np.any(snow_new > 0.0) or np.any((np.asarray(snow_old) == 0.0) & (np.asarray(snow_new) > 0.0))
            except Exception:
                cond = False
            if cond:
                print("[CF_DEBUG_SNOW] snow_old=", np.asarray(snow_old).reshape(-1)[:5])
                print("[CF_DEBUG_SNOW] snow_new=", np.asarray(snow_new).reshape(-1)[:5])
                print("[CF_DEBUG_SNOW] snow_depth_new=", np.asarray(snow_depth_new).reshape(-1)[:5])
                if snow_layers_new is not None:
                    print("[CF_DEBUG_SNOW] snow_layers.ice top=", np.asarray(snow_layers_new.h2osoi_ice).reshape(-1,5)[:5,0])
                    print("[CF_DEBUG_SNOW] snow_layers.liq top=", np.asarray(snow_layers_new.h2osoi_liq).reshape(-1,5)[:5,0])
    except Exception:
        pass
    return force_scalar_snow_state_numpy(out_state) if scalar_snow_only else out_state


# ---------------------------------------------------------------------------
# SNICAR snow grain aging tables and albedo parameterisation
# ---------------------------------------------------------------------------

#: Module-level singleton for the SNICAR aging lookup tables.  Populated by
#: :func:`load_snicar_aging_tables` before the first call to
#: :func:`snow_grain_aging_step`.
_SNICAR_AGING_TABLES: dict[str, np.ndarray] | None = None


def load_snicar_aging_tables(path: str | Path) -> None:
    """Load the Flanner & Zender (2006) grain-aging lookup tables from NetCDF.

    Parameters
    ----------
    path:
        Path to ``snicar_drdt_bst_fit_60_c070416.nc`` (or equivalent).  The
        file must contain variables ``tau`` [hr], ``kappa`` [–], and
        ``drdsdt0`` [µm hr-1], each of shape ``(idx_rhos, idx_Tgrd, idx_T)``
        = ``(8, 31, 11)``.
    """
    global _SNICAR_AGING_TABLES
    with Dataset(str(path)) as ds:
        _SNICAR_AGING_TABLES = {
            "tau":   np.asarray(ds.variables["tau"][:],     dtype=np.float64),  # (8,31,11)
            "kappa": np.asarray(ds.variables["kappa"][:],   dtype=np.float64),
            "drdt0": np.asarray(ds.variables["drdsdt0"][:], dtype=np.float64),  # µm hr-1
        }


def snow_grain_aging_step(
    snw_rds: np.ndarray,
    t_snow: np.ndarray,
    dTdz: np.ndarray,
    rhos: np.ndarray,
    frc_liq: np.ndarray,
    frc_newsnow: np.ndarray,
    frc_refrz: np.ndarray,
    active: np.ndarray,
    dt: float,
) -> np.ndarray:
    """Advance snow effective grain radius by one time step.

    Implements the Flanner & Zender (2006) dry-growth parameterisation and
    the Brun (1989) wet-growth parameterisation, following ELM
    ``SnowAge_grain`` exactly.

    Parameters
    ----------
    snw_rds : (…, nlyr)  current effective radius [µm]
    t_snow  : (…, nlyr)  layer temperature [K]
    dTdz    : (…, nlyr)  |dT/dz| temperature gradient [K m-1]
    rhos    : (…, nlyr)  layer bulk density [kg m-3]
    frc_liq : (…, nlyr)  liquid-water fraction of layer mass [0–1]
    frc_newsnow : (…, nlyr)  fresh-snowfall fraction (non-zero for top layer)
    frc_refrz   : (…, nlyr)  re-frozen fraction [0–1]
    active  : (…,)       number of active snow layers (integer)
    dt      : float       timestep [s]

    Returns
    -------
    np.ndarray, same shape as snw_rds, updated grain radius [µm]
    """
    tables = _SNICAR_AGING_TABLES
    if tables is None:
        # No tables loaded — return unchanged.
        return snw_rds

    tau_tbl   = tables["tau"]    # (8, 31, 11)
    kappa_tbl = tables["kappa"]
    drdt0_tbl = tables["drdt0"]

    snw_rds = np.array(snw_rds, dtype=np.float64, copy=True)
    shape = snw_rds.shape[:-1]
    nlyr  = snw_rds.shape[-1]

    # Flatten all but the layer axis for vectorised indexing.
    rds   = snw_rds.reshape(-1, nlyr)
    T     = t_snow.reshape(-1, nlyr)
    Tg    = dTdz.reshape(-1, nlyr)
    rho   = np.maximum(rhos.reshape(-1, nlyr), 50.0)
    liq   = np.clip(frc_liq.reshape(-1, nlyr), 0.0, 0.1)
    fnew  = np.clip(frc_newsnow.reshape(-1, nlyr), 0.0, 1.0)
    frefrz = np.clip(frc_refrz.reshape(-1, nlyr), 0.0, 1.0)
    act   = np.maximum(-np.asarray(active, dtype=np.int32).reshape(-1), 0)

    dt_hr = dt / 3600.0

    # Lookup table indices (0-based, matching Fortran 1-based after -1).
    T_idx    = np.clip(np.round((T - 223.0) / 5.0).astype(int),  0, 10)   # 0..10
    Tgrd_idx = np.clip(np.round(Tg / 10.0).astype(int),           0, 30)   # 0..30
    rhos_idx = np.clip(np.round((rho - 50.0) / 50.0).astype(int), 0,  7)   # 0..7

    bst_tau   = tau_tbl  [rhos_idx, Tgrd_idx, T_idx]
    bst_kappa = kappa_tbl[rhos_idx, Tgrd_idx, T_idx]
    bst_drdt0 = drdt0_tbl[rhos_idx, Tgrd_idx, T_idx]

    # Dry grain growth: dr/dt = drdt0 * (tau / (dr_fresh + tau))^(1/kappa)
    dr_fresh = np.maximum(rds - SNW_RDS_MIN, 0.0)
    tau_safe = np.maximum(bst_tau, 1.0e-12)
    dr_dry = bst_drdt0 * (tau_safe / (dr_fresh + tau_safe)) ** (1.0 / np.maximum(bst_kappa, 1.0e-12)) * dt_hr

    # Wet grain growth: Brun (1989) — C1 term set to zero as in ELM.
    # dr_wet [µm] = 1e18 * dt * C2 * frc_liq^3 / (4π * r^2)
    rds_safe = np.maximum(rds, SNW_RDS_MIN)
    dr_wet = 1.0e18 * (dt * C2_LIQ_BRUN89 * liq ** 3) / (4.0 * math.pi * rds_safe ** 2)

    dr = dr_dry + dr_wet

    # Mass-weighted blend of aged, fresh, and refrozen snow.
    fnew_clamped = np.minimum(fnew, 1.0)
    frefrz_clamped = np.minimum(frefrz, 1.0)
    sum_fractions = fnew_clamped + frefrz_clamped
    overflow = sum_fractions > 1.0
    frefrz_clamped = np.where(overflow, frefrz_clamped / sum_fractions, frefrz_clamped)
    fnew_clamped   = np.where(overflow, 1.0 - frefrz_clamped, fnew_clamped)
    fold = 1.0 - fnew_clamped - frefrz_clamped

    rds_new = (rds + dr) * fold + SNW_RDS_MIN * fnew_clamped + SNW_RDS_REFRZ * frefrz_clamped
    rds_new = np.clip(rds_new, SNW_RDS_MIN, SNW_RDS_MAX)

    # Zero out inactive layers.
    for k in range(nlyr):
        rds_new[:, k] = np.where(act > k, rds_new[:, k], 0.0)

    return rds_new.reshape(shape + (nlyr,))


def _snow_albedo_from_grain_radius(snw_rds_top: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return clean-snow hemispherical reflectance for VIS and NIR bands.

    Parameterised as a function of the top-layer effective grain radius, fit to
    SNICAR output following Flanner & Zender (2006) / Gardner & Sharp (2010):

    - VIS (0.3–0.7 µm): nearly flat with a weak power-law increase with *r*.
    - NIR (0.7–5.0 µm): exponential decrease with √r.

    Tuned to SNICAR single-layer output at SZA 60° (clean snow, no aerosol):
      r=54µm → (α_vis, α_nir) ≈ (1.00, 0.62);  r=1500µm → (0.97, 0.27).

    Parameters
    ----------
    snw_rds_top : effective grain radius [µm], broadcast-able array.

    Returns
    -------
    alpha_vis, alpha_nir : np.ndarray
    """
    r = np.asarray(snw_rds_top, dtype=np.float64)
    # VIS: 1 – 0.000149 * max(r – r_min, 0)^0.754
    alpha_vis = 1.0 - 0.000149 * np.maximum(r - SNW_RDS_MIN, 0.0) ** 0.754
    # NIR: 0.654 * exp(-0.0448 * sqrt(r)) + 0.15
    alpha_nir = 0.654 * np.exp(-0.0448 * np.sqrt(np.maximum(r, SNW_RDS_MIN))) + 0.15
    return np.clip(alpha_vis, 0.0, 1.0), np.clip(alpha_nir, 0.0, 1.0)


def _snow_albedo_from_grain_radius_torch(snw_rds_top: np.ndarray | torch.Tensor, *, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """Torch equivalent of `_snow_albedo_from_grain_radius`."""
    r = _to_torch(snw_rds_top, device, dtype)
    alpha_vis = 1.0 - 0.000149 * torch.clamp(r - SNW_RDS_MIN, min=0.0) ** 0.754
    alpha_nir = 0.654 * torch.exp(-0.0448 * torch.sqrt(torch.clamp(r, min=SNW_RDS_MIN))) + 0.15
    return torch.clamp(alpha_vis, 0.0, 1.0), torch.clamp(alpha_nir, 0.0, 1.0)


_ELM_SOIL_ALBSAT_20 = np.asarray(
    [
        [0.25, 0.50],
        [0.23, 0.46],
        [0.21, 0.42],
        [0.20, 0.40],
        [0.19, 0.38],
        [0.18, 0.36],
        [0.17, 0.34],
        [0.16, 0.32],
        [0.15, 0.30],
        [0.14, 0.28],
        [0.13, 0.26],
        [0.12, 0.24],
        [0.11, 0.22],
        [0.10, 0.20],
        [0.09, 0.18],
        [0.08, 0.16],
        [0.07, 0.14],
        [0.06, 0.12],
        [0.05, 0.10],
        [0.04, 0.08],
    ],
    dtype=np.float64,
)
_ELM_SOIL_ALBDRY_20 = np.asarray(
    [
        [0.36, 0.61],
        [0.34, 0.57],
        [0.32, 0.53],
        [0.31, 0.51],
        [0.30, 0.49],
        [0.29, 0.48],
        [0.28, 0.45],
        [0.27, 0.43],
        [0.26, 0.41],
        [0.25, 0.39],
        [0.24, 0.37],
        [0.23, 0.35],
        [0.22, 0.33],
        [0.20, 0.31],
        [0.18, 0.29],
        [0.16, 0.27],
        [0.14, 0.25],
        [0.12, 0.23],
        [0.10, 0.21],
        [0.08, 0.16],
    ],
    dtype=np.float64,
)
_ELM_SOIL_ALBEDO_TORCH_CACHE: dict[tuple[str, str], tuple[torch.Tensor, torch.Tensor]] = {}


def _elm_soil_albedo_from_color(
    soil_color: np.ndarray | float | None,
    h2osoi_vol_top: np.ndarray | float | None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """ELM 20-class direct soil albedo for VIS/NIR bands."""
    if soil_color is None or h2osoi_vol_top is None:
        return None
    soil_color_np = np.asarray(soil_color, dtype=np.float64)
    soil_color_np = np.where(np.isfinite(soil_color_np), soil_color_np, 15.0)
    idx = np.clip(np.rint(soil_color_np).astype(np.int64), 1, 20) - 1
    h2osoi_vol_np = np.asarray(h2osoi_vol_top, dtype=np.float64)
    inc = np.maximum(0.11 - 0.40 * np.maximum(h2osoi_vol_np, 0.0), 0.0)
    alb_vis = np.minimum(_ELM_SOIL_ALBSAT_20[idx, 0] + inc, _ELM_SOIL_ALBDRY_20[idx, 0])
    alb_nir = np.minimum(_ELM_SOIL_ALBSAT_20[idx, 1] + inc, _ELM_SOIL_ALBDRY_20[idx, 1])
    return alb_vis, alb_nir


def _elm_soil_albedo_tables_torch(device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    key = (str(device), str(dtype))
    cached = _ELM_SOIL_ALBEDO_TORCH_CACHE.get(key)
    if cached is None:
        cached = (
            torch.as_tensor(_ELM_SOIL_ALBSAT_20, device=device, dtype=dtype),
            torch.as_tensor(_ELM_SOIL_ALBDRY_20, device=device, dtype=dtype),
        )
        _ELM_SOIL_ALBEDO_TORCH_CACHE[key] = cached
    return cached


def _elm_soil_albedo_from_color_torch(
    soil_color: np.ndarray | torch.Tensor | float | None,
    h2osoi_vol_top: np.ndarray | torch.Tensor | float | None,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Torch equivalent of `_elm_soil_albedo_from_color`."""
    if soil_color is None or h2osoi_vol_top is None:
        return None
    albsat, albdry = _elm_soil_albedo_tables_torch(device, dtype)
    soil_color_t = _to_torch(soil_color, device, dtype)
    soil_color_t = torch.where(torch.isfinite(soil_color_t), soil_color_t, torch.full_like(soil_color_t, 15.0))
    idx = torch.clamp(torch.round(soil_color_t).to(torch.long), 1, 20) - 1
    flat_idx = idx.reshape(-1)
    albsat_vis = albsat[:, 0].index_select(0, flat_idx).reshape(idx.shape)
    albsat_nir = albsat[:, 1].index_select(0, flat_idx).reshape(idx.shape)
    albdry_vis = albdry[:, 0].index_select(0, flat_idx).reshape(idx.shape)
    albdry_nir = albdry[:, 1].index_select(0, flat_idx).reshape(idx.shape)
    h2osoi_vol_t = torch.clamp(_to_torch(h2osoi_vol_top, device, dtype), min=0.0)
    inc = torch.clamp(0.11 - 0.40 * h2osoi_vol_t, min=0.0)
    return torch.minimum(albsat_vis + inc, albdry_vis), torch.minimum(albsat_nir + inc, albdry_nir)


def _np_safe_divide(num: np.ndarray, den: np.ndarray, eps: float = 1.0e-12) -> np.ndarray:
    den_safe = np.where(np.abs(den) > eps, den, np.where(den < 0.0, -eps, eps))
    return num / den_safe


def _torch_safe_divide(num: torch.Tensor, den: torch.Tensor, eps: float = 1.0e-12) -> torch.Tensor:
    eps_t = torch.as_tensor(eps, device=den.device, dtype=den.dtype)
    den_safe = torch.where(
        torch.abs(den) > eps_t,
        den,
        torch.where(den < 0.0, -eps_t, eps_t),
    )
    return num / den_safe


def _elm_twostream_transmission(
    *,
    rho: np.ndarray,
    tau: np.ndarray,
    albgrd: np.ndarray,
    albgri: np.ndarray,
    elai: np.ndarray,
    esai: np.ndarray,
    coszen: np.ndarray,
    xl: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Approximate ELM SurfaceAlbedoMod two-stream ground transmission."""
    eps = 1.0e-8
    laisum = np.maximum(np.asarray(elai, dtype=np.float64), 0.0) + np.maximum(np.asarray(esai, dtype=np.float64), 0.0)
    cosz = np.maximum(np.asarray(coszen, dtype=np.float64), 0.001)
    chil = np.clip(np.asarray(xl, dtype=np.float64), -0.4, 0.6)
    chil = np.where(np.abs(chil) <= 0.01, 0.01, chil)
    phi1 = 0.5 - 0.633 * chil - 0.330 * chil * chil
    phi2 = 0.877 * (1.0 - 2.0 * phi1)
    phi2 = np.where(np.abs(phi2) > eps, phi2, eps)
    gdir = phi1 + phi2 * cosz
    twostext = gdir / cosz
    avmu = (1.0 - (phi1 / phi2) * np.log(np.maximum((phi1 + phi2) / np.maximum(phi1, eps), eps))) / phi2
    temp0 = gdir + phi2 * cosz
    temp1 = phi1 * cosz
    temp2 = 1.0 - temp1 / np.maximum(temp0, eps) * np.log(np.maximum((temp1 + temp0) / np.maximum(temp1, eps), eps))

    rho = np.maximum(np.asarray(rho, dtype=np.float64), eps)
    tau = np.maximum(np.asarray(tau, dtype=np.float64), eps)
    albgrd = np.clip(np.asarray(albgrd, dtype=np.float64), eps, 0.999999)
    albgri = np.clip(np.asarray(albgri, dtype=np.float64), eps, 0.999999)
    omegal = np.clip(rho + tau, eps, 0.999999)
    asu = 0.5 * omegal * gdir / np.maximum(temp0, eps) * temp2
    betad = (1.0 + avmu * twostext) / np.maximum(omegal * avmu * twostext, eps) * asu
    betai = 0.5 * ((rho + tau) + (rho - tau) * ((1.0 + chil) / 2.0) ** 2) / omegal

    omega = omegal
    b = 1.0 - omega + omega * betai
    c1 = omega * betai
    tmp0 = avmu * twostext
    d = tmp0 * omega * betad
    f = tmp0 * omega * (1.0 - betad)
    tmp1 = np.maximum(b * b - c1 * c1, eps)
    h = np.sqrt(tmp1) / np.maximum(avmu, eps)
    sigma = tmp0 * tmp0 - tmp1
    sigma = np.where(np.abs(sigma) > eps, sigma, np.where(sigma < 0.0, -eps, eps))
    p1 = b + avmu * h
    p2 = b - avmu * h
    p3 = b + tmp0
    p4 = b - tmp0

    s1 = np.exp(-np.minimum(h * laisum, 40.0))
    s2 = np.exp(-np.minimum(twostext * laisum, 40.0))

    u1 = b - c1 / albgrd
    u2 = b - c1 * albgrd
    u3 = f + c1 * albgrd
    tmp2 = u1 - avmu * h
    tmp3 = u1 + avmu * h
    d1 = p1 * tmp2 / s1 - p2 * tmp3 * s1
    tmp4 = u2 + avmu * h
    tmp5 = u2 - avmu * h
    d2 = tmp4 / s1 - tmp5 * s1
    h1 = -d * p4 - c1 * f
    tmp6 = d - h1 * p3 / sigma
    tmp7 = (d - c1 - h1 / sigma * (u1 + tmp0)) * s2
    h2 = _np_safe_divide(tmp6 * tmp2 / s1 - p2 * tmp7, d1)
    h3 = -_np_safe_divide(tmp6 * tmp3 * s1 - p1 * tmp7, d1)
    h4 = -f * p3 - c1 * d
    tmp8 = h4 / sigma
    tmp9 = (u3 - tmp8 * (u2 - tmp0)) * s2
    h5 = -_np_safe_divide(tmp8 * tmp4 / s1 + tmp9, d2)
    h6 = _np_safe_divide(tmp8 * tmp5 * s1 + tmp9, d2)
    ftdd = s2
    ftid = h4 * s2 / sigma + h5 * s1 + h6 / s1

    u1 = b - c1 / albgri
    u2 = b - c1 * albgri
    tmp2 = u1 - avmu * h
    tmp3 = u1 + avmu * h
    d1 = p1 * tmp2 / s1 - p2 * tmp3 * s1
    tmp4 = u2 + avmu * h
    tmp5 = u2 - avmu * h
    d2 = tmp4 / s1 - tmp5 * s1
    h9 = _np_safe_divide(tmp4, d2 * s1)
    h10 = _np_safe_divide(-tmp5 * s1, d2)
    ftii = h9 * s1 + h10 / s1

    canopy_mask = (laisum > eps) & (np.asarray(coszen, dtype=np.float64) > 0.0)
    ftdd = np.where(canopy_mask, ftdd, 1.0)
    ftid = np.where(canopy_mask, ftid, 0.0)
    ftii = np.where(canopy_mask, ftii, 1.0)
    return (
        np.where(np.isfinite(ftdd), ftdd, 1.0),
        np.where(np.isfinite(ftid), ftid, 0.0),
        np.where(np.isfinite(ftii), ftii, 1.0),
    )


def _elm_twostream_transmission_torch(
    *,
    rho: torch.Tensor,
    tau: torch.Tensor,
    albgrd: torch.Tensor,
    albgri: torch.Tensor,
    elai: torch.Tensor,
    esai: torch.Tensor,
    coszen: torch.Tensor,
    xl: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Torch equivalent of `_elm_twostream_transmission`."""
    eps = torch.as_tensor(1.0e-8, device=elai.device, dtype=elai.dtype)
    laisum = torch.clamp(elai, min=0.0) + torch.clamp(esai, min=0.0)
    coszen_raw = coszen
    cosz = torch.clamp(coszen_raw, min=0.001)
    chil = torch.clamp(xl, min=-0.4, max=0.6)
    chil = torch.where(torch.abs(chil) <= 0.01, torch.full_like(chil, 0.01), chil)
    phi1 = 0.5 - 0.633 * chil - 0.330 * chil * chil
    phi2 = 0.877 * (1.0 - 2.0 * phi1)
    phi2 = torch.where(torch.abs(phi2) > eps, phi2, eps)
    gdir = phi1 + phi2 * cosz
    twostext = gdir / cosz
    avmu = (1.0 - (phi1 / phi2) * torch.log(torch.clamp((phi1 + phi2) / torch.clamp(phi1, min=eps), min=eps))) / phi2
    temp0 = gdir + phi2 * cosz
    temp1 = phi1 * cosz
    temp2 = 1.0 - temp1 / torch.clamp(temp0, min=eps) * torch.log(torch.clamp((temp1 + temp0) / torch.clamp(temp1, min=eps), min=eps))

    rho = torch.clamp(rho, min=eps)
    tau = torch.clamp(tau, min=eps)
    albgrd = torch.clamp(albgrd, min=eps, max=0.999999)
    albgri = torch.clamp(albgri, min=eps, max=0.999999)
    omegal = torch.clamp(rho + tau, min=eps, max=0.999999)
    asu = 0.5 * omegal * gdir / torch.clamp(temp0, min=eps) * temp2
    betad = (1.0 + avmu * twostext) / torch.clamp(omegal * avmu * twostext, min=eps) * asu
    betai = 0.5 * ((rho + tau) + (rho - tau) * ((1.0 + chil) / 2.0) ** 2) / omegal

    omega = omegal
    b = 1.0 - omega + omega * betai
    c1 = omega * betai
    tmp0 = avmu * twostext
    d = tmp0 * omega * betad
    f = tmp0 * omega * (1.0 - betad)
    tmp1 = torch.clamp(b * b - c1 * c1, min=eps)
    h = torch.sqrt(tmp1) / torch.clamp(avmu, min=eps)
    sigma = tmp0 * tmp0 - tmp1
    sigma = torch.where(
        torch.abs(sigma) > eps,
        sigma,
        torch.where(sigma < 0.0, -eps, eps),
    )
    p1 = b + avmu * h
    p2 = b - avmu * h
    p3 = b + tmp0
    p4 = b - tmp0

    s1 = torch.exp(-torch.clamp(h * laisum, max=40.0))
    s2 = torch.exp(-torch.clamp(twostext * laisum, max=40.0))

    u1 = b - c1 / albgrd
    u2 = b - c1 * albgrd
    u3 = f + c1 * albgrd
    tmp2 = u1 - avmu * h
    tmp3 = u1 + avmu * h
    d1 = p1 * tmp2 / s1 - p2 * tmp3 * s1
    tmp4 = u2 + avmu * h
    tmp5 = u2 - avmu * h
    d2 = tmp4 / s1 - tmp5 * s1
    h1 = -d * p4 - c1 * f
    tmp6 = d - h1 * p3 / sigma
    tmp7 = (d - c1 - h1 / sigma * (u1 + tmp0)) * s2
    h2 = _torch_safe_divide(tmp6 * tmp2 / s1 - p2 * tmp7, d1)
    h3 = -_torch_safe_divide(tmp6 * tmp3 * s1 - p1 * tmp7, d1)
    h4 = -f * p3 - c1 * d
    tmp8 = h4 / sigma
    tmp9 = (u3 - tmp8 * (u2 - tmp0)) * s2
    h5 = -_torch_safe_divide(tmp8 * tmp4 / s1 + tmp9, d2)
    h6 = _torch_safe_divide(tmp8 * tmp5 * s1 + tmp9, d2)
    ftdd = s2
    ftid = h4 * s2 / sigma + h5 * s1 + h6 / s1

    u1 = b - c1 / albgri
    u2 = b - c1 * albgri
    tmp2 = u1 - avmu * h
    tmp3 = u1 + avmu * h
    d1 = p1 * tmp2 / s1 - p2 * tmp3 * s1
    tmp4 = u2 + avmu * h
    tmp5 = u2 - avmu * h
    d2 = tmp4 / s1 - tmp5 * s1
    h9 = _torch_safe_divide(tmp4, d2 * s1)
    h10 = _torch_safe_divide(-tmp5 * s1, d2)
    ftii = h9 * s1 + h10 / s1

    canopy_mask = (laisum > eps) & (coszen_raw > 0.0)
    ftdd = torch.where(canopy_mask, ftdd, torch.ones_like(ftdd))
    ftid = torch.where(canopy_mask, ftid, torch.zeros_like(ftid))
    ftii = torch.where(canopy_mask, ftii, torch.ones_like(ftii))
    ftdd = torch.where(torch.isfinite(ftdd), ftdd, torch.ones_like(ftdd))
    ftid = torch.where(torch.isfinite(ftid), ftid, torch.zeros_like(ftid))
    ftii = torch.where(torch.isfinite(ftii), ftii, torch.ones_like(ftii))
    return ftdd, ftid, ftii


def surface_shortwave_diagnostics_torch(
    forc_swrad: np.ndarray | torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
    elai: np.ndarray | torch.Tensor,
    esai: np.ndarray | torch.Tensor,
    atmosphere: "AtmosphericForcingInputs",
    soil_color: np.ndarray | torch.Tensor | float | None,
    h2osoi_vol_top: np.ndarray | torch.Tensor | float | None,
    frac_sno: np.ndarray | torch.Tensor | None = None,
    snow_albedo: tuple[torch.Tensor, torch.Tensor] | None = None,
    rholvis: np.ndarray | torch.Tensor | float,
    rholnir: np.ndarray | torch.Tensor | float,
    taulvis: np.ndarray | torch.Tensor | float,
    taulnir: np.ndarray | torch.Tensor | float,
    rhosvis: np.ndarray | torch.Tensor | float,
    rhosnir: np.ndarray | torch.Tensor | float,
    tausvis: np.ndarray | torch.Tensor | float,
    tausnir: np.ndarray | torch.Tensor | float,
    xl: np.ndarray | torch.Tensor | float,
) -> dict[str, torch.Tensor]:
    """Return ELM-style VIS/NIR direct/diffuse shortwave diagnostics.

    The reflected flux fields follow native history naming: ``FSRVD`` is
    direct visible reflected shortwave, ``FSRVI`` diffuse visible, ``FSRND``
    direct NIR, and ``FSRNI`` diffuse NIR. ``ALBD``/``ALBI`` are the surface
    direct/diffuse albedos with the last dimension ordered VIS, NIR.
    """
    forc_swrad_t = torch.clamp(_to_torch(forc_swrad, device, dtype), min=0.0)
    soil_albedo = _elm_soil_albedo_from_color_torch(
        soil_color,
        h2osoi_vol_top,
        device=device,
        dtype=dtype,
    )
    if soil_albedo is None:
        alb_ground_vis = torch.full_like(forc_swrad_t, 0.15)
        alb_ground_nir = torch.full_like(forc_swrad_t, 0.30)
    else:
        alb_soil_vis, alb_soil_nir = soil_albedo
        if frac_sno is not None and snow_albedo is not None:
            snow_vis, snow_nir = snow_albedo
            frac = torch.clamp(_to_torch(frac_sno, device, dtype), min=0.0, max=1.0)
            alb_ground_vis = alb_soil_vis * (1.0 - frac) + snow_vis * frac
            alb_ground_nir = alb_soil_nir * (1.0 - frac) + snow_nir * frac
        else:
            alb_ground_vis = alb_soil_vis
            alb_ground_nir = alb_soil_nir

    elai_t = _to_torch(elai, device, dtype)
    esai_t = _to_torch(esai, device, dtype)
    coszen = _to_torch(atmosphere.coszen_col, device, dtype)
    xl_t = _to_torch(xl, device, dtype)

    def _band_albedo_transmission(
        rho: torch.Tensor,
        tau: torch.Tensor,
        albgrd: torch.Tensor,
        albgri: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        eps = torch.as_tensor(1.0e-8, device=device, dtype=dtype)
        laisum = torch.clamp(elai_t, min=0.0) + torch.clamp(esai_t, min=0.0)
        coszen_raw = coszen
        cosz = torch.clamp(coszen_raw, min=0.001)
        chil = torch.clamp(xl_t, min=-0.4, max=0.6)
        chil = torch.where(torch.abs(chil) <= 0.01, torch.full_like(chil, 0.01), chil)
        phi1 = 0.5 - 0.633 * chil - 0.330 * chil * chil
        phi2 = 0.877 * (1.0 - 2.0 * phi1)
        phi2 = torch.where(torch.abs(phi2) > eps, phi2, eps)
        gdir = phi1 + phi2 * cosz
        twostext = gdir / cosz
        avmu = (1.0 - (phi1 / phi2) * torch.log(torch.clamp((phi1 + phi2) / torch.clamp(phi1, min=eps), min=eps))) / phi2
        temp0 = gdir + phi2 * cosz
        temp1 = phi1 * cosz
        temp2 = 1.0 - temp1 / torch.clamp(temp0, min=eps) * torch.log(torch.clamp((temp1 + temp0) / torch.clamp(temp1, min=eps), min=eps))

        rho = torch.clamp(rho, min=eps)
        tau = torch.clamp(tau, min=eps)
        albgrd = torch.clamp(albgrd, min=eps, max=0.999999)
        albgri = torch.clamp(albgri, min=eps, max=0.999999)
        omegal = torch.clamp(rho + tau, min=eps, max=0.999999)
        asu = 0.5 * omegal * gdir / torch.clamp(temp0, min=eps) * temp2
        betad = (1.0 + avmu * twostext) / torch.clamp(omegal * avmu * twostext, min=eps) * asu
        betai = 0.5 * ((rho + tau) + (rho - tau) * ((1.0 + chil) / 2.0) ** 2) / omegal

        b = 1.0 - omegal + omegal * betai
        c1 = omegal * betai
        tmp0 = avmu * twostext
        d = tmp0 * omegal * betad
        f = tmp0 * omegal * (1.0 - betad)
        tmp1 = torch.clamp(b * b - c1 * c1, min=eps)
        h = torch.sqrt(tmp1) / torch.clamp(avmu, min=eps)
        sigma = tmp0 * tmp0 - tmp1
        sigma = torch.where(
            torch.abs(sigma) > eps,
            sigma,
            torch.where(sigma < 0.0, -eps, eps),
        )
        p1 = b + avmu * h
        p2 = b - avmu * h
        p3 = b + tmp0
        p4 = b - tmp0
        s1 = torch.exp(-torch.clamp(h * laisum, max=40.0))
        s2 = torch.exp(-torch.clamp(twostext * laisum, max=40.0))

        u1 = b - c1 / albgrd
        u2 = b - c1 * albgrd
        u3 = f + c1 * albgrd
        tmp2 = u1 - avmu * h
        tmp3 = u1 + avmu * h
        d1 = p1 * tmp2 / s1 - p2 * tmp3 * s1
        tmp4 = u2 + avmu * h
        tmp5 = u2 - avmu * h
        d2 = tmp4 / s1 - tmp5 * s1
        h1 = -d * p4 - c1 * f
        tmp6 = d - h1 * p3 / sigma
        tmp7 = (d - c1 - h1 / sigma * (u1 + tmp0)) * s2
        h2 = _torch_safe_divide(tmp6 * tmp2 / s1 - p2 * tmp7, d1)
        h3 = -_torch_safe_divide(tmp6 * tmp3 * s1 - p1 * tmp7, d1)
        h4 = -f * p3 - c1 * d
        tmp8 = h4 / sigma
        tmp9 = (u3 - tmp8 * (u2 - tmp0)) * s2
        h5 = -_torch_safe_divide(tmp8 * tmp4 / s1 + tmp9, d2)
        h6 = _torch_safe_divide(tmp8 * tmp5 * s1 + tmp9, d2)
        albd = h1 / sigma + h2 + h3
        ftid = h4 * s2 / sigma + h5 * s1 + h6 / s1
        ftdd = s2

        u1 = b - c1 / albgri
        u2 = b - c1 * albgri
        tmp2 = u1 - avmu * h
        tmp3 = u1 + avmu * h
        d1 = p1 * tmp2 / s1 - p2 * tmp3 * s1
        tmp4 = u2 + avmu * h
        tmp5 = u2 - avmu * h
        d2 = tmp4 / s1 - tmp5 * s1
        h7 = _torch_safe_divide(c1 * tmp2, d1 * s1)
        h8 = _torch_safe_divide(-c1 * tmp3 * s1, d1)
        h9 = _torch_safe_divide(tmp4, d2 * s1)
        h10 = _torch_safe_divide(-tmp5 * s1, d2)
        albi = h7 + h8
        ftii = h9 * s1 + h10 / s1

        canopy_mask = (laisum > eps) & (coszen_raw > 0.0)
        albd = torch.where(canopy_mask, albd, albgrd)
        albi = torch.where(canopy_mask, albi, albgri)
        ftdd = torch.where(canopy_mask, ftdd, torch.ones_like(ftdd))
        ftid = torch.where(canopy_mask, ftid, torch.zeros_like(ftid))
        ftii = torch.where(canopy_mask, ftii, torch.ones_like(ftii))
        albd = torch.where(torch.isfinite(albd), albd, albgrd)
        albi = torch.where(torch.isfinite(albi), albi, albgri)
        ftdd = torch.where(torch.isfinite(ftdd), ftdd, torch.ones_like(ftdd))
        ftid = torch.where(torch.isfinite(ftid), ftid, torch.zeros_like(ftid))
        ftii = torch.where(torch.isfinite(ftii), ftii, torch.ones_like(ftii))
        return (
            torch.clamp(albd, min=0.0, max=1.0),
            torch.clamp(albi, min=0.0, max=1.0),
            ftdd,
            ftid,
            ftii,
        )

    rho_vis, tau_vis = _weighted_canopy_optics_torch(
        elai=elai_t,
        esai=esai_t,
        rhol=_to_torch(rholvis, device, dtype),
        rhos=_to_torch(rhosvis, device, dtype),
        taul=_to_torch(taulvis, device, dtype),
        taus=_to_torch(tausvis, device, dtype),
    )
    rho_nir, tau_nir = _weighted_canopy_optics_torch(
        elai=elai_t,
        esai=esai_t,
        rhol=_to_torch(rholnir, device, dtype),
        rhos=_to_torch(rhosnir, device, dtype),
        taul=_to_torch(taulnir, device, dtype),
        taus=_to_torch(tausnir, device, dtype),
    )
    albd_vis, albi_vis, ftdd_vis, ftid_vis, ftii_vis = _band_albedo_transmission(
        rho_vis,
        tau_vis,
        alb_ground_vis,
        alb_ground_vis,
    )
    albd_nir, albi_nir, ftdd_nir, ftid_nir, ftii_nir = _band_albedo_transmission(
        rho_nir,
        tau_nir,
        alb_ground_nir,
        alb_ground_nir,
    )

    fsdsvd = torch.clamp(_to_torch(atmosphere.forc_solad_vis, device, dtype), min=0.0)
    fsdsnd = torch.clamp(_to_torch(atmosphere.forc_solad_nir, device, dtype), min=0.0)
    fsdsvi = torch.clamp(_to_torch(atmosphere.forc_solai_vis, device, dtype), min=0.0)
    fsdsni = torch.clamp(_to_torch(atmosphere.forc_solai_nir, device, dtype), min=0.0)
    fsrvd = fsdsvd * albd_vis
    fsrnd = fsdsnd * albd_nir
    fsrvi = fsdsvi * albi_vis
    fsrni = fsdsni * albi_nir
    fsr = fsrvd + fsrnd + fsrvi + fsrni
    fsds_total = fsdsvd + fsdsnd + fsdsvi + fsdsni
    fsa = torch.clamp(fsds_total - fsr, min=0.0)
    fsa_out = torch.minimum(fsa, forc_swrad_t)
    trd_vis = fsdsvd * ftdd_vis
    tri_vis = fsdsvd * ftid_vis + fsdsvi * ftii_vis
    trd_nir = fsdsnd * ftdd_nir
    tri_nir = fsdsnd * ftid_nir + fsdsni * ftii_nir
    sabg = (
        (trd_vis + tri_vis) * (1.0 - alb_ground_vis)
        + (trd_nir + tri_nir) * (1.0 - alb_ground_nir)
    )
    sabg = torch.minimum(torch.clamp(sabg, min=0.0), fsa_out)
    sabv = torch.clamp(fsa_out - sabg, min=0.0)

    return {
        "albgrd_mean": torch.stack((alb_ground_vis, alb_ground_nir), dim=-1),
        "albgri_mean": torch.stack((alb_ground_vis, alb_ground_nir), dim=-1),
        "albd_mean": torch.stack((albd_vis, albd_nir), dim=-1),
        "albi_mean": torch.stack((albi_vis, albi_nir), dim=-1),
        "fsdsvd_mean": fsdsvd,
        "fsdsnd_mean": fsdsnd,
        "fsdsvi_mean": fsdsvi,
        "fsdsni_mean": fsdsni,
        "fsrvd_mean": fsrvd,
        "fsrnd_mean": fsrnd,
        "fsrvi_mean": fsrvi,
        "fsrni_mean": fsrni,
        "fsr_mean": torch.minimum(fsr, forc_swrad_t),
        "fsa_mean": fsa_out,
        "sabg_mean": sabg,
        "sabv_mean": sabv,
    }


def _weighted_canopy_optics(
    *,
    elai: np.ndarray,
    esai: np.ndarray,
    rhol: np.ndarray,
    rhos: np.ndarray,
    taul: np.ndarray,
    taus: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    laisum = np.maximum(elai, 0.0) + np.maximum(esai, 0.0)
    wl = np.maximum(elai, 0.0) / np.maximum(laisum, 1.0e-8)
    ws = np.maximum(esai, 0.0) / np.maximum(laisum, 1.0e-8)
    return (
        np.maximum(rhol * wl + rhos * ws, 1.0e-8),
        np.maximum(taul * wl + taus * ws, 1.0e-8),
    )


def _weighted_canopy_optics_torch(
    *,
    elai: torch.Tensor,
    esai: torch.Tensor,
    rhol: torch.Tensor,
    rhos: torch.Tensor,
    taul: torch.Tensor,
    taus: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    laisum = torch.clamp(elai, min=0.0) + torch.clamp(esai, min=0.0)
    wl = torch.clamp(elai, min=0.0) / torch.clamp(laisum, min=1.0e-8)
    ws = torch.clamp(esai, min=0.0) / torch.clamp(laisum, min=1.0e-8)
    return (
        torch.clamp(rhol * wl + rhos * ws, min=1.0e-8),
        torch.clamp(taul * wl + taus * ws, min=1.0e-8),
    )


def _absorbed_ground_shortwave_native_like(
    forc_swrad: np.ndarray,
    *,
    elai: np.ndarray,
    esai: np.ndarray,
    atmosphere: "AtmosphericForcingInputs",
    soil_albedo: tuple[np.ndarray, np.ndarray],
    frac_sno: np.ndarray | None = None,
    snow_albedo: tuple[np.ndarray, np.ndarray] | None = None,
    target_albedo: tuple[np.ndarray, np.ndarray] | None = None,
    rholvis: np.ndarray,
    rholnir: np.ndarray,
    taulvis: np.ndarray,
    taulnir: np.ndarray,
    rhosvis: np.ndarray,
    rhosnir: np.ndarray,
    tausvis: np.ndarray,
    tausnir: np.ndarray,
    xl: np.ndarray,
) -> np.ndarray:
    alb_soil_vis, alb_soil_nir = soil_albedo
    if frac_sno is not None and snow_albedo is not None:
        snow_vis, snow_nir = snow_albedo
        frac = np.clip(np.asarray(frac_sno, dtype=np.float64), 0.0, 1.0)
        alb_ground_vis = alb_soil_vis * (1.0 - frac) + snow_vis * frac
        alb_ground_nir = alb_soil_nir * (1.0 - frac) + snow_nir * frac
    else:
        alb_ground_vis = alb_soil_vis
        alb_ground_nir = alb_soil_nir
    target_vis, target_nir = target_albedo if target_albedo is not None else (alb_ground_vis, alb_ground_nir)

    elai_np = np.asarray(elai, dtype=np.float64)
    esai_np = np.asarray(esai, dtype=np.float64)
    rho_vis, tau_vis = _weighted_canopy_optics(
        elai=elai_np,
        esai=esai_np,
        rhol=np.asarray(rholvis, dtype=np.float64),
        rhos=np.asarray(rhosvis, dtype=np.float64),
        taul=np.asarray(taulvis, dtype=np.float64),
        taus=np.asarray(tausvis, dtype=np.float64),
    )
    rho_nir, tau_nir = _weighted_canopy_optics(
        elai=elai_np,
        esai=esai_np,
        rhol=np.asarray(rholnir, dtype=np.float64),
        rhos=np.asarray(rhosnir, dtype=np.float64),
        taul=np.asarray(taulnir, dtype=np.float64),
        taus=np.asarray(tausnir, dtype=np.float64),
    )
    ftdd_vis, ftid_vis, ftii_vis = _elm_twostream_transmission(
        rho=rho_vis,
        tau=tau_vis,
        albgrd=alb_ground_vis,
        albgri=alb_ground_vis,
        elai=elai_np,
        esai=esai_np,
        coszen=np.asarray(atmosphere.coszen_col, dtype=np.float64),
        xl=np.asarray(xl, dtype=np.float64),
    )
    ftdd_nir, ftid_nir, ftii_nir = _elm_twostream_transmission(
        rho=rho_nir,
        tau=tau_nir,
        albgrd=alb_ground_nir,
        albgri=alb_ground_nir,
        elai=elai_np,
        esai=esai_np,
        coszen=np.asarray(atmosphere.coszen_col, dtype=np.float64),
        xl=np.asarray(xl, dtype=np.float64),
    )
    solad_vis = np.asarray(atmosphere.forc_solad_vis, dtype=np.float64)
    solad_nir = np.asarray(atmosphere.forc_solad_nir, dtype=np.float64)
    solai_vis = np.asarray(atmosphere.forc_solai_vis, dtype=np.float64)
    solai_nir = np.asarray(atmosphere.forc_solai_nir, dtype=np.float64)
    trd_vis = solad_vis * ftdd_vis
    tri_vis = solad_vis * ftid_vis + solai_vis * ftii_vis
    trd_nir = solad_nir * ftdd_nir
    tri_nir = solad_nir * ftid_nir + solai_nir * ftii_nir
    absorbed = trd_vis * (1.0 - target_vis) + tri_vis * (1.0 - target_vis)
    absorbed = absorbed + trd_nir * (1.0 - target_nir) + tri_nir * (1.0 - target_nir)
    return np.clip(np.where(np.isfinite(absorbed), absorbed, 0.0), 0.0, np.maximum(np.asarray(forc_swrad, dtype=np.float64), 0.0))


def _absorbed_ground_shortwave_native_like_torch(
    forc_swrad: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
    elai: np.ndarray | torch.Tensor,
    esai: np.ndarray | torch.Tensor,
    atmosphere: "AtmosphericForcingInputs",
    soil_albedo: tuple[torch.Tensor, torch.Tensor],
    frac_sno: np.ndarray | torch.Tensor | None = None,
    snow_albedo: tuple[torch.Tensor, torch.Tensor] | None = None,
    target_albedo: tuple[torch.Tensor, torch.Tensor] | None = None,
    rholvis: np.ndarray | torch.Tensor | float,
    rholnir: np.ndarray | torch.Tensor | float,
    taulvis: np.ndarray | torch.Tensor | float,
    taulnir: np.ndarray | torch.Tensor | float,
    rhosvis: np.ndarray | torch.Tensor | float,
    rhosnir: np.ndarray | torch.Tensor | float,
    tausvis: np.ndarray | torch.Tensor | float,
    tausnir: np.ndarray | torch.Tensor | float,
    xl: np.ndarray | torch.Tensor | float,
) -> torch.Tensor:
    alb_soil_vis, alb_soil_nir = soil_albedo
    if frac_sno is not None and snow_albedo is not None:
        snow_vis, snow_nir = snow_albedo
        frac = torch.clamp(_to_torch(frac_sno, device, dtype), min=0.0, max=1.0)
        alb_ground_vis = alb_soil_vis * (1.0 - frac) + snow_vis * frac
        alb_ground_nir = alb_soil_nir * (1.0 - frac) + snow_nir * frac
    else:
        alb_ground_vis = alb_soil_vis
        alb_ground_nir = alb_soil_nir
    target_vis, target_nir = target_albedo if target_albedo is not None else (alb_ground_vis, alb_ground_nir)

    elai_t = _to_torch(elai, device, dtype)
    esai_t = _to_torch(esai, device, dtype)
    rho_vis, tau_vis = _weighted_canopy_optics_torch(
        elai=elai_t,
        esai=esai_t,
        rhol=_to_torch(rholvis, device, dtype),
        rhos=_to_torch(rhosvis, device, dtype),
        taul=_to_torch(taulvis, device, dtype),
        taus=_to_torch(tausvis, device, dtype),
    )
    rho_nir, tau_nir = _weighted_canopy_optics_torch(
        elai=elai_t,
        esai=esai_t,
        rhol=_to_torch(rholnir, device, dtype),
        rhos=_to_torch(rhosnir, device, dtype),
        taul=_to_torch(taulnir, device, dtype),
        taus=_to_torch(tausnir, device, dtype),
    )
    coszen = _to_torch(atmosphere.coszen_col, device, dtype)
    xl_t = _to_torch(xl, device, dtype)
    ftdd_vis, ftid_vis, ftii_vis = _elm_twostream_transmission_torch(
        rho=rho_vis,
        tau=tau_vis,
        albgrd=alb_ground_vis,
        albgri=alb_ground_vis,
        elai=elai_t,
        esai=esai_t,
        coszen=coszen,
        xl=xl_t,
    )
    ftdd_nir, ftid_nir, ftii_nir = _elm_twostream_transmission_torch(
        rho=rho_nir,
        tau=tau_nir,
        albgrd=alb_ground_nir,
        albgri=alb_ground_nir,
        elai=elai_t,
        esai=esai_t,
        coszen=coszen,
        xl=xl_t,
    )
    solad_vis = _to_torch(atmosphere.forc_solad_vis, device, dtype)
    solad_nir = _to_torch(atmosphere.forc_solad_nir, device, dtype)
    solai_vis = _to_torch(atmosphere.forc_solai_vis, device, dtype)
    solai_nir = _to_torch(atmosphere.forc_solai_nir, device, dtype)
    trd_vis = solad_vis * ftdd_vis
    tri_vis = solad_vis * ftid_vis + solai_vis * ftii_vis
    trd_nir = solad_nir * ftdd_nir
    tri_nir = solad_nir * ftid_nir + solai_nir * ftii_nir
    absorbed = trd_vis * (1.0 - target_vis) + tri_vis * (1.0 - target_vis)
    absorbed = absorbed + trd_nir * (1.0 - target_nir) + tri_nir * (1.0 - target_nir)
    absorbed = torch.where(torch.isfinite(absorbed), absorbed, torch.zeros_like(absorbed))
    return torch.minimum(torch.clamp(absorbed, min=0.0), torch.clamp(forc_swrad, min=0.0))


def _absorbed_ground_shortwave(
    forc_swrad: np.ndarray,
    *,
    elai: np.ndarray | None = None,
    esai: np.ndarray | None = None,
    atmosphere: "AtmosphericForcingInputs | None" = None,
    soil_color: np.ndarray | float | None = None,
    h2osoi_vol_top: np.ndarray | float | None = None,
    frac_sno: np.ndarray | None = None,
    snow_albedo: tuple[np.ndarray, np.ndarray] | None = None,
    target_albedo: tuple[np.ndarray, np.ndarray] | None = None,
    rholvis: np.ndarray | float | None = None,
    rholnir: np.ndarray | float | None = None,
    taulvis: np.ndarray | float | None = None,
    taulnir: np.ndarray | float | None = None,
    rhosvis: np.ndarray | float | None = None,
    rhosnir: np.ndarray | float | None = None,
    tausvis: np.ndarray | float | None = None,
    tausnir: np.ndarray | float | None = None,
    xl: np.ndarray | float | None = None,
) -> np.ndarray:
    """Approximate no-snow ``sabg`` used by ELM for ``sabg_snow`` when ``snl == 0``."""
    forc_swrad = np.asarray(forc_swrad, dtype=np.float64)
    soil_albedo = _elm_soil_albedo_from_color(soil_color, h2osoi_vol_top)
    native_inputs = (
        atmosphere is not None
        and elai is not None
        and esai is not None
        and soil_albedo is not None
        and all(v is not None for v in (rholvis, rholnir, taulvis, taulnir, rhosvis, rhosnir, tausvis, tausnir, xl))
    )
    if native_inputs:
        return _absorbed_ground_shortwave_native_like(
            forc_swrad,
            elai=np.asarray(elai, dtype=np.float64),
            esai=np.asarray(esai, dtype=np.float64),
            atmosphere=atmosphere,
            soil_albedo=soil_albedo,
            frac_sno=frac_sno,
            snow_albedo=snow_albedo,
            target_albedo=target_albedo,
            rholvis=np.asarray(rholvis, dtype=np.float64),
            rholnir=np.asarray(rholnir, dtype=np.float64),
            taulvis=np.asarray(taulvis, dtype=np.float64),
            taulnir=np.asarray(taulnir, dtype=np.float64),
            rhosvis=np.asarray(rhosvis, dtype=np.float64),
            rhosnir=np.asarray(rhosnir, dtype=np.float64),
            tausvis=np.asarray(tausvis, dtype=np.float64),
            tausnir=np.asarray(tausnir, dtype=np.float64),
            xl=np.asarray(xl, dtype=np.float64),
        )
    if elai is not None:
        trans = np.exp(-0.5 * np.maximum(np.asarray(elai, dtype=np.float64), 0.0))
    else:
        trans = np.ones_like(forc_swrad)
    if atmosphere is not None:
        sw_vis = np.asarray(atmosphere.forc_solad_vis, dtype=np.float64) + np.asarray(atmosphere.forc_solai_vis, dtype=np.float64)
        sw_nir = np.asarray(atmosphere.forc_solad_nir, dtype=np.float64) + np.asarray(atmosphere.forc_solai_nir, dtype=np.float64)
    else:
        sw_vis = 0.53 * forc_swrad
        sw_nir = 0.47 * forc_swrad
    if soil_albedo is None:
        return trans * (0.85 * sw_vis + 0.65 * sw_nir)
    if target_albedo is not None:
        alb_vis, alb_nir = target_albedo
    else:
        alb_vis, alb_nir = soil_albedo
        if snow_albedo is not None and frac_sno is not None:
            frac = np.clip(np.asarray(frac_sno, dtype=np.float64), 0.0, 1.0)
            snow_vis, snow_nir = snow_albedo
            alb_vis = alb_vis * (1.0 - frac) + snow_vis * frac
            alb_nir = alb_nir * (1.0 - frac) + snow_nir * frac
    return trans * (sw_vis * (1.0 - alb_vis) + sw_nir * (1.0 - alb_nir))


def _absorbed_ground_shortwave_torch(
    forc_swrad: np.ndarray | torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
    elai: np.ndarray | torch.Tensor | None = None,
    esai: np.ndarray | torch.Tensor | None = None,
    atmosphere: "AtmosphericForcingInputs | None" = None,
    soil_color: np.ndarray | torch.Tensor | float | None = None,
    h2osoi_vol_top: np.ndarray | torch.Tensor | float | None = None,
    frac_sno: np.ndarray | torch.Tensor | None = None,
    snow_albedo: tuple[torch.Tensor, torch.Tensor] | None = None,
    target_albedo: tuple[torch.Tensor, torch.Tensor] | None = None,
    rholvis: np.ndarray | torch.Tensor | float | None = None,
    rholnir: np.ndarray | torch.Tensor | float | None = None,
    taulvis: np.ndarray | torch.Tensor | float | None = None,
    taulnir: np.ndarray | torch.Tensor | float | None = None,
    rhosvis: np.ndarray | torch.Tensor | float | None = None,
    rhosnir: np.ndarray | torch.Tensor | float | None = None,
    tausvis: np.ndarray | torch.Tensor | float | None = None,
    tausnir: np.ndarray | torch.Tensor | float | None = None,
    xl: np.ndarray | torch.Tensor | float | None = None,
) -> torch.Tensor:
    """Torch equivalent of `_absorbed_ground_shortwave`."""
    forc_swrad_t = _to_torch(forc_swrad, device, dtype)
    soil_albedo = _elm_soil_albedo_from_color_torch(
        soil_color,
        h2osoi_vol_top,
        device=device,
        dtype=dtype,
    )
    native_inputs = (
        atmosphere is not None
        and elai is not None
        and esai is not None
        and soil_albedo is not None
        and all(v is not None for v in (rholvis, rholnir, taulvis, taulnir, rhosvis, rhosnir, tausvis, tausnir, xl))
    )
    if native_inputs:
        return _absorbed_ground_shortwave_native_like_torch(
            forc_swrad_t,
            device=device,
            dtype=dtype,
            elai=elai,
            esai=esai,
            atmosphere=atmosphere,
            soil_albedo=soil_albedo,
            frac_sno=frac_sno,
            snow_albedo=snow_albedo,
            target_albedo=target_albedo,
            rholvis=rholvis,
            rholnir=rholnir,
            taulvis=taulvis,
            taulnir=taulnir,
            rhosvis=rhosvis,
            rhosnir=rhosnir,
            tausvis=tausvis,
            tausnir=tausnir,
            xl=xl,
        )
    if elai is not None:
        trans = torch.exp(-0.5 * torch.clamp(_to_torch(elai, device, dtype), min=0.0))
    else:
        trans = torch.ones_like(forc_swrad_t)
    if atmosphere is not None:
        sw_vis = _to_torch(atmosphere.forc_solad_vis, device, dtype) + _to_torch(atmosphere.forc_solai_vis, device, dtype)
        sw_nir = _to_torch(atmosphere.forc_solad_nir, device, dtype) + _to_torch(atmosphere.forc_solai_nir, device, dtype)
    else:
        sw_vis = 0.53 * forc_swrad_t
        sw_nir = 0.47 * forc_swrad_t
    if soil_albedo is None:
        return trans * (0.85 * sw_vis + 0.65 * sw_nir)
    if target_albedo is not None:
        alb_vis, alb_nir = target_albedo
    else:
        alb_vis, alb_nir = soil_albedo
        if snow_albedo is not None and frac_sno is not None:
            frac = torch.clamp(_to_torch(frac_sno, device, dtype), min=0.0, max=1.0)
            snow_vis, snow_nir = snow_albedo
            alb_vis = alb_vis * (1.0 - frac) + snow_vis * frac
            alb_nir = alb_nir * (1.0 - frac) + snow_nir * frac
    return trans * (sw_vis * (1.0 - alb_vis) + sw_nir * (1.0 - alb_nir))


def h2osno_from_depth_and_frac(
    snow_depth: np.ndarray,
    frac_sno: np.ndarray,
) -> np.ndarray:
    """Use the ELM bulk-density relationship to recover column snow water equivalent.

    ELM's ``snow_depth`` is the depth over the snow-covered part of the grid cell,
    while ``h2osno`` is gridcell/column water equivalent.  The conversion therefore
    includes ``frac_sno``.  Omitting this factor overestimates snow mass by
    approximately ``1 / frac_sno`` during patchy snow onset events.
    """
    snow_depth = np.asarray(snow_depth, dtype=np.float64)
    frac_sno = np.asarray(frac_sno, dtype=np.float64)
    h2osno = np.zeros_like(snow_depth)
    mask = (snow_depth > 0.0) & (frac_sno > 1.0e-4)
    if np.any(mask):
        # Clamp strictly away from the tanh asymptotes to keep arctanh finite.
        frac_clamped = np.clip(frac_sno[mask], 1.0e-6, 1.0 - 1.0e-6)
        # frac = tanh(snow_depth / (2.5 * SNOW_ZLND * fmelt))
        # => fmelt = snow_depth / (2.5 * SNOW_ZLND * arctanh(frac))
        # fmelt = snowbd / 100  =>  snowbd = fmelt * 100
        atanh_frac = np.arctanh(frac_clamped)
        snowbd = np.clip(
            snow_depth[mask] * 100.0 / (2.5 * SNOW_ZLND * np.maximum(atanh_frac, 1.0e-12)),
            100.0,
            400.0,
        )
        h2osno[mask] = snowbd * snow_depth[mask] * frac_sno[mask]
    return h2osno


def sabg_snow_from_state(
    state: CoupledColumnState,
    forc_swrad: np.ndarray,
    frac_sno: np.ndarray,
    *,
    elai: np.ndarray | None = None,
    esai: np.ndarray | None = None,
    atmosphere: "AtmosphericForcingInputs | None" = None,
    soil_color: np.ndarray | float | None = None,
    h2osoi_vol_top: np.ndarray | float | None = None,
    rholvis: np.ndarray | float | None = None,
    rholnir: np.ndarray | float | None = None,
    taulvis: np.ndarray | float | None = None,
    taulnir: np.ndarray | float | None = None,
    rhosvis: np.ndarray | float | None = None,
    rhosnir: np.ndarray | float | None = None,
    tausvis: np.ndarray | float | None = None,
    tausnir: np.ndarray | float | None = None,
    xl: np.ndarray | float | None = None,
) -> np.ndarray:
    """Approximate ELM ``sabg_snow_patch``.

    ELM computes this as the snow-branch absorbed ground radiation when snow
    layers exist.  When no snow layers exist, ELM explicitly resets
    ``sabg_snow`` to ``sabg`` (actual ground absorbed shortwave).  This is not
    area-weighted by ``frac_sno``.

    When ``state.snow_layers.snw_rds`` is available for active snow, the 2-band
    SNICAR parameterisation is used:

    .. math::
        \\text{sabg\\_snow} = T_\\text{can} \\cdot
            \\bigl[ SW_\\text{vis}(1-\\alpha_\\text{vis})
            + SW_\\text{nir}(1-\\alpha_\\text{nir}) \\bigr]

    where :math:`\\alpha` is computed from the top-layer effective grain radius
    via :func:`_snow_albedo_from_grain_radius`.

    If per-band SW components are supplied via *atmosphere*, they are used
    directly; otherwise ``forc_swrad`` is split 53 % VIS / 47 % NIR.

    When ``snw_rds`` is not available for snow-covered points, falls back to a
    temperature-dependent bulk snow albedo (0.85 cold → 0.55 melting).

    When SAI, optical traits, and direct/diffuse VIS/NIR forcing are available,
    sub-canopy transmission follows the native ELM two-stream equations used by
    ``SurfaceAlbedoMod``.  Otherwise the older :math:`\\exp(-0.5 \\cdot LAI)`
    fallback is used.

    Returns
    -------
    np.ndarray
        ``sabg_snow`` shaped like ``forc_swrad`` [W m-2].
    """
    forc_swrad = np.asarray(forc_swrad, dtype=np.float64)
    frac_sno = np.asarray(frac_sno, dtype=np.float64)
    ground_absorbed = _absorbed_ground_shortwave(
        forc_swrad,
        elai=elai,
        esai=esai,
        atmosphere=atmosphere,
        soil_color=soil_color,
        h2osoi_vol_top=h2osoi_vol_top,
        frac_sno=frac_sno,
        rholvis=rholvis,
        rholnir=rholnir,
        taulvis=taulvis,
        taulnir=taulnir,
        rhosvis=rhosvis,
        rhosnir=rhosnir,
        tausvis=tausvis,
        tausnir=tausnir,
        xl=xl,
    )

    # Sub-canopy shortwave (canopy transmission ≈ exp(-0.5 * LAI)).
    if elai is not None:
        elai_np = np.asarray(elai, dtype=np.float64)
        trans = np.exp(-0.5 * np.maximum(elai_np, 0.0))
    else:
        trans = np.ones_like(forc_swrad)

    # --- Grain-radius path ---
    snw_rds_top: np.ndarray | None = None
    has_snow = frac_sno > 1.0e-8
    if state.h2osno is not None:
        h2osno_np = _to_np(state.h2osno)
        has_snow = has_snow & (np.asarray(h2osno_np, dtype=np.float64) > 1.0e-8)
    if state.snow_depth is not None:
        depth_np = _to_np(state.snow_depth)
        has_snow = has_snow & (np.asarray(depth_np, dtype=np.float64) >= SNOW_MIN_ACTIVE_DEPTH)
    if (
        state.snow_layers is not None
        and state.snow_layers.snw_rds is not None
    ):
        rds = state.snow_layers.snw_rds
        if isinstance(rds, torch.Tensor):
            rds = rds.detach().cpu().numpy()
        rds = np.asarray(rds, dtype=np.float64)
        # Top layer is index 0 when there is snow; if no snow keep None.
        snl = state.snow_layers.snl
        if isinstance(snl, torch.Tensor):
            snl = snl.detach().cpu().numpy()
        # Layer bookkeeping can briefly retain inactive/stale snow layers; it
        # should only further gate caller-provided active snow, never resurrect it.
        has_snow = has_snow & (np.asarray(snl, dtype=np.int32) < 0)
        top_rds = np.where(has_snow, rds[..., 0], SNW_RDS_MIN)
        snw_rds_top = top_rds

    if snw_rds_top is not None:
        alpha_vis, alpha_nir = _snow_albedo_from_grain_radius(snw_rds_top)
        snow_absorbed = _absorbed_ground_shortwave(
            forc_swrad,
            elai=elai,
            esai=esai,
            atmosphere=atmosphere,
            soil_color=soil_color,
            h2osoi_vol_top=h2osoi_vol_top,
            frac_sno=frac_sno,
            snow_albedo=(alpha_vis, alpha_nir),
            target_albedo=(alpha_vis, alpha_nir),
            rholvis=rholvis,
            rholnir=rholnir,
            taulvis=taulvis,
            taulnir=taulnir,
            rhosvis=rhosvis,
            rhosnir=rhosnir,
            tausvis=tausvis,
            tausnir=tausnir,
            xl=xl,
        )
        return np.where(has_snow, snow_absorbed, ground_absorbed)

    # --- Fallback: temperature-dependent bulk albedo ---
    t_sno = np.asarray(coupled_t_grnd_from_state(state), dtype=np.float64)
    t_lo, t_hi = TFRZ - 5.0, TFRZ
    alpha = np.clip(0.85 - 0.30 * (t_sno - t_lo) / (t_hi - t_lo), 0.55, 0.85)
    snow_absorbed = forc_swrad * trans * (1.0 - alpha)
    return np.where(has_snow, snow_absorbed, ground_absorbed)


def sabg_snow_from_state_torch(
    state: CoupledColumnState,
    forc_swrad: np.ndarray | torch.Tensor,
    frac_sno: np.ndarray | torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
    elai: np.ndarray | torch.Tensor | None = None,
    esai: np.ndarray | torch.Tensor | None = None,
    atmosphere: "AtmosphericForcingInputs | None" = None,
    soil_color: np.ndarray | torch.Tensor | float | None = None,
    h2osoi_vol_top: np.ndarray | torch.Tensor | float | None = None,
    rholvis: np.ndarray | torch.Tensor | float | None = None,
    rholnir: np.ndarray | torch.Tensor | float | None = None,
    taulvis: np.ndarray | torch.Tensor | float | None = None,
    taulnir: np.ndarray | torch.Tensor | float | None = None,
    rhosvis: np.ndarray | torch.Tensor | float | None = None,
    rhosnir: np.ndarray | torch.Tensor | float | None = None,
    tausvis: np.ndarray | torch.Tensor | float | None = None,
    tausnir: np.ndarray | torch.Tensor | float | None = None,
    xl: np.ndarray | torch.Tensor | float | None = None,
) -> torch.Tensor:
    """Torch-native approximation of ELM ``sabg_snow_patch``."""
    forc_swrad_t = _to_torch(forc_swrad, device, dtype)
    frac_sno_t = _to_torch(frac_sno, device, dtype)
    ground_absorbed = _absorbed_ground_shortwave_torch(
        forc_swrad,
        device=device,
        dtype=dtype,
        elai=elai,
        esai=esai,
        atmosphere=atmosphere,
        soil_color=soil_color,
        h2osoi_vol_top=h2osoi_vol_top,
        frac_sno=frac_sno,
        rholvis=rholvis,
        rholnir=rholnir,
        taulvis=taulvis,
        taulnir=taulnir,
        rhosvis=rhosvis,
        rhosnir=rhosnir,
        tausvis=tausvis,
        tausnir=tausnir,
        xl=xl,
    )
    if elai is not None:
        trans = torch.exp(-0.5 * torch.clamp(_to_torch(elai, device, dtype), min=0.0))
    else:
        trans = torch.ones_like(forc_swrad_t)

    snw_rds_top: torch.Tensor | None = None
    has_snow = frac_sno_t > 1.0e-8
    if state.h2osno is not None:
        has_snow = has_snow & (_to_torch(state.h2osno, device, dtype) > 1.0e-8)
    if state.snow_depth is not None:
        has_snow = has_snow & (_to_torch(state.snow_depth, device, dtype) >= SNOW_MIN_ACTIVE_DEPTH)
    if state.snow_layers is not None and state.snow_layers.snw_rds is not None:
        rds = _to_torch(state.snow_layers.snw_rds, device, dtype)
        snl = _to_torch(state.snow_layers.snl, device, torch.int32)
        # Layer bookkeeping can briefly retain inactive/stale snow layers; it
        # should only further gate caller-provided active snow, never resurrect it.
        has_snow = has_snow & (snl < 0)
        snw_rds_top = torch.where(has_snow, rds[..., 0], torch.full_like(rds[..., 0], SNW_RDS_MIN))

    if snw_rds_top is not None:
        alpha_vis, alpha_nir = _snow_albedo_from_grain_radius_torch(snw_rds_top, device=device, dtype=dtype)
        snow_absorbed = _absorbed_ground_shortwave_torch(
            forc_swrad,
            device=device,
            dtype=dtype,
            elai=elai,
            esai=esai,
            atmosphere=atmosphere,
            soil_color=soil_color,
            h2osoi_vol_top=h2osoi_vol_top,
            frac_sno=frac_sno,
            snow_albedo=(alpha_vis, alpha_nir),
            target_albedo=(alpha_vis, alpha_nir),
            rholvis=rholvis,
            rholnir=rholnir,
            taulvis=taulvis,
            taulnir=taulnir,
            rhosvis=rhosvis,
            rhosnir=rhosnir,
            tausvis=tausvis,
            tausnir=tausnir,
            xl=xl,
        )
        return torch.where(has_snow, snow_absorbed, ground_absorbed)

    t_sno = _to_torch(coupled_t_grnd_from_state_torch(state, device, dtype), device, dtype)
    t_lo = torch.as_tensor(TFRZ - 5.0, device=device, dtype=dtype)
    t_hi = torch.as_tensor(TFRZ, device=device, dtype=dtype)
    alpha = torch.clamp(0.85 - 0.30 * (t_sno - t_lo) / torch.clamp(t_hi - t_lo, min=1.0e-12), 0.55, 0.85)
    snow_absorbed = forc_swrad_t * trans * (1.0 - alpha)
    return torch.where(has_snow, snow_absorbed, ground_absorbed)


def _snow_ice_fraction_from_state(state: CoupledColumnState) -> np.ndarray:
    layers = state.snow_layers
    if layers is not None:
        if isinstance(layers.h2osoi_ice, torch.Tensor):
            ice = layers.h2osoi_ice.detach().cpu().numpy()
            liq = layers.h2osoi_liq.detach().cpu().numpy()
        else:
            ice = np.asarray(layers.h2osoi_ice, dtype=np.float64)
            liq = np.asarray(layers.h2osoi_liq, dtype=np.float64)
        active = np.sum((ice + liq) > 0.0, axis=-1)
        with np.errstate(divide="ignore", invalid="ignore"):
            frac = np.where(active > 0, np.clip(np.sum(ice, axis=-1) / np.maximum(np.sum(ice + liq, axis=-1), 1.0e-12), 0.0, 1.0), 1.0)
        return frac
    snow = snow_storage_from_state(state)
    depth = snow_depth_from_storage(snow)
    frac = np.zeros_like(snow, dtype=np.float64)
    mask = depth > 0.0
    frac[mask] = 1.0
    return frac


def apply_snow_water_and_capping(
    state: CoupledColumnState,
    snow_melt: np.ndarray | torch.Tensor | None,
    *,
    rain_to_snowpack: np.ndarray | torch.Tensor | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> SnowWaterResult:
    """Apply a layered snow water transport and capping step.

    Melt and rain falling on the snow-covered fraction are converted to liquid
    in the snowpack, routed downward through the active snow layers using the
    irreducible saturation rule, and total snow mass is capped at
    ``H2OSNO_MAX`` by removing excess mass from the bottom snow layer. The
    returned ``qflx_snow_to_soil`` is the liquid mass exiting the bottom of
    the snowpack during this step.
    """
    if isinstance(snow_melt, torch.Tensor) or isinstance(rain_to_snowpack, torch.Tensor) or any(
        isinstance(v, torch.Tensor) for v in (state.h2osno, state.snow_depth, state.snow_t, getattr(state.snow_layers, "snl", None)) if v is not None
    ):
        if device is None:
            if isinstance(snow_melt, torch.Tensor):
                device = snow_melt.device
            elif isinstance(rain_to_snowpack, torch.Tensor):
                device = rain_to_snowpack.device
            else:
                device = torch.device("cpu")
        base_snow = _to_torch(snow_storage_from_state(state), device, dtype)
        melt = torch.zeros_like(base_snow) if snow_melt is None else torch.clamp(_to_torch(snow_melt, device, dtype), min=0.0)
        rain_mass = (
            torch.zeros_like(base_snow)
            if rain_to_snowpack is None
            else torch.clamp(_to_torch(rain_to_snowpack, device, dtype), min=0.0)
        )
        snow_shape = tuple(base_snow.shape)
        melt = torch.clamp(_snow_batch_like_torch(melt, snow_shape, device=device, dtype=dtype), min=0.0)
        rain_mass = torch.clamp(_snow_batch_like_torch(rain_mass, snow_shape, device=device, dtype=dtype), min=0.0)
        snow_layers = _snow_layers_to_device(state.snow_layers, device, dtype)
        if snow_layers is None:
            snow_depth0 = snow_depth_from_storage_torch(base_snow, device=device, dtype=dtype)
            snow_layers = _snow_layers_from_storage_torch(
                base_snow,
                snow_depth0,
                state.snow_t,
                frac_sno=state.frac_sno,
                device=device,
                dtype=dtype,
            )
        ice = torch.as_tensor(snow_layers.h2osoi_ice, device=device, dtype=dtype).clone()
        liq = torch.as_tensor(snow_layers.h2osoi_liq, device=device, dtype=dtype).clone()
        dz = torch.as_tensor(snow_layers.dz, device=device, dtype=dtype).clone()
        snl = torch.as_tensor(snow_layers.snl, device=device, dtype=torch.int32)
        t_snow = torch.as_tensor(snow_layers.t_snow, device=device, dtype=dtype)
        frac_sno_eff = torch.clamp(
            _snow_batch_like_torch(state.frac_sno, tuple(snl.shape), device=device, dtype=dtype),
            min=1.0e-12,
            max=1.0,
        )
        nsnow = torch.clamp(-snl.to(torch.int32), min=0)
        inactive_scalar = (nsnow == 0) & (base_snow > 1.0e-8) & (torch.sum(torch.clamp(ice + liq, min=0.0), dim=-1) <= 1.0e-8)
        if torch.any(inactive_scalar):
            ice[..., 0] = torch.where(inactive_scalar, base_snow, ice[..., 0])

        q_snow_to_soil = torch.zeros_like(base_snow)
        scalar_snow = (nsnow == 0) & (base_snow > 1.0e-8)
        if torch.any(scalar_snow) and torch.any(rain_mass > 0.0):
            scalar_rain = torch.where(scalar_snow, rain_mass, torch.zeros_like(rain_mass))
            q_snow_to_soil = q_snow_to_soil + scalar_rain
            rain_mass = torch.where(scalar_snow, torch.zeros_like(rain_mass), rain_mass)
        if torch.any(scalar_snow) and torch.any(melt > 0.0):
            scalar_available = torch.clamp(ice[..., 0] + liq[..., 0], min=0.0)
            scalar_melt = torch.where(
                scalar_snow,
                torch.minimum(melt, scalar_available),
                torch.zeros_like(melt),
            )
            ice_m = torch.minimum(scalar_melt, torch.clamp(ice[..., 0], min=0.0))
            ice[..., 0] = torch.clamp(ice[..., 0] - ice_m, min=0.0)
            liq_m = torch.minimum(scalar_melt - ice_m, torch.clamp(liq[..., 0], min=0.0))
            liq[..., 0] = torch.clamp(liq[..., 0] - liq_m, min=0.0)
            q_snow_to_soil = q_snow_to_soil + ice_m + liq_m
            melt = torch.where(scalar_snow, torch.clamp(melt - scalar_melt, min=0.0), melt)
            scalar_remaining = torch.clamp(ice[..., 0] + liq[..., 0], min=0.0)
            scalar_scale = torch.where(
                scalar_available > 1.0e-12,
                scalar_remaining / torch.clamp(scalar_available, min=1.0e-12),
                torch.zeros_like(scalar_available),
            )
            scalar_depth = torch.where(
                scalar_remaining > 1.0e-8,
                dz[..., 0] * torch.clamp(scalar_scale, min=0.0, max=1.0),
                torch.zeros_like(dz[..., 0]),
            )
            dz[..., 0] = torch.where(scalar_snow, scalar_depth, dz[..., 0])

        if torch.any(rain_mass > 0.0) and torch.any(nsnow > 0):
            liq[..., 0] = liq[..., 0] + torch.where(nsnow > 0, rain_mass, torch.zeros_like(rain_mass))

        # Convert thermodynamic melt into liquid within the snowpack, using the
        # uppermost layers first.
        if torch.any(melt > 0.0) and torch.any(nsnow > 0):
            remaining = melt.clone()
            for k in range(MAX_SNOW_LAYERS):
                layer_mask = nsnow > k
                if not torch.any(layer_mask):
                    continue
                convert = torch.minimum(remaining, torch.clamp(ice[..., k], min=0.0))
                ice[..., k] = torch.clamp(ice[..., k] - convert, min=0.0)
                liq[..., k] = liq[..., k] + convert
                remaining = torch.clamp(remaining - convert, min=0.0)
            if torch.any(remaining > 0.0):
                liq[..., 0] = liq[..., 0] + remaining

        # Route excess liquid downward layer by layer.
        for k in range(MAX_SNOW_LAYERS):
            layer_mask = nsnow > k
            if not torch.any(layer_mask):
                continue
            dzk = dz[..., k]
            icek = torch.clamp(ice[..., k], min=0.0)
            liqk = torch.clamp(liq[..., k], min=0.0)
            totalk = icek + liqk
            vol_ice = torch.where(
                totalk > 0.0,
                torch.clamp(icek / torch.clamp(dzk * frac_sno_eff * DENICE, min=1.0e-12), max=1.0),
                torch.zeros_like(totalk),
            )
            eff_porosity = torch.clamp(1.0 - vol_ice, min=0.0)
            vol_liq = torch.where(
                totalk > 0.0,
                torch.minimum(
                    eff_porosity,
                    liqk / torch.clamp(dzk * frac_sno_eff * DENH2O, min=1.0e-12),
                ),
                torch.zeros_like(totalk),
            )
            qout = torch.clamp((vol_liq - SNOW_IRREDUCIBLE_SAT * eff_porosity) * dzk * frac_sno_eff * DENH2O, min=0.0)
            if k < MAX_SNOW_LAYERS - 1:
                # next_mask: per-element flag for whether the next layer is active.
                # Liquid drains to layer k+1 where it is active; otherwise it exits
                # the snowpack bottom and goes directly to soil.
                next_mask = nsnow > (k + 1)
                if torch.any(next_mask):
                    next_dz = dz[..., k + 1]
                    next_ice = torch.clamp(ice[..., k + 1], min=0.0)
                    next_liq = torch.clamp(liq[..., k + 1], min=0.0)
                    next_total = next_ice + next_liq
                    next_vol_ice = torch.where(
                        next_total > 0.0,
                        torch.clamp(next_ice / torch.clamp(next_dz * frac_sno_eff * DENICE, min=1.0e-12), max=1.0),
                        torch.zeros_like(next_total),
                    )
                    next_eff_porosity = torch.clamp(1.0 - next_vol_ice, min=0.0)
                    next_vol_liq = torch.where(
                        next_total > 0.0,
                        torch.minimum(
                            next_eff_porosity,
                            next_liq / torch.clamp(next_dz * frac_sno_eff * DENH2O, min=1.0e-12),
                        ),
                        torch.zeros_like(next_total),
                    )
                    next_capacity = torch.clamp(
                        (1.0 - next_vol_ice - next_vol_liq) * next_dz * frac_sno_eff * DENH2O,
                        min=0.0,
                    )
                    # Only apply capacity limit for elements where the next layer is active.
                    qout = torch.where(next_mask, torch.minimum(qout, next_capacity), qout)
                # Route to layer k+1 where active; drain to soil where k is the last active layer.
                liq[..., k + 1] = liq[..., k + 1] + torch.where(next_mask, qout, torch.zeros_like(qout))
                q_snow_to_soil = q_snow_to_soil + torch.where(next_mask, torch.zeros_like(qout), qout)
            else:
                q_snow_to_soil = q_snow_to_soil + qout
            liq[..., k] = torch.clamp(liq[..., k] - qout, min=0.0)

        h2osno_new = torch.sum(torch.clamp(ice + liq, min=0.0), dim=-1)

        # Cap total snow mass by removing excess from the bottom snow layer.
        cap_excess = torch.clamp(h2osno_new - H2OSNO_MAX, min=0.0)
        qcap_ice = torch.zeros_like(cap_excess)
        qcap_liq = torch.zeros_like(cap_excess)
        if torch.any(cap_excess > 0.0):
            bottom_mass = torch.clamp(ice[..., -1] + liq[..., -1], min=0.0)
            remove = torch.minimum(cap_excess, bottom_mass * (1.0 - MIN_SNOW_TO_KEEP))
            frac_ice = torch.where(bottom_mass > 0.0, ice[..., -1] / torch.clamp(bottom_mass, min=1.0e-12), torch.zeros_like(bottom_mass))
            remove_ice = remove * frac_ice
            remove_liq = remove * (1.0 - frac_ice)
            ice[..., -1] = torch.clamp(ice[..., -1] - remove_ice, min=0.0)
            liq[..., -1] = torch.clamp(liq[..., -1] - remove_liq, min=0.0)
            qcap_ice = remove_ice
            qcap_liq = remove_liq
            h2osno_new = torch.sum(torch.clamp(ice + liq, min=0.0), dim=-1)
        pre_drain_mass = torch.clamp(base_snow + rain_mass, min=0.0)
        mass_loss = torch.clamp(pre_drain_mass - h2osno_new, min=0.0)
        compact_mask = (nsnow > 0) & (mass_loss > 1.0e-8) & (h2osno_new > 1.0e-8)
        if torch.any(compact_mask):
            loss_frac = torch.clamp(mass_loss / torch.clamp(h2osno_new, min=1.0e-12), min=0.0, max=1.0)
            scale = torch.clamp(1.0 - loss_frac, min=0.0, max=1.0)
            layer_axis_shape = (1,) * len(tuple(nsnow.shape)) + (MAX_SNOW_LAYERS,)
            layer_index = torch.arange(MAX_SNOW_LAYERS, device=device).reshape(layer_axis_shape)
            active_layers = layer_index < nsnow[..., None]
            min_dz = (torch.clamp(ice, min=0.0) / DENICE + torch.clamp(liq, min=0.0) / DENH2O) / torch.clamp(
                frac_sno_eff[..., None],
                min=1.0e-12,
            )
            dz_compact = torch.maximum(dz * scale[..., None], min_dz)
            dz = torch.where(compact_mask[..., None] & active_layers, dz_compact, dz)
        snow_layers_new = SnowLayerState(
            snl=snl,
            dz=dz,
            h2osoi_ice=ice,
            h2osoi_liq=liq,
            t_snow=t_snow,
        )
        snow_depth_new = _snow_depth_from_layers_torch(snow_layers_new, device=device, dtype=dtype)
        state_frac_sno = torch.clamp(
            _snow_batch_like_torch(state.frac_sno, tuple(h2osno_new.shape), device=device, dtype=dtype),
            min=0.0,
            max=1.0,
        )
        frac_sno_new = torch.where(
            h2osno_new > 1.0e-8,
            state_frac_sno,
            torch.zeros_like(h2osno_new),
        )
        snow_layers_new = _rebalance_snow_layers_torch(snow_layers_new, frac_sno_new)
        h2osno_new = _snow_mass_from_layers_torch(snow_layers_new, device=device, dtype=dtype)
        snow_depth_new = _snow_depth_from_layers_torch(snow_layers_new, device=device, dtype=dtype)
        frac_sno_new = torch.where(
            h2osno_new > 1.0e-8,
            state_frac_sno,
            torch.zeros_like(h2osno_new),
        )
        int_snow_base = (
            _snow_batch_like_torch(state.int_snow, tuple(h2osno_new.shape), device=device, dtype=dtype)
            if state.int_snow is not None
            else base_snow
        )
        int_snow_new = torch.where(
            h2osno_new > 1.0e-8,
            torch.maximum(int_snow_base + rain_mass, h2osno_new),
            torch.zeros_like(h2osno_new),
        )
        return SnowWaterResult(
            state=CoupledColumnState(
                h2osoi_liq=_to_torch(state.h2osoi_liq, device, dtype),
                h2osoi_ice=_to_torch(state.h2osoi_ice, device, dtype),
                h2ocan=_to_torch(state.h2ocan, device, dtype),
                h2osfc=_to_torch(state.h2osfc, device, dtype),
                frac_h2osfc=_to_torch(state.frac_h2osfc, device, dtype),
                frac_sno=frac_sno_new,
                zwt=_to_torch(state.zwt, device, dtype),
                t_soisno=_to_torch(state.t_soisno, device, dtype),
                wa=_to_torch(state.wa, device, dtype) if state.wa is not None else None,
                t_h2osfc=_to_torch(state.t_h2osfc, device, dtype) if state.t_h2osfc is not None else None,
                t10=_to_torch(state.t10, device, dtype) if state.t10 is not None else None,
                t10_history=_to_torch(state.t10_history, device, dtype) if state.t10_history is not None else None,
                h2osno=h2osno_new,
                snow_depth=snow_depth_new,
                snow_t=_to_torch(state.snow_t, device, dtype) if state.snow_t is not None else None,
                snow_layers=snow_layers_new,
                int_snow=int_snow_new,
            ),
            qflx_snow_to_soil=q_snow_to_soil,
            qflx_snwcp_ice=qcap_ice,
            qflx_snwcp_liq=qcap_liq,
        )

    h2osno_old = snow_storage_from_state(state)
    snow_layers = state.snow_layers
    if snow_layers is None:
        snow_depth0 = snow_depth_from_storage(h2osno_old)
        snow_layers = _snow_layers_from_storage_numpy(
            h2osno_old,
            snow_depth0,
            snow_t=_to_np(state.snow_t) if state.snow_t is not None else None,
            frac_sno=_to_np(state.frac_sno) if state.frac_sno is not None else None,
        )
    melt = np.zeros_like(h2osno_old) if snow_melt is None else np.maximum(np.asarray(snow_melt, dtype=np.float64), 0.0)
    rain_mass = (
        np.zeros_like(h2osno_old)
        if rain_to_snowpack is None
        else np.maximum(np.asarray(rain_to_snowpack, dtype=np.float64), 0.0)
    )
    melt = np.maximum(_snow_batch_like_numpy(melt, h2osno_old.shape), 0.0)
    rain_mass = np.maximum(_snow_batch_like_numpy(rain_mass, h2osno_old.shape), 0.0)
    ice = np.array(snow_layers.h2osoi_ice, dtype=np.float64, copy=True)
    liq = np.array(snow_layers.h2osoi_liq, dtype=np.float64, copy=True)
    dz = np.array(snow_layers.dz, dtype=np.float64, copy=True)
    snl = np.asarray(snow_layers.snl, dtype=np.int32)
    t_snow = np.asarray(snow_layers.t_snow, dtype=np.float64)
    frac_sno_eff = np.clip(_snow_batch_like_numpy(_to_np(state.frac_sno), snl.shape), 1.0e-12, 1.0)
    nsnow = np.maximum(-snl.astype(np.int32), 0)
    inactive_scalar = (nsnow == 0) & (h2osno_old > 1.0e-8) & (np.sum(np.maximum(ice + liq, 0.0), axis=-1) <= 1.0e-8)
    if np.any(inactive_scalar):
        ice[..., 0] = np.where(inactive_scalar, h2osno_old, ice[..., 0])

    q_snow_to_soil = np.zeros_like(h2osno_old)
    scalar_snow = (nsnow == 0) & (h2osno_old > 1.0e-8)
    if np.any(scalar_snow) and np.any(rain_mass > 0.0):
        scalar_rain = np.where(scalar_snow, rain_mass, 0.0)
        q_snow_to_soil = q_snow_to_soil + scalar_rain
        rain_mass = np.where(scalar_snow, 0.0, rain_mass)
    if np.any(scalar_snow) and np.any(melt > 0.0):
        scalar_available = np.maximum(ice[..., 0] + liq[..., 0], 0.0)
        scalar_melt = np.where(scalar_snow, np.minimum(melt, scalar_available), 0.0)
        ice_m = np.minimum(scalar_melt, np.maximum(ice[..., 0], 0.0))
        ice[..., 0] = np.maximum(ice[..., 0] - ice_m, 0.0)
        liq_m = np.minimum(scalar_melt - ice_m, np.maximum(liq[..., 0], 0.0))
        liq[..., 0] = np.maximum(liq[..., 0] - liq_m, 0.0)
        q_snow_to_soil = q_snow_to_soil + ice_m + liq_m
        melt = np.where(scalar_snow, np.maximum(melt - scalar_melt, 0.0), melt)
        scalar_remaining = np.maximum(ice[..., 0] + liq[..., 0], 0.0)
        scalar_scale = np.where(
            scalar_available > 1.0e-12,
            scalar_remaining / np.maximum(scalar_available, 1.0e-12),
            0.0,
        )
        scalar_depth = np.where(
            scalar_remaining > 1.0e-8,
            dz[..., 0] * np.clip(scalar_scale, 0.0, 1.0),
            0.0,
        )
        dz[..., 0] = np.where(scalar_snow, scalar_depth, dz[..., 0])

    if np.any(rain_mass > 0.0) and np.any(nsnow > 0):
        liq[..., 0] = liq[..., 0] + np.where(nsnow > 0, rain_mass, 0.0)

    if np.any(melt > 0.0) and np.any(nsnow > 0):
        remaining = melt.copy()
        for k in range(MAX_SNOW_LAYERS):
            if not np.any(nsnow > k):
                continue
            convert = np.minimum(remaining, np.maximum(ice[..., k], 0.0))
            ice[..., k] = np.maximum(ice[..., k] - convert, 0.0)
            liq[..., k] = liq[..., k] + convert
            remaining = np.maximum(remaining - convert, 0.0)
        if np.any(remaining > 0.0):
            liq[..., 0] = liq[..., 0] + remaining

    for k in range(MAX_SNOW_LAYERS):
        if not np.any(nsnow > k):
            continue
        dzk = dz[..., k]
        icek = np.maximum(ice[..., k], 0.0)
        liqk = np.maximum(liq[..., k], 0.0)
        totalk = icek + liqk
        vol_ice = np.where(
            totalk > 0.0,
            np.minimum(1.0, icek / np.maximum(dzk * frac_sno_eff * DENICE, 1.0e-12)),
            0.0,
        )
        eff_porosity = np.maximum(1.0 - vol_ice, 0.0)
        vol_liq = np.where(
            totalk > 0.0,
            np.minimum(eff_porosity, liqk / np.maximum(dzk * frac_sno_eff * DENH2O, 1.0e-12)),
            0.0,
        )
        qout = np.maximum((vol_liq - SNOW_IRREDUCIBLE_SAT * eff_porosity) * dzk * frac_sno_eff * DENH2O, 0.0)
        if k < MAX_SNOW_LAYERS - 1:
            # next_mask: per-element flag for whether the next layer is active.
            # Liquid drains to layer k+1 where it is active; otherwise it exits
            # the snowpack bottom and goes directly to soil.
            next_mask = nsnow > (k + 1)
            if np.any(next_mask):
                next_dz = dz[..., k + 1]
                next_ice = np.maximum(ice[..., k + 1], 0.0)
                next_liq = np.maximum(liq[..., k + 1], 0.0)
                next_total = next_ice + next_liq
                next_vol_ice = np.where(
                    next_total > 0.0,
                    np.minimum(1.0, next_ice / np.maximum(next_dz * frac_sno_eff * DENICE, 1.0e-12)),
                    0.0,
                )
                next_eff_porosity = np.maximum(1.0 - next_vol_ice, 0.0)
                next_vol_liq = np.where(
                    next_total > 0.0,
                    np.minimum(next_eff_porosity, next_liq / np.maximum(next_dz * frac_sno_eff * DENH2O, 1.0e-12)),
                    0.0,
                )
                next_capacity = np.maximum((1.0 - next_vol_ice - next_vol_liq) * next_dz * frac_sno_eff * DENH2O, 0.0)
                # Only apply capacity limit for elements where the next layer is active.
                qout = np.where(next_mask, np.minimum(qout, next_capacity), qout)
            # Route to layer k+1 where active; drain to soil where k is the last active layer.
            liq[..., k + 1] = liq[..., k + 1] + np.where(next_mask, qout, 0.0)
            q_snow_to_soil = q_snow_to_soil + np.where(next_mask, 0.0, qout)
        else:
            q_snow_to_soil = q_snow_to_soil + qout
        liq[..., k] = np.maximum(liq[..., k] - qout, 0.0)

    h2osno_new = np.sum(np.maximum(ice + liq, 0.0), axis=-1)
    cap_excess = np.maximum(h2osno_new - H2OSNO_MAX, 0.0)
    qcap_ice = np.zeros_like(cap_excess)
    qcap_liq = np.zeros_like(cap_excess)
    if np.any(cap_excess > 0.0):
        bottom_mass = np.maximum(ice[..., -1] + liq[..., -1], 0.0)
        remove = np.minimum(cap_excess, bottom_mass * (1.0 - MIN_SNOW_TO_KEEP))
        frac_ice = np.where(bottom_mass > 0.0, ice[..., -1] / np.maximum(bottom_mass, 1.0e-12), 0.0)
        remove_ice = remove * frac_ice
        remove_liq = remove * (1.0 - frac_ice)
        ice[..., -1] = np.maximum(ice[..., -1] - remove_ice, 0.0)
        liq[..., -1] = np.maximum(liq[..., -1] - remove_liq, 0.0)
        qcap_ice = remove_ice
        qcap_liq = remove_liq
        h2osno_new = np.sum(np.maximum(ice + liq, 0.0), axis=-1)

    pre_drain_mass = np.maximum(h2osno_old + rain_mass, 0.0)
    mass_loss = np.maximum(pre_drain_mass - h2osno_new, 0.0)
    compact_mask = (nsnow > 0) & (mass_loss > 1.0e-8) & (h2osno_new > 1.0e-8)
    if np.any(compact_mask):
        loss_frac = np.clip(mass_loss / np.maximum(h2osno_new, 1.0e-12), 0.0, 1.0)
        scale = np.clip(1.0 - loss_frac, 0.0, 1.0)
        layer_shape = (1,) * nsnow.ndim + (MAX_SNOW_LAYERS,)
        layer_index = np.arange(MAX_SNOW_LAYERS, dtype=np.int32).reshape(layer_shape)
        active_layers = layer_index < nsnow[..., None]
        min_dz = (np.maximum(ice, 0.0) / DENICE + np.maximum(liq, 0.0) / DENH2O) / np.maximum(
            frac_sno_eff[..., None],
            1.0e-12,
        )
        dz_compact = np.maximum(dz * scale[..., None], min_dz)
        dz = np.where(compact_mask[..., None] & active_layers, dz_compact, dz)

    snow_layers_new = SnowLayerState(
        snl=snl,
        dz=dz,
        h2osoi_ice=ice,
        h2osoi_liq=liq,
        t_snow=t_snow,
    )
    snow_depth_new = _snow_depth_from_layers_numpy(snow_layers_new)
    state_frac_sno = np.clip(_snow_batch_like_numpy(_to_np(state.frac_sno), h2osno_new.shape), 0.0, 1.0)
    frac_sno_new = np.where(
        h2osno_new > 1.0e-8,
        state_frac_sno,
        0.0,
    )
    snow_layers_new = _rebalance_snow_layers_numpy(snow_layers_new, frac_sno_new)
    h2osno_new = _snow_mass_from_layers_numpy(snow_layers_new)
    snow_depth_new = _snow_depth_from_layers_numpy(snow_layers_new)
    frac_sno_new = np.where(
        h2osno_new > 1.0e-8,
        state_frac_sno,
        0.0,
    )
    int_snow_base = (
        _snow_batch_like_numpy(_to_np(state.int_snow), h2osno_new.shape)
        if state.int_snow is not None
        else h2osno_old
    )
    int_snow_new = np.where(
        h2osno_new > 1.0e-8,
        np.maximum(int_snow_base + rain_mass, h2osno_new),
        0.0,
    )
    return SnowWaterResult(
        state=CoupledColumnState(
            h2osoi_liq=_to_np(state.h2osoi_liq),
            h2osoi_ice=_to_np(state.h2osoi_ice),
            h2ocan=_to_np(state.h2ocan),
            h2osfc=_to_np(state.h2osfc),
            frac_h2osfc=_to_np(state.frac_h2osfc),
            frac_sno=frac_sno_new,
            zwt=_to_np(state.zwt),
            t_soisno=_to_np(state.t_soisno),
            wa=_to_np(state.wa) if state.wa is not None else None,
            t_h2osfc=_to_np(state.t_h2osfc) if state.t_h2osfc is not None else None,
            t10=_to_np(state.t10) if state.t10 is not None else None,
            t10_history=_to_np(state.t10_history) if state.t10_history is not None else None,
            h2osno=h2osno_new,
            snow_depth=snow_depth_new,
            snow_t=_to_np(state.snow_t) if state.snow_t is not None else None,
            snow_layers=snow_layers_new,
            int_snow=int_snow_new,
        ),
        qflx_snow_to_soil=q_snow_to_soil,
        qflx_snwcp_ice=qcap_ice,
        qflx_snwcp_liq=qcap_liq,
    )


def update_t10_state(
    state: CoupledColumnState,
    t_ref2m: np.ndarray | torch.Tensor,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """ELM-style recursive 10-day running mean for 2 m temperature.

    Native ELM's accumulMod ``runmean`` is not a finite boxcar after the
    accumulation period is reached. It updates recursively with
    ``new = ((period - 1) * old + current) / period``.
    """
    window = max(1, int(round((10.0 * 86400.0) / dt)))
    tref_np = _to_numpy(t_ref2m)
    assert tref_np is not None
    tref = np.asarray(tref_np, dtype=np.float64)
    if state.t10 is not None:
        prev = _to_np(state.t10)
    elif state.t10_history is not None:
        prev = _to_np(state.t10_history).mean(axis=-1)
    else:
        prev = tref
    t10 = ((window - 1) * prev + tref) / window
    history = np.broadcast_to(t10[..., None], t10.shape + (window,)).copy()
    return t10, history


def _sanitize_running_mean_reference(
    t_ref2m: np.ndarray | torch.Tensor,
    forc_t: np.ndarray | torch.Tensor,
) -> np.ndarray | torch.Tensor:
    """Fallback to forcing temperature when the 2 m reference is invalid."""
    if isinstance(t_ref2m, torch.Tensor) or isinstance(forc_t, torch.Tensor):
        ref_tensor = t_ref2m if isinstance(t_ref2m, torch.Tensor) else forc_t
        assert isinstance(ref_tensor, torch.Tensor)
        tref = _to_torch(t_ref2m, ref_tensor.device, ref_tensor.dtype)
        forc = _to_torch(forc_t, ref_tensor.device, ref_tensor.dtype)
        valid = torch.isfinite(tref) & (torch.abs(tref) < 1.0e5)
        return torch.where(valid, tref, forc)
    tref_np = _to_numpy(t_ref2m)
    forc_np = _to_numpy(forc_t)
    assert tref_np is not None and forc_np is not None
    tref = np.asarray(tref_np, dtype=np.float64)
    forc = np.asarray(forc_np, dtype=np.float64)
    valid = np.isfinite(tref) & (np.abs(tref) < 1.0e5)
    return np.where(valid, tref, forc)


def update_t10_state_torch(
    state: CoupledColumnState,
    t_ref2m: np.ndarray | torch.Tensor,
    dt: float,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Torch-native ELM-style recursive 10-day running mean for ``t10``."""
    window = max(1, int(round((10.0 * 86400.0) / dt)))
    tref = _to_torch(t_ref2m, device, dtype)
    if state.t10 is not None:
        prev = _to_torch(state.t10, device, dtype)
    elif state.t10_history is not None:
        prev = _to_torch(state.t10_history, device, dtype).mean(dim=-1)
    else:
        prev = tref
    t10 = ((window - 1) * prev + tref) / window
    # Do not clone the full (ncell, nens, window) history tensor every step.
    # state.t10 is always set after initialisation, so coupled_t10_from_state_torch
    # returns state.t10 directly and never consults the history.
    # Cloning a (ncell, nens, window) tensor (potentially hundreds of MB) each
    # timestep is wasteful; pass the existing history through unchanged.
    history = state.t10_history
    return t10, history


def exact_btran_and_root_fraction(
    state: CoupledColumnState,
    static: CoupledStaticParameters,
    *,
    soil_diag: dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return CLM/ELM-style ``btran`` and the layer root fractions."""
    rootfr = np.asarray(static.rootfr, dtype=np.float64)
    if static.smpso is None or static.smpsc is None:
        root = np.maximum(rootfr, 0.0)
        rootr = np.broadcast_to(root, state.h2osfc.shape + root.shape)
        btran = np.ones(state.h2osfc.shape, dtype=np.float64)
        return btran, rootr

    # Check per-state cache (populated on the first call within each timestep).
    static_id = id(static)
    if state._btran_cache is not None and static_id in state._btran_cache:
        return state._btran_cache[static_id]

    if soil_diag is None:
        soil_diag = coupled_soil_state_diagnostics(state, static, nprofile=state.t_soisno.shape[-1])
    nlev_hydro = static.hydrology_numpy.dz.shape[-1]
    tempk = _to_np(state.t_soisno)[..., :nlev_hydro]
    h2osoi_liqvol = np.asarray(soil_diag["h2osoi_liqvol"], dtype=np.float64)
    eff_porosity = np.asarray(soil_diag["eff_porosity"], dtype=np.float64)
    smp_l = np.asarray(soil_diag["smp_l"], dtype=np.float64)
    watsat = np.asarray(static.hydrology_numpy.watsat, dtype=np.float64)
    # Broadcast watsat to match [batch, patch, nlev] shape of liqvol
    if watsat.ndim < smp_l.ndim:
        watsat = np.reshape(watsat, (1,) * (smp_l.ndim - watsat.ndim) + watsat.shape)
    btran, rootr_norm = compute_btran_and_root_fraction(
        liqvol=h2osoi_liqvol,
        tempk=tempk,
        frac_h2osfc=_to_np(state.frac_h2osfc),
        rootfr=rootfr,
        smpso=float(static.smpso),
        smpsc=float(static.smpsc),
        tc_stress_c=float(static.tc_stress_c),
        h2osfc_mm=_to_np(state.h2osfc),
        eff_porosity=eff_porosity,
        smp_l=smp_l,
        watsat=watsat,
    )
    result = (btran, rootr_norm)
    # Memoize on the state object so repeat calls within the same timestep are free.
    if state._btran_cache is None:
        state._btran_cache = {}
    state._btran_cache[static_id] = result
    return result


def exact_btran_and_root_fraction_torch(
    state: CoupledColumnState,
    static: CoupledStaticParameters,
    *,
    device: torch.device,
    dtype: torch.dtype,
    soil_diag: dict[str, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Torch-native CLM/ELM-style ``btran`` and layer root fractions."""
    rootfr = (
        static.rootfr_torch
        if static.rootfr_torch is not None and static.rootfr_torch.device == device and static.rootfr_torch.dtype == dtype
        else torch.as_tensor(np.maximum(np.asarray(static.rootfr, dtype=np.float64), 0.0), device=device, dtype=dtype)
    )
    rootfr_norm = (
        static.rootfr_norm_torch
        if static.rootfr_norm_torch is not None and static.rootfr_norm_torch.device == device and static.rootfr_norm_torch.dtype == dtype
        else rootfr / torch.clamp(rootfr.sum(), min=1.0e-12)
    )
    nlev_hydro = static.hydrology_numpy.dz.shape[-1]
    rootfr = rootfr[:nlev_hydro]
    rootfr_norm = rootfr_norm[:nlev_hydro]
    if static.smpso is None or static.smpsc is None:
        rootr = torch.broadcast_to(rootfr_norm, _to_torch(state.h2osfc, device, dtype).shape + rootfr_norm.shape)
        btran = torch.ones_like(_to_torch(state.h2osfc, device, dtype))
        return btran, rootr

    # Use hydrology_torch params directly — they are already correctly shaped
    # tensors on device (either (1,1,nlev) for scalar params or (ncell,1,nlev)
    # for per-cell params).  Reading from hydrology_numpy and applying
    # [None, None, :] breaks when watsat/sucsat/bsw are (ncell, nlev).
    ht = static.hydrology_torch
    dz     = ht.dz.to(device=device, dtype=dtype)
    watsat = ht.watsat.to(device=device, dtype=dtype)
    sucsat = ht.sucsat.to(device=device, dtype=dtype)
    bsw    = ht.bsw.to(device=device, dtype=dtype)
    h2osoi_liq = _to_torch(state.h2osoi_liq, device, dtype)[..., :nlev_hydro]
    h2osoi_ice = _to_torch(state.h2osoi_ice, device, dtype)[..., :nlev_hydro]
    h2osoi_liqvol = h2osoi_liq / (dz * DENH2O)
    vol_ice = torch.minimum(watsat, h2osoi_ice / (dz * DENICE))
    eff_porosity = torch.maximum(watsat - vol_ice, torch.tensor(0.01, device=device, dtype=dtype))
    smp_l = -sucsat * torch.clamp(h2osoi_liqvol / torch.clamp(eff_porosity, min=1.0e-12), min=0.01) ** (-bsw)
    smp_l = torch.maximum(smp_l, torch.tensor(-1.0e8, device=device, dtype=dtype))
    tempk = _to_torch(state.t_soisno, device, dtype)[..., :nlev_hydro]
    if watsat.ndim < smp_l.ndim:
        watsat = watsat.reshape((1,) * (smp_l.ndim - watsat.ndim) + watsat.shape)
    rootfr_unf = torch.where(tempk > (TFRZ + float(static.tc_stress_c)), rootfr, torch.zeros_like(rootfr))
    root_sum_active = torch.sum(rootfr_unf, dim=-1, keepdim=True)
    rootfr_unf = torch.where(root_sum_active > 1.0e-12, rootfr_unf / root_sum_active, torch.zeros_like(rootfr_unf))
    smp_node = torch.maximum(torch.as_tensor(float(static.smpsc), device=device, dtype=dtype), smp_l)
    denom = torch.clamp(torch.as_tensor(float(static.smpso - static.smpsc), device=device, dtype=dtype), min=1.0e-12)
    rresis = torch.minimum(
        (eff_porosity / torch.clamp(watsat, min=1.0e-12)) * (smp_node - float(static.smpsc)) / denom,
        torch.ones_like(smp_node),
    )
    h2osfc_mm = _to_torch(state.h2osfc, device, dtype).reshape(h2osoi_liqvol.shape[:-1])
    floodf = torch.where(
        h2osfc_mm <= 1000.0,
        torch.clamp(1.0 - 0.001 * torch.clamp(h2osfc_mm, min=0.0), min=0.0),
        torch.zeros_like(h2osfc_mm),
    )
    active = (h2osoi_liqvol > 0.0) & (tempk > (TFRZ + float(static.tc_stress_c)))
    root_contrib = torch.where(active, rootfr_unf * rresis * floodf[..., None], torch.zeros_like(rootfr_unf))
    btran = torch.sum(root_contrib, dim=-1)
    rootr_norm = torch.where(btran[..., None] > 1.0e-12, root_contrib / btran[..., None], torch.zeros_like(root_contrib))
    return btran, rootr_norm


def effective_root_fraction(
    state: CoupledColumnState,
    static: CoupledStaticParameters,
) -> np.ndarray:
    """Return the normalized layer root fraction used for transpiration sinks."""
    if isinstance(state.h2osfc, torch.Tensor) or isinstance(state.t_soisno, torch.Tensor):
        _, rootr = exact_btran_and_root_fraction_torch(
            state,
            static,
            device=state.h2osfc.device if isinstance(state.h2osfc, torch.Tensor) else state.t_soisno.device,
            dtype=state.h2osfc.dtype if isinstance(state.h2osfc, torch.Tensor) else state.t_soisno.dtype,
        )
        return np.asarray(rootr.detach().cpu().numpy(), dtype=np.float64)
    _, rootr = exact_btran_and_root_fraction(state, static)
    return rootr


def coupled_btran_from_state(
    state: CoupledColumnState,
    static: CoupledStaticParameters,
) -> np.ndarray:
    """Return the patch-level transpiration wetness factor used by ELM-style stress."""
    if isinstance(state.h2osfc, torch.Tensor) or isinstance(state.t_soisno, torch.Tensor):
        btran, _ = exact_btran_and_root_fraction_torch(
            state,
            static,
            device=state.h2osfc.device if isinstance(state.h2osfc, torch.Tensor) else state.t_soisno.device,
            dtype=state.h2osfc.dtype if isinstance(state.h2osfc, torch.Tensor) else state.t_soisno.dtype,
        )
        return np.asarray(btran.detach().cpu().numpy(), dtype=np.float64)
    btran, _ = exact_btran_and_root_fraction(state, static)
    return btran


def coupled_btran_from_state_torch(
    state: CoupledColumnState,
    static: CoupledStaticParameters,
    *,
    device: torch.device,
    dtype: torch.dtype,
    h2osfc_storage: torch.Tensor,
    liqvol: torch.Tensor,
    eff_porosity: torch.Tensor,
    smp_l: torch.Tensor,
) -> torch.Tensor:
    """Torch-native CLM/ELM-style btran formulation."""
    if static.smpso is None or static.smpsc is None:
        return torch.ones_like(h2osfc_storage[..., 0] if h2osfc_storage.ndim > 0 else h2osfc_storage)

    rootfr = (
        static.rootfr_norm_torch
        if static.rootfr_norm_torch is not None and static.rootfr_norm_torch.device == device and static.rootfr_norm_torch.dtype == dtype
        else torch.as_tensor(np.maximum(np.asarray(static.rootfr, dtype=np.float64), 0.0), device=device, dtype=dtype)
    )
    if static.rootfr_norm_torch is None or static.rootfr_norm_torch.device != device or static.rootfr_norm_torch.dtype != dtype:
        rootfr = rootfr / torch.clamp(rootfr.sum(), min=1.0e-12)
    rootfr = rootfr[:liqvol.shape[-1]]

    tempk = _to_torch(state.t_soisno, device, dtype)[..., :liqvol.shape[-1]]
    active = (liqvol > 0.0) & (tempk > (TFRZ + float(static.tc_stress_c)))

    smp_node = torch.maximum(torch.as_tensor(float(static.smpsc), device=device, dtype=dtype), smp_l)
    denom = torch.clamp(torch.as_tensor(float(static.smpso - static.smpsc), device=device, dtype=dtype), min=1.0e-12)
    ep_over_ws = eff_porosity / torch.clamp(_to_torch(static.hydrology_torch.watsat, device, dtype), min=1.0e-12)
    rresis = torch.minimum(ep_over_ws * (smp_node - float(static.smpsc)) / denom, torch.ones_like(smp_node))

    if h2osfc_storage.ndim == liqvol.ndim - 1:
        h2osfc_mm = h2osfc_storage
    else:
        h2osfc_mm = h2osfc_storage.reshape(liqvol.shape[:-1])
    floodf = torch.where(
        h2osfc_mm <= 1000.0,
        torch.clamp(1.0 - 0.001 * torch.clamp(h2osfc_mm, min=0.0), min=0.0),
        torch.zeros_like(h2osfc_mm),
    )
    rresis = torch.where(active, rresis * floodf[..., None], torch.zeros_like(rresis))

    rootfr_unf = torch.where(tempk > (TFRZ + float(static.tc_stress_c)), rootfr, torch.zeros_like(rootfr))
    root_sum_active = torch.sum(rootfr_unf, dim=-1, keepdim=True)
    rootfr_unf = torch.where(root_sum_active > 1.0e-12, rootfr_unf / root_sum_active, torch.zeros_like(rootfr_unf))
    btran = torch.sum(rootfr_unf * rresis, dim=-1)
    return btran


def distribute_root_uptake(
    qflx_tran_veg: np.ndarray | torch.Tensor,
    state: CoupledColumnState,
    static: CoupledStaticParameters,
    backend: str,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> np.ndarray | torch.Tensor:
    if backend == "torch":
        assert device is not None and dtype is not None
        _, rootr = exact_btran_and_root_fraction_torch(
            state,
            static,
            device=device,
            dtype=dtype,
        )
        tran_t = _to_torch(qflx_tran_veg, device, dtype)
        return tran_t[..., None] * rootr
    rootr = np.array(effective_root_fraction(state, static), copy=True)
    tran = np.asarray(qflx_tran_veg, dtype=np.float64)
    return tran[..., None] * rootr


def build_texture_static_parameters(
    texture: SoilTextureProfile,
    rootfr: np.ndarray,
    *,
    dz: np.ndarray | None = None,
    z: np.ndarray | None = None,
    zi: np.ndarray | None = None,
    wtfact: float | None = None,
    fover: float | None = None,
    micro_sigma: float | None = None,
    pc: float = 0.4,
    h2osfc_thresh: float | None = None,
    topo_slope: float | np.ndarray | None = None,
    device: torch.device,
    dtype: torch.dtype,
) -> CoupledStaticParameters:
    """Build consistent hydrology and soil-thermal parameters from texture."""
    if dz is None or z is None or zi is None:
        dz, z, zi = elm_soil_grid(nlevsoi=len(texture.sand_pct))
    nlevbed = None
    if len(texture.sand_pct) > 10:
        nlevbed = 10
    hydro_np = hydraulic_params_from_texture(
        dz=dz,
        z=z,
        zi=zi,
        texture=texture,
        rootfr=rootfr,
        topo_slope=topo_slope,
    )
    if wtfact is not None:
        hydro_np.wtfact = np.array([[wtfact]], dtype=np.float64)
    if fover is not None:
        hydro_np.fover = np.array([[fover]], dtype=np.float64)
    if h2osfc_thresh is not None:
        hydro_np.h2osfc_thresh = np.array([[h2osfc_thresh]], dtype=np.float64)
    hydro_torch = SoilWaterCoreParameters(
        dz=torch.as_tensor(hydro_np.dz, device=device, dtype=dtype),
        z=torch.as_tensor(hydro_np.z, device=device, dtype=dtype),
        zi=torch.as_tensor(hydro_np.zi, device=device, dtype=dtype),
        watsat=torch.as_tensor(hydro_np.watsat, device=device, dtype=dtype),
        sucsat=torch.as_tensor(hydro_np.sucsat, device=device, dtype=dtype),
        bsw=torch.as_tensor(hydro_np.bsw, device=device, dtype=dtype),
        hksat=torch.as_tensor(hydro_np.hksat, device=device, dtype=dtype),
        smpmin=torch.as_tensor(hydro_np.smpmin, device=device, dtype=dtype),
        wtfact=torch.as_tensor(hydro_np.wtfact, device=device, dtype=dtype),
        fover=torch.as_tensor(hydro_np.fover, device=device, dtype=dtype),
        fdrain=torch.as_tensor(hydro_np.fdrain, device=device, dtype=dtype),
        h2osfc_thresh=torch.as_tensor(hydro_np.h2osfc_thresh, device=device, dtype=dtype),
        pc=torch.as_tensor(hydro_np.pc, device=device, dtype=dtype),
        topo_slope=None if topo_slope is None else torch.as_tensor(topo_slope, device=device, dtype=dtype),
    )
    soil_np = thermal_params_from_texture(dz=dz, z=z, zi=zi, texture=texture, nlevbed=nlevbed)
    soil_torch = thermal_params_from_texture_torch(
        texture=texture,
        dz=dz,
        z=z,
        zi=zi,
        nlevbed=nlevbed,
        device=device,
        dtype=dtype,
    )
    rootfr_np = np.asarray(rootfr, dtype=np.float64)
    rootfr_t = torch.as_tensor(np.maximum(rootfr_np, 0.0), device=device, dtype=dtype)
    rootfr_norm_t = rootfr_t / torch.clamp(rootfr_t.sum(), min=1.0e-12)
    return CoupledStaticParameters(
        rootfr=rootfr_np,
        hydrology_numpy=hydro_np,
        hydrology_torch=hydro_torch,
        soil_temperature_numpy=soil_np,
        soil_temperature_torch=soil_torch,
        micro_sigma=0.03 if micro_sigma is None else micro_sigma,
        pc=float(pc),
        topo_slope=None if topo_slope is None else float(np.asarray(topo_slope, dtype=np.float64).reshape(())),
        rootfr_torch=rootfr_t,
        rootfr_norm_torch=rootfr_norm_t,
    )


def build_static_parameters_from_surfdata(
    surfdata_file: Path | str,
    *,
    device: torch.device,
    dtype: torch.dtype,
    gridcell: int = 0,
    nlevsoi: int | None = None,
    rootfr: np.ndarray | None = None,
    organic_max: float = 130.0,
) -> tuple[CoupledStaticParameters, SurfdataSiteInfo, np.ndarray, np.ndarray, np.ndarray]:
    """Build coupled static parameters directly from location-specific surfdata fields."""
    site = load_site_info_from_surfdata(
        surfdata_file,
        gridcell=gridcell,
        organic_max=organic_max,
    )
    nlev = site.sand_pct.size if nlevsoi is None else int(nlevsoi)
    dz, z, zi = elm_soil_grid(nlevsoi=nlev)
    if rootfr is None:
        beta = 0.966
        zi_cm_full = np.concatenate([[0.0], zi * 100.0])
        rootfr = beta ** zi_cm_full[:-1] - beta ** zi_cm_full[1:]
        rootfr = np.maximum(rootfr, 0.0)
        rootfr /= rootfr.sum()
    else:
        rootfr = _extend_rootfr_to_nlev(rootfr, nlev)
    texture = SoilTextureProfile(
        sand_pct=_extend_profile_to_nlev(site.sand_pct, nlev),
        clay_pct=_extend_profile_to_nlev(site.clay_pct, nlev),
        organic_frac=_extend_profile_to_nlev(site.organic_frac, nlev),
    )
    static = build_texture_static_parameters(
        texture,
        rootfr,
        dz=dz,
        z=z,
        zi=zi,
        wtfact=site.fmax,
        fover=0.5,
        micro_sigma=site.micro_sigma_m,
        pc=site.pc,
        h2osfc_thresh=h2osfc_threshold_from_micro_sigma(site.micro_sigma_m, pc=site.pc),
        topo_slope=site.slope,
        device=device,
        dtype=dtype,
    )
    static.n_melt = 200.0 / max(10.0, site.std_elev_m)
    static.soil_color = site.soil_color
    return static, site, dz, z, zi


def predict_surface_targets(
    bundle: SurfaceModelBundle,
    surface_features: np.ndarray | torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
    already_zero_feature_names: set[str] | frozenset[str] | None = None,
) -> dict[str, np.ndarray] | dict[str, torch.Tensor]:
    already_zero_feature_names = already_zero_feature_names or frozenset()
    if isinstance(surface_features, torch.Tensor):
        x = _to_torch(surface_features, device, dtype)
        x_mean = (
            bundle.x_mean_torch
            if bundle.x_mean_torch is not None and bundle.x_mean_torch.device == device and bundle.x_mean_torch.dtype == dtype
            else torch.as_tensor(bundle.x_mean, device=device, dtype=dtype)
        )
        x_std = (
            bundle.x_std_safe_torch
            if bundle.x_std_safe_torch is not None and bundle.x_std_safe_torch.device == device and bundle.x_std_safe_torch.dtype == dtype
            else torch.as_tensor(np.where(bundle.x_std > 0.0, bundle.x_std, 1.0), device=device, dtype=dtype)
        )
        x_scaled = (x - x_mean) / x_std
        constant_mask = bundle.x_float32_constant_mask_torch
        if x_scaled.dtype != torch.float64 and constant_mask is not None:
            if constant_mask.device != x_scaled.device:
                constant_mask = constant_mask.to(device=x_scaled.device)
            x_scaled[..., constant_mask] = 0.0
        batch_shape = tuple(x_scaled.shape[:-1])
        x2 = x_scaled.reshape(-1, x_scaled.shape[-1])
        input_idx_map = _bundle_input_index_map(tuple(bundle.input_names))
        target_idx_map = _bundle_input_index_map(tuple(bundle.target_names))
        with torch.inference_mode():
            y = bundle.model(x2)
            # Apply per-target input masks: re-run forward with blocked features zeroed
            # and splice the masked-target columns into the output.
            if bundle.target_feature_masks:
                # Group targets by unique feature-mask signature to minimise extra passes.
                mask_key_to_targets: dict[tuple[str, ...], list[int]] = {}
                for t_name, f_names in bundle.target_feature_masks.items():
                    if t_name not in bundle.target_names:
                        continue
                    key = tuple(sorted(f_names))
                    mask_key_to_targets.setdefault(key, []).append(bundle.target_names.index(t_name))
                for f_names_key, t_indices in mask_key_to_targets.items():
                    if all(f_name in already_zero_feature_names for f_name in f_names_key):
                        continue
                    x2_masked = x2.clone()
                    for f_name in f_names_key:
                        idx = input_idx_map.get(f_name)
                        if idx is not None:
                            x2_masked[:, idx] = 0.0
                    y_masked = bundle.model(x2_masked)
                    for t_idx in t_indices:
                        y[:, t_idx] = y_masked[:, t_idx]
            y = _apply_surface_energy_smp_l_clip_for_targets(
                bundle,
                x2,
                y,
                input_idx_map,
                already_zero_feature_names=already_zero_feature_names,
            )
            y = y.reshape(batch_shape + (len(bundle.target_names),))
        y_phys = postprocess_surface_predictions_torch(y, bundle, surface_features=x)
        targets = _target_map_torch(y_phys, bundle.target_names)
        return _apply_patch0_zero_surface_targets(targets, bundle, batch_shape, like=y_phys)

    x_np = _to_numpy(surface_features)
    assert x_np is not None
    x_scaled = (x_np - bundle.x_mean) / np.where(bundle.x_std > 0.0, bundle.x_std, 1.0)
    x = _to_torch(x_scaled, device, dtype)
    batch_shape = tuple(x.shape[:-1])
    x2 = x.reshape(-1, x.shape[-1])
    input_idx_map = _bundle_input_index_map(tuple(bundle.input_names))
    with torch.inference_mode():
        y = bundle.model(x2)
        # Apply per-target input masks: re-run forward with blocked features zeroed
        # and splice the masked-target columns into the output.
        if bundle.target_feature_masks:
            # Group targets by unique feature-mask signature to minimise extra passes.
            mask_key_to_targets: dict[tuple[str, ...], list[int]] = {}
            for t_name, f_names in bundle.target_feature_masks.items():
                if t_name not in bundle.target_names:
                    continue
                key = tuple(sorted(f_names))
                mask_key_to_targets.setdefault(key, []).append(bundle.target_names.index(t_name))
            for f_names_key, t_indices in mask_key_to_targets.items():
                if all(f_name in already_zero_feature_names for f_name in f_names_key):
                    continue
                x2_masked = x2.clone()
                for f_name in f_names_key:
                    idx = input_idx_map.get(f_name)
                    if idx is not None:
                        x2_masked[:, idx] = 0.0
                y_masked = bundle.model(x2_masked)
                for t_idx in t_indices:
                    y[:, t_idx] = y_masked[:, t_idx]
        y = _apply_surface_energy_smp_l_clip_for_targets(
            bundle,
            x2,
            y,
            input_idx_map,
            already_zero_feature_names=already_zero_feature_names,
        )
        y = y.reshape(batch_shape + (len(bundle.target_names),))
    y_phys = postprocess_surface_predictions(y.detach().cpu().numpy(), bundle, surface_features=x_np)
    targets = _target_map(y_phys, bundle.target_names)
    return _apply_patch0_zero_surface_targets(targets, bundle, batch_shape, like=y_phys)


def _hydrology_nlev(static: CoupledStaticParameters) -> int:
    return int(np.asarray(static.hydrology_numpy.dz).shape[-1])


def _splice_hydrology_array(
    original: np.ndarray | torch.Tensor,
    hydro_updated: np.ndarray | torch.Tensor,
) -> np.ndarray | torch.Tensor:
    nlev = int(hydro_updated.shape[-1])
    if original.shape[-1] == nlev:
        return hydro_updated
    if isinstance(original, torch.Tensor) or isinstance(hydro_updated, torch.Tensor):
        device = hydro_updated.device if isinstance(hydro_updated, torch.Tensor) else original.device
        dtype = hydro_updated.dtype if isinstance(hydro_updated, torch.Tensor) else original.dtype
        out = _to_torch(original, device, dtype).clone()
        out[..., :nlev] = _to_torch(hydro_updated, device, dtype)
        return out
    out = np.array(original, dtype=np.float64, copy=True)
    out[..., :nlev] = np.asarray(hydro_updated, dtype=np.float64)
    return out


def _numpy_hydrology_step(
    state: CoupledColumnState,
    static: CoupledStaticParameters,
    surface_targets: Mapping[str, np.ndarray],
    dt: float,
) -> tuple[HydrologyResult, CoupledColumnState]:
    nlev_hydro = _hydrology_nlev(static)
    if "qflx_rootsoi" in surface_targets:
        qroot = np.asarray(surface_targets["qflx_rootsoi"], dtype=np.float64)[..., :nlev_hydro]
    else:
        qroot = distribute_root_uptake(
            np.asarray(surface_targets["qflx_tran_veg"], dtype=np.float64),
            state,
            static,
            "numpy",
        )
    forcing = HydrologyForcing(
        qflx_top_soil=np.asarray(surface_targets["qflx_prec_grnd"], dtype=np.float64),
        qflx_prec_grnd=np.asarray(surface_targets["qflx_prec_grnd"], dtype=np.float64),
        qflx_evap_grnd=np.asarray(surface_targets["qflx_ev_soil"], dtype=np.float64),
        qflx_ev_soil=np.asarray(surface_targets["qflx_ev_soil"], dtype=np.float64),
        qflx_ev_h2osfc=np.asarray(surface_targets["qflx_ev_h2osfc"], dtype=np.float64),
        qflx_rootsoi=qroot,
        qflx_tran_veg=np.asarray(surface_targets["qflx_tran_veg"], dtype=np.float64),
    )
    hydro_state = HydrologyState(
        h2osoi_liq=_to_np(state.h2osoi_liq)[..., :nlev_hydro],
        h2osoi_ice=_to_np(state.h2osoi_ice)[..., :nlev_hydro],
        h2osfc=_to_np(state.h2osfc),
        zwt=_to_np(state.zwt),
        frac_h2osfc=_to_np(state.frac_h2osfc),
        frac_sno=_active_snow_fraction_numpy(state),
        t_soisno=_to_np(state.t_soisno)[..., :nlev_hydro],
    )
    result = hydrology_no_drainage_step(hydro_state, static.hydrology_numpy, forcing, dt)
    next_state = CoupledColumnState(
        h2osoi_liq=_splice_hydrology_array(_to_np(state.h2osoi_liq), result.state.h2osoi_liq),
        h2osoi_ice=_splice_hydrology_array(_to_np(state.h2osoi_ice), result.state.h2osoi_ice),
        h2ocan=_to_np(state.h2ocan),
        h2osfc=result.state.h2osfc,
        frac_h2osfc=frac_h2osfc_from_storage(result.state.h2osfc, static.micro_sigma),
        frac_sno=result.state.frac_sno,
        zwt=result.state.zwt,
        t_soisno=_to_np(state.t_soisno),
        t_h2osfc=_to_np(state.t_h2osfc) if state.t_h2osfc is not None else None,
        t10=_to_np(state.t10) if state.t10 is not None else None,
        t10_history=_to_np(state.t10_history) if state.t10_history is not None else None,
        h2osno=_to_np(state.h2osno) if state.h2osno is not None else None,
        snow_depth=_to_np(state.snow_depth) if state.snow_depth is not None else None,
        snow_t=_to_np(state.snow_t) if state.snow_t is not None else None,
        snow_layers=state.snow_layers,
        int_snow=_to_np(state.int_snow) if state.int_snow is not None else None,
    )
    return result, next_state


def _numpy_hydrology_step_full(
    state: CoupledColumnState,
    static: CoupledStaticParameters,
    surface_targets: Mapping[str, np.ndarray],
    dt: float,
) -> tuple[HydrologyResult, CoupledColumnState]:
    nlev_hydro = _hydrology_nlev(static)
    if "qflx_rootsoi" in surface_targets:
        qroot = np.asarray(surface_targets["qflx_rootsoi"], dtype=np.float64)[..., :nlev_hydro]
    else:
        qroot = distribute_root_uptake(
            np.asarray(surface_targets["qflx_tran_veg"], dtype=np.float64),
            state,
            static,
            "numpy",
        )
    forcing = HydrologyForcing(
        qflx_top_soil=np.asarray(surface_targets["qflx_prec_grnd"], dtype=np.float64),
        qflx_prec_grnd=np.asarray(surface_targets["qflx_prec_grnd"], dtype=np.float64),
        qflx_evap_grnd=np.asarray(surface_targets["qflx_ev_soil"], dtype=np.float64),
        qflx_ev_soil=np.asarray(surface_targets["qflx_ev_soil"], dtype=np.float64),
        qflx_ev_h2osfc=np.asarray(surface_targets["qflx_ev_h2osfc"], dtype=np.float64),
        qflx_rootsoi=qroot,
        qflx_tran_veg=np.asarray(surface_targets["qflx_tran_veg"], dtype=np.float64),
    )
    hydro_state = HydrologyState(
        h2osoi_liq=_to_np(state.h2osoi_liq)[..., :nlev_hydro],
        h2osoi_ice=_to_np(state.h2osoi_ice)[..., :nlev_hydro],
        h2osfc=_to_np(state.h2osfc),
        zwt=_to_np(state.zwt),
        frac_h2osfc=_to_np(state.frac_h2osfc),
        frac_sno=_active_snow_fraction_numpy(state),
        t_soisno=_to_np(state.t_soisno)[..., :nlev_hydro],
    )
    result = hydrology_step(hydro_state, static.hydrology_numpy, forcing, dt, wa=state.wa)
    next_state = CoupledColumnState(
        h2osoi_liq=_splice_hydrology_array(_to_np(state.h2osoi_liq), result.state.h2osoi_liq),
        h2osoi_ice=_splice_hydrology_array(_to_np(state.h2osoi_ice), result.state.h2osoi_ice),
        h2ocan=_to_np(state.h2ocan),
        h2osfc=result.state.h2osfc,
        frac_h2osfc=frac_h2osfc_from_storage(result.state.h2osfc, static.micro_sigma),
        frac_sno=result.state.frac_sno,
        zwt=result.state.zwt,
        wa=result.wa,
        t_soisno=_to_np(state.t_soisno),
        t_h2osfc=_to_np(state.t_h2osfc) if state.t_h2osfc is not None else None,
        t10=_to_np(state.t10) if state.t10 is not None else None,
        t10_history=_to_np(state.t10_history) if state.t10_history is not None else None,
        h2osno=_to_np(state.h2osno) if state.h2osno is not None else None,
        snow_depth=_to_np(state.snow_depth) if state.snow_depth is not None else None,
        snow_t=_to_np(state.snow_t) if state.snow_t is not None else None,
        snow_layers=state.snow_layers,
        int_snow=_to_np(state.int_snow) if state.int_snow is not None else None,
    )
    return result, next_state


def _torch_hydrology_step(
    state: CoupledColumnState,
    static: CoupledStaticParameters,
    surface_targets: Mapping[str, np.ndarray],
    dt: float,
    *,
    device: torch.device,
    dtype: torch.dtype,
    workspace: SoilHydrologyTorchWorkspace | None = None,
) -> tuple[SoilHydrologyTorchResult, CoupledColumnState]:
    nlev_hydro = _hydrology_nlev(static)
    qflx_tran_veg = _to_torch(surface_targets["qflx_tran_veg"], device, dtype)
    if "qflx_rootsoi" in surface_targets:
        qroot = _to_torch(surface_targets["qflx_rootsoi"], device, dtype)[..., :nlev_hydro]
    else:
        qroot = distribute_root_uptake(qflx_tran_veg, state, static, "torch", device=device, dtype=dtype)
    result = hydrology_no_drainage_torch(
        SoilHydrologyNoDrainageState(
            h2osoi_liq=_to_torch(state.h2osoi_liq, device, dtype)[..., :nlev_hydro],
            h2osoi_ice=_to_torch(state.h2osoi_ice, device, dtype)[..., :nlev_hydro],
            h2osfc=_to_torch(state.h2osfc, device, dtype),
            t_soisno=_to_torch(state.t_soisno, device, dtype)[..., :nlev_hydro],
            zwt=_to_torch(state.zwt, device, dtype),
        ),
        static.hydrology_torch,
        qflx_top_soil=_to_torch(surface_targets["qflx_prec_grnd"], device, dtype),
        qflx_evap_grnd=_to_torch(surface_targets["qflx_ev_soil"], device, dtype),
        qflx_ev_soil=_to_torch(surface_targets["qflx_ev_soil"], device, dtype),
        qflx_ev_h2osfc=_to_torch(surface_targets["qflx_ev_h2osfc"], device, dtype),
        frac_h2osfc=_to_torch(state.frac_h2osfc, device, dtype),
        frac_sno=_active_snow_fraction_torch(state, device=device, dtype=dtype),
        qflx_rootsoi=qroot,
        dt=dt,
        workspace=workspace,
    )
    frac_h2osfc = frac_h2osfc_from_storage_torch(result.h2osfc, static.micro_sigma, device=device, dtype=dtype)
    h2osoi_ice = _to_torch(state.h2osoi_ice, device, dtype)
    next_state = CoupledColumnState(
        h2osoi_liq=_splice_hydrology_array(_to_torch(state.h2osoi_liq, device, dtype), result.h2osoi_liq),
        h2osoi_ice=h2osoi_ice,
        h2ocan=_to_torch(state.h2ocan, device, dtype),
        h2osfc=result.h2osfc,
        frac_h2osfc=frac_h2osfc,
        frac_sno=_active_snow_fraction_torch(state, device=device, dtype=dtype),
        zwt=result.zwt if result.zwt is not None else _to_torch(state.zwt, device, dtype),
        wa=result.wa,
        t_soisno=_to_torch(state.t_soisno, device, dtype),
        t_h2osfc=_to_torch(state.t_h2osfc, device, dtype) if state.t_h2osfc is not None else None,
        t10=_to_torch(state.t10, device, dtype) if state.t10 is not None else None,
        t10_history=_to_torch(state.t10_history, device, dtype) if state.t10_history is not None else None,
        h2osno=_to_torch(state.h2osno, device, dtype) if state.h2osno is not None else None,
        snow_depth=_to_torch(state.snow_depth, device, dtype) if state.snow_depth is not None else None,
        snow_t=_to_torch(state.snow_t, device, dtype) if state.snow_t is not None else None,
        snow_layers=_snow_layers_to_device(state.snow_layers, device, dtype),
        int_snow=_to_torch(state.int_snow, device, dtype) if state.int_snow is not None else None,
    )
    return result, next_state


def _torch_hydrology_step_full(
    state: CoupledColumnState,
    static: CoupledStaticParameters,
    surface_targets: Mapping[str, np.ndarray],
    dt: float,
    *,
    device: torch.device,
    dtype: torch.dtype,
    workspace: SoilHydrologyTorchWorkspace | None = None,
    profile_sections: dict[str, float] | None = None,
) -> tuple[SoilHydrologyTorchResult, CoupledColumnState]:
    nlev_hydro = _hydrology_nlev(static)
    qflx_tran_veg = _to_torch(surface_targets["qflx_tran_veg"], device, dtype)
    if "qflx_rootsoi" in surface_targets:
        qroot = _to_torch(surface_targets["qflx_rootsoi"], device, dtype)[..., :nlev_hydro]
    else:
        qroot = distribute_root_uptake(qflx_tran_veg, state, static, "torch", device=device, dtype=dtype)
    result = hydrology_torch(
        SoilHydrologyNoDrainageState(
            h2osoi_liq=_to_torch(state.h2osoi_liq, device, dtype)[..., :nlev_hydro],
            h2osoi_ice=_to_torch(state.h2osoi_ice, device, dtype)[..., :nlev_hydro],
            h2osfc=_to_torch(state.h2osfc, device, dtype),
            t_soisno=_to_torch(state.t_soisno, device, dtype)[..., :nlev_hydro],
            zwt=_to_torch(state.zwt, device, dtype),
        ),
        static.hydrology_torch,
        qflx_top_soil=_to_torch(surface_targets["qflx_prec_grnd"], device, dtype),
        qflx_evap_grnd=_to_torch(surface_targets["qflx_ev_soil"], device, dtype),
        qflx_ev_soil=_to_torch(surface_targets["qflx_ev_soil"], device, dtype),
        qflx_ev_h2osfc=_to_torch(surface_targets["qflx_ev_h2osfc"], device, dtype),
        frac_h2osfc=_to_torch(state.frac_h2osfc, device, dtype),
        frac_sno=_active_snow_fraction_torch(state, device=device, dtype=dtype),
        qflx_rootsoi=qroot,
        dt=dt,
        wa=_to_torch(state.wa, device, dtype) if state.wa is not None else None,
        workspace=workspace,
        profile_sections=profile_sections,
    )
    frac_h2osfc = frac_h2osfc_from_storage_torch(result.h2osfc, static.micro_sigma, device=device, dtype=dtype)
    h2osoi_ice = _to_torch(state.h2osoi_ice, device, dtype)
    next_state = CoupledColumnState(
        h2osoi_liq=_splice_hydrology_array(_to_torch(state.h2osoi_liq, device, dtype), result.h2osoi_liq),
        h2osoi_ice=h2osoi_ice,
        h2ocan=_to_torch(state.h2ocan, device, dtype),
        h2osfc=result.h2osfc,
        frac_h2osfc=frac_h2osfc,
        frac_sno=_to_torch(state.frac_sno, device, dtype),
        zwt=result.zwt if result.zwt is not None else _to_torch(state.zwt, device, dtype),
        wa=result.wa,
        t_soisno=_to_torch(state.t_soisno, device, dtype),
        t_h2osfc=_to_torch(state.t_h2osfc, device, dtype) if state.t_h2osfc is not None else None,
        t10=_to_torch(state.t10, device, dtype) if state.t10 is not None else None,
        t10_history=_to_torch(state.t10_history, device, dtype) if state.t10_history is not None else None,
        h2osno=_to_torch(state.h2osno, device, dtype) if state.h2osno is not None else None,
        snow_depth=_to_torch(state.snow_depth, device, dtype) if state.snow_depth is not None else None,
        snow_t=_to_torch(state.snow_t, device, dtype) if state.snow_t is not None else None,
        snow_layers=_snow_layers_to_device(state.snow_layers, device, dtype),
        int_snow=_to_torch(state.int_snow, device, dtype) if state.int_snow is not None else None,
    )
    return result, next_state


def _numpy_soil_temperature_step(
    state: CoupledColumnState,
    static: CoupledStaticParameters,
    surface_targets: Mapping[str, np.ndarray],
    dt: float,
    *,
    snow_enabled: bool = True,
) -> tuple[SoilTemperatureResult, CoupledColumnState]:
    forcing = SoilTemperatureForcing(
        hs_soil=np.asarray(surface_targets["hs_soil"], dtype=np.float64),
        hs_top_snow=np.asarray(surface_targets.get("hs_top_snow", surface_targets["hs_soil"]), dtype=np.float64),
        hs_h2osfc=np.asarray(surface_targets["hs_h2osfc"], dtype=np.float64),
        dhsdT=np.asarray(surface_targets["dhsdT"], dtype=np.float64),
        eflx_bot=np.zeros_like(np.asarray(surface_targets["hs_soil"], dtype=np.float64)),
    )
    result = soil_temperature_step(
        SoilTemperatureState(
            t_soisno=_to_np(state.t_soisno),
            h2osoi_liq=_to_np(state.h2osoi_liq),
            h2osoi_ice=_to_np(state.h2osoi_ice),
            frac_sno=_to_np(state.frac_sno),
            t_h2osfc=_to_np(state.t_h2osfc) if state.t_h2osfc is not None else None,
            h2osfc=_to_np(state.h2osfc),
            frac_h2osfc=_to_np(state.frac_h2osfc),
            h2osno=_to_np(state.h2osno) if state.h2osno is not None else None,
            snow_depth=_to_np(state.snow_depth) if state.snow_depth is not None else None,
            snow_t=_to_np(state.snow_t) if state.snow_t is not None else None,
            snow_layers=state.snow_layers,
        ),
        static.soil_temperature_numpy,
        forcing,
        dt,
        snow_enabled=snow_enabled,
    )
    next_state = CoupledColumnState(
        h2osoi_liq=result.state.h2osoi_liq,
        h2osoi_ice=result.state.h2osoi_ice,
        h2ocan=_to_np(state.h2ocan),
        h2osfc=result.state.h2osfc if result.state.h2osfc is not None else _to_np(state.h2osfc),
        frac_h2osfc=(
            result.state.frac_h2osfc
            if result.state.frac_h2osfc is not None
            else frac_h2osfc_from_storage_torch(
                result.state.h2osfc if result.state.h2osfc is not None else state.h2osfc,
                static.micro_sigma,
                device=torch.device("cpu"),
                dtype=torch.float64,
            )
        ),
        frac_sno=_to_np(state.frac_sno),
        zwt=_to_np(state.zwt),
        wa=_to_np(state.wa) if state.wa is not None else None,
        t_soisno=result.state.t_soisno,
        t_h2osfc=result.state.t_h2osfc,
        t10=_to_np(state.t10) if state.t10 is not None else None,
        t10_history=_to_np(state.t10_history) if state.t10_history is not None else None,
        h2osno=result.state.h2osno if result.state.h2osno is not None else (_to_np(state.h2osno) if state.h2osno is not None else None),
        snow_depth=result.state.snow_depth if result.state.snow_depth is not None else (_to_np(state.snow_depth) if state.snow_depth is not None else None),
        snow_t=result.state.snow_t if result.state.snow_t is not None else (_to_np(state.snow_t) if state.snow_t is not None else None),
        snow_layers=(
            result.state.snow_layers
            if result.state.snow_layers is not None
            else _snow_layers_from_storage_numpy(
                result.state.h2osno if result.state.h2osno is not None else state.h2osno,
                result.state.snow_depth if result.state.snow_depth is not None else state.snow_depth,
                result.state.snow_t if result.state.snow_t is not None else state.snow_t,
                frac_sno=state.frac_sno,
            )
        ),
        int_snow=_to_np(state.int_snow) if state.int_snow is not None else None,
    )
    return result, next_state


def _torch_soil_temperature_step(
    state: CoupledColumnState,
    static: CoupledStaticParameters,
    surface_targets: Mapping[str, np.ndarray],
    dt: float,
    *,
    device: torch.device,
    dtype: torch.dtype,
    workspace: SoilTemperatureTorchWorkspace | None = None,
    allow_numpy_thermal_fallback: bool = True,
    snow_enabled: bool = True,
) -> tuple[SoilTemperatureResultTorch, CoupledColumnState]:
    if allow_numpy_thermal_fallback:
        if state.snow_layers is not None and getattr(state.snow_layers, "snl", None) is not None:
            snl = torch.as_tensor(state.snow_layers.snl, device=device, dtype=torch.int32)
            use_numpy_thermal = bool(torch.any(snl < 0).detach().cpu().item())
        else:
            snow_mass = _to_torch(state.h2osno, device, dtype) if state.h2osno is not None else torch.zeros_like(_to_torch(state.h2osfc, device, dtype))
            snow_depth = _to_torch(state.snow_depth, device, dtype) if state.snow_depth is not None else torch.zeros_like(snow_mass)
            use_numpy_thermal = bool(torch.any((snow_mass > MIN_SNOW_LAYER_MASS) & (snow_depth > 0.0)).detach().cpu().item())
    else:
        use_numpy_thermal = False
    if use_numpy_thermal:
        targets_np = {
            name: _surface_target_to_numpy(value)
            for name, value in surface_targets.items()
        }
        result_np, state_np = _numpy_soil_temperature_step(
            coupled_state_to_cpu(state),
            static,
            targets_np,
            dt,
            snow_enabled=snow_enabled,
        )
        return result_np, coupled_state_to_device(state_np, device, dtype)

    forcing = SoilTemperatureForcingTorch(
        hs_soil=_to_torch(surface_targets["hs_soil"], device, dtype),
        hs_top_snow=_to_torch(surface_targets.get("hs_top_snow", surface_targets["hs_soil"]), device, dtype),
        hs_h2osfc=_to_torch(surface_targets["hs_h2osfc"], device, dtype),
        dhsdT=_to_torch(surface_targets["dhsdT"], device, dtype),
        eflx_bot=torch.zeros_like(_to_torch(surface_targets["hs_soil"], device, dtype)),
    )
    result = soil_temperature_step_torch(
        SoilTemperatureStateTorch(
            t_soisno=_to_torch(state.t_soisno, device, dtype),
            h2osoi_liq=_to_torch(state.h2osoi_liq, device, dtype),
            h2osoi_ice=_to_torch(state.h2osoi_ice, device, dtype),
            frac_sno=_to_torch(state.frac_sno, device, dtype),
            t_h2osfc=None if state.t_h2osfc is None else _to_torch(state.t_h2osfc, device, dtype),
            h2osfc=_to_torch(state.h2osfc, device, dtype),
            frac_h2osfc=_to_torch(state.frac_h2osfc, device, dtype),
            h2osno=_to_torch(state.h2osno, device, dtype) if state.h2osno is not None else None,
            snow_depth=_to_torch(state.snow_depth, device, dtype) if state.snow_depth is not None else None,
            snow_t=_to_torch(state.snow_t, device, dtype) if state.snow_t is not None else None,
            snow_layers=_snow_layers_to_device(state.snow_layers, device, dtype),
        ),
        static.soil_temperature_torch,
        forcing,
        dt,
        workspace=workspace,
        snow_enabled=snow_enabled,
    )
    next_state = CoupledColumnState(
        h2osoi_liq=result.state.h2osoi_liq,
        h2osoi_ice=result.state.h2osoi_ice,
        h2ocan=_to_torch(state.h2ocan, device, dtype),
        h2osfc=result.state.h2osfc if result.state.h2osfc is not None else _to_torch(state.h2osfc, device, dtype),
        frac_h2osfc=(
            result.state.frac_h2osfc
            if result.state.frac_h2osfc is not None
            else frac_h2osfc_from_storage_torch(
                result.state.h2osfc if result.state.h2osfc is not None else state.h2osfc,
                static.micro_sigma,
                device=device,
                dtype=dtype,
            )
        ),
        frac_sno=_to_torch(state.frac_sno, device, dtype),
        zwt=_to_torch(state.zwt, device, dtype),
        wa=_to_torch(state.wa, device, dtype) if state.wa is not None else None,
        t_soisno=result.state.t_soisno,
        t_h2osfc=result.state.t_h2osfc,
        t10=_to_torch(state.t10, device, dtype) if state.t10 is not None else None,
        t10_history=_to_torch(state.t10_history, device, dtype) if state.t10_history is not None else None,
        h2osno=result.state.h2osno if result.state.h2osno is not None else (_to_torch(state.h2osno, device, dtype) if state.h2osno is not None else None),
        snow_depth=result.state.snow_depth if result.state.snow_depth is not None else (_to_torch(state.snow_depth, device, dtype) if state.snow_depth is not None else None),
        snow_t=result.state.snow_t if result.state.snow_t is not None else (_to_torch(state.snow_t, device, dtype) if state.snow_t is not None else None),
        snow_layers=(
            result.state.snow_layers
            if result.state.snow_layers is not None
            else _snow_layers_from_storage_torch(
                result.state.h2osno if result.state.h2osno is not None else state.h2osno,
                result.state.snow_depth if result.state.snow_depth is not None else state.snow_depth,
                result.state.snow_t if result.state.snow_t is not None else state.snow_t,
                frac_sno=state.frac_sno,
                device=device,
                dtype=dtype,
            )
        ),
        int_snow=_to_torch(state.int_snow, device, dtype) if state.int_snow is not None else None,
    )
    return result, next_state


def _soil_temperature_state_from_coupled_torch(state: CoupledColumnState) -> SoilTemperatureStateTorch:
    return SoilTemperatureStateTorch(
        t_soisno=state.t_soisno,
        h2osoi_liq=state.h2osoi_liq,
        h2osoi_ice=state.h2osoi_ice,
        frac_sno=state.frac_sno,
        snow_layers=state.snow_layers,
        t_h2osfc=state.t_h2osfc,
        h2osfc=state.h2osfc,
        frac_h2osfc=state.frac_h2osfc,
        h2osno=state.h2osno,
        snow_depth=state.snow_depth,
        snow_t=state.snow_t,
    )


def _cell_scatter_soil_temperature_state_torch(
    base: SoilTemperatureStateTorch,
    indices: torch.Tensor,
    update: SoilTemperatureStateTorch,
) -> SoilTemperatureStateTorch:
    return SoilTemperatureStateTorch(
        t_soisno=_cell_scatter_value(base.t_soisno, indices, update.t_soisno),
        h2osoi_liq=_cell_scatter_value(base.h2osoi_liq, indices, update.h2osoi_liq),
        h2osoi_ice=_cell_scatter_value(base.h2osoi_ice, indices, update.h2osoi_ice),
        frac_sno=_cell_scatter_value(base.frac_sno, indices, update.frac_sno),
        snow_layers=_cell_scatter_snow_layers(base.snow_layers, indices, update.snow_layers),
        t_h2osfc=_cell_scatter_value(base.t_h2osfc, indices, update.t_h2osfc),
        h2osfc=_cell_scatter_value(base.h2osfc, indices, update.h2osfc),
        frac_h2osfc=_cell_scatter_value(base.frac_h2osfc, indices, update.frac_h2osfc),
        h2osno=_cell_scatter_value(base.h2osno, indices, update.h2osno),
        snow_depth=_cell_scatter_value(base.snow_depth, indices, update.snow_depth),
        snow_t=_cell_scatter_value(base.snow_t, indices, update.snow_t),
    )


def _cell_scatter_optional_result_tensor(base, indices: torch.Tensor, update, like: torch.Tensor):
    if base is None and update is None:
        return None
    if base is None:
        base = torch.zeros_like(like)
    return _cell_scatter_value(base, indices, update)


def _merge_soil_temperature_result_torch(
    base_result: SoilTemperatureResultTorch,
    base_state: CoupledColumnState,
    indices: torch.Tensor,
    update_result: SoilTemperatureResultTorch,
) -> SoilTemperatureResultTorch:
    base_result_state = _soil_temperature_state_from_coupled_torch(base_state)
    return SoilTemperatureResultTorch(
        state=_cell_scatter_soil_temperature_state_torch(base_result_state, indices, update_result.state),
        eflx_fgr12=_cell_scatter_value(base_result.eflx_fgr12, indices, update_result.eflx_fgr12),
        fn_h2osfc_old=_cell_scatter_value(base_result.fn_h2osfc_old, indices, update_result.fn_h2osfc_old),
        fn_h2osfc_new=_cell_scatter_value(base_result.fn_h2osfc_new, indices, update_result.fn_h2osfc_new),
        imelt=_cell_scatter_value(base_result.imelt, indices, update_result.imelt),
        qflx_snow_melt=_cell_scatter_optional_result_tensor(
            base_result.qflx_snow_melt,
            indices,
            update_result.qflx_snow_melt,
            base_result.eflx_fgr12,
        ),
        qflx_scalar_snow_melt=_cell_scatter_optional_result_tensor(
            getattr(base_result, "qflx_scalar_snow_melt", None),
            indices,
            getattr(update_result, "qflx_scalar_snow_melt", None),
            base_result.eflx_fgr12,
        ),
        qflx_snow_freeze=_cell_scatter_optional_result_tensor(
            base_result.qflx_snow_freeze,
            indices,
            update_result.qflx_snow_freeze,
            base_result.eflx_fgr12,
        ),
        qflx_soil_melt=_cell_scatter_optional_result_tensor(
            getattr(base_result, "qflx_soil_melt", None),
            indices,
            getattr(update_result, "qflx_soil_melt", None),
            base_result.eflx_fgr12,
        ),
        qflx_soil_freeze=_cell_scatter_optional_result_tensor(
            getattr(base_result, "qflx_soil_freeze", None),
            indices,
            getattr(update_result, "qflx_soil_freeze", None),
            base_result.eflx_fgr12,
        ),
        qflx_soil_melt_profile=_cell_scatter_optional_result_tensor(
            getattr(base_result, "qflx_soil_melt_profile", None),
            indices,
            getattr(update_result, "qflx_soil_melt_profile", None),
            base_result.state.t_soisno,
        ),
        qflx_soil_freeze_profile=_cell_scatter_optional_result_tensor(
            getattr(base_result, "qflx_soil_freeze_profile", None),
            indices,
            getattr(update_result, "qflx_soil_freeze_profile", None),
            base_result.state.t_soisno,
        ),
        phase_hm_h2osfc=_cell_scatter_optional_result_tensor(
            base_result.phase_hm_h2osfc,
            indices,
            update_result.phase_hm_h2osfc,
            base_result.eflx_fgr12,
        ),
        phase_xm_h2osfc=_cell_scatter_optional_result_tensor(
            base_result.phase_xm_h2osfc,
            indices,
            update_result.phase_xm_h2osfc,
            base_result.eflx_fgr12,
        ),
        phase_temp1_h2osfc=_cell_scatter_optional_result_tensor(
            base_result.phase_temp1_h2osfc,
            indices,
            update_result.phase_temp1_h2osfc,
            base_result.eflx_fgr12,
        ),
    )


def _torch_soil_temperature_step_grouped_snow(
    state: CoupledColumnState,
    static: CoupledStaticParameters,
    surface_targets: Mapping[str, np.ndarray],
    dt: float,
    *,
    device: torch.device,
    dtype: torch.dtype,
    workspace: SoilTemperatureTorchWorkspace | None = None,
    allow_numpy_thermal_fallback: bool = False,
    snow_enabled: bool = True,
) -> tuple[SoilTemperatureResultTorch, CoupledColumnState]:
    if not snow_enabled:
        return _torch_soil_temperature_step(
            state,
            static,
            surface_targets,
            dt,
            device=device,
            dtype=dtype,
            workspace=workspace,
            allow_numpy_thermal_fallback=allow_numpy_thermal_fallback,
            snow_enabled=False,
        )

    state = _repair_snow_layer_state_shape_torch(state, device=device, dtype=dtype)
    active_idx = _active_snow_cell_indices_torch(state, device=device, dtype=dtype)
    ncell = int(_to_torch(state.h2osfc, device, dtype).shape[0])
    if active_idx.numel() > 0:
        active_idx = active_idx[(active_idx >= 0) & (active_idx < ncell)]
    if active_idx.numel() == 0 or active_idx.numel() >= ncell:
        return _torch_soil_temperature_step(
            state,
            static,
            surface_targets,
            dt,
            device=device,
            dtype=dtype,
            workspace=workspace,
            allow_numpy_thermal_fallback=allow_numpy_thermal_fallback,
            snow_enabled=True,
        )

    base_input_state = _state_with_inactive_snow_layers_torch(state, device=device, dtype=dtype)
    base_result, base_state = _torch_soil_temperature_step(
        base_input_state,
        static,
        surface_targets,
        dt,
        device=device,
        dtype=dtype,
        workspace=workspace,
        allow_numpy_thermal_fallback=False,
        snow_enabled=True,
    )

    active_state_in = _cell_index_coupled_state(state, active_idx)
    active_static = _cell_index_static_parameters(static, active_idx, ncell=ncell)
    active_targets = {
        name: _cell_index_select(surface_targets[name], active_idx, ncell=ncell)
        for name in ("hs_soil", "hs_top_snow", "hs_h2osfc", "dhsdT")
        if name in surface_targets
    }
    active_result, active_state = _torch_soil_temperature_step(
        active_state_in,
        active_static,
        active_targets,
        dt,
        device=device,
        dtype=dtype,
        workspace=None,
        allow_numpy_thermal_fallback=allow_numpy_thermal_fallback,
        snow_enabled=True,
    )
    return (
        _merge_soil_temperature_result_torch(base_result, base_state, active_idx, active_result),
        _cell_scatter_coupled_state(base_state, active_idx, active_state),
    )


def coupled_step(
    bundle: SurfaceModelBundle,
    surface_features: np.ndarray | torch.Tensor,
    state: CoupledColumnState,
    static: CoupledStaticParameters,
    dt: float,
    *,
    hydrology_backend: str = "torch",
    soil_temperature_backend: str = "torch",
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
    surface_target_overrides: Mapping[str, np.ndarray | float] | None = None,
    hydrology_workspace: SoilHydrologyTorchWorkspace | None = None,
    soil_temperature_workspace: SoilTemperatureTorchWorkspace | None = None,
    profile_sections: dict[str, float] | None = None,
    snow_model_enabled: bool = True,
    snow_scalar_only: bool = False,
    copy_surface_features: bool = True,
    snow_state_already_disabled: bool = False,
    precomputed_surface_targets: "dict[str, np.ndarray | torch.Tensor] | None" = None,
    precomputed_canopy_state: "CoupledColumnState | None" = None,
) -> CoupledStepResult:
    """Run one coupled step.

    Parameters
    ----------
    surface_features
        Surface-emulator feature tensor in the exact bundle input order,
        shaped ``(..., nfeature)``.
    state
        Dynamic soil/surface-water state for the same leading batch shape.
    static
        Texture-derived static parameter objects for both backends.
    """
    if device is None:
        device = torch.device("cpu")

    def _profiled(name: str, fn):
        if profile_sections is None:
            return fn()
        synchronize(device)
        t0 = time.perf_counter()
        out = fn()
        synchronize(device)
        profile_sections[name] = profile_sections.get(name, 0.0) + (time.perf_counter() - t0)
        return out

    working_state = state if (snow_model_enabled or snow_state_already_disabled) else _zero_snow_state(state)
    if isinstance(surface_features, torch.Tensor):
        working_features = surface_features.clone() if copy_surface_features else surface_features
    else:
        working_features = np.array(surface_features, dtype=np.float64, copy=copy_surface_features)
    if not snow_model_enabled:
        idx_map = _bundle_input_index_map(tuple(bundle.input_names))
        rain_idx = idx_map.get("forc_rain")
        snow_idx = idx_map.get("forc_snow")
        if rain_idx is not None and snow_idx is not None:
            working_features[..., rain_idx] = working_features[..., rain_idx] + working_features[..., snow_idx]
            working_features[..., snow_idx] = 0.0

    if precomputed_surface_targets is not None:
        # Blended-patch mode: per-patch canopy interception, MLP inference, and
        # canopy evaporation have already been run and blended by area weight by
        # the caller.  Skip directly to snow, hydrology, and soil temperature.
        surface_targets = dict(precomputed_surface_targets)
        canopy_state = precomputed_canopy_state if precomputed_canopy_state is not None else working_state
    else:
        canopy_precip_targets, canopy_interception_state = _profiled(
            "canopy_interception",
            lambda: apply_canopy_interception(
                bundle,
                working_features,
                working_state,
                dt,
            ),
        )
        idx_map = _bundle_input_index_map(tuple(bundle.input_names))
        h2ocan_idx = idx_map.get("h2ocan")
        if h2ocan_idx is not None:
            if isinstance(working_features, torch.Tensor):
                working_features[..., h2ocan_idx] = _to_torch(canopy_interception_state.h2ocan, device, dtype)
            else:
                working_features[..., h2ocan_idx] = np.asarray(canopy_interception_state.h2ocan, dtype=np.float64)

        surface_targets = _profiled(
            "predict_surface_targets",
            lambda: predict_surface_targets(
                bundle,
                working_features,
                device=device,
                dtype=dtype,
            ),
        )
        if surface_target_overrides is not None:
            first_target = next(iter(surface_targets.values()))
            batch_shape = tuple(first_target.shape)
            for name, value in surface_target_overrides.items():
                if name.startswith("_"):
                    continue
                if isinstance(first_target, torch.Tensor):
                    arr_t = _to_torch(value, device, dtype)
                    target_shape = batch_shape + tuple(arr_t.shape[len(batch_shape):])
                    if arr_t.shape == target_shape:
                        surface_targets[name] = arr_t
                    else:
                        surface_targets[name] = torch.broadcast_to(arr_t, target_shape)
                else:
                    arr = np.asarray(value, dtype=np.float64)
                    target_shape = batch_shape + tuple(arr.shape[len(batch_shape):])
                    if arr.shape == target_shape:
                        surface_targets[name] = arr
                    else:
                        surface_targets[name] = np.broadcast_to(arr, target_shape)
        if "hs_top_snow" not in surface_targets:
            if isinstance(next(iter(surface_targets.values())), torch.Tensor):
                surface_targets["hs_top_snow"] = surface_targets["hs_soil"]
            else:
                surface_targets["hs_top_snow"] = np.asarray(surface_targets["hs_soil"], dtype=np.float64)
        if not snow_model_enabled and "qflx_ev_snow" in surface_targets:
            if isinstance(surface_targets["qflx_ev_snow"], torch.Tensor):
                surface_targets["qflx_ev_snow"] = torch.zeros_like(surface_targets["qflx_ev_snow"])
            else:
                surface_targets["qflx_ev_snow"] = np.zeros_like(np.asarray(surface_targets["qflx_ev_snow"], dtype=np.float64))
        _ensure_surface_targets(bundle, surface_targets)
        surface_targets.update(canopy_precip_targets)
        if surface_target_overrides is not None:
            first_target = next(iter(surface_targets.values()))
            batch_shape = tuple(first_target.shape)
            for name in ("qflx_prec_grnd", "qflx_snow_grnd", "qflx_rain_grnd", "qflx_leafdrip"):
                if name not in surface_target_overrides:
                    continue
                value = surface_target_overrides[name]
                if isinstance(first_target, torch.Tensor):
                    arr_t = _to_torch(value, device, dtype)
                    target_shape = batch_shape + tuple(arr_t.shape[len(batch_shape):])
                    surface_targets[name] = arr_t if arr_t.shape == target_shape else torch.broadcast_to(arr_t, target_shape)
                else:
                    arr = np.asarray(value, dtype=np.float64)
                    target_shape = batch_shape + tuple(arr.shape[len(batch_shape):])
                    surface_targets[name] = arr if arr.shape == target_shape else np.broadcast_to(arr, target_shape)
            if "_h2ocan_after_interception" in surface_target_overrides:
                value = surface_target_overrides["_h2ocan_after_interception"]
                if isinstance(first_target, torch.Tensor):
                    h2ocan_after_interception = _to_torch(value, device, dtype)
                else:
                    h2ocan_after_interception = np.asarray(value, dtype=np.float64)
                canopy_interception_state = replace(canopy_interception_state, h2ocan=h2ocan_after_interception)
        surface_targets, canopy_state = _profiled(
            "canopy_evaporation",
            lambda: apply_canopy_evaporation(
                surface_targets,
                canopy_interception_state,
                dt,
                device=device,
                dtype=dtype,
            ),
        )
    if snow_model_enabled:
        snow_state = _profiled(
            "update_snow_state",
            lambda: update_snow_state(
                canopy_state,
                bundle,
                working_features,
                surface_targets,
                dt,
                static=static,
                scalar_snow_only=snow_scalar_only,
            ),
        )
    else:
        snow_state = canopy_state if snow_state_already_disabled else _zero_snow_state(canopy_state)
    if snow_model_enabled:
        if isinstance(snow_state.h2osno, torch.Tensor):
            snow_state = _repair_snow_fraction_depth_torch(snow_state, device=device, dtype=dtype)
        else:
            snow_state = _repair_snow_fraction_depth_numpy(snow_state)
        if snow_scalar_only:
            snow_state = (
                force_scalar_snow_state_torch(snow_state, device=device, dtype=dtype)
                if isinstance(snow_state.h2osno, torch.Tensor)
                else force_scalar_snow_state_numpy(snow_state)
            )
    use_torch_targets = any(isinstance(v, torch.Tensor) for v in surface_targets.values())
    if not use_torch_targets or hydrology_backend in ("numpy", "numpy_full") or soil_temperature_backend == "numpy":
        surface_targets = {
            name: _surface_target_to_numpy(value)
            for name, value in surface_targets.items()
        }
    if soil_temperature_backend == "numpy":
        soil_result, final_state = _profiled(
            "soil_temperature_numpy",
            lambda: _numpy_soil_temperature_step(
                snow_state,
                static,
                surface_targets,
                dt,
                snow_enabled=snow_model_enabled,
            ),
        )
    elif soil_temperature_backend == "torch":
        soil_result, final_state = _profiled(
            "soil_temperature_torch",
            lambda: _torch_soil_temperature_step_grouped_snow(
                snow_state,
                static,
                surface_targets,
                dt,
                device=device,
                dtype=dtype,
                workspace=soil_temperature_workspace,
                allow_numpy_thermal_fallback=False,
                snow_enabled=snow_model_enabled,
            ),
        )
    else:
        raise ValueError(f"Unsupported soil-temperature backend: {soil_temperature_backend}")

    thermal_state = final_state
    if snow_model_enabled:
        if not isinstance(final_state.h2osno, torch.Tensor):
            final_state = _repair_snow_fraction_depth_numpy(final_state)
            thermal_state = final_state
        else:
            final_state = _repair_snow_fraction_depth_torch(final_state, device=device, dtype=dtype)
            thermal_state = final_state
        if snow_scalar_only:
            final_state = (
                force_scalar_snow_state_torch(final_state, device=device, dtype=dtype)
                if isinstance(final_state.h2osno, torch.Tensor)
                else force_scalar_snow_state_numpy(final_state)
            )
            thermal_state = final_state
    snow_melt = None if not snow_model_enabled else getattr(soil_result, "qflx_snow_melt", None)
    scalar_snow_melt = None if not snow_model_enabled else getattr(soil_result, "qflx_scalar_snow_melt", None)
    if snow_model_enabled:
        combined_snow_melt = snow_melt
        if scalar_snow_melt is not None:
            if combined_snow_melt is None:
                combined_snow_melt = scalar_snow_melt
            elif isinstance(combined_snow_melt, torch.Tensor) or isinstance(scalar_snow_melt, torch.Tensor):
                combined_snow_melt = _to_torch(combined_snow_melt, device, dtype) + _to_torch(
                    scalar_snow_melt,
                    device,
                    dtype,
                )
            else:
                combined_snow_melt = np.asarray(combined_snow_melt, dtype=np.float64) + np.asarray(
                    scalar_snow_melt,
                    dtype=np.float64,
                )
        final_state = _apply_post_melt_snow_fraction_update(
            final_state,
            combined_snow_melt,
            dt,
            static,
            device=device,
            dtype=dtype,
        )
        if snow_scalar_only:
            final_state = (
                force_scalar_snow_state_torch(final_state, device=device, dtype=dtype)
                if isinstance(final_state.h2osno, torch.Tensor)
                else force_scalar_snow_state_numpy(final_state)
            )
    qflx_snow_to_soil = None
    snow_water_result = None
    snow_rain_frac = None
    snow_rain_flux = None
    rain_to_snowpack = None
    snow_water_needed = False
    if snow_model_enabled:
        rain_source = surface_targets.get("qflx_rain_grnd")
        if rain_source is None:
            rain_source = surface_targets["qflx_prec_grnd"]
            if "qflx_snow_grnd" in surface_targets:
                if isinstance(rain_source, torch.Tensor) or isinstance(surface_targets["qflx_snow_grnd"], torch.Tensor):
                    rain_source = torch.clamp(
                        _to_torch(rain_source, device, dtype) - _to_torch(surface_targets["qflx_snow_grnd"], device, dtype),
                        min=0.0,
                    )
                else:
                    rain_source = np.maximum(
                        np.asarray(rain_source, dtype=np.float64) - np.asarray(surface_targets["qflx_snow_grnd"], dtype=np.float64),
                        0.0,
                    )
        if isinstance(rain_source, torch.Tensor) or isinstance(final_state.frac_sno, torch.Tensor):
            snow_rain_flux = _to_torch(rain_source, device, dtype)
            snow_rain_frac = _active_snow_fraction_torch(final_state, device=device, dtype=dtype)
            rain_to_snowpack = snow_rain_flux * snow_rain_frac * dt
            h2osno_now = (
                _to_torch(final_state.h2osno, device, dtype)
                if final_state.h2osno is not None
                else torch.zeros_like(snow_rain_flux)
            )
            qflx_snow_to_soil = torch.zeros_like(snow_rain_flux)
            if scalar_snow_melt is not None:
                qflx_snow_to_soil = qflx_snow_to_soil + torch.clamp(
                    _snow_batch_like_torch(_to_torch(scalar_snow_melt, device, dtype), tuple(qflx_snow_to_soil.shape), device=device, dtype=dtype),
                    min=0.0,
                ) * dt
            snow_water_needed = bool(
                torch.any((h2osno_now > 1.0e-8) & (snow_rain_frac > 0.0)).detach().cpu().item()
                or torch.any(rain_to_snowpack > 0.0).detach().cpu().item()
            )
        else:
            snow_rain_flux = np.asarray(rain_source, dtype=np.float64)
            snow_rain_frac = _active_snow_fraction_numpy(final_state)
            rain_to_snowpack = snow_rain_flux * snow_rain_frac * dt
            h2osno_now = (
                _to_np(final_state.h2osno)
                if final_state.h2osno is not None
                else np.zeros_like(snow_rain_flux, dtype=np.float64)
            )
            qflx_snow_to_soil = np.zeros_like(snow_rain_flux, dtype=np.float64)
            if scalar_snow_melt is not None:
                qflx_snow_to_soil = qflx_snow_to_soil + np.maximum(
                    _snow_batch_like_numpy(np.asarray(scalar_snow_melt, dtype=np.float64), qflx_snow_to_soil.shape),
                    0.0,
                ) * dt
            snow_water_needed = bool(
                np.any((h2osno_now > 1.0e-8) & (snow_rain_frac > 0.0))
                or np.any(rain_to_snowpack > 0.0)
            )
    if snow_model_enabled and snow_water_needed:
        # The thermal solver (_snow_soil_temperature_step) already converted
        # melted ice to liquid inside result.state.snow_layers and reduced
        # h2osno accordingly.  Pass melt=None so apply_snow_water_and_capping
        # just routes the existing excess liquid downward and drains to soil,
        # without double-counting the ice→liq conversion.
        snow_water_result = _profiled(
            "snow_water",
            lambda: apply_snow_water_and_capping(
                final_state,
                None,
                rain_to_snowpack=rain_to_snowpack,
                device=device,
                dtype=dtype,
            ),
        )
        final_state = snow_water_result.state
        if final_state.h2osfc is not None:
            if isinstance(final_state.h2osfc, torch.Tensor):
                final_state = CoupledColumnState(
                    h2osoi_liq=final_state.h2osoi_liq,
                    h2osoi_ice=final_state.h2osoi_ice,
                    h2ocan=final_state.h2ocan,
                    h2osfc=final_state.h2osfc,
                    frac_h2osfc=frac_h2osfc_from_storage_torch(final_state.h2osfc, static.micro_sigma, device=device, dtype=dtype),
                    frac_sno=final_state.frac_sno,
                    zwt=final_state.zwt,
                    wa=final_state.wa,
                    t_soisno=final_state.t_soisno,
                    t_h2osfc=final_state.t_h2osfc,
                    t10=final_state.t10,
                    t10_history=final_state.t10_history,
                    h2osno=final_state.h2osno,
                    snow_depth=final_state.snow_depth,
                    snow_t=final_state.snow_t,
                    snow_layers=final_state.snow_layers,
                    int_snow=final_state.int_snow,
                )
            else:
                final_state = CoupledColumnState(
                    h2osoi_liq=final_state.h2osoi_liq,
                    h2osoi_ice=final_state.h2osoi_ice,
                    h2ocan=final_state.h2ocan,
                    h2osfc=final_state.h2osfc,
                    frac_h2osfc=frac_h2osfc_from_storage(final_state.h2osfc, static.micro_sigma),
                    frac_sno=final_state.frac_sno,
                    zwt=final_state.zwt,
                    wa=final_state.wa,
                    t_soisno=final_state.t_soisno,
                    t_h2osfc=final_state.t_h2osfc,
                    t10=final_state.t10,
                    t10_history=final_state.t10_history,
                    h2osno=final_state.h2osno,
                    snow_depth=final_state.snow_depth,
                    snow_t=final_state.snow_t,
                    snow_layers=final_state.snow_layers,
                    int_snow=final_state.int_snow,
                )
        if isinstance(snow_water_result.qflx_snow_to_soil, torch.Tensor):
            qflx_snow_to_soil = (
                _to_torch(qflx_snow_to_soil, device, dtype)
                if qflx_snow_to_soil is not None
                else torch.zeros_like(snow_water_result.qflx_snow_to_soil)
            ) + snow_water_result.qflx_snow_to_soil
        else:
            qflx_snow_to_soil = (
                _to_np(qflx_snow_to_soil)
                if qflx_snow_to_soil is not None
                else np.zeros_like(snow_water_result.qflx_snow_to_soil, dtype=np.float64)
            ) + snow_water_result.qflx_snow_to_soil
        if isinstance(final_state.h2osno, torch.Tensor):
            final_state = _repair_snow_fraction_depth_torch(final_state, device=device, dtype=dtype)
        else:
            final_state = _repair_snow_fraction_depth_numpy(final_state)
    if snow_model_enabled:
        if isinstance(final_state.h2osno, torch.Tensor) or isinstance(final_state.h2osfc, torch.Tensor):
            final_state, inactive_snow_to_soil = _drain_inactive_snow_storage_torch(
                final_state,
                device=device,
                dtype=dtype,
            )
            qflx_snow_to_soil = (
                _to_torch(qflx_snow_to_soil, device, dtype)
                if qflx_snow_to_soil is not None
                else torch.zeros_like(inactive_snow_to_soil)
            ) + inactive_snow_to_soil
        else:
            final_state, inactive_snow_to_soil = _drain_inactive_snow_storage_numpy(final_state)
            qflx_snow_to_soil = (
                _to_np(qflx_snow_to_soil)
                if qflx_snow_to_soil is not None
                else np.zeros_like(inactive_snow_to_soil, dtype=np.float64)
            ) + inactive_snow_to_soil
        if snow_scalar_only:
            final_state = (
                force_scalar_snow_state_torch(final_state, device=device, dtype=dtype)
                if isinstance(final_state.h2osno, torch.Tensor)
                else force_scalar_snow_state_numpy(final_state)
            )
        # qflx_snow_to_soil is kg/m² (mass per step); convert to kg/m²/s for
        # the hydrology top-soil flux. Native ELM sends only bare-fraction
        # rain directly to soil; snowfall is already in h2osno, and rain over
        # snow first drains through SnowWater.
        if isinstance(surface_targets["qflx_prec_grnd"], torch.Tensor):
            snow_frac = _to_torch(snow_rain_frac, device, dtype)
            surface_targets["qflx_prec_grnd"] = torch.clamp(1.0 - snow_frac, min=0.0, max=1.0) * _to_torch(
                snow_rain_flux,
                device,
                dtype,
            ) + _to_torch(
                qflx_snow_to_soil,
                device,
                dtype,
            ) / dt
        else:
            snow_frac = np.clip(np.asarray(snow_rain_frac, dtype=np.float64), 0.0, 1.0)
            surface_targets["qflx_prec_grnd"] = (1.0 - snow_frac) * np.asarray(snow_rain_flux, dtype=np.float64) + _to_np(
                qflx_snow_to_soil,
            ) / dt

    if hydrology_backend == "numpy":
        hydrology_result, hydro_state = _profiled(
            "hydrology_numpy",
            lambda: _numpy_hydrology_step(final_state, static, surface_targets, dt),
        )
    elif hydrology_backend == "numpy_full":
        hydrology_result, hydro_state = _profiled(
            "hydrology_numpy_full",
            lambda: _numpy_hydrology_step_full(
                final_state,
                static,
                surface_targets,
                dt,
            ),
        )
    elif hydrology_backend == "torch":
        hydrology_result, hydro_state = _profiled(
            "hydrology_torch",
            lambda: _torch_hydrology_step(
                final_state,
                static,
                surface_targets,
                dt,
                device=device,
                dtype=dtype,
                workspace=hydrology_workspace,
            ),
        )
    elif hydrology_backend == "full":
        hydrology_result, hydro_state = _profiled(
            "hydrology_torch_full",
            lambda: _torch_hydrology_step_full(
                final_state,
                static,
                surface_targets,
                dt,
                device=device,
                dtype=dtype,
                workspace=hydrology_workspace,
                profile_sections=profile_sections,
            ),
        )
    else:
        raise ValueError(f"Unsupported hydrology backend: {hydrology_backend}")

    _t10_ref: np.ndarray | torch.Tensor | None = surface_targets.get("t_ref2m")
    _forc_t_ref: np.ndarray | torch.Tensor | None = surface_targets.get("forc_t")
    if _forc_t_ref is None and "forc_t" in bundle.input_names:
        _forc_t_ref = surface_features[..., bundle.input_names.index("forc_t")]
    if _t10_ref is None and _forc_t_ref is not None:
        # Fall back to air temperature when the MLP does not predict t_ref2m.
        _t10_ref = _forc_t_ref
    if _t10_ref is not None:
        if _forc_t_ref is None:
            if "forc_t" in bundle.input_names:
                _forc_t_ref = surface_features[..., bundle.input_names.index("forc_t")]
            else:
                _forc_t_ref = _t10_ref
        _t10_ref = _sanitize_running_mean_reference(_t10_ref, _forc_t_ref)
        if isinstance(final_state.t_soisno, torch.Tensor):
            t10, t10_history = update_t10_state_torch(final_state, _t10_ref, dt, device=device, dtype=dtype)
        else:
            t10, t10_history = update_t10_state(final_state, _t10_ref, dt)
        final_state = CoupledColumnState(
            h2osoi_liq=final_state.h2osoi_liq,
            h2osoi_ice=final_state.h2osoi_ice,
            h2ocan=final_state.h2ocan,
            h2osfc=final_state.h2osfc,
            frac_h2osfc=final_state.frac_h2osfc,
            frac_sno=final_state.frac_sno,
            zwt=final_state.zwt,
            wa=final_state.wa,
            t_soisno=final_state.t_soisno,
            t_h2osfc=final_state.t_h2osfc,
            t10=t10,
            t10_history=t10_history,
            h2osno=final_state.h2osno,
            snow_depth=final_state.snow_depth,
            snow_t=final_state.snow_t,
            snow_layers=final_state.snow_layers,
            int_snow=final_state.int_snow,
        )
    if snow_model_enabled:
        if isinstance(final_state.h2osno, torch.Tensor):
            final_state = _repair_snow_fraction_depth_torch(final_state, device=device, dtype=dtype)
        else:
            final_state = _repair_snow_fraction_depth_numpy(final_state)
        if snow_scalar_only:
            final_state = (
                force_scalar_snow_state_torch(final_state, device=device, dtype=dtype)
                if isinstance(final_state.h2osno, torch.Tensor)
                else force_scalar_snow_state_numpy(final_state)
            )

    if not snow_model_enabled and snow_state_already_disabled:
        h2osno_final = working_state.h2osno
        snow_depth_final = working_state.snow_depth
        frac_sno_final = working_state.frac_sno
        snow_t_final = None
        snow_layers_final = None
        int_snow_final = working_state.int_snow
    else:
        snow_depth_final = final_state.snow_depth
        if snow_depth_final is None:
            snow_depth_final = (
                snow_depth_from_storage_torch(final_state.h2osno, device=device, dtype=dtype)
                if isinstance(final_state.h2osno, torch.Tensor)
                else snow_depth_from_storage(final_state.h2osno)
            )
        if not snow_model_enabled:
            final_state = _zero_snow_state(final_state)
        h2osno_final = final_state.h2osno
        frac_sno_final = final_state.frac_sno
        snow_t_final = final_state.snow_t
        snow_layers_final = final_state.snow_layers
        int_snow_final = final_state.int_snow
    if h2osno_final is not None:
        if isinstance(h2osno_final, torch.Tensor):
            snow_mass_final = _to_torch(h2osno_final, device, dtype)
            snow_depth_final = torch.where(snow_mass_final > 1.0e-8, _to_torch(snow_depth_final, device, dtype), torch.zeros_like(snow_mass_final))
            if frac_sno_final is not None:
                frac_sno_final = torch.where(
                    snow_mass_final > 1.0e-8,
                    torch.clamp(_to_torch(frac_sno_final, device, dtype), min=0.0, max=1.0),
                    torch.zeros_like(snow_mass_final),
                )
            else:
                frac_sno_final = frac_sno_from_snow_storage_torch(snow_mass_final, snow_depth_final, device=device, dtype=dtype)
            if int_snow_final is not None:
                int_snow_final = torch.where(
                    snow_mass_final > 1.0e-8,
                    torch.maximum(_to_torch(int_snow_final, device, dtype), snow_mass_final),
                    torch.zeros_like(snow_mass_final),
                )
        else:
            snow_mass_final = np.asarray(h2osno_final, dtype=np.float64)
            snow_depth_final = np.where(snow_mass_final > 1.0e-8, np.asarray(snow_depth_final, dtype=np.float64), 0.0)
            if frac_sno_final is not None:
                frac_sno_final = np.clip(
                    np.where(snow_mass_final > 1.0e-8, np.asarray(frac_sno_final, dtype=np.float64), 0.0),
                    0.0,
                    1.0,
                )
            else:
                frac_sno_final = frac_sno_from_snow_storage(snow_mass_final, snow_depth_final)
            if int_snow_final is not None:
                int_snow_final = np.where(
                    snow_mass_final > 1.0e-8,
                    np.maximum(np.asarray(int_snow_final, dtype=np.float64), snow_mass_final),
                    0.0,
                )
    else:
        frac_sno_final = (
                frac_sno_final
                if frac_sno_final is not None
                else (
                    torch.zeros_like(_to_torch(snow_depth_final, device, dtype))
                    if isinstance(snow_depth_final, torch.Tensor)
                    else np.zeros_like(np.asarray(snow_depth_final, dtype=np.float64))
            )
        )

    if snow_model_enabled and h2osno_final is not None:
        if isinstance(h2osno_final, torch.Tensor):
            snow_mass_final_t = _to_torch(h2osno_final, device, dtype)
            snow_depth_final_t = _to_torch(snow_depth_final, device, dtype)
            frac_sno_final = torch.where(
                snow_mass_final_t > 1.0e-8,
                _to_torch(frac_sno_final, device, dtype),
                torch.zeros_like(snow_mass_final_t),
            )
            snow_depth_final = torch.where(
                snow_mass_final_t > 1.0e-8,
                snow_depth_final_t,
                torch.zeros_like(snow_depth_final_t),
            )
            if int_snow_final is not None:
                int_snow_final = torch.where(
                    snow_mass_final_t > 1.0e-8,
                    torch.maximum(_to_torch(int_snow_final, device, dtype), snow_mass_final_t),
                    torch.zeros_like(snow_mass_final_t),
                )
        else:
            snow_mass_final_np = np.asarray(h2osno_final, dtype=np.float64)
            snow_depth_final_np = np.asarray(snow_depth_final, dtype=np.float64)
            frac_sno_final = np.where(
                snow_mass_final_np > 1.0e-8,
                np.asarray(frac_sno_final, dtype=np.float64),
                0.0,
            )
            snow_depth_final = np.where(snow_mass_final_np > 1.0e-8, snow_depth_final_np, 0.0)
            if int_snow_final is not None:
                int_snow_final = np.where(
                    snow_mass_final_np > 1.0e-8,
                    np.maximum(np.asarray(int_snow_final, dtype=np.float64), snow_mass_final_np),
                    0.0,
                )

    return CoupledStepResult(
        surface_targets=surface_targets,
        hydrology_backend=hydrology_backend,
        soil_temperature_backend=soil_temperature_backend,
        hydrology_result=hydrology_result,
        soil_temperature_result=soil_result,
        qflx_snow_to_soil=qflx_snow_to_soil,
        state=CoupledColumnState(
            h2osoi_liq=hydro_state.h2osoi_liq,
            h2osoi_ice=hydro_state.h2osoi_ice,
            h2ocan=hydro_state.h2ocan,
            h2osfc=hydro_state.h2osfc,
            frac_h2osfc=hydro_state.frac_h2osfc,
            zwt=hydro_state.zwt,
            wa=hydro_state.wa,
            t_soisno=final_state.t_soisno,
            t_h2osfc=final_state.t_h2osfc,
            t10=final_state.t10,
            t10_history=final_state.t10_history,
            h2osno=h2osno_final,
            snow_depth=snow_depth_final,
            frac_sno=frac_sno_final,
            snow_t=snow_t_final,
            snow_layers=snow_layers_final,
            int_snow=int_snow_final,
        ),
        canopy_state=canopy_state,
        snow_state=snow_state,
        thermal_state=thermal_state,
        snow_water_result=snow_water_result,
    )


def build_demo_texture_and_root_profile(nlev: int) -> tuple[SoilTextureProfile, np.ndarray]:
    dz, z, zi = elm_soil_grid(nlevsoi=nlev)
    beta = 0.966
    zi_cm_full = np.concatenate([[0.0], zi * 100.0])
    rootfr = beta ** zi_cm_full[:-1] - beta ** zi_cm_full[1:]
    rootfr = np.maximum(rootfr, 0.0)
    rootfr /= rootfr.sum()
    z_ctrl = np.array([0.0, 0.30, 1.00, 3.80], dtype=np.float64)
    sand_ctrl = np.array([58.0, 50.0, 40.0, 32.0], dtype=np.float64)
    clay_ctrl = np.array([12.0, 20.0, 28.0, 36.0], dtype=np.float64)
    org_ctrl = np.array([0.08, 0.02, 0.005, 0.0], dtype=np.float64)
    texture = SoilTextureProfile(
        sand_pct=np.interp(z, z_ctrl, sand_ctrl),
        clay_pct=np.interp(z, z_ctrl, clay_ctrl),
        organic_frac=np.interp(z, z_ctrl, org_ctrl),
    )
    return texture, rootfr


def build_demo_state(ncell: int, nens: int, nlev: int) -> CoupledColumnState:
    dz, _, _ = elm_soil_grid(nlevsoi=nlev)
    shape = (ncell, nens)
    t0 = np.full(shape, 285.0, dtype=np.float64)
    snow_layers = _snow_layers_from_storage_numpy(
        np.zeros(shape, dtype=np.float64),
        np.zeros(shape, dtype=np.float64),
        snow_t=np.full(shape + (MAX_SNOW_LAYERS,), 273.15, dtype=np.float64),
    )
    return CoupledColumnState(
        h2osoi_liq=np.broadcast_to(0.30 * dz * 1000.0, shape + (nlev,)).copy(),
        h2osoi_ice=np.zeros(shape + (nlev,), dtype=np.float64),
        h2ocan=np.full(shape, 0.05, dtype=np.float64),
        h2osfc=np.full(shape, 2.0, dtype=np.float64),
        frac_h2osfc=frac_h2osfc_from_storage(np.full(shape, 2.0, dtype=np.float64), 0.03),
        frac_sno=np.zeros(shape, dtype=np.float64),
        zwt=np.full(shape, 4.0, dtype=np.float64),
        t_soisno=np.full(shape + (nlev,), 285.0, dtype=np.float64),
        t_h2osfc=np.full(shape, 285.0, dtype=np.float64),
        t10=t0,
        t10_history=np.broadcast_to(t0[..., None], shape + (240,)).copy(),
        h2osno=np.zeros(shape, dtype=np.float64),
        snow_depth=np.zeros(shape, dtype=np.float64),
        snow_t=np.full(shape + (5,), 273.15, dtype=np.float64),
        snow_layers=snow_layers,
        int_snow=np.zeros(shape, dtype=np.float64),
    )


def slice_coupled_state(state: CoupledColumnState, start: int, stop: int) -> CoupledColumnState:
    """Slice the leading batch axis of a coupled state."""
    def sl(x: np.ndarray | torch.Tensor | None):
        if x is None:
            return None
        return x[start:stop]

    return CoupledColumnState(
        h2osoi_liq=sl(state.h2osoi_liq),
        h2osoi_ice=sl(state.h2osoi_ice),
        h2ocan=sl(state.h2ocan),
        h2osfc=sl(state.h2osfc),
        frac_h2osfc=sl(state.frac_h2osfc),
        frac_sno=sl(state.frac_sno),
        zwt=sl(state.zwt),
        t_soisno=sl(state.t_soisno),
        wa=sl(state.wa),
        t_h2osfc=sl(state.t_h2osfc),
        t10=sl(state.t10),
        t10_history=sl(state.t10_history),
        h2osno=sl(state.h2osno),
        snow_depth=sl(state.snow_depth),
        snow_t=sl(state.snow_t),
        snow_layers=_snow_layers_slice(state.snow_layers, start, stop),
        int_snow=sl(state.int_snow),
    )


def coupled_state_to_device(state: CoupledColumnState, device: torch.device, dtype: torch.dtype) -> CoupledColumnState:
    def tensor_copy(x):
        return torch.as_tensor(np.array(x, copy=True) if isinstance(x, np.ndarray) else x, device=device, dtype=dtype)

    return CoupledColumnState(
        h2osoi_liq=None if state.h2osoi_liq is None else tensor_copy(state.h2osoi_liq),
        h2osoi_ice=None if state.h2osoi_ice is None else tensor_copy(state.h2osoi_ice),
        h2ocan=None if state.h2ocan is None else tensor_copy(state.h2ocan),
        h2osfc=None if state.h2osfc is None else tensor_copy(state.h2osfc),
        frac_h2osfc=None if state.frac_h2osfc is None else tensor_copy(state.frac_h2osfc),
        frac_sno=None if state.frac_sno is None else tensor_copy(state.frac_sno),
        zwt=None if state.zwt is None else tensor_copy(state.zwt),
        t_soisno=None if state.t_soisno is None else tensor_copy(state.t_soisno),
        wa=None if state.wa is None else tensor_copy(state.wa),
        t_h2osfc=None if state.t_h2osfc is None else tensor_copy(state.t_h2osfc),
        t10=None if state.t10 is None else tensor_copy(state.t10),
        t10_history=None if state.t10_history is None else tensor_copy(state.t10_history),
        h2osno=None if state.h2osno is None else tensor_copy(state.h2osno),
        snow_depth=None if state.snow_depth is None else tensor_copy(state.snow_depth),
        snow_t=None if state.snow_t is None else tensor_copy(state.snow_t),
        snow_layers=_snow_layers_to_device(state.snow_layers, device, dtype),
        int_snow=None if state.int_snow is None else tensor_copy(state.int_snow),
    )


def coupled_state_to_cpu(state: CoupledColumnState) -> CoupledColumnState:
    def arr(x):
        if x is None:
            return None
        return np.asarray(x.detach().cpu().numpy(), dtype=np.float64) if isinstance(x, torch.Tensor) else np.asarray(x, dtype=np.float64)

    return CoupledColumnState(
        h2osoi_liq=arr(state.h2osoi_liq),
        h2osoi_ice=arr(state.h2osoi_ice),
        h2ocan=arr(state.h2ocan),
        h2osfc=arr(state.h2osfc),
        frac_h2osfc=arr(state.frac_h2osfc),
        frac_sno=arr(state.frac_sno),
        zwt=arr(state.zwt),
        t_soisno=arr(state.t_soisno),
        wa=arr(state.wa),
        t_h2osfc=arr(state.t_h2osfc),
        t10=arr(state.t10),
        t10_history=arr(state.t10_history),
        h2osno=arr(state.h2osno),
        snow_depth=arr(state.snow_depth),
        snow_t=arr(state.snow_t),
        snow_layers=_snow_layers_to_cpu(state.snow_layers),
        int_snow=arr(state.int_snow),
    )


def concatenate_coupled_states(states: list[CoupledColumnState]) -> CoupledColumnState:
    if not states:
        raise ValueError("No coupled states to concatenate.")

    def cat(field: str):
        values = [getattr(state, field) for state in states if getattr(state, field) is not None]
        if not values:
            return None
        first = values[0]
        if isinstance(first, torch.Tensor):
            return torch.cat(values, dim=0)
        return np.concatenate([np.asarray(value, dtype=np.float64) for value in values], axis=0)

    return CoupledColumnState(
        h2osoi_liq=cat("h2osoi_liq"),
        h2osoi_ice=cat("h2osoi_ice"),
        h2ocan=cat("h2ocan"),
        h2osfc=cat("h2osfc"),
        frac_h2osfc=cat("frac_h2osfc"),
        frac_sno=cat("frac_sno"),
        zwt=cat("zwt"),
        t_soisno=cat("t_soisno"),
        wa=cat("wa"),
        t_h2osfc=cat("t_h2osfc"),
        t10=cat("t10"),
        t10_history=cat("t10_history"),
        h2osno=cat("h2osno"),
        snow_depth=cat("snow_depth"),
        snow_t=cat("snow_t"),
        snow_layers=_snow_layers_cat([state.snow_layers for state in states]),
        int_snow=cat("int_snow"),
    )


def build_demo_atmospheric_forcing(batch_shape: tuple[int, ...]) -> AtmosphericForcingInputs:
    return AtmosphericForcingInputs(
        forc_t=np.full(batch_shape, 285.0, dtype=np.float64),
        forc_q=np.full(batch_shape, 0.008, dtype=np.float64),
        forc_pbot=np.full(batch_shape, 101325.0, dtype=np.float64),
        forc_lwrad=np.full(batch_shape, 320.0, dtype=np.float64),
        forc_rain=np.full(batch_shape, 1.0e-5, dtype=np.float64),
        forc_snow=np.zeros(batch_shape, dtype=np.float64),
        forc_swrad=np.full(batch_shape, 400.0, dtype=np.float64),
        forc_solad_vis=np.full(batch_shape, 120.0, dtype=np.float64),
        forc_solad_nir=np.full(batch_shape, 120.0, dtype=np.float64),
        forc_solai_vis=np.full(batch_shape, 80.0, dtype=np.float64),
        forc_solai_nir=np.full(batch_shape, 80.0, dtype=np.float64),
        wind_speed=np.full(batch_shape, 2.5, dtype=np.float64),
        forc_rho=np.full(batch_shape, 1.2, dtype=np.float64),
        forc_pco2=np.full(batch_shape, 4.2e-4, dtype=np.float64),
        dayl=np.full(batch_shape, 43200.0, dtype=np.float64),
        max_dayl=np.full(batch_shape, 54000.0, dtype=np.float64),
        coszen_col=np.full(batch_shape, 0.7, dtype=np.float64),
    )


def build_demo_atmospheric_forcing_torch(
    batch_shape: tuple[int, ...],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> AtmosphericForcingInputs:
    return AtmosphericForcingInputs(
        forc_t=torch.full(batch_shape, 285.0, device=device, dtype=dtype),
        forc_q=torch.full(batch_shape, 0.008, device=device, dtype=dtype),
        forc_pbot=torch.full(batch_shape, 101325.0, device=device, dtype=dtype),
        forc_lwrad=torch.full(batch_shape, 320.0, device=device, dtype=dtype),
        forc_rain=torch.full(batch_shape, 1.0e-5, device=device, dtype=dtype),
        forc_snow=torch.zeros(batch_shape, device=device, dtype=dtype),
        forc_swrad=torch.full(batch_shape, 400.0, device=device, dtype=dtype),
        forc_solad_vis=torch.full(batch_shape, 120.0, device=device, dtype=dtype),
        forc_solad_nir=torch.full(batch_shape, 120.0, device=device, dtype=dtype),
        forc_solai_vis=torch.full(batch_shape, 80.0, device=device, dtype=dtype),
        forc_solai_nir=torch.full(batch_shape, 80.0, device=device, dtype=dtype),
        wind_speed=torch.full(batch_shape, 2.5, device=device, dtype=dtype),
        forc_rho=torch.full(batch_shape, 1.2, device=device, dtype=dtype),
        forc_pco2=torch.full(batch_shape, 4.2e-4, device=device, dtype=dtype),
        dayl=torch.full(batch_shape, 43200.0, device=device, dtype=dtype),
        max_dayl=torch.full(batch_shape, 54000.0, device=device, dtype=dtype),
        coszen_col=torch.full(batch_shape, 0.7, device=device, dtype=dtype),
    )


def build_demo_vegetation_surface_inputs(
    state: CoupledColumnState,
) -> VegetationSurfaceInputs:
    batch_shape = state.h2osfc.shape
    if state.t10 is not None:
        t10 = state.t10
    elif state.t10_history is not None:
        t10 = _to_np(state.t10_history).mean(axis=-1)
    else:
        t10 = _to_np(state.t_soisno)[..., 0]
    return VegetationSurfaceInputs(
        frac_veg_nosno_patch=np.ones(batch_shape, dtype=np.float64),
        elai_patch=np.full(batch_shape, 2.0, dtype=np.float64),
        esai_patch=np.full(batch_shape, 0.2, dtype=np.float64),
        snow_depth=np.zeros(batch_shape, dtype=np.float64),
        t10=t10,
        htop_patch=np.full(batch_shape, 1.0, dtype=np.float64),
    )


def build_demo_vegetation_surface_inputs_torch(
    state: CoupledColumnState,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> VegetationSurfaceInputs:
    batch_shape = state.h2osfc.shape
    if state.t10 is not None:
        if isinstance(state.t10, torch.Tensor):
            t10 = state.t10.to(device=device, dtype=dtype)
        else:
            t10 = torch.as_tensor(state.t10, device=device, dtype=dtype)
    elif state.t10_history is not None:
        t10_hist = _to_torch(state.t10_history, device, dtype)
        t10 = t10_hist.mean(dim=-1)
    else:
        t10 = torch.as_tensor(state.t_soisno[..., 0], device=device, dtype=dtype)
    return VegetationSurfaceInputs(
        frac_veg_nosno_patch=torch.ones(batch_shape, device=device, dtype=dtype),
        elai_patch=torch.full(batch_shape, 2.0, device=device, dtype=dtype),
        esai_patch=torch.full(batch_shape, 0.2, device=device, dtype=dtype),
        snow_depth=torch.zeros(batch_shape, device=device, dtype=dtype),
        t10=t10,
        htop_patch=torch.full(batch_shape, 1.0, device=device, dtype=dtype),
    )


def build_demo_surface_geometry(batch_shape: tuple[int, ...]) -> SurfaceGeometryInputs:
    return SurfaceGeometryInputs(
        forc_hgt_u_patch=np.full(batch_shape, 30.0, dtype=np.float64),
        z0mg_col=np.full(batch_shape, 0.05, dtype=np.float64),
        ugust=np.full(batch_shape, 0.1, dtype=np.float64),
    )


def build_demo_surface_geometry_torch(
    batch_shape: tuple[int, ...],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> SurfaceGeometryInputs:
    return SurfaceGeometryInputs(
        forc_hgt_u_patch=torch.full(batch_shape, 30.0, device=device, dtype=dtype),
        z0mg_col=torch.full(batch_shape, 0.05, device=device, dtype=dtype),
        ugust=torch.full(batch_shape, 0.1, device=device, dtype=dtype),
    )


def build_surface_features_from_inputs(
    bundle: SurfaceModelBundle,
    atmosphere: AtmosphericForcingInputs,
    vegetation: VegetationSurfaceInputs,
    geometry: SurfaceGeometryInputs,
    state: CoupledColumnState,
    static: CoupledStaticParameters,
    feature_overrides: Mapping[str, np.ndarray | float] | None = None,
    snow_model_enabled: bool = True,
) -> np.ndarray:
    """Build a physically structured surface-feature tensor."""
    batch_shape = state.h2osfc.shape
    features = np.broadcast_to(bundle.x_mean, batch_shape + bundle.x_mean.shape).copy()
    # Compute soil_diag once for all layers; reuse for both btran and feature extraction.
    nlev_full = state.t_soisno.shape[-1]
    soil_diag = coupled_soil_state_diagnostics(state, static, nprofile=nlev_full)
    t_grnd = coupled_t_grnd_from_state(state)
    t10 = coupled_t10_from_state(state)
    feature_frac_sno = _total_snow_fraction_numpy(state)
    feature_snow_depth = _total_snow_depth_numpy(state)

    updates: dict[str, np.ndarray | float] = {
        "forc_t": atmosphere.forc_t,
        "forc_q": atmosphere.forc_q,
        "forc_pbot": atmosphere.forc_pbot,
        "forc_vpd": vapor_pressure_deficit_from_forcing(
            atmosphere.forc_t,
            atmosphere.forc_q,
            atmosphere.forc_pbot,
        ),
        "forc_lwrad": atmosphere.forc_lwrad,
        "forc_rain": atmosphere.forc_rain,
        "forc_snow": atmosphere.forc_snow,
        "forc_swrad": atmosphere.forc_swrad,
        "forc_solad_vis": atmosphere.forc_solad_vis,
        "forc_solad_nir": atmosphere.forc_solad_nir,
        "forc_solai_vis": atmosphere.forc_solai_vis,
        "forc_solai_nir": atmosphere.forc_solai_nir,
        "wind_speed": atmosphere.wind_speed,
        "forc_rho": atmosphere.forc_rho,
        "forc_pco2": atmosphere.forc_pco2,
        "frac_veg_nosno_patch": vegetation.frac_veg_nosno_patch,
        "elai_patch": vegetation.elai_patch,
        "esai_patch": vegetation.esai_patch,
        "frac_sno": feature_frac_sno,
        "snow_depth": feature_snow_depth,
        "zwt_col": _to_np(state.zwt),
        "h2ocan": _to_np(state.h2ocan),
        "t10": t10,
        "dayl": atmosphere.dayl,
        "max_dayl": atmosphere.max_dayl,
        "forc_hgt_u_patch": geometry.forc_hgt_u_patch,
        "z0mg_col": geometry.z0mg_col,
        "frac_h2osfc": _to_np(state.frac_h2osfc),
        "coszen_col": atmosphere.coszen_col,
        "t_grnd": t_grnd,
        "t_h2osfc": _to_np(state.t_h2osfc) if state.t_h2osfc is not None else _to_np(state.t_soisno)[..., 0],
        "ugust": geometry.ugust,
        "pft_type": 0.0 if static.pft_type is None else static.pft_type,
        "smpso": 0.0 if static.smpso is None else static.smpso,
        "smpsc": 0.0 if static.smpsc is None else static.smpsc,
        "soil_color": 0.0 if static.soil_color is None else static.soil_color,
        "leafcn": 0.0 if static.leafcn is None else static.leafcn,
        "slatop": 0.0 if static.slatop is None else static.slatop,
        "flnr": 0.0 if static.flnr is None else static.flnr,
        "mbbopt": 0.0 if static.mbbopt is None else static.mbbopt,
        "z0mr": 0.0 if static.z0mr is None else static.z0mr,
        "rholvis": 0.0 if static.rholvis is None else static.rholvis,
        "rholnir": 0.0 if static.rholnir is None else static.rholnir,
        "taulvis": 0.0 if static.taulvis is None else static.taulvis,
        "taulnir": 0.0 if static.taulnir is None else static.taulnir,
        "rhosvis": 0.0 if static.rhosvis is None else static.rhosvis,
        "rhosnir": 0.0 if static.rhosnir is None else static.rhosnir,
        "tausvis": 0.0 if static.tausvis is None else static.tausvis,
        "tausnir": 0.0 if static.tausnir is None else static.tausnir,
        "xl": 0.0 if static.xl is None else static.xl,
    }
    if vegetation.htop_patch is not None:
        updates["htop_patch"] = vegetation.htop_patch
    if "btran" in bundle.input_names:
        # Reuse already-computed soil_diag to avoid a second coupled_soil_state_diagnostics call.
        btran, _ = exact_btran_and_root_fraction(state, static, soil_diag=soil_diag)
        updates["btran"] = btran
    if "sabg_snow" in bundle.input_names:
        updates["sabg_snow"] = sabg_snow_from_state(
            state,
            atmosphere.forc_swrad,
            feature_frac_sno,
            elai=vegetation.elai_patch,
            esai=vegetation.esai_patch,
            atmosphere=atmosphere,
            soil_color=static.soil_color,
            h2osoi_vol_top=soil_diag["h2osoi_vol"][..., 0],
            rholvis=static.rholvis,
            rholnir=static.rholnir,
            taulvis=static.taulvis,
            taulnir=static.taulnir,
            rhosvis=static.rhosvis,
            rhosnir=static.rhosnir,
            tausvis=static.tausvis,
            tausnir=static.tausnir,
            xl=static.xl,
        )
    if not snow_model_enabled:
        updates["forc_rain"] = np.asarray(atmosphere.forc_rain, dtype=np.float64) + np.asarray(atmosphere.forc_snow, dtype=np.float64)
        updates["forc_snow"] = np.zeros_like(np.asarray(atmosphere.forc_snow, dtype=np.float64))
        updates["frac_sno"] = np.zeros_like(_to_np(state.frac_sno))
        updates["snow_depth"] = np.zeros_like(_to_np(state.snow_depth) if state.snow_depth is not None else np.asarray(vegetation.snow_depth, dtype=np.float64))
        if "sabg_snow" in bundle.input_names:
            updates["sabg_snow"] = _absorbed_ground_shortwave(
                atmosphere.forc_swrad,
                elai=vegetation.elai_patch,
                esai=vegetation.esai_patch,
                atmosphere=atmosphere,
                soil_color=static.soil_color,
                h2osoi_vol_top=soil_diag["h2osoi_vol"][..., 0],
                rholvis=static.rholvis,
                rholnir=static.rholnir,
                taulvis=static.taulvis,
                taulnir=static.taulnir,
                rhosvis=static.rhosvis,
                rhosnir=static.rhosnir,
                tausvis=static.tausvis,
                tausnir=static.tausnir,
                xl=static.xl,
            )
    if "soilbeta_col" in bundle.input_names:
        updates["soilbeta_col"] = coupled_soilbeta_from_state(
            state,
            static,
            frac_sno=np.asarray(updates["frac_sno"], dtype=np.float64),
            frac_h2osfc=np.asarray(updates["frac_h2osfc"], dtype=np.float64),
        )
    update_feature_columns(bundle, features, updates)

    # Bulk-assign soil profile features (avoids one dict entry + broadcast per level)
    nlev = soil_diag["smp_l"].shape[-1]
    input_names_t = tuple(bundle.input_names)
    for diag_key, prefix in (
        ("smp_l", "smp_l"),
        ("eff_porosity", "eff_porosity"),
        ("h2osoi_liqvol", "h2osoi_liqvol"),
    ):
        fidx, lidx = _soil_profile_feature_indices(input_names_t, prefix, nlev)
        if fidx:
            features[..., fidx] = soil_diag[diag_key][..., lidx]
    fidx, lidx = _soil_profile_feature_indices(input_names_t, "rootfr", nlev)
    if fidx:
        features[..., fidx] = static.rootfr[np.array(lidx, dtype=np.intp)]
    fidx, lidx = _soil_profile_feature_indices(input_names_t, "t_soisno", nlev)
    if fidx:
        features[..., fidx] = _to_np(state.t_soisno)[..., lidx]
    if feature_overrides:
        update_feature_columns(bundle, features, feature_overrides)
    valid = np.isfinite(features) & (np.abs(features) < 1.0e35)
    if not np.all(valid):
        features = np.where(valid, features, 0.0)
    return features


def build_surface_features_from_inputs_torch(
    bundle: SurfaceModelBundle,
    atmosphere: AtmosphericForcingInputs,
    vegetation: VegetationSurfaceInputs,
    geometry: SurfaceGeometryInputs,
    state: CoupledColumnState,
    static: CoupledStaticParameters,
    *,
    device: torch.device,
    dtype: torch.dtype,
    features: torch.Tensor | None = None,
    feature_overrides: Mapping[str, np.ndarray | float | torch.Tensor] | None = None,
    snow_model_enabled: bool = True,
    sanitize_features: bool = True,
    skip_feature_names: frozenset[str] | set[str] | None = None,
    precomputed_state_features: Mapping[str, torch.Tensor | tuple[int, ...] | None] | None = None,
    update_static_features: bool = True,
) -> torch.Tensor:
    """Torch-native feature assembly used by the coupled benchmark path."""
    common = precomputed_state_features
    batch_shape = state.h2osfc.shape if common is None else common["batch_shape"]
    assert isinstance(batch_shape, tuple)
    idx_map = _bundle_input_index_map(tuple(bundle.input_names))
    if features is None:
        features = torch.as_tensor(
            np.array(np.broadcast_to(bundle.x_mean, batch_shape + bundle.x_mean.shape), copy=True),
            device=device,
            dtype=dtype,
        )
    if common is None:
        common = precompute_surface_feature_state_torch(state, static, device=device, dtype=dtype)
    h2ocan_t = common["h2ocan_t"]
    h2osfc_t = common["h2osfc_t"]
    frac_h2osfc_t = common["frac_h2osfc_t"]
    frac_sno_t = common["frac_sno_t"]
    t_soisno_t = common["t_soisno_t"]
    t_h2osfc_t = common["t_h2osfc_t"]
    snow_depth_t = common["snow_depth_t"]
    t_grnd = common["t_grnd"]
    t10 = common["t10"]
    liqvol = common["liqvol"]
    eff_porosity = common["eff_porosity"]
    smp_l = common["smp_l"]
    assert isinstance(h2ocan_t, torch.Tensor)
    assert isinstance(h2osfc_t, torch.Tensor)
    assert isinstance(frac_h2osfc_t, torch.Tensor)
    assert isinstance(frac_sno_t, torch.Tensor)
    assert isinstance(t_soisno_t, torch.Tensor)
    assert isinstance(t_h2osfc_t, torch.Tensor)
    assert snow_depth_t is None or isinstance(snow_depth_t, torch.Tensor)
    assert isinstance(t_grnd, torch.Tensor)
    assert isinstance(t10, torch.Tensor)
    assert isinstance(liqvol, torch.Tensor)
    assert isinstance(eff_porosity, torch.Tensor)
    assert isinstance(smp_l, torch.Tensor)
    nlev_hydro = int(liqvol.shape[-1])

    forc_t_t = _to_torch(atmosphere.forc_t, device, dtype)
    forc_q_t = _to_torch(atmosphere.forc_q, device, dtype)
    forc_pbot_t = _to_torch(atmosphere.forc_pbot, device, dtype)
    updates: dict[str, np.ndarray | float | torch.Tensor] = {
        "forc_t": forc_t_t,
        "forc_q": forc_q_t,
        "forc_pbot": forc_pbot_t,
        "forc_vpd": vapor_pressure_deficit_from_forcing_torch(
            forc_t_t,
            forc_q_t,
            forc_pbot_t,
        ),
        "forc_lwrad": _to_torch(atmosphere.forc_lwrad, device, dtype),
        "forc_rain": _to_torch(atmosphere.forc_rain, device, dtype),
        "forc_snow": _to_torch(atmosphere.forc_snow, device, dtype),
        "forc_swrad": _to_torch(atmosphere.forc_swrad, device, dtype),
        "forc_solad_vis": _to_torch(atmosphere.forc_solad_vis, device, dtype),
        "forc_solad_nir": _to_torch(atmosphere.forc_solad_nir, device, dtype),
        "forc_solai_vis": _to_torch(atmosphere.forc_solai_vis, device, dtype),
        "forc_solai_nir": _to_torch(atmosphere.forc_solai_nir, device, dtype),
        "wind_speed": _to_torch(atmosphere.wind_speed, device, dtype),
        "forc_rho": _to_torch(atmosphere.forc_rho, device, dtype),
        "forc_pco2": _to_torch(atmosphere.forc_pco2, device, dtype),
        "frac_veg_nosno_patch": _to_torch(vegetation.frac_veg_nosno_patch, device, dtype),
        "elai_patch": _to_torch(vegetation.elai_patch, device, dtype),
        "esai_patch": _to_torch(vegetation.esai_patch, device, dtype),
        "frac_sno": frac_sno_t,
        "snow_depth": snow_depth_t if snow_depth_t is not None else _to_torch(vegetation.snow_depth, device, dtype),
        "zwt_col": _to_torch(state.zwt, device, dtype),
        "h2ocan": h2ocan_t,
        "t10": t10,
        "dayl": _to_torch(atmosphere.dayl, device, dtype),
        "max_dayl": _to_torch(atmosphere.max_dayl, device, dtype),
        "forc_hgt_u_patch": _to_torch(geometry.forc_hgt_u_patch, device, dtype),
        "z0mg_col": _to_torch(geometry.z0mg_col, device, dtype),
        "frac_h2osfc": frac_h2osfc_t,
        "coszen_col": _to_torch(atmosphere.coszen_col, device, dtype),
        "t_grnd": t_grnd,
        "t_h2osfc": t_h2osfc_t,
        "ugust": _to_torch(geometry.ugust, device, dtype),
    }
    if update_static_features:
        updates.update({
            "pft_type": 0.0 if static.pft_type is None else static.pft_type,
            "smpso": 0.0 if static.smpso is None else static.smpso,
            "smpsc": 0.0 if static.smpsc is None else static.smpsc,
            "soil_color": 0.0 if static.soil_color is None else static.soil_color,
            "leafcn": 0.0 if static.leafcn is None else static.leafcn,
            "slatop": 0.0 if static.slatop is None else static.slatop,
            "flnr": 0.0 if static.flnr is None else static.flnr,
            "mbbopt": 0.0 if static.mbbopt is None else static.mbbopt,
            "z0mr": 0.0 if static.z0mr is None else static.z0mr,
            "rholvis": 0.0 if static.rholvis is None else static.rholvis,
            "rholnir": 0.0 if static.rholnir is None else static.rholnir,
            "taulvis": 0.0 if static.taulvis is None else static.taulvis,
            "taulnir": 0.0 if static.taulnir is None else static.taulnir,
            "rhosvis": 0.0 if static.rhosvis is None else static.rhosvis,
            "rhosnir": 0.0 if static.rhosnir is None else static.rhosnir,
            "tausvis": 0.0 if static.tausvis is None else static.tausvis,
            "tausnir": 0.0 if static.tausnir is None else static.tausnir,
            "xl": 0.0 if static.xl is None else static.xl,
        })
    if vegetation.htop_patch is not None:
        updates["htop_patch"] = _to_torch(vegetation.htop_patch, device, dtype)
    if "btran" in bundle.input_names:
        if static.smpso is None or static.smpsc is None:
            updates["btran"] = torch.ones(batch_shape, device=device, dtype=dtype)
        else:
            updates["btran"] = coupled_btran_from_state_torch(
                state,
                static,
                device=device,
                dtype=dtype,
                h2osfc_storage=h2osfc_t,
                liqvol=liqvol,
                eff_porosity=eff_porosity,
                smp_l=smp_l,
            )
    if "sabg_snow" in bundle.input_names:
        updates["sabg_snow"] = sabg_snow_from_state_torch(
            state,
            atmosphere.forc_swrad,
            frac_sno_t,
            device=device,
            dtype=dtype,
            elai=vegetation.elai_patch,
            esai=vegetation.esai_patch,
            atmosphere=atmosphere,
            soil_color=static.soil_color,
            h2osoi_vol_top=common["h2osoi_vol"][..., 0],
            rholvis=static.rholvis,
            rholnir=static.rholnir,
            taulvis=static.taulvis,
            taulnir=static.taulnir,
            rhosvis=static.rhosvis,
            rhosnir=static.rhosnir,
            tausvis=static.tausvis,
            tausnir=static.tausnir,
            xl=static.xl,
        )
    if not snow_model_enabled:
        updates["forc_rain"] = _to_torch(atmosphere.forc_rain, device, dtype) + _to_torch(atmosphere.forc_snow, device, dtype)
        updates["forc_snow"] = torch.zeros(batch_shape, device=device, dtype=dtype)
        updates["frac_sno"] = torch.zeros(batch_shape, device=device, dtype=dtype)
        updates["snow_depth"] = torch.zeros(batch_shape, device=device, dtype=dtype)
        if "sabg_snow" in bundle.input_names:
            updates["sabg_snow"] = _absorbed_ground_shortwave_torch(
                atmosphere.forc_swrad,
                device=device,
                dtype=dtype,
                elai=vegetation.elai_patch,
                esai=vegetation.esai_patch,
                atmosphere=atmosphere,
                soil_color=static.soil_color,
                h2osoi_vol_top=common["h2osoi_vol"][..., 0],
                rholvis=static.rholvis,
                rholnir=static.rholnir,
                taulvis=static.taulvis,
                taulnir=static.taulnir,
                rhosvis=static.rhosvis,
                rhosnir=static.rhosnir,
                tausvis=static.tausvis,
                tausnir=static.tausnir,
                xl=static.xl,
            )
    if "soilbeta_col" in bundle.input_names:
        updates["soilbeta_col"] = coupled_soilbeta_from_state_torch(
            state,
            static,
            device=device,
            dtype=dtype,
            h2osoi_liq=common["h2osoi_liq_t"],
            h2osoi_ice=common["h2osoi_ice_t"],
            frac_sno=updates["frac_sno"],
            frac_h2osfc=updates["frac_h2osfc"],
        )
    if skip_feature_names:
        updates = {name: value for name, value in updates.items() if name not in skip_feature_names}
    _batch_update_feature_columns_torch(features, idx_map, updates)

    nlev = smp_l.shape[-1]
    input_names_t = tuple(bundle.input_names)
    for prefix, values in (
        ("smp_l", smp_l),
        ("eff_porosity", eff_porosity),
        ("h2osoi_liqvol", liqvol),
    ):
        fidx, lidx = _soil_profile_feature_indices(input_names_t, prefix, nlev)
        if fidx:
            _index_copy_feature_columns_torch(features, fidx, values[..., lidx])

    if update_static_features:
        rootfr = (
            static.rootfr_norm_torch
            if static.rootfr_norm_torch is not None and static.rootfr_norm_torch.device == device and static.rootfr_norm_torch.dtype == dtype
            else torch.as_tensor(np.maximum(np.asarray(static.rootfr, dtype=np.float64), 0.0), device=device, dtype=dtype)
        )
        if static.rootfr_norm_torch is None or static.rootfr_norm_torch.device != device or static.rootfr_norm_torch.dtype != dtype:
            rootfr = rootfr / torch.clamp(rootfr.sum(), min=1.0e-12)
        rootfr = rootfr[:nlev]
        fidx, lidx = _soil_profile_feature_indices(input_names_t, "rootfr", nlev)
        if fidx:
            root_values = torch.broadcast_to(rootfr[lidx], batch_shape + (len(lidx),))
            _index_copy_feature_columns_torch(features, fidx, root_values)

    fidx, lidx = _soil_profile_feature_indices(input_names_t, "t_soisno", nlev)
    if fidx:
        _index_copy_feature_columns_torch(features, fidx, t_soisno_t[..., lidx])

    if feature_overrides:
        override_updates: dict[str, np.ndarray | float | torch.Tensor] = {}
        for name, value in feature_overrides.items():
            if isinstance(value, torch.Tensor):
                override_updates[name] = value.to(device=device, dtype=dtype)
            else:
                override_updates[name] = _to_torch(value, device, dtype)
        _batch_update_feature_columns_torch(features, idx_map, override_updates)

    if sanitize_features:
        valid = torch.isfinite(features) & (torch.abs(features) < 1.0e35)
        if features.device.type == "cpu":
            if not torch.all(valid):
                features = torch.where(valid, features, torch.zeros_like(features))
        else:
            features = torch.where(valid, features, torch.zeros_like(features))
    return features


_bundle_registry: dict[int, "SurfaceModelBundle"] = {}


@lru_cache(maxsize=16)
def _manifold_clip_params(
    input_names: tuple[str, ...],
    bundle_id: int,
    nsigma: float,
    skip_features: frozenset[str] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pre-compute the static clipping arrays for constrain_surface_features_to_training_manifold."""
    bundle = _bundle_registry[bundle_id]
    _mean = np.asarray(bundle.x_mean, dtype=np.float64)
    _std = np.asarray(bundle.x_std, dtype=np.float64)
    n_feat = len(input_names)
    constrain_mask = np.ones(n_feat, dtype=bool)
    if skip_features:
        for i, name in enumerate(input_names):
            if name in skip_features:
                constrain_mask[i] = False
    tiny = ((_std < 1.0e-10) & constrain_mask).nonzero()[0]
    active_idx = (constrain_mask & ~(_std < 1.0e-10)).nonzero()[0]
    lower = _mean[active_idx] - nsigma * _std[active_idx]
    upper = _mean[active_idx] + nsigma * _std[active_idx]
    smpl_mask = np.array([input_names[i] in SMPL_FEATURES for i in active_idx])
    if smpl_mask.any():
        lower[smpl_mask] = _mean[active_idx[smpl_mask]] - (2.0 * nsigma) * _std[active_idx[smpl_mask]]
        upper[smpl_mask] = _mean[active_idx[smpl_mask]] + (2.0 * nsigma) * _std[active_idx[smpl_mask]]
    return tiny, active_idx, lower, upper


def constrain_surface_features_to_training_manifold(
    bundle: SurfaceModelBundle,
    features: np.ndarray,
    *,
    nsigma: float = 4.0,
    skip_features: frozenset[str] | set[str] | None = EXTERNAL_FORCING_FEATURES,
) -> np.ndarray:
    """Clip features to a simple training-manifold envelope in z-score space.

    This is intentionally conservative: each feature is limited to
    ``x_mean +/- nsigma * x_std``. Features with near-zero training variance are
    snapped back to the training mean to avoid pathological extrapolation on
    effectively constant inputs.

    ``skip_features`` names features that are never clipped (default:
    external forcings).
    Atmospheric forcing values are externally prescribed and physically valid at
    any magnitude, so clipping them to training statistics would corrupt the
    forcing. Soil matric potential features (``smp_l``) are clipped more
    loosely than the rest of the state because they can legitimately become
    drier than the training mean in long replays, and hard clipping them too
    tightly suppresses transpiration.
    """
    skip_fs = frozenset(skip_features) if skip_features is not None else None
    bid = id(bundle)
    _bundle_registry[bid] = bundle  # kept alive by this dict
    tiny_idx, active_idx, lower, upper = _manifold_clip_params(
        tuple(bundle.input_names),
        bid,
        nsigma,
        skip_fs,
    )
    x_mean = np.asarray(bundle.x_mean, dtype=np.float64)
    x_out = np.asarray(features, dtype=np.float64)
    if not x_out.flags.writeable:
        x_out = x_out.copy()
    if len(tiny_idx) > 0:
        x_out[..., tiny_idx] = x_mean[tiny_idx]
    if len(active_idx) > 0:
        np.clip(x_out[..., active_idx], lower, upper, out=x_out[..., active_idx])
    return x_out


def make_demo_surface_features(
    bundle: SurfaceModelBundle,
    state: CoupledColumnState,
    static: CoupledStaticParameters,
    *,
    seed: int,
    snow_model_enabled: bool = True,
) -> np.ndarray:
    del seed
    batch_shape = state.h2osfc.shape
    atmosphere = build_demo_atmospheric_forcing(batch_shape)
    vegetation = build_demo_vegetation_surface_inputs(state)
    geometry = build_demo_surface_geometry(batch_shape)
    return build_surface_features_from_inputs(
        bundle,
        atmosphere,
        vegetation,
        geometry,
        state,
        static,
        snow_model_enabled=snow_model_enabled,
    )


def make_demo_surface_features_torch(
    bundle: SurfaceModelBundle,
    state: CoupledColumnState,
    static: CoupledStaticParameters,
    *,
    atmosphere: AtmosphericForcingInputs | None = None,
    vegetation: VegetationSurfaceInputs | None = None,
    geometry: SurfaceGeometryInputs | None = None,
    template: torch.Tensor | None = None,
    device: torch.device,
    dtype: torch.dtype,
    snow_model_enabled: bool = True,
) -> torch.Tensor:
    atmosphere = build_demo_atmospheric_forcing(state.h2osfc.shape) if atmosphere is None else atmosphere
    vegetation = build_demo_vegetation_surface_inputs(state) if vegetation is None else vegetation
    geometry = build_demo_surface_geometry(state.h2osfc.shape) if geometry is None else geometry
    return build_surface_features_from_inputs_torch(
        bundle,
        atmosphere,
        vegetation,
        geometry,
        state,
        static,
        device=device,
        dtype=dtype,
        features=template,
        snow_model_enabled=snow_model_enabled,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run or benchmark the coupled surface-surrogate, hydrology, and soil-temperature driver.",
    )
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--patch-index", type=int, default=1)
    parser.add_argument("--hydrology-backend", choices=("numpy", "numpy_full", "torch", "full"), default="torch")
    parser.add_argument("--soil-temperature-backend", choices=("numpy", "torch"), default="torch")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--ncell", type=int, default=1000)
    parser.add_argument("--ensemble-size", type=int, default=100)
    parser.add_argument("--cells-per-chunk", type=int, default=0)
    parser.add_argument("--warmup-iters", type=int, default=1)
    parser.add_argument("--timed-iters", type=int, default=3)
    parser.add_argument("--profile-sections", action="store_true", help="Print per-section benchmark timings.")
    parser.add_argument("--disable-snow-model", action="store_true", help="Bypass snow evolution and treat snowfall as rain.")
    parser.add_argument(
        "--snow-scalar-only",
        action="store_true",
        help="Keep scalar snow storage active but never promote to resolved snow layers.",
    )
    parser.add_argument("--dt", type=float, default=3600.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument(
        "--tf32-matmul",
        choices=("highest", "high", "medium"),
        default="highest",
        help="CUDA float32 matmul precision policy.",
    )
    parser.add_argument("--describe-inputs", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    configure_tf32_matmul(args.tf32_matmul, device)
    snow_model_enabled = not args.disable_snow_model
    model_file = args.model_dir / f"patch_{args.patch_index}" / "model.pt"
    bundle = load_surface_model_bundle(model_file, device=device, dtype=dtype, use_compile=args.compile)

    if args.describe_inputs:
        print(describe_inputs(bundle))
        return

    texture, rootfr = build_demo_texture_and_root_profile(nlev=10)
    dz, z, zi = elm_soil_grid(nlevsoi=10)
    static = build_texture_static_parameters(
        texture,
        rootfr,
        dz=dz,
        z=z,
        zi=zi,
        device=device,
        dtype=dtype,
    )
    state = build_demo_state(args.ncell, args.ensemble_size, nlev=10)
    if snow_model_enabled and args.snow_scalar_only:
        state = force_scalar_snow_state_numpy(state)
    hydrology_workspace = SoilHydrologyTorchWorkspace() if device.type != "cpu" and args.hydrology_backend in ("torch", "full") else None
    soil_temperature_workspace = SoilTemperatureTorchWorkspace() if device.type != "cpu" and args.soil_temperature_backend == "torch" else None
    profile_sections: dict[str, float] | None = {} if args.profile_sections else None
    cells_per_chunk = args.ncell if args.cells_per_chunk <= 0 else min(args.cells_per_chunk, args.ncell)
    use_chunking = cells_per_chunk < args.ncell
    use_torch_backend = args.hydrology_backend in ("torch", "full") or args.soil_temperature_backend == "torch"
    chunk_spans = [(start, min(start + cells_per_chunk, args.ncell)) for start in range(0, args.ncell, cells_per_chunk)]

    def profile_call(name: str, fn):
        if profile_sections is None:
            return fn()
        synchronize(device)
        t0 = time.perf_counter()
        out = fn()
        synchronize(device)
        profile_sections[name] = profile_sections.get(name, 0.0) + (time.perf_counter() - t0)
        return out

    def run_chunked_pass(
        chunk_state: CoupledColumnState,
        *,
        start: int,
        stop: int,
    ) -> tuple[CoupledColumnState, list[float], dict[str, float], int, CoupledStepResult | None]:
        chunk_weight = (stop - start) * args.ensemble_size
        last_result: CoupledStepResult | None = None
        hydrology_workspace = SoilHydrologyTorchWorkspace() if device.type != "cpu" and args.hydrology_backend in ("torch", "full") else None
        soil_temperature_workspace = SoilTemperatureTorchWorkspace() if device.type != "cpu" and args.soil_temperature_backend == "torch" else None
        if use_torch_backend:
            chunk_state = coupled_state_to_device(chunk_state, device, dtype)
            atmosphere = profile_call("feature_build", lambda: build_demo_atmospheric_forcing_torch(chunk_state.h2osfc.shape, device=device, dtype=dtype))
            vegetation = profile_call("feature_build", lambda: build_demo_vegetation_surface_inputs_torch(chunk_state, device=device, dtype=dtype))
            geometry = profile_call("feature_build", lambda: build_demo_surface_geometry_torch(chunk_state.h2osfc.shape, device=device, dtype=dtype))
            feature_template = torch.as_tensor(
                np.array(np.broadcast_to(bundle.x_mean, chunk_state.h2osfc.shape + bundle.x_mean.shape), copy=True),
                device=device,
                dtype=dtype,
            )
            features = profile_call(
                "feature_build",
                lambda: make_demo_surface_features_torch(
                    bundle,
                    chunk_state,
                    static,
                    atmosphere=atmosphere,
                    vegetation=vegetation,
                    geometry=geometry,
                    template=feature_template,
                    device=device,
                    dtype=dtype,
                    snow_model_enabled=snow_model_enabled,
                ),
            )
        else:
            atmosphere = profile_call("feature_build", lambda: build_demo_atmospheric_forcing(chunk_state.h2osfc.shape))
            vegetation = profile_call("feature_build", lambda: build_demo_vegetation_surface_inputs(chunk_state))
            geometry = profile_call("feature_build", lambda: build_demo_surface_geometry(chunk_state.h2osfc.shape))
            features = profile_call(
                "feature_build",
                lambda: build_surface_features_from_inputs(
                    bundle,
                    atmosphere,
                    vegetation,
                    geometry,
                    chunk_state,
                    static,
                    snow_model_enabled=snow_model_enabled,
                ),
            )

        for _ in range(args.warmup_iters):
            result = coupled_step(
                bundle,
                features,
                chunk_state,
                static,
                args.dt,
                hydrology_backend=args.hydrology_backend,
                soil_temperature_backend=args.soil_temperature_backend,
                device=device,
                dtype=dtype,
                hydrology_workspace=hydrology_workspace,
                soil_temperature_workspace=soil_temperature_workspace,
                profile_sections=profile_sections,
                snow_model_enabled=snow_model_enabled,
                snow_scalar_only=args.snow_scalar_only,
            )
            chunk_state = result.state
            last_result = result
            if use_torch_backend:
                features = profile_call(
                    "feature_build",
                    lambda: make_demo_surface_features_torch(
                        bundle,
                        chunk_state,
                        static,
                        atmosphere=atmosphere,
                        vegetation=vegetation,
                        geometry=geometry,
                        template=feature_template,
                        device=device,
                        dtype=dtype,
                        snow_model_enabled=snow_model_enabled,
                    ),
                )
            else:
                features = profile_call(
                    "feature_build",
                    lambda: build_surface_features_from_inputs(
                        bundle,
                        atmosphere,
                        vegetation,
                        geometry,
                        chunk_state,
                        static,
                        snow_model_enabled=snow_model_enabled,
                    ),
                )

        chunk_timings: list[float] = []
        for _ in range(args.timed_iters):
            start_time = time.perf_counter()
            result = coupled_step(
                bundle,
                features,
                chunk_state,
                static,
                args.dt,
                hydrology_backend=args.hydrology_backend,
                soil_temperature_backend=args.soil_temperature_backend,
                device=device,
                dtype=dtype,
                hydrology_workspace=hydrology_workspace,
                soil_temperature_workspace=soil_temperature_workspace,
                profile_sections=profile_sections,
                snow_model_enabled=snow_model_enabled,
                snow_scalar_only=args.snow_scalar_only,
            )
            chunk_timings.append(time.perf_counter() - start_time)
            chunk_state = result.state
            last_result = result
            if use_torch_backend:
                features = profile_call(
                    "feature_build",
                    lambda: make_demo_surface_features_torch(
                        bundle,
                        chunk_state,
                        static,
                        atmosphere=atmosphere,
                        vegetation=vegetation,
                        geometry=geometry,
                        template=feature_template,
                        device=device,
                        dtype=dtype,
                        snow_model_enabled=snow_model_enabled,
                    ),
                )
            else:
                features = profile_call(
                    "feature_build",
                    lambda: build_surface_features_from_inputs(
                        bundle,
                        atmosphere,
                        vegetation,
                        geometry,
                        chunk_state,
                        static,
                        snow_model_enabled=snow_model_enabled,
                    ),
                )

        if use_torch_backend:
            chunk_state_out = coupled_state_to_cpu(chunk_state)
            qflx_prec_grnd = last_result.surface_targets["qflx_prec_grnd"] if last_result is not None else None
            if isinstance(qflx_prec_grnd, torch.Tensor):
                qflx_prec_grnd = qflx_prec_grnd.detach().cpu().numpy()
        else:
            chunk_state_out = chunk_state
            qflx_prec_grnd = last_result.surface_targets["qflx_prec_grnd"] if last_result is not None else None

        if qflx_prec_grnd is None:
            qflx_prec_grnd_mean = float("nan")
        else:
            qflx_prec_grnd_mean = float(np.asarray(qflx_prec_grnd, dtype=np.float64).mean())
        sanity = {
            "mean_qflx_prec_grnd": qflx_prec_grnd_mean,
            "mean_h2osfc": float(np.asarray(chunk_state_out.h2osfc, dtype=np.float64).mean()),
            "mean_t_top": float(np.asarray(chunk_state_out.t_soisno[..., 0], dtype=np.float64).mean()),
        }
        return chunk_state_out, chunk_timings, sanity, chunk_weight, last_result

    if not use_chunking:
        if use_torch_backend:
            state = coupled_state_to_device(state, device, dtype)
            feature_template = torch.as_tensor(
                np.array(np.broadcast_to(bundle.x_mean, state.h2osfc.shape + bundle.x_mean.shape), copy=True),
                device=device,
                dtype=dtype,
            )
            features = profile_call(
                "feature_build",
                lambda: make_demo_surface_features_torch(
                    bundle,
                    state,
                    static,
                    template=feature_template,
                    device=device,
                    dtype=dtype,
                    snow_model_enabled=snow_model_enabled,
                ),
            )
        else:
            features = profile_call("feature_build", lambda: make_demo_surface_features(bundle, state, static, seed=args.seed, snow_model_enabled=snow_model_enabled))

    try:
        if use_chunking:
            timings = [0.0 for _ in range(args.timed_iters)]
            sanity_num = {"mean_qflx_prec_grnd": 0.0, "mean_h2osfc": 0.0, "mean_t_top": 0.0}
            sanity_weight = 0
            chunk_states: list[CoupledColumnState] = []
            result: CoupledStepResult | None = None
            for start, stop in chunk_spans:
                chunk_state = slice_coupled_state(state, start, stop)
                chunk_state_out, chunk_timings, chunk_sanity, chunk_weight, chunk_result = run_chunked_pass(
                    chunk_state,
                    start=start,
                    stop=stop,
                )
                for i, elapsed in enumerate(chunk_timings):
                    timings[i] += elapsed
                for key in sanity_num:
                    sanity_num[key] += chunk_sanity[key] * chunk_weight
                sanity_weight += chunk_weight
                chunk_states.append(chunk_state_out)
                result = chunk_result
            state = concatenate_coupled_states(chunk_states)
            sanity = {key: value / max(sanity_weight, 1) for key, value in sanity_num.items()}
            assert result is not None
        else:
            for _ in range(args.warmup_iters):
                result = coupled_step(
                    bundle,
                    features,
                    state,
                    static,
                    args.dt,
                    hydrology_backend=args.hydrology_backend,
                    soil_temperature_backend=args.soil_temperature_backend,
                    device=device,
                    dtype=dtype,
                    hydrology_workspace=hydrology_workspace,
                    soil_temperature_workspace=soil_temperature_workspace,
                    profile_sections=profile_sections,
                    snow_model_enabled=snow_model_enabled,
                    snow_scalar_only=args.snow_scalar_only,
                )
                state = result.state
                if device.type != "cpu":
                    features = profile_call(
                        "feature_build",
                        lambda: make_demo_surface_features_torch(
                            bundle,
                            state,
                            static,
                            template=feature_template,
                            device=device,
                            dtype=dtype,
                            snow_model_enabled=snow_model_enabled,
                        ),
                    )
                else:
                    features = profile_call("feature_build", lambda: make_demo_surface_features(bundle, state, static, seed=args.seed, snow_model_enabled=snow_model_enabled))

            timings = []
            for _ in range(args.timed_iters):
                start = time.perf_counter()
                result = coupled_step(
                    bundle,
                    features,
                    state,
                    static,
                    args.dt,
                    hydrology_backend=args.hydrology_backend,
                    soil_temperature_backend=args.soil_temperature_backend,
                    device=device,
                    dtype=dtype,
                    hydrology_workspace=hydrology_workspace,
                    soil_temperature_workspace=soil_temperature_workspace,
                    profile_sections=profile_sections,
                    snow_model_enabled=snow_model_enabled,
                    snow_scalar_only=args.snow_scalar_only,
                )
                elapsed = time.perf_counter() - start
                timings.append(elapsed)
                state = result.state
                if device.type != "cpu":
                    features = profile_call(
                        "feature_build",
                        lambda: make_demo_surface_features_torch(
                            bundle,
                            state,
                            static,
                            template=feature_template,
                            device=device,
                            dtype=dtype,
                            snow_model_enabled=snow_model_enabled,
                        ),
                    )
                else:
                    features = profile_call("feature_build", lambda: make_demo_surface_features(bundle, state, static, seed=args.seed, snow_model_enabled=snow_model_enabled))
    except KeyError as exc:
        raise SystemExit(str(exc)) from exc

    if not use_chunking:
        assert result is not None
        qflx_prec_grnd = result.surface_targets["qflx_prec_grnd"]
        if isinstance(qflx_prec_grnd, torch.Tensor):
            qflx_prec_grnd = qflx_prec_grnd.detach().cpu().numpy()
        h2osfc = result.state.h2osfc
        if isinstance(h2osfc, torch.Tensor):
            h2osfc = h2osfc.detach().cpu().numpy()
        t_top = result.state.t_soisno[..., 0]
        if isinstance(t_top, torch.Tensor):
            t_top = t_top.detach().cpu().numpy()
        sanity = {
            "mean_qflx_prec_grnd": float(np.asarray(qflx_prec_grnd, dtype=np.float64).mean()),
            "mean_h2osfc": float(np.asarray(h2osfc, dtype=np.float64).mean()),
            "mean_t_top": float(np.asarray(t_top, dtype=np.float64).mean()),
        }

    nsample = args.ncell * args.ensemble_size
    mean_s = float(np.mean(timings))
    median_s = float(np.median(timings))
    print(
        f"device={device.type} dtype={dtype} patch={bundle.patch_index} "
        f"hydrology={args.hydrology_backend} soil_temperature={args.soil_temperature_backend} "
        f"cells_per_chunk={cells_per_chunk if use_chunking else args.ncell} tf32_matmul={args.tf32_matmul}"
    )
    print()
    print(" cells    nens      samples       mean_s     median_s      samples/s    us/sample")
    print("--------------------------------------------------------------------------------------")
    print(
        f"{args.ncell:6d} {args.ensemble_size:7d} {nsample:12d} "
        f"{mean_s:12.6e} {median_s:12.6e} {nsample / mean_s:14.3f} {1.0e6 * mean_s / nsample:12.3f}"
    )
    print(
        "sanity: "
        f"mean_qflx_prec_grnd={sanity['mean_qflx_prec_grnd']:.6e} "
        f"mean_h2osfc={sanity['mean_h2osfc']:.6e} "
        f"mean_t_top={sanity['mean_t_top']:.6e}"
    )
    if profile_sections is not None:
        print()
        print("profile sections (total_s / us_per_sample):")
        for name, total_s in sorted(profile_sections.items(), key=lambda item: item[1], reverse=True):
            print(f"  {name:<28} {total_s:10.6f} / {1.0e6 * total_s / nsample:10.3f}")


if __name__ == "__main__":
    main()
