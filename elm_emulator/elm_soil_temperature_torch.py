"""Torch implementation of the ELM-like soil-temperature solver.

Current scope matches the NumPy module:
- non-urban
- no snow
- optional standing surface water
- no phase-change / enthalpy correction
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import math

import numpy as np
import torch

from .elm_soil_temperature import (
    CAPR,
    CNFAC,
    CPLIQ,
    CPICE,
    DENH2O,
    DENICE,
    GRAV,
    HFUS,
    TFRZ,
    TKAIR,
    TKICE,
    TKWAT,
    SoilTextureProfile,
    pedotransfer_cosby1984_table5,
)


MAX_SNOW_LAYERS = 5
MIN_SNOW_LAYER_MASS = 1.0e-6
RHO_ICE = 917.0
SNOW_GEOMETRY_MIN_DENSITY = 50.0
SNOW_DEPTH_FRAC_MIN = 1.0e-3
SNOW_K_TMPS = torch.tensor([223.0, 248.0, 263.0, 268.0, 273.0])
SNOW_K_COE1 = torch.tensor([2.564, 2.172, 1.985, 1.883, 1.776])
SNOW_K_COE2 = torch.tensor([-0.059, 0.015, 0.073, 0.107, 0.147])
SNOW_K_COE3 = torch.tensor([0.0205, 0.0252, 0.0336, 0.0386, 0.0455])


OM_TKM = 0.25
OM_TKD = 0.05
OM_CSOL = 2.5
ZSAPRIC = 0.5


def _to_torch(x, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.as_tensor(x, device=device, dtype=dtype)


def _safe_snow_depth_from_mass_fraction_torch(
    h2osno: torch.Tensor,
    density: torch.Tensor,
    frac_sno: torch.Tensor,
) -> torch.Tensor:
    """Convert SWE to snow depth without letting tiny snow fractions explode."""
    mass = torch.clamp(
        torch.where(torch.isfinite(h2osno), h2osno, torch.zeros_like(h2osno)),
        min=0.0,
    )
    rho = torch.clamp(
        torch.where(torch.isfinite(density), density, torch.full_like(density, 200.0)),
        min=SNOW_GEOMETRY_MIN_DENSITY,
        max=DENICE,
    )
    frac = torch.clamp(
        torch.where(torch.isfinite(frac_sno), frac_sno, torch.zeros_like(frac_sno)),
        min=0.0,
        max=1.0,
    )
    depth_from_frac = mass / torch.clamp(rho * frac, min=1.0e-12)
    max_depth_from_mass = mass / SNOW_GEOMETRY_MIN_DENSITY
    depth_from_ice = mass / DENICE
    return torch.where(
        frac > SNOW_DEPTH_FRAC_MIN,
        torch.minimum(depth_from_frac, max_depth_from_mass),
        depth_from_ice,
    )


def _broadcast_layers(arr: torch.Tensor, batch_shape: tuple[int, ...], nlev: int) -> torch.Tensor:
    return torch.broadcast_to(arr, batch_shape + (nlev,))


def _broadcast_to_layers(arr: torch.Tensor, batch_shape: tuple[int, ...], nlev: int) -> torch.Tensor:
    return _broadcast_layers(arr, batch_shape, nlev)


def _broadcast_batch(arr: torch.Tensor, batch_shape: tuple[int, ...]) -> torch.Tensor:
    return torch.broadcast_to(arr, batch_shape)


def _batched_tridiagonal_torch(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    r: torch.Tensor,
    workspace: SoilTemperatureTorchWorkspace | None = None,
) -> torch.Tensor:
    """Solve batched tridiagonal systems (Thomas algorithm).

    All arrays are shaped ``(nbatch, n)``.

    The *workspace* argument is accepted for API compatibility but ignored;
    fresh temporaries are always allocated so this function is fully
    ``torch.compile``-friendly (no workspace.tensor() / torch.full_like
    graph breaks).
    """
    nbatch, n = b.shape
    cp = torch.empty_like(c)
    dp = torch.empty_like(r)
    x = torch.empty_like(r)

    cp[:, 0] = c[:, 0] / b[:, 0]
    dp[:, 0] = r[:, 0] / b[:, 0]

    for j in range(1, n):
        denom = b[:, j] - a[:, j] * cp[:, j - 1]
        if j < n - 1:
            cp[:, j] = c[:, j] / denom
        dp[:, j] = (r[:, j] - a[:, j] * dp[:, j - 1]) / denom

    x[:, -1] = dp[:, -1]
    for j in range(n - 2, -1, -1):
        x[:, j] = dp[:, j] - cp[:, j] * x[:, j + 1]
    return x


@dataclass
class SoilThermalParametersTorch:
    dz: torch.Tensor
    z: torch.Tensor
    zi: torch.Tensor
    watsat: torch.Tensor
    bsw: torch.Tensor
    sucsat: torch.Tensor
    tkmg: torch.Tensor
    tkdry: torch.Tensor
    csol: torch.Tensor


@dataclass
class SoilTemperatureStateTorch:
    t_soisno: torch.Tensor
    h2osoi_liq: torch.Tensor
    h2osoi_ice: torch.Tensor
    frac_sno: Optional[torch.Tensor] = None
    snow_layers: Optional[object] = None
    t_h2osfc: Optional[torch.Tensor] = None
    h2osfc: Optional[torch.Tensor] = None
    frac_h2osfc: Optional[torch.Tensor] = None
    h2osno: Optional[torch.Tensor] = None
    snow_depth: Optional[torch.Tensor] = None
    snow_t: Optional[torch.Tensor] = None


@dataclass
class SnowLayerState:
    snl: torch.Tensor
    dz: torch.Tensor
    h2osoi_ice: torch.Tensor
    h2osoi_liq: torch.Tensor
    t_snow: torch.Tensor
    snw_rds: Optional[torch.Tensor] = None


@dataclass
class SoilTemperatureForcingTorch:
    hs_soil: torch.Tensor
    dhsdT: torch.Tensor
    eflx_bot: torch.Tensor
    hs_top_snow: Optional[torch.Tensor] = None
    hs_h2osfc: Optional[torch.Tensor] = None


@dataclass
class SoilTemperatureResultTorch:
    state: SoilTemperatureStateTorch
    eflx_fgr12: torch.Tensor
    fn_h2osfc_old: torch.Tensor
    fn_h2osfc_new: torch.Tensor
    imelt: torch.Tensor
    qflx_snow_melt: Optional[torch.Tensor] = None
    qflx_scalar_snow_melt: Optional[torch.Tensor] = None
    qflx_snow_freeze: Optional[torch.Tensor] = None
    qflx_soil_melt: Optional[torch.Tensor] = None
    qflx_soil_freeze: Optional[torch.Tensor] = None
    qflx_soil_melt_profile: Optional[torch.Tensor] = None
    qflx_soil_freeze_profile: Optional[torch.Tensor] = None
    phase_hm_h2osfc: Optional[torch.Tensor] = None
    phase_xm_h2osfc: Optional[torch.Tensor] = None
    phase_temp1_h2osfc: Optional[torch.Tensor] = None


@dataclass
class SoilTemperatureTorchWorkspace:
    """Reusable scratch buffers for the Torch soil-temperature solver."""

    a: torch.Tensor | None = None
    b: torch.Tensor | None = None
    c: torch.Tensor | None = None
    r: torch.Tensor | None = None
    fact: torch.Tensor | None = None
    fn_old: torch.Tensor | None = None
    fn_new: torch.Tensor | None = None
    cp: torch.Tensor | None = None
    dp: torch.Tensor | None = None
    x: torch.Tensor | None = None

    def tensor(self, name: str, shape: tuple[int, ...], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        buf = getattr(self, name)
        if buf is None or tuple(buf.shape) != shape or buf.device != device or buf.dtype != dtype:
            buf = torch.empty(shape, device=device, dtype=dtype)
            setattr(self, name, buf)
        return buf


def thermal_params_from_texture_torch(
    texture: SoilTextureProfile,
    dz: np.ndarray,
    z: np.ndarray,
    zi: np.ndarray,
    *,
    nlevbed: int | None = None,
    device: torch.device,
    dtype: torch.dtype,
) -> SoilThermalParametersTorch:
    sand = np.asarray(texture.sand_pct, dtype=np.float64)
    clay = np.asarray(texture.clay_pct, dtype=np.float64)
    organic_frac = np.clip(np.asarray(texture.organic_frac, dtype=np.float64), 0.0, 1.0)
    watsat_min, bsw_min, sucsat_min, _ = pedotransfer_cosby1984_table5(sand, clay)
    om_watsat = np.maximum(0.93 - 0.1 * (z / ZSAPRIC), 0.83)
    om_b = np.minimum(2.7 + 9.3 * (z / ZSAPRIC), 12.0)
    om_sucsat = np.minimum(10.3 - 0.2 * (z / ZSAPRIC), 10.1)
    bd = (1.0 - watsat_min) * 2.7e3
    watsat = (1.0 - organic_frac) * watsat_min + organic_frac * om_watsat
    bsw = (1.0 - organic_frac) * (2.91 + 0.159 * clay) + organic_frac * om_b
    sucsat = (1.0 - organic_frac) * sucsat_min + organic_frac * om_sucsat
    mineral_tkm = np.where(sand + clay > 0.0, (8.80 * sand + 2.92 * clay) / (sand + clay), 0.0)
    tkm = (1.0 - organic_frac) * mineral_tkm + OM_TKM * organic_frac
    tkmg = tkm ** (1.0 - watsat)
    tkdry = ((0.135 * bd + 64.7) / (2.7e3 - 0.947 * bd)) * (1.0 - organic_frac) + OM_TKD * organic_frac
    csol = (((1.0 - organic_frac) * (2.128 * sand + 2.385 * clay) / np.maximum(sand + clay, 1.0e-12)) + OM_CSOL * organic_frac) * 1.0e6
    if nlevbed is not None:
        csol = np.array(csol, copy=True)
        csol[int(nlevbed) :] = 2.0e6
    return SoilThermalParametersTorch(
        dz=_to_torch(dz, device, dtype),
        z=_to_torch(z, device, dtype),
        zi=_to_torch(zi, device, dtype),
        watsat=_to_torch(watsat, device, dtype),
        bsw=_to_torch(bsw, device, dtype),
        sucsat=_to_torch(sucsat, device, dtype),
        tkmg=_to_torch(tkmg, device, dtype),
        tkdry=_to_torch(tkdry, device, dtype),
        csol=_to_torch(csol, device, dtype),
    )


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


def _elm_snow_layer_depths(snow_depth: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ELM's active snow-layer count and top-to-bottom layer depths."""
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


def _snow_layers_from_storage_fallback(
    h2osno: torch.Tensor,
    snow_depth: torch.Tensor,
    snow_t: torch.Tensor | None = None,
) -> "SnowLayerState":
    h2osno = torch.as_tensor(h2osno)
    snow_depth = torch.as_tensor(snow_depth)
    shape = h2osno.shape
    snl = torch.zeros(shape, device=h2osno.device, dtype=torch.int32)
    dz = torch.zeros(shape + (MAX_SNOW_LAYERS,), device=h2osno.device, dtype=h2osno.dtype)
    ice = torch.zeros_like(dz)
    liq = torch.zeros_like(dz)
    temp = torch.zeros_like(dz)
    if snow_t is not None:
        temp_seed = torch.broadcast_to(torch.as_tensor(snow_t, device=h2osno.device, dtype=h2osno.dtype), shape + (MAX_SNOW_LAYERS,))
    else:
        temp_seed = torch.broadcast_to(torch.full(shape, TFRZ, device=h2osno.device, dtype=h2osno.dtype)[..., None], shape + (MAX_SNOW_LAYERS,))
    nsnow, dz_template = _elm_snow_layer_depths(snow_depth)
    for k in range(1, MAX_SNOW_LAYERS + 1):
        mask = nsnow == k
        if not torch.any(mask):
            continue
        snl = torch.where(mask, torch.full_like(snl, -k), snl)
        dz[mask, :k] = torch.clamp(dz_template[mask, :k], min=1.0e-6)
        mass_frac = dz_template[mask, :k] / torch.clamp(snow_depth[mask, None], min=1.0e-12)
        ice[mask, :k] = torch.clamp(h2osno[mask, None] * mass_frac, min=0.0)
        temp[mask, :k] = temp_seed[mask, :k]
    inactive_scalar = (nsnow == 0) & (h2osno > 1.0e-8)
    if torch.any(inactive_scalar):
        dz[inactive_scalar, 0] = torch.clamp(snow_depth[inactive_scalar], min=0.0)
        ice[inactive_scalar, 0] = h2osno[inactive_scalar]
        temp[inactive_scalar, 0] = temp_seed[inactive_scalar, 0]
    return SnowLayerState(snl=snl, dz=dz, h2osoi_ice=ice, h2osoi_liq=liq, t_snow=temp)


def _snow_thermal_conductivity_torch(tempk: torch.Tensor, density: torch.Tensor) -> torch.Tensor:
    tempk = torch.as_tensor(tempk)
    density = torch.clamp(torch.as_tensor(density), min=0.0, max=RHO_ICE)
    dens_ratio = density / RHO_ICE
    k_curve = torch.stack(
        [
            SNOW_K_COE1[i] * dens_ratio**2 - SNOW_K_COE2[i] * dens_ratio + SNOW_K_COE3[i]
            for i in range(SNOW_K_TMPS.numel())
        ],
        dim=0,
    )
    k_vals = torch.full_like(tempk, float(SNOW_K_COE3[-1]))
    for i in range(SNOW_K_TMPS.numel() - 1):
        lo = float(SNOW_K_TMPS[i])
        hi = float(SNOW_K_TMPS[i + 1])
        mask = (tempk >= lo) & (tempk <= hi)
        interp = k_curve[i] + (tempk - lo) * (k_curve[i + 1] - k_curve[i]) / max(hi - lo, 1.0e-12)
        k_vals = torch.where(mask, interp, k_vals)
    k_vals = torch.where(tempk < float(SNOW_K_TMPS[0]), k_curve[0], k_vals)
    k_vals = torch.where(tempk > float(SNOW_K_TMPS[-1]), k_curve[-1], k_vals)
    return torch.clamp(k_vals, min=1.0e-3)


def _soil_phase_change_torch(
    t: torch.Tensor,
    hliq: torch.Tensor,
    hice: torch.Tensor,
    params: SoilThermalParametersTorch,
    fact: torch.Tensor,
    dhsdT: torch.Tensor,
    frac_h2osfc: torch.Tensor,
    dt: float,
    scalar_snow_mask: torch.Tensor | None = None,
    scalar_h2osno: torch.Tensor | None = None,
    scalar_snow_depth: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor]:
    batch_shape = t.shape[:-1]
    nlev = t.shape[-1]
    t2 = t.reshape(-1, nlev).clone()
    t0 = t2.clone()
    hliq2 = hliq.reshape(-1, nlev).clone()
    hice2 = hice.reshape(-1, nlev).clone()
    fact2 = fact.reshape(-1, nlev)
    dz = _broadcast_layers(params.dz, batch_shape, nlev).reshape(-1, nlev)
    watsat = _broadcast_layers(params.watsat, batch_shape, nlev).reshape(-1, nlev)
    bsw = _broadcast_layers(params.bsw, batch_shape, nlev).reshape(-1, nlev)
    sucsat = _broadcast_layers(params.sucsat, batch_shape, nlev).reshape(-1, nlev)
    frac_h2osfc2 = frac_h2osfc.reshape(-1)
    dhsdT2 = dhsdT.reshape(-1)
    if scalar_h2osno is None or scalar_snow_depth is None:
        scalar_h2osno2 = None
        scalar_snow_depth2 = None
        scalar_present = torch.zeros(t2.shape[0], device=t.device, dtype=torch.bool)
    else:
        scalar_h2osno2 = scalar_h2osno.reshape(-1).clone()
        scalar_snow_depth2 = scalar_snow_depth.reshape(-1).clone()
        mask_in = (
            torch.ones(t2.shape[0], device=t.device, dtype=torch.bool)
            if scalar_snow_mask is None
            else scalar_snow_mask.reshape(-1).to(device=t.device, dtype=torch.bool)
        )
        scalar_present = mask_in & (scalar_h2osno2 > MIN_SNOW_LAYER_MASS) & (scalar_snow_depth2 > 0.0)
    qflx_scalar_snow_melt = torch.zeros(t2.shape[0], device=t.device, dtype=t.dtype)

    imelt = torch.zeros_like(t2, dtype=torch.int32)
    wice0 = hice2.clone()
    wliq0 = hliq2.clone()
    wmass0 = hice2 + hliq2
    tinc = torch.zeros_like(t2)
    hm = torch.zeros_like(t2)

    melt_mask = (hice2 > 0.0) & (t2 > TFRZ)
    imelt[melt_mask] = 1
    tinc[melt_mask] = TFRZ - t2[melt_mask]
    t2[melt_mask] = TFRZ

    # Vectorized supercool computation (no if-any guard)
    smp_cold = HFUS * (TFRZ - t2) / (GRAV * torch.clamp(t2, min=1.0)) * 1000.0
    sup_cold = watsat * torch.pow(torch.clamp(smp_cold / torch.clamp(sucsat, min=1.0e-12), min=0.0), -1.0 / bsw) * dz * 1000.0
    supercool = torch.where(t2 < TFRZ, sup_cold, torch.zeros_like(t2))

    freeze_mask = (hliq2 > supercool) & (t2 < TFRZ)
    imelt[freeze_mask] = 2
    tinc[freeze_mask] = TFRZ - t2[freeze_mask]
    t2[freeze_mask] = TFRZ

    scalar_top_melt = scalar_present & (t2[:, 0] > TFRZ)
    imelt[:, 0] = torch.where(scalar_top_melt, torch.ones_like(imelt[:, 0]), imelt[:, 0])
    tinc[:, 0] = torch.where(scalar_top_melt, TFRZ - t2[:, 0], tinc[:, 0])
    t2[:, 0] = torch.where(scalar_top_melt, torch.full_like(t2[:, 0], TFRZ), t2[:, 0])

    valid_fact = torch.isfinite(fact2) & (torch.abs(fact2) > 1.0e-30)
    invalid_active = (imelt > 0) & ~valid_fact
    imelt = torch.where(invalid_active, torch.zeros_like(imelt), imelt)
    tinc = torch.where(invalid_active, torch.zeros_like(tinc), tinc)
    t2 = torch.where(invalid_active, t0, t2)
    inv_fact = torch.where(
        valid_fact,
        torch.reciprocal(torch.where(valid_fact, fact2, torch.ones_like(fact2))),
        torch.zeros_like(fact2),
    )

    hm[:, 0] = (1.0 - frac_h2osfc2) * dhsdT2 * tinc[:, 0] - tinc[:, 0] * inv_fact[:, 0]
    if nlev > 1:
        hm[:, 1:] = -tinc[:, 1:] * inv_fact[:, 1:]

    bad_melt = (imelt == 1) & (hm < 0.0)
    bad_freeze = (imelt == 2) & (hm > 0.0)
    imelt[bad_melt | bad_freeze] = 0
    hm[bad_melt | bad_freeze] = 0.0

    xm = hm * dt / HFUS

    if scalar_h2osno2 is not None and scalar_snow_depth2 is not None:
        scalar_melt = scalar_top_melt & (imelt[:, 0] == 1) & (xm[:, 0] > 0.0)
        old_snow = scalar_h2osno2.clone()
        melt_snow = torch.where(
            scalar_melt,
            torch.minimum(xm[:, 0], torch.clamp(old_snow, min=0.0)),
            torch.zeros_like(old_snow),
        )
        scalar_h2osno2 = torch.where(scalar_melt, torch.clamp(old_snow - melt_snow, min=0.0), scalar_h2osno2)
        scalar_snow_depth2 = torch.where(
            scalar_melt & (old_snow > 1.0e-12),
            scalar_snow_depth2 * scalar_h2osno2 / torch.clamp(old_snow, min=1.0e-12),
            scalar_snow_depth2,
        )
        qflx_scalar_snow_melt = melt_snow / max(dt, 1.0e-12)
        heatr = hm[:, 0] - HFUS * melt_snow / max(dt, 1.0e-12)
        leftover = scalar_melt & (heatr > 0.0)
        hm[:, 0] = torch.where(scalar_melt, torch.where(leftover, heatr, torch.zeros_like(heatr)), hm[:, 0])
        xm[:, 0] = torch.where(
            scalar_melt,
            torch.where(leftover, heatr * dt / HFUS, torch.zeros_like(heatr)),
            xm[:, 0],
        )
        imelt[:, 0] = torch.where(scalar_melt & ~leftover, torch.zeros_like(imelt[:, 0]), imelt[:, 0])

    # Vectorized phase change over all levels at once (replaces Python loop)
    melt_all   = (imelt == 1) & (torch.abs(hm) > 0.0)   # (nbatch, nlev)
    freeze_all = (imelt == 2) & (torch.abs(hm) > 0.0)

    # Melt branch
    hice_melt  = torch.clamp(wice0 - xm, min=0.0)
    heatr_melt = hm - HFUS * (wice0 - hice_melt) / dt
    hliq_melt  = torch.clamp(wmass0 - hice_melt, min=0.0)
    dt_melt = fact2 * heatr_melt
    denom0 = 1.0 - (1.0 - frac_h2osfc2) * fact2[:, 0] * dhsdT2
    safe_denom0 = torch.where(torch.abs(denom0) > 1.0e-30, denom0, torch.ones_like(denom0))
    dt_melt = dt_melt.clone()
    dt_melt[:, 0] = torch.where(
        torch.abs(denom0) > 1.0e-30,
        fact2[:, 0] * heatr_melt[:, 0] / safe_denom0,
        dt_melt[:, 0],
    )
    t_melt = t2 + dt_melt

    # Freeze branch
    target_ice = torch.where(
        wmass0 < supercool,
        torch.zeros_like(wmass0),
        torch.minimum(wmass0 - supercool, wice0 - xm),
    )
    hice_frz   = torch.clamp(target_ice, min=0.0)
    heatr_frz  = hm - HFUS * (wice0 - hice_frz) / dt
    hliq_frz   = torch.clamp(wmass0 - hice_frz, min=0.0)
    dt_frz = fact2 * heatr_frz
    dt_frz = dt_frz.clone()
    dt_frz[:, 0] = torch.where(
        torch.abs(denom0) > 1.0e-30,
        fact2[:, 0] * heatr_frz[:, 0] / safe_denom0,
        dt_frz[:, 0],
    )
    t_frz = t2 + dt_frz

    hice2 = torch.where(melt_all, hice_melt, torch.where(freeze_all, hice_frz, hice2))
    hliq2 = torch.where(melt_all, hliq_melt, torch.where(freeze_all, hliq_frz, hliq2))
    t2    = torch.where(melt_all, t_melt,    torch.where(freeze_all, t_frz,    t2))

    return (
        t2.reshape(batch_shape + (nlev,)),
        hliq2.reshape(batch_shape + (nlev,)),
        hice2.reshape(batch_shape + (nlev,)),
        imelt.reshape(batch_shape + (nlev,)),
        None if scalar_h2osno2 is None else scalar_h2osno2.reshape(batch_shape),
        None if scalar_snow_depth2 is None else scalar_snow_depth2.reshape(batch_shape),
        qflx_scalar_snow_melt.reshape(batch_shape),
    )


def _snow_soil_temperature_step_torch(
    state: SoilTemperatureStateTorch,
    params: SoilThermalParametersTorch,
    forcing: SoilTemperatureForcingTorch,
    dt: float,
    workspace: SoilTemperatureTorchWorkspace | None = None,
) -> SoilTemperatureResultTorch:
    t = state.t_soisno
    hliq = state.h2osoi_liq
    hice = state.h2osoi_ice
    batch_shape = t.shape[:-1]
    nlev = t.shape[-1]
    device = t.device
    dtype = t.dtype
    nbatch = t.reshape(-1, nlev).shape[0]

    hs_soil = _broadcast_batch(forcing.hs_soil, batch_shape).reshape(-1)
    hs_top_snow = _broadcast_batch(
        hs_soil if forcing.hs_top_snow is None else forcing.hs_top_snow,
        batch_shape,
    ).reshape(-1)
    hs_h2osfc = _broadcast_batch(
        torch.zeros((), device=device, dtype=dtype) if forcing.hs_h2osfc is None else forcing.hs_h2osfc,
        batch_shape,
    ).reshape(-1)
    dhsdT = _broadcast_batch(forcing.dhsdT, batch_shape).reshape(-1)
    eflx_bot = _broadcast_batch(forcing.eflx_bot, batch_shape).reshape(-1)

    snow_layers_state = state.snow_layers
    if snow_layers_state is not None:
        snow_snl = torch.broadcast_to(torch.as_tensor(snow_layers_state.snl, device=device, dtype=torch.int32), batch_shape).reshape(-1).clone()
        snow_dz_state = torch.broadcast_to(torch.as_tensor(snow_layers_state.dz, device=device, dtype=dtype), batch_shape + (MAX_SNOW_LAYERS,)).reshape(-1, MAX_SNOW_LAYERS).clone()
        snow_ice_state = torch.broadcast_to(torch.as_tensor(snow_layers_state.h2osoi_ice, device=device, dtype=dtype), batch_shape + (MAX_SNOW_LAYERS,)).reshape(-1, MAX_SNOW_LAYERS).clone()
        snow_liq_state = torch.broadcast_to(torch.as_tensor(snow_layers_state.h2osoi_liq, device=device, dtype=dtype), batch_shape + (MAX_SNOW_LAYERS,)).reshape(-1, MAX_SNOW_LAYERS).clone()
        if state.snow_t is None:
            snow_t_state = torch.broadcast_to(
                torch.as_tensor(snow_layers_state.t_snow, device=device, dtype=dtype),
                batch_shape + (MAX_SNOW_LAYERS,),
            ).reshape(-1, MAX_SNOW_LAYERS).clone()
        else:
            snow_t_state = _broadcast_to_layers(
                torch.as_tensor(state.snow_t, device=device, dtype=dtype),
                batch_shape,
                MAX_SNOW_LAYERS,
            ).reshape(-1, MAX_SNOW_LAYERS).clone()
        h2osno = torch.sum(torch.clamp(snow_ice_state + snow_liq_state, min=0.0), dim=-1).reshape(-1).clone()
        nsnow_v = torch.clamp(-snow_snl, min=0)
        layer_index = torch.arange(MAX_SNOW_LAYERS, device=device).reshape(1, MAX_SNOW_LAYERS)
        snow_depth = torch.sum(
            torch.where(
                layer_index < nsnow_v[:, None],
                torch.clamp(snow_dz_state, min=0.0),
                torch.zeros_like(snow_dz_state),
            ),
            dim=-1,
        ).reshape(-1).clone()
    else:
        snow_snl = torch.full((nbatch,), -_snow_layer_count(float((state.snow_depth.reshape(-1)[0].item() if state.snow_depth is not None else 0.0))), device=device, dtype=torch.int32)
        h2osno = _broadcast_batch(0.0 if state.h2osno is None else state.h2osno, batch_shape).reshape(-1).clone()
        snow_depth = _broadcast_batch(0.0 if state.snow_depth is None else state.snow_depth, batch_shape).reshape(-1).clone()
        if state.snow_t is None:
            snow_t_state = torch.broadcast_to(torch.minimum(t[..., :1], torch.full_like(t[..., :1], TFRZ)), batch_shape + (MAX_SNOW_LAYERS,)).clone()
        else:
            snow_t_state = _broadcast_to_layers(state.snow_t, batch_shape, MAX_SNOW_LAYERS).clone()
        snow_layers_state = _snow_layers_from_storage_fallback(
            h2osno.reshape(batch_shape),
            snow_depth.reshape(batch_shape),
            snow_t_state,
        )
        snow_snl = torch.as_tensor(snow_layers_state.snl, device=device, dtype=torch.int32).reshape(-1).clone()
        snow_dz_state = torch.as_tensor(snow_layers_state.dz, device=device, dtype=dtype).reshape(-1, MAX_SNOW_LAYERS).clone()
        snow_ice_state = torch.as_tensor(snow_layers_state.h2osoi_ice, device=device, dtype=dtype).reshape(-1, MAX_SNOW_LAYERS).clone()
        snow_liq_state = torch.as_tensor(snow_layers_state.h2osoi_liq, device=device, dtype=dtype).reshape(-1, MAX_SNOW_LAYERS).clone()
        snow_t_state = torch.as_tensor(snow_layers_state.t_snow, device=device, dtype=dtype).reshape(-1, MAX_SNOW_LAYERS).clone()

    h2osfc = _broadcast_batch(0.0 if state.h2osfc is None else state.h2osfc, batch_shape).reshape(-1).clone()
    frac_h2osfc = _broadcast_batch(0.0 if state.frac_h2osfc is None else state.frac_h2osfc, batch_shape).reshape(-1).clone()
    t_h2osfc = _broadcast_batch(0.0 if state.t_h2osfc is None else state.t_h2osfc, batch_shape).reshape(-1).clone()

    dz = _broadcast_to_layers(params.dz, batch_shape, nlev).reshape(-1, nlev)
    z = _broadcast_to_layers(params.z, batch_shape, nlev).reshape(-1, nlev)
    cv_soil, thk_soil, _ = soil_thermal_properties_torch(state, params)
    cv_soil2 = cv_soil.reshape(-1, nlev)
    thk_soil2 = thk_soil.reshape(-1, nlev)
    t2 = t.reshape(-1, nlev)

    def _selected_params(rows: torch.Tensor) -> SoilThermalParametersTorch:
        def select(arr: torch.Tensor) -> torch.Tensor:
            arr_b = _broadcast_layers(arr, batch_shape, nlev).reshape(-1, nlev)
            return arr_b.index_select(0, rows)

        return SoilThermalParametersTorch(
            dz=select(params.dz),
            z=select(params.z),
            zi=select(params.zi),
            watsat=select(params.watsat),
            bsw=select(params.bsw),
            sucsat=select(params.sucsat),
            tkmg=select(params.tkmg),
            tkdry=select(params.tkdry),
            csol=select(params.csol),
        )

    base_result = _soil_temperature_step_torch_no_snow(
        state,
        params,
        forcing,
        dt,
        workspace=workspace,
        snow_enabled=True,
    )

    t_pc = base_result.state.t_soisno.reshape(-1, nlev).clone()
    hliq_pc = base_result.state.h2osoi_liq.reshape(-1, nlev).clone()
    hice_pc = base_result.state.h2osoi_ice.reshape(-1, nlev).clone()
    t_h2osfc_out = base_result.state.t_h2osfc.reshape(-1).clone()
    h2osfc_out = base_result.state.h2osfc.reshape(-1).clone()
    frac_h2osfc_out = base_result.state.frac_h2osfc.reshape(-1).clone()
    h2osno_out = base_result.state.h2osno.reshape(-1).clone()
    snow_depth_out = base_result.state.snow_depth.reshape(-1).clone()
    snow_t_new = snow_t_state.reshape(-1, MAX_SNOW_LAYERS).clone()
    eflx_fgr12 = base_result.eflx_fgr12.reshape(-1).clone()
    fn_h2osfc_old_out = base_result.fn_h2osfc_old.reshape(-1).clone()
    fn_h2osfc_new_out = base_result.fn_h2osfc_new.reshape(-1).clone()
    imelt_pc = base_result.imelt.reshape(-1, nlev).clone()
    qflx_soil_melt_profile = (
        torch.zeros((nbatch, nlev), device=device, dtype=dtype)
        if base_result.qflx_soil_melt_profile is None
        else base_result.qflx_soil_melt_profile.reshape(-1, nlev).clone()
    )
    qflx_soil_freeze_profile = (
        torch.zeros((nbatch, nlev), device=device, dtype=dtype)
        if base_result.qflx_soil_freeze_profile is None
        else base_result.qflx_soil_freeze_profile.reshape(-1, nlev).clone()
    )
    qflx_snow_melt = torch.zeros(nbatch, device=device, dtype=dtype)
    qflx_snow_freeze = torch.zeros(nbatch, device=device, dtype=dtype)
    if base_result.phase_hm_h2osfc is None:
        phase_hm_h2osfc = torch.full(batch_shape, torch.nan, device=device, dtype=dtype)
        phase_xm_h2osfc = torch.full(batch_shape, torch.nan, device=device, dtype=dtype)
        phase_temp1_h2osfc = torch.full(batch_shape, torch.nan, device=device, dtype=dtype)
    else:
        phase_hm_h2osfc = base_result.phase_hm_h2osfc.clone()
        phase_xm_h2osfc = base_result.phase_xm_h2osfc.clone()
        phase_temp1_h2osfc = base_result.phase_temp1_h2osfc.clone()
    phase_hm_flat = phase_hm_h2osfc.reshape(-1)
    phase_xm_flat = phase_xm_h2osfc.reshape(-1)
    phase_temp1_flat = phase_temp1_h2osfc.reshape(-1)

    nsnow_all = torch.clamp(-snow_snl.to(torch.int32), min=0)
    frac_sno_flat = (
        _broadcast_batch(state.frac_sno, batch_shape).reshape(-1)
        if state.frac_sno is not None
        else torch.ones(nbatch, device=device, dtype=dtype)
    )
    hliq2 = hliq.reshape(-1, nlev)
    hice2 = hice.reshape(-1, nlev)

    for nsnow in range(1, MAX_SNOW_LAYERS + 1):
        rows = torch.nonzero(nsnow_all == nsnow, as_tuple=False).reshape(-1)
        if rows.numel() == 0:
            continue

        m = rows.numel()
        snow_dz = snow_dz_state.index_select(0, rows)[:, :nsnow].clone()
        snow_ice = snow_ice_state.index_select(0, rows)[:, :nsnow].clone()
        snow_liq = snow_liq_state.index_select(0, rows)[:, :nsnow].clone()
        snow_temp_layer = snow_t_state.index_select(0, rows)[:, :nsnow].clone()
        frac_sno_eff = torch.clamp(frac_sno_flat.index_select(0, rows), min=1.0e-12)
        h2osno_g = h2osno.index_select(0, rows).clone()
        snow_depth_g = snow_depth.index_select(0, rows).clone()
        h2osfc_g = h2osfc.index_select(0, rows).clone()
        frac_h2osfc_g = frac_h2osfc.index_select(0, rows).clone()
        t_h2osfc_g = t_h2osfc.index_select(0, rows).clone()
        t_g = t2.index_select(0, rows)
        z_g = z.index_select(0, rows)
        cv_soil_g = cv_soil2.index_select(0, rows)
        thk_soil_g = thk_soil2.index_select(0, rows)
        hs_top_snow_g = hs_top_snow.index_select(0, rows)
        hs_h2osfc_g = hs_h2osfc.index_select(0, rows)
        dhsdT_g = dhsdT.index_select(0, rows)
        eflx_bot_g = eflx_bot.index_select(0, rows)

        snow_cv = torch.clamp((CPLIQ * snow_liq + CPICE * snow_ice) / frac_sno_eff[:, None], min=1.0e-3)
        snow_density = torch.clamp(
            (snow_ice + snow_liq) / torch.clamp(snow_dz * frac_sno_eff[:, None], min=1.0e-12),
            min=0.0,
            max=RHO_ICE,
        )
        snow_k = _snow_thermal_conductivity_torch(snow_temp_layer, snow_density)
        snow_z = -torch.sum(snow_dz, dim=1, keepdim=True) + torch.cumsum(snow_dz, dim=1) - 0.5 * snow_dz
        wice0 = snow_ice.clone()
        wmass0 = snow_ice + snow_liq

        temp = torch.cat([snow_temp_layer, t_g], dim=1)
        cv_col = torch.cat([snow_cv, cv_soil_g], dim=1)
        z_col = torch.cat([snow_z, z_g], dim=1)
        layer_k = torch.cat([snow_k, thk_soil_g], dim=1)
        ncol = nsnow + nlev

        k_int = torch.zeros((m, ncol), device=device, dtype=dtype)
        fn_old = torch.zeros_like(k_int)
        k_int[:, :-1] = (
            2.0
            * layer_k[:, :-1]
            * layer_k[:, 1:]
            / torch.clamp(layer_k[:, :-1] + layer_k[:, 1:], min=1.0e-12)
        )
        fn_old[:, :-1] = k_int[:, :-1] * (temp[:, 1:] - temp[:, :-1]) / torch.clamp(
            z_col[:, 1:] - z_col[:, :-1],
            min=1.0e-12,
        )
        fn_old[:, -1] = eflx_bot_g

        fact = dt / torch.clamp(cv_col, min=1.0e-12)
        a = torch.zeros((m, ncol), device=device, dtype=dtype)
        bb = torch.zeros_like(a)
        cc = torch.zeros_like(a)
        r = torch.zeros_like(a)

        dzp0 = torch.clamp(z_col[:, 1] - z_col[:, 0], min=1.0e-12)
        bb[:, 0] = 1.0 + (1.0 - CNFAC) * fact[:, 0] * k_int[:, 0] / dzp0 - fact[:, 0] * dhsdT_g
        cc[:, 0] = -(1.0 - CNFAC) * fact[:, 0] * k_int[:, 0] / dzp0
        r[:, 0] = temp[:, 0] + fact[:, 0] * ((hs_top_snow_g - dhsdT_g * temp[:, 0]) + CNFAC * fn_old[:, 0])

        if ncol > 2:
            mid = slice(1, ncol - 1)
            prev = slice(0, ncol - 2)
            nxt = slice(2, ncol)
            dzm = torch.clamp(z_col[:, mid] - z_col[:, prev], min=1.0e-12)
            dzp = torch.clamp(z_col[:, nxt] - z_col[:, mid], min=1.0e-12)
            a[:, mid] = -(1.0 - CNFAC) * fact[:, mid] * k_int[:, prev] / dzm
            bb[:, mid] = 1.0 + (1.0 - CNFAC) * fact[:, mid] * (k_int[:, mid] / dzp + k_int[:, prev] / dzm)
            cc[:, mid] = -(1.0 - CNFAC) * fact[:, mid] * k_int[:, mid] / dzp
            r[:, mid] = temp[:, mid] + CNFAC * fact[:, mid] * (fn_old[:, mid] - fn_old[:, prev])

        dzm_bot = torch.clamp(z_col[:, -1] - z_col[:, -2], min=1.0e-12)
        a[:, -1] = -(1.0 - CNFAC) * fact[:, -1] * k_int[:, -2] / dzm_bot
        bb[:, -1] = 1.0 + (1.0 - CNFAC) * fact[:, -1] * k_int[:, -2] / dzm_bot
        r[:, -1] = temp[:, -1] - CNFAC * fact[:, -1] * fn_old[:, -2] + fact[:, -1] * fn_old[:, -1]

        solved = _batched_tridiagonal_torch(a, bb, cc, r, workspace=workspace)
        snow_t_solved = torch.minimum(solved[:, :nsnow], torch.full_like(solved[:, :nsnow], TFRZ))
        t_solved = solved[:, nsnow:]

        has_h2osfc = (h2osfc_g > 1.0e-8) & (frac_h2osfc_g > 1.0e-8)
        c_h2osfc = torch.clamp(CPLIQ * h2osfc_g / torch.clamp(frac_h2osfc_g, min=1.0e-12), min=1.0e-6)
        dz_h2osfc = torch.clamp(1.0e-3 * h2osfc_g / torch.clamp(frac_h2osfc_g, min=1.0e-12), min=1.0e-6)
        zh2osfc = 0.5e-3 * h2osfc_g
        thk1 = thk_soil_g[:, 0]
        z1 = z_g[:, 0]
        tk_h2osfc = TKWAT * thk1 * (z1 + zh2osfc) / torch.clamp(TKWAT * z1 + thk1 * zh2osfc, min=1.0e-12)
        dzm_h2osfc = 0.5 * dz_h2osfc + z1
        fn_h2osfc_old_g = tk_h2osfc * (t_g[:, 0] - t_h2osfc_g) / torch.clamp(dzm_h2osfc, min=1.0e-12)
        bb_h2osfc = (
            1.0
            + (1.0 - CNFAC) * (dt / c_h2osfc) * tk_h2osfc / torch.clamp(dzm_h2osfc, min=1.0e-12)
            - (dt / c_h2osfc) * dhsdT_g
        )
        c_h2osfc_soil = -(1.0 - CNFAC) * (dt / c_h2osfc) * tk_h2osfc / torch.clamp(dzm_h2osfc, min=1.0e-12)
        r_h2osfc = t_h2osfc_g + (dt / c_h2osfc) * (
            hs_h2osfc_g - dhsdT_g * t_h2osfc_g + CNFAC * fn_h2osfc_old_g
        )
        t_h2osfc_candidate = (r_h2osfc - c_h2osfc_soil * t_solved[:, 0]) / torch.clamp(bb_h2osfc, min=1.0e-12)
        fn_h2osfc_new_g = tk_h2osfc * (t_solved[:, 0] - t_h2osfc_candidate) / torch.clamp(dzm_h2osfc, min=1.0e-12)
        t_h2osfc_g = torch.where(has_h2osfc, t_h2osfc_candidate, t_h2osfc_g)
        fn_h2osfc_old_g = torch.where(has_h2osfc, fn_h2osfc_old_g, torch.zeros_like(fn_h2osfc_old_g))
        fn_h2osfc_new_g = torch.where(has_h2osfc, fn_h2osfc_new_g, torch.zeros_like(fn_h2osfc_new_g))

        mass_mask = h2osno_g > 0.0
        snow_xm = frac_sno_eff[:, None] * snow_cv * (solved[:, :nsnow] - TFRZ) / HFUS
        top_tinc = TFRZ - solved[:, 0]
        top_hm = frac_sno_eff * (dhsdT_g * top_tinc - top_tinc / torch.clamp(fact[:, 0], min=1.0e-30))
        snow_xm = snow_xm.clone()
        snow_xm[:, 0] = top_hm * dt / HFUS
        melt_mask = (snow_xm > 0.0) & mass_mask[:, None]
        freeze_mask = (snow_xm < 0.0) & mass_mask[:, None]
        melt_j = torch.minimum(snow_xm, wice0)
        snow_ice_after = torch.where(
            melt_mask,
            torch.clamp(wice0 - melt_j, min=0.0),
            torch.where(freeze_mask, torch.minimum(wmass0, wice0 - snow_xm), snow_ice),
        )
        snow_liq_after = torch.clamp(wmass0 - snow_ice_after, min=0.0)
        qflx_snow_melt_g = torch.sum(torch.where(melt_mask, torch.clamp(melt_j, min=0.0), torch.zeros_like(melt_j)), dim=1) / max(dt, 1.0e-12)
        qflx_snow_freeze_g = torch.sum(
            torch.where(freeze_mask, torch.clamp(snow_ice_after - wice0, min=0.0), torch.zeros_like(snow_ice_after)),
            dim=1,
        ) / max(dt, 1.0e-12)
        snow_ice = torch.where(mass_mask[:, None], snow_ice_after, snow_ice)
        snow_liq = torch.where(mass_mask[:, None], snow_liq_after, snow_liq)
        snow_temp_layer = torch.where(
            (snow_liq > 0.0) & (snow_ice > 0.0) & mass_mask[:, None],
            torch.full_like(snow_temp_layer, TFRZ),
            snow_temp_layer,
        )
        h2osno_g = torch.where(
            mass_mask,
            torch.sum(torch.clamp(snow_ice + snow_liq, min=0.0), dim=1),
            h2osno_g,
        )

        old_snow = h2osno_g.clone()
        old_h2osfc = h2osfc_g.clone()
        phase_mask = (h2osfc_g > 0.0) & (t_h2osfc_g <= TFRZ)
        rho_avg = torch.where(
            snow_depth_g > 0.0,
            torch.minimum(
                torch.full_like(h2osno_g, 800.0),
                old_snow / torch.clamp(frac_sno_eff * torch.clamp(snow_depth_g, min=1.0e-12), min=1.0e-12),
            ),
            torch.full_like(h2osno_g, 200.0),
        )
        tinc = TFRZ - t_h2osfc_g
        c_phase = CPLIQ * old_h2osfc / torch.clamp(frac_h2osfc_g, min=1.0e-12)
        hm = frac_h2osfc_g * (dhsdT_g * tinc - tinc * c_phase / max(dt, 1.0e-12))
        xm = hm * dt / HFUS
        temp1 = old_h2osfc + xm
        phase_keep = phase_mask & (temp1 >= 0.0)
        phase_absorb = phase_mask & ~phase_keep
        phase_hm_g = torch.where(phase_mask, hm, torch.full_like(hm, torch.nan))
        phase_xm_g = torch.where(phase_mask, xm, torch.full_like(xm, torch.nan))
        phase_temp1_g = torch.where(phase_mask, temp1, torch.full_like(temp1, torch.nan))

        last_ice = snow_ice[:, -1]
        last_ice = torch.where(phase_keep, torch.clamp(last_ice - xm, min=0.0), last_ice)
        last_ice = torch.where(phase_absorb, last_ice + old_h2osfc, last_ice)
        snow_ice = snow_ice.clone()
        snow_ice[:, -1] = last_ice

        c1 = (
            frac_sno_eff * (snow_cv[:, -1] - dhsdT_g * dt)
            if nsnow == 1
            else frac_sno_eff * snow_cv[:, -1]
        )
        c2_keep = torch.where(frac_h2osfc_g != 0.0, -CPLIQ * xm - frac_h2osfc_g * dhsdT_g * dt, torch.zeros_like(xm))
        c2_absorb = torch.where(frac_h2osfc_g != 0.0, frac_h2osfc_g * (c_phase - dhsdT_g * dt), torch.zeros_like(xm))
        denom_keep = c1 + c2_keep
        denom_absorb = c1 + c2_absorb
        mixed_keep = (c1 * snow_temp_layer[:, -1] + c2_keep * t_h2osfc_g) / torch.clamp(denom_keep, min=1.0e-12)
        mixed_absorb = (c1 * snow_temp_layer[:, -1] + c2_absorb * t_h2osfc_g) / torch.clamp(denom_absorb, min=1.0e-12)
        mix_keep = phase_keep & (denom_keep > 0.0)
        mix_absorb = phase_absorb & (denom_absorb > 0.0)
        last_temp = snow_temp_layer[:, -1]
        last_temp = torch.where(mix_keep, mixed_keep, last_temp)
        last_temp = torch.where(mix_absorb, mixed_absorb, last_temp)
        snow_temp_layer = snow_temp_layer.clone()
        snow_temp_layer[:, -1] = last_temp
        t_h2osfc_g = torch.where(mix_keep, mixed_keep, t_h2osfc_g)
        t_h2osfc_g = torch.where(mix_absorb, mixed_absorb, t_h2osfc_g)

        rho_absorb = (old_snow * rho_avg + old_h2osfc * DENICE) / torch.clamp(old_snow + old_h2osfc, min=1.0e-12)
        h2osno_g = torch.where(phase_keep, torch.clamp(old_snow - xm, min=0.0), h2osno_g)
        h2osno_g = torch.where(phase_absorb, old_snow + old_h2osfc, h2osno_g)
        h2osfc_g = torch.where(phase_keep, torch.clamp(old_h2osfc + xm, min=0.0), h2osfc_g)
        h2osfc_g = torch.where(phase_absorb, torch.zeros_like(h2osfc_g), h2osfc_g)
        rho_depth = torch.where(phase_absorb, rho_absorb, rho_avg)
        snow_depth_phase = _safe_snow_depth_from_mass_fraction_torch(
            h2osno_g,
            rho_depth,
            frac_sno_eff,
        )
        snow_depth_g = torch.where(phase_mask, snow_depth_phase, snow_depth_g)
        snow_temp_layer[:, -1] = torch.where(
            (snow_liq[:, -1] > 0.0) & (snow_ice[:, -1] > 0.0),
            torch.full_like(snow_temp_layer[:, -1], TFRZ),
            snow_temp_layer[:, -1],
        )

        snow_dz_state[rows, :nsnow] = snow_dz
        snow_ice_state[rows, :nsnow] = snow_ice
        snow_liq_state[rows, :nsnow] = snow_liq
        snow_t_state[rows, :nsnow] = snow_temp_layer
        snow_t_new[rows, :nsnow] = snow_t_solved

        fn_new = torch.zeros((m, ncol), device=device, dtype=dtype)
        fn_new[:, :-1] = k_int[:, :-1] * (solved[:, 1:] - solved[:, :-1]) / torch.clamp(
            z_col[:, 1:] - z_col[:, :-1],
            min=1.0e-12,
        )
        fn_new[:, -1] = eflx_bot_g
        fact_soil = fact[:, nsnow:]
        fn_old_soil = fn_old[:, nsnow:]
        fn_new_soil = fn_new[:, nsnow:]
        params_g = _selected_params(rows)
        t_pc_g, hliq_pc_g, hice_pc_g, imelt_g, _, _, _ = _soil_phase_change_torch(
            t_solved,
            hliq2.index_select(0, rows),
            hice2.index_select(0, rows),
            params_g,
            fact_soil,
            dhsdT_g,
            torch.zeros((m,), device=device, dtype=dtype),
            dt,
        )
        soil_ice_delta_g = hice_pc_g.reshape(m, nlev) - hice2.index_select(0, rows)
        qflx_soil_melt_profile_g = torch.clamp(-soil_ice_delta_g, min=0.0) / max(dt, 1.0e-12)
        qflx_soil_freeze_profile_g = torch.clamp(soil_ice_delta_g, min=0.0) / max(dt, 1.0e-12)

        t_pc.index_copy_(0, rows, t_pc_g.reshape(m, nlev))
        hliq_pc.index_copy_(0, rows, hliq_pc_g.reshape(m, nlev))
        hice_pc.index_copy_(0, rows, hice_pc_g.reshape(m, nlev))
        imelt_pc.index_copy_(0, rows, imelt_g.reshape(m, nlev))
        qflx_soil_melt_profile.index_copy_(0, rows, qflx_soil_melt_profile_g)
        qflx_soil_freeze_profile.index_copy_(0, rows, qflx_soil_freeze_profile_g)
        t_h2osfc_out.index_copy_(0, rows, t_h2osfc_g)
        h2osfc_out.index_copy_(0, rows, h2osfc_g)
        frac_h2osfc_out.index_copy_(0, rows, frac_h2osfc_g)
        h2osno_out.index_copy_(0, rows, h2osno_g)
        snow_depth_out.index_copy_(0, rows, snow_depth_g)
        eflx_fgr12.index_copy_(0, rows, -CNFAC * fn_old_soil[:, 0] - (1.0 - CNFAC) * fn_new_soil[:, 0])
        fn_h2osfc_old_out.index_copy_(0, rows, fn_h2osfc_old_g)
        fn_h2osfc_new_out.index_copy_(0, rows, fn_h2osfc_new_g)
        qflx_snow_melt.index_copy_(0, rows, qflx_snow_melt_g)
        qflx_snow_freeze.index_copy_(0, rows, qflx_snow_freeze_g)
        phase_hm_flat.index_copy_(0, rows, phase_hm_g)
        phase_xm_flat.index_copy_(0, rows, phase_xm_g)
        phase_temp1_flat.index_copy_(0, rows, phase_temp1_g)

    snow_layers_new = SnowLayerState(
        snl=snow_snl.reshape(batch_shape),
        dz=snow_dz_state.reshape(batch_shape + (MAX_SNOW_LAYERS,)),
        h2osoi_ice=snow_ice_state.reshape(batch_shape + (MAX_SNOW_LAYERS,)),
        h2osoi_liq=snow_liq_state.reshape(batch_shape + (MAX_SNOW_LAYERS,)),
        t_snow=snow_t_state.reshape(batch_shape + (MAX_SNOW_LAYERS,)),
    )

    return SoilTemperatureResultTorch(
        state=SoilTemperatureStateTorch(
            t_soisno=t_pc.reshape(batch_shape + (nlev,)),
            h2osoi_liq=hliq_pc.reshape(batch_shape + (nlev,)),
            h2osoi_ice=hice_pc.reshape(batch_shape + (nlev,)),
            frac_sno=_broadcast_batch(0.0 if state.frac_sno is None else state.frac_sno, batch_shape).reshape(batch_shape),
            t_h2osfc=t_h2osfc_out.reshape(batch_shape),
            h2osfc=h2osfc_out.reshape(batch_shape),
            frac_h2osfc=frac_h2osfc_out.reshape(batch_shape),
            h2osno=h2osno_out.reshape(batch_shape),
            snow_depth=snow_depth_out.reshape(batch_shape),
            snow_t=snow_t_new.reshape(batch_shape + (MAX_SNOW_LAYERS,)),
            snow_layers=snow_layers_new,
        ),
        eflx_fgr12=eflx_fgr12.reshape(batch_shape),
        fn_h2osfc_old=fn_h2osfc_old_out.reshape(batch_shape),
        fn_h2osfc_new=fn_h2osfc_new_out.reshape(batch_shape),
        imelt=imelt_pc,
        qflx_snow_melt=qflx_snow_melt.reshape(batch_shape),
        qflx_scalar_snow_melt=torch.zeros(batch_shape, device=device, dtype=dtype),
        qflx_snow_freeze=qflx_snow_freeze.reshape(batch_shape),
        qflx_soil_melt=torch.sum(qflx_soil_melt_profile.reshape(batch_shape + (nlev,)), dim=-1),
        qflx_soil_freeze=torch.sum(qflx_soil_freeze_profile.reshape(batch_shape + (nlev,)), dim=-1),
        qflx_soil_melt_profile=qflx_soil_melt_profile.reshape(batch_shape + (nlev,)),
        qflx_soil_freeze_profile=qflx_soil_freeze_profile.reshape(batch_shape + (nlev,)),
        phase_hm_h2osfc=phase_hm_h2osfc,
        phase_xm_h2osfc=phase_xm_h2osfc,
        phase_temp1_h2osfc=phase_temp1_h2osfc,
    )


def soil_thermal_properties_torch(
    state: SoilTemperatureStateTorch,
    params: SoilThermalParametersTorch,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    t = state.t_soisno
    hliq = state.h2osoi_liq
    hice = state.h2osoi_ice
    batch_shape = t.shape[:-1]
    nlev = t.shape[-1]

    dz = _broadcast_layers(params.dz, batch_shape, nlev)
    z = _broadcast_layers(params.z, batch_shape, nlev)
    zi = _broadcast_layers(params.zi, batch_shape, nlev)
    watsat = _broadcast_layers(params.watsat, batch_shape, nlev)
    tkmg = _broadcast_layers(params.tkmg, batch_shape, nlev)
    tkdry = _broadcast_layers(params.tkdry, batch_shape, nlev)
    csol = _broadcast_layers(params.csol, batch_shape, nlev)

    satw = (hliq / DENH2O + hice / DENICE) / torch.clamp(dz * watsat, min=1.0e-12)
    satw = torch.clamp(satw, 0.0, 1.0)
    dke = torch.where(
        satw > 1.0e-7,
        torch.where(t >= TFRZ, torch.clamp(torch.log10(torch.clamp(satw, min=1.0e-12)) + 1.0, min=0.0), satw),
        torch.zeros_like(satw),
    )
    fl = torch.where(
        (hliq / (DENH2O * dz) + hice / (DENICE * dz)) > 0.0,
        (hliq / (DENH2O * dz)) / torch.clamp(hliq / (DENH2O * dz) + hice / (DENICE * dz), min=1.0e-12),
        torch.zeros_like(satw),
    )
    dksat = tkmg * (TKWAT ** (fl * watsat)) * (TKICE ** ((1.0 - fl) * watsat))
    thk = torch.where(satw > 1.0e-7, dke * dksat + (1.0 - dke) * tkdry, tkdry)

    tk = torch.zeros_like(thk)
    if nlev > 1:
        num = thk[..., :-1] * thk[..., 1:] * (z[..., 1:] - z[..., :-1])
        den = thk[..., :-1] * (z[..., 1:] - zi[..., :-1]) + thk[..., 1:] * (zi[..., :-1] - z[..., :-1])
        tk[..., :-1] = num / torch.clamp(den, min=1.0e-12)
    tk[..., -1] = 0.0

    cv = csol * (1.0 - watsat) * dz + hice * CPICE + hliq * CPLIQ
    return cv, thk, tk


def _soil_temperature_step_torch_no_snow(
    state: SoilTemperatureStateTorch,
    params: SoilThermalParametersTorch,
    forcing: SoilTemperatureForcingTorch,
    dt: float,
    workspace: SoilTemperatureTorchWorkspace | None = None,
    *,
    snow_enabled: bool = True,
) -> SoilTemperatureResultTorch:
    t = state.t_soisno
    hliq = state.h2osoi_liq
    hice = state.h2osoi_ice
    batch_shape = t.shape[:-1]
    nlev = t.shape[-1]
    device = t.device
    dtype = t.dtype

    zero_batch = (
        torch.zeros((), device=device, dtype=dtype)
        if state.h2osno is None or state.snow_depth is None
        else None
    )
    h2osno = _broadcast_batch(zero_batch if state.h2osno is None else state.h2osno, batch_shape)
    snow_depth = _broadcast_batch(zero_batch if state.snow_depth is None else state.snow_depth, batch_shape)
    h2osno = h2osno.reshape(-1)
    snow_depth = snow_depth.reshape(-1)
    scalar_snow_mask = (h2osno > MIN_SNOW_LAYER_MASS) & (snow_depth > 0.0)
    if state.snow_layers is not None and getattr(state.snow_layers, "snl", None) is not None:
        snl_layers = torch.broadcast_to(
            torch.as_tensor(state.snow_layers.snl, device=device, dtype=torch.int32),
            batch_shape,
        ).reshape(-1)
        dz_layers = _broadcast_layers(
            torch.as_tensor(state.snow_layers.dz, device=device, dtype=dtype),
            batch_shape,
            MAX_SNOW_LAYERS,
        ).reshape(-1, MAX_SNOW_LAYERS)
        ice_layers = _broadcast_layers(
            torch.as_tensor(state.snow_layers.h2osoi_ice, device=device, dtype=dtype),
            batch_shape,
            MAX_SNOW_LAYERS,
        ).reshape(-1, MAX_SNOW_LAYERS)
        liq_layers = _broadcast_layers(
            torch.as_tensor(state.snow_layers.h2osoi_liq, device=device, dtype=dtype),
            batch_shape,
            MAX_SNOW_LAYERS,
        ).reshape(-1, MAX_SNOW_LAYERS)
        nsnow = torch.clamp(-snl_layers, min=0)
        layer_index = torch.arange(MAX_SNOW_LAYERS, device=device).reshape(1, MAX_SNOW_LAYERS)
        active_layers = layer_index < nsnow[:, None]
        layer_mass = torch.sum(torch.clamp(ice_layers + liq_layers, min=0.0), dim=-1)
        active_depth = torch.sum(
            torch.where(active_layers, torch.clamp(dz_layers, min=0.0), torch.zeros_like(dz_layers)),
            dim=-1,
        )
        scalar_depth = torch.clamp(dz_layers[:, 0], min=0.0)
        has_layer_storage = layer_mass > 1.0e-8
        h2osno = torch.where(has_layer_storage, layer_mass, h2osno)
        snow_depth = torch.where(
            nsnow > 0,
            active_depth,
            torch.where(has_layer_storage, scalar_depth, snow_depth),
        )
        scalar_snow_mask = (nsnow == 0) & (h2osno > MIN_SNOW_LAYER_MASS) & (snow_depth > 0.0)

    hs_soil = _broadcast_batch(forcing.hs_soil, batch_shape).reshape(-1)
    dhsdT = _broadcast_batch(forcing.dhsdT, batch_shape).reshape(-1)
    eflx_bot = _broadcast_batch(forcing.eflx_bot, batch_shape).reshape(-1)
    hs_h2osfc = _broadcast_batch(
        torch.zeros((), device=device, dtype=dtype) if forcing.hs_h2osfc is None else forcing.hs_h2osfc,
        batch_shape,
    ).reshape(-1)

    h2osfc = _broadcast_batch(
        torch.zeros((), device=device, dtype=dtype) if state.h2osfc is None else state.h2osfc,
        batch_shape,
    ).reshape(-1)
    frac_h2osfc = _broadcast_batch(
        torch.zeros((), device=device, dtype=dtype) if state.frac_h2osfc is None else state.frac_h2osfc,
        batch_shape,
    ).reshape(-1)
    if state.t_h2osfc is None:
        t_h2osfc = t[..., 0].reshape(-1).clone()
    else:
        t_h2osfc = _broadcast_batch(state.t_h2osfc, batch_shape).reshape(-1)
    has_h2osfc = (h2osfc > 1.0e-8) & (frac_h2osfc > 1.0e-8)

    dz = _broadcast_layers(params.dz, batch_shape, nlev)
    z = _broadcast_layers(params.z, batch_shape, nlev)
    cv, thk, tk = soil_thermal_properties_torch(state, params)

    t2 = t.reshape(-1, nlev)
    cv2 = cv.reshape(-1, nlev)
    tk2 = tk.reshape(-1, nlev)
    thk2 = thk.reshape(-1, nlev)
    dz2 = dz.reshape(-1, nlev)
    z2 = z.reshape(-1, nlev)

    c_h2osfc = torch.where(has_h2osfc, torch.clamp(CPLIQ * h2osfc / torch.clamp(frac_h2osfc, min=1.0e-12), min=1.0e-6), torch.full_like(h2osfc, 1.0e-6))
    dz_h2osfc = torch.where(has_h2osfc, torch.clamp(1.0e-3 * h2osfc / torch.clamp(frac_h2osfc, min=1.0e-12), min=1.0e-6), torch.full_like(h2osfc, 1.0e-6))
    tk_h2osfc = TKWAT * thk2[:, 0] * (z2[:, 0] + 0.5e-3 * h2osfc) / torch.clamp(TKWAT * z2[:, 0] + thk2[:, 0] * (0.5e-3 * h2osfc), min=1.0e-12)

    nunknown = nlev + 1
    if workspace is None:
        fact = torch.empty_like(t2)
        fn_old = torch.empty_like(t2)
        a = torch.empty((t2.shape[0], nunknown), device=device, dtype=dtype)
        b = torch.empty_like(a)
        c = torch.empty_like(a)
        r = torch.empty_like(a)
    else:
        fact = workspace.tensor("fact", t2.shape, device, dtype)
        fn_old = workspace.tensor("fn_old", t2.shape, device, dtype)
        a = workspace.tensor("a", (t2.shape[0], nunknown), device, dtype)
        b = workspace.tensor("b", (t2.shape[0], nunknown), device, dtype)
        c = workspace.tensor("c", (t2.shape[0], nunknown), device, dtype)
        r = workspace.tensor("r", (t2.shape[0], nunknown), device, dtype)
    top_den = 0.5 * (z2[:, 0] + CAPR * z2[:, 1])
    fact[:, 0] = dt / cv2[:, 0] * dz2[:, 0] / torch.clamp(top_den, min=1.0e-12)
    fn_old[:, 0] = tk2[:, 0] * (t2[:, 1] - t2[:, 0]) / torch.clamp(z2[:, 1] - z2[:, 0], min=1.0e-12)
    # Vectorized over interior layers [1, nlev-2]
    if nlev > 2:
        _jsi = slice(1, nlev - 1)
        _jp1i = slice(2, nlev)
        fact[:, _jsi]   = dt / cv2[:, _jsi]
        fn_old[:, _jsi] = tk2[:, _jsi] * (t2[:, _jp1i] - t2[:, _jsi]) / torch.clamp(z2[:, _jp1i] - z2[:, _jsi], min=1.0e-12)
    fact[:, -1] = dt / cv2[:, -1]
    fn_old[:, -1] = eflx_bot

    dzm_h2osfc = 0.5 * dz_h2osfc + z2[:, 0]
    fn_h2osfc_old = tk_h2osfc * (t2[:, 0] - t_h2osfc) / torch.clamp(dzm_h2osfc, min=1.0e-12)

    b[:, 0] = 1.0 + (1.0 - CNFAC) * (dt / c_h2osfc) * tk_h2osfc / torch.clamp(dzm_h2osfc, min=1.0e-12) - (dt / c_h2osfc) * dhsdT
    c[:, 0] = -(1.0 - CNFAC) * (dt / c_h2osfc) * tk_h2osfc / torch.clamp(dzm_h2osfc, min=1.0e-12)
    r[:, 0] = t_h2osfc + (dt / c_h2osfc) * (hs_h2osfc - dhsdT * t_h2osfc + CNFAC * fn_h2osfc_old)

    dzp = z2[:, 1] - z2[:, 0]
    b[:, 1] = 1.0 + (1.0 - CNFAC) * fact[:, 0] * tk2[:, 0] / torch.clamp(dzp, min=1.0e-12) - fact[:, 0] * dhsdT
    c[:, 1] = -(1.0 - CNFAC) * fact[:, 0] * tk2[:, 0] / torch.clamp(dzp, min=1.0e-12)
    r[:, 1] = t2[:, 0] + fact[:, 0] * ((hs_soil - dhsdT * t2[:, 0]) + CNFAC * fn_old[:, 0])
    b[:, 1] = torch.where(
        has_h2osfc,
        b[:, 1]
        + frac_h2osfc * ((1.0 - CNFAC) * fact[:, 0] * tk_h2osfc / torch.clamp(dzm_h2osfc, min=1.0e-12) + fact[:, 0] * dhsdT),
        b[:, 1],
    )
    a[:, 1] = torch.where(
        has_h2osfc,
        -frac_h2osfc * (1.0 - CNFAC) * fact[:, 0] * tk_h2osfc / torch.clamp(dzm_h2osfc, min=1.0e-12),
        torch.zeros_like(hs_soil),
    )
    r[:, 1] = torch.where(
        has_h2osfc,
        r[:, 1] - frac_h2osfc * fact[:, 0] * ((hs_soil - dhsdT * t2[:, 0]) + CNFAC * fn_h2osfc_old),
        r[:, 1],
    )

    # Vectorized tridiagonal build for interior soil layers [j=1..nlev-2, jj=2..nlev-1]
    if nlev > 2:
        _jj = slice(2, nlev)       # jj = j+1, j in [1, nlev-2]
        _j  = slice(1, nlev - 1)   # j in [1, nlev-2]
        _jm1 = slice(0, nlev - 2)  # j-1
        _jp1 = slice(2, nlev)      # j+1
        dzm_v = torch.clamp(z2[:, _j] - z2[:, _jm1], min=1.0e-12)
        dzp_v = torch.clamp(z2[:, _jp1] - z2[:, _j], min=1.0e-12)
        a[:, _jj] = -(1.0 - CNFAC) * fact[:, _j] * tk2[:, _jm1] / dzm_v
        b[:, _jj] = 1.0 + (1.0 - CNFAC) * fact[:, _j] * (tk2[:, _j] / dzp_v + tk2[:, _jm1] / dzm_v)
        c[:, _jj] = -(1.0 - CNFAC) * fact[:, _j] * tk2[:, _j] / dzp_v
        r[:, _jj] = t2[:, _j] + CNFAC * fact[:, _j] * (fn_old[:, _j] - fn_old[:, _jm1])

    dzm = z2[:, -1] - z2[:, -2]
    a[:, -1] = -(1.0 - CNFAC) * fact[:, -1] * tk2[:, -2] / torch.clamp(dzm, min=1.0e-12)
    b[:, -1] = 1.0 + (1.0 - CNFAC) * fact[:, -1] * tk2[:, -2] / torch.clamp(dzm, min=1.0e-12)
    r[:, -1] = t2[:, -1] - CNFAC * fact[:, -1] * fn_old[:, -2] + fact[:, -1] * fn_old[:, -1]

    t_all_new = _batched_tridiagonal_torch(a, b, c, r, workspace=workspace)
    t_new = t_all_new[:, 1:]
    t_h2osfc_new = torch.where(has_h2osfc, t_all_new[:, 0], t_new[:, 0])
    fn_h2osfc_old = torch.where(has_h2osfc, fn_h2osfc_old, torch.zeros_like(fn_h2osfc_old))
    fn_h2osfc_new = torch.where(
        has_h2osfc,
        tk_h2osfc * (t_new[:, 0] - t_h2osfc_new) / torch.clamp(dzm_h2osfc, min=1.0e-12),
        torch.zeros_like(fn_h2osfc_old),
    )
    eflx_fgr12 = -CNFAC * fn_old[:, 0] - (1.0 - CNFAC) * (
        tk2[:, 0] * (t_new[:, 1] - t_new[:, 0]) / torch.clamp(z2[:, 1] - z2[:, 0], min=1.0e-12)
    )

    t_pc, hliq_pc, hice_pc, imelt, h2osno_pc, snow_depth_pc, qflx_scalar_snow_melt = _soil_phase_change_torch(
        t_new.reshape(batch_shape + (nlev,)),
        hliq,
        hice,
        params,
        fact.reshape(batch_shape + (nlev,)),
        dhsdT.reshape(batch_shape),
        frac_h2osfc.reshape(batch_shape),
        dt,
        scalar_snow_mask=scalar_snow_mask.reshape(batch_shape),
        scalar_h2osno=h2osno.reshape(batch_shape),
        scalar_snow_depth=snow_depth.reshape(batch_shape),
    )
    soil_ice_delta = hice_pc - hice
    qflx_soil_freeze_profile = torch.clamp(soil_ice_delta, min=0.0) / max(dt, 1.0e-12)
    qflx_soil_melt_profile = torch.clamp(-soil_ice_delta, min=0.0) / max(dt, 1.0e-12)
    if h2osno_pc is not None:
        h2osno = h2osno_pc.reshape(-1)
    if snow_depth_pc is not None:
        snow_depth = snow_depth_pc.reshape(-1)
    phase_hm_h2osfc = None
    phase_xm_h2osfc = None
    phase_temp1_h2osfc = None
    if snow_enabled:
        phase_hm_h2osfc = torch.full(batch_shape, torch.nan, device=device, dtype=dtype)
        phase_xm_h2osfc = torch.full(batch_shape, torch.nan, device=device, dtype=dtype)
        phase_temp1_h2osfc = torch.full(batch_shape, torch.nan, device=device, dtype=dtype)
        phase_hm_flat = phase_hm_h2osfc.reshape(-1)
        phase_xm_flat = phase_xm_h2osfc.reshape(-1)
        phase_temp1_flat = phase_temp1_h2osfc.reshape(-1)
        frac_sno_eff_flat = (
            _broadcast_batch(state.frac_sno, batch_shape).reshape(-1)
            if state.frac_sno is not None
            else torch.zeros_like(h2osfc)
        )
        phase_mask = has_h2osfc & (t_h2osfc <= TFRZ) & (h2osfc > 0.0)
        tinc = TFRZ - t_h2osfc
        c_phase = CPLIQ * h2osfc / torch.clamp(frac_h2osfc, min=1.0e-12)
        hm = frac_h2osfc * (dhsdT * tinc - tinc * c_phase / max(dt, 1.0e-12))
        xm = hm * dt / HFUS
        temp1 = h2osfc + xm
        phase_hm_flat[:] = torch.where(phase_mask, hm, phase_hm_flat)
        phase_xm_flat[:] = torch.where(phase_mask, xm, phase_xm_flat)
        phase_temp1_flat[:] = torch.where(phase_mask, temp1, phase_temp1_flat)
        apply_phase = phase_mask & (temp1 >= 0.0)
        z_avg = frac_sno_eff_flat * snow_depth
        rho_avg = torch.where(
            z_avg > 0.0,
            torch.minimum(torch.full_like(h2osno, 800.0), h2osno / torch.clamp(z_avg, min=1.0e-12)),
            torch.full_like(h2osno, 200.0),
        )
        h2osno = torch.where(apply_phase, h2osno - xm, h2osno)
        h2osfc = torch.where(apply_phase, h2osfc + xm, h2osfc)
        snow_depth = torch.where(
            apply_phase,
            _safe_snow_depth_from_mass_fraction_torch(h2osno, rho_avg, frac_sno_eff_flat),
            snow_depth,
        )

    return SoilTemperatureResultTorch(
        state=SoilTemperatureStateTorch(
            t_soisno=t_pc,
            h2osoi_liq=hliq_pc,
            h2osoi_ice=hice_pc,
            t_h2osfc=t_h2osfc_new.reshape(batch_shape),
            h2osfc=h2osfc.reshape(batch_shape),
            frac_h2osfc=frac_h2osfc.reshape(batch_shape),
            h2osno=h2osno.reshape(batch_shape),
            snow_depth=snow_depth.reshape(batch_shape),
            snow_t=state.snow_t,
        ),
        eflx_fgr12=eflx_fgr12.reshape(batch_shape),
        fn_h2osfc_old=fn_h2osfc_old.reshape(batch_shape),
        fn_h2osfc_new=fn_h2osfc_new.reshape(batch_shape),
        imelt=imelt,
        qflx_snow_melt=qflx_scalar_snow_melt,
        qflx_scalar_snow_melt=qflx_scalar_snow_melt,
        qflx_snow_freeze=torch.zeros(batch_shape, device=device, dtype=dtype),
        qflx_soil_melt=torch.sum(qflx_soil_melt_profile, dim=-1),
        qflx_soil_freeze=torch.sum(qflx_soil_freeze_profile, dim=-1),
        qflx_soil_melt_profile=qflx_soil_melt_profile,
        qflx_soil_freeze_profile=qflx_soil_freeze_profile,
        phase_hm_h2osfc=phase_hm_h2osfc,
        phase_xm_h2osfc=phase_xm_h2osfc,
        phase_temp1_h2osfc=phase_temp1_h2osfc,
    )


def soil_temperature_step_torch(
    state: SoilTemperatureStateTorch,
    params: SoilThermalParametersTorch,
    forcing: SoilTemperatureForcingTorch,
    dt: float,
    workspace: SoilTemperatureTorchWorkspace | None = None,
    *,
    snow_enabled: bool = True,
) -> SoilTemperatureResultTorch:
    t = state.t_soisno
    batch_shape = t.shape[:-1]
    device = t.device
    dtype = t.dtype

    if snow_enabled:
        if state.snow_layers is not None and getattr(state.snow_layers, "snl", None) is not None:
            snl = torch.broadcast_to(
                torch.as_tensor(state.snow_layers.snl, device=device, dtype=torch.int32),
                batch_shape,
            )
            has_active_snow = torch.any(snl < 0)
        else:
            zero_batch = (
                torch.zeros((), device=device, dtype=dtype)
                if state.h2osno is None or state.snow_depth is None
                else None
            )
            h2osno = _broadcast_batch(zero_batch if state.h2osno is None else state.h2osno, batch_shape)
            snow_depth = _broadcast_batch(zero_batch if state.snow_depth is None else state.snow_depth, batch_shape)
            has_active_snow = torch.any((h2osno > MIN_SNOW_LAYER_MASS) & (snow_depth >= 0.01))
        if has_active_snow:
            return _snow_soil_temperature_step_torch(state, params, forcing, dt, workspace=workspace)

    return _soil_temperature_step_torch_no_snow(
        state,
        params,
        forcing,
        dt,
        workspace=workspace,
        snow_enabled=snow_enabled,
    )


__all__ = [
    "SoilThermalParametersTorch",
    "SoilTemperatureStateTorch",
    "SoilTemperatureForcingTorch",
    "SoilTemperatureResultTorch",
    "SoilTemperatureTorchWorkspace",
    "thermal_params_from_texture_torch",
    "soil_thermal_properties_torch",
    "soil_temperature_step_torch",
]
