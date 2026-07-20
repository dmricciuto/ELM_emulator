"""Vectorized Python port of the core ELM soil-temperature step.

This module follows the non-urban, no-snow, no-standing-water soil path in
``SoilTemperatureMod.F90`` as closely as practical in NumPy:

- ``SoilStateType`` texture-derived thermal properties
- ``SoilTemperatureMod::SoilThermProp``
- ``SoilTemperatureMod::ComputeHeatDiffFluxAndFactor``
- ``SoilTemperatureMod::SetRHSVec_SoilNonUrban``
- ``SoilTemperatureMod::SetMatrix_SoilNonUrban``
- ``BandDiagonalMod::BandDiagonal`` restricted to the soil-only tridiagonal case

Scope
-----
Implemented:
- ELM-like thermal-property calculation from texture and current water/ice state
- implicit Crank-Nicolson soil-temperature update
- lower-boundary heat-flux forcing
- top conductive flux diagnostic ``eflx_fgr12``

Deliberately omitted in this first version:
- snow layers
- standing surface water temperature solve
- urban columns
- phase change / enthalpy correction
- the broader canopy/ground flux assembly in ``ComputeGroundHeatFluxAndDeriv``

That keeps the interface aligned with the surrogate split you were converging on:
the surface model provides the net upper boundary heat flux and its temperature
derivative, while this module updates the soil column.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .elm_hydrology import SoilTextureProfile, elm_soil_grid, pedotransfer_cosby1984_table5


R8 = np.float64

DENH2O = 1000.0
DENICE = 917.0
TFRZ = 273.15
TKAIR = 0.023
TKICE = 2.290
TKWAT = 0.57
CPLIQ = 4188.0
CPICE = 2117.27
CAPR = 0.34
CNFAC = 0.5
THK_BEDROCK = 3.0
CSOL_BEDROCK = 2.0e6
OM_TKM = 0.25
OM_TKD = 0.05
OM_CSOL = 2.5
ZSAPRIC = 0.5
HFUS = 3.337e5
GRAV = 9.80616


def _as_r8(x: np.ndarray | float) -> np.ndarray:
    return np.asarray(x, dtype=R8)


def _broadcast_to_layers(arr: np.ndarray, batch_shape: tuple[int, ...], nlev: int) -> np.ndarray:
    arr = _as_r8(arr)
    return np.broadcast_to(arr, batch_shape + (nlev,))


def _broadcast_to_batch(arr: np.ndarray, batch_shape: tuple[int, ...]) -> np.ndarray:
    arr = _as_r8(arr)
    return np.broadcast_to(arr, batch_shape)


def _batched_tridiagonal(a: np.ndarray, b: np.ndarray, c: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Solve many tridiagonal systems vectorized over batch."""
    nbatch, n = b.shape
    cp = np.zeros_like(c)
    dp = np.zeros_like(r)
    x = np.zeros_like(r)

    cp[:, 0] = c[:, 0] / b[:, 0]
    dp[:, 0] = r[:, 0] / b[:, 0]

    for j in range(1, n):
        denom = b[:, j] - a[:, j] * cp[:, j - 1]
        cp[:, j] = np.where(j < n - 1, c[:, j] / denom, 0.0)
        dp[:, j] = (r[:, j] - a[:, j] * dp[:, j - 1]) / denom

    x[:, -1] = dp[:, -1]
    for j in range(n - 2, -1, -1):
        x[:, j] = dp[:, j] - cp[:, j] * x[:, j + 1]

    return x


@dataclass
class SoilThermalParameters:
    """Static or slowly varying soil-column thermal parameters."""

    dz: np.ndarray
    z: np.ndarray
    zi: np.ndarray
    watsat: np.ndarray
    bsw: np.ndarray
    sucsat: np.ndarray
    tkmg: np.ndarray
    tkdry: np.ndarray
    csol: np.ndarray
    nlevbed: Optional[np.ndarray] = None


@dataclass
class SoilTemperatureState:
    """Current soil thermal and moisture state."""

    t_soisno: np.ndarray
    h2osoi_liq: np.ndarray
    h2osoi_ice: np.ndarray
    frac_sno: Optional[np.ndarray] = None
    snow_layers: Optional[object] = None
    t_h2osfc: Optional[np.ndarray] = None
    h2osfc: Optional[np.ndarray] = None
    frac_h2osfc: Optional[np.ndarray] = None
    h2osno: Optional[np.ndarray] = None
    snow_depth: Optional[np.ndarray] = None
    snow_t: Optional[np.ndarray] = None


@dataclass
class SnowLayerState:
    snl: np.ndarray
    dz: np.ndarray
    h2osoi_ice: np.ndarray
    h2osoi_liq: np.ndarray
    t_snow: np.ndarray
    snw_rds: Optional[np.ndarray] = None


@dataclass
class SoilTemperatureForcing:
    """Upper and lower boundary forcing for one soil-temperature step."""

    hs_soil: np.ndarray
    dhsdT: np.ndarray
    eflx_bot: np.ndarray
    hs_top_snow: Optional[np.ndarray] = None
    hs_h2osfc: Optional[np.ndarray] = None


@dataclass
class SoilTemperatureResult:
    """Updated state and diagnostics from one temperature solve."""

    state: SoilTemperatureState
    cv: np.ndarray
    thk: np.ndarray
    tk: np.ndarray
    fact: np.ndarray
    fn_old: np.ndarray
    fn_new: np.ndarray
    eflx_fgr12: np.ndarray
    tk_h2osfc: np.ndarray
    fn_h2osfc_old: np.ndarray
    fn_h2osfc_new: np.ndarray
    imelt: np.ndarray
    qflx_snow_melt: np.ndarray | None = None
    qflx_scalar_snow_melt: np.ndarray | None = None
    qflx_snow_freeze: np.ndarray | None = None
    phase_hm_h2osfc: np.ndarray | None = None
    phase_xm_h2osfc: np.ndarray | None = None
    phase_temp1_h2osfc: np.ndarray | None = None


MAX_SNOW_LAYERS = 5
MIN_SNOW_LAYER_MASS = 1.0e-6
RHO_ICE = 917.0
SNOW_GEOMETRY_MIN_DENSITY = 50.0
SNOW_DEPTH_FRAC_MIN = 1.0e-3
SNOW_K_TMPS = np.array([223.0, 248.0, 263.0, 268.0, 273.0], dtype=R8)
SNOW_K_COE1 = np.array([2.564, 2.172, 1.985, 1.883, 1.776], dtype=R8)
SNOW_K_COE2 = np.array([-0.059, 0.015, 0.073, 0.107, 0.147], dtype=R8)
SNOW_K_COE3 = np.array([0.0205, 0.0252, 0.0336, 0.0386, 0.0455], dtype=R8)


def _safe_snow_depth_from_mass_fraction(h2osno: float, density: float, frac_sno: float) -> float:
    """Convert SWE to snow depth without letting tiny snow fractions explode."""
    mass = max(float(h2osno), 0.0) if np.isfinite(h2osno) else 0.0
    rho = float(density) if np.isfinite(density) else 200.0
    rho = min(max(rho, SNOW_GEOMETRY_MIN_DENSITY), DENICE)
    frac = float(frac_sno) if np.isfinite(frac_sno) else 0.0
    frac = min(max(frac, 0.0), 1.0)
    if frac > SNOW_DEPTH_FRAC_MIN:
        return min(mass / max(rho * frac, 1.0e-12), mass / SNOW_GEOMETRY_MIN_DENSITY)
    return mass / DENICE


def _snow_layer_count(depth: float) -> int:
    """ELM-like first-pass snow layer count from snow depth."""
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


def _elm_snow_layer_depths(snow_depth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ELM's active snow-layer count and top-to-bottom layer depths."""
    depth = np.asarray(snow_depth, dtype=R8)
    nsnow = np.zeros(depth.shape, dtype=np.int32)
    dz = np.zeros(depth.shape + (MAX_SNOW_LAYERS,), dtype=R8)

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


def _snow_layers_from_storage_fallback(
    h2osno: np.ndarray,
    snow_depth: np.ndarray,
    snow_t: np.ndarray | None = None,
) -> "SnowLayerState":
    h2osno = np.asarray(h2osno, dtype=R8)
    snow_depth = np.asarray(snow_depth, dtype=R8)
    shape = h2osno.shape
    snl = np.zeros(shape, dtype=np.int32)
    dz = np.zeros(shape + (MAX_SNOW_LAYERS,), dtype=R8)
    ice = np.zeros_like(dz)
    liq = np.zeros_like(dz)
    temp = np.zeros_like(dz)
    if snow_t is not None:
        temp_seed = np.broadcast_to(np.asarray(snow_t, dtype=R8), shape + (MAX_SNOW_LAYERS,))
    else:
        temp_seed = np.broadcast_to(np.full(shape, TFRZ, dtype=R8)[..., None], shape + (MAX_SNOW_LAYERS,))
    nsnow, dz_template = _elm_snow_layer_depths(snow_depth)
    for k in range(1, MAX_SNOW_LAYERS + 1):
        mask = nsnow == k
        if not np.any(mask):
            continue
        snl = np.where(mask, -k, snl)
        dz[mask, :k] = np.maximum(dz_template[mask, :k], 1.0e-6)
        mass_frac = dz_template[mask, :k] / np.maximum(snow_depth[mask, None], 1.0e-12)
        ice[mask, :k] = np.maximum(h2osno[mask, None] * mass_frac, 0.0)
        temp[mask, :k] = temp_seed[mask, :k]
    inactive_scalar = (nsnow == 0) & (h2osno > 1.0e-8)
    if np.any(inactive_scalar):
        dz[inactive_scalar, 0] = np.maximum(snow_depth[inactive_scalar], 0.0)
        ice[inactive_scalar, 0] = h2osno[inactive_scalar]
        temp[inactive_scalar, 0] = temp_seed[inactive_scalar, 0]
    return SnowLayerState(snl=snl, dz=dz, h2osoi_ice=ice, h2osoi_liq=liq, t_snow=temp)


def _interface_conductivity(k1: float, k2: float) -> float:
    return 2.0 * k1 * k2 / max(k1 + k2, 1.0e-12)


def _snow_thermal_conductivity(tempk: np.ndarray, density: np.ndarray) -> np.ndarray:
    """ELM-like snow thermal conductivity parameterization."""
    tempk = np.asarray(tempk, dtype=R8)
    density = np.clip(np.asarray(density, dtype=R8), 0.0, RHO_ICE)
    dens_ratio = density / RHO_ICE
    k_vals = np.empty_like(tempk, dtype=R8)
    for i in range(SNOW_K_TMPS.size):
        k_vals[...] = 0.0  # keep shape visible to mypy-free numpy broadcasting
    k_curve = np.stack(
        [
            SNOW_K_COE1[i] * dens_ratio**2 - SNOW_K_COE2[i] * dens_ratio + SNOW_K_COE3[i]
            for i in range(SNOW_K_TMPS.size)
        ],
        axis=0,
    )
    k_vals = np.full_like(tempk, SNOW_K_COE3[-1], dtype=R8)
    for i in range(SNOW_K_TMPS.size - 1):
        lo = SNOW_K_TMPS[i]
        hi = SNOW_K_TMPS[i + 1]
        mask = (tempk >= lo) & (tempk <= hi)
        if np.any(mask):
            k_vals = np.where(
                mask,
                k_curve[i]
                + (tempk - lo) * (k_curve[i + 1] - k_curve[i]) / max(hi - lo, 1.0e-12),
                k_vals,
            )
    k_vals = np.where(tempk < SNOW_K_TMPS[0], k_curve[0], k_vals)
    k_vals = np.where(tempk > SNOW_K_TMPS[-1], k_curve[-1], k_vals)
    return np.maximum(k_vals, 1.0e-3)


def thermal_params_from_texture(
    dz: np.ndarray,
    z: np.ndarray,
    zi: np.ndarray,
    texture: SoilTextureProfile,
    *,
    watsat: Optional[np.ndarray] = None,
    nlevbed: Optional[np.ndarray] = None,
) -> SoilThermalParameters:
    """Build ELM-like thermal parameters from texture and organic fraction.

    This mirrors the default non-lake branch in ``SoilStateType.F90`` for the
    thermal quantities consumed by ``SoilTemperatureMod``.
    """
    sand = _as_r8(texture.sand_pct)
    clay = _as_r8(texture.clay_pct)
    organic_frac = np.clip(_as_r8(texture.organic_frac), 0.0, 1.0)
    dz = _as_r8(dz)
    z = _as_r8(z)
    zi = _as_r8(zi)

    if sand.shape != clay.shape or sand.shape != organic_frac.shape:
        raise ValueError("texture arrays must have the same shape")

    if sand.ndim != 1:
        raise ValueError("texture arrays must be 1D over soil level")

    nlev = sand.shape[0]
    if dz.shape != (nlev,) or z.shape != (nlev,) or zi.shape != (nlev,):
        raise ValueError("dz, z, zi must be 1D and match the soil-level count")

    watsat_min, bsw_min, sucsat_min, _ = pedotransfer_cosby1984_table5(sand, clay)
    if watsat is None:
        mineral_watsat = watsat_min
    else:
        mineral_watsat = _as_r8(watsat)
        if mineral_watsat.shape != (nlev,):
            raise ValueError("watsat must match the soil-level count")

    om_watsat = np.maximum(0.93 - 0.1 * (z / ZSAPRIC), 0.83)
    om_b = np.minimum(2.7 + 9.3 * (z / ZSAPRIC), 12.0)
    om_sucsat = np.minimum(10.3 - 0.2 * (z / ZSAPRIC), 10.1)
    bd = (1.0 - mineral_watsat) * 2.7e3
    bulk_watsat = (1.0 - organic_frac) * mineral_watsat + organic_frac * om_watsat
    bulk_bsw = (1.0 - organic_frac) * (2.91 + 0.159 * clay) + organic_frac * om_b
    bulk_sucsat = (1.0 - organic_frac) * sucsat_min + organic_frac * om_sucsat

    with np.errstate(divide="ignore", invalid="ignore"):
        mineral_tkm = np.where(
            sand + clay > 0.0,
            (8.80 * sand + 2.92 * clay) / (sand + clay),
            0.0,
        )
        tkm = (1.0 - organic_frac) * mineral_tkm + OM_TKM * organic_frac

    tkmg = tkm ** (1.0 - bulk_watsat)
    tkdry = (
        ((0.135 * bd + 64.7) / (2.7e3 - 0.947 * bd)) * (1.0 - organic_frac)
        + OM_TKD * organic_frac
    )
    csol = (
        ((1.0 - organic_frac) * (2.128 * sand + 2.385 * clay) / np.maximum(sand + clay, 1.0e-12))
        + OM_CSOL * organic_frac
    ) * 1.0e6

    if nlevbed is not None:
        nlevbed_arr = _as_r8(nlevbed).astype(np.int64)
        if nlevbed_arr.ndim != 0:
            raise ValueError("nlevbed must be scalar for this helper")
        csol = csol.copy()
        csol[int(nlevbed_arr) :] = CSOL_BEDROCK

    return SoilThermalParameters(
        dz=dz,
        z=z,
        zi=zi,
        watsat=bulk_watsat,
        bsw=bulk_bsw,
        sucsat=bulk_sucsat,
        tkmg=tkmg,
        tkdry=tkdry,
        csol=csol,
        nlevbed=nlevbed,
    )


def _soil_phase_change(
    t: np.ndarray,
    hliq: np.ndarray,
    hice: np.ndarray,
    params: SoilThermalParameters,
    fact: np.ndarray,
    dhsdT: np.ndarray,
    frac_h2osfc: np.ndarray,
    dt: float,
    scalar_snow_mask: np.ndarray | None = None,
    scalar_h2osno: np.ndarray | None = None,
    scalar_snow_depth: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray]:
    """Apply the soil-layer part of ``Phasechange_beta`` for no-snow columns."""
    batch_shape = t.shape[:-1]
    nlev = t.shape[-1]
    t2 = t.reshape(-1, nlev).copy()
    t0 = t2.copy()
    hliq2 = hliq.reshape(-1, nlev).copy()
    hice2 = hice.reshape(-1, nlev).copy()
    fact2 = fact.reshape(-1, nlev)
    nbatch = t2.shape[0]

    dz = _broadcast_to_layers(params.dz, batch_shape, nlev).reshape(-1, nlev)
    watsat = _broadcast_to_layers(params.watsat, batch_shape, nlev).reshape(-1, nlev)
    bsw = _broadcast_to_layers(params.bsw, batch_shape, nlev).reshape(-1, nlev)
    sucsat = _broadcast_to_layers(params.sucsat, batch_shape, nlev).reshape(-1, nlev)
    frac_h2osfc2 = frac_h2osfc.reshape(-1)
    dhsdT2 = dhsdT.reshape(-1)
    if scalar_h2osno is None or scalar_snow_depth is None:
        scalar_h2osno2 = None
        scalar_snow_depth2 = None
        scalar_present = np.zeros(nbatch, dtype=bool)
    else:
        scalar_h2osno2 = np.asarray(scalar_h2osno, dtype=R8).reshape(-1).copy()
        scalar_snow_depth2 = np.asarray(scalar_snow_depth, dtype=R8).reshape(-1).copy()
        mask_in = (
            np.ones(nbatch, dtype=bool)
            if scalar_snow_mask is None
            else np.asarray(scalar_snow_mask, dtype=bool).reshape(-1)
        )
        scalar_present = mask_in & (scalar_h2osno2 > MIN_SNOW_LAYER_MASS) & (scalar_snow_depth2 > 0.0)
    qflx_scalar_snow_melt = np.zeros(nbatch, dtype=R8)

    imelt = np.zeros((nbatch, nlev), dtype=np.int32)
    wice0 = hice2.copy()
    wliq0 = hliq2.copy()
    wmass0 = hice2 + hliq2
    tinc = np.zeros_like(t2)
    hm = np.zeros_like(t2)

    melt_mask = (hice2 > 0.0) & (t2 > TFRZ)
    imelt[melt_mask] = 1
    tinc[melt_mask] = TFRZ - t2[melt_mask]
    t2[melt_mask] = TFRZ

    supercool = np.zeros_like(t2)
    cold = t2 < TFRZ
    if np.any(cold):
        smp = HFUS * (TFRZ - t2[cold]) / (GRAV * t2[cold]) * 1000.0
        supercool[cold] = watsat[cold] * (smp / sucsat[cold]) ** (-1.0 / bsw[cold])
        supercool[cold] = supercool[cold] * dz[cold] * 1000.0

    freeze_mask = (hliq2 > supercool) & (t2 < TFRZ)
    imelt[freeze_mask] = 2
    tinc[freeze_mask] = TFRZ - t2[freeze_mask]
    t2[freeze_mask] = TFRZ

    # Native ELM treats sub-centimeter snow (h2osno > 0, snl == 0) as a
    # scalar snow store. Excess heat in the top soil layer melts that store
    # before any remaining heat is applied to regular soil phase change.
    scalar_top_melt = scalar_present & (t2[:, 0] > TFRZ)
    if np.any(scalar_top_melt):
        imelt[scalar_top_melt, 0] = 1
        tinc[scalar_top_melt, 0] = TFRZ - t2[scalar_top_melt, 0]
        t2[scalar_top_melt, 0] = TFRZ

    active = imelt > 0
    if np.any(active):
        valid_fact = np.isfinite(fact2) & (np.abs(fact2) > 1.0e-30)
        invalid_active = active & ~valid_fact
        if np.any(invalid_active):
            imelt[invalid_active] = 0
            tinc[invalid_active] = 0.0
            t2[invalid_active] = t0[invalid_active]
            active = imelt > 0
        inv_fact = np.zeros_like(fact2)
        np.divide(1.0, fact2, out=inv_fact, where=valid_fact)

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
            if np.any(scalar_melt):
                old_snow = scalar_h2osno2.copy()
                melt_snow = np.where(
                    scalar_melt,
                    np.minimum(xm[:, 0], np.maximum(old_snow, 0.0)),
                    0.0,
                )
                scalar_h2osno2 = np.where(scalar_melt, np.maximum(old_snow - melt_snow, 0.0), scalar_h2osno2)
                scalar_snow_depth2 = np.where(
                    scalar_melt & (old_snow > 1.0e-12),
                    scalar_snow_depth2 * scalar_h2osno2 / np.maximum(old_snow, 1.0e-12),
                    scalar_snow_depth2,
                )
                qflx_scalar_snow_melt = melt_snow / max(dt, 1.0e-12)
                heatr = hm[:, 0] - HFUS * melt_snow / max(dt, 1.0e-12)
                leftover = scalar_melt & (heatr > 0.0)
                hm[:, 0] = np.where(scalar_melt, np.where(leftover, heatr, 0.0), hm[:, 0])
                xm[:, 0] = np.where(
                    scalar_melt,
                    np.where(leftover, heatr * dt / HFUS, 0.0),
                    xm[:, 0],
                )
                imelt[:, 0] = np.where(scalar_melt & ~leftover, 0, imelt[:, 0])

        # --- melt (fully vectorized over batch and layer) ---
        melt = (imelt == 1) & (np.abs(hm) > 0.0)
        if np.any(melt):
            hice2_melt = np.maximum(0.0, wice0 - xm)
            heatr_melt = hm - HFUS * (wice0 - hice2_melt) / dt
            hliq2_melt = np.maximum(0.0, wmass0 - hice2_melt)
            # surface layer correction for dhsdT
            dt_melt = fact2 * heatr_melt
            denom0 = 1.0 - (1.0 - frac_h2osfc2) * fact2[:, 0] * dhsdT2
            np.divide(
                fact2[:, 0] * heatr_melt[:, 0],
                denom0,
                out=dt_melt[:, 0],
                where=np.abs(denom0) > 1.0e-30,
            )
            nz = melt & (np.abs(heatr_melt) > 0.0)
            hice2 = np.where(melt, hice2_melt, hice2)
            hliq2 = np.where(melt, hliq2_melt, hliq2)
            t2 = np.where(nz, t2 + dt_melt, t2)

        # --- freeze (fully vectorized over batch and layer) ---
        freeze = (imelt == 2) & (np.abs(hm) > 0.0)
        if np.any(freeze):
            target_ice = np.where(
                wmass0 < supercool,
                0.0,
                np.minimum(wmass0 - supercool, wice0 - xm),
            )
            hice2_frz = np.maximum(0.0, target_ice)
            heatr_frz = hm - HFUS * (wice0 - hice2_frz) / dt
            hliq2_frz = np.maximum(0.0, wmass0 - hice2_frz)
            dt_frz = fact2 * heatr_frz
            denom0 = 1.0 - (1.0 - frac_h2osfc2) * fact2[:, 0] * dhsdT2
            np.divide(
                fact2[:, 0] * heatr_frz[:, 0],
                denom0,
                out=dt_frz[:, 0],
                where=np.abs(denom0) > 1.0e-30,
            )
            nz = freeze & (np.abs(heatr_frz) > 0.0)
            hice2 = np.where(freeze, hice2_frz, hice2)
            hliq2 = np.where(freeze, hliq2_frz, hliq2)
            t2 = np.where(nz, t2 + dt_frz, t2)

    return (
        t2.reshape(batch_shape + (nlev,)),
        hliq2.reshape(batch_shape + (nlev,)),
        hice2.reshape(batch_shape + (nlev,)),
        imelt.reshape(batch_shape + (nlev,)),
        None if scalar_h2osno2 is None else scalar_h2osno2.reshape(batch_shape),
        None if scalar_snow_depth2 is None else scalar_snow_depth2.reshape(batch_shape),
        qflx_scalar_snow_melt.reshape(batch_shape),
    )


def soil_thermal_properties(
    state: SoilTemperatureState,
    params: SoilThermalParameters,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute ELM-like layer heat capacity and conductivity for soil only."""
    t = _as_r8(state.t_soisno)
    hliq = _as_r8(state.h2osoi_liq)
    hice = _as_r8(state.h2osoi_ice)

    if t.shape != hliq.shape or t.shape != hice.shape:
        raise ValueError("t_soisno, h2osoi_liq, and h2osoi_ice must have the same shape")
    if t.ndim < 1:
        raise ValueError("soil arrays must have at least one dimension")

    batch_shape = t.shape[:-1]
    nlev = t.shape[-1]

    dz = _broadcast_to_layers(params.dz, batch_shape, nlev)
    z = _broadcast_to_layers(params.z, batch_shape, nlev)
    zi = _broadcast_to_layers(params.zi, batch_shape, nlev)
    watsat = _broadcast_to_layers(params.watsat, batch_shape, nlev)
    tkmg = _broadcast_to_layers(params.tkmg, batch_shape, nlev)
    tkdry = _broadcast_to_layers(params.tkdry, batch_shape, nlev)
    csol = _broadcast_to_layers(params.csol, batch_shape, nlev)

    satw = (hliq / DENH2O + hice / DENICE) / np.maximum(dz * watsat, 1.0e-12)
    satw = np.clip(satw, 0.0, 1.0)

    dke = np.where(
        satw > 1.0e-7,
        np.where(t >= TFRZ, np.maximum(0.0, np.log10(np.maximum(satw, 1.0e-12)) + 1.0), satw),
        0.0,
    )
    fl = np.where(
        (hliq / (DENH2O * dz) + hice / (DENICE * dz)) > 0.0,
        (hliq / (DENH2O * dz))
        / np.maximum(hliq / (DENH2O * dz) + hice / (DENICE * dz), 1.0e-12),
        0.0,
    )
    dksat = tkmg * TKWAT ** (fl * watsat) * TKICE ** ((1.0 - fl) * watsat)
    thk = np.where(satw > 1.0e-7, dke * dksat + (1.0 - dke) * tkdry, tkdry)

    tk = np.zeros_like(thk)
    if nlev > 1:
        num = thk[..., :-1] * thk[..., 1:] * (z[..., 1:] - z[..., :-1])
        den = thk[..., :-1] * (z[..., 1:] - zi[..., :-1]) + thk[..., 1:] * (zi[..., :-1] - z[..., :-1])
        tk[..., :-1] = num / np.maximum(den, 1.0e-12)
    tk[..., -1] = 0.0

    cv = csol * (1.0 - watsat) * dz + hice * CPICE + hliq * CPLIQ
    return cv, thk, tk


def soil_temperature_step(
    state: SoilTemperatureState,
    params: SoilThermalParameters,
    forcing: SoilTemperatureForcing,
    dt: float,
    *,
    snow_enabled: bool = True,
) -> SoilTemperatureResult:
    """Advance the soil-temperature column by one ELM-like implicit step.

    This is the soil-only branch of ``SoilTemperatureMod``:
    - no snow
    - no standing water
    - non-urban

    The upper-boundary forcing is the net soil-surface heat flux ``hs_soil``,
    optional standing-water heat flux ``hs_h2osfc``, and the temperature
    derivative ``dhsdT``. That is the same implicit interface ELM uses
    internally.
    """
    t = _as_r8(state.t_soisno)
    hliq = _as_r8(state.h2osoi_liq)
    hice = _as_r8(state.h2osoi_ice)

    if t.shape != hliq.shape or t.shape != hice.shape:
        raise ValueError("state arrays must have the same shape")
    if t.ndim < 1:
        raise ValueError("soil arrays must have at least one dimension")

    batch_shape = t.shape[:-1]
    nlev = t.shape[-1]
    if nlev < 2:
        raise ValueError("soil-temperature solve requires at least 2 soil layers")

    h2osno = np.array(_broadcast_to_batch(0.0 if state.h2osno is None else state.h2osno, batch_shape).reshape(-1), copy=True)
    snow_depth = np.array(_broadcast_to_batch(0.0 if state.snow_depth is None else state.snow_depth, batch_shape).reshape(-1), copy=True)
    active_snow = np.zeros_like(h2osno, dtype=bool)
    if snow_enabled:
        if state.snow_layers is not None and getattr(state.snow_layers, "snl", None) is not None:
            snl_layers = np.broadcast_to(np.asarray(state.snow_layers.snl, dtype=np.int32), batch_shape).reshape(-1)
            active_snow = snl_layers < 0
        else:
            active_snow = (h2osno > MIN_SNOW_LAYER_MASS) & (snow_depth >= 0.01)
    if snow_enabled and np.any(active_snow):
        return _snow_soil_temperature_step(state, params, forcing, dt)

    hs_soil = _broadcast_to_batch(forcing.hs_soil, batch_shape).reshape(-1)
    hs_top_snow = _broadcast_to_batch(
        forcing.hs_soil if forcing.hs_top_snow is None else forcing.hs_top_snow,
        batch_shape,
    ).reshape(-1)
    dhsdT = _broadcast_to_batch(forcing.dhsdT, batch_shape).reshape(-1)
    eflx_bot = _broadcast_to_batch(forcing.eflx_bot, batch_shape).reshape(-1)
    hs_h2osfc = _broadcast_to_batch(
        0.0 if forcing.hs_h2osfc is None else forcing.hs_h2osfc,
        batch_shape,
    ).reshape(-1)

    h2osfc = np.array(_broadcast_to_batch(0.0 if state.h2osfc is None else state.h2osfc, batch_shape).reshape(-1), copy=True)
    frac_h2osfc = np.array(_broadcast_to_batch(
        0.0 if state.frac_h2osfc is None else state.frac_h2osfc,
        batch_shape,
    ).reshape(-1), copy=True)
    if state.t_h2osfc is None:
        t_h2osfc = t[..., 0].reshape(-1).copy()
    else:
        t_h2osfc = np.array(_broadcast_to_batch(state.t_h2osfc, batch_shape).reshape(-1), copy=True)
    snow_depth = np.array(
        _broadcast_to_batch(
            0.0 if state.snow_depth is None else state.snow_depth,
            batch_shape,
        ).reshape(-1),
        copy=True,
    )
    scalar_snow_mask = (h2osno > MIN_SNOW_LAYER_MASS) & (snow_depth > 0.0)
    if state.snow_layers is not None and getattr(state.snow_layers, "snl", None) is not None:
        snl_layers = np.broadcast_to(np.asarray(state.snow_layers.snl, dtype=np.int32), batch_shape).reshape(-1)
        dz_layers = _broadcast_to_layers(
            np.asarray(state.snow_layers.dz, dtype=R8),
            batch_shape,
            MAX_SNOW_LAYERS,
        ).reshape(-1, MAX_SNOW_LAYERS)
        ice_layers = _broadcast_to_layers(
            np.asarray(state.snow_layers.h2osoi_ice, dtype=R8),
            batch_shape,
            MAX_SNOW_LAYERS,
        ).reshape(-1, MAX_SNOW_LAYERS)
        liq_layers = _broadcast_to_layers(
            np.asarray(state.snow_layers.h2osoi_liq, dtype=R8),
            batch_shape,
            MAX_SNOW_LAYERS,
        ).reshape(-1, MAX_SNOW_LAYERS)
        nsnow = np.maximum(-snl_layers, 0)
        layer_index = np.arange(MAX_SNOW_LAYERS, dtype=np.int32)[None, :]
        active_layers = layer_index < nsnow[:, None]
        layer_mass = np.sum(np.maximum(ice_layers + liq_layers, 0.0), axis=-1)
        active_depth = np.sum(np.where(active_layers, np.maximum(dz_layers, 0.0), 0.0), axis=-1)
        scalar_depth = np.maximum(dz_layers[:, 0], 0.0)
        has_layer_storage = layer_mass > 1.0e-8
        h2osno = np.where(has_layer_storage, layer_mass, h2osno)
        snow_depth = np.where(nsnow > 0, active_depth, np.where(has_layer_storage, scalar_depth, snow_depth))
        scalar_snow_mask = (nsnow == 0) & (h2osno > MIN_SNOW_LAYER_MASS) & (snow_depth > 0.0)

    has_h2osfc = (h2osfc > 1.0e-8) & (frac_h2osfc > 1.0e-8)

    dz = _broadcast_to_layers(params.dz, batch_shape, nlev)
    z = _broadcast_to_layers(params.z, batch_shape, nlev)
    zi = _broadcast_to_layers(params.zi, batch_shape, nlev)

    cv, thk, tk = soil_thermal_properties(state, params)
    tk_h2osfc = np.zeros(batch_shape, dtype=R8).reshape(-1)

    t2 = t.reshape(-1, nlev)
    cv2 = cv.reshape(-1, nlev)
    tk2 = tk.reshape(-1, nlev)
    dz2 = dz.reshape(-1, nlev)
    z2 = z.reshape(-1, nlev)
    zi2 = zi.reshape(-1, nlev)

    c_h2osfc = np.where(has_h2osfc, np.maximum(1.0e-6, CPLIQ * h2osfc / np.maximum(frac_h2osfc, 1.0e-12)), 1.0e-6)
    dz_h2osfc = np.where(has_h2osfc, np.maximum(1.0e-6, 1.0e-3 * h2osfc / np.maximum(frac_h2osfc, 1.0e-12)), 1.0e-6)
    zh2osfc = 0.5e-3 * h2osfc
    thk1 = thk.reshape(-1, nlev)[:, 0]
    z1 = z2[:, 0]
    tk_h2osfc[:] = TKWAT * thk1 * (z1 + zh2osfc) / np.maximum(TKWAT * z1 + thk1 * zh2osfc, 1.0e-12)

    fact = np.zeros_like(t2)
    fn_old = np.zeros_like(t2)
    fn_h2osfc_old = np.zeros_like(hs_soil)

    top_den = 0.5 * (z2[:, 0] + CAPR * z2[:, 1])
    fact[:, 0] = dt / cv2[:, 0] * dz2[:, 0] / np.maximum(top_den, 1.0e-12)
    fact[:, 1:] = dt / cv2[:, 1:]
    dz_mid = np.maximum(z2[:, 1:] - z2[:, :-1], 1.0e-12)
    fn_old[:, :-1] = tk2[:, :-1] * (t2[:, 1:] - t2[:, :-1]) / dz_mid
    fn_old[:, -1] = eflx_bot
    dzm_h2osfc = 0.5 * dz_h2osfc + z2[:, 0]
    fn_h2osfc_old[:] = tk_h2osfc * (t2[:, 0] - t_h2osfc) / np.maximum(dzm_h2osfc, 1.0e-12)

    nunknown = nlev + 1
    a = np.zeros((t2.shape[0], nunknown), dtype=R8)
    b = np.zeros((t2.shape[0], nunknown), dtype=R8)
    c = np.zeros((t2.shape[0], nunknown), dtype=R8)
    r = np.zeros((t2.shape[0], nunknown), dtype=R8)

    b[:, 0] = 1.0 + (1.0 - CNFAC) * (dt / c_h2osfc) * tk_h2osfc / np.maximum(dzm_h2osfc, 1.0e-12) - (dt / c_h2osfc) * dhsdT
    c[:, 0] = -(1.0 - CNFAC) * (dt / c_h2osfc) * tk_h2osfc / np.maximum(dzm_h2osfc, 1.0e-12)
    r[:, 0] = t_h2osfc + (dt / c_h2osfc) * (hs_h2osfc - dhsdT * t_h2osfc + CNFAC * fn_h2osfc_old)

    dzp = z2[:, 1] - z2[:, 0]
    b[:, 1] = 1.0 + (1.0 - CNFAC) * fact[:, 0] * tk2[:, 0] / np.maximum(dzp, 1.0e-12) - fact[:, 0] * dhsdT
    c[:, 1] = -(1.0 - CNFAC) * fact[:, 0] * tk2[:, 0] / np.maximum(dzp, 1.0e-12)
    r[:, 1] = t2[:, 0] + fact[:, 0] * ((hs_soil - dhsdT * t2[:, 0]) + CNFAC * fn_old[:, 0])
    wet = has_h2osfc
    b[wet, 1] += frac_h2osfc[wet] * (
        (1.0 - CNFAC) * fact[wet, 0] * tk_h2osfc[wet] / np.maximum(dzm_h2osfc[wet], 1.0e-12)
        + fact[wet, 0] * dhsdT[wet]
    )
    a[wet, 1] = -frac_h2osfc[wet] * (1.0 - CNFAC) * fact[wet, 0] * tk_h2osfc[wet] / np.maximum(dzm_h2osfc[wet], 1.0e-12)
    r[wet, 1] -= frac_h2osfc[wet] * fact[wet, 0] * ((hs_soil[wet] - dhsdT[wet] * t2[wet, 0]) + CNFAC * fn_h2osfc_old[wet])

    # Interior layers jj = 2..nlev-1 (j = 1..nlev-2)
    if nlev > 2:
        dzm_int = np.maximum(z2[:, 1:-1] - z2[:, :-2], 1.0e-12)
        dzp_int = np.maximum(z2[:, 2:] - z2[:, 1:-1], 1.0e-12)
        f_int = (1.0 - CNFAC) * fact[:, 1:-1]
        a[:, 2:nlev] = -f_int * tk2[:, :-2] / dzm_int
        b[:, 2:nlev] = 1.0 + f_int * (tk2[:, 1:-1] / dzp_int + tk2[:, :-2] / dzm_int)
        c[:, 2:nlev] = -f_int * tk2[:, 1:-1] / dzp_int
        r[:, 2:nlev] = t2[:, 1:-1] + CNFAC * fact[:, 1:-1] * (fn_old[:, 1:-1] - fn_old[:, :-2])
    # Bottom layer jj = nlev (j = nlev-1)
    dzm_bot = np.maximum(z2[:, -1] - z2[:, -2], 1.0e-12)
    a[:, -1] = -(1.0 - CNFAC) * fact[:, -1] * tk2[:, -2] / dzm_bot
    b[:, -1] = 1.0 + (1.0 - CNFAC) * fact[:, -1] * tk2[:, -2] / dzm_bot
    r[:, -1] = t2[:, -1] - CNFAC * fact[:, -1] * fn_old[:, -2] + fact[:, -1] * fn_old[:, -1]

    t_all_new = _batched_tridiagonal(a, b, c, r)
    t_new = t_all_new[:, 1:]
    t_h2osfc_new = np.where(has_h2osfc, t_all_new[:, 0], t_new[:, 0])

    fn_new = np.empty_like(t_new)
    fn_new[:, :-1] = tk2[:, :-1] * (t_new[:, 1:] - t_new[:, :-1]) / dz_mid
    fn_new[:, -1] = eflx_bot
    fn_h2osfc_old = np.where(has_h2osfc, fn_h2osfc_old, 0.0)
    fn_h2osfc_new = np.where(
        has_h2osfc,
        tk_h2osfc * (t_new[:, 0] - t_h2osfc_new) / np.maximum(dzm_h2osfc, 1.0e-12),
        0.0,
    )

    eflx_fgr12 = -CNFAC * fn_old[:, 0] - (1.0 - CNFAC) * fn_new[:, 0]

    t_pc, hliq_pc, hice_pc, imelt, h2osno_pc, snow_depth_pc, qflx_scalar_snow_melt = _soil_phase_change(
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
    if h2osno_pc is not None:
        h2osno = h2osno_pc.reshape(-1)
    if snow_depth_pc is not None:
        snow_depth = snow_depth_pc.reshape(-1)
    phase_hm_h2osfc = None
    phase_xm_h2osfc = None
    phase_temp1_h2osfc = None
    if snow_enabled:
        phase_hm_h2osfc = np.full(batch_shape, np.nan, dtype=R8)
        phase_xm_h2osfc = np.full(batch_shape, np.nan, dtype=R8)
        phase_temp1_h2osfc = np.full(batch_shape, np.nan, dtype=R8)
        phase_hm_h2osfc_flat = phase_hm_h2osfc.reshape(-1)
        phase_xm_h2osfc_flat = phase_xm_h2osfc.reshape(-1)
        phase_temp1_h2osfc_flat = phase_temp1_h2osfc.reshape(-1)
        frac_sno_eff_flat = (
            np.asarray(state.frac_sno, dtype=R8).reshape(-1)
            if state.frac_sno is not None
            else np.zeros(h2osfc.shape[0], dtype=R8)
        )
        has_h2osfc = (h2osfc > 1.0e-8) & (frac_h2osfc > 1.0e-8)
        if np.any(has_h2osfc):
            for bidx in np.flatnonzero(has_h2osfc):
                if t_h2osfc[bidx] <= TFRZ and h2osfc[bidx] > 0.0:
                    tinc = TFRZ - t_h2osfc[bidx]
                    c_h2osfc = CPLIQ * h2osfc[bidx] / max(frac_h2osfc[bidx], 1.0e-12)
                    hm = frac_h2osfc[bidx] * (dhsdT[bidx] * tinc - tinc * c_h2osfc / max(dt, 1.0e-12))
                    xm = hm * dt / HFUS
                    phase_hm_h2osfc_flat[bidx] = hm
                    phase_xm_h2osfc_flat[bidx] = xm
                    phase_temp1_h2osfc_flat[bidx] = h2osfc[bidx] + xm
                    z_avg = frac_sno_eff_flat[bidx] * snow_depth[bidx]
                    rho_avg = min(800.0, h2osno[bidx] / max(z_avg, 1.0e-12)) if z_avg > 0.0 else 200.0
                    if phase_temp1_h2osfc_flat[bidx] >= 0.0:
                        h2osno[bidx] -= xm
                        h2osfc[bidx] += xm
                        t_h2osfc[bidx] = TFRZ
                        snow_depth[bidx] = _safe_snow_depth_from_mass_fraction(
                            h2osno[bidx],
                            rho_avg,
                            frac_sno_eff_flat[bidx],
                        )

    return SoilTemperatureResult(
        state=SoilTemperatureState(
            t_soisno=t_pc,
            h2osoi_liq=hliq_pc,
            h2osoi_ice=hice_pc,
            t_h2osfc=t_h2osfc_new.reshape(batch_shape),
            h2osfc=h2osfc.reshape(batch_shape),
            frac_h2osfc=frac_h2osfc.reshape(batch_shape),
            h2osno=h2osno.reshape(batch_shape),
            snow_depth=snow_depth.reshape(batch_shape),
            snow_t=None if state.snow_t is None else _broadcast_to_layers(state.snow_t, batch_shape, MAX_SNOW_LAYERS),
        ),
        cv=cv,
        thk=thk,
        tk=tk,
        fact=fact.reshape(batch_shape + (nlev,)),
        fn_old=fn_old.reshape(batch_shape + (nlev,)),
        fn_new=fn_new.reshape(batch_shape + (nlev,)),
        eflx_fgr12=eflx_fgr12.reshape(batch_shape),
        tk_h2osfc=tk_h2osfc.reshape(batch_shape),
        fn_h2osfc_old=fn_h2osfc_old.reshape(batch_shape),
        fn_h2osfc_new=fn_h2osfc_new.reshape(batch_shape),
        imelt=imelt,
        qflx_snow_melt=qflx_scalar_snow_melt,
        qflx_scalar_snow_melt=qflx_scalar_snow_melt,
        qflx_snow_freeze=np.zeros(batch_shape, dtype=R8),
        phase_hm_h2osfc=phase_hm_h2osfc,
        phase_xm_h2osfc=phase_xm_h2osfc,
        phase_temp1_h2osfc=phase_temp1_h2osfc,
    )


def _snow_soil_temperature_step(
    state: SoilTemperatureState,
    params: SoilThermalParameters,
    forcing: SoilTemperatureForcing,
    dt: float,
) -> SoilTemperatureResult:
    """First-pass implicit snow+soil solve for non-urban columns.

    This keeps the existing soil and standing-water machinery out of the snow
    branch for now and focuses on the dominant missing behavior: snow
    insulation through active layers above the soil column.
    """
    t = _as_r8(state.t_soisno)
    hliq = _as_r8(state.h2osoi_liq)
    hice = _as_r8(state.h2osoi_ice)
    batch_shape = t.shape[:-1]
    nlev = t.shape[-1]
    flat_shape = (-1, nlev)
    nbatch = t.reshape(flat_shape).shape[0]

    hs_soil = _broadcast_to_batch(forcing.hs_soil, batch_shape).reshape(-1)
    hs_top_snow = _broadcast_to_batch(
        hs_soil if forcing.hs_top_snow is None else forcing.hs_top_snow,
        batch_shape,
    ).reshape(-1)
    dhsdT = _broadcast_to_batch(forcing.dhsdT, batch_shape).reshape(-1)
    eflx_bot = _broadcast_to_batch(forcing.eflx_bot, batch_shape).reshape(-1)
    hs_h2osfc = _broadcast_to_batch(0.0 if forcing.hs_h2osfc is None else forcing.hs_h2osfc, batch_shape).reshape(-1)
    h2osno = _broadcast_to_batch(0.0 if state.h2osno is None else state.h2osno, batch_shape).reshape(-1).copy()
    snow_depth = _broadcast_to_batch(0.0 if state.snow_depth is None else state.snow_depth, batch_shape).reshape(-1).copy()
    h2osfc = _broadcast_to_batch(0.0 if state.h2osfc is None else state.h2osfc, batch_shape).reshape(-1).copy()
    frac_h2osfc = _broadcast_to_batch(
        0.0 if state.frac_h2osfc is None else state.frac_h2osfc,
        batch_shape,
    ).reshape(-1).copy()
    t_h2osfc = _broadcast_to_batch(
        0.0 if state.t_h2osfc is None else state.t_h2osfc,
        batch_shape,
    ).reshape(-1).copy()
    snow_layers_state = state.snow_layers
    if snow_layers_state is not None:
        snow_snl = np.asarray(snow_layers_state.snl, dtype=np.int32).reshape(-1).copy()
        snow_dz_state = np.asarray(snow_layers_state.dz, dtype=np.float64).reshape(-1, MAX_SNOW_LAYERS).copy()
        snow_ice_state = np.asarray(snow_layers_state.h2osoi_ice, dtype=np.float64).reshape(-1, MAX_SNOW_LAYERS).copy()
        snow_liq_state = np.asarray(snow_layers_state.h2osoi_liq, dtype=np.float64).reshape(-1, MAX_SNOW_LAYERS).copy()
        if state.snow_t is None:
            snow_t_state = np.asarray(snow_layers_state.t_snow, dtype=np.float64).reshape(-1, MAX_SNOW_LAYERS).copy()
        else:
            snow_t_state = _broadcast_to_layers(state.snow_t, batch_shape, MAX_SNOW_LAYERS).reshape(-1, MAX_SNOW_LAYERS).copy()
        _nsnow_v = np.maximum(-snow_snl, 0)
        snow_depth = np.sum(
            snow_dz_state * (np.arange(MAX_SNOW_LAYERS)[None, :] < _nsnow_v[:, None]).astype(R8),
            axis=-1,
        )
        h2osno = np.sum(np.maximum(snow_ice_state + snow_liq_state, 0.0), axis=-1).reshape(-1)
    else:
        snow_snl = np.full(nbatch, -_snow_layer_count(float(snow_depth[0]) if snow_depth.size > 0 else 0.0), dtype=np.int32)
        if state.snow_t is None:
            snow_t_state = np.broadcast_to(np.minimum(t[..., :1], TFRZ), batch_shape + (MAX_SNOW_LAYERS,)).copy()
        else:
            snow_t_state = _broadcast_to_layers(state.snow_t, batch_shape, MAX_SNOW_LAYERS).copy()
        snow_layers_state = _snow_layers_from_storage_fallback(
            h2osno.reshape(batch_shape),
            snow_depth.reshape(batch_shape),
            snow_t=snow_t_state,
        )
        snow_snl = np.asarray(snow_layers_state.snl, dtype=np.int32).reshape(-1).copy()
        snow_dz_state = np.asarray(snow_layers_state.dz, dtype=np.float64).reshape(-1, MAX_SNOW_LAYERS).copy()
        snow_ice_state = np.asarray(snow_layers_state.h2osoi_ice, dtype=np.float64).reshape(-1, MAX_SNOW_LAYERS).copy()
        snow_liq_state = np.asarray(snow_layers_state.h2osoi_liq, dtype=np.float64).reshape(-1, MAX_SNOW_LAYERS).copy()
        snow_t_state = np.asarray(snow_layers_state.t_snow, dtype=np.float64).reshape(-1, MAX_SNOW_LAYERS).copy()

    dz = _broadcast_to_layers(params.dz, batch_shape, nlev).reshape(flat_shape)
    z = _broadcast_to_layers(params.z, batch_shape, nlev).reshape(flat_shape)
    cv_soil, thk_soil, tk_soil = soil_thermal_properties(state, params)
    cv_soil2 = cv_soil.reshape(flat_shape)
    thk_soil2 = thk_soil.reshape(flat_shape)
    tk_soil2 = tk_soil.reshape(flat_shape)
    t2 = t.reshape(flat_shape)

    t_new = t2.copy()
    snow_t_new = snow_t_state.reshape(-1, MAX_SNOW_LAYERS).copy()
    fn_old_soil = np.zeros_like(t2)
    fn_new_soil = np.zeros_like(t2)
    fact_soil = np.zeros_like(t2)
    tk_h2osfc_out = np.zeros(nbatch, dtype=R8)
    fn_h2osfc_old_out = np.zeros(nbatch, dtype=R8)
    fn_h2osfc_new_out = np.zeros(nbatch, dtype=R8)
    imelt = np.zeros_like(t2, dtype=np.int32)
    qflx_snow_melt = np.zeros(nbatch, dtype=R8)
    qflx_snow_freeze = np.zeros(nbatch, dtype=R8)
    phase_hm_h2osfc_flat = np.full(nbatch, np.nan, dtype=R8)
    phase_xm_h2osfc_flat = np.full(nbatch, np.nan, dtype=R8)
    phase_temp1_h2osfc_flat = np.full(nbatch, np.nan, dtype=R8)

    for bidx in range(nbatch):
        nsnow = int(max(-snow_snl[bidx], 0)) if snow_layers_state is not None else _snow_layer_count(float(snow_depth[bidx]))
        if nsnow == 0:
            continue

        frac_sno_eff = 1.0
        if state.frac_sno is not None:
            frac_sno_eff = max(float(np.asarray(state.frac_sno, dtype=R8).reshape(-1)[bidx]), 1.0e-12)
        snow_dz = np.array(snow_dz_state[bidx, :nsnow], dtype=np.float64, copy=True)
        snow_ice = np.array(snow_ice_state[bidx, :nsnow], dtype=np.float64, copy=True)
        snow_liq = np.array(snow_liq_state[bidx, :nsnow], dtype=np.float64, copy=True)
        snow_temp = np.array(snow_t_state[bidx, :nsnow], dtype=np.float64, copy=True)
        snow_cv = np.maximum((CPLIQ * snow_liq + CPICE * snow_ice) / frac_sno_eff, 1.0e-3)
        snow_density = np.clip((snow_ice + snow_liq) / np.maximum(snow_dz * frac_sno_eff, 1.0e-12), 0.0, RHO_ICE)
        snow_k = _snow_thermal_conductivity(snow_temp, snow_density)
        snow_z = -float(np.sum(snow_dz)) + np.cumsum(snow_dz) - 0.5 * snow_dz
        wice0 = snow_ice.copy()
        wmass0 = snow_ice + snow_liq

        temp = np.concatenate([snow_temp, t2[bidx]])
        cv_col = np.concatenate([snow_cv, cv_soil2[bidx]])
        z_col = np.concatenate([snow_z, z[bidx]])
        layer_k = np.concatenate([snow_k, thk_soil2[bidx]])
        ncol = temp.size

        k_int = np.zeros(ncol, dtype=R8)
        fn_old = np.zeros(ncol, dtype=R8)
        for j in range(ncol - 1):
            k_int[j] = _interface_conductivity(float(layer_k[j]), float(layer_k[j + 1]))
            fn_old[j] = k_int[j] * (temp[j + 1] - temp[j]) / max(z_col[j + 1] - z_col[j], 1.0e-12)
        fn_old[-1] = eflx_bot[bidx]

        fact = dt / np.maximum(cv_col, 1.0e-12)
        a = np.zeros((1, ncol), dtype=R8)
        bb = np.zeros((1, ncol), dtype=R8)
        c = np.zeros((1, ncol), dtype=R8)
        r = np.zeros((1, ncol), dtype=R8)

        dzp = max(z_col[1] - z_col[0], 1.0e-12)
        bb[0, 0] = 1.0 + (1.0 - CNFAC) * fact[0] * k_int[0] / dzp - fact[0] * dhsdT[bidx]
        c[0, 0] = -(1.0 - CNFAC) * fact[0] * k_int[0] / dzp
        r[0, 0] = temp[0] + fact[0] * ((hs_top_snow[bidx] - dhsdT[bidx] * temp[0]) + CNFAC * fn_old[0])

        for j in range(1, ncol - 1):
            dzm = max(z_col[j] - z_col[j - 1], 1.0e-12)
            dzp = max(z_col[j + 1] - z_col[j], 1.0e-12)
            a[0, j] = -(1.0 - CNFAC) * fact[j] * k_int[j - 1] / dzm
            bb[0, j] = 1.0 + (1.0 - CNFAC) * fact[j] * (k_int[j] / dzp + k_int[j - 1] / dzm)
            c[0, j] = -(1.0 - CNFAC) * fact[j] * k_int[j] / dzp
            r[0, j] = temp[j] + CNFAC * fact[j] * (fn_old[j] - fn_old[j - 1])

        dzm = max(z_col[-1] - z_col[-2], 1.0e-12)
        a[0, -1] = -(1.0 - CNFAC) * fact[-1] * k_int[-2] / dzm
        bb[0, -1] = 1.0 + (1.0 - CNFAC) * fact[-1] * k_int[-2] / dzm
        r[0, -1] = temp[-1] - CNFAC * fact[-1] * fn_old[-2] + fact[-1] * fn_old[-1]

        solved = _batched_tridiagonal(a, bb, c, r)[0]
        snow_t_new[bidx, :nsnow] = np.minimum(solved[:nsnow], TFRZ)
        t_new[bidx] = solved[nsnow:]

        if float(h2osfc[bidx]) > 1.0e-8 and float(frac_h2osfc[bidx]) > 1.0e-8:
            c_h2osfc = max(1.0e-6, CPLIQ * float(h2osfc[bidx]) / max(float(frac_h2osfc[bidx]), 1.0e-12))
            dz_h2osfc = max(1.0e-6, 1.0e-3 * float(h2osfc[bidx]) / max(float(frac_h2osfc[bidx]), 1.0e-12))
            zh2osfc = 0.5e-3 * float(h2osfc[bidx])
            thk1 = float(thk_soil2[bidx, 0])
            z1 = float(z[bidx, 0])
            tk_h2osfc = TKWAT * thk1 * (z1 + zh2osfc) / max(TKWAT * z1 + thk1 * zh2osfc, 1.0e-12)
            dzm_h2osfc = 0.5 * dz_h2osfc + z1
            fn_h2osfc_old = tk_h2osfc * (float(t2[bidx, 0]) - float(t_h2osfc[bidx])) / max(dzm_h2osfc, 1.0e-12)
            bb_h2osfc = (
                1.0
                + (1.0 - CNFAC) * (dt / c_h2osfc) * tk_h2osfc / max(dzm_h2osfc, 1.0e-12)
                - (dt / c_h2osfc) * float(dhsdT[bidx])
            )
            c_h2osfc_soil = -(1.0 - CNFAC) * (dt / c_h2osfc) * tk_h2osfc / max(dzm_h2osfc, 1.0e-12)
            r_h2osfc = float(t_h2osfc[bidx]) + (dt / c_h2osfc) * (
                float(hs_h2osfc[bidx]) - float(dhsdT[bidx]) * float(t_h2osfc[bidx]) + CNFAC * fn_h2osfc_old
            )
            t_h2osfc[bidx] = (r_h2osfc - c_h2osfc_soil * float(t_new[bidx, 0])) / max(bb_h2osfc, 1.0e-12)
            tk_h2osfc_out[bidx] = tk_h2osfc
            fn_h2osfc_old_out[bidx] = fn_h2osfc_old
            fn_h2osfc_new_out[bidx] = tk_h2osfc * (float(t_new[bidx, 0]) - float(t_h2osfc[bidx])) / max(dzm_h2osfc, 1.0e-12)

        if float(h2osno[bidx]) > 0.0:
            snow_xm = frac_sno_eff * snow_cv * (solved[:nsnow] - TFRZ) / HFUS
            top_tinc = TFRZ - solved[0]
            top_hm = frac_sno_eff * (dhsdT[bidx] * top_tinc - top_tinc / max(fact[0], 1.0e-30))
            snow_xm[0] = top_hm * dt / HFUS
            actual_snow_melt = 0.0
            for j in range(nsnow):
                if snow_xm[j] > 0.0:
                    melt_j = min(snow_xm[j], wice0[j])
                    actual_snow_melt += max(0.0, melt_j)
                    snow_ice[j] = max(0.0, wice0[j] - melt_j)
                elif snow_xm[j] < 0.0:
                    snow_ice[j] = min(wmass0[j], wice0[j] - snow_xm[j])
                    qflx_snow_freeze[bidx] += max(0.0, snow_ice[j] - wice0[j]) / max(dt, 1.0e-12)
                snow_liq[j] = max(0.0, wmass0[j] - snow_ice[j])
                if snow_liq[j] > 0.0 and snow_ice[j] > 0.0:
                    snow_temp[j] = TFRZ
            qflx_snow_melt[bidx] = actual_snow_melt / max(dt, 1.0e-12)
            # Native ELM Phasechange_beta conserves resolved snow-layer mass here:
            # melted snow ice becomes snow liquid, then SnowWater routes/retains it.
            h2osno[bidx] = float(np.sum(np.maximum(snow_ice + snow_liq, 0.0)))

        if float(h2osfc[bidx]) > 0.0 and float(t_h2osfc[bidx]) <= TFRZ + 1.0e-10 and nsnow > 0:
            old_snow = float(h2osno[bidx])
            old_h2osfc = float(h2osfc[bidx])
            frac_sno_local = max(frac_sno_eff, 1.0e-12)
            rho_avg = min(800.0, old_snow / max(frac_sno_local * max(float(snow_depth[bidx]), 1.0e-12), 1.0e-12)) if float(snow_depth[bidx]) > 0.0 else 200.0
            tinc = TFRZ - float(t_h2osfc[bidx])
            c_h2osfc = CPLIQ * old_h2osfc / max(float(frac_h2osfc[bidx]), 1.0e-12)
            hm = float(frac_h2osfc[bidx]) * (float(dhsdT[bidx]) * tinc - tinc * c_h2osfc / max(dt, 1.0e-12))
            xm = hm * dt / HFUS
            phase_hm_h2osfc_flat[bidx] = hm
            phase_xm_h2osfc_flat[bidx] = xm
            phase_temp1_h2osfc_flat[bidx] = old_h2osfc + xm
            if old_h2osfc + xm >= 0.0:
                h2osno[bidx] = max(0.0, old_snow - xm)
                h2osfc[bidx] = max(0.0, old_h2osfc + xm)
                snow_ice[nsnow - 1] = max(0.0, snow_ice[nsnow - 1] - xm)
                if nsnow == 1:
                    c1 = frac_sno_local * (snow_cv[nsnow - 1] - float(dhsdT[bidx]) * dt)
                else:
                    c1 = frac_sno_local * snow_cv[nsnow - 1]
                c2 = (-CPLIQ * xm - float(frac_h2osfc[bidx]) * float(dhsdT[bidx]) * dt) if float(frac_h2osfc[bidx]) != 0.0 else 0.0
                if c1 + c2 > 0.0:
                    mixed_t = (c1 * snow_temp[nsnow - 1] + c2 * float(t_h2osfc[bidx])) / (c1 + c2)
                    snow_temp[nsnow - 1] = mixed_t
                    t_h2osfc[bidx] = mixed_t
            else:
                rho_avg = (old_snow * rho_avg + old_h2osfc * DENICE) / max(old_snow + old_h2osfc, 1.0e-12)
                h2osno[bidx] = old_snow + old_h2osfc
                h2osfc[bidx] = 0.0
                snow_ice[nsnow - 1] = snow_ice[nsnow - 1] + old_h2osfc
                if nsnow == 1:
                    c1 = frac_sno_local * (snow_cv[nsnow - 1] - float(dhsdT[bidx]) * dt)
                else:
                    c1 = frac_sno_local * snow_cv[nsnow - 1]
                c2 = float(frac_h2osfc[bidx]) * (c_h2osfc - float(dhsdT[bidx]) * dt) if float(frac_h2osfc[bidx]) != 0.0 else 0.0
                if c1 + c2 > 0.0:
                    mixed_t = (c1 * snow_temp[nsnow - 1] + c2 * float(t_h2osfc[bidx])) / (c1 + c2)
                    snow_temp[nsnow - 1] = mixed_t
                    t_h2osfc[bidx] = mixed_t
            snow_depth[bidx] = _safe_snow_depth_from_mass_fraction(
                h2osno[bidx],
                rho_avg,
                frac_sno_eff,
            )
            if snow_liq[nsnow - 1] > 0.0 and snow_ice[nsnow - 1] > 0.0:
                snow_temp[nsnow - 1] = TFRZ

        snow_dz_state[bidx, :nsnow] = snow_dz
        snow_ice_state[bidx, :nsnow] = snow_ice
        snow_liq_state[bidx, :nsnow] = snow_liq
        snow_t_state[bidx, :nsnow] = snow_temp

        fn_new = np.zeros(ncol, dtype=R8)
        for j in range(ncol - 1):
            fn_new[j] = k_int[j] * (solved[j + 1] - solved[j]) / max(z_col[j + 1] - z_col[j], 1.0e-12)
        fn_new[-1] = eflx_bot[bidx]
        fact_soil[bidx] = fact[nsnow:]
        fn_old_soil[bidx] = fn_old[nsnow:]
        fn_new_soil[bidx] = fn_new[nsnow:]

    t_pc, hliq_pc, hice_pc, imelt, _, _, _ = _soil_phase_change(
        t_new.reshape(batch_shape + (nlev,)),
        hliq,
        hice,
        params,
        fact_soil.reshape(batch_shape + (nlev,)),
        dhsdT.reshape(batch_shape),
        np.zeros(batch_shape, dtype=R8),
        dt,
    )

    snow_layers_new = SnowLayerState(
        snl=snow_snl.reshape(batch_shape),
        dz=snow_dz_state.reshape(batch_shape + (MAX_SNOW_LAYERS,)),
        h2osoi_ice=snow_ice_state.reshape(batch_shape + (MAX_SNOW_LAYERS,)),
        h2osoi_liq=snow_liq_state.reshape(batch_shape + (MAX_SNOW_LAYERS,)),
        t_snow=snow_t_state.reshape(batch_shape + (MAX_SNOW_LAYERS,)),
    )

    active_snow = np.maximum(-snow_snl, 0) > 0
    has_h2osfc = (h2osfc > 1.0e-8) & (frac_h2osfc > 1.0e-8) & ~active_snow
    if np.any(has_h2osfc):
        for bidx in np.where(has_h2osfc)[0]:
            if t_h2osfc[bidx] <= TFRZ and h2osfc[bidx] > 0.0:
                tinc = TFRZ - t_h2osfc[bidx]
                c_h2osfc = CPLIQ * h2osfc[bidx] / max(frac_h2osfc[bidx], 1.0e-12)
                hm = frac_h2osfc[bidx] * (dhsdT[bidx] * tinc - tinc * c_h2osfc / max(dt, 1.0e-12))
                xm = hm * dt / HFUS
                temp1 = h2osfc[bidx] + xm
                phase_hm_h2osfc_flat[bidx] = hm
                phase_xm_h2osfc_flat[bidx] = xm
                phase_temp1_h2osfc_flat[bidx] = temp1
                frac_sno_eff_b = max(
                    float(np.asarray(state.frac_sno, dtype=R8).reshape(-1)[bidx]) if state.frac_sno is not None else 0.0,
                    0.0,
                )
                z_avg = frac_sno_eff_b * snow_depth[bidx]
                rho_avg = min(800.0, h2osno[bidx] / max(z_avg, 1.0e-12)) if z_avg > 0.0 else 200.0
                if temp1 >= 0.0:
                    # ELM uses the signed xm directly: negative xm freezes h2osfc
                    # into the snowpack; positive xm would represent the opposite
                    # sign convention in the residual energy term.
                    h2osno[bidx] -= xm
                    h2osfc[bidx] += xm
                    t_h2osfc[bidx] = TFRZ
                    snow_depth[bidx] = _safe_snow_depth_from_mass_fraction(
                        h2osno[bidx],
                        rho_avg,
                        frac_sno_eff_b,
                    )

    phase_hm_h2osfc = phase_hm_h2osfc_flat.reshape(batch_shape)
    phase_xm_h2osfc = phase_xm_h2osfc_flat.reshape(batch_shape)
    phase_temp1_h2osfc = phase_temp1_h2osfc_flat.reshape(batch_shape)

    eflx_fgr12 = -CNFAC * fn_old_soil[:, 0] - (1.0 - CNFAC) * fn_new_soil[:, 0]
    return SoilTemperatureResult(
        state=SoilTemperatureState(
            t_soisno=t_pc,
            h2osoi_liq=hliq_pc,
            h2osoi_ice=hice_pc,
            frac_sno=np.asarray(state.frac_sno, dtype=R8) if state.frac_sno is not None else None,
            t_h2osfc=t_h2osfc.reshape(batch_shape),
            h2osfc=h2osfc.reshape(batch_shape),
            frac_h2osfc=frac_h2osfc.reshape(batch_shape),
            h2osno=h2osno.reshape(batch_shape),
            snow_depth=snow_depth.reshape(batch_shape),
            snow_t=snow_t_new.reshape(batch_shape + (MAX_SNOW_LAYERS,)),
            snow_layers=snow_layers_new,
        ),
        cv=cv_soil,
        thk=thk_soil,
        tk=tk_soil,
        fact=fact_soil.reshape(batch_shape + (nlev,)),
        fn_old=fn_old_soil.reshape(batch_shape + (nlev,)),
        fn_new=fn_new_soil.reshape(batch_shape + (nlev,)),
        eflx_fgr12=eflx_fgr12.reshape(batch_shape),
        tk_h2osfc=tk_h2osfc_out.reshape(batch_shape),
        fn_h2osfc_old=fn_h2osfc_old_out.reshape(batch_shape),
        fn_h2osfc_new=fn_h2osfc_new_out.reshape(batch_shape),
        imelt=imelt,
        qflx_snow_melt=qflx_snow_melt.reshape(batch_shape),
        qflx_scalar_snow_melt=np.zeros(batch_shape, dtype=R8),
        qflx_snow_freeze=qflx_snow_freeze.reshape(batch_shape),
        phase_hm_h2osfc=phase_hm_h2osfc,
        phase_xm_h2osfc=phase_xm_h2osfc,
        phase_temp1_h2osfc=phase_temp1_h2osfc,
    )


__all__ = [
    "SoilTextureProfile",
    "SoilThermalParameters",
    "SoilTemperatureState",
    "SoilTemperatureForcing",
    "SoilTemperatureResult",
    "elm_soil_grid",
    "pedotransfer_cosby1984_table5",
    "thermal_params_from_texture",
    "soil_thermal_properties",
    "soil_temperature_step",
]
