"""Shared transpiration-wetness helpers for training and replay."""

from __future__ import annotations

import numpy as np


def inverse_frac_h2osfc(frac: np.ndarray) -> np.ndarray:
    frac = np.clip(np.asarray(frac, dtype=np.float64), 0.0, 0.999999)
    return -5.0 * np.log(np.maximum(1.0 - frac, 1.0e-12))


def compute_btran_and_root_fraction(
    liqvol: np.ndarray,
    tempk: np.ndarray,
    frac_h2osfc: np.ndarray,
    rootfr: np.ndarray,
    smpso: float | np.ndarray,
    smpsc: float | np.ndarray,
    tc_stress_c: float,
    h2osfc_mm: np.ndarray | None = None,
    eff_porosity: np.ndarray | None = None,
    smp_l: np.ndarray | None = None,
    watsat: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return btran and the normalized layer uptake weights.

    Mirrors ELM's ``calc_root_moist_stress_clm45default`` in
    ``SoilMoistStressMod.F90``.

    Parameters
    ----------
    liqvol:
        Volumetric liquid water content per soil layer (m3/m3),
        i.e. h2osoi_liq / (dz * denh2o).
    tempk:
        Soil layer temperature (K).
    frac_h2osfc:
        Fraction of land covered by surface water (used to infer h2osfc_mm
        when h2osfc_mm is not supplied directly).
    rootfr:
        Root fraction per layer (sums to 1 over active layers).
    smpso:
        Soil water potential at full stomatal opening (mm, e.g. -66 000).
        May be a scalar or an array broadcastable to ``liqvol.shape[:-1]``.
    smpsc:
        Soil water potential at full stomatal closure (mm, e.g. -255 000).
        May be a scalar or an array broadcastable to ``liqvol.shape[:-1]``.
    tc_stress_c:
        Temperature threshold below which root uptake is zero (°C, e.g. -2).
    h2osfc_mm:
        Surface water depth (mm).  If None, inferred from frac_h2osfc.
    eff_porosity:
        Effective porosity per layer (watsat − vol_ice, m3/m3).  Required
        when smp_l is not supplied.
    smp_l:
        Pre-computed matrix potential per layer (mm, negative suction).
        Required for faithful ELM behavior.  When available from the native
        training file, this is ELM's h2osoi_smp-derived value.
    watsat:
        Saturated volumetric water content per layer (porosity, m3/m3).
        Used for the ``eff_porosity/watsat`` scaling in rresis (ELM line
        ``rresis = min((eff_porosity/watsat)*(smp_node-smpsc)/(smpso-smpsc), 1)``).
        When None the scaling factor is 1.0 (matches the old behaviour).
    """
    liqvol = np.asarray(liqvol, dtype=np.float64)
    tempk = np.asarray(tempk, dtype=np.float64)
    frac_h2osfc = np.asarray(frac_h2osfc, dtype=np.float64)
    rootfr = np.maximum(np.asarray(rootfr, dtype=np.float64), 0.0)
    smpso_arr = np.asarray(smpso, dtype=np.float64)
    smpsc_arr = np.asarray(smpsc, dtype=np.float64)

    active = (liqvol > 0.0) & (tempk > (273.15 + float(tc_stress_c)))

    if smp_l is None:
        raise ValueError("compute_btran_and_root_fraction requires smp_l; saturation/theta is not a valid ELM rresis fallback.")
    # Use pre-computed Brooks-Corey matrix potential (mm) directly, matching
    # ELM's h2osoi_smp-derived smp_node path.
    smp_node = np.asarray(smp_l, dtype=np.float64)

    smpsc_expanded = smpsc_arr[..., None] if smpsc_arr.ndim > 0 else smpsc_arr
    smp_node = np.maximum(smpsc_expanded, smp_node)
    denom = smpso_arr - smpsc_arr
    denom = np.where(np.abs(denom) < 1.0e-12, 1.0e-12, denom)

    if watsat is not None and eff_porosity is not None:
        ep_over_ws = np.asarray(eff_porosity, dtype=np.float64) / np.maximum(
            np.asarray(watsat, dtype=np.float64), 1.0e-12
        )
    else:
        ep_over_ws = 1.0

    denom_expanded = denom[..., None] if np.ndim(denom) > 0 else denom
    rresis = np.minimum(ep_over_ws * (smp_node - smpsc_expanded) / denom_expanded, 1.0)

    if h2osfc_mm is None:
        h2osfc_mm = inverse_frac_h2osfc(frac_h2osfc)
    else:
        h2osfc_mm = np.asarray(h2osfc_mm, dtype=np.float64)
    floodf = np.where(h2osfc_mm <= 1000.0, np.maximum(0.0, 1.0 - 0.001 * np.maximum(h2osfc_mm, 0.0)), 0.0)
    rresis = np.where(active, rresis * floodf[..., None], 0.0)

    rootfr_unf = np.where(tempk > (273.15 + float(tc_stress_c)), rootfr, 0.0)
    root_sum = np.sum(rootfr_unf, axis=-1, keepdims=True)
    rootfr_unf = np.divide(rootfr_unf, root_sum, out=np.zeros_like(rootfr_unf), where=root_sum > 1.0e-12)
    root_contrib = rootfr_unf * rresis
    btran = np.sum(root_contrib, axis=-1)
    rootr_norm = np.zeros_like(root_contrib)
    np.divide(root_contrib, btran[..., None], out=rootr_norm, where=btran[..., None] > 1.0e-12)
    return btran, rootr_norm
