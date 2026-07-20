"""Vectorized Python port of the core ELM soil hydrology step.

This module follows the natural-vegetation, non-urban, non-VIC branch of the
ELM hydrology path as closely as practical in NumPy:

- ``SoilHydrologyMod::SurfaceRunoff``
- ``SoilHydrologyMod::Infiltration``
- ``SoilWaterMovementMod::soilwater_zengdecker2009``

It is intended for large offline ensembles, where the leading dimensions are
typically ``(ncell, nens)`` and the last dimension is soil layer.

Scope
-----
Implemented:
- surface runoff for the standard fsat formulation
- infiltration excess, surface-water storage, and drainage back to soil
- batched Zeng-Decker soil-water solve with a tridiagonal Richards-equation step
- matric potential and hydraulic conductivity diagnostics

Deliberately omitted for the first version:
- VIC runoff / infiltration option
- urban landunits
- HUM_HOL / polygon / marsh logic
- perched water table and lateral aquifer exchange
- explicit drainage / water-table update from ``SoilHydrologyMod::Drainage``

Those omitted branches are where ELM becomes highly configuration-specific.
The core batched solve here is still useful, and the interface is structured so
the missing branches can be added later.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


R8 = np.float64

DENH2O = 1000.0
DENICE = 917.0
E_ICE = 5.0
TFRZ = 273.15
HFUS = 3.337e5
GRAV = 9.80616


def _as_r8(x: np.ndarray | float) -> np.ndarray:
    return np.asarray(x, dtype=R8)


def _flatten_batch(arr: np.ndarray) -> tuple[np.ndarray, tuple[int, ...]]:
    """Flatten all but the last dimension."""
    arr = _as_r8(arr)
    if arr.ndim < 1:
        raise ValueError("expected at least 1 dimension")
    batch_shape = arr.shape[:-1]
    return arr.reshape((-1, arr.shape[-1])), batch_shape


def _flatten_vec(arr: np.ndarray, batch_shape: tuple[int, ...]) -> np.ndarray:
    arr = _as_r8(arr)
    nbatch = int(np.prod(batch_shape)) if batch_shape else 1
    if arr.size == nbatch:
        return arr.reshape(-1)
    return np.broadcast_to(arr, batch_shape).reshape(-1)


def _broadcast_optional_vec(
    arr: np.ndarray | float | None,
    batch_shape: tuple[int, ...],
    *,
    default: float = 0.0,
) -> np.ndarray:
    if arr is None:
        return np.full(batch_shape, default, dtype=R8).reshape(-1)
    return np.broadcast_to(_as_r8(arr), batch_shape).reshape(-1)


def _restore(arr: np.ndarray, batch_shape: tuple[int, ...]) -> np.ndarray:
    return arr.reshape(batch_shape + (arr.shape[-1],))


def _restore_vec(arr: np.ndarray, batch_shape: tuple[int, ...]) -> np.ndarray:
    return arr.reshape(batch_shape)


def _repair_liquid_water_floor(
    h2osoi_liq: np.ndarray,
    h2osfc: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the same basic watmin repair used later in ELM drainage.

    In the full model this bookkeeping occurs in ``Drainage``. For the
    standalone no-drainage wrapper we still need it, otherwise the top layer can
    go negative in dry periods and the plotting driver becomes hard to interpret.
    """
    watmin = 1.0e-8
    hliq = h2osoi_liq.copy()
    hsurf = h2osfc.copy()
    nbatch, nlev = hliq.shape

    for j in range(nlev - 1):
        xs = np.where(hliq[:, j] < watmin, watmin - hliq[:, j], 0.0)
        hliq[:, j] += xs
        hliq[:, j + 1] -= xs

    xs1 = np.maximum(0.0, -hliq[:, 0])
    hliq[:, 0] += xs1
    hsurf += xs1

    xs = np.where(hliq[:, -1] < watmin, watmin - hliq[:, -1], 0.0)
    hliq[:, -1] += xs

    return hliq, hsurf


def _redistribute_saturation_excess(
    h2osoi_liq: np.ndarray,
    h2osoi_ice: np.ndarray,
    eff_porosity: np.ndarray,
    dz: np.ndarray,
    h2osfc: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the saturation-excess cleanup from the ELM drainage path.

    Water above effective porosity is pushed upward layer by layer; any excess
    at the surface is moved into ponded surface water.
    """
    hliq = h2osoi_liq.copy()
    hice = h2osoi_ice.copy()
    effp = eff_porosity.copy()
    dzmm = dz * 1.0e3
    hsurf = h2osfc.copy()
    nbatch, nlev = hliq.shape

    for j in range(nlev - 1, 0, -1):
        max_liq = effp[:, j] * dzmm[:, j]
        excess = np.maximum(hliq[:, j] - max_liq, 0.0)
        hliq[:, j] = np.minimum(max_liq, hliq[:, j])
        hliq[:, j - 1] += excess

    top_capacity = np.maximum(0.0, effp[:, 0] * dzmm[:, 0])
    top_excess = np.maximum(hliq[:, 0] - top_capacity, 0.0)
    hliq[:, 0] -= top_excess
    hsurf += top_excess

    return hliq, hsurf


def _batched_tridiagonal(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    r: np.ndarray,
) -> np.ndarray:
    """Solve many tridiagonal systems with vectorization over batch.

    Arrays are shaped ``(nbatch, n)``.
    """
    nbatch, n = b.shape
    cp = np.zeros_like(c)
    dp = np.zeros_like(r)
    x = np.zeros_like(r)
    eps = 1.0e-12

    def _safe_denom(denom: np.ndarray) -> np.ndarray:
        sign = np.where(denom >= 0.0, 1.0, -1.0)
        return np.where(np.abs(denom) < eps, sign * eps, denom)

    b0 = _safe_denom(b[:, 0])
    cp[:, 0] = c[:, 0] / b0
    dp[:, 0] = r[:, 0] / b0

    for j in range(1, n):
        denom = _safe_denom(b[:, j] - a[:, j] * cp[:, j - 1])
        # avoid branch inside loop — always compute cp, zero it at last step after
        cp[:, j] = c[:, j] / denom
        dp[:, j] = (r[:, j] - a[:, j] * dp[:, j - 1]) / denom
    cp[:, -1] = 0.0

    x[:, -1] = dp[:, -1]
    for j in range(n - 2, -1, -1):
        x[:, j] = dp[:, j] - cp[:, j] * x[:, j + 1]

    return x


def _clapp_hornberger_smp(
    vliq: np.ndarray,
    watsat: np.ndarray,
    sucsat: np.ndarray,
    bsw: np.ndarray,
    smpmin: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return soil matric potential and d(smp)/d(vliq)."""
    s_node = np.clip(vliq / watsat, 0.01, 1.0)
    smp = -sucsat * s_node ** (-bsw)
    # `smpmin` may be stored as either a signed floor (-1e8 mm) or as a
    # positive magnitude (1e8 mm). In both cases the physical clipping bound is
    # the large negative floor.
    smp_floor = -np.abs(smpmin[:, None])
    smp = np.maximum(smp, smp_floor)
    dsmpdw = -bsw * smp / np.maximum(vliq, 1.0e-12)
    return smp, dsmpdw


def _clapp_hornberger_hk(
    vliq: np.ndarray,
    watsat: np.ndarray,
    hksat: np.ndarray,
    bsw: np.ndarray,
    icefrac: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return unsaturated hydraulic conductivity at interfaces."""
    s1 = 0.5 * (vliq[:, :-1] + vliq[:, 1:]) / (0.5 * (watsat[:, :-1] + watsat[:, 1:]))
    s1 = np.clip(s1, 0.0, 1.0)
    s2 = hksat[:, :-1] * s1 ** (2.0 * bsw[:, :-1] + 2.0)
    imped = 10.0 ** (-E_ICE * 0.5 * (icefrac[:, :-1] + icefrac[:, 1:]))
    hk = imped * s1 * s2
    dhkdw = (
        imped
        * (2.0 * bsw[:, :-1] + 3.0)
        * s2
        / np.maximum(watsat[:, :-1] + watsat[:, 1:], 1.0e-12)
    )
    return hk, dhkdw


@dataclass
class HydrologyState:
    """Column state for the soil hydrology step.

    All arrays may have arbitrary leading batch dimensions. Soil-layer arrays
    must have the layer dimension last.
    """

    h2osoi_liq: np.ndarray
    h2osoi_ice: np.ndarray
    h2osfc: np.ndarray
    zwt: np.ndarray
    frac_h2osfc: np.ndarray
    frac_sno: np.ndarray
    t_soisno: np.ndarray
    nlevbed: Optional[np.ndarray] = None


@dataclass
class HydrologyParameters:
    """Static or slowly varying soil-column parameters."""

    dz: np.ndarray
    z: np.ndarray
    zi: np.ndarray
    watsat: np.ndarray
    sucsat: np.ndarray
    bsw: np.ndarray
    hksat: np.ndarray
    smpmin: np.ndarray
    wtfact: np.ndarray
    fover: np.ndarray
    fdrain: np.ndarray
    h2osfc_thresh: np.ndarray
    pc: np.ndarray
    rootfr: np.ndarray
    topo_slope: np.ndarray | None = None


@dataclass
class SoilTextureProfile:
    """Layered soil descriptors used to derive ELM hydraulic parameters.

    Percentages use ELM conventions:
    - ``sand_pct`` / ``clay_pct`` in percent [0, 100]
    - ``organic_frac`` as [0, 1]
    """

    sand_pct: np.ndarray
    clay_pct: np.ndarray
    organic_frac: np.ndarray


def elm_soil_grid(
    nlevsoi: int = 10,
    scalez: float = 0.025,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the standard ELM exponential soil layer grid.

    Mirrors ``elmgridMod::init_elm_vertgrid`` (scalez = 0.025 m).
    Only the first ``nlevsoi`` layers are returned; ELM uses nlevsoi=10
    for the hydrologically active column, with deeper layers treated as
    bedrock.

    Returns
    -------
    dz : array (nlevsoi,)  layer thickness [m]
    z  : array (nlevsoi,)  node centre depth [m]
    zi : array (nlevsoi,)  lower interface depth [m]  (zi[0] is bottom of layer 0)
    """
    nlevgrnd = max(nlevsoi, 15)
    j = np.arange(1, nlevgrnd + 1, dtype=R8)
    zsoi = scalez * (np.exp(0.5 * (j - 0.5)) - 1.0)

    dzsoi = np.empty(nlevgrnd, dtype=R8)
    dzsoi[0] = 0.5 * (zsoi[0] + zsoi[1])
    dzsoi[1:-1] = 0.5 * (zsoi[2:] - zsoi[:-2])
    dzsoi[-1] = zsoi[-1] - zsoi[-2]

    zisoi = np.empty(nlevgrnd + 1, dtype=R8)
    zisoi[0] = 0.0
    zisoi[1:-1] = 0.5 * (zsoi[:-1] + zsoi[1:])
    zisoi[-1] = zsoi[-1] + 0.5 * dzsoi[-1]

    return dzsoi[:nlevsoi], zsoi[:nlevsoi], zisoi[1 : nlevsoi + 1]


def pedotransfer_cosby1984_table5(
    sand_pct: np.ndarray,
    clay_pct: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """ELM default mineral-soil pedotransfer function.

    Mirrors ``FuncPedotransferMod::pedotransf_cosby1984_table5``.
    Returns ``(watsat, bsw, sucsat, hksat)`` with units matching ELM:
    - ``watsat`` in m3/m3
    - ``bsw`` dimensionless
    - ``sucsat`` in mm
    - ``hksat`` in mm/s
    """
    sand = _as_r8(sand_pct)
    clay = _as_r8(clay_pct)
    watsat = 0.489 - 0.00126 * sand
    bsw = 2.91 + 0.159 * clay
    sucsat = 10.0 * (10.0 ** (1.88 - 0.0131 * sand))
    hksat = 0.0070556 * (10.0 ** (-0.884 + 0.0153 * sand))
    return watsat, bsw, sucsat, hksat


def hydraulic_params_from_texture(
    dz: np.ndarray,
    z: np.ndarray,
    zi: np.ndarray,
    texture: SoilTextureProfile,
    rootfr: np.ndarray,
    topo_slope: np.ndarray | None = None,
) -> HydrologyParameters:
    """Build ELM-like hydrology parameters from texture and organic fraction.

    This follows the default non-lake branch in ``SoilStateType.F90`` closely
    for the hydraulic quantities used by this Python hydrology kernel.
    """
    sand = _as_r8(texture.sand_pct)
    clay = _as_r8(texture.clay_pct)
    organic_frac = np.clip(_as_r8(texture.organic_frac), 0.0, 1.0)
    dz = _as_r8(dz)
    z = _as_r8(z)
    zi = _as_r8(zi)
    rootfr = _as_r8(rootfr)

    if sand.shape != clay.shape or sand.shape != organic_frac.shape:
        raise ValueError("texture arrays must have the same shape")

    watsat_min, bsw_min, sucsat_min, hksat_min = pedotransfer_cosby1984_table5(sand, clay)

    zsapric = 0.5
    om_watsat = np.maximum(0.93 - 0.1 * (z / zsapric), 0.83)
    om_b = np.minimum(2.7 + 9.3 * (z / zsapric), 12.0)
    om_sucsat = np.minimum(10.3 - 0.2 * (z / zsapric), 10.1)
    om_hksat = np.maximum(0.28 - 0.2799 * (z / zsapric), 1.0e-4)

    pcalpha = 0.5
    pcbeta = 0.139
    perc_norm = (1.0 - pcalpha) ** (-pcbeta)
    perc_frac = np.zeros_like(organic_frac)
    high_organic = organic_frac > pcalpha
    perc_frac[high_organic] = perc_norm * (organic_frac[high_organic] - pcalpha) ** pcbeta
    uncon_frac = (1.0 - organic_frac) + (1.0 - perc_frac) * organic_frac

    uncon_hksat = np.where(
        organic_frac < 1.0,
        uncon_frac
        / (
            (1.0 - organic_frac) / np.maximum(hksat_min, 1.0e-20)
            + ((1.0 - perc_frac) * organic_frac) / om_hksat
        ),
        0.0,
    )

    watsat = (1.0 - organic_frac) * watsat_min + organic_frac * om_watsat
    bsw = (1.0 - organic_frac) * (2.91 + 0.159 * clay) + organic_frac * om_b
    sucsat = (1.0 - organic_frac) * sucsat_min + organic_frac * om_sucsat
    hksat = uncon_frac * uncon_hksat + (perc_frac * organic_frac) * om_hksat

    return HydrologyParameters(
        dz=dz,
        z=z,
        zi=zi,
        watsat=watsat,
        sucsat=sucsat,
        bsw=bsw,
        hksat=hksat,
        smpmin=np.array([[-1.0e8]], dtype=R8),
        wtfact=np.array([[0.35]], dtype=R8),
        fover=np.array([[0.5]], dtype=R8),
        fdrain=np.array([[2.5]], dtype=R8),
        h2osfc_thresh=np.array([[1.0]], dtype=R8),
        pc=np.array([[0.4]], dtype=R8),
        rootfr=rootfr,
        topo_slope=None if topo_slope is None else _as_r8(topo_slope),
    )


@dataclass
class HydrologyForcing:
    """Surface and plant-water forcing for one hydrology step."""

    qflx_top_soil: np.ndarray
    qflx_evap_grnd: np.ndarray
    qflx_ev_soil: np.ndarray
    qflx_ev_h2osfc: np.ndarray
    qflx_prec_grnd: Optional[np.ndarray] = None
    qflx_rootsoi: Optional[np.ndarray] = None
    qflx_tran_veg: Optional[np.ndarray] = None


@dataclass
class HydrologyResult:
    state: HydrologyState
    wa: Optional[np.ndarray]
    qflx_surf: np.ndarray
    qflx_infl: np.ndarray
    qflx_h2osfc_surf: np.ndarray
    qflx_h2osfc_drain: np.ndarray
    qflx_infl_excess: np.ndarray
    qcharge: np.ndarray
    qflx_deep_drain: np.ndarray
    h2osoi_vol: np.ndarray
    eff_porosity: np.ndarray
    smp_l: np.ndarray
    hk_l: np.ndarray
    qflx_drain: Optional[np.ndarray] = None
    qflx_rsub_sat: Optional[np.ndarray] = None
    qin_layer1: Optional[np.ndarray] = None
    qout_layer1: Optional[np.ndarray] = None
    qroot_layer1: Optional[np.ndarray] = None
    dW1dt_layer1: Optional[np.ndarray] = None
    zwt_after_qcharge: Optional[np.ndarray] = None
    zwt_after_drainage: Optional[np.ndarray] = None


def compute_root_sink(
    forcing: HydrologyForcing,
    params: HydrologyParameters,
    batch_shape: tuple[int, ...],
) -> np.ndarray:
    """Return per-layer root sink [mm/s]."""
    if forcing.qflx_rootsoi is not None:
        qroot, _ = _flatten_batch(forcing.qflx_rootsoi)
        return qroot

    if forcing.qflx_tran_veg is None:
        raise ValueError("provide either qflx_rootsoi or qflx_tran_veg")

    tran = _flatten_vec(forcing.qflx_tran_veg, batch_shape)
    rootfr = np.broadcast_to(_as_r8(params.rootfr), batch_shape + params.rootfr.shape[-1:])
    rootfr, _ = _flatten_batch(rootfr)
    root_sum = np.maximum(rootfr.sum(axis=1, keepdims=True), 1.0e-12)
    return rootfr / root_sum * tran[:, None]


def surface_runoff(
    state: HydrologyState,
    params: HydrologyParameters,
    forcing: HydrologyForcing,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Port of the standard ELM surface-runoff branch.

    Returns ``(qflx_surf, fsat, icefrac)``.
    """
    h2osoi_liq, batch_shape = _flatten_batch(state.h2osoi_liq)
    h2osoi_ice, _ = _flatten_batch(state.h2osoi_ice)
    dz = np.broadcast_to(_as_r8(params.dz), batch_shape + params.dz.shape[-1:])
    dz, _ = _flatten_batch(dz)
    watsat = np.broadcast_to(_as_r8(params.watsat), batch_shape + params.watsat.shape[-1:])
    watsat, _ = _flatten_batch(watsat)
    zwt = _flatten_vec(state.zwt, batch_shape)
    wtfact = _flatten_vec(params.wtfact, batch_shape)
    fover = _flatten_vec(params.fover, batch_shape)
    qflx_top_soil = _flatten_vec(forcing.qflx_top_soil, batch_shape)

    vol_ice = np.minimum(watsat, h2osoi_ice / (dz * DENICE))
    icefrac = np.minimum(1.0, vol_ice / np.maximum(watsat, 1.0e-12))
    fsat = wtfact * np.exp(-0.5 * fover * zwt)
    qflx_surf = fsat * qflx_top_soil

    return (
        _restore_vec(qflx_surf, batch_shape),
        _restore_vec(fsat, batch_shape),
        _restore(icefrac, batch_shape),
    )


def infiltration(
    state: HydrologyState,
    params: HydrologyParameters,
    forcing: HydrologyForcing,
    qflx_surf: np.ndarray,
    fsat: np.ndarray,
    dt: float,
) -> tuple[HydrologyState, dict[str, np.ndarray]]:
    """Simplified natural-vegetation infiltration port from ELM."""
    h2osoi_liq, batch_shape = _flatten_batch(state.h2osoi_liq)
    h2osoi_ice, _ = _flatten_batch(state.h2osoi_ice)
    t_soisno, _ = _flatten_batch(state.t_soisno)
    h2osfc = _flatten_vec(state.h2osfc, batch_shape)
    frac_h2osfc = _flatten_vec(state.frac_h2osfc, batch_shape)
    frac_sno = _flatten_vec(state.frac_sno, batch_shape)
    qflx_top_soil = _flatten_vec(forcing.qflx_top_soil, batch_shape)
    qflx_evap_grnd = _flatten_vec(forcing.qflx_evap_grnd, batch_shape)
    qflx_ev_soil = _flatten_vec(forcing.qflx_ev_soil, batch_shape)
    qflx_ev_h2osfc = _flatten_vec(forcing.qflx_ev_h2osfc, batch_shape)
    qflx_surf_f = _flatten_vec(qflx_surf, batch_shape)
    fsat_f = _flatten_vec(fsat, batch_shape)

    dz = np.broadcast_to(_as_r8(params.dz), batch_shape + params.dz.shape[-1:])
    dz, _ = _flatten_batch(dz)
    watsat = np.broadcast_to(_as_r8(params.watsat), batch_shape + params.watsat.shape[-1:])
    watsat, _ = _flatten_batch(watsat)
    hksat = np.broadcast_to(_as_r8(params.hksat), batch_shape + params.hksat.shape[-1:])
    hksat, _ = _flatten_batch(hksat)
    h2osfc_thresh = _flatten_vec(params.h2osfc_thresh, batch_shape)
    pc = _flatten_vec(params.pc, batch_shape)

    vol_ice = np.minimum(watsat, h2osoi_ice / (dz * DENICE))
    eff_porosity = np.maximum(0.01, watsat - vol_ice)
    icefrac = np.minimum(1.0, vol_ice / np.maximum(watsat, 1.0e-12))

    has_active_snow = frac_sno > 1.0e-12
    fsno = np.where(has_active_snow, frac_sno, 0.0)
    qflx_evap = np.where(has_active_snow, qflx_ev_soil, qflx_evap_grnd)

    qflx_in_soil = (1.0 - frac_h2osfc) * (qflx_top_soil - qflx_surf_f)
    qflx_in_h2osfc = frac_h2osfc * (qflx_top_soil - qflx_surf_f)

    soil_evap_area = np.clip(1.0 - fsno - frac_h2osfc, 0.0, 1.0)
    qflx_soil_evap = soil_evap_area * qflx_evap
    # The emulator predicts potential soil evaporation independently of the
    # current prognostic water state.  Keep the native negative top-boundary
    # flux form, but do not let positive evaporation pull layer 1 below the
    # small residual pore-water reserve in a single hydrology step.
    layer1_reserve = 0.01 * eff_porosity[:, 0] * dz[:, 0] * DENH2O
    layer1_available = np.maximum(h2osoi_liq[:, 0] - layer1_reserve, 0.0) / dt
    evap_supply = np.maximum(qflx_in_soil, 0.0) + layer1_available
    qflx_soil_evap = np.where(qflx_soil_evap > 0.0, np.minimum(qflx_soil_evap, evap_supply), qflx_soil_evap)
    qflx_in_soil = qflx_in_soil - qflx_soil_evap
    qflx_in_h2osfc = qflx_in_h2osfc - frac_h2osfc * qflx_ev_h2osfc

    qinmax = (1.0 - fsat_f) * np.min(10.0 ** (-E_ICE * icefrac[:, :3]) * hksat[:, :3], axis=1)
    qflx_infl_excess = np.maximum(0.0, qflx_in_soil - (1.0 - frac_h2osfc) * qinmax)
    qflx_infl = qflx_in_soil - qflx_infl_excess
    qflx_in_h2osfc = qflx_in_h2osfc + qflx_infl_excess

    active_storage = np.maximum(h2osfc - h2osfc_thresh, 0.0)
    mu = 1.0
    topo_slope_deg = _broadcast_optional_vec(params.topo_slope, batch_shape, default=2.0)
    frac_infclust = np.where(frac_h2osfc <= pc, 0.0, (frac_h2osfc - pc) ** mu)
    k_wet = np.sin(np.deg2rad(topo_slope_deg))
    qflx_h2osfc_surf = np.minimum(k_wet * frac_infclust * active_storage, active_storage / dt)
    qflx_h2osfc_surf = np.where(qflx_h2osfc_surf < 1.0e-8, 0.0, qflx_h2osfc_surf)
    qflx_in_h2osfc = qflx_in_h2osfc - qflx_h2osfc_surf

    # Do not let potential surface-water evaporation/runoff borrow from soil.
    # If the predicted h2osfc sink exceeds available storage plus incoming water,
    # cap the net h2osfc loss at zero storage rather than charging the deficit to
    # qflx_infl/layer 1.
    qflx_in_h2osfc = np.maximum(qflx_in_h2osfc, -h2osfc / dt)
    h2osfc = h2osfc + qflx_in_h2osfc * dt

    qflx_h2osfc_drain = np.zeros_like(h2osfc)
    negative_storage = h2osfc < 0.0
    if np.any(negative_storage):
        h2osfc[negative_storage] = 0.0

    nonnegative_storage = ~negative_storage
    if np.any(nonnegative_storage):
        qflx_h2osfc_drain[nonnegative_storage] = np.minimum(
            frac_h2osfc[nonnegative_storage] * qinmax[nonnegative_storage],
            np.maximum(h2osfc[nonnegative_storage], 0.0) / dt,
        )
        h2osfc[nonnegative_storage] = np.maximum(
            0.0,
            h2osfc[nonnegative_storage] - qflx_h2osfc_drain[nonnegative_storage] * dt,
        )
        qflx_infl[nonnegative_storage] = qflx_infl[nonnegative_storage] + qflx_h2osfc_drain[nonnegative_storage]

    new_state = HydrologyState(
        h2osoi_liq=_restore(h2osoi_liq, batch_shape),
        h2osoi_ice=_restore(h2osoi_ice, batch_shape),
        h2osfc=_restore_vec(h2osfc, batch_shape),
        zwt=state.zwt.copy(),
        frac_h2osfc=state.frac_h2osfc.copy(),
        frac_sno=state.frac_sno.copy(),
        t_soisno=state.t_soisno.copy(),
        nlevbed=state.nlevbed.copy() if state.nlevbed is not None else None,
    )

    fluxes = {
        "qflx_infl": _restore_vec(qflx_infl, batch_shape),
        "qflx_h2osfc_surf": _restore_vec(qflx_h2osfc_surf, batch_shape),
        "qflx_h2osfc_drain": _restore_vec(qflx_h2osfc_drain, batch_shape),
        "qflx_infl_excess": _restore_vec(qflx_infl_excess, batch_shape),
        "eff_porosity": _restore(eff_porosity, batch_shape),
        "icefrac": _restore(icefrac, batch_shape),
    }
    return new_state, fluxes


def water_table(
    state: HydrologyState,
    params: HydrologyParameters,
    qcharge: np.ndarray,
    wa: np.ndarray,
    dt: float,
) -> tuple[HydrologyState, np.ndarray]:
    """Simplified port of the default ELM water-table update.

    This keeps the standard qcharge-driven groundwater movement and excludes the
    optional HUM_HOL / perched-water-table branches.
    """
    h2osoi_liq, batch_shape = _flatten_batch(state.h2osoi_liq)
    h2osoi_ice, _ = _flatten_batch(state.h2osoi_ice)
    zwt = _flatten_vec(state.zwt, batch_shape).copy()
    wa_f = _flatten_vec(wa, batch_shape).copy()
    qcharge_f = _flatten_vec(qcharge, batch_shape)

    z = np.broadcast_to(_as_r8(params.z), batch_shape + params.z.shape[-1:])
    zi = np.broadcast_to(_as_r8(params.zi), batch_shape + params.zi.shape[-1:])
    dz = np.broadcast_to(_as_r8(params.dz), batch_shape + params.dz.shape[-1:])
    watsat = np.broadcast_to(_as_r8(params.watsat), batch_shape + params.watsat.shape[-1:])
    sucsat = np.broadcast_to(_as_r8(params.sucsat), batch_shape + params.sucsat.shape[-1:])
    bsw = np.broadcast_to(_as_r8(params.bsw), batch_shape + params.bsw.shape[-1:])
    z, _ = _flatten_batch(z)
    zi, _ = _flatten_batch(zi)
    dz, _ = _flatten_batch(dz)
    watsat, _ = _flatten_batch(watsat)
    sucsat, _ = _flatten_batch(sucsat)
    bsw, _ = _flatten_batch(bsw)

    nbatch, nlev = h2osoi_liq.shape
    # jwt = last layer whose bottom interface is above zwt (vectorized)
    # Native ELM keeps the "water table below the soil column" case distinct
    # from "water table in the bottom soil layer".  `jwt` is the number of soil
    # layer interfaces above the water table, so it ranges from 0 (in/top of
    # layer 1) through nlev (below the active soil column).
    jwt = np.sum(zwt[:, None] > zi, axis=1).astype(np.int64)

    rous = watsat[:, -1] * (1.0 - (1.0 + 1.0e3 * zwt / sucsat[:, -1]) ** (-1.0 / bsw[:, -1]))
    rous = np.maximum(rous, 0.02)

    below_column = jwt >= nlev
    if np.any(below_column):
        qcharge_amt = qcharge_f[below_column] * dt
        wa_f[below_column] = wa_f[below_column] + qcharge_amt
        zwt[below_column] = zwt[below_column] - qcharge_amt / 1000.0 / rous[below_column]

    within_column = ~below_column
    rows = np.where(within_column)[0]
    for row in rows:
        qtot = qcharge_f[row] * dt
        if qtot > 0.0:
            for j in range(min(jwt[row], nlev - 1), -1, -1):
                sy = watsat[row, j] * (1.0 - (1.0 + 1.0e3 * zwt[row] / sucsat[row, j]) ** (-1.0 / bsw[row, j]))
                sy = max(sy, 0.02)
                layer = min(qtot, sy * (zwt[row] - (zi[row, j - 1] if j > 0 else 0.0)) * 1.0e3)
                layer = max(layer, 0.0)
                zwt[row] -= layer / sy / 1000.0
                qtot -= layer
                if qtot <= 0.0:
                    break
        else:
            for j in range(jwt[row] + 1, nlev):
                sy = watsat[row, j] * (1.0 - (1.0 + 1.0e3 * zwt[row] / sucsat[row, j]) ** (-1.0 / bsw[row, j]))
                sy = max(sy, 0.02)
                layer = max(qtot, -(sy * (zi[row, j] - zwt[row]) * 1.0e3))
                layer = min(layer, 0.0)
                qtot -= layer
                if qtot >= 0.0:
                    zwt[row] -= layer / sy / 1000.0
                    break
                zwt[row] = zi[row, j]
            if qtot > 0.0:
                zwt[row] -= qtot / 1000.0 / rous[row]

    new_state = HydrologyState(
        h2osoi_liq=state.h2osoi_liq.copy(),
        h2osoi_ice=state.h2osoi_ice.copy(),
        h2osfc=state.h2osfc.copy(),
        zwt=_restore_vec(zwt, batch_shape),
        frac_h2osfc=state.frac_h2osfc.copy(),
        frac_sno=state.frac_sno.copy(),
        t_soisno=state.t_soisno.copy(),
        nlevbed=state.nlevbed.copy() if state.nlevbed is not None else None,
    )
    return new_state, _restore_vec(wa_f, batch_shape)


def drainage(
    state: HydrologyState,
    params: HydrologyParameters,
    h2osoi_vol: np.ndarray,
    eff_porosity: np.ndarray,
    hk_l: np.ndarray,
    qcharge: np.ndarray,
    wa: np.ndarray,
    dt: float,
    smp_l: np.ndarray | None = None,
) -> tuple[HydrologyState, dict[str, np.ndarray]]:
    """Simplified default ELM drainage.

    Included pieces:
    - topographic baseflow ``rsub_top`` using the standard exponential form
    - saturation excess from water above effective porosity
    - lower-bound moisture floor repair

    Excluded pieces:
    - VIC baseflow branch
    - perched drainage under frozen layers
    - HUM_HOL and lateral aquifer exchange
    """
    h2osoi_liq, batch_shape = _flatten_batch(state.h2osoi_liq)
    h2osoi_ice, _ = _flatten_batch(state.h2osoi_ice)
    h2osfc = _flatten_vec(state.h2osfc, batch_shape).copy()
    zwt = _flatten_vec(state.zwt, batch_shape).copy()
    wa_f = _flatten_vec(wa, batch_shape).copy()
    h2osoi_vol_f, _ = _flatten_batch(h2osoi_vol)
    eff_porosity_f, _ = _flatten_batch(eff_porosity)
    hk_l_f, _ = _flatten_batch(hk_l)
    smp_l_f = None
    if smp_l is not None:
        smp_l_f, _ = _flatten_batch(smp_l)

    z = np.broadcast_to(_as_r8(params.z), batch_shape + params.z.shape[-1:])
    zi = np.broadcast_to(_as_r8(params.zi), batch_shape + params.zi.shape[-1:])
    dz = np.broadcast_to(_as_r8(params.dz), batch_shape + params.dz.shape[-1:])
    watsat = np.broadcast_to(_as_r8(params.watsat), batch_shape + params.watsat.shape[-1:])
    sucsat = np.broadcast_to(_as_r8(params.sucsat), batch_shape + params.sucsat.shape[-1:])
    bsw = np.broadcast_to(_as_r8(params.bsw), batch_shape + params.bsw.shape[-1:])
    fdrain = _flatten_vec(params.fdrain, batch_shape)
    z, _ = _flatten_batch(z)
    zi, _ = _flatten_batch(zi)
    dz, _ = _flatten_batch(dz)
    watsat, _ = _flatten_batch(watsat)
    sucsat, _ = _flatten_batch(sucsat)
    bsw, _ = _flatten_batch(bsw)
    dzmm = dz * 1.0e3

    nbatch, nlev = h2osoi_liq.shape
    # Native jwt is the number of soil interfaces above the water table.  This
    # preserves a distinct nlev state when zwt is below the soil column; using
    # searchsorted(...)-1 aliases that case to the bottom soil layer.
    jwt = np.sum(zwt[:, None] > zi, axis=1).astype(np.int64)
    below_column = zwt > zi[:, -1]

    topo_slope = _broadcast_optional_vec(params.topo_slope, batch_shape, default=0.0)
    rsub_top_globalmax = 5.5e-3
    rsub_top_max = np.minimum(10.0 * np.sin(np.radians(topo_slope)), rsub_top_globalmax)
    rsub_top = rsub_top_max * np.exp(-fdrain * zwt)
    rsub_top_tot = -rsub_top * dt

    rous = watsat[:, -1] * (1.0 - (1.0 + 1.0e3 * zwt / sucsat[:, -1]) ** (-1.0 / bsw[:, -1]))
    rous = np.maximum(rous, 0.02)

    for row in range(nbatch):
        if below_column[row]:
            # Non-variable-soil-thickness ELM treats below-column topographic
            # drainage as aquifer water loss, not as removal from the bottom
            # soil layer.
            wa_f[row] += rsub_top_tot[row]
            zwt[row] -= rsub_top_tot[row] / 1000.0 / rous[row]
            continue
        if jwt[row] == nlev - 1:
            wa_f[row] = wa_f[row] + rsub_top_tot[row]
            zwt[row] = zwt[row] - rsub_top_tot[row] / 1000.0 / rous[row]
        else:
            rem = rsub_top_tot[row]
            for j in range(jwt[row] + 1, nlev):
                sy = watsat[row, j] * (1.0 - (1.0 + 1.0e3 * zwt[row] / sucsat[row, j]) ** (-1.0 / bsw[row, j]))
                sy = max(sy, 0.02)
                layer = max(rem, -(sy * (zi[row, j] - zwt[row]) * 1.0e3))
                layer = min(layer, 0.0)
                h2osoi_liq[row, j] += layer
                rem -= layer
                if rem >= 0.0:
                    zwt[row] -= layer / sy / 1000.0
                    break
                zwt[row] = zi[row, j]
            if rem < 0.0:
                wa_f[row] += rem
                zwt[row] = zwt[row] - rem / 1000.0 / rous[row]
            rsub_top[row] = -((rsub_top_tot[row] - rem) / dt)

    qflx_rsub_sat = np.zeros(nbatch, dtype=R8)
    # Saturation-excess redistribution: bottom-to-top, vectorized
    for j in range(nlev - 1, 0, -1):
        xsi = np.maximum(h2osoi_liq[:, j] - eff_porosity_f[:, j] * dzmm[:, j], 0.0)
        h2osoi_liq[:, j] -= xsi
        h2osoi_liq[:, j - 1] += xsi

    xs1 = np.maximum(
        np.maximum(h2osoi_liq[:, 0], 0.0)
        - np.maximum(0.0, watsat[:, 0] * dzmm[:, 0] - h2osoi_ice[:, 0]),
        0.0,
    )
    h2osoi_liq[:, 0] -= xs1
    h2osfc += xs1

    # watmin repair: vectorized top-to-bottom
    watmin = 1.0e-8
    xs_all = np.maximum(watmin - h2osoi_liq[:, :-1], 0.0)
    h2osoi_liq[:, :-1] += xs_all
    h2osoi_liq[:, 1:] -= xs_all

    xs = np.maximum(watmin - h2osoi_liq[:, -1], 0.0)
    h2osoi_liq[:, -1] += xs
    rsub_top -= xs / dt

    qflx_drain = qflx_rsub_sat + rsub_top

    new_state = HydrologyState(
        h2osoi_liq=_restore(h2osoi_liq, batch_shape),
        h2osoi_ice=state.h2osoi_ice.copy(),
        h2osfc=_restore_vec(h2osfc, batch_shape),
        zwt=_restore_vec(np.clip(zwt, 0.0, 80.0), batch_shape),
        frac_h2osfc=state.frac_h2osfc.copy(),
        frac_sno=state.frac_sno.copy(),
        t_soisno=state.t_soisno.copy(),
        nlevbed=state.nlevbed.copy() if state.nlevbed is not None else None,
    )
    diagnostics = {
        "qflx_drain": _restore_vec(qflx_drain, batch_shape),
        "qflx_rsub_sat": _restore_vec(qflx_rsub_sat, batch_shape),
        "rsub_top": _restore_vec(rsub_top, batch_shape),
        "wa": _restore_vec(wa_f, batch_shape),
    }
    return new_state, diagnostics


def soil_water_zengdecker2009(
    state: HydrologyState,
    params: HydrologyParameters,
    qflx_infl: np.ndarray,
    qflx_rootsoi: np.ndarray,
    dt: float,
) -> tuple[HydrologyState, dict[str, np.ndarray]]:
    """Vectorized port of the ELM Zeng-Decker soil-water update."""
    h2osoi_liq, batch_shape = _flatten_batch(state.h2osoi_liq)
    h2osoi_ice, _ = _flatten_batch(state.h2osoi_ice)
    t_soisno, _ = _flatten_batch(state.t_soisno)
    qin_top = _flatten_vec(qflx_infl, batch_shape)
    qroot, _ = _flatten_batch(qflx_rootsoi)
    zwt = _flatten_vec(state.zwt, batch_shape)

    dz = np.broadcast_to(_as_r8(params.dz), batch_shape + params.dz.shape[-1:])
    z = np.broadcast_to(_as_r8(params.z), batch_shape + params.z.shape[-1:])
    zi = np.broadcast_to(_as_r8(params.zi), batch_shape + params.zi.shape[-1:])
    watsat = np.broadcast_to(_as_r8(params.watsat), batch_shape + params.watsat.shape[-1:])
    sucsat = np.broadcast_to(_as_r8(params.sucsat), batch_shape + params.sucsat.shape[-1:])
    bsw = np.broadcast_to(_as_r8(params.bsw), batch_shape + params.bsw.shape[-1:])
    hksat = np.broadcast_to(_as_r8(params.hksat), batch_shape + params.hksat.shape[-1:])
    dz, _ = _flatten_batch(dz)
    z, _ = _flatten_batch(z)
    zi, _ = _flatten_batch(zi)
    watsat, _ = _flatten_batch(watsat)
    sucsat, _ = _flatten_batch(sucsat)
    bsw, _ = _flatten_batch(bsw)
    hksat, _ = _flatten_batch(hksat)
    smpmin = _flatten_vec(params.smpmin, batch_shape)
    smp_floor = -np.abs(smpmin)

    nbatch, nlev = h2osoi_liq.shape
    dzmm = dz * 1.0e3
    zmm = z * 1.0e3
    zimm = np.concatenate([np.zeros((nbatch, 1), dtype=R8), zi * 1.0e3], axis=1)
    zwtmm = zwt * 1.0e3

    vol_ice = np.minimum(watsat, h2osoi_ice / (dz * DENICE))
    icefrac = np.minimum(1.0, vol_ice / np.maximum(watsat, 1.0e-12))
    vliq = np.maximum(h2osoi_liq, 1.0e-6) / (dz * DENH2O)
    h2osoi_vol = h2osoi_liq / (dz * DENH2O) + h2osoi_ice / (dz * DENICE)

    # Keep below-column water tables distinct so the extra aquifer row is
    # coupled into the solve instead of aliasing them to the bottom soil layer.
    jwt = np.sum(zwt[:, None] > zi, axis=1).astype(np.int64)

    vwc_zwt = watsat[:, -1].copy()
    rows = np.arange(nbatch)
    jwt1 = np.minimum(jwt, nlev - 1)
    frozen = t_soisno[rows, jwt1] < TFRZ
    if np.any(frozen):
        j = nlev - 1
        rowf = rows[frozen]
        smp1 = HFUS * (TFRZ - t_soisno[rowf, j]) / (GRAV * t_soisno[rowf, j]) * 1000.0
        smp1 = np.maximum(sucsat[rowf, j], smp1)
        vwc_zwt[rowf] = watsat[rowf, j] * (smp1 / sucsat[rowf, j]) ** (-1.0 / bsw[rowf, j])
        vwc_zwt[rowf] = np.minimum(vwc_zwt[rowf], 0.5 * (watsat[rowf, j] + h2osoi_vol[rowf, j]))

    vol_eq = np.zeros((nbatch, nlev), dtype=R8)
    zq = np.zeros((nbatch, nlev + 1), dtype=R8)
    # Fully vectorized over both batch and layer dimensions
    # zwtmm: (nbatch,)  zimm: (nbatch, nlev+1)  watsat/sucsat/bsw: (nbatch, nlev)
    _zwtmm = zwtmm[:, None]  # (nbatch, 1) for broadcasting
    _zi0 = zimm[:, :nlev]    # (nbatch, nlev) lower interface of each layer
    _zi1 = zimm[:, 1:]       # (nbatch, nlev) upper interface of each layer

    sat_mask  = _zwtmm <= _zi0
    mid_mask  = (_zwtmm > _zi0) & (_zwtmm < _zi1)
    deep_mask = _zwtmm >= _zi1

    vol_eq = np.where(sat_mask, watsat, vol_eq)

    # mid case
    _exp = 1.0 - 1.0 / bsw
    _dz_layer = np.maximum(_zi1 - _zi0, 1.0e-12)
    _dz_above = np.maximum(_zwtmm - _zi0, 1.0e-12)
    temp0_mid = ((sucsat + _dz_above) / sucsat) ** _exp
    voleq1_mid = (-sucsat * watsat / _exp / _dz_above * (1.0 - temp0_mid))
    vol_eq_mid = (voleq1_mid * (_zwtmm - _zi0) + watsat * (_zi1 - _zwtmm)) / _dz_layer
    vol_eq = np.where(mid_mask, vol_eq_mid, vol_eq)

    # deep case
    _dz_to_bot = np.maximum(_zwtmm - _zi1, 1.0e-12)
    _dz_to_top = np.maximum(_zwtmm - _zi0, 1.0e-12)
    tempi_deep = ((sucsat + _dz_to_bot) / sucsat) ** _exp
    temp0_deep = ((sucsat + _dz_to_top) / sucsat) ** _exp
    vol_eq_deep = (-sucsat * watsat / _exp / _dz_layer * (tempi_deep - temp0_deep))
    vol_eq = np.where(deep_mask, vol_eq_deep, vol_eq)

    vol_eq = np.clip(vol_eq, 0.0, watsat)
    zq[:, :nlev] = np.maximum(-sucsat * np.maximum(vol_eq / watsat, 0.01) ** (-bsw), smp_floor[:, None])

    below_column = jwt >= nlev
    zq[:, -1] = zq[:, -2]
    if np.any(below_column):
        j = nlev - 1
        delta_z_zwt = np.maximum(zwtmm[below_column] - zimm[below_column, nlev], 1.0)
        temp0 = ((sucsat[below_column, j] + delta_z_zwt) / sucsat[below_column, j]) ** (
            1.0 - 1.0 / bsw[below_column, j]
        )
        vol_eq_aq = -sucsat[below_column, j] * watsat[below_column, j] / (
            1.0 - 1.0 / bsw[below_column, j]
        ) / delta_z_zwt * (1.0 - temp0)
        vol_eq_aq = np.clip(vol_eq_aq, 0.0, watsat[below_column, j])
        zq[below_column, nlev] = np.maximum(
            -sucsat[below_column, j] * np.maximum(vol_eq_aq / watsat[below_column, j], 0.01) ** (-bsw[below_column, j]),
            smp_floor[below_column],
        )

    smp, dsmpdw = _clapp_hornberger_smp(vliq, watsat, sucsat, bsw, smpmin)
    hk, dhkdw = _clapp_hornberger_hk(vliq, watsat, hksat, bsw, icefrac)

    nsolve = nlev + 1
    a = np.zeros((nbatch, nsolve), dtype=R8)
    b = np.zeros((nbatch, nsolve), dtype=R8)
    c = np.zeros((nbatch, nsolve), dtype=R8)
    r = np.zeros((nbatch, nsolve), dtype=R8)

    qin = np.zeros((nbatch, nsolve), dtype=R8)
    qout = np.zeros((nbatch, nsolve), dtype=R8)
    qroot_ext = np.zeros((nbatch, nsolve), dtype=R8)
    qroot_ext[:, :nlev] = qroot

    # Fully vectorized tridiagonal assembly over all layers
    # Inter-layer fluxes: qout[j] = flux from layer j to j+1 (downward positive)
    # qin[j] = flux from layer j-1 to j = qout[j-1]
    den_mid = np.maximum(zmm[:, 1:] - zmm[:, :-1], 1.0e-12)  # (nbatch, nlev-1)
    num_mid = (smp[:, 1:] - smp[:, :-1]) - (zq[:, 1:nlev] - zq[:, :nlev-1])
    q_mid = -hk * num_mid / den_mid  # flux between adjacent layers
    dqdw_upper = -(-hk * dsmpdw[:, :-1] + num_mid * dhkdw) / den_mid
    dqdw_lower = -(hk * dsmpdw[:, 1:]  + num_mid * dhkdw) / den_mid

    qin[:, 0] = qin_top
    qin[:, 1:nlev] = q_mid
    qout[:, : nlev - 1] = q_mid

    # Native ELM solves an extra aquifer unknown when the water table is below
    # the soil column.  The bottom soil layer is then coupled to that unknown;
    # otherwise the bottom soil boundary is no-flow and the extra row is
    # hydrologically inactive.
    jbot = nlev - 1
    s_node_aq = np.clip(0.5 * (vwc_zwt + vliq[:, jbot]) / np.maximum(watsat[:, jbot], 1.0e-12), 0.01, 1.0)
    smp_aq = -sucsat[:, jbot] * s_node_aq ** (-bsw[:, jbot])
    smp_aq = np.maximum(smp_aq, smp_floor)
    dsmpdw_aq = -bsw[:, jbot] * smp_aq / np.maximum(s_node_aq * watsat[:, jbot], 1.0e-12)
    s1_bottom = np.clip(vliq[:, jbot] / np.maximum(watsat[:, jbot], 1.0e-12), 0.0, 1.0)
    s2_bottom = hksat[:, jbot] * s1_bottom ** (2.0 * bsw[:, jbot] + 2.0)
    imped_bottom = 10.0 ** (-E_ICE * icefrac[:, jbot])
    hk_bottom = imped_bottom * s1_bottom * s2_bottom
    dhkdw_bottom = (
        imped_bottom
        * (2.0 * bsw[:, jbot] + 3.0)
        * s2_bottom
        / np.maximum(2.0 * watsat[:, jbot], 1.0e-12)
    )
    zmm_aq = 0.5 * (zwtmm + zmm[:, jbot])
    dzmm_aq = np.maximum(zwtmm - zmm[:, jbot], 1.0e-12)
    den_bottom = np.maximum(zmm_aq - zmm[:, jbot], 1.0e-12)
    num_bottom = (smp_aq - smp[:, jbot]) - (zq[:, nlev] - zq[:, jbot])
    qout_bottom = -hk_bottom * num_bottom / den_bottom
    dqbot_dw_soil = -(-hk_bottom * dsmpdw[:, jbot] + num_bottom * dhkdw_bottom) / den_bottom
    dqbot_dw_aq = -(hk_bottom * dsmpdw_aq + num_bottom * dhkdw_bottom) / den_bottom
    qout_bottom = np.where(below_column, qout_bottom, 0.0)
    dqbot_dw_soil = np.where(below_column, dqbot_dw_soil, 0.0)
    dqbot_dw_aq = np.where(below_column, dqbot_dw_aq, 0.0)
    qout[:, jbot] = qout_bottom
    qin[:, nlev] = qout_bottom

    r[:, :] = qin - qout - qroot_ext

    # top layer
    b[:, 0] = dzmm[:, 0] / dt + dqdw_upper[:, 0]
    c[:, 0] = dqdw_lower[:, 0]

    # interior layers 1..nlev-2
    if nlev > 2:
        a[:, 1 : nlev - 1] = -dqdw_upper[:, :-1]
        b[:, 1 : nlev - 1] = dzmm[:, 1:-1] / dt - dqdw_lower[:, :-1] + dqdw_upper[:, 1:]
        c[:, 1 : nlev - 1] = dqdw_lower[:, 1:]

    # bottom soil layer
    a[:, jbot] = -dqdw_upper[:, -1]
    b[:, jbot] = dzmm[:, jbot] / dt - dqdw_lower[:, -1] + dqbot_dw_soil
    c[:, jbot] = dqbot_dw_aq

    # aquifer row: active only when water table is below the soil column
    a[:, nlev] = np.where(below_column, -dqbot_dw_soil, 0.0)
    b[:, nlev] = np.where(below_column, dzmm_aq / dt - dqbot_dw_aq, dzmm[:, jbot] / dt)

    dwat = _batched_tridiagonal(a, b, c, r)
    h2osoi_liq_new = h2osoi_liq + dwat[:, :nlev] * dzmm

    qcharge = np.zeros(nbatch, dtype=R8)
    in_column = jwt < nlev
    if np.any(in_column):
        jj = np.minimum(jwt[in_column], nlev - 1)
        rows = np.arange(nbatch)[in_column]
        s_node = np.clip(h2osoi_vol[in_column, jj] / watsat[in_column, jj], 0.01, 1.0)
        s1 = np.minimum(1.0, s_node)
        imped = 10.0 ** (-E_ICE * icefrac[in_column, jj])
        ka = imped * hksat[in_column, jj] * s1 ** (2.0 * bsw[in_column, jj] + 3.0)
        above = np.maximum(jwt[in_column] - 1, 0)
        smp1 = np.maximum(smp_floor[in_column], smp[in_column, above])
        wh = smp1 - zq[in_column, above]
        dist = np.where(
            jwt[in_column] == 0,
            (zwt[in_column] + 1.0e-3) * 1000.0,
            np.maximum((zwt[in_column] - z[in_column, above]) * 1000.0 * 2.0, 1.0e-6),
        )
        qcharge[rows] = np.clip(-ka * (0.0 - wh) / dist, -10.0 / dt, 10.0 / dt)
    if np.any(below_column):
        qcharge[below_column] = dwat[below_column, nlev] * dzmm_aq[below_column] / dt

    h2osoi_vol_new = h2osoi_liq_new / (dz * DENH2O) + h2osoi_ice / (dz * DENICE)
    dWdt = (h2osoi_liq_new - h2osoi_liq) / dt
    qout_actual = np.zeros((nbatch, nlev), dtype=R8)
    qout_actual[:, 0] = qin_top - qroot[:, 0] - dWdt[:, 0]
    for j in range(1, nlev):
        qout_actual[:, j] = qout_actual[:, j - 1] - qroot[:, j] - dWdt[:, j]
    dW1dt_layer1 = dWdt[:, 0]
    smp_l, _ = _clapp_hornberger_smp(
        np.maximum(h2osoi_liq_new, 1.0e-6) / (dz * DENH2O),
        watsat,
        sucsat,
        bsw,
        smpmin,
    )
    hk_l, _ = _clapp_hornberger_hk(
        np.maximum(h2osoi_liq_new, 1.0e-6) / (dz * DENH2O),
        watsat,
        hksat,
        bsw,
        np.minimum(1.0, np.minimum(watsat, h2osoi_ice / (dz * DENICE)) / np.maximum(watsat, 1.0e-12)),
    )

    new_state = HydrologyState(
        h2osoi_liq=_restore(h2osoi_liq_new, batch_shape),
        h2osoi_ice=state.h2osoi_ice.copy(),
        h2osfc=state.h2osfc.copy(),
        zwt=state.zwt.copy(),
        frac_h2osfc=state.frac_h2osfc.copy(),
        frac_sno=state.frac_sno.copy(),
        t_soisno=state.t_soisno.copy(),
        nlevbed=state.nlevbed.copy() if state.nlevbed is not None else None,
    )
    diagnostics = {
        "qcharge": _restore_vec(qcharge, batch_shape),
        "h2osoi_vol": _restore(h2osoi_vol_new, batch_shape),
        "eff_porosity": _restore(watsat - np.minimum(watsat, h2osoi_ice / (dz * DENICE)), batch_shape),
        "smp_l": _restore(smp_l, batch_shape),
        "hk_l": _restore(
            np.concatenate([hk_l, hk_l[:, -1:]], axis=1)[:, :nlev],
            batch_shape,
        ),
        "qin_layer1": _restore_vec(qin[:, 0], batch_shape),
        "qout_layer1": _restore_vec(qout_actual[:, 0], batch_shape),
        "qroot_layer1": _restore_vec(qroot[:, 0], batch_shape),
        "dW1dt_layer1": _restore_vec(dW1dt_layer1, batch_shape),
    }
    return new_state, diagnostics


def hydrology_no_drainage_step(
    state: HydrologyState,
    params: HydrologyParameters,
    forcing: HydrologyForcing,
    dt: float,
) -> HydrologyResult:
    """Run a vectorized hydrology step using the ELM no-drainage decomposition.

    Positive ``qcharge`` is consumed immediately as a simple lower-boundary
    deep-drainage sink rather than being accumulated in a groundwater store.
    This makes the standalone column usable for multi-year synthetic tests.
    """
    qflx_surf, fsat, _ = surface_runoff(state, params, forcing)
    state2, infil = infiltration(state, params, forcing, qflx_surf=qflx_surf, fsat=fsat, dt=dt)
    batch_shape = state.h2osoi_liq.shape[:-1]
    qroot = compute_root_sink(forcing, params, batch_shape)
    state3, diag = soil_water_zengdecker2009(state2, params, qflx_infl=infil["qflx_infl"], qflx_rootsoi=qroot, dt=dt)
    # Update the water table depth from qcharge so that the Zeng-Decker
    # equilibrium profile (vol_eq / zq) stays consistent with the actual
    # saturated-zone position on the next timestep.  Without this, zwt is
    # frozen at its initial value: after a rain event wets the upper layers
    # above the stale equilibrium, the solver generates spurious downward
    # hydraulic fluxes that drain those layers immediately.
    wa_zero = np.zeros_like(state.zwt)
    state3, _ = water_table(state3, params, diag["qcharge"], wa_zero, dt=dt)
    hliq_flat, _ = _flatten_batch(state3.h2osoi_liq)
    hice_flat, _ = _flatten_batch(state3.h2osoi_ice)
    hsurf_flat = _flatten_vec(state3.h2osfc, batch_shape)
    effp_flat, _ = _flatten_batch(diag["eff_porosity"])
    dz = np.broadcast_to(_as_r8(params.dz), batch_shape + params.dz.shape[-1:])
    dz_flat, _ = _flatten_batch(dz)
    qdeep = np.maximum(_flatten_vec(diag["qcharge"], batch_shape), 0.0)
    bottom_available = np.maximum(hliq_flat[:, -1], 0.0) / dt
    qdeep = np.minimum(qdeep, bottom_available)
    hliq_flat[:, -1] = np.maximum(0.0, hliq_flat[:, -1] - qdeep * dt)
    hliq_flat, hsurf_flat = _redistribute_saturation_excess(
        hliq_flat,
        hice_flat,
        effp_flat,
        dz_flat,
        hsurf_flat,
    )
    hliq_repaired, hsurf_repaired = _repair_liquid_water_floor(hliq_flat, hsurf_flat, dt)
    state3 = HydrologyState(
        h2osoi_liq=_restore(hliq_repaired, batch_shape),
        h2osoi_ice=state3.h2osoi_ice,
        h2osfc=_restore_vec(hsurf_repaired, batch_shape),
        zwt=state3.zwt,
        frac_h2osfc=state3.frac_h2osfc,
        frac_sno=state3.frac_sno,
        t_soisno=state3.t_soisno,
        nlevbed=state3.nlevbed,
    )
    diag["h2osoi_vol"] = state3.h2osoi_liq / (dz * DENH2O) + state3.h2osoi_ice / (dz * DENICE)
    return HydrologyResult(
        state=state3,
        wa=None,
        qflx_surf=qflx_surf,
        qflx_infl=infil["qflx_infl"],
        qflx_h2osfc_surf=infil["qflx_h2osfc_surf"],
        qflx_h2osfc_drain=infil["qflx_h2osfc_drain"],
        qflx_infl_excess=infil["qflx_infl_excess"],
        qcharge=diag["qcharge"],
        qflx_deep_drain=_restore_vec(qdeep, batch_shape),
        qflx_drain=None,
        qflx_rsub_sat=None,
        h2osoi_vol=diag["h2osoi_vol"],
        eff_porosity=diag["eff_porosity"],
        smp_l=diag["smp_l"],
        hk_l=diag["hk_l"],
        qin_layer1=diag["qin_layer1"],
        qout_layer1=diag["qout_layer1"],
        qroot_layer1=diag["qroot_layer1"],
        dW1dt_layer1=diag["dW1dt_layer1"],
        zwt_after_qcharge=state3.zwt,
        zwt_after_drainage=state3.zwt,
    )


def hydrology_step(
    state: HydrologyState,
    params: HydrologyParameters,
    forcing: HydrologyForcing,
    dt: float,
    wa: Optional[np.ndarray] = None,
) -> HydrologyResult:
    """Run the standard non-VIC, non-HUM_HOL ELM hydrology sequence.

    This is the closest current Python analogue of:
    ``SurfaceRunoff -> Infiltration -> SoilWater -> WaterTable -> Drainage``.
    """
    if wa is None:
        wa = np.zeros(state.h2osfc.shape, dtype=R8)

    qflx_surf, fsat, _ = surface_runoff(state, params, forcing)
    state2, infil = infiltration(state, params, forcing, qflx_surf=qflx_surf, fsat=fsat, dt=dt)
    batch_shape = state.h2osoi_liq.shape[:-1]
    qroot = compute_root_sink(forcing, params, batch_shape)
    state3, diag = soil_water_zengdecker2009(
        state2,
        params,
        qflx_infl=infil["qflx_infl"],
        qflx_rootsoi=qroot,
        dt=dt,
    )
    state4, wa2 = water_table(state3, params, qcharge=diag["qcharge"], wa=wa, dt=dt)
    state5, drain = drainage(
        state4,
        params,
        h2osoi_vol=diag["h2osoi_vol"],
        eff_porosity=diag["eff_porosity"],
        hk_l=diag["hk_l"],
        qcharge=diag["qcharge"],
        wa=wa2,
        dt=dt,
        smp_l=diag["smp_l"],
    )
    return HydrologyResult(
        state=state5,
        wa=drain["wa"],
        qflx_surf=qflx_surf,
        qflx_infl=infil["qflx_infl"],
        qflx_h2osfc_surf=infil["qflx_h2osfc_surf"],
        qflx_h2osfc_drain=infil["qflx_h2osfc_drain"],
        qflx_infl_excess=infil["qflx_infl_excess"],
        qcharge=diag["qcharge"],
        qflx_deep_drain=diag["qcharge"],
        qflx_drain=drain["qflx_drain"],
        qflx_rsub_sat=drain["qflx_rsub_sat"],
        h2osoi_vol=diag["h2osoi_vol"],
        eff_porosity=diag["eff_porosity"],
        smp_l=diag["smp_l"],
        hk_l=diag["hk_l"],
        qin_layer1=diag["qin_layer1"],
        qout_layer1=diag["qout_layer1"],
        qroot_layer1=diag["qroot_layer1"],
        dW1dt_layer1=diag["dW1dt_layer1"],
        zwt_after_qcharge=state4.zwt,
        zwt_after_drainage=state5.zwt,
    )


__all__ = [
    "HydrologyState",
    "HydrologyParameters",
    "SoilTextureProfile",
    "HydrologyForcing",
    "HydrologyResult",
    "elm_soil_grid",
    "pedotransfer_cosby1984_table5",
    "hydraulic_params_from_texture",
    "surface_runoff",
    "infiltration",
    "soil_water_zengdecker2009",
    "water_table",
    "drainage",
    "hydrology_no_drainage_step",
    "hydrology_step",
]
