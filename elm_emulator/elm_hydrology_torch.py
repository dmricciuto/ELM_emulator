"""PyTorch port of the batched ELM soil-water core.

This module ports the heavy part of the Python hydrology prototype to Torch:
- the batched tridiagonal solver
- the Zeng-Decker unsaturated soil-water update

It is intended for device execution on CPU / MPS / CUDA and accepts arbitrary
leading batch dimensions with the layer dimension last.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import torch


DENH2O = 1000.0
DENICE = 917.0
E_ICE = 5.0
TFRZ = 273.15
HFUS = 3.337e5
GRAV = 9.80616


@dataclass
class SoilWaterCoreState:
    h2osoi_liq: torch.Tensor
    h2osoi_ice: torch.Tensor
    t_soisno: torch.Tensor
    zwt: torch.Tensor


@dataclass
class SoilHydrologyNoDrainageState:
    h2osoi_liq: torch.Tensor
    h2osoi_ice: torch.Tensor
    h2osfc: torch.Tensor
    t_soisno: torch.Tensor
    zwt: torch.Tensor


@dataclass
class SoilWaterCoreParameters:
    dz: torch.Tensor
    z: torch.Tensor
    zi: torch.Tensor
    watsat: torch.Tensor
    sucsat: torch.Tensor
    bsw: torch.Tensor
    hksat: torch.Tensor
    smpmin: torch.Tensor
    wtfact: torch.Tensor | None = None
    fover: torch.Tensor | None = None
    fdrain: torch.Tensor | None = None
    h2osfc_thresh: torch.Tensor | None = None
    pc: torch.Tensor | None = None
    topo_slope: torch.Tensor | None = None
    rsub_top_globalmax: torch.Tensor | None = None


@dataclass
class SoilWaterCoreResult:
    h2osoi_liq: torch.Tensor
    h2osoi_vol: torch.Tensor
    qcharge: torch.Tensor
    smp_l: torch.Tensor
    hk_l: torch.Tensor
    qin_layer1: torch.Tensor
    qout_layer1: torch.Tensor
    qroot_layer1: torch.Tensor
    dW1dt_layer1: torch.Tensor


@dataclass
class SoilHydrologyTorchResult:
    h2osoi_liq: torch.Tensor
    h2osoi_vol: torch.Tensor
    h2osfc: torch.Tensor
    qflx_surf: torch.Tensor
    qflx_infl: torch.Tensor
    qflx_soil_evap: torch.Tensor
    qflx_h2osfc_surf: torch.Tensor
    qflx_h2osfc_drain: torch.Tensor
    qflx_infl_excess: torch.Tensor
    qcharge: torch.Tensor
    smp_l: torch.Tensor
    hk_l: torch.Tensor
    qflx_drain: torch.Tensor | None = None
    qflx_rsub_sat: torch.Tensor | None = None
    zwt: torch.Tensor | None = None
    wa: torch.Tensor | None = None
    qflx_deep_drain: torch.Tensor | None = None
    eff_porosity: torch.Tensor | None = None
    qin_layer1: torch.Tensor | None = None
    qout_layer1: torch.Tensor | None = None
    qroot_layer1: torch.Tensor | None = None
    dW1dt_layer1: torch.Tensor | None = None
    zwt_after_qcharge: torch.Tensor | None = None
    zwt_after_drainage: torch.Tensor | None = None


@dataclass
class SoilHydrologyTorchWorkspace:
    """Reusable scratch buffers for the Torch hydrology solver."""

    a: torch.Tensor | None = None
    b: torch.Tensor | None = None
    c: torch.Tensor | None = None
    r: torch.Tensor | None = None
    qin: torch.Tensor | None = None
    qout: torch.Tensor | None = None
    vol_eq: torch.Tensor | None = None
    zq: torch.Tensor | None = None
    cp: torch.Tensor | None = None
    dp: torch.Tensor | None = None
    x: torch.Tensor | None = None

    def tensor(self, name: str, shape: tuple[int, ...], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        buf = getattr(self, name)
        if buf is None or tuple(buf.shape) != shape or buf.device != device or buf.dtype != dtype:
            buf = torch.empty(shape, device=device, dtype=dtype)
            setattr(self, name, buf)
        return buf


def _broadcast_last_like(x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.broadcast_to(x.to(device=target.device, dtype=target.dtype), target.shape)


def _flatten_last(x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
    batch_shape = tuple(x.shape[:-1])
    return x.reshape(-1, x.shape[-1]), batch_shape


def _flatten_vec(x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
    batch_shape = tuple(x.shape)
    return x.reshape(-1), batch_shape


def _restore_last(x: torch.Tensor, batch_shape: tuple[int, ...]) -> torch.Tensor:
    return x.reshape(batch_shape + (x.shape[-1],))


def _restore_vec(x: torch.Tensor, batch_shape: tuple[int, ...]) -> torch.Tensor:
    return x.reshape(batch_shape)


def _sync_for_profile(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()


def _profiled(
    profile_sections: dict[str, float] | None,
    name: str,
    device: torch.device,
    fn,
):
    if profile_sections is None:
        return fn()
    _sync_for_profile(device)
    t0 = time.perf_counter()
    out = fn()
    _sync_for_profile(device)
    profile_sections[name] = profile_sections.get(name, 0.0) + (time.perf_counter() - t0)
    return out


def _expand_last(
    x: torch.Tensor,
    batch_shape: tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    return torch.broadcast_to(x.to(device=device, dtype=dtype), batch_shape + (x.shape[-1],)).reshape(-1, x.shape[-1])


def _expand_vec(
    x: torch.Tensor,
    batch_shape: tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    x = x.to(device=device, dtype=dtype)
    nbatch = 1
    for s in batch_shape:
        nbatch *= s
    if x.numel() == nbatch:
        return x.reshape(-1)
    return torch.broadcast_to(x, batch_shape).reshape(-1)


def batched_tridiagonal_torch(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    r: torch.Tensor,
    workspace: SoilHydrologyTorchWorkspace | None = None,
) -> torch.Tensor:
    """Solve batched tridiagonal systems (Thomas algorithm).

    All arrays are shaped ``(nbatch, n)``.

    The *workspace* argument is accepted for API compatibility but ignored;
    fresh temporaries are always allocated so this function is fully
    ``torch.compile``-friendly (no workspace.tensor() graph breaks).
    """
    nbatch, n = b.shape
    cp = torch.empty_like(c)
    dp = torch.empty_like(r)
    x = torch.empty_like(r)

    # safe denominator: magnitude >= 1e-12, sign preserved
    def _sd(raw: torch.Tensor) -> torch.Tensor:
        a2 = raw.abs().clamp(min=1.0e-12)
        return torch.where(raw >= 0.0, a2, -a2)

    cp[:, 0] = c[:, 0] / _sd(b[:, 0])
    dp[:, 0] = r[:, 0] / _sd(b[:, 0])

    for j in range(1, n):
        denom = _sd(b[:, j] - a[:, j] * cp[:, j - 1])
        if j < n - 1:
            cp[:, j] = c[:, j] / denom
        dp[:, j] = (r[:, j] - a[:, j] * dp[:, j - 1]) / denom

    x[:, -1] = dp[:, -1]
    for j in range(n - 2, -1, -1):
        x[:, j] = dp[:, j] - cp[:, j] * x[:, j + 1]

    return x


def _clapp_hornberger_smp_torch(
    vliq: torch.Tensor,
    watsat: torch.Tensor,
    sucsat: torch.Tensor,
    bsw: torch.Tensor,
    smpmin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    s_node = torch.clamp(vliq / watsat, min=0.01, max=1.0)
    smp = -sucsat * torch.pow(s_node, -bsw)
    smp_floor = -torch.abs(smpmin[:, None])
    smp = torch.maximum(smp, smp_floor)
    dsmpdw = -bsw * smp / torch.clamp(vliq, min=1.0e-12)
    return smp, dsmpdw


def _clapp_hornberger_hk_torch(
    vliq: torch.Tensor,
    watsat: torch.Tensor,
    hksat: torch.Tensor,
    bsw: torch.Tensor,
    icefrac: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    s1 = 0.5 * (vliq[:, :-1] + vliq[:, 1:]) / (0.5 * (watsat[:, :-1] + watsat[:, 1:]))
    s1 = torch.clamp(s1, min=0.0, max=1.0)
    s2 = hksat[:, :-1] * torch.pow(s1, 2.0 * bsw[:, :-1] + 2.0)
    imped = torch.pow(
        torch.full_like(s1, 10.0),
        -E_ICE * 0.5 * (icefrac[:, :-1] + icefrac[:, 1:]),
    )
    hk = imped * s1 * s2
    dhkdw = imped * (2.0 * bsw[:, :-1] + 3.0) * s2 / torch.clamp(
        watsat[:, :-1] + watsat[:, 1:],
        min=1.0e-12,
    )
    return hk, dhkdw


def _redistribute_saturation_excess_torch(
    h2osoi_liq: torch.Tensor,
    h2osoi_ice: torch.Tensor,
    eff_porosity: torch.Tensor,
    dz: torch.Tensor,
    h2osfc: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Torch port of the saturation-excess cleanup from the ELM drainage path."""
    hliq = h2osoi_liq.clone()
    dzmm = dz * 1.0e3
    hsurf = h2osfc.clone()
    nlev = hliq.shape[1]

    for j in range(nlev - 1, 0, -1):
        max_liq = eff_porosity[:, j] * dzmm[:, j]
        excess = torch.clamp(hliq[:, j] - max_liq, min=0.0)
        hliq[:, j] = torch.minimum(max_liq, hliq[:, j])
        hliq[:, j - 1] = hliq[:, j - 1] + excess

    top_capacity = torch.clamp(eff_porosity[:, 0] * dzmm[:, 0], min=0.0)
    top_excess = torch.clamp(hliq[:, 0] - top_capacity, min=0.0)
    hliq[:, 0] = hliq[:, 0] - top_excess
    hsurf = hsurf + top_excess

    return hliq, hsurf


def _repair_liquid_water_floor_torch(
    h2osoi_liq: torch.Tensor,
    h2osfc: torch.Tensor,
    dt: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Torch port of the watmin floor repair from the ELM drainage path."""
    watmin = 1.0e-8
    hliq = h2osoi_liq.clone()
    hsurf = h2osfc.clone()
    nlev = hliq.shape[1]

    for j in range(nlev - 1):
        xs = torch.where(
            hliq[:, j] < watmin,
            watmin - hliq[:, j],
            torch.zeros_like(hliq[:, j]),
        )
        hliq[:, j] = hliq[:, j] + xs
        hliq[:, j + 1] = hliq[:, j + 1] - xs

    xs1 = torch.clamp(-hliq[:, 0], min=0.0)
    hliq[:, 0] = hliq[:, 0] + xs1
    hsurf = hsurf + xs1

    xs = torch.where(
        hliq[:, -1] < watmin,
        watmin - hliq[:, -1],
        torch.zeros_like(hliq[:, -1]),
    )
    hliq[:, -1] = hliq[:, -1] + xs

    return hliq, hsurf


def surface_runoff_torch(
    h2osoi_liq: torch.Tensor,
    h2osoi_ice: torch.Tensor,
    zwt: torch.Tensor,
    qflx_top_soil: torch.Tensor,
    dz: torch.Tensor,
    watsat: torch.Tensor,
    wtfact: torch.Tensor,
    fover: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    vol_ice = torch.minimum(watsat, h2osoi_ice / (dz * DENICE))
    icefrac = torch.minimum(torch.ones_like(watsat), vol_ice / torch.clamp(watsat, min=1.0e-12))
    fsat = wtfact * torch.exp(-0.5 * fover * zwt)
    qflx_surf = fsat * qflx_top_soil
    return qflx_surf, fsat, icefrac


def infiltration_torch(
    h2osoi_liq: torch.Tensor,
    h2osoi_ice: torch.Tensor,
    h2osfc: torch.Tensor,
    frac_h2osfc: torch.Tensor,
    frac_sno: torch.Tensor,
    t_soisno: torch.Tensor,
    qflx_top_soil: torch.Tensor,
    qflx_evap_grnd: torch.Tensor,
    qflx_ev_soil: torch.Tensor,
    qflx_ev_h2osfc: torch.Tensor,
    qflx_surf: torch.Tensor,
    fsat: torch.Tensor,
    dz: torch.Tensor,
    watsat: torch.Tensor,
    hksat: torch.Tensor,
    h2osfc_thresh: torch.Tensor,
    pc: torch.Tensor,
    topo_slope: torch.Tensor | None,
    dt: float,
) -> dict[str, torch.Tensor]:
    vol_ice = torch.minimum(watsat, h2osoi_ice / (dz * DENICE))
    eff_porosity = torch.clamp(watsat - vol_ice, min=0.01)
    icefrac = torch.minimum(torch.ones_like(watsat), vol_ice / torch.clamp(watsat, min=1.0e-12))

    has_active_snow = frac_sno > 1.0e-12
    fsno = torch.where(has_active_snow, frac_sno, torch.zeros_like(frac_sno))
    qflx_evap = torch.where(has_active_snow, qflx_ev_soil, qflx_evap_grnd)

    qflx_in_soil = (1.0 - frac_h2osfc) * (qflx_top_soil - qflx_surf)
    qflx_in_h2osfc = frac_h2osfc * (qflx_top_soil - qflx_surf)
    soil_evap_area = torch.clamp(1.0 - fsno - frac_h2osfc, min=0.0, max=1.0)
    qflx_soil_evap = soil_evap_area * qflx_evap
    # Mirror the NumPy path: preserve negative infiltration as the top soil
    # evaporation boundary, but cap positive evaporation by available layer-1
    # liquid plus same-step incoming water.
    layer1_reserve = 0.01 * eff_porosity[:, 0] * dz[:, 0] * DENH2O
    layer1_available = torch.clamp(h2osoi_liq[:, 0] - layer1_reserve, min=0.0) / dt
    evap_supply = torch.clamp(qflx_in_soil, min=0.0) + layer1_available
    qflx_soil_evap = torch.where(qflx_soil_evap > 0.0, torch.minimum(qflx_soil_evap, evap_supply), qflx_soil_evap)
    qflx_in_soil = qflx_in_soil - qflx_soil_evap
    qflx_in_h2osfc = qflx_in_h2osfc - frac_h2osfc * qflx_ev_h2osfc

    qinmax = (1.0 - fsat) * torch.min(torch.pow(torch.full_like(icefrac[:, :3], 10.0), -E_ICE * icefrac[:, :3]) * hksat[:, :3], dim=1).values
    qflx_infl_excess = torch.clamp(qflx_in_soil - (1.0 - frac_h2osfc) * qinmax, min=0.0)
    qflx_infl = qflx_in_soil - qflx_infl_excess
    qflx_in_h2osfc = qflx_in_h2osfc + qflx_infl_excess

    active_storage = torch.clamp(h2osfc - h2osfc_thresh, min=0.0)
    frac_infclust = torch.where(
        frac_h2osfc <= pc,
            torch.zeros_like(frac_h2osfc),
        (frac_h2osfc - pc) ** 1.0,
    )
    topo_slope_deg = torch.full_like(frac_h2osfc, 2.0) if topo_slope is None else topo_slope
    k_wet = torch.sin(torch.deg2rad(topo_slope_deg))
    qflx_h2osfc_surf = torch.minimum(k_wet * frac_infclust * active_storage, active_storage / dt)
    qflx_h2osfc_surf = torch.where(
        qflx_h2osfc_surf < 1.0e-8,
        torch.zeros_like(qflx_h2osfc_surf),
        qflx_h2osfc_surf,
    )
    qflx_in_h2osfc = qflx_in_h2osfc - qflx_h2osfc_surf

    # Do not let potential surface-water evaporation/runoff borrow from soil.
    # Cap the net h2osfc loss at zero storage instead of transferring any
    # negative-storage deficit into qflx_infl/layer 1.
    qflx_in_h2osfc = torch.maximum(qflx_in_h2osfc, -h2osfc / dt)
    h2osfc_new = h2osfc + qflx_in_h2osfc * dt

    h2osfc_new = torch.clamp(h2osfc_new, min=0.0)

    qflx_h2osfc_drain = torch.minimum(
        frac_h2osfc * qinmax,
        torch.clamp(h2osfc_new, min=0.0) / dt,
    )
    h2osfc_new = torch.clamp(h2osfc_new - qflx_h2osfc_drain * dt, min=0.0)
    qflx_infl = qflx_infl + qflx_h2osfc_drain

    return {
        "h2osfc": h2osfc_new,
        "qflx_infl": qflx_infl,
        "qflx_soil_evap": qflx_soil_evap,
        "qflx_h2osfc_surf": qflx_h2osfc_surf,
        "qflx_h2osfc_drain": qflx_h2osfc_drain,
        "qflx_infl_excess": qflx_infl_excess,
        "eff_porosity": eff_porosity,
        "icefrac": icefrac,
    }


def soil_water_zengdecker2009_torch(
    state: SoilWaterCoreState,
    params: SoilWaterCoreParameters,
    qflx_infl: torch.Tensor,
    qflx_rootsoi: torch.Tensor,
    dt: float,
    workspace: SoilHydrologyTorchWorkspace | None = None,
) -> SoilWaterCoreResult:
    """Torch port of the heavy unsaturated soil-water solve."""
    h2osoi_liq, batch_shape = _flatten_last(state.h2osoi_liq)
    h2osoi_ice, _ = _flatten_last(state.h2osoi_ice)
    t_soisno, _ = _flatten_last(state.t_soisno)
    qin_top, _ = _flatten_vec(qflx_infl)
    qroot, _ = _flatten_last(qflx_rootsoi)
    zwt, _ = _flatten_vec(state.zwt)

    nlev = h2osoi_liq.shape[-1]
    nbatch = h2osoi_liq.shape[0]
    device = h2osoi_liq.device
    dtype = h2osoi_liq.dtype

    def expand_last(x: torch.Tensor) -> torch.Tensor:
        return torch.broadcast_to(x.to(device=device, dtype=dtype), batch_shape + (x.shape[-1],)).reshape(nbatch, -1)

    dz = expand_last(params.dz)
    z = expand_last(params.z)
    zi = expand_last(params.zi)
    watsat = expand_last(params.watsat)
    sucsat = expand_last(params.sucsat)
    bsw = expand_last(params.bsw)
    hksat = expand_last(params.hksat)
    smpmin = torch.broadcast_to(params.smpmin.to(device=device, dtype=dtype), batch_shape).reshape(nbatch)
    smp_floor = -torch.abs(smpmin)

    dzmm = dz * 1.0e3
    zmm = z * 1.0e3
    zimm = torch.cat([torch.zeros((nbatch, 1), device=device, dtype=dtype), zi * 1.0e3], dim=1)
    zwtmm = zwt * 1.0e3

    vol_ice = torch.minimum(watsat, h2osoi_ice / (dz * DENICE))
    icefrac = torch.minimum(torch.ones_like(watsat), vol_ice / torch.clamp(watsat, min=1.0e-12))
    vliq = torch.clamp(h2osoi_liq, min=1.0e-6) / (dz * DENH2O)
    h2osoi_vol = h2osoi_liq / (dz * DENH2O) + h2osoi_ice / (dz * DENICE)

    # Native ELM keeps below-column water tables distinct (jwt == nlev) so
    # the soil-water solve can couple the bottom soil layer to an aquifer row.
    jwt = torch.sum((zwt[:, None] > zi).to(torch.long), dim=1)

    vwc_zwt = watsat[:, -1].clone()
    rows = torch.arange(nbatch, device=device)
    jwt1 = torch.clamp(jwt, max=nlev - 1)
    frozen = t_soisno[rows, jwt1] < TFRZ
    # Vectorized frozen-soil correction for vwc_zwt (no if-any guard)
    _j = nlev - 1
    smp1_frz = HFUS * (TFRZ - t_soisno[:, _j]) / (GRAV * torch.clamp(t_soisno[:, _j], min=1.0)) * 1000.0
    smp1_frz = torch.maximum(sucsat[:, _j], smp1_frz)
    vwc_zwt_frz = watsat[:, _j] * torch.pow(smp1_frz / torch.clamp(sucsat[:, _j], min=1.0e-12), -1.0 / bsw[:, _j])
    vwc_zwt_frz = torch.minimum(vwc_zwt_frz, 0.5 * (watsat[:, _j] + h2osoi_vol[:, _j]))
    vwc_zwt = torch.where(frozen, vwc_zwt_frz, vwc_zwt)

    if workspace is None:
        vol_eq = torch.empty((nbatch, nlev + 1), device=device, dtype=dtype)
        zq = torch.empty((nbatch, nlev + 1), device=device, dtype=dtype)
        a = torch.empty((nbatch, nlev + 1), device=device, dtype=dtype)
        b = torch.empty((nbatch, nlev + 1), device=device, dtype=dtype)
        c = torch.empty((nbatch, nlev + 1), device=device, dtype=dtype)
        r = torch.empty((nbatch, nlev + 1), device=device, dtype=dtype)
        qin = torch.empty((nbatch, nlev + 1), device=device, dtype=dtype)
        qout = torch.empty((nbatch, nlev + 1), device=device, dtype=dtype)
    else:
        vol_eq = workspace.tensor("vol_eq", (nbatch, nlev + 1), device, dtype)
        zq = workspace.tensor("zq", (nbatch, nlev + 1), device, dtype)
        a = workspace.tensor("a", (nbatch, nlev + 1), device, dtype)
        b = workspace.tensor("b", (nbatch, nlev + 1), device, dtype)
        c = workspace.tensor("c", (nbatch, nlev + 1), device, dtype)
        r = workspace.tensor("r", (nbatch, nlev + 1), device, dtype)
        qin = workspace.tensor("qin", (nbatch, nlev + 1), device, dtype)
        qout = workspace.tensor("qout", (nbatch, nlev + 1), device, dtype)
    a.zero_()
    b.zero_()
    c.zero_()
    r.zero_()
    qin.zero_()
    qout.zero_()

    # --- Vectorized vol_eq computation over all levels at once (no Python loop) ---
    zwtmm_col = zwtmm[:, None]                          # (nbatch, 1)
    zim_lo    = zimm[:, :nlev]                          # (nbatch, nlev)
    zim_hi    = zimm[:, 1:nlev + 1]                     # (nbatch, nlev)
    dz_lay_c  = torch.clamp(zim_hi - zim_lo, min=1.0e-12)
    sat_mask_all = zwtmm_col <= zim_lo
    mid_mask_all = (zwtmm_col > zim_lo) & (zwtmm_col < zim_hi)
    exp_lvl      = 1.0 - 1.0 / bsw                     # (nbatch, nlev)
    denom_lvl    = torch.clamp(exp_lvl, min=1.0e-12)
    # mid branch — compute unconditionally, guarded by torch.where
    ratio_mid  = torch.clamp((sucsat + zwtmm_col - zim_lo) / torch.clamp(sucsat, min=1.0e-12), min=1.0e-12)
    temp0_mid  = torch.pow(ratio_mid, exp_lvl)
    dz_mid     = torch.clamp(zwtmm_col - zim_lo, min=1.0e-12)
    voleq1_mid = -sucsat * watsat / denom_lvl / dz_mid * (1.0 - temp0_mid)
    vol_eq_mid = (voleq1_mid * (zwtmm_col - zim_lo) + watsat * (zim_hi - zwtmm_col)) / dz_lay_c
    # deep branch — compute unconditionally, guarded by torch.where
    ratio_dlo  = torch.clamp((sucsat + zwtmm_col - zim_lo)  / torch.clamp(sucsat, min=1.0e-12), min=1.0e-12)
    ratio_dhi  = torch.clamp((sucsat + zwtmm_col - zim_hi)  / torch.clamp(sucsat, min=1.0e-12), min=1.0e-12)
    temp0_deep = torch.pow(ratio_dlo, exp_lvl)
    tempi_deep = torch.pow(ratio_dhi, exp_lvl)
    vol_eq_deep = -sucsat * watsat / denom_lvl / dz_lay_c * (tempi_deep - temp0_deep)
    vol_eq_all = torch.where(sat_mask_all, watsat,
                  torch.where(mid_mask_all, vol_eq_mid, vol_eq_deep))
    vol_eq_all = torch.clamp(vol_eq_all, min=0.0)
    vol_eq_all = torch.minimum(vol_eq_all, watsat)
    zq_all     = -sucsat * torch.pow(torch.clamp(vol_eq_all / torch.clamp(watsat, min=1.0e-12), min=0.01), -bsw)
    zq_all     = torch.maximum(zq_all, smp_floor[:, None])
    vol_eq[:, :nlev] = vol_eq_all
    zq[:, :nlev]     = zq_all

    # --- Below-column aquifer zq (vectorized, no if-any guard) ---
    below_column = jwt >= nlev
    zq[:, -1]    = zq[:, -2]
    _j           = nlev - 1
    delta_z_zwt  = torch.clamp(zwtmm - zimm[:, nlev], min=1.0)
    ratio_aq     = torch.clamp((sucsat[:, _j] + delta_z_zwt) / torch.clamp(sucsat[:, _j], min=1.0e-12), min=1.0e-12)
    temp0_aq     = torch.pow(ratio_aq, 1.0 - 1.0 / bsw[:, _j])
    vol_eq_aq    = (-sucsat[:, _j] * watsat[:, _j]
                    / torch.clamp(1.0 - 1.0 / bsw[:, _j], min=1.0e-12)
                    / delta_z_zwt * (1.0 - temp0_aq))
    vol_eq_aq    = torch.clamp(vol_eq_aq, min=0.0)
    vol_eq_aq    = torch.minimum(vol_eq_aq, watsat[:, _j])
    zq_aq        = -sucsat[:, _j] * torch.pow(torch.clamp(vol_eq_aq / torch.clamp(watsat[:, _j], min=1.0e-12), min=0.01), -bsw[:, _j])
    zq_aq        = torch.maximum(zq_aq, smp_floor)
    zq[:, -1]    = torch.where(below_column, zq_aq, zq[:, -1])

    smp, dsmpdw = _clapp_hornberger_smp_torch(vliq, watsat, sucsat, bsw, smpmin)
    hk, dhkdw = _clapp_hornberger_hk_torch(vliq, watsat, hksat, bsw, icefrac)

    qin[:, 0] = qin_top
    den = torch.clamp(zmm[:, 1] - zmm[:, 0], min=1.0e-12)
    num = (smp[:, 1] - smp[:, 0]) - (zq[:, 1] - zq[:, 0])
    qout[:, 0] = -hk[:, 0] * num / den
    dqodw1 = -(-hk[:, 0] * dsmpdw[:, 0] + num * dhkdw[:, 0]) / den
    dqodw2 = -(hk[:, 0] * dsmpdw[:, 1] + num * dhkdw[:, 0]) / den
    r[:, 0] = qin[:, 0] - qout[:, 0] - qroot[:, 0]
    b[:, 0] = dzmm[:, 0] / dt + dqodw1
    c[:, 0] = dqodw2

    # --- Vectorized tridiagonal build for interior layers [1, nlev-2] ---
    if nlev > 2:
        _js  = slice(1, nlev - 1)     # j
        _jm1 = slice(0, nlev - 2)     # j-1
        _jp1 = slice(2, nlev)         # j+1
        _den_in  = torch.clamp(zmm[:, _js] - zmm[:, _jm1], min=1.0e-12)
        _num_in  = (smp[:, _js] - smp[:, _jm1]) - (zq[:, _js] - zq[:, _jm1])
        _qin_j   = -hk[:, _jm1] * _num_in / _den_in
        _dqidw0  = -(-hk[:, _jm1] * dsmpdw[:, _jm1] + _num_in * dhkdw[:, _jm1]) / _den_in
        _dqidw1  = -(hk[:, _jm1] * dsmpdw[:, _js]  + _num_in * dhkdw[:, _jm1]) / _den_in
        _den_out = torch.clamp(zmm[:, _jp1] - zmm[:, _js], min=1.0e-12)
        _num_out = (smp[:, _jp1] - smp[:, _js]) - (zq[:, _jp1] - zq[:, _js])
        _qout_j  = -hk[:, _js] * _num_out / _den_out
        _dqodw1  = -(-hk[:, _js] * dsmpdw[:, _js]  + _num_out * dhkdw[:, _js]) / _den_out
        _dqodw2  = -(hk[:, _js] * dsmpdw[:, _jp1]  + _num_out * dhkdw[:, _js]) / _den_out
        qin[:, _js]  = _qin_j
        qout[:, _js] = _qout_j
        r[:, _js]    = _qin_j - _qout_j - qroot[:, _js]
        a[:, _js]    = -_dqidw0
        b[:, _js]    = dzmm[:, _js] / dt - _dqidw1 + _dqodw1
        c[:, _js]    = _dqodw2

    j = nlev - 1
    s_node_aq = torch.clamp(0.5 * (vwc_zwt + vliq[:, j]) / torch.clamp(watsat[:, j], min=1.0e-12), min=0.01, max=1.0)
    smp_aq = -sucsat[:, j] * torch.pow(s_node_aq, -bsw[:, j])
    smp_aq = torch.maximum(smp_aq, smp_floor)
    dsmpdw_aq = -bsw[:, j] * smp_aq / torch.clamp(s_node_aq * watsat[:, j], min=1.0e-12)
    s1_bottom = torch.clamp(vliq[:, j] / torch.clamp(watsat[:, j], min=1.0e-12), min=0.0, max=1.0)
    s2_bottom = hksat[:, j] * torch.pow(s1_bottom, 2.0 * bsw[:, j] + 2.0)
    imped_bottom = torch.pow(torch.full_like(s1_bottom, 10.0), -E_ICE * icefrac[:, j])
    hk_bottom = imped_bottom * s1_bottom * s2_bottom
    dhkdw_bottom = (
        imped_bottom
        * (2.0 * bsw[:, j] + 3.0)
        * s2_bottom
        / torch.clamp(2.0 * watsat[:, j], min=1.0e-12)
    )
    zmm_aq = 0.5 * (zwtmm + zmm[:, j])
    dzmm_aq = torch.clamp(zwtmm - zmm[:, j], min=1.0e-12)
    den_bottom = torch.clamp(zmm_aq - zmm[:, j], min=1.0e-12)
    num_bottom = (smp_aq - smp[:, j]) - (zq[:, nlev] - zq[:, j])
    qout_bottom = -hk_bottom * num_bottom / den_bottom
    dqbot_dw_soil = -(-hk_bottom * dsmpdw[:, j] + num_bottom * dhkdw_bottom) / den_bottom
    dqbot_dw_aq = -(hk_bottom * dsmpdw_aq + num_bottom * dhkdw_bottom) / den_bottom
    qout_bottom = torch.where(below_column, qout_bottom, torch.zeros_like(qout_bottom))
    dqbot_dw_soil = torch.where(below_column, dqbot_dw_soil, torch.zeros_like(dqbot_dw_soil))
    dqbot_dw_aq = torch.where(below_column, dqbot_dw_aq, torch.zeros_like(dqbot_dw_aq))

    den_in = torch.clamp(zmm[:, j] - zmm[:, j - 1], min=1.0e-12)
    num_in = (smp[:, j] - smp[:, j - 1]) - (zq[:, j] - zq[:, j - 1])
    qin[:, j] = -hk[:, j - 1] * num_in / den_in
    dqidw0 = -(-hk[:, j - 1] * dsmpdw[:, j - 1] + num_in * dhkdw[:, j - 1]) / den_in
    dqidw1 = -(hk[:, j - 1] * dsmpdw[:, j] + num_in * dhkdw[:, j - 1]) / den_in
    qout[:, j] = qout_bottom
    r[:, j] = qin[:, j] - qout[:, j] - qroot[:, j]
    a[:, j] = -dqidw0
    b[:, j] = dzmm[:, j] / dt - dqidw1 + dqbot_dw_soil
    c[:, j] = dqbot_dw_aq
    qin[:, nlev] = qout_bottom
    r[:, nlev] = qin[:, nlev]
    a[:, nlev] = torch.where(below_column, -dqbot_dw_soil, torch.zeros_like(dqbot_dw_soil))
    b[:, nlev] = torch.where(below_column, dzmm_aq / dt - dqbot_dw_aq, dzmm[:, j] / dt)

    dwat = batched_tridiagonal_torch(a, b, c, r, workspace=workspace)
    h2osoi_liq_new = h2osoi_liq + dwat[:, :nlev] * dzmm

    qcharge = torch.zeros(nbatch, device=device, dtype=dtype)
    in_column = jwt < nlev
    # Vectorized qcharge for in-column water table (no if-any guard)
    jj_col    = torch.clamp(jwt, max=nlev - 1)           # (nbatch,)
    above_col = torch.clamp(jwt - 1, min=0)              # (nbatch,)
    s_node_c  = torch.clamp(h2osoi_vol[rows, jj_col] / torch.clamp(watsat[rows, jj_col], min=1.0e-12), min=0.01, max=1.0)
    imped_c   = torch.pow(torch.full_like(s_node_c, 10.0), -E_ICE * icefrac[rows, jj_col])
    ka_c      = imped_c * hksat[rows, jj_col] * torch.pow(s_node_c, 2.0 * bsw[rows, jj_col] + 3.0)
    smp1_c    = torch.maximum(smp_floor, smp[rows, above_col])
    wh_c      = smp1_c - zq[rows, above_col]
    dist_c    = torch.where(
        jwt == 0,
        (zwt + 1.0e-3) * 1000.0,
        torch.clamp((zwt - z[rows, above_col]) * 1000.0 * 2.0, min=1.0e-6),
    )
    qcharge_in = torch.clamp(-ka_c * (0.0 - wh_c) / dist_c, min=-10.0 / dt, max=10.0 / dt)
    qcharge    = torch.where(in_column, qcharge_in, qcharge)
    # Vectorized below-column qcharge
    qcharge    = torch.where(below_column, dwat[:, nlev] * dzmm_aq / dt, qcharge)

    h2osoi_vol_new = h2osoi_liq_new / (dz * DENH2O) + h2osoi_ice / (dz * DENICE)
    dW1dt_layer1 = (h2osoi_liq_new[:, 0] - h2osoi_liq[:, 0]) / dt
    vliq_new = torch.clamp(h2osoi_liq_new, min=1.0e-6) / (dz * DENH2O)
    smp_l, _ = _clapp_hornberger_smp_torch(
        vliq_new,
        watsat,
        sucsat,
        bsw,
        smpmin,
    )
    hk_l, _ = _clapp_hornberger_hk_torch(
        vliq_new,
        watsat,
        hksat,
        bsw,
        torch.minimum(
            torch.ones_like(watsat),
            torch.minimum(watsat, h2osoi_ice / (dz * DENICE)) / torch.clamp(watsat, min=1.0e-12),
        ),
    )
    hk_l = torch.cat([hk_l, hk_l[:, -1:]], dim=1)[:, :nlev]

    return SoilWaterCoreResult(
        h2osoi_liq=_restore_last(h2osoi_liq_new, batch_shape),
        h2osoi_vol=_restore_last(h2osoi_vol_new, batch_shape),
        qcharge=_restore_vec(qcharge, batch_shape),
        smp_l=_restore_last(smp_l, batch_shape),
        hk_l=_restore_last(hk_l, batch_shape),
        qin_layer1=_restore_vec(qin[:, 0], batch_shape),
        qout_layer1=_restore_vec(qout[:, 0], batch_shape),
        qroot_layer1=_restore_vec(qroot[:, 0], batch_shape),
        dW1dt_layer1=_restore_vec(dW1dt_layer1, batch_shape),
    )


def hydrology_no_drainage_torch(
    state: SoilHydrologyNoDrainageState,
    params: SoilWaterCoreParameters,
    qflx_top_soil: torch.Tensor,
    qflx_evap_grnd: torch.Tensor,
    qflx_ev_soil: torch.Tensor,
    qflx_ev_h2osfc: torch.Tensor,
    frac_h2osfc: torch.Tensor,
    frac_sno: torch.Tensor,
    qflx_rootsoi: torch.Tensor,
    dt: float,
    workspace: SoilHydrologyTorchWorkspace | None = None,
) -> SoilHydrologyTorchResult:
    h2osoi_liq, batch_shape = _flatten_last(state.h2osoi_liq)
    h2osoi_ice, _ = _flatten_last(state.h2osoi_ice)
    t_soisno, _ = _flatten_last(state.t_soisno)
    h2osfc, _ = _flatten_vec(state.h2osfc)
    zwt, _ = _flatten_vec(state.zwt)
    qflx_top_soil_f, _ = _flatten_vec(qflx_top_soil)
    qflx_evap_grnd_f, _ = _flatten_vec(qflx_evap_grnd)
    qflx_ev_soil_f, _ = _flatten_vec(qflx_ev_soil)
    qflx_ev_h2osfc_f, _ = _flatten_vec(qflx_ev_h2osfc)
    frac_h2osfc_f, _ = _flatten_vec(frac_h2osfc)
    frac_sno_f, _ = _flatten_vec(frac_sno)
    qroot, _ = _flatten_last(qflx_rootsoi)

    device = h2osoi_liq.device
    dtype = h2osoi_liq.dtype
    dz = _expand_last(params.dz, batch_shape, device, dtype)
    watsat = _expand_last(params.watsat, batch_shape, device, dtype)
    hksat = _expand_last(params.hksat, batch_shape, device, dtype)
    wtfact = _expand_vec(params.wtfact, batch_shape, device, dtype)
    fover = _expand_vec(params.fover, batch_shape, device, dtype)
    fdrain = _expand_vec(
        params.fdrain if params.fdrain is not None else torch.full_like(params.fover, 2.5),
        batch_shape,
        device,
        dtype,
    )
    h2osfc_thresh = _expand_vec(params.h2osfc_thresh, batch_shape, device, dtype)
    pc = _expand_vec(params.pc, batch_shape, device, dtype)
    topo_slope_exp = (
        _expand_vec(params.topo_slope, batch_shape, device, dtype)
        if params.topo_slope is not None else None
    )

    qflx_surf, fsat, _ = surface_runoff_torch(
        h2osoi_liq=h2osoi_liq,
        h2osoi_ice=h2osoi_ice,
        zwt=zwt,
        qflx_top_soil=qflx_top_soil_f,
        dz=dz,
        watsat=watsat,
        wtfact=wtfact,
        fover=fover,
    )
    infil = infiltration_torch(
        h2osoi_liq=h2osoi_liq,
        h2osoi_ice=h2osoi_ice,
        h2osfc=h2osfc,
        frac_h2osfc=frac_h2osfc_f,
        frac_sno=frac_sno_f,
        t_soisno=t_soisno,
        qflx_top_soil=qflx_top_soil_f,
        qflx_evap_grnd=qflx_evap_grnd_f,
        qflx_ev_soil=qflx_ev_soil_f,
        qflx_ev_h2osfc=qflx_ev_h2osfc_f,
        qflx_surf=qflx_surf,
        fsat=fsat,
        dz=dz,
        watsat=watsat,
        hksat=hksat,
        h2osfc_thresh=h2osfc_thresh,
        pc=pc,
        topo_slope=topo_slope_exp,
        dt=dt,
    )
    core = soil_water_zengdecker2009_torch(
        SoilWaterCoreState(
            h2osoi_liq=state.h2osoi_liq,
            h2osoi_ice=state.h2osoi_ice,
            t_soisno=state.t_soisno,
            zwt=state.zwt,
        ),
        params,
        qflx_infl=_restore_vec(infil["qflx_infl"], batch_shape),
        qflx_rootsoi=qflx_rootsoi,
        dt=dt,
        workspace=workspace,
    )

    # --- post-process: mirrors hydrology_no_drainage_step in elm_hydrology.py ---
    zwt_after_qcharge, _ = water_table_torch(
        zwt=state.zwt,
        qcharge=core.qcharge,
        z=params.z,
        zi=params.zi,
        watsat=params.watsat,
        sucsat=params.sucsat,
        bsw=params.bsw,
        dt=dt,
        wa=torch.zeros_like(state.zwt),
    )

    # qdeep: consume positive qcharge as bottom-layer baseflow drain
    hliq_flat, _ = _flatten_last(core.h2osoi_liq)
    hsurf_flat = infil["h2osfc"]          # (nbatch,) flat
    effp_flat = infil["eff_porosity"]     # (nbatch, nlev) flat

    qdeep = torch.clamp(core.qcharge.reshape(-1), min=0.0)
    bottom_available = torch.clamp(hliq_flat[:, -1], min=0.0) / dt
    qdeep = torch.minimum(qdeep, bottom_available)
    hliq_flat = hliq_flat.clone()
    hliq_flat[:, -1] = torch.clamp(hliq_flat[:, -1] - qdeep * dt, min=0.0)

    # rsub_top: exponential topographic baseflow — mirrors elm_hydrology.py's
    # drainage().  Removes min(rsub_top_mm, total_available) from the column,
    # draining layers bottom-to-top (loop over nlev ≤ 15 iterations).
    # origflag=0: native reads the global cap from clm_params.nc and uses
    # 10 mm/s when it is absent.
    rsub_top_globalmax = _expand_vec(
        params.rsub_top_globalmax
        if params.rsub_top_globalmax is not None
        else torch.full_like(state.zwt, 10.0),
        batch_shape,
        device,
        dtype,
    )
    if params.topo_slope is not None:
        topo_slope_f = _expand_vec(params.topo_slope, batch_shape, device, dtype)
        rsub_top_max = torch.minimum(
            10.0 * torch.sin(torch.deg2rad(topo_slope_f)),
            rsub_top_globalmax,
        )
    else:
        rsub_top_max = torch.zeros((h2osoi_liq.shape[0],), device=device, dtype=dtype)
    zwt_flat, _ = _flatten_vec(zwt_after_qcharge)
    zi_full = _expand_last(params.zi, batch_shape, device, dtype)
    icefrac = torch.clamp(
        torch.minimum(watsat, h2osoi_ice / (dz * DENICE))
        / torch.clamp(watsat, min=1.0e-12),
        max=1.0,
    )
    jwt = (zwt_flat[:, None] > zi_full).long().sum(dim=1)
    ice_start = torch.clamp(jwt - 1, min=0)
    active_ice = torch.arange(h2osoi_liq.shape[-1], device=device)[None, :] >= ice_start[:, None]
    ice_dz = torch.where(active_ice, dz, torch.zeros_like(dz))
    imped = torch.pow(
        torch.full_like(zwt_flat, 10.0),
        -E_ICE * torch.sum(icefrac * ice_dz, dim=1)
        / torch.clamp(torch.sum(ice_dz, dim=1), min=1.0e-12),
    )
    rsub_top_rate = imped * rsub_top_max * torch.exp(-fdrain * zwt_flat)
    # Vectorized bottom-to-top drain: distribute rsub_top budget across
    # layers starting from the bottom, without a sequential Python loop.
    available = torch.clamp(hliq_flat - 1.0e-8, min=0.0)          # (nbatch, nlev)
    # cumavail[j] = sum of available water in layers j … nlev-1
    cumavail = torch.flip(
        torch.cumsum(torch.flip(available, dims=[-1]), dim=-1), dims=[-1]
    )
    actual_drain = torch.minimum(rsub_top_rate * dt, cumavail[:, 0])
    # For each layer: drain what's available after layers below are exhausted
    cum_below = torch.cat(
        [cumavail[:, 1:],
         torch.zeros(hliq_flat.shape[0], 1, device=hliq_flat.device, dtype=hliq_flat.dtype)],
        dim=-1,
    )
    layer_drain_v = torch.clamp(
        torch.minimum(available, torch.clamp(actual_drain[:, None] - cum_below, min=0.0)),
        min=0.0,
    )
    hliq_flat = hliq_flat - layer_drain_v
    # Update zwt: draining actual_drain mm deepens the water table
    watsat_bottom = _expand_last(params.watsat, batch_shape, device, dtype)[:, -1]
    rous = torch.clamp(watsat_bottom, min=0.02)
    zwt_flat = zwt_flat + actual_drain / (1.0e3 * rous)
    qdeep = qdeep + actual_drain / dt     # combine into reported deep drain flux

    hliq_flat, hsurf_flat = _redistribute_saturation_excess_torch(
        hliq_flat, h2osoi_ice, effp_flat, dz, hsurf_flat
    )
    dW1dt_layer1 = (hliq_flat[:, 0] - h2osoi_liq[:, 0]) / dt
    hliq_repaired, hsurf_repaired = _repair_liquid_water_floor_torch(hliq_flat, hsurf_flat, dt)

    h2osoi_liq_out = _restore_last(hliq_repaired, batch_shape)
    h2osfc_out = _restore_vec(hsurf_repaired, batch_shape)
    dz_broad = torch.broadcast_to(
        params.dz.to(device=device, dtype=dtype),
        batch_shape + (params.dz.shape[-1],),
    )
    h2osoi_vol_out = (
        h2osoi_liq_out / (dz_broad * DENH2O)
        + state.h2osoi_ice.to(device=device, dtype=dtype) / (dz_broad * DENICE)
    )

    return SoilHydrologyTorchResult(
        h2osoi_liq=h2osoi_liq_out,
        h2osoi_vol=h2osoi_vol_out,
        h2osfc=h2osfc_out,
        qflx_surf=_restore_vec(qflx_surf, batch_shape),
        qflx_infl=_restore_vec(infil["qflx_infl"], batch_shape),
        qflx_soil_evap=_restore_vec(infil["qflx_soil_evap"], batch_shape),
        qflx_h2osfc_surf=_restore_vec(infil["qflx_h2osfc_surf"], batch_shape),
        qflx_h2osfc_drain=_restore_vec(infil["qflx_h2osfc_drain"], batch_shape),
        qflx_infl_excess=_restore_vec(infil["qflx_infl_excess"], batch_shape),
        qcharge=core.qcharge,
        smp_l=core.smp_l,
        hk_l=core.hk_l,
        qflx_deep_drain=_restore_vec(qdeep, batch_shape),
        eff_porosity=_restore_last(effp_flat, batch_shape),
        zwt=_restore_vec(torch.clamp(zwt_flat, min=0.0, max=80.0), batch_shape),
        zwt_after_qcharge=zwt_after_qcharge,
        zwt_after_drainage=_restore_vec(torch.clamp(zwt_flat, min=0.0, max=80.0), batch_shape),
        qin_layer1=core.qin_layer1,
        qout_layer1=core.qout_layer1,
        qroot_layer1=core.qroot_layer1,
        dW1dt_layer1=_restore_vec(dW1dt_layer1, batch_shape),
    )


def water_table_torch(
    zwt: torch.Tensor,
    qcharge: torch.Tensor,
    z: torch.Tensor,
    zi: torch.Tensor,
    watsat: torch.Tensor,
    sucsat: torch.Tensor,
    bsw: torch.Tensor,
    dt: float,
    wa: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Vectorized water-table update driven by qcharge.

    Uses a fixed nlev-level scan so the function is fully vectorized and
    torch.compile-friendly (no per-element Python loops or .item() calls).
    For typical model time steps (dt ~ 1800 s) qcharge * dt is far smaller
    than a single layer thickness, so the one-sweep result is exact; for
    larger charges the scan through all nlev levels provides multi-layer
    accuracy.
    """
    batch_shape = tuple(zwt.shape)
    zwt_f, _ = _flatten_vec(zwt)
    zwt_f = zwt_f.clone()
    qcharge_f, _ = _flatten_vec(qcharge)
    nbatch = zwt_f.shape[0]
    device = zwt_f.device
    dtype = zwt_f.dtype

    if wa is None:
        wa_f = torch.zeros_like(zwt_f)
    else:
        wa_f, _ = _flatten_vec(wa)
        wa_f = wa_f.clone()

    zf   = _expand_last(z,      batch_shape, device, dtype)
    zif  = _expand_last(zi,     batch_shape, device, dtype)
    wsat = _expand_last(watsat, batch_shape, device, dtype)
    ssat = _expand_last(sucsat, batch_shape, device, dtype)
    bswf = _expand_last(bsw,    batch_shape, device, dtype)
    nlev = zf.shape[1]

    # Native ELM keeps below-column water tables distinct (jwt == nlev)
    # instead of aliasing them to the bottom soil layer.
    jwt = (zwt_f[:, None] > zif).long().sum(dim=1)

    rous_last = wsat[:, -1] * (1.0 - (1.0 + 1.0e3 * zwt_f / torch.clamp(ssat[:, -1], min=1.0e-12)) ** (-1.0 / bswf[:, -1]))
    rous_last = torch.clamp(rous_last, min=0.02)

    below   = jwt >= nlev
    qtot    = qcharge_f * dt      # mm equivalent

    # --- below-column: aquifer update (vectorized) ---
    wa_f  = wa_f  + torch.where(below, qtot, torch.zeros_like(qtot))
    zwt_f = zwt_f - torch.where(below, qtot / 1000.0 / rous_last, torch.zeros_like(qtot))

    # --- in-column: multi-level scan (fixed nlev iterations, fully vectorized) ---
    # Rising water table (qtot > 0): scan upward from jwt to 0
    rising  = qtot > 0.0
    falling = qtot < 0.0
    qtot_r  = torch.where(rising  & ~below, qtot,  torch.zeros_like(qtot))
    qtot_f  = torch.where(falling & ~below, qtot,  torch.zeros_like(qtot))

    for j_abs in range(nlev - 1, -1, -1):   # shallow → deep scan for rising
        active = rising & ~below & (jwt >= j_abs) & (qtot_r > 0.0)
        ztop_j = zif[:, j_abs - 1] if j_abs > 0 else torch.zeros(nbatch, device=device, dtype=dtype)
        sy_j   = wsat[:, j_abs] * (1.0 - (1.0 + 1.0e3 * zwt_f / torch.clamp(ssat[:, j_abs], min=1.0e-12)) ** (-1.0 / bswf[:, j_abs]))
        sy_j   = torch.clamp(sy_j, min=0.02)
        cap_j  = torch.clamp(sy_j * (zwt_f - ztop_j) * 1.0e3, min=0.0)
        abs_j  = torch.where(active, torch.minimum(qtot_r, cap_j), torch.zeros_like(qtot_r))
        zwt_f  = zwt_f - abs_j / sy_j / 1.0e3
        qtot_r = qtot_r - abs_j

    # Leftover rising charge goes into aquifer
    wa_f  = wa_f  + torch.where(rising & ~below & (qtot_r > 0.0), qtot_r, torch.zeros_like(qtot_r))
    zwt_f = zwt_f - torch.where(rising & ~below & (qtot_r > 0.0), qtot_r / 1000.0 / rous_last, torch.zeros_like(qtot_r))

    for j_abs in range(nlev):               # shallow → deep scan for falling
        # Native's 1-based loop is j=jwt+1,nlevbed.  With zero-based layer
        # indices, the first layer is therefore j_abs == jwt (the layer that
        # currently contains the water table), not jwt+1.
        active = falling & ~below & (jwt <= j_abs) & (qtot_f < 0.0)
        sy_j   = wsat[:, j_abs] * (1.0 - (1.0 + 1.0e3 * zwt_f / torch.clamp(ssat[:, j_abs], min=1.0e-12)) ** (-1.0 / bswf[:, j_abs]))
        sy_j   = torch.clamp(sy_j, min=0.02)
        layer_cap = -torch.clamp(sy_j * (zif[:, j_abs] - zwt_f) * 1.0e3, min=0.0)  # negative
        drain_j   = torch.where(active, torch.maximum(qtot_f, layer_cap), torch.zeros_like(qtot_f))
        # If fully drained through this layer, move zwt to layer bottom.
        at_bottom = active & (drain_j <= layer_cap)
        zwt_f   = torch.where(at_bottom, zif[:, j_abs],
                    zwt_f - torch.where(active & ~at_bottom, drain_j / sy_j / 1.0e3, torch.zeros_like(drain_j)))
        qtot_f  = qtot_f - drain_j

    return _restore_vec(torch.clamp(zwt_f, min=0.0, max=80.0), batch_shape), _restore_vec(wa_f, batch_shape)


def drainage_torch(
    h2osoi_liq: torch.Tensor,
    h2osoi_ice: torch.Tensor,
    h2osfc: torch.Tensor,
    zwt: torch.Tensor,
    z: torch.Tensor,
    zi: torch.Tensor,
    dz: torch.Tensor,
    watsat: torch.Tensor,
    sucsat: torch.Tensor,
    bsw: torch.Tensor,
    eff_porosity: torch.Tensor,
    fdrain: torch.Tensor | None,
    topo_slope: torch.Tensor | None,
    dt: float,
    rsub_top_globalmax: torch.Tensor | None = None,
    wa: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Torch port of the narrowed default drainage path."""
    hliq, batch_shape = _flatten_last(h2osoi_liq)
    hliq = hliq.clone()
    hice, _ = _flatten_last(h2osoi_ice)
    hsurf, _ = _flatten_vec(h2osfc)
    hsurf = hsurf.clone()
    zwt_f, _ = _flatten_vec(zwt)
    zwt_f = zwt_f.clone()
    nbatch = hliq.shape[0]
    device = hliq.device
    dtype = hliq.dtype
    if wa is None:
        wa_f = torch.zeros_like(zwt_f)
    else:
        wa_f, _ = _flatten_vec(wa)
        wa_f = wa_f.clone()

    zf = _expand_last(z, batch_shape, device, dtype)
    zif = _expand_last(zi, batch_shape, device, dtype)
    dzf = _expand_last(dz, batch_shape, device, dtype)
    watsatf = _expand_last(watsat, batch_shape, device, dtype)
    sucsatf = _expand_last(sucsat, batch_shape, device, dtype)
    bswf = _expand_last(bsw, batch_shape, device, dtype)
    effp, _ = _flatten_last(eff_porosity)
    if fdrain is None:
        fdrain = torch.full_like(zwt, 2.5)
    fdrain_f = _expand_vec(fdrain, batch_shape, device, dtype)
    nlev = hliq.shape[1]
    dzmm = dzf * 1.0e3

    jwt = (zwt_f[:, None] > zif).long().sum(dim=1)

    topo_slope_f = (_expand_vec(topo_slope, batch_shape, device, dtype)
                    if topo_slope is not None else torch.zeros_like(zwt_f))
    globalmax_f = (
        _expand_vec(rsub_top_globalmax, batch_shape, device, dtype)
        if rsub_top_globalmax is not None else torch.full_like(zwt_f, 10.0)
    )
    rsub_top_max = torch.minimum(10.0 * torch.sin(torch.deg2rad(topo_slope_f)), globalmax_f)
    vol_ice = torch.minimum(watsatf, hice / (dzf * DENICE))
    icefrac = torch.clamp(vol_ice / torch.clamp(watsatf, min=1.0e-12), max=1.0)
    ice_start = torch.clamp(jwt - 1, min=0)
    active_ice = torch.arange(nlev, device=device)[None, :] >= ice_start[:, None]
    ice_dz = torch.where(active_ice, dzmm, torch.zeros_like(dzmm))
    imped = torch.pow(
        torch.full_like(zwt_f, 10.0),
        -E_ICE * torch.sum(icefrac * ice_dz, dim=1)
        / torch.clamp(torch.sum(ice_dz, dim=1), min=1.0e-12),
    )
    rsub_top = imped * rsub_top_max * torch.exp(-fdrain_f * zwt_f)
    rsub_top_tot = -rsub_top * dt

    rous = watsatf[:, -1] * (1.0 - (1.0 + 1.0e3 * zwt_f / sucsatf[:, -1]) ** (-1.0 / bswf[:, -1]))
    rous = torch.clamp(rous, min=0.02)

    # rsub_top_tot is always <= 0 (drainage), so this is the falling-table case.
    # Vectorized scan over nlev levels: drain each layer from jwt+1 downward.
    below_col = zwt_f > zif[:, -1]
    # jwt == nlev-1 means the table is inside the bottom soil layer.  Native
    # ELM only takes the direct-aquifer branch for jwt == nlev (strictly below
    # the active soil column).
    direct_aquifer = below_col
    wa_f   = wa_f   + torch.where(direct_aquifer, rsub_top_tot,             torch.zeros_like(rsub_top_tot))
    zwt_f  = zwt_f  - torch.where(direct_aquifer, rsub_top_tot / 1000.0 / rous, torch.zeros_like(rsub_top_tot))
    rem    = torch.where(~direct_aquifer, rsub_top_tot, torch.zeros_like(rsub_top_tot))  # <= 0

    for j_abs in range(nlev):
        # Match ELM Drainage's j=jwt+1,nlevbed loop after converting from its
        # 1-based layer number to our zero-based index.
        active = ~direct_aquifer & (jwt <= j_abs) & (rem < 0.0)
        sy_j   = watsatf[:, j_abs] * (1.0 - (1.0 + 1.0e3 * zwt_f / torch.clamp(sucsatf[:, j_abs], min=1.0e-12)) ** (-1.0 / bswf[:, j_abs]))
        sy_j   = torch.clamp(sy_j, min=0.02)
        layer_cap = -torch.clamp(sy_j * (zif[:, j_abs] - zwt_f) * 1.0e3, min=0.0)   # <= 0
        drain_j   = torch.where(active, torch.maximum(rem, layer_cap), torch.zeros_like(rem))
        hliq[:, j_abs] = hliq[:, j_abs] + drain_j
        rem    = rem - drain_j
        # If fully drained through this layer, move zwt to layer bottom; otherwise update in-layer
        at_bottom = active & (drain_j <= layer_cap)
        zwt_f  = torch.where(at_bottom, zif[:, j_abs],
                    zwt_f - torch.where(active & ~at_bottom, drain_j / sy_j / 1.0e3, torch.zeros_like(drain_j)))

    # Residual that couldn't fit in soil column goes into aquifer
    wa_f  = wa_f  + torch.where(~direct_aquifer & (rem < 0.0), rem, torch.zeros_like(rem))
    zwt_f = zwt_f - torch.where(~direct_aquifer & (rem < 0.0), rem / 1000.0 / rous, torch.zeros_like(rem))
    # Keep the full requested flux: native counts residual drainage supplied
    # by the aquifer as drainage too.

    qflx_rsub_sat = torch.zeros_like(zwt_f)
    # Match the NumPy drainage path's bottom-to-top saturation-excess cascade.
    for j_abs in range(nlev - 1, 0, -1):
        xsi = torch.clamp(hliq[:, j_abs] - effp[:, j_abs] * dzmm[:, j_abs], min=0.0)
        hliq[:, j_abs] = hliq[:, j_abs] - xsi
        hliq[:, j_abs - 1] = hliq[:, j_abs - 1] + xsi

    xs1 = torch.clamp(
        torch.clamp(hliq[:, 0], min=0.0)
        - torch.clamp(watsatf[:, 0] * dzmm[:, 0] - hice[:, 0], min=0.0),
        min=0.0,
    )
    hliq[:, 0] = hliq[:, 0] - xs1
    hsurf = hsurf + xs1

    watmin = 1.0e-8
    # Vectorized watmin floor: borrow deficit from next layer in one pass.
    _xs = torch.clamp(watmin - hliq[:, :-1], min=0.0)
    hliq[:, :-1] = hliq[:, :-1] + _xs
    hliq[:, 1:]  = hliq[:, 1:]  - _xs

    xs = torch.clamp(watmin - hliq[:, -1], min=0.0)
    hliq[:, -1] = hliq[:, -1] + xs
    rsub_top = rsub_top - xs / dt
    qflx_drain = qflx_rsub_sat + rsub_top

    return (
        _restore_last(hliq, batch_shape),
        _restore_vec(hsurf, batch_shape),
        _restore_vec(torch.clamp(zwt_f, min=0.0, max=80.0), batch_shape),
        _restore_vec(qflx_drain, batch_shape),
        _restore_vec(wa_f, batch_shape),
    )


def hydrology_torch(
    state: SoilHydrologyNoDrainageState,
    params: SoilWaterCoreParameters,
    qflx_top_soil: torch.Tensor,
    qflx_evap_grnd: torch.Tensor,
    qflx_ev_soil: torch.Tensor,
    qflx_ev_h2osfc: torch.Tensor,
    frac_h2osfc: torch.Tensor,
    frac_sno: torch.Tensor,
    qflx_rootsoi: torch.Tensor,
    dt: float,
    wa: torch.Tensor | None = None,
    workspace: SoilHydrologyTorchWorkspace | None = None,
    profile_sections: dict[str, float] | None = None,
) -> SoilHydrologyTorchResult:
    """Full narrowed hydrology path on device."""
    h2osoi_liq, batch_shape = _flatten_last(state.h2osoi_liq)
    h2osoi_ice, _ = _flatten_last(state.h2osoi_ice)
    t_soisno, _ = _flatten_last(state.t_soisno)
    h2osfc, _ = _flatten_vec(state.h2osfc)
    zwt, _ = _flatten_vec(state.zwt)
    qflx_top_soil_f, _ = _flatten_vec(qflx_top_soil)
    qflx_evap_grnd_f, _ = _flatten_vec(qflx_evap_grnd)
    qflx_ev_soil_f, _ = _flatten_vec(qflx_ev_soil)
    qflx_ev_h2osfc_f, _ = _flatten_vec(qflx_ev_h2osfc)
    frac_h2osfc_f, _ = _flatten_vec(frac_h2osfc)
    frac_sno_f, _ = _flatten_vec(frac_sno)

    device = h2osoi_liq.device
    dtype = h2osoi_liq.dtype
    dz = _expand_last(params.dz, batch_shape, device, dtype)
    watsat = _expand_last(params.watsat, batch_shape, device, dtype)
    hksat = _expand_last(params.hksat, batch_shape, device, dtype)
    wtfact = _expand_vec(params.wtfact, batch_shape, device, dtype)
    fover = _expand_vec(params.fover, batch_shape, device, dtype)
    h2osfc_thresh = _expand_vec(params.h2osfc_thresh, batch_shape, device, dtype)
    pc = _expand_vec(params.pc, batch_shape, device, dtype)
    topo_slope_exp = (
        _expand_vec(params.topo_slope, batch_shape, device, dtype)
        if params.topo_slope is not None else None
    )

    qflx_surf, fsat, _ = _profiled(
        profile_sections,
        "hydrology_full_surface_runoff",
        device,
        lambda: surface_runoff_torch(
            h2osoi_liq=h2osoi_liq,
            h2osoi_ice=h2osoi_ice,
            zwt=zwt,
            qflx_top_soil=qflx_top_soil_f,
            dz=dz,
            watsat=watsat,
            wtfact=wtfact,
            fover=fover,
        ),
    )
    infil = _profiled(
        profile_sections,
        "hydrology_full_infiltration",
        device,
        lambda: infiltration_torch(
            h2osoi_liq=h2osoi_liq,
            h2osoi_ice=h2osoi_ice,
            h2osfc=h2osfc,
            frac_h2osfc=frac_h2osfc_f,
            frac_sno=frac_sno_f,
            t_soisno=t_soisno,
            qflx_top_soil=qflx_top_soil_f,
            qflx_evap_grnd=qflx_evap_grnd_f,
            qflx_ev_soil=qflx_ev_soil_f,
            qflx_ev_h2osfc=qflx_ev_h2osfc_f,
            qflx_surf=qflx_surf,
            fsat=fsat,
            dz=dz,
            watsat=watsat,
            hksat=hksat,
            h2osfc_thresh=h2osfc_thresh,
            pc=pc,
            topo_slope=topo_slope_exp,
            dt=dt,
        ),
    )
    core = _profiled(
        profile_sections,
        "hydrology_full_soil_water_core",
        device,
        lambda: soil_water_zengdecker2009_torch(
            SoilWaterCoreState(
                h2osoi_liq=state.h2osoi_liq,
                h2osoi_ice=state.h2osoi_ice,
                t_soisno=state.t_soisno,
                zwt=state.zwt,
            ),
            params,
            qflx_infl=_restore_vec(infil["qflx_infl"], batch_shape),
            qflx_rootsoi=qflx_rootsoi,
            dt=dt,
            workspace=workspace,
        ),
    )
    eff_porosity = _restore_last(infil["eff_porosity"], batch_shape)
    zwt2, wa2 = _profiled(
        profile_sections,
        "hydrology_full_water_table",
        device,
        lambda: water_table_torch(
            zwt=state.zwt,
            qcharge=core.qcharge,
            z=params.z,
            zi=params.zi,
            watsat=params.watsat,
            sucsat=params.sucsat,
            bsw=params.bsw,
            dt=dt,
            wa=wa,
        ),
    )
    hliq2, hsurf2, zwt3, qdrain, wa3 = _profiled(
        profile_sections,
        "hydrology_full_drainage",
        device,
        lambda: drainage_torch(
            h2osoi_liq=core.h2osoi_liq,
            h2osoi_ice=state.h2osoi_ice,
            h2osfc=_restore_vec(infil["h2osfc"], batch_shape),
            zwt=zwt2,
            z=params.z,
            zi=params.zi,
            dz=params.dz,
            watsat=params.watsat,
            sucsat=params.sucsat,
            bsw=params.bsw,
            eff_porosity=eff_porosity,
            fdrain=params.fdrain,
            topo_slope=params.topo_slope,
            rsub_top_globalmax=params.rsub_top_globalmax,
            dt=dt,
            wa=wa2,
        ),
    )
    return SoilHydrologyTorchResult(
        h2osoi_liq=hliq2,
        h2osoi_vol=core.h2osoi_vol,
        h2osfc=hsurf2,
        qflx_surf=_restore_vec(qflx_surf, batch_shape),
        qflx_infl=_restore_vec(infil["qflx_infl"], batch_shape),
        qflx_soil_evap=_restore_vec(infil["qflx_soil_evap"], batch_shape),
        qflx_h2osfc_surf=_restore_vec(infil["qflx_h2osfc_surf"], batch_shape),
        qflx_h2osfc_drain=_restore_vec(infil["qflx_h2osfc_drain"], batch_shape),
        qflx_infl_excess=_restore_vec(infil["qflx_infl_excess"], batch_shape),
        qcharge=core.qcharge,
        smp_l=core.smp_l,
        hk_l=core.hk_l,
        qflx_drain=qdrain,
        qflx_rsub_sat=torch.zeros_like(qdrain),
        qflx_deep_drain=core.qcharge,
        eff_porosity=eff_porosity,
        zwt=zwt3,
        wa=wa3,
        zwt_after_qcharge=zwt2,
        zwt_after_drainage=zwt3,
        qin_layer1=core.qin_layer1,
        qout_layer1=core.qout_layer1,
        qroot_layer1=core.qroot_layer1,
        dW1dt_layer1=core.dW1dt_layer1,
    )


__all__ = [
    "SoilWaterCoreState",
    "SoilHydrologyNoDrainageState",
    "SoilWaterCoreParameters",
    "SoilWaterCoreResult",
    "SoilHydrologyTorchResult",
    "batched_tridiagonal_torch",
    "surface_runoff_torch",
    "infiltration_torch",
    "soil_water_zengdecker2009_torch",
    "hydrology_no_drainage_torch",
    "water_table_torch",
    "drainage_torch",
    "hydrology_torch",
    "SoilHydrologyTorchWorkspace",
]
