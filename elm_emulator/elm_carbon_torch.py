#!/usr/bin/env python3
"""PyTorch port of ELM CTC carbon cycle (plant + soil decomposition).

Shapes
------
All state tensors use *batch-first* layout:

    scalars  :  ``(ncell, nens)``
    vertical :  ``(ncell, nens, nlev)``   (nlev = number of soil layers)

The PFT dimension is **not** batched here — call once per PFT and accumulate
litter externally, matching the existing hydrology/temperature solver pattern.

Moisture scalar
---------------
Uses the ELM log-matric-potential formulation:

    f_w = clamp( (log(-ψ) - log(-ψ_min)) / (log(-ψ_wet) - log(-ψ_min)), 0, 1 )

``smp_l`` is expected in **mm** (same sign convention as the hydrology solver —
negative values for unsaturated soil; positive / zero = saturated).

Parameters
----------
``CarbonParamsTorch`` is the single pre-computed parameter struct (replaces
``CarbonPFTParams`` + ``CarbonDecompParams``).  Build it once with
``build_carbon_params_torch()`` then reuse across timesteps.

Integration guide
-----------------
1. ``carbon_params = build_carbon_params_torch(nc_file, pft, device=device)``
2. ``plant_state  = init_plant_state_torch(ncell, nens, carbon_params, device)``
3. ``soil_state   = init_soil_state_torch(ncell, nens, nlev, device=device)``
4. Each hour::

       plant_state, litter, diag = carbon_plant_step_torch(
           plant_state, gpp, tref, t_pheno, dayl, rootfr, carbon_params, dt)
       soil_state, hr = carbon_soil_step_torch(
           soil_state, litter, t_soil, smp_l, dz, carbon_params, dt)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from netCDF4 import Dataset

from .elm_hydrology import elm_soil_grid  # reuse soil layer geometry

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
_SECS_PER_DAY: float = 86400.0
_SECS_PER_YEAR: float = 365.25 * _SECS_PER_DAY
_ELM_SPINUP_SECS_PER_YEAR: float = 365.0 * _SECS_PER_DAY
_MAX_ACCEL_DECOMP_RATE: float = 1.0 / _SECS_PER_YEAR
_TFRZ: float = 273.15
_MR_TREF: float = 20.0 + _TFRZ        # K — reference for maintenance resp.

# Dynamic woody-stem allocation (ELM AllocationMod.F90, CNAllocationMod).
# When stem_leaf == -1 (ELM sentinel) use the native sigmoid:
#   f3 = max(2.7 / (1 + exp(-0.004*(annsum_npp - 300))) - 0.4,  0.2)
# annsum_npp is the previous year's NPP in gC m⁻² yr⁻¹.
# Reference value at NPP=300: f3 = max(2.7/2 - 0.4, 0.2) = 0.95
_ALLOC_SIGMOID_A: float = 2.7
_ALLOC_SIGMOID_K: float = 0.004
_ALLOC_SIGMOID_X0: float = 300.0
_ALLOC_SIGMOID_B: float = 0.4
_ALLOC_SIGMOID_MIN: float = 0.2

# Moisture scalar limits (mm of matric potential, negative)
_PSI_WET_MM: float = -10.0             # ≈ field capacity (−0.01 MPa)
_PSI_MIN_MM: float = -1e8              # wilting / near-zero decomp (−100 MPa)
_LOG_PSI_WET: float = float(np.log(-_PSI_WET_MM))   # ln(10) ≈ 2.303
_LOG_PSI_MIN: float = float(np.log(-_PSI_MIN_MM))   # ln(1e8) ≈ 18.421
_MPa_TO_MM: float = 1e6 / 9.81   # convert MPa → mm H₂O (matric potential)


# ---------------------------------------------------------------------------
# Parameter struct
# ---------------------------------------------------------------------------

@dataclass
class CarbonParamsTorch:
    """Combined PFT + decomposition parameters as 0-d / 1-d torch tensors.

    All rate constants are stored in **s⁻¹**.
    Decomposition constants from CLM/ELM params are converted from day⁻¹;
    plant turnover constants are converted from year⁻¹.
    All tensors live on ``device`` with ``dtype``.
    """
    device: torch.device
    dtype: torch.dtype
    is_evergreen: bool
    is_season_decid: bool
    is_stress_decid: bool
    is_any_decid: bool
    use_dynamic_stem_flag: bool
    som_diffus_value: float

    # ------------------------------------------------------------------
    # PFT-level scalars  (shape: scalar tensor)
    # ------------------------------------------------------------------
    leafcn:    torch.Tensor   # leaf C:N
    frootcn:   torch.Tensor
    livewdcn:  torch.Tensor
    deadwdcn:  torch.Tensor

    froot_leaf: torch.Tensor   # allometric ratios
    stem_leaf:  torch.Tensor
    croot_stem: torch.Tensor
    flivewd:    torch.Tensor   # fraction of stem/croot allocation → live pool (rest → dead)

    leaf_long:  torch.Tensor   # longevity (yr)
    froot_long: torch.Tensor

    grperc:     torch.Tensor   # growth respiration fraction

    lf_flab: torch.Tensor; lf_fcel: torch.Tensor; lf_flig: torch.Tensor
    fr_flab: torch.Tensor; fr_fcel: torch.Tensor; fr_flig: torch.Tensor

    evergreen:    torch.Tensor   # bool stored as float (1 / 0)
    season_decid: torch.Tensor

    q10_mr: torch.Tensor   # Q10 for maintenance respiration

    # Derived scalars (computed once from leaf_long / froot_long etc.)
    leaf_turn_rate:  torch.Tensor   # s⁻¹
    froot_turn_rate: torch.Tensor
    lwtop_rate:      torch.Tensor   # s⁻¹  live-wood→dead-wood turnover (lwtop_ann / secs_per_yr)
    r_mort:          torch.Tensor   # background mortality rate (s⁻¹)

    # Allometric fractions — static fallback for non-woody PFTs
    f_leaf:   torch.Tensor
    f_froot:  torch.Tensor
    f_lstem:  torch.Tensor
    f_lcroot: torch.Tensor

    # Dynamic woody-stem allocation (ELM AllocationMod sigmoid)
    # use_dynamic_stem == 1 when stem_leaf param == -1 (ELM sentinel)
    woody:            torch.Tensor
    use_dynamic_stem: torch.Tensor

    # Stress-deciduous phenology (ELM PhenologyMod)
    # ELM native phenology uses soil layer 3 temperature, t_soisno(c,3).
    # Onset: wet-soil days trigger leaf-out; if enough freezing days occurred
    # during dormancy, ELM additionally requires a GDD sum before onset.
    # Offset: t_pheno < tc_stress OR soilpsi(layer 3) < soilpsi_off OR dayl < crit_dayl_stress
    stress_decid:      torch.Tensor
    soilpsi_on_mm:     torch.Tensor   # mm H₂O onset psi threshold (from MPa)
    soilpsi_off_mm:    torch.Tensor   # mm H₂O offset psi threshold (from MPa)
    crit_dayl_stress:  torch.Tensor   # s, daylength offset threshold
    tc_stress:         torch.Tensor   # K, cold-stress temperature
    crit_onset_fdd:    torch.Tensor   # d, freezing days needed to activate GDD gate
    crit_onset_swi:    torch.Tensor   # d, wet-soil days needed for onset

    # ------------------------------------------------------------------
    # Decomposition scalars
    # ------------------------------------------------------------------
    k_l1:   torch.Tensor; k_l2:   torch.Tensor; k_l3:   torch.Tensor
    k_s1:   torch.Tensor; k_s2:   torch.Tensor; k_s3:   torch.Tensor
    k_s4:   torch.Tensor; k_frag: torch.Tensor

    cwd_flig: torch.Tensor

    rf_l1s1: torch.Tensor; rf_l2s2: torch.Tensor; rf_l3s3: torch.Tensor
    rf_s1s2: torch.Tensor; rf_s2s3: torch.Tensor; rf_s3s4: torch.Tensor

    q10_hr:            torch.Tensor
    q10_hr_at_frz:     torch.Tensor
    som_diffus:        torch.Tensor   # m² s⁻¹
    decomp_depth_efolding: torch.Tensor  # m

    # Python scalar copies avoid host/device sync when choosing accelerated rates.
    k_l1_value:   float; k_l2_value:   float; k_l3_value:   float
    k_s1_value:   float; k_s2_value:   float; k_s3_value:   float
    k_s4_value:   float; k_frag_value: float

    br_mr:          torch.Tensor   # base MR rate (gC gN⁻¹ s⁻¹ at MR_TREF)
    crit_dayl:      torch.Tensor   # s
    crit_onset_gdd: torch.Tensor   # °C·d
    ndays_on:       torch.Tensor   # d
    ndays_off:      torch.Tensor   # d
    fstor2tran:     torch.Tensor   # fraction storage→transfer at onset

    # Pre-computed depth scaling vector — shape (1, 1, nlev)
    depth_scalar: torch.Tensor

    # Surface-litter vertical profile — shape (1, 1, nlev)
    # Aboveground (leaf + stem CWD) litter distributed with shallow exponential
    # e-folding depth (~0.1 m), normalised so sum == 1 on an area basis.
    surf_litter_profile: torch.Tensor


# ---------------------------------------------------------------------------
# State dataclasses
# ---------------------------------------------------------------------------

@dataclass
class PlantStateTorch:
    """Live plant carbon pools, shape ``(ncell, nens)`` per field."""
    leafc:      torch.Tensor
    frootc:     torch.Tensor
    livestemc:  torch.Tensor
    deadstemc:  torch.Tensor
    livecrootc: torch.Tensor
    deadcrootc: torch.Tensor

    leafc_xfer:      torch.Tensor
    frootc_xfer:     torch.Tensor
    livestemc_xfer:  torch.Tensor
    deadstemc_xfer:  torch.Tensor
    livecrootc_xfer: torch.Tensor
    deadcrootc_xfer: torch.Tensor

    leafc_stor:      torch.Tensor
    frootc_stor:     torch.Tensor
    livestemc_stor:  torch.Tensor
    deadstemc_stor:  torch.Tensor
    livecrootc_stor: torch.Tensor
    deadcrootc_stor: torch.Tensor

    pheno_state: torch.Tensor   # int32, (ncell, nens)  0=dormant 1=onset 2=growing 3=offset
    gdd:         torch.Tensor   # °C·d, (ncell, nens)
    onset_day:   torch.Tensor   # d elapsed in onset,  (ncell, nens)
    offset_day:  torch.Tensor   # d elapsed in offset, (ncell, nens)
    onset_fdd:   torch.Tensor   # d FDD accumulated since last offset (stress-decid)
    onset_swi:   torch.Tensor   # d wet-soil index accumulated while dormant
    onset_gddflag: torch.Tensor # 1 if freeze gate is active for this dormancy
    annsum_npp:  torch.Tensor   # gC m⁻² yr⁻¹, previous year's NPP (updated externally)

    def total(self) -> torch.Tensor:
        return (self.leafc + self.frootc + self.livestemc + self.deadstemc
                + self.livecrootc + self.deadcrootc
                + self.leafc_xfer + self.frootc_xfer + self.livestemc_xfer
                + self.deadstemc_xfer + self.livecrootc_xfer + self.deadcrootc_xfer
                + self.leafc_stor + self.frootc_stor + self.livestemc_stor
                + self.deadstemc_stor + self.livecrootc_stor + self.deadcrootc_stor)


@dataclass
class SoilStateTorch:
    """Dead organic matter / SOM, shape ``(ncell, nens, nlev)`` per field."""
    cwdc:   torch.Tensor
    litr1c: torch.Tensor
    litr2c: torch.Tensor
    litr3c: torch.Tensor
    soil1c: torch.Tensor
    soil2c: torch.Tensor
    soil3c: torch.Tensor
    soil4c: torch.Tensor

    def total(self) -> torch.Tensor:
        return (self.cwdc + self.litr1c + self.litr2c + self.litr3c
                + self.soil1c + self.soil2c + self.soil3c + self.soil4c)


@dataclass
class LitterFluxTorch:
    """Litter fluxes from plant to soil, shape ``(ncell, nens, nlev)``, gC m⁻² s⁻¹."""
    litr1_flux: torch.Tensor
    litr2_flux: torch.Tensor
    litr3_flux: torch.Tensor
    cwd_flux:   torch.Tensor


@dataclass
class CarbonStepResultTorch:
    plant_state: PlantStateTorch
    soil_state:  SoilStateTorch
    gpp:  torch.Tensor   # (ncell, nens)
    npp:  torch.Tensor
    mr:   torch.Tensor
    gr:   torch.Tensor
    hr:   torch.Tensor   # column-integrated HR
    nee:  torch.Tensor


# ---------------------------------------------------------------------------
# Parameter loading
# ---------------------------------------------------------------------------

def build_carbon_params_torch(
    nc_file: str | Path,
    pft: int,
    dz: np.ndarray | None = None,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> CarbonParamsTorch:
    """Load ``CarbonParamsTorch`` from ``clm_params_c180524.nc``.

    Parameters
    ----------
    nc_file
        Path to ``clm_params_c180524.nc``.
    pft
        0-based PFT index.
    dz
        Layer thicknesses (m).  Defaults to ``elm_soil_grid()`` geometry.
    device, dtype
        Target device and dtype for all tensors.
    """
    device = torch.device(device)

    if dz is None:
        _z, dz_np, _zi = elm_soil_grid()
    else:
        dz_np = np.asarray(dz, dtype=np.float64)

    def _t(x: float) -> torch.Tensor:
        return torch.tensor(x, device=device, dtype=dtype)

    def _get(ds: Dataset, name: str, pft_idx: int | None = None) -> float:
        arr = np.asarray(ds.variables[name][:]).flatten()
        if pft_idx is not None and arr.size > 1:
            return float(arr[pft_idx])
        return float(arr[0])

    with Dataset(nc_file, "r") as ds:
        p = lambda n: _get(ds, n, pft)
        c = lambda n: _get(ds, n, None)

        leafcn    = p("leafcn")
        frootcn   = p("frootcn")
        livewdcn  = p("livewdcn")
        deadwdcn  = p("deadwdcn")
        froot_leaf  = p("froot_leaf")
        stem_leaf_raw = p("stem_leaf")
        croot_stem  = p("croot_stem")
        woody_flag  = p("woody")
        flivewd     = p("flivewd")
        leaf_long   = p("leaf_long")
        froot_long  = p("froot_long")
        grperc      = p("grperc")
        lf_flab = p("lf_flab"); lf_fcel = p("lf_fcel"); lf_flig = p("lf_flig")
        fr_flab = p("fr_flab"); fr_fcel = p("fr_fcel"); fr_flig = p("fr_flig")
        evergreen    = p("evergreen")
        season_decid = p("season_decid")
        q10_mr       = c("q10_mr")

        k_l1   = c("k_l1")  / _SECS_PER_DAY
        k_l2   = c("k_l2")  / _SECS_PER_DAY
        k_l3   = c("k_l3")  / _SECS_PER_DAY
        k_s1   = c("k_s1")  / _SECS_PER_DAY
        k_s2   = c("k_s2")  / _SECS_PER_DAY
        k_s3   = c("k_s3")  / _SECS_PER_DAY
        k_s4   = c("k_s4")  / _SECS_PER_DAY
        k_frag = c("k_frag") / _SECS_PER_DAY
        cwd_flig   = c("cwd_flig")
        rf_l1s1    = c("rf_l1s1"); rf_l2s2 = c("rf_l2s2"); rf_l3s3 = c("rf_l3s3")
        rf_s1s2    = c("rf_s1s2"); rf_s2s3 = c("rf_s2s3"); rf_s3s4 = c("rf_s3s4")
        q10_hr     = c("q10_hr")
        som_diffus             = c("som_diffus")
        decomp_depth_efolding  = c("decomp_depth_efolding")
        br_mr          = c("br_mr")
        crit_dayl      = c("crit_dayl")
        crit_onset_gdd = c("crit_onset_gdd")
        ndays_on       = c("ndays_on")
        ndays_off      = c("ndays_off")
        fstor2tran     = c("fstor2tran")
        lwtop_ann      = c("lwtop_ann")
        stress_decid_f  = p("stress_decid")
        soilpsi_on_MPa  = c("soilpsi_on")
        soilpsi_off_MPa = c("soilpsi_off")
        crit_dayl_str   = c("crit_dayl_stress")
        tc_stress_C     = c("tc_stress")
        crit_onset_fdd  = c("crit_onset_fdd")
        crit_onset_swi  = c("crit_onset_swi")

    # stem_leaf: ELM uses -1 as a sentinel meaning "compute dynamically from
    # annsum_npp".  Store static fractions as a non-woody / first-year fallback.
    use_dynamic_stem = woody_flag > 0.5 and stem_leaf_raw < 0.0
    is_evergreen = evergreen > 0.5
    is_season_decid = season_decid > 0.5 and not is_evergreen
    is_stress_decid = stress_decid_f > 0.5 and not is_evergreen
    is_any_decid = is_season_decid or is_stress_decid
    if use_dynamic_stem:
        # First-year fallback: evaluate sigmoid at annsum_npp=300 gC/m2/yr
        stem_leaf = max(
            _ALLOC_SIGMOID_A / (1.0 + float(np.exp(-_ALLOC_SIGMOID_K * (300.0 - _ALLOC_SIGMOID_X0))))
            - _ALLOC_SIGMOID_B,
            _ALLOC_SIGMOID_MIN,
        )
    else:
        stem_leaf = max(0.0, stem_leaf_raw)

    # Allometric fractions (static / first-year fallback)
    a_total = froot_leaf + stem_leaf + croot_stem * stem_leaf + 1.0
    a_total = max(a_total, 1e-9)
    f_leaf   = 1.0        / a_total
    f_froot  = froot_leaf / a_total
    f_lstem  = stem_leaf  / a_total
    f_lcroot = (croot_stem * stem_leaf) / a_total

    # Live-wood-to-dead-wood annual turnover rate (scalar, not per-PFT in params)
    # Derived rates (s⁻¹) — guard against bare-ground PFTs that have leaf_long=0
    leaf_turn_rate  = 1.0 / (max(leaf_long,  1e-6) * _SECS_PER_YEAR)
    froot_turn_rate = 1.0 / (max(froot_long, 1e-6) * _SECS_PER_YEAR)
    lwtop_rate      = lwtop_ann / _SECS_PER_YEAR
    r_mort          = 0.02      / _SECS_PER_YEAR

    # Depth scaling: (1, 1, nlev)
    nlev = len(dz_np)
    z_mid = (np.cumsum(dz_np) - 0.5 * dz_np).astype(np.float64)
    depth_scl = np.exp(-z_mid / decomp_depth_efolding).astype(np.float32)
    depth_scalar = torch.tensor(depth_scl, device=device, dtype=dtype).reshape(1, 1, nlev)
    q10_hr_at_frz = q10_hr ** ((_TFRZ - (_TFRZ + 25.0)) / 10.0)

    # Surface-litter profile: exponential with e-folding depth 0.1 m,
    # weighted by dz and normalised to sum to 1 (area-integrated = input flux).
    _surf_efolding = 0.1  # m — shallow surface litter distribution
    w_surf = np.exp(-z_mid / _surf_efolding) * dz_np
    w_surf /= w_surf.sum()
    surf_litter_profile = torch.tensor(w_surf.astype(np.float32), device=device, dtype=dtype).reshape(1, 1, nlev)

    return CarbonParamsTorch(
        device=device, dtype=dtype,
        is_evergreen=is_evergreen,
        is_season_decid=is_season_decid,
        is_stress_decid=is_stress_decid,
        is_any_decid=is_any_decid,
        use_dynamic_stem_flag=use_dynamic_stem,
        som_diffus_value=float(som_diffus),
        leafcn=_t(leafcn), frootcn=_t(frootcn), livewdcn=_t(livewdcn), deadwdcn=_t(deadwdcn),
        froot_leaf=_t(froot_leaf), stem_leaf=_t(stem_leaf),
        croot_stem=_t(croot_stem), flivewd=_t(flivewd),
        leaf_long=_t(leaf_long), froot_long=_t(froot_long),
        grperc=_t(grperc),
        lf_flab=_t(lf_flab), lf_fcel=_t(lf_fcel), lf_flig=_t(lf_flig),
        fr_flab=_t(fr_flab), fr_fcel=_t(fr_fcel), fr_flig=_t(fr_flig),
        evergreen=_t(evergreen), season_decid=_t(season_decid),
        q10_mr=_t(q10_mr),
        leaf_turn_rate=_t(leaf_turn_rate), froot_turn_rate=_t(froot_turn_rate),
        lwtop_rate=_t(lwtop_rate), r_mort=_t(r_mort),
        f_leaf=_t(f_leaf), f_froot=_t(f_froot), f_lstem=_t(f_lstem), f_lcroot=_t(f_lcroot),
        woody=_t(woody_flag), use_dynamic_stem=_t(float(use_dynamic_stem)),
        stress_decid=_t(stress_decid_f),
        soilpsi_on_mm=_t(soilpsi_on_MPa * _MPa_TO_MM),
        soilpsi_off_mm=_t(soilpsi_off_MPa * _MPa_TO_MM),
        crit_dayl_stress=_t(crit_dayl_str),
        tc_stress=_t(tc_stress_C + _TFRZ),
        crit_onset_fdd=_t(crit_onset_fdd),
        crit_onset_swi=_t(crit_onset_swi),
        k_l1=_t(k_l1), k_l2=_t(k_l2), k_l3=_t(k_l3),
        k_s1=_t(k_s1), k_s2=_t(k_s2), k_s3=_t(k_s3), k_s4=_t(k_s4), k_frag=_t(k_frag),
        cwd_flig=_t(cwd_flig),
        rf_l1s1=_t(rf_l1s1), rf_l2s2=_t(rf_l2s2), rf_l3s3=_t(rf_l3s3),
        rf_s1s2=_t(rf_s1s2), rf_s2s3=_t(rf_s2s3), rf_s3s4=_t(rf_s3s4),
        q10_hr=_t(q10_hr), q10_hr_at_frz=_t(q10_hr_at_frz),
        som_diffus=_t(som_diffus), decomp_depth_efolding=_t(decomp_depth_efolding),
        k_l1_value=k_l1, k_l2_value=k_l2, k_l3_value=k_l3,
        k_s1_value=k_s1, k_s2_value=k_s2, k_s3_value=k_s3,
        k_s4_value=k_s4, k_frag_value=k_frag,
        br_mr=_t(br_mr), crit_dayl=_t(crit_dayl), crit_onset_gdd=_t(crit_onset_gdd),
        ndays_on=_t(ndays_on), ndays_off=_t(ndays_off), fstor2tran=_t(fstor2tran),
        depth_scalar=depth_scalar,
        surf_litter_profile=surf_litter_profile,
    )


# ---------------------------------------------------------------------------
# Initialisation helpers
# ---------------------------------------------------------------------------

def init_plant_state_torch(
    ncell: int,
    nens: int,
    params: CarbonParamsTorch,
    *,
    leafc_init: float = 50.0,
) -> PlantStateTorch:
    """Cold-start plant state from a prescribed leaf C pool."""
    dev, dt = params.device, params.dtype
    z  = torch.zeros(ncell, nens, device=dev, dtype=dt)
    lc = leafc_init
    a_lstem  = float(params.stem_leaf.item())
    a_lcroot = float(params.croot_stem.item()) * a_lstem
    a_froot  = float(params.froot_leaf.item())

    # pheno=0 (dormant) for season_decid and stress_decid; pheno=2 (growing) for all others.
    pheno_init = 0 if params.is_any_decid else 2

    return PlantStateTorch(
        leafc=z + lc,
        frootc=z + lc * a_froot,
        livestemc=z + lc * a_lstem,
        deadstemc=torch.zeros_like(z),
        livecrootc=z + lc * a_lcroot,
        deadcrootc=torch.zeros_like(z),
        leafc_xfer=torch.zeros_like(z), frootc_xfer=torch.zeros_like(z),
        livestemc_xfer=torch.zeros_like(z), deadstemc_xfer=torch.zeros_like(z),
        livecrootc_xfer=torch.zeros_like(z), deadcrootc_xfer=torch.zeros_like(z),
        leafc_stor=torch.zeros_like(z), frootc_stor=torch.zeros_like(z),
        livestemc_stor=torch.zeros_like(z), deadstemc_stor=torch.zeros_like(z),
        livecrootc_stor=torch.zeros_like(z), deadcrootc_stor=torch.zeros_like(z),
        pheno_state=torch.full((ncell, nens), pheno_init, device=dev, dtype=torch.int32),
        gdd=torch.zeros_like(z), onset_day=torch.zeros_like(z), offset_day=torch.zeros_like(z),
        onset_fdd=torch.zeros_like(z),
        onset_swi=torch.zeros_like(z),
        onset_gddflag=torch.zeros_like(z),
        annsum_npp=torch.full((ncell, nens), 300.0, device=dev, dtype=dt),
    )


def init_soil_state_torch(
    ncell: int,
    nens: int,
    nlev: int = 10,
    *,
    som_init: float = 1000.0,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> SoilStateTorch:
    """Cold-start soil state with uniformly distributed SOM."""
    device = torch.device(device)
    z = torch.zeros(ncell, nens, nlev, device=device, dtype=dtype)
    s = som_init / nlev
    return SoilStateTorch(
        cwdc=z.clone(), litr1c=z.clone(), litr2c=z.clone(), litr3c=z.clone(),
        soil1c=z + s * 0.05,
        soil2c=z + s * 0.20,
        soil3c=z + s * 0.60,
        soil4c=z + s * 0.15,
    )


def rootfr_profile_torch(
    roota: float,
    rootb: float,
    z_mid: np.ndarray | torch.Tensor,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    ncell: int = 1,
    nens: int = 1,
) -> torch.Tensor:
    """Jackson (1996) exponential root fraction profile, shape ``(1, 1, nlev)``."""
    device = torch.device(device)
    if isinstance(z_mid, torch.Tensor):
        z = z_mid.to(device=device, dtype=dtype)
    else:
        z = torch.as_tensor(z_mid, device=device, dtype=dtype)
    frac = torch.exp(-roota * z) + torch.exp(-rootb * z)
    frac = frac.clamp(min=0.0)
    total = frac.sum()
    frac = frac / total.clamp(min=1e-15)
    return frac.reshape(1, 1, -1)   # broadcast-ready


# ---------------------------------------------------------------------------
# Environmental scalars
# ---------------------------------------------------------------------------

def _f_temp_decomp_torch(
    t_soil: torch.Tensor,
    q10: torch.Tensor,
    val_at_frz: torch.Tensor,
) -> torch.Tensor:
    """Q10 temperature scalar with frozen-soil suppression."""
    ref = _TFRZ + 25.0
    above = torch.pow(q10, (t_soil - ref) / 10.0)
    # Below 0°C: Q10 value at 0°C × exp(0.4 × (T − 273.15))
    below = val_at_frz * torch.exp(0.4 * (t_soil - _TFRZ))
    return torch.where(t_soil >= _TFRZ, above, below.clamp(min=0.0))


def _f_water_decomp_torch(smp_l: torch.Tensor) -> torch.Tensor:
    """Log-matric-potential moisture scalar.

    ``smp_l`` in mm (negative = unsaturated; ≥ 0 = saturated → f_w = 1).
    """
    # Clamp to avoid log(0) — use a small negative floor
    psi = smp_l.clamp(max=-1e-3)   # mm, negative; ensure < 0
    log_psi = torch.log(-psi)       # log(|ψ|), all positive
    f_w = (log_psi - _LOG_PSI_MIN) / (_LOG_PSI_WET - _LOG_PSI_MIN)
    f_w = f_w.clamp(0.0, 1.0)
    # Override: saturated (smp_l ≥ 0) → f_w = 1
    return torch.where(smp_l >= 0.0, 1.0, f_w)


def elm_native_decomp_spinup_factors(params: CarbonParamsTorch) -> dict[str, float]:
    """ELM DecompCascadeCN accelerated-decomposition factors for spinup_state=1.

    Native ELM derives these from the base pool turnover rates, so the accelerated
    first-order rate is roughly one turnover per year for the slow litter/SOM pools
    and one half-year turnover for CWD.  Environmental temperature, moisture, and
    depth scalars still multiply the accelerated rates at each timestep; ELM's
    scalar-average correction begins after spinup year 40 and is intentionally not
    included here.
    """

    def _pool_factor(k_value: float) -> float:
        if not np.isfinite(k_value) or k_value <= 0.0:
            return 1.0
        return max(1.0, 1.0 / (k_value * _ELM_SPINUP_SECS_PER_YEAR))

    def _cwd_factor(k_value: float) -> float:
        if not np.isfinite(k_value) or k_value <= 0.0:
            return 1.0
        return max(1.0, 0.5 / (k_value * _ELM_SPINUP_SECS_PER_YEAR))

    return {
        "litr1": _pool_factor(params.k_l1_value),
        "litr2": _pool_factor(params.k_l2_value),
        "litr3": _pool_factor(params.k_l3_value),
        "cwd": _cwd_factor(params.k_frag_value),
        "soil1": _pool_factor(params.k_s1_value),
        "soil2": _pool_factor(params.k_s2_value),
        "soil3": _pool_factor(params.k_s3_value),
        "soil4": _pool_factor(params.k_s4_value),
    }


# ---------------------------------------------------------------------------
# Plant step
# ---------------------------------------------------------------------------

def carbon_plant_step_torch(
    state: PlantStateTorch,
    gpp: torch.Tensor,       # (ncell, nens)  gC m⁻² s⁻¹
    tref: torch.Tensor,      # (ncell, nens)  K — canopy/air temperature for MR
    t_pheno: torch.Tensor,   # (ncell, nens)  K — phenology temperature, ELM t_soisno layer 3
    dayl: torch.Tensor,      # (ncell, nens)  s — current daylength
    rootfr: torch.Tensor,    # (1, 1, nlev) or (ncell, nens, nlev) root fraction
    params: CarbonParamsTorch,
    dt: float,
    smp_l: torch.Tensor | None = None,   # (ncell, nens, nlev) mm, for stress-decid
    crit_onset_gdd: torch.Tensor | None = None,  # (ncell, nens) °C·d, ELM dynamic threshold
    accel_fac: float = 1.0,
    prev_dayl: torch.Tensor | None = None,  # (ncell, nens) s — previous timestep/day daylength
) -> tuple[PlantStateTorch, LitterFluxTorch, dict[str, torch.Tensor]]:
    """Single vectorized plant carbon step.

    Returns ``(new_state, litter_fluxes, diag)`` where ``diag`` contains
    ``"gpp"``, ``"mr"``, ``"gr"``, ``"npp"`` all in gC m⁻² s⁻¹, plus
    boolean phenology transition flags ``"onset"`` and ``"offset"``.
    """
    p = params
    dev, dt_t = p.device, p.dtype
    dt_day = dt / 86400.0
    inv_dt = 1.0 / dt
    nlev = rootfr.shape[-1]
    batch_shape = state.leafc.shape   # (ncell, nens)

    is_evergreen    = p.is_evergreen
    is_season_decid = p.is_season_decid
    is_stress_decid = p.is_stress_decid
    is_any_decid    = p.is_any_decid

    # ---- unpack state ----
    leafc      = state.leafc
    frootc     = state.frootc
    livestemc  = state.livestemc
    deadstemc  = state.deadstemc
    livecrootc = state.livecrootc
    deadcrootc = state.deadcrootc
    leafc_xfer = state.leafc_xfer; frootc_xfer = state.frootc_xfer
    livestemc_xfer  = state.livestemc_xfer;  deadstemc_xfer  = state.deadstemc_xfer
    livecrootc_xfer = state.livecrootc_xfer; deadcrootc_xfer = state.deadcrootc_xfer
    leafc_stor = state.leafc_stor; frootc_stor = state.frootc_stor
    livestemc_stor  = state.livestemc_stor;  deadstemc_stor  = state.deadstemc_stor
    livecrootc_stor = state.livecrootc_stor; deadcrootc_stor = state.deadcrootc_stor
    pheno      = state.pheno_state
    gdd        = state.gdd
    on_day     = state.onset_day
    off_day    = state.offset_day
    annsum_npp = state.annsum_npp   # (ncell, nens) gC m⁻² yr⁻¹
    onset_fdd  = state.onset_fdd    # (ncell, nens) d
    onset_swi  = state.onset_swi    # (ncell, nens) d
    onset_gddflag = state.onset_gddflag
    if crit_onset_gdd is None:
        crit_onset_gdd = p.crit_onset_gdd
    if prev_dayl is None:
        ws_flag = torch.ones(batch_shape, device=dev, dtype=torch.bool)
        season_offset_ok = torch.ones(batch_shape, device=dev, dtype=torch.bool)
    else:
        # ELM seasonal-deciduous phenology only accumulates onset GDD on the
        # winter-to-summer side of the daylength cycle, and only tests offset
        # after daylength starts declining past summer solstice.
        ws_flag = dayl >= prev_dayl
        season_offset_ok = ~ws_flag

    # ==================================================================
    # 1. Phenology
    # ==================================================================
    onset_event = torch.zeros(batch_shape, device=dev, dtype=torch.bool)
    offset_event = torch.zeros(batch_shape, device=dev, dtype=torch.bool)

    def _storage_to_transfer(trigger: torch.Tensor) -> None:
        nonlocal leafc_xfer, frootc_xfer, livestemc_xfer, deadstemc_xfer
        nonlocal livecrootc_xfer, deadcrootc_xfer
        nonlocal leafc_stor, frootc_stor, livestemc_stor, deadstemc_stor
        nonlocal livecrootc_stor, deadcrootc_stor

        xf = p.fstor2tran
        remain = 1.0 - xf
        leafc_xfer      = torch.where(trigger, leafc_xfer      + xf * leafc_stor,      leafc_xfer)
        frootc_xfer     = torch.where(trigger, frootc_xfer     + xf * frootc_stor,     frootc_xfer)
        livestemc_xfer  = torch.where(trigger, livestemc_xfer  + xf * livestemc_stor,  livestemc_xfer)
        deadstemc_xfer  = torch.where(trigger, deadstemc_xfer  + xf * deadstemc_stor,  deadstemc_xfer)
        livecrootc_xfer = torch.where(trigger, livecrootc_xfer + xf * livecrootc_stor, livecrootc_xfer)
        deadcrootc_xfer = torch.where(trigger, deadcrootc_xfer + xf * deadcrootc_stor, deadcrootc_xfer)

        leafc_stor      = torch.where(trigger, leafc_stor      * remain, leafc_stor)
        frootc_stor     = torch.where(trigger, frootc_stor     * remain, frootc_stor)
        livestemc_stor  = torch.where(trigger, livestemc_stor  * remain, livestemc_stor)
        deadstemc_stor  = torch.where(trigger, deadstemc_stor  * remain, deadstemc_stor)
        livecrootc_stor = torch.where(trigger, livecrootc_stor * remain, livecrootc_stor)
        deadcrootc_stor = torch.where(trigger, deadcrootc_stor * remain, deadcrootc_stor)

    if is_season_decid:
        dormant = pheno == 0
        gdd_incr = (t_pheno - _TFRZ).clamp(min=0.0) * dt_day
        gdd = torch.where(dormant & season_offset_ok, 0.0, gdd)
        gdd = torch.where(dormant & ws_flag, gdd + gdd_incr, gdd)

        # Dormant → onset (GDD threshold)
        onset_trigger = dormant & ws_flag & (gdd >= crit_onset_gdd)
        onset_event = onset_event | onset_trigger
        gdd    = torch.where(onset_trigger, 0.0, gdd)
        pheno  = torch.where(onset_trigger, 1, pheno)
        on_day = torch.where(onset_trigger, 0.0, on_day)

        # Storage → transfer at onset trigger
        _storage_to_transfer(onset_trigger)

    if is_stress_decid:
        # Native ELM uses third-layer soil water potential: soilpsi(c,3).
        if smp_l is not None:
            soilpsi_pheno = smp_l[..., 2]   # (ncell, nens)
        else:
            soilpsi_pheno = torch.zeros(batch_shape, device=dev, dtype=dt_t)

        dormant = pheno == 0
        freeze_gate = onset_gddflag > 0.5

        # ELM accumulates freezing days first. Only after enough freezing days
        # does it require a GDD sum before wet-soil onset can proceed.
        onset_fdd = torch.where(
            dormant & (~freeze_gate) & (t_pheno < _TFRZ),
            onset_fdd + dt_day,
            onset_fdd,
        )
        freeze_trigger = dormant & (~freeze_gate) & (onset_fdd > p.crit_onset_fdd)
        onset_gddflag = torch.where(freeze_trigger, 1.0, onset_gddflag)
        onset_fdd = torch.where(freeze_trigger, 0.0, onset_fdd)
        onset_swi = torch.where(freeze_trigger, 0.0, onset_swi)
        freeze_gate = onset_gddflag > 0.5

        gdd = torch.where(
            dormant & freeze_gate & (t_pheno > _TFRZ),
            gdd + (t_pheno - _TFRZ) * dt_day,
            gdd,
        )
        onset_swi = torch.where(
            dormant & (soilpsi_pheno >= p.soilpsi_on_mm),
            onset_swi + dt_day,
            onset_swi,
        )

        onset_s = (
            dormant
            & (onset_swi > p.crit_onset_swi)
            & ((~freeze_gate) | (gdd >= crit_onset_gdd))
            & (dayl > p.crit_dayl_stress)
        )
        onset_event = onset_event | onset_s
        onset_fdd = torch.where(onset_s, 0.0, onset_fdd)
        onset_swi = torch.where(onset_s, 0.0, onset_swi)
        onset_gddflag = torch.where(onset_s, 0.0, onset_gddflag)
        gdd = torch.where(onset_s, 0.0, gdd)
        pheno  = torch.where(onset_s, 1, pheno)
        on_day = torch.where(onset_s, 0.0, on_day)

        _storage_to_transfer(onset_s)

        # Offset: growing → senescence when cold OR dry OR short daylength
        offset_s = (
            (pheno == 2)
            & ((t_pheno < p.tc_stress)
               | (soilpsi_pheno < p.soilpsi_off_mm)
               | (dayl < p.crit_dayl_stress))
        )
        offset_event = offset_event | offset_s
        pheno   = torch.where(offset_s, 3, pheno)
        off_day = torch.where(offset_s, 0.0, off_day)

    # ==================================================================
    # 2. Onset: transfer → active over ndays_on
    # ==================================================================
    if is_any_decid:
        in_onset  = (pheno == 1)
        remaining_onset_days = (p.ndays_on - on_day).clamp(min=dt_day)
        onset_rate_dt = torch.where(in_onset, (dt_day / remaining_onset_days).clamp(max=1.0), 0.0)
        lx = torch.where(in_onset, leafc_xfer      * onset_rate_dt, 0.0)
        fx = torch.where(in_onset, frootc_xfer     * onset_rate_dt, 0.0)
        ls = torch.where(in_onset, livestemc_xfer  * onset_rate_dt, 0.0)
        ds = torch.where(in_onset, deadstemc_xfer  * onset_rate_dt, 0.0)
        lc = torch.where(in_onset, livecrootc_xfer * onset_rate_dt, 0.0)
        dc = torch.where(in_onset, deadcrootc_xfer * onset_rate_dt, 0.0)
        leafc      = leafc      + lx
        frootc     = frootc     + fx
        livestemc  = livestemc  + ls
        deadstemc  = deadstemc  + ds
        livecrootc = livecrootc + lc
        deadcrootc = deadcrootc + dc
        leafc_xfer      = leafc_xfer      - lx
        frootc_xfer     = frootc_xfer     - fx
        livestemc_xfer  = livestemc_xfer  - ls
        deadstemc_xfer  = deadstemc_xfer  - ds
        livecrootc_xfer = livecrootc_xfer - lc
        deadcrootc_xfer = deadcrootc_xfer - dc
        on_day = torch.where(in_onset, on_day + dt_day, on_day)
        xfer_remaining = (
            leafc_xfer + frootc_xfer + livestemc_xfer + deadstemc_xfer
            + livecrootc_xfer + deadcrootc_xfer
        )
        onset_done = in_onset & ((on_day >= p.ndays_on) | (xfer_remaining <= 0.0))
        pheno  = torch.where(onset_done, 2, pheno)
        on_day = torch.where(onset_done, 0.0, on_day)

    # ==================================================================
    # 3. Maintenance respiration
    # ==================================================================
    f_mr = torch.pow(p.q10_mr, (tref - _MR_TREF) / 10.0)
    br   = p.br_mr

    def _mr_pool(pool: torch.Tensor, cn: torch.Tensor) -> torch.Tensor:
        n_pool = pool / cn.clamp(min=1e-15)
        return br * n_pool * f_mr

    mr_leaf   = _mr_pool(leafc,      p.leafcn)
    mr_froot  = _mr_pool(frootc,     p.frootcn)
    mr_lstem  = _mr_pool(livestemc,  p.livewdcn)
    mr_lcroot = _mr_pool(livecrootc, p.livewdcn)
    mr_total  = mr_leaf + mr_froot + mr_lstem + mr_lcroot   # gC m⁻² s⁻¹

    # MR scalar: scale down when C-limited
    mr_scalar = torch.where(
        mr_total > 1e-15,
        (1.0 - (-( gpp - mr_total)).clamp(min=0.0) / (mr_total * dt + 1e-15)).clamp(0.0, 1.0),
        1.0,
    )
    mr_paid  = mr_total * mr_scalar
    avail_c2 = gpp - mr_paid   # ≥ 0 by construction

    # ==================================================================
    # 4. Allocation
    # ==================================================================
    alloc_gross = avail_c2.clamp(min=0.0) * dt   # gC m⁻²
    if not is_evergreen:
        alloc_gross = torch.where(pheno == 0, 0.0, alloc_gross)
    gr_step     = alloc_gross * p.grperc
    alloc_net   = alloc_gross * (1.0 - p.grperc)

    # Dynamic stem:leaf ratio for woody PFTs (ELM AllocationMod.F90).
    # f3 = max(2.7 / (1 + exp(-0.004*(annsum_npp - 300))) - 0.4,  0.2)
    # Non-woody PFTs use the static fractions from params.
    if p.use_dynamic_stem_flag:
        stem_leaf_dyn = (_ALLOC_SIGMOID_A
                         / (1.0 + torch.exp(-_ALLOC_SIGMOID_K * (annsum_npp - _ALLOC_SIGMOID_X0)))
                         - _ALLOC_SIGMOID_B).clamp(min=_ALLOC_SIGMOID_MIN)   # (ncell, nens)
        a_tot = (1.0 + p.froot_leaf
                 + stem_leaf_dyn + p.croot_stem * stem_leaf_dyn).clamp(min=1e-9)
        f_leaf_d   = 1.0               / a_tot
        f_froot_d  = p.froot_leaf      / a_tot
        f_lstem_d  = stem_leaf_dyn     / a_tot
        f_lcroot_d = p.croot_stem * stem_leaf_dyn / a_tot
    else:
        f_leaf_d   = p.f_leaf
        f_froot_d  = p.f_froot
        f_lstem_d  = p.f_lstem
        f_lcroot_d = p.f_lcroot

    # flivewd: fraction of stem/croot allocation → live pool; rest → dead pool.
    # (ELM AllocationMod: livestemc += alloc * f_lstem * flivewd;
    #                      deadstemc += alloc * f_lstem * (1 - flivewd))
    flivewd = p.flivewd

    if is_evergreen:
        leafc      = leafc      + alloc_net * f_leaf_d
        frootc     = frootc     + alloc_net * f_froot_d
        livestemc  = livestemc  + alloc_net * f_lstem_d  * flivewd
        deadstemc  = deadstemc  + alloc_net * f_lstem_d  * (1.0 - flivewd)
        livecrootc = livecrootc + alloc_net * f_lcroot_d * flivewd
        deadcrootc = deadcrootc + alloc_net * f_lcroot_d * (1.0 - flivewd)
    else:
        # Deciduous displayed leaf/froot growth comes from transfer pools during
        # onset. New NPP replenishes storage for the next leaf-out instead of
        # increasing in-season LAI.
        go_storage = alloc_net

        leafc_stor  = leafc_stor  + go_storage * f_leaf_d
        frootc_stor = frootc_stor + go_storage * f_froot_d
        livestemc_stor  = livestemc_stor  + go_storage * f_lstem_d  * flivewd
        deadstemc_stor  = deadstemc_stor  + go_storage * f_lstem_d  * (1.0 - flivewd)
        livecrootc_stor = livecrootc_stor + go_storage * f_lcroot_d * flivewd
        deadcrootc_stor = deadcrootc_stor + go_storage * f_lcroot_d * (1.0 - flivewd)

    # ==================================================================
    # 5. Offset: leaves and fine roots senesce to LITTER over ndays_off
    # ==================================================================
    # For deciduous: senescing leafc/frootc goes to litter (not storage).
    # Storage for next year is built from GPP allocation during dormancy.
    offset_leaf_litter = 0.0
    offset_froot_litter = 0.0

    if is_season_decid:
        # Season-deciduous offset is triggered by daylength falling below crit_dayl.
        # (Stress-deciduous offset is already triggered in section 1 above.)
        offset_trigger = (pheno == 2) & season_offset_ok & (dayl < p.crit_dayl)
        offset_event = offset_event | offset_trigger
        pheno   = torch.where(offset_trigger, 3, pheno)
        off_day = torch.where(offset_trigger, 0.0, off_day)

    if is_any_decid:
        in_offset = (pheno == 3)
        remaining_offset_days = (p.ndays_off - off_day).clamp(min=dt_day)
        offset_rate_dt = (dt_day / remaining_offset_days).clamp(max=1.0)
        lt = torch.where(in_offset, leafc  * offset_rate_dt, 0.0)
        ft = torch.where(in_offset, frootc * offset_rate_dt, 0.0)
        leafc  = leafc  - lt   # senesced leaves → litter (see section 8)
        frootc = frootc - ft   # senesced roots  → litter (see section 8)
        offset_leaf_litter  = lt
        offset_froot_litter = ft
        off_day = torch.where(in_offset, off_day + dt_day, off_day)
        offset_done = in_offset & (off_day >= p.ndays_off)
        pheno   = torch.where(offset_done, 0, pheno)
        off_day = torch.where(offset_done, 0.0, off_day)
        gdd     = torch.where(offset_done, 0.0, gdd)

    # ==================================================================
    # 6. Turnover (continuous)
    # ==================================================================
    # Evergreen: continuous leaf/froot turnover at 1/leaf_long and 1/froot_long.
    # Deciduous (season or stress): senescence handled by phenological offset
    # (section 5) → litter; no additional continuous rate to avoid double-counting.
    if is_any_decid:
        leaf_turn_flux = 0.0
        froot_turn_flux = 0.0
    else:
        leaf_turn_flux  = leafc  * p.leaf_turn_rate  * dt
        froot_turn_flux = frootc * p.froot_turn_rate * dt
        leafc  = leafc  - leaf_turn_flux
        frootc = frootc - froot_turn_flux

    # Live wood → dead wood  (lwtop_ann = 0.7 yr⁻¹ from params)
    lstem_to_dstem   = livestemc  * p.lwtop_rate * dt
    lcroot_to_dcroot = livecrootc * p.lwtop_rate * dt
    livestemc  = livestemc  - lstem_to_dstem
    deadstemc  = deadstemc  + lstem_to_dstem
    livecrootc = livecrootc - lcroot_to_dcroot
    deadcrootc = deadcrootc + lcroot_to_dcroot

    # ==================================================================
    # 7. Background mortality — applied to ALL 18 plant pools
    # ==================================================================
    # Whole-tree mortality: every pool loses 2%/yr.  Storage/transfer pools
    # are treated the same as active pools (the whole organism dies).
    # Litter routing: soft-tissue pools → leaf/froot litter fractions;
    # wood pools → CWD.  Storage/xfer follow the same routing as their
    # corresponding active pool type.
    # accel_fac > 1 speeds up dead-wood turnover during accelerated spinup.
    r_mort_dt      = p.r_mort * dt
    r_mort_dt_wood = r_mort_dt * accel_fac   # applied to deadstemc/deadcrootc pools

    # --- active pools ---
    mort_leaf   = leafc       * r_mort_dt
    mort_froot  = frootc      * r_mort_dt
    mort_lstem  = livestemc   * r_mort_dt
    mort_dstem  = deadstemc   * r_mort_dt_wood
    mort_lcroot = livecrootc  * r_mort_dt
    mort_dcroot = deadcrootc  * r_mort_dt_wood

    # --- storage pools ---
    mort_leaf_stor   = leafc_stor        * r_mort_dt
    mort_froot_stor  = frootc_stor       * r_mort_dt
    mort_lstem_stor  = livestemc_stor    * r_mort_dt
    mort_dstem_stor  = deadstemc_stor    * r_mort_dt_wood
    mort_lcroot_stor = livecrootc_stor   * r_mort_dt
    mort_dcroot_stor = deadcrootc_stor   * r_mort_dt_wood

    # --- transfer pools ---
    mort_leaf_xfer   = leafc_xfer        * r_mort_dt
    mort_froot_xfer  = frootc_xfer       * r_mort_dt
    mort_lstem_xfer  = livestemc_xfer    * r_mort_dt
    mort_dstem_xfer  = deadstemc_xfer    * r_mort_dt_wood
    mort_lcroot_xfer = livecrootc_xfer   * r_mort_dt
    mort_dcroot_xfer = deadcrootc_xfer   * r_mort_dt_wood

    leafc       = leafc       - mort_leaf
    frootc      = frootc      - mort_froot
    livestemc   = livestemc   - mort_lstem
    deadstemc   = deadstemc   - mort_dstem
    livecrootc  = livecrootc  - mort_lcroot
    deadcrootc  = deadcrootc  - mort_dcroot

    leafc_stor       = leafc_stor       - mort_leaf_stor
    frootc_stor      = frootc_stor      - mort_froot_stor
    livestemc_stor   = livestemc_stor   - mort_lstem_stor
    deadstemc_stor   = deadstemc_stor   - mort_dstem_stor
    livecrootc_stor  = livecrootc_stor  - mort_lcroot_stor
    deadcrootc_stor  = deadcrootc_stor  - mort_dcroot_stor

    leafc_xfer       = leafc_xfer       - mort_leaf_xfer
    frootc_xfer      = frootc_xfer      - mort_froot_xfer
    livestemc_xfer   = livestemc_xfer   - mort_lstem_xfer
    deadstemc_xfer   = deadstemc_xfer   - mort_dstem_xfer
    livecrootc_xfer  = livecrootc_xfer  - mort_lcroot_xfer
    deadcrootc_xfer  = deadcrootc_xfer  - mort_dcroot_xfer

    # ==================================================================
    # 8. Litter fluxes → soil layers
    # ==================================================================
    # All soft-tissue mortality (active + stor + xfer) → leaf/froot litter.
    # All wood mortality (active + stor + xfer) → CWD.
    total_leaf_litter = (
        leaf_turn_flux + offset_leaf_litter    # one of these is always 0
        + mort_leaf + mort_leaf_stor + mort_leaf_xfer
    ) * inv_dt                                        # gC m⁻² s⁻¹
    lf_lab = total_leaf_litter * p.lf_flab
    lf_cel = total_leaf_litter * p.lf_fcel
    lf_lig = total_leaf_litter * p.lf_flig

    total_froot_litter = (
        froot_turn_flux + offset_froot_litter  # one of these is always 0
        + mort_froot + mort_froot_stor + mort_froot_xfer
    ) * inv_dt
    fr_lab = total_froot_litter * p.fr_flab
    fr_cel = total_froot_litter * p.fr_fcel
    fr_lig = total_froot_litter * p.fr_flig

    cwd_ag = (
        mort_lstem  + mort_dstem
        + mort_lstem_stor  + mort_dstem_stor
        + mort_lstem_xfer  + mort_dstem_xfer
    ) * inv_dt
    cwd_bg = (
        mort_lcroot + mort_dcroot
        + mort_lcroot_stor + mort_dcroot_stor
        + mort_lcroot_xfer + mort_dcroot_xfer
    ) * inv_dt

    # Aboveground litter → exponential surface profile (not just layer 0)
    surf = p.surf_litter_profile   # (1, 1, nlev)
    # Below-ground litter → rootfr-weighted layers.
    litr1_flux = lf_lab.unsqueeze(-1) * surf + fr_lab.unsqueeze(-1) * rootfr
    litr2_flux = lf_cel.unsqueeze(-1) * surf + fr_cel.unsqueeze(-1) * rootfr
    litr3_flux = lf_lig.unsqueeze(-1) * surf + fr_lig.unsqueeze(-1) * rootfr
    cwd_flux_t = cwd_ag.unsqueeze(-1) * surf + cwd_bg.unsqueeze(-1) * rootfr

    # ==================================================================
    # 9. Clamp all pools ≥ 0
    # ==================================================================
    def _c(x: torch.Tensor) -> torch.Tensor:
        return x.clamp(min=0.0)

    new_state = PlantStateTorch(
        leafc=_c(leafc), frootc=_c(frootc),
        livestemc=_c(livestemc), deadstemc=_c(deadstemc),
        livecrootc=_c(livecrootc), deadcrootc=_c(deadcrootc),
        leafc_xfer=_c(leafc_xfer), frootc_xfer=_c(frootc_xfer),
        livestemc_xfer=_c(livestemc_xfer), deadstemc_xfer=_c(deadstemc_xfer),
        livecrootc_xfer=_c(livecrootc_xfer), deadcrootc_xfer=_c(deadcrootc_xfer),
        leafc_stor=_c(leafc_stor), frootc_stor=_c(frootc_stor),
        livestemc_stor=_c(livestemc_stor), deadstemc_stor=_c(deadstemc_stor),
        livecrootc_stor=_c(livecrootc_stor), deadcrootc_stor=_c(deadcrootc_stor),
        pheno_state=pheno, gdd=_c(gdd), onset_day=_c(on_day), offset_day=_c(off_day),
        onset_fdd=_c(onset_fdd),
        onset_swi=_c(onset_swi),
        onset_gddflag=_c(onset_gddflag),
        annsum_npp=annsum_npp,   # unchanged here — caller updates at year end
    )

    litter = LitterFluxTorch(
        litr1_flux=litr1_flux, litr2_flux=litr2_flux,
        litr3_flux=litr3_flux, cwd_flux=cwd_flux_t,
    )
    diag = {
        "gpp": gpp,
        "mr":  mr_paid,
        "gr":  gr_step * inv_dt,
        "npp": alloc_net * inv_dt,
        "onset": onset_event,
        "offset": offset_event,
    }
    return new_state, litter, diag


# ---------------------------------------------------------------------------
# Soil step: CTC decomposition + vertical bioturbation
# ---------------------------------------------------------------------------

def carbon_soil_step_torch(
    state: SoilStateTorch,
    litter: LitterFluxTorch,
    t_soil: torch.Tensor,     # (ncell, nens, nlev)  K
    smp_l: torch.Tensor,      # (ncell, nens, nlev)  mm  (negative = unsaturated)
    dz: torch.Tensor | np.ndarray,   # (nlev,)  m
    params: CarbonParamsTorch,
    dt: float,
    accel_fac: float = 1.0,
    accel_mode: str = "constant",
) -> tuple[SoilStateTorch, torch.Tensor]:
    """Single CTC decomposition + bioturbation step.

    Returns ``(new_state, hr)`` where ``hr`` has shape ``(ncell, nens, nlev)``
    in gC m⁻² s⁻¹.
    """
    p = params
    dev, dtype = p.device, p.dtype

    if not isinstance(dz, torch.Tensor):
        dz = torch.as_tensor(dz, device=dev, dtype=dtype)
    dz = dz.to(device=dev, dtype=dtype)

    # ------------------------------------------------------------------
    # Environmental scalars  (ncell, nens, nlev)
    # ------------------------------------------------------------------
    f_temp  = _f_temp_decomp_torch(t_soil, p.q10_hr, p.q10_hr_at_frz)
    f_water = _f_water_decomp_torch(smp_l)
    rate_scalar = f_temp * f_water * p.depth_scalar   # broadcasts (1,1,nlev)

    accel_mode_norm = str(accel_mode).strip().lower().replace("_", "-")
    if accel_mode_norm == "native":
        accel_mode_norm = "elm-native"
    if accel_mode_norm not in {"constant", "elm-native"}:
        raise ValueError(f"Unsupported carbon acceleration mode: {accel_mode!r}")

    native_factors = (
        elm_native_decomp_spinup_factors(p)
        if accel_mode_norm == "elm-native"
        else None
    )

    def _k_eff_constant(k: torch.Tensor, k_value: float) -> torch.Tensor:
        # Accelerate only genuinely slow pools. If applying accel_fac would make
        # the native first-order turnover faster than one year, leave it native.
        if accel_fac <= 1.0 or k_value * accel_fac > _MAX_ACCEL_DECOMP_RATE:
            return k
        return k * accel_fac

    def _k_eff(k: torch.Tensor, k_value: float, pool_name: str) -> torch.Tensor:
        if native_factors is not None:
            return k * native_factors[pool_name]
        return _k_eff_constant(k, k_value)

    def _decay_flux(pool: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        # Exact first-order decay over dt, bounded to <= pool. This keeps the
        # daily spinup step stable for CLM/ELM rates supplied in day^-1.
        return pool * (-torch.expm1(-(k * rate_scalar * dt)))

    # ------------------------------------------------------------------
    # Decomposition fluxes  (gC m⁻² per step from exact first-order decay)
    # ------------------------------------------------------------------
    cwdc   = state.cwdc
    litr1c = state.litr1c
    litr2c = state.litr2c
    litr3c = state.litr3c
    soil1c = state.soil1c
    soil2c = state.soil2c
    soil3c = state.soil3c
    soil4c = state.soil4c

    cwd_frag  = _decay_flux(cwdc, _k_eff(p.k_frag, p.k_frag_value, "cwd"))
    cwd_to_l2 = cwd_frag * (1.0 - p.cwd_flig)
    cwd_to_l3 = cwd_frag * p.cwd_flig

    l1f = _decay_flux(litr1c, _k_eff(p.k_l1, p.k_l1_value, "litr1"))
    l1_to_s1 = l1f * (1.0 - p.rf_l1s1);  hr_l1 = l1f * p.rf_l1s1

    l2f = _decay_flux(litr2c, _k_eff(p.k_l2, p.k_l2_value, "litr2"))
    l2_to_s2 = l2f * (1.0 - p.rf_l2s2);  hr_l2 = l2f * p.rf_l2s2

    l3f = _decay_flux(litr3c, _k_eff(p.k_l3, p.k_l3_value, "litr3"))
    l3_to_s3 = l3f * (1.0 - p.rf_l3s3);  hr_l3 = l3f * p.rf_l3s3

    s1f = _decay_flux(soil1c, _k_eff(p.k_s1, p.k_s1_value, "soil1"))
    s1_to_s2 = s1f * (1.0 - p.rf_s1s2);  hr_s1 = s1f * p.rf_s1s2

    s2f = _decay_flux(soil2c, _k_eff(p.k_s2, p.k_s2_value, "soil2"))
    s2_to_s3 = s2f * (1.0 - p.rf_s2s3);  hr_s2 = s2f * p.rf_s2s3

    s3f = _decay_flux(soil3c, _k_eff(p.k_s3, p.k_s3_value, "soil3"))
    s3_to_s4 = s3f * (1.0 - p.rf_s3s4);  hr_s3 = s3f * p.rf_s3s4

    s4f = _decay_flux(soil4c, _k_eff(p.k_s4, p.k_s4_value, "soil4"))
    hr_s4 = s4f

    hr = (hr_l1 + hr_l2 + hr_l3 + hr_s1 + hr_s2 + hr_s3 + hr_s4) / dt

    # ------------------------------------------------------------------
    # Pool updates
    # ------------------------------------------------------------------
    cwdc   = cwdc   + litter.cwd_flux   * dt - cwd_frag
    litr1c = litr1c + litter.litr1_flux * dt - l1f
    litr2c = litr2c + litter.litr2_flux * dt + cwd_to_l2 - l2f
    litr3c = litr3c + litter.litr3_flux * dt + cwd_to_l3 - l3f
    soil1c = soil1c + l1_to_s1 - s1f
    soil2c = soil2c + l2_to_s2 + s1_to_s2 - s2f
    soil3c = soil3c + l3_to_s3 + s2_to_s3 - s3f
    soil4c = soil4c + s3_to_s4 - s4f

    # ------------------------------------------------------------------
    # Bioturbation diffusion (explicit, all 8 pools)
    # ------------------------------------------------------------------
    D = p.som_diffus_value
    if D > 0.0:
        dz_iface = 0.5 * (dz[:-1] + dz[1:])   # (nlev-1,)

        def _diffuse(pool: torch.Tensor, D_eff: float = D) -> torch.Tensor:
            """Explicit diffusion step, pool shape (ncell, nens, nlev)."""
            conc = pool / dz.unsqueeze(0).unsqueeze(0)          # gC m⁻³
            # Inter-layer flux: D × dC/dz  (positive = downward)
            flux = D_eff * (conc[..., :-1] - conc[..., 1:]) / dz_iface   # (…, nlev-1)
            dp = torch.zeros_like(pool)
            dp[..., :-1] -= flux * dt
            dp[..., 1:]  += flux * dt
            return pool + dp

        if native_factors is not None:
            cwdc   = _diffuse(cwdc)
            litr1c = _diffuse(litr1c, D_eff=D * native_factors["litr1"])
            litr2c = _diffuse(litr2c, D_eff=D * native_factors["litr2"])
            litr3c = _diffuse(litr3c, D_eff=D * native_factors["litr3"])
            soil1c = _diffuse(soil1c, D_eff=D * native_factors["soil1"])
            soil2c = _diffuse(soil2c, D_eff=D * native_factors["soil2"])
            soil3c = _diffuse(soil3c, D_eff=D * native_factors["soil3"])
            soil4c = _diffuse(soil4c, D_eff=D * native_factors["soil4"])
        else:
            cwdc   = _diffuse(cwdc)
            litr1c = _diffuse(litr1c)
            litr2c = _diffuse(litr2c)
            litr3c = _diffuse(litr3c)
            soil1c = _diffuse(soil1c)
            soil2c = _diffuse(soil2c)
            soil3c = _diffuse(soil3c)
            soil4c = _diffuse(soil4c, D_eff=D * accel_fac)

    def _c(x: torch.Tensor) -> torch.Tensor:
        return x.clamp(min=0.0)

    new_state = SoilStateTorch(
        cwdc=_c(cwdc), litr1c=_c(litr1c), litr2c=_c(litr2c), litr3c=_c(litr3c),
        soil1c=_c(soil1c), soil2c=_c(soil2c), soil3c=_c(soil3c), soil4c=_c(soil4c),
    )
    return new_state, hr


# ---------------------------------------------------------------------------
# Combined step
# ---------------------------------------------------------------------------

def carbon_step_torch(
    plant_state: PlantStateTorch,
    soil_state: SoilStateTorch,
    gpp: torch.Tensor,        # (ncell, nens)
    tref: torch.Tensor,       # (ncell, nens)
    t10: torch.Tensor,        # (ncell, nens)
    t_soil: torch.Tensor,     # (ncell, nens, nlev)
    smp_l: torch.Tensor,      # (ncell, nens, nlev)  mm
    dayl: torch.Tensor,       # (ncell, nens)
    rootfr: torch.Tensor,     # (1, 1, nlev) or (ncell, nens, nlev)
    dz: torch.Tensor | np.ndarray,   # (nlev,)
    params: CarbonParamsTorch,
    dt: float,
    prev_dayl: torch.Tensor | None = None,
) -> CarbonStepResultTorch:
    """Run one coupled plant + soil carbon step."""
    new_plant, litter, pdiag = carbon_plant_step_torch(
        plant_state, gpp, tref, t10, dayl, rootfr, params, dt,
        prev_dayl=prev_dayl,
    )
    new_soil, hr = carbon_soil_step_torch(
        soil_state, litter, t_soil, smp_l, dz, params, dt,
    )
    hr_col = hr.sum(dim=-1)   # (ncell, nens)
    mr = pdiag["mr"]; gr = pdiag["gr"]; npp = pdiag["npp"]
    return CarbonStepResultTorch(
        plant_state=new_plant, soil_state=new_soil,
        gpp=gpp, npp=npp, mr=mr, gr=gr,
        hr=hr_col, nee=hr_col + mr + gr - gpp,
    )
