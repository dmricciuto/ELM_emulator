"""Regional NetCDF diagnostic output helpers."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

from .regional_time import _days_in_month

_WATER_DIAG_NLEV = 10
_WATER_STORAGE_DIAG_KEYS = (
    "h2osoi_liq_total_mean",
    "h2osoi_ice_total_mean",
    "h2ocan_mean",
    "wa_mean",
    "tws_mean",
)
_WATER_LAYER_DIAG_KEYS = tuple(
    key
    for prefix in ("smp_l", "rresis", "rootr", "qflx_rootsoi")
    for lev in range(1, _WATER_DIAG_NLEV + 1)
    for key in (f"{prefix}_lev{lev}_mean",)
)
_REGIONAL_DAILY_DIAG_KEYS = (
    "forc_t_mean",
    "forc_rain_mean",
    "forc_snow_mean",
    "fsds_mean",
    "tg_mean",
    "eflx_fgr12_mean",
    "hs_soil_mean",
    "hs_top_snow_mean",
    "hs_h2osfc_mean",
    "dhsdT_mean",
    "fpsn_mean",
    "gpp_gc_m2_d_mean",
    "lai_mean",
    "t_soil_mean",
    "t_soisno_lev1_mean",
    "t_soisno_lev3_mean",
    "t_soisno_lev5_mean",
    "t_veg_mean",
    "liq_mean",
    "h2osoi_liqvol_lev1_mean",
    "h2osoi_liqvol_lev2_mean",
    "h2osoi_liqvol_lev3_mean",
    "h2osoi_liqvol_lev4_mean",
    "h2osoi_liqvol_lev5_mean",
    "h2osfc_mean",
    "h2osno_mean",
    "frac_sno_mean",
    "active_frac_sno_mean",
    "zwt_mean",
    "btran_mean",
    "qvegt_mean",
    "qvege_mean",
    "qsoil_mean",
    "qflx_tran_veg_mean",
    "qflx_ev_soil_mean",
    "qflx_evap_soi_mean",
    "eflx_sh_tot_mean",
    "qflx_surf_mean",
    "qflx_h2osfc_surf_mean",
    "qflx_drain_mean",
    "qflx_deep_drain_mean",
    "qflx_runoff_tot_mean",
    "qflx_evap_tot_mean",
    "eflx_lh_tot_mean",
    "snow_depth_mean",
    "active_snow_depth_mean",
    "snow_depth_eff_mean",
) + _WATER_STORAGE_DIAG_KEYS + _WATER_LAYER_DIAG_KEYS
_REGIONAL_DAILY_ENSEMBLE_DIAG_KEYS = (
    "fpsn_ens_std",
    "qflx_runoff_tot_ens_std",
)
_REGIONAL_CARBON_DIAG_KEYS = (
    "carbon_gpp_daily",
    "carbon_npp_daily",
    "carbon_hr_daily",
    "carbon_gpp_daily_mean",
    "carbon_npp_daily_mean",
    "carbon_hr_daily_mean",
    "carbon_totvegc",
    "carbon_totsomc",
    "carbon_deadstemc",
    "carbon_totecosysc",
)
_REGIONAL_TRAIT_STD_TARGET_MAP = {
    "fpsn_patch": "fpsn_trait_std",
    "eflx_sh_tot": "eflx_sh_tot_trait_std",
    "qflx_tran_veg": "qflx_tran_veg_trait_std",
    "qflx_evap_canopy": "qflx_evap_canopy_trait_std",
    "qflx_ev_soil": "qflx_ev_soil_trait_std",
    "qflx_ev_snow": "qflx_ev_snow_trait_std",
    "qflx_ev_h2osfc": "qflx_ev_h2osfc_trait_std",
    "t_veg": "t_veg_trait_std",
    "t_ref2m": "t_ref2m_trait_std",
    "hs_soil": "hs_soil_trait_std",
    "hs_top_snow": "hs_top_snow_trait_std",
    "hs_h2osfc": "hs_h2osfc_trait_std",
    "dhsdT": "dhsdT_trait_std",
    "laisun_patch": "laisun_patch_trait_std",
    "laisha_patch": "laisha_patch_trait_std",
    "psnsun_patch": "psnsun_patch_trait_std",
    "psnsha_patch": "psnsha_patch_trait_std",
}
_REGIONAL_DAILY_TRAIT_DIAG_KEYS = tuple(_REGIONAL_TRAIT_STD_TARGET_MAP.values())
_REGIONAL_DAY_NIGHT_FLUX_OUTPUT_KEYS = (
    "gpp_gc_m2_d_day_mean",
    "gpp_gc_m2_d_night_mean",
    "qflx_evap_tot_day_mean",
    "qflx_evap_tot_night_mean",
    "eflx_sh_tot_day_mean",
    "eflx_sh_tot_night_mean",
    "daytime_sample_count",
    "nighttime_sample_count",
)
_REGIONAL_DAILY_OUTPUT_KEYS = (
    _REGIONAL_DAILY_DIAG_KEYS
    + _REGIONAL_DAILY_ENSEMBLE_DIAG_KEYS
    + _REGIONAL_CARBON_DIAG_KEYS
    + _REGIONAL_DAILY_TRAIT_DIAG_KEYS
    + _REGIONAL_DAY_NIGHT_FLUX_OUTPUT_KEYS
)

# ---------------------------------------------------------------------------
# Regional diagnostic output
# ---------------------------------------------------------------------------

def _write_regional_daily_output(
    output_path: "Path",
    *,
    surf: dict,
    year: int,
    month: int,
    daily_diag: dict[str, np.ndarray],
    append: bool,
    calendar: str = "gregorian",
    frequency: str = "daily",
    period_index: list[tuple[int, int, int, int]] | None = None,
    day_night_fsds_threshold: float | None = None,
) -> None:
    """Write daily, 5-day, or monthly regional diagnostics to one appendable NetCDF file."""
    if not daily_diag:
        return
    frequency = str(frequency).lower()
    if frequency not in {"daily", "5day", "monthly"}:
        raise ValueError(f"Unsupported regional output frequency: {frequency}")
    if period_index is not None and frequency != "5day":
        raise ValueError("period_index is only supported for frequency='5day'")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nday = int(next(iter(daily_diag.values())).shape[0])
    ncell = int(np.asarray(surf["lat_deg"]).size)
    mode = "a" if append and output_path.exists() else "w"

    metadata = {
        "forc_t_mean": ("mean 2-m air temperature", "K"),
        "forc_rain_mean": ("mean rain forcing", "kg m-2 s-1"),
        "forc_snow_mean": ("mean snow forcing", "kg m-2 s-1"),
        "fsds_mean": ("mean incoming shortwave radiation", "W m-2"),
        "tg_mean": ("mean ground temperature", "K"),
        "eflx_fgr12_mean": ("mean ground heat flux into soil/snow", "W m-2"),
        "hs_soil_mean": ("mean surface heat flux into soil", "W m-2"),
        "hs_top_snow_mean": ("mean surface heat flux into top snow layer", "W m-2"),
        "hs_h2osfc_mean": ("mean surface heat flux into surface water", "W m-2"),
        "dhsdT_mean": ("mean surface heat flux temperature derivative", "W m-2 K-1"),
        "fpsn_mean": ("mean net photosynthesis", "umol CO2 m-2 s-1"),
        "gpp_gc_m2_d_mean": ("mean GPP proxy converted from fpsn_patch", "gC m-2 d-1"),
        "fpsn_ens_std": ("ensemble standard deviation of net photosynthesis", "umol CO2 m-2 s-1"),
        "lai_mean": ("mean area-weighted leaf area index", "m2 m-2"),
        "t_soil_mean": ("mean top-layer soil temperature", "K"),
        "t_soisno_lev1_mean": ("mean layer-1 soil temperature", "K"),
        "t_soisno_lev3_mean": ("mean layer-3 soil temperature", "K"),
        "t_soisno_lev5_mean": ("mean layer-5 soil temperature", "K"),
        "t_veg_mean": ("mean vegetation temperature", "K"),
        "liq_mean": ("mean top-layer volumetric soil water", "m3 m-3"),
        "h2osoi_liqvol_lev1_mean": ("mean layer-1 volumetric liquid soil water", "m3 m-3"),
        "h2osoi_liqvol_lev2_mean": ("mean layer-2 volumetric liquid soil water", "m3 m-3"),
        "h2osoi_liqvol_lev3_mean": ("mean layer-3 volumetric liquid soil water", "m3 m-3"),
        "h2osoi_liqvol_lev4_mean": ("mean layer-4 volumetric liquid soil water", "m3 m-3"),
        "h2osoi_liqvol_lev5_mean": ("mean layer-5 volumetric liquid soil water", "m3 m-3"),
        "h2osfc_mean": ("mean surface water storage", "mm"),
        "h2osno_mean": ("mean snow water equivalent", "kg m-2"),
        "frac_sno_mean": ("mean raw snow-covered fraction", "1"),
        "active_frac_sno_mean": ("mean active snow-covered fraction used by replay features", "1"),
        "zwt_mean": ("mean water table depth", "m"),
        "btran_mean": ("mean transpiration beta factor", "1"),
        "qvegt_mean": ("mean canopy transpiration", "mm s-1"),
        "qvege_mean": ("mean canopy evaporation", "mm s-1"),
        "qsoil_mean": ("mean ELM-style ground evaporation", "mm s-1"),
        "qflx_tran_veg_mean": ("mean vegetation transpiration flux", "kg m-2 s-1"),
        "qflx_ev_soil_mean": ("mean unweighted bare-soil evaporation component", "kg m-2 s-1"),
        "qflx_evap_soi_mean": ("mean weighted ground evaporation flux", "kg m-2 s-1"),
        "eflx_sh_tot_mean": ("mean sensible heat flux", "W m-2"),
        "qflx_surf_mean": ("mean surface runoff flux", "kg m-2 s-1"),
        "qflx_h2osfc_surf_mean": ("mean surface-water runoff flux", "kg m-2 s-1"),
        "qflx_drain_mean": ("mean subsurface drainage flux", "kg m-2 s-1"),
        "qflx_deep_drain_mean": ("mean deep drainage flux", "kg m-2 s-1"),
        "qflx_runoff_tot_mean": ("mean total runoff flux", "kg m-2 s-1"),
        "qflx_runoff_tot_ens_std": ("ensemble standard deviation of total runoff flux", "kg m-2 s-1"),
        "qflx_evap_tot_mean": ("mean ELM-style total evaporation flux", "kg m-2 s-1"),
        "eflx_lh_tot_mean": ("mean ELM-style total latent heat flux", "W m-2"),
        "gpp_gc_m2_d_day_mean": (
            "daytime mean GPP proxy converted from fpsn_patch where FSDS exceeds the day/night threshold",
            "gC m-2 d-1",
        ),
        "gpp_gc_m2_d_night_mean": (
            "nighttime mean GPP proxy converted from fpsn_patch where FSDS is at or below the day/night threshold",
            "gC m-2 d-1",
        ),
        "qflx_evap_tot_day_mean": (
            "daytime mean ELM-style total evaporation flux where FSDS exceeds the day/night threshold",
            "kg m-2 s-1",
        ),
        "qflx_evap_tot_night_mean": (
            "nighttime mean ELM-style total evaporation flux where FSDS is at or below the day/night threshold",
            "kg m-2 s-1",
        ),
        "eflx_sh_tot_day_mean": (
            "daytime mean sensible heat flux where FSDS exceeds the day/night threshold",
            "W m-2",
        ),
        "eflx_sh_tot_night_mean": (
            "nighttime mean sensible heat flux where FSDS is at or below the day/night threshold",
            "W m-2",
        ),
        "daytime_sample_count": (
            "number of hourly diagnostic samples with FSDS exceeding the day/night threshold",
            "1",
        ),
        "nighttime_sample_count": (
            "number of hourly diagnostic samples with FSDS at or below the day/night threshold",
            "1",
        ),
        "snow_depth_mean": ("mean raw snow depth", "m"),
        "active_snow_depth_mean": ("mean active snow depth used by replay features", "m"),
        "snow_depth_eff_mean": ("mean snow-fraction-weighted snow depth", "m"),
        "h2osoi_liq_total_mean": ("mean total soil liquid water storage over hydrology layers", "mm"),
        "h2osoi_ice_total_mean": ("mean total soil ice storage over hydrology layers", "mm"),
        "h2ocan_mean": ("mean canopy-intercepted water storage", "mm"),
        "wa_mean": ("mean unconfined aquifer water storage", "mm"),
        "tws_mean": ("mean replay total water storage: soil liquid+ice, canopy, surface, snow, aquifer", "mm"),
        "carbon_gpp_daily": ("online carbon daily gross primary production", "gC m-2 d-1"),
        "carbon_npp_daily": ("online carbon daily net primary production", "gC m-2 d-1"),
        "carbon_hr_daily": ("online carbon daily heterotrophic respiration", "gC m-2 d-1"),
        "carbon_gpp_daily_mean": ("monthly mean online carbon daily gross primary production", "gC m-2 d-1"),
        "carbon_npp_daily_mean": ("monthly mean online carbon daily net primary production", "gC m-2 d-1"),
        "carbon_hr_daily_mean": ("monthly mean online carbon daily heterotrophic respiration", "gC m-2 d-1"),
        "carbon_totvegc": ("online carbon total vegetation carbon", "gC m-2"),
        "carbon_totsomc": ("online carbon total soil organic matter carbon, soil1c through soil4c", "gC m-2"),
        "carbon_deadstemc": ("online carbon active dead stem carbon", "gC m-2"),
        "carbon_totecosysc": ("online carbon total ecosystem carbon, vegetation plus CWD, litter, and SOM", "gC m-2"),
    }
    metadata.update({
        "fpsn_trait_std": (
            "same-state independent-PFT trait standard deviation of net photosynthesis",
            "umol CO2 m-2 s-1",
        ),
        "eflx_sh_tot_trait_std": (
            "same-state independent-PFT trait standard deviation of sensible heat flux",
            "W m-2",
        ),
        "qflx_tran_veg_trait_std": (
            "same-state independent-PFT trait standard deviation of vegetation transpiration flux",
            "kg m-2 s-1",
        ),
        "qflx_evap_canopy_trait_std": (
            "same-state independent-PFT trait standard deviation of canopy evaporation MLP target",
            "kg m-2 s-1",
        ),
        "qflx_ev_soil_trait_std": (
            "same-state independent-PFT trait standard deviation of bare-soil evaporation component",
            "kg m-2 s-1",
        ),
        "qflx_ev_snow_trait_std": (
            "same-state independent-PFT trait standard deviation of snow evaporation component",
            "kg m-2 s-1",
        ),
        "qflx_ev_h2osfc_trait_std": (
            "same-state independent-PFT trait standard deviation of surface-water evaporation component",
            "kg m-2 s-1",
        ),
        "t_veg_trait_std": (
            "same-state independent-PFT trait standard deviation of vegetation temperature",
            "K",
        ),
        "t_ref2m_trait_std": (
            "same-state independent-PFT trait standard deviation of 2-m reference air temperature",
            "K",
        ),
        "hs_soil_trait_std": (
            "same-state independent-PFT trait standard deviation of ground heat flux target",
            "W m-2",
        ),
        "hs_top_snow_trait_std": (
            "same-state independent-PFT trait standard deviation of top-snow heat flux target",
            "W m-2",
        ),
        "hs_h2osfc_trait_std": (
            "same-state independent-PFT trait standard deviation of surface-water heat flux target",
            "W m-2",
        ),
        "dhsdT_trait_std": (
            "same-state independent-PFT trait standard deviation of heat-flux temperature derivative target",
            "W m-2 K-1",
        ),
        "laisun_patch_trait_std": (
            "same-state independent-PFT trait standard deviation of sunlit LAI target",
            "m2 m-2",
        ),
        "laisha_patch_trait_std": (
            "same-state independent-PFT trait standard deviation of shaded LAI target",
            "m2 m-2",
        ),
        "psnsun_patch_trait_std": (
            "same-state independent-PFT trait standard deviation of sunlit photosynthesis",
            "umol CO2 m-2 s-1",
        ),
        "psnsha_patch_trait_std": (
            "same-state independent-PFT trait standard deviation of shaded photosynthesis",
            "umol CO2 m-2 s-1",
        ),
    })
    for lev in range(1, _WATER_DIAG_NLEV + 1):
        metadata[f"smp_l_lev{lev}_mean"] = (
            f"mean layer-{lev} soil matric potential used for BTRAN",
            "mm",
        )
        metadata[f"rresis_lev{lev}_mean"] = (
            f"mean layer-{lev} area-weighted root moisture resistance factor",
            "1",
        )
        metadata[f"rootr_lev{lev}_mean"] = (
            f"mean layer-{lev} effective root uptake fraction diagnosed from qflx_rootsoi/qflx_tran_veg",
            "1",
        )
        metadata[f"qflx_rootsoi_lev{lev}_mean"] = (
            f"mean layer-{lev} root water extraction flux passed to hydrology",
            "kg m-2 s-1",
        )

    def _five_day_mean(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        rows = []
        for start_day in range(0, nday, 5):
            rows.append(np.nanmean(values[start_day:start_day + 5], axis=0))
        return np.stack(rows, axis=0).astype(np.float32, copy=False)

    if frequency == "daily":
        output_keys = tuple(key for key in _REGIONAL_DAILY_OUTPUT_KEYS if key in daily_diag)
        output_diag = {
            key: np.asarray(daily_diag[key], dtype=np.float32)
            for key in output_keys
        }
        nout = nday
    elif frequency == "5day":
        output_keys = tuple(key for key in _REGIONAL_DAILY_OUTPUT_KEYS if key in daily_diag)
        if period_index is None:
            output_diag = {
                key: _five_day_mean(np.asarray(daily_diag[key], dtype=np.float32))
                for key in output_keys
            }
            nout = int(math.ceil(nday / 5.0))
        else:
            nout = len(period_index)
            if nout <= 0:
                return
            output_diag = {
                key: np.asarray(daily_diag[key], dtype=np.float32)
                for key in output_keys
            }
    else:
        output_keys = tuple(key for key in _REGIONAL_DAILY_OUTPUT_KEYS if key in daily_diag)
        output_diag = {
            key: np.nanmean(np.asarray(daily_diag[key], dtype=np.float32), axis=0, keepdims=True)
            for key in output_keys
        }
        nout = 1

    def _ensure_output_var(ds: "Dataset", key: str, ncell: int) -> None:
        if key in ds.variables:
            return
        long_name, units = metadata[key]
        var = ds.createVariable(
            key,
            "f4",
            ("time", "cell"),
            chunksizes=(1, ncell),
            fill_value=np.float32(np.nan),
        )
        var.long_name = long_name
        var.units = units
        var.cell_methods = "time: mean"

    with Dataset(output_path, mode) as ds:
        if mode == "w":
            ds.createDimension("time", None)
            ds.createDimension("cell", ncell)
            ds.title = f"Regional surface emulator {frequency} diagnostic means"
            ds.history = "Created by benchmark_replay.py"
            ds.output_frequency = frequency
            if day_night_fsds_threshold is not None:
                ds.day_night_fsds_threshold = float(day_night_fsds_threshold)
                ds.day_night_fsds_threshold_units = "W m-2"

            gridcell_var = ds.createVariable("gridcell_index", "i4", ("cell",))
            gridcell_var.long_name = "1-based surfdata gridcell index"
            gridcell_var[:] = np.asarray(surf["active_idx"], dtype=np.int32) + 1

            lat_var = ds.createVariable("lat", "f4", ("cell",))
            lat_var.units = "degrees_north"
            lat_var[:] = np.asarray(surf["lat_deg"], dtype=np.float32)

            lon_var = ds.createVariable("lon", "f4", ("cell",))
            lon_var.units = "degrees_east"
            lon_var[:] = np.asarray(surf["lon_deg"], dtype=np.float32)

            row_var = ds.createVariable("grid_row", "i4", ("cell",))
            row_var.long_name = "met/surfdata grid row index"
            row_var[:] = np.asarray(surf["grid_row"], dtype=np.int32)

            col_var = ds.createVariable("grid_col", "i4", ("cell",))
            col_var.long_name = "met/surfdata grid column index"
            col_var[:] = np.asarray(surf["grid_col"], dtype=np.int32)

            if "met_grid_row" in surf:
                met_row_var = ds.createVariable("met_grid_row", "i4", ("cell",))
                met_row_var.long_name = "forcing source grid row index"
                met_row_var[:] = np.asarray(surf["met_grid_row"], dtype=np.int32)

            if "met_grid_col" in surf:
                met_col_var = ds.createVariable("met_grid_col", "i4", ("cell",))
                met_col_var.long_name = "forcing source grid column index"
                met_col_var[:] = np.asarray(surf["met_grid_col"], dtype=np.int32)

            clone_cell_metadata = {
                "synthetic_flnr_factor": ("multiplicative flnr perturbation factor", "1"),
                "synthetic_mbbopt_factor": ("multiplicative mbbopt perturbation factor", "1"),
                "synthetic_pct_sand_delta": ("additive PCT_SAND perturbation", "percentage points"),
                "synthetic_pct_sand_lev1": ("perturbed layer-1 PCT_SAND", "%"),
                "synthetic_pct_sand_mean": ("perturbed mean PCT_SAND over hydrology layers", "%"),
                "precip_treatment_area_fraction": ("fraction of cell area inside precipitation treatment", "1"),
                "precip_treatment_multiplier": ("multiplicative precipitation forcing factor", "1"),
            }
            for key, (long_name, units) in clone_cell_metadata.items():
                if key not in surf:
                    continue
                var = ds.createVariable(key, "f4", ("cell",))
                var.long_name = long_name
                var.units = units
                var[:] = np.asarray(surf[key], dtype=np.float32)
            if "synthetic_clone_grid_rows" in surf:
                ds.synthetic_clone_grid_rows = int(surf["synthetic_clone_grid_rows"])
                ds.synthetic_clone_grid_cols = int(surf["synthetic_clone_grid_cols"])
                ds.synthetic_clone_corr_length_cells = float(surf["synthetic_clone_corr_length_cells"])
                ds.synthetic_clone_seed = int(surf["synthetic_clone_seed"])
            for key in (
                "precip_treatment_area_ha_requested",
                "precip_treatment_area_ha_effective",
                "precip_treatment_reduction",
                "precip_treatment_cell_size_m",
                "precip_treatment_side_m",
                "precip_treatment_center_x_m",
                "precip_treatment_center_y_m",
                "precip_treatment_start_date",
                "precip_treatment_end_date",
                "precip_treatment_cell_count_any_overlap",
                "precip_treatment_cell_equivalent_count",
            ):
                if key in surf:
                    setattr(ds, key, surf[key])

            date_var = ds.createVariable("date", "i4", ("time",))
            if frequency == "monthly":
                date_var.long_name = "date as YYYYMM"
            elif frequency == "5day":
                date_var.long_name = "period start date as YYYYMMDD"
            else:
                date_var.long_name = "date as YYYYMMDD"
            ds.createVariable("year", "i4", ("time",))
            ds.createVariable("month", "i4", ("time",))
            ds.createVariable("day", "i4", ("time",))
            period_length_var = ds.createVariable("period_length_days", "i4", ("time",))
            period_length_var.long_name = "number of days averaged into this record"
            period_length_var.units = "days"

            for key in output_keys:
                _ensure_output_var(ds, key, ncell)
        else:
            existing_frequency = str(getattr(ds, "output_frequency", "daily")).lower()
            if existing_frequency != frequency:
                raise ValueError(
                    f"{output_path} was created with output_frequency={existing_frequency!r}, "
                    f"but current run requested {frequency!r}"
                )
            existing_ncell = len(ds.dimensions["cell"])
            if existing_ncell != ncell:
                raise ValueError(
                    f"{output_path} has cell dimension {existing_ncell}, "
                    f"but current run has {ncell}"
                )
            for key in output_keys:
                _ensure_output_var(ds, key, ncell)

        start = len(ds.dimensions["time"])
        days_in_month = _days_in_month(year, month, calendar)
        days = np.arange(1, nday + 1, dtype=np.int32)
        if period_index is None and nday > days_in_month:
            raise ValueError(f"daily diagnostics have {nday} days for {year}-{month:02d}")
        if frequency == "daily":
            ds.variables["year"][start:start + nout] = year
            ds.variables["month"][start:start + nout] = month
            ds.variables["day"][start:start + nout] = days
            ds.variables["date"][start:start + nout] = year * 10000 + month * 100 + days
            if "period_length_days" not in ds.variables:
                period_length_var = ds.createVariable("period_length_days", "i4", ("time",))
                period_length_var.long_name = "number of days averaged into this record"
                period_length_var.units = "days"
            ds.variables["period_length_days"][start:start + nout] = np.ones(nout, dtype=np.int32)
        elif frequency == "5day":
            if period_index is None:
                period_start_days = np.arange(1, nday + 1, 5, dtype=np.int32)
                period_lengths = np.minimum(5, nday - (period_start_days - 1)).astype(np.int32)
                period_years = np.full(nout, year, dtype=np.int32)
                period_months = np.full(nout, month, dtype=np.int32)
                period_days = period_start_days
            else:
                period_records = np.asarray(period_index, dtype=np.int32)
                if period_records.shape != (nout, 4):
                    raise ValueError(
                        f"period_index has shape {period_records.shape}, expected {(nout, 4)}"
                    )
                period_years = period_records[:, 0]
                period_months = period_records[:, 1]
                period_days = period_records[:, 2]
                period_lengths = period_records[:, 3]
            ds.variables["year"][start:start + nout] = period_years
            ds.variables["month"][start:start + nout] = period_months
            ds.variables["day"][start:start + nout] = period_days
            ds.variables["date"][start:start + nout] = (
                period_years * 10000 + period_months * 100 + period_days
            )
            if "period_length_days" not in ds.variables:
                period_length_var = ds.createVariable("period_length_days", "i4", ("time",))
                period_length_var.long_name = "number of days averaged into this record"
                period_length_var.units = "days"
            ds.variables["period_length_days"][start:start + nout] = period_lengths
        else:
            ds.variables["year"][start:start + nout] = np.asarray([year], dtype=np.int32)
            ds.variables["month"][start:start + nout] = np.asarray([month], dtype=np.int32)
            ds.variables["day"][start:start + nout] = np.asarray([0], dtype=np.int32)
            ds.variables["date"][start:start + nout] = np.asarray([year * 100 + month], dtype=np.int32)
            if "period_length_days" not in ds.variables:
                period_length_var = ds.createVariable("period_length_days", "i4", ("time",))
                period_length_var.long_name = "number of days averaged into this record"
                period_length_var.units = "days"
            ds.variables["period_length_days"][start:start + nout] = np.asarray([nday], dtype=np.int32)

        for key in output_keys:
            values = np.asarray(output_diag[key], dtype=np.float32)
            if values.shape != (nout, ncell):
                raise ValueError(
                    f"{key} {frequency} diagnostics have shape {values.shape}, "
                    f"expected {(nout, ncell)}"
                )
            ds.variables[key][start:start + nout, :] = values

    print(f"  saved {frequency} diagnostics {output_path} ({nout} records)")


def _write_regional_single_cell_timeseries(
    output_path: "Path",
    *,
    records: dict[str, list[np.ndarray]],
    selected_cell_info: dict | None,
) -> None:
    """Write hourly single-cell diagnostics accumulated across months."""
    if not records:
        return
    max_records = max((len(values) for values in records.values() if values), default=0)
    if max_records <= 0:
        return
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data: dict[str, np.ndarray] = {}
    for key, values in records.items():
        if not values:
            continue
        if len(values) < max_records:
            sample = np.asarray(values[-1], dtype=np.float64)
            pad = np.full_like(sample, np.nan, dtype=np.float64)
            values = [*values, *[pad.copy() for _ in range(max_records - len(values))]]
        arr = np.stack(values, axis=0)
        if arr.ndim == 3 and arr.shape[1] == 1:
            arr = arr[:, 0, :]
        else:
            arr = arr.reshape(arr.shape[0], -1)
        data[key] = np.asarray(arr)

    ntime = int(next(iter(data.values())).shape[0])
    nens = int(next(iter(data.values())).shape[1])
    time_keys = {"hour_index", "year", "month", "day", "hour_of_day"}
    int_keys = time_keys | {"physics_step", "snow_snl"}

    with Dataset(output_path, "w") as ds:
        ds.createDimension("time", ntime)
        ds.createDimension("ensemble", nens)
        ds.title = "Regional single-cell hourly replay diagnostics"
        ds.history = "Created by benchmark_replay.py"
        if selected_cell_info is not None:
            ds.lat = float(selected_cell_info["lat"])
            ds.lon = float(selected_cell_info["lon"])
            ds.active_cell = int(selected_cell_info.get("active_cell", -1))

        for key in ("hour_index", "year", "month", "day", "hour_of_day"):
            if key not in data:
                continue
            var = ds.createVariable(key, "i4", ("time",))
            var[:] = data[key][:, 0].astype(np.int32, copy=False)

        for key, arr in data.items():
            if key in time_keys:
                continue
            if key in int_keys or key.startswith("imelt_lev"):
                var = ds.createVariable(key, "i4", ("time", "ensemble"), fill_value=np.int32(-9999))
                var[:] = np.where(np.isfinite(arr), arr, -9999).astype(np.int32, copy=False)
            else:
                var = ds.createVariable(key, "f4", ("time", "ensemble"), fill_value=np.float32(np.nan))
                var[:] = arr.astype(np.float32, copy=False)
                if key == "mlp_ood_max_abs_z":
                    var.long_name = "MLP input maximum absolute standardized anomaly"
                    var.units = "training sigma"
                elif key == "mlp_ood_frac":
                    var.long_name = "Fraction of MLP inputs beyond OOD threshold"
                    var.units = "fraction"
                elif key.startswith("mlp_ood_abs_z_"):
                    var.long_name = "MLP input absolute standardized anomaly"
                    var.units = "training sigma"
                    var.source_feature_name = key[len("mlp_ood_abs_z_"):]

    print(f"  saved single-cell hourly diagnostics {output_path} ({ntime} hours)")
