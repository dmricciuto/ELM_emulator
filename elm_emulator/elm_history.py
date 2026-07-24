"""ELM-style h0 history output for emulator regional replays."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

from .elm_hydrology import elm_soil_grid
from .regional_time import _days_in_month

_FILL_FLOAT = np.float32(1.0e36)
_FILL_INT = np.int32(-9999)
_BASE_YEAR = 1850
_SECSPDAY = 86400.0
_LV = 2.501e6
_PeriodIndex = tuple[int, int, int, int] | tuple[int, int, int, int, int]


def infer_elm_history_case_name(output_dir: Path) -> str:
    """Infer a case name from an ELM/OLMT run directory path."""
    output_dir = Path(output_dir)
    if output_dir.name == "run" and output_dir.parent.name:
        return output_dir.parent.name
    return output_dir.name or "elm_emulator"


def _days_since_base(year: int, month: int, day: int, calendar: str) -> int:
    year = int(year)
    month = int(month)
    day = int(day)
    if calendar == "noleap":
        days = (year - _BASE_YEAR) * 365
        days += sum(_days_in_month(year, m, calendar) for m in range(1, month))
        return days + day - 1
    import datetime as _dt

    return (_dt.date(year, month, day) - _dt.date(_BASE_YEAR, 1, 1)).days


def _ymd_from_days_since_base(offset: int, calendar: str) -> tuple[int, int, int]:
    offset = int(offset)
    if calendar == "noleap":
        year = _BASE_YEAR + offset // 365
        doy0 = offset % 365
        month = 1
        while True:
            nd = _days_in_month(year, month, calendar)
            if doy0 < nd:
                return year, month, doy0 + 1
            doy0 -= nd
            month += 1
    import datetime as _dt

    date = _dt.date(_BASE_YEAR, 1, 1) + _dt.timedelta(days=offset)
    return date.year, date.month, date.day


def _seconds_since_base(year: int, month: int, day: int, second_of_day: int, calendar: str) -> int:
    second = int(second_of_day)
    if second < 0 or second >= int(_SECSPDAY):
        raise ValueError(f"second_of_day must be in [0, 86400), got {second_of_day}")
    return _days_since_base(year, month, day, calendar) * int(_SECSPDAY) + second


def _ymd_sec_from_seconds_since_base(offset_seconds: int, calendar: str) -> tuple[int, int, int, int]:
    day_offset, second = divmod(int(offset_seconds), int(_SECSPDAY))
    year, month, day = _ymd_from_days_since_base(day_offset, calendar)
    return year, month, day, int(second)


def _period_start_year(period: _PeriodIndex) -> int:
    return int(period[0])


def _period_seconds(period: _PeriodIndex, calendar: str) -> tuple[int, int]:
    if len(period) == 4:
        year, month, day, length_days = period
        start = _seconds_since_base(int(year), int(month), int(day), 0, calendar)
        length_seconds = int(length_days) * int(_SECSPDAY)
    elif len(period) == 5:
        year, month, day, start_second, length_seconds_raw = period
        start = _seconds_since_base(int(year), int(month), int(day), int(start_second), calendar)
        length_seconds = int(length_seconds_raw)
    else:
        raise ValueError(
            "period_index entries must be (year, month, day, length_days) "
            "or (year, month, day, start_second, length_seconds)"
        )
    if length_seconds <= 0:
        raise ValueError(f"history period length must be positive, got {length_seconds}")
    return start, start + length_seconds


def _char_row(text: str, width: int = 16) -> np.ndarray:
    return np.asarray(list(text[:width].ljust(width)), dtype="S1")


def _as_float_array(value: np.ndarray) -> np.ndarray:
    arr = np.asanyarray(value)
    if np.ma.isMaskedArray(arr):
        arr = arr.filled(np.nan)
    return np.asarray(arr, dtype=np.float32)


def _normalise_scalar(value: np.ndarray, ntime: int, ncell: int) -> np.ndarray:
    arr = _as_float_array(value)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.shape != (ntime, ncell):
        raise ValueError(f"scalar h0 field has shape {arr.shape}, expected {(ntime, ncell)}")
    return arr


def _normalise_vertical(value: np.ndarray, ntime: int, ncell: int) -> np.ndarray:
    arr = _as_float_array(value)
    if arr.ndim == 2:
        arr = arr[None, :, :]
    if arr.ndim != 3:
        raise ValueError(f"vertical h0 field has shape {arr.shape}, expected 3 dimensions")
    if arr.shape[0] != ntime:
        raise ValueError(f"vertical h0 field has {arr.shape[0]} time records, expected {ntime}")
    if arr.shape[1] == ncell:
        return arr
    if arr.shape[2] == ncell:
        return np.moveaxis(arr, 2, 1)
    raise ValueError(f"vertical h0 field has shape {arr.shape}; no lndgrid dimension of {ncell}")


def _normalise_numrad(value: np.ndarray, ntime: int, ncell: int) -> np.ndarray:
    arr = _as_float_array(value)
    if arr.ndim == 2:
        arr = arr[None, :, :]
    if arr.shape == (ntime, ncell, 2):
        return arr
    if arr.shape == (ntime, 2, ncell):
        return np.moveaxis(arr, 1, 2)
    raise ValueError(f"numrad h0 field has shape {arr.shape}, expected {(ntime, ncell, 2)}")


def _first_present(diag: dict[str, np.ndarray], names: tuple[str, ...]) -> np.ndarray | None:
    for name in names:
        if name in diag:
            return diag[name]
    return None


def _maybe_scaled(
    diag: dict[str, np.ndarray],
    names: tuple[str, ...],
    *,
    scale: float = 1.0,
    offset: float = 0.0,
) -> np.ndarray | None:
    value = _first_present(diag, names)
    if value is None:
        return None
    return _as_float_array(value) * np.float32(scale) + np.float32(offset)


def _estimate_area_km2(surf: dict, ncell: int) -> np.ndarray:
    if "area_km2" in surf:
        area = _as_float_array(surf["area_km2"])
        if area.shape == (ncell,):
            return area
    if "cell_area_km2" in surf:
        area = _as_float_array(surf["cell_area_km2"])
        if area.shape == (ncell,):
            return area
    if "precip_treatment_cell_size_m" in surf:
        size_m = float(surf["precip_treatment_cell_size_m"])
        if np.isfinite(size_m) and size_m > 0.0:
            return np.full(ncell, (size_m * 0.001) ** 2, dtype=np.float32)

    lat = _as_float_array(surf.get("lat_deg", np.zeros(ncell, dtype=np.float32)))
    lat_centers = np.asarray(surf.get("map_lat_centers", []), dtype=np.float64)
    lon_centers = np.asarray(surf.get("map_lon_centers", []), dtype=np.float64)
    if lat_centers.size > 1 and lon_centers.size > 1:
        dlat = float(np.nanmedian(np.abs(np.diff(np.sort(np.unique(lat_centers))))))
        dlon = float(np.nanmedian(np.abs(np.diff(np.sort(np.unique(lon_centers))))))
        if dlat > 0.0 and dlon > 0.0:
            radius_m = 6_371_000.0
            dlon_rad = np.deg2rad(dlon)
            lat1 = np.deg2rad(lat.astype(np.float64) - 0.5 * dlat)
            lat2 = np.deg2rad(lat.astype(np.float64) + 0.5 * dlat)
            area = radius_m**2 * dlon_rad * (np.sin(lat2) - np.sin(lat1)) / 1.0e6
            return np.maximum(area, 0.0).astype(np.float32)
    return np.ones(ncell, dtype=np.float32)


_SCALAR_SPECS: dict[str, tuple[tuple[str, ...], str, str, float]] = {
    "GPP": (("gpp_gc_m2_d_mean", "carbon_gpp_daily", "carbon_gpp_daily_mean"), "gross primary production", "gC/m^2/s", 1.0 / _SECSPDAY),
    "FPSN": (("fpsn_mean",), "photosynthesis", "umol/m2s", 1.0),
    "QFLX_EVAP_TOT": (("qflx_evap_tot_mean",), "total evaporation flux", "kg/m^2/s", 1.0),
    "EFLX_LH_TOT": (("eflx_lh_tot_mean",), "total latent heat flux [+ to atm]", "W/m^2", 1.0),
    "FSH": (("eflx_sh_tot_mean",), "sensible heat", "W/m^2", 1.0),
    "QOVER": (("qflx_surf_mean",), "surface runoff", "mm/s", 1.0),
    "QH2OSFC": (("qflx_h2osfc_surf_mean",), "surface water runoff", "mm/s", 1.0),
    "QRUNOFF": (("qflx_runoff_tot_mean",), "total liquid runoff", "mm/s", 1.0),
    "QVEGT": (("qvegt_mean", "qflx_tran_veg_mean"), "canopy transpiration", "mm/s", 1.0),
    "QVEGE": (("qvege_mean",), "canopy evaporation", "mm/s", 1.0),
    "QSOIL": (("qsoil_mean", "qflx_evap_soi_mean"), "ground evaporation", "mm/s", 1.0),
    "TLAI": (("lai_mean",), "total projected leaf area index", "1", 1.0),
    "TBOT": (("forc_t_mean",), "atmospheric air temperature", "K", 1.0),
    "QBOT": (("forc_q_mean",), "atmospheric specific humidity", "kg/kg", 1.0),
    "PBOT": (("forc_pbot_mean",), "atmospheric pressure", "Pa", 1.0),
    "PSRF": (("forc_pbot_mean",), "surface pressure", "Pa", 1.0),
    "FLDS": (("forc_lwrad_mean",), "atmospheric longwave radiation", "W/m^2", 1.0),
    "FIRE": (("eflx_lwrad_out_mean", "fire_mean"), "emitted infrared (longwave) radiation", "W/m^2", 1.0),
    "FIRA": (("eflx_lwrad_net_mean", "fira_mean"), "net infrared (longwave) radiation", "W/m^2", 1.0),
    "WIND": (("forc_wind_mean",), "atmospheric wind velocity magnitude", "m/s", 1.0),
    "RH2M": (("rh2m_mean",), "2m relative humidity", "%", 1.0),
    "FSDS": (("fsds_mean",), "atmospheric incident solar radiation", "W/m^2", 1.0),
    "FSA": (("fsa_mean",), "absorbed solar radiation", "W/m^2", 1.0),
    "FSR": (("fsr_mean",), "reflected solar radiation", "W/m^2", 1.0),
    "FSDSVD": (("fsdsvd_mean",), "direct vis incident solar radiation", "W/m^2", 1.0),
    "FSDSVI": (("fsdsvi_mean",), "diffuse vis incident solar radiation", "W/m^2", 1.0),
    "FSDSND": (("fsdsnd_mean",), "direct nir incident solar radiation", "W/m^2", 1.0),
    "FSDSNI": (("fsdsni_mean",), "diffuse nir incident solar radiation", "W/m^2", 1.0),
    "FSRVD": (("fsrvd_mean",), "direct vis reflected solar radiation", "W/m^2", 1.0),
    "FSRVI": (("fsrvi_mean",), "diffuse vis reflected solar radiation", "W/m^2", 1.0),
    "FSRND": (("fsrnd_mean",), "direct nir reflected solar radiation", "W/m^2", 1.0),
    "FSRNI": (("fsrni_mean",), "diffuse nir reflected solar radiation", "W/m^2", 1.0),
    "SABG": (("sabg_mean",), "solar radiation absorbed by ground", "W/m^2", 1.0),
    "SABV": (("sabv_mean",), "solar radiation absorbed by vegetation", "W/m^2", 1.0),
    "FGR": (("eflx_fgr12_mean", "eflx_fgr12"), "heat flux into soil/snow including snow melt", "W/m^2", 1.0),
    "FGR12": (("eflx_fgr12_mean", "eflx_fgr12"), "heat flux between soil layers 1 and 2", "W/m^2", 1.0),
    "HS_SOIL": (("hs_soil_mean", "hs_soil"), "surface heat flux into soil", "W/m^2", 1.0),
    "HS_TOPSNOW": (("hs_top_snow_mean", "hs_top_snow"), "surface heat flux into top snow layer", "W/m^2", 1.0),
    "HS_H2OSFC": (("hs_h2osfc_mean", "hs_h2osfc"), "surface heat flux into surface water", "W/m^2", 1.0),
    "DHSDT": (("dhsdT_mean", "dhsdT"), "surface heat flux temperature derivative", "W/m^2/K", 1.0),
    "TG": (("tg_mean", "t_grnd_mean"), "ground temperature", "K", 1.0),
    "TV": (("t_veg_mean",), "vegetation temperature", "K", 1.0),
    "RAIN": (("forc_rain_mean",), "atmospheric rain", "mm/s", 1.0),
    "SNOW": (("forc_snow_mean",), "atmospheric snow", "mm/s", 1.0),
    "BTRAN": (("btran_mean",), "transpiration beta factor", "1", 1.0),
    "SNOWDP": (("snow_depth_eff_mean", "snow_depth_mean", "active_snow_depth_mean"), "gridcell mean snow height", "m", 1.0),
    "H2OSFC": (("h2osfc_mean",), "surface water storage", "mm", 1.0),
    "ZWT": (("zwt_mean",), "water table depth", "m", 1.0),
    "TOTVEGC": (("carbon_totvegc",), "total vegetation carbon, excluding cpool", "gC/m^2", 1.0),
    "TOTECOSYSC": (("carbon_totecosysc",), "total ecosystem carbon", "gC/m^2", 1.0),
    "HR": (("carbon_hr_daily", "carbon_hr_daily_mean"), "total heterotrophic respiration", "gC/m^2/s", 1.0 / _SECSPDAY),
}

_NUMRAD_SPECS: dict[str, tuple[tuple[str, ...], str, str]] = {
    "ALBGRD": (("albgrd_mean",), "ground albedo (direct)", "proportion"),
    "ALBGRI": (("albgri_mean",), "ground albedo (indirect)", "proportion"),
    "ALBD": (("albd_mean",), "surface albedo (direct)", "proportion"),
    "ALBI": (("albi_mean",), "surface albedo (indirect)", "proportion"),
}


class ElmHistoryWriter:
    """Append ELM-style h0 yearly history files under a run directory."""

    def __init__(
        self,
        output_dir: Path,
        *,
        case_name: str,
        calendar: str = "noleap",
        surface_dataset: Path | None = None,
        clm_params: Path | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.case_name = str(case_name)
        self.calendar = str(calendar)
        self.surface_dataset = None if surface_dataset is None else Path(surface_dataset)
        self.clm_params = None if clm_params is None else Path(clm_params)
        self._opened_years: set[int] = set()

    def path_for_year(self, year: int) -> Path:
        return self.output_dir / f"{self.case_name}.elm.h0.{int(year):04d}-01-01-00000.nc"

    def write(
        self,
        *,
        surf: dict,
        diag: dict[str, np.ndarray],
        period_index: list[_PeriodIndex],
    ) -> list[Path]:
        """Write one or more period means.

        ``period_index`` entries are either ``(start_year, start_month,
        start_day, length_days)`` or ``(start_year, start_month, start_day,
        start_second, length_seconds)``. Records are labelled at interval ends,
        matching the physically comparable native ELM hourly h0 records after
        the initial ``time=0`` history record.
        """
        if not diag or not period_index:
            return []
        ntime = len(period_index)
        ncell = int(np.asarray(surf["lat_deg"]).size)
        for key, value in tuple(diag.items()):
            arr = np.asanyarray(value)
            if arr.shape[0] != ntime and not (arr.ndim == 1 and ntime == 1):
                raise ValueError(
                    f"{key} has {arr.shape[0] if arr.ndim else 0} records, "
                    f"but period_index has {ntime}"
                )

        groups: dict[int, list[int]] = {}
        for idx, period in enumerate(period_index):
            groups.setdefault(_period_start_year(period), []).append(idx)

        written: list[Path] = []
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for year, indices in groups.items():
            path = self.path_for_year(year)
            mode = "a" if year in self._opened_years and path.exists() else "w"
            with Dataset(path, mode) as ds:
                if mode == "w":
                    self._initialise_file(ds, surf, year=year)
                self._append_records(
                    ds,
                    surf=surf,
                    diag={key: self._select_records(value, indices, ntime) for key, value in diag.items()},
                    period_index=[period_index[i] for i in indices],
                )
            self._opened_years.add(year)
            written.append(path)
        return written

    @staticmethod
    def _select_records(value: np.ndarray, indices: list[int], ntime: int) -> np.ndarray:
        arr = np.asanyarray(value)
        if arr.ndim == 1 and ntime == 1:
            return arr
        return arr[indices]

    def _initialise_file(self, ds: Dataset, surf: dict, *, year: int) -> None:
        ncell = int(np.asarray(surf["lat_deg"]).size)
        _dz15, z15, _zi15 = elm_soil_grid(nlevsoi=15)
        _dz10, z10, _zi10 = elm_soil_grid(nlevsoi=10)

        ds.createDimension("lndgrid", ncell)
        ds.createDimension("gridcell", ncell)
        ds.createDimension("levgrnd", 15)
        ds.createDimension("levsoi", 10)
        ds.createDimension("numrad", 2)
        ds.createDimension("hist_interval", 2)
        ds.createDimension("string_length", 16)
        ds.createDimension("time", None)

        ds.title = "ELM History file information"
        ds.source = "ELM surface emulator"
        ds.product = "model-output"
        ds.realm = "land"
        ds.case = self.case_name
        ds.history = f"created by ELM_emulator on {time.strftime('%Y-%m-%d %H:%M:%S %Z')}"
        ds.Conventions = "CF-1.7"
        ds.comment = "NOTE: emulator h0 contains a comparable subset of ELM land history fields."
        ds.calendar = self.calendar
        if self.surface_dataset is not None:
            ds.Surface_dataset = str(self.surface_dataset)
        if self.clm_params is not None:
            ds.PFT_physiological_constants_dataset = str(self.clm_params)

        levgrnd = ds.createVariable("levgrnd", "f4", ("levgrnd",))
        levgrnd.long_name = "coordinate soil levels"
        levgrnd.units = "m"
        levgrnd[:] = z15.astype(np.float32)

        levsoi = ds.createVariable("levsoi", "f4", ("levsoi",))
        levsoi.long_name = "coordinate soil levels (equivalent to top nlevsoi levels of levgrnd)"
        levsoi.units = "m"
        levsoi[:] = z10.astype(np.float32)

        time_var = ds.createVariable("time", "f8", ("time",))
        time_var.long_name = "time"
        time_var.units = f"days since {int(year):04d}-01-01 00:00:00"
        time_var.calendar = self.calendar

        bounds = ds.createVariable("time_bounds", "f8", ("time", "hist_interval"))
        bounds.long_name = "history time interval endpoints"

        for name, long_name, units in (
            ("mcdate", "current date (YYYYMMDD)", ""),
            ("mcsec", "current seconds of current date", "s"),
            ("mdcur", "current day (from base day)", ""),
            ("mscur", "current seconds of current day", ""),
            ("nstep", "time step", ""),
        ):
            var = ds.createVariable(name, "i4", ("time",), fill_value=_FILL_INT)
            var.long_name = long_name
            if units:
                var.units = units

        ds.createVariable("date_written", "S1", ("time", "string_length"))
        ds.createVariable("time_written", "S1", ("time", "string_length"))

        for name, values, long_name, units in (
            ("lon", surf["lon_deg"], "coordinate longitude", "degrees_east"),
            ("lat", surf["lat_deg"], "coordinate latitude", "degrees_north"),
            ("area", _estimate_area_km2(surf, ncell), "grid cell areas", "km^2"),
            ("topo", surf.get("std_elev", np.zeros(ncell, dtype=np.float32)), "grid cell topography", "m"),
            ("landfrac", np.ones(ncell, dtype=np.float32), "land fraction", ""),
        ):
            var = ds.createVariable(name, "f4", ("lndgrid",), fill_value=_FILL_FLOAT)
            var.long_name = long_name
            if units:
                var.units = units
            var.missing_value = _FILL_FLOAT
            var[:] = _as_float_array(values)

        for name, long_name in (
            ("landmask", "land/ocean mask (0.=ocean and 1.=land)"),
            ("pftmask", "pft real/fake mask (0.=fake and 1.=real)"),
        ):
            var = ds.createVariable(name, "i4", ("lndgrid",), fill_value=_FILL_INT)
            var.long_name = long_name
            var.missing_value = _FILL_INT
            var[:] = np.ones(ncell, dtype=np.int32)

    def _ensure_scalar_var(self, ds: Dataset, name: str, long_name: str, units: str) -> None:
        if name in ds.variables:
            return
        var = ds.createVariable(
            name,
            "f4",
            ("time", "lndgrid"),
            chunksizes=(1, len(ds.dimensions["lndgrid"])),
            fill_value=_FILL_FLOAT,
        )
        var.long_name = long_name
        var.units = units
        var.cell_methods = "time: mean"
        var.missing_value = _FILL_FLOAT

    def _ensure_vertical_var(self, ds: Dataset, name: str, long_name: str, units: str) -> None:
        if name in ds.variables:
            return
        var = ds.createVariable(
            name,
            "f4",
            ("time", "levgrnd", "lndgrid"),
            chunksizes=(1, len(ds.dimensions["levgrnd"]), len(ds.dimensions["lndgrid"])),
            fill_value=_FILL_FLOAT,
        )
        var.long_name = long_name
        var.units = units
        var.cell_methods = "time: mean"
        var.missing_value = _FILL_FLOAT

    def _ensure_numrad_var(self, ds: Dataset, name: str, long_name: str, units: str) -> None:
        if name in ds.variables:
            return
        if "numrad" not in ds.dimensions:
            ds.createDimension("numrad", 2)
        var = ds.createVariable(
            name,
            "f4",
            ("time", "numrad", "lndgrid"),
            chunksizes=(1, len(ds.dimensions["numrad"]), len(ds.dimensions["lndgrid"])),
            fill_value=_FILL_FLOAT,
        )
        var.long_name = long_name
        var.units = units
        var.cell_methods = "time: mean"
        var.missing_value = _FILL_FLOAT

    def _append_records(
        self,
        ds: Dataset,
        *,
        surf: dict,
        diag: dict[str, np.ndarray],
        period_index: list[_PeriodIndex],
    ) -> None:
        ntime = len(period_index)
        ncell = int(np.asarray(surf["lat_deg"]).size)
        start = len(ds.dimensions["time"])

        now_date = time.strftime("%m/%d/%y")
        now_time = time.strftime("%H:%M:%S")
        date_rows = np.stack([_char_row(now_date) for _ in range(ntime)], axis=0)
        time_rows = np.stack([_char_row(now_time) for _ in range(ntime)], axis=0)

        origin_year = _period_start_year(period_index[0])
        origin_second = _seconds_since_base(origin_year, 1, 1, 0, self.calendar)
        time_values = np.empty(ntime, dtype=np.float64)
        time_bounds = np.empty((ntime, 2), dtype=np.float64)
        mcdate = np.empty(ntime, dtype=np.int32)
        mcsec = np.empty(ntime, dtype=np.int32)
        mdcur = np.empty(ntime, dtype=np.int32)
        nstep = np.empty(ntime, dtype=np.int32)
        for idx, period in enumerate(period_index):
            start_second, stop_second = _period_seconds(period, self.calendar)
            label_second = stop_second
            bounds_start = start_second
            bounds_stop = stop_second
            current_year, current_month, current_day, current_sec = _ymd_sec_from_seconds_since_base(
                label_second,
                self.calendar,
            )
            time_values[idx] = float(label_second - origin_second) / _SECSPDAY
            time_bounds[idx, :] = (
                float(bounds_start - origin_second) / _SECSPDAY,
                float(bounds_stop - origin_second) / _SECSPDAY,
            )
            mcdate[idx] = current_year * 10000 + current_month * 100 + current_day
            mcsec[idx] = int(current_sec)
            mdcur[idx] = int((label_second - origin_second) // int(_SECSPDAY))
            nstep[idx] = int(round(float(label_second - origin_second) / 3600.0))

        ds.variables["time"][start:start + ntime] = time_values
        ds.variables["time_bounds"][start:start + ntime, :] = time_bounds
        ds.variables["mcdate"][start:start + ntime] = mcdate
        ds.variables["mcsec"][start:start + ntime] = mcsec
        ds.variables["mdcur"][start:start + ntime] = mdcur
        ds.variables["mscur"][start:start + ntime] = mcsec
        ds.variables["nstep"][start:start + ntime] = nstep
        ds.variables["date_written"][start:start + ntime, :] = date_rows
        ds.variables["time_written"][start:start + ntime, :] = time_rows

        for name, (source_names, long_name, units, scale) in _SCALAR_SPECS.items():
            values = _maybe_scaled(diag, source_names, scale=scale)
            if values is None:
                continue
            values = _normalise_scalar(values, ntime, ncell)
            self._ensure_scalar_var(ds, name, long_name, units)
            ds.variables[name][start:start + ntime, :] = np.ma.masked_invalid(values)

        for name, (source_names, long_name, units) in _NUMRAD_SPECS.items():
            values = _first_present(diag, source_names)
            if values is None:
                continue
            values = _normalise_numrad(values, ntime, ncell)
            out = np.moveaxis(values, 2, 1)
            self._ensure_numrad_var(ds, name, long_name, units)
            ds.variables[name][start:start + ntime, :, :] = np.ma.masked_invalid(out)

        for name, source_name, long_name in (
            ("FCTR", "qvegt_mean", "canopy transpiration"),
            ("FCEV", "qvege_mean", "canopy evaporation"),
            ("FGEV", "qsoil_mean", "ground evaporation"),
        ):
            values = _maybe_scaled(diag, (source_name,), scale=_LV)
            if values is None:
                continue
            values = _normalise_scalar(values, ntime, ncell)
            self._ensure_scalar_var(ds, name, long_name, "W/m^2")
            ds.variables[name][start:start + ntime, :] = np.ma.masked_invalid(values)

        gpp = _maybe_scaled(diag, ("GPP", "gpp_gc_m2_d_mean", "carbon_gpp_daily", "carbon_gpp_daily_mean"), scale=1.0 / _SECSPDAY)
        hr = _maybe_scaled(diag, ("HR", "carbon_hr_daily", "carbon_hr_daily_mean"), scale=1.0 / _SECSPDAY)
        npp_daily = _maybe_scaled(diag, ("carbon_npp_daily", "carbon_npp_daily_mean"), scale=1.0 / _SECSPDAY)
        if gpp is not None and hr is not None and npp_daily is not None:
            er = gpp + hr - npp_daily
            nee = er - gpp
            for name, values, long_name in (
                ("ER", er, "total ecosystem respiration, autotrophic + heterotrophic"),
                ("NEE", nee, "net ecosystem exchange of carbon, positive for source"),
            ):
                values = _normalise_scalar(values, ntime, ncell)
                self._ensure_scalar_var(ds, name, long_name, "gC/m^2/s")
                ds.variables[name][start:start + ntime, :] = np.ma.masked_invalid(values)

        tsoi = _first_present(diag, ("elm_h0_TSOI", "TSOI"))
        if tsoi is None:
            layers = [
                _maybe_scaled(diag, (f"t_soisno_lev{lev}_mean",), scale=1.0)
                for lev in range(1, 16)
            ]
            if any(layer is not None for layer in layers):
                base = np.full((ntime, ncell, 15), np.nan, dtype=np.float32)
                for lev, layer in enumerate(layers):
                    if layer is not None:
                        base[:, :, lev] = _normalise_scalar(layer, ntime, ncell)
                tsoi = base
        if tsoi is not None:
            values = _normalise_vertical(tsoi, ntime, ncell)
            out = np.full((ntime, 15, ncell), np.nan, dtype=np.float32)
            nlev = min(15, values.shape[2])
            out[:, :nlev, :] = np.moveaxis(values[:, :, :nlev], 2, 1)
            self._ensure_vertical_var(ds, "TSOI", "soil temperature (vegetated landunits only)", "K")
            ds.variables["TSOI"][start:start + ntime, :, :] = np.ma.masked_invalid(out)

        h2osoi = _first_present(diag, ("elm_h0_H2OSOI", "H2OSOI"))
        if h2osoi is None:
            layers = [
                _maybe_scaled(diag, (f"h2osoi_liqvol_lev{lev}_mean",), scale=1.0)
                for lev in range(1, 11)
            ]
            if any(layer is not None for layer in layers):
                base = np.zeros((ntime, ncell, 15), dtype=np.float32)
                for lev, layer in enumerate(layers):
                    if layer is not None:
                        base[:, :, lev] = _normalise_scalar(layer, ntime, ncell)
                h2osoi = base
        if h2osoi is not None:
            values = _normalise_vertical(h2osoi, ntime, ncell)
            out = np.zeros((ntime, 15, ncell), dtype=np.float32)
            nlev = min(15, values.shape[2])
            out[:, :nlev, :] = np.moveaxis(values[:, :, :nlev], 2, 1)
            self._ensure_vertical_var(ds, "H2OSOI", "volumetric soil water (vegetated landunits only)", "mm3/mm3")
            ds.variables["H2OSOI"][start:start + ntime, :, :] = np.ma.masked_invalid(out)
