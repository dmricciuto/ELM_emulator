"""Regional surfdata loading and domain transformation helpers."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

from .regional_met import _detect_regional_met_format, _trendy_met_file
from .regional_time import _days_in_month


def _organic_fraction_from_density(organic_kg_m3: np.ndarray, organic_max: float = 130.0) -> np.ndarray:
    """Convert organic matter density (kg m-3) to ELM-style volume fraction."""
    frac = np.asarray(organic_kg_m3, dtype=np.float64) / float(organic_max)
    return np.clip(frac * frac, 0.0, 1.0)


def load_regional_surfdata(
    surfdata_file: Path,
    *,
    met_dir: Path | None = None,
    min_lon_deg: float | None = -95.0,
    max_lon_deg: float | None = None,
    min_lat_deg: float | None = None,
    max_lat_deg: float | None = None,
    met_format: str = "auto",
    nlev_hydro: int = 10,
    nlev_thermal: int = 15,
    organic_max: float = 130.0,
) -> dict:
    """Load all per-cell arrays needed for the regional benchmark.

    Returns a dict with keys:
      ``active_idx``  – 1-D int array of cell indices with PFTDATA_MASK > 0
      ``lat_deg``, ``lon_deg``  – (ncell,) floats
      ``grid_row``, ``grid_col`` – (ncell,) int indices into (361, 625) met grid
      ``pct_nat_pft`` – (17, ncell) gridcell PFT coverage after unsupported
          landunits are folded into PFT 0 when landunit fractions are available
      ``monthly_lai``, ``monthly_sai`` – (12, 17, ncell) monthly LAI/SAI per PFT
      ``monthly_htop``, ``monthly_hbot`` – (12, 17, ncell) monthly canopy heights
      ``sand``, ``clay``, ``organic`` – (ncell, nlev_hydro) texture arrays [0–100 / kg m-3]
      ``organic_frac`` – (ncell, nlev_hydro) organic volume fraction
      ``soil_color``, ``fmax``, ``slope``, ``std_elev`` – (ncell,)
    """
    def _lon_west(values: np.ndarray) -> np.ndarray:
        return ((np.asarray(values, dtype=np.float64) + 180.0) % 360.0) - 180.0

    def _lon_bound(value: float | None) -> float | None:
        if value is None or not np.isfinite(float(value)):
            return None
        return float(((float(value) + 180.0) % 360.0) - 180.0)

    def _bbox_keep_mask(lat_values: np.ndarray, lon_values: np.ndarray) -> np.ndarray:
        keep = np.ones(np.asarray(lat_values).shape, dtype=bool)
        if min_lat_deg is not None and np.isfinite(float(min_lat_deg)):
            keep &= np.asarray(lat_values, dtype=np.float64) >= float(min_lat_deg)
        if max_lat_deg is not None and np.isfinite(float(max_lat_deg)):
            keep &= np.asarray(lat_values, dtype=np.float64) <= float(max_lat_deg)
        lon_min = _lon_bound(min_lon_deg)
        lon_max = _lon_bound(max_lon_deg)
        if lon_min is not None or lon_max is not None:
            lon_values_west = _lon_west(lon_values)
            if lon_min is not None and lon_max is not None:
                if lon_min <= lon_max:
                    keep &= (lon_values_west >= lon_min) & (lon_values_west <= lon_max)
                else:
                    keep &= (lon_values_west >= lon_min) | (lon_values_west <= lon_max)
            elif lon_min is not None:
                keep &= lon_values_west >= lon_min
            elif lon_max is not None:
                keep &= lon_values_west <= lon_max
        return keep

    def _bbox_label() -> str:
        def fmt(value: float | None) -> str:
            return "none" if value is None else f"{float(value):g}"

        return (
            f"lat=[{fmt(min_lat_deg)}, {fmt(max_lat_deg)}], "
            f"lon=[{fmt(min_lon_deg)}, {fmt(max_lon_deg)}]"
        )

    def _apply_index_filter(index: np.ndarray) -> None:
        nonlocal active, lat, lon, grid_row, grid_col, met_index
        active = active[index]
        lat = lat[index]
        lon = lon[index]
        grid_row = grid_row[index]
        grid_col = grid_col[index]
        met_index = met_index[index]

    resolved_met_format = (
        _detect_regional_met_format(Path(met_dir), met_format)
        if met_dir is not None
        else str(met_format).lower()
    )
    with Dataset(surfdata_file) as ds:
        lat_var = ds.variables["LATIXY"]
        is_gridded = lat_var.ndim == 2
        if is_gridded:
            if resolved_met_format != "trendy":
                raise ValueError(
                    "2-D regional surfdata currently requires --regional-met-format trendy "
                    "so the compressed forcing points can be coordinate-matched."
                )
            if met_dir is None:
                raise ValueError("TRENDY surfdata loading requires --met-dir")
            lat2 = np.asarray(ds.variables["LATIXY"][:], dtype=np.float32)
            lon2 = np.asarray(ds.variables["LONGXY"][:], dtype=np.float32)
            lon2_west = ((lon2.astype(np.float64) + 180.0) % 360.0) - 180.0
            nlat, nlon = lat2.shape
            with Dataset(_trendy_met_file(Path(met_dir), "TBOT")) as met_ds:
                met_lat = np.asarray(met_ds.variables["LATIXY"][:], dtype=np.float32)
                met_lon = np.asarray(met_ds.variables["LONGXY"][:], dtype=np.float32)
            coord_to_rc: dict[tuple[float, float], tuple[int, int]] = {}
            for row in range(nlat):
                for col in range(nlon):
                    coord_to_rc[(round(float(lon2_west[row, col]), 4), round(float(lat2[row, col]), 4))] = (row, col)
            rows: list[int] = []
            cols: list[int] = []
            missing: list[tuple[float, float]] = []
            for lon_i, lat_i in zip(met_lon, met_lat):
                rc = coord_to_rc.get((round(float(lon_i), 4), round(float(lat_i), 4)))
                if rc is None:
                    missing.append((float(lon_i), float(lat_i)))
                else:
                    rows.append(rc[0])
                    cols.append(rc[1])
            if missing:
                sample = ", ".join(f"({lon:.3f},{lat:.3f})" for lon, lat in missing[:5])
                raise ValueError(f"Could not match {len(missing)} TRENDY forcing points to surfdata coordinates; first: {sample}")
            grid_row = np.asarray(rows, dtype=np.int32)
            grid_col = np.asarray(cols, dtype=np.int32)
            active = (grid_row.astype(np.int64) * int(nlon) + grid_col.astype(np.int64)).astype(np.int64)
            met_index = np.arange(met_lat.size, dtype=np.int64)
            lat = met_lat.astype(np.float32, copy=False)
            lon = met_lon.astype(np.float32, copy=False)
            map_lat_centers = np.nanmean(lat2, axis=1).astype(np.float32)
            map_lon_centers = np.nanmean(lon2_west, axis=0).astype(np.float32)
        else:
            mask      = np.asarray(ds.variables["PFTDATA_MASK"][:], dtype=np.int32)
            active    = np.where(mask > 0)[0]
            # Read each variable fully then apply mask in numpy — fancy indexing
            # directly against a NetCDF4 variable is extremely slow (seek-per-chunk).
            lat       = np.asarray(ds.variables["LATIXY"][:],   dtype=np.float32)[active]
            lon       = np.asarray(ds.variables["LONGXY"][:],   dtype=np.float32)[active]
            if "gridYID" in ds.variables and "gridXID" in ds.variables:
                grid_row  = np.asarray(ds.variables["gridYID"][:],  dtype=np.int32)[active]
                grid_col  = np.asarray(ds.variables["gridXID"][:],  dtype=np.int32)[active]
            elif resolved_met_format == "site":
                # CLM1PT point surfdata generated by OLMT has no regional met
                # grid indices.  Use local one-cell indices; site forcing is
                # selected by gridcell order in all_hourly.nc instead.
                grid_row = np.arange(active.size, dtype=np.int32)
                grid_col = np.zeros(active.size, dtype=np.int32)
            elif resolved_met_format == "trendy":
                # CLM1PT point surfdata can also be driven by compressed TRENDY
                # forcing. Native ELM chooses the closest TRENDY coordinate for
                # the current domain point; do the same here and use that
                # compressed land-point index when reading the forcing files.
                if met_dir is None:
                    raise ValueError("TRENDY point surfdata loading requires --met-dir")
                with Dataset(_trendy_met_file(Path(met_dir), "TBOT")) as met_ds:
                    met_lat = np.asarray(met_ds.variables["LATIXY"][:], dtype=np.float64)
                    met_lon = np.asarray(met_ds.variables["LONGXY"][:], dtype=np.float64)
                met_indices = []
                for lat_i, lon_i in zip(lat.astype(np.float64), lon.astype(np.float64)):
                    lon_cmp = met_lon.copy()
                    if lon_i < 0.0:
                        lon_cmp[lon_cmp >= 180.0] -= 360.0
                    elif lon_i >= 180.0:
                        lon_cmp[lon_cmp < 0.0] += 360.0
                    met_indices.append(int(np.argmin((met_lat - lat_i) ** 2 + (lon_cmp - lon_i) ** 2)))
                met_index = np.asarray(met_indices, dtype=np.int64)
                grid_row = met_index.astype(np.int32, copy=True)
                grid_col = np.zeros(active.size, dtype=np.int32)
            else:
                raise KeyError(
                    f"{surfdata_file} is missing gridYID/gridXID; use "
                    "--regional-met-format site for CLM1PT point surfdata."
                )
            if resolved_met_format == "site":
                met_index = np.arange(active.size, dtype=np.int64)
            elif "gridYID" in ds.variables and "gridXID" in ds.variables:
                met_index = active.astype(np.int64, copy=True)

    # ------------------------------------------------------------------
    # Invalidate cells whose met-grid point is ocean/coastal fill (-9999).
    # Load one met variable (TBOT, first timestep) and build a valid mask
    # over the 2-D met grid, then filter surfdata cells accordingly.
    # ------------------------------------------------------------------
    if met_dir is not None and resolved_met_format == "tessfa":
        import glob as _glob
        _tpqwl = sorted(_glob.glob(str(Path(met_dir) / "clmforc.Daymet_ERA5_TESSFA.4km.TPQWL.*.nc")))
        if not _tpqwl:
            raise FileNotFoundError(f"No TPQWL met files found under {met_dir}")
        with Dataset(_tpqwl[0]) as _ds_met:
            _tbot0 = np.asarray(_ds_met.variables["TBOT"][0], dtype=np.float32)  # (nlat, nlon)
        _met_valid = (_tbot0 > -9000)  # True = valid land cell
        _cell_valid = _met_valid[grid_row, grid_col]  # (ncell,) bool
        n_bad = int((~_cell_valid).sum())
        if n_bad > 0:
            print(f"  [load_regional_surfdata] dropping {n_bad} cells with fill-value met data "
                  f"(ocean/coastal) out of {len(active)}")
        # Re-index everything to valid cells only.
        _apply_index_filter(np.where(_cell_valid)[0])

    _keep_bbox = _bbox_keep_mask(lat, lon)
    n_drop_bbox = int((~_keep_bbox).sum())
    if n_drop_bbox > 0:
        print(
            "  [load_regional_surfdata] dropping "
            f"{n_drop_bbox:,} cells outside bounds {_bbox_label()}"
        )
    _apply_index_filter(np.where(_keep_bbox)[0])
    if active.size == 0:
        raise ValueError(f"No active regional cells remain after applying bounds {_bbox_label()}")

    pct_natveg_original: np.ndarray | None = None
    pct_unsupported_landunit: np.ndarray | None = None
    with Dataset(surfdata_file) as ds:
        def _sample_landunit_fraction(name: str) -> np.ndarray | None:
            if name not in ds.variables:
                return None
            data = np.asarray(ds.variables[name][:], dtype=np.float32)
            if name == "PCT_URBAN":
                data = data.sum(axis=0)
            if ds.variables["LATIXY"].ndim == 2:
                return data[row, col]
            return data[active]

        if ds.variables["LATIXY"].ndim == 2:
            row = np.asarray(grid_row, dtype=np.int64)
            col = np.asarray(grid_col, dtype=np.int64)
            pct       = np.asarray(ds.variables["PCT_NAT_PFT"][:], dtype=np.float32)[:, row, col]
            m_lai     = np.asarray(ds.variables["MONTHLY_LAI"][:], dtype=np.float32)[:, :, row, col]
            m_sai     = np.asarray(ds.variables["MONTHLY_SAI"][:], dtype=np.float32)[:, :, row, col]
            m_htop    = np.asarray(ds.variables["MONTHLY_HEIGHT_TOP"][:], dtype=np.float32)[:, :, row, col]
            m_hbot    = np.asarray(ds.variables["MONTHLY_HEIGHT_BOT"][:], dtype=np.float32)[:, :, row, col]
            sand_full = np.asarray(ds.variables["PCT_SAND"][:nlev_hydro], dtype=np.float32)[:, row, col]
            clay_full = np.asarray(ds.variables["PCT_CLAY"][:nlev_hydro], dtype=np.float32)[:, row, col]
            org_full  = np.asarray(ds.variables["ORGANIC"][:nlev_hydro], dtype=np.float32)[:, row, col]
            soil_color = np.asarray(ds.variables["SOIL_COLOR"][:], dtype=np.float32)[row, col]
            fmax      = np.asarray(ds.variables["FMAX"][:], dtype=np.float32)[row, col]
            slope     = np.asarray(ds.variables["SLOPE"][:], dtype=np.float32)[row, col]
            std_elev  = np.asarray(ds.variables["STD_ELEV"][:], dtype=np.float32)[row, col]
        else:
            pct       = np.asarray(ds.variables["PCT_NAT_PFT"][:], dtype=np.float32)[:, active]
            m_lai     = np.asarray(ds.variables["MONTHLY_LAI"][:], dtype=np.float32)[:, :, active]
            m_sai     = np.asarray(ds.variables["MONTHLY_SAI"][:], dtype=np.float32)[:, :, active]
            m_htop    = np.asarray(ds.variables["MONTHLY_HEIGHT_TOP"][:], dtype=np.float32)[:, :, active]
            m_hbot    = np.asarray(ds.variables["MONTHLY_HEIGHT_BOT"][:], dtype=np.float32)[:, :, active]
            sand_full = np.asarray(ds.variables["PCT_SAND"][:nlev_hydro], dtype=np.float32)[:, active]
            clay_full = np.asarray(ds.variables["PCT_CLAY"][:nlev_hydro], dtype=np.float32)[:, active]
            org_full  = np.asarray(ds.variables["ORGANIC"][:nlev_hydro],  dtype=np.float32)[:, active]
            soil_color = np.asarray(ds.variables["SOIL_COLOR"][:], dtype=np.float32)[active]
            fmax      = np.asarray(ds.variables["FMAX"][:],    dtype=np.float32)[active]
            slope     = np.asarray(ds.variables["SLOPE"][:],   dtype=np.float32)[active]
            std_elev  = np.asarray(ds.variables["STD_ELEV"][:], dtype=np.float32)[active]
        pct_natveg_original = _sample_landunit_fraction("PCT_NATVEG")

    if pct_natveg_original is not None:
        # PCT_NAT_PFT is a PFT distribution inside the natural landunit. For the
        # one-column replay approximation, represent unsupported landunits
        # (lake/glacier/wetland/urban/crop) as PFT 0 bare ground.
        pct_natveg = np.clip(pct_natveg_original.astype(np.float32, copy=False), 0.0, 100.0)
        pct_unsupported_landunit = np.clip(100.0 - pct_natveg, 0.0, 100.0).astype(np.float32, copy=False)
        pct_sum = pct.sum(axis=0).astype(np.float32, copy=False)
        scale = np.divide(
            pct_natveg,
            pct_sum,
            out=np.zeros_like(pct_natveg, dtype=np.float32),
            where=pct_sum > 1.0e-12,
        )
        pct = pct * scale[None, :]
        pct[0, :] += pct_unsupported_landunit
        pct_total = pct.sum(axis=0)
        valid_total = pct_total > 1.0e-12
        pct[:, valid_total] *= (100.0 / pct_total[valid_total])[None, :]
        n_folded = int(np.count_nonzero(pct_unsupported_landunit > 1.0e-6))
        if n_folded:
            print(
                "  [landunits] folded unsupported landunit area into PFT 0: "
                f"cells={n_folded:,}/{pct_unsupported_landunit.size:,} "
                f"mean={float(np.nanmean(pct_unsupported_landunit)):.2f}% "
                f"max={float(np.nanmax(pct_unsupported_landunit)):.2f}%"
            )

    # Transpose soil arrays to (ncell, nlev).
    sand = sand_full.T   # (ncell, nlev)
    clay = clay_full.T
    organic_kg = org_full.T
    organic_frac = _organic_fraction_from_density(organic_kg.astype(np.float64), organic_max)
    return dict(
        active_idx=active,
        met_index=met_index,
        met_format=resolved_met_format,
        map_nlat=int(nlat) if 'nlat' in locals() else max(433, int(grid_row.max()) + 1 if grid_row.size else 433),
        map_nlon=int(nlon) if 'nlon' in locals() else max(1321, int(grid_col.max()) + 1 if grid_col.size else 1321),
        map_lat_centers=map_lat_centers if 'map_lat_centers' in locals() else None,
        map_lon_centers=map_lon_centers if 'map_lon_centers' in locals() else None,
        map_lat_min=float(np.nanmin(map_lat_centers)) if 'map_lat_centers' in locals() else (float(np.nanmin(lat)) if lat.size else np.nan),
        map_lat_max=float(np.nanmax(map_lat_centers)) if 'map_lat_centers' in locals() else (float(np.nanmax(lat)) if lat.size else np.nan),
        map_lon_min=float(np.nanmin(map_lon_centers)) if 'map_lon_centers' in locals() else (float(np.nanmin(((lon.astype(np.float64) + 180.0) % 360.0) - 180.0)) if lon.size else np.nan),
        map_lon_max=float(np.nanmax(map_lon_centers)) if 'map_lon_centers' in locals() else (float(np.nanmax(((lon.astype(np.float64) + 180.0) % 360.0) - 180.0)) if lon.size else np.nan),
        pct_natveg_original=pct_natveg_original,
        pct_unsupported_landunit=pct_unsupported_landunit,
        lat_deg=lat,
        lon_deg=lon,
        grid_row=grid_row,
        grid_col=grid_col,
        pct_nat_pft=pct,
        monthly_lai=m_lai,
        monthly_sai=m_sai,
        monthly_htop=m_htop,
        monthly_hbot=m_hbot,
        sand=sand,
        clay=clay,
        organic_frac=organic_frac.astype(np.float32),
        soil_color=soil_color,
        fmax=fmax,
        slope=slope,
        std_elev=std_elev,
    )


def _nearest_regional_cell_index(surf: dict, lat: float, lon: float) -> tuple[int, float]:
    """Return nearest active regional cell index and approximate distance in km."""
    lat_arr = np.asarray(surf["lat_deg"], dtype=np.float64)
    lon_arr = np.asarray(surf["lon_deg"], dtype=np.float64)
    lon_query = float(lon)
    if np.nanmax(lon_arr) > 180.0 and lon_query < 0.0:
        lon_query += 360.0
    elif np.nanmax(lon_arr) <= 180.0 and lon_query > 180.0:
        lon_query -= 360.0
    dlat = np.deg2rad(lat_arr - float(lat))
    dlon = np.deg2rad(((lon_arr - lon_query + 180.0) % 360.0) - 180.0)
    lat_mid = np.deg2rad(0.5 * (lat_arr + float(lat)))
    dist_rad = np.sqrt(dlat * dlat + (np.cos(lat_mid) * dlon) ** 2)
    idx = int(np.nanargmin(dist_rad))
    return idx, float(dist_rad[idx] * 6371.0)


def _subset_regional_surf_cells(surf: dict, indices: np.ndarray | list[int]) -> dict:
    """Subset a regional surfdata dict to selected active-cell indices."""
    idx = np.asarray(indices, dtype=np.int64)
    axis0_keys = {
        "active_idx", "met_index", "lat_deg", "lon_deg", "grid_row", "grid_col",
        "sand", "clay", "organic_frac", "soil_color", "fmax", "slope", "std_elev",
        "pct_natveg_original", "pct_unsupported_landunit",
    }
    axis1_keys = {"pct_nat_pft"}
    axis2_keys = {"monthly_lai", "monthly_sai", "monthly_htop", "monthly_hbot"}
    out: dict = {}
    for key, value in surf.items():
        if value is None:
            out[key] = None
            continue
        if key in {"met_format"}:
            out[key] = value
            continue
        arr = np.asarray(value)
        if key in axis0_keys:
            out[key] = arr[idx].copy()
        elif key in axis1_keys:
            out[key] = arr[:, idx].copy()
        elif key in axis2_keys:
            out[key] = arr[:, :, idx].copy()
        else:
            out[key] = value
    return out


def _correlated_standard_normal_field(
    nrow: int,
    ncol: int,
    *,
    corr_length_cells: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return a zero-mean/unit-std Gaussian random field with smooth structure."""
    nrow_i = int(nrow)
    ncol_i = int(ncol)
    if nrow_i <= 0 or ncol_i <= 0:
        raise ValueError("Synthetic clone grid dimensions must be positive")
    field = rng.standard_normal((nrow_i, ncol_i)).astype(np.float64)
    corr = float(corr_length_cells)
    if np.isfinite(corr) and corr > 0.0:
        ky = np.fft.fftfreq(nrow_i)[:, None]
        kx = np.fft.rfftfreq(ncol_i)[None, :]
        filt = np.exp(-0.5 * (2.0 * np.pi * corr) ** 2 * (kx * kx + ky * ky))
        field = np.fft.irfftn(
            np.fft.rfftn(field, axes=(0, 1)) * filt,
            s=(nrow_i, ncol_i),
            axes=(0, 1),
        )
    field = np.asarray(field, dtype=np.float64)
    field -= float(np.mean(field))
    std = float(np.std(field))
    if not np.isfinite(std) or std <= 1.0e-12:
        field = rng.standard_normal((nrow_i, ncol_i)).astype(np.float64)
        field -= float(np.mean(field))
        std = max(float(np.std(field)), 1.0e-12)
    return (field / std).astype(np.float32)


def _clone_single_regional_cell_to_synthetic_grid(
    surf: dict,
    *,
    nrow: int,
    ncol: int,
    corr_length_cells: float,
    flnr_relative_sd: float,
    mbbopt_relative_sd: float,
    sand_absolute_sd: float,
    trait_min_factor: float,
    trait_max_factor: float,
    sand_min_pct: float,
    sand_max_pct: float,
    grid_spacing_km: float,
    seed: int,
) -> dict:
    """Clone one selected regional cell to a synthetic grid with smooth perturbations."""
    nrow_i = int(nrow)
    ncol_i = int(ncol)
    ncell = nrow_i * ncol_i
    if nrow_i <= 0 or ncol_i <= 0:
        raise ValueError("Synthetic clone grid dimensions must be positive")
    if int(np.asarray(surf["lat_deg"]).size) != 1:
        raise ValueError(
            "Synthetic clone-grid mode requires a single source cell. "
            "Use --lat/--lon to select the MOFlux gridcell first."
        )
    if trait_min_factor <= 0.0 or trait_max_factor <= 0.0 or trait_min_factor > trait_max_factor:
        raise ValueError("Synthetic trait factor bounds must be positive and min <= max")
    if sand_min_pct < 0.0 or sand_max_pct > 100.0 or sand_min_pct > sand_max_pct:
        raise ValueError("Synthetic PCT_SAND bounds must satisfy 0 <= min <= max <= 100")

    axis0_keys = {
        "active_idx", "met_index", "lat_deg", "lon_deg", "grid_row", "grid_col",
        "sand", "clay", "organic_frac", "soil_color", "fmax", "slope", "std_elev",
        "pct_natveg_original", "pct_unsupported_landunit",
    }
    axis1_keys = {"pct_nat_pft"}
    axis2_keys = {"monthly_lai", "monthly_sai", "monthly_htop", "monthly_hbot"}
    out: dict = {}
    for key, value in surf.items():
        if value is None:
            out[key] = None
            continue
        if key in {"met_format"}:
            out[key] = value
            continue
        arr = np.asarray(value)
        if key in axis0_keys:
            out[key] = np.broadcast_to(arr[:1], (ncell,) + arr.shape[1:]).copy()
        elif key in axis1_keys:
            out[key] = np.broadcast_to(arr[:, :1], (arr.shape[0], ncell)).copy()
        elif key in axis2_keys:
            out[key] = np.broadcast_to(arr[:, :, :1], (arr.shape[0], arr.shape[1], ncell)).copy()
        else:
            out[key] = value

    base_grid_row = int(np.asarray(surf["grid_row"])[0])
    base_grid_col = int(np.asarray(surf["grid_col"])[0])
    base_met_index = int(np.asarray(surf.get("met_index", [0]))[0])
    rows, cols = np.indices((nrow_i, ncol_i), dtype=np.int32)
    out["active_idx"] = np.arange(ncell, dtype=np.int64)
    out["grid_row"] = rows.reshape(-1).astype(np.int32)
    out["grid_col"] = cols.reshape(-1).astype(np.int32)
    out["met_grid_row"] = np.full(ncell, base_grid_row, dtype=np.int32)
    out["met_grid_col"] = np.full(ncell, base_grid_col, dtype=np.int32)
    out["met_index"] = np.full(ncell, base_met_index, dtype=np.int64)

    base_lat = float(np.asarray(surf["lat_deg"])[0])
    base_lon = float(np.asarray(surf["lon_deg"])[0])
    spacing_km = max(float(grid_spacing_km), 0.0)
    if spacing_km > 0.0:
        lat_step = spacing_km / 111.32
        lon_step = spacing_km / (111.32 * max(math.cos(math.radians(base_lat)), 0.10))
        map_lat = base_lat + (np.arange(nrow_i, dtype=np.float32) - 0.5 * (nrow_i - 1)) * lat_step
        map_lon = base_lon + (np.arange(ncol_i, dtype=np.float32) - 0.5 * (ncol_i - 1)) * lon_step
    else:
        map_lat = np.arange(nrow_i, dtype=np.float32)
        map_lon = np.arange(ncol_i, dtype=np.float32)
    out["map_nlat"] = nrow_i
    out["map_nlon"] = ncol_i
    out["map_lat_centers"] = map_lat.astype(np.float32, copy=False)
    out["map_lon_centers"] = map_lon.astype(np.float32, copy=False)
    out["map_lat_min"] = float(np.min(map_lat)) if map_lat.size else np.nan
    out["map_lat_max"] = float(np.max(map_lat)) if map_lat.size else np.nan
    out["map_lon_min"] = float(np.min(map_lon)) if map_lon.size else np.nan
    out["map_lon_max"] = float(np.max(map_lon)) if map_lon.size else np.nan

    rng = np.random.default_rng(int(seed))
    z_flnr = _correlated_standard_normal_field(
        nrow_i, ncol_i, corr_length_cells=corr_length_cells, rng=rng
    ).reshape(-1)
    z_mbbopt = _correlated_standard_normal_field(
        nrow_i, ncol_i, corr_length_cells=corr_length_cells, rng=rng
    ).reshape(-1)
    z_sand = _correlated_standard_normal_field(
        nrow_i, ncol_i, corr_length_cells=corr_length_cells, rng=rng
    ).reshape(-1)

    flnr_factor = np.clip(
        1.0 + float(flnr_relative_sd) * z_flnr,
        float(trait_min_factor),
        float(trait_max_factor),
    ).astype(np.float32)
    mbbopt_factor = np.clip(
        1.0 + float(mbbopt_relative_sd) * z_mbbopt,
        float(trait_min_factor),
        float(trait_max_factor),
    ).astype(np.float32)
    sand_delta = (float(sand_absolute_sd) * z_sand).astype(np.float32)

    sand = np.asarray(out["sand"], dtype=np.float32).copy()
    clay = np.asarray(out["clay"], dtype=np.float32).copy()
    sand = np.clip(
        sand + sand_delta[:, None],
        float(sand_min_pct),
        float(sand_max_pct),
    ).astype(np.float32, copy=False)
    clay_cap = np.maximum(0.0, 99.0 - sand)
    clay = np.minimum(clay, clay_cap).astype(np.float32, copy=False)
    out["sand"] = sand
    out["clay"] = clay
    out["synthetic_flnr_factor"] = flnr_factor
    out["synthetic_mbbopt_factor"] = mbbopt_factor
    out["synthetic_pct_sand_delta"] = sand_delta
    out["synthetic_pct_sand_lev1"] = sand[:, 0].astype(np.float32, copy=False)
    out["synthetic_pct_sand_mean"] = sand.mean(axis=1).astype(np.float32, copy=False)
    out["synthetic_clone_grid_rows"] = nrow_i
    out["synthetic_clone_grid_cols"] = ncol_i
    out["synthetic_clone_corr_length_cells"] = float(corr_length_cells)
    out["synthetic_clone_seed"] = int(seed)
    return out


def _add_center_square_precip_treatment(
    surf: dict,
    *,
    area_ha: float,
    reduction: float,
    cell_size_m: float,
    center_x_m: float = 0.0,
    center_y_m: float = 0.0,
) -> dict:
    """Add a centered square precipitation treatment multiplier to ``surf``.

    The square is represented with fractional overlap on edge cells so the
    effective treated area matches ``area_ha`` even when the side length does
    not fall exactly on synthetic clone-grid cell edges.
    """
    area_ha = float(area_ha)
    reduction = float(reduction)
    cell_size_m = float(cell_size_m)
    if area_ha <= 0.0:
        return surf
    if not np.isfinite(area_ha) or area_ha < 0.0:
        raise ValueError("--regional-precip-treatment-area-ha must be finite and non-negative")
    if not np.isfinite(reduction) or reduction < 0.0 or reduction > 1.0:
        raise ValueError("--regional-precip-treatment-reduction must be in [0, 1]")
    if not np.isfinite(cell_size_m) or cell_size_m <= 0.0:
        raise ValueError("--regional-precip-treatment-cell-size-m must be positive")

    rows = np.asarray(surf["grid_row"], dtype=np.int32)
    cols = np.asarray(surf["grid_col"], dtype=np.int32)
    nrow = int(surf.get("synthetic_clone_grid_rows", int(rows.max()) + 1))
    ncol = int(surf.get("synthetic_clone_grid_cols", int(cols.max()) + 1))
    x = (cols.astype(np.float64) + 0.5 - 0.5 * ncol) * cell_size_m
    y = (0.5 * nrow - (rows.astype(np.float64) + 0.5)) * cell_size_m

    side_m = math.sqrt(area_ha * 10000.0)
    half = 0.5 * side_m
    x0 = float(center_x_m) - half
    x1 = float(center_x_m) + half
    y0 = float(center_y_m) - half
    y1 = float(center_y_m) + half
    cell_half = 0.5 * cell_size_m
    overlap_x = np.maximum(
        0.0,
        np.minimum(x + cell_half, x1) - np.maximum(x - cell_half, x0),
    )
    overlap_y = np.maximum(
        0.0,
        np.minimum(y + cell_half, y1) - np.maximum(y - cell_half, y0),
    )
    area_fraction = np.clip((overlap_x * overlap_y) / (cell_size_m * cell_size_m), 0.0, 1.0)
    multiplier = np.asarray(1.0 - reduction * area_fraction, dtype=np.float32)

    out = dict(surf)
    out["precip_treatment_area_fraction"] = area_fraction.astype(np.float32)
    out["precip_treatment_multiplier"] = multiplier
    out["precip_treatment_area_ha_requested"] = float(area_ha)
    out["precip_treatment_area_ha_effective"] = float(area_fraction.sum() * cell_size_m * cell_size_m / 10000.0)
    out["precip_treatment_reduction"] = float(reduction)
    out["precip_treatment_cell_size_m"] = float(cell_size_m)
    out["precip_treatment_side_m"] = float(side_m)
    out["precip_treatment_center_x_m"] = float(center_x_m)
    out["precip_treatment_center_y_m"] = float(center_y_m)
    out["precip_treatment_cell_count_any_overlap"] = int(np.count_nonzero(area_fraction > 0.0))
    out["precip_treatment_cell_equivalent_count"] = float(area_fraction.sum())
    return out


def _parse_optional_yyyymmdd(value: int | str | None, *, name: str) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if len(text) != 8 or not text.isdigit():
        raise ValueError(f"{name} must be formatted as YYYYMMDD")
    year = int(text[:4])
    month = int(text[4:6])
    day = int(text[6:8])
    if month < 1 or month > 12:
        raise ValueError(f"{name} month must be in [1, 12]")
    if day < 1 or day > 31:
        raise ValueError(f"{name} day must be in [1, 31]")
    return int(text)


def _month_date_bounds(year: int, month: int, *, calendar: str) -> tuple[int, int]:
    first = int(year) * 10000 + int(month) * 100 + 1
    last = int(year) * 10000 + int(month) * 100 + _days_in_month(int(year), int(month), calendar)
    return first, last


def _month_intersects_date_window(
    year: int,
    month: int,
    *,
    calendar: str,
    start_date: int | None,
    end_date: int | None,
) -> bool:
    first, last = _month_date_bounds(year, month, calendar=calendar)
    if start_date is not None and last < int(start_date):
        return False
    if end_date is not None and first > int(end_date):
        return False
    return True


def _synthetic_trait_value_for_surf(
    surf: dict,
    factor_key: str,
    base_value: float,
) -> float | np.ndarray:
    """Return scalar trait value or per-cell ``(ncell, 1)`` value for clone grids."""
    if factor_key not in surf:
        return float(base_value)
    factor = np.asarray(surf[factor_key], dtype=np.float32).reshape(-1, 1)
    return (float(base_value) * factor).astype(np.float32, copy=False)
