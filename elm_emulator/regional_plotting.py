"""Regional diagnostic map plotting helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Diagnostic map plotting
# ---------------------------------------------------------------------------

def _segments_from_shapely_geometry(geom: object) -> "list[tuple[np.ndarray, np.ndarray]]":
    """Convert shapely line/polygon geometry into lon/lat line segments."""
    segments: list[tuple[np.ndarray, np.ndarray]] = []
    gtype = getattr(geom, "geom_type", "")
    if gtype == "LineString":
        coords = np.asarray(getattr(geom, "coords"), dtype=np.float32)
        if coords.shape[0] >= 2:
            segments.append((coords[:, 0], coords[:, 1]))
    elif gtype == "MultiLineString":
        for part in getattr(geom, "geoms", []):
            segments.extend(_segments_from_shapely_geometry(part))
    elif gtype == "Polygon":
        exterior = getattr(geom, "exterior", None)
        if exterior is not None:
            coords = np.asarray(exterior.coords, dtype=np.float32)
            if coords.shape[0] >= 2:
                segments.append((coords[:, 0], coords[:, 1]))
    elif gtype == "MultiPolygon":
        for part in getattr(geom, "geoms", []):
            segments.extend(_segments_from_shapely_geometry(part))
    return segments


def _load_natural_earth_segments(
    *,
    category: str,
    name: str,
    resolution: str = "50m",
) -> "list[tuple[np.ndarray, np.ndarray]] | None":
    """Return Natural Earth line segments through Cartopy if cached/available."""
    try:
        from cartopy.io import shapereader

        path = shapereader.natural_earth(
            resolution=resolution,
            category=category,
            name=name,
        )
        reader = shapereader.Reader(path)
        segments: list[tuple[np.ndarray, np.ndarray]] = []
        for geom in reader.geometries():
            segments.extend(_segments_from_shapely_geometry(geom))
        return segments
    except Exception as exc:
        print(f"  Natural Earth {name} unavailable ({exc})")
        return None


def _plot_diag_maps(
    diag: "dict[str, np.ndarray]",
    n_months: int,
    grid_row: np.ndarray,
    grid_col: np.ndarray,
    nlat: int,
    nlon: int,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    out_dir: "Path",
    label: str,
    map_lat_centers: np.ndarray | None = None,
    map_lon_centers: np.ndarray | None = None,
) -> None:
    """Reconstruct (nlat, nlon) grids from (ncell,) arrays and save PNG maps."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    coast_segs = _load_natural_earth_segments(category="physical", name="coastline")
    border_segs = _load_natural_earth_segments(category="cultural", name="admin_0_boundary_lines_land")

    def _edges_from_sorted_centers(centers: np.ndarray, fallback_min: float, fallback_max: float) -> tuple[float, float]:
        centers = np.asarray(centers, dtype=np.float64)
        finite = centers[np.isfinite(centers)]
        if finite.size < 2:
            return float(fallback_min), float(fallback_max)
        diffs = np.diff(finite)
        diffs = diffs[np.isfinite(diffs) & (np.abs(diffs) > 0.0)]
        if diffs.size == 0:
            return float(fallback_min), float(fallback_max)
        step = float(np.nanmedian(np.abs(diffs)))
        return float(finite[0] - 0.5 * step), float(finite[-1] + 0.5 * step)

    lat_centers = None if map_lat_centers is None else np.asarray(map_lat_centers, dtype=np.float64)
    lon_centers = None if map_lon_centers is None else np.asarray(map_lon_centers, dtype=np.float64)
    row_order = None
    col_order = None
    row_slice = slice(0, nlat)
    col_slice = slice(0, nlon)
    if lat_centers is not None and lat_centers.ndim == 1 and lat_centers.size == nlat:
        row_order = np.argsort(lat_centers)
        lat_sorted = lat_centers[row_order]
    else:
        lat_sorted = None
    if lon_centers is not None and lon_centers.ndim == 1 and lon_centers.size == nlon:
        col_order = np.argsort(lon_centers)
        lon_sorted = lon_centers[col_order]
    else:
        lon_sorted = None

    if grid_row.size:
        if row_order is not None:
            row_pos = np.empty(nlat, dtype=np.int64)
            row_pos[row_order] = np.arange(nlat, dtype=np.int64)
            used_rows = row_pos[np.asarray(grid_row, dtype=np.int64)]
        else:
            used_rows = np.asarray(grid_row, dtype=np.int64)
        row_start = max(0, int(np.nanmin(used_rows)))
        row_stop = min(nlat, int(np.nanmax(used_rows)) + 1)
        row_slice = slice(row_start, row_stop)
    if grid_col.size:
        if col_order is not None:
            col_pos = np.empty(nlon, dtype=np.int64)
            col_pos[col_order] = np.arange(nlon, dtype=np.int64)
            used_cols = col_pos[np.asarray(grid_col, dtype=np.int64)]
        else:
            used_cols = np.asarray(grid_col, dtype=np.int64)
        col_start = max(0, int(np.nanmin(used_cols)))
        col_stop = min(nlon, int(np.nanmax(used_cols)) + 1)
        col_slice = slice(col_start, col_stop)

    if lat_sorted is not None:
        plot_lat_min, plot_lat_max = _edges_from_sorted_centers(lat_sorted[row_slice], lat_min, lat_max)
    else:
        plot_lat_min, plot_lat_max = float(lat_min), float(lat_max)
    if lon_sorted is not None:
        plot_lon_min, plot_lon_max = _edges_from_sorted_centers(lon_sorted[col_slice], lon_min, lon_max)
    else:
        plot_lon_min, plot_lon_max = float(lon_min), float(lon_max)

    # Display rows/columns in increasing geographic coordinates before using
    # origin="lower". TRENDY global surfdata stores longitude as 0..360, while
    # the maps are labeled -180..180.
    extent = [plot_lon_min, plot_lon_max, plot_lat_min, plot_lat_max]

    configs = {
        "forc_t_mean": {
            "title": "Mean 2-m Air Temperature",
            "units": "K",
            "cmap": "RdBu_r",
        },
        "fpsn_mean": {
            "title": "Mean Net Photosynthesis (fpsn_patch)",
            "units": "µmol CO₂ m⁻² s⁻¹",
            "cmap": "YlGn",
        },
        "gpp_gc_m2_d_mean": {
            "title": "Mean GPP Proxy from Photosynthesis",
            "units": "gC m⁻² d⁻¹",
            "cmap": "YlGn",
        },
        "lai_mean": {
            "title": "Mean Leaf Area Index (area-weighted)",
            "units": "m² m⁻²",
            "cmap": "Greens",
        },
        "t_soil_mean": {
            "title": "Mean Top-Layer Soil Temperature",
            "units": "K",
            "cmap": "RdBu_r",
        },
        "t_veg_mean": {
            "title": "Mean Vegetation Temperature",
            "units": "K",
            "cmap": "RdBu_r",
        },
        "liq_mean": {
            "title": "Mean Top-Layer Volumetric Soil Water",
            "units": "m³ m⁻³",
            "cmap": "Blues",
        },
        "h2osfc_mean": {
            "title": "Mean Surface Water Storage",
            "units": "mm",
            "cmap": "PuBu",
        },
        "h2osno_mean": {
            "title": "Mean Snow Water Equivalent",
            "units": "kg m-2",
            "cmap": "Blues",
        },
        "frac_sno_mean": {
            "title": "Mean Raw Snow Fraction",
            "units": "fraction",
            "cmap": "Blues",
        },
        "active_frac_sno_mean": {
            "title": "Mean Active Snow Fraction",
            "units": "fraction",
            "cmap": "Blues",
        },
        "qflx_tran_veg_mean": {
            "title": "Mean Vegetation Transpiration",
            "units": "mm d⁻¹",
            "cmap": "YlGnBu",
            "scale": 86400.0,
        },
        "eflx_sh_tot_mean": {
            "title": "Mean Sensible Heat Flux",
            "units": "W m⁻²",
            "cmap": "RdBu_r",
        },
        "qflx_evap_tot_mean": {
            "title": "Mean Total Evaporation",
            "units": "mm d⁻¹",
            "cmap": "YlGnBu",
            "scale": 86400.0,
        },
        "eflx_lh_tot_mean": {
            "title": "Mean Latent Heat Flux",
            "units": "W m⁻²",
            "cmap": "YlGnBu",
        },
        "qflx_surf_mean": {
            "title": "Mean Surface Runoff",
            "units": "mm d⁻¹",
            "cmap": "PuBuGn",
            "scale": 86400.0,
        },
        "qflx_h2osfc_surf_mean": {
            "title": "Mean Surface Water Runoff",
            "units": "mm d⁻¹",
            "cmap": "PuBuGn",
            "scale": 86400.0,
        },
        "qflx_drain_mean": {
            "title": "Mean Subsurface Drainage",
            "units": "mm d⁻¹",
            "cmap": "PuBuGn",
            "scale": 86400.0,
        },
        "qflx_deep_drain_mean": {
            "title": "Mean Deep Drainage",
            "units": "mm d⁻¹",
            "cmap": "PuBuGn",
            "scale": 86400.0,
        },
        "qflx_runoff_tot_mean": {
            "title": "Mean Total Runoff",
            "units": "mm d⁻¹",
            "cmap": "PuBuGn",
            "scale": 86400.0,
        },
        "snow_depth_mean": {
            "title": "Mean Snow Depth",
            "units": "m",
            "cmap": "Blues",
        },
        "active_snow_depth_mean": {
            "title": "Mean Active Snow Depth",
            "units": "m",
            "cmap": "Blues",
        },
        "snow_depth_eff_mean": {
            "title": "Mean Snow Depth x Snow Fraction",
            "units": "m",
            "cmap": "Blues",
        },
        "mlp_ood_max_abs_z": {
            "title": "Mean MLP Input Max |z|",
            "units": "training σ",
            "cmap": "magma",
        },
        "mlp_ood_frac": {
            "title": "Mean Fraction of MLP Inputs Beyond OOD Threshold",
            "units": "fraction",
            "cmap": "viridis",
        },
        "synthetic_flnr_factor": {
            "title": "Synthetic flnr Factor",
            "units": "1",
            "cmap": "viridis",
        },
        "synthetic_mbbopt_factor": {
            "title": "Synthetic mbbopt Factor",
            "units": "1",
            "cmap": "viridis",
        },
        "synthetic_pct_sand_delta": {
            "title": "Synthetic PCT_SAND Perturbation",
            "units": "percentage points",
            "cmap": "RdBu_r",
        },
        "synthetic_pct_sand_lev1": {
            "title": "Synthetic Layer-1 PCT_SAND",
            "units": "%",
            "cmap": "YlOrBr",
        },
        "synthetic_pct_sand_mean": {
            "title": "Synthetic Mean PCT_SAND",
            "units": "%",
            "cmap": "YlOrBr",
        },
    }

    for key, arr in diag.items():
        arr = arr / n_months
        if key.startswith("mlp_ood_abs_z_"):
            feature_name = key[len("mlp_ood_abs_z_"):]
            cfg = {
                "title": f"Mean MLP |z| for {feature_name}",
                "units": "training σ",
                "cmap": "magma",
            }
        else:
            cfg = configs.get(key, {"title": key, "units": "", "cmap": "viridis"})
        scale = cfg.get("scale", 1.0)

        vals = (arr * scale).astype(np.float32)
        # Mask obvious fill values before inserting into grid.
        vals[(vals < -9000) | (vals > 1e10)] = np.nan

        grid2d = np.full((nlat, nlon), np.nan, dtype=np.float32)
        grid2d[grid_row, grid_col] = vals
        if row_order is not None:
            grid2d = grid2d[row_order, :]
        if col_order is not None:
            grid2d = grid2d[:, col_order]
        grid2d = grid2d[row_slice, col_slice]

        valid = grid2d[np.isfinite(grid2d)]
        vmin = float(valid.min()) if valid.size > 0 else None
        vmax = float(valid.max()) if valid.size > 0 else None

        fig, ax = plt.subplots(figsize=(13, 6))
        im = ax.imshow(
            grid2d,
            origin="lower",          # row 0 = lat_min (south) → correct orientation
            extent=extent,
            aspect="auto",
            cmap=cfg["cmap"],
            interpolation="nearest",
            vmin=vmin,
            vmax=vmax,
        )
        if coast_segs:
            for seg_lon, seg_lat in coast_segs:
                ax.plot(seg_lon, seg_lat, color="black", linewidth=0.45, alpha=0.7, transform=ax.transData)
        if border_segs:
            for seg_lon, seg_lat in border_segs:
                ax.plot(seg_lon, seg_lat, color="black", linewidth=0.7, alpha=0.9, transform=ax.transData)

        cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
        cbar.set_label(cfg["units"])
        ax.set_title(f"{cfg['title']}  ({label})")
        ax.set_xlabel("Longitude (°)")
        ax.set_ylabel("Latitude (°)")
        ax.set_xlim(plot_lon_min, plot_lon_max)
        ax.set_ylim(plot_lat_min, plot_lat_max)

        key_out_dir = out_dir / "ood" if key.startswith("mlp_ood") else out_dir
        key_out_dir.mkdir(parents=True, exist_ok=True)
        out_path = key_out_dir / f"regional_{key}_{label}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {out_path}")




def _filter_regional_plot_diag_maps(
    diag_maps: dict[str, np.ndarray],
    plot_variables: list[str] | tuple[str, ...] | None,
) -> dict[str, np.ndarray]:
    """Return only requested regional map variables, preserving insertion order."""
    if not plot_variables:
        return dict(diag_maps)
    wanted = {str(name) for name in plot_variables}
    return {key: value for key, value in diag_maps.items() if key in wanted}
