#!/usr/bin/env python3
"""Build a 1/24-degree MODIS LAI climatology and insert it into ELM surfdata.

The source is the 500 m MCD15A3H Collection 6.1 product hosted as cloud-
optimized GeoTIFFs by Microsoft Planetary Computer.  One four-day composite
nearest the middle of each month is sampled in each requested year.  Only
main-algorithm LAI retrievals (including saturated-but-usable retrievals) are
retained before area averaging to the destination grid.

MCD15A3H supplies gridcell LAI rather than ELM-PFT LAI.  To disaggregate it,
this script derives one domain-wide monthly LAI template for every ELM PFT
from the input surfdata.  Within each destination cell it scales those PFT
templates so their PCT_NATVEG * PCT_NAT_PFT weighted mean matches MODIS.  This
preserves distinct PFT seasonal cycles without fitting an unidentifiable ML
model.  MONTHLY_SAI and every other data variable are left unchanged.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import shutil
import time
import urllib.parse
import urllib.request
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from netCDF4 import Dataset
from pyproj import Transformer
from rasterio.merge import merge
from rasterio.warp import Resampling, reproject


STAC_SEARCH = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
SIGN_URL = "https://planetarycomputer.microsoft.com/api/sas/v1/sign?href="
COLLECTION = "modis-15A3H-061"
LAI_MAX = 10.0
EVERGREEN_PFTS = (1, 2, 4, 5, 9)
DECIDUOUS_WOODY_PFTS = (3, 6, 7, 8, 10, 11)
SEASONAL_HERBACEOUS_PFTS = (12, 13, 14, 15, 16)
WINTER_MONTHS = (11, 0, 1)  # December, January, February; zero-based.


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--surfdata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--climatology-output", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        default=[2015, 2016, 2017, 2018, 2019],
        help="MODIS years used for the mid-month climatology.",
    )
    parser.add_argument("--workers", type=int, default=3, help="Concurrent month composites to process.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def request_json(url: str, payload: dict | None = None, retries: int = 5) -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"User-Agent": "ELM-emulator-MODIS-LAI/1.0"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers, method="POST" if data else "GET")
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return json.load(response)
        except Exception:
            if attempt + 1 == retries:
                raise
            time.sleep(2 ** attempt)
    raise AssertionError("unreachable")


def stac_items_for_midmonth(year: int, month: int, bbox: list[float], cache: Path) -> list[dict]:
    cache.mkdir(parents=True, exist_ok=True)
    cache_path = cache / f"{COLLECTION}_{year:04d}_{month:02d}.json"
    if cache_path.exists():
        result = json.loads(cache_path.read_text())
    else:
        result = request_json(
            STAC_SEARCH,
            {
                "collections": [COLLECTION],
                "bbox": bbox,
                "datetime": f"{year:04d}-{month:02d}-10T00:00:00Z/{year:04d}-{month:02d}-20T23:59:59Z",
                "limit": 100,
            },
        )
        cache_path.write_text(json.dumps(result, indent=2) + "\n")
    features = result.get("features", [])
    if not features:
        raise RuntimeError(f"No {COLLECTION} items found for {year:04d}-{month:02d}")

    starts = sorted({item["properties"]["start_datetime"] for item in features})
    target = datetime(year, month, 15, tzinfo=timezone.utc)
    selected_start = min(
        starts,
        key=lambda value: abs(
            (datetime.fromisoformat(value.replace("Z", "+00:00")) - target).total_seconds()
        ),
    )
    selected = [item for item in features if item["properties"]["start_datetime"] == selected_start]
    if not selected:
        raise AssertionError("Selected MODIS composite has no tiles")
    return selected


def signed_asset(item: dict, name: str) -> str:
    href = item["assets"][name]["href"]
    response = request_json(SIGN_URL + urllib.parse.quote(href, safe=""))
    return str(response["href"])


def target_grid(surfdata: Path) -> dict:
    with Dataset(surfdata) as ds:
        lat_cell = np.asarray(ds.variables["LATIXY"][:], dtype=np.float64)
        lon_cell = np.asarray(ds.variables["LONGXY"][:], dtype=np.float64)
    lon_cell = ((lon_cell + 180.0) % 360.0) - 180.0
    lat = np.unique(np.round(lat_cell, 8))
    lon = np.unique(np.round(lon_cell, 8))
    if lat.size * lon.size != lat_cell.size:
        raise ValueError("Surfdata coordinates do not form a complete rectilinear grid")
    dy = float(np.median(np.diff(lat)))
    dx = float(np.median(np.diff(lon)))
    west = float(lon.min() - dx / 2.0)
    east = float(lon.max() + dx / 2.0)
    south = float(lat.min() - dy / 2.0)
    north = float(lat.max() + dy / 2.0)
    return {
        "lat_cell": lat_cell,
        "lon_cell": lon_cell,
        "lat": lat,
        "lon": lon,
        "dx": dx,
        "dy": dy,
        "bbox": [west, south, east, north],
        "transform": Affine.translation(west, north) * Affine.scale(dx, -dy),
    }


def transformed_bounds(bbox: list[float], dst_crs) -> tuple[float, float, float, float]:
    west, south, east, north = bbox
    edge = np.linspace(0.0, 1.0, 101)
    lon = np.concatenate(
        [west + edge * (east - west), west + edge * (east - west), np.full(101, west), np.full(101, east)]
    )
    lat = np.concatenate(
        [np.full(101, south), np.full(101, north), south + edge * (north - south), south + edge * (north - south)]
    )
    transformer = Transformer.from_crs("EPSG:4326", dst_crs, always_xy=True)
    x, y = transformer.transform(lon, lat)
    return float(np.min(x)), float(np.min(y)), float(np.max(x)), float(np.max(y))


def mosaic_asset(items: list[dict], asset: str, bounds=None):
    with contextlib.ExitStack() as stack:
        sources = [stack.enter_context(rasterio.open(signed_asset(item, asset))) for item in items]
        if bounds is None:
            bounds = transformed_bounds(GLOBAL_GRID["bbox"], sources[0].crs)
        data, transform = merge(sources, bounds=bounds, nodata=255, dtype=np.uint8)
        return data[0], transform, sources[0].crs, bounds


def aggregate_composite(items: list[dict], grid: dict) -> tuple[np.ndarray, np.ndarray]:
    lai_raw, src_transform, src_crs, bounds = mosaic_asset(items, "Lai_500m")
    qc, qc_transform, qc_crs, _ = mosaic_asset(items, "FparLai_QC", bounds=bounds)
    if lai_raw.shape != qc.shape or src_transform != qc_transform or src_crs != qc_crs:
        raise ValueError("MODIS LAI and QC mosaics are not aligned")

    scf_qc = (qc >> 5) & 0b111
    valid = (lai_raw <= 100) & ((qc & 0b1) == 0) & (scf_qc <= 1)
    source = np.where(valid, lai_raw.astype(np.float32) * 0.1, np.nan).astype(np.float32)
    support = valid.astype(np.float32)
    shape = (grid["lat"].size, grid["lon"].size)
    lai_north = np.full(shape, np.nan, dtype=np.float32)
    support_north = np.zeros(shape, dtype=np.float32)
    reproject(
        source,
        lai_north,
        src_transform=src_transform,
        src_crs=src_crs,
        src_nodata=np.nan,
        dst_transform=grid["transform"],
        dst_crs="EPSG:4326",
        dst_nodata=np.nan,
        resampling=Resampling.average,
    )
    reproject(
        support,
        support_north,
        src_transform=src_transform,
        src_crs=src_crs,
        src_nodata=None,
        dst_transform=grid["transform"],
        dst_crs="EPSG:4326",
        dst_nodata=0.0,
        resampling=Resampling.average,
    )
    # NetCDF latitude is stored south-to-north; rasterio destinations are north-to-south.
    return np.flipud(lai_north), np.flipud(support_north)


def build_climatology(grid: dict, years: list[int], work_dir: Path, workers: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    annual = np.full((len(years), 12, grid["lat"].size, grid["lon"].size), np.nan, dtype=np.float32)
    support = np.zeros_like(annual)
    def process(yi: int, year: int, month: int):
        composite_dir = work_dir / "composites"
        composite_dir.mkdir(parents=True, exist_ok=True)
        composite_cache = composite_dir / f"{COLLECTION}_{year:04d}_{month:02d}.npz"
        if composite_cache.exists():
            with np.load(composite_cache) as cached:
                return yi, year, month, int(cached["nitems"]), cached["lai"], cached["support"]
        items = stac_items_for_midmonth(year, month, grid["bbox"], work_dir / "stac")
        lai, good_support = aggregate_composite(items, grid)
        np.savez_compressed(composite_cache, lai=lai, support=good_support, nitems=len(items))
        return yi, year, month, len(items), lai, good_support

    jobs = [(yi, year, month) for yi, year in enumerate(years) for month in range(1, 13)]
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = [executor.submit(process, *job) for job in jobs]
        for future in as_completed(futures):
            yi, year, month, nitems, lai, good_support = future.result()
            annual[yi, month - 1] = lai
            support[yi, month - 1] = good_support
            good = np.isfinite(lai)
            print(
                f"{year:04d}-{month:02d}: tiles={nitems} "
                f"valid_target_cells={good.mean() * 100.0:.2f}% "
                f"mean_500m_support={good_support[good].mean() * 100.0:.1f}%",
                flush=True,
            )
    with warnings.catch_warnings():
        # Ocean cells legitimately contain no valid MODIS retrievals in any year.
        warnings.simplefilter("ignore", category=RuntimeWarning)
        climatology = np.nanmedian(annual, axis=0).astype(np.float32)
        mean_support = np.nanmean(np.where(np.isfinite(annual), support, np.nan), axis=0).astype(np.float32)
    observation_count = np.isfinite(annual).sum(axis=0).astype(np.uint8)
    return climatology, observation_count, mean_support


def write_climatology(path: Path, lai: np.ndarray, count: np.ndarray, support: np.ndarray, grid: dict, years: list[int], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {path}; pass --overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    with Dataset(path, "w", format="NETCDF4_CLASSIC") as ds:
        ds.createDimension("month", 12)
        ds.createDimension("lat", grid["lat"].size)
        ds.createDimension("lon", grid["lon"].size)
        ds.createVariable("month", "i2", ("month",))[:] = np.arange(1, 13)
        ds.createVariable("lat", "f8", ("lat",))[:] = grid["lat"]
        ds.createVariable("lon", "f8", ("lon",))[:] = grid["lon"]
        var = ds.createVariable("LAI", "f4", ("month", "lat", "lon"), fill_value=np.float32(np.nan), zlib=True, complevel=4)
        var[:] = lai
        var.units = "m2 m-2"
        var.long_name = "mid-month MODIS LAI climatology area-averaged to 1/24 degree"
        ds.createVariable("observation_count", "i1", ("month", "lat", "lon"), zlib=True)[:] = count.astype(np.int8)
        ds.createVariable("mean_good_pixel_fraction", "f4", ("month", "lat", "lon"), fill_value=np.float32(np.nan), zlib=True)[:] = support
        ds.source = "NASA MCD15A3H Collection 6.1 via Microsoft Planetary Computer"
        ds.source_doi = "10.5067/MODIS/MCD15A3H.061"
        ds.climatology_years = ",".join(str(year) for year in years)
        ds.temporal_sampling = "one 4-day composite nearest the 15th of each month in each year"
        ds.quality_screen = "Lai_500m <= 100; MODLAND_QC=0; SCF_QC in {0,1}; scale factor 0.1"
        ds.history = f"{datetime.now(timezone.utc).isoformat()}: created by scripts/build_modis_lai_surface.py"
    print(f"Wrote {path}", flush=True)


def weighted_pft_templates(old_lai: np.ndarray, pct_pft: np.ndarray, pct_natveg: np.ndarray, landfrac: np.ndarray) -> np.ndarray:
    weights = (pct_pft / 100.0) * (pct_natveg[None, :] / 100.0) * landfrac[None, :]
    template = np.zeros((12, old_lai.shape[1]), dtype=np.float64)
    for month in range(12):
        for pft in range(1, old_lai.shape[1]):
            valid = (weights[pft] > 0.0) & np.isfinite(old_lai[month, pft]) & (old_lai[month, pft] >= 0.0)
            if np.any(valid):
                template[month, pft] = np.average(old_lai[month, pft, valid], weights=weights[pft, valid])
    return template


def disaggregate_to_pfts(surfdata: Path, climatology: np.ndarray, grid: dict) -> tuple[np.ndarray, dict]:
    with Dataset(surfdata) as ds:
        old = np.asarray(ds.variables["MONTHLY_LAI"][:], dtype=np.float64)
        pct = np.asarray(ds.variables["PCT_NAT_PFT"][:], dtype=np.float64)
        pct_natveg = np.asarray(ds.variables["PCT_NATVEG"][:], dtype=np.float64)
        landfrac = np.asarray(ds.variables["LANDFRAC_PFT"][:], dtype=np.float64)

    row = np.searchsorted(grid["lat"], np.round(grid["lat_cell"], 8))
    col = np.searchsorted(grid["lon"], np.round(grid["lon_cell"], 8))
    target = climatology[:, row, col].astype(np.float64)
    template = weighted_pft_templates(old, pct, pct_natveg, landfrac)
    # These are the same one-column area weights used by regional_surfdata:
    # natural-vegetation PFT fractions are reduced by PCT_NATVEG, while the
    # unsupported remainder is represented as bare PFT 0. LANDFRAC_PFT is an
    # outer gridcell mask, not a within-column PFT weight.
    weights = (pct / 100.0) * (pct_natveg[None, :] / 100.0)
    weights[0] = 0.0
    new = old.copy()
    update = np.isfinite(target) & (landfrac[None, :] > 0.0)
    new[:, 1:, :][np.broadcast_to(update[:, None, :], new[:, 1:, :].shape)] = 0.0
    new[:, 0, :][update] = 0.0

    def allocate(template_pc: np.ndarray, weight_pc: np.ndarray, requested: np.ndarray):
        """Allocate a requested aggregate with capped proportional PFT scaling."""
        template_pc = np.asarray(template_pc, dtype=np.float64)
        weight_pc = np.asarray(weight_pc, dtype=np.float64)
        requested = np.asarray(requested, dtype=np.float64)
        base = np.sum(weight_pc * template_pc, axis=0)
        capacity = LAI_MAX * np.sum(weight_pc * (template_pc > 0.0), axis=0)
        desired = np.minimum(np.maximum(requested, 0.0), capacity)
        solvable = np.isfinite(desired) & (base > 1.0e-12) & (desired > 0.0)
        result = np.zeros_like(template_pc)
        if not np.any(solvable):
            return result, np.zeros_like(requested), capacity
        lo = np.zeros(solvable.sum(), dtype=np.float64)
        hi = np.maximum(desired[solvable] / base[solvable], 1.0)
        weight_s = weight_pc[:, solvable]
        template_s = template_pc[:, solvable]
        for _ in range(16):
            aggregate_hi = np.sum(weight_s * np.minimum(template_s * hi[None, :], LAI_MAX), axis=0)
            grow = aggregate_hi < desired[solvable] - 1.0e-10
            if not np.any(grow):
                break
            hi[grow] *= 2.0
        for _ in range(36):
            mid = 0.5 * (lo + hi)
            aggregate_mid = np.sum(weight_s * np.minimum(template_s * mid[None, :], LAI_MAX), axis=0)
            low = aggregate_mid < desired[solvable]
            lo[low] = mid[low]
            hi[~low] = mid[~low]
        result[:, solvable] = np.minimum(template_s * (0.5 * (lo + hi))[None, :], LAI_MAX)
        achieved = np.sum(weight_pc * result, axis=0)
        return result, achieved, capacity

    evergreen = np.asarray([p for p in EVERGREEN_PFTS if p < old.shape[1]], dtype=np.int64)
    seasonal = np.asarray(
        [p for p in DECIDUOUS_WOODY_PFTS + SEASONAL_HERBACEOUS_PFTS if p < old.shape[1]],
        dtype=np.int64,
    )

    # Winter MODIS LAI is assigned exclusively to evergreen PFTs. The average
    # inferred evergreen patch LAI becomes the baseline used outside winter.
    winter_evergreen = []
    for month in WINTER_MONTHS:
        template_pc = np.broadcast_to(template[month, evergreen, None], (evergreen.size, old.shape[2]))
        allocated, _, _ = allocate(template_pc, weights[evergreen], target[month])
        allocated[:, ~update[month]] = 0.0
        new[month, evergreen, :] = allocated
        winter_evergreen.append(allocated)
    evergreen_baseline = np.mean(winter_evergreen, axis=0)
    capacity = np.zeros_like(target)
    desired = np.zeros_like(target)
    for month in WINTER_MONTHS:
        capacity[month] = LAI_MAX * np.sum(weights[evergreen] * (template[month, evergreen, None] > 0.0), axis=0)
        desired[month] = np.minimum(np.maximum(target[month], 0.0), capacity[month])

    # Outside winter, hold the evergreen component at its MODIS-calibrated
    # dormant-season baseline. Any observed rise or decline above that baseline
    # is therefore timed entirely by the local MODIS seasonal trajectory.
    nonwinter_months = np.asarray([month for month in range(12) if month not in WINTER_MONTHS])
    remaining = np.zeros_like(target)
    evergreen_aggregate_by_month = np.zeros_like(target)
    for month in nonwinter_months:
        evergreen_lai = evergreen_baseline.copy()
        evergreen_aggregate = np.sum(weights[evergreen] * evergreen_lai, axis=0)
        too_high = np.isfinite(target[month]) & (evergreen_aggregate > target[month])
        if np.any(too_high):
            reduced, _, _ = allocate(evergreen_lai[:, too_high], weights[evergreen][:, too_high], target[month, too_high])
            evergreen_lai[:, too_high] = reduced
            evergreen_aggregate[too_high] = np.sum(weights[evergreen][:, too_high] * reduced, axis=0)
        evergreen_lai[:, ~update[month]] = 0.0
        new[month, evergreen, :] = evergreen_lai
        evergreen_aggregate_by_month[month] = np.sum(weights[evergreen] * evergreen_lai, axis=0)
        remaining[month] = np.maximum(target[month] - evergreen_aggregate_by_month[month], 0.0)

    # Infer the local green-up / senescence curve directly from MODIS. Solve
    # PFT amplitudes once at the cell's seasonal peak, then multiply them by
    # this observed 0..1 trajectory. The legacy ELM curves affect relative
    # PFT peak amplitudes only, never the transition timing.
    peak_remaining = np.max(remaining[nonwinter_months], axis=0)
    seasonal_peak_template = np.max(template[nonwinter_months][:, seasonal], axis=0)
    seasonal_template_pc = np.broadcast_to(
        seasonal_peak_template[:, None], (seasonal.size, old.shape[2])
    )
    seasonal_peak_lai, seasonal_peak_aggregate, seasonal_peak_capacity = allocate(
        seasonal_template_pc, weights[seasonal], peak_remaining
    )
    phenology = np.zeros_like(target)
    for month in nonwinter_months:
        phenology[month] = np.divide(
            remaining[month],
            peak_remaining,
            out=np.zeros_like(peak_remaining),
            where=peak_remaining > 1.0e-12,
        )
        phenology[month] = np.clip(phenology[month], 0.0, 1.0)
        seasonal_lai = seasonal_peak_lai * phenology[month][None, :]
        seasonal_lai[:, ~update[month]] = 0.0
        new[month, seasonal, :] = seasonal_lai
        capacity[month] = (
            evergreen_aggregate_by_month[month]
            + seasonal_peak_capacity * phenology[month]
        )
        desired[month] = np.minimum(np.maximum(target[month], 0.0), capacity[month])

    aggregate = np.einsum("pc,mpc->mc", weights, new)
    residual = aggregate - desired
    applicable = update & (capacity > 1.0e-12)
    stats = {
        "template": template,
        "target": target,
        "desired": desired,
        "aggregate": aggregate,
        "modis_phenology": phenology,
        "applicable": applicable,
        "capacity_limited": update & (target > capacity + 1.0e-7),
        "winter_unrepresented": update[np.asarray(WINTER_MONTHS)] & (capacity[np.asarray(WINTER_MONTHS)] <= 1.0e-12) & (target[np.asarray(WINTER_MONTHS)] > 0.0),
        "max_abs_residual": float(np.nanmax(np.abs(residual[applicable]))),
        "updated_fraction": float(update.mean()),
    }
    return new, stats


def write_surface_copy(source: Path, output: Path, monthly_lai: np.ndarray, years: list[int], stats: dict, overwrite: bool) -> None:
    if output.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {output}; pass --overwrite")
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, output)
    with Dataset(output, "r+") as ds:
        var = ds.variables["MONTHLY_LAI"]
        var[:] = monthly_lai
        var.setncattr("source", "NASA MCD15A3H Collection 6.1 500 m LAI")
        var.setncattr("climatology_years", ",".join(str(year) for year in years))
        var.setncattr(
            "pft_disaggregation",
            "winter MODIS LAI allocated only to evergreen PFTs; evergreen baseline carried through the year; "
            "March-November deciduous/grass/crop timing follows the local normalized MODIS transition from "
            "winter baseline to seasonal peak; legacy PFT templates set relative peak amplitudes only; all "
            "allocations use PCT_NATVEG*PCT_NAT_PFT weights and LAI<=10",
        )
        var.setncattr("modis_updated_month_cell_fraction", stats["updated_fraction"])
        ds.setncattr("MODIS_LAI_source", "MCD15A3H.061; doi:10.5067/MODIS/MCD15A3H.061")
        ds.setncattr("MODIS_LAI_climatology_years", ",".join(str(year) for year in years))
        history = str(getattr(ds, "History_Log", "")).rstrip()
        entry = f"{datetime.now(timezone.utc).isoformat()}: MONTHLY_LAI updated by scripts/build_modis_lai_surface.py; MONTHLY_SAI unchanged"
        ds.setncattr("History_Log", f"{history}\n{entry}" if history else entry)
    print(f"Wrote {output}", flush=True)


def main() -> None:
    global GLOBAL_GRID
    args = parse_args()
    if not args.years:
        raise SystemExit("--years requires at least one year")
    args.work_dir.mkdir(parents=True, exist_ok=True)
    GLOBAL_GRID = target_grid(args.surfdata)
    climatology, count, support = build_climatology(GLOBAL_GRID, args.years, args.work_dir, args.workers)
    write_climatology(args.climatology_output, climatology, count, support, GLOBAL_GRID, args.years, args.overwrite)
    new_lai, stats = disaggregate_to_pfts(args.surfdata, climatology, GLOBAL_GRID)
    write_surface_copy(args.surfdata, args.output, new_lai, args.years, stats, args.overwrite)
    print(
        f"Updated {stats['updated_fraction'] * 100.0:.2f}% of month-cell combinations; "
        f"capacity-limited={stats['capacity_limited'].mean() * 100.0:.3f}%; "
        f"winter-unrepresented={stats['winter_unrepresented'].mean() * 100.0:.3f}%; "
        f"max aggregate residual={stats['max_abs_residual']:.3g}",
        flush=True,
    )


GLOBAL_GRID: dict = {}


if __name__ == "__main__":
    main()
