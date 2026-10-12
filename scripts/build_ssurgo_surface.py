#!/usr/bin/env python3
"""Build an ELM regional surfdata file using USDA gSSURGO soils.

The official state gSSURGO packages contain map-unit polygons plus SSURGO
component and horizon tables.  This script:

1. thickness-weights horizon properties onto the ten standard ELM layers;
2. component-weights those profiles to one profile per SSURGO map unit;
3. rasterizes map units on a subgrid of the destination 1/24-degree grid;
4. averages subgrid samples into each destination cell; and
5. overwrites PCT_SAND, PCT_CLAY, and ORGANIC in a copy of the surfdata file.

ORGANIC is organic-matter density in kg m-3, calculated as representative
organic matter mass fraction times representative 1/3-bar bulk density.
Original surfdata values are retained wherever SSURGO lacks spatial coverage
or a required property.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyogrio
import rasterio.features
import shapely
from affine import Affine
from netCDF4 import Dataset
from pyogrio.raw import read as ogr_read
from pyproj import Transformer


ALL_STATES = (
    "TX", "OK", "KS", "AR", "LA", "MS", "AL", "TN", "MO", "IL", "IN",
    "KY", "OH", "WV", "VA", "NC", "SC", "GA", "FL", "MD", "DE", "DC",
    "NJ", "PA",
)

# Standard ELM/CLM 10-layer soil thicknesses.  The cumulative bottoms extend
# to 3.8018 m.  SSURGO's deepest described horizon is extended downward so the
# lower ELM layers do not abruptly revert to a different soil product.
ELM_DZ_M = np.asarray(
    [
        0.01751282,
        0.02757897,
        0.04547003,
        0.07496741,
        0.12355132,
        0.20378255,
        0.33598063,
        0.55393840,
        0.91329000,
        1.50576000,
    ],
    dtype=np.float64,
)
ELM_BOTTOM_CM = 100.0 * np.cumsum(ELM_DZ_M)
ELM_TOP_CM = np.concatenate(([0.0], ELM_BOTTOM_CM[:-1]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--surfdata", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--states", nargs="+", default=list(ALL_STATES))
    parser.add_argument(
        "--subgrid-factor",
        type=int,
        default=8,
        help="Samples per destination-cell axis used for polygon area averaging.",
    )
    parser.add_argument("--polygon-chunk-size", type=int, default=50_000)
    parser.add_argument(
        "--overwrite-output",
        action="store_true",
        help="Replace an existing output file. The input surfdata is never modified.",
    )
    parser.add_argument(
        "--profiles-only",
        action="store_true",
        help="Build cached map-unit profiles without rasterizing or writing surfdata.",
    )
    return parser.parse_args()


def gdb_path(raw_dir: Path, state: str) -> str:
    archive = (raw_dir / f"gSSURGO_{state}.zip").resolve()
    if not archive.exists():
        raise FileNotFoundError(archive)
    return f"/vsizip/{archive}/gSSURGO_{state}.gdb"


def read_table(path: str, layer: str, columns: list[str]) -> pd.DataFrame:
    meta, _, _, values = ogr_read(path, layer=layer, columns=columns)
    return pd.DataFrame({name: value for name, value in zip(meta["fields"], values)})


def build_state_profiles(state: str, raw_dir: Path, cache_dir: Path) -> Path:
    cache = cache_dir / f"mapunit_elm10_{state}.npz"
    if cache.exists():
        print(f"  {state}: using cached map-unit profiles {cache.name}", flush=True)
        return cache

    path = gdb_path(raw_dir, state)
    component = read_table(path, "component", ["mukey", "cokey", "comppct_r"])
    horizon = read_table(
        path,
        "chorizon",
        [
            "cokey", "hzdept_r", "hzdepb_r", "sandtotal_r", "claytotal_r",
            "om_r", "dbthirdbar_r", "dbovendry_r",
        ],
    )
    component = component.dropna(subset=["mukey", "cokey", "comppct_r"]).copy()
    component["mukey"] = pd.to_numeric(component["mukey"], errors="coerce")
    component["cokey"] = pd.to_numeric(component["cokey"], errors="coerce")
    component["comppct_r"] = pd.to_numeric(component["comppct_r"], errors="coerce")
    component = component.dropna().astype({"mukey": "int64", "cokey": "int64"})

    for name in horizon.columns:
        horizon[name] = pd.to_numeric(horizon[name], errors="coerce")
    horizon = horizon.dropna(subset=["cokey", "hzdept_r", "hzdepb_r"]).copy()
    horizon = horizon[horizon["hzdepb_r"] > horizon["hzdept_r"]]
    horizon["cokey"] = horizon["cokey"].astype("int64")
    merged = horizon.merge(component, on="cokey", how="inner", validate="many_to_one")

    # Convert organic matter (%) and bulk density (g cm-3) to organic-matter
    # density (kg m-3). Prefer 1/3-bar density; oven-dry is a fallback.
    bulk_density = merged["dbthirdbar_r"].where(
        merged["dbthirdbar_r"].notna(), merged["dbovendry_r"]
    )
    merged["organic_density"] = merged["om_r"] * 0.01 * bulk_density * 1000.0

    # Extend the first/last described horizons to cover the complete ELM
    # profile. This is preferable to creating a discontinuity to the legacy
    # global soil product at the SSURGO description depth.
    ordered = merged.sort_values(["cokey", "hzdept_r", "hzdepb_r"])
    first = ordered.groupby("cokey", sort=False).head(1).copy()
    first = first[first["hzdept_r"] > 0.0]
    first["hzdepb_r"] = first["hzdept_r"]
    first["hzdept_r"] = 0.0
    last = ordered.groupby("cokey", sort=False).tail(1).copy()
    deepest_cm = float(ELM_BOTTOM_CM[-1])
    last = last[last["hzdepb_r"] < deepest_cm]
    last["hzdept_r"] = last["hzdepb_r"]
    last["hzdepb_r"] = deepest_cm
    merged = pd.concat([merged, first, last], ignore_index=True)

    mukeys = np.sort(merged["mukey"].unique().astype(np.int64))
    profile = np.full((mukeys.size, 3, 10), np.nan, dtype=np.float32)
    properties = ("sandtotal_r", "claytotal_r", "organic_density")
    mukey_values = merged["mukey"].to_numpy(dtype=np.int64)
    component_weight = merged["comppct_r"].to_numpy(dtype=np.float64)
    hz_top = merged["hzdept_r"].to_numpy(dtype=np.float64)
    hz_bottom = merged["hzdepb_r"].to_numpy(dtype=np.float64)
    key_pos = np.searchsorted(mukeys, mukey_values)

    for lev, (layer_top, layer_bottom) in enumerate(zip(ELM_TOP_CM, ELM_BOTTOM_CM)):
        overlap = np.maximum(
            np.minimum(hz_bottom, layer_bottom) - np.maximum(hz_top, layer_top),
            0.0,
        )
        base_weight = component_weight * overlap
        for prop_idx, prop_name in enumerate(properties):
            values = merged[prop_name].to_numpy(dtype=np.float64)
            valid = (base_weight > 0.0) & np.isfinite(values)
            numerator = np.bincount(
                key_pos[valid],
                weights=base_weight[valid] * values[valid],
                minlength=mukeys.size,
            )
            denominator = np.bincount(
                key_pos[valid], weights=base_weight[valid], minlength=mukeys.size
            )
            good = denominator > 0.0
            profile[good, prop_idx, lev] = (numerator[good] / denominator[good]).astype(
                np.float32
            )

    np.savez_compressed(
        cache,
        mukey=mukeys,
        profile=profile,
        elm_top_cm=ELM_TOP_CM,
        elm_bottom_cm=ELM_BOTTOM_CM,
        property_names=np.asarray(["PCT_SAND", "PCT_CLAY", "ORGANIC"]),
    )
    valid_pct = 100.0 * np.isfinite(profile).mean(axis=(0, 2))
    print(
        f"  {state}: {mukeys.size:,} map units; valid profiles "
        f"sand={valid_pct[0]:.1f}% clay={valid_pct[1]:.1f}% organic={valid_pct[2]:.1f}%",
        flush=True,
    )
    return cache


def target_grid(surfdata: Path) -> dict[str, np.ndarray | float | int | Affine]:
    with Dataset(surfdata) as ds:
        lat_cell = np.asarray(ds.variables["LATIXY"][:], dtype=np.float64)
        lon_cell = np.asarray(ds.variables["LONGXY"][:], dtype=np.float64)
    lon_cell = ((lon_cell + 180.0) % 360.0) - 180.0
    lat_centers = np.unique(np.round(lat_cell, 8))
    lon_centers = np.unique(np.round(lon_cell, 8))
    if lat_centers.size * lon_centers.size != lat_cell.size:
        raise ValueError("Destination LATIXY/LONGXY do not form a complete rectilinear grid")
    dy = float(np.median(np.diff(lat_centers)))
    dx = float(np.median(np.diff(lon_centers)))
    if not (math.isclose(dx, 1.0 / 24.0, rel_tol=0.0, abs_tol=2.0e-5) and
            math.isclose(dy, 1.0 / 24.0, rel_tol=0.0, abs_tol=2.0e-5)):
        raise ValueError(f"Expected a 1/24-degree grid, found dx={dx}, dy={dy}")
    return {
        "lat_cell": lat_cell,
        "lon_cell": lon_cell,
        "lat_centers": lat_centers,
        "lon_centers": lon_centers,
        "dx": dx,
        "dy": dy,
        "west": float(lon_centers.min() - 0.5 * dx),
        "north": float(lat_centers.max() + 0.5 * dy),
    }


def rasterize_mapunits(
    states: list[str],
    raw_dir: Path,
    work_dir: Path,
    grid: dict,
    factor: int,
    chunk_size: int,
) -> Path:
    cache = work_dir / f"mukey_subgrid_x{factor}_{'-'.join(states)}.npy"
    metadata_path = cache.with_suffix(".json")
    if cache.exists() and metadata_path.exists():
        print(f"Using cached map-unit raster {cache.name}", flush=True)
        return cache

    nlat = int(np.asarray(grid["lat_centers"]).size)
    nlon = int(np.asarray(grid["lon_centers"]).size)
    height, width = nlat * factor, nlon * factor
    transform = Affine.translation(float(grid["west"]), float(grid["north"])) * Affine.scale(
        float(grid["dx"]) / factor, -float(grid["dy"]) / factor
    )
    mapunit = np.lib.format.open_memmap(cache, mode="w+", dtype=np.int32, shape=(height, width))
    mapunit[:] = 0
    transformer = Transformer.from_crs("EPSG:5070", "EPSG:4326", always_xy=True)

    for state in states:
        path = gdb_path(raw_dir, state)
        info = pyogrio.read_info(path, layer="MUPOLYGON")
        feature_count = int(info["features"])
        print(f"  {state}: rasterizing {feature_count:,} polygons", flush=True)
        for start in range(0, feature_count, chunk_size):
            meta, _, geometry_wkb, fields = ogr_read(
                path,
                layer="MUPOLYGON",
                columns=["MUKEY"],
                skip_features=start,
                max_features=min(chunk_size, feature_count - start),
            )
            if geometry_wkb is None or not fields:
                continue
            mukey = pd.to_numeric(pd.Series(fields[0]), errors="coerce").to_numpy(dtype=np.float64)
            valid = np.isfinite(mukey) & (geometry_wkb != None)  # noqa: E711
            if not np.any(valid):
                continue
            geometry = shapely.from_wkb(geometry_wkb[valid])
            geometry = shapely.transform(geometry, transformer.transform, interleaved=False)
            values = mukey[valid].astype(np.int32)
            rasterio.features.rasterize(
                zip(geometry, values),
                out=mapunit,
                transform=transform,
                fill=0,
                all_touched=False,
                dtype=np.int32,
            )
            print(
                f"    {min(start + chunk_size, feature_count):,}/{feature_count:,}",
                end="\r",
                flush=True,
            )
        print("", flush=True)
        mapunit.flush()

    metadata_path.write_text(
        json.dumps(
            {
                "states": states,
                "factor": factor,
                "shape": [height, width],
                "transform": list(transform),
                "created_utc": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        )
        + "\n"
    )
    return cache


def combine_profiles(profile_paths: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    keys: list[np.ndarray] = []
    values: list[np.ndarray] = []
    for path in profile_paths:
        with np.load(path) as data:
            keys.append(np.asarray(data["mukey"], dtype=np.int64))
            values.append(np.asarray(data["profile"], dtype=np.float32))
    mukey = np.concatenate(keys)
    profile = np.concatenate(values, axis=0)
    order = np.argsort(mukey, kind="stable")
    mukey = mukey[order]
    profile = profile[order]
    keep = np.concatenate(([True], mukey[1:] != mukey[:-1]))
    return mukey[keep], profile[keep]


def aggregate_to_destination(
    raster_path: Path,
    mukey: np.ndarray,
    profile: np.ndarray,
    grid: dict,
    factor: int,
) -> tuple[np.ndarray, np.ndarray]:
    nlat = int(np.asarray(grid["lat_centers"]).size)
    nlon = int(np.asarray(grid["lon_centers"]).size)
    mapunit = np.load(raster_path, mmap_mode="r")
    expected_shape = (nlat * factor, nlon * factor)
    if mapunit.shape != expected_shape:
        raise ValueError(f"Cached map-unit raster has shape {mapunit.shape}, expected {expected_shape}")

    flat_key = np.asarray(mapunit).reshape(-1).astype(np.int64, copy=False)
    key_pos = np.searchsorted(mukey, flat_key)
    key_pos_clip = np.minimum(key_pos, mukey.size - 1)
    spatial_valid = (flat_key > 0) & (key_pos < mukey.size) & (mukey[key_pos_clip] == flat_key)
    output = np.full((3, 10, nlat, nlon), np.nan, dtype=np.float32)
    coverage = np.zeros((3, 10, nlat, nlon), dtype=np.float32)

    for prop in range(3):
        for lev in range(10):
            fine = np.full(flat_key.size, np.nan, dtype=np.float32)
            values = profile[key_pos_clip[spatial_valid], prop, lev]
            fine[spatial_valid] = values
            fine = fine.reshape(nlat, factor, nlon, factor)
            valid = np.isfinite(fine)
            count = valid.sum(axis=(1, 3))
            total = np.where(valid, fine, 0.0).sum(axis=(1, 3), dtype=np.float64)
            good = count > 0
            output[prop, lev, good] = (total[good] / count[good]).astype(np.float32)
            coverage[prop, lev] = count.astype(np.float32) / float(factor * factor)
    return output, coverage


def grid_to_cell_order(values: np.ndarray, grid: dict) -> np.ndarray:
    lat_cell = np.asarray(grid["lat_cell"])
    lon_cell = np.asarray(grid["lon_cell"])
    lat_centers = np.asarray(grid["lat_centers"])
    lon_centers = np.asarray(grid["lon_centers"])
    row = np.rint((lat_centers.max() - lat_cell) / float(grid["dy"])).astype(np.int64)
    col = np.rint((lon_cell - lon_centers.min()) / float(grid["dx"])).astype(np.int64)
    return values[..., row, col]


def write_surface_copy(
    source: Path,
    output: Path,
    values_grid: np.ndarray,
    coverage_grid: np.ndarray,
    grid: dict,
    states: list[str],
    factor: int,
    overwrite: bool,
) -> None:
    if output.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {output}; pass --overwrite-output to replace it")
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, output)
    values = grid_to_cell_order(values_grid, grid)
    coverage = grid_to_cell_order(coverage_grid, grid)
    var_names = ("PCT_SAND", "PCT_CLAY", "ORGANIC")

    # Enforce a physically valid mineral texture triangle after spatial averaging.
    texture_sum = values[0] + values[1]
    over = np.isfinite(texture_sum) & (texture_sum > 100.0)
    scale = np.ones_like(texture_sum, dtype=np.float32)
    scale[over] = 100.0 / texture_sum[over]
    values[0] *= scale
    values[1] *= scale

    now = datetime.now(timezone.utc).isoformat()
    with Dataset(output, "r+") as ds:
        old_values = np.stack(
            [np.asarray(ds.variables[name][:], dtype=np.float64) for name in var_names]
        )
        new_values = values.astype(np.float64)
        valid_masks = np.isfinite(new_values)
        # Sand and clay describe one mineral texture and must be updated as a
        # pair.  Mixing one SSURGO property with the legacy counterpart can
        # put an otherwise valid texture outside the sand-clay triangle.
        texture_valid = valid_masks[0] & valid_masks[1]
        valid_masks[0] = texture_valid
        valid_masks[1] = texture_valid
        valid_masks[0] &= (new_values[0] >= 0.0) & (new_values[0] <= 100.0)
        valid_masks[1] &= (new_values[1] >= 0.0) & (new_values[1] <= 100.0)
        valid_masks[2] &= (new_values[2] >= 0.0) & (new_values[2] <= 1300.0)

        merged = old_values.copy()
        for prop in range(3):
            merged[prop, valid_masks[prop]] = new_values[prop, valid_masks[prop]]

        # Guard against roundoff at the edge of the mineral texture triangle.
        merged_texture_sum = merged[0] + merged[1]
        merged_over = np.isfinite(merged_texture_sum) & (merged_texture_sum > 100.0)
        merged_scale = np.ones_like(merged_texture_sum)
        merged_scale[merged_over] = 100.0 / merged_texture_sum[merged_over]
        merged[0] *= merged_scale
        merged[1] *= merged_scale

        for prop, name in enumerate(var_names):
            variable = ds.variables[name]
            valid = valid_masks[prop]
            variable[:] = merged[prop]
            variable.setncattr(
                "source",
                "USDA-NRCS FY2026 gSSURGO state databases downloaded 2026-08-14",
            )
            variable.setncattr(
                "ssurgo_processing",
                "component- and horizon-thickness-weighted; deepest horizon extended to 3.8018 m; "
                f"map-unit polygons averaged using {factor}x{factor} subcell samples",
            )
            variable.setncattr("ssurgo_valid_fraction", float(valid.mean()))
            variable.setncattr(
                "ssurgo_mean_spatial_coverage",
                float(np.nanmean(coverage[prop])),
            )

        ds.setncattr("SSURGO_source", "USDA-NRCS gSSURGO by State, FY2026 official packages")
        ds.setncattr("SSURGO_states", ",".join(states))
        ds.setncattr("SSURGO_download_date", "2026-08-14")
        ds.setncattr("SSURGO_subgrid_factor", factor)
        history = str(getattr(ds, "History_Log", "")).rstrip()
        entry = (
            f"{now}: PCT_SAND, PCT_CLAY, and ORGANIC updated from USDA-NRCS gSSURGO "
            "by scripts/build_ssurgo_surface.py"
        )
        ds.setncattr("History_Log", f"{history}\n{entry}" if history else entry)

    print(f"Wrote {output}", flush=True)
    for prop, name in enumerate(var_names):
        valid = valid_masks[prop]
        applied = values[prop][valid]
        print(
            f"  {name}: SSURGO values for {valid.mean() * 100.0:.2f}% of layer-cells; "
            f"range={np.nanmin(applied):.4g}..{np.nanmax(applied):.4g}",
            flush=True,
        )


def main() -> None:
    args = parse_args()
    states = [state.upper() for state in args.states]
    unknown = sorted(set(states) - set(ALL_STATES))
    if unknown:
        raise SystemExit(f"States outside the configured surface-domain set: {unknown}")
    if args.subgrid_factor < 1:
        raise SystemExit("--subgrid-factor must be positive")
    args.work_dir.mkdir(parents=True, exist_ok=True)
    profile_dir = args.work_dir / "profiles"
    profile_dir.mkdir(parents=True, exist_ok=True)

    profile_paths = [
        build_state_profiles(state, args.raw_dir, profile_dir) for state in states
    ]
    if args.profiles_only:
        return
    grid = target_grid(args.surfdata)
    raster_path = rasterize_mapunits(
        states,
        args.raw_dir,
        args.work_dir,
        grid,
        args.subgrid_factor,
        args.polygon_chunk_size,
    )
    mukey, profile = combine_profiles(profile_paths)
    values, coverage = aggregate_to_destination(
        raster_path, mukey, profile, grid, args.subgrid_factor
    )
    write_surface_copy(
        args.surfdata,
        args.output,
        values,
        coverage,
        grid,
        states,
        args.subgrid_factor,
        args.overwrite_output,
    )


if __name__ == "__main__":
    main()
