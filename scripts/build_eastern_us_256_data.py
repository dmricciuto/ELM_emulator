#!/usr/bin/env python3
"""Build matching 256-cell eastern-US ELM surface and domain files.

The output surface file uses the same one-dimensional ``gridcell`` layout as
``data/surface_data/surfdata_sg256.nc``.  Cells are sampled reproducibly from
the land points in the requested latitude/longitude box.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import xarray as xr


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
WORKSPACE_ROOT = REPO_ROOT.parent
DEFAULT_SURFACE = REPO_ROOT / "data/surface_data/surfdata_hcru_TRENDY.nc4"
DEFAULT_DOMAIN = (
    WORKSPACE_ROOT
    / "e3sm_inputdata/share/domains/domain.clm/domain.lnd.360x720_cruncep.100429.nc"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data/surface_data"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--surface-source", type=Path, default=DEFAULT_SURFACE)
    parser.add_argument("--domain-source", type=Path, default=DEFAULT_DOMAIN)
    parser.add_argument(
        "--surface-output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "surfdata_eastern_us_256.nc",
    )
    parser.add_argument(
        "--domain-output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "domain_eastern_us_256.nc",
    )
    parser.add_argument(
        "--point-list-output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "eastern_us_256_cells.csv",
    )
    parser.add_argument("--count", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lat-min", type=float, default=24.0)
    parser.add_argument("--lat-max", type=float, default=49.5)
    parser.add_argument("--lon-min", type=float, default=-100.0)
    parser.add_argument("--lon-max", type=float, default=-66.5)
    return parser.parse_args()


def longitude_180(values: np.ndarray) -> np.ndarray:
    return (np.asarray(values, dtype=np.float64) + 180.0) % 360.0 - 180.0


def select_cells(
    domain: xr.Dataset,
    surface: xr.Dataset,
    *,
    count: int,
    seed: int,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
) -> tuple[np.ndarray, np.ndarray]:
    if domain.sizes.get("nj") != surface.sizes.get("lsmlat") or domain.sizes.get(
        "ni"
    ) != surface.sizes.get("lsmlon"):
        raise ValueError("Surface and domain source grids do not have matching dimensions")

    lat = np.asarray(domain["yc"].values, dtype=np.float64)
    lon = longitude_180(domain["xc"].values)
    surface_lat = np.asarray(surface["LATIXY"].values, dtype=np.float64)
    surface_lon = longitude_180(surface["LONGXY"].values)
    if not np.allclose(lat, surface_lat) or not np.allclose(lon, surface_lon):
        raise ValueError("Surface and domain latitude/longitude grids are not aligned")

    eligible = (
        (np.asarray(domain["mask"].values) > 0)
        & (lat >= lat_min)
        & (lat <= lat_max)
        & (lon >= lon_min)
        & (lon <= lon_max)
    )
    rows, cols = np.nonzero(eligible)
    if rows.size < count:
        raise ValueError(f"Only {rows.size} eligible land cells found; requested {count}")

    rng = np.random.default_rng(seed)
    chosen = np.sort(rng.choice(rows.size, size=count, replace=False))
    return rows[chosen], cols[chosen]


def subset_surface(source: xr.Dataset, rows: np.ndarray, cols: np.ndarray) -> xr.Dataset:
    # Read one compact rectangular slab before point selection. Direct netCDF4
    # vector indexing performs thousands of small reads and is much slower.
    row_start = int(rows.min())
    col_start = int(cols.min())
    row_index = xr.DataArray(rows - row_start, dims="gridcell")
    col_index = xr.DataArray(cols - col_start, dims="gridcell")
    variables: dict[str, xr.DataArray] = {}
    for name, variable in source.data_vars.items():
        if "lsmlat" in variable.dims and "lsmlon" in variable.dims:
            slab = variable.isel(
                lsmlat=slice(row_start, int(rows.max()) + 1),
                lsmlon=slice(col_start, int(cols.max()) + 1),
            ).load()
            variables[name] = slab.isel(lsmlat=row_index, lsmlon=col_index)
        else:
            variables[name] = variable
    # OLMT's sg256-style extraction writes data variables but not dimension
    # coordinate variables or global metadata.
    return xr.Dataset(variables)


def subset_domain(source: xr.Dataset, rows: np.ndarray, cols: np.ndarray) -> xr.Dataset:
    row_index = xr.DataArray(rows, dims="cell")
    col_index = xr.DataArray(cols, dims="cell")
    variables: dict[str, xr.DataArray] = {}
    for name, variable in source.data_vars.items():
        if "nj" in variable.dims and "ni" in variable.dims:
            selected = variable.isel(nj=row_index, ni=col_index).rename(cell="ni")
            variables[name] = selected.expand_dims(nj=[0]).transpose("nj", "ni", ...)
        else:
            variables[name] = variable
    return xr.Dataset(variables)


def write_point_list(
    path: Path, domain: xr.Dataset, rows: np.ndarray, cols: np.ndarray
) -> None:
    lat = np.asarray(domain["yc"].values)[rows, cols]
    lon = longitude_180(np.asarray(domain["xc"].values)[rows, cols])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["cell_index", "source_row", "source_col", "lat", "lon"])
        for cell, (row, col, cell_lat, cell_lon) in enumerate(
            zip(rows, cols, lat, lon, strict=True)
        ):
            writer.writerow(
                [cell, int(row), int(col), f"{cell_lat:.8f}", f"{cell_lon:.8f}"]
            )


def main() -> None:
    args = parse_args()
    if args.count <= 0:
        raise ValueError("--count must be positive")

    with xr.open_dataset(args.surface_source, decode_times=False) as surface, xr.open_dataset(
        args.domain_source, decode_times=False
    ) as domain:
        rows, cols = select_cells(
            domain,
            surface,
            count=args.count,
            seed=args.seed,
            lat_min=args.lat_min,
            lat_max=args.lat_max,
            lon_min=args.lon_min,
            lon_max=args.lon_max,
        )
        surface_subset = subset_surface(surface, rows, cols).load()
        domain_subset = subset_domain(domain, rows, cols).load()
        write_point_list(args.point_list_output, domain, rows, cols)

    args.surface_output.parent.mkdir(parents=True, exist_ok=True)
    args.domain_output.parent.mkdir(parents=True, exist_ok=True)
    surface_subset.to_netcdf(args.surface_output, format="NETCDF4_CLASSIC")
    domain_subset.to_netcdf(args.domain_output, format="NETCDF4_CLASSIC")

    lat = np.asarray(domain_subset["yc"].values).reshape(-1)
    lon = longitude_180(np.asarray(domain_subset["xc"].values).reshape(-1))
    print(f"Wrote {args.surface_output} ({surface_subset.sizes['gridcell']} cells)")
    print(f"Wrote {args.domain_output} ({domain_subset.sizes['ni']} cells)")
    print(f"Wrote {args.point_list_output}")
    print(
        f"Selected bounds: lat {lat.min():.2f}..{lat.max():.2f}, "
        f"lon {lon.min():.2f}..{lon.max():.2f}"
    )


if __name__ == "__main__":
    main()
