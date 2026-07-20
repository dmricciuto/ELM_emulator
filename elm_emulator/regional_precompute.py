"""CPU-side solar geometry and vegetation precompute helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .regional_time import _day_of_year_start, _days_in_month, _days_in_year


def compute_coszen_daylength(
    lat_rad: np.ndarray,   # (ncell,)
    lon_deg: np.ndarray,   # (ncell,)
    doy: int,
    utc_hour: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Cosine of solar zenith angle and daylength (seconds) per cell.

    Uses the Spencer (1971) declination formula, consistent with ELM.
    """
    B = 2.0 * np.pi * (doy - 1) / 365.0
    decl = (
        0.006918 - 0.399912 * np.cos(B) + 0.070257 * np.sin(B)
        - 0.006758 * np.cos(2.0 * B) + 0.000907 * np.sin(2.0 * B)
        - 0.002697 * np.cos(3.0 * B) + 0.001480 * np.sin(3.0 * B)
    )
    # Local apparent solar time → hour angle (0 = solar noon).
    local_solar_hour = utc_hour + lon_deg / 15.0
    ha = 2.0 * np.pi * (local_solar_hour / 24.0 - 0.5)
    coszen = (
        np.sin(lat_rad) * np.sin(decl)
        + np.cos(lat_rad) * np.cos(decl) * np.cos(ha)
    )
    # Daylength: solar hour angle at sunrise/sunset.
    ha_sr = np.arccos(np.clip(-np.tan(lat_rad) * np.tan(decl), -1.0, 1.0))
    dayl = 2.0 * ha_sr / (2.0 * np.pi) * 86400.0  # seconds
    return coszen, dayl


def compute_max_daylength(lat_rad: np.ndarray) -> np.ndarray:
    """Annual maximum daylength (seconds), matching native ELM initialization.

    ELM uses a hardwired present-day maximum declination of +/-23.4667 degrees,
    with the negative value in the southern hemisphere.
    """
    lat_rad = np.asarray(lat_rad, dtype=np.float64)
    max_decl = np.where(lat_rad < 0.0, -0.409571, 0.409571)
    ha_sr = np.arccos(np.clip(-np.tan(lat_rad) * np.tan(max_decl), -1.0, 1.0))
    return 2.0 * ha_sr / (2.0 * np.pi) * 86400.0


def interp_monthly_lai_to_step(
    monthly_lai: np.ndarray,  # (12, npft_active, ncell)
    doy: int,
) -> np.ndarray:
    """Linearly interpolate monthly LAI to a specific day-of-year.

    Monthly values are treated as valid at the 15th of each month.
    Returns ``(npft_active, ncell)`` float32.
    """
    # Day of year for the 15th of each month (non-leap year).
    month_mid_doy = np.array(
        [15, 46, 74, 105, 135, 166, 196, 227, 258, 288, 319, 349], dtype=np.float32
    )
    # Wrap DOY around so we can interpolate across year boundaries.
    doy_f = float(doy)
    # Find surrounding months.
    if doy_f <= month_mid_doy[0]:
        m0, m1 = 11, 0
        d0, d1 = month_mid_doy[11] - 365.0, month_mid_doy[0]
    elif doy_f > month_mid_doy[-1]:
        m0, m1 = 11, 0
        d0, d1 = month_mid_doy[11], month_mid_doy[0] + 365.0
    else:
        m1 = int(np.searchsorted(month_mid_doy, doy_f))
        m0 = m1 - 1
        d0, d1 = month_mid_doy[m0], month_mid_doy[m1]
    t = float(np.clip((doy_f - d0) / (d1 - d0), 0.0, 1.0))
    return (
        (1.0 - t) * monthly_lai[m0].astype(np.float32)
        + t         * monthly_lai[m1].astype(np.float32)
    )


def _precompute_month_cpu(
    surf: dict,
    year: int,
    month: int,
    *,
    calendar: str = "gregorian",
    n1h: int | None = None,
) -> dict:
    """CPU-only precomputation for one month (run in prefetch thread).

    Returns a dict with keys:
      ``coszen_all``  – (n1h, ncell) float32 numpy array
      ``dayl_all``    – (n1h, ncell) float32 numpy array
      ``prev_dayl_all`` – (n1h, ncell) float32 numpy array
      ``max_dayl_np`` – (ncell,)     float32 numpy array
      ``lai_by_doy``  – dict[int, (17, ncell) float32 numpy array]
      ``sai_by_doy``  – dict[int, (17, ncell) float32 numpy array]
      ``htop_by_doy`` – dict[int, (17, ncell) float32 numpy array]
      ``hbot_by_doy`` – dict[int, (17, ncell) float32 numpy array]
      ``n1h``         – int
    """
    lat_rad = np.asarray(surf["lat_deg"], dtype=np.float64) * (np.pi / 180.0)
    lon_deg = np.asarray(surf["lon_deg"], dtype=np.float64)
    ncell   = lat_rad.shape[0]

    days_in_month = _days_in_month(year, month, calendar)
    if n1h is None:
        n1h = days_in_month * 24
    else:
        n1h = int(n1h)
        if n1h <= 0:
            raise ValueError("n1h override for _precompute_month_cpu must be positive")
        if n1h % 24 != 0:
            raise ValueError(f"n1h override must be a whole number of days, got {n1h}")
    day_of_year_start = _day_of_year_start(year, month, calendar)

    coszen_all = np.empty((n1h, ncell), dtype=np.float32)
    dayl_all   = np.empty((n1h, ncell), dtype=np.float32)
    prev_dayl_all = np.empty((n1h, ncell), dtype=np.float32)
    days_in_prev_year = _days_in_year(year - 1, calendar)
    for s in range(n1h):
        doy_s = day_of_year_start + s // 24
        prev_doy_s = doy_s - 1 if doy_s > 1 else days_in_prev_year
        utc_s = (s % 24) + 0.5
        cz, dl = compute_coszen_daylength(lat_rad, lon_deg, doy_s, utc_s)
        _, prev_dl = compute_coszen_daylength(lat_rad, lon_deg, prev_doy_s, utc_s)
        coszen_all[s] = cz
        dayl_all[s]   = dl
        prev_dayl_all[s] = prev_dl
    max_dayl_np = compute_max_daylength(lat_rad).astype(np.float32)

    unique_doys = sorted(set(day_of_year_start + s // 24 for s in range(n1h)))
    lai_by_doy: dict[int, np.ndarray] = {}
    sai_by_doy: dict[int, np.ndarray] = {}
    htop_by_doy: dict[int, np.ndarray] = {}
    hbot_by_doy: dict[int, np.ndarray] = {}
    for d in unique_doys:
        lai_by_doy[d] = interp_monthly_lai_to_step(surf["monthly_lai"], d).astype(np.float32)
        sai_by_doy[d] = interp_monthly_lai_to_step(surf["monthly_sai"], d).astype(np.float32)
        htop_by_doy[d] = interp_monthly_lai_to_step(surf["monthly_htop"], d).astype(np.float32)
        hbot_by_doy[d] = interp_monthly_lai_to_step(surf["monthly_hbot"], d).astype(np.float32)

    return {
        "coszen_all":  coszen_all,
        "dayl_all":    dayl_all,
        "prev_dayl_all": prev_dayl_all,
        "max_dayl_np": max_dayl_np,
        "lai_by_doy":  lai_by_doy,
        "sai_by_doy":  sai_by_doy,
        "htop_by_doy": htop_by_doy,
        "hbot_by_doy": hbot_by_doy,
        "n1h":         n1h,
    }


def _hash_precompute_input_array(hasher: "hashlib._Hash", name: str, value: np.ndarray) -> None:
    arr = np.ascontiguousarray(np.asarray(value))
    hasher.update(name.encode("utf-8"))
    hasher.update(str(arr.shape).encode("utf-8"))
    hasher.update(str(arr.dtype).encode("utf-8"))
    hasher.update(arr.view(np.uint8))


def _precompute_month_cache_path(
    cache_dir: Path,
    surf: dict,
    year: int,
    month: int,
    *,
    calendar: str,
    n1h: int,
) -> Path:
    hasher = hashlib.sha256()
    metadata = {
        "version": 1,
        "year": int(year),
        "month": int(month),
        "calendar": str(calendar),
        "n1h": int(n1h),
    }
    hasher.update(json.dumps(metadata, sort_keys=True).encode("utf-8"))
    for key in (
        "active_idx",
        "lat_deg",
        "lon_deg",
        "monthly_lai",
        "monthly_sai",
        "monthly_htop",
        "monthly_hbot",
    ):
        _hash_precompute_input_array(hasher, key, surf[key])
    digest = hasher.hexdigest()[:24]
    return Path(cache_dir) / (
        f"regional_precompute_{int(year):04d}{int(month):02d}_"
        f"{str(calendar)}_h{int(n1h)}_{digest}.npz"
    )


def _load_precompute_month_cache(
    cache_dir: Path,
    surf: dict,
    year: int,
    month: int,
    *,
    calendar: str,
    n1h: int,
) -> dict | None:
    path = _precompute_month_cache_path(
        Path(cache_dir),
        surf,
        year,
        month,
        calendar=calendar,
        n1h=n1h,
    )
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            doys = np.asarray(data["doy"], dtype=np.int32)
            pc = {
                "coszen_all": np.asarray(data["coszen_all"], dtype=np.float32),
                "dayl_all": np.asarray(data["dayl_all"], dtype=np.float32),
                "prev_dayl_all": np.asarray(data["prev_dayl_all"], dtype=np.float32),
                "max_dayl_np": np.asarray(data["max_dayl_np"], dtype=np.float32),
                "n1h": int(np.asarray(data["n1h"]).reshape(-1)[0]),
            }
            for key in ("lai", "sai", "htop", "hbot"):
                stack = np.asarray(data[f"{key}_by_doy"], dtype=np.float32)
                pc[f"{key}_by_doy"] = {
                    int(doy): stack[i]
                    for i, doy in enumerate(doys)
                }
        print(f"  [precompute cache] loaded {path}")
        return pc
    except Exception as exc:
        print(f"  [precompute cache] ignoring unreadable cache {path}: {exc}")
        return None


def _save_precompute_month_cache(
    cache_dir: Path,
    surf: dict,
    year: int,
    month: int,
    *,
    calendar: str,
    n1h: int,
    precomputed: dict,
) -> None:
    path = _precompute_month_cache_path(
        Path(cache_dir),
        surf,
        year,
        month,
        calendar=calendar,
        n1h=n1h,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    doys = np.asarray(sorted(precomputed["lai_by_doy"]), dtype=np.int32)
    tmp_path = path.with_name(path.name + ".tmp.npz")
    np.savez(
        tmp_path,
        n1h=np.asarray([int(precomputed["n1h"])], dtype=np.int32),
        doy=doys,
        coszen_all=np.asarray(precomputed["coszen_all"], dtype=np.float32),
        dayl_all=np.asarray(precomputed["dayl_all"], dtype=np.float32),
        prev_dayl_all=np.asarray(precomputed["prev_dayl_all"], dtype=np.float32),
        max_dayl_np=np.asarray(precomputed["max_dayl_np"], dtype=np.float32),
        lai_by_doy=np.stack([precomputed["lai_by_doy"][int(d)] for d in doys], axis=0).astype(np.float32, copy=False),
        sai_by_doy=np.stack([precomputed["sai_by_doy"][int(d)] for d in doys], axis=0).astype(np.float32, copy=False),
        htop_by_doy=np.stack([precomputed["htop_by_doy"][int(d)] for d in doys], axis=0).astype(np.float32, copy=False),
        hbot_by_doy=np.stack([precomputed["hbot_by_doy"][int(d)] for d in doys], axis=0).astype(np.float32, copy=False),
    )
    tmp_path.replace(path)
    print(f"  [precompute cache] saved {path}")


def _load_or_precompute_month_cpu(
    surf: dict,
    year: int,
    month: int,
    *,
    calendar: str = "gregorian",
    n1h: int | None = None,
    cache_dir: Path | None = None,
) -> dict:
    if n1h is None:
        n1h = _days_in_month(year, month, calendar) * 24
    n1h = int(n1h)
    if cache_dir is not None:
        cached = _load_precompute_month_cache(
            Path(cache_dir),
            surf,
            year,
            month,
            calendar=calendar,
            n1h=n1h,
        )
        if cached is not None:
            return cached
    pc = _precompute_month_cpu(surf, year, month, calendar=calendar, n1h=n1h)
    if cache_dir is not None:
        _save_precompute_month_cache(
            Path(cache_dir),
            surf,
            year,
            month,
            calendar=calendar,
            n1h=n1h,
            precomputed=pc,
        )
    return pc
