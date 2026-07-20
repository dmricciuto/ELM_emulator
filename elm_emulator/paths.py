"""Repository-local default paths used by the CLI wrappers."""

from __future__ import annotations

from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent

DEFAULT_MODEL_DIR = REPO_ROOT / "artifacts" / "canopyflux_models"
DEFAULT_TESSFA_DOMAIN2_SURFDATA = (
    REPO_ROOT / "data" / "surface_data" / "surfdata.TESSFA_DOMAIN2.4km.1d.NLCD.c240827.nc4"
)
DEFAULT_SG256_SURFDATA = REPO_ROOT / "data" / "surface_data" / "surfdata_sg256.nc"
DEFAULT_CLM_PARAMS_SP = REPO_ROOT / "data" / "params" / "clm_params_SP.nc"
DEFAULT_CLM_PARAMS_C180524 = REPO_ROOT / "data" / "params" / "clm_params_c180524.nc"
DEFAULT_USMOZ_SURFDATA = REPO_ROOT / "data" / "site" / "usmoz" / "surfdata.nc"
DEFAULT_USMOZ_CLM_PARAMS = REPO_ROOT / "data" / "site" / "usmoz" / "clm_params.nc"
