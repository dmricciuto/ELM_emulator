#!/usr/bin/env python3
"""Train one canopy-flux emulator model per patch index with PyTorch."""

from __future__ import annotations

import argparse
import json
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable

import numpy as np
from netCDF4 import Dataset
from sklearn.metrics import r2_score

from .btran_utils import compute_btran_and_root_fraction
from .coupled_surface_hydrology_soiltemp import (
    organic_fraction_from_surfdata,
    vapor_pressure_deficit_from_forcing,
)
from .elm_hydrology import SoilTextureProfile, elm_soil_grid, hydraulic_params_from_texture
from .paths import DEFAULT_MODEL_DIR

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise SystemExit(
        "PyTorch is required for this utility. Install torch first, then rerun."
    ) from exc


# These are not part of the learned canopy-flux contract. CH4 diagnostics are
# only present when CH4 is active, while canopy/snow routing is handled by the
# ported surface-water physics in the coupled replay.
SKIP_TARGET_NAMES = {
    "grnd_ch4_cond_patch",
    "canopy_cond_patch",
    "h2ocan",
    "qflx_prec_grnd",
    "qflx_rain_grnd",
    "qflx_snow_grnd",
    "qflx_leafdrip",
    "qflx_snwcp_ice",
    "qflx_snwcp_liq",
}

DROP_INPUT_PREFIXES = ("smp_l_lev", "rootfr_lev", "t_soisno_lev", "h2osoi_liqvol_lev", "eff_porosity_lev")
KEEP_DROPPED_PREFIX_INPUT_NAMES = frozenset({
    "smp_l_lev1",
    "h2osoi_liqvol_lev1",
    "t_soisno_lev1",
    "soilbeta_col",
})
DEFAULT_DROP_INPUT_NAMES = (
    "forc_hgt_u_patch",
    "z0mg_col",
    "ugust",
    "forc_solad_vis",
    "forc_solad_nir",
    "forc_solai_vis",
    "forc_solai_nir",
    "pft_type",
)
OPTIONAL_PARAMETER_INPUT_NAMES = ("pft_type",)
FLOAT32_CONSTANT_FEATURE_RTOL = 10.0 * np.finfo(np.float32).eps
FLOAT32_CONSTANT_FEATURE_ATOL = 1.0e-12
PHOTOSYNTHESIS_TARGET_NAMES = ("psnsun_patch", "psnsha_patch", "fpsn_patch")
PATCH0_OMITTED_TARGET_NAMES = {
    "fpsn_patch",
    "psnsun_patch",
    "psnsha_patch",
    "laisun_patch",
    "laisha_patch",
    "qflx_tran_veg",
    "qflx_evap_canopy",
}
PHYSIOLOGY_PARAMETER_FEATURE_NAMES = ("flnr", "mbbopt")
PHYSIOLOGY_PARAMETER_TARGET_NAMES = (
    "fpsn_patch",
    "psnsun_patch",
    "psnsha_patch",
    "qflx_tran_veg",
)
NONNEGATIVE_TARGET_NAMES = {
    "cgrndl",
    "gh_canopy",
    "h2ocan",
    "fpsn_patch",
    "laisun_patch",
    "laisha_patch",
    "rssun_patch",
    "rssha_patch",
    "psnsun_patch",
    "psnsha_patch",
}
DEFAULT_LOG1P_TARGET_NAMES = ("gh_canopy", "psnsun_patch")
RECOMMENDED_LOG1P_TARGET_NAMES = ("gh_canopy", "psnsun_patch", "psnsha_patch")
ZERO_RADIATION_THRESHOLD = 1.0e-12
ZERO_LAI_THRESHOLD = 1.0e-2
ZERO_BTRAN_THRESHOLD = 1.0e-2
UNIVERSAL_PATCH_INDEX = -1


@dataclass
class PatchDataset:
    patch_index: int
    features: np.ndarray
    targets: np.ndarray
    target_valid_mask: np.ndarray
    years: np.ndarray
    months: np.ndarray
    days: np.ndarray
    secs: np.ndarray
    nsteps: np.ndarray
    pft_indices: np.ndarray | None = None
    gridcell_indices: np.ndarray | None = None
    dropped_nonfinite: int = 0
    feature_nonfinite_counts: np.ndarray | None = None
    target_nonfinite_counts: np.ndarray | None = None
    feature_large_counts: np.ndarray | None = None
    target_large_counts: np.ndarray | None = None


class PatchMLP(nn.Module):
    """Simple fully connected multi-output regressor."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_layers: tuple[int, ...],
        *,
        pft_embedding_dim: int = 0,
        pft_input_index: int | None = None,
        pft_embedding_categories: Iterable[int] | None = None,
        pft_x_mean: float = 0.0,
        pft_x_std: float = 1.0,
    ) -> None:
        super().__init__()
        categories = tuple(sorted({int(value) for value in (pft_embedding_categories or ())}))
        use_pft_embedding = pft_embedding_dim > 0 and pft_input_index is not None and len(categories) > 0
        self.pft_input_index = int(pft_input_index) if use_pft_embedding else None
        self.pft_embedding_dim = int(pft_embedding_dim) if use_pft_embedding else 0
        if use_pft_embedding:
            self.pft_unknown_embedding_index = len(categories)
            self.pft_embedding = nn.Embedding(len(categories) + 1, self.pft_embedding_dim)
            max_category = max(max(categories), 0)
            lookup = torch.full(
                (max_category + 1,),
                self.pft_unknown_embedding_index,
                dtype=torch.long,
            )
            for embedding_index, category in enumerate(categories):
                if category >= 0:
                    lookup[category] = embedding_index
            self.register_buffer("pft_embedding_lookup", lookup, persistent=True)
            self.register_buffer("pft_x_mean", torch.tensor(float(pft_x_mean)), persistent=True)
            self.register_buffer("pft_x_std", torch.tensor(float(pft_x_std) if pft_x_std != 0.0 else 1.0), persistent=True)
        else:
            self.pft_unknown_embedding_index = None
            self.pft_embedding = None

        layers: list[nn.Module] = []
        prev_dim = input_dim + self.pft_embedding_dim
        for width in hidden_layers:
            layers.append(nn.Linear(prev_dim, width))
            layers.append(nn.GELU())
            prev_dim = width
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def _pft_embedding_indices(self, x: torch.Tensor) -> torch.Tensor:
        assert self.pft_input_index is not None
        assert self.pft_embedding is not None
        assert self.pft_unknown_embedding_index is not None
        raw_pft = torch.round(x[..., self.pft_input_index] * self.pft_x_std + self.pft_x_mean).to(torch.long)
        unknown = torch.full_like(raw_pft, self.pft_unknown_embedding_index)
        valid = (raw_pft >= 0) & (raw_pft < self.pft_embedding_lookup.numel())
        safe_raw = torch.where(valid, raw_pft, torch.zeros_like(raw_pft))
        return torch.where(valid, self.pft_embedding_lookup[safe_raw], unknown)

    def reset_unknown_pft_embedding_to_mean(self) -> None:
        if self.pft_embedding is None or self.pft_unknown_embedding_index is None:
            return
        if self.pft_unknown_embedding_index <= 0:
            return
        with torch.no_grad():
            learned = self.pft_embedding.weight[: self.pft_unknown_embedding_index]
            self.pft_embedding.weight[self.pft_unknown_embedding_index].copy_(learned.mean(dim=0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pft_embedding is not None:
            pft_embedding = self.pft_embedding(self._pft_embedding_indices(x))
            x_continuous = x.clone()
            assert self.pft_input_index is not None
            x_continuous[..., self.pft_input_index] = 0.0
            x = torch.cat((x_continuous, pft_embedding), dim=-1)
        return self.network(x)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train one canopy-flux emulator model per patch index.",
    )
    parser.add_argument(
        "training_file",
        type=Path,
        help="Path to CASEID.canopyflux_training.nc",
    )
    parser.add_argument(
        "--surfdata-file",
        type=Path,
        default=None,
        help="Optional surfdata.nc used to derive row-specific btran from the stored training-state columns.",
    )
    parser.add_argument(
        "--surfdata-cell-stride",
        type=int,
        default=None,
        help=(
            "Number of 1-based cell_index slots per surfdata gridcell when cell_index stores "
            "a regular PFT/patch slot id instead of a compact gridcell id. Prefer storing an "
            "actual gridcell_index in the training file; this option is only an explicit fallback."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help="Directory where per-patch models and metrics are written.",
    )
    parser.add_argument(
        "--drop-input-names",
        nargs="*",
        default=list(DEFAULT_DROP_INPUT_NAMES),
        help=(
            "Exact input feature names to remove before training. Defaults to "
            "the features unavailable in the regional replay plus optional "
            "diagnostic features. Pass the option "
            "with no names to disable exact-name dropping."
        ),
    )
    parser.add_argument(
        "--include-pft-type-feature",
        action="store_true",
        help=(
            "Opt in to using PFT identity as an input feature. For legacy "
            "training files, pft_type is appended from the parameter matrix."
        ),
    )
    parser.add_argument(
        "--pft-embedding-dim",
        type=int,
        default=0,
        help=(
            "Use a learned categorical embedding for pft_type with this vector width. "
            "When positive, pft_type is retained as categorical metadata and is not "
            "used as an ordinal continuous predictor."
        ),
    )
    parser.add_argument(
        "--patch-index",
        type=int,
        nargs="*",
        default=None,
        help="Optional subset of patch indices to train.",
    )
    parser.add_argument(
        "--universal-model",
        action="store_true",
        help=(
            "Train one shared model over all selected patch/PFT rows instead of "
            "one model per patch. The shared artifact is written as patch_-1."
        ),
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=512,
        help="Skip patches with fewer than this many filtered samples.",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.2,
        help="Fraction of each non-holdout patch sample pool used for random validation.",
    )
    parser.add_argument(
        "--hidden-layers",
        type=int,
        nargs="+",
        default=(256, 256, 128),
        help="Hidden layer widths for the MLP regressor.",
    )
    parser.add_argument(
        "--max-iter",
        dest="max_epochs",
        type=int,
        default=200,
        help="Maximum training epochs per patch model.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4096,
        help="Mini-batch size used during training. Larger values, e.g. 65536, can better saturate CUDA.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1.0e-3,
        help="Adam learning rate.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1.0e-5,
        help="Adam weight decay.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Early-stopping patience in epochs.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "mps", "cuda", "cpu"),
        default="auto",
        help="Torch device selection.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducible training.",
    )
    parser.add_argument(
        "--spval-threshold",
        type=float,
        default=1.0e30,
        help="Drop rows containing feature or target magnitudes larger than this threshold.",
    )
    parser.add_argument(
        "--plot-targets",
        nargs="+",
        default=None,
        help="Validation targets to plot. Defaults to all targets in the current training contract.",
    )
    parser.add_argument(
        "--plot-gridcell-index",
        nargs="+",
        type=int,
        default=[1, 100, 200],
        help=(
            "Global gridcell indices to plot as separate full time series figures. "
            "Each plot includes train, validation, and holdout predictions when available."
        ),
    )
    parser.add_argument(
        "--holdout-gridcell-index",
        nargs="+",
        type=int,
        default=[1, 100, 200],
        help=(
            "Global gridcell indices to exclude from training and normal validation. "
            "These are reported separately as spatial holdout diagnostics."
        ),
    )
    parser.add_argument(
        "--qflx-tran-weight",
        type=float,
        default=2.0,
        help="Extra sample-weight multiplier driven by qflx_tran_veg magnitude.",
    )
    parser.add_argument(
        "--psnsun-weight",
        type=float,
        default=2.0,
        help="Extra sample-weight multiplier driven by psnsun_patch magnitude.",
    )
    parser.add_argument(
        "--qflx-prec-weight",
        type=float,
        default=3.0,
        help=(
            "Legacy extra sample-weight multiplier driven by qflx_prec_grnd magnitude. "
            "No effect when qflx_prec_grnd is excluded from the learned target contract."
        ),
    )
    parser.add_argument(
        "--fpsn-weight",
        type=float,
        default=1.0,
        help="Extra sample-weight multiplier driven by fpsn_patch magnitude.",
    )
    parser.add_argument(
        "--psn-low-value-weight",
        type=float,
        default=4.0,
        help=(
            "Extra per-target loss multiplier for valid photosynthesis samples "
            "where the target value is near zero."
        ),
    )
    parser.add_argument(
        "--psn-low-value-threshold",
        type=float,
        default=0.25,
        help="Photosynthesis target value at or below which --psn-low-value-weight is applied.",
    )
    parser.add_argument(
        "--hs-soil-weight",
        type=float,
        default=1.5,
        help="Extra sample-weight multiplier driven by |hs_soil| magnitude.",
    )
    parser.add_argument(
        "--hs-h2osfc-weight",
        type=float,
        default=1.5,
        help="Extra sample-weight multiplier driven by |hs_h2osfc| magnitude.",
    )
    parser.add_argument(
        "--dhsdT-weight",
        type=float,
        default=1.0,
        help="Extra sample-weight multiplier driven by |dhsdT| magnitude.",
    )
    parser.add_argument(
        "--no-nighttime-psn-mask",
        dest="nighttime_psn_mask",
        action="store_false",
        default=True,
        help=(
            "Disable nighttime masking of photosynthesis targets (psnsun_patch, "
            "psnsha_patch, fpsn_patch).  By default, samples where forc_swrad <= 0 "
            "are excluded from the loss for those targets so the model is forced to "
            "learn the daytime photosynthesis signal rather than fitting near-zero."
        ),
    )
    parser.add_argument(
        "--log1p-targets",
        nargs="+",
        default=list(RECOMMENDED_LOG1P_TARGET_NAMES),
        help=(
            "Nonnegative targets trained in log1p space. The default includes "
            "psnsha_patch so shaded photosynthesis gets better near-zero resolution."
        ),
    )
    parser.add_argument(
        "--importance-targets",
        nargs="+",
        default=("psnsun_patch", "psnsha_patch"),
        help="Targets for which permutation feature importance is estimated on validation data.",
    )
    parser.add_argument(
        "--sensitivity-targets",
        nargs="+",
        default=(
            "fpsn_patch",
            "psnsun_patch",
            "psnsha_patch",
            "qflx_tran_veg",
            "qflx_ev_soil",
            "eflx_sh_tot",
        ),
        help=(
            "Targets for which a validation-set feature sensitivity analysis is written. "
            "Uses permutation-RMSE over the validation set."
        ),
    )
    parser.add_argument(
        "--sensitivity-max-samples",
        type=int,
        default=4096,
        help="Maximum number of validation samples used for the sensitivity analysis.",
    )
    parser.add_argument(
        "--sensitivity-top-k",
        type=int,
        default=15,
        help="Number of top features to include in each sensitivity bar plot.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of patches to train in parallel using threads. CUDA-safe; each thread shares the same device.",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=0,
        help="Write per-target val MSE to training_log.jsonl every this many epochs. 0 disables.",
    )
    parser.add_argument(
        "--importance-max-samples",
        type=int,
        default=4096,
        help="Maximum number of validation samples used for permutation importance.",
    )
    parser.add_argument(
        "--target-feature-masks",
        nargs="*",
        default=None,
        metavar="TARGET:FEAT1,FEAT2,...",
        help=(
            "Zero out specific input features when computing the loss for specific targets. "
            "Format: TARGET_NAME:FEAT_NAME1,FEAT_NAME2  (repeat for multiple targets). "
            "Example: --target-feature-masks qflx_tran_veg:h2ocan,smp_l_lev1  "
            "At inference the same mask is applied via the bundle."
        ),
    )
    parser.add_argument(
        "--mask-sabg-snow-for-photosynthesis",
        action="store_true",
        help=(
            "Convenience mask that prevents fpsn_patch, psnsun_patch, and psnsha_patch "
            "from using sabg_snow. The feature remains available to other targets."
        ),
    )
    parser.add_argument(
        "--mask-physiology-params-for-non-physiology-targets",
        action="store_true",
        help=(
            "Convenience mask that prevents non-physiology targets from using flnr/mbbopt. "
            "The parameters remain available to photosynthesis and transpiration targets."
        ),
    )
    parser.add_argument(
        "--state-target-lag-steps",
        type=int,
        default=1,
        help=(
            "Shift selected state-like targets forward by this many timesteps relative to the input features. "
            "Use 0 to disable lagging."
        ),
    )
    parser.add_argument(
        "--state-target-names",
        nargs="+",
        default=["t_veg", "t_ref2m"],
        help=(
            "Targets to lag relative to the input features. "
            "These are treated as prognostic state-like outputs."
        ),
    )
    return parser.parse_args()


def parse_target_feature_masks(
    raw: list[str] | None,
    input_names: list[str],
    target_names: list[str],
) -> dict[int, list[int]]:
    """Parse ``--target-feature-masks`` spec into {target_idx: [feature_idx, ...]}.

    Example input: ["qflx_tran_veg:h2ocan,smp_l_lev1", "fpsn_patch:sabg_snow"]
    Returns a dict keyed by target column index.
    Unknown target or feature names are warned and skipped.
    """
    if not raw:
        return {}
    result: dict[int, list[int]] = {}
    for spec in raw:
        if ":" not in spec:
            print(f"  [mask] WARNING: ignoring malformed spec '{spec}' (expected TARGET:FEAT1,FEAT2)")
            continue
        target_name, feat_str = spec.split(":", 1)
        target_name = target_name.strip()
        if target_name not in target_names:
            print(f"  [mask] WARNING: target '{target_name}' not in target list, skipping mask.")
            continue
        t_idx = target_names.index(target_name)
        f_indices: list[int] = []
        for fname in feat_str.split(","):
            fname = fname.strip()
            if fname not in input_names:
                print(f"  [mask] WARNING: feature '{fname}' not in input list, skipping.")
                continue
            f_indices.append(input_names.index(fname))
        if f_indices:
            result[t_idx] = f_indices
            feat_labels = [input_names[i] for i in f_indices]
            print(f"  [mask] {target_name}: zeroing features {feat_labels} during training and inference.")
    return result


def build_target_feature_masks(
    args: argparse.Namespace,
    input_names: list[str],
    target_names: list[str],
) -> dict[int, list[int]]:
    """Resolve explicit and convenience per-target input masks."""
    raw = list(args.target_feature_masks or [])
    if args.mask_sabg_snow_for_photosynthesis:
        raw.extend(
            f"{target}:sabg_snow"
            for target in ("fpsn_patch", "psnsun_patch", "psnsha_patch")
        )
    if args.mask_physiology_params_for_non_physiology_targets:
        param_features = ",".join(PHYSIOLOGY_PARAMETER_FEATURE_NAMES)
        physiology_targets = set(PHYSIOLOGY_PARAMETER_TARGET_NAMES)
        raw.extend(
            f"{target}:{param_features}"
            for target in target_names
            if target not in physiology_targets
        )
    return parse_target_feature_masks(raw, input_names, target_names)


def resolve_surfdata_file(training_file: Path, surfdata_file: Path | None) -> Path | None:
    if surfdata_file is not None:
        return surfdata_file
    return None


def select_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        raise RuntimeError("Requested CUDA device, but torch.cuda.is_available() is False.")
    if requested == "mps":
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        raise RuntimeError("Requested MPS device, but torch.backends.mps.is_available() is False.")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _decode_name_attr(raw_attr: object, expected_size: int, prefix: str) -> list[str]:
    if raw_attr is None:
        return [f"{prefix}_{idx}" for idx in range(expected_size)]
    if isinstance(raw_attr, bytes):
        text = raw_attr.decode("utf-8")
    else:
        text = str(raw_attr)
    names = [item.strip() for item in text.split(",") if item.strip()]
    if prefix == "feature" and len(names) == expected_size - 1:
        return names
    if len(names) != expected_size:
        return [f"{prefix}_{idx}" for idx in range(expected_size)]
    return names


def _read_matrix(var) -> np.ndarray:
    data = np.asarray(var[:], dtype=np.float64)
    dims = getattr(var, "dimensions", ())
    if dims in (("sample", "feature"), ("sample", "target"), ("sample", "param")):
        return data
    if dims in (("feature", "sample"), ("target", "sample"), ("param", "sample")):
        return data.T
    if data.ndim != 2:
        raise ValueError(f"Expected a rank-2 training array, got shape {data.shape}")
    if data.shape[0] >= data.shape[1]:
        return data
    return data.T


def _read_surfdata_profile(var, nlev: int) -> np.ndarray:
    dims = getattr(var, "dimensions", ())
    data = np.asarray(var[:], dtype=np.float64)
    if "gridcell" not in dims or "nlevsoi" not in dims:
        raise ValueError(f"Expected {getattr(var, 'name', 'surfdata profile')} to have gridcell and nlevsoi dimensions.")
    grid_axis = dims.index("gridcell")
    lev_axis = dims.index("nlevsoi")
    profile = np.moveaxis(data, (grid_axis, lev_axis), (0, 1))
    if profile.ndim != 2:
        profile = profile.reshape(profile.shape[0], profile.shape[1], -1)
        if profile.shape[-1] != 1:
            raise ValueError(f"Expected rank-2 surfdata profile, got shape {data.shape}.")
        profile = profile[..., 0]
    if profile.shape[1] < nlev:
        raise ValueError("Surfdata profile has fewer soil layers than the training file.")
    return profile[:, :nlev]


def _load_surfdata_watsat_profiles(surfdata_file: Path, nlev: int) -> np.ndarray:
    with Dataset(surfdata_file) as ds:
        missing = [name for name in ("PCT_SAND", "PCT_CLAY", "ORGANIC") if name not in ds.variables]
        if missing:
            raise KeyError(f"Cannot derive btran watsat without surfdata variables: {', '.join(missing)}")
        sand = _read_surfdata_profile(ds.variables["PCT_SAND"], nlev)
        clay = _read_surfdata_profile(ds.variables["PCT_CLAY"], nlev)
        organic_density = _read_surfdata_profile(ds.variables["ORGANIC"], nlev)

    organic_frac = organic_fraction_from_surfdata(organic_density)
    dz, z, zi = elm_soil_grid(nlevsoi=nlev)
    rootfr = np.full(nlev, 1.0 / float(nlev), dtype=np.float64)
    texture = SoilTextureProfile(
        sand_pct=sand,
        clay_pct=clay,
        organic_frac=organic_frac,
    )
    hydro = hydraulic_params_from_texture(
        dz=dz,
        z=z,
        zi=zi,
        texture=texture,
        rootfr=rootfr,
    )
    return np.asarray(hydro.watsat, dtype=np.float64)


def _map_training_index_to_surfdata_gridcell(
    training_index: np.ndarray | None,
    n_surf_cells: int,
    *,
    index_name: str = "cell_index",
    surfdata_cell_stride: int | None = None,
) -> np.ndarray | None:
    if training_index is None:
        if n_surf_cells == 1:
            return None
        raise ValueError(
            "Cannot map multi-cell surfdata to training rows because the training file has no "
            "gridcell_index or compatible cell_index."
        )

    index_values = np.asarray(training_index, dtype=np.int64)
    if index_values.size == 0:
        return np.empty(0, dtype=np.int64)
    if np.any(index_values < 0):
        raise ValueError(f"{index_name} contains negative values; cannot map to surfdata gridcells.")

    index_min = int(np.min(index_values))
    index_max = int(np.max(index_values))
    if n_surf_cells == 1:
        print("  [load] surfdata has one gridcell; using that profile for all training rows.")
        return np.zeros(index_values.shape, dtype=np.int64)

    if surfdata_cell_stride is not None:
        if surfdata_cell_stride <= 0:
            raise ValueError("--surfdata-cell-stride must be positive.")
        mapped = (index_values - 1) // int(surfdata_cell_stride)
        if np.any(mapped < 0) or np.any(mapped >= n_surf_cells):
            raise ValueError(
                f"--surfdata-cell-stride={surfdata_cell_stride} maps {index_name} range "
                f"{index_min}..{index_max} outside surfdata gridcell range 0..{n_surf_cells - 1}."
            )
        print(
            f"  [load] mapping {index_name} to surfdata gridcell with explicit "
            f"1-based slot stride {surfdata_cell_stride}; "
            f"{np.unique(mapped).size} of {n_surf_cells} surfdata cells are used."
        )
        return mapped

    if index_min == 0 and index_max < n_surf_cells:
        print(f"  [load] mapping {index_name} directly as 0-based surfdata gridcell indices.")
        return index_values.copy()

    if index_min >= 1 and index_max <= n_surf_cells:
        print(f"  [load] mapping {index_name} directly as 1-based surfdata gridcell indices.")
        return index_values - 1

    raise ValueError(
        f"Cannot infer surfdata gridcell mapping: {index_name} range is {index_min}..{index_max}, "
        f"but surfdata has {n_surf_cells} gridcells. Pass --surfdata-cell-stride if {index_name} "
        "stores a regular 1-based PFT/patch slot id, or regenerate the training file with an "
        "actual gridcell_index/surfdata_gridcell_index variable."
    )


def _watsat_for_training_rows(
    surfdata_file: Path,
    training_index: np.ndarray | None,
    nlev: int,
    *,
    index_name: str = "cell_index",
    surfdata_cell_stride: int | None = None,
) -> np.ndarray:
    watsat_by_cell = _load_surfdata_watsat_profiles(surfdata_file, nlev)
    mapped = _map_training_index_to_surfdata_gridcell(
        training_index,
        watsat_by_cell.shape[0],
        index_name=index_name,
        surfdata_cell_stride=surfdata_cell_stride,
    )
    if mapped is None:
        return watsat_by_cell[0]
    return watsat_by_cell[mapped]


def _read_surfdata_gridcell_scalar(surfdata_file: Path, variable_name: str) -> np.ndarray:
    with Dataset(surfdata_file) as ds:
        if variable_name not in ds.variables:
            raise KeyError(f"Cannot append surfdata feature without {variable_name} in {surfdata_file}.")
        var = ds.variables[variable_name]
        values = np.asarray(var[:], dtype=np.float64)
        dims = tuple(getattr(var, "dimensions", ()))

    if "gridcell" in dims:
        values = np.moveaxis(values, dims.index("gridcell"), 0)
    elif values.ndim != 1:
        raise ValueError(f"Expected {variable_name} to have a gridcell dimension or be rank-1.")

    if values.ndim == 1:
        return values
    trailing = int(np.prod(values.shape[1:]))
    if trailing != 1:
        raise ValueError(f"Expected scalar {variable_name} per gridcell, got shape {values.shape}.")
    return values.reshape(values.shape[0])


def _append_surfdata_input_features(
    features: np.ndarray,
    feature_names: list[str],
    surfdata_file: Path,
    training_index: np.ndarray | None,
    *,
    index_name: str = "cell_index",
    surfdata_cell_stride: int | None = None,
) -> tuple[np.ndarray, list[str]]:
    requested = (("soil_color", "SOIL_COLOR"),)
    fmap = {name: idx for idx, name in enumerate(feature_names)}
    appended_columns: list[np.ndarray] = []
    appended_names: list[str] = []
    for feature_name, surfdata_name in requested:
        if feature_name in fmap:
            continue
        by_cell = _read_surfdata_gridcell_scalar(surfdata_file, surfdata_name)
        mapped = _map_training_index_to_surfdata_gridcell(
            training_index,
            by_cell.shape[0],
            index_name=index_name,
            surfdata_cell_stride=surfdata_cell_stride,
        )
        if mapped is None:
            values = np.full(features.shape[0], by_cell[0], dtype=np.float64)
        else:
            values = by_cell[mapped]
        if values.shape[0] != features.shape[0]:
            raise ValueError(
                f"Surfdata-derived {feature_name!r} has {values.shape[0]} rows, "
                f"but training features have {features.shape[0]} rows."
            )
        appended_columns.append(values[:, None])
        appended_names.append(feature_name)

    if not appended_columns:
        return features, feature_names
    print(f"  [features] appending surfdata-derived input features: {appended_names}")
    augmented = np.concatenate([features, *appended_columns], axis=1)
    return augmented, feature_names + appended_names


def _append_btran_feature(
    features: np.ndarray,
    feature_names: list[str],
    parameters: np.ndarray,
    param_names: list[str],
    watsat: np.ndarray | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Append CLM-style btran derived from stored training state.

    This uses the stored soil-water state columns plus the per-patch
    ``smpso`` / ``smpsc`` parameters already present in the training file.
    It is intentionally aligned with the CLM-style drought-stress algebra used
    in the coupled replay, but remains self-contained for training-file use.
    """
    if "btran" in feature_names:
        keep_indices = [idx for idx, name in enumerate(feature_names) if name != "btran"]
        print("  [load] ignoring stored btran feature and deriving btran from state columns.")
        features = features[:, keep_indices]
        feature_names = [feature_names[idx] for idx in keep_indices]
    required = [
        "h2osoi_liqvol_lev1",
        "t_soisno_lev1",
        "rootfr_lev1",
    ]
    missing = [name for name in required if name not in feature_names]
    if missing:
        raise KeyError(f"Cannot derive btran without feature columns: {', '.join(missing)}")

    fmap = {name: idx for idx, name in enumerate(feature_names)}
    pmap = {name: idx for idx, name in enumerate(param_names)}
    nrow = features.shape[0]
    nlev = sum(1 for name in feature_names if name.startswith("h2osoi_liqvol_lev"))
    nroot = sum(1 for name in feature_names if name.startswith("rootfr_lev"))
    if nlev == 0:
        raise KeyError("Cannot derive btran without soil moisture layers.")
    if nroot == 0:
        raise KeyError("Cannot derive btran without root fraction layers.")
    if any(f"rootfr_lev{lev}" not in fmap for lev in range(1, nroot + 1)):
        raise KeyError("Cannot derive btran without root fraction layers.")
    def _row_parameter(name: str) -> np.ndarray:
        """Return a row-wise PFT parameter, preferring explicit feature columns."""
        if name in fmap:
            return np.asarray(features[:, fmap[name]], dtype=np.float64)
        if name not in pmap:
            raise KeyError(f"Cannot derive btran without {name} as a feature or parameter.")
        values = np.asarray(parameters[:, pmap[name]], dtype=np.float64)
        if values.shape[0] != nrow:
            raise ValueError(
                f"Parameter column {name!r} has {values.shape[0]} rows, "
                f"but training features have {nrow} rows."
            )
        return values

    smpso = _row_parameter("smpso")
    smpsc = _row_parameter("smpsc")
    h2osfc = np.asarray(features[:, fmap["h2osfc"]], dtype=np.float64) if "h2osfc" in fmap else None
    frac_h2osfc = np.asarray(features[:, fmap["frac_h2osfc"]], dtype=np.float64) if "frac_h2osfc" in fmap else np.zeros(features.shape[0], dtype=np.float64)
    tc_stress_c = -2.0
    tempk = np.stack([features[:, fmap[f"t_soisno_lev{lev}"]] for lev in range(1, nlev + 1)], axis=-1)
    liqvol = np.stack([features[:, fmap[f"h2osoi_liqvol_lev{lev}"]] for lev in range(1, nlev + 1)], axis=-1)
    rootfr = np.stack([features[:, fmap[f"rootfr_lev{lev}"]] for lev in range(1, nroot + 1)], axis=-1)
    eff_porosity = None
    if all(f"eff_porosity_lev{lev}" in fmap for lev in range(1, nlev + 1)):
        eff_porosity = np.stack([features[:, fmap[f"eff_porosity_lev{lev}"]] for lev in range(1, nlev + 1)], axis=-1)
    # Use ELM-computed matrix potential (smp_l) directly when available in the
    # training features.  This avoids the need to re-derive smp_node via the
    # Brooks-Corey curve (which would require sucsat/bsw not stored in features).
    smp_l = None
    if all(f"smp_l_lev{lev}" in fmap for lev in range(1, nlev + 1)):
        smp_l = np.stack([features[:, fmap[f"smp_l_lev{lev}"]] for lev in range(1, nlev + 1)], axis=-1)
    if watsat is not None:
        watsat = np.asarray(watsat, dtype=np.float64)
        if watsat.shape[-1] < nlev:
            raise KeyError("Surfdata-derived watsat has fewer layers than the training soil profile.")
        watsat = watsat[..., :nlev]
        if watsat.ndim == 2 and watsat.shape[0] not in (1, nrow):
            raise ValueError(
                f"Row-specific watsat has {watsat.shape[0]} rows, but training features have {nrow} rows."
            )
    if nroot > nlev:
        rootfr_for_btran = rootfr[:, :nlev]
    else:
        rootfr_for_btran = rootfr
    btran_col, _ = compute_btran_and_root_fraction(
        liqvol=liqvol,
        tempk=tempk,
        frac_h2osfc=frac_h2osfc,
        rootfr=rootfr_for_btran,
        smpso=smpso,
        smpsc=smpsc,
        tc_stress_c=tc_stress_c,
        h2osfc_mm=h2osfc,
        eff_porosity=eff_porosity,
        smp_l=smp_l,
        watsat=watsat,
    )
    btran_col = btran_col[..., None]
    augmented = np.concatenate([features, btran_col], axis=1)
    return augmented, feature_names + ["btran"]


def _append_parameter_input_features(
    features: np.ndarray,
    feature_names: list[str],
    parameters: np.ndarray,
    param_names: list[str],
    names_to_append: Iterable[str] | None,
) -> tuple[np.ndarray, list[str]]:
    requested = [str(name).strip() for name in (names_to_append or ()) if str(name).strip()]
    if not requested:
        return features, feature_names

    fmap = {name: idx for idx, name in enumerate(feature_names)}
    pmap = {name: idx for idx, name in enumerate(param_names)}
    appended_columns: list[np.ndarray] = []
    appended_names: list[str] = []
    for name in requested:
        if name in fmap:
            continue
        if name not in pmap:
            raise KeyError(f"Cannot append requested input feature {name!r}; it is not in param_names.")
        values = np.asarray(parameters[:, pmap[name]], dtype=np.float64)
        if values.shape[0] != features.shape[0]:
            raise ValueError(
                f"Parameter column {name!r} has {values.shape[0]} rows, "
                f"but training features have {features.shape[0]} rows."
            )
        appended_columns.append(values[:, None])
        appended_names.append(name)

    if not appended_columns:
        return features, feature_names
    print(f"  [features] appending parameter-derived input features: {appended_names}")
    augmented = np.concatenate([features, *appended_columns], axis=1)
    return augmented, feature_names + appended_names


def _append_derived_forcing_input_features(
    features: np.ndarray,
    feature_names: list[str],
) -> tuple[np.ndarray, list[str]]:
    """Append locally derived forcing features that are reproducible in replay."""
    fmap = {name: idx for idx, name in enumerate(feature_names)}
    if "forc_vpd" in fmap:
        return features, feature_names
    required = ("forc_t", "forc_q", "forc_pbot")
    missing = [name for name in required if name not in fmap]
    if missing:
        raise KeyError(f"Cannot derive forc_vpd without feature columns: {', '.join(missing)}")
    forc_vpd = vapor_pressure_deficit_from_forcing(
        features[:, fmap["forc_t"]],
        features[:, fmap["forc_q"]],
        features[:, fmap["forc_pbot"]],
    )
    print("  [features] appending forcing-derived input feature: forc_vpd")
    return np.concatenate([features, forc_vpd[:, None]], axis=1), feature_names + ["forc_vpd"]


def _drop_feature_prefixes(
    features: np.ndarray,
    feature_names: list[str],
    prefixes: tuple[str, ...],
    keep_names: Iterable[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    keep_exact = frozenset(keep_names or ())
    keep_indices = [
        idx
        for idx, name in enumerate(feature_names)
        if name in keep_exact or not name.startswith(prefixes)
    ]
    if len(keep_indices) == len(feature_names):
        return features, feature_names
    return features[:, keep_indices], [feature_names[idx] for idx in keep_indices]


def _drop_feature_names(
    features: np.ndarray,
    feature_names: list[str],
    names_to_drop: Iterable[str] | None,
) -> tuple[np.ndarray, list[str], list[str]]:
    drop_names = {str(name).strip() for name in (names_to_drop or ()) if str(name).strip()}
    if not drop_names:
        return features, feature_names, []
    keep_indices = [idx for idx, name in enumerate(feature_names) if name not in drop_names]
    if len(keep_indices) == len(feature_names):
        return features, feature_names, []
    dropped = [feature_names[idx] for idx in range(len(feature_names)) if idx not in keep_indices]
    return features[:, keep_indices], [feature_names[idx] for idx in keep_indices], dropped


def _drop_float32_constant_features(
    features: np.ndarray,
    feature_names: list[str],
    keep_names: Iterable[str] | None = None,
) -> tuple[np.ndarray, list[str], list[str], list[int]]:
    """Drop features with variation below float32 resolution for this patch.

    The MLP is trained and normally inferred in float32. If a feature's standard
    deviation is smaller than the representable spacing around its mean, the
    training z-score is numerical noise that cannot be reproduced reliably in
    replay. Drop only those per-patch constants; if the feature is meaningfully
    variable in another dataset/patch, it is retained automatically.
    """
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0)
    threshold = np.maximum(
        FLOAT32_CONSTANT_FEATURE_ATOL,
        FLOAT32_CONSTANT_FEATURE_RTOL * np.maximum(np.abs(mean), 1.0),
    )
    drop_mask = std <= threshold
    keep_exact = frozenset(keep_names or ())
    for idx, name in enumerate(feature_names):
        if name == "btran" or name in keep_exact:
            drop_mask[idx] = False
    keep_indices = [idx for idx, drop in enumerate(drop_mask) if not drop]
    if len(keep_indices) == len(feature_names):
        return features, feature_names, [], list(range(len(feature_names)))
    dropped = [feature_names[idx] for idx, drop in enumerate(drop_mask) if drop]
    return features[:, keep_indices], [feature_names[idx] for idx in keep_indices], dropped, keep_indices


def load_training_data(
    path: Path,
    surfdata_file: Path | None = None,
    surfdata_cell_stride: int | None = None,
    drop_input_names: Iterable[str] | None = DEFAULT_DROP_INPUT_NAMES,
    parameter_input_names: Iterable[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, dict[str, np.ndarray], dict[str, list[str]]]:
    with Dataset(path, "r") as ds:
        patch_index = np.asarray(ds.variables["patch_index"][:], dtype=np.int64)
        cell_index = None
        cell_index_name = None
        for candidate_name in ("pft_index", "cell_index", "pfts1d_pft_index"):
            if candidate_name in ds.variables:
                cell_index = np.asarray(ds.variables[candidate_name][:], dtype=np.int64)
                cell_index_name = candidate_name
                break
        surfdata_gridcell_index = None
        surfdata_gridcell_name = None
        for candidate_name in ("surfdata_gridcell_index", "gridcell_index", "pfts1d_gridcell_index"):
            if candidate_name in ds.variables:
                surfdata_gridcell_index = np.asarray(ds.variables[candidate_name][:], dtype=np.int64)
                surfdata_gridcell_name = candidate_name
                break
        features = _read_matrix(ds.variables["features"])
        targets = _read_matrix(ds.variables["targets"])
        parameters = _read_matrix(ds.variables["parameters"])
        if features.shape[0] != patch_index.size:
            raise ValueError(
                f"Feature rows ({features.shape[0]}) do not match patch_index size ({patch_index.size})"
            )
        if targets.shape[0] != patch_index.size:
            raise ValueError(
                f"Target rows ({targets.shape[0]}) do not match patch_index size ({patch_index.size})"
            )

        time_info = {
            "nstep": np.asarray(ds.variables["nstep"][:], dtype=np.int64),
            "year": np.asarray(ds.variables["year"][:], dtype=np.int64),
            "month": np.asarray(ds.variables["month"][:], dtype=np.int64),
            "day": np.asarray(ds.variables["day"][:], dtype=np.int64),
            "sec": np.asarray(ds.variables["sec"][:], dtype=np.int64),
        }
        if cell_index is not None:
            time_info["pft_index"] = cell_index
        if surfdata_gridcell_index is not None:
            time_info["gridcell_index"] = surfdata_gridcell_index
        names = {
            "feature_names": _decode_name_attr(
                getattr(ds, "feature_names", None), features.shape[1], "feature"
            ),
            "target_names": _decode_name_attr(
                getattr(ds, "target_names", None), targets.shape[1], "target"
            ),
            "param_names": _decode_name_attr(
                getattr(ds, "param_names", None), parameters.shape[1], "param"
            ),
        }

    keep_target_indices = [
        idx for idx, name in enumerate(names["target_names"]) if name not in SKIP_TARGET_NAMES
    ]
    dropped_target_names = [name for name in names["target_names"] if name in SKIP_TARGET_NAMES]
    targets = targets[:, keep_target_indices]
    names["target_names"] = [names["target_names"][idx] for idx in keep_target_indices]
    if features.shape[1] == len(names["feature_names"]) + 1 and "btran" not in names["feature_names"]:
        print(
            "  [load] WARNING: feature matrix has one legacy trailing column without a name; "
            "dropping it and deriving btran from the stored state columns."
        )
        features = features[:, :-1]
    watsat = None
    if surfdata_file is not None:
        surfdata_index = surfdata_gridcell_index if surfdata_gridcell_index is not None else cell_index
        surfdata_index_name = surfdata_gridcell_name if surfdata_gridcell_name is not None else cell_index_name
        if surfdata_gridcell_name is not None:
            print(f"  [load] mapping surfdata with training variable {surfdata_gridcell_name}.")
        watsat = _watsat_for_training_rows(
            surfdata_file,
            surfdata_index,
            sum(1 for name in names["feature_names"] if name.startswith("h2osoi_liqvol_lev")),
            index_name=surfdata_index_name or "cell_index",
            surfdata_cell_stride=surfdata_cell_stride,
        )
    elif any(name.startswith("eff_porosity_lev") for name in names["feature_names"]):
        print(
            "  [load] WARNING: deriving btran without surfdata watsat; "
            "eff_porosity/watsat scaling will be omitted."
        )
    features, names["feature_names"] = _append_parameter_input_features(
        features,
        names["feature_names"],
        parameters,
        names["param_names"],
        parameter_input_names,
    )
    if surfdata_file is not None:
        features, names["feature_names"] = _append_surfdata_input_features(
            features,
            names["feature_names"],
            surfdata_file,
            surfdata_index,
            index_name=surfdata_index_name or "cell_index",
            surfdata_cell_stride=surfdata_cell_stride,
        )
    features, names["feature_names"] = _append_derived_forcing_input_features(
        features,
        names["feature_names"],
    )
    features, names["feature_names"] = _append_btran_feature(
        features,
        names["feature_names"],
        parameters,
        names["param_names"],
        watsat=watsat,
    )
    features, names["feature_names"] = _drop_feature_prefixes(
        features,
        names["feature_names"],
        DROP_INPUT_PREFIXES,
        keep_names=KEEP_DROPPED_PREFIX_INPUT_NAMES,
    )
    features, names["feature_names"], dropped_input_names = _drop_feature_names(
        features,
        names["feature_names"],
        drop_input_names,
    )
    if dropped_input_names:
        print(f"  [features] dropping exact input names unavailable in regional replay: {dropped_input_names}")
    names["input_names"] = names["feature_names"]
    names["dropped_target_names"] = dropped_target_names
    names["dropped_input_names"] = dropped_input_names
    return patch_index, cell_index, features, targets, time_info, names


def iter_patch_datasets(
    patch_index: np.ndarray,
    features: np.ndarray,
    targets: np.ndarray,
    time_info: dict[str, np.ndarray],
    selected_patch_indices: set[int] | None,
    spval_threshold: float,
    *,
    cell_index: np.ndarray | None = None,
    input_names: list[str] | None = None,
    target_names: list[str] | None = None,
    state_target_lag_steps: int = 0,
    state_target_names: list[str] | None = None,
    nighttime_psn_mask: bool = True,
) -> Iterable[PatchDataset]:
    for patch_id in np.unique(patch_index):
        patch_id_int = int(patch_id)
        if selected_patch_indices is not None and patch_id_int not in selected_patch_indices:
            continue
        mask = patch_index == patch_id
        patch_features = features[mask, :]
        patch_targets = targets[mask, :]
        feature_nonfinite_counts = np.count_nonzero(~np.isfinite(patch_features), axis=0)
        target_nonfinite_counts = np.count_nonzero(~np.isfinite(patch_targets), axis=0)
        feature_large_counts = np.count_nonzero(np.abs(patch_features) > spval_threshold, axis=0)
        target_large_counts = np.count_nonzero(np.abs(patch_targets) > spval_threshold, axis=0)
        feature_valid_mask = (
            np.all(np.isfinite(patch_features), axis=1)
            & np.all(np.abs(patch_features) <= spval_threshold, axis=1)
        )
        target_valid_mask = np.isfinite(patch_targets) & (np.abs(patch_targets) <= spval_threshold)
        dropped_nonfinite = int(np.count_nonzero(~feature_valid_mask))
        filtered_features = patch_features[feature_valid_mask, :]
        filtered_targets = patch_targets[feature_valid_mask, :]
        filtered_target_valid = target_valid_mask[feature_valid_mask, :]
        filtered_years = time_info["year"][mask][feature_valid_mask]
        filtered_months = time_info["month"][mask][feature_valid_mask]
        filtered_days = time_info["day"][mask][feature_valid_mask]
        filtered_secs = time_info["sec"][mask][feature_valid_mask]
        filtered_nsteps = time_info["nstep"][mask][feature_valid_mask]
        filtered_pft_indices = (
            time_info["pft_index"][mask][feature_valid_mask]
            if "pft_index" in time_info
            else None
        )
        filtered_gridcell_indices = (
            time_info["gridcell_index"][mask][feature_valid_mask]
            if "gridcell_index" in time_info
            else None
        )

        if filtered_nsteps.size > 0:
            order = np.argsort(filtered_nsteps, kind="stable")
            filtered_features = filtered_features[order, :]
            filtered_targets = filtered_targets[order, :]
            filtered_target_valid = filtered_target_valid[order, :]
            filtered_years = filtered_years[order]
            filtered_months = filtered_months[order]
            filtered_days = filtered_days[order]
            filtered_secs = filtered_secs[order]
            filtered_nsteps = filtered_nsteps[order]
            if filtered_pft_indices is not None:
                filtered_pft_indices = filtered_pft_indices[order]
            if filtered_gridcell_indices is not None:
                filtered_gridcell_indices = filtered_gridcell_indices[order]

        if state_target_lag_steps > 0 and target_names and state_target_names:
            lagged_indices = [
                target_names.index(name)
                for name in state_target_names
                if name in target_names
            ]
            if lagged_indices:
                # Apply lag per PFT sample identity so we never pair features from
                # one subgrid record with targets from another.
                if cell_index is None:
                    raise ValueError(
                        "State-target lagging requires a training file with 'pft_index' or 'cell_index'. "
                        "Regenerate or copy the correct training file, or rerun with "
                        "--state-target-lag-steps 0 to disable lagging."
                    )
                cell_ids = cell_index[mask][feature_valid_mask][order]

                all_feat, all_tgt, all_tgt_valid = [], [], []
                all_years, all_months, all_days, all_secs, all_nsteps = [], [], [], [], []
                all_pft_indices, all_gridcell_indices = [], []

                for cid in np.unique(cell_ids):
                    cmask = cell_ids == cid
                    c_feat    = filtered_features[cmask, :]
                    c_tgt     = filtered_targets[cmask, :]
                    c_valid   = filtered_target_valid[cmask, :]
                    c_years   = filtered_years[cmask]
                    c_months  = filtered_months[cmask]
                    c_days    = filtered_days[cmask]
                    c_secs    = filtered_secs[cmask]
                    c_nsteps  = filtered_nsteps[cmask]
                    c_pft_indices = filtered_pft_indices[cmask] if filtered_pft_indices is not None else None
                    c_gridcell_indices = (
                        filtered_gridcell_indices[cmask]
                        if filtered_gridcell_indices is not None
                        else None
                    )

                    if c_tgt.shape[0] <= state_target_lag_steps:
                        continue
                    ac = c_tgt.shape[0] - state_target_lag_steps
                    lagged_tgt   = np.array(c_tgt[:ac, :],   copy=True)
                    lagged_valid = np.array(c_valid[:ac, :], copy=True)
                    for t_idx in lagged_indices:
                        lagged_tgt[:, t_idx]   = c_tgt[state_target_lag_steps:, t_idx]
                        lagged_valid[:, t_idx] = c_valid[state_target_lag_steps:, t_idx]

                    all_feat.append(c_feat[:ac, :])
                    all_tgt.append(lagged_tgt)
                    all_tgt_valid.append(lagged_valid)
                    all_years.append(c_years[:ac])
                    all_months.append(c_months[:ac])
                    all_days.append(c_days[:ac])
                    all_secs.append(c_secs[:ac])
                    all_nsteps.append(c_nsteps[:ac])
                    if c_pft_indices is not None:
                        all_pft_indices.append(c_pft_indices[:ac])
                    if c_gridcell_indices is not None:
                        all_gridcell_indices.append(c_gridcell_indices[:ac])

                if not all_feat:
                    continue
                filtered_features      = np.concatenate(all_feat,       axis=0)
                filtered_targets       = np.concatenate(all_tgt,        axis=0)
                filtered_target_valid  = np.concatenate(all_tgt_valid,  axis=0)
                filtered_years         = np.concatenate(all_years)
                filtered_months        = np.concatenate(all_months)
                filtered_days          = np.concatenate(all_days)
                filtered_secs          = np.concatenate(all_secs)
                filtered_nsteps        = np.concatenate(all_nsteps)
                filtered_pft_indices = (
                    np.concatenate(all_pft_indices)
                    if all_pft_indices
                    else None
                )
                filtered_gridcell_indices = (
                    np.concatenate(all_gridcell_indices)
                    if all_gridcell_indices
                    else None
                )

        # Mask nighttime samples out of photosynthesis targets so the loss is
        # only computed on samples where solar radiation is actually present.
        # Mask samples where photosynthesis is trivially zero out of the psn
        # targets so the model must learn the real photosynthesis signal.
        # Two conditions make psn trivially zero:
        #   1. Night: forc_swrad <= 0  (no incoming solar)
        #   2. No canopy: elai_patch == 0  (no exposed leaf area)
        if (
            nighttime_psn_mask
            and target_names is not None
            and input_names is not None
        ):
            zero_psn = np.zeros(filtered_features.shape[0], dtype=bool)
            if "forc_swrad" in input_names:
                zero_psn |= filtered_features[:, input_names.index("forc_swrad")] <= ZERO_RADIATION_THRESHOLD
            if "elai_patch" in input_names:
                zero_psn |= filtered_features[:, input_names.index("elai_patch")] <= ZERO_LAI_THRESHOLD
            if "btran" in input_names:
                zero_psn |= filtered_features[:, input_names.index("btran")] < ZERO_BTRAN_THRESHOLD
            if zero_psn.any():
                for tname in PHOTOSYNTHESIS_TARGET_NAMES:
                    if tname in target_names:
                        filtered_target_valid[zero_psn, target_names.index(tname)] = False

        # Mask qflx_tran_veg when elai_patch == 0: transpiration is trivially
        # zero with no exposed leaf area and including those samples would swamp
        # the signal from the leaf-on period.
        if (
            target_names is not None
            and input_names is not None
            and "elai_patch" in input_names
            and "qflx_tran_veg" in target_names
        ):
            no_canopy = filtered_features[:, input_names.index("elai_patch")] == 0.0
            if no_canopy.any():
                filtered_target_valid[no_canopy, target_names.index("qflx_tran_veg")] = False

        yield PatchDataset(
            patch_index=patch_id_int,
            features=filtered_features,
            targets=filtered_targets,
            target_valid_mask=filtered_target_valid,
            years=filtered_years,
            months=filtered_months,
            days=filtered_days,
            secs=filtered_secs,
            nsteps=filtered_nsteps,
            pft_indices=filtered_pft_indices,
            gridcell_indices=filtered_gridcell_indices,
            dropped_nonfinite=dropped_nonfinite,
            feature_nonfinite_counts=feature_nonfinite_counts,
            target_nonfinite_counts=target_nonfinite_counts,
            feature_large_counts=feature_large_counts,
            target_large_counts=target_large_counts,
        )


def split_train_validation(
    sample_count: int,
    val_fraction: float,
    random_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"Expected 0 < val_fraction < 1, got {val_fraction}")
    rng = np.random.default_rng(random_seed)
    all_idx = np.arange(sample_count, dtype=np.int64)
    n_val = max(1, min(int(np.floor(sample_count * val_fraction)), sample_count - 1))
    rng.shuffle(all_idx)
    val_idx = np.sort(all_idx[:n_val])
    train_idx = np.sort(all_idx[n_val:])
    return train_idx, val_idx


def standardize_train_validation(
    train_data: np.ndarray,
    val_data: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = train_data.mean(axis=0)
    std = train_data.std(axis=0)
    std = np.where(std > 0.0, std, 1.0)
    train_scaled = (train_data - mean) / std
    val_scaled = (val_data - mean) / std
    return train_scaled, val_scaled, mean, std


def standardize_targets_with_mask(
    train_targets: np.ndarray,
    train_valid_mask: np.ndarray,
    val_targets: np.ndarray,
    target_names: list[str],
    log1p_target_names: Iterable[str] = DEFAULT_LOG1P_TARGET_NAMES,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    transformed_train_targets = np.array(train_targets, copy=True)
    transformed_val_targets = np.array(val_targets, copy=True)
    log1p_targets = set(log1p_target_names or DEFAULT_LOG1P_TARGET_NAMES)

    for idx, name in enumerate(target_names):
        if name in log1p_targets:
            transformed_train_targets[:, idx] = np.log1p(np.maximum(transformed_train_targets[:, idx], 0.0))
            transformed_val_targets[:, idx] = np.log1p(np.maximum(transformed_val_targets[:, idx], 0.0))

    y_mean = np.zeros(train_targets.shape[1], dtype=np.float64)
    y_std = np.ones(train_targets.shape[1], dtype=np.float64)
    for idx in range(train_targets.shape[1]):
        valid = train_valid_mask[:, idx]
        if np.any(valid):
            y_mean[idx] = transformed_train_targets[valid, idx].mean()
            std = transformed_train_targets[valid, idx].std()
            y_std[idx] = std if std > 0.0 else 1.0
    train_scaled = (transformed_train_targets - y_mean) / y_std
    val_scaled = (transformed_val_targets - y_mean) / y_std
    return train_scaled, val_scaled, y_mean, y_std


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    valid_mask: np.ndarray,
    target_names: list[str],
) -> dict[str, object]:
    rmse_per_target = np.zeros(y_true.shape[1], dtype=np.float64)
    mae_per_target = np.zeros(y_true.shape[1], dtype=np.float64)
    zero_fraction_per_target = np.zeros(y_true.shape[1], dtype=np.float64)
    r2_per_target = np.zeros(y_true.shape[1], dtype=np.float64)
    for idx in range(y_true.shape[1]):
        valid = valid_mask[:, idx]
        if np.any(valid):
            diff = y_true[valid, idx] - y_pred[valid, idx]
            rmse_per_target[idx] = float(np.sqrt(np.mean(diff ** 2)))
            mae_per_target[idx] = float(np.mean(np.abs(diff)))
            zero_fraction_per_target[idx] = float(np.mean(np.isclose(y_true[valid, idx], 0.0)))
            if np.count_nonzero(valid) > 1:
                r2_per_target[idx] = float(r2_score(y_true[valid, idx], y_pred[valid, idx]))
            else:
                r2_per_target[idx] = np.nan
        else:
            rmse_per_target[idx] = np.nan
            mae_per_target[idx] = np.nan
            zero_fraction_per_target[idx] = np.nan
            r2_per_target[idx] = np.nan
    return {
        "rmse_mean": float(np.nanmean(rmse_per_target)),
        "rmse_max": float(np.nanmax(rmse_per_target)),
        "mae_mean": float(np.nanmean(mae_per_target)),
        "r2_mean": float(np.nanmean(r2_per_target)),
        "rmse_by_target": {
            name: float(value) if np.isfinite(value) else None
            for name, value in zip(target_names, rmse_per_target)
        },
        "mae_by_target": {
            name: float(value) if np.isfinite(value) else None
            for name, value in zip(target_names, mae_per_target)
        },
        "r2_by_target": {
            name: float(value) if np.isfinite(value) else None
            for name, value in zip(target_names, r2_per_target)
        },
        "zero_fraction_by_target": {
            name: float(value) if np.isfinite(value) else None
            for name, value in zip(target_names, zero_fraction_per_target)
        },
        "valid_count_by_target": {
            name: int(value)
            for name, value in zip(target_names, np.count_nonzero(valid_mask, axis=0))
        },
        "rmse_per_target": rmse_per_target.tolist(),
        "mae_per_target": mae_per_target.tolist(),
        "r2_per_target": r2_per_target.tolist(),
        "zero_fraction_per_target": zero_fraction_per_target.tolist(),
        "valid_count_per_target": np.count_nonzero(valid_mask, axis=0).tolist(),
    }


def _predict_numpy(
    model: nn.Module,
    x_scaled: np.ndarray,
    device: torch.device,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    target_names: list[str],
    x_raw: np.ndarray | None = None,
    input_names: list[str] | None = None,
    target_feature_masks: dict[int, list[int]] | None = None,
    log1p_target_names: Iterable[str] = DEFAULT_LOG1P_TARGET_NAMES,
) -> np.ndarray:
    with torch.no_grad():
        x_tensor = torch.tensor(x_scaled, dtype=torch.float32, device=device)
        y_pred_scaled_tensor = model(x_tensor)
        y_pred_scaled_tensor = _apply_input_masks_for_targets(
            y_pred_scaled_tensor,
            x_tensor,
            model,  # type: ignore[arg-type]
            target_feature_masks or {},
        )
        y_pred_scaled = y_pred_scaled_tensor.cpu().numpy()
    y_pred = y_pred_scaled * y_std + y_mean
    return postprocess_predictions(
        y_pred,
        target_names,
        x_raw=x_raw,
        input_names=input_names,
        log1p_target_names=log1p_target_names,
    )


def postprocess_predictions(
    y_pred: np.ndarray,
    target_names: list[str],
    *,
    x_raw: np.ndarray | None = None,
    input_names: list[str] | None = None,
    log1p_target_names: Iterable[str] = DEFAULT_LOG1P_TARGET_NAMES,
) -> np.ndarray:
    y_checked = np.array(y_pred, copy=True)
    log1p_targets = set(log1p_target_names)

    for idx, name in enumerate(target_names):
        if name in log1p_targets:
            y_checked[:, idx] = np.expm1(y_checked[:, idx])
        if name in NONNEGATIVE_TARGET_NAMES:
            y_checked[:, idx] = np.maximum(y_checked[:, idx], 0.0)

    target_idx_map = {name: idx for idx, name in enumerate(target_names)}
    if (
        x_raw is not None
        and input_names is not None
        and "elai_patch" in input_names
        and "laisun_patch" in target_idx_map
        and "laisha_patch" in target_idx_map
    ):
        lai_idx = input_names.index("elai_patch")
        elai = np.maximum(np.asarray(x_raw[:, lai_idx], dtype=np.float64), 0.0)
        laisun_idx = target_idx_map["laisun_patch"]
        laisha_idx = target_idx_map["laisha_patch"]
        lai_sum = y_checked[:, laisun_idx] + y_checked[:, laisha_idx]
        scale = np.minimum(1.0, elai / np.maximum(lai_sum, 1.0e-12))
        y_checked[:, laisun_idx] *= scale
        y_checked[:, laisha_idx] *= scale

    if x_raw is not None and input_names is not None:
        zero_psn = np.zeros(y_checked.shape[0], dtype=bool)
        has_zero_rule = False
        if "forc_swrad" in input_names:
            sw_idx = input_names.index("forc_swrad")
            zero_psn |= np.asarray(x_raw[:, sw_idx], dtype=np.float64) <= ZERO_RADIATION_THRESHOLD
            has_zero_rule = True
        if "elai_patch" in input_names:
            lai_idx = input_names.index("elai_patch")
            zero_psn |= np.asarray(x_raw[:, lai_idx], dtype=np.float64) <= ZERO_LAI_THRESHOLD
            has_zero_rule = True
        if "btran" in input_names:
            btran_idx = input_names.index("btran")
            zero_psn |= np.asarray(x_raw[:, btran_idx], dtype=np.float64) < ZERO_BTRAN_THRESHOLD
            has_zero_rule = True
        if has_zero_rule and np.any(zero_psn):
            for target_name in PHOTOSYNTHESIS_TARGET_NAMES:
                if target_name in target_names:
                    y_checked[zero_psn, target_names.index(target_name)] = 0.0

    return y_checked


def _normalized_target_signal(
    y: np.ndarray,
    valid_mask: np.ndarray,
    target_names: list[str],
    target_name: str,
) -> np.ndarray:
    if target_name not in target_names:
        return np.zeros(y.shape[0], dtype=np.float64)

    idx = target_names.index(target_name)
    signal = np.zeros(y.shape[0], dtype=np.float64)
    valid = valid_mask[:, idx]
    if not np.any(valid):
        return signal

    positive = np.maximum(y[valid, idx], 0.0)
    scale = np.quantile(positive, 0.95)
    if not np.isfinite(scale) or scale <= 0.0:
        return signal

    signal[valid] = np.clip(positive / scale, 0.0, 1.0)
    return signal


def _normalized_abs_target_signal(
    y: np.ndarray,
    valid_mask: np.ndarray,
    target_names: list[str],
    target_name: str,
) -> np.ndarray:
    if target_name not in target_names:
        return np.zeros(y.shape[0], dtype=np.float64)

    idx = target_names.index(target_name)
    signal = np.zeros(y.shape[0], dtype=np.float64)
    valid = valid_mask[:, idx]
    if not np.any(valid):
        return signal

    magnitude = np.abs(y[valid, idx])
    scale = np.quantile(magnitude, 0.95)
    if not np.isfinite(scale) or scale <= 0.0:
        return signal

    signal[valid] = np.clip(magnitude / scale, 0.0, 1.0)
    return signal


def compute_sample_weights(
    y: np.ndarray,
    valid_mask: np.ndarray,
    target_names: list[str],
    qflx_tran_weight: float,
    psnsun_weight: float,
    qflx_prec_weight: float,
    fpsn_weight: float,
    hs_soil_weight: float,
    hs_h2osfc_weight: float,
    dhsdT_weight: float,
) -> np.ndarray:
    weights = np.ones(y.shape[0], dtype=np.float64)
    weights += qflx_tran_weight * _normalized_target_signal(
        y, valid_mask, target_names, "qflx_tran_veg"
    )
    weights += psnsun_weight * _normalized_target_signal(
        y, valid_mask, target_names, "psnsun_patch"
    )
    weights += qflx_prec_weight * _normalized_target_signal(
        y, valid_mask, target_names, "qflx_prec_grnd"
    )
    weights += fpsn_weight * _normalized_target_signal(
        y, valid_mask, target_names, "fpsn_patch"
    )
    weights += hs_soil_weight * _normalized_abs_target_signal(
        y, valid_mask, target_names, "hs_soil"
    )
    weights += hs_soil_weight * _normalized_abs_target_signal(
        y, valid_mask, target_names, "hs_top_snow"
    )
    weights += hs_h2osfc_weight * _normalized_abs_target_signal(
        y, valid_mask, target_names, "hs_h2osfc"
    )
    weights += dhsdT_weight * _normalized_abs_target_signal(
        y, valid_mask, target_names, "dhsdT"
    )
    return weights


def compute_target_loss_weights(
    y: np.ndarray,
    valid_mask: np.ndarray,
    target_names: list[str],
    psn_low_value_weight: float,
    psn_low_value_threshold: float,
) -> np.ndarray:
    weights = np.ones(y.shape, dtype=np.float64)
    if psn_low_value_weight <= 1.0:
        return weights
    for target_name in PHOTOSYNTHESIS_TARGET_NAMES:
        if target_name not in target_names:
            continue
        idx = target_names.index(target_name)
        low_valid = valid_mask[:, idx] & (np.asarray(y[:, idx], dtype=np.float64) <= psn_low_value_threshold)
        weights[low_valid, idx] *= psn_low_value_weight
    return weights


def _importance_prediction_bounds(
    y_eval: np.ndarray,
    valid: np.ndarray,
    target_idx: int,
    target_name: str,
) -> tuple[float, float]:
    """Return broad target-space bounds for permutation diagnostics.

    Permuting correlated environmental/state features can create combinations
    that never occur in the ELM trajectory. For log1p-trained targets, those
    off-manifold inputs can explode through expm1 and swamp the sensitivity
    plot with impossible values. Use a generous validation-domain bound so the
    diagnostic reports large degradation without treating runaway extrapolation
    as a physically meaningful RMSE.
    """
    values = np.asarray(y_eval[valid, target_idx], dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return -np.inf, np.inf

    lower = 0.0 if target_name in NONNEGATIVE_TARGET_NAMES else float(np.min(values))
    upper = float(np.max(values))
    spread = upper - lower
    if not np.isfinite(spread) or spread <= 0.0:
        spread = max(abs(upper), abs(lower), 1.0)
    upper = upper + 0.25 * spread
    if target_name in NONNEGATIVE_TARGET_NAMES:
        upper = max(upper, 1.0)
    else:
        lower = lower - 0.25 * spread
    return lower, upper


def estimate_permutation_importance(
    model: nn.Module,
    x_val_scaled: np.ndarray,
    x_val_raw: np.ndarray | None,
    y_val: np.ndarray,
    y_val_valid: np.ndarray,
    input_names: list[str],
    target_names: list[str],
    y_mean: np.ndarray,
    y_std: np.ndarray,
    device: torch.device,
    requested_targets: tuple[str, ...] | list[str],
    max_samples: int,
    random_seed: int,
    target_feature_masks: dict[int, list[int]] | None = None,
    log1p_target_names: Iterable[str] = DEFAULT_LOG1P_TARGET_NAMES,
) -> dict[str, list[dict[str, float | str | None]]]:
    target_indices = [target_names.index(name) for name in requested_targets if name in target_names]
    if not target_indices or x_val_scaled.shape[0] == 0:
        return {}

    if x_val_scaled.shape[0] > max_samples:
        rng = np.random.default_rng(random_seed)
        sample_idx = np.sort(rng.choice(x_val_scaled.shape[0], size=max_samples, replace=False))
        x_eval = x_val_scaled[sample_idx, :]
        x_eval_raw = x_val_raw[sample_idx, :] if x_val_raw is not None else None
        y_eval = y_val[sample_idx, :]
        valid_eval = y_val_valid[sample_idx, :]
    else:
        x_eval = x_val_scaled
        x_eval_raw = x_val_raw
        y_eval = y_val
        valid_eval = y_val_valid

    baseline_pred = _predict_numpy(
        model,
        x_eval,
        device,
        y_mean,
        y_std,
        target_names,
        x_raw=x_eval_raw,
        input_names=input_names,
        target_feature_masks=target_feature_masks,
        log1p_target_names=log1p_target_names,
    )
    importance: dict[str, list[dict[str, float | str | None]]] = {}
    rng = np.random.default_rng(random_seed)

    for target_idx in target_indices:
        target_name = target_names[target_idx]
        valid = valid_eval[:, target_idx]
        if not np.any(valid):
            importance[target_name] = []
            continue

        pred_min, pred_max = _importance_prediction_bounds(
            y_eval,
            valid,
            target_idx,
            target_name,
        )
        baseline_target_pred = np.clip(baseline_pred[:, target_idx], pred_min, pred_max)
        baseline_rmse = float(
            np.sqrt(np.mean((baseline_target_pred[valid] - y_eval[valid, target_idx]) ** 2))
        )
        feature_rows: list[dict[str, float | str | None]] = []
        valid_indices = np.flatnonzero(valid)
        for feature_idx, feature_name in enumerate(input_names):
            x_perm = np.array(x_eval, copy=True)
            x_perm_raw = np.array(x_eval_raw, copy=True) if x_eval_raw is not None else None
            # Shuffle only within the target's scored domain. For photosynthesis
            # targets, invalid rows are mostly night/no-canopy/drought-zero rows;
            # mixing those into daytime valid samples exaggerates seasonal/state
            # proxies such as t10.
            perm_idx = rng.permutation(valid_indices)
            x_perm[valid_indices, feature_idx] = x_perm[perm_idx, feature_idx]
            if x_perm_raw is not None:
                x_perm_raw[valid_indices, feature_idx] = x_perm_raw[perm_idx, feature_idx]
            perm_pred = _predict_numpy(
                model,
                x_perm,
                device,
                y_mean,
                y_std,
                target_names,
                x_raw=x_perm_raw,
                input_names=input_names,
                target_feature_masks=target_feature_masks,
                log1p_target_names=log1p_target_names,
            )
            perm_target_pred = np.clip(perm_pred[:, target_idx], pred_min, pred_max)
            perm_rmse = float(
                np.sqrt(np.mean((perm_target_pred[valid] - y_eval[valid, target_idx]) ** 2))
            )
            feature_rows.append(
                {
                    "name": feature_name,
                    "baseline_rmse": baseline_rmse,
                    "permuted_rmse": perm_rmse,
                    "rmse_increase": perm_rmse - baseline_rmse,
                    "prediction_clip_min": pred_min,
                    "prediction_clip_max": pred_max,
                }
            )

        feature_rows.sort(key=lambda row: float(row["rmse_increase"]), reverse=True)
        importance[target_name] = feature_rows

    return importance


def write_feature_sensitivity_csv(
    path: Path,
    target_name: str,
    rows: list[dict[str, float | str | None]],
) -> None:
    import csv

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "target_name",
                "feature_name",
                "baseline_rmse",
                "permuted_rmse",
                "rmse_increase",
                "prediction_clip_min",
                "prediction_clip_max",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    target_name,
                    row["name"],
                    row["baseline_rmse"],
                    row["permuted_rmse"],
                    row["rmse_increase"],
                    row.get("prediction_clip_min"),
                    row.get("prediction_clip_max"),
                ]
            )


def write_feature_sensitivity_plot(
    path: Path,
    target_name: str,
    rows: list[dict[str, float | str | None]],
    top_k: int,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    if not rows:
        return

    top_rows = rows[: max(1, min(top_k, len(rows)))]
    labels = [str(row["name"]) for row in top_rows][::-1]
    deltas = [float(row["rmse_increase"]) for row in top_rows][::-1]
    baseline_rmse = float(top_rows[0]["baseline_rmse"])

    fig, ax = plt.subplots(figsize=(10, max(4.0, 0.35 * len(top_rows) + 1.5)))
    y = np.arange(len(top_rows))
    ax.barh(y, deltas, color="black", alpha=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Permutation RMSE increase")
    ax.set_ylabel("Feature")
    ax.set_title(f"{target_name} sensitivity  (baseline RMSE = {baseline_rmse:.4g})")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _apply_input_masks_for_targets(
    pred_standard: torch.Tensor,
    xb: torch.Tensor,
    model: "PatchMLP",
    target_feature_masks: dict[int, list[int]],
) -> torch.Tensor:
    """Return predictions where masked-target columns come from a masked-feature forward pass.

    For each (target_idx, feature_indices) entry, a second forward pass is run with those
    input features zeroed (mean-scaled = 0). The returned tensor has the same shape as
    `pred_standard` but with masked-target columns replaced by the masked-forward output.
    Only one extra forward pass is made per unique feature-mask signature.
    """
    if not target_feature_masks:
        return pred_standard

    # Group targets by their feature-mask signature to minimise extra forward passes.
    mask_key_to_targets: dict[tuple[int, ...], list[int]] = {}
    for t_idx, f_indices in target_feature_masks.items():
        key = tuple(sorted(f_indices))
        mask_key_to_targets.setdefault(key, []).append(t_idx)

    result = pred_standard
    for f_indices_tuple, t_indices in mask_key_to_targets.items():
        xb_masked = xb.clone()
        for f_idx in f_indices_tuple:
            xb_masked[:, f_idx] = 0.0
        pred_masked = model(xb_masked)
        # Replace only the target columns that belong to this mask group.
        cols = torch.tensor(t_indices, device=xb.device)
        result = result.clone()
        result[:, cols] = pred_masked[:, cols]
    return result


def _iter_tensor_minibatches(
    tensors: tuple[torch.Tensor, ...],
    batch_size: int,
    *,
    shuffle: bool,
) -> Iterable[tuple[torch.Tensor, ...]]:
    """Yield mini-batches directly from in-memory tensors.

    PyTorch's DataLoader is excellent for CPU datasets that need worker-side
    loading, but for tensors already resident on CUDA it still walks samples
    through Python before collating each batch. Direct tensor indexing keeps
    batching on the selected device and avoids that per-sample overhead.
    """
    if not tensors:
        return
    sample_count = int(tensors[0].shape[0])
    if sample_count == 0:
        return
    if any(int(tensor.shape[0]) != sample_count for tensor in tensors):
        raise ValueError("All mini-batch tensors must have the same leading dimension.")

    batch_size = max(1, min(int(batch_size), sample_count))
    if shuffle:
        order = torch.randperm(sample_count, device=tensors[0].device)
        for start in range(0, sample_count, batch_size):
            idx = order.narrow(0, start, min(batch_size, sample_count - start))
            yield tuple(tensor.index_select(0, idx) for tensor in tensors)
    else:
        for start in range(0, sample_count, batch_size):
            end = min(start + batch_size, sample_count)
            yield tuple(tensor[start:end] for tensor in tensors)


def _predict_scaled_tensor(
    model: "PatchMLP",
    x_tensor: torch.Tensor,
    target_feature_masks: dict[int, list[int]],
    batch_size: int,
) -> np.ndarray:
    pred_chunks: list[torch.Tensor] = []
    with torch.no_grad():
        for (xb,) in _iter_tensor_minibatches((x_tensor,), batch_size, shuffle=False):
            pred_std = model(xb)
            pred = _apply_input_masks_for_targets(pred_std, xb, model, target_feature_masks)
            pred_chunks.append(pred.detach().cpu())
    if not pred_chunks:
        return np.empty((0, 0), dtype=np.float32)
    return torch.cat(pred_chunks, dim=0).numpy()


def train_patch_model(
    patch_data: PatchDataset,
    target_names: list[str],
    input_names: list[str],
    args: argparse.Namespace,
    device: torch.device,
    target_feature_masks: dict[int, list[int]] | None = None,
    log_dir: Path | None = None,
) -> dict[str, object]:
    sample_count = patch_data.features.shape[0]
    holdout_gridcells = list(args.holdout_gridcell_index or [])
    holdout_idx = np.empty(0, dtype=np.int64)
    split_pool_idx = np.arange(sample_count, dtype=np.int64)
    if holdout_gridcells and patch_data.gridcell_indices is not None:
        holdout_mask = np.isin(patch_data.gridcell_indices, holdout_gridcells)
        holdout_idx = np.flatnonzero(holdout_mask)
        if holdout_idx.size > 0:
            split_pool_idx = np.flatnonzero(~holdout_mask)
            if split_pool_idx.size < 2:
                raise ValueError(
                    "Held-out gridcells include too many samples; fewer than two samples "
                    "remain for the train/validation split."
                )
            print(
                f"  [split] patch {patch_data.patch_index}: excluding "
                f"{holdout_idx.size} spatial-holdout samples from gridcells {holdout_gridcells}"
            )
    elif holdout_gridcells:
        print(
            f"  [split] patch {patch_data.patch_index}: cannot hold out gridcells "
            "because gridcell_index is unavailable."
        )
    pool_train_idx, pool_val_idx = split_train_validation(
        split_pool_idx.size,
        args.val_fraction,
        args.random_seed,
    )
    train_idx = np.sort(split_pool_idx[pool_train_idx])
    val_idx = np.sort(split_pool_idx[pool_val_idx])

    input_names = list(input_names)
    x_train = patch_data.features[train_idx, :]
    x_val = patch_data.features[val_idx, :]
    x_holdout = patch_data.features[holdout_idx, :] if holdout_idx.size > 0 else None
    y_train = patch_data.targets[train_idx, :]
    y_val = patch_data.targets[val_idx, :]
    y_holdout = patch_data.targets[holdout_idx, :] if holdout_idx.size > 0 else None
    y_train_valid = patch_data.target_valid_mask[train_idx, :]
    y_val_valid = patch_data.target_valid_mask[val_idx, :]
    y_holdout_valid = patch_data.target_valid_mask[holdout_idx, :] if holdout_idx.size > 0 else None

    keep_constant_input_names = {"pft_type"} if args.pft_embedding_dim > 0 else None
    x_train, input_names, dropped_constant_input_names, keep_feature_indices = _drop_float32_constant_features(
        x_train,
        input_names,
        keep_names=keep_constant_input_names,
    )
    if dropped_constant_input_names:
        x_val = x_val[:, keep_feature_indices]
        if x_holdout is not None:
            x_holdout = x_holdout[:, keep_feature_indices]
        print(
            f"  [features] patch {patch_data.patch_index}: dropping float32-constant inputs "
            f"{dropped_constant_input_names}"
        )

    x_train_scaled, x_val_scaled, x_mean, x_std = standardize_train_validation(x_train, x_val)
    x_holdout_scaled = (x_holdout - x_mean) / x_std if x_holdout is not None else None
    y_train_scaled, y_val_scaled, y_mean, y_std = standardize_targets_with_mask(
        y_train, y_train_valid, y_val, target_names, args.log1p_targets
    )
    y_train_filled = np.where(y_train_valid, y_train_scaled, 0.0)
    y_val_filled = np.where(y_val_valid, y_val_scaled, 0.0)
    train_sample_weights = compute_sample_weights(
        y_train,
        y_train_valid,
        target_names,
        args.qflx_tran_weight,
        args.psnsun_weight,
        args.qflx_prec_weight,
        args.fpsn_weight,
        args.hs_soil_weight,
        args.hs_h2osfc_weight,
        args.dhsdT_weight,
    )
    val_sample_weights = compute_sample_weights(
        y_val,
        y_val_valid,
        target_names,
        args.qflx_tran_weight,
        args.psnsun_weight,
        args.qflx_prec_weight,
        args.fpsn_weight,
        args.hs_soil_weight,
        args.hs_h2osfc_weight,
        args.dhsdT_weight,
    )
    train_target_loss_weights = compute_target_loss_weights(
        y_train,
        y_train_valid,
        target_names,
        args.psn_low_value_weight,
        args.psn_low_value_threshold,
    )
    val_target_loss_weights = compute_target_loss_weights(
        y_val,
        y_val_valid,
        target_names,
        args.psn_low_value_weight,
        args.psn_low_value_threshold,
    )

    x_train_tensor = torch.tensor(x_train_scaled, dtype=torch.float32, device=device)
    y_train_tensor = torch.tensor(y_train_filled, dtype=torch.float32, device=device)
    y_train_mask_tensor = torch.tensor(y_train_valid.astype(np.float32), dtype=torch.float32, device=device)
    train_weight_tensor = torch.tensor(train_sample_weights.astype(np.float32), dtype=torch.float32, device=device)
    train_target_weight_tensor = torch.tensor(train_target_loss_weights.astype(np.float32), dtype=torch.float32, device=device)
    x_val_tensor = torch.tensor(x_val_scaled, dtype=torch.float32, device=device)
    y_val_tensor = torch.tensor(y_val_filled, dtype=torch.float32, device=device)
    y_val_mask_tensor = torch.tensor(y_val_valid.astype(np.float32), dtype=torch.float32, device=device)
    val_weight_tensor = torch.tensor(val_sample_weights.astype(np.float32), dtype=torch.float32, device=device)
    val_target_weight_tensor = torch.tensor(val_target_loss_weights.astype(np.float32), dtype=torch.float32, device=device)
    x_holdout_tensor = (
        torch.tensor(x_holdout_scaled, dtype=torch.float32, device=device)
        if x_holdout_scaled is not None
        else None
    )

    pft_embedding_categories: tuple[int, ...] = ()
    pft_embedding_input_index = None
    pft_embedding_x_mean = 0.0
    pft_embedding_x_std = 1.0
    if args.pft_embedding_dim > 0:
        if "pft_type" not in input_names:
            raise ValueError(
                "PFT embeddings require pft_type in the input feature list. "
                "This should be automatic when --pft-embedding-dim is positive."
            )
        pft_embedding_input_index = input_names.index("pft_type")
        finite_pft = np.asarray(x_train[:, pft_embedding_input_index], dtype=np.float64)
        finite_pft = finite_pft[np.isfinite(finite_pft)]
        pft_embedding_categories = tuple(
            sorted({int(round(value)) for value in finite_pft})
        )
        if not pft_embedding_categories:
            raise ValueError("Cannot train PFT embeddings because no finite pft_type values are available.")
        pft_embedding_x_mean = float(x_mean[pft_embedding_input_index])
        pft_embedding_x_std = float(x_std[pft_embedding_input_index])
        print(
            f"  [features] patch {patch_data.patch_index}: using pft_type embedding "
            f"dim={args.pft_embedding_dim} for categories {list(pft_embedding_categories)}"
        )

    model = PatchMLP(
        input_dim=x_train.shape[1],
        output_dim=y_train.shape[1],
        hidden_layers=tuple(args.hidden_layers),
        pft_embedding_dim=args.pft_embedding_dim,
        pft_input_index=pft_embedding_input_index,
        pft_embedding_categories=pft_embedding_categories,
        pft_x_mean=pft_embedding_x_mean,
        pft_x_std=pft_embedding_x_std,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    # Re-resolve masks after per-patch feature dropping. A mask feature that was
    # dropped is already unavailable to the model, so parse_target_feature_masks
    # will warn and skip it.
    del target_feature_masks
    _masks = build_target_feature_masks(
        args,
        input_names=input_names,
        target_names=target_names,
    )

    best_state = None
    best_val_loss = float("inf")
    epochs_since_improvement = 0
    history_train_loss: list[float] = []
    history_val_loss: list[float] = []

    for _epoch in range(args.max_epochs):
        model.train()
        batch_loss_acc = torch.zeros(1, device=device)
        batch_count = 0
        for xb, yb, ymask, sample_weight, target_weight in _iter_tensor_minibatches(
            (x_train_tensor, y_train_tensor, y_train_mask_tensor, train_weight_tensor, train_target_weight_tensor),
            args.batch_size,
            shuffle=True,
        ):
            sample_weight = sample_weight.unsqueeze(1)
            optimizer.zero_grad(set_to_none=True)
            pred_std = model(xb)
            # For masked targets, substitute predictions from a forward pass where
            # the blocked features are zeroed (scaled mean = 0).
            pred = _apply_input_masks_for_targets(pred_std, xb, model, _masks)
            sqerr = (pred - yb) ** 2
            weighted_mask = ymask * sample_weight * target_weight
            loss = (sqerr * weighted_mask).sum() / weighted_mask.sum().clamp_min(1.0)
            loss.backward()
            optimizer.step()
            batch_loss_acc = batch_loss_acc + loss.detach()
            batch_count += 1

        mean_train_loss = float(batch_loss_acc.cpu()) / max(batch_count, 1)
        history_train_loss.append(mean_train_loss)

        model.eval()
        with torch.no_grad():
            val_pred_std = model(x_val_tensor)
            val_pred = _apply_input_masks_for_targets(val_pred_std, x_val_tensor, model, _masks)
            val_sqerr = (val_pred - y_val_tensor) ** 2
            val_weighted_mask = y_val_mask_tensor * val_weight_tensor.unsqueeze(1) * val_target_weight_tensor
            val_loss = float(((val_sqerr * val_weighted_mask).sum() / val_weighted_mask.sum().clamp_min(1.0)).detach().cpu())
            if log_dir is not None and args.log_interval > 0 and (_epoch + 1) % args.log_interval == 0:
                per_target_mse = (val_sqerr * y_val_mask_tensor).sum(0) / y_val_mask_tensor.sum(0).clamp_min(1.0)
                log_row = {"epoch": _epoch + 1, "val_loss": val_loss,
                           "per_target_mse": dict(zip(target_names, per_target_mse.cpu().tolist()))}
                with (log_dir / "training_log.jsonl").open("a", encoding="utf-8") as _lf:
                    _lf.write(json.dumps(log_row) + "\n")
        history_val_loss.append(val_loss)

        best_marker = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            epochs_since_improvement = 0
            best_marker = " *"
        else:
            epochs_since_improvement += 1
            if epochs_since_improvement >= args.patience:
                break

        print(
            f"  epoch {_epoch + 1:4d}  train={mean_train_loss:.4f}  val={val_loss:.4f}"
            f"  patience={epochs_since_improvement}/{args.patience}{best_marker}",
            flush=True,
        )

    if best_state is None:
        raise RuntimeError("Training failed to produce a valid model state.")

    model.load_state_dict(best_state)
    model.reset_unknown_pft_embedding_to_mean()
    model.eval()
    y_train_pred_scaled = _predict_scaled_tensor(model, x_train_tensor, _masks, args.batch_size)
    y_val_pred_scaled = _predict_scaled_tensor(model, x_val_tensor, _masks, args.batch_size)
    y_holdout_pred_scaled = (
        _predict_scaled_tensor(model, x_holdout_tensor, _masks, args.batch_size)
        if x_holdout_tensor is not None
        else None
    )

    y_train_pred = y_train_pred_scaled * y_std + y_mean
    y_val_pred = y_val_pred_scaled * y_std + y_mean
    y_holdout_pred = (
        y_holdout_pred_scaled * y_std + y_mean
        if y_holdout_pred_scaled is not None
        else None
    )
    y_train_pred = postprocess_predictions(
        y_train_pred,
        target_names,
        x_raw=x_train,
        input_names=input_names,
        log1p_target_names=args.log1p_targets,
    )
    y_val_pred = postprocess_predictions(
        y_val_pred,
        target_names,
        x_raw=x_val,
        input_names=input_names,
        log1p_target_names=args.log1p_targets,
    )
    if y_holdout_pred is not None:
        y_holdout_pred = postprocess_predictions(
            y_holdout_pred,
            target_names,
            x_raw=x_holdout,
            input_names=input_names,
            log1p_target_names=args.log1p_targets,
        )
    y_all_pred = np.empty_like(patch_data.targets, dtype=np.float64)
    y_all_pred[train_idx, :] = y_train_pred
    y_all_pred[val_idx, :] = y_val_pred
    if y_holdout_pred is not None:
        y_all_pred[holdout_idx, :] = y_holdout_pred
    sample_split = np.full(sample_count, "train", dtype=object)
    sample_split[val_idx] = "validation"
    if holdout_idx.size > 0:
        sample_split[holdout_idx] = "holdout"

    train_metrics = compute_metrics(y_train, y_train_pred, y_train_valid, target_names)
    val_metrics = compute_metrics(y_val, y_val_pred, y_val_valid, target_names)
    holdout_metrics = (
        compute_metrics(y_holdout, y_holdout_pred, y_holdout_valid, target_names)
        if y_holdout is not None
        and y_holdout_pred is not None
        and y_holdout_valid is not None
        else None
    )
    val_metrics["target_names"] = target_names
    if holdout_metrics is not None:
        holdout_metrics["target_names"] = target_names
    val_metrics["feature_importance_by_target"] = estimate_permutation_importance(
        model=model,
        x_val_scaled=x_val_scaled,
        y_val=y_val,
        y_val_valid=y_val_valid,
        input_names=input_names,
        target_names=target_names,
        y_mean=y_mean,
        y_std=y_std,
        device=device,
        requested_targets=args.importance_targets,
        max_samples=args.importance_max_samples,
        random_seed=args.random_seed,
        x_val_raw=x_val,
        target_feature_masks=_masks,
        log1p_target_names=args.log1p_targets,
    )
    sensitivity_targets = tuple(args.sensitivity_targets)
    val_metrics["feature_sensitivity_by_target"] = estimate_permutation_importance(
        model=model,
        x_val_scaled=x_val_scaled,
        x_val_raw=x_val,
        y_val=y_val,
        y_val_valid=y_val_valid,
        input_names=input_names,
        target_names=target_names,
        y_mean=y_mean,
        y_std=y_std,
        device=device,
        requested_targets=sensitivity_targets,
        max_samples=args.sensitivity_max_samples,
        random_seed=args.random_seed,
        target_feature_masks=_masks,
        log1p_target_names=args.log1p_targets,
    )

    # Convert index-keyed masks back to name-keyed for portable storage.
    target_feature_masks_by_name: dict[str, list[str]] = {
        target_names[t_idx]: [input_names[f_idx] for f_idx in f_indices]
        for t_idx, f_indices in _masks.items()
    }

    def _sample_range(indices: np.ndarray) -> dict[str, object]:
        return {
            "nstep_first": int(patch_data.nsteps[indices[0]]),
            "nstep_last": int(patch_data.nsteps[indices[-1]]),
            "date_first": [
                int(patch_data.years[indices[0]]),
                int(patch_data.months[indices[0]]),
                int(patch_data.days[indices[0]]),
                int(patch_data.secs[indices[0]]),
            ],
            "date_last": [
                int(patch_data.years[indices[-1]]),
                int(patch_data.months[indices[-1]]),
                int(patch_data.days[indices[-1]]),
                int(patch_data.secs[indices[-1]]),
            ],
        }

    return {
        "state_dict": {key: value.detach().cpu().numpy() for key, value in model.state_dict().items()},
        "hidden_layers": list(args.hidden_layers),
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
        "input_names": input_names,
        "target_names": target_names,
        "target_feature_masks": target_feature_masks_by_name,
        "log1p_target_names": list(args.log1p_targets),
        "pft_embedding_dim": int(args.pft_embedding_dim),
        "pft_embedding_input_name": "pft_type" if pft_embedding_input_index is not None else None,
        "pft_embedding_input_index": pft_embedding_input_index,
        "pft_embedding_categories": list(pft_embedding_categories),
        "pft_embedding_unknown_index": (
            len(pft_embedding_categories) if pft_embedding_input_index is not None else None
        ),
        "pft_embedding_x_mean": pft_embedding_x_mean,
        "pft_embedding_x_std": pft_embedding_x_std,
        "surfdata_file": str(args.surfdata_file) if args.surfdata_file is not None else None,
        "surfdata_cell_stride": args.surfdata_cell_stride,
        "dropped_constant_input_names": dropped_constant_input_names,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "holdout_metrics": holdout_metrics,
        "holdout_gridcell_index": holdout_gridcells,
        "holdout_sample_count": int(holdout_idx.size),
        "y_val_true": y_val,
        "y_val_pred": y_val_pred,
        "y_all_true": patch_data.targets,
        "y_all_pred": y_all_pred,
        "sample_split": sample_split,
        "plot_nsteps": patch_data.nsteps,
        "plot_gridcell_index": patch_data.gridcell_indices,
        "state_target_lag_steps": int(args.state_target_lag_steps),
        "state_target_names": list(args.state_target_names),
        "training_history": {
            "train_loss": history_train_loss,
            "val_loss": history_val_loss,
            "best_val_loss": best_val_loss,
        },
        "train_range": _sample_range(train_idx),
        "val_range": _sample_range(val_idx),
        "holdout_range": _sample_range(holdout_idx) if holdout_idx.size > 0 else None,
    }


def write_validation_plots(
    patch_dir: Path,
    target_names: list[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    requested_targets: tuple[str, ...] | list[str] | None,
    *,
    all_y_true: np.ndarray | None = None,
    all_y_pred: np.ndarray | None = None,
    sample_split: np.ndarray | None = None,
    gridcell_index: np.ndarray | None = None,
    nsteps: np.ndarray | None = None,
    selected_gridcell_indices: tuple[int, ...] | list[int] | None = None,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    if requested_targets is None:
        plot_indices = list(range(len(target_names)))
    else:
        plot_indices = [target_names.index(name) for name in requested_targets if name in target_names]
        for name in ("laisun_patch", "laisha_patch", "psnsun_patch", "psnsha_patch"):
            if name in target_names:
                idx = target_names.index(name)
                if idx not in plot_indices:
                    plot_indices.append(idx)
    if not plot_indices:
        return

    nplots = len(plot_indices)
    fig_ts, axes_ts = plt.subplots(nplots, 1, figsize=(10, 3 * nplots), squeeze=False)
    fig_sc, axes_sc = plt.subplots(nplots, 1, figsize=(6, 3.5 * nplots), squeeze=False)
    subset_count = min(1000, y_true.shape[0])
    fig_ts_subset, axes_ts_subset = plt.subplots(nplots, 1, figsize=(10, 3 * nplots), squeeze=False)

    for ax_row, target_idx in enumerate(plot_indices):
        name = target_names[target_idx]
        truth = y_true[:, target_idx]
        pred = y_pred[:, target_idx]

        ax = axes_ts[ax_row, 0]
        ax.plot(truth, label="truth", linewidth=1.0)
        ax.plot(pred, label="pred", linewidth=1.0)
        ax.set_title(name)
        ax.set_xlabel("Validation sample")
        ax.set_ylabel("Value")
        ax.legend(loc="best")

        ax = axes_ts_subset[ax_row, 0]
        ax.plot(truth[:subset_count], label="truth", linewidth=1.0)
        ax.plot(pred[:subset_count], label="pred", linewidth=1.0)
        ax.set_title(f"{name} (First {subset_count})")
        ax.set_xlabel("Validation sample")
        ax.set_ylabel("Value")
        ax.legend(loc="best")

        ax = axes_sc[ax_row, 0]
        ax.scatter(truth, pred, s=6, alpha=0.4)
        data_min = float(min(np.min(truth), np.min(pred)))
        data_max = float(max(np.max(truth), np.max(pred)))
        ax.plot([data_min, data_max], [data_min, data_max], "k--", linewidth=1.0)
        ax.set_title(name)
        ax.set_xlabel("Truth")
        ax.set_ylabel("Prediction")

    fig_ts.tight_layout()
    fig_ts_subset.tight_layout()
    fig_sc.tight_layout()
    fig_ts.savefig(patch_dir / "validation_timeseries.png", dpi=150)
    fig_ts_subset.savefig(patch_dir / "validation_timeseries_first1000.png", dpi=150)
    fig_sc.savefig(patch_dir / "validation_scatter.png", dpi=150)
    plt.close(fig_ts)
    plt.close(fig_ts_subset)
    plt.close(fig_sc)

    if (
        all_y_true is None
        or all_y_pred is None
        or sample_split is None
        or gridcell_index is None
        or not selected_gridcell_indices
    ):
        return

    x_all = np.asarray(nsteps) if nsteps is not None else np.arange(all_y_true.shape[0])
    gridcell_index = np.asarray(gridcell_index)
    sample_split = np.asarray(sample_split)
    for gridcell_id in selected_gridcell_indices:
        cell_mask = gridcell_index == int(gridcell_id)
        if not np.any(cell_mask):
            print(f"  [plot] gridcell {gridcell_id} is not present in this patch; skipping.")
            continue
        order = np.argsort(x_all[cell_mask], kind="stable")
        x_cell = x_all[cell_mask][order]
        true_cell = all_y_true[cell_mask, :][order, :]
        pred_cell = all_y_pred[cell_mask, :][order, :]
        split_cell = sample_split[cell_mask][order]
        train_mask = split_cell == "train"
        val_mask = split_cell == "validation"
        holdout_mask = split_cell == "holdout"

        fig_cell, axes_cell = plt.subplots(nplots, 1, figsize=(11, 3 * nplots), squeeze=False)
        for ax_row, target_idx in enumerate(plot_indices):
            name = target_names[target_idx]
            truth = true_cell[:, target_idx]
            pred = pred_cell[:, target_idx]
            ax = axes_cell[ax_row, 0]
            ax.plot(x_cell, truth, color="black", linewidth=1.0, label="truth")
            if np.any(train_mask):
                ax.scatter(
                    x_cell[train_mask],
                    pred[train_mask],
                    s=8,
                    alpha=0.45,
                    label="pred train",
                )
            if np.any(val_mask):
                ax.scatter(
                    x_cell[val_mask],
                    pred[val_mask],
                    s=10,
                    alpha=0.75,
                    label="pred validation",
                )
            if np.any(holdout_mask):
                ax.scatter(
                    x_cell[holdout_mask],
                    pred[holdout_mask],
                    s=12,
                    alpha=0.85,
                    label="pred holdout",
                )
            ax.set_title(f"{name}  gridcell {gridcell_id}")
            ax.set_xlabel("nstep" if nsteps is not None else "Sample")
            ax.set_ylabel("Value")
            ax.legend(loc="best")
        fig_cell.tight_layout()
        fig_cell.savefig(patch_dir / f"validation_timeseries_gridcell_{int(gridcell_id)}.png", dpi=150)
        plt.close(fig_cell)


def write_feature_sensitivity_artifacts(
    patch_dir: Path,
    val_metrics: dict[str, object],
    requested_targets: tuple[str, ...] | list[str] | None,
    top_k: int,
) -> None:
    sensitivity = val_metrics.get("feature_sensitivity_by_target", {})
    if not isinstance(sensitivity, dict) or not sensitivity:
        return

    if requested_targets is None:
        target_names = list(sensitivity.keys())
    else:
        target_names = [name for name in requested_targets if name in sensitivity]

    for target_name in target_names:
        rows = sensitivity.get(target_name, [])
        if not rows:
            continue
        csv_path = patch_dir / f"feature_sensitivity_{target_name}.csv"
        png_path = patch_dir / f"feature_sensitivity_{target_name}.png"
        write_feature_sensitivity_csv(csv_path, target_name, rows)
        write_feature_sensitivity_plot(png_path, target_name, rows, top_k=top_k)


def write_patch_artifacts(
    output_dir: Path, patch_id: int, results: dict[str, object], sample_count: int
) -> None:
    patch_dir = output_dir / f"patch_{patch_id}"
    patch_dir.mkdir(parents=True, exist_ok=True)

    model_bundle = {
        "state_dict": results["state_dict"],
        "hidden_layers": results["hidden_layers"],
        "x_mean": results["x_mean"],
        "x_std": results["x_std"],
        "y_mean": results["y_mean"],
        "y_std": results["y_std"],
        "input_names": results["input_names"],
        "target_names": results["target_names"],
        "target_feature_masks": results.get("target_feature_masks", {}),
        "log1p_target_names": results.get("log1p_target_names", list(DEFAULT_LOG1P_TARGET_NAMES)),
        "pft_embedding_dim": results.get("pft_embedding_dim", 0),
        "pft_embedding_input_name": results.get("pft_embedding_input_name"),
        "pft_embedding_input_index": results.get("pft_embedding_input_index"),
        "pft_embedding_categories": results.get("pft_embedding_categories", []),
        "pft_embedding_unknown_index": results.get("pft_embedding_unknown_index"),
        "pft_embedding_x_mean": results.get("pft_embedding_x_mean", 0.0),
        "pft_embedding_x_std": results.get("pft_embedding_x_std", 1.0),
        "surfdata_file": results.get("surfdata_file"),
        "surfdata_cell_stride": results.get("surfdata_cell_stride"),
        "dropped_target_names": results.get("dropped_target_names", []),
        "patch_omitted_target_names": results.get("patch_omitted_target_names", []),
        "dropped_input_names": results.get("dropped_input_names", []),
        "dropped_constant_input_names": results.get("dropped_constant_input_names", []),
        "state_target_lag_steps": results.get("state_target_lag_steps", 0),
        "state_target_names": results.get("state_target_names", []),
        "patch_index": patch_id,
    }
    torch.save(model_bundle, patch_dir / "model.pt")

    metrics = {
        "patch_index": patch_id,
        "sample_count": sample_count,
        "dropped_nonfinite": int(results["dropped_nonfinite"]),
        "bad_feature_columns": results["bad_feature_columns"],
        "bad_target_columns": results["bad_target_columns"],
        "train_metrics": results["train_metrics"],
        "val_metrics": results["val_metrics"],
        "holdout_metrics": results.get("holdout_metrics"),
        "holdout_gridcell_index": results.get("holdout_gridcell_index", []),
        "holdout_sample_count": results.get("holdout_sample_count", 0),
        "train_range": results["train_range"],
        "val_range": results["val_range"],
        "holdout_range": results.get("holdout_range"),
        "training_history": results["training_history"],
        "log1p_target_names": results.get("log1p_target_names", list(DEFAULT_LOG1P_TARGET_NAMES)),
        "surfdata_file": results.get("surfdata_file"),
        "surfdata_cell_stride": results.get("surfdata_cell_stride"),
        "dropped_target_names": results.get("dropped_target_names", []),
        "patch_omitted_target_names": results.get("patch_omitted_target_names", []),
        "dropped_input_names": results.get("dropped_input_names", []),
        "dropped_constant_input_names": results.get("dropped_constant_input_names", []),
        "state_target_lag_steps": results.get("state_target_lag_steps", 0),
        "state_target_names": results.get("state_target_names", []),
    }
    with (patch_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)


_print_lock = threading.Lock()


def _target_subset_for_patch(
    patch_index: int,
    target_names: list[str],
) -> tuple[list[str], list[int], list[str]]:
    if int(patch_index) != 0:
        return list(target_names), list(range(len(target_names))), []
    keep_indices = [
        idx for idx, name in enumerate(target_names)
        if name not in PATCH0_OMITTED_TARGET_NAMES
    ]
    omitted = [
        name for name in target_names
        if name in PATCH0_OMITTED_TARGET_NAMES
    ]
    return [target_names[idx] for idx in keep_indices], keep_indices, omitted


def _subset_patch_targets(
    patch_data: PatchDataset,
    keep_indices: list[int],
) -> PatchDataset:
    if len(keep_indices) == patch_data.targets.shape[1]:
        return patch_data
    keep = np.asarray(keep_indices, dtype=np.intp)
    return replace(
        patch_data,
        targets=patch_data.targets[:, keep],
        target_valid_mask=patch_data.target_valid_mask[:, keep],
        target_nonfinite_counts=(
            patch_data.target_nonfinite_counts[keep]
            if patch_data.target_nonfinite_counts is not None
            else None
        ),
        target_large_counts=(
            patch_data.target_large_counts[keep]
            if patch_data.target_large_counts is not None
            else None
        ),
    )


def _collect_patch_diagnostics(
    patch_data: PatchDataset,
    names: dict[str, list[str]],
    args: argparse.Namespace,
) -> tuple[int, int, list[dict], list[dict], dict[str, object] | None]:
    """Return (sample_count, raw_sample_count, bad_feature_columns, bad_target_columns, skip_entry)."""
    sample_count = patch_data.features.shape[0]
    raw_sample_count = sample_count + patch_data.dropped_nonfinite
    bad_feature_columns = []
    bad_target_columns = []
    if patch_data.feature_nonfinite_counts is not None:
        bad_feature_columns = [
            {
                "name": names["input_names"][idx],
                "nonfinite_count": int(count),
                "large_count": int(patch_data.feature_large_counts[idx]),
                "all_bad": int(count + patch_data.feature_large_counts[idx]) == raw_sample_count,
            }
            for idx, count in enumerate(patch_data.feature_nonfinite_counts)
            if count > 0 or int(patch_data.feature_large_counts[idx]) > 0
        ]
    if patch_data.target_nonfinite_counts is not None:
        bad_target_columns = [
            {
                "name": names["target_names"][idx],
                "nonfinite_count": int(count),
                "large_count": int(patch_data.target_large_counts[idx]),
                "all_bad": int(count + patch_data.target_large_counts[idx]) == raw_sample_count,
            }
            for idx, count in enumerate(patch_data.target_nonfinite_counts)
            if count > 0 or int(patch_data.target_large_counts[idx]) > 0
        ]
    skip_entry = None
    if sample_count < args.min_samples:
        skip_entry = {
            "patch_index": patch_data.patch_index,
            "sample_count": sample_count,
            "raw_sample_count": raw_sample_count,
            "dropped_nonfinite": patch_data.dropped_nonfinite,
            "bad_feature_columns": bad_feature_columns,
            "bad_target_columns": bad_target_columns,
            "status": "skipped",
            "reason": f"fewer than min_samples={args.min_samples}",
            "dropped_target_names": names["dropped_target_names"],
            "dropped_input_names": names.get("dropped_input_names", []),
            "state_target_lag_steps": args.state_target_lag_steps,
            "state_target_names": args.state_target_names,
        }
    return sample_count, raw_sample_count, bad_feature_columns, bad_target_columns, skip_entry


def _train_and_write_patch(
    patch_data: PatchDataset,
    names: dict[str, list[str]],
    args: argparse.Namespace,
    device: torch.device,
    sample_count: int,
    raw_sample_count: int,
    bad_feature_columns: list[dict],
    bad_target_columns: list[dict],
) -> dict[str, object]:
    target_names, keep_target_indices, patch_omitted_target_names = _target_subset_for_patch(
        patch_data.patch_index,
        names["target_names"],
    )
    if patch_omitted_target_names:
        patch_data = _subset_patch_targets(patch_data, keep_target_indices)
        bad_target_columns = [
            entry
            for entry in bad_target_columns
            if entry.get("name") not in patch_omitted_target_names
        ]
    dropped_target_names = list(names["dropped_target_names"]) + list(patch_omitted_target_names)
    patch_dir = args.output_dir / f"patch_{patch_data.patch_index}"
    patch_dir.mkdir(parents=True, exist_ok=True)
    with _print_lock:
        print(
            f"Training patch {patch_data.patch_index} with {sample_count} filtered samples "
            f"({patch_data.dropped_nonfinite} dropped)"
        )
        if patch_omitted_target_names:
            print(
                "  patch 0: omitting vegetation-only targets from training: "
                + ", ".join(patch_omitted_target_names)
            )
    results = train_patch_model(
        patch_data=patch_data,
        target_names=target_names,
        input_names=names["input_names"],
        args=args,
        device=device,
        log_dir=patch_dir,
    )
    write_patch_artifacts(
        output_dir=args.output_dir,
        patch_id=patch_data.patch_index,
        results={
            **results,
            "dropped_nonfinite": patch_data.dropped_nonfinite,
            "bad_feature_columns": bad_feature_columns,
            "bad_target_columns": bad_target_columns,
            "dropped_target_names": dropped_target_names,
            "patch_omitted_target_names": patch_omitted_target_names,
            "dropped_input_names": names.get("dropped_input_names", []),
        },
        sample_count=sample_count,
    )
    write_validation_plots(
        patch_dir=args.output_dir / f"patch_{patch_data.patch_index}",
        target_names=target_names,
        y_true=results["y_val_true"],
        y_pred=results["y_val_pred"],
        requested_targets=args.plot_targets,
        all_y_true=results.get("y_all_true"),
        all_y_pred=results.get("y_all_pred"),
        sample_split=results.get("sample_split"),
        gridcell_index=results.get("plot_gridcell_index"),
        nsteps=results.get("plot_nsteps"),
        selected_gridcell_indices=args.plot_gridcell_index,
    )
    write_feature_sensitivity_artifacts(
        patch_dir=args.output_dir / f"patch_{patch_data.patch_index}",
        val_metrics=results["val_metrics"],
        requested_targets=args.sensitivity_targets,
        top_k=args.sensitivity_top_k,
    )
    return {
        "patch_index": patch_data.patch_index,
        "sample_count": sample_count,
        "raw_sample_count": raw_sample_count,
        "dropped_nonfinite": patch_data.dropped_nonfinite,
        "bad_feature_columns": bad_feature_columns,
        "bad_target_columns": bad_target_columns,
        "status": "trained",
        "val_rmse_mean": results["val_metrics"]["rmse_mean"],
        "val_r2_mean": results["val_metrics"]["r2_mean"],
        "dropped_target_names": dropped_target_names,
        "patch_omitted_target_names": patch_omitted_target_names,
        "dropped_input_names": names.get("dropped_input_names", []),
        "state_target_lag_steps": args.state_target_lag_steps,
        "state_target_names": args.state_target_names,
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_random_seed(args.random_seed)
    device = select_device(args.device)
    print(f"Using torch device: {device}")
    args.surfdata_file = resolve_surfdata_file(args.training_file, args.surfdata_file)
    if args.pft_embedding_dim < 0:
        raise ValueError(f"--pft-embedding-dim must be nonnegative, got {args.pft_embedding_dim}")

    drop_input_names = list(args.drop_input_names or [])
    parameter_input_names: tuple[str, ...] = ()
    if args.pft_embedding_dim > 0:
        parameter_input_names = OPTIONAL_PARAMETER_INPUT_NAMES
        drop_input_names = [name for name in drop_input_names if name != "pft_type"]
    elif args.include_pft_type_feature:
        parameter_input_names = OPTIONAL_PARAMETER_INPUT_NAMES
        drop_input_names = [name for name in drop_input_names if name != "pft_type"]
    elif "pft_type" not in drop_input_names:
        drop_input_names.append("pft_type")

    selected_patch_indices = set(args.patch_index) if args.patch_index else None
    patch_index, cell_index, features, targets, time_info, names = load_training_data(
        args.training_file,
        args.surfdata_file,
        surfdata_cell_stride=args.surfdata_cell_stride,
        drop_input_names=drop_input_names,
        parameter_input_names=parameter_input_names,
    )
    if args.universal_model:
        original_patch_index = patch_index
        if selected_patch_indices is not None:
            selected = np.array(sorted(selected_patch_indices), dtype=patch_index.dtype)
            keep = np.isin(patch_index, selected)
            if not np.any(keep):
                raise ValueError(
                    "No rows remain after applying --patch-index "
                    f"{sorted(selected_patch_indices)} for --universal-model."
                )
            print(
                f"Training one universal model from patches {sorted(selected_patch_indices)} "
                f"({int(np.count_nonzero(keep))} rows before filtering)."
            )
            patch_index = patch_index[keep]
            features = features[keep, :]
            targets = targets[keep, :]
            if cell_index is not None:
                cell_index = cell_index[keep]
            time_info = {
                key: value[keep] if isinstance(value, np.ndarray) and value.shape[:1] == keep.shape else value
                for key, value in time_info.items()
            }
        else:
            print(
                "Training one universal model from all patches "
                f"({patch_index.size} rows before filtering)."
            )
        print(
            "  [universal] source patches present: "
            + ", ".join(str(int(p)) for p in np.unique(original_patch_index if selected_patch_indices is None else patch_index))
        )
        patch_index = np.full(patch_index.shape, UNIVERSAL_PATCH_INDEX, dtype=np.int64)
        selected_patch_indices = {UNIVERSAL_PATCH_INDEX}

    # Collect all patch work items (fast — just array slicing and masking).
    work_items: list[tuple[PatchDataset, int, int, list, list]] = []
    summary: list[dict[str, object]] = []
    for patch_data in iter_patch_datasets(
        patch_index=patch_index,
        features=features,
        targets=targets,
        time_info=time_info,
        selected_patch_indices=selected_patch_indices,
        spval_threshold=args.spval_threshold,
        cell_index=cell_index,
        input_names=names["input_names"],
        target_names=names["target_names"],
        state_target_lag_steps=args.state_target_lag_steps,
        state_target_names=args.state_target_names,
        nighttime_psn_mask=args.nighttime_psn_mask,
    ):
        sample_count, raw_sample_count, bad_feat, bad_tgt, skip_entry = _collect_patch_diagnostics(
            patch_data, names, args
        )
        if skip_entry is not None:
            summary.append(skip_entry)
            continue
        work_items.append((patch_data, sample_count, raw_sample_count, bad_feat, bad_tgt))

    def _submit(item: tuple) -> dict[str, object]:
        patch_data, sample_count, raw_sample_count, bad_feat, bad_tgt = item
        return _train_and_write_patch(
            patch_data=patch_data,
            names=names,
            args=args,
            device=device,
            sample_count=sample_count,
            raw_sample_count=raw_sample_count,
            bad_feature_columns=bad_feat,
            bad_target_columns=bad_tgt,
        )

    if args.num_workers > 1:
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {executor.submit(_submit, item): item for item in work_items}
            for future in as_completed(futures):
                summary.append(future.result())
    else:
        for item in work_items:
            summary.append(_submit(item))

    summary.sort(key=lambda e: e["patch_index"])
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"Wrote per-patch model artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
