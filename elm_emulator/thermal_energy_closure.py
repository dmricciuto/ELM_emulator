"""Energy-closed upper boundaries for the soil/snow thermal solver.

Native ELM constructs the net heat flux into soil from absorbed radiation,
longwave exchange, ground sensible heat, and ground evaporation.  New emulator
checkpoints predict those primitive terms and use these helpers instead of
learning the composite surface heat fluxes and ``dhsdT`` independently.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - torch is required by coupled replay
    torch = None


STEFAN_BOLTZMANN = 5.670374419e-8
LATENT_HEAT_VAPORIZATION = 2.501e6
SOIL_EMISSIVITY = 0.96

ENERGY_CLOSED_SOIL_TARGET_NAMES = frozenset(
    {
        "sabg_soil",
        "dlrad",
        "eflx_sh_soil",
        "qflx_ev_soil",
        "cgrnds",
        "cgrndl",
    }
)

ENERGY_CLOSED_SURFACE_TARGET_NAMES = ENERGY_CLOSED_SOIL_TARGET_NAMES | frozenset(
    {
        "sabg_snow_top",
        "eflx_sh_snow",
        "eflx_sh_h2osfc",
        "qflx_ev_snow",
        "qflx_ev_h2osfc",
    }
)


def has_energy_closed_soil_targets(targets_or_names: Mapping | list[str] | tuple[str, ...]) -> bool:
    """Return whether all primitive snow-free soil-energy targets are present."""

    names = set(targets_or_names.keys() if isinstance(targets_or_names, Mapping) else targets_or_names)
    return ENERGY_CLOSED_SOIL_TARGET_NAMES.issubset(names)


def has_energy_closed_surface_targets(targets_or_names: Mapping | list[str] | tuple[str, ...]) -> bool:
    """Return whether soil, snow, and surface-water closure primitives exist."""

    names = set(targets_or_names.keys() if isinstance(targets_or_names, Mapping) else targets_or_names)
    return ENERGY_CLOSED_SURFACE_TARGET_NAMES.issubset(names)


def derive_surface_thermal_boundaries_numpy(
    targets: Mapping[str, np.ndarray],
    *,
    forc_lwrad: np.ndarray,
    frac_veg_nosno: np.ndarray,
    t_soil_top: np.ndarray,
    t_snow_top: np.ndarray,
    t_h2osfc: np.ndarray,
    t_grnd: np.ndarray,
    htvp: np.ndarray | float = LATENT_HEAT_VAPORIZATION,
    emg: np.ndarray | float = SOIL_EMISSIVITY,
    enforce_nonnegative_conductance: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Derive ELM's soil, top-snow, and standing-water thermal boundaries."""

    if not has_energy_closed_surface_targets(targets):
        missing = sorted(ENERGY_CLOSED_SURFACE_TARGET_NAMES.difference(targets))
        raise KeyError(f"Missing full surface energy-closure targets: {', '.join(missing)}")
    emg_a = np.asarray(emg)
    htvp_a = np.asarray(htvp)
    common_radiation = np.asarray(targets["dlrad"]) + (
        1.0 - np.asarray(frac_veg_nosno)
    ) * emg_a * np.asarray(forc_lwrad)
    cgrnds = np.asarray(targets["cgrnds"])
    cgrndl = np.asarray(targets["cgrndl"])
    if enforce_nonnegative_conductance:
        cgrnds = np.maximum(cgrnds, 0.0)
        cgrndl = np.maximum(cgrndl, 0.0)

    def _boundary(sabg, temperature, sensible, evaporation):
        emitted = emg_a * STEFAN_BOLTZMANN * np.asarray(temperature) ** 4
        latent = htvp_a * np.asarray(evaporation)
        heat = np.asarray(sabg) + common_radiation - emitted - np.asarray(sensible) - latent
        return heat, emitted, latent

    hs_soil, lw_soil, latent_soil = _boundary(
        targets["sabg_soil"], t_soil_top, targets["eflx_sh_soil"], targets["qflx_ev_soil"]
    )
    hs_top_snow, lw_snow, latent_snow = _boundary(
        targets["sabg_snow_top"], t_snow_top, targets["eflx_sh_snow"], targets["qflx_ev_snow"]
    )
    hs_h2osfc, lw_h2osfc, latent_h2osfc = _boundary(
        targets["sabg_soil"], t_h2osfc, targets["eflx_sh_h2osfc"], targets["qflx_ev_h2osfc"]
    )
    longwave_derivative = 4.0 * emg_a * STEFAN_BOLTZMANN * np.asarray(t_grnd) ** 3
    dhsdT = -(cgrnds + htvp_a * cgrndl + longwave_derivative)
    return hs_soil, hs_top_snow, hs_h2osfc, dhsdT, {
        "common_radiation": common_radiation,
        "longwave_emitted_soil": lw_soil,
        "longwave_emitted_snow": lw_snow,
        "longwave_emitted_h2osfc": lw_h2osfc,
        "latent_soil": latent_soil,
        "latent_snow": latent_snow,
        "latent_h2osfc": latent_h2osfc,
        "longwave_derivative": longwave_derivative,
    }


def derive_soil_thermal_boundary_numpy(
    targets: Mapping[str, np.ndarray],
    *,
    forc_lwrad: np.ndarray,
    frac_veg_nosno: np.ndarray,
    t_soil_top: np.ndarray,
    t_grnd: np.ndarray,
    htvp: np.ndarray | float = LATENT_HEAT_VAPORIZATION,
    emg: np.ndarray | float = SOIL_EMISSIVITY,
    enforce_nonnegative_conductance: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Derive ``hs_soil`` and ``dhsdT`` from primitive NumPy targets.

    Fluxes follow ELM's sign convention: positive ``hs_soil`` is energy into
    the soil.  Ground evaporation is in kg m-2 s-1 (numerically mm s-1).
    """

    if not has_energy_closed_soil_targets(targets):
        missing = sorted(ENERGY_CLOSED_SOIL_TARGET_NAMES.difference(targets))
        raise KeyError(f"Missing energy-closure targets: {', '.join(missing)}")
    sabg_soil = np.asarray(targets["sabg_soil"])
    dlrad = np.asarray(targets["dlrad"])
    sh_soil = np.asarray(targets["eflx_sh_soil"])
    qsoil = np.asarray(targets["qflx_ev_soil"])
    cgrnds = np.asarray(targets["cgrnds"])
    cgrndl = np.asarray(targets["cgrndl"])
    if enforce_nonnegative_conductance:
        cgrnds = np.maximum(cgrnds, 0.0)
        cgrndl = np.maximum(cgrndl, 0.0)
    emg_a = np.asarray(emg)
    htvp_a = np.asarray(htvp)
    t_soil_a = np.asarray(t_soil_top)
    t_grnd_a = np.asarray(t_grnd)
    radiation_in = (
        sabg_soil
        + dlrad
        + (1.0 - np.asarray(frac_veg_nosno)) * emg_a * np.asarray(forc_lwrad)
    )
    longwave_emitted = emg_a * STEFAN_BOLTZMANN * t_soil_a**4
    latent_soil = htvp_a * qsoil
    hs_soil = radiation_in - longwave_emitted - sh_soil - latent_soil
    longwave_derivative = 4.0 * emg_a * STEFAN_BOLTZMANN * t_grnd_a**3
    dhsdT = -(cgrnds + htvp_a * cgrndl + longwave_derivative)
    components = {
        "radiation_in": radiation_in,
        "longwave_emitted": longwave_emitted,
        "sensible_soil": sh_soil,
        "latent_soil": latent_soil,
        "longwave_derivative": longwave_derivative,
    }
    return hs_soil, dhsdT, components


def derive_soil_thermal_boundary_torch(
    targets: Mapping[str, "torch.Tensor"],
    *,
    forc_lwrad: "torch.Tensor",
    frac_veg_nosno: "torch.Tensor",
    t_soil_top: "torch.Tensor",
    t_grnd: "torch.Tensor",
    htvp: "torch.Tensor | float" = LATENT_HEAT_VAPORIZATION,
    emg: "torch.Tensor | float" = SOIL_EMISSIVITY,
    enforce_nonnegative_conductance: bool = True,
) -> tuple["torch.Tensor", "torch.Tensor", dict[str, "torch.Tensor"]]:
    """Torch equivalent of :func:`derive_soil_thermal_boundary_numpy`."""

    if torch is None:  # pragma: no cover
        raise RuntimeError("PyTorch is required for Torch energy closure")
    if not has_energy_closed_soil_targets(targets):
        missing = sorted(ENERGY_CLOSED_SOIL_TARGET_NAMES.difference(targets))
        raise KeyError(f"Missing energy-closure targets: {', '.join(missing)}")
    like = targets["qflx_ev_soil"]

    def _as_tensor(value):
        return value.to(device=like.device, dtype=like.dtype) if isinstance(value, torch.Tensor) else torch.as_tensor(value, device=like.device, dtype=like.dtype)

    sabg_soil = _as_tensor(targets["sabg_soil"])
    dlrad = _as_tensor(targets["dlrad"])
    sh_soil = _as_tensor(targets["eflx_sh_soil"])
    qsoil = _as_tensor(targets["qflx_ev_soil"])
    cgrnds = _as_tensor(targets["cgrnds"])
    cgrndl = _as_tensor(targets["cgrndl"])
    if enforce_nonnegative_conductance:
        cgrnds = torch.clamp(cgrnds, min=0.0)
        cgrndl = torch.clamp(cgrndl, min=0.0)
    emg_t = _as_tensor(emg)
    htvp_t = _as_tensor(htvp)
    t_soil_t = _as_tensor(t_soil_top)
    t_grnd_t = _as_tensor(t_grnd)
    radiation_in = sabg_soil + dlrad + (1.0 - _as_tensor(frac_veg_nosno)) * emg_t * _as_tensor(forc_lwrad)
    longwave_emitted = emg_t * STEFAN_BOLTZMANN * t_soil_t.pow(4)
    latent_soil = htvp_t * qsoil
    hs_soil = radiation_in - longwave_emitted - sh_soil - latent_soil
    longwave_derivative = 4.0 * emg_t * STEFAN_BOLTZMANN * t_grnd_t.pow(3)
    dhsdT = -(cgrnds + htvp_t * cgrndl + longwave_derivative)
    components = {
        "radiation_in": radiation_in,
        "longwave_emitted": longwave_emitted,
        "sensible_soil": sh_soil,
        "latent_soil": latent_soil,
        "longwave_derivative": longwave_derivative,
    }
    return hs_soil, dhsdT, components


def derive_surface_thermal_boundaries_torch(
    targets: Mapping[str, "torch.Tensor"],
    *,
    forc_lwrad: "torch.Tensor",
    frac_veg_nosno: "torch.Tensor",
    t_soil_top: "torch.Tensor",
    t_snow_top: "torch.Tensor",
    t_h2osfc: "torch.Tensor",
    t_grnd: "torch.Tensor",
    htvp: "torch.Tensor | float" = LATENT_HEAT_VAPORIZATION,
    emg: "torch.Tensor | float" = SOIL_EMISSIVITY,
    enforce_nonnegative_conductance: bool = True,
) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor", dict[str, "torch.Tensor"]]:
    """Torch equivalent of :func:`derive_surface_thermal_boundaries_numpy`."""

    if torch is None:  # pragma: no cover
        raise RuntimeError("PyTorch is required for Torch energy closure")
    if not has_energy_closed_surface_targets(targets):
        missing = sorted(ENERGY_CLOSED_SURFACE_TARGET_NAMES.difference(targets))
        raise KeyError(f"Missing full surface energy-closure targets: {', '.join(missing)}")
    like = targets["qflx_ev_soil"]

    def _as_tensor(value):
        return value.to(device=like.device, dtype=like.dtype) if isinstance(value, torch.Tensor) else torch.as_tensor(value, device=like.device, dtype=like.dtype)

    emg_t = _as_tensor(emg)
    htvp_t = _as_tensor(htvp)
    common_radiation = _as_tensor(targets["dlrad"]) + (
        1.0 - _as_tensor(frac_veg_nosno)
    ) * emg_t * _as_tensor(forc_lwrad)
    cgrnds = _as_tensor(targets["cgrnds"])
    cgrndl = _as_tensor(targets["cgrndl"])
    if enforce_nonnegative_conductance:
        cgrnds = torch.clamp(cgrnds, min=0.0)
        cgrndl = torch.clamp(cgrndl, min=0.0)

    def _boundary(sabg, temperature, sensible, evaporation):
        emitted = emg_t * STEFAN_BOLTZMANN * _as_tensor(temperature).pow(4)
        latent = htvp_t * _as_tensor(evaporation)
        heat = _as_tensor(sabg) + common_radiation - emitted - _as_tensor(sensible) - latent
        return heat, emitted, latent

    hs_soil, lw_soil, latent_soil = _boundary(
        targets["sabg_soil"], t_soil_top, targets["eflx_sh_soil"], targets["qflx_ev_soil"]
    )
    hs_top_snow, lw_snow, latent_snow = _boundary(
        targets["sabg_snow_top"], t_snow_top, targets["eflx_sh_snow"], targets["qflx_ev_snow"]
    )
    hs_h2osfc, lw_h2osfc, latent_h2osfc = _boundary(
        targets["sabg_soil"], t_h2osfc, targets["eflx_sh_h2osfc"], targets["qflx_ev_h2osfc"]
    )
    longwave_derivative = 4.0 * emg_t * STEFAN_BOLTZMANN * _as_tensor(t_grnd).pow(3)
    dhsdT = -(cgrnds + htvp_t * cgrndl + longwave_derivative)
    return hs_soil, hs_top_snow, hs_h2osfc, dhsdT, {
        "common_radiation": common_radiation,
        "longwave_emitted_soil": lw_soil,
        "longwave_emitted_snow": lw_snow,
        "longwave_emitted_h2osfc": lw_h2osfc,
        "latent_soil": latent_soil,
        "latent_snow": latent_snow,
        "latent_h2osfc": latent_h2osfc,
        "longwave_derivative": longwave_derivative,
    }
