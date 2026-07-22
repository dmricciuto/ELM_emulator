from __future__ import annotations

import numpy as np
import pytest

from elm_emulator.coupled_surface_hydrology_soiltemp import (
    SURFACE_ENERGY_SMP_L_LOWER_BOUND,
    SurfaceModelBundle,
    predict_surface_targets,
)


torch = pytest.importorskip("torch")


class _CopySmpModel(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        smp = x[:, :1]
        return torch.cat((smp, smp), dim=1)


class _ThermalFluxModel(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.empty((x.shape[0], 3), device=x.device, dtype=x.dtype)
        out[:, 0] = 10.0
        out[:, 1] = -1000.0
        out[:, 2] = -2.5
        return out


def _bundle() -> SurfaceModelBundle:
    return SurfaceModelBundle(
        patch_index=13,
        input_names=["smp_l_lev1"],
        target_names=["hs_soil", "t_ref2m"],
        x_mean=np.asarray([0.0], dtype=np.float64),
        x_std=np.asarray([1.0], dtype=np.float64),
        y_mean=np.asarray([0.0, 0.0], dtype=np.float64),
        y_std=np.asarray([1.0, 1.0], dtype=np.float64),
        model=_CopySmpModel(),
    )


def _thermal_bundle() -> SurfaceModelBundle:
    return SurfaceModelBundle(
        patch_index=13,
        input_names=["frac_h2osfc"],
        target_names=["hs_soil", "hs_h2osfc", "dhsdT"],
        x_mean=np.asarray([0.0], dtype=np.float64),
        x_std=np.asarray([1.0], dtype=np.float64),
        y_mean=np.asarray([0.0, 0.0, 0.0], dtype=np.float64),
        y_std=np.asarray([1.0, 1.0, 1.0], dtype=np.float64),
        model=_ThermalFluxModel(),
    )


def test_smp_l_clip_only_applies_to_surface_energy_targets_numpy() -> None:
    features = np.asarray([[-4.25e7]], dtype=np.float64)

    targets = predict_surface_targets(
        _bundle(),
        features,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    np.testing.assert_allclose(targets["hs_soil"], [SURFACE_ENERGY_SMP_L_LOWER_BOUND])
    np.testing.assert_allclose(targets["t_ref2m"], [-4.25e7])


def test_smp_l_clip_only_applies_to_surface_energy_targets_torch() -> None:
    features = torch.tensor([[-4.25e7]], dtype=torch.float32)

    targets = predict_surface_targets(
        _bundle(),
        features,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert torch.allclose(targets["hs_soil"], torch.tensor([SURFACE_ENERGY_SMP_L_LOWER_BOUND]))
    assert torch.allclose(targets["t_ref2m"], torch.tensor([-4.25e7]))


def test_hs_h2osfc_only_drives_dhsdt_when_fraction_is_active_numpy() -> None:
    features = np.asarray([[0.0], [0.2]], dtype=np.float64)

    targets = predict_surface_targets(
        _thermal_bundle(),
        features,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    np.testing.assert_allclose(targets["hs_h2osfc"], [-1000.0, -1000.0])
    np.testing.assert_allclose(targets["dhsdT"], [-2.5, -50.0])


def test_hs_h2osfc_only_drives_dhsdt_when_fraction_is_active_torch() -> None:
    features = torch.tensor([[0.0], [0.2]], dtype=torch.float32)

    targets = predict_surface_targets(
        _thermal_bundle(),
        features,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert torch.allclose(targets["hs_h2osfc"], torch.tensor([-1000.0, -1000.0]))
    assert torch.allclose(targets["dhsdT"], torch.tensor([-2.5, -50.0]))
