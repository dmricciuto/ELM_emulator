import numpy as np
import pytest

from elm_emulator.coupled_surface_hydrology_soiltemp import _ensure_surface_targets
from elm_emulator.thermal_energy_closure import (
    LATENT_HEAT_VAPORIZATION,
    SOIL_EMISSIVITY,
    STEFAN_BOLTZMANN,
    derive_surface_thermal_boundaries_numpy,
    derive_surface_thermal_boundaries_torch,
    derive_soil_thermal_boundary_numpy,
    derive_soil_thermal_boundary_torch,
    has_energy_closed_soil_targets,
    has_energy_closed_surface_targets,
)
from elm_emulator.train_patch_models import _energy_closure_loss_torch


def _primitive_targets(array):
    return {
        "sabg_soil": array([120.0, 80.0]),
        "dlrad": array([25.0, 15.0]),
        "eflx_sh_soil": array([30.0, 12.0]),
        "qflx_ev_soil": array([2.0e-5, -1.0e-6]),
        "cgrnds": array([8.0, 5.0]),
        "cgrndl": array([2.0e-6, 1.0e-6]),
    }


def test_numpy_boundary_matches_native_elm_decomposition():
    targets = _primitive_targets(np.asarray)
    forc_lwrad = np.array([300.0, 280.0])
    frac_veg = np.array([1.0, 0.25])
    t_soil = np.array([290.0, 275.0])
    t_grnd = np.array([291.0, 276.0])

    hs_soil, dhsdT, components = derive_soil_thermal_boundary_numpy(
        targets,
        forc_lwrad=forc_lwrad,
        frac_veg_nosno=frac_veg,
        t_soil_top=t_soil,
        t_grnd=t_grnd,
    )

    expected_hs = (
        targets["sabg_soil"]
        + targets["dlrad"]
        + (1.0 - frac_veg) * SOIL_EMISSIVITY * forc_lwrad
        - SOIL_EMISSIVITY * STEFAN_BOLTZMANN * t_soil**4
        - targets["eflx_sh_soil"]
        - LATENT_HEAT_VAPORIZATION * targets["qflx_ev_soil"]
    )
    expected_derivative = -(
        targets["cgrnds"]
        + LATENT_HEAT_VAPORIZATION * targets["cgrndl"]
        + 4.0 * SOIL_EMISSIVITY * STEFAN_BOLTZMANN * t_grnd**3
    )
    np.testing.assert_allclose(hs_soil, expected_hs)
    np.testing.assert_allclose(dhsdT, expected_derivative)
    np.testing.assert_allclose(
        hs_soil,
        components["radiation_in"]
        - components["longwave_emitted"]
        - components["sensible_soil"]
        - components["latent_soil"],
    )


def test_torch_boundary_matches_numpy_and_propagates_gradient():
    torch = pytest.importorskip("torch")
    np_targets = _primitive_targets(np.asarray)
    torch_targets = {
        name: torch.tensor(value, dtype=torch.float64, requires_grad=True)
        for name, value in np_targets.items()
    }
    kwargs_np = {
        "forc_lwrad": np.array([300.0, 280.0]),
        "frac_veg_nosno": np.array([1.0, 0.25]),
        "t_soil_top": np.array([290.0, 275.0]),
        "t_grnd": np.array([291.0, 276.0]),
    }
    hs_np, derivative_np, _ = derive_soil_thermal_boundary_numpy(np_targets, **kwargs_np)
    kwargs_torch = {
        name: torch.tensor(value, dtype=torch.float64)
        for name, value in kwargs_np.items()
    }
    hs_t, derivative_t, _ = derive_soil_thermal_boundary_torch(
        torch_targets,
        **kwargs_torch,
    )

    np.testing.assert_allclose(hs_t.detach().numpy(), hs_np)
    np.testing.assert_allclose(derivative_t.detach().numpy(), derivative_np)
    (hs_t.sum() + derivative_t.sum()).backward()
    assert torch_targets["qflx_ev_soil"].grad is not None
    assert torch_targets["cgrnds"].grad is not None


def test_full_surface_boundaries_close_soil_snow_and_standing_water():
    torch = pytest.importorskip("torch")
    targets = _primitive_targets(np.asarray)
    targets.update(
        {
            "sabg_snow_top": np.array([55.0, 20.0]),
            "eflx_sh_snow": np.array([12.0, 7.0]),
            "eflx_sh_h2osfc": np.array([18.0, 9.0]),
            "qflx_ev_snow": np.array([3.0e-6, 1.0e-6]),
            "qflx_ev_h2osfc": np.array([6.0e-6, 2.0e-6]),
        }
    )
    kwargs = {
        "forc_lwrad": np.array([300.0, 280.0]),
        "frac_veg_nosno": np.array([1.0, 0.25]),
        "t_soil_top": np.array([290.0, 275.0]),
        "t_snow_top": np.array([268.0, 270.0]),
        "t_h2osfc": np.array([285.0, 273.5]),
        "t_grnd": np.array([281.0, 273.0]),
    }
    boundaries_np = derive_surface_thermal_boundaries_numpy(targets, **kwargs)
    common = targets["dlrad"] + (1.0 - kwargs["frac_veg_nosno"]) * SOIL_EMISSIVITY * kwargs["forc_lwrad"]
    expected_snow = (
        targets["sabg_snow_top"]
        + common
        - SOIL_EMISSIVITY * STEFAN_BOLTZMANN * kwargs["t_snow_top"] ** 4
        - targets["eflx_sh_snow"]
        - LATENT_HEAT_VAPORIZATION * targets["qflx_ev_snow"]
    )
    expected_water = (
        targets["sabg_soil"]
        + common
        - SOIL_EMISSIVITY * STEFAN_BOLTZMANN * kwargs["t_h2osfc"] ** 4
        - targets["eflx_sh_h2osfc"]
        - LATENT_HEAT_VAPORIZATION * targets["qflx_ev_h2osfc"]
    )
    np.testing.assert_allclose(boundaries_np[1], expected_snow)
    np.testing.assert_allclose(boundaries_np[2], expected_water)
    assert has_energy_closed_surface_targets(targets)

    torch_targets = {
        name: torch.tensor(value, dtype=torch.float64, requires_grad=True)
        for name, value in targets.items()
    }
    torch_kwargs = {
        name: torch.tensor(value, dtype=torch.float64) for name, value in kwargs.items()
    }
    boundaries_t = derive_surface_thermal_boundaries_torch(torch_targets, **torch_kwargs)
    for actual, expected in zip(boundaries_t[:4], boundaries_np[:4]):
        np.testing.assert_allclose(actual.detach().numpy(), expected)
    sum(value.sum() for value in boundaries_t[:4]).backward()
    assert torch_targets["sabg_snow_top"].grad is not None
    assert torch_targets["eflx_sh_h2osfc"].grad is not None


def test_primitive_contract_replaces_required_composite_targets():
    targets = _primitive_targets(np.asarray)
    targets.update(
        {
            "qflx_tran_veg": np.array([0.0]),
            "qflx_ev_h2osfc": np.array([0.0]),
            "hs_h2osfc": np.array([0.0]),
        }
    )
    assert has_energy_closed_soil_targets(targets)
    _ensure_surface_targets(None, targets)


def test_incomplete_primitive_contract_does_not_hide_missing_composites():
    targets = _primitive_targets(np.asarray)
    targets.pop("dlrad")
    targets.update(
        {
            "qflx_tran_veg": np.array([0.0]),
            "qflx_ev_h2osfc": np.array([0.0]),
            "hs_h2osfc": np.array([0.0]),
        }
    )
    assert not has_energy_closed_soil_targets(targets)
    with pytest.raises(KeyError, match="hs_soil, dhsdT"):
        _ensure_surface_targets(None, targets)


def test_closure_loss_accepts_float32_roundoff_in_zero_surface_fractions():
    torch = pytest.importorskip("torch")
    input_names = [
        "forc_lwrad",
        "frac_veg_nosno_patch",
        "t_soisno_lev1",
        "t_grnd",
        "frac_sno",
        "frac_h2osfc",
    ]
    target_names = [
        "qflx_ev_soil",
        "eflx_sh_soil",
        "sabg_soil",
        "dlrad",
        "cgrnds",
        "cgrndl",
    ]
    x_phys = torch.tensor(
        [[300.0, 1.0, 290.0, 291.0, 1.4901161e-8, 0.0]],
        dtype=torch.float32,
    )
    true_phys = torch.tensor(
        [[2.0e-5, 30.0, 120.0, 25.0, 8.0, 2.0e-6]],
        dtype=torch.float32,
    )
    pred_phys = true_phys.clone()
    pred_phys[:, target_names.index("eflx_sh_soil")] += 5.0
    pred_phys.requires_grad_()
    closure, bias, derivative = _energy_closure_loss_torch(
        pred_phys,
        true_phys,
        torch.ones_like(true_phys),
        x_phys,
        input_names=input_names,
        target_names=target_names,
        x_mean=torch.zeros(len(input_names)),
        x_std=torch.ones(len(input_names)),
        y_mean=torch.zeros(len(target_names)),
        y_std=torch.ones(len(target_names)),
        heat_scale=50.0,
        derivative_scale=10.0,
    )
    assert closure.item() > 0.0
    assert bias.item() > 0.0
    assert derivative.item() == pytest.approx(0.0)


def test_closure_loss_uses_active_snow_and_surface_water_boundaries():
    torch = pytest.importorskip("torch")
    input_names = [
        "forc_lwrad",
        "frac_veg_nosno_patch",
        "t_soisno_lev1",
        "t_grnd",
        "frac_sno",
        "frac_h2osfc",
        "t_snow_top",
        "t_h2osfc",
    ]
    target_names = [
        "qflx_ev_soil",
        "qflx_ev_snow",
        "qflx_ev_h2osfc",
        "eflx_sh_soil",
        "eflx_sh_snow",
        "eflx_sh_h2osfc",
        "sabg_soil",
        "sabg_snow_top",
        "dlrad",
        "cgrnds",
        "cgrndl",
    ]
    x_phys = torch.tensor(
        [
            [300.0, 1.0, 290.0, 281.0, 0.7, 0.0, 268.0, 285.0],
            [300.0, 1.0, 290.0, 285.0, 0.0, 0.8, 268.0, 285.0],
        ],
        dtype=torch.float32,
    )
    true_phys = torch.tensor(
        [
            [2e-5, 3e-6, 6e-6, 30.0, 12.0, 18.0, 120.0, 55.0, 25.0, 8.0, 2e-6],
            [2e-5, 3e-6, 6e-6, 30.0, 12.0, 18.0, 120.0, 55.0, 25.0, 8.0, 2e-6],
        ],
        dtype=torch.float32,
    )
    pred_phys = true_phys.clone()
    pred_phys[0, target_names.index("eflx_sh_snow")] += 8.0
    pred_phys[1, target_names.index("eflx_sh_h2osfc")] += 6.0
    pred_phys.requires_grad_()
    closure, bias, derivative = _energy_closure_loss_torch(
        pred_phys,
        true_phys,
        torch.ones_like(true_phys),
        x_phys,
        input_names=input_names,
        target_names=target_names,
        x_mean=torch.zeros(len(input_names)),
        x_std=torch.ones(len(input_names)),
        y_mean=torch.zeros(len(target_names)),
        y_std=torch.ones(len(target_names)),
        heat_scale=50.0,
        derivative_scale=10.0,
    )
    assert closure.item() > 0.0
    assert bias.item() > 0.0
    assert derivative.item() == pytest.approx(0.0)
    closure.backward()
    assert pred_phys.grad is not None
