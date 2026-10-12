from __future__ import annotations

import numpy as np
import pytest

from elm_emulator.coupled_surface_hydrology_soiltemp import (
    CoupledColumnState,
    SurfaceModelBundle,
    apply_canopy_evaporation,
    predict_surface_targets,
)
from elm_emulator.elm_hydrology import (
    HydrologyForcing,
    HydrologyState,
    SoilTextureProfile,
    elm_soil_grid,
    hydraulic_params_from_texture,
    hydrology_no_drainage_step,
)


torch = pytest.importorskip("torch")


class _NegativeFluxModel(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.full((x.shape[0], 6), -1.0e-6, device=x.device, dtype=x.dtype)


def _signed_flux_bundle() -> SurfaceModelBundle:
    return SurfaceModelBundle(
        patch_index=13,
        input_names=["dummy"],
        target_names=[
            "qflx_evap_canopy",
            "qflx_ev_soil",
            "qflx_ev_snow",
            "qflx_ev_h2osfc",
            "qflx_tran_veg",
            "qflx_prec_grnd",
        ],
        x_mean=np.asarray([0.0], dtype=np.float64),
        x_std=np.asarray([1.0], dtype=np.float64),
        y_mean=np.zeros(6, dtype=np.float64),
        y_std=np.ones(6, dtype=np.float64),
        model=_NegativeFluxModel(),
    )


def _coupled_state(ncell: int = 2) -> CoupledColumnState:
    return CoupledColumnState(
        h2osoi_liq=np.full((ncell, 10), 30.0, dtype=np.float64),
        h2osoi_ice=np.zeros((ncell, 10), dtype=np.float64),
        h2ocan=np.asarray([0.10, 0.001], dtype=np.float64)[:ncell],
        h2osfc=np.zeros(ncell, dtype=np.float64),
        frac_h2osfc=np.zeros(ncell, dtype=np.float64),
        frac_sno=np.zeros(ncell, dtype=np.float64),
        zwt=np.full(ncell, 5.0, dtype=np.float64),
        t_soisno=np.full((ncell, 15), 293.15, dtype=np.float64),
    )


def test_non_transpiration_evaporation_targets_are_signed_numpy() -> None:
    targets = predict_surface_targets(
        _signed_flux_bundle(),
        np.asarray([[0.0]], dtype=np.float64),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    np.testing.assert_allclose(targets["qflx_evap_canopy"], [-1.0e-6])
    np.testing.assert_allclose(targets["qflx_ev_soil"], [-1.0e-6])
    np.testing.assert_allclose(targets["qflx_ev_snow"], [-1.0e-6])
    np.testing.assert_allclose(targets["qflx_ev_h2osfc"], [-1.0e-6])
    np.testing.assert_allclose(targets["qflx_tran_veg"], [0.0])
    np.testing.assert_allclose(targets["qflx_prec_grnd"], [0.0])


def test_non_transpiration_evaporation_targets_are_signed_torch() -> None:
    targets = predict_surface_targets(
        _signed_flux_bundle(),
        torch.zeros((1, 1), dtype=torch.float32),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert torch.allclose(targets["qflx_evap_canopy"], torch.tensor([-1.0e-6]))
    assert torch.allclose(targets["qflx_ev_soil"], torch.tensor([-1.0e-6]))
    assert torch.allclose(targets["qflx_ev_snow"], torch.tensor([-1.0e-6]))
    assert torch.allclose(targets["qflx_ev_h2osfc"], torch.tensor([-1.0e-6]))
    assert torch.allclose(targets["qflx_tran_veg"], torch.tensor([0.0]))
    assert torch.allclose(targets["qflx_prec_grnd"], torch.tensor([0.0]))


def test_negative_canopy_evaporation_adds_canopy_water_numpy() -> None:
    dt = 3600.0
    targets, state = apply_canopy_evaporation(
        {"qflx_evap_canopy": np.asarray([-1.0e-6, 1.0e-6], dtype=np.float64)},
        _coupled_state(),
        dt,
    )

    np.testing.assert_allclose(targets["qflx_evap_canopy"][0], -1.0e-6)
    np.testing.assert_allclose(state.h2ocan[0], 0.10 + dt * 1.0e-6)
    np.testing.assert_allclose(targets["qflx_evap_canopy"][1], 0.001 / dt)
    np.testing.assert_allclose(state.h2ocan[1], 0.0)


def test_negative_canopy_evaporation_adds_canopy_water_torch() -> None:
    dt = 3600.0
    targets, state = apply_canopy_evaporation(
        {"qflx_evap_canopy": torch.tensor([-1.0e-6, 1.0e-6], dtype=torch.float32)},
        _coupled_state(),
        dt,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert torch.allclose(targets["qflx_evap_canopy"][0], torch.tensor(-1.0e-6))
    assert torch.allclose(state.h2ocan[0], torch.tensor(0.10 + dt * 1.0e-6))
    assert torch.allclose(targets["qflx_evap_canopy"][1], torch.tensor(0.001 / dt))
    assert torch.allclose(state.h2ocan[1], torch.tensor(0.0))


def test_negative_soil_evaporation_enters_top_soil_boundary() -> None:
    dz, z, zi = elm_soil_grid(10)
    params = hydraulic_params_from_texture(
        dz,
        z,
        zi,
        SoilTextureProfile(
            sand_pct=np.full(10, 40.0, dtype=np.float64),
            clay_pct=np.full(10, 20.0, dtype=np.float64),
            organic_frac=np.zeros(10, dtype=np.float64),
        ),
        rootfr=np.full(10, 0.1, dtype=np.float64),
    )
    state = HydrologyState(
        h2osoi_liq=(0.25 * dz * 1000.0)[None, :],
        h2osoi_ice=np.zeros((1, 10), dtype=np.float64),
        h2osfc=np.zeros(1, dtype=np.float64),
        zwt=np.full(1, 5.0, dtype=np.float64),
        frac_h2osfc=np.zeros(1, dtype=np.float64),
        frac_sno=np.zeros(1, dtype=np.float64),
        t_soisno=np.full((1, 10), 293.15, dtype=np.float64),
    )
    zero = HydrologyForcing(
        qflx_top_soil=np.zeros(1, dtype=np.float64),
        qflx_evap_grnd=np.zeros(1, dtype=np.float64),
        qflx_ev_soil=np.zeros(1, dtype=np.float64),
        qflx_ev_h2osfc=np.zeros(1, dtype=np.float64),
        qflx_tran_veg=np.zeros(1, dtype=np.float64),
    )
    dew = HydrologyForcing(
        qflx_top_soil=np.zeros(1, dtype=np.float64),
        qflx_evap_grnd=np.full(1, -1.0e-6, dtype=np.float64),
        qflx_ev_soil=np.full(1, -1.0e-6, dtype=np.float64),
        qflx_ev_h2osfc=np.zeros(1, dtype=np.float64),
        qflx_tran_veg=np.zeros(1, dtype=np.float64),
    )

    zero_result = hydrology_no_drainage_step(state, params, zero, dt=3600.0)
    dew_result = hydrology_no_drainage_step(state, params, dew, dt=3600.0)

    assert dew_result.qflx_infl[0] > zero_result.qflx_infl[0]
    np.testing.assert_allclose(zero_result.qflx_soil_evap, [0.0])
    np.testing.assert_allclose(dew_result.qflx_soil_evap, [-1.0e-6])
    assert dew_result.state.h2osoi_liq[0, 0] > zero_result.state.h2osoi_liq[0, 0]
