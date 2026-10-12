import numpy as np

from elm_emulator import coupled_surface_hydrology_soiltemp as coupled

import torch


def _rain_on_snow_state(*, torch_state: bool):
    dz = np.array([[0.02, 0.05, 0.08, 0.12, 0.18]], dtype=np.float64)
    ice = np.array([[3.0, 8.0, 15.0, 20.0, 20.0]], dtype=np.float64)
    liq = np.array([[0.1, 0.2, 0.3, 0.4, 0.5]], dtype=np.float64)
    temp = np.full((1, 5), coupled.TFRZ, dtype=np.float64)
    rds = np.full((1, 5), 300.0, dtype=np.float64)

    def value(array, *, integer=False):
        if not torch_state:
            return array
        dtype = torch.int32 if integer else torch.float64
        return torch.as_tensor(array, dtype=dtype)

    layers = coupled.SnowLayerState(
        snl=value(np.array([-5], dtype=np.int32), integer=True),
        dz=value(dz),
        h2osoi_ice=value(ice),
        h2osoi_liq=value(liq),
        t_snow=value(temp),
        snw_rds=value(rds),
    )
    return coupled.CoupledColumnState(
        h2osoi_liq=value(np.full((1, 3), 10.0)),
        h2osoi_ice=value(np.zeros((1, 3))),
        h2ocan=value(np.zeros(1)),
        h2osfc=value(np.zeros(1)),
        frac_h2osfc=value(np.zeros(1)),
        frac_sno=value(np.full(1, 0.8)),
        zwt=value(np.ones(1)),
        t_soisno=value(np.full((1, 3), 272.0)),
        h2osno=value(np.full(1, 67.5)),
        snow_depth=value(np.full(1, 0.45)),
        snow_t=value(temp),
        snow_layers=layers,
        int_snow=value(np.full(1, 100.0)),
    )


def test_three_hour_step_subcycles_rain_through_snowpack():
    state = _rain_on_snow_state(torch_state=True)
    rain = torch.tensor([30.0], dtype=torch.float64)
    one_sweep = coupled.apply_snow_water_and_capping(
        state,
        None,
        rain_to_snowpack=rain,
        dt=1800.0,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    three_hour = coupled.apply_snow_water_and_capping(
        state,
        None,
        rain_to_snowpack=rain,
        dt=10800.0,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )

    # One receiving-capacity-limited sweep strands most rain in the top snow
    # layers. Six native-cadence sweeps move substantially more out the bottom.
    assert three_hour.qflx_snow_to_soil.item() > one_sweep.qflx_snow_to_soil.item() + 15.0
    assert three_hour.state.snow_layers.h2osoi_liq.sum().item() < one_sweep.state.snow_layers.h2osoi_liq.sum().item()

    initial_plus_rain = state.h2osno + rain
    torch.testing.assert_close(
        three_hour.state.h2osno + three_hour.qflx_snow_to_soil,
        initial_plus_rain,
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_subcycled_snow_water_numpy_matches_torch():
    rain_np = np.array([30.0], dtype=np.float64)
    numpy_result = coupled.apply_snow_water_and_capping(
        _rain_on_snow_state(torch_state=False),
        None,
        rain_to_snowpack=rain_np,
        dt=10800.0,
    )
    torch_result = coupled.apply_snow_water_and_capping(
        _rain_on_snow_state(torch_state=True),
        None,
        rain_to_snowpack=torch.as_tensor(rain_np),
        dt=10800.0,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )

    np.testing.assert_allclose(
        torch_result.qflx_snow_to_soil.detach().numpy(),
        numpy_result.qflx_snow_to_soil,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        torch_result.state.h2osno.detach().numpy(),
        numpy_result.state.h2osno,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        torch_result.state.snow_layers.h2osoi_liq.detach().numpy(),
        numpy_result.state.snow_layers.h2osoi_liq,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
