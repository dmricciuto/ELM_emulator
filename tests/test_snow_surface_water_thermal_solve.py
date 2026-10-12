import torch

from elm_emulator.elm_hydrology import SoilTextureProfile, elm_soil_grid
from elm_emulator.elm_soil_temperature_torch import (
    TKAIR,
    TKICE,
    SnowLayerState,
    SoilTemperatureForcingTorch,
    SoilTemperatureStateTorch,
    _batched_pentadiagonal_torch,
    _snow_thermal_conductivity_torch,
    soil_temperature_step_torch,
    thermal_params_from_texture_torch,
)


def test_batched_pentadiagonal_matches_dense_solve() -> None:
    torch.manual_seed(7)
    batch, n = 4, 9
    dtype = torch.float64
    lower2 = 0.04 * torch.randn(batch, n, dtype=dtype)
    lower1 = 0.08 * torch.randn(batch, n, dtype=dtype)
    upper1 = 0.08 * torch.randn(batch, n, dtype=dtype)
    upper2 = 0.04 * torch.randn(batch, n, dtype=dtype)
    lower2[:, :2] = 0.0
    lower1[:, :1] = 0.0
    upper1[:, -1:] = 0.0
    upper2[:, -2:] = 0.0
    diagonal = torch.full((batch, n), 2.5, dtype=dtype)
    rhs = torch.randn(batch, n, dtype=dtype)

    matrix = (
        torch.diag_embed(diagonal)
        + torch.diag_embed(lower1[:, 1:], offset=-1)
        + torch.diag_embed(lower2[:, 2:], offset=-2)
        + torch.diag_embed(upper1[:, :-1], offset=1)
        + torch.diag_embed(upper2[:, :-2], offset=2)
    )
    expected = torch.linalg.solve(matrix, rhs.unsqueeze(-1)).squeeze(-1)
    actual = _batched_pentadiagonal_torch(
        lower2, lower1, diagonal, upper1, upper2, rhs
    )
    torch.testing.assert_close(actual, expected, rtol=1.0e-12, atol=1.0e-12)


def test_surface_water_flux_changes_soil_temperature_under_snow() -> None:
    dtype = torch.float64
    nlev = 4
    dz, z, zi = elm_soil_grid(nlevsoi=nlev)
    texture = SoilTextureProfile(
        sand_pct=torch.full((nlev,), 45.0, dtype=dtype).numpy(),
        clay_pct=torch.full((nlev,), 20.0, dtype=dtype).numpy(),
        organic_frac=torch.zeros(nlev, dtype=dtype).numpy(),
    )
    params = thermal_params_from_texture_torch(
        texture,
        dz,
        z,
        zi,
        device=torch.device("cpu"),
        dtype=dtype,
    )
    snow_dz = torch.zeros((1, 5), dtype=dtype)
    snow_ice = torch.zeros_like(snow_dz)
    snow_liq = torch.zeros_like(snow_dz)
    snow_temp = torch.full_like(snow_dz, 268.0)
    snow_rds = torch.zeros_like(snow_dz)
    snow_dz[:, 0] = 0.05
    snow_ice[:, 0] = 10.0
    snow_rds[:, 0] = 321.0
    state = SoilTemperatureStateTorch(
        t_soisno=torch.tensor([[278.0, 277.0, 276.0, 275.0]], dtype=dtype),
        h2osoi_liq=torch.full((1, nlev), 10.0, dtype=dtype),
        h2osoi_ice=torch.zeros((1, nlev), dtype=dtype),
        frac_sno=torch.tensor([0.7], dtype=dtype),
        snow_layers=SnowLayerState(
            snl=torch.tensor([-1], dtype=torch.int32),
            dz=snow_dz,
            h2osoi_ice=snow_ice,
            h2osoi_liq=snow_liq,
            t_snow=snow_temp,
            snw_rds=snow_rds,
        ),
        t_h2osfc=torch.tensor([278.0], dtype=dtype),
        h2osfc=torch.tensor([10.0], dtype=dtype),
        frac_h2osfc=torch.tensor([0.3], dtype=dtype),
        h2osno=torch.tensor([10.0], dtype=dtype),
        snow_depth=torch.tensor([0.05], dtype=dtype),
        snow_t=snow_temp,
    )

    def advance(hs_h2osfc: float):
        forcing = SoilTemperatureForcingTorch(
            hs_soil=torch.tensor([0.0], dtype=dtype),
            hs_top_snow=torch.tensor([0.0], dtype=dtype),
            hs_h2osfc=torch.tensor([hs_h2osfc], dtype=dtype),
            dhsdT=torch.tensor([-4.0], dtype=dtype),
            eflx_bot=torch.tensor([0.0], dtype=dtype),
        )
        return soil_temperature_step_torch(
            state,
            params,
            forcing,
            1800.0,
            snow_enabled=True,
            blend_partial_snow_soil_temperature=True,
        )

    cold_result = advance(-100.0)
    warm_result = advance(100.0)
    cold_water_soil = cold_result.state.t_soisno[..., 0]
    warm_water_soil = warm_result.state.t_soisno[..., 0]
    assert torch.all(warm_water_soil > cold_water_soil)
    torch.testing.assert_close(cold_result.state.snow_layers.snw_rds, snow_rds)
    torch.testing.assert_close(warm_result.state.snow_layers.snw_rds, snow_rds)

    melt_dt = 1800.0
    melt_result = soil_temperature_step_torch(
        state,
        params,
        SoilTemperatureForcingTorch(
            hs_soil=torch.tensor([0.0], dtype=dtype),
            hs_top_snow=torch.tensor([600.0], dtype=dtype),
            hs_h2osfc=torch.tensor([0.0], dtype=dtype),
            dhsdT=torch.tensor([-4.0], dtype=dtype),
            eflx_bot=torch.tensor([0.0], dtype=dtype),
        ),
        melt_dt,
        snow_enabled=True,
        blend_partial_snow_soil_temperature=True,
    )
    expected_melt_profile = torch.clamp(
        state.snow_layers.h2osoi_ice
        - melt_result.state.snow_layers.h2osoi_ice,
        min=0.0,
    ) / melt_dt
    assert torch.sum(expected_melt_profile) > 0.0
    torch.testing.assert_close(
        melt_result.qflx_snow_melt_profile,
        expected_melt_profile,
    )
    torch.testing.assert_close(
        torch.sum(melt_result.qflx_snow_melt_profile, dim=-1),
        melt_result.qflx_snow_melt,
    )
    torch.testing.assert_close(
        melt_result.state.snow_t,
        melt_result.state.snow_layers.t_snow,
    )


def test_default_snow_conductivity_matches_native_jordan_option() -> None:
    dtype = torch.float64
    density = torch.tensor([100.0, 200.0, 300.0, 400.0], dtype=dtype)
    temperature = torch.full_like(density, 273.0)
    expected = TKAIR + (
        7.75e-5 * density + 1.105e-6 * density**2
    ) * (TKICE - TKAIR)
    actual = _snow_thermal_conductivity_torch(temperature, density)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_internal_snow_absorption_warms_internal_layer_and_keeps_state_synced() -> None:
    dtype = torch.float64
    nlev = 4
    dz, z, zi = elm_soil_grid(nlevsoi=nlev)
    texture = SoilTextureProfile(
        sand_pct=torch.full((nlev,), 45.0, dtype=dtype).numpy(),
        clay_pct=torch.full((nlev,), 20.0, dtype=dtype).numpy(),
        organic_frac=torch.zeros(nlev, dtype=dtype).numpy(),
    )
    params = thermal_params_from_texture_torch(
        texture, dz, z, zi, device=torch.device("cpu"), dtype=dtype
    )
    snow_dz = torch.zeros((1, 5), dtype=dtype)
    snow_dz[0, :2] = torch.tensor([0.02, 0.08], dtype=dtype)
    snow_ice = torch.zeros_like(snow_dz)
    snow_ice[0, :2] = torch.tensor([4.0, 16.0], dtype=dtype)
    snow_liq = torch.zeros_like(snow_dz)
    snow_temp = torch.full_like(snow_dz, 268.0)
    state = SoilTemperatureStateTorch(
        t_soisno=torch.tensor([[274.0, 273.0, 272.0, 271.0]], dtype=dtype),
        h2osoi_liq=torch.full((1, nlev), 10.0, dtype=dtype),
        h2osoi_ice=torch.zeros((1, nlev), dtype=dtype),
        frac_sno=torch.tensor([0.8], dtype=dtype),
        snow_layers=SnowLayerState(
            snl=torch.tensor([-2], dtype=torch.int32),
            dz=snow_dz,
            h2osoi_ice=snow_ice,
            h2osoi_liq=snow_liq,
            t_snow=snow_temp,
        ),
        snow_t=snow_temp,
        h2osno=torch.tensor([20.0], dtype=dtype),
        snow_depth=torch.tensor([0.1], dtype=dtype),
        h2osfc=torch.tensor([0.0], dtype=dtype),
        frac_h2osfc=torch.tensor([0.0], dtype=dtype),
        t_h2osfc=torch.tensor([273.0], dtype=dtype),
    )

    def advance(total_absorption: float):
        return soil_temperature_step_torch(
            state,
            params,
            SoilTemperatureForcingTorch(
                hs_soil=torch.tensor([0.0], dtype=dtype),
                hs_top_snow=torch.tensor([0.0], dtype=dtype),
                hs_h2osfc=torch.tensor([0.0], dtype=dtype),
                dhsdT=torch.tensor([-4.0], dtype=dtype),
                eflx_bot=torch.tensor([0.0], dtype=dtype),
                sabg_snow_top=torch.tensor([10.0], dtype=dtype),
                sabg_snow_total=torch.tensor([total_absorption], dtype=dtype),
            ),
            1800.0,
            snow_enabled=True,
        )

    top_only = advance(10.0)
    with_internal = advance(70.0)
    assert with_internal.state.snow_t[0, 1] > top_only.state.snow_t[0, 1]
    torch.testing.assert_close(
        with_internal.state.snow_t,
        with_internal.state.snow_layers.t_snow,
    )
