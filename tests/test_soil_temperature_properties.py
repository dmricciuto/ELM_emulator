import numpy as np

from elm_emulator.elm_hydrology import SoilTextureProfile, elm_soil_grid
from elm_emulator.elm_soil_temperature import (
    THK_BEDROCK,
    SoilTemperatureState,
    soil_thermal_properties,
    thermal_params_from_texture,
)


def _texture(nlev: int) -> SoilTextureProfile:
    return SoilTextureProfile(
        sand_pct=np.full(nlev, 45.0),
        clay_pct=np.full(nlev, 20.0),
        organic_frac=np.zeros(nlev),
    )


def test_soil_thermal_conductivity_uses_bedrock_below_nlevbed() -> None:
    nlev = 4
    dz, z, zi = elm_soil_grid(nlevsoi=nlev)
    params = thermal_params_from_texture(dz, z, zi, _texture(nlev), nlevbed=2)
    state = SoilTemperatureState(
        t_soisno=np.full(nlev, 290.0),
        h2osoi_liq=np.full(nlev, 20.0),
        h2osoi_ice=np.zeros(nlev),
    )

    _, thk, tk = soil_thermal_properties(state, params)

    np.testing.assert_allclose(thk[2:], THK_BEDROCK)
    assert not np.allclose(thk[:2], THK_BEDROCK)
    assert tk[-1] == 0.0


def test_torch_soil_thermal_conductivity_uses_bedrock_below_nlevbed() -> None:
    torch = __import__("pytest").importorskip("torch")
    from elm_emulator.elm_soil_temperature_torch import (
        SoilTemperatureStateTorch,
        soil_thermal_properties_torch,
        thermal_params_from_texture_torch,
    )

    nlev = 4
    dz, z, zi = elm_soil_grid(nlevsoi=nlev)
    params = thermal_params_from_texture_torch(
        _texture(nlev),
        dz,
        z,
        zi,
        nlevbed=2,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    state = SoilTemperatureStateTorch(
        t_soisno=torch.full((nlev,), 290.0, dtype=torch.float64),
        h2osoi_liq=torch.full((nlev,), 20.0, dtype=torch.float64),
        h2osoi_ice=torch.zeros(nlev, dtype=torch.float64),
    )

    _, thk, tk = soil_thermal_properties_torch(state, params)

    np.testing.assert_allclose(thk[2:].numpy(), THK_BEDROCK)
    assert not np.allclose(thk[:2].numpy(), THK_BEDROCK)
    assert float(tk[-1]) == 0.0
