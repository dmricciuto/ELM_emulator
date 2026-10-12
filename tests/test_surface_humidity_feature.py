from __future__ import annotations

import numpy as np
import torch

from elm_emulator.coupled_surface_hydrology_soiltemp import (
    diagnostic_soil_surface_humidity,
    diagnostic_soil_surface_humidity_torch,
)


def test_soil_surface_humidity_numpy_torch_match() -> None:
    temperature = np.asarray([275.0, 290.0, 305.0], dtype=np.float64)
    matric_potential = np.asarray([-1.0e3, -1.0e5, -1.0e7], dtype=np.float64)
    pressure = np.asarray([90000.0, 100000.0, 95000.0], dtype=np.float64)
    forc_q = np.asarray([0.003, 0.008, 0.015], dtype=np.float64)

    expected = diagnostic_soil_surface_humidity(
        temperature,
        matric_potential,
        pressure,
        forc_q,
    )
    actual = diagnostic_soil_surface_humidity_torch(
        torch.as_tensor(temperature),
        torch.as_tensor(matric_potential),
        torch.as_tensor(pressure),
        torch.as_tensor(forc_q),
    ).numpy()

    np.testing.assert_allclose(actual, expected, rtol=1.0e-12, atol=0.0)
    assert np.all(expected >= 0.0)


def test_drier_soil_has_lower_surface_humidity() -> None:
    humidity = diagnostic_soil_surface_humidity(
        soil_temperature=np.asarray([295.0, 295.0]),
        soil_matric_potential_mm=np.asarray([-1.0e3, -1.0e7]),
        forc_pbot=100000.0,
        forc_q=0.0,
    )

    assert humidity[1] < humidity[0]
