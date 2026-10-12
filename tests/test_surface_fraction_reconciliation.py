from dataclasses import replace

import numpy as np
import torch

from elm_emulator.coupled_surface_hydrology_soiltemp import (
    reconcile_snow_surface_water_fractions_numpy,
    reconcile_snow_surface_water_fractions_torch,
)
from elm_emulator.replay_initialization import regional_initial_state


def _numpy_state(frac_sno, frac_h2osfc, h2osno):
    state = regional_initial_state(
        len(frac_sno),
        4,
        nens=1,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    return replace(
        state,
        frac_sno=np.asarray(frac_sno, dtype=np.float64).reshape(-1, 1),
        frac_h2osfc=np.asarray(frac_h2osfc, dtype=np.float64).reshape(-1, 1),
        h2osno=np.asarray(h2osno, dtype=np.float64).reshape(-1, 1),
    )


def test_native_surface_fraction_reconciliation_numpy() -> None:
    state = _numpy_state(
        frac_sno=[0.8, 0.995, 0.999, 0.3],
        frac_h2osfc=[0.4, 0.2, 0.005, 0.4],
        h2osno=[10.0, 10.0, 10.0, 0.0],
    )
    out = reconcile_snow_surface_water_fractions_numpy(state)

    np.testing.assert_allclose(out.frac_h2osfc[:, 0], [0.2, 0.01, 0.005, 0.4])
    np.testing.assert_allclose(out.frac_sno[:, 0], [0.8, 0.99, 0.995, 0.3])
    assert np.all(out.frac_sno + out.frac_h2osfc <= 1.0 + 1.0e-12)


def test_native_surface_fraction_reconciliation_torch_matches_numpy() -> None:
    state = _numpy_state(
        frac_sno=[0.8, 0.995, 0.999, 0.3],
        frac_h2osfc=[0.4, 0.2, 0.005, 0.4],
        h2osno=[10.0, 10.0, 10.0, 0.0],
    )
    state_t = replace(
        state,
        frac_sno=torch.as_tensor(state.frac_sno),
        frac_h2osfc=torch.as_tensor(state.frac_h2osfc),
        h2osno=torch.as_tensor(state.h2osno),
    )
    out = reconcile_snow_surface_water_fractions_torch(
        state_t,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )

    np.testing.assert_allclose(out.frac_h2osfc.numpy()[:, 0], [0.2, 0.01, 0.005, 0.4])
    np.testing.assert_allclose(out.frac_sno.numpy()[:, 0], [0.8, 0.99, 0.995, 0.3])
    assert bool(torch.all(out.frac_sno + out.frac_h2osfc <= 1.0 + 1.0e-12))
