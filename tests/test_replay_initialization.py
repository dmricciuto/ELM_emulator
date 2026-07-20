from __future__ import annotations

import numpy as np
import torch

from elm_emulator.replay_initialization import (
    load_regional_restart,
    prepare_regional_initial_state,
    regional_initial_state,
    save_regional_restart,
    tile_state,
)


def test_tile_state_broadcasts_1x1_state() -> None:
    state = regional_initial_state(
        1,
        4,
        nens=1,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    tiled = tile_state(state, ncell=3, nens=2)

    assert tiled.h2osoi_liq.shape == (3, 2, 4)
    assert tiled.h2osfc.shape == (3, 2)
    assert tiled.snow_layers is not None
    assert tiled.snow_layers.t_snow.shape == (3, 2, 5)


def test_regional_restart_roundtrip(tmp_path) -> None:
    state = regional_initial_state(
        2,
        4,
        nens=1,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    surf = {
        "active_idx": np.array([10, 11], dtype=np.int64),
        "grid_row": np.array([1, 1], dtype=np.int32),
        "grid_col": np.array([2, 3], dtype=np.int32),
        "lat_deg": np.array([31.0, 31.5], dtype=np.float64),
        "lon_deg": np.array([-99.0, -98.5], dtype=np.float64),
    }
    pft_ids = [7]
    pft_weights = np.ones((1, 2), dtype=np.float64)
    path = tmp_path / "restart.nc"

    save_regional_restart(
        path,
        state=state,
        surf=surf,
        pft_ids=pft_ids,
        pft_weights=pft_weights,
        start_year=1980,
        start_month=1,
        completed_year=1980,
        completed_month=1,
        completed_hours_last_month=744,
        next_year=1980,
        next_month=2,
        calendar="gregorian",
        snow_model_enabled=True,
        dtype=torch.float32,
        regional_max_hours=None,
    )
    loaded = load_regional_restart(
        path,
        surf=surf,
        pft_ids=pft_ids,
        pft_weights=pft_weights,
        nens=1,
        expected_start_year=1980,
        expected_start_month=2,
        snow_model_enabled=True,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert loaded.h2osoi_liq.shape == state.h2osoi_liq.shape
    assert torch.allclose(loaded.h2osoi_liq, state.h2osoi_liq)
    assert torch.allclose(loaded.t_soisno, state.t_soisno)


def test_prepare_regional_initial_state_cold_start() -> None:
    surf = {
        "lat_deg": np.array([31.0], dtype=np.float64),
        "lon_deg": np.array([-99.0], dtype=np.float64),
    }

    state = prepare_regional_initial_state(
        surf=surf,
        pft_ids=[7],
        pft_weights=np.ones((1, 1), dtype=np.float64),
        ncell=1,
        nlev=4,
        nlevbed=4,
        nens=1,
        device=torch.device("cpu"),
        dtype=torch.float32,
        restart_in=None,
        start_year=1980,
        start_month=1,
        snow_model_enabled=True,
        snow_scalar_only=False,
    )

    assert state.h2osoi_liq.shape == (1, 1, 4)
