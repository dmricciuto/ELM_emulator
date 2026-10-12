from __future__ import annotations

import torch

from elm_emulator.benchmark_replay import _reduce_regional_shortwave_diagnostic


def _reduce(key: str, value: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return _reduce_regional_shortwave_diagnostic(
        key,
        value,
        pft_weights=weights,
        universal_pft_weights=weights,
        universal_pft_count=weights.shape[0],
        device=torch.device("cpu"),
        dtype=torch.float32,
    )


def test_cell_major_shortwave_forcing_survives_100k_cell_reduction() -> None:
    weights = torch.full((8, 100_000), 1.0 / 8.0)
    value = torch.arange(100_000, dtype=torch.float32)[:, None]
    reduced = _reduce("fsdsvd_mean", value, weights)
    torch.testing.assert_close(reduced, value[:, 0])


def test_pft_major_shortwave_field_is_area_blended() -> None:
    weights = torch.tensor([[0.25, 0.75], [0.75, 0.25]], dtype=torch.float32)
    value = torch.tensor(
        [[[2.0], [4.0]], [[6.0], [8.0]]],
        dtype=torch.float32,
    )
    reduced = _reduce("fsrvd_mean", value, weights)
    torch.testing.assert_close(reduced, torch.tensor([5.0, 5.0]))


def test_numrad_supports_cell_and_pft_major_layouts() -> None:
    weights = torch.tensor([[0.25, 0.75], [0.75, 0.25]], dtype=torch.float32)
    cell_major = torch.tensor([[[0.1, 0.2]], [[0.3, 0.4]]])
    torch.testing.assert_close(_reduce("albgrd_mean", cell_major, weights), cell_major[:, 0])

    pft_major = torch.stack((cell_major, cell_major + 0.2), dim=0)
    expected = torch.tensor([[0.25, 0.35], [0.35, 0.45]])
    torch.testing.assert_close(_reduce("albd_mean", pft_major, weights), expected)
