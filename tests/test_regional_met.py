import numpy as np
import torch

from elm_emulator.regional_met import _MET_VAR_IDX, interp_met_to_1h_device


def test_elm_stream_interpolation_matches_cpl_bypass_linear_and_nearest_precip():
    step = 6
    ncore = 2
    nsteps = ncore + 2
    data = torch.zeros((nsteps, 1, 7), dtype=torch.float32)

    # Padded 6-hour records as native cpl-bypass sees them at 00, 06, 12, 18.
    source = torch.tensor([0.0, 6.0, 12.0, 18.0], dtype=torch.float32)
    for name in ("TBOT", "QBOT", "PSRF", "FLDS", "WIND"):
        data[:, 0, _MET_VAR_IDX[name]] = source

    data[:, 0, _MET_VAR_IDX["PRECTmms"]] = torch.tensor([1.0, 2.0, 4.0, 8.0])
    data[:, 0, _MET_VAR_IDX["FSDS"]] = torch.tensor([10.0, 20.0, 40.0, 80.0])

    out = interp_met_to_1h_device(
        data,
        step,
        elm_stream_methods=True,
        padding_steps=1,
    )

    label_hours = np.arange(1, ncore * step + 1, dtype=np.int64)
    tod_hours = label_hours % 24
    day_index = label_hours // 24
    lower = day_index * (24 // step) + ((tod_hours + (step // 2 - 1)) // step)
    wt1 = 1.0 - np.mod(tod_hours + 24 - step / 2, step) / step
    expected_linear = source.numpy()[lower] * wt1 + source.numpy()[lower + 1] * (1.0 - wt1)

    np.testing.assert_allclose(
        out[:, 0, _MET_VAR_IDX["TBOT"]].numpy(),
        expected_linear,
        rtol=1.0e-6,
    )
    np.testing.assert_allclose(
        out[:, 0, _MET_VAR_IDX["PRECTmms"]].numpy(),
        np.repeat([2.0, 4.0], step),
        rtol=1.0e-6,
    )
    np.testing.assert_allclose(
        out[:, 0, _MET_VAR_IDX["FSDS"]].numpy(),
        np.repeat([20.0, 40.0], step),
        rtol=1.0e-6,
    )
