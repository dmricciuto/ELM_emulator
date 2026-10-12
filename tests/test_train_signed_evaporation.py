from __future__ import annotations

import numpy as np
import pytest

from elm_emulator.train_patch_models import (
    compute_metrics,
    compute_target_loss_weights,
    postprocess_predictions,
    standardize_targets_with_mask,
)


def test_signed_evaporation_targets_cannot_use_log1p_transform() -> None:
    y = np.asarray([[-1.0e-6], [2.0e-6]], dtype=np.float64)
    valid = np.ones_like(y, dtype=bool)

    with pytest.raises(ValueError, match="Signed evaporation targets cannot be trained in log1p"):
        standardize_targets_with_mask(
            y,
            valid,
            y,
            ["qflx_ev_soil"],
            log1p_target_names=["qflx_ev_soil"],
        )


def test_negative_evaporation_loss_weight_is_target_specific() -> None:
    y = np.asarray(
        [
            [-2.0e-6, 1.0e-6, -2.0e-6],
            [1.0e-6, -3.0e-6, 2.0e-6],
        ],
        dtype=np.float64,
    )
    valid = np.ones_like(y, dtype=bool)

    weights = compute_target_loss_weights(
        y,
        valid,
        ["qflx_ev_soil", "qflx_evap_canopy", "qflx_tran_veg"],
        psn_low_value_weight=1.0,
        psn_low_value_threshold=0.0,
        negative_evap_weight=6.0,
        negative_evap_threshold=1.0e-12,
    )

    np.testing.assert_allclose(weights[:, 0], [6.0, 1.0])
    np.testing.assert_allclose(weights[:, 1], [1.0, 6.0])
    np.testing.assert_allclose(weights[:, 2], [1.0, 1.0])


def test_high_lai_qsoil_loss_weight_is_target_specific() -> None:
    x = np.asarray([[2.9], [3.0], [5.0]], dtype=np.float64)
    y = np.asarray(
        [
            [1.0e-6, 10.0],
            [2.0e-6, 20.0],
            [3.0e-6, 30.0],
        ],
        dtype=np.float64,
    )
    valid = np.ones_like(y, dtype=bool)

    weights = compute_target_loss_weights(
        y,
        valid,
        ["qflx_ev_soil", "fpsn_patch"],
        psn_low_value_weight=1.0,
        psn_low_value_threshold=0.0,
        x=x,
        input_names=["elai_patch"],
        qsoil_high_lai_weight=8.0,
        qsoil_high_lai_threshold=3.0,
    )

    np.testing.assert_allclose(weights[:, 0], [1.0, 8.0, 8.0])
    np.testing.assert_allclose(weights[:, 1], [1.0, 1.0, 1.0])


def test_metrics_report_negative_evaporation_diagnostics() -> None:
    y_true = np.asarray([[-2.0e-6], [0.0], [3.0e-6]], dtype=np.float64)
    y_pred = np.asarray([[-1.0e-6], [1.0e-7], [2.0e-6]], dtype=np.float64)
    valid = np.ones_like(y_true, dtype=bool)

    metrics = compute_metrics(y_true, y_pred, valid, ["qflx_ev_soil"])

    assert metrics["negative_fraction_by_target"]["qflx_ev_soil"] == pytest.approx(1.0 / 3.0)
    assert metrics["pred_negative_fraction_by_target"]["qflx_ev_soil"] == pytest.approx(1.0 / 3.0)
    assert metrics["target_min_by_target"]["qflx_ev_soil"] == pytest.approx(-2.0e-6)
    assert metrics["pred_min_by_target"]["qflx_ev_soil"] == pytest.approx(-1.0e-6)


def test_transpiration_is_nonnegative_but_canopy_evaporation_remains_signed() -> None:
    predictions = np.asarray([[-2.0e-6, -3.0e-6]], dtype=np.float64)

    processed = postprocess_predictions(
        predictions,
        ["qflx_tran_veg", "qflx_evap_canopy"],
    )

    np.testing.assert_allclose(processed[:, 0], [0.0])
    np.testing.assert_allclose(processed[:, 1], [-3.0e-6])
