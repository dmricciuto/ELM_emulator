from __future__ import annotations

import numpy as np

from elm_emulator.regional_plotting import _filter_regional_plot_diag_maps


def test_filter_regional_plot_maps_excludes_profiles_and_nested_outputs() -> None:
    diagnostics = {
        "t_soisno_lev3_mean": np.ones(4, dtype=np.float32),
        "elm_h0_TSOI": np.ones((4, 15), dtype=np.float32),
        "_daily": {"t_soil_mean": np.ones((2, 4), dtype=np.float32)},
        "h2osoi_liqvol_lev1_mean": np.ones(4, dtype=np.float32),
    }

    filtered = _filter_regional_plot_diag_maps(diagnostics, None)

    assert list(filtered) == ["t_soisno_lev3_mean", "h2osoi_liqvol_lev1_mean"]


def test_filter_regional_plot_maps_applies_requested_names_after_shape_filter() -> None:
    diagnostics = {
        "t_soisno_lev3_mean": np.ones(4, dtype=np.float32),
        "elm_h0_TSOI": np.ones((4, 15), dtype=np.float32),
        "lai_mean": np.ones(4, dtype=np.float32),
    }

    filtered = _filter_regional_plot_diag_maps(
        diagnostics,
        ["elm_h0_TSOI", "lai_mean"],
    )

    assert list(filtered) == ["lai_mean"]


def test_filter_regional_plot_maps_includes_selected_monthly_carbon_pools() -> None:
    diagnostics = {
        "gpp_gc_m2_d_mean": np.ones(4, dtype=np.float32),
    }
    monthly_diagnostics = {
        "carbon_totvegc": np.full((1, 4), 2000.0, dtype=np.float32),
        "carbon_totsomc": np.full((1, 4), 5000.0, dtype=np.float32),
        "profile_not_a_map": np.ones((1, 4, 10), dtype=np.float32),
    }

    filtered = _filter_regional_plot_diag_maps(
        diagnostics,
        ["gpp_gc_m2_d_mean", "carbon_totvegc", "carbon_totsomc"],
        monthly_diag_maps=monthly_diagnostics,
    )

    assert list(filtered) == [
        "gpp_gc_m2_d_mean",
        "carbon_totvegc",
        "carbon_totsomc",
    ]
    np.testing.assert_array_equal(filtered["carbon_totvegc"], monthly_diagnostics["carbon_totvegc"][0])
