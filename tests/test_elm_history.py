from __future__ import annotations

import numpy as np
from netCDF4 import Dataset

from elm_emulator.elm_history import ElmHistoryWriter, infer_elm_history_case_name


def test_infer_elm_history_case_name_from_olmt_run_dir(tmp_path):
    run_dir = tmp_path / "my_case" / "run"
    assert infer_elm_history_case_name(run_dir) == "my_case"
    assert infer_elm_history_case_name(tmp_path / "plain_output") == "plain_output"


def test_elm_history_writer_creates_h0_file(tmp_path):
    surf = {
        "lat_deg": np.asarray([31.25, 31.75], dtype=np.float32),
        "lon_deg": np.asarray([-99.25, -99.75], dtype=np.float32),
        "map_lat_centers": np.asarray([31.25, 31.75], dtype=np.float32),
        "map_lon_centers": np.asarray([-99.75, -99.25], dtype=np.float32),
    }
    diag = {
        "gpp_gc_m2_d_mean": np.asarray([[1.728, 0.864]], dtype=np.float32),
        "qflx_evap_tot_mean": np.asarray([[1.0e-5, 2.0e-5]], dtype=np.float32),
        "eflx_sh_tot_mean": np.asarray([[25.0, 30.0]], dtype=np.float32),
        "forc_q_mean": np.asarray([[0.010, 0.012]], dtype=np.float32),
        "forc_pbot_mean": np.asarray([[99000.0, 98000.0]], dtype=np.float32),
        "forc_lwrad_mean": np.asarray([[360.0, 355.0]], dtype=np.float32),
        "forc_wind_mean": np.asarray([[3.0, 4.0]], dtype=np.float32),
        "rh2m_mean": np.asarray([[55.0, 60.0]], dtype=np.float32),
        "fsa_mean": np.asarray([[180.0, 170.0]], dtype=np.float32),
        "fsr_mean": np.asarray([[20.0, 30.0]], dtype=np.float32),
        "sabg_mean": np.asarray([[90.0, 95.0]], dtype=np.float32),
        "sabv_mean": np.asarray([[90.0, 75.0]], dtype=np.float32),
        "albd_mean": np.asarray([[[0.10, 0.25], [0.12, 0.27]]], dtype=np.float32),
        "elm_h0_TSOI": np.full((1, 2, 15), 285.0, dtype=np.float32),
        "elm_h0_H2OSOI": np.full((1, 2, 10), 0.25, dtype=np.float32),
    }

    writer = ElmHistoryWriter(
        tmp_path / "case_a" / "run",
        case_name="case_a",
        calendar="noleap",
    )
    paths = writer.write(
        surf=surf,
        diag=diag,
        period_index=[(1980, 1, 1, 31)],
    )

    assert len(paths) == 1
    assert paths[0].name == "case_a.elm.h0.1980-01-01-00000.nc"
    with Dataset(paths[0]) as ds:
        assert len(ds.dimensions["time"]) == 1
        assert len(ds.dimensions["lndgrid"]) == 2
        assert ds.variables["GPP"].shape == (1, 2)
        np.testing.assert_allclose(ds.variables["GPP"][0, :], [2.0e-5, 1.0e-5])
        np.testing.assert_allclose(ds.variables["QFLX_EVAP_TOT"][0, :], [1.0e-5, 2.0e-5])
        np.testing.assert_allclose(ds.variables["QBOT"][0, :], [0.010, 0.012])
        np.testing.assert_allclose(ds.variables["PBOT"][0, :], [99000.0, 98000.0])
        np.testing.assert_allclose(ds.variables["FLDS"][0, :], [360.0, 355.0])
        np.testing.assert_allclose(ds.variables["WIND"][0, :], [3.0, 4.0])
        np.testing.assert_allclose(ds.variables["RH2M"][0, :], [55.0, 60.0])
        np.testing.assert_allclose(ds.variables["FSA"][0, :], [180.0, 170.0])
        np.testing.assert_allclose(ds.variables["FSR"][0, :], [20.0, 30.0])
        np.testing.assert_allclose(ds.variables["SABG"][0, :], [90.0, 95.0])
        np.testing.assert_allclose(ds.variables["SABV"][0, :], [90.0, 75.0])
        assert ds.variables["ALBD"].dimensions == ("time", "numrad", "lndgrid")
        np.testing.assert_allclose(ds.variables["ALBD"][0, :, :], [[0.10, 0.12], [0.25, 0.27]])
        np.testing.assert_allclose(ds.variables["TSOI"][0, :, :], 285.0)
        np.testing.assert_allclose(ds.variables["H2OSOI"][0, :10, :], 0.25)
        np.testing.assert_allclose(ds.variables["H2OSOI"][0, 10:, :], 0.0)
        assert int(ds.variables["mcdate"][0]) == 19800201


def test_elm_history_writer_hourly_periods(tmp_path):
    surf = {
        "lat_deg": np.asarray([35.25], dtype=np.float32),
        "lon_deg": np.asarray([-83.25], dtype=np.float32),
    }
    diag = {
        "tg_mean": np.asarray([[283.0], [284.0]], dtype=np.float32),
        "eflx_fgr12_mean": np.asarray([[12.0], [13.0]], dtype=np.float32),
        "fsa_mean": np.asarray([[100.0], [110.0]], dtype=np.float32),
        "sabg_mean": np.asarray([[40.0], [45.0]], dtype=np.float32),
        "sabv_mean": np.asarray([[60.0], [65.0]], dtype=np.float32),
        "elm_h0_TSOI": np.full((2, 1, 15), 280.0, dtype=np.float32),
        "elm_h0_H2OSOI": np.full((2, 1, 10), 0.30, dtype=np.float32),
    }

    writer = ElmHistoryWriter(
        tmp_path / "hourly_case" / "run",
        case_name="hourly_case",
        calendar="noleap",
    )
    paths = writer.write(
        surf=surf,
        diag=diag,
        period_index=[
            (2012, 7, 1, 0, 3600),
            (2012, 7, 1, 3600, 3600),
        ],
    )

    with Dataset(paths[0]) as ds:
        assert len(ds.dimensions["time"]) == 2
        assert ds.variables["time"].units == "days since 2012-01-01 00:00:00"
        np.testing.assert_allclose(ds.variables["time"][:], [181.0 + 1.0 / 24.0, 181.0 + 2.0 / 24.0])
        np.testing.assert_allclose(np.diff(ds.variables["time"][:]), [1.0 / 24.0])
        np.testing.assert_allclose(ds.variables["time_bounds"][:, 1] - ds.variables["time_bounds"][:, 0], 1.0 / 24.0)
        np.testing.assert_array_equal(ds.variables["mcdate"][:], [20120701, 20120701])
        np.testing.assert_array_equal(ds.variables["mcsec"][:], [3600, 7200])
        np.testing.assert_array_equal(ds.variables["nstep"][:], [4345, 4346])
        np.testing.assert_allclose(ds.variables["TG"][:, 0], [283.0, 284.0])
        np.testing.assert_allclose(ds.variables["FGR"][:, 0], [12.0, 13.0])
        np.testing.assert_allclose(ds.variables["SABG"][:, 0] + ds.variables["SABV"][:, 0], ds.variables["FSA"][:, 0])
