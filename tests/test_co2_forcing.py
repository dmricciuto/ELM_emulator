from __future__ import annotations

from pathlib import Path

import numpy as np
from netCDF4 import Dataset

from elm_emulator.benchmark_replay import (
    _DEFAULT_CO2_FILE_RELATIVE,
    _co2_ppm_from_file,
    _default_co2_file_for_met_dir,
)


def test_default_co2_file_is_inferred_from_inputdata_root() -> None:
    met_dir = Path("/tmp/inputdata/atm/datm7/atm_forcing.example/cpl_bypass_full")
    assert _default_co2_file_for_met_dir(met_dir) == Path("/tmp/inputdata") / _DEFAULT_CO2_FILE_RELATIVE


def test_co2_stream_uses_month_midpoint_interpolation(tmp_path: Path) -> None:
    path = tmp_path / "co2.nc"
    with Dataset(path, "w") as ds:
        ds.createDimension("time", 3)
        ds.createDimension("lat", 1)
        ds.createDimension("lon", 1)
        date = ds.createVariable("date", "i4", ("time",))
        date[:] = np.asarray([20000116, 20000214, 20000316], dtype=np.int32)
        co2 = ds.createVariable("CO2", "f4", ("time", "lat", "lon"))
        co2[:, 0, 0] = np.asarray([370.0, 371.0, 372.0], dtype=np.float32)

    assert _co2_ppm_from_file(path, 2000, 1, calendar="noleap") == 370.0
    assert _co2_ppm_from_file(path, 2000, 2, calendar="noleap") == 371.0
