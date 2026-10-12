import numpy as np
from netCDF4 import Dataset
import torch

from elm_emulator import coupled_surface_hydrology_soiltemp as coupled


def _constant_aging_tables():
    shape = (8, 31, 11)
    return {
        "tau": np.full(shape, 100.0, dtype=np.float64),
        "kappa": np.full(shape, 2.0, dtype=np.float64),
        "drdt0": np.full(shape, 5.0, dtype=np.float64),
    }


def test_torch_grain_aging_matches_numpy(monkeypatch):
    monkeypatch.setattr(coupled, "_SNICAR_AGING_TABLES", _constant_aging_tables())
    coupled._SNICAR_AGING_TABLES_TORCH.clear()

    rds = np.array([[54.526, 100.0, 0.0], [250.0, 0.0, 0.0]], dtype=np.float64)
    temp = np.array([[268.0, 265.0, 273.15], [272.0, 273.15, 273.15]], dtype=np.float64)
    grad = np.array([[20.0, 10.0, 0.0], [5.0, 0.0, 0.0]], dtype=np.float64)
    rho = np.array([[150.0, 250.0, 50.0], [350.0, 50.0, 50.0]], dtype=np.float64)
    frac_liq = np.array([[0.0, 0.02, 0.0], [0.01, 0.0, 0.0]], dtype=np.float64)
    frac_new = np.array([[0.1, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float64)
    frac_refrz = np.array([[0.0, 0.05, 0.0], [0.0, 0.0, 0.0]], dtype=np.float64)
    snl = np.array([-2, -1], dtype=np.int32)

    expected = coupled.snow_grain_aging_step(
        rds, temp, grad, rho, frac_liq, frac_new, frac_refrz, snl, 1800.0
    )
    actual = coupled.snow_grain_aging_step_torch(
        torch.as_tensor(rds),
        torch.as_tensor(temp),
        torch.as_tensor(grad),
        torch.as_tensor(rho),
        torch.as_tensor(frac_liq),
        torch.as_tensor(frac_new),
        torch.as_tensor(frac_refrz),
        torch.as_tensor(snl),
        1800.0,
    ).numpy()

    np.testing.assert_allclose(actual, expected, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_array_equal(actual[0, 2:], 0.0)
    np.testing.assert_array_equal(actual[1, 1:], 0.0)


def test_torch_grain_aging_is_inert_without_tables(monkeypatch):
    monkeypatch.setattr(coupled, "_SNICAR_AGING_TABLES", None)
    coupled._SNICAR_AGING_TABLES_TORCH.clear()
    rds = torch.tensor([[coupled.SNW_RDS_MIN]], dtype=torch.float64)
    zeros = torch.zeros_like(rds)
    actual = coupled.snow_grain_aging_step_torch(
        rds, zeros + 268.0, zeros, zeros + 150.0, zeros, zeros, zeros,
        torch.tensor([-1], dtype=torch.int32), 1800.0,
    )
    torch.testing.assert_close(actual, rds)


def test_loader_transposes_standard_elm_netcdf_order(tmp_path):
    path = tmp_path / "snicar_aging.nc"
    source = np.arange(11 * 31 * 8, dtype=np.float64).reshape(11, 31, 8)
    with Dataset(path, "w") as dataset:
        dataset.createDimension("T", 11)
        dataset.createDimension("dTdz", 31)
        dataset.createDimension("sno_dns", 8)
        for name in ("tau", "kappa", "drdsdt0"):
            variable = dataset.createVariable(name, "f8", ("T", "dTdz", "sno_dns"))
            variable[:] = source + (1.0 if name == "kappa" else 0.0)

    coupled.load_snicar_aging_tables(path)
    assert coupled._SNICAR_AGING_TABLES["tau"].shape == (8, 31, 11)
    assert coupled._SNICAR_AGING_TABLES["tau"][7, 30, 10] == source[10, 30, 7]


def test_snow_water_step_preserves_grain_radius():
    dtype = torch.float64
    layer_shape = (1, 5)
    rds = torch.zeros(layer_shape, dtype=dtype)
    rds[:, :2] = torch.tensor([250.0, 400.0], dtype=dtype)
    layers = coupled.SnowLayerState(
        snl=torch.tensor([-2], dtype=torch.int32),
        dz=torch.tensor([[0.02, 0.08, 0.0, 0.0, 0.0]], dtype=dtype),
        h2osoi_ice=torch.tensor([[4.0, 16.0, 0.0, 0.0, 0.0]], dtype=dtype),
        h2osoi_liq=torch.zeros(layer_shape, dtype=dtype),
        t_snow=torch.full(layer_shape, 270.0, dtype=dtype),
        snw_rds=rds,
    )
    state = coupled.CoupledColumnState(
        h2osoi_liq=torch.full((1, 3), 10.0, dtype=dtype),
        h2osoi_ice=torch.zeros((1, 3), dtype=dtype),
        h2ocan=torch.zeros(1, dtype=dtype),
        h2osfc=torch.zeros(1, dtype=dtype),
        frac_h2osfc=torch.zeros(1, dtype=dtype),
        frac_sno=torch.full((1,), 0.8, dtype=dtype),
        zwt=torch.ones(1, dtype=dtype),
        t_soisno=torch.full((1, 3), 272.0, dtype=dtype),
        h2osno=torch.full((1,), 20.0, dtype=dtype),
        snow_depth=torch.full((1,), 0.1, dtype=dtype),
        snow_t=layers.t_snow,
        snow_layers=layers,
        int_snow=torch.full((1,), 20.0, dtype=dtype),
    )
    result = coupled.apply_snow_water_and_capping(
        state,
        torch.zeros(1, dtype=dtype),
        device=torch.device("cpu"),
        dtype=dtype,
    )
    torch.testing.assert_close(result.state.snow_layers.snw_rds, rds)
