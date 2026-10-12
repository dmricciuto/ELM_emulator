import numpy as np
import torch

from elm_emulator.coupled_surface_hydrology_soiltemp import (
    ELM_HYDROLOGY_LEVELS,
    build_texture_static_parameters,
)
from elm_emulator.elm_hydrology import SoilTextureProfile, elm_soil_grid
from elm_emulator.elm_hydrology_torch import drainage_torch, water_table_torch


def test_static_builder_keeps_bedrock_out_of_hydrology() -> None:
    dtype = torch.float64
    device = torch.device("cpu")
    dz, z, zi = elm_soil_grid(15)
    texture = SoilTextureProfile(
        sand_pct=np.full(15, 40.0),
        clay_pct=np.full(15, 20.0),
        organic_frac=np.zeros(15),
    )
    static = build_texture_static_parameters(
        texture,
        rootfr=np.full(15, 1.0 / 15.0),
        dz=dz,
        z=z,
        zi=zi,
        device=device,
        dtype=dtype,
    )

    assert static.hydrology_numpy.dz.shape[-1] == ELM_HYDROLOGY_LEVELS
    assert static.hydrology_torch.dz.shape[-1] == ELM_HYDROLOGY_LEVELS
    assert static.rootfr.shape[-1] == ELM_HYDROLOGY_LEVELS
    assert static.soil_temperature_numpy.dz.shape[-1] == 15
    assert static.soil_temperature_torch.dz.shape[-1] == 15
    np.testing.assert_allclose(static.rootfr.sum(), 1.0)


def test_falling_water_table_starts_in_its_current_layer() -> None:
    dtype = torch.float64
    zwt = torch.tensor([0.30], dtype=dtype)
    z = torch.tensor([0.10, 0.30, 0.60], dtype=dtype)
    zi = torch.tensor([0.20, 0.45, 0.80], dtype=dtype)
    watsat = torch.tensor([0.42, 0.35, 0.28], dtype=dtype)
    sucsat = torch.tensor([80.0, 160.0, 320.0], dtype=dtype)
    bsw = torch.tensor([3.0, 5.0, 8.0], dtype=dtype)
    dt = 100.0
    qcharge = torch.tensor([-1.0e-4], dtype=dtype)

    updated, _ = water_table_torch(
        zwt=zwt,
        qcharge=qcharge,
        z=z,
        zi=zi,
        watsat=watsat,
        sucsat=sucsat,
        bsw=bsw,
        dt=dt,
        wa=torch.tensor([4000.0], dtype=dtype),
    )

    # zwt=0.30 lies in zero-based layer 1.  Native ELM's falling-table loop
    # consumes the charge in that layer before considering layer 2.
    sy = watsat[1] * (1.0 - (1.0 + 1.0e3 * zwt[0] / sucsat[1]) ** (-1.0 / bsw[1]))
    sy = torch.clamp(sy, min=0.02)
    expected = zwt[0] - (qcharge[0] * dt) / sy / 1000.0
    torch.testing.assert_close(updated[0], expected)


def test_drainage_removes_water_from_current_water_table_layer_first() -> None:
    dtype = torch.float64
    hliq = torch.tensor([[20.0, 20.0, 20.0]], dtype=dtype)
    hice = torch.zeros_like(hliq)
    dz = torch.tensor([0.20, 0.25, 0.35], dtype=dtype)
    z = torch.tensor([0.10, 0.30, 0.60], dtype=dtype)
    zi = torch.tensor([0.20, 0.45, 0.80], dtype=dtype)
    watsat = torch.tensor([0.42, 0.35, 0.28], dtype=dtype)
    sucsat = torch.tensor([80.0, 160.0, 320.0], dtype=dtype)
    bsw = torch.tensor([3.0, 5.0, 8.0], dtype=dtype)

    hliq_out, _, _, qdrain, _ = drainage_torch(
        h2osoi_liq=hliq,
        h2osoi_ice=hice,
        h2osfc=torch.zeros(1, dtype=dtype),
        zwt=torch.tensor([0.30], dtype=dtype),
        z=z,
        zi=zi,
        dz=dz,
        watsat=watsat,
        sucsat=sucsat,
        bsw=bsw,
        eff_porosity=watsat[None, :],
        fdrain=torch.tensor([2.5], dtype=dtype),
        topo_slope=torch.tensor([1.0], dtype=dtype),
        dt=10.0,
        wa=torch.tensor([4000.0], dtype=dtype),
    )

    assert qdrain[0] > 0.0
    torch.testing.assert_close(hliq_out[0, 0], hliq[0, 0])
    assert hliq_out[0, 1] < hliq[0, 1]
    torch.testing.assert_close(hliq_out[0, 2], hliq[0, 2])


def test_bottom_soil_layer_is_not_treated_as_direct_aquifer() -> None:
    dtype = torch.float64
    hliq = torch.tensor([[20.0, 20.0, 20.0]], dtype=dtype)
    dz = torch.tensor([0.20, 0.25, 0.35], dtype=dtype)
    z = torch.tensor([0.10, 0.30, 0.60], dtype=dtype)
    zi = torch.tensor([0.20, 0.45, 0.80], dtype=dtype)
    watsat = torch.tensor([0.42, 0.35, 0.28], dtype=dtype)
    sucsat = torch.tensor([80.0, 160.0, 320.0], dtype=dtype)
    bsw = torch.tensor([3.0, 5.0, 8.0], dtype=dtype)
    wa = torch.tensor([4000.0], dtype=dtype)

    hliq_out, _, _, qdrain, wa_out = drainage_torch(
        h2osoi_liq=hliq,
        h2osoi_ice=torch.zeros_like(hliq),
        h2osfc=torch.zeros(1, dtype=dtype),
        # Between the second and third interfaces: inside bottom layer 3.
        zwt=torch.tensor([0.60], dtype=dtype),
        z=z,
        zi=zi,
        dz=dz,
        watsat=watsat,
        sucsat=sucsat,
        bsw=bsw,
        eff_porosity=watsat[None, :],
        fdrain=torch.tensor([2.5], dtype=dtype),
        topo_slope=torch.tensor([1.0], dtype=dtype),
        dt=10.0,
        wa=wa,
    )

    assert qdrain[0] > 0.0
    assert hliq_out[0, 2] < hliq[0, 2]
    torch.testing.assert_close(wa_out, wa)


def test_origflag0_uses_native_global_cap_fallback() -> None:
    dtype = torch.float64
    dz = torch.tensor([0.20, 0.25, 0.35], dtype=dtype)
    zi = torch.tensor([0.20, 0.45, 0.80], dtype=dtype)
    watsat = torch.tensor([0.42, 0.35, 0.28], dtype=dtype)
    sucsat = torch.tensor([80.0, 160.0, 320.0], dtype=dtype)
    bsw = torch.tensor([3.0, 5.0, 8.0], dtype=dtype)
    slope = torch.tensor([0.72217597], dtype=dtype)
    zwt = torch.tensor([1.0], dtype=dtype)

    _, _, _, qdrain, _ = drainage_torch(
        h2osoi_liq=torch.full((1, 3), 20.0, dtype=dtype),
        h2osoi_ice=torch.zeros((1, 3), dtype=dtype),
        h2osfc=torch.zeros(1, dtype=dtype),
        zwt=zwt,
        z=torch.tensor([0.10, 0.30, 0.60], dtype=dtype),
        zi=zi,
        dz=dz,
        watsat=watsat,
        sucsat=sucsat,
        bsw=bsw,
        eff_porosity=watsat[None, :],
        fdrain=torch.tensor([2.5], dtype=dtype),
        topo_slope=slope,
        dt=10.0,
        wa=torch.tensor([4000.0], dtype=dtype),
    )
    expected = 10.0 * torch.sin(torch.deg2rad(slope)) * torch.exp(-2.5 * zwt)
    torch.testing.assert_close(qdrain, expected)


def test_origflag0_applies_depth_weighted_ice_impedance() -> None:
    dtype = torch.float64
    dz = torch.tensor([0.20, 0.25, 0.35], dtype=dtype)
    zi = torch.tensor([0.20, 0.45, 0.80], dtype=dtype)
    watsat = torch.tensor([0.42, 0.35, 0.28], dtype=dtype)
    icefrac = 0.2
    hice = (icefrac * watsat * dz * 917.0)[None, :]
    common = dict(
        h2osoi_liq=torch.full((1, 3), 20.0, dtype=dtype),
        h2osoi_ice=hice,
        h2osfc=torch.zeros(1, dtype=dtype),
        zwt=torch.tensor([1.0], dtype=dtype),
        z=torch.tensor([0.10, 0.30, 0.60], dtype=dtype),
        zi=zi,
        dz=dz,
        watsat=watsat,
        sucsat=torch.tensor([80.0, 160.0, 320.0], dtype=dtype),
        bsw=torch.tensor([3.0, 5.0, 8.0], dtype=dtype),
        eff_porosity=watsat[None, :] - hice / (dz * 917.0),
        fdrain=torch.tensor([2.5], dtype=dtype),
        topo_slope=torch.tensor([1.0], dtype=dtype),
        dt=10.0,
        wa=torch.tensor([4000.0], dtype=dtype),
    )
    _, _, _, qdrain, _ = drainage_torch(**common)
    unfrozen = 10.0 * torch.sin(torch.deg2rad(common["topo_slope"])) * torch.exp(
        -2.5 * common["zwt"]
    )
    torch.testing.assert_close(qdrain, unfrozen * 10.0 ** (-5.0 * icefrac))


def test_aquifer_residual_is_included_in_reported_drainage() -> None:
    dtype = torch.float64
    dz = torch.tensor([0.20, 0.25, 0.35], dtype=dtype)
    zi = torch.tensor([0.20, 0.45, 0.80], dtype=dtype)
    watsat = torch.tensor([0.42, 0.35, 0.28], dtype=dtype)
    wa = torch.tensor([4000.0], dtype=dtype)
    zwt = torch.tensor([0.60], dtype=dtype)
    slope = torch.tensor([10.0], dtype=dtype)
    dt = 3600.0
    _, _, _, qdrain, wa_out = drainage_torch(
        h2osoi_liq=torch.full((1, 3), 100.0, dtype=dtype),
        h2osoi_ice=torch.zeros((1, 3), dtype=dtype),
        h2osfc=torch.zeros(1, dtype=dtype),
        zwt=zwt,
        z=torch.tensor([0.10, 0.30, 0.60], dtype=dtype),
        zi=zi,
        dz=dz,
        watsat=watsat,
        sucsat=torch.tensor([80.0, 160.0, 320.0], dtype=dtype),
        bsw=torch.tensor([3.0, 5.0, 8.0], dtype=dtype),
        eff_porosity=watsat[None, :],
        fdrain=torch.tensor([2.5], dtype=dtype),
        topo_slope=slope,
        dt=dt,
        wa=wa,
    )
    expected = 10.0 * torch.sin(torch.deg2rad(slope)) * torch.exp(-2.5 * zwt)
    torch.testing.assert_close(qdrain, expected)
    assert wa_out[0] < wa[0]
