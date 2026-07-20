# Architecture

The package is organized around three reusable workflows:

1. `benchmark_replay.py`
   - site, regional, clone-grid, treatment, restart, and NetCDF-output replay
2. `train_patch_models.py`
   - canopy-flux MLP training and artifact writing
3. `coupled_surface_hydrology_soiltemp.py`
   - coupled surface surrogate, hydrology, soil temperature, snow, and optional carbon mechanics

Runtime support modules are intentionally small in number:

- `elm_hydrology.py`
- `elm_hydrology_torch.py`
- `elm_soil_temperature.py`
- `elm_soil_temperature_torch.py`
- `elm_carbon_torch.py`
- `btran_utils.py`
- `paths.py`

The old `surface_emulator` directory contained many analysis-specific scripts for US-MOz
footprints, BACI inversions, plotting, and ad hoc diagnostics. Those were not copied into
this repository because they are downstream experiments, not core replay infrastructure.

## Boundaries

Core modules should remain importable without relying on sibling-script side effects. New
analysis scripts should live outside this package or under a clearly separate `experiments/`
directory if they become necessary.

Model artifacts live under `artifacts/canopyflux_models/`. Large forcing/training files should
stay external and be passed by path.
