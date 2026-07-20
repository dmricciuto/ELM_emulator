from __future__ import annotations


def test_core_modules_import() -> None:
    import elm_emulator.benchmark_replay  # noqa: F401
    import elm_emulator.coupled_surface_hydrology_soiltemp  # noqa: F401
    import elm_emulator.train_patch_models  # noqa: F401


def test_cli_commands_registered() -> None:
    from elm_emulator.cli import COMMANDS

    assert {"replay", "train", "coupled-benchmark"} <= set(COMMANDS)
