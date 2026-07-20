"""Top-level command dispatcher for the ELM emulator package."""

from __future__ import annotations

import sys
from collections.abc import Callable


COMMANDS: dict[str, tuple[str, str]] = {
    "replay": ("elm_emulator.benchmark_replay", "Run site, regional, or clone-grid replay"),
    "train": ("elm_emulator.train_patch_models", "Train canopy-flux patch models"),
    "coupled-benchmark": (
        "elm_emulator.coupled_surface_hydrology_soiltemp",
        "Run the synthetic coupled surface/hydrology/soil-temperature benchmark",
    ),
}


def _usage() -> str:
    rows = "\n".join(f"  {name:<17} {desc}" for name, (_, desc) in COMMANDS.items())
    return (
        "Usage: elm-emulator <command> [args]\n\n"
        "Commands:\n"
        f"{rows}\n\n"
        "Examples:\n"
        "  elm-emulator replay --regional --help\n"
        "  elm-emulator train --help\n"
    )


def _load_main(module_name: str) -> Callable[[], None]:
    module = __import__(module_name, fromlist=["main"])
    return module.main


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        print(_usage())
        return

    command = sys.argv[1]
    if command not in COMMANDS:
        raise SystemExit(f"Unknown command {command!r}.\n\n{_usage()}")

    module_name, _ = COMMANDS[command]
    sys.argv = [f"elm-emulator {command}", *sys.argv[2:]]
    _load_main(module_name)()


if __name__ == "__main__":
    main()
