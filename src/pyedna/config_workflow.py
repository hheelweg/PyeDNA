"""Runtime-configuration CLI workflow."""

import os
import shutil
from pathlib import Path

from pyedna.config import (
    AMBERTOOLS_EXECUTABLES,
    CONFIG_PATH,
    PMEMD_EXECUTABLES,
    amber_data_path,
    amber_executable,
    get_config,
)


CONFIG_TEMPLATE = """[amber]
ambertools_home = "/path/to/ambertools26"
pmemd_home = "/path/to/pmemd26"

[nab]
home = "/path/to/AmberClassic"

[libraries]
dye_dir = "/path/to/dye_library"
dna_dir = "/path/to/dna_library"
linker_dir = "/path/to/linker_library"
"""


def run_config(command: str) -> None:
    if command == "init":
        init_config()
    elif command == "show":
        show_config()
    elif command == "check":
        check_config()
    else:
        raise RuntimeError(f"Unknown config command: {command}")


def init_config() -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)

    if CONFIG_PATH.exists():
        print(f"PyeDNA config already exists: {CONFIG_PATH}")
        print("Leaving existing config unchanged.")
        return

    CONFIG_PATH.write_text(CONFIG_TEMPLATE)
    print(f"Created PyeDNA config: {CONFIG_PATH}")


def show_config() -> None:
    config = get_config()

    print(f"config.path = {CONFIG_PATH}")
    print(f"amber.ambertools_home = {config.amber.ambertools_home}")
    print(f"amber.pmemd_home = {config.amber.pmemd_home}")
    print(f"nab.home = {config.nab.home}")
    print(f"libraries.dye_dir = {config.libraries.dye_dir}")
    print(f"libraries.dna_dir = {config.libraries.dna_dir}")
    print(f"libraries.linker_dir = {config.libraries.linker_dir}")


def check_config() -> None:
    failures = []

    def check(label: str, passed: bool, detail: str = "") -> None:
        status = "PASS" if passed else "FAIL"
        suffix = f" - {detail}" if detail else ""
        print(f"{status} {label}{suffix}")
        if not passed:
            failures.append(label)

    check("CONFIG_PATH exists", CONFIG_PATH.is_file(), str(CONFIG_PATH))

    try:
        config = get_config()
    except RuntimeError as exc:
        check("get_config() succeeds", False, str(exc))
        raise RuntimeError("PyeDNA config check failed.") from exc

    check("get_config() succeeds", True)
    check(
        "amber.ambertools_home exists",
        config.amber.ambertools_home.exists(),
        str(config.amber.ambertools_home),
    )
    check(
        "amber.pmemd_home exists",
        config.amber.pmemd_home.exists(),
        str(config.amber.pmemd_home),
    )
    check("nab.home exists", config.nab.home.exists(), str(config.nab.home))
    check(
        "libraries.dye_dir exists",
        config.libraries.dye_dir.exists(),
        str(config.libraries.dye_dir),
    )
    check(
        "libraries.dna_dir exists",
        config.libraries.dna_dir.exists(),
        str(config.libraries.dna_dir),
    )
    check(
        "libraries.linker_dir exists",
        config.libraries.linker_dir.exists(),
        str(config.libraries.linker_dir),
    )
    check(
        "<nab.home>/bin/nab exists",
        (config.nab.home / "bin" / "nab").is_file(),
    )
    for name in ("antechamber", "parmchk2", "resp", "respgen", "sander", "tleap"):
        try:
            executable = amber_executable(name)
        except RuntimeError as exc:
            check(f"AmberTools executable {name} exists", False, str(exc))
        else:
            check(f"AmberTools executable {name} exists", True, str(executable))
    for name in ("pmemd",):
        try:
            executable = amber_executable(name)
        except RuntimeError as exc:
            check(f"pmemd executable {name} exists", False, str(exc))
        else:
            check(f"pmemd executable {name} exists", True, str(executable))
    for label, parts in (
        ("AmberTools DNA.OL15.lib exists", ("dat", "leap", "lib", "DNA.OL15.lib")),
        ("AmberTools leaprc.DNA.OL15 exists", ("dat", "leap", "cmd", "leaprc.DNA.OL15")),
    ):
        try:
            data_file = amber_data_path(*parts)
        except RuntimeError as exc:
            check(label, False, str(exc))
        else:
            check(label, True, str(data_file))
    optional = sorted(
        name
        for name in (AMBERTOOLS_EXECUTABLES | PMEMD_EXECUTABLES)
        if name not in {"antechamber", "parmchk2", "pmemd", "resp", "respgen", "sander", "tleap"}
    )
    for name in optional:
        try:
            executable = amber_executable(name)
        except RuntimeError:
            continue
        else:
            check(f"optional Amber executable {name} exists", True, str(executable))
    check("gcc is available", shutil.which("gcc") is not None)

    conda_prefix = os.environ.get("CONDA_PREFIX")
    conda_path = Path(conda_prefix) if conda_prefix else None
    check(
        "CONDA_PREFIX exists",
        conda_path is not None and conda_path.exists(),
        conda_prefix or "",
    )
    check(
        "$CONDA_PREFIX/lib/libgfortran.so exists",
        conda_path is not None
        and (conda_path / "lib" / "libgfortran.so").is_file(),
    )

    if failures:
        raise RuntimeError(
            "PyeDNA config check failed: " + ", ".join(failures)
        )
