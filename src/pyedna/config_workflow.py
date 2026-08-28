"""Runtime-configuration CLI workflow."""

import os
import shutil
from pathlib import Path

from pyedna.config import CONFIG_PATH, get_config


CONFIG_TEMPLATE = """[amber]
home = "/path/to/amber"

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
    print(f"amber.home = {config.amber.home}")
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
    check("amber.home exists", config.amber.home.exists(), str(config.amber.home))
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
