import os
from dataclasses import dataclass
from functools import cache
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


CONFIG_PATH = Path.home() / ".config" / "pyedna" / "config.toml"



@dataclass(frozen=True)
class AmberConfig:
    home: Path

@dataclass(frozen=True)
class NabConfig:
    home: Path

@dataclass(frozen=True)
class LibraryConfig:
    dye_dir: Path
    dna_dir: Path
    linker_dir: Path


@dataclass(frozen=True)
class PyeDNAConfig:
    amber: AmberConfig
    nab: NabConfig
    libraries: LibraryConfig


def _get_path(data: dict, section: str, key: str) -> Path:
    try:
        value = data[section][key]
    except KeyError as exc:
        raise RuntimeError(
            f"Missing '{section}.{key}' in PyeDNA config: {CONFIG_PATH}"
        ) from exc

    path = Path(value).expanduser()

    if not path.exists():
        raise RuntimeError(
            f"PyeDNA config path does not exist: {section}.{key} = {path}"
        )

    return path


@cache
def get_config() -> PyeDNAConfig:
    if not CONFIG_PATH.is_file():
        raise RuntimeError(
            f"PyeDNA config file not found: {CONFIG_PATH}"
        )

    with CONFIG_PATH.open("rb") as file:
        data = tomllib.load(file)

    return PyeDNAConfig(
        amber=AmberConfig(
            home=_get_path(data, "amber", "home"),
        ),
        nab=NabConfig(
            home=_get_path(data, "nab", "home"),
        ),
        libraries=LibraryConfig(
            dye_dir=_get_path(data, "libraries", "dye_dir"),
            dna_dir=_get_path(data, "libraries", "dna_dir"),
            linker_dir=_get_path(data, "libraries", "linker_dir"),
        ),
    )


def amber_executable(name: str) -> Path:
    path = get_config().amber.home / "bin" / name

    if not path.is_file():
        raise RuntimeError(f"Amber executable not found: {path}")

    return path


def amber_environment() -> dict[str, str]:
    config = get_config()
    amber_home = config.amber.home

    env = os.environ.copy()
    env["AMBERHOME"] = str(amber_home)
    env["PATH"] = f"{amber_home / 'bin'}:{env.get('PATH', '')}"
    env["LD_LIBRARY_PATH"] = (
        f"{amber_home / 'lib'}"
        f"{':' + env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else ''}"
    )

    return env