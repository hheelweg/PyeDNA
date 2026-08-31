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
    ambertools_home: Path
    pmemd_home: Path

    @property
    def home(self) -> Path:
        """Return the AmberTools root for legacy callers."""
        return self.ambertools_home

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


AMBERTOOLS_EXECUTABLES = {
    "antechamber",
    "cpptraj",
    "parmchk2",
    "prepgen",
    "resp",
    "respgen",
    "sander",
    "tleap",
}

PMEMD_EXECUTABLES = {
    "pmemd",
    "pmemd.MPI",
    "pmemd.cuda",
}


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


def _get_amber_config(data: dict) -> AmberConfig:
    amber = data.get("amber", {})

    if "ambertools_home" in amber or "pmemd_home" in amber:
        return AmberConfig(
            ambertools_home=_get_path(data, "amber", "ambertools_home"),
            pmemd_home=_get_path(data, "amber", "pmemd_home"),
        )

    home = _get_path(data, "amber", "home")
    return AmberConfig(ambertools_home=home, pmemd_home=home)


@cache
def get_config() -> PyeDNAConfig:
    if not CONFIG_PATH.is_file():
        raise RuntimeError(
            f"PyeDNA config file not found: {CONFIG_PATH}"
        )

    with CONFIG_PATH.open("rb") as file:
        data = tomllib.load(file)

    return PyeDNAConfig(
        amber=_get_amber_config(data),
        nab=NabConfig(
            home=_get_path(data, "nab", "home"),
        ),
        libraries=LibraryConfig(
            dye_dir=_get_path(data, "libraries", "dye_dir"),
            dna_dir=_get_path(data, "libraries", "dna_dir"),
            linker_dir=_get_path(data, "libraries", "linker_dir"),
        ),
    )


def _amber_executable_root(name: str) -> Path:
    config = get_config()

    if name in AMBERTOOLS_EXECUTABLES:
        return config.amber.ambertools_home
    if name in PMEMD_EXECUTABLES:
        return config.amber.pmemd_home

    known = sorted(AMBERTOOLS_EXECUTABLES | PMEMD_EXECUTABLES)
    raise RuntimeError(
        f"Unknown Amber executable {name!r}. "
        f"Add it to the PyeDNA executable map if it is required. "
        f"Known executables: {', '.join(known)}"
    )


def amber_executable(name: str) -> Path:
    path = _amber_executable_root(name) / "bin" / name

    if not path.is_file():
        raise RuntimeError(f"Amber executable not found: {path}")

    return path


def amber_data_path(*parts: str) -> Path:
    config = get_config()
    path = config.amber.ambertools_home.joinpath(*parts)

    if not path.exists():
        raise RuntimeError(f"AmberTools data file not found: {path}")

    return path


def _prepend_env_path(env: dict[str, str], key: str, path: Path) -> None:
    current = env.get(key)
    env[key] = f"{path}{':' + current if current else ''}"


def _unique_paths(paths: tuple[Path, ...]) -> tuple[Path, ...]:
    unique = []
    for path in paths:
        if path not in unique:
            unique.append(path)
    return tuple(unique)


def amber_environment(name: str | None = None) -> dict[str, str]:
    config = get_config()
    ambertools_home = config.amber.ambertools_home
    pmemd_home = config.amber.pmemd_home

    env = os.environ.copy()

    if name is not None and name in PMEMD_EXECUTABLES:
        env["PMEMDHOME"] = str(pmemd_home)
        env["AMBERHOME"] = str(ambertools_home)
        roots = _unique_paths((ambertools_home, pmemd_home))
        for root in roots:
            _prepend_env_path(env, "LD_LIBRARY_PATH", root / "lib")
        for root in roots:
            _prepend_env_path(env, "PATH", root / "bin")
    else:
        env["AMBERHOME"] = str(ambertools_home)
        _prepend_env_path(env, "PATH", ambertools_home / "bin")
        _prepend_env_path(env, "LD_LIBRARY_PATH", ambertools_home / "lib")

    return env
