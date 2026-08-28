# Installation

PyeDNA is a Python package that orchestrates external scientific software. Installing PyeDNA installs the Python dependencies declared in `pyproject.toml`; it does not install AmberTools, Amber/pmemd, AmberClassic/NAB, HADDOCK3, ACPYPE, CUDA, or GPU4PySCF.

> **Important**
>
> The current PyeDNA quantum trajectory workflow requires GPU4PySCF. A CPU-only execution path is not currently implemented for that workflow. Users intending to run quantum trajectory analysis must install the validated CUDA/CuPy/GPU4PySCF stack and run on a compatible NVIDIA GPU.

Component parameterization, HADDOCK docking, MD, and quantum analysis can be computationally expensive. Submit those workflows only on appropriate compute resources.

## 1. PyeDNA Python Package

PyeDNA currently requires Python `>=3.12`, as declared in `pyproject.toml`.

For normal scientific use, the intended released-user installation is:

```bash
pip install pyedna
```

For local or pre-release testing from a built wheel:

```bash
pip install pyedna-<version>-py3-none-any.whl
```

Developer collaborators who intend to modify PyeDNA should use an editable install from a source checkout:

```bash
git clone <repository>
cd PyeDNA
pip install -e .
```

Editable installation links the active Python environment to the checkout, so code changes are reflected immediately. Scientific collaborators who only run PyeDNA workflows should use the normal package or wheel installation path.

The package dependencies include `pyscf==2.8.0`. PyeDNA pins this version because the GPU4PySCF stack currently used by the project is known to be compatible with it.

## 2. External Scientific Software

The following runtime dependencies must be installed separately and configured for the machine or cluster.

| Dependency | Role in PyeDNA |
| --- | --- |
| AmberTools 26-compatible installation | Small-molecule parameterization, RESP fitting, `tleap` Amber setup, minimization, and AmberTools data files. |
| Amber / pmemd26-compatible installation | `pmemd` and `pmemd.cuda` executables for MD equilibration and production. |
| AmberClassic / NAB | Generated DNA structures when `dna.source = "generate"`. |
| HADDOCK3 | Structure docking after PyeDNA writes `docking_config.cfg`. |
| ACPYPE | Conversion of dye-linker components into HADDOCK/CNS topology and parameter files. |
| NVIDIA/CUDA environment | Required for the current GPU4PySCF quantum trajectory workflow and for GPU Amber MD with `pmemd.cuda`. |

PyeDNA currently resolves AmberTools executables from `amber.ambertools_home` and pmemd executables from `amber.pmemd_home` in the runtime config:

```toml
[amber]
ambertools_home = "/path/to/ambertools26"
pmemd_home = "/path/to/pmemd26"
```

The configured AmberTools installation is expected to provide the executables currently invoked by PyeDNA workflows: `antechamber`, `parmchk2`, `resp`, `respgen`, `sander`, and `tleap`.

The configured Amber/pmemd installation is expected to provide `pmemd` and `pmemd.cuda`.

PyeDNA resolves AmberTools data from `ambertools_home`. It directly reads `dat/leap/lib/DNA.OL15.lib` for OL15 reference atom charges during linker RESP restraint generation, and `tleap` loads DNA force-field data through `leaprc.DNA.OL15`. Users do not configure individual OL15 data files such as `frcmod.DNA.OL15` as separate PyeDNA runtime settings.

## 3. GPU Python Dependencies

The currently validated GPU Python stack is:

```text
cupy-cuda11x==13.4.1
gpu4pyscf-cuda11x==1.4.3
pyscf==2.8.0
```

`pyscf==2.8.0` is installed as a PyeDNA package dependency. CuPy and GPU4PySCF are currently installed separately in the active environment.

The current component ESP workflow imports GPU4PySCF for electrostatic-potential generation. Trajectory quantum analysis with the PySCF backend also imports GPU4PySCF and CuPy. There is currently no documented CPU fallback for the quantum trajectory workflow.

## 4. Runtime Configuration

Create the per-user runtime configuration:

```bash
pyedna config init
```

Edit `~/.config/pyedna/config.toml` for the local cluster or workstation. The runtime config stores machine-specific paths and is not expected to live in the repository.

NAB is configured separately:

```toml
[nab]
home = "/path/to/AmberClassic"
```

PyeDNA currently expects `<nab.home>/bin/nab`, `gcc` on `PATH`, and an active Conda environment containing a linkable `libgfortran.so`. The current validated Conda runtime prerequisite is:

```bash
conda install -c conda-forge libgfortran5
```

Do not hardcode user- or machine-specific paths into Python source files.

## 5. Validation

After editing the runtime config, inspect and validate it:

```bash
pyedna config show
pyedna config check
```

`pyedna config check` validates the config file, configured directories, required AmberTools and pmemd executables, required AmberTools OL15 data files, `<nab.home>/bin/nab`, `gcc`, `CONDA_PREFIX`, and `$CONDA_PREFIX/lib/libgfortran.so`.

## Author's note: ACPYPE / teLeap compatibility

Pip-installed ACPYPE bundles its own Amber/teLeap binaries. On some Linux systems, the bundled `teLeap` can fail because required shared libraries are not available, even when the external AmberTools `tleap` works correctly.

The project author currently uses an environment-specific workaround that redirects ACPYPE's bundled `teLeap` to the configured external AmberTools `tleap`. Installation details may change in future. If ACPYPE fails while direct `tleap` works, consult this note before changing PyeDNA inputs.

<!-- AUTHOR NOTE:
Add the currently recommended ACPYPE/teLeap workaround here.
Include any cluster-specific wrapper instructions only if desired.
-->
