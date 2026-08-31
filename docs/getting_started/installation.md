# Installation

PyeDNA is a Python package that orchestrates different external scientific software. Installing PyeDNA installs the Python dependencies declared in `pyproject.toml`; it does not install AmberTools, Amber/pmemd, AmberClassic/NAB, HADDOCK3, ACPYPE, CUDA, or GPU4PySCF. Those installations are further detailled below. 

**Note**:
Component parameterization, HADDOCK docking, MD, and quantum analysis can be computationally expensive. Submit those workflows only on appropriate compute resources.

## 1. PyeDNA Python Package

PyeDNA should be installed in a dedicated Python environment. The currently supported Python version is `>=3.12`, as declared in `pyproject.toml`. Older Python versions might also be supported, 

We recommend using a Conda environment:

```bash
conda create -n pyedna_env python=3.13 -y
conda activate pyedna_env
```

### Scientific users

For the current pre-release version, install PyeDNA from a wheel provided with the corresponding release:

```bash
pip install pyedna-<version>-py3-none-any.whl
```

The future goal is installation directly from a Python package index:

```bash
pip install <PyeDNA-distribution-name>
```

Scientific users do not need to clone the PyeDNA source repository. 

> **Note**:
>
> This is not implemented yet but will be once PyeDNA is release-ready. So currently installation from the wheel provided (likely in `PyeDNA/dist/`) is recommended. 


### Developer collaborators

Collaborators who intend to modify PyeDNA should install it from a source checkout:

```bash
git clone <repository>
cd PyeDNA
pip install -e .
```

Scientific collaborators who only run PyeDNA workflows should use the release installation described above.

The package dependencies include `pyscf==2.8.0`, which is currently required for compatibility with the validated GPU4PySCF stack.

We recommend installing this into it's own dedicated Conda environement (e.g. `pyedna_env`) as suggested above. 

## 2. External Scientific Software

The following runtime dependencies must be installed separately and configured for the machine or cluster.

| Dependency | Role in PyeDNA |
| --- | --- |
| AmberTools 26-compatible installation | Small-molecule parameterization, RESP fitting, `tleap` Amber setup, and AmberTools data files. |
| Amber / pmemd26-compatible installation | `pmemd`, `pmemd.MPI`, and `pmemd.cuda` executables for MD minimization, equilibration, and production. |
| AmberClassic / NAB | Generated DNA structures when `dna.source = "generate"`. |
| HADDOCK3 | Structure docking after PyeDNA writes `docking_config.cfg`. |
| ACPYPE | Conversion of dye-linker components into HADDOCK/CNS topology and parameter files. |
| NVIDIA/CUDA environment | Required for the current GPU4PySCF quantum trajectory workflow and for GPU Amber MD with `pmemd.cuda`. |

### Amber and AmberTools

PyeDNA's MD engine is built entirely on Amber and AmberTools. Current version supports [Amber26](https://ambermd.org/GetAmber.php#ambertools) and [AmberTools26](https://ambermd.org/GetAmber.php#ambertools). Please refer to their documentation for the installation. 

> **Important: GPU resources**
>
> Note that in order to use availale GPU resources properly one might need to specify proper CUDA path `-DCUDA_TOOLKIT_ROOT_DIR=/path/to/cuda` in `run_cmake` when compiling Amber26. One also needs to set `-DCUDA=TRUE` before running `./run_cmake`.  

PyeDNA currently resolves AmberTools executables from `amber.ambertools_home` and pmemd executables from `amber.pmemd_home` in the runtime config where they need to be fed manually as:

```toml
[amber]
ambertools_home = "/path/to/ambertools26"
pmemd_home = "/path/to/pmemd26"
```

The configured AmberTools installation is expected to provide the executables currently invoked by PyeDNA workflows: `antechamber`, `parmchk2`, `resp`, `respgen`, `sander`, and `tleap`. 

The configured Amber/pmemd installation must provide the executable required by the requested MD resources: `pmemd` for serial CPU jobs, `pmemd.MPI` for CPU MPI jobs, and `pmemd.cuda` for GPU jobs. The MD workflow uses `pmemd` when no CUDA device is visible to the process and one SLURM task is requested, `pmemd.MPI` when no CUDA device is visible and multiple SLURM tasks are requested, and `pmemd.cuda` when a CUDA device is visible.

> **Important**
> 
> The specification of the two Amber home directories is related to the split of relevant executables between AmberTools26 and licensed Amber26. This might be subject to change for future Amber versions.

PyeDNA resolves AmberTools data from `ambertools_home`. It directly reads `dat/leap/lib/DNA.OL15.lib` for OL15 reference atom charges during linker RESP restraint generation, and `tleap` loads DNA force-field data through `leaprc.DNA.OL15`. Users do not configure individual OL15 data files such as `frcmod.DNA.OL15` as separate PyeDNA runtime settings.


### HADDOCK3 and ACPYPE

HADDOCK3 and ACPYPE should be installed into the same active Python environment used for PyeDNA:

```bash
conda activate pyedna_env
pip install haddock3 acpype
```

[HADDOCK3](https://github.com/haddocking/haddock3) is used during the structure workflow to generate physically plausible DNA–dye arrangements subject to the configured attachment restraints. 

[ACPYPE](https://github.com/alanwilter/acpype) is used to convert dye-linker components into the CNS-compatible topology and parameter files required by HADDOCK3 for non-standard residues.

> **Disclaimer: ACPYPE and `teLeap`!**
>
> Pip-installed ACPYPE (`pip install acpype`) bundles its own Amber/teLeap binaries. On some Linux systems, the bundled `teLeap` can fail because required shared libraries are not available, even when the external AmberTools `tleap` works correctly.

One workaround for this, given Amber(Tools) has been succesfully installed is provided here: We can use an environment-specific workaround that redirects ACPYPE's bundled `teLeap` to the configured external AmberTools `tleap`. Installation details may change in future. If ACPYPE fails while direct `tleap` works, consult this note before changing PyeDNA inputs. You need to find the path to ACPYPE's `teLeap` first, something like `/path/to/acpype/amber_linux/bin/teLeap`, and then also the home directory of the AmberTools installation, with 

```
export AMBERTOOLS_HOME="/path/to/ambertools26"
```

One can then build the wrapper for ACPYPE's `teLeap` pointing to the AmberTools installation. 

```
ACPYPE_TLEAP="/path/to/acpype/amber_linux/bin/teLeap"

mv "$ACPYPE_TLEAP" "${ACPYPE_TLEAP}.original"

cat > "$ACPYPE_TLEAP" <<EOF
#!/usr/bin/env bash
export AMBERHOME="$AMBERTOOLS_HOME"
export PATH="\$AMBERHOME/bin:\$PATH"
export LD_LIBRARY_PATH="\$AMBERHOME/lib\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}"
exec "\$AMBERHOME/bin/tleap" "\$@"
EOF

chmod +x "$ACPYPE_TLEAP"
```

### Nucleid Acid Builder (NAB)

In order to create DNA structures e.g. from sequence strings, PyeDNA uses NAB, currently available in [AmberClassic](https://github.com/Amber-MD/AmberClassic.git). Refer to their documentation for installation guidelines. The current runtime config uses `nab.home` for the AmberClassic root directory and expects `<nab.home>/bin/nab`: 

```toml
[amber]
[nab]
home = "/path/to/AmberClassic"
```

PyeDNA currently expects `<nab.home>/bin/nab`, `gcc` on `PATH`, and an active Conda environment containing a linkable `libgfortran.so`. The current validated Conda runtime prerequisite is:

```bash
conda activate pyedna_env
conda install -c conda-forge libgfortran5
```

Do not hardcode user- or machine-specific paths into Python source files.

### GPU4PySCF

> **Important**
>
> The current PyeDNA quantum trajectory workflow requires [GPU4PySCF](https://github.com/pyscf/gpu4pyscf). A CPU-only execution path is not currently implemented for that workflow. Users intending to run quantum trajectory analysis must install the validated CUDA/CuPy/GPU4PySCF stack and run on a compatible NVIDIA GPU.
>
> Component parameterization is different: `pyedna components create-dye` and `pyedna components create-linker` can generate RESP electrostatic-potential files with plain CPU PySCF, and use GPU4PySCF automatically only when the package stack is installed and a CUDA device is visible to the process.

The currently validated GPU Python stack is:

```text
cupy-cuda11x==13.4.1
gpu4pyscf-cuda11x==1.4.3
pyscf==2.8.0
```

> **Note**
>
> The versions we install here need to be adjusted to the available CUDA version, e.g. CUDA 11.5 requires `-cuda11x` for both CuPy and GPU4PySCF. 

`pyscf==2.8.0` is installed as a PyeDNA package dependency. More recent versions of PySCF might fail (this has not been thoroughly tested). The appropriate versions of CuPy and GPU4PySCF need to be installed separately in the active Conda environment as

```
pip install "cupy-cuda11x==13.4.1"
pip install "gpu4pyscf-cuda11x==1.4.3"
```

One can also install the appropriate version of cuTENSOR.

For component parameterization, the same scientific TOML file and CLI command are used for CPU and GPU jobs. Backend selection is controlled by the runtime resources allocated to the process:

- CPU jobs should request no GPU resources. PyeDNA falls back to CPU PySCF even if CuPy and GPU4PySCF are installed in the active environment but no CUDA device is visible.
- GPU jobs should request a GPU through the scheduler, for example with `#SBATCH --gres=gpu:1`. If CuPy, GPU4PySCF, and a visible CUDA device are all available, PyeDNA uses GPU4PySCF for PySCF-backed geometry optimization and RESP ESP generation.

Amber RESP fitting itself is CPU AmberTools software. GPU4PySCF accelerates the PySCF quantum calculation used to generate the electrostatic-potential target for RESP. Trajectory quantum analysis with the PySCF backend still imports GPU4PySCF and CuPy. **Note:** There is currently no documented CPU fallback for the quantum trajectory workflow.


## 3. Runtime Configuration

External software that has not been installed in the active Conda environment (e.g. `pyedna_env`), as mentioned above, needs to be added to the runtime configuration of PyeDNA. 
In order to create the per-user runtime configuration:

```bash
pyedna config init
```

Edit `~/.config/pyedna/config.toml` for the local cluster or workstation. The runtime config stores machine-specific paths and is not expected to live in the repository.

### Validation

After editing the runtime config, inspect and validate it:

```bash
pyedna config show
pyedna config check
```

`pyedna config check` validates the config file, configured directories, required AmberTools executables, required `pmemd`, optional mapped Amber executables such as `pmemd.MPI` and `pmemd.cuda`, required AmberTools OL15 data files, `<nab.home>/bin/nab`, `gcc`, `CONDA_PREFIX`, and `$CONDA_PREFIX/lib/libgfortran.so`.
 More information can be found in the [configuration.md](configuration.md). 

 > **Important**
 >
 > Make sure to run PyeDNA through the CLI **only** with activated Conda environment, i.e. `conda activate pyedna_env`.


