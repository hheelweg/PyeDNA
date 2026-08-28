# PyeDNA 🧬
#### Create DNA structures, attach dye molecules, run AMBER MD, analyze trajectories...

*credits: Maria A. Castellanos*

Future high-throughput platform for creating DNA/chromophore structures, performing all-atom MD and analyzing trajectories with classical and quantum methods.
Currenty has the following functions implemented:

- Create double stranded DNA helix (`double_helix`) with [NAB](https://github.com/Amber-MD/AmberClassic.git)
- Create custom dye-library from ChemDraw (`.cdx`) input to geometry-optimized (classical + quantum) input files and GAFF forcefield parameters for AMBER MD
- Attach dye molecules (currently only `CY3`, `CY5`) in desired orientation
- Run all-atom AMBER MD on the DNA/dye composite with GPU support
- Analyze trajectories classically and quantum-mechanically (DFT/TDDFT) with `pyscf` and `gpu4pyscf`

Future versions will include:
- Creation of more complex DNA structures 
- Curated library with topologies and GAFF parameters for dyes (`libraries.dye_dir`)
- More functionality to analyze trajectories classically
- More functionality to analyze trajectories quantum-mechanically
- Extension to perform high-troughput analysis of optoelectronic properties 
- ...

Stay tuned for more.


### Installation

Install PyeDNA as a Python package in the active scientific Python environment. For normal use, the intended released-user workflow is:

```bash
pip install pyedna
```

For developer collaborators modifying the source checkout:

```bash
pip install -e .
```

Create and validate the machine-specific runtime configuration:

```bash
pyedna config init
pyedna config show
pyedna config check
```

Edit `~/.config/pyedna/config.toml` to set paths to AmberTools, pmemd, AmberClassic/NAB, and the molecular libraries. Workflow-specific scientific settings remain in TOML files such as `structure.toml`, `md.toml`, and `traj.toml`.


### Requirements

#### NAB
In order to create/customize DNA structures, a local installation of the Nucleic Acid Builder ([NAB](https://github.com/Amber-MD/AmberClassic.git)) is required. The current runtime config uses `nab.home` for the AmberClassic root directory and expects `<nab.home>/bin/nab`.

#### Amber / AmberTools
For the Molecular Dynamics simulation we require AmberTools and Amber/pmemd. Split AmberTools and pmemd installations are configured with `amber.ambertools_home` and `amber.pmemd_home` in `~/.config/pyedna/config.toml`. **Note**: We use GPU-assisted MD executables like `pmemd.cuda` for running the MD simulations. Make sure that the Amber code is compiled with the right CUDA version of your computing cluster or local machine.

#### Python
Python package dependencies are declared in `pyproject.toml`. PyeDNA currently requires Python `>=3.12` and pins `pyscf==2.8.0` for compatibility with the validated GPU4PySCF stack.

The current quantum trajectory workflow requires GPU4PySCF and has no CPU-only execution path. The currently validated GPU Python stack is `cupy-cuda11x==13.4.1`, `gpu4pyscf-cuda11x==1.4.3`, and `pyscf==2.8.0`.


### Usage

PyeDNA workflows are launched through the installed `pyedna` CLI:

```bash
pyedna components create-dye dye.toml
pyedna components create-linker linker.toml
pyedna components create-dyelnk dyelnk.toml
pyedna structure prepare structure.toml
pyedna structure dock structure.toml
pyedna structure finalize structure.toml
pyedna structure amber structure.toml
pyedna md run md.toml
pyedna analysis trajectory traj.toml
```

Scheduler scripts may wrap these commands on HPC systems, but the repository `jobs/` directory is not required for normal installed-package use.
