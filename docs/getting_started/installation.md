# Installation

PyeDNA is developed for a Linux HPC cluster with Conda and external scientific software. The repository itself does not include Amber, AmberTools, HADDOCK3, ORCA, or quantum-chemistry executables.

## Required Software

- Python environment with the packages listed in `requirements.txt`.
- AmberTools/Amber 24, including `antechamber`, `parmchk2`, `resp`, `respgen`, `tleap`, `sander`, and GPU-capable `pmemd.cuda` for MD.
- NAB through AmberClassic or an Amber installation that provides `nab`, if generating DNA with `dna.source = "generate"`.
- HADDOCK3, used by the structure workflow after PyeDNA writes HADDOCK input files.
- PySCF and GPU4PySCF for current component parameterization and quantum analysis paths.
- ORCA is present in `config.sh.mask`; current trajectory analysis also accepts `backend = "orca"` in quantum jobs.

## Repository Setup

Clone the repository and work from the checked-out branch:

```bash
cd /path/to/PyeDNA
cp config.sh.mask config.sh
```

Edit `config.sh` for the local cluster or workstation. Do not hardcode those paths into Python source files.

## Running Workflows

PyeDNA workflows are normally launched through scripts in `jobs/`:

```bash
sbatch "$PYEDNA_HOME/jobs/components/create_dye.sh" dye.toml
sbatch "$PYEDNA_HOME/jobs/components/create_linker.sh" linker.toml
bash "$PYEDNA_HOME/jobs/components/create_dyelnk.sh" dyelnk.toml
bash "$PYEDNA_HOME/jobs/structure/create_structure.sh" structure.toml
sbatch "$PYEDNA_HOME/jobs/md/do_md_gpu.sh" md.toml
python "$PYEDNA_HOME/scripts/analyze_traj.py" traj.toml
```

Component parameterization, HADDOCK, MD, and quantum analysis can be computationally expensive. Submit those workflows only on appropriate resources.
