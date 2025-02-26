# PyeDNA 🧬
#### Create DNA structures, attache dye molecules, run AMBER MD, analyze trajectories...

*credits: Maria A. Castellanos*

Future high-throughput platform for creating DNA/chromophore structures, performing all-atom MD and analyzing trajectories with classical and quantum methods.
Currenty has the following functions implemented:

- Create double stranded DNA helix (`double_helix`) with [NAB](https://github.com/Amber-MD/AmberClassic.git)
- Attach dye molecules (currently only `CY3`, `CY5`) in desired orientation
- Run all-atom AMBER MD on the DNA/dye composite with GPU support
- Analyze trajectories classicaly and quantum-mechanically (DFT/TDDFT) with `pyscf` and `gpu4pyscf`

Future versions will include:
- Creation of more complex DNA structure
- Curated `DYE_DIR` with topologies and GAFF parameters for dyes
- Module for constructing dye input structures from ChemDraw (`.cdx`) files
- More functionality to analyze trajectories classically
- More functionality to analyze trajectories quantum-mechanically
- Extension to perform high-troughput analysis of optoelectronic properties 
- ...

Stay tuned for more 🚨


### Installation

In order to make sure user-specific environment variables are set, the user needs to set up a `config.sh` file in `PYEDNA_HOME`. A mask (`config.sh.mask`) is provided in the root directory. Navigate to `PYEDNA_HOME`, and then type.  

```
cp config.sh.mask config.sh
nano config.sh
```

Then set the Python environment `[env-name]`, as well as the paths to `AMBERHOME` and the (custom) `DYE_DIR` in order to reference constructed (custom) dyes.


### Requirements

add this! 


### Usage

Before running *any* type of calculation, make sure the `PYEDNA_HOME` variable is set up correctly. In order to do that, run the following command in the shell

```
export PYEDNA_HOME="/path/to/PyeDNA"
```
One can also add this to the `~/.zshrc` or `~/.bashrc` for a permanent addition to the shell configuration.
Before executing job scripts from `jobs` directory, type

```
export $PATH:/path/to/PyeDNA/jobs
```

Before executing bash scripts from `bin` directory, type

```
export $PATH:/path/to/PyeDNA/bin
```