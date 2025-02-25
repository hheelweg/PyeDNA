# PyeDNA 🧬
#### Create DNA structures, attache dye molecules, run AMBER MD, analyze trajectories...

*credits: Maria A. Castellanos*

Future high-throughput platform for creating DNA/chromophore structures, performing all-atom MD and analyzing trajectories with classical and quantum methods.
Currenty has the folling functions implemented:

- Create double stranded DNA helix (`double_helix`) with NAB
- Attach dye molecules (currently only `CY3`, `CY5`) in desired orientation
- Run AMBER MD on the DNA/dye composite
- Analyze trajectories classicaly and quantum-mechanically (DFT/TDDFT) with `pyscf`

Future versions will include:
- Creation of more complex DNA structure
- Modules for construct dye input structures from ChemDraw (`.cdx`) files
- More functionality to analyze trajectories classically
- ...

Stay tuned 🚨