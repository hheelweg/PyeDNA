## 02 — Run molecular dynamics

PyeDNA automatically prepares and executes a complete AMBER molecular dynamics
workflow for the DNA–dye system generated in Step 1.

The workflow consists of

1. energy minimization,
2. restrained equilibration,
3. pressure equilibration,
4. production molecular dynamics.

All AMBER input files are generated automatically from the parameters provided
in `md.params`.

---

### Required input

The MD workflow requires the structure generated during the previous step:

```text
dna_1nt.prmtop
dna_1nt.rst7
```

These files contain the complete solvated DNA–dye system together with all
force-field information.
The workflow additionally requires a parameter file

```text
md.params
```

located in the current working directory.
A typical directory is therefore

```text
md/
├── dna_1nt.prmtop
├── dna_1nt.rst7
└── md.params
```

---

### MD parameters (`md.params`)

An example parameter file is

```python
res_fstrong = 500
res_fweaker = 10
res_fweak   = 5

min_maxcyc  = 2000

temp        = 300
pres        = 1
gamma_ln    = 5
taup        = 5
dt          = 0.002

eq1_nstlim  = 500000
eq2_nstlim  = 500000

prod_nstlim = 100000000
prod_ntpr   = 50000
prod_ntwx   = 100000
prod_ntwr   = 100000
```

Only a small number of parameters usually need to be modified.

#### DNA restraints

During minimization and equilibration, harmonic restraints are applied to
selected DNA residues to preserve the overall DNA geometry while allowing the
solvent and dyes to relax.

PyeDNA performs three successive stages using

- strong restraints,
- weaker restraints,
- weak restraints,

before the production simulation.

For most systems the default restraint constants are sufficient.

---

#### Energy minimization

```python
min_maxcyc
```

defines the maximum number of minimization iterations.

This stage removes steric clashes introduced during structure construction and
relaxes the solvent before molecular dynamics begins.

---

#### Thermostat and barostat

The default MD protocol uses

- Langevin dynamics for temperature control,
- isotropic pressure coupling,
- a 2 fs timestep.

The corresponding parameters are

```python
temp
pres
gamma_ln
taup
dt
```

The defaults are appropriate for standard room-temperature simulations and
normally do not require modification.

---

#### Equilibration

Two equilibration stages are performed.

```python
eq1_nstlim
eq2_nstlim
```

define their lengths.
During equilibration the solvent density stabilizes while positional restraints
on the DNA are gradually relaxed.

---

#### Production simulation

The production run generates the trajectory used for all subsequent analysis.
Its duration is determined by

```python
prod_nstlim
```

while

```python
prod_ntpr
```

controls how often energies are written,

```python
prod_ntwx
```

controls how frequently trajectory frames are saved,

and

```python
prod_ntwr
```

determines the restart-file interval.
For most users these are the parameters most frequently adjusted.

---

### Default simulation protocol

PyeDNA automatically performs the following sequence:

#### 1. Energy minimization

The complete solvated system is minimized while strong restraints are applied
to selected DNA residues.
This removes steric clashes introduced during structure preparation.

#### 2. Heating and restrained equilibration

The system is heated to the target temperature while gradually reducing the DNA
restraints.
The solvent and dye molecules relax around the DNA without allowing large
distortions of the duplex.

#### 3. Pressure equilibration

The simulation continues under constant pressure until the solvent density and
simulation box stabilize.

#### 4. Production MD

All production data are generated during this stage.
Coordinates, energies and restart files are written at the user-specified
intervals.

---

### Running on a GPU

PyeDNA executes the production workflow through the supplied

```text
do_md_gpu.sh
```

script, which launches the AMBER GPU engine (`pmemd.cuda`).
The exact GPU resources depend on the local cluster or workstation.
Typical settings that users may wish to modify include

- SLURM partition,
- GPU type,
- number of GPUs,
- wall time,
- number of CPU cores used for auxiliary tasks.

These settings are independent of the molecular dynamics parameters in
`md.params`.

---

### Generated files

The workflow automatically creates

- AMBER input files for every simulation stage,
- minimization outputs,
- equilibration restart files,
- production restart files,
- production trajectory (`*.nc`),
- AMBER log (`*.out`) files.

The principal outputs for subsequent analysis are

```text
dna_1nt.prmtop
dna_1nt.nc
dna_1nt.rst7
```

where the NetCDF trajectory (`.nc`) contains the complete molecular dynamics
trajectory.