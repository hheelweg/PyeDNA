# Configuration

PyeDNA uses two kinds of configuration:

- Runtime config: machine- and user-specific installation paths in `~/.config/pyedna/config.toml`.
- Scientific config: workflow settings in component TOMLs, `structure.toml`, `md.toml`, and `traj.toml`.

The runtime config is not expected to live inside the repository. It stores local executable roots and library directories that differ between clusters, workstations, and users.

## Runtime Config File

Create a template with:

```bash
pyedna config init
```

The current schema is:

```toml
[amber]
ambertools_home = "/path/to/ambertools26"
pmemd_home = "/path/to/pmemd26"

[nab]
home = "/path/to/AmberClassic"

[libraries]
dye_dir = "/path/to/dye_library"
dna_dir = "/path/to/dna_library"
linker_dir = "/path/to/linker_library"
```

| Setting | Required for | Meaning |
| --- | --- | --- |
| `amber.ambertools_home` | component parameterization, dye-linker assembly, Amber setup, minimization, AmberTools data | Root of the AmberTools installation. PyeDNA resolves AmberTools executables and Amber data relative to this root. |
| `amber.pmemd_home` | MD equilibration and production | Root of the Amber/pmemd installation. PyeDNA resolves pmemd-family executables from this root. |
| `nab.home` | generated DNA structures | Root of the AmberClassic/NAB installation. PyeDNA expects `<nab.home>/bin/nab`. |
| `libraries.dye_dir` | dye library lookup and dye library output | User's dye parameter/template library. |
| `libraries.dna_dir` | library DNA input | User's DNA structure library containing reusable DNA PDB templates named `<dna.name>.pdb`. |
| `libraries.linker_dir` | linker library lookup and linker library output | User's linker parameter/template library, including curated DNA-linker compatibility parameters. |

The current code still accepts a legacy `[amber].home` form when split AmberTools/pmemd paths are absent, but new runtime configs should use `ambertools_home` and `pmemd_home`.

For AmberTools commands, PyeDNA sets `AMBERHOME` to `amber.ambertools_home` and prepends the AmberTools `bin` and `lib` directories. For pmemd commands, PyeDNA also sets `PMEMDHOME` to `amber.pmemd_home` and prepends both AmberTools and pmemd `bin` and `lib` directories.

## Config CLI

```bash
pyedna config init
pyedna config show
pyedna config check
```

`pyedna config init` creates `~/.config/pyedna/config.toml` from a template. It creates parent directories as needed and leaves an existing config unchanged.

`pyedna config show` prints the resolved config path and resolved values for `amber.ambertools_home`, `amber.pmemd_home`, `nab.home`, and the three library directories.

`pyedna config check` validates:

- `~/.config/pyedna/config.toml` exists and can be loaded;
- configured AmberTools, pmemd, NAB, dye-library, DNA-library, and linker-library directories exist;
- `<nab.home>/bin/nab` exists;
- required AmberTools executables `antechamber`, `parmchk2`, `resp`, `respgen`, `sander`, and `tleap` exist;
- required pmemd executables `pmemd` and `pmemd.cuda` exist;
- AmberTools data files `dat/leap/lib/DNA.OL15.lib` and `dat/leap/cmd/leaprc.DNA.OL15` exist;
- optional mapped Amber executables such as `cpptraj` and `prepgen` are reported when present;
- `gcc`, `CONDA_PREFIX`, and `$CONDA_PREFIX/lib/libgfortran.so` are available for NAB.

## Scientific Config Files

Scientific workflow settings remain in TOML files that are passed to the CLI:

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

These files store scientific choices such as SMILES strings, atom-map IDs, attachment residues, force-field identifiers, MD stages, trajectory selections, and analysis groups. They should be portable across machines except for ordinary input/output file locations chosen by the user.

## Library Layout

When `output.directory = "library"` is used for components, PyeDNA writes final reusable files into:

```text
<libraries.dye_dir>/<dye_code>/<amber_forcefield>/
<libraries.linker_dir>/<linker_code>/<amber_forcefield>/<dna_forcefield>/
```

The structure workflow resolves dye and linker inputs from:

```text
<libraries.dye_dir>/<dye>/<dye_forcefield>/<dye>.mol2
<libraries.dye_dir>/<dye>/<dye>.attach
<libraries.linker_dir>/<linker>/<dye_forcefield>/<dna_forcefield>/<linker>3.mol2
<libraries.linker_dir>/<linker>/<dye_forcefield>/<dna_forcefield>/<linker>5.mol2
<libraries.linker_dir>/connect/<dye_forcefield>/<dna_forcefield>/connectparams.frcmod
```

The compatibility file may also be named `connectparms.frcmod` for dye-linker assembly, but final Amber setup reports the canonical `connectparams.frcmod` path when missing DNA-linker parameters are detected.

## External Software Roles

AmberTools creates component charge and parameter files, and `tleap` creates final Amber topology and coordinate inputs. Amber/pmemd runs GPU MD stages. AmberClassic/NAB generates simple DNA templates. HADDOCK3 samples docked DNA-dye arrangements before final Amber preparation. ACPYPE prepares HADDOCK/CNS topology inputs for dye-linker components. PySCF/GPU4PySCF perform geometry optimization, electrostatic-potential generation, and quantum trajectory analysis.
