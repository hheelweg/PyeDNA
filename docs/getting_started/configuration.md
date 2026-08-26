# Configuration

PyeDNA reads machine-specific paths and executable locations from the shell environment, normally by sourcing `config.sh` from `PYEDNA_HOME`.

## Environment Variables

| Variable | Required for | Meaning |
| --- | --- | --- |
| `PYEDNA_HOME` | all job wrappers; DNA generation; HADDOCK templates | Root directory of the PyeDNA repository. |
| `PYTHONPATH` | Python entry points when not installed as a package | Should include the repository scripts path in the current mask. |
| `AMBERHOME` | component parameterization, Amber setup, MD | Amber/AmberTools installation root. |
| `PATH` containing `$AMBERHOME/bin` | Amber executables | Makes `antechamber`, `resp`, `tleap`, `sander`, and `pmemd.cuda` available. |
| `ORCAHOME` | ORCA quantum backend | ORCA installation root, when ORCA jobs are used. |
| `DYE_DIR` | dye library lookup and dye library output | Root directory for reusable dye templates. |
| `LNK_DIR` | linker library lookup and linker library output | Root directory for reusable linker templates and DNA-linker compatibility parameters. |
| `DNA_DIR` | library DNA input | Directory containing reusable DNA PDB templates named `<dna.name>.pdb`. |

`config.sh.mask` currently includes `PYEDNA_HOME`, `AMBERHOME`, `ORCAHOME`, `DYE_DIR`, and `DNA_DIR`. The implementation also uses `LNK_DIR` for linker workflows and dye-linker assembly; add it to local `config.sh` when using those workflows.

## Library Layout

When `output.directory = "library"` is used for components, PyeDNA writes final reusable files into:

```text
DYE_DIR/<dye_code>/<amber_forcefield>/
LNK_DIR/<linker_code>/<amber_forcefield>/<dna_forcefield>/
```

The structure workflow resolves dye and linker inputs from:

```text
DYE_DIR/<dye>/<dye_forcefield>/<dye>.mol2
DYE_DIR/<dye>/<dye>.attach
LNK_DIR/<linker>/<dye_forcefield>/<dna_forcefield>/<linker>3.mol2
LNK_DIR/<linker>/<dye_forcefield>/<dna_forcefield>/<linker>5.mol2
LNK_DIR/connect/<dye_forcefield>/<dna_forcefield>/connectparams.frcmod
```

The compatibility file may also be named `connectparms.frcmod` for dye-linker assembly, but final Amber setup reports the canonical `connectparams.frcmod` path when missing DNA-linker parameters are detected.

## External Software Roles

AmberTools creates component charge and parameter files, and `tleap` creates final Amber topology and coordinate inputs. HADDOCK3 samples docked DNA-dye arrangements before final Amber preparation. PySCF/GPU4PySCF perform geometry optimization, electrostatic potential generation, and quantum trajectory analysis.
