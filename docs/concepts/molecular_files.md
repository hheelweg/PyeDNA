# Molecular Files

This page summarizes the main molecular file types used by PyeDNA.

| File type | Where it appears | Role |
| --- | --- | --- |
| PDB (`.pdb`) | DNA templates, HADDOCK inputs/results, final structures, solvated structures | Stores atom coordinates, residue names, residue IDs, chain IDs, and segment IDs. |
| SDF (`.sdf`) | component parameterization intermediates | Stores small-molecule structures used by RDKit and AmberTools. |
| XYZ (`.xyz`) | QM geometry optimization intermediates | Stores atom symbols and coordinates for quantum calculations. |
| MOL2 (`.mol2`) | dye/linker libraries, linked dye-linker components | Stores coordinates, atom names, atom types, bonds, residue names, and partial charges. |
| FRCMOD (`.frcmod`) | dye/linker libraries and compatibility parameters | Stores Amber parameter terms missing from the base force fields. |
| attach (`.attach`) | dye/linker metadata | Records attachment atom names such as `LINKER`, `3CONNECT`, `5CONNECT`, `5'`, `3'`, and optional Amber atom mappings. |
| PRMTOP (`.prmtop`) | Amber setup output and MD/analysis input | Amber topology containing atoms, residues, parameters, charges, bonds, solvent metadata, and molecule metadata. |
| RST7 (`.rst7`) / NCRST (`.ncrst`) | Amber setup and MD stage restarts | Amber coordinate/restart files. |
| NetCDF (`.nc`) | MD trajectories | Binary Amber trajectory format used by MD and trajectory analysis. |
| TOML (`.toml`) | user workflow configuration | Human-editable configuration for PyeDNA workflows. |
| JSONL (`.jsonl`) | trajectory analysis outputs | One JSON record per result, suitable for streaming and later loading. |
| CSV (`.csv`) | structure bond and mapping metadata | Tabular metadata used to reconstruct connectivity and Amber inputs. |

## File Resolution

Relative user-supplied paths are generally resolved from the current working directory. Library files are resolved from runtime configuration paths such as `libraries.dye_dir`, `libraries.linker_dir`, and `libraries.dna_dir`.

## Important Metadata Files

`structures/bonds.csv` records final covalent bonds that must be passed to `tleap`. `resid_mapping.json` records attachment residue mapping used later by trajectory analysis.
