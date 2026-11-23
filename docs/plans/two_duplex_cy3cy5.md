# Two-duplex CY3/CY5 system without HADDOCK: as built (2026-09-21)

Full plan with rationale: `~/.claude/plans/write-the-plan-here-eager-crown.md`. This file records
what was actually built and where it lives. Code and outputs are in `code/dye_dna/` (a sibling of
this repo, spelled `dye_DNA` on disk; macOS is case-insensitive). Cluster mirror:
`/home/gridsan/nborodin/dye_dna` on SuperCloud.

## What the system is

Two parallel 22-bp B-DNA duplexes in TIP3P with 82 Na+ (neutralize-only). Duplex 1 (resids 1–44)
is the equilibrated CY5/CY3 0nt duplex from `cg_ff/atomistic/cy3cy5/0nt` (CY5 = 11, CY3 = 12,
composite dye residues, OL15 + GAFF2), coordinates untouched. Duplex 2 (resids 45–88) is a NAB
B-DNA of the same undyed sequence `TGCACTCTCGAATGACCGAGCT`, placed as the partner helix sits in
the origami build `42_PERDIX_Triangle_cy3cy5_0nt_C22` (dyes on staple C, tucked into the
inter-helix gap). That placement reproduces the origami's groove pattern on both duplexes:

| Quantity | Origami C22 edge (measured) | Built |
| --- | --- | --- |
| inter-axis spacing at the dyes | 22.5 Å (20.6 Å global fit) | 22.5 Å |
| dye-centroid → partner azimuth | −52.8° (toward the gap, in a groove) | −52.8° |
| dyed-duplex minor-groove register toward partner | 105° | 98° (fixed by dye chemistry) |
| partner minor-groove register toward dyed duplex | −103.4° | −103.4° |
| strand sense between helices | antiparallel | antiparallel |
| dye centroid from partner axis | 16.5 Å | 18.6 Å |
| min dye atom ↔ partner P | 3.4 Å | 4.6 Å |
| min inter-duplex heavy-atom distance | 4.4 Å (P–P) | 3.24 Å, no pair < 2.5 Å |

Alternatives built and superseded: partner on the far side per the `..._0nt` (staple-B) origami
edge at 20.7 Å (dyes away, two backbone O–O contacts at 1.7/2.3 Å); dyes facing straight at the
partner (0°) at 21 Å (0.67 Å interpenetration) and at 26 Å (clean but wider than any origami edge).

## Files in `code/dye_dna/`

| File | Role |
| --- | --- |
| `helix_frame.py` | shared PDB parsing, PCA axes, signed azimuths, minor-groove register, Rodrigues rotations |
| `measure_origami_frame.py` → `origami_frame.json` | measures spacing, dye azimuth, register, strand sense in the origami |
| `build_two_duplex.py` | NAB plain duplex via PyeDNA, placement, `run/combined.pdb`, `run/tleap.in`, `run/placement.json` |
| `test_combined_pdb.py` | bookkeeping, geometry, clearance (> 2.0 Å), origami fidelity, rigidity of the NAB block |
| `test_prmtop.py` | residue labels, 8 termini, dye junction bonds 10-11-12-13, no inter-duplex bonds, 4 solute molecules, net charge, spacing in rst7 |
| `inputs/` | dyed PDB, old dye library (`CY?_del.mol2/frcmod`, `connectparms.frcmod`), triangle PDB, NAB template |
| `run/` | `combined.pdb`, `tleap.in`, `tleap.log`, `two_duplex.prmtop/.rst7`, `two_duplex_solvated.pdb`, `structures/two_duplex_1.pdb` (placeholder), `md.toml`, `md_test.toml`, `view_combined.vmd` |

Commands (Mac):

```
P=/opt/anaconda3/envs/pyedna_env/bin/python
$P measure_origami_frame.py
$P measure_origami_frame.py --pdb inputs/triangle_DX_42bp_cy3cy5_0nt_C22_dry.pdb --staple 392-433 --dyes 413,414 --out origami_frame_C22.json
$P build_two_duplex.py --seq TGCACTCTCGAATGACCGAGCT --frame origami_frame_C22.json --side origami --spacing 22.5 --no-register-scan
$P test_combined_pdb.py
(cd run && /opt/anaconda3/envs/AmberTools25/bin/tleap -f tleap.in > tleap.log)
$P test_prmtop.py
```

tleap result: `Errors = 0; Warnings = 3` (one-sided connection notes for the dye residues, and the
−82 pre-ion charge). 68694 atoms, 21914 WAT, 82 Na+, 0 Cl-.

## PyeDNA change (this repo)

`md.toml` `[system]` gained optional `prmtop` / `rst7`. When set, `prepare` must be absent from
`workflow.stages`; `_ModelSimulation.copy_topology` copies the files into the run directory and
resolves restraints from the prmtop. Files: `src/pyedna/md/config.py`, `src/pyedna/md/simulation.py`,
`docs/run_md/do_md.md`. Dry run on the Mac confirmed: restraint analysis lists exactly 8 terminal
residues (1, 22, 23, 44, 45, 66, 67, 88), all five stage inputs render.

## Deviations from the plan

- Reference origami is the C22 build (dyes toward the gap), not the staple-B build the plan
  measured (dyes away). User decision: dyes must point toward the partner.
- Spacing 22.5 Å = C22 local spacing at the dyes, instead of the plan's 21 Å.
- Partner register at the C22 value; no clearance rotation performed (`--no-register-scan`).
- Clearance 3.24 Å; default 2.0 Å gate untouched for this build.

## Gate

No MD, not even `md_test.toml`, until the user has inspected `run/combined.pdb` and
`run/two_duplex_solvated.pdb` in VMD (`vmd -e run/view_combined.vmd`) and approved.
