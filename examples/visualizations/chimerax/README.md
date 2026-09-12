# ChimeraX visualization example

This directory contains a small example workflow for viewing prepared PyeDNA structures in ChimeraX.

`chimera_style.cxc` defines the canonical example styling for DNA, dyes, and linkers.
Dye colors are keyed by dye residue name, such as `CY3`, `CY5`, `PDI`, and `SQA`.
Linker styling is keyed by linker residue names, such as `C33`/`C35`, `C63`/`C65`, and `DE3`/`DE5`.

`view_structures.sh` opens every `.pdb` file in `structures/` in one ChimeraX session, then applies `chimera_style.cxc`.

Expected working directory layout:

```text
.
|-- view_structures.sh
|-- chimera_style.cxc
`-- structures/
    |-- something_1.pdb
    `-- something_2.pdb
```

Minimal usage:

```bash
cp examples/visualizations/chimerax/view_structures.sh .
cp examples/visualizations/chimerax/chimera_style.cxc .
./view_structures.sh
```

Set `CHIMERAX=/path/to/chimerax` if your ChimeraX executable is not named `chimerax`.
